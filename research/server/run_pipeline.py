"""Restartable, stage-oriented server runner for the verified benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGES = ["check", "download", "preprocess", "split", "tune", "train", "verify", "report", "test"]
DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execute(arguments: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    print("RUN", " ".join(arguments), flush=True)
    with log.open("a") as stream:
        process = subprocess.Popen(arguments, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, arguments)


def stage_complete(stage: str) -> bool:
    manifest = json.loads((ROOT / "research/server/data_manifest.json").read_text())
    if stage == "download":
        path = ROOT / "data/raw/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv"
        return path.is_file() and path.stat().st_size == manifest["csv"]["bytes"] and digest(path) == manifest["csv"]["sha256"]
    if stage == "preprocess":
        report = ROOT / "research/results/preprocessing_four_verification.json"
        outputs = [ROOT / "data/processed_four" / f"{name}.parquet" for name in DATASETS]
        return (report.is_file() and json.loads(report.read_text()).get("passed") is True
                and all(path.is_file() and digest(path) == manifest["processed_parquet_sha256"][path.stem]
                        for path in outputs))
    if stage == "split":
        report = ROOT / "research/results/minibatch_prepare.json"
        pilots = [ROOT / "data/minibatch_splits" / dataset / f"pilot_{split}.parquet"
                  for dataset in DATASETS for split in ("train", "val", "test")]
        return (report.is_file() and json.loads(report.read_text()).get("complete") is True
                and all(path.is_file() for path in pilots))
    if stage == "tune":
        return (ROOT / "research/artifacts/minibatch_tuning/selection.json").is_file()
    if stage == "train":
        runs = ROOT / "research/artifacts/minibatch_runs/runs.csv"
        if not runs.is_file():
            return False
        import pandas as pd
        return len(pd.read_csv(runs)) == 72
    if stage == "verify":
        path = ROOT / "research/results/minibatch_verification.json"
        return path.is_file() and json.loads(path.read_text()).get("runs_checked") == 72
    if stage == "report":
        return (ROOT / "research/MINIBATCH_REPORT_VI.md").is_file() and (ROOT / "research/results/minibatch/summary.csv").is_file()
    return False


def run_stage(stage: str, args) -> None:
    py = sys.executable
    logs = ROOT / "research/artifacts/server_logs"
    if stage == "check":
        execute([py, "research/server/check_server.py", "--output", "research/results/server_environment.json"], logs / "00_check.log")
    elif stage == "download":
        if stage_complete("preprocess"):
            print("SKIP download: verified processed Parquet files are present", flush=True)
            return
        if not stage_complete(stage):
            execute([py, "research/download_kaggle.py"], logs / "01_download.log")
        execute([py, "research/verify_kaggle_data.py"], logs / "01_inventory.log")
    elif stage == "preprocess":
        if not stage_complete(stage):
            output = ROOT / "data/processed_four"
            if output.exists():
                raise RuntimeError("data/processed_four exists but is incomplete or has a checksum mismatch; inspect it before retrying")
            execute([py, "research/preprocess_four.py", "--source", "data/raw/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv", "--output", "data/processed_four", "--report", "research/results/preprocessing_four.json"], logs / "02_preprocess.log")
            execute([py, "research/verify_preprocessed_four.py"], logs / "02_verify.log")
    elif stage == "split":
        if not stage_complete(stage):
            output = ROOT / "data/minibatch_splits"
            if output.exists():
                raise RuntimeError("data/minibatch_splits exists but is incomplete; inspect it before retrying")
            execute([py, "-m", "nids_minibatch.prepare", "--source", "data/processed_four", "--output", "data/minibatch_splits", "--report", "research/results/minibatch_prepare.json", "--threads", str(args.threads)], logs / "03_split.log")
    elif stage == "tune":
        if not stage_complete(stage):
            execute([py, "-m", "nids_minibatch.tune", "--data", "data/minibatch_splits", "--output", "research/artifacts/minibatch_tuning", "--threads", str(args.threads), "--device", args.device], logs / "04_tune.log")
    elif stage == "train":
        selection = json.loads((ROOT / "research/artifacts/minibatch_tuning/selection.json").read_text())["selected"]
        command = [py, "-m", "nids_minibatch.training", "--data", "data/minibatch_splits", "--output", "research/artifacts/minibatch_runs", "--datasets", *DATASETS, "--tasks", "multiclass", "binary", "--models", "edge_mlp", "sage", "sage_edge", "--seeds", "11", "22", "33", "--epochs", "60", "--patience", "10", "--batch-size", str(selection["batch_size"]), "--fanout", *map(str, selection["fanout"]), "--threads", str(args.threads), "--device", args.device]
        if (ROOT / "research/artifacts/minibatch_runs").exists():
            command.append("--resume")
        if not stage_complete(stage):
            execute(command, logs / "05_train.log")
    elif stage == "verify":
        execute([py, "research/validate_minibatch_results.py", "--data", "data/minibatch_splits", "--runs", "research/artifacts/minibatch_runs", "--output", "research/results/minibatch_verification.json", "--threads", str(args.threads)], logs / "06_verify.log")
    elif stage == "report":
        execute([py, "research/build_minibatch_report.py", "--runs", "research/artifacts/minibatch_runs", "--tuning", "research/artifacts/minibatch_tuning", "--prepare", "research/results/minibatch_prepare.json", "--verification", "research/results/minibatch_verification.json", "--output", "research/results/minibatch", "--report", "research/MINIBATCH_REPORT_VI.md"], logs / "07_report.log")
    elif stage == "test":
        execute([py, "-m", "pytest", "tests", "-q"], logs / "08_test.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else [args.stage]
    for stage in selected:
        print(f"\n=== {stage.upper()} ===", flush=True)
        run_stage(stage, args)
    print("SERVER PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
