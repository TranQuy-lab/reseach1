"""Restartable, benchmark-gated full-data server pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
MODELS = ["edge_mlp", "sage", "sage_edge"]
STAGES = ["check", "split", "benchmark", "train", "verify", "report", "test"]
DEFAULT_MAX_TRAIN_STEPS = 20_000
DEFAULT_EVAL_EVERY_STEPS = 1_000


def execute(arguments: list[str], log: Path, check: bool = True) -> int:
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
        code = process.wait()
    if check and code:
        raise subprocess.CalledProcessError(code, arguments)
    return code


def train_command(output: str, datasets: list[str], models: list[str], seeds: list[int],
                  tasks: list[str], epochs: int, patience: int, threads: int,
                  max_batches: int = 0, prediction_cap: int = 100_000,
                  max_steps: int = 0, eval_every_steps: int = 0) -> list[str]:
    command = [
        sys.executable, "-m", "nids_minibatch.training", "--data", "data/full_splits",
        "--output", output, "--datasets", *datasets, "--tasks", *tasks,
        "--models", *models, "--seeds", *map(str, seeds), "--epochs", str(epochs),
        "--patience", str(patience), "--batch-size", "4096", "--fanout", "15", "10",
        "--threads", str(threads), "--device", "cuda", "--scope", "full",
        "--eval-every", "3", "--amp", "--prediction-cap", str(prediction_cap),
        "--max-train-batches", str(max_batches),
    ]
    if max_steps:
        command.extend(["--max-train-steps", str(max_steps),
                        "--eval-every-steps", str(eval_every_steps)])
    return command


def require_launch_gate() -> dict:
    path = ROOT / "research/results/full_benchmark_estimate.json"
    if not path.is_file():
        raise RuntimeError("Run the bounded benchmark before full training")
    value = json.loads(path.read_text())
    if value.get("safe_to_launch_72") is not True:
        raise RuntimeError("Full-data launch gate failed; inspect full_benchmark_estimate.json")
    budget = value.get("training_budget", {})
    if budget != {
        "max_train_steps": DEFAULT_MAX_TRAIN_STEPS,
        "eval_every_steps": DEFAULT_EVAL_EVERY_STEPS,
    }:
        raise RuntimeError("Launch gate training budget differs from the locked protocol")
    return value


def run_stage(stage: str, threads: int) -> None:
    logs = ROOT / "research/artifacts/full_server_logs"
    if stage == "check":
        execute([sys.executable, "research/server/check_server.py", "--output",
                 "research/results/server_environment.json", "--scope", "full",
                 "--require-cuda"], logs / "00_check.log")
    elif stage == "split":
        report = ROOT / "research/results/full_prepare.json"
        if report.is_file():
            value = json.loads(report.read_text())
            if (value.get("complete") is True
                    and value.get("manifest_schema_version") == 2
                    and value.get("protocol") == "PROTOCOL_FULL_DATA_VI.md"):
                return
        execute([sys.executable, "-m", "nids_minibatch.prepare", "--source",
                 "data/processed_four", "--output", "data/full_splits", "--report",
                 str(report.relative_to(ROOT)), "--threads", str(threads),
                 "--protocol", "PROTOCOL_FULL_DATA_VI.md"],
                logs / "01_split.log")
    elif stage == "benchmark":
        bench = ROOT / "research/artifacts/full_benchmark"
        failures = []
        for dataset in DATASETS:
            output = bench / dataset
            if (output / "runs.csv").is_file() and len(__import__("pandas").read_csv(output / "runs.csv")) == 3:
                continue
            command = train_command(str(output.relative_to(ROOT)), [dataset], MODELS, [11],
                                    ["multiclass"], 1, 1, threads, 500, 1_000)
            if output.exists():
                command.append("--resume")
            code = execute(command, logs / f"02_benchmark_{dataset}.log", check=False)
            if code:
                failures.append(dataset)
        if failures:
            raise RuntimeError(f"Benchmark failed for: {failures}")
        execute([sys.executable, "research/estimate_full_runtime.py", "--benchmarks",
                 "research/artifacts/full_benchmark", "--prepare", "research/results/full_prepare.json",
                 "--environment", "research/results/server_environment.json", "--output",
                 "research/results/full_benchmark_estimate.json", "--max-train-steps",
                 str(DEFAULT_MAX_TRAIN_STEPS), "--eval-every-steps",
                 str(DEFAULT_EVAL_EVERY_STEPS)], logs / "02_estimate.log")
    elif stage == "train":
        gate = require_launch_gate()
        budget = gate["training_budget"]
        output = ROOT / "research/artifacts/full_runs"
        command = train_command(str(output.relative_to(ROOT)), DATASETS, MODELS,
                                [11, 22, 33], ["multiclass", "binary"], 1, 10, threads,
                                max_steps=budget["max_train_steps"],
                                eval_every_steps=budget["eval_every_steps"])
        if output.exists():
            command.append("--resume")
        execute(command, logs / "03_train.log")
    elif stage == "verify":
        execute([sys.executable, "research/validate_minibatch_results.py", "--data",
                 "data/full_splits", "--runs", "research/artifacts/full_runs", "--output",
                 "research/results/full_verification.json", "--threads", str(threads)],
                logs / "04_verify.log")
    elif stage == "report":
        execute([sys.executable, "research/build_full_report.py", "--runs",
                 "research/artifacts/full_runs", "--verification",
                 "research/results/full_verification.json", "--benchmark",
                 "research/results/full_benchmark_estimate.json", "--prepare",
                 "research/results/full_prepare.json", "--output",
                 "research/results/full", "--report", "research/FULL_DATA_REPORT_VI.md"],
                logs / "05_report.log")
    elif stage == "test":
        execute([sys.executable, "-m", "pytest", "tests", "-q"], logs / "06_test.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["prepare"], required=True,
                        help="prepare runs check, split and bounded benchmark only")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    selected = ["check", "split", "benchmark"] if args.stage == "prepare" else [args.stage]
    for stage in selected:
        print(f"\n=== FULL DATA: {stage.upper()} ===", flush=True)
        run_stage(stage, args.threads)


if __name__ == "__main__":
    main()
