"""Restartable, benchmark-gated full-data server pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nids_minibatch.training import experiment_source_sha256, sha256_file


DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
MODELS = ["edge_mlp", "sage", "sage_edge"]
STAGES = ["check", "split", "benchmark", "train", "verify", "report", "test"]
DEFAULT_TRAIN_PASSES = 2
DEFAULT_MIN_TRAIN_STEPS = 1_500
DEFAULT_EVALS_PER_PASS = 4
DEFAULT_NUM_WORKERS = 4


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
                  train_passes: int = 0, min_train_steps: int = 0,
                  evals_per_pass: int = DEFAULT_EVALS_PER_PASS,
                  num_workers: int = DEFAULT_NUM_WORKERS) -> list[str]:
    command = [
        sys.executable, "-m", "nids_minibatch.training", "--data", "data/full_splits",
        "--output", output, "--datasets", *datasets, "--tasks", *tasks,
        "--models", *models, "--seeds", *map(str, seeds), "--epochs", str(epochs),
        "--patience", str(patience), "--batch-size", "4096", "--fanout", "15", "10",
        "--threads", str(threads), "--device", "cuda", "--scope", "full",
        "--eval-every", "3", "--amp", "--prediction-cap", str(prediction_cap),
        "--max-train-batches", str(max_batches), "--num-workers", str(num_workers),
    ]
    if train_passes:
        command.extend(["--train-passes", str(train_passes),
                        "--min-train-steps", str(min_train_steps),
                        "--evals-per-pass", str(evals_per_pass)])
    return command


def require_launch_gate(num_workers: int = DEFAULT_NUM_WORKERS) -> dict:
    path = ROOT / "research/results/full_benchmark_estimate.json"
    if not path.is_file():
        raise RuntimeError("Run the bounded benchmark before full training")
    value = json.loads(path.read_text())
    if value.get("safe_to_launch_72") is not True:
        raise RuntimeError("Full-data launch gate failed; inspect full_benchmark_estimate.json")
    budget = value.get("training_budget", {})
    if budget != {
        "train_passes": DEFAULT_TRAIN_PASSES,
        "min_train_steps": DEFAULT_MIN_TRAIN_STEPS,
        "evals_per_pass": DEFAULT_EVALS_PER_PASS,
        "minimum_eval_interval_steps": 500,
    }:
        raise RuntimeError("Launch gate training budget differs from the locked protocol")
    if value.get("execution") != {"num_workers": num_workers, "batch_size": 4096}:
        raise RuntimeError("Launch gate execution settings differ from the train command")
    return value


def benchmark_complete(output: Path, dataset: str, num_workers: int) -> bool:
    runs_file = output / "runs.csv"
    provenance_file = output / "provenance.json"
    if not runs_file.is_file() or not provenance_file.is_file():
        return False
    import pandas as pd
    table = pd.read_csv(runs_file)
    provenance = json.loads(provenance_file.read_text())
    return (
        len(table) == 3
        and set(table.model) == set(MODELS)
        and set(table.dataset) == {dataset}
        and provenance.get("max_train_batches") == 500
        and provenance.get("num_workers") == num_workers
        and provenance.get("models") == MODELS
        and provenance.get("source_sha256") == experiment_source_sha256()
        and provenance.get("protocol_sha256") == sha256_file(
            ROOT / "research/PROTOCOL_FULL_DATA_VI.md"
        )
    )


def run_stage(stage: str, threads: int, num_workers: int,
              budget_usd: float, hourly_price_usd: float) -> None:
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
            if benchmark_complete(output, dataset, num_workers):
                continue
            if output.exists() and (output / "provenance.json").is_file():
                provenance = json.loads((output / "provenance.json").read_text())
                if (provenance.get("source_sha256") != experiment_source_sha256()
                        or provenance.get("num_workers") != num_workers):
                    raise RuntimeError(
                        f"Stale benchmark at {output}; archive that directory before rerunning"
                    )
            command = train_command(str(output.relative_to(ROOT)), [dataset], MODELS, [11],
                                    ["multiclass"], 1, 1, threads, 500, 1_000,
                                    num_workers=num_workers)
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
                 "research/results/full_benchmark_estimate.json", "--train-passes",
                 str(DEFAULT_TRAIN_PASSES), "--min-train-steps",
                 str(DEFAULT_MIN_TRAIN_STEPS), "--evals-per-pass",
                 str(DEFAULT_EVALS_PER_PASS), "--num-workers", str(num_workers),
                 "--budget-usd", str(budget_usd), "--hourly-price-usd",
                 str(hourly_price_usd)],
                logs / "02_estimate.log")
    elif stage == "train":
        gate = require_launch_gate(num_workers)
        budget = gate["training_budget"]
        output = ROOT / "research/artifacts/full_runs"
        command = train_command(str(output.relative_to(ROOT)), DATASETS, MODELS,
                                [11, 22, 33], ["multiclass", "binary"], 1, 10, threads,
                                train_passes=budget["train_passes"],
                                min_train_steps=budget["min_train_steps"],
                                evals_per_pass=budget["evals_per_pass"],
                                num_workers=num_workers)
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
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--budget-usd", type=float, default=6.0)
    parser.add_argument("--hourly-price-usd", type=float, default=0.0)
    args = parser.parse_args()
    if (args.threads < 1 or args.num_workers < 0 or args.budget_usd < 0
            or args.hourly_price_usd < 0):
        parser.error("invalid threads, workers, budget or hourly price")
    selected = ["check", "split", "benchmark"] if args.stage == "prepare" else [args.stage]
    for stage in selected:
        print(f"\n=== FULL DATA: {stage.upper()} ===", flush=True)
        run_stage(stage, args.threads, args.num_workers,
                  args.budget_usd, args.hourly_price_usd)


if __name__ == "__main__":
    main()
