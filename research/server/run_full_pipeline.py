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
# Locked full-data budget: complete passes over each train split instead of one
# global step count. Fanout and batch size come from the tuning record, which
# selected batch 2048 with fanout [10, 5] on validation macro-F1.
DEFAULT_PASSES = 2
DEFAULT_MIN_TRAIN_STEPS = 1_500
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_FANOUT = (10, 5)
DEFAULT_NUM_WORKERS = 4
DEFAULT_EVAL_EVERY_STEPS = 1_000
DEFAULT_VALIDATIONS = 0
DEFAULT_TASKS = 2
DEFAULT_SEEDS = (11, 22, 33)
DEFAULT_VRAM_GIB = 24.0
DEFAULT_RAM_GIB = 40.0
DEFAULT_MIN_FREE_DISK_GIB = 12.0
GATE_PATH = ROOT / "research/results/full_benchmark_estimate.json"


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
                  budget_json: Path | None = None,
                  num_workers: int = DEFAULT_NUM_WORKERS,
                  fanout: tuple[int, int] = DEFAULT_FANOUT,
                  batch_size: int = DEFAULT_BATCH_SIZE) -> list[str]:
    """Build one training command; a budget document switches it to step mode."""
    command = [
        sys.executable, "-m", "nids_minibatch.training", "--data", "data/full_splits",
        "--output", output, "--datasets", *datasets, "--tasks", *tasks,
        "--models", *models, "--seeds", *map(str, seeds), "--epochs", str(epochs),
        "--patience", str(patience), "--batch-size", str(batch_size),
        "--fanout", *map(str, fanout), "--threads", str(threads),
        "--device", "cuda", "--scope", "full", "--eval-every", "1", "--amp",
        "--prediction-cap", str(prediction_cap),
        "--max-train-batches", str(max_batches),
        "--num-workers", str(num_workers),
    ]
    if budget_json is not None:
        command.extend(["--budget-json", str(budget_json)])
    return command


def require_launch_gate() -> dict:
    """Refuse to train unless the locked gate matches the current protocol."""
    if not GATE_PATH.is_file():
        raise RuntimeError("Run the bounded benchmark before full training")
    value = json.loads(GATE_PATH.read_text())
    if value.get("safe_to_launch_72") is not True:
        raise RuntimeError("Full-data launch gate failed; inspect full_benchmark_estimate.json")
    budget = value.get("training_budget", {})
    if budget.get("mode") != "passes" or budget.get("passes") != DEFAULT_PASSES \
            or budget.get("min_train_steps") != DEFAULT_MIN_TRAIN_STEPS \
            or budget.get("batch_size") != DEFAULT_BATCH_SIZE:
        raise RuntimeError("Launch gate training budget differs from the locked protocol")
    missing = [dataset for dataset in DATASETS if dataset not in budget.get("per_dataset", {})]
    if missing:
        raise RuntimeError(f"Launch gate budget lacks datasets: {missing}")
    return value


def run_stage(stage: str, args: argparse.Namespace) -> None:
    logs = ROOT / "research/artifacts/full_server_logs"
    datasets = list(args.datasets)
    if stage == "check":
        execute([sys.executable, "research/server/check_server.py", "--output",
                 "research/results/server_environment.json"], logs / "00_check.log")
    elif stage == "split":
        report = ROOT / "research/results/full_prepare.json"
        if report.is_file() and json.loads(report.read_text()).get("complete") is True:
            return
        execute([sys.executable, "-m", "nids_minibatch.prepare", "--source",
                 "data/processed_four", "--output", "data/full_splits", "--report",
                 str(report.relative_to(ROOT)), "--threads", str(args.threads)],
                logs / "01_split.log")
    elif stage == "benchmark":
        bench = ROOT / "research/artifacts/full_benchmark"
        failures = []
        for dataset in datasets:
            output = bench / dataset
            if (output / "runs.csv").is_file() and len(__import__("pandas").read_csv(output / "runs.csv")) == 3:
                continue
            command = train_command(str(output.relative_to(ROOT)), [dataset], MODELS, [11],
                                    ["multiclass"], 1, 1, args.threads, 500, 1_000,
                                    num_workers=args.num_workers)
            if output.exists():
                command.append("--resume")
            code = execute(command, logs / f"02_benchmark_{dataset}.log", check=False)
            if code:
                failures.append(dataset)
        if failures:
            raise RuntimeError(f"Benchmark failed for: {failures}")
        estimate_flags = [
            "--passes", str(args.passes),
            "--min-train-steps", str(args.min_train_steps),
            "--batch-size", str(DEFAULT_BATCH_SIZE),
            "--eval-every-steps", str(args.eval_every_steps),
            "--validations", str(args.validations),
            "--tasks", str(DEFAULT_TASKS),
            "--models", str(len(MODELS)),
            "--seeds", str(len(args.seeds)),
            "--vram-gib", str(args.vram_gib),
            "--ram-gib", str(args.ram_gib),
            "--min-free-disk-gib", str(args.min_free_disk_gib),
        ]
        execute([sys.executable, "research/estimate_full_runtime.py", "--benchmarks",
                 "research/artifacts/full_benchmark", "--prepare", "research/results/full_prepare.json",
                 "--environment", "research/results/server_environment.json", "--output",
                 str(GATE_PATH.relative_to(ROOT)), *estimate_flags], logs / "02_estimate.log")
    elif stage == "train":
        require_launch_gate()
        output = ROOT / "research/artifacts/full_runs"
        command = train_command(str(output.relative_to(ROOT)), datasets, MODELS,
                                list(args.seeds), ["multiclass", "binary"], 1, 10,
                                args.threads, budget_json=GATE_PATH,
                                num_workers=args.num_workers)
        if output.exists():
            command.append("--resume")
        execute(command, logs / "03_train.log")
    elif stage == "verify":
        execute([sys.executable, "research/validate_minibatch_results.py", "--data",
                 "data/full_splits", "--runs", "research/artifacts/full_runs", "--output",
                 "research/results/full_verification.json", "--threads", str(args.threads)],
                logs / "04_verify.log")
    elif stage == "report":
        execute([sys.executable, "research/build_full_report.py", "--runs",
                 "research/artifacts/full_runs", "--verification",
                 "research/results/full_verification.json", "--benchmark",
                 str(GATE_PATH.relative_to(ROOT)), "--output",
                 "research/results/full", "--report", "research/FULL_DATA_REPORT_VI.md"],
                logs / "05_report.log")
    elif stage == "test":
        execute([sys.executable, "-m", "pytest", "tests", "-q"], logs / "06_test.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["prepare"], required=True,
                        help="prepare runs check, split and bounded benchmark only")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help="Restrict the benchmark and train stages to these datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument("--min-train-steps", type=int, default=DEFAULT_MIN_TRAIN_STEPS)
    parser.add_argument("--eval-every-steps", type=int, default=DEFAULT_EVAL_EVERY_STEPS)
    parser.add_argument("--validations", type=int, default=DEFAULT_VALIDATIONS,
                        help="Target full validations per run; 0 keeps eval-every-steps")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--vram-gib", type=float, default=DEFAULT_VRAM_GIB)
    parser.add_argument("--ram-gib", type=float, default=DEFAULT_RAM_GIB)
    parser.add_argument("--min-free-disk-gib", type=float, default=DEFAULT_MIN_FREE_DISK_GIB)
    args = parser.parse_args()
    if min(args.threads, args.passes, args.min_train_steps, args.eval_every_steps,
           args.num_workers) < 1:
        parser.error("threads, budget, interval and workers must be positive")
    if args.validations < 0:
        parser.error("validations must be nonnegative")
    if not args.seeds or min(args.seeds) < 0:
        parser.error("seeds must be nonnegative")
    selected = ["check", "split", "benchmark"] if args.stage == "prepare" else [args.stage]
    for stage in selected:
        print(f"\n=== FULL DATA: {stage.upper()} ===", flush=True)
        run_stage(stage, args)


if __name__ == "__main__":
    main()
