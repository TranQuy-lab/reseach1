"""Run or verify only endpoint-holdout dataset/task cells that pass the gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path("research/PROTOCOL_PAPER_EXTENSION_VI.md")
SEEDS = (11, 22, 33, 44, 55)
MODELS = ("edge_mlp", "sage", "sage_edge")


def execute(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "verify"], required=True)
    parser.add_argument("--prepare", type=Path,
                        default=Path("research/results/endpoint_holdout_prepare.json"))
    parser.add_argument("--data", type=Path, default=Path("data/endpoint_holdout_splits"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/artifacts/endpoint_holdout_runs"))
    parser.add_argument("--verification-output", type=Path,
                        default=Path("research/results/endpoint_holdout_verification"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prediction-cap", type=int, default=100_000)
    args = parser.parse_args()
    if args.threads < 1 or args.num_workers < 0 or args.prediction_cap < 0:
        parser.error("invalid execution settings")
    prepare = json.loads(args.prepare.read_text())
    if prepare.get("complete") is not True:
        raise RuntimeError("Endpoint prepare gate is incomplete")
    eligible = prepare.get("eligible_dataset_tasks", [])
    if not eligible:
        raise RuntimeError("No dataset/task passed the endpoint class-support gate")
    args.output.mkdir(parents=True, exist_ok=True)
    args.verification_output.mkdir(parents=True, exist_ok=True)

    completed = []
    for cell in eligible:
        dataset, task = cell["dataset"], cell["task"]
        cell_name = f"{dataset}__{task}"
        run_root = args.output / cell_name
        if args.mode == "train":
            command = [
                sys.executable, "-u", "-m", "nids_minibatch.training",
                "--data", str(args.data), "--output", str(run_root),
                "--datasets", dataset, "--tasks", task,
                "--models", *MODELS, "--seeds", *map(str, SEEDS),
                "--epochs", "1", "--patience", "10", "--batch-size", "4096",
                "--fanout", "15", "10", "--threads", str(args.threads),
                "--device", "cuda", "--scope", "full", "--eval-every", "3",
                "--amp", "--prediction-cap", str(args.prediction_cap),
                "--max-train-batches", "0", "--num-workers", str(args.num_workers),
                "--train-passes", "2", "--min-train-steps", "1500",
                "--evals-per-pass", "4", "--protocol", str(PROTOCOL),
            ]
            if run_root.exists():
                command.append("--resume")
        else:
            command = [
                sys.executable, "-u", "research/validate_minibatch_results.py",
                "--data", str(args.data), "--runs", str(run_root),
                "--output", str(args.verification_output / f"{cell_name}.json"),
                "--threads", str(args.threads),
            ]
        execute(command)
        completed.append(cell)

    manifest = {
        "mode": args.mode,
        "prepare_manifest": args.prepare.as_posix(),
        "eligible_dataset_tasks": eligible,
        "completed_dataset_tasks": completed,
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "complete": completed == eligible,
    }
    destination = (
        args.output / "matrix_manifest.json" if args.mode == "train"
        else args.verification_output / "matrix_manifest.json"
    )
    destination.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
