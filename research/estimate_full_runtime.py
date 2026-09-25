"""Build a conservative, step-based launch gate from bounded GPU benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def validation_count(max_steps: int, first_pass_steps: int,
                     every_steps: int) -> int:
    """Count first-pass, interval, and final-budget validations conservatively."""
    if min(max_steps, first_pass_steps, every_steps) < 1:
        raise ValueError("step counts must be positive")
    if max_steps < first_pass_steps:
        raise ValueError("budget must include at least one complete data pass")
    return 1 + math.ceil((max_steps - first_pass_steps) / every_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=20_000)
    parser.add_argument("--eval-every-steps", type=int, default=1_000)
    parser.add_argument("--safety-factor", type=float, default=1.35)
    parser.add_argument("--minimum-windows", type=int, default=3)
    args = parser.parse_args()
    if min(args.max_train_steps, args.eval_every_steps, args.minimum_windows) < 1:
        parser.error("step budgets and minimum-windows must be positive")

    prepare = json.loads(args.prepare.read_text())
    environment = json.loads(args.environment.read_text())
    run_files = sorted(args.benchmarks.glob("*/runs.csv"))
    frames = [pd.read_csv(path) for path in run_files]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected = 4 * 3
    reasons: list[str] = []
    if len(table) != expected:
        reasons.append(f"benchmark incomplete: expected {expected} rows, found {len(table)}")

    details = []
    if len(table) == expected:
        for row in table.itertuples(index=False):
            run_dir = args.benchmarks / row.dataset / (
                f"{row.dataset}__multiclass__{row.model}__seed11"
            )
            history = json.loads((run_dir / "history.json").read_text())
            first = history[0]
            profile = first.get("batch_timing") or {}
            windows = profile.get("seconds_per_batch_windows", [])
            if len(windows) < args.minimum_windows:
                reasons.append(
                    f"insufficient post-warmup windows: {row.dataset}/{row.model} "
                    f"({len(windows)} < {args.minimum_windows})"
                )
                continue
            train_rows = prepare["datasets"][row.dataset]["split_rows"]["train"]["rows"]
            first_pass_steps = math.ceil(train_rows / 4096)
            if args.max_train_steps < first_pass_steps:
                reasons.append(
                    f"step budget below one full pass: {row.dataset} needs {first_pass_steps}"
                )
                continue
            median_batch = float(np.median(windows))
            p90_batch = float(np.quantile(windows, 0.9))
            details.append({
                "dataset": row.dataset,
                "model": row.model,
                "train_rows": train_rows,
                "steps_per_full_pass": first_pass_steps,
                "benchmark_edges": int(first["train_edges"]),
                "benchmark_batches": int(first["train_batches"]),
                "post_warmup_windows": len(windows),
                "median_seconds_per_batch": median_batch,
                "p90_seconds_per_batch": p90_batch,
                "mad_seconds_per_batch": float(
                    np.median(np.abs(np.asarray(windows) - median_batch))
                ),
                "validation_runs_planned": validation_count(
                    args.max_train_steps, first_pass_steps, args.eval_every_steps
                ),
                "full_validation_seconds": first["validation_seconds"],
                "final_evaluation_seconds": row.seconds_final_evaluation,
                "peak_rss_gib": row.peak_rss_kib_process / 1024**2,
                "peak_cuda_gib": row.peak_cuda_bytes / 1024**3,
            })

    raw_seconds = 0.0
    for item in details:
        per_run = (
            args.max_train_steps * item["p90_seconds_per_batch"]
            + item["validation_runs_planned"] * item["full_validation_seconds"]
            + item["final_evaluation_seconds"]
        )
        # Two tasks and three seeds for every dataset/model pair.
        raw_seconds += 6 * per_run
    if len(table):
        for _, group in table.groupby("dataset"):
            raw_seconds += float(group.seconds_dataset_load.max())
            raw_seconds += 2 * float(group.seconds_graph_prepare.max())

    estimate = {
        "raw_hours": raw_seconds / 3600,
        "planning_hours": raw_seconds * args.safety_factor / 3600,
        "planning_days": raw_seconds * args.safety_factor / 86400,
    }
    if details:
        max_vram = max(x["peak_cuda_gib"] for x in details)
        max_rss = max(x["peak_rss_gib"] for x in details)
        if max_vram >= 29:
            reasons.append(f"CUDA peak {max_vram:.1f} GiB leaves insufficient margin on 32 GiB")
        if max_rss >= 110:
            reasons.append(f"RAM peak {max_rss:.1f} GiB leaves insufficient margin on 125 GiB")
    free_disk = float(environment.get("free_disk_gib", 0))
    if free_disk < 12:
        reasons.append(f"only {free_disk:.1f} GiB disk free after split/benchmark")
    if estimate["planning_days"] > 14:
        reasons.append("step-budget planning estimate exceeds 14-day launch limit")

    result = {
        "safe_to_launch_72": not reasons,
        "reasons": reasons,
        "training_budget": {
            "max_train_steps": args.max_train_steps,
            "eval_every_steps": args.eval_every_steps,
        },
        "safety_factor": args.safety_factor,
        "minimum_post_warmup_windows": args.minimum_windows,
        "benchmark_rows": len(table),
        "step_budget_estimate": estimate,
        "details": details,
        "policy": "Do not launch 72 runs unless safe_to_launch_72 is true.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if reasons:
        raise SystemExit("full-data launch gate failed")


if __name__ == "__main__":
    main()
