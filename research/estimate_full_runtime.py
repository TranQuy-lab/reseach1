"""Build a conservative, step-based launch gate from bounded GPU benchmarks.

The gate owns the locked full-data budget: per-dataset optimizer steps derived
from complete passes over each training split, a planned number of full-graph
validations, and the host resource margins that decide whether the 72 runs may
launch. `--passes`/`--min-train-steps` keep every dataset comparable by data
exposure instead of sharing one global step count that meant ~49 passes for
NF-UNSW-NB15-v2 and ~3 for NF-BoT-IoT-v2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # direct `python research/estimate_full_runtime.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nids_minibatch.budget import (
    DEFAULT_BATCH_SIZE, DEFAULT_EVAL_EVERY_STEPS, DEFAULT_MIN_TRAIN_STEPS,
    DEFAULT_PASSES, build_training_budget, validation_count,
)

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_EVAL_EVERY_STEPS",
           "DEFAULT_MIN_TRAIN_STEPS", "DEFAULT_PASSES", "build_training_budget",
           "main", "validation_count"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                        help="Complete passes over each train split that fund a run")
    parser.add_argument("--min-train-steps", type=int, default=DEFAULT_MIN_TRAIN_STEPS,
                        help="Step floor so small datasets still train")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-every-steps", type=int, default=DEFAULT_EVAL_EVERY_STEPS,
                        help="Fixed validation interval when --validations is 0")
    parser.add_argument("--validations", type=int, default=0,
                        help="Target full validations per run; 0 keeps the fixed interval")
    parser.add_argument("--tasks", type=int, default=2, help="Tasks per dataset/model/seed")
    parser.add_argument("--models", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--safety-factor", type=float, default=1.35)
    parser.add_argument("--minimum-windows", type=int, default=3)
    parser.add_argument("--vram-gib", type=float, default=24.0,
                        help="Installed GPU memory of the target host")
    parser.add_argument("--ram-gib", type=float, default=40.0,
                        help="Installed system memory of the target host")
    parser.add_argument("--vram-margin-gib", type=float, default=4.0)
    parser.add_argument("--ram-margin-gib", type=float, default=6.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=12.0)
    parser.add_argument("--max-planning-days", type=float, default=14.0)
    args = parser.parse_args()
    if min(args.passes, args.min_train_steps, args.batch_size, args.eval_every_steps,
           args.minimum_windows, args.tasks, args.models, args.seeds) < 1:
        parser.error("budgets, batch size, windows and run counts must be positive")
    if min(args.vram_gib, args.ram_gib, args.max_planning_days) <= 0:
        parser.error("host memory and the planning horizon must be positive")

    prepare = json.loads(args.prepare.read_text())
    environment = json.loads(args.environment.read_text())
    split_rows = {
        dataset: int(info["split_rows"]["train"]["rows"])
        for dataset, info in prepare["datasets"].items()
    }
    training_budget = build_training_budget(
        split_rows, passes=args.passes, min_train_steps=args.min_train_steps,
        batch_size=args.batch_size, eval_every_steps=args.eval_every_steps,
        target_validations=args.validations,
    )
    per_dataset = training_budget["per_dataset"]

    run_files = sorted(args.benchmarks.glob("*/runs.csv"))
    frames = [pd.read_csv(path) for path in run_files]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected = len(per_dataset) * args.models
    reasons: list[str] = []
    if len(table) != expected:
        reasons.append(f"benchmark incomplete: expected {expected} rows, found {len(table)}")

    details = []
    if len(table) == expected:
        for row in table.itertuples(index=False):
            budget = per_dataset.get(row.dataset)
            if budget is None:
                reasons.append(f"benchmark dataset {row.dataset} is absent from the split")
                continue
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
            median_batch = float(np.median(windows))
            p90_batch = float(np.quantile(windows, 0.9))
            details.append({
                "dataset": row.dataset,
                "model": row.model,
                "train_rows": budget["train_rows"],
                "steps_per_full_pass": budget["steps_per_pass"],
                "max_train_steps": budget["max_train_steps"],
                "passes_effective": budget["passes_effective"],
                "eval_every_steps": budget["eval_every_steps"],
                "validation_runs_planned": budget["planned_validations"],
                "benchmark_edges": int(first["train_edges"]),
                "benchmark_batches": int(first["train_batches"]),
                "post_warmup_windows": len(windows),
                "median_seconds_per_batch": median_batch,
                "p90_seconds_per_batch": p90_batch,
                "mad_seconds_per_batch": float(
                    np.median(np.abs(np.asarray(windows) - median_batch))
                ),
                "full_validation_seconds": first["validation_seconds"],
                "final_evaluation_seconds": row.seconds_final_evaluation,
                "peak_rss_gib": row.peak_rss_kib_process / 1024**2,
                "peak_cuda_gib": row.peak_cuda_bytes / 1024**3,
            })

    combos = args.tasks * args.models * args.seeds
    raw_seconds = 0.0
    for item in details:
        per_run = (
            item["max_train_steps"] * item["p90_seconds_per_batch"]
            + item["validation_runs_planned"] * item["full_validation_seconds"]
            + item["final_evaluation_seconds"]
        )
        raw_seconds += combos * per_run
    if len(table):
        for _, group in table.groupby("dataset"):
            raw_seconds += float(group.seconds_dataset_load.max())
            raw_seconds += args.tasks * float(group.seconds_graph_prepare.max())

    total_optimizer_steps = combos * sum(
        entry["max_train_steps"] for entry in per_dataset.values()
    )
    estimate = {
        "runs": combos * len(per_dataset),
        "optimizer_steps": total_optimizer_steps,
        "raw_hours": raw_seconds / 3600,
        "planning_hours": raw_seconds * args.safety_factor / 3600,
        "planning_days": raw_seconds * args.safety_factor / 86400,
    }
    resource_limits = {
        "vram_gib": args.vram_gib,
        "ram_gib": args.ram_gib,
        "vram_budget_gib": args.vram_gib - args.vram_margin_gib,
        "ram_budget_gib": args.ram_gib - args.ram_margin_gib,
        "min_free_disk_gib": args.min_free_disk_gib,
        "max_planning_days": args.max_planning_days,
    }
    if details:
        max_vram = max(x["peak_cuda_gib"] for x in details)
        max_rss = max(x["peak_rss_gib"] for x in details)
        if max_vram > resource_limits["vram_budget_gib"]:
            reasons.append(
                f"CUDA peak {max_vram:.1f} GiB leaves less than "
                f"{args.vram_margin_gib:.1f} GiB margin on {args.vram_gib:.0f} GiB"
            )
        if max_rss > resource_limits["ram_budget_gib"]:
            reasons.append(
                f"RAM peak {max_rss:.1f} GiB leaves less than "
                f"{args.ram_margin_gib:.1f} GiB margin on {args.ram_gib:.0f} GiB"
            )
    free_disk = float(environment.get("free_disk_gib", 0))
    if free_disk < args.min_free_disk_gib:
        reasons.append(
            f"only {free_disk:.1f} GiB disk free after split/benchmark"
        )
    if estimate["planning_days"] > args.max_planning_days:
        reasons.append(
            f"step-budget planning estimate exceeds {args.max_planning_days:.0f}-day limit"
        )

    result = {
        "safe_to_launch_72": not reasons,
        "reasons": reasons,
        "training_budget": training_budget,
        "resource_limits": resource_limits,
        "benchmark_rows": len(table),
        "step_budget_estimate": estimate,
        "details": details,
        "policy": "Do not launch the 72 runs unless safe_to_launch_72 is true.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if reasons:
        raise SystemExit("full-data launch gate failed")


if __name__ == "__main__":
    main()
