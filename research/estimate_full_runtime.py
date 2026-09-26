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


def dataset_budget(train_rows: int, batch_size: int, train_passes: int,
                   min_train_steps: int, evals_per_pass: int,
                   minimum_eval_interval_steps: int = 500) -> dict:
    if min(train_rows, batch_size, train_passes, min_train_steps,
           evals_per_pass, minimum_eval_interval_steps) < 1:
        raise ValueError("dataset budget inputs must be positive")
    steps_per_pass = math.ceil(train_rows / batch_size)
    max_steps = max(min_train_steps, train_passes * steps_per_pass)
    eval_every_steps = max(
        minimum_eval_interval_steps, math.ceil(steps_per_pass / evals_per_pass)
    )
    return {
        "steps_per_full_pass": steps_per_pass,
        "max_train_steps": max_steps,
        "eval_every_steps": eval_every_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-passes", type=int, default=2)
    parser.add_argument("--min-train-steps", type=int, default=1_500)
    parser.add_argument("--evals-per-pass", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--budget-usd", type=float, default=6.0)
    parser.add_argument("--hourly-price-usd", type=float, default=0.0,
                        help="0 reports break-even price without enforcing a cost gate")
    parser.add_argument("--safety-factor", type=float, default=1.35)
    parser.add_argument("--minimum-windows", type=int, default=3)
    args = parser.parse_args()
    if min(args.train_passes, args.min_train_steps, args.evals_per_pass,
           args.minimum_windows) < 1 or min(args.num_workers, args.budget_usd,
                                             args.hourly_price_usd) < 0:
        parser.error("budget values must be positive and num-workers nonnegative")

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
            budget = dataset_budget(
                train_rows, 4096, args.train_passes, args.min_train_steps,
                args.evals_per_pass,
            )
            first_pass_steps = budget["steps_per_full_pass"]
            median_batch = float(np.median(windows))
            p90_batch = float(np.quantile(windows, 0.9))
            details.append({
                "dataset": row.dataset,
                "model": row.model,
                "train_rows": train_rows,
                "steps_per_full_pass": first_pass_steps,
                "max_train_steps": budget["max_train_steps"],
                "eval_every_steps": budget["eval_every_steps"],
                "benchmark_edges": int(first["train_edges"]),
                "benchmark_batches": int(first["train_batches"]),
                "post_warmup_windows": len(windows),
                "median_seconds_per_batch": median_batch,
                "p90_seconds_per_batch": p90_batch,
                "mad_seconds_per_batch": float(
                    np.median(np.abs(np.asarray(windows) - median_batch))
                ),
                "validation_runs_planned": validation_count(
                    budget["max_train_steps"], first_pass_steps,
                    budget["eval_every_steps"],
                ),
                "full_validation_seconds": first["validation_seconds"],
                "final_evaluation_seconds": row.seconds_final_evaluation,
                "independent_verification_seconds_estimate": row.seconds_final_evaluation,
                "peak_rss_gib": row.peak_rss_kib_process / 1024**2,
                "peak_cuda_gib": row.peak_cuda_bytes / 1024**3,
            })

    raw_seconds = 0.0
    for item in details:
        per_run = (
            item["max_train_steps"] * item["p90_seconds_per_batch"]
            + item["validation_runs_planned"] * item["full_validation_seconds"]
            + item["final_evaluation_seconds"]
            + item["independent_verification_seconds_estimate"]
        )
        # Two tasks and three seeds for every dataset/model pair.
        raw_seconds += 6 * per_run
    if len(table):
        for _, group in table.groupby("dataset"):
            raw_seconds += float(group.seconds_dataset_load.max())
            raw_seconds += float(group.seconds_graph_prepare.max())

    estimate = {
        "raw_hours": raw_seconds / 3600,
        "planning_hours": raw_seconds * args.safety_factor / 3600,
        "planning_days": raw_seconds * args.safety_factor / 86400,
    }
    planning_cost = (estimate["planning_hours"] * args.hourly_price_usd
                     if args.hourly_price_usd else None)
    cost_estimate = {
        "budget_usd": args.budget_usd,
        "hourly_price_usd": args.hourly_price_usd or None,
        "planning_cost_usd": planning_cost,
        "max_hourly_price_for_budget_usd": (
            args.budget_usd / estimate["planning_hours"]
            if estimate["planning_hours"] > 0 else None
        ),
        "cost_gate_evaluated": bool(args.hourly_price_usd),
    }
    total_optimizer_steps = sum(item["max_train_steps"] * 6 for item in details)
    legacy_optimizer_steps = 72 * 20_000
    dataset_step_plan = {}
    for item in details:
        dataset_step_plan[item["dataset"]] = {
            "train_rows": item["train_rows"],
            "steps_per_full_pass": item["steps_per_full_pass"],
            "max_train_steps": item["max_train_steps"],
            "eval_every_steps": item["eval_every_steps"],
            "validation_runs_planned": item["validation_runs_planned"],
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
    if planning_cost is not None and planning_cost > args.budget_usd:
        reasons.append(
            f"planning cost ${planning_cost:.2f} exceeds ${args.budget_usd:.2f} budget"
        )

    result = {
        "safe_to_launch_72": not reasons,
        "reasons": reasons,
        "training_budget": {
            "train_passes": args.train_passes,
            "min_train_steps": args.min_train_steps,
            "evals_per_pass": args.evals_per_pass,
            "minimum_eval_interval_steps": 500,
        },
        "execution": {"num_workers": args.num_workers, "batch_size": 4096},
        "safety_factor": args.safety_factor,
        "minimum_post_warmup_windows": args.minimum_windows,
        "benchmark_rows": len(table),
        "step_budget_estimate": estimate,
        "cost_estimate": cost_estimate,
        "optimizer_step_plan": {
            "datasets": dataset_step_plan,
            "total_for_72_runs": total_optimizer_steps,
            "legacy_fixed_20000_total": legacy_optimizer_steps,
            "fraction_reduced": (1 - total_optimizer_steps / legacy_optimizer_steps
                                 if details else None),
        },
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
