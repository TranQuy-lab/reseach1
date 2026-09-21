"""Convert bounded full-data benchmark artifacts into a conservative launch gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def evaluations(epochs: int, every: int = 3) -> int:
    return len({1, epochs, *range(every, epochs + 1, every)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-factor", type=float, default=1.35)
    args = parser.parse_args()

    prepare = json.loads(args.prepare.read_text())
    environment = json.loads(args.environment.read_text())
    run_files = sorted(args.benchmarks.glob("*/runs.csv"))
    frames = [pd.read_csv(path) for path in run_files]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected = 4 * 3
    reasons = []
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
            train_rows = prepare["datasets"][row.dataset]["split_rows"]["train"]["rows"]
            if not first["train_edges"] or not first["train_seconds"]:
                reasons.append(f"invalid throughput sample: {row.dataset}/{row.model}")
                continue
            epoch_train = first["train_seconds"] * train_rows / first["train_edges"]
            details.append({
                "dataset": row.dataset,
                "model": row.model,
                "train_rows": train_rows,
                "benchmark_edges": int(first["train_edges"]),
                "benchmark_train_seconds": first["train_seconds"],
                "estimated_full_epoch_train_seconds": epoch_train,
                "full_validation_seconds": first["validation_seconds"],
                "final_evaluation_seconds": row.seconds_final_evaluation,
                "peak_rss_gib": row.peak_rss_kib_process / 1024**2,
                "peak_cuda_gib": row.peak_cuda_bytes / 1024**3,
            })

    scenarios = {}
    for epochs in (10, 30, 60):
        seconds = 0.0
        for item in details:
            per_run = (
                epochs * item["estimated_full_epoch_train_seconds"]
                + evaluations(epochs) * item["full_validation_seconds"]
                + item["final_evaluation_seconds"]
            )
            # Two tasks and three seeds for every dataset/model pair.
            seconds += 6 * per_run
        # Frames/graphs are prepared once per task; use the conservative maximum
        # observed setup time for every dataset and both tasks.
        if len(table):
            for dataset, group in table.groupby("dataset"):
                seconds += float(group.seconds_dataset_load.max())
                seconds += 2 * float(group.seconds_graph_prepare.max())
        scenarios[str(epochs)] = {
            "raw_hours": seconds / 3600,
            "planning_hours": seconds * args.safety_factor / 3600,
            "planning_days": seconds * args.safety_factor / 86400,
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
    if scenarios.get("60", {}).get("planning_days", math.inf) > 14:
        reasons.append("60-epoch planning estimate exceeds 14-day launch limit")

    result = {
        "safe_to_launch_72": not reasons,
        "reasons": reasons,
        "safety_factor": args.safety_factor,
        "benchmark_rows": len(table),
        "scenario_estimates": scenarios,
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
