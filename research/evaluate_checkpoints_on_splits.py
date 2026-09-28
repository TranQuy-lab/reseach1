"""Evaluate locked checkpoints on an alternate graph without retraining.

This implements the RW and WR arms in the paper-extension protocol. The
persisted training preprocessor and checkpoint are reused verbatim; only the
evaluation graph is replaced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nids_minibatch.data import Preprocessor, make_graph
from nids_minibatch.models import build_model
from nids_minibatch.schema import FEATURES
from nids_minibatch.training import (
    EVALUATION_MODE,
    full_probabilities,
    load_frames,
    metrics,
    resolve_device,
    save_predictions,
    sha256_file,
    write_json,
)


def negative_control_error(run_dir: Path, graph, probability: np.ndarray,
                           classes: list[str]) -> float:
    """Compare EdgeMLP probabilities by source-row ID across graph variants."""
    stored = pd.read_parquet(run_dir / "test_predictions.parquet")
    positions = pd.Series(
        np.arange(len(graph.source_row_id), dtype=np.int64),
        index=pd.Index(graph.source_row_id),
    )
    offsets = positions.reindex(stored.source_row_id).to_numpy()
    if pd.isna(offsets).any():
        raise AssertionError("Negative-control rows are missing from alternate split")
    offsets = offsets.astype(np.int64)
    baseline = stored[[f"p_{name}" for name in classes]].to_numpy()
    return float(np.max(np.abs(probability[offsets] - baseline)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Alternate split root used only for evaluation")
    parser.add_argument("--runs", type=Path, required=True,
                        help="Source checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["RW", "WR"], required=True)
    parser.add_argument("--models", nargs="+", choices=["edge_mlp", "sage", "sage_edge"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--prediction-cap", type=int, default=100_000)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.prediction_cap < 0:
        parser.error("threads must be positive and prediction-cap nonnegative")
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output exists; refusing overwrite: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(args.threads)
    # Match the locked training/verification policy: report nondeterministic
    # kernels without aborting an otherwise replayable checkpoint evaluation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(args.device)

    source_provenance = json.loads((args.runs / "provenance.json").read_text())
    table = pd.read_csv(args.runs / "runs.csv")
    selected_models = args.models or list(source_provenance["models"])
    table = table[table.model.isin(selected_models)].copy()
    expected = (
        len(source_provenance["datasets"]) * len(source_provenance["tasks"])
        * len(selected_models) * len(source_provenance["seeds"])
    )
    if len(table) != expected or table[["dataset", "task", "model", "seed"]].duplicated().any():
        raise ValueError(f"Expected {expected} source checkpoints, found {len(table)}")
    storage_dtype = (
        torch.float16 if source_provenance.get("edge_storage_dtype") == "torch.float16"
        else torch.float32
    )
    existing = args.output / "runs.csv"
    rows = pd.read_csv(existing).to_dict("records") if args.resume and existing.is_file() else []
    completed = {
        (row["dataset"], row["task"], row["model"], int(row["seed"]))
        for row in rows
    }
    largest_negative_control_error = 0.0

    for dataset in source_provenance["datasets"]:
        frames = load_frames(args.data, dataset, "full")
        for task in source_provenance["tasks"]:
            task_rows = table[(table.dataset == dataset) & (table.task == task)]
            first = task_rows.iloc[0]
            first_dir = args.runs / (
                f"{dataset}__{task}__{first.model}__seed{int(first.seed)}"
            )
            pre = Preprocessor.from_dict(json.loads(
                (first_dir / "preprocessor.json").read_text()
            ))
            graphs = {
                split: make_graph(
                    frames[split], pre.transform(frames[split]),
                    pre.labels(frames[split]), storage_dtype,
                )
                for split in ("val", "test")
            }
            for source_row in task_rows.itertuples(index=False):
                key = (dataset, task, source_row.model, int(source_row.seed))
                if key in completed:
                    print(f"SKIP {key}", flush=True)
                    continue
                run_id = f"{dataset}__{task}__{source_row.model}__seed{int(source_row.seed)}"
                source_dir = args.runs / run_id
                stored_pre = Preprocessor.from_dict(json.loads(
                    (source_dir / "preprocessor.json").read_text()
                ))
                if not stored_pre.equivalent(pre):
                    raise AssertionError(f"Preprocessor mismatch: {run_id}")
                config = json.loads((source_dir / "config.json").read_text())
                model = build_model(
                    source_row.model, len(FEATURES), len(pre.classes),
                    config["hidden"], config["dropout"],
                ).to(device)
                checkpoint = source_dir / "model.pt"
                checkpoint_hash = sha256_file(checkpoint)
                model.load_state_dict(torch.load(
                    checkpoint, map_location="cpu", weights_only=True,
                ))
                started = time.perf_counter()
                val_probability = full_probabilities(
                    model, source_row.model, graphs["val"], False,
                )
                test_probability = full_probabilities(
                    model, source_row.model, graphs["test"], False,
                )
                elapsed = time.perf_counter() - started
                if sha256_file(checkpoint) != checkpoint_hash:
                    raise AssertionError(f"Checkpoint changed during evaluation: {run_id}")
                control_error = None
                if source_row.model == "edge_mlp" and args.condition == "RW":
                    control_error = negative_control_error(
                        source_dir, graphs["test"], test_probability, pre.classes,
                    )
                    largest_negative_control_error = max(
                        largest_negative_control_error, control_error,
                    )
                    if control_error > 1e-7:
                        raise AssertionError(
                            f"EdgeMLP negative control changed for {run_id}: {control_error}"
                        )
                dest = args.output / run_id
                if dest.exists():
                    raise RuntimeError(f"Partial evaluation refuses overwrite: {dest}")
                dest.mkdir()
                prediction_rows = save_predictions(
                    dest / "test_predictions.parquet", graphs["test"],
                    test_probability, pre.classes, args.prediction_cap or None,
                )
                result = {
                    "dataset": dataset, "task": task, "model": source_row.model,
                    "seed": int(source_row.seed), "condition": args.condition,
                    "source_checkpoint": checkpoint.as_posix(),
                    "source_checkpoint_sha256": checkpoint_hash,
                    "evaluation_mode": EVALUATION_MODE,
                    "seconds_evaluation": elapsed,
                    "negative_control_max_abs_error": control_error,
                    "test_prediction_rows_stored": prediction_rows,
                    "test_prediction_rows_total": len(graphs["test"].labels),
                    "val": metrics(
                        graphs["val"].labels.numpy(), val_probability, pre.classes,
                    ),
                    "test": metrics(
                        graphs["test"].labels.numpy(), test_probability, pre.classes,
                    ),
                }
                write_json(dest / "metrics.json", result)
                row = {
                    "dataset": dataset, "task": task, "model": source_row.model,
                    "seed": int(source_row.seed), "condition": args.condition,
                    "seconds_evaluation": elapsed,
                    "negative_control_max_abs_error": control_error,
                    **{
                        f"{split}_{metric}": result[split][metric]
                        for split in ("val", "test")
                        for metric in ("macro_f1", "weighted_f1", "accuracy")
                    },
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(existing, index=False)
                print(
                    f"DONE {args.condition} {run_id} "
                    f"test_macro_f1={result['test']['macro_f1']:.6f}",
                    flush=True,
                )
                del model, val_probability, test_probability
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del graphs
        del frames

    provenance = {
        "condition": args.condition,
        "data_root": args.data.as_posix(),
        "source_runs": args.runs.as_posix(),
        "source_runs_provenance_sha256": sha256_file(args.runs / "provenance.json"),
        "source_runs_table_sha256": sha256_file(args.runs / "runs.csv"),
        "data_manifest": args.data_manifest.as_posix() if args.data_manifest else None,
        "data_manifest_sha256": (
            sha256_file(args.data_manifest) if args.data_manifest else None
        ),
        "models": selected_models,
        "datasets": source_provenance["datasets"],
        "tasks": source_provenance["tasks"],
        "seeds": source_provenance["seeds"],
        "evaluation_mode": EVALUATION_MODE,
        "largest_negative_control_error": largest_negative_control_error,
        "complete": len(rows) == expected,
    }
    write_json(args.output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
