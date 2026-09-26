"""Rebuild validation/test artifacts from existing full-data checkpoints.

This migration never trains or modifies model checkpoints.  It exists for
completed runs whose CUDA BF16 full-graph GNN evaluation was not reproducible.
The operation is resumable: each repaired run is marked in metrics.json and
runs.csv is updated atomically after every checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from nids_minibatch.data import Preprocessor
from nids_minibatch.models import build_model
from nids_minibatch.schema import FEATURES
from nids_minibatch.training import (
    EVALUATION_MODE,
    experiment_source_sha256,
    full_probabilities,
    load_frames,
    metrics,
    prepare_graphs,
    preprocessor_for_task,
    relabel_graphs,
    save_predictions,
    sha256_file,
    write_json,
)


ROW_FIELDS = (
    "dataset", "task", "model", "seed", "seconds_fit_and_evaluate",
    "seconds_fit", "seconds_final_evaluation", "seconds_dataset_load",
    "seconds_graph_prepare", "best_epoch", "epochs_ran", "parameters",
    "best_step", "steps_ran", "peak_rss_kib_process", "peak_cuda_bytes",
    "step_budget", "steps_per_full_pass", "eval_every_steps",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_row(result: dict) -> dict:
    row = {key: result[key] for key in ROW_FIELDS}
    row.update({
        f"{split}_{metric}": result[split][metric]
        for split in ("val", "test")
        for metric in ("macro_f1", "weighted_f1", "accuracy")
    })
    return row


def write_table(path: Path, rows: dict[tuple[str, str, str, int], dict]) -> None:
    frame = pd.DataFrame(rows.values())
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_summary(runs: Path, rows: dict[tuple[str, str, str, int], dict]) -> None:
    frame = pd.DataFrame(rows.values())
    summary = frame.groupby(["dataset", "task", "model"])[
        ["test_macro_f1", "test_weighted_f1", "test_accuracy",
         "seconds_fit_and_evaluate"]
    ].agg(["mean", "std"])
    temporary = runs / "summary.csv.tmp"
    summary.to_csv(temporary)
    temporary.replace(runs / "summary.csv")


def scalar_metrics(value: dict) -> dict:
    return {key: value[key] for key in ("macro_f1", "weighted_f1", "accuracy")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")

    torch.set_num_threads(args.threads)
    provenance_path = args.runs / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("scope") != "full":
        raise ValueError("This migration is restricted to full-data runs")
    device = torch.device(str(provenance.get("effective_device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoints requested but CUDA is unavailable")
    storage_dtype = (
        torch.float16
        if provenance.get("edge_storage_dtype") == "torch.float16"
        else torch.float32
    )

    table = pd.read_csv(args.runs / "runs.csv")
    expected = (
        len(provenance["datasets"]) * len(provenance["tasks"])
        * len(provenance["models"]) * len(provenance["seeds"])
    )
    if len(table) != expected or table[
        ["dataset", "task", "model", "seed"]
    ].duplicated().any():
        raise ValueError(f"Expected {expected} unique runs, found {len(table)}")
    rows = {
        (row["dataset"], row["task"], row["model"], int(row["seed"])): row
        for row in table.to_dict("records")
    }

    provenance.setdefault("training_source_sha256", provenance["source_sha256"])
    provenance["evaluation_source_sha256"] = experiment_source_sha256()
    rebuild = provenance.setdefault("evaluation_rebuild", {})
    rebuild.update({
        "status": "running",
        "mode": EVALUATION_MODE,
        "reason": "Replace non-reproducible BF16 full-graph GNN evaluation",
        "started_at_utc": rebuild.get("started_at_utc", utc_now()),
        "checkpoints_modified": False,
    })
    write_json(provenance_path, provenance)

    completed = 0
    for dataset in provenance["datasets"]:
        load_started = time.perf_counter()
        frames = load_frames(args.data, dataset, "full")
        dataset_load_seconds = time.perf_counter() - load_started
        graph_started = time.perf_counter()
        feature_pre, shared_graphs = prepare_graphs(frames, "multiclass", storage_dtype)
        graph_prepare_seconds = time.perf_counter() - graph_started
        for task in provenance["tasks"]:
            pre = preprocessor_for_task(feature_pre, task)
            graphs = (
                shared_graphs if task == "multiclass"
                else relabel_graphs(shared_graphs, frames, pre)
            )
            for name in provenance["models"]:
                for seed_value in provenance["seeds"]:
                    seed = int(seed_value)
                    key = (dataset, task, name, seed)
                    run_id = f"{dataset}__{task}__{name}__seed{seed}"
                    run_dir = args.runs / run_id
                    config = json.loads((run_dir / "config.json").read_text())
                    stored_pre = Preprocessor.from_dict(json.loads(
                        (run_dir / "preprocessor.json").read_text()
                    ))
                    if stored_pre.as_dict() != pre.as_dict():
                        raise AssertionError(f"Preprocessor mismatch: {run_id}")
                    result_path = run_dir / "metrics.json"
                    result = json.loads(result_path.read_text())
                    if (result.get("evaluation_mode") == EVALUATION_MODE
                            and (run_dir / "test_predictions.parquet").is_file()):
                        rows[key] = result_row(result)
                        completed += 1
                        print(f"SKIP {run_id}", flush=True)
                        continue

                    checkpoint_path = run_dir / "model.pt"
                    checkpoint_sha256 = sha256_file(checkpoint_path)
                    model = build_model(
                        name, len(FEATURES), len(pre.classes),
                        config["hidden"], config["dropout"],
                    ).to(device)
                    model.load_state_dict(torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True,
                    ))
                    started = time.perf_counter()
                    val_probability = full_probabilities(
                        model, name, graphs["val"], provenance.get("amp", False),
                    )
                    test_probability = full_probabilities(
                        model, name, graphs["test"], provenance.get("amp", False),
                    )
                    evaluation_seconds = time.perf_counter() - started
                    if sha256_file(checkpoint_path) != checkpoint_sha256:
                        raise AssertionError(f"Checkpoint changed: {run_id}")

                    result.setdefault("evaluation_rebuild", {
                        "previous_evaluation_mode": "cuda_bf16_amp",
                        "previous_seconds_final_evaluation": result[
                            "seconds_final_evaluation"
                        ],
                        "previous_val": scalar_metrics(result["val"]),
                        "previous_test": scalar_metrics(result["test"]),
                        "checkpoint_sha256": checkpoint_sha256,
                    })
                    result["evaluation_rebuild"]["rebuilt_at_utc"] = utc_now()
                    result["evaluation_mode"] = EVALUATION_MODE
                    result["seconds_dataset_load"] = dataset_load_seconds
                    result["seconds_graph_prepare"] = graph_prepare_seconds
                    result["seconds_final_evaluation"] = evaluation_seconds
                    result["seconds_fit_and_evaluate"] = (
                        result["seconds_fit"] + evaluation_seconds
                    )
                    result["val"] = metrics(
                        graphs["val"].labels.numpy(), val_probability, pre.classes,
                    )
                    result["test"] = metrics(
                        graphs["test"].labels.numpy(), test_probability, pre.classes,
                    )
                    result["test_prediction_rows_stored"] = save_predictions(
                        run_dir / "test_predictions.parquet", graphs["test"],
                        test_probability, pre.classes,
                        provenance.get("prediction_cap") or None,
                    )
                    result["test_prediction_rows_total"] = len(graphs["test"].labels)
                    write_json(result_path, result)
                    rows[key] = result_row(result)
                    write_table(args.runs / "runs.csv", rows)
                    completed += 1
                    print(
                        f"REBUILT {run_id} ({completed}/{expected}) "
                        f"test_macro_f1={result['test']['macro_f1']:.6f}",
                        flush=True,
                    )
                    del model, val_probability, test_probability
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if task != "multiclass":
                del graphs
        del shared_graphs, frames

    write_table(args.runs / "runs.csv", rows)
    write_summary(args.runs, rows)
    rebuild.update({
        "status": "complete",
        "completed_at_utc": utc_now(),
        "runs_rebuilt_or_confirmed": completed,
    })
    write_json(provenance_path, provenance)
    print(f"Rebuilt or confirmed {completed}/{expected} runs", flush=True)


if __name__ == "__main__":
    main()
