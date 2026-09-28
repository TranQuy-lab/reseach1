"""Run fair tabular baselines on the existing full-data splits.

This script deliberately consumes ``data/full_splits`` and the persisted
Preprocessor from a verified E-GraphSAGE run. It does not resplit data, fit
preprocessing on validation/test, or overwrite existing runs.

The models are external tabular comparators for the paper. Metrics are
computed on the complete validation/test split; only an optional bounded
prediction artifact is stored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from nids_minibatch.data import Preprocessor
from nids_minibatch.schema import DATASETS, FEATURES

# Cost-controlled paper comparator: one RF seed on the four multiclass cells.
# The existing 120-run GNN study remains the primary result; this arm is a
# bounded external comparator, not a claim about every tabular algorithm.
MODELS = ("random_forest",)
TASKS = ("multiclass",)
DEFAULT_SEEDS = (11,)
PROTOCOL = Path("research/PROTOCOL_RF_BOUNDED_VI.md")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_split(root: Path, dataset: str, split: str) -> pd.DataFrame:
    paths = sorted((root / dataset / f"split={split}").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing full split: {root / dataset / f'split={split}'}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    required = set(FEATURES) | {"Attack", "Label", "source_row_id", "flow_group_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset}/{split}: missing columns {sorted(missing)}")
    return frame


def load_persisted_preprocessor(runs_root: Path, dataset: str, task: str) -> Preprocessor:
    path = runs_root / f"{dataset}__{task}__edge_mlp__seed11" / "preprocessor.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing verified preprocessor {path}; run the full E-GraphSAGE pipeline first"
        )
    return Preprocessor.from_dict(json.loads(path.read_text()))


def target(pre: Preprocessor, frame: pd.DataFrame) -> np.ndarray:
    return pre.labels(frame)


def make_model(name: str, seed: int, n_jobs: int, n_estimators: int,
               max_depth: int | None):
    if name != "random_forest":
        raise ValueError(f"This bounded protocol only permits random_forest, got {name}")
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        max_features="sqrt",
    )


def score(model, x: np.ndarray, y: np.ndarray, classes: list[str]) -> dict[str, Any]:
    pred = model.predict(x)
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
    }


def run_one(data_root: Path, runs_root: Path, output_root: Path, dataset: str,
            task: str, model_name: str, seed: int, n_jobs: int,
            n_estimators: int, max_depth: int | None, prediction_cap: int,
            resume: bool = False) -> dict[str, Any]:
    run_id = f"{dataset}__{task}__{model_name}__seed{seed}"
    dest = output_root / run_id
    metrics_path = dest / "metrics.json"
    required_outputs = [
        metrics_path, dest / "config.json", dest / "model.joblib",
        dest / "preprocessor.json", dest / "test_predictions.parquet",
    ]
    if all(path.is_file() for path in required_outputs):
        print(f"SKIP {run_id}", flush=True)
        return json.loads(metrics_path.read_text())
    if dest.exists():
        if not resume:
            raise RuntimeError(f"Partial existing run refuses overwrite: {dest}")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    started = time.perf_counter()
    pre = load_persisted_preprocessor(runs_root, dataset, task)
    frames = {split: load_split(data_root, dataset, split) for split in ("train", "val", "test")}
    x = {split: pre.transform(frame).astype(np.float32, copy=False) for split, frame in frames.items()}
    y = {split: target(pre, frame) for split, frame in frames.items()}
    expected_classes = np.arange(len(pre.classes), dtype=np.int64)
    if not np.array_equal(np.unique(y["train"]), expected_classes):
        raise ValueError(
            f"{dataset}/{task}: train split does not contain every persisted class"
        )
    fit_started = time.perf_counter()
    model = make_model(model_name, seed, n_jobs, n_estimators, max_depth)
    model.fit(x["train"], y["train"])
    fit_seconds = time.perf_counter() - fit_started
    val_metrics = score(model, x["val"], y["val"], pre.classes)
    test_metrics = score(model, x["test"], y["test"], pre.classes)
    elapsed = time.perf_counter() - started
    cap = min(prediction_cap, len(y["test"])) if prediction_cap else len(y["test"])
    offsets = np.linspace(0, len(y["test"]) - 1, cap, dtype=np.int64) if cap else np.empty(0, dtype=np.int64)
    pred_frame = pd.DataFrame({
        "row_offset": offsets,
        "source_row_id": frames["test"].source_row_id.to_numpy()[offsets],
        "y_true": y["test"][offsets],
        "y_pred": model.predict(x["test"][offsets]),
    })
    if hasattr(model, "predict_proba"):
        raw_probability = model.predict_proba(x["test"][offsets])
        probability = np.zeros((len(offsets), len(pre.classes)), dtype=np.float64)
        probability[:, np.asarray(model.classes_, dtype=np.int64)] = raw_probability
        for i, name in enumerate(pre.classes):
            pred_frame[f"p_{name}"] = probability[:, i]
    pred_frame.to_parquet(dest / "test_predictions.parquet", index=False)
    joblib.dump(model, dest / "model.joblib", compress=3)
    config = {
        "dataset": dataset, "task": task, "model": model_name, "seed": seed,
        "features": list(FEATURES), "classes": pre.classes, "scope": "full",
        "n_estimators_or_max_iter": n_estimators, "max_depth": max_depth,
        "early_stopping": None,
        "class_weight": "balanced",
        "n_jobs": n_jobs, "prediction_cap": prediction_cap,
        "preprocessor_source": str((runs_root / f"{dataset}__{task}__edge_mlp__seed11" / "preprocessor.json").as_posix()),
    }
    write_json(dest / "preprocessor.json", pre.as_dict())
    config["preprocessor_sha256"] = sha256_file(dest / "preprocessor.json")
    write_json(dest / "config.json", config)
    result = {
        **config, "seconds_fit_and_evaluate": elapsed, "seconds_fit": fit_seconds,
        "train_rows": int(len(y["train"])), "val_rows": int(len(y["val"])),
        "test_rows": int(len(y["test"])), "val": val_metrics, "test": test_metrics,
    }
    write_json(metrics_path, result)
    print(f"DONE {run_id} test_macro_f1={test_metrics['macro_f1']:.6f} seconds={elapsed:.1f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/full_splits"))
    parser.add_argument("--runs", type=Path, default=Path("research/artifacts/full_runs_5seed"))
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/tabular_rf_bounded_multiclass"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--n-estimators", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--prediction-cap", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.n_estimators < 1 or args.prediction_cap < 0:
        parser.error("n-jobs, n-estimators and prediction-cap must be valid")
    if not PROTOCOL.is_file():
        parser.error(f"missing locked protocol: {PROTOCOL}")
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output exists; pass --resume or choose a new directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    rows, started = [], time.perf_counter()
    for dataset in args.datasets:
        for task in args.tasks:
            for model_name in args.models:
                for seed in args.seeds:
                    result = run_one(
                        args.data, args.runs, args.output, dataset, task, model_name, seed,
                        args.n_jobs, args.n_estimators, args.max_depth,
                        args.prediction_cap, args.resume,
                    )
                    rows.append({
                        "dataset": dataset, "task": task, "model": model_name, "seed": seed,
                        "seconds_fit_and_evaluate": result["seconds_fit_and_evaluate"],
                        "seconds_fit": result["seconds_fit"],
                        "train_rows": result["train_rows"], "val_rows": result["val_rows"],
                        "test_rows": result["test_rows"],
                        "val_macro_f1": result["val"]["macro_f1"],
                        "val_weighted_f1": result["val"]["weighted_f1"],
                        "val_accuracy": result["val"]["accuracy"],
                        "test_macro_f1": result["test"]["macro_f1"],
                        "test_weighted_f1": result["test"]["weighted_f1"],
                        "test_accuracy": result["test"]["accuracy"],
                    })
    fields = list(rows[0]) if rows else []
    with (args.output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "task", "model"], as_index=False)[
        ["test_macro_f1", "test_weighted_f1", "test_accuracy", "seconds_fit_and_evaluate"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(value).rstrip("_") for value in summary.columns]
    summary.to_csv(args.output / "summary.csv", index=False)
    write_json(args.output / "provenance.json", {
        "scope": "full", "datasets": args.datasets, "tasks": args.tasks,
        "models": args.models, "seeds": args.seeds, "features": list(FEATURES),
        "n_jobs": args.n_jobs, "n_estimators": args.n_estimators, "max_depth": args.max_depth,
        "prediction_cap": args.prediction_cap, "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "protocol_path": PROTOCOL.as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": sha256_file(Path(__file__)),
        "gnn_runs_root": args.runs.as_posix(),
        "note": "External tabular comparators on the locked full-data splits.",
    })


if __name__ == "__main__":
    main()
