"""Run fair tabular baselines on the existing full-data splits.

This script deliberately consumes ``data/full_splits`` and the persisted
Preprocessor from a verified E-GraphSAGE run. It does not resplit data, fit
preprocessing on validation/test, or overwrite existing runs.

The models are secondary comparators for the paper, not replacements for the
paper-faithful E-GraphSAGE reference. Metrics are computed on the complete
validation/test split; only an optional bounded prediction artifact is stored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from nids_minibatch.data import Preprocessor
from nids_minibatch.schema import DATASETS, FEATURES

MODELS = ("random_forest", "extra_trees", "hist_gradient_boosting")
TASKS = ("multiclass", "binary")
DEFAULT_SEEDS = (11, 22, 33)


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
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            max_features="sqrt",
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            max_features="sqrt",
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    weights = len(y) / (len(counts) * np.maximum(counts, 1))
    return weights[y]


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
            n_estimators: int, max_depth: int | None, prediction_cap: int) -> dict[str, Any]:
    run_id = f"{dataset}__{task}__{model_name}__seed{seed}"
    dest = output_root / run_id
    metrics_path = dest / "metrics.json"
    if metrics_path.is_file() and (dest / "config.json").is_file():
        print(f"SKIP {run_id}", flush=True)
        return json.loads(metrics_path.read_text())
    if dest.exists():
        raise RuntimeError(f"Partial existing run refuses overwrite: {dest}")
    dest.mkdir(parents=True)
    started = time.perf_counter()
    pre = load_persisted_preprocessor(runs_root, dataset, task)
    frames = {split: load_split(data_root, dataset, split) for split in ("train", "val", "test")}
    x = {split: pre.transform(frame).astype(np.float32, copy=False) for split, frame in frames.items()}
    y = {split: target(pre, frame) for split, frame in frames.items()}
    fit_started = time.perf_counter()
    model = make_model(model_name, seed, n_jobs, n_estimators, max_depth)
    fit_kwargs = {}
    if model_name == "hist_gradient_boosting":
        fit_kwargs["sample_weight"] = balanced_sample_weight(y["train"])
    model.fit(x["train"], y["train"], **fit_kwargs)
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
        prob = model.predict_proba(x["test"][offsets])
        for i, name in enumerate(pre.classes):
            pred_frame[f"p_{name}"] = prob[:, i]
    pred_frame.to_parquet(dest / "test_predictions.parquet", index=False)
    import joblib
    joblib.dump(model, dest / "model.joblib", compress=3)
    config = {
        "dataset": dataset, "task": task, "model": model_name, "seed": seed,
        "features": list(FEATURES), "classes": pre.classes, "scope": "full",
        "n_estimators_or_max_iter": n_estimators, "max_depth": max_depth,
        "class_weight": "balanced" if model_name != "hist_gradient_boosting" else "sample_weight",
        "n_jobs": n_jobs, "prediction_cap": prediction_cap,
        "preprocessor_source": str((runs_root / f"{dataset}__{task}__edge_mlp__seed11" / "preprocessor.json").as_posix()),
    }
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
    parser.add_argument("--runs", type=Path, default=Path("research/artifacts/full_runs"))
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/tabular_full_runs"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--prediction-cap", type=int, default=100_000)
    args = parser.parse_args()
    if args.n_jobs < 1 or args.n_estimators < 1 or args.prediction_cap < 0:
        parser.error("n-jobs, n-estimators and prediction-cap must be valid")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, started = [], time.perf_counter()
    for dataset in args.datasets:
        for task in args.tasks:
            for model_name in args.models:
                for seed in args.seeds:
                    result = run_one(
                        args.data, args.runs, args.output, dataset, task, model_name, seed,
                        args.n_jobs, args.n_estimators, args.max_depth, args.prediction_cap,
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
    write_json(args.output / "provenance.json", {
        "scope": "full", "datasets": args.datasets, "tasks": args.tasks,
        "models": args.models, "seeds": args.seeds, "features": list(FEATURES),
        "n_jobs": args.n_jobs, "n_estimators": args.n_estimators, "max_depth": args.max_depth,
        "prediction_cap": args.prediction_cap, "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "note": "Secondary tabular comparators; not the primary E-GraphSAGE baseline.",
    })


if __name__ == "__main__":
    main()
