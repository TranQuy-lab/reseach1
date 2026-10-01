#!/usr/bin/env python3
"""B3 (TN-2) - Strong tabular comparators on the locked split (Gate B).

Protocol fairness rules taken from research/PROTOCOL_FULL_DATA_VI.md and
ke_hoach_nang-cap-bai-bao-quoc-te.md:
  * identical split, identical train-fitted preprocessing (scalers reused from the
    archived GNN runs, not refit);
  * no IP/port/flow identifier ever enters the feature matrix;
  * hyper-parameter selection on the validation split only;
  * the test split is scored once, at the end, for the selected configuration;
  * class imbalance handled with class_weight="balanced" in every comparator;
  * every configuration tried is recorded, including the losers.

Models:
  hist_gradient_boosting  - primary strong tabular comparator (memory efficient)
  extra_trees             - bagged randomised trees
  random_forest           - classic reference comparator

Output: results/tn2_tabular_runs.csv, results/tn2_tuning.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import resource

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "work/features"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "tn2_tabular_runs.csv"
TUNE = OUT / "tn2_tuning.csv"

DATASETS = ["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"]

GRIDS = {
    "hist_gradient_boosting": [
        {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31},
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31},
        {"learning_rate": 0.1, "max_iter": 400, "max_leaf_nodes": 63},
    ],
    "extra_trees": [
        {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 200, "max_features": 0.3, "min_samples_leaf": 2},
    ],
    "random_forest": [
        {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1},
    ],
}


def build(name: str, cfg: dict, seed: int, n_jobs: int):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=seed, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            **cfg)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, class_weight="balanced",
                                    n_jobs=n_jobs, **cfg)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, class_weight="balanced",
                                      n_jobs=n_jobs, **cfg)
    raise ValueError(name)


def score(model, X, y) -> dict:
    p = model.predict(X)
    return {"macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y, p))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=["multiclass", "binary"])
    ap.add_argument("--models", nargs="+", default=list(GRIDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--tune-seed", type=int, default=11,
                    help="hyper-parameters are selected on validation with this single "
                         "seed; the selected configuration is then refit for every seed")
    args = ap.parse_args()

    done = set()
    if RUNS.exists():
        prev = pd.read_csv(RUNS)
        done = set(zip(prev.dataset, prev.task, prev.model, prev.seed))

    tune_rows = []
    for dataset in args.datasets:
        Xtr = np.load(WORK / f"{dataset}__train.npy", mmap_mode="r")
        Xva = np.load(WORK / f"{dataset}__val.npy", mmap_mode="r")
        Xte = np.load(WORK / f"{dataset}__test.npy", mmap_mode="r")
        print(f"\n### {dataset}: train={len(Xtr):,d} val={len(Xva):,d} test={len(Xte):,d}",
              flush=True)
        for task in args.tasks:
            ytr = np.load(WORK / f"{dataset}__train__y_{task}.npy")
            yva = np.load(WORK / f"{dataset}__val__y_{task}.npy")
            yte = np.load(WORK / f"{dataset}__test__y_{task}.npy")
            for name in args.models:
                pending = [s for s in args.seeds if (dataset, task, name, s) not in done]
                if not pending:
                    print("skip", dataset, task, name, flush=True)
                    continue
                # ---- stage 1: select configuration on validation, tune seed only ----
                best_cfg, best_val = None, -np.inf
                for cfg in GRIDS[name]:
                    t0 = time.perf_counter()
                    model = build(name, cfg, args.tune_seed, args.n_jobs)
                    model.fit(Xtr, ytr)
                    fit_s = time.perf_counter() - t0
                    v = score(model, Xva, yva)
                    tune_rows.append({"dataset": dataset, "task": task, "model": name,
                                      "seed": args.tune_seed, "config": json.dumps(cfg),
                                      "fit_seconds": fit_s,
                                      "val_macro_f1": v["macro_f1"],
                                      "val_weighted_f1": v["weighted_f1"],
                                      "val_accuracy": v["accuracy"]})
                    pd.DataFrame(tune_rows).to_csv(TUNE, index=False)
                    print(f"  tune {name:24s} s{args.tune_seed} {cfg} -> val_macro_f1="
                          f"{v['macro_f1']:.4f} ({fit_s:.0f}s)", flush=True)
                    if v["macro_f1"] > best_val:
                        best_cfg, best_val = cfg, v["macro_f1"]
                    del model
                # ---- stage 2: refit the selected configuration for each seed ----
                for seed in pending:
                    cfg = best_cfg
                    t0 = time.perf_counter()
                    model = build(name, cfg, seed, args.n_jobs)
                    model.fit(Xtr, ytr)
                    fit_s = time.perf_counter() - t0
                    v = score(model, Xva, yva)
                    t0 = time.perf_counter()
                    te = score(model, Xte, yte)
                    test_s = time.perf_counter() - t0
                    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
                    row = {"dataset": dataset, "task": task, "model": name, "seed": seed,
                           "config": json.dumps(cfg), "n_train": int(len(Xtr)),
                           "peak_rss_gib": peak_rss_gib,
                           "parameters": int(getattr(model, "n_iter_", 0) or 0),
                           "selected_by": "validation_macro_f1",
                           "val_macro_f1": v["macro_f1"], "val_weighted_f1": v["weighted_f1"],
                           "val_accuracy": v["accuracy"],
                           "test_macro_f1": te["macro_f1"],
                           "test_weighted_f1": te["weighted_f1"],
                           "test_accuracy": te["accuracy"],
                           "fit_seconds": fit_s, "predict_seconds": test_s}
                    pd.DataFrame([row]).to_csv(RUNS, mode="a",
                                               header=not RUNS.exists(), index=False)
                    print(f"[{time.strftime('%H:%M:%S')}] {dataset} {task} {name} s{seed} "
                          f"TEST macro_f1={te['macro_f1']:.4f} weighted={te['weighted_f1']:.4f} "
                          f"peakRSS={peak_rss_gib:.2f}GiB", flush=True)
                    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
