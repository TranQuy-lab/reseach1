#!/usr/bin/env python3
"""TN-9 - Tabular comparator on the endpoint-disjoint split.

Motivation
----------
On NF-ToN-IoT-v2 multiclass the capacity-matched flow-only MLP falls from 0.696 to
0.409 when endpoints are held out, i.e. it leans on endpoint familiarity for 0.29
macro-F1. Yet on the locked split a HistGradientBoosting comparator that never sees
endpoint identity reaches 0.866, far above any graph variant.

The question that decides how to read both facts: does the strong tabular model also
collapse on unseen endpoints? If it holds, then the flow features do carry the
fine-grained attack-type signal and the neural models' endpoint reliance is a
self-inflicted weakness rather than a property of the data.

Uses the feature matrices materialised by scripts/16 (scaler fit on the holdout
split's own train), and the same locked hyper-parameter grid and selection rule.

Outputs: results/tn9_holdout_tabular.csv, results/tn9_holdout_tabular_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parents[1]
HOLDOUT_FEAT = HERE / "work/holdout_features"
LOCKED_FEAT = HERE / "work/features"
OUT = HERE / "results"

GRID = [
    {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_iter": 400, "max_leaf_nodes": 63},
]


def score(model, X, y) -> dict:
    p = model.predict(X)
    return {"macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y, p))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NF-ToN-IoT-v2"])
    ap.add_argument("--tasks", nargs="+", default=["multiclass"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--tune-seed", type=int, default=11)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    runs = OUT / "tn9_holdout_tabular.csv"
    rows = []
    for ds in args.datasets:
        for task in args.tasks:
            root = HOLDOUT_FEAT
            Xtr = np.load(root / f"{ds}__{task}__train.npy", mmap_mode="r")
            Xva = np.load(root / f"{ds}__{task}__val.npy", mmap_mode="r")
            Xte = np.load(root / f"{ds}__{task}__test.npy", mmap_mode="r")
            ytr = np.load(root / f"{ds}__{task}__train__y.npy")
            yva = np.load(root / f"{ds}__{task}__val__y.npy")
            yte = np.load(root / f"{ds}__{task}__test__y.npy")
            print(f"### {ds} {task} (endpoint-holdout): train={len(ytr):,d} "
                  f"val={len(yva):,d} test={len(yte):,d}", flush=True)
            best_cfg, best_val = None, -np.inf
            for cfg in GRID:
                t0 = time.perf_counter()
                m = HistGradientBoostingClassifier(random_state=args.tune_seed,
                                                   class_weight="balanced",
                                                   early_stopping=True,
                                                   validation_fraction=0.1,
                                                   n_iter_no_change=20, **cfg)
                m.fit(Xtr, ytr)
                v = score(m, Xva, yva)
                print(f"  tune {cfg} val={v['macro_f1']:.4f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
                if v["macro_f1"] > best_val:
                    best_cfg, best_val = cfg, v["macro_f1"]
                del m
            for seed in args.seeds:
                t0 = time.perf_counter()
                m = HistGradientBoostingClassifier(random_state=seed,
                                                   class_weight="balanced",
                                                   early_stopping=True,
                                                   validation_fraction=0.1,
                                                   n_iter_no_change=20, **best_cfg)
                m.fit(Xtr, ytr)
                v = score(m, Xva, yva)
                te = score(m, Xte, yte)
                rows.append({"dataset": ds, "task": task, "model": "hist_gradient_boosting",
                             "split": "endpoint_holdout", "seed": seed,
                             "config": json.dumps(best_cfg),
                             "val_macro_f1": v["macro_f1"],
                             "test_macro_f1": te["macro_f1"],
                             "test_weighted_f1": te["weighted_f1"],
                             "test_accuracy": te["accuracy"],
                             "fit_seconds": time.perf_counter() - t0})
                pd.DataFrame(rows).to_csv(runs, index=False)
                print(f"[{time.strftime('%H:%M:%S')}] {ds} {task} HGB holdout s{seed} "
                      f"TEST={te['macro_f1']:.4f}", flush=True)
                del m

    df = pd.DataFrame(rows)
    # compare against the locked-split MLP on both splits
    comp = []
    locked_mlp = OUT / "tn1_mlp_runs.csv"
    holdout_mlp = OUT / "tn4_holdout_runs.csv"
    for ds in args.datasets:
        for task in args.tasks:
            hb = df[(df.dataset == ds) & (df.task == task)].test_macro_f1.mean()
            if locked_mlp.exists():
                a = pd.read_csv(locked_mlp)
                a = a[(a.dataset == ds) & (a.task == task)]
                for model in ("mlp_h128_1l", "mlp_h273_2l"):
                    g = a[a.model == model]
                    if len(g):
                        comp.append({"dataset": ds, "task": task, "model": model,
                                     "split": "locked", "test_macro_f1": g.test_macro_f1.mean()})
            if holdout_mlp.exists():
                b = pd.read_csv(holdout_mlp)
                b = b[(b.dataset == ds) & (b.task == task)]
                for model in ("mlp_h128_1l", "mlp_h273_2l"):
                    g = b[b.model == model]
                    if len(g):
                        comp.append({"dataset": ds, "task": task, "model": model,
                                     "split": "endpoint_holdout",
                                     "test_macro_f1": g.test_macro_f1.mean()})
            comp.append({"dataset": ds, "task": task, "model": "hist_gradient_boosting",
                         "split": "endpoint_holdout", "test_macro_f1": hb})
    c = pd.DataFrame(comp)
    c.to_csv(OUT / "tn9_holdout_comparison.csv", index=False)
    summary = {
        "holdout_tabular": df.groupby(["dataset", "task"]).test_macro_f1
            .agg(["count", "mean", "std"]).round(4).reset_index().to_dict(orient="records"),
        "comparison": c.round(4).to_dict(orient="records"),
    }
    (OUT / "tn9_holdout_tabular_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
