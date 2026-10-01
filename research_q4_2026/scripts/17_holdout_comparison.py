#!/usr/bin/env python3
"""B8 - Locked split versus endpoint-holdout split, same model, same budget.

Compares results/tn4_holdout_runs.csv (endpoint-disjoint) with
results/tn1_mlp_runs.csv (locked flow_group_id split) for identical architectures,
seeds and budget rules. The only intended difference is whether the test flows can
involve endpoints that were seen during training.

Outputs: results/holdout_comparison.csv, results/holdout_summary.json,
         figures/fig10_holdout_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
FIG = HERE / "figures"
RNG = np.random.default_rng(99)
SHORT = {"NF-UNSW-NB15-v2": "UNSW", "NF-ToN-IoT-v2": "ToN"}


def boot(d, n=20000):
    if len(d) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    hp = RES / "tn4_holdout_runs.csv"
    lp = RES / "tn1_mlp_runs.csv"
    if not hp.exists() or not lp.exists():
        print("need both tn4_holdout_runs.csv and tn1_mlp_runs.csv")
        return 1
    h = pd.read_csv(hp)
    l = pd.read_csv(lp)
    key = ["dataset", "task", "model", "seed"]
    l = l.rename(columns={"test_macro_f1": "locked_macro_f1",
                          "test_weighted_f1": "locked_weighted_f1"})
    h = h.rename(columns={"test_macro_f1": "holdout_macro_f1",
                          "test_weighted_f1": "holdout_weighted_f1"})
    m = l[key + ["locked_macro_f1", "locked_weighted_f1"]].merge(
        h[key + ["holdout_macro_f1", "holdout_weighted_f1"]], on=key)
    m["delta_locked_minus_holdout"] = m.locked_macro_f1 - m.holdout_macro_f1
    m.to_csv(RES / "holdout_comparison.csv", index=False)

    rows = []
    for (ds, task, model), g in m.groupby(["dataset", "task", "model"]):
        d = g.delta_locked_minus_holdout.to_numpy(dtype=float)
        lo, hi = boot(d)
        rows.append({"dataset": ds, "task": task, "model": model, "n": len(d),
                     "locked_mean": g.locked_macro_f1.mean(),
                     "holdout_mean": g.holdout_macro_f1.mean(),
                     "mean_drop": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                     "relative_drop_percent": float(100 * d.mean() / g.locked_macro_f1.mean())})
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "holdout_summary.csv", index=False)
    out = {
        "n_pairs": int(len(m)),
        "cells": summ.to_dict(orient="records"),
        "overall_mean_drop": float(m.delta_locked_minus_holdout.mean()),
        "overall_relative_drop_percent": float(
            100 * m.delta_locked_minus_holdout.mean() / m.locked_macro_f1.mean()),
    }
    (RES / "holdout_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(summ.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2))

    if len(summ):
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(summ)), 4.8))
        x = np.arange(len(summ))
        ax.bar(x - 0.2, summ.locked_mean, width=0.4, label="locked split",
               color="#4C72B0")
        ax.bar(x + 0.2, summ.holdout_mean, width=0.4, label="endpoint holdout",
               color="#C44E52")
        labels = [f"{SHORT.get(r.dataset, r.dataset)}\n{r.task[:4]}·{r.model}"
                  for r in summ.itertuples()]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("test macro-F1")
        ax.legend(fontsize=8)
        ax.set_title("Same model and budget, endpoint-disjoint test set")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(FIG / "fig10_holdout_comparison.png", dpi=170)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
