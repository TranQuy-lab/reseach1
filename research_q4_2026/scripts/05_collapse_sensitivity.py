#!/usr/bin/env python3
"""A5 - Optimization-collapse identification and ranking sensitivity.

Observation that motivates this script (found in A2/A4): the topology-only variant
`sage` shows seed-dependent bimodal outcomes (e.g. CSE-CIC multiclass DDoS test F1
= 0.99/0.64/0.97/0.66/0.99) while `edge_mlp` is stable. This script tests whether
the reported model ranking survives removal of runs that a *prespecified,
metric-free* rule labels as optimisation-collapsed.

Collapse rule (prespecified here, applied identically to all three models, uses
only the validation trajectory, never the test set):

    collapse  :=  val_std >= 0.05   OR   drop_after_best >= 0.10

where val_std is the SD of the run's validation macro-F1 trajectory and
drop_after_best the maximum fall below the trajectory maximum.

Outputs
-------
results/collapse_flags.csv
results/collapse_rates.csv
results/ranking_sensitivity.csv
results/collapse_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"

VAL_STD_T = 0.05
DROP_T = 0.10
CONTRASTS = [("sage", "edge_mlp"), ("sage_edge", "edge_mlp"), ("sage_edge", "sage")]


def boot(d: np.ndarray, n: int = 10000, rng=None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(7)
    if len(d) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    reg = pd.read_csv(OUT / "run_registry.csv")
    conv = pd.read_csv(OUT / "convergence_per_run.csv")
    df = reg.merge(conv[["run_id", "val_std", "drop_after_best", "best_is_last_eval",
                         "classification"]], on="run_id", how="left")
    df["collapsed"] = (df.val_std >= VAL_STD_T) | (df.drop_after_best >= DROP_T)
    df.to_csv(OUT / "collapse_flags.csv", index=False)

    rates = df.groupby(["dataset", "task", "model"]).agg(
        n=("run_id", "size"), collapsed=("collapsed", "sum")).reset_index()
    rates["collapse_rate"] = rates.collapsed / rates.n
    rates.to_csv(OUT / "collapse_rates.csv", index=False)
    by_model = df.groupby("model").collapsed.agg(["sum", "size", "mean"]).reset_index()
    by_model.columns = ["model", "n_collapsed", "n_runs", "collapse_rate"]

    # ---- ranking sensitivity: all runs vs non-collapsed runs ---------------
    rows = []
    for (ds, task) in df.groupby(["dataset", "task"]).groups:
        cell = df[(df.dataset == ds) & (df.task == task)]
        w_all = cell.pivot_table(index="seed", columns="model", values="test_macro_f1")
        keep = cell[~cell.collapsed]
        w_keep = keep.pivot_table(index="seed", columns="model", values="test_macro_f1")
        for a, b in CONTRASTS:
            for label, w in (("all_runs", w_all), ("non_collapsed_only", w_keep)):
                if a not in w or b not in w:
                    rows.append({"dataset": ds, "task": task, "comparison": f"{a}_minus_{b}",
                                 "subset": label, "n": 0, "mean_delta": np.nan,
                                 "ci_lo": np.nan, "ci_hi": np.nan})
                    continue
                d = (w[a] - w[b]).dropna().to_numpy(dtype=float)
                lo, hi = boot(d)
                rows.append({"dataset": ds, "task": task, "comparison": f"{a}_minus_{b}",
                             "subset": label, "n": len(d), "mean_delta": float(d.mean()),
                             "ci_lo": lo, "ci_hi": hi,
                             "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    rs = pd.DataFrame(rows)
    rs.to_csv(OUT / "ranking_sensitivity.csv", index=False)

    wide = rs.pivot_table(index=["dataset", "task", "comparison"], columns="subset",
                          values="mean_delta")
    wide = wide.dropna()
    sign_change = ((np.sign(wide["all_runs"]) != np.sign(wide["non_collapsed_only"]))
                   .sum()) if len(wide) else 0
    corr = float(wide["all_runs"].corr(wide["non_collapsed_only"])) if len(wide) > 2 else np.nan

    summary = {
        "rule": {"val_std_threshold": VAL_STD_T, "drop_after_best_threshold": DROP_T},
        "collapse_by_model": by_model.to_dict(orient="records"),
        "collapse_by_dataset": df.groupby("dataset").collapsed.agg(["sum", "size", "mean"])
            .reset_index().rename(columns={"sum": "n_collapsed", "size": "n_runs",
                                           "mean": "collapse_rate"}).to_dict(orient="records"),
        "n_cells_compared": int(len(wide)),
        "sign_changes_all_vs_noncollapsed": int(sign_change),
        "corr_mean_delta_all_vs_noncollapsed": corr,
        "noncollapsed_contrast_direction_counts": {
            f"{k[0]}|sign{k[1]:+.0f}": int(v) for k, v in
            (rs[rs.subset == "non_collapsed_only"].assign(sign=lambda x: np.sign(x.mean_delta))
               .groupby("comparison")["sign"].value_counts().items())},
        "all_contrast_direction_counts": {
            f"{k[0]}|sign{k[1]:+.0f}": int(v) for k, v in
            (rs[rs.subset == "all_runs"].assign(sign=lambda x: np.sign(x.mean_delta))
               .groupby("comparison")["sign"].value_counts().items())},
    }
    (OUT / "collapse_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print()
    pd.set_option("display.width", 220)
    print(by_model.to_string(index=False))
    print()
    print(rates[rates.collapsed > 0].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
