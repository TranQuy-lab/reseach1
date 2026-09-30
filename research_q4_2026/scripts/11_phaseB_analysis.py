#!/usr/bin/env python3
"""B5 - Phase B synthesis: capacity, topology and tabular comparators.

Consumes:
  results/run_registry.csv     archived GNN runs (edge_mlp, sage, sage_edge)
  results/tn1_mlp_runs.csv     capacity-matched / depth controls on the same split
  results/tn2_tabular_runs.csv RF / ExtraTrees / HistGradientBoosting

Key contrasts (paired by nominal seed, same split, same preprocessing):
  capacity   : mlp_h273_2l - edge_mlp        (capacity/depth alone, NO topology)
  topology   : sage        - mlp_h273_2l     (topology at MATCHED capacity) <- central
  direct-edge: sage_edge   - sage            (direct flow path into the head)
  tabular    : best_tabular - edge_mlp / sage_edge

Outputs: results/phaseB_*.csv, results/phaseB_summary.json, figures/fig5..fig7
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261002)

SHORT = {"NF-UNSW-NB15-v2": "UNSW", "NF-ToN-IoT-v2": "ToN",
         "NF-CSE-CIC-IDS2018-v2": "CSE-CIC", "NF-BoT-IoT-v2": "BoT-IoT"}
ORDER = ["edge_mlp", "mlp_h128_1l", "mlp_h128_2l", "mlp_h273_2l", "sage", "sage_edge",
         "hist_gradient_boosting", "extra_trees", "random_forest"]
LABEL = {
    "edge_mlp": "edge_mlp (archive, 5.4k)",
    "mlp_h128_1l": "MLP h128x1 (5.4k, control)",
    "mlp_h128_2l": "MLP h128x2",
    "mlp_h273_2l": "MLP h273x2 (86k = sage capacity)",
    "sage": "sage (archive, 87k)",
    "sage_edge": "sage_edge (archive, 87k)",
    "hist_gradient_boosting": "HistGradientBoosting",
    "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest",
}


def boot(d, n=20000):
    if len(d) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_all() -> pd.DataFrame:
    reg = pd.read_csv(OUT / "run_registry.csv")[
        ["dataset", "task", "model", "seed", "test_macro_f1", "test_weighted_f1",
         "test_accuracy", "parameters"]]
    frames = [reg]
    for path in ["tn1_mlp_runs.csv", "tn2_tabular_runs.csv"]:
        p = OUT / path
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "parameters" not in d:
            d["parameters"] = np.nan
        frames.append(d[["dataset", "task", "model", "seed", "test_macro_f1",
                         "test_weighted_f1", "test_accuracy", "parameters"]])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    allr = load_all()
    rows = []
    for (ds, task), g in allr.groupby(["dataset", "task"]):
        for model, gg in g.groupby("model"):
            v = gg.test_macro_f1.dropna()
            if not len(v):
                continue
            rows.append({"dataset": ds, "task": task, "model": model, "n_seeds": len(v),
                         "test_macro_f1_mean": v.mean(), "test_macro_f1_std": v.std(ddof=1)
                         if len(v) > 1 else np.nan,
                         "test_macro_f1_min": v.min(), "test_macro_f1_max": v.max(),
                         "weighted_f1_mean": gg.test_weighted_f1.mean(),
                         "parameters": gg.parameters.dropna().iloc[0]
                         if gg.parameters.notna().any() else np.nan})
    tab = pd.DataFrame(rows)
    pivot = tab.pivot_table(index=["dataset", "task"], columns="model",
                            values="test_macro_f1_mean")
    present = [m for m in ORDER if m in pivot.columns]
    pivot = pivot[present]
    pivot.to_csv(OUT / "phaseB_matrix.csv")
    print(pivot.round(4).to_string())

    # ---- central contrasts -------------------------------------------------
    wide = allr.pivot_table(index=["dataset", "task", "seed"], columns="model",
                            values="test_macro_f1")
    contrast_specs = [
        ("capacity_no_topology", "mlp_h273_2l", "edge_mlp"),
        ("topology_at_matched_capacity", "sage", "mlp_h273_2l"),
        ("topology_at_matched_capacity_edge", "sage_edge", "mlp_h273_2l"),
        ("topology_confounded_archive", "sage", "edge_mlp"),
        ("direct_edge_path", "sage_edge", "sage"),
        ("best_tabular_vs_edge_mlp", "hist_gradient_boosting", "edge_mlp"),
        ("best_tabular_vs_topo", "hist_gradient_boosting", "sage_edge"),
        ("control_vs_archive", "mlp_h128_1l", "edge_mlp"),
    ]
    crows = []
    for (ds, task), g in wide.groupby(level=[0, 1]):
        for name, a, b in contrast_specs:
            if a not in g.columns or b not in g.columns:
                continue
            d = (g[a] - g[b]).dropna().to_numpy(dtype=float)
            if len(d) < 2:
                continue
            lo, hi = boot(d)
            try:
                _, p = stats.wilcoxon(d) if np.any(d != 0) else (np.nan, 1.0)
            except ValueError:
                p = 1.0
            crows.append({"dataset": ds, "task": task, "contrast": name,
                          "a": a, "b": b, "n": len(d), "mean_delta": float(d.mean()),
                          "sd": float(d.std(ddof=1)), "ci_lo": lo, "ci_hi": hi,
                          "ci_excludes_zero": bool(lo > 0 or hi < 0),
                          "p_wilcoxon": float(p),
                          "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan})
    cl = pd.DataFrame(crows)
    cl.to_csv(OUT / "phaseB_contrasts.csv", index=False)

    pooled = cl.groupby("contrast").agg(
        k=("mean_delta", "size"), pooled_mean=("mean_delta", "mean"),
        n_positive=("mean_delta", lambda s: int((s > 0).sum())),
        n_negative=("mean_delta", lambda s: int((s < 0).sum()))).reset_index()
    pooled.to_csv(OUT / "phaseB_pooled.csv", index=False)
    print()
    print(cl.round(4).to_string(index=False))
    print()
    print(pooled.to_string(index=False))

    # ---- figure: full matrix ----------------------------------------------
    cells = [f"{SHORT[d]}\n{t}" for d, t in pivot.index]
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(present)), 7))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([LABEL.get(m, m) for m in present], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        color="w", fontsize=7.5)
    fig.colorbar(im, label="test macro-F1 (mean over seeds)")
    ax.set_title("Phase B: archived GNN, capacity controls and tabular comparators "
                 "on the locked split")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_phaseB_matrix.png", dpi=170)
    plt.close(fig)

    # ---- figure: central contrasts ---------------------------------------
    key = cl[cl.contrast.isin(["capacity_no_topology", "topology_at_matched_capacity",
                               "topology_confounded_archive", "direct_edge_path"])]
    if len(key):
        fig, ax = plt.subplots(figsize=(9.5, 6))
        order = ["capacity_no_topology", "topology_confounded_archive",
                 "topology_at_matched_capacity", "direct_edge_path"]
        ylab, ypos = [], []
        y = 0
        colours = {"capacity_no_topology": "#4C72B0",
                   "topology_confounded_archive": "#C44E52",
                   "topology_at_matched_capacity": "#55A868",
                   "direct_edge_path": "#8172B2"}
        for c in order:
            sub = key[key.contrast == c]
            for r in sub.itertuples():
                ax.errorbar(r.mean_delta, y,
                            xerr=[[r.mean_delta - r.ci_lo], [r.ci_hi - r.mean_delta]],
                            fmt="o", color=colours[c], capsize=3, ms=6)
                ylab.append(f"{SHORT[r.dataset]}·{r.task[:4]}")
                ypos.append(y)
                y += 1
            y += 1
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylab, fontsize=7)
        ax.set_xlabel("Δ test macro-F1 (paired by seed, 95% bootstrap CI)")
        handles = [plt.Line2D([], [], marker="o", ls="", color=colours[c],
                              label=c.replace("_", " ")) for c in order]
        ax.legend(handles=handles, fontsize=8, loc="lower right")
        ax.set_title("Separating capacity from topology")
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        fig.savefig(FIG / "fig6_capacity_vs_topology.png", dpi=170)
        plt.close(fig)

    summary = {
        "models_present": present,
        "n_cells": int(len(pivot)),
        "pooled_by_contrast": pooled.to_dict(orient="records"),
        "matrix": {f"{d}|{t}": {m: (None if not np.isfinite(v) else float(v))
                                for m, v in row.items()}
                   for (d, t), row in pivot.iterrows()},
    }
    (OUT / "phaseB_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
