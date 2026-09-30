#!/usr/bin/env python3
"""A6 - Figures for the evidence audit and the mechanism story."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["edge_mlp", "sage", "sage_edge"]
COL = {"edge_mlp": "#4C72B0", "sage": "#C44E52", "sage_edge": "#55A868"}
DS = ["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"]
SHORT = {"NF-UNSW-NB15-v2": "UNSW", "NF-ToN-IoT-v2": "ToN",
         "NF-CSE-CIC-IDS2018-v2": "CSE-CIC", "NF-BoT-IoT-v2": "BoT-IoT"}


def fig1_ranking_with_ci():
    pc = pd.read_csv(OUT / "paired_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, task in zip(axes, ["multiclass", "binary"]):
        sub = pc[pc.task == task]
        cells = [f"{SHORT[d]}" for d in DS]
        y = np.arange(len(DS))
        for j, (cmp_, colour, marker) in enumerate([
                ("sage_minus_edge_mlp", "#C44E52", "o"),
                ("sage_edge_minus_edge_mlp", "#55A868", "s"),
                ("sage_edge_minus_sage", "#8172B2", "^")]):
            means, los, his = [], [], []
            for d in DS:
                r = sub[(sub.dataset == d) & (sub.comparison == cmp_)]
                means.append(r.mean_delta.iloc[0]); los.append(r.boot_ci_lo.iloc[0])
                his.append(r.boot_ci_hi.iloc[0])
            off = (j - 1) * 0.22
            ax.errorbar(means, y + off, xerr=[np.array(means) - np.array(los),
                                              np.array(his) - np.array(means)],
                        fmt=marker, color=colour, capsize=3, ms=6,
                        label=cmp_.replace("_minus_", " − ").replace("_", " "))
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(cells)
        ax.set_title(f"{task}  (paired seed deltas, 95% bootstrap CI)")
        ax.set_xlabel("Δ macro-F1")
        ax.grid(alpha=0.25, axis="x")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Model contrasts: no contrast is stable across all datasets", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_contrast_forest.png", dpi=170)
    plt.close(fig)


def fig2_instability_vs_score():
    cf = pd.read_csv(OUT / "collapse_flags.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for m in MODELS:
        g = cf[cf.model == m]
        ax.scatter(g.val_std, g.test_macro_f1, s=26, alpha=0.75, c=COL[m], label=m,
                   edgecolors="none")
    ax.axvline(0.05, color="k", ls=":", lw=1.2)
    ax.text(0.0505, 0.03, "collapse rule\nval_std ≥ 0.05", fontsize=7.5, va="bottom")
    ax.set_xlabel("validation-curve SD (instability)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Instability predicts a lower test score (ρ = −0.54, n = 120)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    conv = pd.read_csv(OUT / "convergence_per_run.csv")
    counts = (conv.assign(cat=lambda x: np.select(
        [x.best_is_last_eval & (x.last_slope > 0), x.decay_after_best],
        ["still rising", "decay after best"], default="plateau"))
        .groupby(["model", "cat"]).size().unstack(fill_value=0))
    counts = counts.reindex(MODELS)
    bottom = np.zeros(len(counts))
    colours = {"still rising": "#DD8452", "decay after best": "#C44E52", "plateau": "#55A868"}
    for c in ["still rising", "decay after best", "plateau"]:
        if c in counts:
            ax.bar(counts.index, counts[c], bottom=bottom, label=c, color=colours[c])
            bottom += counts[c].to_numpy()
    ax.set_ylabel("runs (of 40 per model)")
    ax.set_title("Budget-conditioned behaviour of each variant")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_instability_and_budget.png", dpi=170)
    plt.close(fig)


def fig3_collapse_rates():
    cr = pd.read_csv(OUT / "collapse_flags.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    r = cr.groupby("model").collapsed.mean().reindex(MODELS)
    ax.bar(r.index, r.values, color=[COL[m] for m in r.index])
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.006, f"{v*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("collapse rate"); ax.set_ylim(0, 0.32)
    ax.set_title("Optimisation collapse by variant (n = 40 runs each)")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    cell = cr.groupby(["dataset", "task", "model"]).collapsed.mean().reset_index()
    piv = cell.pivot_table(index=["dataset", "task"], columns="model", values="collapsed")
    piv = piv.reindex([(d, t) for d in DS for t in ["multiclass", "binary"]])
    labels = [f"{SHORT[d]}\n{t}" for d, t in piv.index]
    x = np.arange(len(piv))
    w = 0.26
    for j, m in enumerate(MODELS):
        ax.bar(x + (j - 1) * w, piv[m].fillna(0).values, width=w, color=COL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("collapse rate"); ax.set_title("Collapse is concentrated in BoT-IoT `sage`")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_collapse_rates.png", dpi=170)
    plt.close(fig)


def fig4_learning_curves():
    lc = pd.read_csv(OUT / "learning_curves.csv")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=False)
    for i, d in enumerate(DS):
        for j, task in enumerate(["multiclass", "binary"]):
            ax = axes[j][i]
            sub = lc[(lc.dataset == d) & (lc.task == task)]
            for m in MODELS:
                g = sub[sub.model == m]
                if g.empty:
                    continue
                for seed, gg in g.groupby("seed"):
                    gg = gg.sort_values("eval_index")
                    ax.plot(gg.eval_index, gg.val_macro_f1, color=COL[m], alpha=0.55, lw=1.3)
                mean = g.groupby("eval_index").val_macro_f1.mean()
                ax.plot(mean.index, mean.values, color=COL[m], lw=2.6, label=m,
                        marker="o", ms=3)
            ax.set_title(f"{SHORT[d]} · {task}", fontsize=10)
            ax.set_xlabel("validation event"); ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_ylabel("validation macro-F1")
            if i == 3 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Archived validation learning curves (thin = per seed, thick = mean); "
                 "the budget is the right-hand edge", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_learning_curves.png", dpi=150)
    plt.close(fig)


def main() -> int:
    fig1_ranking_with_ci()
    fig2_instability_vs_score()
    fig3_collapse_rates()
    fig4_learning_curves()
    print("figures written to", FIG)
    for p in sorted(FIG.iterdir()):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
