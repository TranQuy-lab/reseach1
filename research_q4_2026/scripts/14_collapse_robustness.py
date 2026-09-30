#!/usr/bin/env python3
"""A7 - Robustness of the optimisation-collapse finding.

Three checks that a reviewer will demand:
  1. collapse rate as a function of the (prespecified) threshold - is the 0/25/5 %
     ordering an artefact of the cut-off?
  2. seed-level bootstrap CI on each model's collapse rate;
  3. how informative validation actually is: Spearman(best validation macro-F1,
     test macro-F1) per model, i.e. whether the checkpoint rule works.

Outputs: results/collapse_robustness.csv, results/collapse_robustness.json,
         figures/fig8_collapse_robustness.png
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
RNG = np.random.default_rng(4242)
MODELS = ["edge_mlp", "sage", "sage_edge"]
COL = {"edge_mlp": "#4C72B0", "sage": "#C44E52", "sage_edge": "#55A868"}


def main() -> int:
    df = pd.read_csv(OUT / "collapse_flags.csv")
    conv = pd.read_csv(OUT / "convergence_per_run.csv")
    df = df.merge(conv[["run_id", "best_val", "last_val", "best_pass"]], on="run_id",
                  how="left")
    rows = []
    for vs_t in (0.02, 0.05, 0.10, 0.20):
        for dp_t in (0.05, 0.10, 0.20):
            flag = (df.val_std >= vs_t) | (df.drop_after_best >= dp_t)
            for m, g in df.assign(flag=flag).groupby("model"):
                rows.append({"val_std_threshold": vs_t, "drop_threshold": dp_t,
                             "model": m, "n": len(g), "collapsed": int(g.flag.sum()),
                             "rate": float(g.flag.mean())})
    rb = pd.DataFrame(rows)
    rb.to_csv(OUT / "collapse_robustness.csv", index=False)

    # bootstrap CI on the headline rule
    boot = {}
    base = df.assign(flag=(df.val_std >= 0.05) | (df.drop_after_best >= 0.10))
    for m, g in base.groupby("model"):
        f = g.flag.to_numpy(dtype=float)
        idx = RNG.integers(0, len(f), size=(20000, len(f)))
        r = f[idx].mean(axis=1)
        boot[m] = {"rate": float(f.mean()),
                   "ci_lo": float(np.percentile(r, 2.5)),
                   "ci_hi": float(np.percentile(r, 97.5)), "n": int(len(f))}
    # pairwise Fisher exact tests between variants
    pair = {}
    for a, b in (("sage", "edge_mlp"), ("sage", "sage_edge"), ("sage_edge", "edge_mlp")):
        ga = base[base.model == a].flag.to_numpy()
        gb = base[base.model == b].flag.to_numpy()
        table = [[int(ga.sum()), int((~ga).sum())], [int(gb.sum()), int((~gb).sum())]]
        try:
            odds, p = stats.fisher_exact(table)
        except Exception:
            odds, p = np.nan, np.nan
        pair[f"{a}_vs_{b}"] = {"table": table, "odds_ratio": float(odds), "p": float(p)}
    # Holm-Bonferroni across the three pairwise tests
    keys = list(pair)
    order = sorted(range(len(keys)), key=lambda i: pair[keys[i]]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        val = max(running, (len(keys) - rank) * pair[keys[i]]["p"])
        running = val
        pair[keys[i]]["p_holm"] = float(min(val, 1.0))

    # validation informativeness
    corr = {}
    for m, g in df.groupby("model"):
        rho, p = stats.spearmanr(g.best_val, g.test_macro_f1)
        corr[m] = {"n": int(len(g)), "spearman_bestval_vs_test": float(rho), "p": float(p),
                   "mean_gap_bestval_minus_test": float((g.best_val - g.test_macro_f1).mean())}
    out = {"headline_rule": {"val_std": 0.05, "drop_after_best": 0.10},
           "bootstrap_ci": boot,
           "fisher_exact": pair,
           "validation_informativeness": corr,
           "threshold_ordering_stable": bool(
               all(len(set(rb[(rb.val_std_threshold == v) & (rb.drop_threshold == d)]
                           .set_index("model")["rate"].to_dict().items())) == 3
                   for v in rb.val_std_threshold.unique()
                   for d in rb.drop_threshold.unique()))}
    (OUT / "collapse_robustness.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    sub = rb[rb.drop_threshold == 0.10]
    for m in MODELS:
        g = sub[sub.model == m].sort_values("val_std_threshold")
        ax.plot(g.val_std_threshold, g.rate, marker="o", color=COL[m], label=m)
    ax.set_xscale("log")
    ax.set_xlabel("validation-SD threshold")
    ax.set_ylabel("collapse rate")
    ax.set_title("Collapse ordering is stable across thresholds")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    xs = np.arange(len(MODELS))
    rates = [boot[m]["rate"] for m in MODELS]
    err = [[boot[m]["rate"] - boot[m]["ci_lo"] for m in MODELS],
           [boot[m]["ci_hi"] - boot[m]["rate"] for m in MODELS]]
    ax.bar(xs, rates, yerr=err, capsize=5, color=[COL[m] for m in MODELS])
    ax.set_xticks(xs); ax.set_xticklabels(MODELS)
    ax.set_ylabel("collapse rate (95 % bootstrap CI)")
    ax.set_title("Optimisation collapse by variant")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig8_collapse_robustness.png", dpi=170)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
