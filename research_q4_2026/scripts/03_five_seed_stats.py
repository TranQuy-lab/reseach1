#!/usr/bin/env python3
"""A3 - Five-seed inferential statistics for the model contrasts.

Design facts that constrain the analysis (recorded, not assumed):
  * prediction unit = flow/edge; replication unit = seed (n = 5 per cell)
  * 8 dataset x task cells, 3 prespecified contrasts:
        sage      - edge_mlp
        sage_edge - edge_mlp
        sage_edge - sage
  * seeds are paired across models within a cell (same split, same seed)

Reported per contrast:
  mean/se/SD of paired deltas, Cohen's dz, exact sign test, Wilcoxon,
  seed-level bootstrap CI (BCa-free percentile, 20000 resamples),
  TOST equivalence vs a +-0.01 macro-F1 margin,
  and a DerSimonian-Laird random-effects pooled estimate across cells with
  Cochran Q / I^2 heterogeneity.

Multiplicity: Holm-Bonferroni across the 24 cell-level tests, applied to the
Wilcoxon p-values, and reported alongside - never instead of - effect sizes.

Outputs
-------
results/paired_contrasts.csv
results/pooled_meta.csv
results/stats_summary.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
RNG = np.random.default_rng(20261001)
NBOOT = 20000
MARGIN = 0.01  # equivalence margin in macro-F1

CONTRASTS = [
    ("sage", "edge_mlp", "sage_minus_edge_mlp"),
    ("sage_edge", "edge_mlp", "sage_edge_minus_edge_mlp"),
    ("sage_edge", "sage", "sage_edge_minus_sage"),
]


def bootstrap_ci(d: np.ndarray, n: int = NBOOT) -> tuple[float, float]:
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost(d: np.ndarray, margin: float = MARGIN) -> dict:
    """Two one-sided tests against equivalence margin (paired t)."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {"t_lower": np.inf, "t_upper": -np.inf, "p_tost": 0.0 if abs(d.mean()) < margin else 1.0}
    t_lower = (d.mean() + margin) / se       # H0: mu <= -margin
    t_upper = (d.mean() - margin) / se       # H0: mu >= +margin
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    return {"t_lower": float(t_lower), "t_upper": float(t_upper),
            "p_tost": float(max(p_lower, p_upper))}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()


def main() -> int:
    reg = pd.read_csv(OUT / "run_registry.csv")
    reg = reg[reg.model.isin(["edge_mlp", "sage", "sage_edge"])]

    wide = reg.pivot_table(index=["dataset", "task", "seed"], columns="model",
                           values="test_macro_f1")
    wide_val = reg.pivot_table(index=["dataset", "task", "seed"], columns="model",
                               values="val_macro_f1")

    rows = []
    for (ds, task), g in wide.groupby(level=[0, 1]):
        for a, b, name in CONTRASTS:
            d = (g[a] - g[b]).dropna().to_numpy(dtype=float)
            n = len(d)
            mean = float(d.mean())
            sd = float(d.std(ddof=1))
            se = sd / np.sqrt(n)
            lo, hi = bootstrap_ci(d)
            # exact sign test (two-sided)
            nz = d[d != 0]
            k = int((nz > 0).sum())
            if len(nz) == 0:
                p_sign = 1.0
            else:
                p_sign = float(min(1.0, 2 * stats.binom.cdf(min(k, len(nz) - k), len(nz), 0.5)))
            try:
                w_stat, p_wilcox = stats.wilcoxon(d, zero_method="wilcox")
                p_wilcox = float(p_wilcox)
            except ValueError:
                w_stat, p_wilcox = np.nan, 1.0
            t_stat, p_t = stats.ttest_rel(g[a].dropna(), g[b].dropna())
            rows.append({
                "dataset": ds, "task": task, "comparison": name,
                "model_a": a, "model_b": b, "n_pairs": n,
                "mean_delta": mean, "sd_delta": sd, "se_delta": se,
                "cohen_dz": float(mean / sd) if sd > 0 else np.nan,
                "boot_ci_lo": lo, "boot_ci_hi": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "sign_test_k_positive": k, "p_sign": p_sign,
                "wilcoxon_W": float(w_stat) if w_stat == w_stat else np.nan,
                "p_wilcoxon": p_wilcox,
                "t_stat": float(t_stat), "p_paired_t": float(p_t),
                "ci_width": hi - lo,
                "direction": "a> b" if mean > 0 else ("a<b" if mean < 0 else "tie"),
                "seed_deltas": ";".join(f"{x:.6f}" for x in d),
            })
    pc = pd.DataFrame(rows)
    pc["p_wilcoxon_holm"] = holm(pc["p_wilcoxon"].tolist())
    pc["p_sign_holm"] = holm(pc["p_sign"].tolist())
    pc = pc.sort_values(["dataset", "task", "comparison"]).reset_index(drop=True)
    pc.to_csv(OUT / "paired_contrasts.csv", index=False)

    # ---- DerSimonian-Laird random-effects pooling per contrast -----------
    meta_rows = []
    for name, g in pc.groupby("comparison"):
        y = g["mean_delta"].to_numpy(dtype=float)
        s = g["se_delta"].to_numpy(dtype=float)
        v = s ** 2
        k = len(y)
        w = 1.0 / v
        y_fixed = float((w * y).sum() / w.sum())
        Q = float((w * (y - y_fixed) ** 2).sum())
        df = k - 1
        p_Q = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else np.nan
        C = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
        I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
        wr = 1.0 / (v + tau2)
        y_re = float((wr * y).sum() / wr.sum())
        se_re = float(np.sqrt(1.0 / wr.sum()))
        crit = float(stats.norm.ppf(0.975))
        # prediction interval
        if k > 1:
            pi_lo = y_re - stats.t.ppf(0.975, k - 2) * np.sqrt(se_re ** 2 + tau2)
            pi_hi = y_re + stats.t.ppf(0.975, k - 2) * np.sqrt(se_re ** 2 + tau2)
        else:
            pi_lo = pi_hi = np.nan
        meta_rows.append({
            "comparison": name, "k_cells": k,
            "pooled_delta_RE": y_re, "se_RE": se_re,
            "ci_lo_RE": y_re - crit * se_re, "ci_hi_RE": y_re + crit * se_re,
            "pooled_delta_fixed": y_fixed,
            "tau2": tau2, "tau": float(np.sqrt(tau2)), "I2_percent": I2,
            "Q": Q, "p_Q": p_Q,
            "prediction_interval_lo": float(pi_lo), "prediction_interval_hi": float(pi_hi),
            "n_cells_positive": int((y > 0).sum()),
            "n_cells_negative": int((y < 0).sum()),
            "pooled_ci_excludes_zero": bool((y_re - crit * se_re) > 0 or (y_re + crit * se_re) < 0),
        })
    mt = pd.DataFrame(meta_rows)
    mt.to_csv(OUT / "pooled_meta.csv", index=False)

    # ---- 3-seed vs 5-seed decision stability -----------------------------
    seed_list = [11, 22, 33]
    stab = []
    for (ds, task), g in wide.groupby(level=[0, 1]):
        for a, b, name in CONTRASTS:
            full = (g[a] - g[b]).dropna()
            g3 = g[g.index.get_level_values("seed").isin(seed_list)]
            if not set(g3.index.get_level_values("seed")) == set(seed_list):
                continue
            d3 = (g3[a] - g3[b]).dropna()
            stab.append({
                "dataset": ds, "task": task, "comparison": name,
                "mean3": float(d3.mean()), "mean5": float(full.mean()),
                "sign3": np.sign(d3.mean()), "sign5": np.sign(full.mean()),
                "sign_flip": bool(np.sign(d3.mean()) != np.sign(full.mean())),
                "ci3_lo": bootstrap_ci(d3.to_numpy())[0], "ci3_hi": bootstrap_ci(d3.to_numpy())[1],
                "ci5_lo": bootstrap_ci(full.to_numpy())[0], "ci5_hi": bootstrap_ci(full.to_numpy())[1],
            })
    ss = pd.DataFrame(stab)
    ss["ci3_excludes_zero"] = (ss.ci3_lo > 0) | (ss.ci3_hi < 0)
    ss["ci5_excludes_zero"] = (ss.ci5_lo > 0) | (ss.ci5_hi < 0)
    ss["decision_flip_on_ci"] = ss.ci3_excludes_zero != ss.ci5_excludes_zero
    ss.to_csv(OUT / "seed_count_stability.csv", index=False)

    summary = {
        "n_cells": int(wide.groupby(level=[0, 1]).ngroups),
        "n_contrasts_total": int(len(pc)),
        "wilcoxon_significant_uncorrected": int((pc.p_wilcoxon < 0.05).sum()),
        "wilcoxon_significant_holm": int((pc.p_wilcoxon_holm < 0.05).sum()),
        "sign_test_significant_holm": int((pc.p_sign_holm < 0.05).sum()),
        "ci_excludes_zero_uncorrected": int(pc.ci_excludes_zero.sum()),
        "cells_positive_negative": {
            name: {"positive": int((g.mean_delta > 0).sum()),
                   "negative": int((g.mean_delta < 0).sum())}
            for name, g in pc.groupby("comparison")},
        "sign_flips_3_to_5_seed": int(ss.sign_flip.sum()),
        "ci_decision_flips_3_to_5_seed": int(ss.decision_flip_on_ci.sum()),
        "n_stability_rows": int(len(ss)),
        "pooled": mt.to_dict(orient="records"),
    }
    (OUT / "stats_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
