#!/usr/bin/env python3
"""A4 - Rare-class behaviour, per-class variance, and the instability -> score link.

Uses the per-run metrics.json (full per-class precision/recall/F1/support for the
test split) for all 120 archived runs. Read-only over the repository.

Outputs
-------
results/per_class_raw.csv + per_class_summary.csv
results/rare_class_5seed.csv
results/instability_vs_score.csv
results/rare_class_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/noble-tran/nghiencuu/repo_reseach1")
RES = REPO / "research"
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"

IGNORE = {"accuracy", "macro avg", "weighted avg"}


def main() -> int:
    reg = pd.read_csv(OUT / "run_registry.csv")
    rows = []
    for r in reg.itertuples():
        d = REPO / r.artifact_dir
        met = json.loads((d / "metrics.json").read_text())
        for split in ("val", "test"):
            pc = met.get(split, {}).get("per_class", {})
            for cls, vals in pc.items():
                if cls in IGNORE or not isinstance(vals, dict):
                    continue
                if "f1-score" not in vals:
                    continue
                rows.append({
                    "dataset": r.dataset, "task": r.task, "model": r.model, "seed": r.seed,
                    "split": split, "class": cls,
                    "precision": vals.get("precision"), "recall": vals.get("recall"),
                    "f1": vals.get("f1-score"), "support": vals.get("support"),
                })
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / "per_class_raw.csv", index=False)

    test = pc[pc.split == "test"]
    agg = test.groupby(["dataset", "task", "class", "model"]).agg(
        support=("support", "first"),
        f1_mean=("f1", "mean"), f1_std=("f1", lambda s: s.std(ddof=1)),
        f1_min=("f1", "min"), f1_max=("f1", "max"),
        recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
    ).reset_index()
    agg["f1_range"] = agg.f1_max - agg.f1_min
    agg["cv"] = agg.f1_std / agg.f1_mean.replace(0, np.nan)

    # per-class paired contrast (topology - flow only), same seeds
    wide = test.pivot_table(index=["dataset", "task", "class", "seed"], columns="model",
                            values="f1")
    deltas = []
    for (ds, task, cls), g in wide.groupby(level=[0, 1, 2]):
        for a, b, name in [("sage", "edge_mlp", "sage_minus_edge_mlp"),
                           ("sage_edge", "edge_mlp", "sage_edge_minus_edge_mlp"),
                           ("sage_edge", "sage", "sage_edge_minus_sage")]:
            if a not in g or b not in g:
                continue
            dd = (g[a] - g[b]).dropna().to_numpy(dtype=float)
            if len(dd) < 2:
                continue
            deltas.append({
                "dataset": ds, "task": task, "class": cls, "comparison": name,
                "support": float(agg[(agg.dataset == ds) & (agg.task == task) &
                                     (agg["class"] == cls)].support.iloc[0]),
                "mean_delta": float(dd.mean()), "sd_delta": float(dd.std(ddof=1)),
            })
    dl = pd.DataFrame(deltas)
    dl = dl.drop(columns=["support"], errors="ignore")
    dlw = dl.pivot_table(index=["dataset", "task", "class"], columns="comparison",
                         values=["mean_delta", "sd_delta"])
    dlw.columns = [f"{a}_{b}" for a, b in dlw.columns]
    dlw = dlw.reset_index()
    agg = agg.merge(dlw, on=["dataset", "task", "class"], how="left")
    agg.to_csv(OUT / "per_class_summary.csv", index=False)

    rare = agg[agg.support < 1000].sort_values(["support", "dataset", "class", "model"])
    rare.to_csv(OUT / "rare_class_5seed.csv", index=False)

    # ---- instability of validation curve vs achieved test score -----------
    conv = pd.read_csv(OUT / "convergence_per_run.csv")
    m = conv.merge(reg[["run_id", "seed"]], on="run_id")
    by_cell = m.groupby(["dataset", "task", "model"]).agg(
        mean_val_std=("val_std", "mean"),
        mean_drop_after_best=("drop_after_best", "mean"),
        mean_best_val=("best_val", "mean"),
        test_mean=("test_macro_f1", "mean"),
    ).reset_index()
    by_cell.to_csv(OUT / "instability_vs_score.csv", index=False)

    # run-level correlation, and within-model (to remove model identity)
    run_lvl = m[["val_std", "drop_after_best", "test_macro_f1", "best_val", "model"]].dropna()
    rho_all, p_all = stats.spearmanr(run_lvl.val_std, run_lvl.test_macro_f1)
    rho_drop, p_drop = stats.spearmanr(run_lvl.drop_after_best, run_lvl.test_macro_f1)
    within = {}
    for mdl, g in run_lvl.groupby("model"):
        if len(g) > 5:
            r, p = stats.spearmanr(g.val_std, g.test_macro_f1)
            within[mdl] = {"n": int(len(g)), "spearman_rho": float(r), "p": float(p)}
    # partial: gap between best validation and test (generalisation gap)
    run_lvl = run_lvl.assign(gap=run_lvl.best_val - run_lvl.test_macro_f1)

    out = {
        "n_per_class_records": int(len(pc)),
        "n_test_class_cells": int(len(agg)),
        "rare_class_cells_support_lt_1000": int(len(rare)),
        "rare_class_list": rare[["dataset", "task", "class", "support"]].drop_duplicates()
                              .to_dict(orient="records"),
        "classes_with_highest_seed_variance": agg.sort_values("f1_std", ascending=False)
            .head(12)[["dataset", "task", "class", "model", "support", "f1_mean", "f1_std"]]
            .to_dict(orient="records"),
        "spearman_valstd_vs_testmacro": {"rho": float(rho_all), "p": float(p_all),
                                         "n": int(len(run_lvl))},
        "spearman_dropafterbest_vs_testmacro": {"rho": float(rho_drop), "p": float(p_drop)},
        "spearman_within_model": within,
        "mean_generalisation_gap_by_model": run_lvl.groupby("model").gap.mean().to_dict(),
    }
    (OUT / "rare_class_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
