#!/usr/bin/env python3
"""A2 - Convergence / budget diagnostics from the ARCHIVED validation learning curves.

This is a *post-hoc* (explicitly exploratory) diagnostic on the 120 archived runs.
It does NOT replace the prespecified convergence-first protocol; it measures how
much of the two-pass comparison is budget-conditioned and separates three rival
explanations that the proposal lists:

  R1 underfitting         -> validation macro-F1 still rising at the budget boundary
  R2 optimization decay   -> validation macro-F1 falls after an early best checkpoint
  R3 genuine plateau      -> best checkpoint reached early and later evals flat

Outputs
-------
results/convergence_per_run.csv
results/convergence_by_cell.csv
results/convergence_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"

EPS = 0.005          # "flat" band in macro-F1
DECAY_TOL = 0.005    # drop that counts as decay


def diagnose(g: pd.DataFrame) -> dict:
    g = g.sort_values("eval_index")
    v = g["val_macro_f1"].to_numpy(dtype=float)
    steps = g["step"].to_numpy(dtype=float)
    passes = g["pass_no"].to_numpy(dtype=float)
    n = len(v)
    best_i = int(np.argmax(v))
    best_v = float(v[best_i])
    last_v = float(v[-1])
    first_v = float(v[0])

    # R1: still rising at the boundary?
    if n >= 2:
        last_slope = float(v[-1] - v[-2])
    else:
        last_slope = float("nan")

    # earliest eval that is within EPS of the run's own best and never leaves the band
    plateau_i = None
    for i in range(n):
        if np.all(v[i:] >= best_v - EPS):
            plateau_i = i
            break

    # R2: decay after best
    after = v[best_i:]
    drop_after_best = float(best_v - after.min())

    run_max = float(np.max(v))
    run_min = float(np.min(v))
    return {
        "n_evals": n,
        "first_val": first_v,
        "last_val": last_v,
        "best_val": best_v,
        "best_eval_index": best_i,
        "best_pass": float(passes[best_i]),
        "best_step": float(steps[best_i]),
        "last_step": float(steps[-1]),
        "evals_from_best_to_end": n - 1 - best_i,
        "best_is_first_eval": bool(best_i == 0),
        "best_is_last_eval": bool(best_i == n - 1),
        "last_minus_best": float(last_v - best_v),
        "last_slope": last_slope,
        "still_rising_at_boundary": bool(best_i == n - 1 and last_slope > 0),
        "plateau_eval_index": plateau_i,
        "plateau_pass": (float(passes[plateau_i]) if plateau_i is not None else None),
        "plateau_within_budget": bool(plateau_i is not None),
        "drop_after_best": drop_after_best,
        "val_range": run_max - run_min,
        "val_std": float(np.std(v, ddof=1)) if n > 1 else float("nan"),
        # trajectory shape flags
        "monotone_nondecreasing": bool(np.all(np.diff(v) >= -1e-12)),
        "decay_after_best": bool(drop_after_best > DECAY_TOL),
        "improved_over_first": bool(last_v > first_v + EPS),
        "regressed_below_first": bool(last_v < first_v - EPS),
    }


def classify(r: pd.Series) -> str:
    """Three-way classification of the budget-conditioned evidence."""
    if r["best_is_last_eval"] and r["last_slope"] > 0:
        return "R1_still_rising"
    if r["decay_after_best"]:
        return "R2_optimization_decay"
    if r["plateau_within_budget"]:
        return "R3_plateau"
    return "R4_undetermined"


def main() -> int:
    cf = pd.read_csv(OUT / "learning_curves.csv")
    reg = pd.read_csv(OUT / "run_registry.csv")

    per_run = []
    for rid, g in cf.groupby("run_id"):
        d = diagnose(g)
        d["run_id"] = rid
        per_run.append(d)
    pr = pd.DataFrame(per_run)
    pr = pr.merge(reg[["run_id", "dataset", "task", "model", "seed", "test_macro_f1",
                       "val_macro_f1", "best_step", "steps_per_full_pass", "steps_ran",
                       "step_budget", "min_train_steps", "train_passes_configured"]],
                  on="run_id", how="left")
    pr["classification"] = pr.apply(classify, axis=1)
    pr["effective_passes_run"] = pr["steps_ran"] / pr["steps_per_full_pass"]
    pr["budget_binding"] = np.where(pr["steps_ran"] <= pr["min_train_steps"],
                                    "min_train_steps", "dataset_passes")
    pr = pr.sort_values(["dataset", "task", "model", "seed"]).reset_index(drop=True)
    pr.to_csv(OUT / "convergence_per_run.csv", index=False)

    agg = pr.groupby(["dataset", "task", "model"]).agg(
        n=("run_id", "size"),
        mean_best_val=("best_val", "mean"),
        mean_last_val=("last_val", "mean"),
        mean_last_minus_best=("last_minus_best", "mean"),
        mean_best_pass=("best_pass", "mean"),
        frac_best_is_first_eval=("best_is_first_eval", "mean"),
        frac_best_is_last_eval=("best_is_last_eval", "mean"),
        frac_still_rising=("still_rising_at_boundary", "mean"),
        frac_decay_after_best=("decay_after_best", "mean"),
        mean_drop_after_best=("drop_after_best", "mean"),
        mean_val_std=("val_std", "mean"),
        mean_effective_passes=("effective_passes_run", "mean"),
    ).reset_index()
    cls = pr.pivot_table(index=["dataset", "task", "model"], columns="classification",
                         values="run_id", aggfunc="count").fillna(0).astype(int).reset_index()
    agg = agg.merge(cls, on=["dataset", "task", "model"], how="left")
    agg.to_csv(OUT / "convergence_by_cell.csv", index=False)

    summary = {
        "n_runs": int(len(pr)),
        "classification_counts": pr["classification"].value_counts().to_dict(),
        "classification_share": (pr["classification"].value_counts(normalize=True)
                                 .round(4).to_dict()),
        "frac_best_is_first_eval": float(pr["best_is_first_eval"].mean()),
        "frac_best_is_last_eval": float(pr["best_is_last_eval"].mean()),
        "frac_decay_after_best": float(pr["decay_after_best"].mean()),
        "frac_improved_over_first": float(pr["improved_over_first"].mean()),
        "mean_last_minus_best": float(pr["last_minus_best"].mean()),
        "by_model_classification": {
            m: g["classification"].value_counts().to_dict()
            for m, g in pr.groupby("model")},
        "by_dataset_classification": {
            d: g["classification"].value_counts().to_dict()
            for d, g in pr.groupby("dataset")},
        "by_task_classification": {
            t: g["classification"].value_counts().to_dict()
            for t, g in pr.groupby("task")},
        "budget_binding_counts": pr["budget_binding"].value_counts().to_dict(),
        "effective_passes_by_dataset": {
            d: {"mean": float(g["effective_passes_run"].mean()),
                "min": float(g["effective_passes_run"].min()),
                "max": float(g["effective_passes_run"].max())}
            for d, g in pr.groupby("dataset")},
    }
    (OUT / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
