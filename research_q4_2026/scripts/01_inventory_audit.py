#!/usr/bin/env python3
"""A1 - Inventory + consistency audit of the archived E-GraphSAGE evidence base.

Read-only over the source repository. Writes outputs into this working directory.
No file inside TranQuy-lab/reseach1 is modified.

Outputs
-------
results/run_registry.csv        one row per archived run (120 GNN + tabular RF)
results/learning_curves.csv     long-format validation curves from history.json
results/audit_inventory.json    machine-checkable audit facts
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/noble-tran/nghiencuu/repo_reseach1")
RES = REPO / "research"
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

GNN_DIRS = [RES / "artifacts/full_runs", RES / "artifacts/full_runs_seeds44_55"]
TAB_DIR = RES / "artifacts/tabular_rf_bounded_multiclass"

RUN_RE = re.compile(r"^(?P<dataset>.+?)__(?P<task>binary|multiclass)__(?P<model>.+?)__seed(?P<seed>\d+)$")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_gnn() -> tuple[list[dict], list[dict]]:
    registry, curves = [], []
    for base in GNN_DIRS:
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            m = RUN_RE.match(d.name)
            if not m:
                continue
            cfg = json.loads((d / "config.json").read_text())
            met = json.loads((d / "metrics.json").read_text())
            hist = json.loads((d / "history.json").read_text())
            spp = cfg.get("steps_per_full_pass") or 1
            best_step = met.get("best_step")
            row = {
                "run_id": d.name,
                "artifact_dir": str(d.relative_to(REPO)),
                "source_batch": base.name,
                "dataset": m.group("dataset"),
                "task": m.group("task"),
                "model": m.group("model"),
                "seed": int(m.group("seed")),
                "n_evals": len(hist),
                "steps_ran": met.get("steps_ran"),
                "step_budget": met.get("step_budget"),
                "steps_per_full_pass": spp,
                "eval_every_steps": met.get("eval_every_steps"),
                "train_passes_configured": cfg.get("train_passes"),
                "min_train_steps": cfg.get("min_train_steps"),
                "batch_size": cfg.get("batch_size"),
                "fanout": "-".join(map(str, cfg.get("fanout") or [])),
                "hidden": cfg.get("hidden"),
                "dropout": cfg.get("dropout"),
                "learning_rate": cfg.get("learning_rate"),
                "amp": cfg.get("amp"),
                "scope": cfg.get("scope"),
                "patience": cfg.get("patience"),
                "best_epoch": met.get("best_epoch"),
                "best_step": best_step,
                "best_pass": (math.ceil(best_step / spp) if best_step else None),
                "parameters": met.get("parameters"),
                "seconds_fit_and_evaluate": met.get("seconds_fit_and_evaluate"),
                "seconds_fit": met.get("seconds_fit"),
                "peak_rss_kib_process": met.get("peak_rss_kib_process"),
                "peak_cuda_bytes": met.get("peak_cuda_bytes"),
                "checkpoint_parameter_max_abs_error": met.get("checkpoint_parameter_max_abs_error"),
                "val_macro_f1": met.get("val", {}).get("macro_f1"),
                "val_weighted_f1": met.get("val", {}).get("weighted_f1"),
                "val_accuracy": met.get("val", {}).get("accuracy"),
                "test_macro_f1": met.get("test", {}).get("macro_f1"),
                "test_weighted_f1": met.get("test", {}).get("weighted_f1"),
                "test_accuracy": met.get("test", {}).get("accuracy"),
                "sha256_metrics": sha256(d / "metrics.json"),
            }
            registry.append(row)
            for i, h in enumerate(hist):
                curves.append({
                    "run_id": d.name,
                    "dataset": row["dataset"],
                    "task": row["task"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "eval_index": i,
                    "epoch": h.get("epoch"),
                    "step": h.get("step"),
                    "pass_no": (math.ceil(h["step"] / spp) if h.get("step") else None),
                    "loss": h.get("loss"),
                    "val_loss": h.get("val_loss"),
                    "val_macro_f1": h.get("val_macro_f1"),
                    "train_edges": h.get("train_edges"),
                    "train_batches": h.get("train_batches"),
                    "train_seconds": h.get("train_seconds"),
                    "validation_seconds": h.get("validation_seconds"),
                    "eligible_after_full_pass": h.get("eligible_after_full_pass"),
                })
    return registry, curves


def collect_tabular() -> list[dict]:
    rows = []
    if not TAB_DIR.exists():
        return rows
    for d in sorted(p for p in TAB_DIR.iterdir() if p.is_dir()):
        m = RUN_RE.match(d.name)
        if not m:
            continue
        cfg = json.loads((d / "config.json").read_text())
        met = json.loads((d / "metrics.json").read_text())
        rows.append({
            "run_id": d.name,
            "artifact_dir": str(d.relative_to(REPO)),
            "dataset": m.group("dataset"),
            "task": m.group("task"),
            "model": m.group("model"),
            "seed": int(m.group("seed")),
            "seconds_fit_and_evaluate": met.get("seconds_fit_and_evaluate"),
            "parameters_cfg": json.dumps({k: v for k, v in cfg.items() if k in
                                          ("n_estimators", "max_depth", "bounded_trees", "protocol")}),
            "val_macro_f1": met.get("val", {}).get("macro_f1"),
            "test_macro_f1": met.get("test", {}).get("macro_f1"),
            "test_weighted_f1": met.get("test", {}).get("weighted_f1"),
            "test_accuracy": met.get("test", {}).get("accuracy"),
        })
    return rows


def main() -> int:
    reg, curves = collect_gnn()
    tab = collect_tabular()
    df = pd.DataFrame(reg).sort_values(["dataset", "task", "model", "seed"]).reset_index(drop=True)
    cf = pd.DataFrame(curves).sort_values(["run_id", "eval_index"]).reset_index(drop=True)
    td = pd.DataFrame(tab).sort_values(["dataset", "seed"]).reset_index(drop=True)

    df.to_csv(OUT / "run_registry.csv", index=False)
    cf.to_csv(OUT / "learning_curves.csv", index=False)
    td.to_csv(OUT / "tabular_registry.csv", index=False)

    audit: dict = {"n_gnn_runs": int(len(df)), "n_tabular_runs": int(len(td))}

    # --- completeness of the 4x2x3x5 design -------------------------------
    cells = df.groupby(["dataset", "task", "model"]).size()
    audit["design_cells"] = int(len(cells))
    audit["seeds_per_cell_complete"] = bool((cells == 5).all())
    audit["cell_sizes"] = {f"{a}|{b}|{c}": int(v) for (a, b, c), v in cells.items()}

    # --- published runs.csv must match per-run metrics.json ---------------
    pub = pd.read_csv(RES / "results/full_5seed/runs.csv")
    audit["published_runs_rows"] = int(len(pub))
    key = ["dataset", "task", "model", "seed"]
    merged = pub.merge(df, on=key, suffixes=("_pub", "_met"), how="outer", indicator=True)
    audit["published_vs_artifact_merge"] = merged["_merge"].value_counts().to_dict()
    diffs = {}
    for col in ["test_macro_f1", "test_weighted_f1", "test_accuracy", "val_macro_f1",
                "seconds_fit_and_evaluate", "best_step", "steps_ran"]:
        a, b = merged[f"{col}_pub"], merged[f"{col}_met"]
        both = a.notna() & b.notna()
        d = (a[both].astype(float) - b[both].astype(float)).abs()
        diffs[col] = {"max_abs_diff": float(d.max()) if len(d) else None, "n": int(both.sum())}
    audit["published_vs_artifact_max_abs_diff"] = diffs

    # --- recompute summary.csv stats from runs.csv ------------------------
    rec = pub.groupby(["dataset", "task", "model"]).agg(
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", lambda s: s.std(ddof=1)),
    ).reset_index()
    pubsum = pd.read_csv(RES / "results/full_5seed/summary.csv")
    chk = pubsum.merge(rec, on=["dataset", "task", "model"], suffixes=("_file", "_recomputed"))
    audit["summary_mean_recompute_max_abs_diff"] = float(
        (chk["test_macro_f1_mean_file"] - chk["test_macro_f1_mean_recomputed"]).abs().max())
    audit["summary_std_recompute_max_abs_diff"] = float(
        (chk["test_macro_f1_std_file"] - chk["test_macro_f1_std_recomputed"]).abs().max())

    # --- 3-seed vs 5-seed std inflation (the key stability fact) ----------
    three = pub[pub.seed.isin([11, 22, 33])].groupby(["dataset", "task", "model"]).agg(
        mean3=("test_macro_f1", "mean"), std3=("test_macro_f1", lambda s: s.std(ddof=1))).reset_index()
    five = pub.groupby(["dataset", "task", "model"]).agg(
        mean5=("test_macro_f1", "mean"), std5=("test_macro_f1", lambda s: s.std(ddof=1))).reset_index()
    infl = three.merge(five, on=["dataset", "task", "model"])
    infl["std_ratio_5_over_3"] = infl["std5"] / infl["std3"].replace(0, np.nan)
    infl["mean_shift_5_minus_3"] = infl["mean5"] - infl["mean3"]
    infl.to_csv(OUT / "seed_inflation_3_vs_5.csv", index=False)
    audit["std_ratio_median"] = float(infl["std_ratio_5_over_3"].median())
    audit["std_ratio_max"] = float(infl["std_ratio_5_over_3"].max())
    audit["cells_where_std_grew_over_1p5x"] = int((infl["std_ratio_5_over_3"] > 1.5).sum())

    # --- verification file facts -----------------------------------------
    ver = json.loads((RES / "results/full_verification_5seed.json").read_text())
    audit["verification"] = {
        k: ver.get(k) for k in ["passed", "runs_checked", "largest_probability_replay_error",
                                "largest_metric_recomputation_error", "protocol_sha256",
                                "evaluation_mode", "probability_tolerance"]
    }
    audit["verification_artifact_sha256_count"] = len(ver.get("artifact_sha256", {}))

    # --- checkpoint parameter stability ----------------------------------
    audit["checkpoint_parameter_max_abs_error_all_zero"] = bool(
        (df["checkpoint_parameter_max_abs_error"].fillna(-1) == 0.0).all())

    # --- budget provenance: is the budget really pass-based? ---------------
    budget = df.groupby(["dataset", "task"]).agg(
        spp=("steps_per_full_pass", "first"),
        steps_ran_min=("steps_ran", "min"), steps_ran_max=("steps_ran", "max"),
        budget_min=("step_budget", "min"), budget_max=("step_budget", "max"),
        passes_min=("train_passes_configured", "min"),
        min_steps=("min_train_steps", "first"),
    ).reset_index()
    budget["budget_binding"] = np.where(
        budget["steps_ran_max"] <= budget["min_steps"], "min_train_steps", "dataset_passes")
    budget["effective_passes"] = budget["steps_ran_max"] / budget["spp"]
    budget.to_csv(OUT / "budget_provenance.csv", index=False)
    audit["budget_binding_by_cell"] = {
        f"{r.dataset}|{r.task}": r.budget_binding for r in budget.itertuples()}

    (OUT / "audit_inventory.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps(audit, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
