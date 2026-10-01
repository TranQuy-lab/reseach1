#!/usr/bin/env python3
"""TN-6 - Deployment-relevant error profile from the archived confusion matrices.

The repository never stored per-run test probabilities for the GNN runs (the files
are gitignored), so PR-AUC cannot be recovered. The full test confusion matrices
ARE stored, which is enough for the security-relevant quantity: at each model's own
operating point, what is the per-class false-positive (false-alarm) and false-negative
rate?

For class k with confusion matrix C (rows = true, columns = predicted):
    TP = C[k,k]
    FN = sum(C[k,:]) - TP
    FP = sum(C[:,k]) - TP
    TN = total - TP - FN - FP
    FPR_k = FP / (FP + TN)      false-alarm rate for class k
    FNR_k = FN / (FN + TP)      miss rate for class k

For the binary task the class named Benign gives the operational false-alarm rate of
the detector; for multiclass we additionally report the expected false alarms per
million flows.

Outputs: results/operational_metrics_per_run.csv,
         results/operational_metrics_summary.csv,
         results/operational_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/noble-tran/nghiencuu/repo_reseach1")
RES = REPO / "research"
OUT = Path(__file__).resolve().parents[1] / "results"
IGNORE = {"accuracy", "macro avg", "weighted avg"}
DATASETS = ["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"]


def main() -> int:
    reg = pd.read_csv(OUT / "run_registry.csv")
    rows = []
    for r in reg.itertuples():
        met = json.loads((REPO / r.artifact_dir / "metrics.json").read_text())
        for split in ("val", "test"):
            block = met.get(split, {})
            cm = block.get("confusion_matrix")
            per = block.get("per_class", {})
            classes = [c for c in per if c not in IGNORE]
            if not cm or not classes:
                continue
            C = np.asarray(cm, dtype=np.float64)
            if C.shape[0] != len(classes):
                continue
            total = C.sum()
            for k, cls in enumerate(classes):
                tp = C[k, k]
                fn = C[k, :].sum() - tp
                fp = C[:, k].sum() - tp
                tn = total - tp - fn - fp
                rows.append({
                    "dataset": r.dataset, "task": r.task, "model": r.model, "seed": r.seed,
                    "split": split, "class": cls,
                    "support": float(tp + fn),
                    "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
                    "recall_tpr": float(tp / (tp + fn)) if tp + fn else np.nan,
                    "fpr": float(fp / (fp + tn)) if fp + tn else np.nan,
                    "fnr": float(fn / (tp + fn)) if tp + fn else np.nan,
                    "fp_per_million": float(1e6 * fp / (fp + tn)) if fp + tn else np.nan,
                    "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
                })
        # binary detector-level false-alarm rate (class "Attack" predicted when Benign)
        if r.task == "binary":
            met2 = json.loads((REPO / r.artifact_dir / "metrics.json").read_text())
            cm = met2.get("test", {}).get("confusion_matrix")
            per = met2.get("test", {}).get("per_class", {})
            classes = [c for c in per if c not in IGNORE]
            if cm and "Benign" in classes:
                C = np.asarray(cm, dtype=np.float64)
                bi = classes.index("Benign")
                tn = C[bi, bi]
                fp = C[bi, :].sum() - tn
                rows.append({
                    "dataset": r.dataset, "task": "binary_detector", "model": r.model,
                    "seed": r.seed, "split": "test", "class": "false_alarm",
                    "support": float(C[bi, :].sum()),
                    "precision": np.nan, "recall_tpr": float(tn / C[bi, :].sum()),
                    "fpr": float(fp / C[bi, :].sum()),
                    "fnr": np.nan, "fp_per_million": float(1e6 * fp / C[bi, :].sum()),
                    "tp": float(tn), "fp": float(fp), "fn": np.nan, "tn": np.nan,
                })
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / "operational_metrics_per_run.csv", index=False)

    test = per_run[(per_run.split == "test") & (per_run.task != "binary_detector")]
    summ = test.groupby(["dataset", "task", "model", "class"]).agg(
        support=("support", "first"), fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
        fnr_mean=("fnr", "mean"), recall_mean=("recall_tpr", "mean"),
        fp_per_million_mean=("fp_per_million", "mean")).reset_index()
    summ.to_csv(OUT / "operational_metrics_summary.csv", index=False)

    # macro false-alarm rate per model/cell
    macro = test.groupby(["dataset", "task", "model"]).agg(
        macro_fpr=("fpr", "mean"), macro_fnr=("fnr", "mean"),
        worst_class_fpr=("fpr", "max")).reset_index()
    macro.to_csv(OUT / "operational_macro.csv", index=False)

    det = per_run[per_run.task == "binary_detector"]
    det_s = det.groupby(["dataset", "model"]).agg(
        false_alarm_rate=("fpr", "mean"), fp_per_million=("fp_per_million", "mean"),
        benign_recall=("recall_tpr", "mean")).reset_index()

    out = {
        "n_per_class_records": int(len(per_run)),
        "macro_false_alarm_by_cell": macro.round(5).to_dict(orient="records"),
        "binary_detector_false_alarm": det_s.round(5).to_dict(orient="records"),
        "worst_false_alarm_classes": summ.sort_values("fpr_mean", ascending=False)
            .head(12).round(5).to_dict(orient="records"),
        "note": "Derived from archived full-test confusion matrices at each model's own "
                "argmax operating point; the repository does not store GNN probabilities, "
                "so PR-AUC is not recoverable.",
    }
    (OUT / "operational_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
