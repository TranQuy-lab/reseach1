#!/usr/bin/env python3
"""B6 - Endpoint-holdout feasibility analysis (Gate C evidence).

Compares the locked flow_group_id split with the endpoint-disjoint designs and
quantifies what the holdout costs: retained rows, class coverage and label shift.
This is the feasibility gate that must pass before any "unseen host" claim.

Outputs: results/endpoint_feasibility.json, results/endpoint_class_shift.csv,
         figures/fig7_endpoint_feasibility.png
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
SPLITS = HERE / "data/full_splits"
ENDP = HERE / "data/endpoint_splits"
FIG = HERE / "figures"
DATASETS = ["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"]
SHORT = {"NF-UNSW-NB15-v2": "UNSW", "NF-ToN-IoT-v2": "ToN",
         "NF-CSE-CIC-IDS2018-v2": "CSE-CIC", "NF-BoT-IoT-v2": "BoT-IoT"}


def class_counts(root: Path, dataset: str, mode: str) -> dict[str, dict[str, int]]:
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=4")
    glob = (root / dataset / "split=*" / "*.parquet").as_posix()
    if not (root / dataset).exists():
        return {}
    rows = con.execute(
        f"SELECT split, Attack, count(*) n FROM read_parquet('{glob}', hive_partitioning=true) "
        f"GROUP BY split, Attack").fetchall()
    con.close()
    out: dict[str, dict[str, int]] = {}
    for split, attack, n in rows:
        out.setdefault(split, {})[attack] = int(n)
    return out


def main() -> int:
    report_paths = sorted(RES.glob("endpoint_split_report_*.json"))
    report = {}
    for p in report_paths:
        report.update(json.loads(p.read_text()).get("datasets", {}))
    out = {"endpoint_split_reports": report, "comparison": {}}
    shift_rows = []

    locked = {d: class_counts(SPLITS, d, "locked") for d in DATASETS}
    holdout = {d: class_counts(ENDP, f"{d}__ipport__holdout", "holdout") for d in DATASETS}

    for d in DATASETS:
        lk, ho = locked.get(d), holdout.get(d)
        if not lk or not ho:
            continue
        lk_test = lk.get("test", {})
        ho_test = ho.get("test", {})
        lk_tot = sum(lk_test.values())
        ho_tot = sum(ho_test.values())
        classes = sorted(set(lk_test) | set(ho_test))
        # total-variation distance between the two test label distributions
        tv = 0.5 * sum(abs(lk_test.get(c, 0) / max(lk_tot, 1) - ho_test.get(c, 0) / max(ho_tot, 1))
                       for c in classes)
        out["comparison"][d] = {
            "locked_test_rows": lk_tot,
            "holdout_test_rows": ho_tot,
            "holdout_retained_fraction": (
                report[d]["retained_rows"] / report[d]["source_rows"]
                if d in report else None),
            "holdout_dropped_fraction": (
                report[d]["dropped_fraction"] if d in report else None),
            "classes_locked_test": len(lk_test),
            "classes_holdout_test": len(ho_test),
            "test_label_total_variation_distance": tv,
            "holdout_endpoints_in_train": (
                report[d].get("holdout_endpoints_also_in_train") if d in report else None),
        }
        for c in classes:
            shift_rows.append({
                "dataset": d, "class": c,
                "locked_share": lk_test.get(c, 0) / max(lk_tot, 1),
                "holdout_share": ho_test.get(c, 0) / max(ho_tot, 1),
                "locked_n": lk_test.get(c, 0), "holdout_n": ho_test.get(c, 0),
            })

    sh = pd.DataFrame(shift_rows)
    sh.to_csv(RES / "endpoint_class_shift.csv", index=False)
    (RES / "endpoint_feasibility.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["comparison"], indent=2, default=str))

    # ---- figure -----------------------------------------------------------
    have = [d for d in DATASETS if d in out["comparison"]]
    if have:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax = axes[0]
        x = np.arange(len(have))
        lk = [out["comparison"][d]["locked_test_rows"] for d in have]
        ho = [out["comparison"][d]["holdout_test_rows"] for d in have]
        ax.bar(x - 0.2, lk, width=0.4, label="locked flow-group split", color="#4C72B0")
        ax.bar(x + 0.2, ho, width=0.4, label="endpoint-disjoint holdout", color="#C44E52")
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in have])
        ax.set_ylabel("test flows (log)"); ax.legend(fontsize=8)
        ax.set_title("Test-set size under each split design")
        ax.grid(alpha=0.25, axis="y")

        ax = axes[1]
        tv = [out["comparison"][d]["test_label_total_variation_distance"] for d in have]
        rf = [out["comparison"][d]["holdout_retained_fraction"] or np.nan for d in have]
        ax.bar(x - 0.2, tv, width=0.4, label="label total-variation distance", color="#55A868")
        ax.bar(x + 0.2, rf, width=0.4, label="retained row fraction", color="#8172B2")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in have])
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        ax.set_title("Cost of the holdout: label shift and data discarded")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(FIG / "fig7_endpoint_feasibility.png", dpi=170)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
