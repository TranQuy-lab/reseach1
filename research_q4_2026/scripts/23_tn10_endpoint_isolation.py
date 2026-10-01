#!/usr/bin/env python3
"""TN-10 - Isolating endpoint familiarity correctly.

Why the previous comparison was confounded
------------------------------------------
The endpoint-disjoint split changes several things at once: which endpoints appear in
train, the size and composition of the training set, the label distribution
(total-variation distance 0.293 on NF-ToN-IoT-v2 multiclass) and which flows survive
at all (only 63% are retained). TN-9 showed that a HistGradientBoosting model - which
never observes endpoint identity as a feature - also falls from 0.866 to 0.472 on that
split, proving that most of the apparent "endpoint familiarity" effect is actually
distribution shift created by the split construction.

The clean design
----------------
Hold everything fixed and vary only the test subset:
  * train on the LOCKED train split with the LOCKED scaler (unchanged);
  * score the LOCKED test split, partitioned by whether each flow's endpoints were
    present in train:
        all test flows
        both endpoints seen in train
        neither endpoint seen in train   <- the endpoint-disjoint subset
        exactly one endpoint seen
Because the model, the training data, the preprocessing and the overall test
population are identical, any difference between the subsets is attributable to
endpoint familiarity and nothing else.

Outputs: results/tn10_endpoint_subset_flags.npz (aligned to the test memmap),
         results/tn10_endpoint_isolation.csv,
         results/tn10_endpoint_isolation.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parents[1]
SPLITS = HERE / "data/full_splits"
FEAT = HERE / "work/features"
OUT = HERE / "results"
S = "IPV4_SRC_ADDR || ':' || CAST(L4_SRC_PORT AS VARCHAR)"
D = "IPV4_DST_ADDR || ':' || CAST(L4_DST_PORT AS VARCHAR)"
GRID = [
    {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_iter": 400, "max_leaf_nodes": 63},
]


def _files(ds: str, split: str):
    return sorted((SPLITS / ds / f"split={split}").glob("*.parquet"))


def endpoint_seen_flags(ds: str, threads: int = 8) -> Path:
    """Per-test-row booleans (src seen, dst seen) aligned to the feature memmap."""
    dest = OUT / f"tn10_flags_{ds}.npz"
    if dest.exists():
        return dest
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=?", [threads])
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp = OUT / f"tn10_tmp_{ds}"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    train = (SPLITS / ds / "split=train" / "*.parquet").as_posix()
    con.execute(f"""
        CREATE TEMP TABLE train_ep AS
        SELECT {S} AS ep FROM read_parquet('{train}')
        UNION SELECT {D} AS ep FROM read_parquet('{train}')
    """)
    con.execute("CREATE INDEX idx_ep ON train_ep(ep)")
    print(f"  [{ds}] train endpoint set built", flush=True)
    n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in _files(ds, "test"))
    src_seen = np.zeros(n_total, dtype=bool)
    dst_seen = np.zeros(n_total, dtype=bool)
    pos = 0
    for f in _files(ds, "test"):
        q = f"""
            SELECT (t1.ep IS NOT NULL) AS s_seen, (t2.ep IS NOT NULL) AS d_seen
            FROM read_parquet('{f.as_posix()}') r
            LEFT JOIN train_ep t1 ON t1.ep = {S}
            LEFT JOIN train_ep t2 ON t2.ep = {D}
        """
        tbl = con.execute(q).fetchnumpy()
        n = len(tbl["s_seen"])
        src_seen[pos:pos + n] = np.asarray(tbl["s_seen"], dtype=bool)
        dst_seen[pos:pos + n] = np.asarray(tbl["d_seen"], dtype=bool)
        pos += n
    assert pos == n_total
    np.savez(dest, src_seen=src_seen, dst_seen=dst_seen)
    con.close()
    for f in tmp.rglob("*"):
        if f.is_file():
            f.unlink()
    tmp.rmdir()
    return dest


def _write_summary() -> None:
    """Regenerate the summary from the raw file so a timeout cannot lose it."""
    raw = OUT / "tn10_endpoint_isolation.csv"
    if not raw.exists():
        return
    df = pd.read_csv(raw)
    agg = df.groupby(["dataset", "task", "subset"]).agg(
        n_seeds=("seed", "size"), n_flows=("n_flows", "first"),
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        weighted_f1=("weighted_f1", "mean")).reset_index()
    base = agg[agg.subset == "all_test"].set_index(["dataset", "task"]).macro_f1
    agg["delta_vs_all_test"] = [
        r.macro_f1 - base.loc[(r.dataset, r.task)] for r in agg.itertuples()]
    agg.to_csv(OUT / "tn10_endpoint_isolation_summary.csv", index=False)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NF-ToN-IoT-v2"])
    ap.add_argument("--tasks", nargs="+", default=["multiclass"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--n-jobs", type=int, default=14)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        flags = endpoint_seen_flags(ds)
        z = np.load(flags)
        src_seen, dst_seen = z["src_seen"], z["dst_seen"]
        both = src_seen & dst_seen
        neither = (~src_seen) & (~dst_seen)
        one = src_seen ^ dst_seen
        print(f"  [{ds}] test flows: both-seen={both.sum():,d} "
              f"neither-seen={neither.sum():,d} one-seen={one.sum():,d}", flush=True)
        for task in args.tasks:
            Xtr = np.load(FEAT / f"{ds}__train.npy", mmap_mode="r")
            Xva = np.load(FEAT / f"{ds}__val.npy", mmap_mode="r")
            Xte = np.load(FEAT / f"{ds}__test.npy", mmap_mode="r")
            ytr = np.load(FEAT / f"{ds}__train__y_{task}.npy")
            yva = np.load(FEAT / f"{ds}__val__y_{task}.npy")
            yte = np.load(FEAT / f"{ds}__test__y_{task}.npy")
            best_cfg, best_val = None, -np.inf
            for cfg in GRID:
                m = HistGradientBoostingClassifier(random_state=11, class_weight="balanced",
                                                   early_stopping=True,
                                                   validation_fraction=0.1,
                                                   n_iter_no_change=20, **cfg)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                v = float(f1_score(yva, m.predict(Xva), average="macro", zero_division=0))
                print(f"  tune {cfg} val={v:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
                if v > best_val:
                    best_cfg, best_val = cfg, v
                del m
            for seed in args.seeds:
                m = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced",
                                                   early_stopping=True,
                                                   validation_fraction=0.1,
                                                   n_iter_no_change=20, **best_cfg)
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                for label, mask in (("all_test", np.ones(len(yte), dtype=bool)),
                                    ("both_endpoints_seen", both),
                                    ("one_endpoint_seen", one),
                                    ("neither_endpoint_seen", neither)):
                    if mask.sum() < 100:
                        continue
                    rows.append({
                        "dataset": ds, "task": task, "model": "hist_gradient_boosting",
                        "seed": seed, "subset": label, "n_flows": int(mask.sum()),
                        "macro_f1": float(f1_score(yte[mask], pred[mask], average="macro",
                                                   zero_division=0)),
                        "weighted_f1": float(f1_score(yte[mask], pred[mask],
                                                      average="weighted", zero_division=0)),
                        "accuracy": float((yte[mask] == pred[mask]).mean()),
                    })
                pd.DataFrame(rows).to_csv(OUT / "tn10_endpoint_isolation.csv", index=False)
                _write_summary()
                print(f"[{time.strftime('%H:%M:%S')}] {ds} {task} s{seed} done", flush=True)
                del m
    agg = _write_summary()
    out = {"by_subset": agg.round(4).to_dict(orient="records"),
           "interpretation": "Model, training data, preprocessing and test population are "
                             "identical across subsets; only endpoint familiarity varies."}
    (OUT / "tn10_endpoint_isolation.json").write_text(json.dumps(out, indent=2, default=str))
    print(agg.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
