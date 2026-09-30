#!/usr/bin/env python3
"""B4 (TN-3) - Endpoint-disjoint split construction (Gate C, CPU-only part).

Builds a split in which no endpoint appears in more than one of train/val/test.
This is what the archived flow_group_id split cannot provide: in that split
92.7-100 % of the validation/test endpoints already occur in train
(see results/full_prepare_reproduction.json), so it measures within-environment
flow classification only.

Method
------
1. DuckDB densifies the distinct endpoints to int64 ids (no python dicts).
2. Flows become (src_id, dst_id) pairs; a flow links its two endpoints.
3. Union-find (vectorised, path-halving) gives connected components, so two hosts
   that ever talk share a component and therefore a split.
4. Component -> split with the same discipline as the locked protocol:
   hash(component_id, 20260920) % 10 -> 0-6 train, 7 val, 8-9 test.
5. Write split=... Parquet, then verify endpoint disjointness, class coverage and
   row conservation.

Endpoint mode `ipport` uses IP+port; mode `ip` uses the IP address only (a stricter
host-level holdout). Writes only into the working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "data/processed_four"
OUT = HERE / "data/endpoint_splits"
RES = HERE / "results"
DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
SPLIT_SEED = 20260920


def connected_components(n_nodes: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component id per node using scipy's C implementation.

    A python/NumPy union-find over tens of millions of edges is far too slow, so
    the same connected-component semantics are obtained from a sparse adjacency
    matrix via scipy.sparse.csgraph.connected_components.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as cc
    data = np.ones(len(a), dtype=np.int8)
    g = coo_matrix((data, (a, b)), shape=(n_nodes, n_nodes))
    n_comp, labels = cc(g, directed=False, return_labels=True)
    return labels.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--endpoint-mode", choices=["ipport", "ip"], default="ipport")
    ap.add_argument("--strategy", choices=["holdout", "flowhash", "component"],
                    default="holdout",
                    help="holdout: endpoints split train/holdout (70/30); a flow is kept "
                         "only when both endpoints share a group, and holdout flows are "
                         "split into val/test by hashed flow group. flowhash: 70/10/20 "
                         "endpoint groups. component: connected-component holdout")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--memory-limit", default="8GB")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    report_path = RES / f"endpoint_split_report_{args.endpoint_mode}_{args.strategy}.json"
    report: dict = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"split_seed": SPLIT_SEED, "endpoint_mode": args.endpoint_mode, "strategy": args.strategy,
                   "split_rule": "hash(component_id, seed) % 10: 0-6 train, 7 val, 8-9 test"})
    report.setdefault("datasets", {})

    for dataset in args.datasets:
        if args.resume and dataset in report["datasets"]:
            print("skip (done)", dataset, flush=True)
            continue
        src = SRC / f"{dataset}.parquet"
        ds_out = OUT / f"{dataset}__{args.endpoint_mode}__{args.strategy}"
        if ds_out.exists():
            print("skip (exists)", ds_out, flush=True)
            continue
        t0 = time.perf_counter()
        con = duckdb.connect()
        con.execute("SET threads=?", [args.threads])
        con.execute("SET memory_limit=?", [args.memory_limit])
        con.execute("SET preserve_insertion_order=false")
        tmp = ds_out.with_name(ds_out.name + "_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")

        if args.endpoint_mode == "ipport":
            skey = "IPV4_SRC_ADDR || ':' || CAST(L4_SRC_PORT AS VARCHAR)"
            dkey = "IPV4_DST_ADDR || ':' || CAST(L4_DST_PORT AS VARCHAR)"
        else:
            skey, dkey = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"

        print(f"[{dataset}] densify endpoints ...", flush=True)
        con.execute(f"""
            CREATE TEMP TABLE ep AS
            SELECT e, CAST(row_number() OVER (ORDER BY e) - 1 AS BIGINT) AS eid
            FROM (SELECT DISTINCT e FROM (
                    SELECT {skey} e FROM read_parquet('{src.as_posix()}')
                    UNION SELECT {dkey} e FROM read_parquet('{src.as_posix()}')))
        """)
        n_ep = con.execute("SELECT count(*) FROM ep").fetchone()[0]
        print(f"[{dataset}] endpoints={n_ep:,d} ({time.perf_counter()-t0:.0f}s)", flush=True)

        if args.strategy == "holdout":
            # Two endpoint groups: train (70 %) and holdout (30 %). A flow survives
            # only when both endpoints are in the same group, so no holdout endpoint
            # ever occurs in train. Holdout flows are then split 1/3 validation,
            # 2/3 test by hashed flow group. val and test therefore share endpoints
            # with each other but never with train; this is stated explicitly.
            print(f"[{dataset}] assign endpoint groups (train/holdout) ...", flush=True)
            con.execute(f"""
                CREATE TEMP TABLE ep_split AS
                SELECT e AS endpoint,
                       CASE WHEN (hash(e, {SPLIT_SEED}) % 10) < 7 THEN 'train'
                            ELSE 'holdout' END AS grp
                FROM ep
            """)
            print(f"[{dataset}] writing split parquet (holdout) ...", flush=True)
            ds_out.mkdir(parents=True)
            con.execute(f"""
                COPY (
                    SELECT r.*,
                           CASE WHEN ls.grp = 'train' THEN 'train'
                                WHEN (hash(CAST(r.flow_group_id AS VARCHAR), {SPLIT_SEED}) % 3) = 0
                                     THEN 'val'
                                ELSE 'test' END AS split
                    FROM read_parquet('{src.as_posix()}') r
                    JOIN ep_split ls ON ls.endpoint = {skey}
                    JOIN ep_split ld ON ld.endpoint = {dkey}
                    WHERE ls.grp = ld.grp
                ) TO '{ds_out.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY (split), COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """)
            all_glob = (ds_out / "split=*" / "*.parquet").as_posix()
            rows = dict(con.execute(
                f"SELECT split, count(*) FROM read_parquet('{all_glob}', hive_partitioning=true) "
                f"GROUP BY split").fetchall())
            src_rows = con.execute("SELECT count(*) FROM read_parquet(?)",
                                   [str(src)]).fetchone()[0]
            leak = con.execute(f"""
                WITH tr AS (SELECT {skey} k FROM read_parquet('{ds_out.as_posix()}/split=train/*.parquet')
                            UNION SELECT {dkey} k FROM read_parquet('{ds_out.as_posix()}/split=train/*.parquet')),
                     ho AS (SELECT {skey} k FROM read_parquet('{ds_out.as_posix()}/split=test/*.parquet')
                            UNION SELECT {dkey} k FROM read_parquet('{ds_out.as_posix()}/split=test/*.parquet')
                            UNION SELECT {skey} k FROM read_parquet('{ds_out.as_posix()}/split=val/*.parquet')
                            UNION SELECT {dkey} k FROM read_parquet('{ds_out.as_posix()}/split=val/*.parquet'))
                SELECT count(*) FROM (SELECT k FROM ho INTERSECT SELECT k FROM tr)
            """).fetchone()[0]
            classes = {sp: set(x[0] for x in con.execute(
                f"SELECT DISTINCT Attack FROM read_parquet('{ds_out.as_posix()}/split={sp}/*.parquet')"
            ).fetchall()) for sp in ("train", "val", "test")}
            info = {
                "strategy": "holdout",
                "source_rows": int(src_rows),
                "retained_rows": int(sum(rows.values())),
                "dropped_rows": int(src_rows - sum(rows.values())),
                "dropped_fraction": float((src_rows - sum(rows.values())) / src_rows),
                "split_rows": {k: int(v) for k, v in rows.items()},
                "n_endpoints": int(n_ep),
                "holdout_endpoints_also_in_train": int(leak),
                "row_conservation_ok": bool(sum(rows.values()) <= src_rows),
                "classes_in_train_not_in_split": {
                    sp: sorted(classes["train"] - classes[sp]) for sp in classes},
                "classes_total": len(classes["train"]),
                "note": "val and test share endpoints with each other; neither shares "
                        "endpoints with train",
                "elapsed_seconds": round(time.perf_counter() - t0, 1),
            }
            report["datasets"][dataset] = info
            report_path.write_text(json.dumps(report, indent=2))
            print(json.dumps({dataset: info}, indent=1), flush=True)
            con.close()
            for f in tmp.rglob("*"):
                if f.is_file():
                    f.unlink()
            tmp.rmdir()
            continue

        if args.strategy == "flowhash":
            # Endpoint-level disjoint holdout: an endpoint is assigned to exactly one
            # group; a flow is retained only when both endpoints share that group, so
            # no validation/test endpoint can occur in train. Mixed flows are dropped
            # and the dropped fraction is reported.
            print(f"[{dataset}] assign endpoint groups ...", flush=True)
            con.execute(f"""
                CREATE TEMP TABLE ep_split AS
                SELECT e AS endpoint,
                       CASE WHEN (hash(e, {SPLIT_SEED}) % 10) < 7 THEN 'train'
                            WHEN (hash(e, {SPLIT_SEED}) % 10) = 7 THEN 'val'
                            ELSE 'test' END AS split
                FROM ep
            """)
            print(f"[{dataset}] writing split parquet (flowhash) ...", flush=True)
            ds_out.mkdir(parents=True)
            con.execute(f"""
                COPY (
                    SELECT r.*, ls.split AS split
                    FROM read_parquet('{src.as_posix()}') r
                    JOIN ep_split ls ON ls.endpoint = {skey}
                    JOIN ep_split ld ON ld.endpoint = {dkey}
                    WHERE ls.split = ld.split
                ) TO '{ds_out.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY (split), COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """)
            all_glob = (ds_out / "split=*" / "*.parquet").as_posix()
            rows = dict(con.execute(
                f"SELECT split, count(*) FROM read_parquet('{all_glob}', hive_partitioning=true) "
                f"GROUP BY split").fetchall())
            src_rows = con.execute("SELECT count(*) FROM read_parquet(?)",
                                   [str(src)]).fetchone()[0]
            leak = con.execute(f"""
                WITH e AS (
                  SELECT split, {skey} k FROM read_parquet('{all_glob}', hive_partitioning=true)
                  UNION ALL
                  SELECT split, {dkey} k FROM read_parquet('{all_glob}', hive_partitioning=true))
                SELECT count(*) FROM (SELECT k FROM e GROUP BY k HAVING count(DISTINCT split) > 1)
            """).fetchone()[0]
            classes = {sp: set(x[0] for x in con.execute(
                f"SELECT DISTINCT Attack FROM read_parquet('{ds_out.as_posix()}/split={sp}/*.parquet')"
            ).fetchall()) for sp in ("train", "val", "test")}
            info = {
                "strategy": "flowhash",
                "source_rows": int(src_rows),
                "retained_rows": int(sum(rows.values())),
                "dropped_rows": int(src_rows - sum(rows.values())),
                "dropped_fraction": float((src_rows - sum(rows.values())) / src_rows),
                "split_rows": {k: int(v) for k, v in rows.items()},
                "n_endpoints": int(n_ep),
                "endpoints_in_multiple_splits": int(leak),
                "row_conservation_ok": bool(sum(rows.values()) <= src_rows),
                "classes_in_train_not_in_split": {
                    sp: sorted(classes["train"] - classes[sp]) for sp in classes},
                "classes_total": len(classes["train"]),
                "elapsed_seconds": round(time.perf_counter() - t0, 1),
            }
            report["datasets"][dataset] = info
            report_path.write_text(json.dumps(report, indent=2))
            print(json.dumps({dataset: info}, indent=1), flush=True)
            con.close()
            for f in tmp.rglob("*"):
                if f.is_file():
                    f.unlink()
            tmp.rmdir()
            continue

        print(f"[{dataset}] build flow pair table ...", flush=True)
        con.execute(f"""
            CREATE TEMP TABLE flow_pairs AS
            SELECT s.eid AS s_id, d.eid AS d_id
            FROM read_parquet('{src.as_posix()}') r
            JOIN ep s ON s.e = {skey}
            JOIN ep d ON d.e = {dkey}
        """)
        n_flows = con.execute("SELECT count(*) FROM flow_pairs").fetchone()[0]
        arr = con.execute("SELECT s_id, d_id FROM flow_pairs").fetchnumpy()
        a = np.asarray(arr["s_id"], dtype=np.int64)
        b = np.asarray(arr["d_id"], dtype=np.int64)
        del arr
        con.execute("DROP TABLE flow_pairs")
        print(f"[{dataset}] union-find over {n_flows:,d} flows ...", flush=True)
        comp = connected_components(int(n_ep), a, b)
        n_comp = int(comp.max()) + 1
        print(f"[{dataset}] components={n_comp:,d} ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        del a, b

        # component -> split, deterministic and independent of row order
        h = np.array([int.from_bytes(hashlib.sha256(f"{int(c)}|{SPLIT_SEED}".encode())
                                     .digest()[:8], "little") % 10 for c in range(n_comp)],
                     dtype=np.int8)
        split_code = np.where(h < 7, 0, np.where(h == 7, 1, 2)).astype(np.int8)

        # Register the numpy results as Arrow so the join stays vectorised.
        split_names = np.array(["train", "val", "test"], dtype=object)[split_code]
        tbl_comp = pa.table({
            "comp": pa.array(np.arange(n_comp, dtype=np.int64)),
            "split": pa.array([str(x) for x in split_names], type=pa.string()),
        })
        tbl_epcomp = pa.table({
            "eid": pa.array(np.arange(int(n_ep), dtype=np.int64)),
            "comp": pa.array(comp.astype(np.int64)),
        })
        con.register("arrow_comp", tbl_comp)
        con.register("arrow_epcomp", tbl_epcomp)
        con.execute("CREATE TEMP TABLE comp_split AS SELECT * FROM arrow_comp")
        con.execute("CREATE TEMP TABLE ep_comp AS SELECT * FROM arrow_epcomp")
        con.execute("""
            CREATE TEMP TABLE ep_split AS
            SELECT e.e AS endpoint, c.split AS split
            FROM ep e JOIN ep_comp ec ON ec.eid = e.eid
                      JOIN comp_split c ON c.comp = ec.comp
        """)
        print(f"[{dataset}] writing split parquet ...", flush=True)
        con.execute(f"""
            COPY (
                SELECT r.*, COALESCE(ls.split, ld.split) AS split
                FROM read_parquet('{src.as_posix()}') r
                LEFT JOIN ep_split ls ON ls.endpoint = {skey}
                LEFT JOIN ep_split ld ON ld.endpoint = {dkey}
            ) TO '{ds_out.as_posix()}'
            (FORMAT PARQUET, PARTITION_BY (split), COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """)

        all_glob = (ds_out / "split=*" / "*.parquet").as_posix()
        rows = dict(con.execute(
            f"SELECT split, count(*) FROM read_parquet('{all_glob}', hive_partitioning=true) "
            f"GROUP BY split").fetchall())
        src_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(src)]).fetchone()[0]
        leak = con.execute(f"""
            WITH e AS (
              SELECT split, {skey} k FROM read_parquet('{all_glob}', hive_partitioning=true)
              UNION ALL
              SELECT split, {dkey} k FROM read_parquet('{all_glob}', hive_partitioning=true))
            SELECT count(*) FROM (SELECT k FROM e GROUP BY k HAVING count(DISTINCT split) > 1)
        """).fetchone()[0]
        classes = {sp: set(x[0] for x in con.execute(
            f"SELECT DISTINCT Attack FROM read_parquet('{ds_out.as_posix()}/split={sp}/*.parquet')"
        ).fetchall()) for sp in ("train", "val", "test")}
        missing = {sp: sorted(classes["train"] - classes[sp]) for sp in classes}
        info = {
            "source_rows": int(src_rows), "split_rows": {k: int(v) for k, v in rows.items()},
            "n_endpoints": int(n_ep), "n_components": int(n_comp),
            "endpoints_in_multiple_splits": int(leak),
            "row_conservation_ok": bool(sum(rows.values()) == src_rows),
            "classes_in_train_not_in_split": missing,
            "classes_total": len(classes["train"]),
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }
        report["datasets"][dataset] = info
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps({dataset: info}, indent=1), flush=True)
        con.close()
        for f in tmp.rglob("*"):
            if f.is_file():
                f.unlink()
        tmp.rmdir()

    print("done ->", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
