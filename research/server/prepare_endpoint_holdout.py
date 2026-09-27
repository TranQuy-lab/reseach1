"""Prepare deterministic endpoint-disjoint splits for generalization analysis.

A flow is retained only when both endpoint IPs are assigned to the same split.
This is intentionally a separate protocol: it must never overwrite
``data/full_splits`` or be mixed into the primary 72-run results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import duckdb

from nids_minibatch.schema import DATASETS, REQUIRED, SPLIT_RANGES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quote(path: Path) -> str:
    return str(path).replace("'", "''")


def endpoint_bucket(seed: int, column: str) -> str:
    if column not in {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}:
        raise ValueError(f"unsupported endpoint column: {column}")
    return f"hash(CAST({column} AS VARCHAR) || '|{int(seed)}') % 10"


def split_label(bucket: str) -> str:
    return (
        f"CASE WHEN {bucket} BETWEEN 0 AND 6 THEN 'train' "
        f"WHEN {bucket} = 7 THEN 'val' ELSE 'test' END"
    )


def prepare(source: Path, output: Path, report: Path, threads: int, seed: int) -> dict:
    source, output, report = Path(source), Path(output), Path(report)
    if output.exists():
        raise ValueError(f"Output exists; refusing overwrite: {output}")
    output.mkdir(parents=True)
    temp = output / "_duckdb_tmp"
    temp.mkdir()
    con = duckdb.connect()
    con.execute("SET threads=?", [threads])
    con.execute("SET memory_limit='16GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{quote(temp)}'")
    con.execute("SET max_temp_directory_size='100GB'")
    started = time.perf_counter()
    result = {
        "manifest_schema_version": 1,
        "protocol": "ENDPOINT_HOLDOUT_VI.md",
        "assignment": "hash(ip || seed) % 10; retain only flows whose src/dst buckets agree",
        "seed": seed, "source": {}, "datasets": {},
        "software": {"python": platform.python_version(), "duckdb": duckdb.__version__},
    }
    try:
        for dataset in DATASETS:
            src = source / f"{dataset}.parquet"
            if not src.is_file():
                raise FileNotFoundError(src)
            columns = set(con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(src)]).fetchdf().column_name)
            missing = set(REQUIRED) - columns
            if missing:
                raise ValueError(f"{dataset}: missing required columns {sorted(missing)}")
            dataset_out = output / dataset
            dataset_out.mkdir()
            source_rows = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(src)]).fetchone()[0])
            result["source"][dataset] = {"path": str(src), "rows": source_rows, "sha256": sha256_file(src)}
            # Use one shared deterministic assignment for both endpoint roles.
            # Retaining only equal buckets gives strict split-disjointness for
            # every IP in the output; cross-bucket flows are reported dropped.
            bucket_src = endpoint_bucket(seed, "IPV4_SRC_ADDR")
            bucket_dst = endpoint_bucket(seed, "IPV4_DST_ADDR")
            split_expr = split_label(bucket_src)
            query = f"""
                COPY (
                    SELECT *, {split_expr} AS split
                    FROM read_parquet('{quote(src)}')
                    WHERE {bucket_src} = {bucket_dst}
                ) TO '{quote(dataset_out)}'
                (FORMAT PARQUET, PARTITION_BY (split), COMPRESSION ZSTD,
                 ROW_GROUP_SIZE 100000)
            """
            con.execute(query)
            glob = dataset_out / "split=*" / "*.parquet"
            split_rows = dict(con.execute(
                "SELECT split, count(*) FROM read_parquet(?, hive_partitioning=true) GROUP BY split",
                [str(glob)],
            ).fetchall())
            kept = sum(int(x) for x in split_rows.values())
            # Exact output-IP overlap check: every IP in a split must occur in
            # only that split. This is a hard gate, not a descriptive metric.
            overlap = int(con.execute(
                "WITH ips AS ("
                " SELECT split, CAST(IPV4_SRC_ADDR AS VARCHAR) ip FROM read_parquet(?, hive_partitioning=true)"
                " UNION ALL SELECT split, CAST(IPV4_DST_ADDR AS VARCHAR) ip FROM read_parquet(?, hive_partitioning=true)"
                ") SELECT count(*) FROM (SELECT ip FROM ips GROUP BY ip HAVING count(DISTINCT split)>1)",
                [str(glob), str(glob)],
            ).fetchone()[0])
            if overlap:
                raise AssertionError(f"{dataset}: endpoint overlap across output splits: {overlap}")
            result["datasets"][dataset] = {
                "source_rows": source_rows, "kept_rows": kept,
                "dropped_rows": source_rows - kept, "split_rows": {k: int(v) for k, v in split_rows.items()},
                "cross_split_endpoint_ips": overlap,
            }
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps({"dataset": dataset, "source_rows": source_rows,
                              "kept_rows": kept, "dropped_rows": source_rows - kept}), flush=True)
    finally:
        con.close()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed_four"))
    parser.add_argument("--output", type=Path, default=Path("data/endpoint_holdout_splits"))
    parser.add_argument("--report", type=Path, default=Path("research/results/endpoint_holdout_prepare.json"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    prepare(args.source, args.output, args.report, args.threads, args.seed)


if __name__ == "__main__":
    main()
