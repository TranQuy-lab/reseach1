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


def prepare(source: Path, output: Path, report: Path, threads: int, seed: int,
            min_class_rows: int = 30) -> dict:
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
        "protocol": "PROTOCOL_PAPER_EXTENSION_VI.md",
        "assignment": (
            "hash(ip || seed) % 10; retain flows only when src/dst are assigned "
            "to the same train/val/test split"
        ),
        "seed": seed, "minimum_class_rows_per_split": min_class_rows,
        "source": {}, "datasets": {}, "eligible_dataset_tasks": [],
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
            source_attack_rows = {
                str(name): int(count) for name, count in con.execute(
                    "SELECT CAST(Attack AS VARCHAR), count(*) FROM read_parquet(?) GROUP BY Attack",
                    [str(src)],
                ).fetchall()
            }
            source_binary_rows = {
                str(int(label)): int(count) for label, count in con.execute(
                    "SELECT Label, count(*) FROM read_parquet(?) GROUP BY Label",
                    [str(src)],
                ).fetchall()
            }
            result["source"][dataset] = {"path": str(src), "rows": source_rows, "sha256": sha256_file(src)}
            # Use one shared deterministic assignment for both endpoint roles.
            # Endpoints need the same split, not the exact same 0--9 bucket;
            # requiring bucket equality would unnecessarily discard ~90%.
            bucket_src = endpoint_bucket(seed, "IPV4_SRC_ADDR")
            bucket_dst = endpoint_bucket(seed, "IPV4_DST_ADDR")
            source_split = split_label(bucket_src)
            destination_split = split_label(bucket_dst)
            query = f"""
                COPY (
                    SELECT *, {source_split} AS split
                    FROM read_parquet('{quote(src)}')
                    WHERE {source_split} = {destination_split}
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
            attack_support = {split: {} for split in ("train", "val", "test")}
            for split, attack, count in con.execute(
                "SELECT split, CAST(Attack AS VARCHAR), count(*) "
                "FROM read_parquet(?, hive_partitioning=true) "
                "GROUP BY split, Attack",
                [str(glob)],
            ).fetchall():
                attack_support[str(split)][str(attack)] = int(count)
            binary_support = {split: {} for split in ("train", "val", "test")}
            for split, label, count in con.execute(
                "SELECT split, Label, count(*) "
                "FROM read_parquet(?, hive_partitioning=true) "
                "GROUP BY split, Label",
                [str(glob)],
            ).fetchall():
                binary_support[str(split)][str(int(label))] = int(count)
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
            missing_binary = {
                split: [label for label in sorted(source_binary_rows)
                        if binary_support[split].get(label, 0) < min_class_rows]
                for split in ("train", "val", "test")
            }
            missing_multiclass = {
                split: [label for label in sorted(source_attack_rows)
                        if attack_support[split].get(label, 0) < min_class_rows]
                for split in ("train", "val", "test")
            }
            binary_eligible = not any(missing_binary.values())
            multiclass_eligible = not any(missing_multiclass.values())
            if binary_eligible:
                result["eligible_dataset_tasks"].append({"dataset": dataset, "task": "binary"})
            if multiclass_eligible:
                result["eligible_dataset_tasks"].append({"dataset": dataset, "task": "multiclass"})
            result["datasets"][dataset] = {
                "source_rows": source_rows, "kept_rows": kept,
                "dropped_rows": source_rows - kept, "split_rows": {k: int(v) for k, v in split_rows.items()},
                "cross_split_endpoint_ips": overlap,
                "retained_fraction": kept / source_rows if source_rows else 0.0,
                "source_attack_support": source_attack_rows,
                "source_binary_support": source_binary_rows,
                "split_attack_support": attack_support,
                "split_binary_support": binary_support,
                "retained_attack_fraction": {
                    label: sum(attack_support[split].get(label, 0)
                               for split in ("train", "val", "test")) / count
                    for label, count in source_attack_rows.items()
                },
                "task_eligibility": {
                    "binary": {"eligible": binary_eligible, "below_minimum": missing_binary},
                    "multiclass": {
                        "eligible": multiclass_eligible,
                        "below_minimum": missing_multiclass,
                    },
                },
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
    parser.add_argument("--min-class-rows", type=int, default=30)
    args = parser.parse_args()
    if args.threads < 1 or args.min_class_rows < 1:
        parser.error("threads and min-class-rows must be positive")
    prepare(
        args.source, args.output, args.report, args.threads, args.seed,
        args.min_class_rows,
    )


if __name__ == "__main__":
    main()
