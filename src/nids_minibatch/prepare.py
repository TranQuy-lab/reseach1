"""Create deterministic group splits and bounded local benchmark samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from .schema import DATASETS, REQUIRED, SAMPLE_CAPS, SAMPLE_SEED, SPLIT_SEED


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def split_case(seed: int = SPLIT_SEED) -> str:
    bucket = f"hash(flow_group_id, {int(seed)}) % 10"
    return f"CASE WHEN {bucket} < 7 THEN 'train' WHEN {bucket} = 7 THEN 'val' ELSE 'test' END"


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def prepare(source: Path, output: Path, report: Path, threads: int = 2,
            protocol: str = "PROTOCOL_MINIBATCH_VI.md",
            memory_limit: str = "1GB") -> dict:
    source, output, report = Path(source), Path(output), Path(report)
    if output.exists():
        raise ValueError("Output exists; refusing overwrite")
    output.mkdir(parents=True)
    temp = output / "_duckdb_tmp"
    temp.mkdir()
    con = duckdb.connect()
    con.execute("SET threads=?", [threads])
    con.execute("SET memory_limit=?", [memory_limit])
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{_quote(temp)}'")
    con.execute("SET max_temp_directory_size='12GB'")
    started = time.perf_counter()
    result = {
        "manifest_schema_version": 2,
        "protocol": protocol,
        "split_seed": SPLIT_SEED,
        "sample_seed": SAMPLE_SEED,
        "split_rule": "DuckDB hash(flow_group_id, seed) % 10: 0-6 train, 7 val, 8-9 test",
        "sample_caps_per_attack": SAMPLE_CAPS,
        "source": {},
        "datasets": {},
        "software": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
        },
    }
    try:
        for dataset in DATASETS:
            src = source / f"{dataset}.parquet"
            if not src.is_file():
                raise FileNotFoundError(f"Missing source dataset: {dataset}")
            columns = set(con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(src)]).fetchdf().column_name)
            missing = set(REQUIRED) - columns
            if missing:
                raise ValueError(f"{dataset} missing required columns: {sorted(missing)}")
            dataset_out = output / dataset
            dataset_out.mkdir()
            source_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(src)]).fetchone()[0]
            result["source"][dataset] = {
                "path": src.relative_to(source.parent).as_posix(),
                "rows": source_rows,
                "sha256": sha256_file(src),
            }
            query = f"""
                COPY (
                    SELECT *, {split_case()} AS split
                    FROM read_parquet('{_quote(src)}')
                ) TO '{_quote(dataset_out)}'
                (FORMAT PARQUET, PARTITION_BY (split), COMPRESSION ZSTD,
                 ROW_GROUP_SIZE 100000)
            """
            con.execute(query)
            split_info = {}
            all_glob = dataset_out / "split=*" / "*.parquet"
            counts = con.execute(
                "SELECT split, Attack, count(*) n FROM read_parquet(?, hive_partitioning=true) "
                "GROUP BY split, Attack ORDER BY split, Attack", [str(all_glob)]
            ).fetchall()
            for split, attack, count in counts:
                split_info.setdefault(split, {"rows": 0, "classes": {}})
                split_info[split]["rows"] += count
                split_info[split]["classes"][attack] = count
            for split, cap in SAMPLE_CAPS.items():
                part_glob = dataset_out / f"split={split}" / "*.parquet"
                pilot = dataset_out / f"pilot_{split}.parquet"
                writer = None
                try:
                    for attack in sorted(split_info[split]["classes"]):
                        table = con.execute(
                            "SELECT * FROM read_parquet(?, hive_partitioning=true) "
                            "WHERE Attack=? ORDER BY hash(flow_group_id, source_row_id, ?) LIMIT ?",
                            [str(part_glob), attack, SAMPLE_SEED, cap],
                        ).fetch_arrow_table()
                        if writer is None:
                            writer = pq.ParquetWriter(pilot, table.schema, compression="zstd")
                        writer.write_table(table, row_group_size=100_000)
                finally:
                    if writer is not None:
                        writer.close()
                pilot_counts = dict(con.execute(
                    "SELECT Attack, count(*) FROM read_parquet(?) GROUP BY Attack ORDER BY Attack",
                    [str(pilot)],
                ).fetchall())
                split_info[split]["pilot_rows"] = sum(pilot_counts.values())
                split_info[split]["pilot_classes"] = pilot_counts
                split_info[split]["pilot_sha256"] = sha256_file(pilot)
            if set(split_info) != {"train", "val", "test"}:
                raise AssertionError(f"{dataset}: missing split")
            classes = [set(split_info[s]["classes"]) for s in ("train", "val", "test")]
            if classes[0] != classes[1] or classes[0] != classes[2]:
                raise ValueError(f"{dataset}: at least one split lacks a class")
            if sum(x["rows"] for x in split_info.values()) != source_rows:
                raise AssertionError(f"{dataset}: row conservation failed")
            overlap = con.execute(
                "SELECT count(*) FROM (SELECT flow_group_id FROM read_parquet(?, hive_partitioning=true) "
                "GROUP BY flow_group_id HAVING count(DISTINCT split)>1)", [str(all_glob)]
            ).fetchone()[0]
            if overlap:
                raise AssertionError(f"{dataset}: flow groups cross splits")
            conflicts = con.execute(
                "SELECT split, count(*), coalesce(sum(n), 0) FROM ("
                "SELECT split, flow_group_id, count(*) n FROM read_parquet(?, hive_partitioning=true) "
                "GROUP BY split, flow_group_id HAVING min(Attack) != max(Attack)) "
                "GROUP BY split", [str(all_glob)]
            ).fetchall()
            conflict_by_split = {
                split: {"conflicting_label_groups": int(groups),
                        "rows_in_conflicting_label_groups": int(rows)}
                for split, groups, rows in conflicts
            }
            for split in ("train", "val", "test"):
                split_info[split].update(conflict_by_split.get(split, {
                    "conflicting_label_groups": 0,
                    "rows_in_conflicting_label_groups": 0,
                }))
            ip_overlap = {}
            train_glob = dataset_out / "split=train" / "*.parquet"
            for split in ("val", "test"):
                heldout_glob = dataset_out / f"split={split}" / "*.parquet"
                unique_ips, seen_ips = con.execute(
                    "WITH train_ips AS ("
                    " SELECT IPV4_SRC_ADDR ip FROM read_parquet(?) UNION "
                    " SELECT IPV4_DST_ADDR ip FROM read_parquet(?)"
                    "), heldout_ips AS ("
                    " SELECT IPV4_SRC_ADDR ip FROM read_parquet(?) UNION "
                    " SELECT IPV4_DST_ADDR ip FROM read_parquet(?)"
                    ") SELECT count(*), count(*) FILTER (WHERE t.ip IS NOT NULL) "
                    "FROM heldout_ips h LEFT JOIN train_ips t USING (ip)",
                    [str(train_glob), str(train_glob),
                     str(heldout_glob), str(heldout_glob)],
                ).fetchone()
                ip_overlap[split] = {
                    "unique_holdout_ips": int(unique_ips),
                    "ips_also_in_train": int(seen_ips),
                    "fraction_also_in_train": float(seen_ips / unique_ips),
                }
            result["datasets"][dataset] = {
                "source_rows": source_rows,
                "split_rows": split_info,
                "cross_split_groups": overlap,
                "ip_overlap_with_train": ip_overlap,
            }
            write_json(report.with_name(report.stem + ".partial.json"), result)
            print(json.dumps({"dataset": dataset, "rows": source_rows,
                              "pilot_rows": {s: split_info[s]["pilot_rows"] for s in split_info}}), flush=True)
    finally:
        con.close()
    if temp.exists() and not any(temp.iterdir()):
        temp.rmdir()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    write_json(report, result)
    partial = report.with_name(report.stem + ".partial.json")
    if partial.exists():
        partial.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="1GB",
                        help="DuckDB memory limit; full-data server pipeline uses 16GB")
    parser.add_argument("--protocol", choices=[
        "PROTOCOL_MINIBATCH_VI.md", "PROTOCOL_FULL_DATA_VI.md",
    ], default="PROTOCOL_MINIBATCH_VI.md")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    prepare(args.source, args.output, args.report, args.threads, args.protocol,
            args.memory_limit)


if __name__ == "__main__":
    main()
