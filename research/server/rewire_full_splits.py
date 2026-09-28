"""Create a graph-rewired copy of full splits for an ablation.

Only destination endpoint tuples are permuted within each existing split. Flow
features, labels, source endpoints, row IDs and group IDs remain attached to
those same rows. The output is a separate protocol and never overwrites
``data/full_splits``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import duckdb

from nids_minibatch.schema import DATASETS, FEATURES, REQUIRED


def quote(path: Path) -> str:
    return str(path).replace("'", "''")


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
        "protocol": "PROTOCOL_PAPER_EXTENSION_VI.md",
        "rewire": "destination endpoint tuple permuted within each original split",
        "seed": seed, "datasets": {},
        "software": {"python": platform.python_version(), "duckdb": duckdb.__version__},
    }
    try:
        for dataset in DATASETS:
            dataset_source = source / dataset
            if not dataset_source.is_dir():
                raise FileNotFoundError(dataset_source)
            dataset_out = output / dataset
            dataset_out.mkdir()
            result["datasets"][dataset] = {}
            for split in ("train", "val", "test"):
                glob = dataset_source / f"split={split}" / "*.parquet"
                if not list(glob.parent.glob(glob.name)):
                    raise FileNotFoundError(glob)
                columns = set(con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(glob)]).fetchdf().column_name)
                missing = set(REQUIRED) - columns
                if missing:
                    raise ValueError(f"{dataset}/{split}: missing {sorted(missing)}")
                out_folder = dataset_out / f"split={split}"
                out_folder.mkdir(parents=True)
                # Position rows by source_row_id, then pair them with a
                # separately hash-ordered destination pool. This preserves the
                # destination marginal and rewires row-to-endpoint relations.
                output_file = out_folder / "part-00000.parquet"
                query = f"""
                    COPY (
                        WITH rows AS (
                            SELECT *,
                                   row_number() OVER (ORDER BY source_row_id) AS __rewire_pos
                            FROM read_parquet('{quote(glob)}', hive_partitioning=true)
                        ),
                        pool AS (
                            SELECT row_number() OVER (
                                       ORDER BY hash(CAST(source_row_id AS VARCHAR) || '|{int(seed)}'), source_row_id
                                   ) AS __rewire_pos,
                                   IPV4_DST_ADDR AS __new_dst_ip,
                                   L4_DST_PORT AS __new_dst_port
                            FROM read_parquet('{quote(glob)}', hive_partitioning=true)
                        )
                        SELECT rows.* EXCLUDE (__rewire_pos, IPV4_DST_ADDR, L4_DST_PORT),
                               pool.__new_dst_ip AS IPV4_DST_ADDR,
                               pool.__new_dst_port AS L4_DST_PORT
                        FROM rows JOIN pool USING (__rewire_pos)
                    ) TO '{quote(output_file)}'
                    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
                con.execute(query)
                out_glob = out_folder / "*.parquet"
                count = int(con.execute(
                    "SELECT count(*) FROM read_parquet(?, hive_partitioning=true)",
                    [str(out_glob)],
                ).fetchone()[0])
                source_count = int(con.execute(
                    "SELECT count(*) FROM read_parquet(?, hive_partitioning=true)",
                    [str(glob)],
                ).fetchone()[0])
                if count != source_count:
                    raise AssertionError(
                        f"{dataset}/{split}: row count changed {source_count} -> {count}"
                    )
                duplicate_ids = int(con.execute(
                    "SELECT count(*) - count(DISTINCT source_row_id) "
                    "FROM read_parquet(?, hive_partitioning=true)",
                    [str(out_glob)],
                ).fetchone()[0])
                if duplicate_ids:
                    raise AssertionError(
                        f"{dataset}/{split}: duplicate source_row_id after rewiring"
                    )
                protected = [
                    "IPV4_SRC_ADDR", "L4_SRC_PORT", *FEATURES,
                    "Attack", "Label", "Dataset", "flow_group_id",
                ]
                mismatch_expr = " OR ".join(
                    f"s.{name} IS DISTINCT FROM o.{name}" for name in protected
                )
                protected_mismatches, changed_edges = con.execute(
                    f"""
                    SELECT
                        sum(CASE WHEN {mismatch_expr} THEN 1 ELSE 0 END),
                        sum(CASE WHEN s.IPV4_DST_ADDR IS DISTINCT FROM o.IPV4_DST_ADDR
                                      OR s.L4_DST_PORT IS DISTINCT FROM o.L4_DST_PORT
                                 THEN 1 ELSE 0 END)
                    FROM read_parquet(?, hive_partitioning=true) s
                    JOIN read_parquet(?, hive_partitioning=true) o USING (source_row_id)
                    """,
                    [str(glob), str(out_glob)],
                ).fetchone()
                protected_mismatches = int(protected_mismatches or 0)
                changed_edges = int(changed_edges or 0)
                if protected_mismatches:
                    raise AssertionError(
                        f"{dataset}/{split}: {protected_mismatches} protected rows changed"
                    )
                marginal_difference = int(con.execute(
                    """
                    SELECT count(*) FROM (
                        (SELECT IPV4_DST_ADDR, L4_DST_PORT
                         FROM read_parquet(?, hive_partitioning=true)
                         EXCEPT ALL
                         SELECT IPV4_DST_ADDR, L4_DST_PORT
                         FROM read_parquet(?, hive_partitioning=true))
                        UNION ALL
                        (SELECT IPV4_DST_ADDR, L4_DST_PORT
                         FROM read_parquet(?, hive_partitioning=true)
                         EXCEPT ALL
                         SELECT IPV4_DST_ADDR, L4_DST_PORT
                         FROM read_parquet(?, hive_partitioning=true))
                    )
                    """,
                    [str(glob), str(out_glob), str(out_glob), str(glob)],
                ).fetchone()[0])
                if marginal_difference:
                    raise AssertionError(
                        f"{dataset}/{split}: destination marginal was not preserved"
                    )
                source_self_loops = int(con.execute(
                    "SELECT count(*) FROM read_parquet(?, hive_partitioning=true) "
                    "WHERE IPV4_SRC_ADDR = IPV4_DST_ADDR AND L4_SRC_PORT = L4_DST_PORT",
                    [str(glob)],
                ).fetchone()[0])
                rewired_self_loops = int(con.execute(
                    "SELECT count(*) FROM read_parquet(?, hive_partitioning=true) "
                    "WHERE IPV4_SRC_ADDR = IPV4_DST_ADDR AND L4_SRC_PORT = L4_DST_PORT",
                    [str(out_glob)],
                ).fetchone()[0])
                result["datasets"][dataset][split] = {
                    "rows": count,
                    "protected_row_mismatches": protected_mismatches,
                    "destination_marginal_difference_rows": marginal_difference,
                    "changed_endpoint_rows": changed_edges,
                    "changed_endpoint_fraction": changed_edges / count if count else 0.0,
                    "source_self_loop_rows": source_self_loops,
                    "rewired_self_loop_rows": rewired_self_loops,
                }
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps({
                    "dataset": dataset, "split": split, "rows": count,
                    "changed_endpoint_fraction": changed_edges / count if count else 0.0,
                }), flush=True)
    finally:
        con.close()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/full_splits"))
    parser.add_argument("--output", type=Path, default=Path("data/rewired_full_splits"))
    parser.add_argument("--report", type=Path, default=Path("research/results/graph_rewire_prepare.json"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    prepare(args.source, args.output, args.report, args.threads, args.seed)


if __name__ == "__main__":
    main()
