import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import duckdb

OUTPUT_ROOT = Path("/workspace/gnn_data")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_ROOT / "prep_gnn_fast.log")),
    ],
)
log = logging.getLogger("prep_gnn_fast")

CSV_PATH   = "/root/.cache/kagglehub/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset/versions/1/NF-UQ-NIDS-v2.csv"
THREADS    = 32
MAX_MEMORY = "50GB"

FEATURE_COLS = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

def make_con():
    con = duckdb.connect()
    con.execute(f"SET threads TO {THREADS};")
    con.execute(f"SET max_memory TO '{MAX_MEMORY}';")
    con.execute("SELECT setseed(0.42);")
    log.info("DuckDB ready: threads=%d, max_memory=%s", THREADS, MAX_MEMORY)
    return con

def run():
    t_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    con = make_con()

    out_dir = OUTPUT_ROOT / "e_graphsage"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    clean_features_sql = ",\n    ".join(
        f'COALESCE(CASE WHEN isinf("{c}") OR isnan("{c}") THEN 0 ELSE "{c}" END, 0) AS "{c}"'
        for c in FEATURE_COLS
    )

    log.info("=== BƯỚC 1: Quét CSV 1 lần duy nhất cho cả NF-BoT-IoT-v2 và NF-ToN-IoT-v2 ===")
    t0 = time.time()
    extract_sql = f"""
    CREATE OR REPLACE TABLE e_graphsage_raw AS
    SELECT
        {clean_features_sql},
        Label,
        Attack,
        Dataset,
        IPV4_SRC_ADDR || ':' || CAST(L4_SRC_PORT AS VARCHAR) AS src_node,
        IPV4_DST_ADDR || ':' || CAST(L4_DST_PORT AS VARCHAR) AS dst_node
    FROM read_csv_auto('{CSV_PATH}', parallel=true)
    WHERE Dataset IN ('NF-BoT-IoT-v2', 'NF-ToN-IoT-v2')
      AND Label IS NOT NULL
      AND Attack IS NOT NULL;
    """
    con.execute(extract_sql)
    log.info("Quét và làm sạch CSV hoàn tất trong %.1fs", time.time() - t0)

    all_stats = {}
    tasks = [
        ("NF-BoT-IoT-v2", "bot_iot"),
        ("NF-ToN-IoT-v2", "ton_iot"),
    ]

    for dataset_name, prefix in tasks:
        log.info("=" * 60)
        log.info("Xử lý dataset: %s (prefix: %s)", dataset_name, prefix)
        t_ds = time.time()

        final_parquet = out_dir / f"{dataset_name}_gnn.parquet"
        con.execute(f"""
        COPY (
            SELECT * FROM e_graphsage_raw WHERE Dataset = '{dataset_name}'
        ) TO '{final_parquet}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
        """)
        log.info("[%s] Saved full GNN parquet: %s (%.1fs)", dataset_name, final_parquet, time.time() - t_ds)

        log.info("[%s] Phân chia Train / Val / Test...", dataset_name)
        t_split = time.time()
        con.execute(f"""
        CREATE OR REPLACE TABLE temp_split AS
        SELECT *,
               CASE 
                   WHEN rn <= CAST(cnt * 0.70 AS BIGINT) THEN 'train'
                   WHEN rn <= CAST(cnt * 0.80 AS BIGINT) THEN 'val'
                   ELSE 'test'
               END AS _split_tag
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY Attack ORDER BY random()) AS rn,
                   COUNT(*) OVER (PARTITION BY Attack) AS cnt
            FROM e_graphsage_raw
            WHERE Dataset = '{dataset_name}'
        );
        """)

        counts = dict(con.execute("SELECT _split_tag, count(*) FROM temp_split GROUP BY _split_tag").fetchall())
        train_rows = counts.get('train', 0)
        val_rows = counts.get('val', 0)
        test_rows = counts.get('test', 0)
        total_rows = train_rows + val_rows + test_rows

        train_file = splits_dir / f"{prefix}_train.parquet"
        val_file   = splits_dir / f"{prefix}_val.parquet"
        test_file  = splits_dir / f"{prefix}_test.parquet"

        con.execute(f"COPY (SELECT * EXCLUDE (rn, cnt, _split_tag) FROM temp_split WHERE _split_tag = 'train') TO '{train_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');")
        con.execute(f"COPY (SELECT * EXCLUDE (rn, cnt, _split_tag) FROM temp_split WHERE _split_tag = 'val') TO '{val_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');")
        con.execute(f"COPY (SELECT * EXCLUDE (rn, cnt, _split_tag) FROM temp_split WHERE _split_tag = 'test') TO '{test_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');")

        log.info("[%s] Splits xong: train=%d, val=%d, test=%d (%.1fs)", dataset_name, train_rows, val_rows, test_rows, time.time() - t_split)

        cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM '{final_parquet}'").fetchall()]
        all_stats[dataset_name] = {
            "dataset": dataset_name,
            "parquet": str(final_parquet),
            "columns": cols,
            "node_cols": ["src_node", "dst_node"],
            "feature_cols": FEATURE_COLS,
            "total_rows": total_rows,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
        }

    with open(out_dir / "prep_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    anomal_e_summary = {}
    anomal_e_file = OUTPUT_ROOT / "anomal_e" / "prep_summary.json"
    if anomal_e_file.exists():
        with open(anomal_e_file) as f:
            anomal_e_summary = json.load(f)

    total_time = time.time() - t_start
    global_summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_time_minutes": round(total_time / 60, 2),
        "threads": THREADS,
        "anomal_e": anomal_e_summary,
        "e_graphsage": all_stats,
    }
    with open(OUTPUT_ROOT / "global_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("=== HOÀN TẤT TOÀN BỘ PIPELINE TRONG %.2f PHÚT ===", total_time / 60)
    log.info("=" * 60)

if __name__ == "__main__":
    run()
