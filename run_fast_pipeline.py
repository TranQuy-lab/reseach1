import sys
import os
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pandas as pd

from nids_preprocessing import sample, schema, splitting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fast_pipeline")

def run_fast(
    input_csv: str,
    output_dir: str = "/workspace/output",
    target_sample_rows: int = 2000000,
    min_rows_per_group: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    threads: int = 32,
    max_memory: str = "30GB",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("=== BẮT ĐẦU PIPELINE TỐI ƯU ĐA LUỒNG (DUCKDB + PYTHON) ===")
    logger.info("Cấu hình: threads=%d, max_memory=%s, drop_ports=True", threads, max_memory)

    con = duckdb.connect()
    con.execute(f"SET threads TO {threads};")
    con.execute(f"SET max_memory TO '{max_memory}';")

    clean_parquet = output_dir / "NF-UQ-NIDS-V2_clean.parquet"

    logger.info("Giai đoạn 1: Đọc CSV đa luồng + Xóa IP & Port + Lọc NaN/Inf + Lọc Trùng lặp (DISTINCT)...")
    copy_sql = f"""
    COPY (
        SELECT DISTINCT * EXCLUDE (IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT)
        FROM read_csv_auto('{input_csv}', sample_size=100000)
        WHERE NOT (
            isnan(L7_PROTO) OR isinf(L7_PROTO) OR
            isnan(SRC_TO_DST_SECOND_BYTES) OR isinf(SRC_TO_DST_SECOND_BYTES) OR
            isnan(DST_TO_SRC_SECOND_BYTES) OR isinf(DST_TO_SRC_SECOND_BYTES) OR
            isnan(FTP_COMMAND_RET_CODE) OR isinf(FTP_COMMAND_RET_CODE)
        )
    ) TO '{clean_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(copy_sql)
    t_clean = time.time() - t0
    logger.info("Giai đoạn 1 hoàn tất sau %.2f phút! File sạch đã ghi tại: %s", t_clean / 60, clean_parquet)

    clean_rows = con.execute(f"SELECT count(*) FROM '{clean_parquet}'").fetchone()[0]
    logger.info("Tổng số dòng sau làm sạch: %d dòng", clean_rows)

    logger.info("Giai đoạn 2: Lấy mẫu phân tầng %d dòng (min_rows_per_group=%d)...", target_sample_rows, min_rows_per_group)
    df_clean = pd.read_parquet(clean_parquet)
    logger.info("Nạp sạch vào DataFrame: shape=%s (bộ nhớ: %.2f GB)", df_clean.shape, df_clean.memory_usage(deep=True).sum() / 1e9)

    df_sample, sampling_report = sample.stratified_subsample(
        df_clean,
        target_n_rows=target_sample_rows,
        min_rows_per_group=min_rows_per_group,
        random_state=seed,
    )
    sample_path = output_dir / "NF-UQ-NIDS-V2_sampled.parquet"
    df_sample.to_parquet(sample_path)
    logger.info("Ghi file mẫu phân tầng: %d dòng -> %s", len(df_sample), sample_path)

    logger.info("Giai đoạn 3: Chia tập Train / Val / Test (test=%.1f, val=%.1f, seed=%d)...", test_size, val_size, seed)
    splits = splitting.stratified_split(
        df_sample,
        stratify_col=schema.ATTACK_COL,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
    )
    splitting.save_splits(splits, splits_dir)
    logger.info("Đã lưu các tập splits vào %s:", splits_dir)
    for name, part in splits.items():
        logger.info("  - %s: %d dòng", name, len(part))

    finished_at = datetime.now(timezone.utc).isoformat()
    total_time = time.time() - t0

    summary = {
        "pipeline": "fast_pipeline_multithreaded_duckdb",
        "started_at": started_at,
        "finished_at": finished_at,
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 2),
        "input_csv": str(input_csv),
        "drop_ports": True,
        "clean_rows": clean_rows,
        "sampled_rows": len(df_sample),
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "sampling_report": sampling_report.as_dict() if sampling_report else None,
    }

    summary_file = output_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Đã ghi tổng kết vào: %s", summary_file)

    log_path = output_dir / "results_log.csv"
    row = {
        "run_timestamp": finished_at,
        "input_csv": str(input_csv),
        "clean_rows": clean_rows,
        "drop_ports": True,
        "target_sample_rows": target_sample_rows,
        "sampled_rows": len(df_sample),
        "seed": seed,
        "train_rows": len(splits.get("train", [])),
        "val_rows": len(splits.get("val", [])),
        "test_rows": len(splits.get("test", [])),
        "total_duration_minutes": round(total_time / 60, 2),
    }
    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)
    logger.info("Đã ghi nhật ký vào: %s", log_path)

    logger.info("=== PIPELINE HOÀN TẤT THÀNH CÔNG TRONG %.2f PHÚT! ===", total_time / 60)
    return summary

if __name__ == "__main__":
    csv_path = "/root/.cache/kagglehub/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset/versions/1/NF-UQ-NIDS-v2.csv"
    run_fast(csv_path)
