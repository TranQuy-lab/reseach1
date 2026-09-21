"""
pipeline.py
===========
Orchestrates load -> clean -> (optional sub-dataset filter) -> (optional
stratified sub-sample) -> split (stratified OR cross-dataset) -> save, and
writes a run-summary row that Person A can paste into the team's shared
results_log.csv.

Usage
-----
    # Default: stratified split on the FULL cleaned dataset
    python -m nids_preprocessing.pipeline \\
        --input /path/to/NF-UQ-NIDS-v2.csv \\
        --output-dir ./output \\
        --test-size 0.2 --val-size 0.1 \\
        --seed 42

    # Quick smoke test on the first 200k raw rows only (NOT a stratified
    # sample -- just a fast way to check the code runs before a full pass):
    python -m nids_preprocessing.pipeline --input NF-UQ-NIDS-v2.csv \\
        --output-dir ./smoke_test --sample-rows 200000

    # Real stratified sub-sampling down to a training-sized subset, per
    # Prompt A ràng buộc cứng #4 (stratified by BOTH Attack and Dataset,
    # rare classes protected):
    python -m nids_preprocessing.pipeline --input NF-UQ-NIDS-v2.csv \\
        --output-dir ./output --target-sample-rows 2000000

    # Cross-dataset split instead of stratified: train on 3 sub-datasets,
    # test entirely on the 4th (Prompt A ràng buộc cứng #5b):
    python -m nids_preprocessing.pipeline --input NF-UQ-NIDS-v2.csv \\
        --output-dir ./output --split-strategy cross_dataset \\
        --held-out-dataset NF-BoT-IoT-v2

    # Cross-check a full run against the numbers printed in the reference
    # notebook (only meaningful when run on the *complete*, unfiltered CSV,
    # with no --target-sample-rows/--target-sample-frac applied):
    python -m nids_preprocessing.pipeline --input NF-UQ-NIDS-v2.csv \\
        --output-dir ./output --compare-to-reference
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import cleaning, sample, schema, splitting

logger = logging.getLogger(__name__)


def compare_to_reference(report: cleaning.CleaningReport) -> dict:
    """Compares a CleaningReport against schema.REFERENCE_STATS (the
    numbers printed by the reference notebook). Only meaningful when the
    pipeline was run on the full, unfiltered NF-UQ-NIDS-v2.csv -- on a
    sample or a filtered subset these will legitimately differ, so this
    function does not raise, it only reports.
    """
    ref = schema.REFERENCE_STATS
    checks = {
        "raw_shape": (report.raw_shape, ref["raw_shape"]),
        "rows_with_nan_or_inf": (report.rows_with_nan_or_inf, ref["rows_with_nan_or_inf"]),
        "fully_duplicate_rows": (report.fully_duplicate_rows, ref["fully_duplicate_rows"]),
        "final_shape": (report.final_shape, ref["expected_clean_shape"]),
    }
    result = {}
    all_match = True
    for name, (actual, expected) in checks.items():
        match = actual == expected
        all_match &= match
        result[name] = {"actual": actual, "expected_from_reference": expected, "match": match}
    result["all_match"] = all_match
    level = logging.INFO if all_match else logging.WARNING
    logger.log(
        level,
        "Reference comparison %s: %s",
        "PASSED" if all_match else "FOUND DIFFERENCES",
        result,
    )
    return result


def run(
    input_csv: str | Path,
    output_dir: str | Path,
    sample_rows: int | None = None,
    drop_ports: bool = False,
    datasets: list[str] | None = None,
    target_sample_rows: int | None = None,
    target_sample_frac: float | None = None,
    min_rows_per_group: int = sample.DEFAULT_MIN_ROWS_PER_GROUP,
    split_strategy: str = "stratified",
    held_out_dataset: str | None = None,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify_col: str = schema.ATTACK_COL,
    seed: int = 42,
    do_compare_to_reference: bool = False,
) -> dict:
    """Runs the full pipeline once and returns a JSON-serialisable summary
    dict (also written to `<output_dir>/run_summary.json`).

    `split_strategy` is one of "stratified" (default) or "cross_dataset" --
    the two strategies required side-by-side by Prompt A ràng buộc #5.
    `cross_dataset` requires `held_out_dataset` to be set.
    """
    if split_strategy not in ("stratified", "cross_dataset"):
        raise ValueError(f"split_strategy must be 'stratified' or 'cross_dataset', got {split_strategy!r}")
    if split_strategy == "cross_dataset" and held_out_dataset is None:
        raise ValueError("split_strategy='cross_dataset' requires held_out_dataset to be set")
    if target_sample_rows is not None and target_sample_frac is not None:
        raise ValueError("pass at most one of target_sample_rows / target_sample_frac")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    df = cleaning.load_raw_csv(input_csv, nrows=sample_rows)
    df, report = cleaning.clean_pipeline(df, drop_ports=drop_ports)

    clean_path = output_dir / "NF-UQ-NIDS-V2_clean.parquet"
    df.to_parquet(clean_path)
    logger.info("Wrote cleaned dataset: %s rows -> %s", len(df), clean_path)

    reference_comparison = None
    if do_compare_to_reference:
        reference_comparison = compare_to_reference(report)

    if datasets:
        df = splitting.filter_by_dataset(df, datasets)

    sampling_report = None
    if target_sample_rows is not None or target_sample_frac is not None:
        df, sampling_report = sample.stratified_subsample(
            df,
            target_n_rows=target_sample_rows,
            target_frac=target_sample_frac,
            min_rows_per_group=min_rows_per_group,
            random_state=seed,
        )
        sample_path = output_dir / "NF-UQ-NIDS-V2_sampled.parquet"
        df.to_parquet(sample_path)
        logger.info("Wrote stratified sub-sample: %s rows -> %s", len(df), sample_path)

    if split_strategy == "stratified":
        splits = splitting.stratified_split(
            df,
            stratify_col=stratify_col,
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
        )
    else:
        splits = splitting.cross_dataset_split(
            df,
            held_out_dataset=held_out_dataset,
            val_size=val_size,
            stratify_col=stratify_col,
            random_state=seed,
        )
    split_paths = splitting.save_splits(splits, output_dir / "splits")

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "sample_rows_requested": sample_rows,
        "drop_ports": drop_ports,
        "datasets_filter": datasets,
        "target_sample_rows": target_sample_rows,
        "target_sample_frac": target_sample_frac,
        "min_rows_per_group": min_rows_per_group,
        "sampling_report": sampling_report.as_dict() if sampling_report else None,
        "split_strategy": split_strategy,
        "held_out_dataset": held_out_dataset,
        "test_size": test_size,
        "val_size": val_size,
        "stratify_col": stratify_col,
        "seed": seed,
        "cleaning_report": report.as_dict(),
        "reference_comparison": reference_comparison,
        "clean_parquet_path": str(clean_path),
        "split_paths": split_paths,
        "split_sizes": {name: int(len(part)) for name, part in splits.items()},
    }

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _append_results_log(output_dir, summary)
    return summary


def _append_results_log(output_dir: Path, summary: dict) -> None:
    """Appends one row to `<output_dir>/results_log.csv`.

    NOTE for the team: this uses a generic, self-describing set of columns
    (run timestamp, row counts, split sizes, parameters). If the shared
    results_log.csv schema already agreed on in the planning document uses
    different column names, rename/remap these before merging -- this
    function deliberately does not guess at column names it hasn't been
    shown, per "khong bia" (don't fabricate).
    """
    row = {
        "run_timestamp": summary["finished_at"],
        "input_csv": summary["input_csv"],
        "raw_rows": summary["cleaning_report"]["raw_shape"][0],
        "clean_rows": summary["cleaning_report"]["final_shape"][0],
        "clean_cols": summary["cleaning_report"]["final_shape"][1],
        "rows_with_nan_or_inf": summary["cleaning_report"]["rows_with_nan_or_inf"],
        "duplicate_rows_removed": summary["cleaning_report"]["fully_duplicate_rows"],
        "label_attack_mismatches": summary["cleaning_report"]["label_attack_mismatches"],
        "drop_ports": summary["drop_ports"],
        "datasets_filter": summary["datasets_filter"],
        "target_sample_rows": summary["target_sample_rows"],
        "target_sample_frac": summary["target_sample_frac"],
        "sampled_rows_after": summary["sampling_report"]["total_rows_after"] if summary["sampling_report"] else None,
        "n_groups_protected_as_rare": summary["sampling_report"]["n_groups_protected_as_rare"] if summary["sampling_report"] else None,
        "split_strategy": summary["split_strategy"],
        "held_out_dataset": summary["held_out_dataset"],
        "test_size": summary["test_size"],
        "val_size": summary["val_size"],
        "seed": summary["seed"],
        "train_rows": summary["split_sizes"].get("train"),
        "val_rows": summary["split_sizes"].get("val"),
        "test_rows": summary["split_sizes"].get("test"),
    }
    log_path = output_dir / "results_log.csv"
    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)
    logger.info("Appended run summary to %s", log_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NF-UQ-NIDS-v2 preprocessing pipeline")
    p.add_argument("--input", required=True, help="Path to raw NF-UQ-NIDS-v2.csv")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs into")
    p.add_argument("--sample-rows", type=int, default=None, help="Read only the first N raw rows (smoke test only, NOT a stratified sample)")
    p.add_argument("--drop-ports", action="store_true", help="Also drop L4_SRC_PORT/L4_DST_PORT (see schema.py OPEN DECISION)")
    p.add_argument("--datasets", nargs="+", default=None, choices=schema.VALID_DATASETS, help="Restrict to these sub-dataset(s) only, before sampling/splitting")

    sample_group = p.add_mutually_exclusive_group()
    sample_group.add_argument("--target-sample-rows", type=int, default=None, help="Stratified sub-sample (by Dataset x Attack, rare classes protected) down to ~N rows")
    sample_group.add_argument("--target-sample-frac", type=float, default=None, help="Stratified sub-sample down to this fraction of the cleaned dataset")
    p.add_argument("--min-rows-per-group", type=int, default=sample.DEFAULT_MIN_ROWS_PER_GROUP, help="(Dataset,Attack) groups smaller than this are kept whole rather than sampled down further")

    p.add_argument("--split-strategy", choices=["stratified", "cross_dataset"], default="stratified", help="'stratified': split by Attack class. 'cross_dataset': train on N-1 sub-datasets, test on --held-out-dataset")
    p.add_argument("--held-out-dataset", choices=schema.VALID_DATASETS, default=None, help="Required when --split-strategy=cross_dataset")
    p.add_argument("--test-size", type=float, default=0.2, help="Ignored when --split-strategy=cross_dataset (test set is exactly --held-out-dataset)")
    p.add_argument("--val-size", type=float, default=None)
    p.add_argument("--stratify-col", default=schema.ATTACK_COL, choices=[schema.ATTACK_COL, schema.LABEL_COL])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-to-reference", action="store_true", help="Compare cleaning stats to the reference notebook's numbers (only valid on the full, unfiltered, un-sampled CSV)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    summary = run(
        input_csv=args.input,
        output_dir=args.output_dir,
        sample_rows=args.sample_rows,
        drop_ports=args.drop_ports,
        datasets=args.datasets,
        target_sample_rows=args.target_sample_rows,
        target_sample_frac=args.target_sample_frac,
        min_rows_per_group=args.min_rows_per_group,
        split_strategy=args.split_strategy,
        held_out_dataset=args.held_out_dataset,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify_col=args.stratify_col,
        seed=args.seed,
        do_compare_to_reference=args.compare_to_reference,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
