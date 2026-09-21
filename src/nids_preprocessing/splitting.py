"""
splitting.py
============
Utilities for (a) selecting which of the four merged sub-datasets to train
and test on, and the two split strategies required side-by-side by Prompt A
(ràng buộc cứng #5):
  (b) `stratified_split`     -- stratified train/val/test by Attack class
  (c) `cross_dataset_split`  -- train on (N-1) sub-datasets, test on the
                                 one held out entirely, for a genuine
                                 cross-network generalisation check
so Person B and Person C can both load whichever split they need directly,
and the proposed algorithm and the baselines are always compared on
identical data.

Nothing here is part of the reference notebook (it stops at the cleaned
parquet). This is the "IMPROVEMENT" / extension layer that turns the
verified cleaning output into ready-to-train splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import schema

logger = logging.getLogger(__name__)

MIN_RECOMMENDED_CLASS_COUNT = 10


def filter_by_dataset(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Keep only rows whose `Dataset` column is in `datasets`.

    Validates the requested names against schema.VALID_DATASETS so a typo
    (e.g. "NF-BoT-IoT" instead of "NF-BoT-IoT-v2") fails loudly instead of
    silently returning zero rows.
    """
    unknown = [d for d in datasets if d not in schema.VALID_DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset name(s) {unknown}. Valid options: {schema.VALID_DATASETS}"
        )
    out = df[df[schema.DATASET_COL].isin(datasets)].reset_index(drop=True)
    logger.info("Filtered to datasets=%s -> shape=%s", datasets, out.shape)
    return out


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = schema.ATTACK_COL,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified train/(val)/test split.

    Stratifying on `Attack` (rather than the binary `Label`) is deliberate:
    NF-UQ-NIDS-v2 is heavily imbalanced across attack categories (e.g.
    DDoS/DoS are millions of rows, Worms is 164 rows across the whole
    merged dataset) as shown in the dataset's own class-distribution table.
    Stratifying on the coarser binary Label would let a whole rare attack
    category land entirely in one split by chance; stratifying on Attack
    keeps every class represented in every split whenever it has enough
    rows to do so.

    Parameters
    ----------
    test_size : fraction held out as the final test set.
    val_size  : if given, a further fraction of the *remaining* (non-test)
                data held out as a validation set (e.g. for early stopping /
                hyperparameter tuning). If None, only {"train", "test"} are
                returned.
    random_state : fixed for reproducibility across all three team members --
                use the same value when comparing your algorithm against
                the baselines so everyone trains/tests on the same rows.

    Returns
    -------
    dict with keys "train", "test" (and "val" if val_size is given), each a
    DataFrame with its original columns and index reset.
    """
    _warn_if_rare_classes(df, stratify_col, test_size, val_size)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )

    splits = {}
    if val_size is not None:
        # val_size is expressed as a fraction of the ORIGINAL data; convert
        # to a fraction of what remains after the test split was removed.
        remaining_frac = 1 - test_size
        relative_val_size = val_size / remaining_frac
        train_df, val_df = train_test_split(
            train_df,
            test_size=relative_val_size,
            stratify=train_df[stratify_col],
            random_state=random_state,
        )
        splits["val"] = val_df.reset_index(drop=True)

    splits["train"] = train_df.reset_index(drop=True)
    splits["test"] = test_df.reset_index(drop=True)

    for name, part in splits.items():
        logger.info("Split '%s': shape=%s", name, part.shape)
    return splits


def _warn_if_rare_classes(
    df: pd.DataFrame, stratify_col: str, test_size: float, val_size: float | None
) -> None:
    """Not a hard failure -- a logged warning. sklearn's stratified split
    requires at least 2 members per class and will raise its own error if a
    class has only 1 row; this warns earlier, and also flags classes that
    are technically splittable but so small the resulting split fragment
    is almost meaningless (e.g. 3 test rows for a class of 164)."""
    smallest_frac = min(test_size, val_size) if val_size else test_size
    counts = df[stratify_col].value_counts()
    at_risk = counts[counts * smallest_frac < MIN_RECOMMENDED_CLASS_COUNT]
    if len(at_risk):
        logger.warning(
            "%d class(es) in '%s' will have fewer than %d rows in the "
            "smallest split (e.g. %s). Stratification will still run, but "
            "metrics on these classes may be unstable -- worth flagging in "
            "FUTURE_WORK.md.",
            len(at_risk),
            stratify_col,
            MIN_RECOMMENDED_CLASS_COUNT,
            dict(at_risk.head(5)),
        )


def cross_dataset_split(
    df: pd.DataFrame,
    held_out_dataset: str,
    val_size: float | None = None,
    stratify_col: str = schema.ATTACK_COL,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Train on (N-1) of the four merged sub-datasets, test on the one held
    out entirely. This is the second split strategy required by Prompt A,
    ràng buộc cứng #5b:

        "Cross-dataset: train trên (N-1) sub-dataset gốc, test trên
        sub-dataset còn lại -- để đánh giá khả năng tổng quát hoá thật giữa
        các mạng khác nhau."

    Unlike `stratified_split`, rows are partitioned by their `Dataset`
    origin rather than sampled -- every row of `held_out_dataset` goes to
    test, and every row of the other sub-datasets goes to train/val. This
    measures whether a model generalises to a genuinely different network
    environment, not just to unseen rows from the same environments it
    trained on.

    Parameters
    ----------
    held_out_dataset : one of schema.VALID_DATASETS. All of its rows become
        the test set; none of them leak into train/val.
    val_size : if given, a stratified (by `stratify_col`) fraction of the
        *training* sub-datasets (never the held-out one) is carved out for
        validation/early-stopping. If None, only {"train", "test"} are
        returned.
    random_state : only affects the optional train/val split -- the
        train/test partition itself is fully determined by
        `held_out_dataset` and has no randomness.

    Returns
    -------
    dict with keys "train", "test" (and "val" if val_size is given).
    """
    if held_out_dataset not in schema.VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset name {held_out_dataset!r}. Valid options: {schema.VALID_DATASETS}"
        )
    test_df = df[df[schema.DATASET_COL] == held_out_dataset].reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError(
            f"held_out_dataset={held_out_dataset!r} has 0 rows in this dataframe -- "
            "nothing to test on. Check the Dataset column values before splitting."
        )
    train_pool = df[df[schema.DATASET_COL] != held_out_dataset].reset_index(drop=True)

    _warn_about_classes_unique_to_held_out_dataset(train_pool, test_df, stratify_col)

    splits: dict[str, pd.DataFrame] = {"test": test_df}
    if val_size is not None:
        _warn_if_rare_classes(train_pool, stratify_col, test_size=val_size, val_size=None)
        train_df, val_df = train_test_split(
            train_pool,
            test_size=val_size,
            stratify=train_pool[stratify_col],
            random_state=random_state,
        )
        splits["train"] = train_df.reset_index(drop=True)
        splits["val"] = val_df.reset_index(drop=True)
    else:
        splits["train"] = train_pool

    for name, part in splits.items():
        logger.info("Cross-dataset split '%s' (held out=%s): shape=%s", name, held_out_dataset, part.shape)
    return splits


def _warn_about_classes_unique_to_held_out_dataset(
    train_pool: pd.DataFrame, test_df: pd.DataFrame, stratify_col: str
) -> None:
    """Cross-dataset evaluation has a failure mode `stratified_split`
    doesn't: an Attack category that only ever occurs in the held-out
    sub-dataset. The model will never have seen it during training, and any
    metric computed on it (F1, recall, ...) is measuring extrapolation to a
    completely unseen class, not interpolation -- worth flagging loudly
    rather than letting it silently tank (or inflate, if it happens to
    collapse into a class the model over-predicts) one row of a results
    table with no explanation attached.
    """
    train_classes = set(train_pool[stratify_col].unique())
    test_only_classes = set(test_df[stratify_col].unique()) - train_classes
    if test_only_classes:
        logger.warning(
            "cross_dataset_split: %d class(es) in '%s' appear ONLY in the held-out "
            "test dataset and never in the training pool: %s. The model cannot "
            "possibly have learned these classes -- report their metrics separately "
            "and note this in FUTURE_WORK.md / results_log.csv notes, don't average "
            "them silently into the overall macro F1.",
            len(test_only_classes),
            stratify_col,
            sorted(test_only_classes),
        )


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, str]:
    """Writes each split to `<out_dir>/<name>.parquet`. Returns the mapping
    of split name -> written path (as strings) for logging."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path)
        paths[name] = str(path)
        logger.info("Wrote %s: %s rows -> %s", name, len(part), path)
    return paths
