"""
cleaning.py
===========
Reproduces, stage by stage, the cleaning performed in the team's reference
notebook (``NF-UQ-NIDS-V2-00-Cleaning``), as a set of small, independently
testable, documented functions instead of one linear notebook.

Each function docstring says explicitly whether it is:
  - a direct reproduction of a reference-notebook step ("REFERENCE"), or
  - an added improvement, with the reasoning spelled out ("IMPROVEMENT").

The functions are pure (they return new DataFrames rather than mutating
in place with pandas' `inplace=True`). The reference notebook uses
`inplace=True` throughout; we deliberately avoid that here because
`inplace` chained assignment is a known source of `SettingWithCopyWarning`
/ silent no-ops on newer pandas (>=2.x, and especially with pandas' Copy-
on-Write behaviour, which is the default from pandas 3.0 onward). Returning
new frames is slightly more verbose but behaves identically across pandas
versions -- important since the three of you may not all be pinned to the
exact same pandas version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Row/column counts collected at each cleaning stage, for logging and
    for cross-checking against schema.REFERENCE_STATS."""

    raw_shape: tuple[int, int] | None = None
    after_drop_id_cols_shape: tuple[int, int] | None = None
    rows_with_nan_or_inf: int | None = None
    after_dropna_shape: tuple[int, int] | None = None
    fully_duplicate_rows: int | None = None
    final_shape: tuple[int, int] | None = None
    label_attack_mismatches: int | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = " | ".join(self.notes)
        return d


def load_raw_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """REFERENCE: `pd.read_csv(path, dtype=suggested_dtypes, sep=',', encoding='utf-8')`.

    `nrows` is an addition (not in the reference notebook) so the pipeline
    can be smoke-tested on a small slice of the real 76M-row CSV before
    committing to a full run; it has no effect on correctness of the
    cleaning logic itself.
    """
    df = pd.read_csv(
        path,
        dtype=schema.RAW_DTYPES,
        sep=",",
        encoding="utf-8",
        nrows=nrows,
    )
    logger.info("Loaded raw CSV: shape=%s", df.shape)
    return df


def drop_identifier_columns(df: pd.DataFrame, drop_ports: bool = False) -> pd.DataFrame:
    """REFERENCE (when drop_ports=False, the default): drops IPV4_SRC_ADDR /
    IPV4_DST_ADDR only, exactly as the reference notebook's cell 9 does.

    `drop_ports=True` is the OPEN DECISION described in schema.py: also
    drops L4_SRC_PORT / L4_DST_PORT, matching the stricter IP+port
    exclusion reported for the dataset authors' own NF-UQ-NIDS preprocessing
    elsewhere in the literature. Off by default so the pipeline's default
    output reproduces the reference notebook exactly; set it explicitly if
    the team has decided to follow the stricter variant.
    """
    cols_to_drop = list(schema.IP_COLUMNS)
    if drop_ports:
        cols_to_drop += list(schema.PORT_COLUMNS)
    missing = [c for c in cols_to_drop if c not in df.columns]
    if missing:
        raise KeyError(f"Expected identifier columns not found in dataframe: {missing}")
    out = df.drop(columns=cols_to_drop)
    logger.info("Dropped identifier columns %s -> shape=%s", cols_to_drop, out.shape)
    return out


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """IMPROVEMENT over REFERENCE: reproduces the exact dtypes the reference
    notebook's `df_shrink(df, obj2cat=False, int2uint=False)` produced
    (verified against its cell 11 output, see schema.SHRUNK_DTYPES), but
    applies them with plain pandas `astype` instead of depending on fastai.
    See the long comment in schema.py for the rationale.
    """
    present = {c: t for c, t in schema.SHRUNK_DTYPES.items() if c in df.columns}
    missing = set(schema.SHRUNK_DTYPES) - set(present)
    if missing:
        logger.warning("downcast_dtypes: columns not present, skipped: %s", sorted(missing))
    out = df.astype(present)
    logger.info("Downcast dtypes for %d columns", len(present))
    return out


def remove_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """REFERENCE: cell 12 -- replace +-inf with NaN, then drop any row that
    has at least one NaN. Returns (cleaned_df, n_rows_removed).
    """
    df2 = df.replace([np.inf, -np.inf], np.nan)
    mask_bad = df2.isna().any(axis=1)
    n_bad = int(mask_bad.sum())
    out = df2.loc[~mask_bad].copy()
    logger.info("Removed %d rows with NaN/inf -> shape=%s", n_bad, out.shape)
    return out, n_bad


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """REFERENCE: cell 13 -- drop fully-duplicate rows (all columns equal),
    then reset the index. Returns (cleaned_df, n_duplicates_removed).
    """
    n_dupes = int(df.duplicated().sum())
    out = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d fully-duplicate rows -> shape=%s", n_dupes, out.shape)
    return out, n_dupes


def validate_label_matches_attack(df: pd.DataFrame) -> int:
    """IMPROVEMENT: checks (does not assume) the verified relationship
    Label == 0 <=> Attack == 'Benign' documented in schema.py. Returns the
    number of mismatching rows found (0 on the full official dataset).

    This is cheap insurance: if a team member runs this pipeline on a
    filtered/resampled subset and something upstream corrupted the labels,
    this will catch it instead of silently propagating bad labels into
    Person B/C's model training.
    """
    if schema.LABEL_COL not in df.columns or schema.ATTACK_COL not in df.columns:
        logger.warning("validate_label_matches_attack: Label/Attack column missing, skipped")
        return 0
    expected_label = (df[schema.ATTACK_COL] != schema.BENIGN_VALUE).astype(df[schema.LABEL_COL].dtype)
    mismatches = int((expected_label != df[schema.LABEL_COL]).sum())
    if mismatches:
        logger.warning("Label/Attack mismatch found in %d rows", mismatches)
    return mismatches


def clean_pipeline(
    df: pd.DataFrame,
    drop_ports: bool = False,
    validate_label: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Runs the full cleaning sequence in the exact order used by the
    reference notebook:

        drop ID columns -> downcast dtypes -> drop inf/NaN rows ->
        drop fully-duplicate rows

    and returns the cleaned dataframe plus a CleaningReport you can log to
    your shared results_log.csv or compare against schema.REFERENCE_STATS
    with `pipeline.compare_to_reference`.
    """
    report = CleaningReport(raw_shape=tuple(df.shape))

    df = drop_identifier_columns(df, drop_ports=drop_ports)
    report.after_drop_id_cols_shape = tuple(df.shape)

    df = downcast_dtypes(df)

    df, n_bad = remove_invalid_rows(df)
    report.rows_with_nan_or_inf = n_bad
    report.after_dropna_shape = tuple(df.shape)

    df, n_dupes = remove_duplicates(df)
    report.fully_duplicate_rows = n_dupes
    report.final_shape = tuple(df.shape)

    if validate_label:
        report.label_attack_mismatches = validate_label_matches_attack(df)

    return df, report
