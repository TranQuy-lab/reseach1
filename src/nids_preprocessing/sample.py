"""
sample.py
=========
Stratified sub-sampling of the (tens-of-millions-of-rows) cleaned dataset
down to a manageable training subset.

This module exists to satisfy a hard constraint from the team's planning
prompt for Person A (``ke_hoach_pipeline_prompt_AI_NIDS.md``, Prompt A,
ràng buộc cứng #4):

    "Dataset gốc rất lớn (hàng chục triệu dòng): nêu rõ chiến lược lấy mẫu
    con -- stratified theo CẢ Attack lẫn Dataset gốc, cố định random_state,
    log tỉ lệ giữ lại từng lớp. Không lấy mẫu ngẫu nhiên đơn thuần vì sẽ mất
    hẳn lớp hiếm."

Like ``splitting.py``, nothing here reproduces a step from the reference
cleaning notebook -- the reference notebook only cleans and writes a single
parquet file, it does not subsample. This is entirely IMPROVEMENT /
extension code, written to satisfy the explicit requirement above rather
than left implicit.

Why plain random/`DataFrame.sample()` is not used
---------------------------------------------------
A single global random sample at a fraction `p` keeps each class in
expectation at `p * class_size` rows. For a class as rare as "Worms" in
NF-UNSW-NB15-v2 (164 rows out of ~76M total, i.e. far below 0.001% of the
merged dataset), even a generous subsample fraction can plausibly reduce it
to zero rows, or to a single-digit count too small to split into
train/val/test at all. Plain `df.sample(n)` also does not fix the
*relative* imbalance between the four merged sub-datasets (NF-BoT-IoT-v2 is
almost entirely attack traffic; NF-UNSW-NB15-v2 is almost entirely benign),
so a global sample can silently over- or under-represent whole sub-datasets.

The approach here samples per (Dataset, Attack) group independently, and
explicitly protects any group smaller than `min_rows_per_group` by keeping
it *whole* rather than shrinking it further -- deliberately overshooting the
requested total row count for the sake of not losing a rare class entirely,
and logging when that happens so it's visible, not silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

logger = logging.getLogger(__name__)

DEFAULT_MIN_ROWS_PER_GROUP = 50


@dataclass
class SamplingReport:
    """Per-group before/after row counts, for logging and for
    `data_card.md` / `results_log.csv`. Every number here is read back off
    the actual sampled DataFrame, never estimated."""

    group_cols: tuple[str, str]
    total_rows_before: int
    total_rows_after: int
    target_n_rows: int
    min_rows_per_group: int
    random_state: int
    n_groups_total: int
    n_groups_protected_as_rare: int
    group_retention: dict[tuple, dict[str, int]] = field(default_factory=dict)
    protected_groups: list[tuple] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        # dict/list keyed by tuples aren't JSON-serialisable as-is; stringify
        # the group keys so this can be dumped straight into run_summary.json.
        d["group_retention"] = {str(k): v for k, v in self.group_retention.items()}
        d["protected_groups"] = [str(g) for g in self.protected_groups]
        return d

    def summary_lines(self) -> list[str]:
        """Human-readable lines for data_card.md / console logging."""
        lines = [
            f"Sampled {self.total_rows_after:,} / {self.total_rows_before:,} rows "
            f"(requested target: {self.target_n_rows:,}) across "
            f"{self.n_groups_total} ({', '.join(self.group_cols)}) groups.",
            f"{self.n_groups_protected_as_rare} group(s) were smaller than "
            f"min_rows_per_group={self.min_rows_per_group} and were kept in full "
            f"rather than sampled down further.",
        ]
        if self.protected_groups:
            shown = self.protected_groups[:10]
            more = "" if len(self.protected_groups) <= 10 else f" (+{len(self.protected_groups) - 10} more)"
            lines.append(f"Protected groups: {shown}{more}")
        return lines


def _validate_group_cols(df: pd.DataFrame, group_cols: tuple[str, str]) -> None:
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"stratified_subsample: group column(s) not found in dataframe: {missing}"
        )


def compute_group_targets(
    group_sizes: pd.Series,
    target_n_rows: int,
    min_rows_per_group: int,
) -> pd.Series:
    """Given the real size of every (Dataset, Attack) group and an overall
    target row count, decide how many rows to keep from each group.

    Rule (deliberately simple and auditable, not a fancy allocation
    algorithm): every group is first entitled to a share of `target_n_rows`
    proportional to its share of the full dataset. Any group whose full
    size is below `min_rows_per_group` is then bumped up to keep ALL of its
    rows (protecting rare classes/sub-datasets from being sampled away),
    and any group whose proportional share would exceed its own size is
    capped at its actual size (you cannot sample more rows than exist).

    Because protected groups can push the realised total above
    `target_n_rows`, the returned total is generally >= target_n_rows, not
    exactly equal to it. That is intentional -- see module docstring.
    """
    total_rows = int(group_sizes.sum())
    if total_rows == 0:
        return group_sizes.copy()

    proportional = (group_sizes / total_rows * target_n_rows).round().astype(int)
    # never sample more rows than a group actually has
    proportional = proportional.clip(upper=group_sizes)
    # protect rare groups: keep them whole
    is_rare = group_sizes < min_rows_per_group
    targets = proportional.where(~is_rare, group_sizes)
    # rounding/clipping can floor a non-rare, non-empty group to 0 rows;
    # guarantee at least 1 row survives from every group that had any data
    targets = targets.where(~((targets == 0) & (group_sizes > 0)), 1)
    return targets.astype(int)


def stratified_subsample(
    df: pd.DataFrame,
    target_n_rows: int | None = None,
    target_frac: float | None = None,
    group_cols: tuple[str, str] = (schema.DATASET_COL, schema.ATTACK_COL),
    min_rows_per_group: int = DEFAULT_MIN_ROWS_PER_GROUP,
    random_state: int = 42,
) -> tuple[pd.DataFrame, SamplingReport]:
    """Stratified sub-sample of `df`, grouped jointly by `group_cols`
    (default: (Dataset, Attack), per ràng buộc cứng #4 of Prompt A -- both
    the sub-dataset origin AND the attack category, not just one of them).

    Exactly one of `target_n_rows` / `target_frac` must be given.

    Returns (sampled_df, SamplingReport). The report's `group_retention`
    logs, for every group, how many rows it started with and how many were
    kept -- this is the "log tỉ lệ giữ lại từng lớp" (log the retention
    ratio per class) requirement; nothing here is summarised away.
    """
    if (target_n_rows is None) == (target_frac is None):
        raise ValueError(
            "stratified_subsample: pass exactly one of target_n_rows or target_frac"
        )
    _validate_group_cols(df, group_cols)

    total_rows_before = len(df)
    if target_frac is not None:
        if not (0 < target_frac <= 1):
            raise ValueError(f"target_frac must be in (0, 1], got {target_frac}")
        target_n_rows = int(round(total_rows_before * target_frac))

    group_sizes = df.groupby(list(group_cols), observed=True).size()
    targets = compute_group_targets(group_sizes, target_n_rows, min_rows_per_group)

    rng = np.random.default_rng(random_state)
    sampled_parts = []
    group_retention: dict[tuple, dict[str, int]] = {}
    protected_groups: list[tuple] = []

    for group_key, group_df in df.groupby(list(group_cols), observed=True):
        n_keep = int(targets.loc[group_key])
        n_available = len(group_df)
        if n_keep >= n_available:
            sampled_parts.append(group_df)
            n_keep = n_available
        else:
            # per-group fixed seed derived from the global seed so results
            # are reproducible regardless of pandas groupby iteration order
            group_seed = rng.integers(0, 2**32 - 1)
            idx = np.random.default_rng(group_seed).choice(
                n_available, size=n_keep, replace=False
            )
            sampled_parts.append(group_df.iloc[idx])

        group_retention[group_key] = {"before": n_available, "after": n_keep}
        if n_available < min_rows_per_group:
            protected_groups.append(group_key)

    out = pd.concat(sampled_parts, axis=0).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    report = SamplingReport(
        group_cols=group_cols,
        total_rows_before=total_rows_before,
        total_rows_after=len(out),
        target_n_rows=target_n_rows,
        min_rows_per_group=min_rows_per_group,
        random_state=random_state,
        n_groups_total=len(group_sizes),
        n_groups_protected_as_rare=len(protected_groups),
        group_retention=group_retention,
        protected_groups=protected_groups,
    )
    for line in report.summary_lines():
        logger.info(line)

    return out, report
