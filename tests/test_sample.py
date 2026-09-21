import numpy as np
import pandas as pd
import pytest

from nids_preprocessing import sample, schema


def _make_synthetic_df(n_common=4000, n_rare=8, seed=0) -> pd.DataFrame:
    """SYNTHETIC (not real) data: two sub-datasets, three Attack classes,
    one of which ("rare_attack") appears only `n_rare` times total, split
    across both sub-datasets. Large enough that a naive global fraction
    sample would be expected to lose it entirely.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in ["NF-BoT-IoT-v2", "NF-ToN-IoT-v2"]:
        for attack, count in [("Benign", n_common), ("DoS", n_common // 2)]:
            rows.append(
                pd.DataFrame(
                    {
                        schema.DATASET_COL: dataset,
                        schema.ATTACK_COL: attack,
                        "IN_BYTES": rng.integers(0, 1000, size=count),
                    }
                )
            )
    # the rare class: n_rare rows total, split across both sub-datasets
    rows.append(
        pd.DataFrame(
            {
                schema.DATASET_COL: ["NF-BoT-IoT-v2"] * (n_rare // 2)
                + ["NF-ToN-IoT-v2"] * (n_rare - n_rare // 2),
                schema.ATTACK_COL: "rare_attack",
                "IN_BYTES": rng.integers(0, 1000, size=n_rare),
            }
        )
    )
    return pd.concat(rows, axis=0).reset_index(drop=True)


def test_stratified_subsample_never_loses_the_rare_class():
    df = _make_synthetic_df()
    rare_before = int((df[schema.ATTACK_COL] == "rare_attack").sum())
    assert rare_before == 8  # sanity check on the fixture itself

    # ask for a tiny global fraction that would almost certainly zero out
    # an 8-row class under naive df.sample(frac=...)
    out, report = sample.stratified_subsample(
        df, target_frac=0.02, min_rows_per_group=20, random_state=1
    )
    rare_after = int((out[schema.ATTACK_COL] == "rare_attack").sum())
    assert rare_after == rare_before  # fully protected, not just "some kept"


def test_stratified_subsample_hits_target_rows_on_non_rare_groups():
    df = _make_synthetic_df(n_common=4000, n_rare=8)
    target = 400
    out, report = sample.stratified_subsample(
        df, target_n_rows=target, min_rows_per_group=20, random_state=1
    )
    # total will overshoot `target` a bit because the rare group is
    # protected in full -- but should be in the right ballpark, not e.g. 10x
    assert target <= len(out) <= target + 20
    assert report.total_rows_after == len(out)
    # rare_attack is split across both sub-datasets -> 2 protected groups,
    # one per (Dataset, rare_attack) combination
    assert report.n_groups_protected_as_rare == 2


def test_stratified_subsample_logs_before_after_for_every_group():
    df = _make_synthetic_df()
    out, report = sample.stratified_subsample(
        df, target_n_rows=200, min_rows_per_group=20, random_state=1
    )
    # every (Dataset, Attack) combination present in the input must have a
    # logged retention entry -- nothing summarised away
    expected_groups = set(
        df.groupby([schema.DATASET_COL, schema.ATTACK_COL], observed=True).groups.keys()
    )
    assert set(report.group_retention.keys()) == expected_groups
    for key, counts in report.group_retention.items():
        assert counts["after"] <= counts["before"]
        assert counts["after"] >= 1  # no group is silently zeroed out


def test_stratified_subsample_is_reproducible_with_fixed_seed():
    df = _make_synthetic_df()
    out1, _ = sample.stratified_subsample(df, target_n_rows=300, random_state=7)
    out2, _ = sample.stratified_subsample(df, target_n_rows=300, random_state=7)
    pd.testing.assert_frame_equal(
        out1.sort_values("IN_BYTES").reset_index(drop=True),
        out2.sort_values("IN_BYTES").reset_index(drop=True),
    )


def test_stratified_subsample_rejects_both_or_neither_of_n_rows_and_frac():
    df = _make_synthetic_df()
    with pytest.raises(ValueError):
        sample.stratified_subsample(df)  # neither given
    with pytest.raises(ValueError):
        sample.stratified_subsample(df, target_n_rows=100, target_frac=0.1)  # both given


def test_stratified_subsample_rejects_missing_group_columns():
    df = pd.DataFrame({"IN_BYTES": [1, 2, 3]})
    with pytest.raises(KeyError):
        sample.stratified_subsample(df, target_n_rows=2)


def test_compute_group_targets_never_exceeds_group_size():
    sizes = pd.Series({"a": 10, "b": 1000, "c": 3})
    targets = sample.compute_group_targets(sizes, target_n_rows=50, min_rows_per_group=5)
    assert (targets <= sizes).all()
    assert targets["c"] == 3  # rare group (below min_rows_per_group) kept whole
