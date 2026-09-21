import logging

import numpy as np
import pandas as pd
import pytest

from nids_preprocessing import schema, splitting


def _make_synthetic_df(n=300, seed=0) -> pd.DataFrame:
    """A synthetic (NOT from the real dataset) DataFrame just big and varied
    enough to exercise stratification / filtering logic: three Attack
    classes at very different frequencies, and rows spread across three of
    the four real sub-dataset names."""
    rng = np.random.default_rng(seed)
    attack = rng.choice(
        ["Benign", "DoS", "rare_attack"], size=n, p=[0.6, 0.35, 0.05]
    )
    dataset = rng.choice(
        ["NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"], size=n
    )
    label = (attack != "Benign").astype(np.int8)
    return pd.DataFrame(
        {
            "IN_BYTES": rng.integers(0, 1000, size=n),
            schema.ATTACK_COL: attack,
            schema.LABEL_COL: label,
            schema.DATASET_COL: dataset,
        }
    )


def test_filter_by_dataset_keeps_only_requested_sources():
    df = _make_synthetic_df()
    out = splitting.filter_by_dataset(df, ["NF-BoT-IoT-v2"])
    assert set(out[schema.DATASET_COL]) == {"NF-BoT-IoT-v2"}
    assert len(out) < len(df)


def test_filter_by_dataset_rejects_unknown_name():
    df = _make_synthetic_df()
    with pytest.raises(ValueError):
        splitting.filter_by_dataset(df, ["NF-Typo-v2"])


def test_stratified_split_train_test_proportions_and_coverage():
    df = _make_synthetic_df(n=500)
    splits = splitting.stratified_split(df, test_size=0.2, random_state=1)
    assert set(splits) == {"train", "test"}
    assert len(splits["train"]) + len(splits["test"]) == len(df)
    # every Attack class present in the full data should appear in both splits
    for cls in df[schema.ATTACK_COL].unique():
        assert cls in set(splits["train"][schema.ATTACK_COL])
        assert cls in set(splits["test"][schema.ATTACK_COL])
    # class proportions roughly preserved in the test split (loose tolerance;
    # this is a statistical property, not an exact one)
    full_frac = df[schema.ATTACK_COL].value_counts(normalize=True)
    test_frac = splits["test"][schema.ATTACK_COL].value_counts(normalize=True)
    for cls in full_frac.index:
        assert abs(full_frac[cls] - test_frac[cls]) < 0.05


def test_stratified_split_with_val_partitions_disjointly():
    df = _make_synthetic_df(n=600)
    splits = splitting.stratified_split(df, test_size=0.2, val_size=0.1, random_state=1)
    assert set(splits) == {"train", "val", "test"}
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(df)
    # roughly the requested proportions (within a few rows of rounding)
    assert abs(len(splits["test"]) / len(df) - 0.2) < 0.02
    assert abs(len(splits["val"]) / len(df) - 0.1) < 0.02


def test_stratified_split_warns_on_rare_class(caplog):
    df = _make_synthetic_df(n=200, seed=3)  # rare_attack ~5% of 200 = ~10 rows
    with caplog.at_level(logging.WARNING, logger="nids_preprocessing.splitting"):
        splitting.stratified_split(df, test_size=0.2, random_state=1)
    assert any("fewer than" in rec.message for rec in caplog.records)


def test_cross_dataset_split_test_set_is_exactly_the_held_out_dataset():
    df = _make_synthetic_df(n=500)
    splits = splitting.cross_dataset_split(df, held_out_dataset="NF-BoT-IoT-v2")
    assert set(splits["test"][schema.DATASET_COL]) == {"NF-BoT-IoT-v2"}
    assert "NF-BoT-IoT-v2" not in set(splits["train"][schema.DATASET_COL])


def test_cross_dataset_split_train_test_partition_all_rows_exactly_once():
    df = _make_synthetic_df(n=500)
    splits = splitting.cross_dataset_split(df, held_out_dataset="NF-ToN-IoT-v2")
    assert len(splits["train"]) + len(splits["test"]) == len(df)
    # rows are partitioned by Dataset, not sampled -- every held-out row
    # must show up in test, none of them in train
    n_held_out_in_original = (df[schema.DATASET_COL] == "NF-ToN-IoT-v2").sum()
    assert len(splits["test"]) == n_held_out_in_original


def test_cross_dataset_split_with_val_keeps_val_out_of_held_out_dataset():
    df = _make_synthetic_df(n=900)
    splits = splitting.cross_dataset_split(
        df, held_out_dataset="NF-BoT-IoT-v2", val_size=0.2, random_state=1
    )
    assert set(splits) == {"train", "val", "test"}
    assert "NF-BoT-IoT-v2" not in set(splits["val"][schema.DATASET_COL])
    assert "NF-BoT-IoT-v2" not in set(splits["train"][schema.DATASET_COL])
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(df)


def test_cross_dataset_split_rejects_unknown_dataset_name():
    df = _make_synthetic_df()
    with pytest.raises(ValueError):
        splitting.cross_dataset_split(df, held_out_dataset="NF-Typo-v2")


def test_cross_dataset_split_rejects_dataset_with_zero_rows_present():
    df = _make_synthetic_df()
    df = df[df[schema.DATASET_COL] != "NF-CSE-CIC-IDS2018-v2"]  # remove all its rows
    with pytest.raises(ValueError):
        splitting.cross_dataset_split(df, held_out_dataset="NF-CSE-CIC-IDS2018-v2")


def test_cross_dataset_split_warns_about_classes_unique_to_held_out_set(caplog):
    df = _make_synthetic_df(n=300, seed=2)
    # inject an attack class that exists ONLY in the held-out sub-dataset
    extra = pd.DataFrame(
        {
            "IN_BYTES": [1, 2, 3],
            schema.ATTACK_COL: "only_in_bot_iot",
            schema.LABEL_COL: 1,
            schema.DATASET_COL: "NF-BoT-IoT-v2",
        }
    )
    df = pd.concat([df, extra], ignore_index=True)
    with caplog.at_level(logging.WARNING, logger="nids_preprocessing.splitting"):
        splitting.cross_dataset_split(df, held_out_dataset="NF-BoT-IoT-v2")
    assert any("appear ONLY in the held-out" in rec.message for rec in caplog.records)


def test_save_splits_writes_parquet_files(tmp_path):
    df = _make_synthetic_df(n=100)
    splits = splitting.stratified_split(df, test_size=0.25, random_state=1)
    paths = splitting.save_splits(splits, tmp_path)
    assert set(paths) == {"train", "test"}
    for name, path in paths.items():
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(splits[name])
