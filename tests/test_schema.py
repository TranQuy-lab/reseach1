from nids_preprocessing import schema


def test_reference_stats_arithmetic_is_internally_consistent():
    """The 'expected_clean_shape' constant should equal
    raw_shape - rows_with_nan_or_inf - fully_duplicate_rows on the row axis,
    and raw columns - 2 (dropped IP columns) on the column axis. This just
    guards against a future typo when someone edits these constants."""
    raw_rows, raw_cols = schema.REFERENCE_STATS["raw_shape"]
    expected_rows = (
        raw_rows
        - schema.REFERENCE_STATS["rows_with_nan_or_inf"]
        - schema.REFERENCE_STATS["fully_duplicate_rows"]
    )
    expected_cols = raw_cols - len(schema.IP_COLUMNS)
    assert schema.REFERENCE_STATS["expected_clean_shape"] == (expected_rows, expected_cols)


def test_label_value_counts_sum_to_raw_row_count():
    total = sum(schema.REFERENCE_STATS["label_value_counts"].values())
    assert total == schema.REFERENCE_STATS["raw_shape"][0]


def test_shrunk_dtype_columns_are_subset_of_raw_minus_ip_columns():
    raw_minus_ip = set(schema.RAW_DTYPES) - set(schema.IP_COLUMNS)
    assert set(schema.SHRUNK_DTYPES) <= raw_minus_ip


def test_four_sub_datasets():
    assert len(schema.VALID_DATASETS) == 4
    assert "NF-UNSW-NB15-v2" in schema.VALID_DATASETS
    assert "NF-BoT-IoT-v2" in schema.VALID_DATASETS
    assert "NF-ToN-IoT-v2" in schema.VALID_DATASETS
    assert "NF-CSE-CIC-IDS2018-v2" in schema.VALID_DATASETS
