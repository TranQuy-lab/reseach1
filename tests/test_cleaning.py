import numpy as np
import pandas as pd

from nids_preprocessing import cleaning, schema


def test_load_raw_csv_shape_and_dtypes(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    # 8 real rows + 3 synthetic edge-case rows, 46 columns
    assert df.shape == (11, 46)
    assert df["L4_SRC_PORT"].dtype == np.int16
    assert df["Label"].dtype == np.int32
    assert df["IPV4_SRC_ADDR"].dtype == object


def test_drop_identifier_columns_default_keeps_ports(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    out = cleaning.drop_identifier_columns(df)
    assert "IPV4_SRC_ADDR" not in out.columns
    assert "IPV4_DST_ADDR" not in out.columns
    assert "L4_SRC_PORT" in out.columns  # kept by default, matching reference notebook
    assert "L4_DST_PORT" in out.columns
    assert out.shape[1] == 44


def test_drop_identifier_columns_can_also_drop_ports(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    out = cleaning.drop_identifier_columns(df, drop_ports=True)
    assert "L4_SRC_PORT" not in out.columns
    assert "L4_DST_PORT" not in out.columns
    assert out.shape[1] == 42


def test_downcast_dtypes_matches_reference(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    df = cleaning.drop_identifier_columns(df)
    out = cleaning.downcast_dtypes(df)
    for col, dtype in schema.SHRUNK_DTYPES.items():
        assert out[col].dtype == dtype, f"{col}: expected {dtype}, got {out[col].dtype}"


def test_remove_invalid_rows_catches_the_synthetic_nan_and_inf_rows(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    df = cleaning.drop_identifier_columns(df)
    df = cleaning.downcast_dtypes(df)
    out, n_bad = cleaning.remove_invalid_rows(df)
    # exactly the 2 synthetic rows with a missing value / an inf value
    assert n_bad == 2
    assert out.shape[0] == 9
    assert not out.isna().any().any()
    assert np.isfinite(out.select_dtypes(include=[np.floating]).to_numpy()).all()


def test_remove_duplicates_catches_the_synthetic_duplicate_row(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    df = cleaning.drop_identifier_columns(df)
    df = cleaning.downcast_dtypes(df)
    df, _ = cleaning.remove_invalid_rows(df)
    out, n_dupes = cleaning.remove_duplicates(df)
    assert n_dupes == 1
    assert out.shape[0] == 8  # back down to exactly the 8 real rows
    assert list(out.index) == list(range(8))  # reset_index(drop=True)


def test_clean_pipeline_end_to_end_reproduces_the_8_real_rows(sample_raw_csv_path):
    df = cleaning.load_raw_csv(sample_raw_csv_path)
    clean_df, report = cleaning.clean_pipeline(df)

    assert report.raw_shape == (11, 46)
    assert report.after_drop_id_cols_shape == (11, 44)
    assert report.rows_with_nan_or_inf == 2
    assert report.after_dropna_shape == (9, 44)
    assert report.fully_duplicate_rows == 1
    assert report.final_shape == (8, 44)
    assert report.label_attack_mismatches == 0

    assert clean_df.shape == (8, 44)
    assert set(clean_df["Attack"]) == {"DoS", "Benign", "scanning"}
    assert set(clean_df["Dataset"]) == {"NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"}


def test_validate_label_matches_attack_detects_real_mismatch():
    df = pd.DataFrame(
        {
            "Attack": ["Benign", "DoS", "Benign"],
            "Label": np.array([0, 1, 1], dtype=np.int8),  # 3rd row is wrong on purpose
        }
    )
    mismatches = cleaning.validate_label_matches_attack(df)
    assert mismatches == 1


def test_drop_identifier_columns_raises_on_missing_columns():
    df = pd.DataFrame({"Label": [0, 1]})
    try:
        cleaning.drop_identifier_columns(df)
        assert False, "expected KeyError"
    except KeyError:
        pass
