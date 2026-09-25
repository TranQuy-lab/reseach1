import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from nids_minibatch.prepare import prepare, split_case
from nids_minibatch.schema import DATASETS, FEATURES


def source_frame(dataset, n=1000):
    rng = np.random.default_rng(88)
    frame = pd.DataFrame(rng.integers(0, 100, size=(n, len(FEATURES))), columns=FEATURES)
    frame["L7_PROTO"] = frame["L7_PROTO"].astype(float) + 0.25
    frame["SRC_TO_DST_SECOND_BYTES"] = frame["SRC_TO_DST_SECOND_BYTES"].astype(float)
    frame["DST_TO_SRC_SECOND_BYTES"] = frame["DST_TO_SRC_SECOND_BYTES"].astype(float)
    frame["FTP_COMMAND_RET_CODE"] = frame["FTP_COMMAND_RET_CODE"].astype(float)
    frame["IPV4_SRC_ADDR"] = [f"10.0.{i // 250}.{i % 250 + 1}" for i in range(n)]
    frame["IPV4_DST_ADDR"] = "192.0.2.1"
    frame["L4_SRC_PORT"] = np.arange(n) % 65536
    frame["L4_DST_PORT"] = 443
    frame["Attack"] = np.where(np.arange(n) % 2, "DDoS", "Benign")
    frame["Label"] = (frame.Attack != "Benign").astype("int64")
    frame["Dataset"] = dataset
    frame["source_row_id"] = np.arange(1, n + 1)
    frame["flow_group_id"] = [f"{i:032x}" for i in range(n)]
    return frame


def test_split_case_keeps_equal_groups_together():
    con = duckdb.connect()
    values = pd.DataFrame({"flow_group_id": ["a" * 32, "a" * 32, "b" * 32]})
    con.register("values_table", values)
    got = con.execute(f"SELECT {split_case()} split FROM values_table").fetchall()
    assert got[0] == got[1]
    assert got[0][0] in {"train", "val", "test"}


def test_prepare_four_splits_and_pilots(tmp_path):
    source = tmp_path / "processed_four"
    source.mkdir()
    for dataset in DATASETS:
        source_frame(dataset).to_parquet(source / f"{dataset}.parquet", index=False)
    output = tmp_path / "out"
    report = tmp_path / "report.json"
    result = prepare(source, output, report, threads=1)
    assert result["complete"]
    assert result["protocol"] == "PROTOCOL_MINIBATCH_VI.md"
    assert json.loads(report.read_text())["split_seed"] == 20260920
    for dataset in DATASETS:
        info = result["datasets"][dataset]
        assert info["source_rows"] == 1000
        assert info["cross_split_groups"] == 0
        assert set(info["ip_overlap_with_train"]) == {"val", "test"}
        assert all(
            0 <= item["fraction_also_in_train"] <= 1
            for item in info["ip_overlap_with_train"].values()
        )
        assert sum(x["rows"] for x in info["split_rows"].values()) == 1000
        for split in ("train", "val", "test"):
            assert info["split_rows"][split]["conflicting_label_groups"] == 0
            pilot = pd.read_parquet(output / dataset / f"pilot_{split}.parquet")
            assert len(pilot) == info["split_rows"][split]["rows"]
            assert set(pilot.Attack) == {"Benign", "DDoS"}
    with pytest.raises(ValueError, match="refusing overwrite"):
        prepare(source, output, report, threads=1)
