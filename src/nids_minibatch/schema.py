"""Shared schema for the four NF-UQ-NIDS-v2 constituent datasets."""

from nids_research.schema import FEATURES

DATASETS = [
    "NF-UNSW-NB15-v2",
    "NF-BoT-IoT-v2",
    "NF-ToN-IoT-v2",
    "NF-CSE-CIC-IDS2018-v2",
]

ENDPOINT_COLUMNS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
]

TARGET_COLUMNS = ["Attack", "Label"]
TRACE_COLUMNS = ["Dataset", "source_row_id", "flow_group_id"]
REQUIRED = ENDPOINT_COLUMNS + FEATURES + TARGET_COLUMNS + TRACE_COLUMNS

SPLIT_SEED = 20260920
SAMPLE_SEED = 20260921
SPLIT_MODULUS = 10
SPLIT_RANGES = {"train": tuple(range(0, 7)), "val": (7,), "test": (8, 9)}
SAMPLE_CAPS = {"train": 20_000, "val": 5_000, "test": 10_000}
