"""
schema.py
=========
Column definitions and dtype maps for the NF-UQ-NIDS-v2 dataset.

Every constant in this file is traceable to one of two sources, noted inline:

  [NOTEBOOK]  Copied or derived directly from the executed cell outputs of the
              team's reference notebook, ``NF-UQ-NIDS-V2-00-Cleaning`` (Kaggle).
              Where a notebook cell printed a concrete number (row counts,
              value_counts, dtypes table), that number is reproduced verbatim
              here as a named constant so later runs can be checked against it
              (see ``pipeline.compare_to_reference``).

  [DATASET]   Copied or derived from the official dataset documentation
              page for NF-UQ-NIDS-v2 / the NetFlow V2 family (feature
              descriptions, sub-dataset names, citation), as supplied by the
              team.

Nothing in this file is invented. Where the team still needs to make a
judgment call (e.g. whether to also drop the L4 port columns), that is
called out explicitly as an OPEN DECISION rather than silently defaulted.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. Raw column dtypes, exactly as read by the reference notebook.  [NOTEBOOK]
#    (its `suggested_dtypes` dict, cell 3). The CSV also carries a `Dataset`
#    column that the reference notebook does NOT pass a dtype for -> pandas
#    infers it as `object`; we make that explicit here instead of leaving it
#    implicit, since an unspecified dtype is a common source of silent bugs
#    on a 76M-row file.
# ---------------------------------------------------------------------------

IP_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
PORT_COLUMNS = ["L4_SRC_PORT", "L4_DST_PORT"]
LABEL_COL = "Label"
ATTACK_COL = "Attack"
DATASET_COL = "Dataset"

RAW_DTYPES: dict[str, object] = {
    "IPV4_SRC_ADDR": "object",
    "L4_SRC_PORT": np.int16,
    "IPV4_DST_ADDR": "object",
    "L4_DST_PORT": np.int16,
    "PROTOCOL": np.int16,
    "L7_PROTO": np.float32,
    "IN_BYTES": np.int64,
    "IN_PKTS": np.int32,
    "OUT_BYTES": np.int64,
    "OUT_PKTS": np.int32,
    "TCP_FLAGS": np.int16,
    "CLIENT_TCP_FLAGS": np.int16,
    "SERVER_TCP_FLAGS": np.int16,
    "FLOW_DURATION_MILLISECONDS": np.int64,
    "DURATION_IN": np.int64,
    "DURATION_OUT": np.int64,
    "MIN_TTL": np.int16,
    "MAX_TTL": np.int16,
    "LONGEST_FLOW_PKT": np.int32,
    "SHORTEST_FLOW_PKT": np.int32,
    "MIN_IP_PKT_LEN": np.int32,
    "MAX_IP_PKT_LEN": np.int32,
    "SRC_TO_DST_SECOND_BYTES": np.float32,
    "DST_TO_SRC_SECOND_BYTES": np.float32,
    "RETRANSMITTED_IN_BYTES": np.int32,
    "RETRANSMITTED_IN_PKTS": np.int32,
    "RETRANSMITTED_OUT_BYTES": np.int32,
    "RETRANSMITTED_OUT_PKTS": np.int32,
    "SRC_TO_DST_AVG_THROUGHPUT": np.int64,
    "DST_TO_SRC_AVG_THROUGHPUT": np.int64,
    "NUM_PKTS_UP_TO_128_BYTES": np.int32,
    "NUM_PKTS_128_TO_256_BYTES": np.int32,
    "NUM_PKTS_256_TO_512_BYTES": np.int32,
    "NUM_PKTS_512_TO_1024_BYTES": np.int32,
    "NUM_PKTS_1024_TO_1514_BYTES": np.int32,
    "TCP_WIN_MAX_IN": np.int32,
    "TCP_WIN_MAX_OUT": np.int32,
    "ICMP_TYPE": np.int32,
    "ICMP_IPV4_TYPE": np.int32,
    "DNS_QUERY_ID": np.int32,
    "DNS_QUERY_TYPE": np.int32,
    "DNS_TTL_ANSWER": np.int32,
    "FTP_COMMAND_RET_CODE": np.float32,
    "Label": np.int32,
    "Attack": "object",
    "Dataset": "object",  # made explicit; inferred implicitly in the reference notebook
}

# ---------------------------------------------------------------------------
# 2. Post-cleaning ("shrunk") dtypes.  [NOTEBOOK]
#    Reproduced verbatim from the reference notebook's cell 11 output, i.e.
#    the actual result of `df_shrink(df, obj2cat=False, int2uint=False)` on
#    this dataset after the two IP columns were dropped.
#
#    IMPROVEMENT (documented, not silent): the reference notebook pulls in
#    `fastai` solely to call `df_shrink`. fastai is a heavy dependency
#    (torch + fastai) for a one-line dtype downcast. Because the exact
#    input -> output dtype mapping produced by df_shrink on this dataset is
#    fully visible in the notebook's own printed output, we hard-code that
#    verified mapping below and apply it with plain `astype`. This removes
#    the fastai/torch dependency for every team member while reproducing
#    identical dtypes. If you want to double-check this claim yourself,
#    diff SHRUNK_DTYPES below against cell 11 of the reference notebook.
# ---------------------------------------------------------------------------

SHRUNK_DTYPES: dict[str, object] = {
    "L4_SRC_PORT": np.int16,
    "L4_DST_PORT": np.int16,
    "PROTOCOL": np.int16,
    "L7_PROTO": np.float32,
    "IN_BYTES": np.int32,
    "IN_PKTS": np.int32,
    "OUT_BYTES": np.int32,
    "OUT_PKTS": np.int32,
    "TCP_FLAGS": np.int16,
    "CLIENT_TCP_FLAGS": np.int16,
    "SERVER_TCP_FLAGS": np.int16,
    "FLOW_DURATION_MILLISECONDS": np.int32,
    "DURATION_IN": np.int32,
    "DURATION_OUT": np.int32,
    "MIN_TTL": np.int16,
    "MAX_TTL": np.int16,
    "LONGEST_FLOW_PKT": np.int32,
    "SHORTEST_FLOW_PKT": np.int16,
    "MIN_IP_PKT_LEN": np.int16,
    "MAX_IP_PKT_LEN": np.int32,
    "SRC_TO_DST_SECOND_BYTES": np.float32,
    "DST_TO_SRC_SECOND_BYTES": np.float32,
    "RETRANSMITTED_IN_BYTES": np.int32,
    "RETRANSMITTED_IN_PKTS": np.int16,
    "RETRANSMITTED_OUT_BYTES": np.int32,
    "RETRANSMITTED_OUT_PKTS": np.int16,
    "SRC_TO_DST_AVG_THROUGHPUT": np.int64,
    "DST_TO_SRC_AVG_THROUGHPUT": np.int64,
    "NUM_PKTS_UP_TO_128_BYTES": np.int32,
    "NUM_PKTS_128_TO_256_BYTES": np.int32,
    "NUM_PKTS_256_TO_512_BYTES": np.int32,
    "NUM_PKTS_512_TO_1024_BYTES": np.int32,
    "NUM_PKTS_1024_TO_1514_BYTES": np.int32,
    "TCP_WIN_MAX_IN": np.int32,
    "TCP_WIN_MAX_OUT": np.int32,
    "ICMP_TYPE": np.int32,
    "ICMP_IPV4_TYPE": np.int16,
    "DNS_QUERY_ID": np.int32,
    "DNS_QUERY_TYPE": np.int32,
    "DNS_TTL_ANSWER": np.int32,
    "FTP_COMMAND_RET_CODE": np.float32,
    "Label": np.int8,
    # Attack, Dataset stay `object` -- unchanged by df_shrink(obj2cat=False)
}

# ---------------------------------------------------------------------------
# 3. Sub-dataset names.  [DATASET] -- the four NetFlow v2 sources merged into
#    NF-UQ-NIDS-v2, as named in the dataset documentation and in project memory.
# ---------------------------------------------------------------------------

VALID_DATASETS = [
    "NF-UNSW-NB15-v2",
    "NF-BoT-IoT-v2",
    "NF-ToN-IoT-v2",
    "NF-CSE-CIC-IDS2018-v2",
]

# ---------------------------------------------------------------------------
# 4. Reference statistics.  [NOTEBOOK] -- the exact numbers printed by the
#    reference notebook's own cell outputs, used only to *check* that a run
#    of this pipeline on the same raw CSV reproduces them (see
#    pipeline.compare_to_reference). These are NOT assumptions the pipeline
#    relies on; they are a regression check against a verified source.
# ---------------------------------------------------------------------------

REFERENCE_STATS = {
    "raw_shape": (75_987_976, 46),
    "label_value_counts": {1: 50_822_681, 0: 25_165_295},
    "rows_with_nan_or_inf": 384,
    "fully_duplicate_rows": 13_315_579,
    # 75,987,976 - 384 - 13,315,579, columns = 46 - 2 (dropped IP columns)
    "expected_clean_shape": (75_987_976 - 384 - 13_315_579, 44),
}

# ---------------------------------------------------------------------------
# 5. VERIFIED RELATIONSHIP: Label is fully derived from Attack.  [DATASET]
#    Checked by hand against the Attack/Label value_counts printed in the
#    reference notebook (cells 7-8): summing every non-"Benign" Attack
#    category gives exactly 50,822,681, which equals Label == 1's count, and
#    "Benign" (25,165,295) equals Label == 0's count exactly. So:
#        Label == 0  <=>  Attack == "Benign"
#        Label == 1  <=>  Attack != "Benign"
#    This lets the pipeline validate (not assume) label consistency on any
#    subset, and lets Person B/C treat `Attack` as the ground-truth column
#    (multi-class) and derive `Label` (binary) from it rather than trusting
#    two separately-encoded columns to agree.
# ---------------------------------------------------------------------------

BENIGN_VALUE = "Benign"

# ---------------------------------------------------------------------------
# 6. OPEN DECISION for the team (not defaulted silently):
#    The reference notebook drops only IPV4_SRC_ADDR / IPV4_DST_ADDR and
#    KEEPS L4_SRC_PORT / L4_DST_PORT. However, the dataset authors' own
#    preprocessing for the earlier NF-UQ-NIDS work is reported elsewhere as
#    excluding IP addresses *and* their associated ports from the training
#    features (8 of 12 candidate features kept). The two are not the same
#    choice. `cleaning.drop_identifier_columns(..., drop_ports=False)`
#    defaults to matching the reference notebook exactly (ports kept), so
#    the default pipeline output is reference-reproducible. Pass
#    `drop_ports=True` if the team decides to follow the stricter
#    IP+port exclusion instead -- but that is a team decision to make
#    explicitly (e.g. record it in FUTURE_WORK.md / results_log.csv), not
#    something this script should assume.
# ---------------------------------------------------------------------------

# Short human-readable descriptions for a subset of columns, used only for
# generating documentation/EDA output -- not used in any computation.
# [DATASET] copied from the feature dictionary supplied by the team.
FEATURE_DESCRIPTIONS = {
    "L4_SRC_PORT": "IPv4 source port number",
    "L4_DST_PORT": "IPv4 destination port number",
    "PROTOCOL": "IP protocol identifier byte",
    "L7_PROTO": "Layer 7 protocol (numeric)",
    "IN_BYTES": "Incoming number of bytes",
    "IN_PKTS": "Incoming number of packets",
    "OUT_BYTES": "Outgoing number of bytes",
    "OUT_PKTS": "Outgoing number of packets",
    "TCP_FLAGS": "Cumulative of all TCP flags",
    "FLOW_DURATION_MILLISECONDS": "Flow duration in milliseconds",
    "MIN_TTL": "Min flow TTL",
    "MAX_TTL": "Max flow TTL",
    "Label": "Binary label (0 = benign, 1 = attack)",
    "Attack": "Attack category (multi-class ground truth; 'Benign' <=> Label 0)",
    "Dataset": "Which of the four NetFlow v2 sub-datasets this flow came from",
}
