"""
make_fixture.py
================
Builds tests/fixtures/sample_raw.csv, used by the unit tests.

REAL_ROWS below are copied verbatim from the 8 example rows shown in the
NF-UQ-NIDS-v2 dataset documentation supplied by the team (real flows from
NF-BoT-IoT-v2, NF-ToN-IoT-v2 and NF-CSE-CIC-IDS2018-v2).

SYNTHETIC_EXTRA_ROWS are NOT from the dataset. They are constructed here,
clearly labelled, purely to exercise the cleaning logic's edge cases
(an exact duplicate, a missing value, an infinite value) which the 8 real
rows alone don't happen to contain. Do not mistake these for real traffic.

Run: python make_fixture.py   (writes sample_raw.csv next to this script)
"""

import csv
from pathlib import Path

HEADER = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL",
    "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "Label", "Attack", "Dataset",
]
assert len(HEADER) == 46

# --- REAL rows, copied from the dataset documentation ----------------------
REAL_ROWS = [
    ["192.168.100.148", 65389, "192.168.100.7", 80, 6, 7.0, 420, 3, 0, 0, 2, 2, 0,
     4293092, 1875, 0, 64, 64, 140, 140, 0, 140, 140280.0, 0.0, 140, 1, 0, 0,
     1120000, 0, 0, 3, 0, 0, 0, 512, 0, 35840, 140, 0, 0, 0, 0.0, 1, "DoS", "NF-BoT-IoT-v2"],
    ["192.168.100.148", 11154, "192.168.100.5", 80, 6, 7.0, 280, 2, 40, 1, 22, 2, 20,
     4294499, 453, 0, 64, 64, 140, 40, 40, 140, 280.0, 40.0, 0, 0, 0, 0,
     0, 320000, 1, 2, 0, 0, 0, 512, 0, 0, 0, 0, 0, 0, 0.0, 1, "DoS", "NF-BoT-IoT-v2"],
    ["192.168.1.31", 42062, "192.168.1.79", 1041, 6, 0.0, 44, 1, 40, 1, 22, 2, 20,
     0, 0, 0, 0, 0, 44, 40, 40, 44, 44.0, 40.0, 0, 0, 0, 0,
     352000, 320000, 2, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0.0, 0, "Benign", "NF-ToN-IoT-v2"],
    ["192.168.1.34", 46849, "192.168.1.79", 9110, 6, 0.0, 44, 1, 40, 1, 22, 2, 20,
     0, 0, 0, 0, 0, 44, 40, 40, 44, 44.0, 40.0, 0, 0, 0, 0,
     352000, 320000, 2, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0.0, 0, "Benign", "NF-ToN-IoT-v2"],
    ["192.168.1.30", 50360, "192.168.1.152", 1084, 6, 0.0, 44, 1, 40, 1, 22, 2, 20,
     0, 0, 0, 0, 0, 44, 40, 40, 44, 44.0, 40.0, 0, 0, 0, 0,
     352000, 320000, 2, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0.0, 0, "Benign", "NF-ToN-IoT-v2"],
    ["172.31.66.53", 51860, "77.93.254.178", 443, 6, 91.0, 152, 3, 120, 3, 214, 194, 20,
     0, 0, 0, 128, 128, 52, 40, 40, 52, 152.0, 120.0, 0, 0, 0, 0,
     1216000, 960000, 6, 0, 0, 0, 0, 8192, 0, 0, 0, 0, 0, 0, 0.0, 0, "Benign", "NF-CSE-CIC-IDS2018-v2"],
    ["192.168.1.32", 56402, "192.168.1.169", 9012, 6, 0.0, 232, 4, 132, 3, 31, 30, 19,
     0, 0, 0, 64, 64, 92, 40, 40, 92, 232.0, 132.0, 0, 0, 0, 0,
     1856000, 1056000, 7, 0, 0, 0, 0, 29200, 65535, 0, 0, 0, 0, 0, 0.0, 0, "Benign", "NF-ToN-IoT-v2"],
    ["192.168.1.31", 54001, "192.168.1.180", 22, 6, 92.0, 84, 2, 88, 2, 22, 6, 18,
     4294952, 15, 15, 64, 64, 44, 40, 40, 44, 84.0, 88.0, 0, 0, 0, 0,
     40000, 40000, 4, 0, 0, 0, 0, 1024, 29200, 0, 0, 0, 0, 0, 0.0, 1, "scanning", "NF-ToN-IoT-v2"],
]
assert all(len(r) == 46 for r in REAL_ROWS)

# --- SYNTHETIC edge-case rows (NOT real data) -------------------------------
# NOTE: the missing-value row below blanks out L7_PROTO, a *float* column
# (index 5), not an integer column. This matches how missing/invalid values
# actually surface in the real dataset: the reference notebook's own
# "384 rows with NaN" come entirely from +-inf in float columns being
# replaced with NaN (cell 12) -- inf/NaN cannot occur in a genuinely
# integer-typed column. A blank value in an *integer* column is not a
# scenario the real dataset exhibits, and `pd.read_csv` with strict integer
# dtypes will hard-crash on it rather than clean it (found while building
# this fixture) -- see the "Known limitations" note in README.md.
SYNTHETIC_EXTRA_ROWS = [
    # exact duplicate of REAL_ROWS[0] -> exercises drop_duplicates
    list(REAL_ROWS[0]),
    # copy of REAL_ROWS[1] with L7_PROTO (float col) missing -> exercises NaN removal
    (lambda r: (r.__setitem__(5, ""), r)[1])(list(REAL_ROWS[1])),
    # copy of REAL_ROWS[2] with SRC_TO_DST_SECOND_BYTES = inf -> exercises inf removal
    (lambda r: (r.__setitem__(22, "inf"), r)[1])(list(REAL_ROWS[2])),
]


def main():
    out_path = Path(__file__).parent / "sample_raw.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for row in REAL_ROWS + SYNTHETIC_EXTRA_ROWS:
            writer.writerow(row)
    print(f"Wrote {len(REAL_ROWS) + len(SYNTHETIC_EXTRA_ROWS)} rows to {out_path}")


if __name__ == "__main__":
    main()
