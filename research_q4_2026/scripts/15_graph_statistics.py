#!/usr/bin/env python3
"""A8 - Graph statistics: why relational modelling helps in some cells and not others.

Computes, per dataset on the locked train split:
  * number of flows (edges) and distinct endpoints (nodes) and their ratio;
  * degree distribution (mean/median/p90/max);
  * edge multiplicity: distinct (src,dst) pairs vs flows - how much repeated
    endpoint interaction exists, i.e. how much structure a message-passing layer
    could exploit;
  * share of flows that are self-loops;
  * class-distribution entropy.

The hypothesis this supports: message passing can only add information when the
endpoint graph carries repeated, redundant interaction. Sparse graphs with close to
one flow per endpoint pair give the GNN nothing that the edge features do not
already contain.

Outputs: results/graph_statistics.csv, results/graph_statistics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
SPLITS = HERE / "data/full_splits"
RES = HERE / "results"
DATASETS = ["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"]
SHORT = {"NF-UNSW-NB15-v2": "UNSW", "NF-ToN-IoT-v2": "ToN",
         "NF-CSE-CIC-IDS2018-v2": "CSE-CIC", "NF-BoT-IoT-v2": "BoT-IoT"}


def main() -> int:
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='6GB'")
    rows, detail = [], {}
    for ds in DATASETS:
        train = (SPLITS / ds / "split=train" / "*.parquet").as_posix()
        S = "IPV4_SRC_ADDR || ':' || CAST(L4_SRC_PORT AS VARCHAR)"
        D = "IPV4_DST_ADDR || ':' || CAST(L4_DST_PORT AS VARCHAR)"
        base = con.execute(f"""
            SELECT count(*) AS flows,
                   count(DISTINCT (CASE WHEN {S} < {D} THEN {S} ELSE {D} END)) AS dummy
            FROM read_parquet('{train}')
        """).fetchone()
        flows = int(base[0])
        nodes = int(con.execute(f"""
            SELECT count(DISTINCT e) FROM (
                SELECT {S} e FROM read_parquet('{train}')
                UNION SELECT {D} e FROM read_parquet('{train}'))
        """).fetchone()[0])
        pairs = int(con.execute(f"""
            SELECT count(DISTINCT ({S}, {D})) FROM read_parquet('{train}')
        """).fetchone()[0])
        selfloops = int(con.execute(f"""
            SELECT count(*) FROM read_parquet('{train}') WHERE {S} = {D}
        """).fetchone()[0])
        deg = con.execute(f"""
            WITH ep AS (
                SELECT {S} e FROM read_parquet('{train}')
                UNION ALL
                SELECT {D} e FROM read_parquet('{train}'))
            SELECT e, count(*) d FROM ep GROUP BY e
        """).fetchnumpy()["d"]
        deg = np.asarray(deg, dtype=np.float64)
        ent = con.execute(f"""
            WITH c AS (SELECT Attack, count(*) n FROM read_parquet('{train}') GROUP BY Attack),
                 t AS (SELECT sum(n) AS tot FROM c)
            SELECT -sum((n / t.tot) * ln(n / t.tot)) FROM c, t
        """).fetchone()[0]
        row = {
            "dataset": ds,
            "label": SHORT[ds],
            "train_flows": flows,
            "distinct_endpoints": nodes,
            "edges_per_node": flows / max(nodes, 1),
            "distinct_endpoint_pairs": pairs,
            "flows_per_distinct_pair": flows / max(pairs, 1),
            "repeat_interaction_share": 1.0 - pairs / max(flows, 1),
            "self_loop_share": selfloops / max(flows, 1),
            "degree_mean": float(deg.mean()),
            "degree_median": float(np.median(deg)),
            "degree_p90": float(np.percentile(deg, 90)),
            "degree_max": float(deg.max()),
            "degree_gini": float(1 - 2 * np.trapezoid(np.sort(deg).cumsum() / deg.sum(),
                                                      dx=1 / len(deg))),
            "class_entropy_nats": float(ent),
        }
        rows.append(row)
        detail[ds] = row
        print(json.dumps(row, indent=1), flush=True)
    con.close()
    df = pd.DataFrame(rows)
    df.to_csv(RES / "graph_statistics.csv", index=False)
    (RES / "graph_statistics.json").write_text(json.dumps(detail, indent=2))
    print()
    print(df[["label", "train_flows", "distinct_endpoints", "edges_per_node",
              "flows_per_distinct_pair", "repeat_interaction_share"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
