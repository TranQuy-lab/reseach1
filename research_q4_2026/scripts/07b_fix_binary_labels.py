#!/usr/bin/env python3
"""Regenerate binary label arrays directly from the integer `Label` column.

The repo's binary task uses `frame.Label` verbatim with class order
["Benign", "Attack"] (src/nids_minibatch/data.py line 73). An earlier version of
script 07 mistakenly used a name lookup on that integer column; this repairs the
already-materialised arrays without recomputing the feature matrices.
"""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
SPLITS = HERE / "data/full_splits"
WORK = HERE / "work/features"
DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]

for ds in DATASETS:
    for split in ("train", "val", "test"):
        files = sorted((SPLITS / ds / f"split={split}").glob("*.parquet"))
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        y = np.empty(n, dtype=np.int16)
        pos = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=500_000, columns=["Label"]):
                v = np.asarray(b.column("Label").to_pylist(), dtype=np.int16)
                y[pos:pos + len(v)] = v
                pos += len(v)
        assert pos == n
        assert set(np.unique(y)) <= {0, 1}, (ds, split, np.unique(y))
        np.save(WORK / f"{ds}__{split}__y_binary.npy", y)
        print(f"{ds:24s} {split:5s} n={n:>10,d} classes={np.bincount(y).tolist()}")
print("binary labels repaired")
