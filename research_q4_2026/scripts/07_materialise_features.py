#!/usr/bin/env python3
"""B1 - Materialise the locked split into disk-backed, preprocessed feature matrices.

Purpose
-------
Provide exactly the same inputs the archived GNN runs used, so that new comparators
(capacity-matched MLPs, tabular models) are comparable run-for-run. The scaler is
NOT refit: it is read from the archived ``preprocessor.json`` that the GNN runs
themselves used.

Pipeline reproduced from ``src/nids_minibatch/data.py`` (read-only):
  1. signed log1p on the columns listed in ``log_features``
  2. (x - mean) / scale with the archived train-fitted statistics

Output (per dataset):
  work/features/<dataset>__<split>.npy    float32 memmap, shape (n, 39)
  work/features/<dataset>__<split>__labels.npy   int16 class indices, per task stored in json
  results/preprocessed_manifest.json

Nothing inside the source repository is written to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/noble-tran/nghiencuu/repo_reseach1")
RES = REPO / "research"
HERE = Path(__file__).resolve().parents[1]
SPLITS = HERE / "data/full_splits"
WORK = HERE / "work/features"
OUT = HERE / "results"
WORK.mkdir(parents=True, exist_ok=True)

DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
BATCH = 500_000


def archived_preprocessor(dataset: str, task: str) -> dict:
    """Read the archived scaler; assert it is equivalent across the five seeds.

    Note: seeds 44/55 were produced in a separate batch and their scalers differ
    from seeds 11/22/33 by up to 3e-11 relative on extreme-magnitude columns
    (e.g. SRC_TO_DST_SECOND_BYTES scale ~1.24e12). This is float accumulation-order
    noise, not a preprocessing change, so equivalence is checked on RELATIVE
    drift and the observed drift is recorded.
    """
    seen = None
    drift = 0.0
    for base in (RES / "artifacts/full_runs", RES / "artifacts/full_runs_seeds44_55"):
        for d in sorted(base.glob(f"{dataset}__{task}__edge_mlp__seed*/preprocessor.json")):
            cur = json.loads(d.read_text())
            if seen is None:
                seen = cur
                continue
            a_m, b_m = np.asarray(seen["mean"]), np.asarray(cur["mean"])
            a_s, b_s = np.asarray(seen["scale"]), np.asarray(cur["scale"])
            rel = max(
                float((np.abs(a_m - b_m) / np.maximum(np.abs(a_m), 1e-300)).max()),
                float((np.abs(a_s - b_s) / np.maximum(np.abs(a_s), 1e-300)).max()),
            )
            drift = max(drift, rel)
            assert rel < 1e-6, f"scaler materially differs for {dataset}/{task}: {d} rel={rel:g}"
            assert cur["classes"] == seen["classes"], f"classes differ: {d}"
    if seen is None:
        raise FileNotFoundError(f"no archived preprocessor for {dataset}/{task}")
    seen["_observed_relative_scaler_drift"] = drift
    return seen


def main() -> int:
    manifest: dict = {"datasets": {}}
    for dataset in DATASETS:
        pre_mc = archived_preprocessor(dataset, "multiclass")
        pre_bin = archived_preprocessor(dataset, "binary")
        assert pre_mc["features"] == pre_bin["features"]
        assert np.allclose(pre_mc["mean"], pre_bin["mean"], rtol=1e-12, atol=0)
        assert np.allclose(pre_mc["scale"], pre_bin["scale"], rtol=1e-12, atol=0)
        assert pre_mc["log_features"] == pre_bin["log_features"]

        features = pre_mc["features"]
        mean = np.asarray(pre_mc["mean"], dtype=np.float64)
        scale = np.asarray(pre_mc["scale"], dtype=np.float64)
        log_idx = [features.index(c) for c in pre_mc["log_features"]]
        classes_mc = pre_mc["classes"]
        classes_bin = pre_bin["classes"]
        ds_info = {"features": features, "classes_multiclass": classes_mc,
                   "classes_binary": classes_bin,
                   "observed_relative_scaler_drift_across_seeds":
                       pre_mc["_observed_relative_scaler_drift"],
                   "splits": {}}

        for split in ("train", "val", "test"):
            part = SPLITS / dataset / f"split={split}"
            files = sorted(part.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"missing split files: {part}")
            total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            fx = np.lib.format.open_memmap(WORK / f"{dataset}__{split}.npy", mode="w+",
                                           dtype=np.float32, shape=(total, len(features)))
            y_mc = np.empty(total, dtype=np.int16)
            y_bin = np.empty(total, dtype=np.int16)
            pos = 0
            for f in files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=BATCH,
                                             columns=features + ["Attack", "Label"]):
                    x = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False)
                                         for c in features]).astype(np.float64)
                    if log_idx:
                        x[:, log_idx] = np.sign(x[:, log_idx]) * np.log1p(np.abs(x[:, log_idx]))
                    x = (x - mean) / scale
                    n = x.shape[0]
                    fx[pos:pos + n] = x.astype(np.float32)
                    att = np.asarray(batch.column("Attack").to_pylist())
                    # multiclass: repo maps class NAME via lookup over sorted classes
                    mc_lookup = {name: i for i, name in enumerate(classes_mc)}
                    y_mc[pos:pos + n] = np.array([mc_lookup[v] for v in att], dtype=np.int16)
                    # binary: repo uses the integer `Label` column directly and the
                    # fixed class order ["Benign", "Attack"]
                    assert classes_bin == ["Benign", "Attack"], classes_bin
                    y_bin[pos:pos + n] = np.asarray(
                        batch.column("Label").to_pylist(), dtype=np.int16)
                    pos += n
            fx.flush()
            del fx
            assert pos == total, f"{dataset}/{split}: wrote {pos} of {total}"
            np.save(WORK / f"{dataset}__{split}__y_multiclass.npy", y_mc)
            np.save(WORK / f"{dataset}__{split}__y_binary.npy", y_bin)
            counts_mc = np.bincount(y_mc, minlength=len(classes_mc)).tolist()
            counts_bin = np.bincount(y_bin, minlength=len(classes_bin)).tolist()
            ds_info["splits"][split] = {
                "rows": int(total), "multiclass_counts": counts_mc,
                "binary_counts": counts_bin,
            }
            print(f"{dataset:26s} {split:5s} rows={total:>10,d} "
                  f"mc={counts_mc} bin={counts_bin}", flush=True)
        manifest["datasets"][dataset] = ds_info

    (OUT / "preprocessed_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nwrote", OUT / "preprocessed_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
