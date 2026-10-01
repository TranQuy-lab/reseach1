#!/usr/bin/env python3
"""B7 - Endpoint-holdout evaluation with CPU models (Gate C, partial).

Question: does a flow-only model degrade when its test flows involve endpoints that
never appear in training? This isolates endpoint familiarity from model family,
because the *same* architecture is trained on both splits.

Design
------
* Split A: the locked flow_group_id split (endpoints overlap train heavily).
* Split B: the endpoint-disjoint `holdout` split (`0` holdout endpoints in train).
* Same architecture (`mlp_h128_1l` control and `mlp_h273_2l` capacity-matched),
  same budget rule, same seeds. The scaler is fit on each split's own train,
  because the training population differs - this is stated, not hidden.
* Reported per class so that we can distinguish "everything degrades" from
  "only some classes lose support".

Outputs: results/tn4_holdout_runs.csv, results/holdout_comparison.csv,
         results/holdout_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parents[1]
ENDP = HERE / "data/endpoint_splits"
WORK = HERE / "work/holdout_features"
OUT = HERE / "results"
WORK.mkdir(parents=True, exist_ok=True)

FEATURES = None  # filled from the archived preprocessor
VARIANTS = {"mlp_h128_1l": (1, 128), "mlp_h273_2l": (2, 273)}
BATCH, LR, DROPOUT, MIN_STEPS = 4096, 1e-3, 0.2, 1500


class MLP(nn.Module):
    def __init__(self, d_in, n_classes, depth, hidden):
        super().__init__()
        layers, prev = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = hidden
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def materialise(dataset: str, task: str) -> dict:
    """Fit the scaler on the holdout-train split and cache float32 memmaps."""
    import json as _json
    repo = Path("/home/noble-tran/nghiencuu/repo_reseach1")
    pre = _json.loads((repo / "research/artifacts/full_runs" /
                       f"{dataset}__{task}__edge_mlp__seed11/preprocessor.json").read_text())
    feats = pre["features"]
    log_idx = [feats.index(c) for c in pre["log_features"]]
    root = ENDP / f"{dataset}__ipport__holdout"
    out = {}
    for split in ("train", "val", "test"):
        files = sorted((root / f"split={split}").glob("*.parquet"))
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        dest = WORK / f"{dataset}__{task}__{split}.npy"
        if dest.exists():
            out[split] = {"rows": n, "path": str(dest)}
            continue
        acc, pos = None, 0
        for f in files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=500_000,
                                                    columns=feats + ["Attack", "Label"]):
                x = np.column_stack([b.column(c).to_numpy(zero_copy_only=False)
                                     for c in feats]).astype(np.float64)
                if log_idx:
                    x[:, log_idx] = np.sign(x[:, log_idx]) * np.log1p(np.abs(x[:, log_idx]))
                if acc is None:
                    acc = np.empty((n, len(feats)), dtype=np.float64)
                acc[pos:pos + len(x)] = x
                pos += len(x)
        out[split] = {"rows": n, "path": str(dest)}
        if split == "train":
            mean, scale = acc.mean(axis=0), acc.std(axis=0)
            scale[scale == 0] = 1.0
            np.savez(WORK / f"{dataset}__{task}__scaler.npz", mean=mean, scale=scale)
            np.save(WORK / f"{dataset}__{task}__train.npy", ((acc - mean) / scale).astype(np.float32))
        else:
            sc = np.load(WORK / f"{dataset}__{task}__scaler.npz")
            np.save(dest, ((acc - sc["mean"]) / sc["scale"]).astype(np.float32))
        # labels
        if split == "train" or not (WORK / f"{dataset}__{task}__{split}__y.npy").exists():
            y = np.empty(n, dtype=np.int16)
            p2 = 0
            names = None
            for f in files:
                for b in pq.ParquetFile(f).iter_batches(batch_size=500_000,
                                                        columns=["Attack", "Label"]):
                    if names is None:
                        names = sorted(set(b.column("Attack").to_pylist()))
                    v = b.column("Attack" if task == "multiclass" else "Label").to_pylist()
                    if task == "multiclass":
                        lut = {k: i for i, k in enumerate(names)}
                        v = [lut[x] for x in v]
                    arr = np.asarray(v, dtype=np.int16)
                    y[p2:p2 + len(arr)] = arr
                    p2 += len(arr)
            np.save(WORK / f"{dataset}__{task}__{split}__y.npy", y)
    return out


@torch.no_grad()
def predict(model, X, batch=200_000):
    model.eval()
    out = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        out[i:i + batch] = model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))).argmax(1).numpy()
    return out


def evaluate(model, X, y):
    p = predict(model, X)
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
            "per_class_f1": f1_score(y, p, average=None, zero_division=0).tolist()}


def run_one(dataset, task, variant, seed):
    depth, hidden = VARIANTS[variant]
    Xtr = np.load(WORK / f"{dataset}__{task}__train.npy", mmap_mode="r")
    Xva = np.load(WORK / f"{dataset}__{task}__val.npy", mmap_mode="r")
    Xte = np.load(WORK / f"{dataset}__{task}__test.npy", mmap_mode="r")
    ytr = np.load(WORK / f"{dataset}__{task}__train__y.npy")
    yva = np.load(WORK / f"{dataset}__{task}__val__y.npy")
    yte = np.load(WORK / f"{dataset}__{task}__test__y.npy")
    n_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    n_train = len(ytr)
    spp = math.ceil(n_train / BATCH)
    max_steps = max(MIN_STEPS, 2 * spp)
    eval_every = max(500, math.ceil(spp / 4))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1], n_classes, depth, hidden)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(
        n_train / (n_classes * np.maximum(counts, 1)), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"macro_f1": -1.0, "state": None, "step": 0}
    step, t0 = 0, time.perf_counter()
    while step < max_steps:
        perm = rng.permutation(n_train)
        for i in range(0, n_train, BATCH):
            if step >= max_steps:
                break
            idx = np.sort(perm[i:i + BATCH])
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(torch.from_numpy(np.ascontiguousarray(Xtr[idx]))),
                         torch.from_numpy(ytr[idx].astype(np.int64)))
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                m = evaluate(model, Xva, yva)
                if step >= spp and m["macro_f1"] > best["macro_f1"]:
                    best = {"macro_f1": m["macro_f1"], "step": step,
                            "state": {k: v.clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    val = evaluate(model, Xva, yva)
    test = evaluate(model, Xte, yte)
    return {"dataset": dataset, "task": task, "model": variant, "seed": seed,
            "split": "endpoint_holdout",
            "n_train": n_train, "n_test": len(yte), "steps_ran": step,
            "parameters": sum(p.numel() for p in model.parameters()),
            "val_macro_f1": val["macro_f1"], "test_macro_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"], "test_accuracy": test["accuracy"],
            "test_per_class_f1": json.dumps(test["per_class_f1"]),
            "seconds": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2"])
    ap.add_argument("--tasks", nargs="+", default=["multiclass", "binary"])
    ap.add_argument("--models", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    args = ap.parse_args()
    runs = OUT / "tn4_holdout_runs.csv"
    done = set()
    if runs.exists():
        prev = pd.read_csv(runs)
        done = set(zip(prev.dataset, prev.task, prev.model, prev.seed))
    for dataset in args.datasets:
        for task in args.tasks:
            materialise(dataset, task)
            for variant in args.models:
                for seed in args.seeds:
                    if (dataset, task, variant, seed) in done:
                        continue
                    row = run_one(dataset, task, variant, seed)
                    pd.DataFrame([row]).to_csv(runs, mode="a",
                                               header=not runs.exists(), index=False)
                    print(f"[{time.strftime('%H:%M:%S')}] {dataset} {task} {variant} s{seed} "
                          f"TEST macro_f1={row['test_macro_f1']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
