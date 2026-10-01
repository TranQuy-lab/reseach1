#!/usr/bin/env python3
"""TN-10b - Endpoint isolation with a low-memory model (capacity-matched MLP).

Why this variant exists
-----------------------
TN-10 uses HistGradientBoosting, which peaks at 10.5 GiB on NF-CSE-CIC-IDS2018-v2 and
was OOM-killed on this 13 GiB machine. The capacity-matched flow-only MLP needs ~2 GiB
for the same split, so it can answer the same question on the remaining datasets.

It reuses the endpoint flags cached by scripts/23 (result/tn10_flags_<dataset>.npz), so
no re-computation of endpoint membership is needed, and it follows the identical TN-1
training protocol: batch 4096, Adam 1e-3, balanced cross-entropy, checkpoint by
validation macro-F1 only, test read once.

Outputs: results/tn10b_mlp_isolation.csv, results/tn10b_mlp_isolation_summary.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parents[1]
FEAT = HERE / "work/features"
OUT = HERE / "results"
BATCH, LR, DROPOUT, MIN_STEPS, HIDDEN = 4096, 1e-3, 0.2, 1500, 273


class MLP(nn.Module):
    def __init__(self, d_in, n_classes, depth=2, hidden=HIDDEN):
        super().__init__()
        layers, prev = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = hidden
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, X, batch=200_000):
    model.eval()
    out = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        out[i:i + batch] = model(torch.from_numpy(
            np.ascontiguousarray(X[i:i + batch]))).argmax(1).numpy()
    return out


def run(ds: str, task: str, seed: int) -> dict:
    Xtr = np.load(FEAT / f"{ds}__train.npy", mmap_mode="r")
    Xva = np.load(FEAT / f"{ds}__val.npy", mmap_mode="r")
    Xte = np.load(FEAT / f"{ds}__test.npy", mmap_mode="r")
    ytr = np.load(FEAT / f"{ds}__train__y_{task}.npy")
    yva = np.load(FEAT / f"{ds}__val__y_{task}.npy")
    yte = np.load(FEAT / f"{ds}__test__y_{task}.npy")
    z = np.load(OUT / f"tn10_flags_{ds}.npz")
    src_seen, dst_seen = z["src_seen"], z["dst_seen"]
    subsets = {
        "all_test": np.ones(len(yte), dtype=bool),
        "both_endpoints_seen": src_seen & dst_seen,
        "one_endpoint_seen": src_seen ^ dst_seen,
        "neither_endpoint_seen": (~src_seen) & (~dst_seen),
    }

    n_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    n_train = len(ytr)
    spp = math.ceil(n_train / BATCH)
    max_steps = max(MIN_STEPS, 2 * spp)
    eval_every = max(500, math.ceil(spp / 4))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1], n_classes)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(
        n_train / (n_classes * np.maximum(counts, 1)), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"f1": -1.0, "state": None, "step": 0}
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
                v = float(f1_score(yva, predict(model, Xva), average="macro", zero_division=0))
                if step >= spp and v > best["f1"]:
                    best = {"f1": v, "step": step,
                            "state": {k: t.clone() for k, t in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    pred = predict(model, Xte)

    rows = []
    base = None
    for label, mask in subsets.items():
        if mask.sum() < 100:
            continue
        m = float(f1_score(yte[mask], pred[mask], average="macro", zero_division=0))
        if base is None:
            base = m
        rows.append({
            "dataset": ds, "task": task, "model": "mlp_h273_2l", "seed": seed,
            "subset": label, "n_flows": int(mask.sum()), "macro_f1": m,
            "weighted_f1": float(f1_score(yte[mask], pred[mask], average="weighted",
                                          zero_division=0)),
            "accuracy": float((yte[mask] == pred[mask]).mean()),
            "best_val_macro_f1": best["f1"], "seconds": time.perf_counter() - t0,
        })
    for r in rows:
        r["delta_vs_all_test"] = r["macro_f1"] - base
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NF-CSE-CIC-IDS2018-v2"])
    ap.add_argument("--tasks", nargs="+", default=["multiclass"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    args = ap.parse_args()
    runs = OUT / "tn10b_mlp_isolation.csv"
    for ds in args.datasets:
        for task in args.tasks:
            for seed in args.seeds:
                if runs.exists():
                    prev = pd.read_csv(runs)
                    if ((prev.dataset == ds) & (prev.task == task) &
                            (prev.seed == seed)).any():
                        print("skip", ds, task, seed, flush=True)
                        continue
                rows = run(ds, task, seed)
                pd.DataFrame(rows).to_csv(runs, mode="a",
                                          header=not runs.exists(), index=False)
                agg = pd.read_csv(runs)
                g = agg.groupby(["dataset", "task", "subset"]).agg(
                    n_seeds=("seed", "size"), n_flows=("n_flows", "first"),
                    macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std")
                ).reset_index()
                base = g[g.subset == "all_test"].set_index(["dataset", "task"]).macro_f1
                g["delta_vs_all_test"] = [r.macro_f1 - base.loc[(r.dataset, r.task)]
                                          for r in g.itertuples()]
                g.to_csv(OUT / "tn10b_mlp_isolation_summary.csv", index=False)
                print(f"[{time.strftime('%H:%M:%S')}] {ds} {task} s{seed} done: "
                      + ", ".join(f"{r['subset']}={r['macro_f1']:.4f}"
                                  for r in rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
