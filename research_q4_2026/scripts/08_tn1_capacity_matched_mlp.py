#!/usr/bin/env python3
"""B2 (TN-1) - Capacity-matched flow-only baselines on the locked split.

Closes the evidential hole stated in PROTOCOL_FULL_DATA_VI.md: "Random Forest/GBDT
và baseline cùng capacity phải chạy ở protocol xác nhận riêng trên đúng split".

Archived parameter counts (from the runs' own metrics.json):
    edge_mlp  (1 hidden layer, h=128) : 5,378  (binary) / 5,765  (multiclass)
    sage      (2 message-passing layers, h=128) : 86,530 / 87,301   -> 16.1x

Variants trained here:
    mlp_h128_1l  : replicate of edge_mlp  (harness control - must reproduce archive)
    mlp_h128_2l  : depth control, capacity still small
    mlp_h273_2l  : capacity-matched to `sage` (h = 273 gives 86,270 / 87,092 params)

Protocol reproduced exactly:
    batch 4096, Adam lr 1e-3, dropout 0.2, balanced cross-entropy with train weights
    steps_per_pass = ceil(train_rows / 4096)
    max_steps      = max(1500, 2 * steps_per_pass)
    eval_every     = max(500, ceil(steps_per_pass / 4)); first checkpoint only after a
                     full pass; best checkpoint chosen by validation macro-F1 only.

Output: results/tn1_mlp_runs.csv  (append-safe, resumable)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "work/features"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "tn1_mlp_runs.csv"

DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
VARIANTS = {
    "mlp_h128_1l": (1, 128),
    "mlp_h128_2l": (2, 128),
    "mlp_h273_2l": (2, 273),
}
BATCH = 4096
LR = 1e-3
DROPOUT = 0.2
MIN_STEPS = 1500


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, depth: int, hidden: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
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
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))
        out[i:i + batch] = model(xb).argmax(dim=1).numpy()
    return out


def evaluate(model, X, y):
    pred = predict(model, X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def run_one(dataset: str, task: str, variant: str, seed: int) -> dict:
    depth, hidden = VARIANTS[variant]
    Xtr = np.load(WORK / f"{dataset}__train.npy", mmap_mode="r")
    Xva = np.load(WORK / f"{dataset}__val.npy", mmap_mode="r")
    Xte = np.load(WORK / f"{dataset}__test.npy", mmap_mode="r")
    ytr = np.load(WORK / f"{dataset}__train__y_{task}.npy")
    yva = np.load(WORK / f"{dataset}__val__y_{task}.npy")
    yte = np.load(WORK / f"{dataset}__test__y_{task}.npy")
    n_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    n_train = len(ytr)
    spp = math.ceil(n_train / BATCH)
    max_steps = max(MIN_STEPS, 2 * spp)
    eval_every = max(500, math.ceil(spp / 4))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1], n_classes, depth, hidden)
    n_params = sum(p.numel() for p in model.parameters())

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    weights = torch.tensor(n_train / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    torch.set_num_threads(min(16, torch.get_num_threads() or 16))

    history, best = [], {"macro_f1": -1.0, "state": None, "step": 0}
    step, t0 = 0, time.perf_counter()
    while step < max_steps:
        perm = rng.permutation(n_train)
        for i in range(0, n_train, BATCH):
            if step >= max_steps:
                break
            idx = np.sort(perm[i:i + BATCH])
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx]))
            yb = torch.from_numpy(ytr[idx].astype(np.int64))
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                m = evaluate(model, Xva, yva)
                history.append({"step": step, "loss": float(loss.item()), **m,
                                "eligible": step >= spp})
                if step >= spp and m["macro_f1"] > best["macro_f1"]:
                    best = {"macro_f1": m["macro_f1"], "step": step,
                            "state": {k: v.clone() for k, v in model.state_dict().items()}}
    elapsed = time.perf_counter() - t0
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    val = evaluate(model, Xva, yva)
    test = evaluate(model, Xte, yte)
    return {
        "dataset": dataset, "task": task, "model": variant, "seed": seed,
        "depth": depth, "hidden": hidden, "parameters": n_params,
        "steps_ran": step, "steps_per_full_pass": spp, "step_budget": max_steps,
        "eval_every_steps": eval_every,
        "best_step": best["step"], "best_val_macro_f1": best["macro_f1"],
        "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"],
        "val_accuracy": val["accuracy"],
        "test_macro_f1": test["macro_f1"], "test_weighted_f1": test["weighted_f1"],
        "test_accuracy": test["accuracy"],
        "seconds": elapsed,
        "val_curve": json.dumps(history),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=["multiclass", "binary"])
    ap.add_argument("--models", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    args = ap.parse_args()

    import pandas as pd
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev.dataset, prev.task, prev.model, prev.seed))

    for dataset in args.datasets:
        for task in args.tasks:
            for variant in args.models:
                for seed in args.seeds:
                    key = (dataset, task, variant, seed)
                    if key in done:
                        print("skip", key, flush=True)
                        continue
                    t0 = time.perf_counter()
                    row = run_one(dataset, task, variant, seed)
                    df = pd.DataFrame([row])
                    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
                    print(f"[{time.strftime('%H:%M:%S')}] {dataset:24s} {task:10s} "
                          f"{variant:12s} s{seed} params={row['parameters']:>6d} "
                          f"val={row['best_val_macro_f1']:.4f} test={row['test_macro_f1']:.4f} "
                          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
