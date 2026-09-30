#!/usr/bin/env python3
"""TN-7 - Cross-dataset transfer of a flow-based detector (binary task).

The repository has never answered the security question that matters most for
deployment: does a flow-only detector trained on one network still work on another?
This is answerable without a GPU because the flow-only MLP does not use endpoint
identity at all; only the 39 NetFlow features must be commensurate.

Protocol
--------
* source A: train the capacity-matched flow-only MLP on A-train, select the
  checkpoint on A-val (never on any target);
* target B: rescale B's flows with A's TRUSTED train statistics. B is stored scaled
  by its own scaler, so raw_B = scaled_B * scale_B + mean_B and then
  scaled_by_A = (raw_B - mean_A) / scale_A. This is exact up to float32 rounding and
  avoids materialising the raw matrices again;
* binary task only (Benign / Attack share the same two labels in all four datasets);
* reference points: A's own test score (in-domain) and the majority-class rate on
  every target, so a transfer score can be read against a trivial baseline.

Outputs: results/tn7_transfer_runs.csv, results/tn7_transfer_matrix.csv,
         results/tn7_transfer_summary.json
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
FEAT = HERE / "work/features"
OUT = HERE / "results"
DATASETS = ["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"]
BATCH, LR, DROPOUT, MIN_STEPS = 4096, 1e-3, 0.2, 1500


class MLP(nn.Module):
    def __init__(self, d_in, n_classes, depth=2, hidden=273):
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


def train_source(src: str, seed: int) -> tuple[MLP, float]:
    Xtr = np.load(FEAT / f"{src}__train.npy", mmap_mode="r")
    Xva = np.load(FEAT / f"{src}__val.npy", mmap_mode="r")
    ytr = np.load(FEAT / f"{src}__train__y_binary.npy")
    yva = np.load(FEAT / f"{src}__val__y_binary.npy")
    n_classes = 2
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
    step = 0
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
                p = predict(model, Xva)
                m = float(f1_score(yva, p, average="macro", zero_division=0))
                if step >= spp and m > best["f1"]:
                    best = {"f1": m, "step": step,
                            "state": {k: v.clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["f1"]


def rescaled_target(src: str, tgt: str, split: str) -> np.ndarray:
    """Target features expressed in the source's standardisation."""
    src_sc = json.loads((Path("/home/noble-tran/nghiencuu/repo_reseach1/research/artifacts/"
                              "full_runs") / f"{src}__binary__edge_mlp__seed11" /
                         "preprocessor.json").read_text())
    tgt_sc = json.loads((Path("/home/noble-tran/nghiencuu/repo_reseach1/research/artifacts/"
                              "full_runs") / f"{tgt}__binary__edge_mlp__seed11" /
                         "preprocessor.json").read_text())
    X = np.load(FEAT / f"{tgt}__{split}.npy", mmap_mode="r")
    m_a, s_a = np.asarray(src_sc["mean"], np.float32), np.asarray(src_sc["scale"], np.float32)
    m_b, s_b = np.asarray(tgt_sc["mean"], np.float32), np.asarray(tgt_sc["scale"], np.float32)
    out = np.empty((len(X), X.shape[1]), dtype=np.float32)
    step = 500_000
    for i in range(0, len(X), step):
        block = np.asarray(X[i:i + step], dtype=np.float32)
        raw = block * s_b + m_b
        out[i:i + step] = (raw - m_a) / s_a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2"])
    ap.add_argument("--targets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    args = ap.parse_args()

    import pandas as pd
    runs = OUT / "tn7_transfer_runs.csv"
    done = set()
    if runs.exists():
        prev = pd.read_csv(runs)
        done = set(zip(prev.source, prev.target, prev.seed))

    for src in args.sources:
        for seed in args.seeds:
            if all((src, t, seed) in done for t in args.targets):
                continue
            t0 = time.perf_counter()
            model, best_val = train_source(src, seed)
            print(f"[{time.strftime('%H:%M:%S')}] trained {src} s{seed} "
                  f"best_val={best_val:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
            for tgt in args.targets:
                if (src, tgt, seed) in done:
                    continue
                yte = np.load(FEAT / f"{tgt}__test__y_binary.npy")
                X = rescaled_target(src, tgt, "test") if tgt != src else \
                    np.asarray(np.load(FEAT / f"{tgt}__test.npy", mmap_mode="r"))
                p = predict(model, X)
                majority = float(np.bincount(yte, minlength=2).max() / len(yte))
                row = {"source": src, "target": tgt, "seed": seed,
                       "in_domain": bool(src == tgt),
                       "test_macro_f1": float(f1_score(yte, p, average="macro",
                                                       zero_division=0)),
                       "test_weighted_f1": float(f1_score(yte, p, average="weighted",
                                                          zero_division=0)),
                       "test_accuracy": float(accuracy_score(yte, p)),
                       "majority_class_macro_f1": majority}
                pd.DataFrame([row]).to_csv(runs, mode="a",
                                           header=not runs.exists(), index=False)
                print(f"    {src} -> {tgt} s{seed} macro_f1={row['test_macro_f1']:.4f} "
                      f"(majority {majority:.4f})", flush=True)
            del model
    df = pd.read_csv(runs)
    mat = df.pivot_table(index="source", columns="target", values="test_macro_f1",
                         aggfunc="mean")
    mat.to_csv(OUT / "tn7_transfer_matrix.csv")
    summary = {
        "n_runs": int(len(df)),
        "matrix": mat.round(4).to_dict(),
        "in_domain_mean": float(df[df.in_domain].test_macro_f1.mean()),
        "cross_domain_mean": float(df[~df.in_domain].test_macro_f1.mean()),
        "cross_domain_by_pair": df[~df.in_domain].groupby(["source", "target"])
            .test_macro_f1.mean().round(4).to_dict(),
        "majority_baseline_by_target": df.groupby("target").majority_class_macro_f1
            .first().round(4).to_dict(),
    }
    summary["cross_domain_by_pair"] = {f"{k[0]}->{k[1]}": v
                                       for k, v in summary["cross_domain_by_pair"].items()}
    (OUT / "tn7_transfer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
