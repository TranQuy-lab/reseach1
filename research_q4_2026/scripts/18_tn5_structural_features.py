#!/usr/bin/env python3
"""TN-5 - Does message passing add anything beyond local structural statistics?

Rationale
---------
E-GraphSAGE aggregates edge features over neighbours. A far cheaper model can use
the *summary statistics* of that same neighbourhood: how many flows an endpoint
has, how many distinct partners, how often this exact endpoint pair repeats, and
the training-label attack rate of each endpoint. If a flow-only MLP augmented with
those statistics matches the graph model, then the value attributed to "topology"
is reproducible without any message passing.

Leakage discipline
------------------
Every structural statistic is computed from the TRAIN split only and then applied
to validation and test. Endpoints unseen in train receive count 0 and a neutral
label rate (the train prior). Test labels are never used.

Variants
--------
  mlp_struct      39 flow features + 5 structural counts (no labels)
  mlp_struct_lab  39 flow features + 5 counts + 2 training-label attack rates

Outputs: work/struct/<dataset>__<split>__struct.npy, results/tn5_struct_runs.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parents[1]
SPLITS = HERE / "data/full_splits"
WORK = HERE / "work/struct"
FEAT = HERE / "work/features"
OUT = HERE / "results"
WORK.mkdir(parents=True, exist_ok=True)

BATCH, LR, DROPOUT, MIN_STEPS = 4096, 1e-3, 0.2, 1500
STRUCT_COLS = ["src_n", "dst_n", "src_nd", "dst_nd", "pair_n", "src_ar", "dst_ar"]
VARIANTS = {"mlp_struct": 5, "mlp_struct_lab": 7}


def _files(ds: str, split: str):
    return sorted((SPLITS / ds / f"split={split}").glob("*.parquet"))


def _read_cols(files, cols, batch=250_000):
    """Stream the given columns from a set of parquet files as pandas frames."""
    import pandas as pd
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=batch, columns=cols):
            yield pd.DataFrame({c: b.column(c).to_pylist() for c in cols})


def build_stats(ds: str):
    """Vectorised pass over train -> endpoint/pair statistics and label rates."""
    import pandas as pd
    src_n: dict = {}
    dst_n: dict = {}
    src_nd: dict = {}
    dst_nd: dict = {}
    pair_n: dict = {}
    src_pos: dict = {}
    dst_pos: dict = {}
    src_tot: dict = {}
    dst_tot: dict = {}
    pos = neg = 0
    for df in _read_cols(_files(ds, "train"),
                         ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR",
                          "L4_DST_PORT", "Label"]):
        sk = df["IPV4_SRC_ADDR"].astype(str) + ":" + df["L4_SRC_PORT"].astype(str)
        dk = df["IPV4_DST_ADDR"].astype(str) + ":" + df["L4_DST_PORT"].astype(str)
        y = df["Label"].to_numpy()
        pos += int(y.sum()); neg += int(len(y) - y.sum())
        for key, mp, nd, tot, ps in ((sk, src_n, src_nd, src_tot, src_pos),
                                     (dk, dst_n, dst_nd, dst_tot, dst_pos)):
            g = key.value_counts()
            for k, v in g.items():
                mp[k] = mp.get(k, 0) + int(v)
        gnd = sk.groupby(dk).size()
        for k, v in gnd.items():
            dst_nd[k] = dst_nd.get(k, 0) + 1
        gnd = dk.groupby(sk).size()
        for k, v in gnd.items():
            src_nd[k] = src_nd.get(k, 0) + 1
        gp = pd.DataFrame({"p": list(zip(sk, dk))}).value_counts()
        for k, v in gp.items():
            pair_n[k[0]] = pair_n.get(k[0], 0) + int(v)
        for key, pos_mp, tot_mp in ((sk, src_pos, src_tot), (dk, dst_pos, dst_tot)):
            tmp = pd.DataFrame({"k": key.to_numpy(), "y": y})
            agg = tmp.groupby("k")["y"].agg(["sum", "count"])
            for k, r in agg.iterrows():
                pos_mp[k] = pos_mp.get(k, 0) + int(r["sum"])
                tot_mp[k] = tot_mp.get(k, 0) + int(r["count"])
    prior = pos / max(pos + neg, 1)
    src = {}
    for k, n in src_n.items():
        t = src_tot.get(k, 0)
        src[k] = (n, src_nd.get(k, 1), (src_pos.get(k, 0) / t) if t else prior)
    dst = {}
    for k, n in dst_n.items():
        t = dst_tot.get(k, 0)
        dst[k] = (n, dst_nd.get(k, 1), (dst_pos.get(k, 0) / t) if t else prior)
    return src, dst, pair_n, {"prior": prior}, prior


def materialise(ds: str) -> dict:
    import pickle
    import pandas as pd
    cache = WORK / f"{ds}__stats.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            src, dst, pair_n, meta = pickle.load(fh)
    else:
        src, dst, pair_n, meta, prior = build_stats(ds)
        meta["prior"] = prior
        with cache.open("wb") as fh:
            pickle.dump((src, dst, pair_n, meta), fh)
    prior = meta["prior"]
    src_df = pd.DataFrame([(k, v[0], v[1], v[2]) for k, v in src.items()],
                          columns=["sk", "src_n", "src_nd", "src_ar"])
    dst_df = pd.DataFrame([(k, v[0], v[1], v[2]) for k, v in dst.items()],
                          columns=["dk", "dst_n", "dst_nd", "dst_ar"])
    pair_df = pd.DataFrame([(k[0], k[1], v) for k, v in pair_n.items()],
                           columns=["sk", "dk", "pair_n"])
    info = {}
    for split in ("train", "val", "test"):
        dest = WORK / f"{ds}__{split}__struct.npy"
        files = _files(ds, split)
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        arr = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32,
                                        shape=(total, len(STRUCT_COLS)))
        pos = 0
        for df in _read_cols(files, ["IPV4_SRC_ADDR", "L4_SRC_PORT",
                                     "IPV4_DST_ADDR", "L4_DST_PORT"]):
            df = df.assign(sk=df["IPV4_SRC_ADDR"].astype(str) + ":" + df["L4_SRC_PORT"].astype(str),
                           dk=df["IPV4_DST_ADDR"].astype(str) + ":" + df["L4_DST_PORT"].astype(str))
            df = df.merge(src_df, on="sk", how="left").merge(dst_df, on="dk", how="left")
            df = df.merge(pair_df, on=["sk", "dk"], how="left")
            for c in ["src_n", "dst_n", "src_nd", "dst_nd", "pair_n"]:
                df[c] = df[c].fillna(0)
            for c in ["src_ar", "dst_ar"]:
                df[c] = df[c].fillna(prior)
            n = len(df)
            arr[pos:pos + n] = df[STRUCT_COLS].to_numpy(dtype=np.float32)
            pos += n
        arr.flush()
        del arr
        info[split] = total
        print(f"  {ds} {split}: {total:,d} structural rows", flush=True)
    return info


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
def predict(model, X, y, n_cols, batch=100_000):
    """X holds flow features; y is reused as the structural block to keep RAM low."""
    model.eval()
    out = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))
        sb = torch.from_numpy(np.ascontiguousarray(y[i:i + batch, :n_cols]))
        out[i:i + batch] = model(torch.cat([xb, sb], dim=1)).argmax(1).numpy()
    return out


def run_one(ds, task, variant, seed):
    n_cols = VARIANTS[variant]
    Xtr = np.load(FEAT / f"{ds}__train.npy", mmap_mode="r")
    Xva = np.load(FEAT / f"{ds}__val.npy", mmap_mode="r")
    Xte = np.load(FEAT / f"{ds}__test.npy", mmap_mode="r")
    Str = np.load(WORK / f"{ds}__train__struct.npy", mmap_mode="r")
    Sva = np.load(WORK / f"{ds}__val__struct.npy", mmap_mode="r")
    Ste = np.load(WORK / f"{ds}__test__struct.npy", mmap_mode="r")
    ytr = np.load(FEAT / f"{ds}__train__y_{task}.npy")
    yva = np.load(FEAT / f"{ds}__val__y_{task}.npy")
    yte = np.load(FEAT / f"{ds}__test__y_{task}.npy")

    # log1p on counts, then standardise on train structural statistics
    def prep(S, stats=None):
        A = np.asarray(S[:, :n_cols], dtype=np.float64)
        cnt = A[:, [0, 1, 2, 3, 4]] if n_cols == 7 else A[:, :5]
        cnt = np.sign(cnt) * np.log1p(np.abs(cnt))
        A = np.concatenate([cnt, A[:, 5:n_cols]], axis=1) if n_cols == 7 else cnt
        if stats is None:
            stats = (A.mean(0), A.std(0))
        m, s = stats
        s = np.where(s == 0, 1.0, s)
        return (A - m) / s, stats

    Ttr, st = prep(Str)
    Tva, _ = prep(Sva, st)
    Tte, _ = prep(Ste, st)
    Ttr = Ttr.astype(np.float32); Tva = Tva.astype(np.float32); Tte = Tte.astype(np.float32)

    n_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    n_train = len(ytr)
    spp = math.ceil(n_train / BATCH)
    max_steps = max(MIN_STEPS, 2 * spp)
    eval_every = max(500, math.ceil(spp / 4))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1] + n_cols, n_classes)
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
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx]))
            sb = torch.from_numpy(np.ascontiguousarray(Ttr[idx]))
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(torch.cat([xb, sb], dim=1)),
                         torch.from_numpy(ytr[idx].astype(np.int64)))
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                p = predict(model, Xva, Tva, n_cols)
                m = float(f1_score(yva, p, average="macro", zero_division=0))
                if step >= spp and m > best["macro_f1"]:
                    best = {"macro_f1": m, "step": step,
                            "state": {k: v.clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    pv = predict(model, Xva, Tva, n_cols)
    pt = predict(model, Xte, Tte, n_cols)
    return {"dataset": ds, "task": task, "model": variant, "seed": seed,
            "n_structural_columns": n_cols,
            "parameters": sum(p.numel() for p in model.parameters()),
            "steps_ran": step,
            "best_val_macro_f1": best["macro_f1"],
            "val_macro_f1": float(f1_score(yva, pv, average="macro", zero_division=0)),
            "test_macro_f1": float(f1_score(yte, pt, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(yte, pt, average="weighted", zero_division=0)),
            "test_accuracy": float(accuracy_score(yte, pt)),
            "seconds": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2"])
    ap.add_argument("--tasks", nargs="+", default=["multiclass", "binary"])
    ap.add_argument("--models", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--materialise-only", action="store_true")
    args = ap.parse_args()
    import pandas as pd
    runs = OUT / "tn5_struct_runs.csv"
    done = set()
    if runs.exists():
        prev = pd.read_csv(runs)
        done = set(zip(prev.dataset, prev.task, prev.model, prev.seed))
    for ds in args.datasets:
        print(f"### {ds}", flush=True)
        materialise(ds)
        if args.materialise_only:
            continue
        for task in args.tasks:
            for variant in args.models:
                for seed in args.seeds:
                    if (ds, task, variant, seed) in done:
                        continue
                    row = run_one(ds, task, variant, seed)
                    pd.DataFrame([row]).to_csv(runs, mode="a",
                                               header=not runs.exists(), index=False)
                    print(f"[{time.strftime('%H:%M:%S')}] {ds} {task} {variant} s{seed} "
                          f"params={row['parameters']} TEST={row['test_macro_f1']:.4f}",
                          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
