#!/usr/bin/env python3
"""TN-8 - Is message passing just hand-computable neighbourhood smoothing?

Structural fact about the architecture under test
-------------------------------------------------
`EGraphSAGE` initialises node features as `torch.ones((num_nodes, edge_dim))`
(src/nids_minibatch/models.py). Node representations therefore carry no information
of their own; after two mean-aggregation layers each node embedding is a learned
function of its neighbourhood's *edge features* only. The head then concatenates the
two endpoint embeddings.

That means the computation can be emulated without any learned message passing: for
each flow (u -> v) compute the mean of the 39 edge features over the training flows
*arriving at* u and at v, concatenate them, and give the result to a flow-only MLP
together with the flow's own features.

If this emulation matches or beats `sage`, then in this setting "relational
structure" is reproducible by a closed-form neighbour average; no message-passing
layer is required.

Leakage discipline
------------------
Neighbour means are computed from TRAIN flows only. Endpoints unseen in train get a
zero vector for that side. Test labels are never used.

Protocol
--------
Same budget rule as every other Phase B run. Hidden width 238 keeps the parameter
count at ~87.1k, i.e. matched to `sage` (87,301).

Outputs: work/nbr/<dataset>__<split>__nbr.npy, results/tn8_nbr_runs.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parents[1]
SPLITS = HERE / "data/full_splits"
FEAT = HERE / "work/features"
NBR = HERE / "work/nbr"
OUT = HERE / "results"
NBR.mkdir(parents=True, exist_ok=True)

BATCH, LR, DROPOUT, MIN_STEPS = 4096, 1e-3, 0.2, 1500
HIDDEN = 238
S = "IPV4_SRC_ADDR || ':' || CAST(L4_SRC_PORT AS VARCHAR)"
D = "IPV4_DST_ADDR || ':' || CAST(L4_DST_PORT AS VARCHAR)"


def _features() -> list[str]:
    return json.loads((Path("/home/noble-tran/nghiencuu/repo_reseach1/research/artifacts/"
                            "full_runs") / "NF-UNSW-NB15-v2__binary__edge_mlp__seed11" /
                       "preprocessor.json").read_text())["features"]


def build_neighbour_means(ds: str, feats: list[str], threads: int = 6) -> Path:
    """Per-endpoint mean of edge features over TRAIN flows arriving at that endpoint."""
    dest = NBR / f"{ds}__nbr_in.parquet"
    if dest.exists():
        return dest
    train = (SPLITS / ds / "split=train" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET threads=?", [threads])
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp = NBR / f"{ds}_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    agg = ", ".join(f"avg({c}) AS f{i}" for i, c in enumerate(feats))
    print(f"  [{ds}] aggregating {len(feats)} features per destination endpoint ...",
          flush=True)
    con.execute(f"""
        COPY (SELECT {D} AS ep, {agg} FROM read_parquet('{train}') GROUP BY ep)
        TO '{dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{dest.as_posix()}')").fetchone()[0]
    print(f"  [{ds}] {n:,d} endpoints with neighbour means", flush=True)
    con.close()
    for f in tmp.rglob("*"):
        if f.is_file():
            f.unlink()
    tmp.rmdir()
    return dest


def materialise(ds: str, feats: list[str], threads: int = 6) -> None:
    lut = build_neighbour_means(ds, feats, threads)
    con = duckdb.connect()
    con.execute("SET threads=?", [threads])
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")
    for split in ("train", "val", "test"):
        dest = NBR / f"{ds}__{split}__nbr.npy"
        files = sorted((SPLITS / ds / f"split={split}").glob("*.parquet"))
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        if dest.exists() and np.load(dest, mmap_mode="r").shape == (total, 2 * len(feats)):
            continue
        print(f"  [{ds}/{split}] materialising {total:,d} rows", flush=True)
        arr = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32,
                                        shape=(total, 2 * len(feats)))
        pos = 0
        src_cols = ", ".join(f"s.f{i} AS s{i}" for i in range(len(feats)))
        dst_cols = ", ".join(f"d.f{i} AS d{i}" for i in range(len(feats)))
        for f in files:
            q = f"""
                SELECT {src_cols}, {dst_cols}
                FROM read_parquet('{f.as_posix()}') r
                LEFT JOIN read_parquet('{lut.as_posix()}') s ON s.ep = {S}
                LEFT JOIN read_parquet('{lut.as_posix()}') d ON d.ep = {D}
            """
            tbl = con.execute(q).fetchnumpy()
            n = len(next(iter(tbl.values())))
            block = np.zeros((n, 2 * len(feats)), dtype=np.float32)
            for i in range(len(feats)):
                a = tbl.get(f"s{i}")
                b = tbl.get(f"d{i}")
                if a is not None:
                    block[:, i] = np.nan_to_num(a.astype(np.float32), nan=0.0)
                if b is not None:
                    block[:, len(feats) + i] = np.nan_to_num(b.astype(np.float32), nan=0.0)
            arr[pos:pos + n] = block
            pos += n
        arr.flush()
        del arr
        assert pos == total, (ds, split, pos, total)
    con.close()


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
def predict(model, X, N, batch=100_000):
    model.eval()
    out = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))
        nb = torch.from_numpy(np.ascontiguousarray(N[i:i + batch]))
        out[i:i + batch] = model(torch.cat([xb, nb], dim=1)).argmax(1).numpy()
    return out


def run_one(ds, task, seed):
    Xtr = np.load(FEAT / f"{ds}__train.npy", mmap_mode="r")
    Xva = np.load(FEAT / f"{ds}__val.npy", mmap_mode="r")
    Xte = np.load(FEAT / f"{ds}__test.npy", mmap_mode="r")
    Ntr = np.load(NBR / f"{ds}__train__nbr.npy", mmap_mode="r")
    Nva = np.load(NBR / f"{ds}__val__nbr.npy", mmap_mode="r")
    Nte = np.load(NBR / f"{ds}__test__nbr.npy", mmap_mode="r")
    ytr = np.load(FEAT / f"{ds}__train__y_{task}.npy")
    yva = np.load(FEAT / f"{ds}__val__y_{task}.npy")
    yte = np.load(FEAT / f"{ds}__test__y_{task}.npy")

    # standardise the neighbour block using train statistics only
    stats = NBR / f"{ds}__nbr_stats.npz"
    if stats.exists():
        z = np.load(stats)
        m, sd = z["mean"], z["scale"]
    else:
        acc = np.zeros(Ntr.shape[1], dtype=np.float64)
        sq = np.zeros(Ntr.shape[1], dtype=np.float64)
        for i in range(0, len(Ntr), 500_000):
            b = np.asarray(Ntr[i:i + 500_000], dtype=np.float64)
            acc += b.sum(0); sq += (b ** 2).sum(0)
        m = acc / len(Ntr)
        sd = np.sqrt(np.maximum(sq / len(Ntr) - m ** 2, 1e-12))
        np.savez(stats, mean=m, scale=sd)

    def z_(A):
        out = np.empty(A.shape, dtype=np.float32)
        for i in range(0, len(A), 500_000):
            out[i:i + 500_000] = ((np.asarray(A[i:i + 500_000], dtype=np.float64) - m) / sd
                                  ).astype(np.float32)
        return out

    Ntr_z, Nva_z, Nte_z = z_(Ntr), z_(Nva), z_(Nte)
    n_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    n_train = len(ytr)
    spp = math.ceil(n_train / BATCH)
    max_steps = max(MIN_STEPS, 2 * spp)
    eval_every = max(500, math.ceil(spp / 4))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1] + Ntr.shape[1], n_classes)
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
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx]))
            nb = torch.from_numpy(Ntr_z[idx])
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(torch.cat([xb, nb], dim=1)),
                         torch.from_numpy(ytr[idx].astype(np.int64)))
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                m_ = float(f1_score(yva, predict(model, Xva, Nva_z), average="macro",
                                    zero_division=0))
                if step >= spp and m_ > best["f1"]:
                    best = {"f1": m_, "step": step,
                            "state": {k: v.clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    pv = predict(model, Xva, Nva_z)
    pt = predict(model, Xte, Nte_z)
    return {"dataset": ds, "task": task, "model": "mlp_nbr_mean", "seed": seed,
            "hidden": HIDDEN, "parameters": sum(p.numel() for p in model.parameters()),
            "steps_ran": step, "best_val_macro_f1": best["f1"],
            "val_macro_f1": float(f1_score(yva, pv, average="macro", zero_division=0)),
            "test_macro_f1": float(f1_score(yte, pt, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(yte, pt, average="weighted", zero_division=0)),
            "test_accuracy": float(accuracy_score(yte, pt)),
            "seconds": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NF-UNSW-NB15-v2", "NF-ToN-IoT-v2"])
    ap.add_argument("--tasks", nargs="+", default=["multiclass", "binary"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--materialise-only", action="store_true")
    args = ap.parse_args()
    feats = _features()
    import pandas as pd
    runs = OUT / "tn8_nbr_runs.csv"
    done = set()
    if runs.exists():
        prev = pd.read_csv(runs)
        done = set(zip(prev.dataset, prev.task, prev.seed))
    for ds in args.datasets:
        print(f"### {ds}", flush=True)
        materialise(ds, feats, args.threads)
        if args.materialise_only:
            continue
        for task in args.tasks:
            for seed in args.seeds:
                if (ds, task, seed) in done:
                    continue
                row = run_one(ds, task, seed)
                pd.DataFrame([row]).to_csv(runs, mode="a",
                                           header=not runs.exists(), index=False)
                print(f"[{time.strftime('%H:%M:%S')}] {ds} {task} mlp_nbr_mean s{seed} "
                      f"params={row['parameters']} TEST={row['test_macro_f1']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
