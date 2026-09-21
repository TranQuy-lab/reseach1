"""Tune the four predeclared mini-batch configurations on validation only."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

from .training import fit_model, full_probabilities, load_frames, prepare_graphs, write_json


CANDIDATES = [
    {"batch_size": 2048, "fanout": [10, 5]},
    {"batch_size": 2048, "fanout": [15, 10]},
    {"batch_size": 4096, "fanout": [10, 5]},
    {"batch_size": 4096, "fanout": [15, 10]},
]


def tune(data: Path, output: Path, epochs: int = 30, patience: int = 6,
         threads: int = 4, device: str = "auto"):
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError("Output exists; refusing overwrite")
    output.mkdir(parents=True)
    torch.set_num_threads(threads)
    frames = load_frames(data, "NF-UNSW-NB15-v2")
    pre, graphs = prepare_graphs(frames, "multiclass")
    rows = []
    for i, candidate in enumerate(CANDIDATES, 1):
        started = time.perf_counter()
        model, history, best_epoch = fit_model(
            "sage_edge", graphs, len(pre.classes), seed=11, epochs=epochs,
            patience=patience, batch_size=candidate["batch_size"],
            fanout=tuple(candidate["fanout"]),
            device=device,
        )
        seconds = time.perf_counter() - started
        probability = full_probabilities(model, "sage_edge", graphs["val"])
        score = f1_score(
            graphs["val"].labels.numpy(), probability.argmax(1),
            labels=list(range(len(pre.classes))), average="macro", zero_division=0,
        )
        row = {
            "candidate": i, **candidate, "seed": 11,
            "best_epoch": best_epoch, "epochs_ran": len(history),
            "val_macro_f1": float(score), "seconds": seconds,
            "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        rows.append(row)
        write_json(output / f"candidate_{i}.json", {"config": row, "history": history})
        print(json.dumps(row), flush=True)
    best_score = max(row["val_macro_f1"] for row in rows)
    eligible = [row for row in rows if row["val_macro_f1"] >= best_score - 0.005]
    selected = min(eligible, key=lambda row: (row["seconds"], row["peak_rss_kib_process"], row["candidate"]))
    decision = {
        "selection_metric": "validation macro-F1",
        "tie_rule": "within 0.005 of best: fastest, then lower process peak RSS",
        "best_observed_val_macro_f1": best_score,
        "selected": selected,
        "candidates": rows,
        "test_examined": False,
    }
    write_json(output / "selection.json", decision)
    pd.DataFrame(rows).to_csv(output / "candidates.csv", index=False)
    print("SELECTED " + json.dumps(selected), flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    tune(args.data, args.output, args.epochs, args.patience, args.threads, args.device)


if __name__ == "__main__":
    main()
