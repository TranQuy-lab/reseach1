"""Independent metric, provenance, checkpoint and prediction verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from nids_minibatch.data import Preprocessor, make_graph
from nids_minibatch.models import build_model
from nids_minibatch.schema import FEATURES
from nids_minibatch.training import (
    experiment_source_sha256,
    full_probabilities,
    load_frames,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    provenance = json.loads((args.runs / "provenance.json").read_text())
    requested_device = str(provenance.get("effective_device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint verification requested but CUDA is unavailable")
    device = torch.device(requested_device)
    storage_dtype = (torch.float16 if provenance.get("edge_storage_dtype") == "torch.float16"
                     else torch.float32)
    replay_tolerance = 1e-5 if str(provenance.get("effective_device", "cpu")).startswith("cuda") else 1e-7
    scope = provenance.get("scope", "pilot")
    protocol = "research/PROTOCOL_FULL_DATA_VI.md" if scope == "full" else "research/PROTOCOL_MINIBATCH_VI.md"
    if provenance["protocol_sha256"] != sha256(protocol):
        raise AssertionError("Protocol checksum changed after experiment start")
    current_sources = experiment_source_sha256()
    if provenance.get("source_sha256") != current_sources:
        raise AssertionError("Experiment source code changed after experiment start")
    table = pd.read_csv(args.runs / "runs.csv")
    expected = len(provenance["datasets"]) * len(provenance["tasks"]) * len(provenance["models"]) * len(provenance["seeds"])
    if len(table) != expected or table[["dataset", "task", "model", "seed"]].duplicated().any():
        raise AssertionError(f"Expected {expected} unique runs, found {len(table)}")
    checked, artifact_sha256 = [], {}
    largest_probability_error, largest_metric_error = 0.0, 0.0
    frame_cache = {}
    for row in table.itertuples(index=False):
        key = (row.dataset, row.task)
        if key not in frame_cache:
            frames = load_frames(args.data, row.dataset, scope)
            pre_path = args.runs / f"{row.dataset}__{row.task}__{row.model}__seed{row.seed}" / "preprocessor.json"
            pre = Preprocessor.from_dict(json.loads(pre_path.read_text()))
            frame_cache[key] = (frames, pre)
        frames, pre = frame_cache[key]
        run_dir = args.runs / f"{row.dataset}__{row.task}__{row.model}__seed{row.seed}"
        current_pre = Preprocessor.from_dict(json.loads((run_dir / "preprocessor.json").read_text()))
        if current_pre.as_dict() != pre.as_dict():
            raise AssertionError("Preprocessor differs between runs of one dataset/task")
        graph = make_graph(frames["test"], pre.transform(frames["test"]),
                           pre.labels(frames["test"]), storage_dtype)
        config = json.loads((run_dir / "config.json").read_text())
        model = build_model(row.model, len(FEATURES), len(pre.classes),
                            config["hidden"], config["dropout"]).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
        replay = full_probabilities(model, row.model, graph, provenance.get("amp", False))
        stored = pd.read_parquet(run_dir / "test_predictions.parquet")
        probability = stored[[f"p_{name}" for name in pre.classes]].to_numpy()
        offsets = (stored.row_offset.to_numpy(dtype=np.int64)
                   if "row_offset" in stored else np.arange(len(stored), dtype=np.int64))
        probability_error = float(np.max(np.abs(replay[offsets] - probability)))
        largest_probability_error = max(largest_probability_error, probability_error)
        if probability_error > replay_tolerance or not np.isfinite(probability).all():
            raise AssertionError(f"Prediction replay failed for {run_dir.name}")
        if not np.array_equal(stored.source_row_id.to_numpy(), graph.source_row_id[offsets]):
            raise AssertionError("Prediction row IDs differ")
        if not np.allclose(probability.sum(axis=1), 1, atol=1e-6):
            raise AssertionError("Probabilities do not sum to one")
        truth, pred = graph.labels.numpy(), replay.argmax(1)
        labels = list(range(len(pre.classes)))
        recomputed = {
            "test_macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
            "test_weighted_f1": f1_score(truth, pred, labels=labels, average="weighted", zero_division=0),
            "test_accuracy": accuracy_score(truth, pred),
        }
        metric_error = max(abs(float(getattr(row, name)) - value) for name, value in recomputed.items())
        largest_metric_error = max(largest_metric_error, metric_error)
        if metric_error > 1e-12:
            raise AssertionError(f"Metric recomputation failed for {run_dir.name}")
        metrics_file = json.loads((run_dir / "metrics.json").read_text())
        if metrics_file["replay_max_abs_error"] > replay_tolerance:
            raise AssertionError("Training-time checkpoint replay failed")
        artifact_sha256[run_dir.name] = {
            filename: sha256(run_dir / filename) for filename in (
                "config.json", "preprocessor.json", "model.pt", "history.json",
                "metrics.json", "test_predictions.parquet",
            )
        }
        checked.append(run_dir.name)
        print(f"verified {run_dir.name}", flush=True)
    result = {
        "passed": True,
        "runs_checked": len(checked),
        "largest_probability_replay_error": largest_probability_error,
        "largest_metric_recomputation_error": largest_metric_error,
        "protocol_sha256": provenance["protocol_sha256"],
        "source_sha256": provenance["source_sha256"],
        "probability_tolerance": replay_tolerance,
        "artifact_sha256": artifact_sha256,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
