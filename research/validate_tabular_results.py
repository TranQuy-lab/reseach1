"""Reload every tabular model and independently recompute full-test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from nids_minibatch.data import Preprocessor
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.run_tabular_baselines import load_split, sha256_file


KEYS = ("dataset", "task", "model", "seed")


def artifact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict[str, float]:
    labels = list(range(class_count))
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0,
        )),
        "test_weighted_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0,
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/full_splits"))
    parser.add_argument("--runs", type=Path, default=Path("research/artifacts/tabular_full_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/results/tabular_full_verification.json"))
    args = parser.parse_args()

    provenance = json.loads((args.runs / "provenance.json").read_text())
    protocol = Path(provenance["protocol_path"])
    if sha256_file(protocol) != provenance["protocol_sha256"]:
        raise AssertionError("Tabular extension protocol changed after training")
    generator = Path("research/run_tabular_baselines.py")
    if sha256_file(generator) != provenance["source_sha256"]:
        raise AssertionError("Tabular training source changed after training")
    table = pd.read_csv(args.runs / "runs.csv")
    expected = (
        len(provenance["datasets"]) * len(provenance["tasks"])
        * len(provenance["models"]) * len(provenance["seeds"])
    )
    if len(table) != expected or table[list(KEYS)].duplicated().any():
        raise AssertionError(f"Expected {expected} unique tabular runs, found {len(table)}")

    largest_metric_error = 0.0
    largest_probability_error = 0.0
    artifact_sha256: dict[str, dict[str, str]] = {}
    checked = 0
    for (dataset, task), group in table.groupby(["dataset", "task"], sort=False):
        frame = load_split(args.data, dataset, "test")
        first = group.iloc[0]
        first_dir = args.runs / (
            f"{dataset}__{task}__{first.model}__seed{int(first.seed)}"
        )
        pre = Preprocessor.from_dict(json.loads(
            (first_dir / "preprocessor.json").read_text()
        ))
        x_test = pre.transform(frame)
        y_test = pre.labels(frame)
        for row in group.itertuples(index=False):
            run_id = f"{row.dataset}__{row.task}__{row.model}__seed{int(row.seed)}"
            run_dir = args.runs / run_id
            current_pre = Preprocessor.from_dict(json.loads(
                (run_dir / "preprocessor.json").read_text()
            ))
            if not current_pre.equivalent(pre):
                raise AssertionError(f"Preprocessor mismatch: {run_id}")
            model = joblib.load(run_dir / "model.joblib")
            prediction = model.predict(x_test)
            recomputed = score(y_test, prediction, len(pre.classes))
            metric_error = max(
                abs(float(getattr(row, key)) - value)
                for key, value in recomputed.items()
            )
            largest_metric_error = max(largest_metric_error, metric_error)
            if metric_error > 1e-12:
                raise AssertionError(f"Metric recomputation failed: {run_id}")
            stored = pd.read_parquet(run_dir / "test_predictions.parquet")
            offsets = stored.row_offset.to_numpy(dtype=np.int64)
            if not np.array_equal(
                stored.source_row_id.to_numpy(),
                frame.source_row_id.to_numpy()[offsets],
            ):
                raise AssertionError(f"Prediction row IDs differ: {run_id}")
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(x_test[offsets])
                replay = np.zeros((len(offsets), len(pre.classes)), dtype=np.float64)
                replay[:, np.asarray(model.classes_, dtype=np.int64)] = raw
                persisted = stored[[f"p_{name}" for name in pre.classes]].to_numpy()
                probability_error = float(np.max(np.abs(replay - persisted)))
                largest_probability_error = max(
                    largest_probability_error, probability_error,
                )
                if probability_error > 1e-12:
                    raise AssertionError(f"Probability replay failed: {run_id}")
            files = (
                "config.json", "preprocessor.json", "metrics.json",
                "model.joblib", "test_predictions.parquet",
            )
            artifact_sha256[run_id] = {
                name: artifact_hash(run_dir / name) for name in files
            }
            checked += 1
            print(f"verified {run_id}", flush=True)
        del frame, x_test, y_test

    result = {
        "passed": True,
        "runs_checked": checked,
        "largest_metric_recomputation_error": largest_metric_error,
        "largest_probability_replay_error": largest_probability_error,
        "protocol_sha256": provenance["protocol_sha256"],
        "source_sha256": provenance["source_sha256"],
        "artifact_sha256": artifact_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
