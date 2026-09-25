import json
import sys

import pandas as pd

from nids_minibatch.schema import DATASETS
from research.estimate_full_runtime import main, validation_count


def test_step_validation_schedule_includes_first_pass_and_final_budget():
    assert validation_count(1000, 1000, 100) == 1
    assert validation_count(1250, 1000, 100) == 4


def test_bounded_benchmark_produces_safe_conservative_eta(tmp_path, monkeypatch):
    root = tmp_path / "bench"
    models = ["edge_mlp", "sage", "sage_edge"]
    for dataset in DATASETS:
        folder = root / dataset
        folder.mkdir(parents=True)
        rows = []
        for model in models:
            rows.append({
                "dataset": dataset, "task": "multiclass", "model": model,
                "seed": 11, "seconds_final_evaluation": 2.0,
                "seconds_dataset_load": 1.0, "seconds_graph_prepare": 2.0,
                "peak_rss_kib_process": 2 * 1024**2,
                "peak_cuda_bytes": 2 * 1024**3,
            })
            run_dir = folder / f"{dataset}__multiclass__{model}__seed11"
            run_dir.mkdir()
            (run_dir / "history.json").write_text(json.dumps([{
                "train_edges": 500, "train_seconds": 1.0,
                "train_batches": 500,
                "validation_seconds": 1.0,
                "batch_timing": {
                    "seconds_per_batch_windows": [0.001, 0.0011, 0.0009, 0.0012]
                },
            }]))
        pd.DataFrame(rows).to_csv(folder / "runs.csv", index=False)
    prepare = tmp_path / "prepare.json"
    prepare.write_text(json.dumps({"datasets": {
        dataset: {"split_rows": {"train": {"rows": 1000}}} for dataset in DATASETS
    }}))
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"free_disk_gib": 20}))
    output = tmp_path / "estimate.json"
    monkeypatch.setattr(sys, "argv", [
        "estimate_full_runtime.py", "--benchmarks", str(root),
        "--prepare", str(prepare), "--environment", str(environment),
        "--output", str(output), "--max-train-steps", "1000",
        "--eval-every-steps", "100",
    ])
    main()
    value = json.loads(output.read_text())
    assert value["safe_to_launch_72"] is True
    assert value["benchmark_rows"] == 12
    assert value["training_budget"] == {
        "max_train_steps": 1000, "eval_every_steps": 100,
    }
    assert value["step_budget_estimate"]["planning_hours"] > 0
    assert all(item["post_warmup_windows"] == 4 for item in value["details"])
