import json
import sys

import pandas as pd

from nids_minibatch.schema import DATASETS
from research.estimate_full_runtime import evaluations, main


def test_evaluation_schedule_counts_unique_final_epoch():
    assert evaluations(1) == 1
    assert evaluations(10) == 5  # 1, 3, 6, 9, 10


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
                "validation_seconds": 1.0,
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
        "--output", str(output),
    ])
    main()
    value = json.loads(output.read_text())
    assert value["safe_to_launch_72"] is True
    assert value["benchmark_rows"] == 12
    assert value["scenario_estimates"]["60"]["planning_hours"] > 0
