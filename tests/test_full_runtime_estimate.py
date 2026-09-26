import json
import sys

import pandas as pd
import pytest

from nids_minibatch.schema import DATASETS
from research.estimate_full_runtime import dataset_budget, main, validation_count


def test_step_validation_schedule_includes_first_pass_and_final_budget():
    assert validation_count(1000, 1000, 100) == 1
    assert validation_count(1250, 1000, 100) == 4


def test_dataset_budget_uses_passes_with_a_minimum_for_small_datasets():
    small = dataset_budget(1_000, 4096, 2, 1500, 4)
    large = dataset_budget(10_000_000, 4096, 2, 1500, 4)
    assert small == {
        "steps_per_full_pass": 1,
        "max_train_steps": 1500,
        "eval_every_steps": 500,
    }
    assert large["max_train_steps"] == 2 * large["steps_per_full_pass"]
    assert large["eval_every_steps"] >= 500


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
        "--output", str(output), "--train-passes", "2",
        "--min-train-steps", "1000", "--evals-per-pass", "4",
    ])
    main()
    value = json.loads(output.read_text())
    assert value["safe_to_launch_72"] is True
    assert value["benchmark_rows"] == 12
    assert value["training_budget"] == {
        "train_passes": 2, "min_train_steps": 1000,
        "evals_per_pass": 4, "minimum_eval_interval_steps": 500,
    }
    assert value["execution"] == {"num_workers": 4, "batch_size": 4096}
    assert value["step_budget_estimate"]["planning_hours"] > 0
    assert value["cost_estimate"]["budget_usd"] == 6.0
    assert value["cost_estimate"]["cost_gate_evaluated"] is False
    assert value["cost_estimate"]["planning_cost_usd"] is None
    assert value["cost_estimate"]["max_hourly_price_for_budget_usd"] > 0
    assert value["optimizer_step_plan"]["total_for_72_runs"] == 12 * 6 * 1000
    assert value["optimizer_step_plan"]["fraction_reduced"] > 0
    assert all(item["post_warmup_windows"] == 4 for item in value["details"])


def test_cost_gate_rejects_estimate_above_budget(tmp_path, monkeypatch):
    root = tmp_path / "bench"
    for dataset in DATASETS:
        folder = root / dataset
        folder.mkdir(parents=True)
        rows = []
        for model in ["edge_mlp", "sage", "sage_edge"]:
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
                "train_edges": 500, "train_seconds": 1.0, "train_batches": 500,
                "validation_seconds": 1.0,
                "batch_timing": {"seconds_per_batch_windows": [1.0, 1.1, 0.9]},
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
        "--output", str(output), "--min-train-steps", "1000",
        "--hourly-price-usd", "100", "--budget-usd", "6",
    ])
    with pytest.raises(SystemExit, match="launch gate failed"):
        main()
    value = json.loads(output.read_text())
    assert value["safe_to_launch_72"] is False
    assert value["cost_estimate"]["cost_gate_evaluated"] is True
    assert value["cost_estimate"]["planning_cost_usd"] > 6
    assert any("exceeds $6.00 budget" in reason for reason in value["reasons"])
