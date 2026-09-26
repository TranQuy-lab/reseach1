import json
from pathlib import Path

import pytest

import research.server.run_full_pipeline as full_pipeline
from research.server.run_full_pipeline import (
    DEFAULT_EVALS_PER_PASS, DEFAULT_MIN_TRAIN_STEPS, DEFAULT_TRAIN_PASSES,
    require_launch_gate, train_command,
)


ROOT = Path(__file__).resolve().parents[1]


def test_server_notebooks_are_ordered_and_parseable():
    expected = [
        "00_CHECK_SERVER.ipynb", "01_DOWNLOAD_PREPROCESS.ipynb",
        "02_SPLIT_TUNE.ipynb", "03_TRAIN_72_RUNS.ipynb", "04_VERIFY_REPORT.ipynb",
        "10_FULL_PREPARE_BENCHMARK.ipynb", "11_FULL_TRAIN_72.ipynb",
        "12_FULL_VERIFY_REPORT.ipynb",
    ]
    folder = ROOT / "notebooks/server"
    assert [path.name for path in sorted(folder.glob("*.ipynb"))] == expected
    for filename in expected:
        value = json.loads((folder / filename).read_text())
        assert value["nbformat"] == 4
        assert value["metadata"]["kernelspec"]["name"] == "nids-server"
        assert all(cell["cell_type"] in {"markdown", "code"} for cell in value["cells"])
        for cell in value["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), filename, "exec")


def test_server_manifest_and_required_entrypoints():
    manifest = json.loads((ROOT / "research/server/data_manifest.json").read_text())
    assert sum(manifest["expected_rows"].values()) == 75_987_976
    assert len(manifest["csv"]["sha256"]) == 64
    for relative in [
        "research/server/bootstrap_server.sh", "research/server/run_all.sh",
        "research/server/run_pipeline.py", "research/server/check_server.py",
        "research/PROTOCOL_MINIBATCH_VI.md", "research/validate_minibatch_results.py",
        "research/PROTOCOL_FULL_DATA_VI.md", "research/estimate_full_runtime.py",
        "research/build_full_report.py", "research/server/run_full_pipeline.py",
        "research/server/SERVER_PREFLIGHT_CHECKLIST_VI.md",
    ]:
        assert (ROOT / relative).is_file()


def test_full_train_command_uses_locked_dataset_pass_budget():
    command = train_command(
        "out", ["NF-UNSW-NB15-v2"], ["sage"], [11], ["binary"],
        epochs=1, patience=10, threads=4,
        train_passes=DEFAULT_TRAIN_PASSES,
        min_train_steps=DEFAULT_MIN_TRAIN_STEPS,
        evals_per_pass=DEFAULT_EVALS_PER_PASS,
    )
    assert command[command.index("--train-passes") + 1] == "2"
    assert command[command.index("--min-train-steps") + 1] == "1500"
    assert command[command.index("--evals-per-pass") + 1] == "4"
    assert command[command.index("--num-workers") + 1] == "4"
    assert "--scope" in command and command[command.index("--scope") + 1] == "full"
    assert "--device" in command and command[command.index("--device") + 1] == "cuda"


def test_full_split_uses_server_memory_limit(monkeypatch):
    captured = []
    monkeypatch.setattr(full_pipeline, "execute", lambda command, log: captured.append(command))
    monkeypatch.setattr(full_pipeline.Path, "is_file", lambda self: False)
    full_pipeline.run_stage("split", 12, 4, 6.0, 0.25)
    command = captured[0]
    assert command[command.index("--memory-limit") + 1] == "16GB"


def test_launch_gate_locks_budget_and_worker_count(tmp_path, monkeypatch):
    results = tmp_path / "research/results"
    results.mkdir(parents=True)
    (results / "full_benchmark_estimate.json").write_text(json.dumps({
        "safe_to_launch_72": True,
        "training_budget": {
            "train_passes": 2,
            "min_train_steps": 1500,
            "evals_per_pass": 4,
            "minimum_eval_interval_steps": 500,
        },
        "execution": {"num_workers": 4, "batch_size": 4096},
    }))
    monkeypatch.setattr(full_pipeline, "ROOT", tmp_path)
    assert require_launch_gate(4)["safe_to_launch_72"] is True
    with pytest.raises(RuntimeError, match="execution settings"):
        require_launch_gate(8)
