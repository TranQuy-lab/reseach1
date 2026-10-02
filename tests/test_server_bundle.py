import json
from pathlib import Path

from nids_minibatch.budget import (
    DEFAULT_BATCH_SIZE as BUDGET_BATCH_SIZE,
    DEFAULT_EVAL_EVERY_STEPS as BUDGET_EVAL_EVERY_STEPS,
    DEFAULT_FANOUT as BUDGET_FANOUT,
    DEFAULT_MIN_TRAIN_STEPS as BUDGET_MIN_TRAIN_STEPS,
    DEFAULT_NUM_WORKERS as BUDGET_NUM_WORKERS,
    DEFAULT_PASSES as BUDGET_PASSES,
)
from research.server.run_full_pipeline import (
    DEFAULT_BATCH_SIZE, DEFAULT_EVAL_EVERY_STEPS, DEFAULT_FANOUT,
    DEFAULT_MIN_TRAIN_STEPS, DEFAULT_NUM_WORKERS, DEFAULT_PASSES, GATE_PATH,
    train_command,
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
    ]:
        assert (ROOT / relative).is_file()


def test_full_train_command_uses_the_locked_passes_budget():
    command = train_command(
        "out", ["NF-UNSW-NB15-v2"], ["sage"], [11], ["binary"],
        epochs=1, patience=10, threads=4, budget_json=GATE_PATH,
    )
    assert command[command.index("--budget-json") + 1] == str(GATE_PATH)
    assert command[command.index("--num-workers") + 1] == str(DEFAULT_NUM_WORKERS)
    fanout = command.index("--fanout")
    assert command[fanout + 1:fanout + 3] == [str(value) for value in DEFAULT_FANOUT]
    assert command[command.index("--batch-size") + 1] == str(DEFAULT_BATCH_SIZE)
    # The flat step budget is gone; the gate document carries the per-dataset one.
    assert "--max-train-steps" not in command


def test_pipeline_constants_match_the_shared_budget_module():
    assert (DEFAULT_PASSES, DEFAULT_MIN_TRAIN_STEPS, DEFAULT_BATCH_SIZE,
            DEFAULT_EVAL_EVERY_STEPS, tuple(DEFAULT_FANOUT), DEFAULT_NUM_WORKERS) == (
        BUDGET_PASSES, BUDGET_MIN_TRAIN_STEPS, BUDGET_BATCH_SIZE,
        BUDGET_EVAL_EVERY_STEPS, tuple(BUDGET_FANOUT), BUDGET_NUM_WORKERS,
    )
