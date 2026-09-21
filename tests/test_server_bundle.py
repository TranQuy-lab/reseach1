import json
from pathlib import Path


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
