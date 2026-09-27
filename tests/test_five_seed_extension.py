from pathlib import Path


def test_merge_script_is_locked_to_two_new_seeds():
    source = Path("research/merge_full_seed_runs.py").read_text()
    assert "EXPECTED_SEEDS = (11, 22, 33, 44, 55)" in source
    assert "full_runs_5seed" in source
    assert "Refusing overwrite" in source
    assert "source_sha256" in source and "protocol_sha256" in source


def test_extension_runs_new_seeds_only():
    source = Path("research/server/run_five_seed_extension.sh").read_text()
    assert "--seeds 44 55" in source
    assert "full_runs_seeds44_55" in source
    assert "full_runs_5seed" in source
    assert "full_verification_5seed.json" in source
    assert "--resume" in source
    assert "full_runs\"" in source


def test_extension_does_not_overwrite_primary_directory():
    source = Path("research/server/run_five_seed_extension.sh").read_text()
    assert "--output \"${EXTRA_RUNS}\"" in source
    assert "--output research/artifacts/full_runs" not in source
