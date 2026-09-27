from pathlib import Path


def test_tabular_baseline_cli_and_models():
    source = Path("research/run_tabular_baselines.py").read_text()
    assert "random_forest" in source
    assert "extra_trees" in source
    assert "hist_gradient_boosting" in source
    assert "full_splits" in source
    assert "preprocessor.json" in source


def test_endpoint_holdout_is_separate_and_strict():
    source = Path("research/server/prepare_endpoint_holdout.py").read_text()
    assert "endpoint_holdout_splits" in source or "ENDPOINT_HOLDOUT" in source
    assert "cross_split_endpoint_ips" in source
    assert "IPV4_SRC_ADDR" in source and "IPV4_DST_ADDR" in source
    assert "if output.exists()" in source
    assert "replace('ip'" not in source
    assert "full_splits" not in source.split("default=", 1)[-1].split("\n", 1)[0]


def test_rewire_is_separate_and_non_overwriting():
    source = Path("research/server/rewire_full_splits.py").read_text()
    assert "rewired_full_splits" in source
    assert "if output.exists()" in source
    assert "__new_dst_ip" in source
    assert "__new_dst_port" in source


def test_next_runner_exposes_three_stages():
    source = Path("research/server/run_next_experiments.sh").read_text()
    for stage in ("tabular", "endpoint_prepare", "rewire_prepare"):
        assert stage in source
