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


def test_paper_extension_protocol_locks_claim_specific_controls():
    source = Path("research/PROTOCOL_PAPER_EXTENSION_VI.md").read_text()
    for required in (
        "Random Forest", "ExtraTrees", "HistGradientBoosting",
        "RR", "RW", "WW", "WR", "Endpoint-disjoint", "Budget sensitivity",
        "11, 22, 33, 44, 55", "test macro-F1",
    ):
        assert required in source
    assert "tổng số run mới là cỡ mẫu độc lập" in source
    assert "không dùng test để chọn" in source


def test_next_runner_uses_five_seeds_and_locked_extension_protocol():
    source = Path("research/server/run_next_experiments.sh").read_text()
    assert "--seeds 11 22 33 44 55" in source
    assert "PROTOCOL_PAPER_EXTENSION_VI.md" in source
    for stage in (
        "tabular_verify", "endpoint_verify", "rewire_eval_rw",
        "rewire_eval_wr", "budget4", "budget8",
    ):
        assert stage in source


def test_rewire_has_hard_identity_and_marginal_gates():
    source = Path("research/server/rewire_full_splits.py").read_text()
    assert "protected_row_mismatches" in source
    assert "destination_marginal_difference_rows" in source
    assert "changed_endpoint_fraction" in source
    assert "EXCEPT ALL" in source


def test_endpoint_holdout_has_task_specific_feasibility_gate():
    source = Path("research/server/prepare_endpoint_holdout.py").read_text()
    assert "minimum_class_rows_per_split" in source
    assert "eligible_dataset_tasks" in source
    assert "split_attack_support" in source
    assert "split_binary_support" in source


def test_extension_analysis_preserves_paired_seed_unit():
    source = Path("research/build_paper_extension_report.py").read_text()
    assert 'PAIR_KEYS = ["dataset", "task", "seed"]' in source
    assert "positive_seeds" in source
    assert "computational repeats" in source
