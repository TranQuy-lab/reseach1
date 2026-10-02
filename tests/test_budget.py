"""Step-budget arithmetic, exercised without pandas/torch installed."""

import pytest

from nids_minibatch.budget import (
    build_training_budget, eval_interval, passes_effective, resolve_dataset_budget,
    step_budget, steps_per_pass, validation_count,
)

TRAIN_ROWS = {
    "NF-UNSW-NB15-v2": 1_673_104,
    "NF-BoT-IoT-v2": 26_438_303,
    "NF-ToN-IoT-v2": 11_861_039,
    "NF-CSE-CIC-IDS2018-v2": 13_227_184,
}
EXPECTED_PASSES = {
    "NF-UNSW-NB15-v2": 409,
    "NF-BoT-IoT-v2": 6_455,
    "NF-ToN-IoT-v2": 2_896,
    "NF-CSE-CIC-IDS2018-v2": 3_230,
}


def test_one_pass_matches_the_audited_batch_counts():
    for dataset, rows in TRAIN_ROWS.items():
        assert steps_per_pass(rows, 4096) == EXPECTED_PASSES[dataset]


def test_two_pass_budget_replaces_the_flat_20000_steps():
    budget = {d: step_budget(r, 4096, 2, 1_500) for d, r in TRAIN_ROWS.items()}
    assert budget == {
        "NF-UNSW-NB15-v2": 1_500,
        "NF-BoT-IoT-v2": 12_910,
        "NF-ToN-IoT-v2": 5_792,
        "NF-CSE-CIC-IDS2018-v2": 6_460,
    }
    # 4 datasets x 2 tasks x 3 models x 3 seeds = 72 runs.
    combos = 2 * 3 * 3
    assert combos * sum(budget.values()) == 479_916
    assert combos * 20_000 * 4 > 3 * 479_916


def test_minimum_steps_lift_small_datasets_and_never_drop_a_pass():
    assert step_budget(100, 4096, 2, 1_500) == 1_500
    assert step_budget(TRAIN_ROWS["NF-BoT-IoT-v2"], 4096, 2, 1_500) == 12_910


def test_validation_interval_hits_the_target_count_on_every_dataset():
    for dataset, rows in TRAIN_ROWS.items():
        per_pass = steps_per_pass(rows, 4096)
        budget = step_budget(rows, 4096, 2, 1_500)
        every = eval_interval(budget, per_pass, 8)
        assert validation_count(budget, per_pass, every) == 8


def test_passes_budget_cuts_full_validations_versus_the_flat_budget():
    old = {d: validation_count(20_000, steps_per_pass(r, 4096), 1_000)
           for d, r in TRAIN_ROWS.items()}
    new = {d: validation_count(step_budget(r, 4096, 2, 1_500), steps_per_pass(r, 4096), 1_000)
           for d, r in TRAIN_ROWS.items()}
    assert old == {
        "NF-UNSW-NB15-v2": 21, "NF-BoT-IoT-v2": 15,
        "NF-ToN-IoT-v2": 19, "NF-CSE-CIC-IDS2018-v2": 18,
    }
    assert new == {
        "NF-UNSW-NB15-v2": 3, "NF-BoT-IoT-v2": 8,
        "NF-ToN-IoT-v2": 4, "NF-CSE-CIC-IDS2018-v2": 5,
    }
    # 4 datasets x 2 tasks x 3 models x 3 seeds = 72 runs.
    assert sum(old.values()) * 18 == 1_314
    assert sum(new.values()) * 18 == 360


def test_locked_document_wins_over_passes():
    rows = TRAIN_ROWS["NF-ToN-IoT-v2"]
    document = {"training_budget": {"per_dataset": {
        "NF-ToN-IoT-v2": {"max_train_steps": 5_792, "eval_every_steps": 1_000},
    }}}
    assert resolve_dataset_budget(
        document, "NF-ToN-IoT-v2", train_rows=rows, batch_size=4096, passes=2,
        min_train_steps=1_500, validations=0, fallback_max_steps=0,
        fallback_eval_every_steps=0,
    ) == (5_792, 1_000, 2_896)


def test_epoch_mode_stays_untouched_and_document_gaps_raise():
    assert resolve_dataset_budget(
        None, "NF-ToN-IoT-v2", train_rows=TRAIN_ROWS["NF-ToN-IoT-v2"], batch_size=4096,
        passes=0, min_train_steps=1_500, validations=0, fallback_max_steps=0,
        fallback_eval_every_steps=0,
    ) == (0, 0, 2_896)
    with pytest.raises(ValueError):
        resolve_dataset_budget(
            {"training_budget": {"per_dataset": {}}}, "NF-ToN-IoT-v2",
            train_rows=1_000, batch_size=4096, passes=0, min_train_steps=1_500,
            validations=0, fallback_max_steps=0, fallback_eval_every_steps=0,
        )
    with pytest.raises(ValueError):
        resolve_dataset_budget(
            {"training_budget": {"per_dataset": {
                "NF-ToN-IoT-v2": {"max_train_steps": 2, "eval_every_steps": 5},
            }}}, "NF-ToN-IoT-v2", train_rows=10_000, batch_size=4096, passes=0,
            min_train_steps=1_500, validations=0, fallback_max_steps=0,
            fallback_eval_every_steps=0,
        )


def test_locked_budget_document_is_built_from_split_rows():
    document = build_training_budget(TRAIN_ROWS)
    assert document["mode"] == "passes"
    assert document["passes"] == 2 and document["min_train_steps"] == 1_500
    assert document["eval_every_steps"] == 1_000
    assert {d: e["max_train_steps"] for d, e in document["per_dataset"].items()} == {
        "NF-UNSW-NB15-v2": 1_500,
        "NF-BoT-IoT-v2": 12_910,
        "NF-ToN-IoT-v2": 5_792,
        "NF-CSE-CIC-IDS2018-v2": 6_460,
    }
    assert sum(e["max_train_steps"] for e in document["per_dataset"].values()) == 26_662
    assert sum(e["planned_validations"] for e in document["per_dataset"].values()) == 20
    with pytest.raises(ValueError):
        build_training_budget({})


def test_budget_never_falls_below_one_complete_pass():
    for dataset, rows in TRAIN_ROWS.items():
        per_pass = steps_per_pass(rows, 4096)
        assert step_budget(rows, 4096, 2, 1_500) >= per_pass
        assert passes_effective(step_budget(rows, 4096, 2, 1_500), per_pass) >= 2.0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        steps_per_pass(0, 4096)
    with pytest.raises(ValueError):
        step_budget(100, 4096, 0, 1_500)
    with pytest.raises(ValueError):
        validation_count(100, 1_000, 10)
