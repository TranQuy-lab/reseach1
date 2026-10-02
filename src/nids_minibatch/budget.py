"""Deterministic optimizer-step budget shared by the launch gate and training.

The full-data protocol budgets every run by *complete passes over the training
split* instead of one global step count. A flat 20.000-step budget gave
NF-UNSW-NB15-v2 about 49 passes but NF-BoT-IoT-v2 only about 3, so the two were
never comparable despite sharing a number. This module is pure standard library
so the gate, the training CLI, and the tests cannot drift apart.
"""

from __future__ import annotations

import math

PROTOCOL_BUDGET_MODE = "passes"

# Locked full-data budget constants: complete passes over each train split.
DEFAULT_PASSES = 2
DEFAULT_MIN_TRAIN_STEPS = 1_500
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_EVAL_EVERY_STEPS = 1_000
# Tuning selected batch 2048 with fanout [10, 5] on validation macro-F1; the
# fanout is what sets sampling cost per pass, so keep the tuned value.
DEFAULT_FANOUT = (10, 5)
DEFAULT_NUM_WORKERS = 4


def steps_per_pass(train_rows: int, batch_size: int) -> int:
    """Optimizer steps needed to present every training edge once."""
    if train_rows < 1:
        raise ValueError("train_rows must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return math.ceil(train_rows / batch_size)


def step_budget(train_rows: int, batch_size: int, passes: int, min_steps: int) -> int:
    """Total step budget: `passes` complete passes, never below `min_steps`."""
    if passes < 1:
        raise ValueError("passes must be at least 1")
    if min_steps < 1:
        raise ValueError("min_steps must be at least 1")
    return max(passes * steps_per_pass(train_rows, batch_size), min_steps)


def eval_interval(budget: int, first_pass_steps: int, validations: int) -> int:
    """Step interval that yields about `validations` full validations per run.

    The first validation lands at the end of the first complete pass, so the
    remaining `validations - 1` spread across the rest of the budget. Full-graph
    validation is the expensive part of a step-mode run, which is why targetting
    a validation count beats a fixed 1.000-step interval on small datasets.
    """
    if budget < 1 or first_pass_steps < 1:
        raise ValueError("budget and first_pass_steps must be positive")
    if validations < 2:
        return budget
    span = budget - first_pass_steps
    if span <= 0:
        return budget
    return max(1, math.ceil(span / (validations - 1)))


def validation_count(budget: int, first_pass_steps: int, every_steps: int) -> int:
    """Validations planned: the first-pass one plus interval and final ones."""
    if min(budget, first_pass_steps, every_steps) < 1:
        raise ValueError("step counts must be positive")
    if budget < first_pass_steps:
        raise ValueError("budget must include at least one complete data pass")
    return 1 + math.ceil((budget - first_pass_steps) / every_steps)


def passes_effective(budget: int, first_pass_steps: int) -> float:
    """How many complete passes a budget actually funds."""
    return round(budget / first_pass_steps, 3)


def resolve_dataset_budget(
    document: dict | None, dataset: str, *, train_rows: int, batch_size: int,
    passes: int, min_train_steps: int, validations: int,
    fallback_max_steps: int, fallback_eval_every_steps: int,
) -> tuple[int, int, int]:
    """Resolve `(max_steps, eval_every_steps, steps_per_pass)` for one dataset.

    A locked gate document wins outright, so a paid run cannot silently drift
    from the budget that authorised it. Otherwise the budget is derived from
    `passes`/`min_train_steps` against this dataset's own training rows, which
    is what makes the comparison unit a number of complete passes. A zero
    `fallback_max_steps` keeps legacy epoch mode untouched.
    """
    per_pass = steps_per_pass(train_rows, batch_size)
    if document is not None:
        entry = document.get("training_budget", {}).get("per_dataset", {}).get(dataset)
        if not isinstance(entry, dict):
            raise ValueError(f"budget document has no entry for {dataset}")
        max_steps = int(entry["max_train_steps"])
        every = int(entry["eval_every_steps"])
        if max_steps < per_pass:
            raise ValueError(
                f"budget document gives {dataset} {max_steps} steps, "
                f"below one complete pass ({per_pass})"
            )
        return max_steps, every, per_pass
    if passes:
        max_steps = step_budget(train_rows, batch_size, passes, min_train_steps)
        every = (eval_interval(max_steps, per_pass, validations) if validations
                 else fallback_eval_every_steps)
        if not every:
            raise ValueError("a passes budget requires eval_every_steps or validations")
        return max_steps, every, per_pass
    if fallback_max_steps and fallback_max_steps < per_pass:
        raise ValueError(
            f"max_train_steps={fallback_max_steps} is below one complete pass ({per_pass})"
        )
    return fallback_max_steps, fallback_eval_every_steps, per_pass


def build_training_budget(
    split_rows: dict[str, int], *, passes: int = DEFAULT_PASSES,
    min_train_steps: int = DEFAULT_MIN_TRAIN_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_every_steps: int = DEFAULT_EVAL_EVERY_STEPS,
    target_validations: int = 0,
) -> dict:
    """Locked per-dataset budget document from `{dataset: train_rows}`.

    This is the single source of truth behind both the launch gate and the
    training CLI: the gate writes it, `--budget-json` consumes it, and a run
    refuses to start when a dataset is missing from it.
    """
    if passes < 1:
        raise ValueError("passes must be at least 1")
    if min_train_steps < 1:
        raise ValueError("min_train_steps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if eval_every_steps < 1:
        raise ValueError("eval_every_steps must be positive")
    if target_validations < 0:
        raise ValueError("target_validations must be nonnegative")
    per_dataset: dict[str, dict] = {}
    for dataset, train_rows in split_rows.items():
        per_pass = steps_per_pass(train_rows, batch_size)
        max_steps = step_budget(train_rows, batch_size, passes, min_train_steps)
        every = (eval_interval(max_steps, per_pass, target_validations)
                 if target_validations else eval_every_steps)
        per_dataset[dataset] = {
            "train_rows": train_rows,
            "steps_per_pass": per_pass,
            "max_train_steps": max_steps,
            "eval_every_steps": every,
            "planned_validations": validation_count(max_steps, per_pass, every),
            "passes_effective": passes_effective(max_steps, per_pass),
        }
    if not per_dataset:
        raise ValueError("no datasets supplied for the training budget")
    return {
        "mode": PROTOCOL_BUDGET_MODE,
        "passes": passes,
        "min_train_steps": min_train_steps,
        "batch_size": batch_size,
        "eval_every_steps": eval_every_steps,
        "target_validations": target_validations,
        "per_dataset": per_dataset,
    }

