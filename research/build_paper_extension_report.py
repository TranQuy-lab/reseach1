"""Build prespecified effect tables for the paper-extension experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_KEYS = ["dataset", "task", "seed"]


def descriptive_delta_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    groups = ["family", "dataset", "task", "contrast"]
    return frame.groupby(groups, as_index=False).agg(
        n=("delta", "size"),
        mean_delta=("delta", "mean"),
        std_delta=("delta", "std"),
        median_delta=("delta", "median"),
        min_delta=("delta", "min"),
        max_delta=("delta", "max"),
        positive_seeds=("delta", lambda x: int((x > 0).sum())),
        negative_seeds=("delta", lambda x: int((x < 0).sum())),
        zero_seeds=("delta", lambda x: int((x == 0).sum())),
    )


def paired_model_deltas(full: pd.DataFrame, tabular: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([
        full[[*PAIR_KEYS, "model", "test_macro_f1"]],
        tabular[[*PAIR_KEYS, "model", "test_macro_f1"]],
    ], ignore_index=True)
    wide = combined.pivot(index=PAIR_KEYS, columns="model", values="test_macro_f1")
    rows = []
    for relational in ("sage", "sage_edge"):
        for baseline in ("random_forest", "extra_trees", "hist_gradient_boosting"):
            delta = wide[relational] - wide[baseline]
            for key, value in delta.items():
                rows.append({
                    "family": "tabular", **dict(zip(PAIR_KEYS, key)),
                    "contrast": f"{relational}_minus_{baseline}",
                    "delta": float(value),
                })
    return pd.DataFrame(rows)


def topology_deltas(full: pd.DataFrame, rw: pd.DataFrame,
                    ww: pd.DataFrame, wr: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "RR": full, "RW": rw, "WW": ww, "WR": wr,
    }
    indexed = {
        name: value.set_index([*PAIR_KEYS, "model"]).test_macro_f1
        for name, value in conditions.items()
    }
    rows = []
    for model in ("sage", "sage_edge"):
        for left, right in (("RR", "RW"), ("RR", "WW"), ("RR", "WR")):
            left_values = indexed[left].xs(model, level="model")
            right_values = indexed[right].xs(model, level="model")
            delta = left_values - right_values
            for key, value in delta.items():
                rows.append({
                    "family": "rewire", **dict(zip(PAIR_KEYS, key)),
                    "contrast": f"{model}_{left}_minus_{right}",
                    "delta": float(value),
                })
    return pd.DataFrame(rows)


def budget_deltas(full: pd.DataFrame, pass4: pd.DataFrame,
                  pass8: pd.DataFrame) -> pd.DataFrame:
    frames = {2: full, 4: pass4, 8: pass8}
    indexed = {
        passes: value.set_index([*PAIR_KEYS, "model"]).test_macro_f1
        for passes, value in frames.items()
    }
    rows = []
    for model in ("sage", "sage_edge"):
        base = indexed[2].xs(model, level="model")
        for passes in (4, 8):
            delta = indexed[passes].xs(model, level="model") - base
            for key, value in delta.items():
                rows.append({
                    "family": "budget", **dict(zip(PAIR_KEYS, key)),
                    "contrast": f"{model}_{passes}pass_minus_2pass",
                    "delta": float(value),
                })
    return pd.DataFrame(rows)


def read_runs(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    frame = pd.read_csv(path)
    required = {*PAIR_KEYS, "model", "test_macro_f1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} misses columns: {sorted(missing)}")
    if frame[[*PAIR_KEYS, "model"]].duplicated().any():
        raise ValueError(f"{label} contains duplicate run keys")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path,
                        default=Path("research/results/full_5seed/runs.csv"))
    parser.add_argument("--tabular", type=Path,
                        default=Path("research/artifacts/tabular_full_runs/runs.csv"))
    parser.add_argument("--rw", type=Path,
                        default=Path("research/artifacts/rewire_eval_RW/runs.csv"))
    parser.add_argument("--ww", type=Path,
                        default=Path("research/artifacts/rewired_full_runs/runs.csv"))
    parser.add_argument("--wr", type=Path,
                        default=Path("research/artifacts/rewire_eval_WR/runs.csv"))
    parser.add_argument("--budget4", type=Path,
                        default=Path("research/artifacts/bot_budget_4pass/runs.csv"))
    parser.add_argument("--budget8", type=Path,
                        default=Path("research/artifacts/bot_budget_8pass/runs.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/results/paper_extension"))
    args = parser.parse_args()

    full = read_runs(args.full, "five-seed full results")
    tabular = read_runs(args.tabular, "tabular results")
    rw = read_runs(args.rw, "RW results")
    ww = read_runs(args.ww, "WW results")
    wr = read_runs(args.wr, "WR results")
    pass4 = read_runs(args.budget4, "four-pass results")
    pass8 = read_runs(args.budget8, "eight-pass results")
    bot_base = full[
        (full.dataset == "NF-BoT-IoT-v2")
        & full.model.isin(["sage", "sage_edge"])
    ].copy()
    effects = pd.concat([
        paired_model_deltas(full, tabular),
        topology_deltas(full, rw, ww, wr),
        budget_deltas(bot_base, pass4, pass8),
    ], ignore_index=True)
    if not np.isfinite(effects.delta).all():
        raise ValueError("Non-finite or unmatched paired deltas")
    summary = descriptive_delta_summary(effects)
    args.output.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output / "seed_level_effects.csv", index=False)
    summary.to_csv(args.output / "effect_summary.csv", index=False)
    manifest = {
        "complete": True,
        "primary_outcome": "test_macro_f1",
        "pairing_unit": "dataset/task/seed",
        "families": sorted(effects.family.unique().tolist()),
        "input_rows": {
            "full": len(full), "tabular": len(tabular), "RW": len(rw),
            "WW": len(ww), "WR": len(wr), "budget4": len(pass4),
            "budget8": len(pass8),
        },
        "note": (
            "Descriptive paired effects. Five seeds are computational repeats, "
            "not independent network replications."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
