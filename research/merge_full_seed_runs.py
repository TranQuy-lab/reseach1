"""Merge verified full-data seed directories without retraining or overwriting.

The primary 3-seed directory remains untouched. The output is a separate
120-run directory containing seeds 11,22,33,44,55 and a merged provenance/runs
CSV suitable for the existing independent validator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

EXPECTED_SEEDS = (11, 22, 33, 44, 55)
KEY_FIELDS = ("dataset", "task", "model", "seed")
REQUIRED_ARTIFACTS = ("config.json", "preprocessor.json", "model.pt", "history.json", "metrics.json", "test_predictions.parquet")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (row["dataset"], row["task"], row["model"], int(row["seed"]))


def copy_file(src: str, dst: str) -> str:
    """Hardlink on the same server; copy when the filesystems differ."""
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def copy_tree(src: Path, dst: Path) -> int:
    if dst.exists():
        raise ValueError(f"Refusing overwrite existing run: {dst}")
    dst.mkdir(parents=True)
    linked = 0
    for path in sorted(src.iterdir()):
        target = dst / path.name
        if path.is_dir():
            linked += copy_tree(path, target)
        else:
            mode = copy_file(str(path), str(target))
            linked += int(mode == "hardlink")
    return linked


def validate_source(root: Path, expected_seeds: set[int]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    table_path = root / "runs.csv"
    prov_path = root / "provenance.json"
    if not table_path.is_file() or not prov_path.is_file():
        raise FileNotFoundError(f"Missing runs.csv/provenance.json under {root}")
    table = rows(table_path)
    prov = read_json(prov_path)
    actual = {key(row) for row in table}
    if len(actual) != len(table):
        raise ValueError(f"Duplicate run keys in {root}")
    seeds = {int(row["seed"]) for row in table}
    if seeds != expected_seeds:
        raise ValueError(f"Unexpected seeds in {root}: {sorted(seeds)} != {sorted(expected_seeds)}")
    for row in table:
        run_dir = root / f"{row['dataset']}__{row['task']}__{row['model']}__seed{row['seed']}"
        missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(f"{run_dir}: missing {missing}")
    return table, prov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("research/artifacts/full_runs"))
    parser.add_argument("--additional", type=Path, default=Path("research/artifacts/full_runs_seeds44_55"))
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/full_runs_5seed"))
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(EXPECTED_SEEDS))
    args = parser.parse_args()
    expected = set(args.expected_seeds)
    if args.output.exists():
        raise ValueError(f"Output exists; refusing overwrite: {args.output}")
    base_seeds = {11, 22, 33}
    extra_seeds = expected - base_seeds
    if expected != set(EXPECTED_SEEDS) or extra_seeds != {44, 55}:
        raise ValueError("This merge script is locked to base seeds 11/22/33 plus 44/55")
    base_rows, base_prov = validate_source(args.base, base_seeds)
    extra_rows, extra_prov = validate_source(args.additional, extra_seeds)
    if base_prov.get("datasets") != extra_prov.get("datasets"):
        raise ValueError("Dataset lists differ between base and additional provenance")
    if base_prov.get("tasks") != extra_prov.get("tasks"):
        raise ValueError("Task lists differ between base and additional provenance")
    if base_prov.get("models") != extra_prov.get("models"):
        raise ValueError("Model lists differ between base and additional provenance")
    if base_prov.get("source_sha256") != extra_prov.get("source_sha256"):
        raise ValueError("Source hashes differ; refusing to merge incompatible runs")
    if base_prov.get("protocol_sha256") != extra_prov.get("protocol_sha256"):
        raise ValueError("Protocol hashes differ; refusing to merge incompatible runs")
    merged = base_rows + extra_rows
    merged_keys = {key(row) for row in merged}
    expected_count = len(base_prov["datasets"]) * len(base_prov["tasks"]) * len(base_prov["models"]) * len(expected)
    if len(merged) != expected_count or len(merged_keys) != expected_count:
        raise ValueError(f"Expected {expected_count} unique merged runs, found {len(merged)}")
    args.output.mkdir(parents=True)
    linked = 0
    for source_root, table in ((args.base, base_rows), (args.additional, extra_rows)):
        for row in table:
            run_id = f"{row['dataset']}__{row['task']}__{row['model']}__seed{row['seed']}"
            linked += copy_tree(source_root / run_id, args.output / run_id)
    merged.sort(key=key)
    with (args.output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    provenance = dict(base_prov)
    provenance["seeds"] = sorted(expected)
    provenance["merged_from"] = [str(args.base), str(args.additional)]
    provenance["merge_mode"] = "hardlink_or_copy; source directories untouched"
    provenance["run_count"] = len(merged)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(merged), "hardlinked_files": linked}, indent=2))


if __name__ == "__main__":
    main()
