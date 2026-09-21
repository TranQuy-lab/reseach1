"""Fail-fast server inventory for the verified mini-batch pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import psutil
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-low-resource", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dataset_names = (
        "NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2",
        "NF-CSE-CIC-IDS2018-v2",
    )
    processed_dir = root / "data/processed_four"
    processed_data_present = all(
        (processed_dir / f"{name}.parquet").is_file() for name in dataset_names
    )
    # A fresh run needs room for the 13 GB merged CSV and preprocessing outputs.
    # When all verified Parquet inputs are already staged, only split/run artifacts
    # and the environment need additional space.
    required_free_disk_gib = 12 if processed_data_present else 25
    memory_gib = psutil.virtual_memory().total / 1024**3
    disk_gib = shutil.disk_usage(root).free / 1024**3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu": psutil.cpu_count(),
        "physical_cpu": psutil.cpu_count(logical=False),
        "memory_gib": memory_gib,
        "free_disk_gib": disk_gib,
        "processed_data_present": processed_data_present,
        "required_free_disk_gib": required_free_disk_gib,
        "torch": torch.__version__,
        "torch_geometric": importlib.metadata.version("torch-geometric"),
        "pyg_lib": importlib.metadata.version("pyg-lib"),
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scope": "verified stratified mini-batch benchmark",
    }
    problems = []
    if memory_gib < 12:
        problems.append("RAM below 12 GiB")
    if disk_gib < required_free_disk_gib:
        problems.append(f"free disk below {required_free_disk_gib} GiB")
    result["resource_problems"] = problems
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if problems and not args.allow_low_resource:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
