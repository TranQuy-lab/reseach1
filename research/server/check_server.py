"""Fail-fast environment and data inventory before an expensive run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import psutil
import pyarrow.parquet as pq
import torch


EXPECTED_PACKAGES = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "pyarrow": "22.0.0",
    "scikit-learn": "1.7.2",
    "torch-geometric": "2.7.0",
    "pyg-lib": "0.5.0",
    "torch": "2.8.0",
}

# torch reports usable VRAM, which is slightly below the marketed capacity.
# A 24 GB RTX 3090 reports about 23.56 GiB here.
MINIMUM_FULL_GPU_MEMORY_GIB = 23.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_ready(path: Path, expected_rows: int,
                  expected_sha256: str) -> tuple[bool, str | None, dict]:
    if not path.is_file():
        return False, "missing", {}
    if path.stat().st_size < 1_000_000:
        return False, "too small (possibly a Git LFS pointer)", {}
    with path.open("rb") as stream:
        if stream.read(4) != b"PAR1":
            return False, "invalid Parquet header", {}
    rows = pq.ParquetFile(path).metadata.num_rows
    digest = sha256_file(path)
    details = {"rows": rows, "sha256": digest}
    if rows != expected_rows:
        return False, f"row count {rows}, expected {expected_rows}", details
    if digest != expected_sha256:
        return False, "SHA-256 mismatch", details
    return True, None, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["minibatch", "full"], default="minibatch")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-low-resource", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dataset_names = (
        "NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2",
        "NF-CSE-CIC-IDS2018-v2",
    )
    manifest = json.loads((root / "research/server/data_manifest.json").read_text())
    processed_dir = root / "data/processed_four"
    data_checks = {}
    for name in dataset_names:
        path = processed_dir / f"{name}.parquet"
        ready, reason, details = parquet_ready(
            path, manifest["expected_rows"][name],
            manifest["processed_parquet_sha256"][name],
        )
        data_checks[name] = {
            "path": str(path.relative_to(root)),
            "ready": ready,
            "bytes": path.stat().st_size if path.is_file() else None,
            "problem": reason,
            **details,
        }
    processed_data_present = all(value["ready"] for value in data_checks.values())
    # A fresh run needs room for the 13 GB merged CSV and preprocessing outputs.
    # When all verified Parquet inputs are already staged, only split/run artifacts
    # and the environment need additional space.
    required_free_disk_gib = 12 if processed_data_present else 25
    memory_gib = psutil.virtual_memory().total / 1024**3
    disk_gib = shutil.disk_usage(root).free / 1024**3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    package_versions = {}
    package_problems = []
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        package_versions[package] = actual
        # CUDA/CPU build suffixes are intentionally ignored for semantic checks.
        if actual is None or actual.split("+")[0] != expected:
            package_problems.append(f"{package}={actual!r}, expected {expected}")
    cuda_properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    gpu_memory_gib = cuda_properties.total_memory / 1024**3 if cuda_properties else None
    bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.run(
            ["git", "diff", "--quiet", "--no-ext-diff", "HEAD", "--", ".",
             ":(exclude)*.parquet"], cwd=root, stderr=subprocess.DEVNULL,
        )
        if status.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(status.returncode, status.args)
        git_dirty = status.returncode == 1
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu": psutil.cpu_count(),
        "physical_cpu": psutil.cpu_count(logical=False),
        "memory_gib": memory_gib,
        "free_disk_gib": disk_gib,
        "processed_data_present": processed_data_present,
        "processed_data": data_checks,
        "required_free_disk_gib": required_free_disk_gib,
        "torch": torch.__version__,
        "packages": package_versions,
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": gpu_memory_gib,
        "bf16_supported": bf16_supported,
        "scope": args.scope,
        "git_commit": git_commit,
        "tracked_files_dirty": git_dirty,
    }
    problems = []
    if tuple(map(int, platform.python_version_tuple()[:2])) != (3, 12):
        problems.append(f"Python {platform.python_version()} installed, expected 3.12.x")
    minimum_ram = 120 if args.scope == "full" else 12
    if memory_gib < minimum_ram:
        problems.append(f"RAM below {minimum_ram} GiB")
    if disk_gib < required_free_disk_gib:
        problems.append(f"free disk below {required_free_disk_gib} GiB")
    if not processed_data_present:
        problems.append("one or more processed Parquet files are missing or invalid")
    if package_problems:
        problems.extend(package_problems)
    if args.require_cuda and not torch.cuda.is_available():
        problems.append("CUDA is required but unavailable")
    if args.scope == "full" and torch.cuda.is_available():
        if torch.version.cuda != "12.8":
            problems.append(f"Torch CUDA runtime is {torch.version.cuda!r}, expected '12.8'")
        if not bf16_supported:
            problems.append("GPU does not report BF16 support")
        if (gpu_memory_gib is not None
                and gpu_memory_gib < MINIMUM_FULL_GPU_MEMORY_GIB):
            problems.append(
                f"usable GPU VRAM below {MINIMUM_FULL_GPU_MEMORY_GIB:.0f} GiB"
            )
    if args.scope == "full" and git_dirty is True:
        problems.append("tracked repository files are dirty")
    if args.scope == "full" and git_dirty is None:
        problems.append("Git revision is unavailable")
    result["resource_problems"] = problems
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if problems and not args.allow_low_resource:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
