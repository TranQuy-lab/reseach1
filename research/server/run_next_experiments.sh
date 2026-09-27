#!/usr/bin/env bash
# Các thí nghiệm nối tiếp, tách khỏi 72-run primary pipeline.
# Dùng cùng .venv-server và data/full_splits trên server cũ.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${NIDS_PYTHON:-${ROOT}/.venv-server/bin/python}"
THREADS="${NIDS_THREADS:-12}"
STAGE="${1:-help}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${STAGE}" in
  tabular)
    exec "${PYTHON_BIN}" -u research/run_tabular_baselines.py \
      --data data/full_splits \
      --runs research/artifacts/full_runs \
      --output research/artifacts/tabular_full_runs \
      --n-jobs "${THREADS}" \
      --n-estimators "${NIDS_TABULAR_ESTIMATORS:-100}" \
      --max-depth "${NIDS_TABULAR_MAX_DEPTH:-32}" \
      --prediction-cap "${NIDS_PREDICTION_CAP:-100000}"
    ;;
  endpoint_prepare)
    exec "${PYTHON_BIN}" -u research/server/prepare_endpoint_holdout.py \
      --source data/processed_four \
      --output data/endpoint_holdout_splits \
      --report research/results/endpoint_holdout_prepare.json \
      --threads "${THREADS}"
    ;;
  rewire_prepare)
    exec "${PYTHON_BIN}" -u research/server/rewire_full_splits.py \
      --source data/full_splits \
      --output data/rewired_full_splits \
      --report research/results/graph_rewire_prepare.json \
      --threads "${THREADS}"
    ;;
  help|*)
    cat <<'USAGE'
Usage: run_next_experiments.sh {tabular|endpoint_prepare|rewire_prepare}

  tabular          Run RF, ExtraTrees and HistGradientBoosting on all full splits.
  endpoint_prepare Create strict endpoint-disjoint splits; never overwrites full_splits.
  rewire_prepare   Create graph-rewired copies; never overwrites full_splits.

After endpoint_prepare or rewire_prepare, launch the existing training module
with --data pointing to the generated split root, using a separate output folder.
USAGE
    exit 2
    ;;
esac
