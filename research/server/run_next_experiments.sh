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

train_variant() {
  local data_root="$1" output_root="$2"
  [[ -d "${ROOT}/${data_root}" ]] || { echo "Missing split root: ${data_root}" >&2; exit 1; }
  exec "${PYTHON_BIN}" -u -m nids_minibatch.training \
    --data "${data_root}" \
    --output "${output_root}" \
    --datasets NF-UNSW-NB15-v2 NF-BoT-IoT-v2 NF-ToN-IoT-v2 NF-CSE-CIC-IDS2018-v2 \
    --tasks multiclass binary \
    --models edge_mlp sage sage_edge \
    --seeds 11 22 33 \
    --epochs 1 --patience 10 --batch-size 4096 --fanout 15 10 \
    --threads "${THREADS}" --device cuda --scope full --eval-every 3 --amp \
    --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" --max-train-batches 0 \
    --num-workers "${NIDS_NUM_WORKERS:-4}" \
    --train-passes 2 --min-train-steps 1500 --evals-per-pass 4
}

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
  endpoint_train)
    train_variant data/endpoint_holdout_splits research/artifacts/endpoint_holdout_runs
    ;;
  rewire_train)
    train_variant data/rewired_full_splits research/artifacts/rewired_full_runs
    ;;
  help|*)
    cat <<'USAGE'
Usage: run_next_experiments.sh {tabular|endpoint_prepare|endpoint_train|rewire_prepare|rewire_train}

  tabular          Run RF, ExtraTrees and HistGradientBoosting on all full splits.
  endpoint_prepare Create strict endpoint-disjoint splits; never overwrites full_splits.
  endpoint_train   Train the existing three GNN variants on endpoint splits.
  rewire_prepare   Create graph-rewired copies; never overwrites full_splits.
  rewire_train     Train the existing three GNN variants on rewired splits.

The endpoint/rewire train stages use the locked full-data settings:
2 passes, minimum 1500 steps, batch 4096, fanout [15,10], seeds 11/22/33.
Run validation separately with validate_minibatch_results.py and a distinct
output JSON after each variant.
USAGE
    exit 2
    ;;
esac
