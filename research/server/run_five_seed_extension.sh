#!/usr/bin/env bash
# Chạy bổ sung seed 44/55 rồi hợp nhất thành full_runs_5seed (120 run).
# Không train lại và không ghi đè full_runs (seed 11/22/33).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${NIDS_PYTHON:-${ROOT}/.venv-server/bin/python}"
THREADS="${NIDS_THREADS:-12}"
NUM_WORKERS="${NIDS_NUM_WORKERS:-4}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_RUNS="research/artifacts/full_runs_seeds44_55"
MERGED_RUNS="research/artifacts/full_runs_5seed"
VERIFY_JSON="research/results/full_verification_5seed.json"

if [[ -f "${VERIFY_JSON}" && -d "${MERGED_RUNS}" ]]; then
  printf '5-seed extension already complete: %s\n' "${VERIFY_JSON}"
  exit 0
fi

# Git intentionally excludes the 375MB prediction parquet files. On a fresh
# server, regenerate the primary 72-run predictions from the tracked model.pt
# checkpoints before merging; this never retrains or changes model weights.
BASE_PROBE="research/artifacts/full_runs/NF-UNSW-NB15-v2__multiclass__edge_mlp__seed11/test_predictions.parquet"
if [[ ! -f "${BASE_PROBE}" ]]; then
  printf 'Primary prediction artifacts missing; rebuilding 72 tracked checkpoints.\n'
  "${PYTHON_BIN}" -u research/rebuild_full_evaluation.py \
    --data data/full_splits --runs research/artifacts/full_runs --threads "${THREADS}"
fi

"${PYTHON_BIN}" -u -m nids_minibatch.training \
  --data data/full_splits \
  --output "${EXTRA_RUNS}" \
  --datasets NF-UNSW-NB15-v2 NF-BoT-IoT-v2 NF-ToN-IoT-v2 NF-CSE-CIC-IDS2018-v2 \
  --tasks multiclass binary \
  --models edge_mlp sage sage_edge \
  --seeds 44 55 \
  --epochs 1 --patience 10 --batch-size 4096 --fanout 15 10 \
  --threads "${THREADS}" --device cuda --scope full --eval-every 3 --amp \
  --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" --max-train-batches 0 \
  --num-workers "${NUM_WORKERS}" \
  --train-passes 2 --min-train-steps 1500 --evals-per-pass 4 \
  $(if [[ -d "${EXTRA_RUNS}" ]]; then echo --resume; fi)

"${PYTHON_BIN}" -u research/rebuild_full_evaluation.py \
  --data data/full_splits --runs "${EXTRA_RUNS}" --threads "${THREADS}"

"${PYTHON_BIN}" -u research/merge_full_seed_runs.py \
  --base research/artifacts/full_runs \
  --additional "${EXTRA_RUNS}" \
  --output "${MERGED_RUNS}" \
  --allow-source-mismatch

"${PYTHON_BIN}" -u research/validate_minibatch_results.py \
  --data data/full_splits \
  --runs "${MERGED_RUNS}" \
  --output research/results/full_verification_5seed.json \
  --threads "${THREADS}"

printf '\n5-seed extension complete: %s\n' "${MERGED_RUNS}"
