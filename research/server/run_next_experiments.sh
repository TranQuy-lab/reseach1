#!/usr/bin/env bash
# Các thí nghiệm bài báo, tách khỏi full_runs/full_runs_5seed đã khóa.
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
  local data_root="$1" output_root="$2" train_passes="$3"
  shift 3
  local models=("$@")
  [[ -d "${ROOT}/${data_root}" ]] || { echo "Missing split root: ${data_root}" >&2; exit 1; }
  exec "${PYTHON_BIN}" -u -m nids_minibatch.training \
    --data "${data_root}" \
    --output "${output_root}" \
    --datasets NF-UNSW-NB15-v2 NF-BoT-IoT-v2 NF-ToN-IoT-v2 NF-CSE-CIC-IDS2018-v2 \
    --tasks multiclass binary \
    --models "${models[@]}" \
    --seeds 11 22 33 44 55 \
    --epochs 1 --patience 10 --batch-size 4096 --fanout 15 10 \
    --threads "${THREADS}" --device cuda --scope full --eval-every 3 --amp \
    --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" --max-train-batches 0 \
    --num-workers "${NIDS_NUM_WORKERS:-4}" \
    --train-passes "${train_passes}" --min-train-steps 1500 --evals-per-pass 4 \
    --protocol research/PROTOCOL_PAPER_EXTENSION_VI.md \
    $(if [[ -d "${ROOT}/${output_root}" ]]; then echo --resume; fi)
}

case "${STAGE}" in
  tabular|tabular_train)
    exec "${PYTHON_BIN}" -u research/run_tabular_baselines.py \
      --data data/full_splits \
      --runs research/artifacts/full_runs_5seed \
      --output research/artifacts/tabular_full_runs \
      --seeds 11 22 33 44 55 \
      --n-jobs "${THREADS}" \
      --n-estimators "${NIDS_TABULAR_ESTIMATORS:-100}" \
      --max-depth "${NIDS_TABULAR_MAX_DEPTH:-32}" \
      --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" \
      $(if [[ -d research/artifacts/tabular_full_runs ]]; then echo --resume; fi)
    ;;
  tabular_verify)
    exec "${PYTHON_BIN}" -u research/validate_tabular_results.py \
      --data data/full_splits \
      --runs research/artifacts/tabular_full_runs \
      --output research/results/tabular_full_verification.json
    ;;
  endpoint_prepare)
    exec "${PYTHON_BIN}" -u research/server/prepare_endpoint_holdout.py \
      --source data/processed_four \
      --output data/endpoint_holdout_splits \
      --report research/results/endpoint_holdout_prepare.json \
      --threads "${THREADS}" --min-class-rows 30
    ;;
  rewire_prepare)
    exec "${PYTHON_BIN}" -u research/server/rewire_full_splits.py \
      --source data/full_splits \
      --output data/rewired_full_splits \
      --report research/results/graph_rewire_prepare.json \
      --threads "${THREADS}"
    ;;
  endpoint_train)
    exec "${PYTHON_BIN}" -u research/server/run_endpoint_holdout_matrix.py \
      --mode train --threads "${THREADS}" \
      --num-workers "${NIDS_NUM_WORKERS:-4}" \
      --prediction-cap "${NIDS_PREDICTION_CAP:-100000}"
    ;;
  endpoint_verify)
    exec "${PYTHON_BIN}" -u research/server/run_endpoint_holdout_matrix.py \
      --mode verify --threads "${THREADS}"
    ;;
  rewire_train)
    train_variant data/rewired_full_splits research/artifacts/rewired_full_runs 2 sage sage_edge
    ;;
  rewire_verify)
    exec "${PYTHON_BIN}" -u research/validate_minibatch_results.py \
      --data data/rewired_full_splits \
      --runs research/artifacts/rewired_full_runs \
      --output research/results/rewired_full_verification.json \
      --threads "${THREADS}"
    ;;
  rewire_eval_rw)
    exec "${PYTHON_BIN}" -u research/evaluate_checkpoints_on_splits.py \
      --data data/rewired_full_splits \
      --runs research/artifacts/full_runs_5seed \
      --output research/artifacts/rewire_eval_RW \
      --condition RW --models edge_mlp sage sage_edge \
      --data-manifest research/results/graph_rewire_prepare.json \
      --threads "${THREADS}" --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" \
      $(if [[ -d research/artifacts/rewire_eval_RW ]]; then echo --resume; fi)
    ;;
  rewire_eval_wr)
    exec "${PYTHON_BIN}" -u research/evaluate_checkpoints_on_splits.py \
      --data data/full_splits \
      --runs research/artifacts/rewired_full_runs \
      --output research/artifacts/rewire_eval_WR \
      --condition WR --models sage sage_edge \
      --data-manifest research/results/full_prepare.json \
      --threads "${THREADS}" --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" \
      $(if [[ -d research/artifacts/rewire_eval_WR ]]; then echo --resume; fi)
    ;;
  budget4)
    exec "${PYTHON_BIN}" -u -m nids_minibatch.training \
      --data data/full_splits --output research/artifacts/bot_budget_4pass \
      --datasets NF-BoT-IoT-v2 --tasks multiclass binary \
      --models sage sage_edge --seeds 11 22 33 44 55 \
      --epochs 1 --patience 40 --batch-size 4096 --fanout 15 10 \
      --threads "${THREADS}" --device cuda --scope full --eval-every 3 --amp \
      --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" --max-train-batches 0 \
      --num-workers "${NIDS_NUM_WORKERS:-4}" \
      --train-passes 4 --min-train-steps 1500 --evals-per-pass 4 \
      --protocol research/PROTOCOL_PAPER_EXTENSION_VI.md \
      $(if [[ -d research/artifacts/bot_budget_4pass ]]; then echo --resume; fi)
    ;;
  budget8)
    exec "${PYTHON_BIN}" -u -m nids_minibatch.training \
      --data data/full_splits --output research/artifacts/bot_budget_8pass \
      --datasets NF-BoT-IoT-v2 --tasks multiclass binary \
      --models sage sage_edge --seeds 11 22 33 44 55 \
      --epochs 1 --patience 40 --batch-size 4096 --fanout 15 10 \
      --threads "${THREADS}" --device cuda --scope full --eval-every 3 --amp \
      --prediction-cap "${NIDS_PREDICTION_CAP:-100000}" --max-train-batches 0 \
      --num-workers "${NIDS_NUM_WORKERS:-4}" \
      --train-passes 8 --min-train-steps 1500 --evals-per-pass 4 \
      --protocol research/PROTOCOL_PAPER_EXTENSION_VI.md \
      $(if [[ -d research/artifacts/bot_budget_8pass ]]; then echo --resume; fi)
    ;;
  budget4_verify|budget8_verify)
    PASS="${STAGE#budget}"; PASS="${PASS%_verify}"
    exec "${PYTHON_BIN}" -u research/validate_minibatch_results.py \
      --data data/full_splits --runs "research/artifacts/bot_budget_${PASS}pass" \
      --output "research/results/bot_budget_${PASS}pass_verification.json" \
      --threads "${THREADS}"
    ;;
  help|*)
    cat <<'USAGE'
Usage: run_next_experiments.sh STAGE

  tabular_train    Run RF, ExtraTrees and HGB on all full splits and five seeds.
  tabular_verify   Reload every tabular model and recompute full-test metrics.
  endpoint_prepare Create strict endpoint-disjoint splits; never overwrites full_splits.
  endpoint_train   Train five seeds only for dataset/tasks passing the class gate.
  endpoint_verify  Verify every eligible endpoint-holdout matrix cell.
  rewire_prepare   Create and hard-verify an endpoint-marginal-preserving rewire.
  rewire_train     Train sage/sage_edge on rewired splits (WW).
  rewire_verify    Reload and verify all WW checkpoints.
  rewire_eval_rw   Evaluate real-trained checkpoints on rewired graphs.
  rewire_eval_wr   Evaluate rewire-trained checkpoints on real graphs.
  budget4|budget8  Run BoT sensitivity at four/eight train passes.
  budget4_verify|budget8_verify  Verify the corresponding budget arm.

The endpoint/rewire train stages use the locked full-data settings:
minimum 1500 steps, batch 4096, fanout [15,10], seeds 11/22/33/44/55,
and research/PROTOCOL_PAPER_EXTENSION_VI.md.
USAGE
    exit 2
    ;;
esac
