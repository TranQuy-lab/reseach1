#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${NIDS_PYTHON:-${ROOT}/.venv-server/bin/python}"
THREADS="${NIDS_THREADS:-12}"
NUM_WORKERS="${NIDS_NUM_WORKERS:-4}"
BUDGET_USD="${NIDS_BUDGET_USD:-5.5}"
HOURLY_PRICE_USD="${NIDS_HOURLY_PRICE_USD:?Set the actual server hourly price}"

cd "${ROOT}"
for stage in train reevaluate verify report test; do
  echo "=== RELEASE STAGE: ${stage} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -u research/server/run_full_pipeline.py \
    --stage "${stage}" \
    --threads "${THREADS}" \
    --num-workers "${NUM_WORKERS}" \
    --budget-usd "${BUDGET_USD}" \
    --hourly-price-usd "${HOURLY_PRICE_USD}"
done
