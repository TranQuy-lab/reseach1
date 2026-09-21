#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv-server}"
DEVICE="${DEVICE:-auto}"
THREADS="${THREADS:-8}"

PYTHONPATH=src "${ENV_DIR}/bin/python" -u research/server/run_pipeline.py \
  --stage all --device "${DEVICE}" --threads "${THREADS}"
