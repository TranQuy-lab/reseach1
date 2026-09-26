#!/usr/bin/env bash
# Chuỗi 72 run chính thức: train -> verify -> report -> test
# Dùng interpreter .venv-server (torch 2.8.0+cu128, pyg-lib) đúng như provenance.
set -uo pipefail

cd /workspace/reseach1
PY=.venv-server/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=12

echo "=== FULL SWEEP START $(date -Is) ==="
for stage in train verify report test; do
  echo ""
  echo "=== STAGE: $stage $(date -Is) ==="
  $PY -u research/server/run_full_pipeline.py --stage "$stage" --threads 12 --num-workers 4 --budget-usd 5.5 --hourly-price-usd 0.29534726255012
  code=$?
  if [ $code -ne 0 ]; then
    echo "=== STAGE $stage FAILED exit=$code $(date -Is) ==="
    exit $code
  fi
  echo "=== STAGE $stage OK $(date -Is) ==="
done
echo ""
echo "=== FULL SWEEP DONE $(date -Is) ==="
