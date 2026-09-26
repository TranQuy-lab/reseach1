#!/usr/bin/env bash
# Chay not chuoi release tu stage verify (train + reevaluate da xong).
set -uo pipefail
ROOT=/workspace/reseach1
cd "$ROOT"
PY="$ROOT/.venv-server/bin/python"
export PYTHONPATH=src
for stage in verify report test; do
  echo ""
  echo "=== RELEASE STAGE: $stage $(date -Is) ==="
  "$PY" -u research/server/run_full_pipeline.py --stage "$stage"     --threads 12 --num-workers 4 --budget-usd 5.5 --hourly-price-usd 0.29534726255012
  code=$?
  if [ $code -ne 0 ]; then echo "=== STAGE $stage FAILED exit=$code $(date -Is) ==="; exit $code; fi
  echo "=== STAGE $stage OK $(date -Is) ==="
done
echo ""
echo "=== RELEASE TAIL DONE $(date -Is) ==="
