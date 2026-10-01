#!/usr/bin/env bash
# Sequential TN-2 driver. Shares the machine with TN-1, so OMP threads are capped.
set -u
cd "$(dirname "$0")/.."
. .venv/bin/activate
export OMP_NUM_THREADS=5
J=5
run() { echo "=== $* ==="; python scripts/09_tn2_tabular_comparators.py "$@" || echo "FAILED: $*"; }

run --datasets NF-UNSW-NB15-v2 --tasks multiclass --models hist_gradient_boosting extra_trees random_forest --seeds 11 22 33 --n-jobs $J
run --datasets NF-UNSW-NB15-v2 --tasks binary     --models hist_gradient_boosting extra_trees random_forest --seeds 11 22 33 --n-jobs $J
run --datasets NF-ToN-IoT-v2   --tasks multiclass --models hist_gradient_boosting extra_trees random_forest --seeds 11 22 33 --n-jobs $J
run --datasets NF-ToN-IoT-v2   --tasks binary     --models hist_gradient_boosting extra_trees random_forest --seeds 11 22 33 --n-jobs $J
run --datasets NF-CSE-CIC-IDS2018-v2 --tasks multiclass --models hist_gradient_boosting --seeds 11 22 33 --n-jobs $J
run --datasets NF-CSE-CIC-IDS2018-v2 --tasks binary     --models hist_gradient_boosting --seeds 11 22 33 --n-jobs $J
run --datasets NF-BoT-IoT-v2   --tasks multiclass --models hist_gradient_boosting --seeds 11 22 33 --n-jobs $J
run --datasets NF-BoT-IoT-v2   --tasks binary     --models hist_gradient_boosting --seeds 11 22 33 --n-jobs $J
echo "TN2_DRIVER_DONE"
