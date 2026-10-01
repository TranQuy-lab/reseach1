#!/usr/bin/env bash
# Wait until at least 9 GiB is available, then run the Gate B tabular comparators
# for the large datasets. HistGradientBoosting peaks at ~8.1 GiB on ToN-IoT, so it
# must not overlap the training jobs already running.
set -u
cd "$(dirname "$0")/.."
. .venv/bin/activate
need=9
for i in $(seq 1 240); do
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$avail" -ge "$need" ]; then
    echo "[$(date +%H:%M:%S)] MemAvailable=${avail}GiB >= ${need}GiB, starting TN-2"
    break
  fi
  sleep 30
done
export OMP_NUM_THREADS=8
python scripts/09_tn2_tabular_comparators.py \
  --datasets NF-ToN-IoT-v2 --tasks multiclass binary \
  --models hist_gradient_boosting extra_trees random_forest \
  --seeds 11 22 33 --n-jobs 8
python scripts/09_tn2_tabular_comparators.py \
  --datasets NF-CSE-CIC-IDS2018-v2 --tasks multiclass binary \
  --models hist_gradient_boosting --seeds 11 22 33 --n-jobs 8
echo TN2_LARGE_DONE
