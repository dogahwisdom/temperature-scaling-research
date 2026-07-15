#!/usr/bin/env bash
# Continue remaining vision runs into results/raw_v2, then re-aggregate.
set -euo pipefail
REPO=/home/txdigitalafrica/temperature-scaling-research
PY=/home/txdigitalafrica/mambaforge/bin/python
OUT="$REPO/results/raw_v2"
LOGDIR="$REPO/logs"
cd "$REPO"
mkdir -p "$OUT" "$LOGDIR"

run_one() {
  local model=$1 seed=$2
  local out="$OUT/results_${model}_seed${seed}.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $model seed=$seed already exists"
    return 0
  fi
  echo "[run] $model seed=$seed $(date -Iseconds)"
  PYTHONUNBUFFERED=1 "$PY" -u experiments/vision/train_resnet.py \
    --model_size "$model" --seed "$seed" --epochs 30 \
    --results_dir "$OUT" \
    > "$LOGDIR/vis_${model}_seed${seed}.log" 2>&1
  local rc=$?
  echo "[done] $model seed=$seed rc=$rc $(date -Iseconds)"
  return $rc
}

# Wait for any existing train_resnet to finish
while pgrep -f 'experiments/vision/train_resnet.py' >/dev/null 2>&1; do
  echo "[wait] existing train_resnet still running... $(date -Iseconds)"
  sleep 60
done

for model in resnet18 resnet50 resnet101; do
  for seed in 42 123 456; do
    run_one "$model" "$seed" || echo "[ERROR] $model seed=$seed failed" >&2
  done
done

echo "[aggregate] $(date -Iseconds)"
"$PY" experiments/aggregate.py --results_dir "$OUT" --out_prefix final_results_v2
echo "[ALL VISION DONE] $(date -Iseconds)"
