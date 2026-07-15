#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/txdigitalafrica/temperature-scaling-research"
PY="/home/txdigitalafrica/mambaforge/bin/python"
RES="$ROOT/results/raw_v2"
LOGDIR="$ROOT/logs"
export PYTHONUNBUFFERED=1
cd "$ROOT"
echo "==== VISION START $(date -Is) ===="
for m in resnet18 resnet50 resnet101; do
  for s in 42 123 456; do
    out="$RES/results_${m}_seed${s}.json"
    if [ -f "$out" ]; then echo "[skip] $m seed=$s"; continue; fi
    echo "[run] vision $m seed=$s $(date -Is)"
    "$PY" -u experiments/vision/train_resnet.py \
      --model_size "$m" --seed "$s" --epochs 30 \
      --results_dir "$RES" \
      > "$LOGDIR/vis_${m}_seed${s}.log" 2>&1
    echo "[done] vision $m seed=$s rc=$? $(date -Is)"
  done
done
echo "==== AGGREGATE $(date -Is) ===="
"$PY" -u experiments/aggregate.py --results_dir "$RES" --out_prefix final_results_v2
echo "==== VISION END $(date -Is) ===="
