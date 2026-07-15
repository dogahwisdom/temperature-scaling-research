#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/txdigitalafrica/temperature-scaling-research"
PY="/home/txdigitalafrica/mambaforge/bin/python"
RES="$ROOT/results/raw_v2"
LOGDIR="$ROOT/logs"
export PYTHONUNBUFFERED=1
cd "$ROOT"
echo "==== FINISH RESNET101 START $(date -Is) ====" | tee -a "$LOGDIR/finish_resnet101.log"
for s in 123 456; do
  out="$RES/results_resnet101_seed${s}.json"
  if [ -f "$out" ]; then echo "[skip] resnet101 seed=$s"; continue; fi
  echo "[run] resnet101 seed=$s $(date -Is)" | tee -a "$LOGDIR/finish_resnet101.log"
  "$PY" -u experiments/vision/train_resnet.py \
    --model_size resnet101 --seed "$s" --epochs 30 \
    --results_dir "$RES" \
    > "$LOGDIR/vis_resnet101_seed${s}.log" 2>&1
  echo "[done] resnet101 seed=$s rc=$? $(date -Is)" | tee -a "$LOGDIR/finish_resnet101.log"
done
echo "==== AGGREGATE $(date -Is) ====" | tee -a "$LOGDIR/finish_resnet101.log"
"$PY" -u experiments/aggregate.py --results_dir "$RES" --out_prefix final_results_v2
echo "==== ALL DONE $(date -Is) ====" | tee -a "$LOGDIR/finish_resnet101.log"
