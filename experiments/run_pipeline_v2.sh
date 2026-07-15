#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/txdigitalafrica/temperature-scaling-research"
PY="/home/txdigitalafrica/mambaforge/bin/python"
RES="$ROOT/results/raw_v2"
LOGDIR="$ROOT/logs"
mkdir -p "$RES" "$LOGDIR" "$ROOT/results/logits"
export PYTHONUNBUFFERED=1
cd "$ROOT"

run_lang() {
  local model="$1" ds="$2" seed="$3"
  local slug
  if [ "$ds" = "SNLI" ]; then slug=chaosnli_s; else slug=chaosnli_m; fi
  local out="$RES/results_${model}_${slug}_seed${seed}.json"
  if [ -f "$out" ]; then
    echo "[skip] language $model $ds seed=$seed"
    return 0
  fi
  echo "[run] language $model $ds seed=$seed $(date -Is)"
  "$PY" -u experiments/language/train_bert.py \
    --model_name "$model" --dataset_type "$ds" --seed "$seed" --epochs 1 \
    --results_dir "$RES" \
    > "$LOGDIR/lang_${model}_${ds}_seed${seed}.log" 2>&1
  local rc=$?
  echo "[done] language $model $ds seed=$seed rc=$rc $(date -Is)"
  return $rc
}

run_vis() {
  local model="$1" seed="$2"
  local out="$RES/results_${model}_seed${seed}.json"
  if [ -f "$out" ]; then
    echo "[skip] vision $model seed=$seed"
    return 0
  fi
  echo "[run] vision $model seed=$seed $(date -Is)"
  "$PY" -u experiments/vision/train_resnet.py \
    --model_size "$model" --seed "$seed" --epochs 30 \
    --results_dir "$RES" \
    > "$LOGDIR/vis_${model}_seed${seed}.log" 2>&1
  local rc=$?
  echo "[done] vision $model seed=$seed rc=$rc $(date -Is)"
  return $rc
}

echo "==== PIPELINE START $(date -Is) ===="

# Task 1 priority: corrected BERT-large SNLI
for s in 42 123 456; do run_lang bert-large-uncased SNLI $s || true; done

# Remaining language (untouched configs, but need logits + isotonic)
for m in distilbert-base-uncased bert-base-uncased bert-large-uncased; do
  for d in SNLI MNLI; do
    for s in 42 123 456; do
      # skip already-done bert-large SNLI
      if [ "$m" = "bert-large-uncased" ] && [ "$d" = "SNLI" ]; then continue; fi
      run_lang "$m" "$d" "$s" || true
    done
  done
done

# Vision (all 9)
for m in resnet18 resnet50 resnet101; do
  for s in 42 123 456; do
    run_vis "$m" "$s" || true
  done
done

echo "==== AGGREGATE $(date -Is) ===="
"$PY" -u experiments/aggregate.py --results_dir "$RES" --out_prefix final_results_v2
echo "==== PIPELINE END $(date -Is) ===="
