#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
NAV="$ROOT/crowd_nav"
PY=/root/miniconda3/envs/mamba/bin/python
OUT=/root/gate1/fair_v2
CKPT=runs/mamba_t24/rl_model_ep10000.pth

mkdir -p "$OUT/eval"
cd "$NAV"

for mode in bayes oracle_belief; do
  run="$OUT/$mode"
  log="$OUT/train_${mode}.log"
  rm -rf "$run"
  echo "[$(date -Is)] training $mode"
  "$PY" train.py \
    --occlusion-mode "$mode" \
    --legacy-warm-start "$CKPT" \
    --config configs_gate1/env.config \
    --outdir "$run" --gpu >"$log" 2>&1
  test -f "$run/rl_model_ep1500.pth"
  echo "[$(date -Is)] training $mode complete"
done

for mode in bayes oracle_belief; do
  echo "[$(date -Is)] evaluating $mode"
  "$PY" test.py \
    --policy mamba_rl \
    --model_dir "$OUT/$mode" \
    --weights rl_model_ep1500.pth \
    --env_config configs_gate1/env.config \
    --gpu --episodes 100 --seed 42 --no_progress \
    --occlusion-mode "$mode" --legacy-diagnostic \
    >"$OUT/eval/eval_${mode}.log" 2>&1
  echo "[$(date -Is)] evaluation $mode complete"
done

{
  echo "ORDER17_FAIR_GATE_COMPLETE $(date -Is)"
  for mode in bayes oracle_belief; do
    echo "=== $mode ==="
    grep -E "SUCCESS:|COLLISION:|TIMEOUT:|BELIEF:|hidden-recall|hidden-true-prob" \
      "$OUT/eval/eval_${mode}.log"
  done
} >"$OUT/summary.txt"
cat "$OUT/summary.txt"
