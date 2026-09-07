#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
LOG=/root/gate1/phase1_reward_fix_queue.log
MIN_FREE_BYTES=$((3 * 1024 * 1024 * 1024))

while true; do
  free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d ' ')
  if (( free_bytes >= MIN_FREE_BYTES )); then
    break
  fi
  printf '[%s] WAIT_DISK free_bytes=%s required=%s\n' \
    "$(date -Is)" "$free_bytes" "$MIN_FREE_BYTES"
  sleep 300
done

cd "$ROOT"
printf '[%s] START_SHARED_GPU_PHASE1\n' "$(date -Is)"
exec env ALLOW_SHARED_GPU=1 TRAIN_CUDA_FRACTION=0.28 EVAL_CUDA_FRACTION=0.12 \
  bash scripts/run_order17_phase1_reward_fix.sh >>"$LOG" 2>&1
