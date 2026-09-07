#!/bin/bash
set -u

PID_FILE=/root/order17_fair_gate.pid
OUT=/root/gate1/fair_v2
STATUS=/root/order17_fair_gate_watchdog.log

echo "WATCHDOG_START $(date -Is)" >"$STATUS"
while true; do
  if [[ ! -f "$PID_FILE" ]]; then
    echo "STOP missing pid file $(date -Is)" >>"$STATUS"
    exit 1
  fi
  parent=$(cat "$PID_FILE")
  if ! kill -0 "$parent" 2>/dev/null; then
    echo "DRIVER_EXIT $(date -Is)" >>"$STATUS"
    exit 0
  fi

  fault=$(grep -Eim1 'Traceback|CUDA out of memory|OutOfMemoryError|Killed|RuntimeError:' \
    "$OUT"/train_*.log 2>/dev/null || true)
  if [[ -n "$fault" ]]; then
    echo "STOP fault: $fault $(date -Is)" >>"$STATUS"
    pkill -f 'train.py --occlusion-mode (bayes|oracle_belief)' 2>/dev/null || true
    kill "$parent" 2>/dev/null || true
    exit 2
  fi

  replay=$(grep -h 'REPLAY-STORE' "$OUT"/train_*.log 2>/dev/null | tail -1 || true)
  if [[ -n "$replay" ]]; then
    echo "REPLAY $replay" >>"$STATUS"
    if [[ "$replay" == *'ok=0'* ]]; then
      echo "STOP replay did not accept online trajectories $(date -Is)" >>"$STATUS"
      pkill -f 'train.py --occlusion-mode (bayes|oracle_belief)' 2>/dev/null || true
      kill "$parent" 2>/dev/null || true
      exit 3
    fi
  fi
  sleep 15
done
