#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
NAV="$ROOT/crowd_nav"
PY=/root/miniconda3/envs/mamba/bin/python
OUT=/root/gate1/paper2_slot_audit/holdout_seed43
MODEL=/root/gate1/final_belief_gate_v2/full
WEIGHTS=rl_model_ep1500.pth
CONTRACT=cap_10__slots_15__visible5_hidden_confidence
EPISODES=30
SEED=43

mkdir -p "$OUT"
cd "$ROOT"
"$PY" -m py_compile crowd_nav/test.py crowd_nav/tools/audit_slot_contracts.py
"$PY" crowd_nav/test_occlusion_belief.py --unit >"$OUT/unit.log" 2>&1

cd "$NAV"
for density in 5 10 15 20; do
  CUDA_VISIBLE_DEVICES=0 "$PY" test.py \
    --policy mamba_rl \
    --model_dir "$MODEL" \
    --weights "$WEIGHTS" \
    --env_config configs_gate1/env.config \
    --gpu --episodes "$EPISODES" --seed "$SEED" --no_progress \
    --test_case 0 --human-num-override "$density" \
    --occlusion-mode bayes --belief-features full \
    --slot-candidate-csv "$OUT/candidates_n${density}.csv" \
    --slot-truth-csv "$OUT/truth_n${density}.csv" \
    --run_label "holdout_seed43_n${density}" \
    >"$OUT/eval_n${density}.log" 2>&1
  test -s "$OUT/candidates_n${density}.csv"
  test -s "$OUT/truth_n${density}.csv"
  if grep -Eiq 'Traceback|RuntimeError|ABORT|(^|[[:space:]:])(nan|oom)([[:space:]:,]|$)' \
      "$OUT/eval_n${density}.log"; then
    echo "invalid failure token in density $density" >&2
    exit 20
  fi
done

cd "$ROOT"
"$PY" crowd_nav/tools/audit_slot_contracts.py \
  --candidates "$OUT/candidates_n*.csv" \
  --truth "$OUT/truth_n*.csv" \
  --output-prefix "$OUT/contract" \
  --confirm-contract "$CONTRACT" \
  | tee "$OUT/decision.txt"

test -s "$OUT/contract_confirm.csv"
test -s "$OUT/contract_decision.json"
echo COMPLETE | tee "$OUT/status.txt"
