#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
NAV="$ROOT/crowd_nav"
PY=/root/miniconda3/envs/mamba/bin/python
OUT=/root/gate1/paper2_slot_audit/formal_ep1500
MODEL=/root/gate1/final_belief_gate_v2/full
WEIGHTS=rl_model_ep1500.pth
EPISODES=30
SEED=42

mkdir -p "$OUT"
cd "$ROOT"

"$PY" - "$MODEL/$WEIGHTS" <<'PY' | tee "$OUT/checkpoint_preflight.txt"
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
state = checkpoint.get(
    'policy_state', checkpoint.get(
        'model_state_dict', checkpoint.get('value_state', checkpoint)))
meta = checkpoint.get('meta', {}).get('occlusion', {})
assert checkpoint.get('episode') == 1500, checkpoint.get('episode')
assert meta.get('occlusion_mode') == 'bayes', meta
assert meta.get('belief_features') == 'full', meta
assert meta.get('backbone') == 'mamba', meta
assert any('temporal_encoder.backend' in key for key in state), 'not Mamba'
assert any(key.endswith('value_head.weight') for key in state), 'no scalar V head'
assert checkpoint.get('algo') != 'discrete_mamba', checkpoint.get('algo')
print('path=' + path)
print('episode=1500')
print('occlusion_mode=bayes belief_features=full backbone=mamba')
print('decision=scalar_value_lookahead')
PY

"$PY" crowd_nav/test_occlusion_belief.py --unit >"$OUT/unit.log" 2>&1
"$PY" crowd_nav/tools/audit_ttc_velocity_bias.py \
  >"$OUT/analytic_ttc_bias.txt"

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
    --slot-audit-csv "$OUT/slots_n${density}.csv" \
    --slot-candidate-csv "$OUT/candidates_n${density}.csv" \
    --run_label "full_ep1500_n${density}" \
    >"$OUT/eval_n${density}.log" 2>&1
  test -s "$OUT/slots_n${density}.csv"
  test -s "$OUT/candidates_n${density}.csv"
  if grep -Eiq 'Traceback|RuntimeError|ABORT|(^|[[:space:]:])(nan|oom)([[:space:]:,]|$)' \
      "$OUT/eval_n${density}.log"; then
    echo "invalid failure token in density $density" >&2
    exit 20
  fi
done

"$PY" tools/summarize_occlusion_slots.py "$OUT"/slots_n*.csv \
  | tee "$OUT/slot_summary.txt"
"$PY" tools/summarize_slot_candidates.py "$OUT"/candidates_n*.csv \
  | tee "$OUT/candidate_summary.txt"

echo COMPLETE | tee "$OUT/status.txt"
