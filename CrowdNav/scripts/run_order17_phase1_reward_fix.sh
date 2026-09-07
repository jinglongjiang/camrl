#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
NAV="$ROOT/crowd_nav"
PY=/root/miniconda3/envs/mamba/bin/python
CUDA_RUNNER="$ROOT/scripts/cuda_capped_python.py"
OUT=/root/gate1/phase1_reward_fix
R0=/root/gate1/final_belief_gate_v2
LEGACY=runs/mamba_t24/rl_model_ep10000.pth
SEED=42
MIN_FREE_BYTES=$((3 * 1024 * 1024 * 1024))
ALLOW_SHARED_GPU=${ALLOW_SHARED_GPU:-0}
TRAIN_CUDA_FRACTION=${TRAIN_CUDA_FRACTION:-0.28}
EVAL_CUDA_FRACTION=${EVAL_CUDA_FRACTION:-0.12}

if pgrep -af '[t]rain_interface.py' > /dev/null && [[ "$ALLOW_SHARED_GPU" != 1 ]]; then
  printf 'BLOCKED_ECG_ACTIVE\n' >&2
  exit 75
fi
if pgrep -af '[t]rain_interface.py' > /dev/null; then
  printf 'SHARED_GPU_ECG_ACTIVE train_fraction=%s eval_fraction=%s\n' \
    "$TRAIN_CUDA_FRACTION" "$EVAL_CUDA_FRACTION"
fi

free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d ' ')
if (( free_bytes < MIN_FREE_BYTES )); then
  printf 'BLOCKED_DISK free_bytes=%s required=%s\n' "$free_bytes" "$MIN_FREE_BYTES" >&2
  exit 76
fi

if [[ -e "$OUT/COMPLETE" ]]; then
  printf 'ALREADY_COMPLETE %s\n' "$OUT"
  exit 0
fi
if [[ -e "$OUT/r1/fixed_confidence" || -e "$OUT/r1/full" ]]; then
  printf 'REFUSE_NONEMPTY_R1 %s\n' "$OUT/r1" >&2
  exit 77
fi

mkdir -p "$OUT/eval" "$OUT/r1"
cd "$ROOT"

if [[ ! -e "$OUT/PREFLIGHT_PASS" ]]; then
  "$PY" crowd_nav/test_occlusion_belief.py --unit >"$OUT/preflight_unit.log" 2>&1
  "$PY" crowd_nav/test_occlusion_belief.py --leakage >"$OUT/preflight_leakage.log" 2>&1
  for features in fixed_confidence full; do
    CUDA_MEMORY_FRACTION="$EVAL_CUDA_FRACTION" CUDA_VISIBLE_DEVICES=0 \
      "$PY" "$CUDA_RUNNER" crowd_nav/test_occlusion_belief.py \
      --smoke --modes bayes --backbones mamba --episodes 1 --gpu \
      --belief-features "$features" \
      >"$OUT/preflight_smoke_${features}.log" 2>&1
    grep -q '\[SMOKE\] passed' "$OUT/preflight_smoke_${features}.log"
  done
  touch "$OUT/PREFLIGHT_PASS"
else
  printf '[%s] REUSE_PREFLIGHT_PASS\n' "$(date -Is)" | tee -a "$OUT/progress.log"
fi

run_eval() {
  local reward_arm=$1
  local deadline=$2
  local features=$3
  local model_dir=$4
  local weights=$5
  local env_config=$6
  local label="${reward_arm}_l${deadline}_${features}"
  printf '[%s] EVAL_START label=%s\n' "$(date -Is)" "$label" | tee -a "$OUT/progress.log"
  cd "$NAV"
  CUDA_MEMORY_FRACTION="$EVAL_CUDA_FRACTION" CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$CUDA_RUNNER" test.py \
    --policy mamba_rl \
    --model_dir "$model_dir" \
    --weights "$weights" \
    --env_config "$env_config" \
    --policy_config configs_phase1/policy.config \
    --gpu --episodes 100 --seed "$SEED" --no_progress \
    --time-limit "$deadline" \
    --occlusion-mode bayes --belief-features "$features" \
    --oscillation_csv "$OUT/eval/${label}.csv" \
    --run_label "$label" \
    >"$OUT/eval/${label}.log" 2>&1
  printf '[%s] EVAL_DONE label=%s\n' "$(date -Is)" "$label" | tee -a "$OUT/progress.log"
}

# A single training job is compute-heavy and remains serial.  Evaluation is
# dominated by independent simulator stepping, so the two feature arms may run
# concurrently after the ECG guard has released the GPU.  R0-L25 exact episode
# reproduction below is the admission test: any concurrency-induced numerical
# or seeding change stops the script before new training begins.
run_eval_pair() {
  local reward_arm=$1
  local deadline=$2
  local model_root=$3
  local weights=$4
  local env_config=$5
  local fixed_pid full_pid fixed_rc full_rc

  printf '[%s] EVAL_PAIR_START reward=%s deadline=%s\n' \
    "$(date -Is)" "$reward_arm" "$deadline" | tee -a "$OUT/progress.log"
  run_eval "$reward_arm" "$deadline" fixed_confidence \
    "$model_root/fixed_confidence" "$weights" "$env_config" &
  fixed_pid=$!
  run_eval "$reward_arm" "$deadline" full \
    "$model_root/full" "$weights" "$env_config" &
  full_pid=$!

  set +e
  wait "$fixed_pid"
  fixed_rc=$?
  wait "$full_pid"
  full_rc=$?
  set -e
  if ((fixed_rc != 0 || full_rc != 0)); then
    printf 'EVAL_PAIR_FAIL reward=%s deadline=%s fixed_rc=%s full_rc=%s\n' \
      "$reward_arm" "$deadline" "$fixed_rc" "$full_rc" >&2
    return 1
  fi
  printf '[%s] EVAL_PAIR_DONE reward=%s deadline=%s\n' \
    "$(date -Is)" "$reward_arm" "$deadline" | tee -a "$OUT/progress.log"
}

# R0-L25 already exists and is reused verbatim.  Re-running 1200 old episodes
# adds no scientific cell, so only integrity and row-count checks are retained.
cd "$ROOT"
"$PY" - "$R0/eval" <<'PY' | tee "$OUT/r0_l25_reused.txt"
import csv
import hashlib
import os
import sys

old_dir = sys.argv[1]
for arm in ("fixed_confidence", "full"):
    path = os.path.join(old_dir, arm + ".csv")
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 600:
        raise SystemExit(f"R0_L25_INVALID arm={arm} episodes={len(rows)}")
    print(f"R0_L25_REUSED arm={arm} episodes={len(rows)} sha256={digest}")
PY

run_eval_pair r0 50 "$R0" rl_model_ep1500.pth configs_gate1/env.config

cd "$NAV"
for features in fixed_confidence full; do
  run="$OUT/r1/$features"
  log="$OUT/train_${features}.log"
  printf '[%s] TRAIN_START reward=r1 features=%s seed=%s\n' "$(date -Is)" "$features" "$SEED" | tee -a "$OUT/progress.log"
  CUDA_MEMORY_FRACTION="$TRAIN_CUDA_FRACTION" CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$CUDA_RUNNER" train.py \
    --occlusion-mode bayes \
    --belief-features "$features" \
    --legacy-warm-start "$LEGACY" \
    --config configs_phase1/env.config \
    --outdir "$run" --gpu --seed "$SEED" \
    >"$log" 2>&1
  test -f "$run/rl_model_ep1500.pth"
  grep -q '\[IL-TOKEN-CONTRACT\]' "$log"
  grep -q '\[BELIEF-CONTRACT\]' "$log"
  if grep -Eiq '(^|[[:space:]:])(nan|oom)([[:space:]:,]|$)|Traceback|ABORT' "$log"; then
    printf 'INVALID_FAILURE_TOKEN %s\n' "$log" >&2
    exit 20
  fi
  printf '[%s] TRAIN_DONE reward=r1 features=%s\n' "$(date -Is)" "$features" | tee -a "$OUT/progress.log"
done

cd "$ROOT"
"$PY" - "$OUT/r1" <<'PY' | tee "$OUT/checkpoint_contract.txt"
import os
import sys
import torch

root = sys.argv[1]
for arm in ("fixed_confidence", "full"):
    path = os.path.join(root, arm, "rl_model_ep1500.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {}).get("occlusion", {})
    assert meta.get("belief_features") == arm, (arm, meta)
    state = next((checkpoint[key] for key in
                  ("policy_state", "model_state_dict", "value", "policy", "model", "value_state")
                  if isinstance(checkpoint.get(key), dict)), None)
    assert state is not None
    matches = [(key, value) for key, value in state.items()
               if key.endswith("human_encoder.0.weight")]
    assert len(matches) == 1
    key, weight = matches[0]
    sums = [float(weight[:, col].abs().sum()) for col in (9, 10, 11, 12)]
    assert sums[2] > 0 and sums[3] > 0
    if arm == "full":
        assert sums[0] > 0 and sums[1] > 0
    print(arm, key, "column_abs_sums", sums)
PY

for deadline in 25 50; do
  run_eval_pair r1 "$deadline" "$OUT/r1" rl_model_ep1500.pth configs_phase1/env.config
done

cd "$ROOT"
"$PY" - "$OUT/eval" "$R0/eval" <<'PY' | tee "$OUT/summary.txt"
import csv
import math
import os
import sys

root, old_r0_l25 = sys.argv[1:]

def load(label):
    old_names = {
        "r0_l25_fixed_confidence": "fixed_confidence.csv",
        "r0_l25_full": "full.csv",
    }
    path = (os.path.join(old_r0_l25, old_names[label])
            if label in old_names else os.path.join(root, label + ".csv"))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["scenario"], row["episode"], row["seed"]): row for row in rows}
    assert len(result) == 600, (label, len(result))
    return result

def exact_pair_p(entered, left):
    n = entered + left
    if not n:
        return 1.0
    q = min(entered, left)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(q + 1)) / (2 ** n))

summaries = {}
for reward in ("r0", "r1"):
    for deadline in (25, 50):
        fixed = load(f"{reward}_l{deadline}_fixed_confidence")
        full = load(f"{reward}_l{deadline}_full")
        assert fixed.keys() == full.keys()
        keys = sorted(fixed)
        print(f"CELL reward={reward} deadline={deadline}")
        cell = {}
        for arm, rows in (("fixed", fixed), ("full", full)):
            rates = {outcome: sum(rows[k]["outcome"].strip().lower() == outcome for k in keys) / len(keys)
                     for outcome in ("success", "collision", "timeout")}
            cell[arm] = rates
            print(f"  {arm} SR={rates['success']:.4f} CR={rates['collision']:.4f} TR={rates['timeout']:.4f}")
        for outcome in ("success", "collision", "timeout"):
            entered = sum(full[k]["outcome"].strip().lower() == outcome and
                          fixed[k]["outcome"].strip().lower() != outcome for k in keys)
            left = sum(fixed[k]["outcome"].strip().lower() == outcome and
                       full[k]["outcome"].strip().lower() != outcome for k in keys)
            print(f"  paired_{outcome} entered={entered} left={left} p={exact_pair_p(entered, left):.8g}")
        summaries[(reward, deadline)] = cell

cell = summaries[("r1", 50)]
sr_delta = cell["full"]["success"] - cell["fixed"]["success"]
cr_delta = cell["full"]["collision"] - cell["fixed"]["collision"]
tr_delta = cell["full"]["timeout"] - cell["fixed"]["timeout"]
print(f"PRIMARY_R1_L50 delta_SR={sr_delta:+.4f} delta_CR={cr_delta:+.4f} delta_TR={tr_delta:+.4f}")
if cr_delta < 0 and tr_delta < 0.03:
    verdict = "REWARD_OR_DEADLINE_CONFOUND_DOMINATED_ORIGINAL_FREEZING"
elif cr_delta < 0 and tr_delta > 0.05:
    verdict = "SINGLE_SEED_SAFETY_FREEZING_TRADEOFF_REMAINS_REPLICATE_SEEDS"
elif cr_delta >= 0:
    verdict = "ORIGINAL_COLLISION_EFFECT_DID_NOT_SURVIVE_REWARD_FIX"
else:
    verdict = "INCONCLUSIVE"
print("PHASE1_VERDICT=" + verdict)
PY

touch "$OUT/COMPLETE"
printf '[%s] COMPLETE\n' "$(date -Is)" | tee -a "$OUT/progress.log"
