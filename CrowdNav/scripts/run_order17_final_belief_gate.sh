#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
NAV="$ROOT/crowd_nav"
PY=/root/miniconda3/envs/mamba/bin/python
OUT=/root/gate1/final_belief_gate_v2
CKPT=runs/mamba_t24/rl_model_ep10000.pth
SEED=42

mkdir -p "$OUT/eval"
cd "$ROOT"

"$PY" crowd_nav/test_occlusion_belief.py --unit >"$OUT/preflight_unit.log" 2>&1
"$PY" crowd_nav/test_occlusion_belief.py --leakage >"$OUT/preflight_leakage.log" 2>&1
for features in fixed_confidence full; do
  CUDA_VISIBLE_DEVICES=0 "$PY" crowd_nav/test_occlusion_belief.py \
    --smoke --modes bayes --backbones mamba --episodes 1 --gpu \
    --belief-features "$features" \
    >"$OUT/preflight_smoke_${features}.log" 2>&1
done

cd "$NAV"
for features in fixed_confidence full; do
  run="$OUT/$features"
  log="$OUT/train_${features}.log"
  rm -rf "$run"
  echo "[$(date -Is)] TRAIN_START features=$features seed=$SEED" | tee -a "$OUT/progress.log"
  "$PY" train.py \
    --occlusion-mode bayes \
    --belief-features "$features" \
    --legacy-warm-start "$CKPT" \
    --config configs_gate1/env.config \
    --outdir "$run" --gpu --seed "$SEED" \
    >"$log" 2>&1
  test -f "$run/rl_model_ep1500.pth"
  grep -q '\[IL-TOKEN-CONTRACT\]' "$log"
  grep -q '\[BELIEF-CONTRACT\]' "$log"
  if grep -Eiq '(^|[[:space:]:])(nan|oom)([[:space:]:,]|$)|Traceback|ABORT' "$log"; then
    echo "invalid failure token in $log" >&2
    exit 20
  fi
  echo "[$(date -Is)] TRAIN_DONE features=$features" | tee -a "$OUT/progress.log"
done

"$PY" - "$OUT" <<'PY'
import os, sys, torch

out = sys.argv[1]
for arm in ("fixed_confidence", "full"):
    path = os.path.join(out, arm, "rl_model_ep1500.pth")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {}).get("occlusion", {})
    assert meta.get("belief_features") == arm, (arm, meta)
    state = None
    for key in ("policy_state", "model_state_dict", "value", "policy", "model", "value_state"):
        if isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break
    assert state is not None
    matches = [(k, v) for k, v in state.items()
               if k.endswith("human_encoder.0.weight")]
    assert len(matches) == 1, [k for k, _ in matches]
    key, weight = matches[0]
    sums = [float(weight[:, col].abs().sum()) for col in (9, 10, 11, 12)]
    assert sums[2] > 0 and sums[3] > 0, (arm, sums)
    if arm == "full":
        assert sums[0] > 0 and sums[1] > 0, (arm, sums)
    print(arm, key, "column_abs_sums", sums)
PY

for features in fixed_confidence full; do
  echo "[$(date -Is)] EVAL_START features=$features" | tee -a "$OUT/progress.log"
  "$PY" test.py \
    --policy mamba_rl \
    --model_dir "$OUT/$features" \
    --weights rl_model_ep1500.pth \
    --env_config configs_gate1/env.config \
    --gpu --episodes 100 --seed "$SEED" --no_progress \
    --occlusion-mode bayes --belief-features "$features" \
    --oscillation_csv "$OUT/eval/${features}.csv" \
    --run_label "$features" \
    >"$OUT/eval/${features}.log" 2>&1
  echo "[$(date -Is)] EVAL_DONE features=$features" | tee -a "$OUT/progress.log"
done

"$PY" - "$OUT" <<'PY' | tee "$OUT/summary.txt"
import csv, math, os, sys
from collections import defaultdict

out = sys.argv[1]
def load(arm):
    path = os.path.join(out, "eval", arm + ".csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["scenario"], r["episode"], r["seed"]): r for r in rows}

fixed, full = load("fixed_confidence"), load("full")
assert fixed.keys() == full.keys(), (len(fixed), len(full), len(fixed.keys() ^ full.keys()))
assert len(fixed) == 600, len(fixed)

def success(r): return r["outcome"].strip().lower() == "success"
def collision(r): return r["outcome"].strip().lower() == "collision"

keys = sorted(fixed)
f_sr = sum(success(fixed[k]) for k in keys) / len(keys)
b_sr = sum(success(full[k]) for k in keys) / len(keys)
f_cr = sum(collision(fixed[k]) for k in keys) / len(keys)
b_cr = sum(collision(full[k]) for k in keys) / len(keys)
wins = sum(success(full[k]) and not success(fixed[k]) for k in keys)
losses = sum(success(fixed[k]) and not success(full[k]) for k in keys)
n = wins + losses
if n:
    tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
else:
    p = 1.0

print("ORDER17_FINAL_BELIEF_GATE")
print(f"fixed_confidence SR={f_sr:.4f} CR={f_cr:.4f}")
print(f"full             SR={b_sr:.4f} CR={b_cr:.4f}")
print(f"delta_full_minus_fixed SR={b_sr-f_sr:+.4f} CR={b_cr-f_cr:+.4f}")
print(f"paired_success full_wins={wins} full_losses={losses} exact_p={p:.6g}")
for target in ("collision", "timeout"):
    entered = sum(
        full[k]["outcome"].strip().lower() == target
        and fixed[k]["outcome"].strip().lower() != target for k in keys)
    left = sum(
        fixed[k]["outcome"].strip().lower() == target
        and full[k]["outcome"].strip().lower() != target for k in keys)
    n_pair = entered + left
    if n_pair:
        q = min(entered, left)
        pair_p = min(
            1.0,
            2.0 * sum(math.comb(n_pair, i) for i in range(q + 1))
            / (2 ** n_pair),
        )
    else:
        pair_p = 1.0
    print(
        f"paired_{target} entered={entered} left={left} exact_p={pair_p:.6g}")

by_scene = defaultdict(list)
for k in keys: by_scene[k[0]].append(k)
for scene, scene_keys in sorted(by_scene.items()):
    fs = sum(success(fixed[k]) for k in scene_keys) / len(scene_keys)
    bs = sum(success(full[k]) for k in scene_keys) / len(scene_keys)
    print(f"scene={scene} fixed={fs:.3f} full={bs:.3f} delta={bs-fs:+.3f}")

if b_sr - f_sr >= 0.03 and f_cr - b_cr >= 0.02 and p < 0.05:
    verdict = "PASS_REPLICATE_SEEDS"
elif b_sr - f_sr < 0.03 or b_cr >= f_cr:
    verdict = "NO_GO_BAYES_CONFIDENCE_HAS_NO_USEFUL_INCREMENT"
else:
    verdict = "INCONCLUSIVE_DO_NOT_CLAIM"
print("VERDICT=" + verdict)
PY

echo "[$(date -Is)] COMPLETE" | tee -a "$OUT/progress.log"
