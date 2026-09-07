"""Order 14 section 7: STAGE ACCEPTANCE -- closed-loop greedy evaluation.

The ONLY go/no-go. top-1 and audit MC are telemetry from here on: both were
measured on the frozen 768 audit rows at the ranking loss's own tau grid, so
they reported 0.881 while the deployed policy completed 0 of 15 episodes.

Reports, on the 90 pre-registered stage-acceptance layouts:
  * per-scenario and macro SR / CR / TR                    <- the gate
  * speed-ring confusion of the greedy action stream       <- was the speed
                                                              dimension ranked
  * step-0 speed ring, model vs the ORCA action at the
    identical initial state                                <- cold start

Usage: order14_stage_accept.py <checkpoint.pth> [tag]
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, ".")
from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec, HUMAN_FEATURE_DIM_V7
from crowd_nav.bayesian_dvl.intent_train import (
    DistributionalValueModel, collect_online_episode, collect_raw_orca_episode)
from crowd_nav.bayesian_dvl.evaluation_protocol import STAGE_ACCEPT_SEEDS, STAGE_ACCEPT_PLAN

CKPT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else Path(CKPT).parent.name
ENVCFG = Path("crowd_nav/configs/env_bayesian_dvl.config")
CFG = "crowd_nav/configs/train_intent_bdvl.config"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
MACRO_MIN, PER_SCENARIO_MIN = 0.70, 0.50

cfg = load_intent_training_config(CFG)
at = np.asarray(ActionGridSpec.from_env_config(str(ENVCFG)).build_action_table(), dtype=np.float64)
SPEEDS = np.unique(np.round(np.hypot(at[:, 0], at[:, 1]), 4))
RING = {i: int(np.argmin(np.abs(SPEEDS - round(float(np.hypot(*at[i])), 4)))) for i in range(len(at))}

# Which tensors to score matters and is NOT uniform across artifacts:
#   resume_latest.pth : model_state_dict IS the raw model
#   milestone/final   : model_state_dict is the EMA, raw lives in extra
# Always prefer the RAW weights. At warm-up end the EMA has seen 550 steps
# at decay 0.99, so it still sits most of the way back at initialisation --
# evaluating it would measure the wrong network entirely.
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
extra = ck.get("extra", {}) if isinstance(ck, dict) else {}
if isinstance(extra, dict) and extra.get("raw_model_state_dict") is not None:
    state, which = extra["raw_model_state_dict"], "extra.raw_model_state_dict"
elif "model_state_dict" in ck:
    state, which = ck["model_state_dict"], "model_state_dict (raw for a resume artifact)"
else:
    state, which = ck, "bare state_dict"
print(f"weights source: {which}")
model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V7).to(DEV)
model.load_state_dict(state); model.eval()
print(f"STAGE ACCEPT  tag={TAG}  ckpt={CKPT}  device={DEV}")
print(f"layouts={len(STAGE_ACCEPT_SEEDS)}  gate: macro SR>={MACRO_MIN}, each scenario SR>={PER_SCENARIO_MIN}")
print(f"action_quantiles={cfg.iqn_action_quantiles}  mc_quantiles={cfg.iqn_mc_quantiles}", flush=True)

per = defaultdict(lambda: defaultdict(int))
rings = defaultdict(int)
step0 = []
for sc, seed in zip(STAGE_ACCEPT_PLAN, STAGE_ACCEPT_SEEDS):
    r = collect_online_episode(
        env_config_path=ENVCFG, model=model, action_table=at, scenario=sc,
        episode_seed=int(seed), epsilon=0.0, explore_rng=np.random.default_rng(0),
        is_heldout=False, gamma=cfg.gamma, n_samples=cfg.future_n_samples,
        horizon=cfg.future_horizon, device=DEV, belief_mode="full")
    per[sc][r.outcome] += 1; per[sc]["n"] += 1
    for t in r.transitions:
        rings[RING[int(t.action_index)]] += 1
    # Same initial state, teacher vs model: ORCA is re-run on this layout, so
    # its first action is the one it would take from the identical state.
    orca = collect_raw_orca_episode(ENVCFG, sc, int(seed), gamma=cfg.gamma)
    if r.transitions and orca.steps:
        step0.append((sc, seed, RING[int(orca.steps[0].action_index)],
                      RING[int(r.transitions[0].action_index)], orca.outcome))
    print(f"  {sc:<15} {seed}  {r.outcome:<10} steps={r.steps:>3} path={r.path_length:5.2f} "
          f"clr={r.min_clearance:+.3f}  ORCA={orca.outcome}", flush=True)

print("\n" + "=" * 78 + f"\nSTAGE ACCEPTANCE -- {TAG}\n" + "=" * 78)
print(f"{'scenario':<18}{'n':>5}{'SR':>9}{'CR':>9}{'TR':>9}")
srs = []
for sc in sorted(per):
    d = per[sc]; n = d["n"]; sr = d["success"] / n
    srs.append(sr)
    print(f"{sc:<18}{n:>5}{sr:>9.3f}{d['collision']/n:>9.3f}{d['timeout']/n:>9.3f}")
macro = float(np.mean(srs))
ok = macro >= MACRO_MIN and min(srs) >= PER_SCENARIO_MIN
print(f"{'MACRO':<18}{sum(per[s]['n'] for s in per):>5}{macro:>9.3f}")

tot = sum(rings.values())
print("\nexecuted speed-ring distribution (greedy stream):")
for r in range(len(SPEEDS)):
    print(f"  {SPEEDS[r]:.3f}  {rings[r]:>6}  {100*rings[r]/max(tot,1):>5.1f}%")

hit = sum(1 for _, _, o, m, _ in step0 if o == m)
print(f"\nstep-0 speed ring: model == ORCA on {hit}/{len(step0)} layouts")
by = defaultdict(lambda: [0, 0])
for sc, _, o, m, _ in step0:
    by[sc][1] += 1; by[sc][0] += (o == m)
for sc in sorted(by):
    h, n = by[sc]
    mm = [SPEEDS[m] for s, _, o, m, _ in step0 if s == sc]
    print(f"  {sc:<15} {h}/{n}   model rings: {sorted(set(round(x,3) for x in mm))}")

orca_ok = sum(1 for _, _, _, _, o in step0 if o == "success")
print(f"\nORCA teacher on the same 90: {orca_ok}/{len(step0)} = {orca_ok/max(len(step0),1):.3f}")
print(f"\nGATE: {'PASS' if ok else 'FAIL'}  (macro {macro:.3f} vs {MACRO_MIN}, "
      f"min scenario {min(srs):.3f} vs {PER_SCENARIO_MIN})")
sys.exit(0 if ok else 3)
