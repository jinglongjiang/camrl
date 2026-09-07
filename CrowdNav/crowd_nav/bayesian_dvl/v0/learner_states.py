"""Learner-state collection and the pre-training go/no-go.

The student drives its own greedy policy for the whole episode; ORCA is queried
at every visited state but never takes control. That is the entire point: the
states must be the ones the student actually reaches, and letting the teacher
steer would replace them with the expert manifold that is already covered.

Why this is the remaining gap. The value head is fitted by MC regression on
states ORCA visited, and at deployment it is asked to rank 80 counterfactual
successors of states the STUDENT reached. Adding a ranking term on expert
states was measured to be inert -- the loss sat at 1e-8..1e-5 because ORCA's
equivalence class already ranked first there. Nothing has ever supervised the
ordering where the policy actually operates.

Storage. A state's full candidate tensor is 80 x MAX_HUMANS x 70 floats, 448 KB;
10000 of them will not fit beside the corpus on this disk. What is stored is the
compact half -- one belief-conditioned base row block, the 80 successor robot
states, the predicted crowd -- and the 80 candidate rows are rebuilt at training
time by calling patch_robot_columns, the same function deployment calls. The
rebuild is checked byte-for-byte against build_candidate_inputs before use.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.intent_policy import remaining_time_fraction
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    ActionGridSpec, FROZEN_VALUES, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.intent_train import _ScenarioEpisode
from crowd_nav.bayesian_dvl.ranking import (
    build_action_equivalence_class, derive_action_equivalence_tolerance,
)
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn
from crowd_nav.bayesian_dvl.v0 import lookahead as LA
from crowd_nav.bayesian_dvl.v0.scoring import (
    CandidateInputs, build_candidate_inputs, score_from_inputs,
)
from crowd_sim.envs.utils.action import ActionXY

# Training-distribution layouts, deliberately disjoint from the 90 STAGE_ACCEPT
# gate seeds (3_0xx_xxx). Reusing gate layouts as training data would make the
# gate score states it had been fitted on.
COLLECT_BLOCKS = {"circle": 2_600_000, "square": 2_610_000, "junction_crowd": 2_100_000}


def _pack(ci: CandidateInputs, rp_list, pred, expert_mask) -> dict:
    return dict(
        base_rows=ci.rows[0].copy(),            # any candidate: non-robot columns are shared
        mask=ci.mask.copy(),
        robot_feats=ci.robot_feats.copy(),
        rewards=ci.rewards.astype(np.float32),
        terminal=ci.terminal.copy(),
        expert_mask=expert_mask,
        succ=np.array([[r.px, r.py, r.vx, r.vy, r.radius, r.gx, r.gy, r.v_pref, r.theta]
                       for r in rp_list], dtype=np.float32),
        pred=np.array([[h.track_id, h.px, h.py, h.vx, h.vy, h.radius] for h in pred],
                      dtype=np.float32),
    )


def rebuild_rows(rec) -> np.ndarray:
    """Rebuild the 80 candidate rows from the compact record.

    Calls patch_robot_columns -- the deployment function -- so there is no second
    implementation of what a candidate row contains.
    """
    base = np.asarray(rec["base_rows"]); mask = np.asarray(rec["mask"])
    pred = [HumanObservation(int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                             float(r[4]), float(r[5])) for r in np.asarray(rec["pred"])]
    out = np.empty((len(rec["succ"]),) + base.shape, dtype=np.float32)
    for i, s in enumerate(np.asarray(rec["succ"])):
        rp = RobotObservation(px=float(s[0]), py=float(s[1]), vx=float(s[2]), vy=float(s[3]),
                              radius=float(s[4]), gx=float(s[5]), gy=float(s[6]),
                              v_pref=float(s[7]), theta=float(s[8]))
        out[i] = LA.patch_robot_columns(base, mask, rp, pred)
    return out


def collect(cfg, env_cfg: Path, models: dict, n_target: int, mode: str,
            device: str, max_steps: int, seed0: int = 0) -> List[dict]:
    """Roll the student out; label every visited state with ORCA."""
    at = np.asarray(ActionGridSpec.from_env_config(str(env_cfg)).build_action_table(), dtype=np.float64)
    tol = derive_action_equivalence_tolerance(at)
    names = list(models)
    out: List[dict] = []
    t0 = time.time()
    ep_i = 0
    while len(out) < n_target:
        name = names[ep_i % len(names)]
        sc = list(COLLECT_BLOCKS)[(ep_i // len(names)) % len(COLLECT_BLOCKS)]
        seed = COLLECT_BLOCKS[sc] + seed0 + (ep_i // (len(names) * len(COLLECT_BLOCKS)))
        ep_i += 1
        ep = _ScenarioEpisode(env_cfg, sc, int(seed), is_heldout=False)
        env = ep.env
        bank = IntentBeliefBank(make_candidate_fn(ep.scene), dt=FROZEN_VALUES["dt"],
                                speed=TRACKER_DEFAULTS["speed_prior"])
        rng = np.random.default_rng(int(seed))
        for _ in range(max_steps):
            ep.advance_hidden_state()
            hs = [HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
                  for i, h in enumerate(env.humans)]
            bank.update({h.track_id: (h.px, h.py) for h in hs})
            ro = RobotObservation.from_full_state(env.robot.get_full_state())
            rem = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
            ci = build_candidate_inputs(bank, ro, hs, at, rem, mode=mode, rng=rng,
                                        n_samples=cfg.future_n_samples,
                                        feature_horizon=cfg.future_horizon)
            with torch.no_grad():
                s, _, _ = score_from_inputs(models[name], ci, cfg.gamma, device)
            k = int(torch.argmax(s))

            o = env.robot.act([h.get_observable_state() for h in env.humans])
            em = np.zeros(len(at), dtype=bool)
            em[list(build_action_equivalence_class(o.vx, o.vy, at, tol))] = True
            if not em.all():                      # need negatives to rank against
                pred = LA.predict_humans(bank, hs, mode, np.random.default_rng(int(seed)),
                                         cfg.future_n_samples)
                rps = [LA.robot_successor(ro, float(a[0]), float(a[1])) for a in at]
                rec = _pack(ci, rps, pred, em)
                rec.update(scenario=sc, seed=int(seed), checkpoint=name,
                           greedy=k, sample_seed=int(seed))
                out.append(rec)
            _, _, term, trunc, _ = env.step(ActionXY(float(at[k][0]), float(at[k][1])))
            if term or trunc or len(out) >= n_target:
                break
        if ep_i % 20 == 0:
            print(f"  {len(out)}/{n_target} learner states, {ep_i} episodes, "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"collected {len(out)} learner states from {ep_i} episodes in {time.time()-t0:.0f}s")
    return out
