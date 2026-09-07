"""Pre-built candidate sets for the local ranking loss.

Why precompute. The ranking term needs, for one expert state, all 80 successors
scored through the deployment path -- and the expensive part of that (one
posterior-future build, ~80 ms in the `full` arm) cannot be reused across
optimizer updates because each update wants a different state. Building it
inside the training loop would put ~80 ms of CPU between every pair of GPU
steps. Building it once, ahead of time, costs the same total but happens in one
place and is measurable.

Why not simply keep 256177 x 80: storing every training state's candidate set
would be ~2.7 GB of features on a box with 4 GB free. Only the states the
ranking term will actually visit are built -- one per optimizer update.

The candidate sets come from `build_candidate_inputs`, the same function
deployment calls. There is no second implementation to drift.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from crowd_nav.bayesian_dvl.intent_runtime_config import (
    ActionGridSpec, FROZEN_VALUES, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.intent_train import _ScenarioEpisode
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn
from crowd_nav.bayesian_dvl.v0.scoring import build_candidate_inputs


def build_rank_pool(cfg, corpus_path: Path, env_cfg: Path, mode: str,
                    n_states: int, seed: int, stride: int = 3) -> dict:
    """Sample expert decision points and freeze their 80-candidate inputs.

    States are taken by replaying corpus episodes in order, because the belief
    bank is recursive: the posterior at step t is a function of every earlier
    observation, so a state cannot be reconstructed in isolation. Only the
    sampled steps pay for a feature build; the rest only advance the bank.

    Rows with an empty expert set are skipped. Under the failed-demo rule those
    are exactly the steps from ORCA episodes that ended badly, and a ranking
    label there would be teaching the move that caused the collision.
    """
    at = np.asarray(ActionGridSpec.from_env_config(str(env_cfg)).build_action_table(), dtype=np.float64)
    payload = torch.load(str(corpus_path), map_location="cpu", weights_only=False)
    eps = payload["episodes"]
    order = np.random.default_rng(seed).permutation(len(eps))

    rows, masks, rfeats, rewards, terms, experts = [], [], [], [], [], []
    t0 = time.time()
    for ei in order:
        raw = eps[int(ei)]
        ep = _ScenarioEpisode(env_cfg, str(raw.scenario), int(raw.episode_seed),
                              is_heldout=bool(raw.is_heldout))
        bank = IntentBeliefBank(make_candidate_fn(ep.scene), dt=FROZEN_VALUES["dt"],
                                speed=TRACKER_DEFAULTS["speed_prior"])
        rng = np.random.default_rng(int(raw.episode_seed))
        for k, st in enumerate(raw.steps):
            bank.update({h.track_id: (h.px, h.py) for h in st.humans})
            if k % stride or not st.expert_action_indices:
                continue
            ci = build_candidate_inputs(
                bank, st.robot, st.humans, at, st.remaining_fraction, mode=mode,
                rng=rng, n_samples=cfg.future_n_samples,
                feature_horizon=cfg.future_horizon)
            m = np.zeros(len(at), dtype=bool)
            m[list(st.expert_action_indices)] = True
            if m.all():
                continue                      # no negatives to rank against
            rows.append(ci.rows); masks.append(ci.mask); rfeats.append(ci.robot_feats)
            rewards.append(ci.rewards); terms.append(ci.terminal); experts.append(m)
            if len(rows) >= n_states:
                break
        if len(rows) >= n_states:
            break
        if len(rows) and len(rows) % 200 == 0:
            print(f"  rank pool {len(rows)}/{n_states}, {time.time()-t0:.0f}s", flush=True)
    print(f"rank pool: {len(rows)} states in {time.time()-t0:.0f}s", flush=True)
    return {
        "rows": np.stack(rows).astype(np.float32),
        "mask": np.stack(masks),
        "robot_feats": np.stack(rfeats).astype(np.float32),
        "rewards": np.stack(rewards).astype(np.float32),
        "terminal": np.stack(terms),
        "expert_mask": np.stack(experts),
        "mode": mode, "n_states": len(rows),
        "future_horizon": int(cfg.future_horizon),
        "future_n_samples": int(cfg.future_n_samples),
    }
