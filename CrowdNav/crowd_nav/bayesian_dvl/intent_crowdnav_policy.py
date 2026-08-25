"""CrowdNav-facing adapter for the goal-intent (V6) chain -- Order C3.1.

Plan 2.2 point 17: the goal-intent chain could be trained and evaluated by
its own CLI, but could NOT be loaded like SARL/Mamba-VL -- it was not in
``policy_factory``, so Test5 and the Gazebo/ROS nodes had no way to run it.

This module closes that gap with the SAME duck-typed interface the older
``BayesianDVLPolicy`` adapter already uses (``configure``/``predict``/
``set_phase``/``set_device``/``set_env``/``reset_episode_stats`` plus the
explicit ``expects_joint_state = True`` capability flag that
``Robot.act()`` checks BEFORE falling back to class-name guessing).

Deliberately duck-typed rather than subclassing
``crowd_sim.envs.policy.policy.Policy``: the main chain must not grow a
dependency on that module, and ``Robot.act()`` only needs ``.predict()``.

Belief-free w.r.t. the retired SBK-HMM chain: this imports intent_policy /
intent_tracker / scene_candidates only, never belief.py / rollout.py /
world_model.py / trainer.py (enforced by a subprocess import test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec, FROZEN_VALUES, TRACKER_DEFAULTS
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.intent_policy import (
    HUMAN_FEATURE_DIM_V7, build_intent_human_feature_batch, load_intent_checkpoint, remaining_time_fraction,
    score_candidates_v5,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
from crowd_nav.bayesian_dvl.scene_candidates import circle_scene, make_candidate_fn, square_scene
from crowd_sim.envs.utils.action import ActionXY


class IntentPolicyAdapterError(ValueError):
    pass


class IntentBDVLPolicy:
    """Deployment adapter: observable tracks -> goal posterior -> Q(s,b,a)
    -> one of the 80 discrete actions (deterministic, fixed-midpoint tau).

    ``belief_mode`` selects the ablation arm at DEPLOYMENT time so Test5 /
    Gazebo can run the same checkpoint under full/mean/cv/uniform without a
    second policy class. It defaults to ``full`` (the real method).
    """

    expects_joint_state = True
    kinematics = "holonomic"
    multiagent_training = True
    trainable = False

    def __init__(self, model: Optional[DistributionalValueModel] = None, belief_mode: str = "full"):
        # zero-arg construction, matching policy_factory's `policy_class()`
        # pattern: the real model is built by `configure()`.
        if belief_mode not in ("full", "mean", "cv", "uniform"):
            raise IntentPolicyAdapterError(f"unknown belief_mode {belief_mode!r}")
        self.belief_mode = belief_mode
        self.model = model
        self.phase = None
        self.device = torch.device("cpu")
        self.env = None
        self.time_step = FROZEN_VALUES["dt"]
        self.last_state = None
        self.last_action_index: Optional[int] = None
        self.action_table: Optional[np.ndarray] = None
        self.scene = None
        self.bank: Optional[IntentBeliefBank] = None
        self.future_horizon = 8
        self.future_n_samples = 60
        self.n_eval_taus = 32
        self._planner_seed = 0
        self._planner_rng = np.random.default_rng(0)
        self._global_time = 0.0

    # ---------------- configuration ----------------

    def configure(self, config) -> None:
        """Reads ``[intent_bdvl]``:
            checkpoint_path = path to a V6 checkpoint (normally final_ema.pth)
            env_config_path = env config the 80-action grid comes from
            scene           = 'circle:<radius>' or 'square:<width>'
            belief_mode     = full | mean | cv | uniform   (default full)
            device          = cpu | cuda[:N]               (default cpu)
        """
        section_name = "intent_bdvl"
        if hasattr(config, "has_section") and not config.has_section(section_name):
            raise IntentPolicyAdapterError(f"policy config is missing the [{section_name}] section")
        section = config[section_name] if hasattr(config, "__getitem__") else getattr(config, section_name)

        def _get(key, fallback=None):
            if hasattr(section, "get"):
                try:
                    return section.get(key, fallback)
                except TypeError:  # configparser SectionProxy
                    return section.get(key, fallback=fallback)
            return getattr(section, key, fallback)

        env_config_path = _get("env_config_path", "crowd_nav/configs/env_bayesian_dvl.config")
        checkpoint_path = _get("checkpoint_path", None)
        self.belief_mode = str(_get("belief_mode", "full") or "full").strip()
        if self.belief_mode not in ("full", "mean", "cv", "uniform"):
            raise IntentPolicyAdapterError(f"unknown belief_mode {self.belief_mode!r}")
        self.future_horizon = int(_get("future_horizon", 8) or 8)
        self.future_n_samples = int(_get("future_n_samples", 60) or 60)
        self.n_eval_taus = int(_get("eval_taus", 32) or 32)
        self.set_scene(str(_get("scene", "circle:4.0") or "circle:4.0"))
        self.action_table = np.asarray(
            ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)

        self.model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V7)
        if not checkpoint_path:
            raise IntentPolicyAdapterError(
                "[intent_bdvl] checkpoint_path is required -- refusing to deploy random weights")
        # V6 + training-contract validation happens inside; a retired V5
        # checkpoint fails closed here rather than silently deploying.
        load_intent_checkpoint(str(checkpoint_path), self.model)
        self.model.eval()
        device = str(_get("device", "cpu") or "cpu")
        self.set_device(device)

    def set_scene(self, spec: str) -> None:
        """``'circle:<radius>'`` or ``'square:<width>'``. C1.1: a square
        environment must use square public destinations."""
        kind, _, value = str(spec).partition(":")
        kind = kind.strip().lower()
        size = float(value) if value.strip() else (4.0 if kind == "circle" else 10.0)
        if kind == "circle":
            self.scene = circle_scene(radius=size, n_sectors=8)
        elif kind == "square":
            self.scene = square_scene(width=size, n_rows=4)
        else:
            raise IntentPolicyAdapterError(f"scene must be 'circle:<r>' or 'square:<w>', got {spec!r}")
        self._reset_bank()

    def _reset_bank(self) -> None:
        if self.scene is None:
            return
        self.bank = IntentBeliefBank(make_candidate_fn(self.scene), dt=float(self.time_step), speed=TRACKER_DEFAULTS["speed_prior"])

    # ---------------- CrowdNav duck-typed hooks ----------------

    def set_phase(self, phase) -> None:
        self.phase = phase

    def set_device(self, device) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        if self.model is not None:
            self.model.to(self.device)

    def set_env(self, env) -> None:
        self.env = env

    def set_time_step(self, dt) -> None:
        self.time_step = float(dt)
        self._reset_bank()

    def reset_episode_stats(self, *, suite_seed: Optional[int] = None, episode_seed: Optional[int] = None) -> None:
        """Clear the per-episode belief state. MUST be called between
        episodes: track ids restart, and a stale bank would carry one
        episode's posterior into the next."""
        self._reset_bank()
        self._global_time = 0.0
        self.last_action_index = None
        if episode_seed is not None:
            self._planner_seed = int(episode_seed)
        self._planner_rng = np.random.default_rng(self._planner_seed)

    # alias used by some CrowdNav call sites
    reset = reset_episode_stats

    # ---------------- the decision ----------------

    def predict(self, state):
        """``state`` is a ``JointState`` (self_state + human_states), the
        format ``Robot.act()`` builds because ``expects_joint_state`` is
        True. Returns a single ``ActionXY`` from the frozen 80-action grid.
        """
        if self.model is None or self.action_table is None:
            raise IntentPolicyAdapterError("policy is not configured (call configure() first)")
        if self.bank is None:
            self._reset_bank()
        self.last_state = state

        robot = RobotObservation.from_full_state(state.self_state)
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(state.human_states)
        ]
        # identity is by track id, which here is the stable index CrowdNav
        # presents; a real deployment would feed a tracker's ids instead.
        self.bank.update({h.track_id: (h.px, h.py) for h in humans})

        global_time = float(getattr(self.env, "global_time", self._global_time))
        remaining = remaining_time_fraction(global_time, FROZEN_VALUES["time_limit"])
        human_feats, human_mask = build_intent_human_feature_batch(
            self.bank, robot, humans, mode=self.belief_mode, rng=self._planner_rng,
            horizon=self.future_horizon, n_samples=self.future_n_samples,
        )
        results = score_candidates_v5(
            self.model, robot, human_feats, human_mask, self.action_table, remaining,
            n_taus=self.n_eval_taus, device=str(self.device),
        )
        best = max(results, key=lambda r: r.q_mean)
        self.last_action_index = int(best.action_index)
        self._global_time = global_time + float(self.time_step)
        vx, vy = self.action_table[best.action_index]
        return ActionXY(float(vx), float(vy))
