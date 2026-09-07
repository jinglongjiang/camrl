"""Shared protocol definitions for the Bayesian decision gate.

This experiment asks one question only: does the fitted GDBN belief provide
genuine *decision-level* advantage over constant-velocity (CV) reasoning --
i.e. can it rank candidate robot actions by true counterfactual risk better
than CV -- not just a better one-step predictive distribution. See
``README.md`` for the full pre-registered protocol.

Reuses ``bayesian_pilot/protocol.py``'s ``PROFILES``/``BehaviorScheduler``/
``InterventionORCA`` machinery verbatim (do not fork it -- it is already
calibrated and tested). This module adds only what that file does not have:
variable pedestrian-count environments (5 and 20 person densities) and the
seed/split bookkeeping specific to this gate.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Tuple

import numpy as np

from crowd_nav.bayesian_pilot.protocol import (  # noqa: F401
    BehaviorProfile,
    BehaviorScheduler,
    EVENT_MODES,
    InterventionORCA,
    MODE_NAMES,
    MODE_TO_ID,
    PROFILES,
)
from crowd_nav.contracts import init_grid_from_cfg
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.robot import Robot

THIS_DIR = Path(__file__).resolve().parent
# `frozen_k3` (the actual Route-A model under test) and the whole belief_mdp/
# production policy use configs/env_belief_mdp.config's [policy] grid (80
# actions: 5 speeds x 16 headings, v_min=0.05, include_stop=false) -- see
# belief_mdp/train.py's --env_config default. A previous version of this
# file pointed at configs/env_gdbn.config instead, which has include_stop
# = true (81 actions) and a different v_min -- silently comparing a
# DIFFERENT action set than the one "the real policy considers", exactly the
# kind of bug this experiment exists to avoid.
DEFAULT_ENV_CONFIG = str(THIS_DIR.parent / "configs" / "env_belief_mdp.config")


def lock_action_grid(env_config_path: str = None) -> dict:
    """Deterministically set ``contracts.GRID`` from this experiment's own
    env config, regardless of the caller's cwd. ``contracts.py``'s own
    auto-discovery only ever checks a fixed list of relative paths (never
    env_belief_mdp.config), so without this call GRID silently depends on
    whatever ``configs/env.config`` happens to be reachable from cwd --
    which may not match the 80-action grid the frozen K3 model and the
    production policy actually use."""
    config = configparser.RawConfigParser()
    path = env_config_path or DEFAULT_ENV_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        config.read_file(handle)
    return init_grid_from_cfg(config)

# Data splits and the profile each one draws from. `validation_nonstationary`
# uses the SAME profile as `train_nonstationary` (only a disjoint seed block)
# -- it exists to select K and hyperparameters without touching the final
# test distribution, not to probe a different stress level.
SPLIT_PROFILE = {
    "nominal": "nominal",
    "train_nonstationary": "train_nonstationary",
    "validation_nonstationary": "train_nonstationary",
    "heldout_nonstationary": "heldout_nonstationary",
}
SPLITS = tuple(SPLIT_PROFILE.keys())

# Pedestrian-count densities under test. Both use circle_crossing (matching
# bayesian_pilot's convention); square_crossing is alternated in inside
# collect_dataset.py exactly as bayesian_pilot/collect_nonstationary.py does.
DENSITIES = {
    "5person": 5,
    "20person": 20,
}

# Disjoint seed bases -- see belief_mdp/runtime.py and stage1_gate_audit.py
# for the project's other reserved seed bases (train: 300,000+, quick-eval:
# 810000/910000, select_checkpoint validation: 2,000,000, evaluate.py formal
# test: 5,000,000, stage1_gate_audit: 3,000,000). This experiment is entirely
# separate machinery, so it gets its own fresh block starting at 9,000,000.
SPLIT_SEED_BASE = {
    "train_nonstationary": 9_000_000,
    "validation_nonstationary": 9_100_000,
    "heldout_nonstationary": 9_200_000,
    "nominal": 9_300_000,
}
DENSITY_SEED_OFFSET = {
    "5person": 0,
    "20person": 50_000,
}


def episode_seed(split: str, density: str, seed: int, index: int) -> int:
    base = SPLIT_SEED_BASE[split] + DENSITY_SEED_OFFSET[density]
    return int(base + seed + index * 101)


def episode_test_case(seed_value: int, index: int) -> int:
    return int((seed_value * 13 + index) % 9000)


def load_config(env_config_path: str, human_num: int) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    with open(env_config_path, "r", encoding="utf-8") as handle:
        config.read_file(handle)
    config.set("sim", "human_num", str(int(human_num)))
    config.set("robot", "visible", "true")
    config.set("humans", "policy", "orca")
    return config


def build_environment(
    config: configparser.RawConfigParser,
    scenario: str,
) -> Tuple[CrowdSim, Robot, ORCA]:
    """Build one CrowdSim environment. ``scenario`` is 'circle_crossing' or
    'square_crossing' -- ``config``'s human_num was already set by
    ``load_config``."""
    config.set("sim", "test_sim", scenario)
    env = CrowdSim()
    env.configure(config)
    env.phase = "test"

    robot = Robot(config, "robot")
    expert = ORCA()
    expert.configure(config)
    expert.multiagent_training = True
    expert.set_phase("test")
    robot.set_policy(expert)
    robot.env = env
    env.set_robot(robot)
    return env, robot, expert


def observation_variable(robot, humans, num_humans: int) -> np.ndarray:
    """Robot(9) + up to ``num_humans`` pedestrians (5 each), identity-stable
    (simulation order, NOT TTC-sorted) -- required so that per-pedestrian
    belief trackers and mode-transition statistics stay attached to the same
    physical pedestrian across the whole episode. Width is
    ``9 + 5 * num_humans`` (NOT fixed at 34 -- only the 5-person density
    happens to match that legacy width)."""
    robot_state = robot.get_full_state().to_array().astype(np.float32)
    feats = []
    for human in humans[:num_humans]:
        state = human.get_observable_state()
        feats.extend([state.px, state.py, state.vx, state.vy, state.radius])
    expected = 5 * num_humans
    feats.extend([0.0] * max(0, expected - len(feats)))
    return np.concatenate(
        (robot_state, np.asarray(feats[:expected], dtype=np.float32))
    ).astype(np.float32)
