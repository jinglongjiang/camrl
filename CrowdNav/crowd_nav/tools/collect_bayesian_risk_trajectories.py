#!/usr/bin/env python3
"""Collect mixed-outcome trajectories for Bayesian distributional value training."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import random
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.contracts import (
    GRID,
    action_to_discrete_index,
    discrete_index_to_action,
    init_grid_from_cfg,
)
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState


SCENARIOS = [
    {"name": "baseline_circle", "sim": "circle_crossing", "humans": 5, "circle_radius": 4.0},
    {"name": "baseline_square", "sim": "square_crossing", "humans": 10, "square_width": 10.0},
    {"name": "dense_circle", "sim": "circle_crossing", "humans": 10, "circle_radius": 4.0},
    {"name": "dense_square", "sim": "square_crossing", "humans": 20, "square_width": 10.0},
    {"name": "large_circle", "sim": "circle_crossing", "humans": 12, "circle_radius": 6.0},
    {"name": "large_square", "sim": "square_crossing", "humans": 20, "square_width": 14.0},
]


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    state = checkpoint.get(
        "policy_state",
        checkpoint.get("model_state_dict", checkpoint.get("value", checkpoint)),
    )
    return {
        (key.replace("_orig_mod.", "", 1) if key.startswith("_orig_mod.") else key): value
        for key, value in state.items()
    }


def merge_config(args, scenario):
    env_config = configparser.RawConfigParser()
    policy_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    policy_config.read(args.policy_config)

    if not policy_config.has_section("buffer"):
        policy_config.add_section("buffer")
    policy_config.set(
        "buffer",
        "seq_len",
        policy_config.get("temporal", "T", fallback="24"),
    )
    if not policy_config.has_section("robot"):
        policy_config.add_section("robot")
    policy_config.set("robot", "v_pref", "1.0")

    if not env_config.has_section("env"):
        env_config.add_section("env")
    env_config.set("env", "time_step", str(args.time_step))
    env_config.set("env", "time_limit", str(int(round(args.time_limit))))
    if not env_config.has_section("robot"):
        env_config.add_section("robot")
    env_config.set("robot", "v_pref", "1.0")
    if not env_config.has_section("sim"):
        env_config.add_section("sim")
    env_config.set("sim", "test_sim", scenario["sim"])
    env_config.set("sim", "human_num", str(scenario["humans"]))
    for key in ("circle_radius", "square_width"):
        if key in scenario:
            env_config.set("sim", key, str(scenario[key]))

    for section in env_config.sections():
        if not policy_config.has_section(section):
            policy_config.add_section(section)
        for key, value in env_config.items(section):
            if not policy_config.has_option(section, key):
                policy_config.set(section, key, value)
    if not policy_config.has_section("train"):
        policy_config.add_section("train")
    if not policy_config.has_option("train", "gamma"):
        policy_config.set("train", "gamma", "0.99")
    if not policy_config.has_section("sarl"):
        policy_config.add_section("sarl")
    policy_config.set("sarl", "epsilon_start", "0.0")

    for section in policy_config.sections():
        if not env_config.has_section(section):
            env_config.add_section(section)
        for key, value in policy_config.items(section):
            env_config.set(section, key, value)
    init_grid_from_cfg(policy_config)
    return env_config, policy_config


def outcome_from_info(info) -> str:
    if isinstance(info, dict):
        event = str(info.get("event", "")).lower()
    elif info is None:
        event = ""
    else:
        event = info.__class__.__name__.lower()
    if "reachgoal" in event or "reach_goal" in event or "success" in event:
        return "success"
    if "collision" in event:
        return "collision"
    return "timeout"


def perturb_action(action, probability: float, rng: np.random.Generator):
    if probability <= 0.0 or rng.random() >= probability:
        return action, False
    base_index = action_to_discrete_index(float(action.vx), float(action.vy), grid=GRID)
    include_stop = bool(GRID.get("include_stop", False))
    offset = 1 if include_stop else 0
    if include_stop and base_index == 0:
        base_index = offset
    flat = base_index - offset
    speed_idx = flat % int(GRID["n_speeds"])
    heading_idx = flat // int(GRID["n_speeds"])
    heading_idx = (
        heading_idx + int(rng.choice([-2, -1, 1, 2]))
    ) % int(GRID["n_headings"])
    if rng.random() < 0.35:
        speed_idx = int(np.clip(
            speed_idx + int(rng.choice([-1, 1])),
            0,
            int(GRID["n_speeds"]) - 1,
        ))
    candidate_idx = offset + heading_idx * int(GRID["n_speeds"]) + speed_idx
    vx, vy = discrete_index_to_action(candidate_idx)
    return ActionXY(float(vx), float(vy)), True


def min_clearance(robot, humans) -> float:
    return min(
        (
            np.hypot(robot.px - human.px, robot.py - human.py)
            - robot.radius
            - human.radius
            for human in humans
        ),
        default=float("inf"),
    )


def collect_episode(
    env,
    robot,
    policy,
    episode_seed: int,
    exploration: float,
    rng: np.random.Generator,
) -> Dict:
    if hasattr(policy, "reset_episode_stats"):
        policy.reset_episode_stats()
    env.reset(seed=episode_seed)
    states: List[np.ndarray] = []
    actions: List[List[float]] = []
    rewards: List[float] = []
    clearances: List[float] = []
    perturbations = 0
    done = False
    info = None
    max_steps = int(np.ceil(env.time_limit / env.time_step)) + 1

    while not done and len(states) < max_steps:
        state = JointState(
            robot.get_full_state(),
            [human.get_observable_state() for human in env.humans],
        )
        states.append(
            policy._build_joint_state_34(state.self_state, state.human_states)
            .astype(np.float32)
        )
        action = policy.predict(state)
        action, changed = perturb_action(action, exploration, rng)
        perturbations += int(changed)
        actions.append([float(action.vx), float(action.vy)])
        result = env.step(action)
        if len(result) == 5:
            _, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            _, reward, done, info = result
        rewards.append(float(reward))
        clearances.append(float(min_clearance(robot, env.humans)))

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "clearances": np.asarray(clearances, dtype=np.float32),
        "outcome": outcome_from_info(info),
        "episode_seed": int(episode_seed),
        "exploration": float(exploration),
        "perturbations": int(perturbations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/mamba_vl/rl_model_ep10000_T24.pth")
    parser.add_argument("--env_config", default="configs/env.config")
    parser.add_argument("--policy_config", default="configs/policy.config")
    parser.add_argument("--episodes_per_profile", type=int, default=100)
    parser.add_argument("--exploration", default="0.0,0.08,0.18")
    parser.add_argument("--time_limit", type=float, default=25.0)
    parser.add_argument("--time_step", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--output",
        default="runs/bayesian_distributional/risk_trajectories.pt",
    )
    args = parser.parse_args()

    exploration_profiles = [
        float(value.strip())
        for value in args.exploration.split(",")
        if value.strip()
    ]
    if not exploration_profiles:
        raise ValueError("At least one exploration profile is required")
    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoint = torch_load(args.source)
    source_state = extract_state_dict(checkpoint)

    trajectories = []
    outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
    total = len(SCENARIOS) * len(exploration_profiles) * args.episodes_per_profile
    progress = tqdm(total=total, desc="risk-data", unit="ep")
    for scenario_idx, scenario in enumerate(SCENARIOS):
        env_config, policy_config = merge_config(args, scenario)
        policy = MambaRLPolicy(config=policy_config, device=device)
        policy.load_state_dict(source_state)
        policy.use_sarl_predict = True
        policy.set_phase("test")
        policy.eval()

        env = CrowdSim()
        env.configure(env_config)
        env.phase = "test"
        robot = Robot(env_config, "robot")
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)
        policy.set_env_dt(args.time_step)

        for profile_idx, exploration in enumerate(exploration_profiles):
            for episode_idx in range(args.episodes_per_profile):
                episode_seed = (
                    args.seed
                    + scenario_idx * 1_000_003
                    + profile_idx * 100_003
                    + episode_idx
                ) % (2**31 - 1)
                trajectory = collect_episode(
                    env,
                    robot,
                    policy,
                    episode_seed=episode_seed,
                    exploration=exploration,
                    rng=rng,
                )
                trajectory["scenario"] = scenario["name"]
                trajectories.append(trajectory)
                outcome_counts[trajectory["outcome"]] += 1
                progress.update(1)
                progress.set_postfix(outcome_counts)
    progress.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trajectories": trajectories,
            "meta": {
                "source_checkpoint": str(Path(args.source).resolve()),
                "seed": args.seed,
                "episodes_per_profile": args.episodes_per_profile,
                "exploration_profiles": exploration_profiles,
                "outcomes": outcome_counts,
                "time_limit": args.time_limit,
                "time_step": args.time_step,
            },
        },
        output_path,
    )
    print(f"[SAVE] {output_path}")
    print(f"[OUTCOMES] {outcome_counts}")


if __name__ == "__main__":
    main()
