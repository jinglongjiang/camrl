#!/usr/bin/env python3
"""Collect disjoint train, nominal-test, and nonstationary-test datasets."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.bayesian_pilot.protocol import (  # noqa: E402
    BehaviorScheduler,
    InterventionORCA,
    PROFILES,
)
from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402


def load_config(path: Path) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    with path.open("r", encoding="utf-8") as handle:
        config.read_file(handle)
    config.set("env", "time_limit", "25")
    config.set("env", "time_step", "0.25")
    config.set("env", "test_size", "10000")
    config.set("sim", "human_num", "5")
    config.set("robot", "visible", "true")
    config.set("humans", "policy", "orca")
    return config


def build_environment(config, scenario: str) -> tuple[CrowdSim, Robot, ORCA]:
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


def observation_34(robot, humans) -> np.ndarray:
    robot_state = robot.get_full_state().to_array()
    human_states = [human.get_observable_state().to_array() for human in humans]
    if len(human_states) != 5:
        raise ValueError(f"Pilot requires exactly five pedestrians, got {len(human_states)}")
    return np.concatenate([robot_state, *human_states]).astype(np.float32)


def collect_episode(
    env: CrowdSim,
    robot: Robot,
    robot_policy: ORCA,
    config,
    profile_name: str,
    episode_seed: int,
    test_case: int,
) -> Dict:
    env.reset(seed=episode_seed, options={"test_case": int(test_case)})
    robot_policy.sim = None
    robot_policy._last_pref_vel = None

    policies = []
    for human in env.humans:
        policy = InterventionORCA(config)
        policy.time_step = env.time_step
        policy.reset()
        human.set_policy(policy)
        policies.append(policy)

    scheduler = BehaviorScheduler(PROFILES[profile_name], seed=episode_seed + 7919)
    scheduler.reset(len(policies))

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    mode_history: List[np.ndarray] = []
    done = False
    info = {}
    max_steps = int(np.ceil(env.time_limit / env.time_step)) + 1

    while not done and len(observations) < max_steps:
        observations.append(observation_34(robot, env.humans))
        mode_history.append(np.asarray(scheduler.advance(policies), dtype=np.int8))
        action = robot.act([human.get_observable_state() for human in env.humans])
        actions.append(np.asarray([action.vx, action.vy], dtype=np.float32))
        result = env.step(action)
        _, _, terminated, truncated, info = result
        done = bool(terminated or truncated)

    return {
        "obs": np.stack(observations),
        "act": np.stack(actions),
        "modes": np.stack(mode_history),
        "profile": profile_name,
        "scenario": env.test_sim,
        "seed": int(episode_seed),
        "events": dict(scheduler.counts),
        "outcome": str(info.get("event", "unknown")),
    }


def collect_split(
    *,
    config_path: Path,
    profile_choices: tuple[str, ...],
    episodes: int,
    seed: int,
    output: Path,
    description: str,
):
    records = []
    environments = {}
    progress = tqdm(
        range(int(episodes)),
        desc=description,
        unit="ep",
        dynamic_ncols=True,
    )
    for index in progress:
        scenario = "circle_crossing" if index % 2 == 0 else "square_crossing"
        if scenario not in environments:
            config = load_config(config_path)
            environments[scenario] = (*build_environment(config, scenario), config)
        env, robot, robot_policy, config = environments[scenario]
        profile = profile_choices[index % len(profile_choices)]
        record = collect_episode(
            env,
            robot,
            robot_policy,
            config,
            profile,
            episode_seed=seed + index * 101,
            test_case=(seed * 13 + index) % 9000,
        )
        records.append(record)
        event_total = sum(
            sum(item["events"].values())
            for item in records
        )
        progress.set_postfix(events=event_total, outcome=record["outcome"][0:1])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        obs=np.asarray([record["obs"] for record in records], dtype=object),
        act=np.asarray([record["act"] for record in records], dtype=object),
        modes=np.asarray([record["modes"] for record in records], dtype=object),
        profile=np.asarray([record["profile"] for record in records]),
        scenario=np.asarray([record["scenario"] for record in records]),
        seed=np.asarray([record["seed"] for record in records], dtype=np.int64),
    )
    summary = {
        "output": str(output.resolve()),
        "episodes": len(records),
        "profiles": {
            name: sum(record["profile"] == name for record in records)
            for name in sorted(set(profile_choices))
        },
        "scenarios": {
            name: sum(record["scenario"] == name for record in records)
            for name in ("circle_crossing", "square_crossing")
        },
        "events": {
            mode: sum(record["events"].get(mode, 0) for record in records)
            for mode in ("stop", "slow", "turn_left", "turn_right")
        },
        "outcomes": {
            outcome: sum(record["outcome"] == outcome for record in records)
            for outcome in ("reach_goal", "collision", "timeout")
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", default="configs/env_gdbn.config")
    parser.add_argument("--output_dir", default="runs/bayesian_belief_pilot/data")
    parser.add_argument("--train_episodes", type=int, default=300)
    parser.add_argument("--nominal_test_episodes", type=int, default=80)
    parser.add_argument("--nonstationary_test_episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2407)
    args = parser.parse_args()

    config_path = Path(args.env_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    collect_split(
        config_path=config_path,
        profile_choices=("nominal", "train_nonstationary", "train_nonstationary"),
        episodes=args.train_episodes,
        seed=args.seed,
        output=output_dir / "train.npz",
        description="pilot-train",
    )
    collect_split(
        config_path=config_path,
        profile_choices=("nominal",),
        episodes=args.nominal_test_episodes,
        seed=args.seed + 100000,
        output=output_dir / "test_nominal.npz",
        description="pilot-nominal",
    )
    collect_split(
        config_path=config_path,
        profile_choices=("heldout_nonstationary",),
        episodes=args.nonstationary_test_episodes,
        seed=args.seed + 200000,
        output=output_dir / "test_nonstationary.npz",
        description="pilot-heldout",
    )


if __name__ == "__main__":
    main()

