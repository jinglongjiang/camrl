#!/usr/bin/env python3
"""Collect fresh human position tracks from real CrowdSim episodes for
SBK-HMM fitting (guide.md 10 S0 / B4). Controller is ORCA (or
train_nonstationary's InterventionORCA) driving BOTH robot and humans;
the world model itself never reads controller identity or behavior
labels -- only raw (x,y) positions are saved (guide.md 4.2).

Output: one JSON file, ``{"dt": ..., "tracks": [[[x,y], [x,y], ...], ...]}``,
one list-of-positions per human per episode.
"""

from __future__ import annotations

import argparse
import configparser
import json
import hashlib
import sys
from pathlib import Path

import numpy as np


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402
from crowd_nav.bayesian_pilot.protocol import BehaviorScheduler, InterventionORCA, PROFILES  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402
from crowd_nav.bayesian_dvl.config import FROZEN_VALUES  # noqa: E402


def _make_env(env_config_path: Path, human_num: int):
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not env_config.read(str(env_config_path)):
        raise ValueError(f"env config not found: {env_config_path}")
    env_config.set("sim", "human_num", str(human_num))
    # The legacy crowd_sim Agent constructor resolves policies through
    # crowd_sim.envs.policy.policy_factory, which does not know the new
    # BDVL alias. Construct with ORCA, then explicitly replace the robot
    # policy below; this keeps data collection independent of BDVL weights.
    env_config.set("robot", "policy", "orca")
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "train"
    robot = Robot(env_config, "robot")
    orca = ORCA()
    orca.configure(env_config)
    robot.set_policy(orca)
    robot.visible = True
    robot.env = env
    env.set_robot(robot)
    return env, env_config


def _initial_state_hash(env) -> str:
    state = {
        "robot": [float(env.robot.px), float(env.robot.py), float(env.robot.gx), float(env.robot.gy)],
        "humans": [[float(h.px), float(h.py), float(h.vx), float(h.vy)] for h in env.humans],
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collect(env_config_path: Path, profile: str, human_num: int, suite_seeds, episodes_per_seed: int, max_steps: int | None, scenario: str = "baseline_circle", role: str = "world-train"):
    tracks = []
    episodes = []
    dt = None
    if scenario != "baseline_circle":
        raise ValueError("BDVL world collection is frozen to the 5-person baseline_circle training scenario")
    if role == "formal-test":
        raise ValueError("formal-test data cannot be used by the BDVL collector")
    effective_max_steps = max_steps if max_steps is not None else int(round(float(FROZEN_VALUES["time_limit"]) / float(FROZEN_VALUES["dt"]))) + 1
    for suite_seed in suite_seeds:
        env, env_config = _make_env(env_config_path, human_num)
        dt = env.time_step
        for episode_index in range(episodes_per_seed):
            episode_seed = int(suite_seed) * 100000 + episode_index
            # CrowdSim's reset uses case_counter as the actual layout seed;
            # passing reset(seed=...) alone is not sufficient in this codebase.
            # Use the same episode identity for simulator and metadata.
            env.case_counter["train"] = episode_seed % (2**32 - 1)
            env.reset()
            initial_state_hash = _initial_state_hash(env)

            human_policies = []
            for human in env.humans:
                intervention_policy = InterventionORCA(env_config)
                intervention_policy.time_step = env.time_step
                human.set_policy(intervention_policy)
                human_policies.append(intervention_policy)
            scheduler = BehaviorScheduler(PROFILES[profile], seed=int(suite_seed) * 1000 + episode_index)
            scheduler.reset(len(env.humans))

            positions = [[[float(h.px), float(h.py)]] for h in env.humans]
            for _ in range(effective_max_steps):
                scheduler.advance(human_policies)
                human_obs = [h.get_observable_state() for h in env.humans]
                robot_action = env.robot.act(human_obs)
                _, _, terminated, truncated, _ = env.step(robot_action)
                for i, h in enumerate(env.humans):
                    positions[i].append([float(h.px), float(h.py)])
                if terminated or truncated:
                    break
            tracks.extend(positions)
            episodes.append({
                "episode_id": f"{suite_seed}:{episode_index}",
                "suite_seed": int(suite_seed), "episode_seed": episode_seed,
                "scenario": scenario, "profile": profile, "role": role,
                "human_num": human_num, "controller": "orca",
                "initial_state_hash": initial_state_hash,
                "tracks": positions,
            })
    return dt, tracks, episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--profile", default="nominal", choices=sorted(PROFILES.keys()))
    parser.add_argument("--human-num", type=int, default=5)
    parser.add_argument("--suite-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="diagnostic override; default is the frozen CrowdSim terminal step count")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default="baseline_circle")
    parser.add_argument("--role", default="world-train")
    args = parser.parse_args()

    env_config_path = PACKAGE_ROOT / args.env_config
    if env_config_path.name != "env_bayesian_dvl.config":
        raise SystemExit("BDVL collection must use env_bayesian_dvl.config")
    if args.role not in {"world-train", "world-validation", "IL-train", "RL-train", "checkpoint-validation"}:
        raise SystemExit(f"invalid non-formal collection role: {args.role}")
    dt, tracks, episodes = collect(
        env_config_path, args.profile, args.human_num, args.suite_seeds, args.episodes_per_seed, args.max_steps,
        scenario=args.scenario, role=args.role,
    )

    output_path = PACKAGE_ROOT / args.output
    payload = {
        "schema_version": 2, "dt": dt, "profile": args.profile, "human_num": args.human_num,
        "scenario": args.scenario, "role": args.role, "tracks": tracks, "episodes": episodes,
    }
    atomic_write_json(str(output_path), payload)

    manifest = build_run_manifest(
        repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv),
        source_files=["crowd_nav/tools/collect_bdvl_data.py", "crowd_nav/bayesian_pilot/protocol.py"],
        extra={"n_tracks": len(tracks), "n_episodes": len(episodes), "output_sha256": sha256_of_file(str(output_path)),
               "role": args.role, "profile": args.profile, "scenario": args.scenario},
    )
    manifest_path = str(output_path) + ".manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(f"COLLECT_BDVL_DATA_DONE n_tracks={len(tracks)} dt={dt}")


if __name__ == "__main__":
    main()
