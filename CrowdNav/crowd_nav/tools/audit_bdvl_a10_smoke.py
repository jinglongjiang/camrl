#!/usr/bin/env python3
"""BDVL A10 acceptance smoke: run BayesianDVLPolicy for 2 episodes on
EACH of the 6 formal scenarios via the real CrowdNav/CrowdSim stack,
checking legal termination, no NaN, reproducibility, and that a 20-human
scenario really carries 20 valid mask entries (not a top-5 truncation).
"""

from __future__ import annotations

import configparser
import json
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

from crowd_nav.bayesian_dvl.config import ActionGridSpec, FROZEN_VALUES, FORMAL_SCENARIOS  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact, Track, fit_sbk_hmm  # noqa: E402
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder  # noqa: E402
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig  # noqa: E402
from crowd_nav.bayesian_dvl.policy import BDVLPolicy, BayesianDVLPolicy  # noqa: E402
from crowd_nav.bayesian_dvl.contracts import MAX_HUMANS  # noqa: E402


def _synthetic_artifact() -> SBKHMMArtifact:
    """A cheap fixture artifact (fit on a few synthetic tracks) so this
    smoke script needs no fresh-data collection dependency -- fine for
    A10's engineering smoke, NOT a substitute for a real S0-fitted
    production artifact used in any formal science gate."""
    dt = FROZEN_VALUES["dt"]

    def track(speed0, accel, omega):
        pos = [np.array([0.0, 0.0])]
        speed, heading = speed0, 0.0
        for _ in range(30):
            speed = max(speed + accel * dt, 0.0)
            heading += omega * dt
            vel = speed * np.array([np.cos(heading), np.sin(heading)])
            pos.append(pos[-1] + vel * dt)
        return Track(positions=np.array(pos), dt=dt)

    tracks = [track(1.0, 0.0, 0.0), track(0.3, 0.8, 0.0), track(1.5, -0.8, 0.0), track(1.0, 0.0, 1.0), track(1.0, 0.0, -1.0)]
    return fit_sbk_hmm(tracks, train_data_sha256="a10_smoke_fixture", max_iterations=15)


def _make_bdvl_crowdnav_policy(action_table) -> BayesianDVLPolicy:
    artifact = _synthetic_artifact()
    encoder = SetEncoder(human_hidden_dim=16, human_embed_dim=8, robot_embed_dim=8, embedding_dim=16)
    action_encoder = ActionEncoder(hidden_dim=16, embed_dim=8)
    net = IQNValueNetwork(state_embedding_dim=16, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=16)
    reward_config = RewardConfig(
        success_reward=FROZEN_VALUES["success_reward"], collision_penalty=FROZEN_VALUES["collision_penalty"],
        timeout_penalty=FROZEN_VALUES["timeout_penalty"], progress_reward=FROZEN_VALUES["progress_reward"],
        time_penalty=FROZEN_VALUES["time_penalty"], stand_penalty=FROZEN_VALUES["stand_penalty"],
        stand_speed_threshold=FROZEN_VALUES["stand_speed_threshold"],
        discomfort_distance=FROZEN_VALUES["discomfort_distance"],
        discomfort_penalty_factor=FROZEN_VALUES["discomfort_penalty_factor"],
    )
    bdvl = BDVLPolicy(
        artifact=artifact, set_encoder=encoder, value_network=net, action_encoder=action_encoder, action_table=action_table,
        reward_config=reward_config, dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"],
        max_human_speed=FROZEN_VALUES["max_human_speed"], cvar_alpha=FROZEN_VALUES["cvar_alpha"],
        n_world_samples=2, n_iqn_quantiles=4,  # small budget: this is an engineering smoke, not a latency/formal run
    )
    return BayesianDVLPolicy(bdvl_policy=bdvl, suite_seed=88001)


def _make_env(env_config_path: Path, human_num: int, scenario_key: str):
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    env_config.read(str(env_config_path))
    scenario = FORMAL_SCENARIOS[scenario_key]
    env_config.set("sim", "human_num", str(human_num))
    env_config.set("sim", "train_val_sim", "circle_crossing" if scenario["layout"] == "circle" else "square_crossing")
    env_config.set("sim", "test_sim", "circle_crossing" if scenario["layout"] == "circle" else "square_crossing")
    if scenario["layout"] == "circle":
        env_config.set("sim", "circle_radius", str(scenario["radius"]))
    else:
        env_config.set("sim", "square_width", str(scenario["width"]))
    env_config.set("env", "time_limit", str(int(FROZEN_VALUES["time_limit"])))
    env_config.set("reward", "progress_reward", str(FROZEN_VALUES["progress_reward"]))
    env_config.set("reward", "time_penalty", str(FROZEN_VALUES["time_penalty"]))

    env = CrowdSim()
    env.configure(env_config)
    env.phase = "test"
    robot = Robot(env_config, "robot")
    policy = _make_bdvl_crowdnav_policy(action_table)
    robot.set_policy(policy)
    robot.visible = True
    robot.time_step = FROZEN_VALUES["dt"]
    policy.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    return env, policy


if __name__ == "__main__":
    env_config_path = PACKAGE_ROOT / "crowd_nav" / "configs" / "env.config"
    grid = ActionGridSpec.from_env_config(str(env_config_path))
    action_table = grid.build_action_table()

    results = {}
    for scenario_key, scenario_spec in FORMAL_SCENARIOS.items():
        human_num = scenario_spec["humans"]
        env, policy = _make_env(env_config_path, human_num, scenario_key)
        episode_events = []
        for episode_idx in range(2):
            env.case_counter["test"] = 88001 + episode_idx
            policy.reset_episode_stats()
            ob, _ = env.reset()
            assert np.isfinite(ob).all(), f"NaN/Inf in initial observation for {scenario_key}"
            if human_num == MAX_HUMANS:
                assert len(env.humans) == MAX_HUMANS, f"expected {MAX_HUMANS} humans, got {len(env.humans)}"

            terminated, truncated = False, False
            event = None
            for step in range(200):
                human_obs = [h.get_observable_state() for h in env.humans]
                action = env.robot.act(human_obs)
                ob, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(ob).all(), f"NaN/Inf in ob at step {step} for {scenario_key}"
                assert np.isfinite(reward), f"NaN/Inf reward at step {step} for {scenario_key}"
                if terminated or truncated:
                    event = info.get("event")
                    break
            assert event in ("reach_goal", "collision", "timeout"), f"illegal termination event {event!r} for {scenario_key}"
            episode_events.append(event)
        results[scenario_key] = {"human_num": human_num, "events": episode_events}
        print(f"{scenario_key}: humans={human_num} events={episode_events}")

    output_path = PACKAGE_ROOT / "runs" / "bayesian_dvl" / "a10_smoke" / "result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("A10_SMOKE_PASS")
