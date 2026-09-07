#!/usr/bin/env python3
"""BDVL A2 acceptance: transition.py must agree with the real
``CrowdSim.onestep_lookahead`` on >=10,000 random (state, action) pairs
(guide.md A2 acceptance criterion 1).

For each of many real episode snapshots (collected by rolling out ORCA
robot + ORCA humans so states are physically plausible, not just
uniformly random), this compares transition.step(...) against
env.onestep_lookahead(action) for many random candidate actions per
snapshot, checking: next robot position, event, reward, terminated,
truncated, and dmin, all to floating-point tolerance.
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
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402

from crowd_nav.bayesian_dvl.contracts import HumanObservation, canonicalize  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig, step as bdvl_step  # noqa: E402
from crowd_nav.bayesian_dvl.config import FROZEN_VALUES  # noqa: E402


REWARD_CONFIG = RewardConfig(
    success_reward=FROZEN_VALUES["success_reward"],
    collision_penalty=FROZEN_VALUES["collision_penalty"],
    timeout_penalty=FROZEN_VALUES["timeout_penalty"],
    progress_reward=FROZEN_VALUES["progress_reward"],
    time_penalty=FROZEN_VALUES["time_penalty"],
    stand_penalty=FROZEN_VALUES["stand_penalty"],
    stand_speed_threshold=FROZEN_VALUES["stand_speed_threshold"],
    discomfort_distance=FROZEN_VALUES["discomfort_distance"],
    discomfort_penalty_factor=FROZEN_VALUES["discomfort_penalty_factor"],
)


def _make_env(env_config_path: Path):
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    env_config.read(str(env_config_path))
    # A2 equivalence must use THIS project's actual frozen reward
    # values (0.01/-0.003/0.0), not the stale sparse 0.0/0.0 that
    # plain env.config's own [reward] section still has -- see
    # config.py FROZEN_VALUES docstring / guide.md cross-check.
    env_config.set("reward", "progress_reward", str(FROZEN_VALUES["progress_reward"]))
    env_config.set("reward", "time_penalty", str(FROZEN_VALUES["time_penalty"]))
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "train"
    robot = Robot(env_config, "robot")
    orca_policy = ORCA()
    orca_policy.configure(env_config)
    robot.set_policy(orca_policy)
    robot.visible = True
    robot.env = env
    env.set_robot(robot)
    return env


def _random_candidate_action(rng: np.random.Generator, v_pref: float):
    speed = rng.uniform(0.0, v_pref)
    angle = rng.uniform(0.0, 2 * np.pi)
    return float(speed * np.cos(angle)), float(speed * np.sin(angle))


def _compare_one(env, action_vx: float, action_vy: float, dt: float, time_limit: float) -> dict:
    robot_full_state = env.robot.get_full_state()
    human_obs_states = [h.get_observable_state() for h in env.humans]
    track_ids = list(range(len(env.humans)))
    canonical = canonicalize(robot_full_state, human_obs_states, track_ids)

    # The real env computes human_actions the same way regardless of
    # which candidate robot action is being dry-run evaluated (each
    # human's ORCA decision uses the CURRENT robot state, not the
    # hypothetical action) -- replicate that exactly here.
    human_actions_xy = []
    for human in env.humans:
        ob = [other.get_observable_state() for other in env.humans if other is not human]
        if env.robot.visible:
            ob = ob + [env.robot.get_observable_state()]
        ha = human.act(ob)
        human_actions_xy.append((ha.vx, ha.vy))

    humans = [
        HumanObservation(
            track_id=canonical.human_track_ids[i],
            px=float(canonical.human_features[i, 0]), py=float(canonical.human_features[i, 1]),
            vx=float(canonical.human_features[i, 2]), vy=float(canonical.human_features[i, 3]),
            radius=float(canonical.human_features[i, 4]),
        )
        for i in range(canonical.n_humans)
    ]
    bdvl_result = bdvl_step(
        robot=canonical.robot,
        humans=humans,
        action_vx=action_vx, action_vy=action_vy,
        human_actions=human_actions_xy,
        dt=dt, time_limit=time_limit, global_time=env.global_time,
        reward_config=REWARD_CONFIG,
    )

    real_action = ActionXY(action_vx, action_vy)
    real_ob, real_reward, real_terminated, real_truncated, real_info = env.onestep_lookahead(real_action)
    real_next_robot_px, real_next_robot_py = float(real_ob[0]), float(real_ob[1])

    # crowd_sim.py's onestep_lookahead casts its returned `ob` array to
    # float32 (`.astype(np.float32)`) before returning it; bdvl_result
    # is plain Python float64. A 1e-9 tolerance on the *position*
    # comparison was failing on essentially every sample purely from
    # float32 rounding, not a real transition bug (verified by hand:
    # both sides compute the identical 0.075/-3.975 in float64). Reward
    # is NOT float32-cast in the real env (`float(reward)`), so it
    # keeps the tight tolerance.
    POSITION_TOLERANCE = 1e-5
    ok = True
    reasons = []
    if abs(bdvl_result.next_robot.px - real_next_robot_px) > POSITION_TOLERANCE or abs(bdvl_result.next_robot.py - real_next_robot_py) > POSITION_TOLERANCE:
        ok = False
        reasons.append(
            f"next_robot_pos bdvl=({bdvl_result.next_robot.px},{bdvl_result.next_robot.py}) "
            f"real=({real_next_robot_px},{real_next_robot_py})"
        )
    if abs(bdvl_result.reward - real_reward) > 1e-9:
        ok = False
        reasons.append(f"reward bdvl={bdvl_result.reward} real={real_reward}")
    if bdvl_result.terminated != real_terminated or bdvl_result.truncated != real_truncated:
        ok = False
        reasons.append("terminated/truncated")
    if bdvl_result.event != real_info.get("event"):
        ok = False
        reasons.append(f"event bdvl={bdvl_result.event} real={real_info.get('event')}")
    real_dmin = real_info.get("dmin")
    if real_dmin is not None and abs(bdvl_result.dmin - real_dmin) > 1e-6:
        ok = False
        reasons.append(f"dmin bdvl={bdvl_result.dmin} real={real_dmin}")

    return {"ok": ok, "reasons": reasons}


def run(env_config_relpath: str, n_states: int, actions_per_state: int, seed_start: int):
    env_config_path = PACKAGE_ROOT / env_config_relpath
    env = _make_env(env_config_path)
    dt = env.time_step
    time_limit = env.time_limit

    rng = np.random.default_rng(0)
    total_checks = 0
    mismatches = []
    states_visited = 0
    seed_offset = seed_start

    while total_checks < n_states:
        env.case_counter["train"] = seed_offset
        seed_offset += 1
        env.reset()
        for _ in range(140):
            states_visited += 1
            for _ in range(actions_per_state):
                action_vx, action_vy = _random_candidate_action(rng, env.robot.v_pref)
                result = _compare_one(env, action_vx, action_vy, dt, time_limit)
                total_checks += 1
                if not result["ok"]:
                    mismatches.append(result["reasons"])
                if total_checks >= n_states:
                    break
            if total_checks >= n_states:
                break
            ob = [h.get_observable_state() for h in env.humans]
            robot_action = env.robot.act(ob)
            _, _, terminated, truncated, _ = env.step(robot_action)
            if terminated or truncated:
                break

    return {
        "n_states_checks_requested": n_states,
        "total_checks_run": total_checks,
        "states_visited": states_visited,
        "mismatches": len(mismatches),
        "sample_mismatch_reasons": mismatches[:10],
        "status": "PASS" if len(mismatches) == 0 else "FAIL",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="crowd_nav/configs/env.config")
    parser.add_argument("--n-states", type=int, default=10000)
    parser.add_argument("--actions-per-state", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=61001)
    parser.add_argument("--output", default="runs/bayesian_dvl/a2_transition_equivalence/result.json")
    args = parser.parse_args()

    result = run(args.env_config, args.n_states, args.actions_per_state, args.seed_start)
    output_path = PACKAGE_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
