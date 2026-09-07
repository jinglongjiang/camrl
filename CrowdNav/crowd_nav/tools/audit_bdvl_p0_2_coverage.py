#!/usr/bin/env python3
"""BDVL P0-2: local action-coverage audit (guide.md section 9, P0-2).

For real conflict snapshots collected from ``baseline_circle`` episodes,
branch ALL 80 frozen actions from the SAME snapshot using the real
environment's own ``onestep_lookahead`` (dry-run ``step``, no state
mutation) -- this reuses CrowdSim's actual swept-segment collision
check exactly, rather than a reimplementation that could silently
diverge from it. Reports what fraction of conflict snapshots have at
least one non-collision action among the 80.

R1 fix (independent audit B7, 2026-08-06): ``train_nonstationary`` is
NOT undefined -- it is ``crowd_nav.bayesian_pilot.protocol.PROFILES``,
which this script now reuses directly (``--profile train_nonstationary``)
via the real ``InterventionORCA``/``BehaviorScheduler`` classes, not a
reimplementation. ``--profile nominal`` (the default) keeps using plain
ORCA with no interventions, matching PROFILES["nominal"]'s event_rate=0.

"Conflict" is defined here, explicitly and provisionally, as: the
current nearest human-robot clearance (dmin, center-to-center minus
both radii) is below 0.6m OR the linear-motion time-to-collision
against the nearest human is below 3.0s. This threshold is NOT in
guide.md; it is a reasonable engineering choice pending confirmation,
and is recorded in the output so it can be reviewed/overridden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np  # noqa: E402
import configparser  # noqa: E402

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402
from crowd_nav.bayesian_pilot.protocol import BehaviorScheduler, InterventionORCA, PROFILES  # noqa: E402


CONFLICT_DMIN_THRESHOLD_M = 0.6
CONFLICT_TTC_THRESHOLD_S = 3.0


def _ttc_linear(px, py, vx, vy, radius_sum):
    p = np.array([px, py], dtype=float)
    v = np.array([vx, vy], dtype=float)
    vv = v @ v
    if vv < 1e-8:
        return float("inf")
    b = 2.0 * (p @ v)
    c = (p @ p) - radius_sum * radius_sum
    disc = b * b - 4.0 * vv * c
    if disc <= 0.0:
        return float("inf")
    t1 = (-b - np.sqrt(disc)) / (2.0 * vv)
    return float(t1) if t1 > 1e-6 else float("inf")


def _nearest_human_dmin_ttc(env):
    dmin = float("inf")
    ttc_min = float("inf")
    for human in env.humans:
        px, py = human.px - env.robot.px, human.py - env.robot.py
        radius_sum = human.radius + env.robot.radius
        dist = float(np.hypot(px, py)) - radius_sum
        dmin = min(dmin, dist)
        ttc = _ttc_linear(px, py, human.vx - env.robot.vx, human.vy - env.robot.vy, radius_sum)
        ttc_min = min(ttc_min, ttc)
    return dmin, ttc_min


def _make_env(env_config_path: Path, human_num: int = None):
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    env_config.read(str(env_config_path))
    if human_num is not None:
        env_config.set("sim", "human_num", str(human_num))
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "train"
    robot = Robot(env_config, "robot")
    orca_policy = ORCA()
    orca_policy.configure(env_config)
    robot.set_policy(orca_policy)
    robot.visible = True  # humans must react to the robot to produce real conflicts
    robot.env = env
    env.set_robot(robot)
    return env


def run(env_config_relpath: str, action_table_relpath: str, seeds, episodes_per_seed: int, max_steps: int, human_num: int = None, profile: str = "nominal"):
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}, expected one of {sorted(PROFILES)}")
    env_config_path = PACKAGE_ROOT / env_config_relpath
    action_table = json.loads((PACKAGE_ROOT / action_table_relpath).read_text())
    actions = [ActionXY(vx, vy) for vx, vy in action_table["action_table"]]
    assert len(actions) == 80, len(actions)

    total_states = 0
    conflict_states = 0
    conflict_with_safe_action = 0
    per_seed_conflict_rate = {}

    for seed in seeds:
        env = _make_env(env_config_path, human_num=human_num)
        # IMPORTANT (found by direct testing, not documented anywhere):
        # CrowdSim.reset(seed=...) only reseeds self.np_random / the
        # global np.random state at the top of reset(); the actual human
        # layout is generated a few lines later via
        # np.random.seed(counter_offset[phase] + self.case_counter[phase])
        # (crowd_sim.py:267), which ignores the seed= kwarg entirely and
        # is driven solely by the internal, auto-incrementing
        # case_counter. Passing different seed= values to reset() while
        # case_counter starts fresh at 0 for every new env therefore
        # replays the IDENTICAL sequence of human layouts every time --
        # confirmed empirically (5 "different" seeds produced byte-
        # identical conflict counts before this fix). Seeding via
        # case_counter directly is the only way that actually varies
        # the generated scenarios.
        env.case_counter["train"] = int(seed) * 10000
        seed_conflicts = 0
        seed_conflicts_safe = 0
        for episode_idx in range(episodes_per_seed):
            env.reset()
            # Swap each human's policy for an InterventionORCA wrapper
            # and drive them with the real BehaviorScheduler for this
            # profile -- reuses bayesian_pilot's own RNG-driven
            # scheduling logic exactly, not a reimplementation.
            env_config_for_humans = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
            env_config_for_humans.read(str(env_config_path))
            human_policies = []
            for human in env.humans:
                intervention_policy = InterventionORCA(env_config_for_humans)
                intervention_policy.time_step = env.time_step
                human.set_policy(intervention_policy)
                human_policies.append(intervention_policy)
            scheduler = BehaviorScheduler(PROFILES[profile], seed=int(seed) * 1000 + episode_idx)
            scheduler.reset(len(env.humans))

            for step in range(max_steps):
                scheduler.advance(human_policies)
                total_states += 1
                dmin, ttc = _nearest_human_dmin_ttc(env)
                is_conflict = dmin < CONFLICT_DMIN_THRESHOLD_M or ttc < CONFLICT_TTC_THRESHOLD_S
                if is_conflict:
                    conflict_states += 1
                    seed_conflicts += 1
                    has_safe_action = False
                    for action in actions:
                        _, _, terminated, _, info = env.onestep_lookahead(action)
                        if not (terminated and info.get("event") == "collision"):
                            has_safe_action = True
                            break
                    if has_safe_action:
                        conflict_with_safe_action += 1
                        seed_conflicts_safe += 1

                # advance the real episode with the robot's own ORCA action
                ob = [h.get_observable_state() for h in env.humans]
                robot_action = env.robot.act(ob)
                _, _, terminated, truncated, _ = env.step(robot_action)
                if terminated or truncated:
                    break
        per_seed_conflict_rate[int(seed)] = {
            "conflict_states": seed_conflicts,
            "conflict_with_safe_action": seed_conflicts_safe,
        }

    coverage_rate = (conflict_with_safe_action / conflict_states) if conflict_states else None
    result = {
        "profile": profile,
        "human_num": human_num,
        "conflict_definition": {
            "dmin_threshold_m": CONFLICT_DMIN_THRESHOLD_M,
            "ttc_threshold_s": CONFLICT_TTC_THRESHOLD_S,
            "note": "provisional, not specified in guide.md, pending confirmation",
        },
        "seeds": [int(s) for s in seeds],
        "episodes_per_seed": episodes_per_seed,
        "max_steps": max_steps,
        "total_states_visited": total_states,
        "conflict_states": conflict_states,
        "conflict_states_with_safe_action": conflict_with_safe_action,
        "coverage_rate": coverage_rate,
        "gate_threshold": 0.80,
        "gate_status": (
            "PASS" if coverage_rate is not None and coverage_rate >= 0.80 else "FAIL"
        ),
        "per_seed": per_seed_conflict_rate,
        "action_table_sha256": action_table["table_sha256"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="crowd_nav/configs/env.config")
    parser.add_argument("--action-table", default="crowd_nav/configs/bdvl_action_table.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[71001, 71002, 71003, 71004, 71005])
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--human-num", type=int, default=None, help="override sim.human_num (P0-2's 20-human engineering stress)")
    parser.add_argument("--profile", default="nominal", choices=sorted(PROFILES.keys()))
    parser.add_argument("--output", default="runs/bayesian_dvl/p0_2_local_coverage/nominal_result.json")
    args = parser.parse_args()

    result = run(args.env_config, args.action_table, args.seeds, args.episodes_per_seed, args.max_steps, human_num=args.human_num, profile=args.profile)
    output_path = PACKAGE_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
