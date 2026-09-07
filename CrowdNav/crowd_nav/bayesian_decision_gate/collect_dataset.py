#!/usr/bin/env python3
"""Collect one dataset split (density x profile) for the Bayesian decision gate.

Each episode is stored with everything ``evaluate_action_ranking.py`` needs to
compute *ground-truth* counterfactual risk at every timestep: the full,
identity-stable observation sequence (so ``obs[t+1 : t+1+horizon]`` IS the
real recorded pedestrian future -- not a model's prediction of it), the
actions actually taken, per-pedestrian intervention-mode labels (for
event-window vs normal-window stratification), and episode metadata.

Each episode records TWO distinct seed fields, and evaluation code must never
conflate them:
  - ``episode_seed``: the per-episode simulation RNG seed (unique per
    episode -- see ``protocol.episode_seed``), used only to reproduce/seed
    that one rollout.
  - ``suite_seed``: the experiment-level seed passed via ``--seed`` (one of
    2407/3407/4407/5407/6407 in the default 5-seed formal run), IDENTICAL
    across every episode collected in one ``collect_dataset.py`` invocation.
    This is the field the bootstrap in ``evaluate_action_ranking.py`` must
    group by -- a previous version grouped by ``episode_seed`` instead, which
    is unique per episode, so the "seed-block bootstrap" was silently
    resampling individual episodes (its top-level block count equaled the
    episode count, not the intended 5 experiment seeds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.bayesian_decision_gate.protocol import (  # noqa: E402
    DENSITIES,
    PROFILES,
    SPLIT_PROFILE,
    BehaviorScheduler,
    InterventionORCA,
    build_environment,
    episode_seed,
    episode_test_case,
    load_config,
    observation_variable,
)


def collect_episode(
    env,
    robot,
    robot_policy,
    config,
    profile_name: str,
    num_humans: int,
    episode_seed_value: int,
    suite_seed: int,
    test_case: int,
) -> Dict:
    env.reset(seed=episode_seed_value, options={"test_case": int(test_case)})
    robot_policy.sim = None
    robot_policy._last_pref_vel = None

    policies = []
    for human in env.humans:
        policy = InterventionORCA(config)
        policy.time_step = env.time_step
        policy.reset()
        human.set_policy(policy)
        policies.append(policy)

    scheduler = BehaviorScheduler(
        PROFILES[profile_name], seed=episode_seed_value + 7919
    )
    scheduler.reset(len(policies))

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    mode_history: List[np.ndarray] = []
    done = False
    info: Dict = {}
    max_steps = int(np.ceil(env.time_limit / env.time_step)) + 1

    while not done and len(observations) < max_steps:
        observations.append(observation_variable(robot, env.humans, num_humans))
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
        "num_humans": int(num_humans),
        "episode_seed": int(episode_seed_value),
        "suite_seed": int(suite_seed),
        "events": dict(scheduler.counts),
        "outcome": str(info.get("event", "unknown")),
    }


def collect(
    *,
    env_config_path: Path,
    split: str,
    density: str,
    episodes: int,
    seed: int,
    output: Path,
    suite_seed: int = None,
):
    """``seed`` drives the actual per-episode simulation RNG (via
    ``episode_seed()``) and the output filename. ``suite_seed`` is the
    experiment-level identity used for bootstrap grouping and defaults to
    ``seed`` -- pass a different ``suite_seed`` only when collecting
    additional episodes for an ALREADY-existing experiment seed under a new
    ``seed`` value purely to get disjoint episode_seed/test_case ranges
    (see run_gate.py's ``--topup_heldout``)."""
    if suite_seed is None:
        suite_seed = seed
    profile_name = SPLIT_PROFILE[split]
    num_humans = DENSITIES[density]
    records = []
    environments = {}
    description = f"{split}/{density}"
    progress = tqdm(range(int(episodes)), desc=description, unit="ep", dynamic_ncols=True)
    for index in progress:
        scenario = "circle_crossing" if index % 2 == 0 else "square_crossing"
        if scenario not in environments:
            config = load_config(str(env_config_path), num_humans)
            environments[scenario] = (*build_environment(config, scenario), config)
        env, robot, robot_policy, config = environments[scenario]
        seed_value = episode_seed(split, density, seed, index)
        record = collect_episode(
            env,
            robot,
            robot_policy,
            config,
            profile_name,
            num_humans,
            episode_seed_value=seed_value,
            suite_seed=suite_seed,
            test_case=episode_test_case(seed_value, index),
        )
        records.append(record)
        event_total = sum(sum(item["events"].values()) for item in records)
        progress.set_postfix(events=event_total, outcome=record["outcome"][0:1])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        obs=np.asarray([r["obs"] for r in records], dtype=object),
        act=np.asarray([r["act"] for r in records], dtype=object),
        modes=np.asarray([r["modes"] for r in records], dtype=object),
        profile=np.asarray([r["profile"] for r in records]),
        scenario=np.asarray([r["scenario"] for r in records]),
        num_humans=np.asarray([r["num_humans"] for r in records], dtype=np.int64),
        episode_seed=np.asarray([r["episode_seed"] for r in records], dtype=np.int64),
        suite_seed=np.asarray([r["suite_seed"] for r in records], dtype=np.int64),
        outcome=np.asarray([r["outcome"] for r in records]),
    )
    summary = {
        "output": str(output.resolve()),
        "split": split,
        "density": density,
        "profile": profile_name,
        "num_humans": num_humans,
        "episodes": len(records),
        "scenarios": {
            name: sum(r["scenario"] == name for r in records)
            for name in ("circle_crossing", "square_crossing")
        },
        "events": {
            mode: sum(r["events"].get(mode, 0) for r in records)
            for mode in ("stop", "slow", "turn_left", "turn_right")
        },
        "outcomes": {
            outcome: sum(r["outcome"] == outcome for r in records)
            for outcome in ("reach_goal", "collision", "timeout")
        },
        "mean_steps": float(np.mean([len(r["obs"]) for r in records])) if records else 0.0,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", default=None)
    parser.add_argument(
        "--split",
        required=True,
        choices=("train_nonstationary", "validation_nonstationary", "heldout_nonstationary", "nominal"),
    )
    parser.add_argument("--density", required=True, choices=("5person", "20person"))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument(
        "--suite_seed", type=int, default=None,
        help="Experiment-level seed identity for bootstrap grouping (defaults to --seed). "
        "Pass explicitly only when --seed is offset purely to avoid filename/episode_seed collisions.",
    )
    parser.add_argument("--output_dir", default="runs/bayesian_decision_gate/data")
    args = parser.parse_args()

    from crowd_nav.bayesian_decision_gate.protocol import DEFAULT_ENV_CONFIG

    env_config_path = Path(args.env_config or DEFAULT_ENV_CONFIG).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output = output_dir / f"{args.split}_{args.density}_seed{args.seed}.npz"
    collect(
        env_config_path=env_config_path,
        split=args.split,
        density=args.density,
        episodes=args.episodes,
        seed=args.seed,
        output=output,
        suite_seed=args.suite_seed,
    )


if __name__ == "__main__":
    main()
