"""Closed-loop engineering and formal evaluation for CR-S1."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact
from crowd_nav.bayesian_brne.causal_runtime import CausalBayesianBRNEPolicy
from crowd_nav.bayesian_brne.config import PlannerConfig
from crowd_nav.bayesian_brne.interaction_protocol import (
    RobotControllerState, SCENARIO_TABLE, compute_robot_action, make_scenario,
)
from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation


def _observation(env, step: int, episode_seed: int) -> PolicyObservation:
    return PolicyObservation(
        robot_px=float(env.robot_pos[0]), robot_py=float(env.robot_pos[1]),
        robot_vx=float(env.robot_vel[0]), robot_vy=float(env.robot_vel[1]),
        robot_radius=float(env.robot_radius), robot_gx=float(env.robot_goal[0]),
        robot_gy=float(env.robot_goal[1]), robot_v_pref=float(env.robot_pref_speed),
        timestamp=float(step * env.dt), time_step=float(env.dt),
        humans=[
            TrackObservation(
                track_id=int(human.track_id), px=float(human.pos[0]), py=float(human.pos[1]),
                vx=float(human.vel[0]), vy=float(human.vel[1]), radius=float(human.radius),
                timestamp=float(step * env.dt),
            )
            for human in env.humans
        ],
        episode_seed=int(episode_seed),
    )


def _policy(method: str, artifact: CausalResponseArtifact, brne_root: str,
            *, smoke: bool = False) -> CausalBayesianBRNEPolicy:
    if method == "causal_full":
        selected, response, mean = artifact, True, False
    elif method == "causal_zero_response":
        selected, response, mean = artifact.zero_response(), False, False
    elif method == "causal_posterior_mean":
        selected, response, mean = artifact, True, True
    else:
        raise ValueError(method)
    config = PlannerConfig(
        horizon_steps=4 if smoke else 12,
        num_samples=8 if smoke else 64,
        max_outer_iterations=2 if smoke else 10,
        equilibrium_iterations=2 if smoke else 10,
        brne_root=brne_root,
    )
    policy = CausalBayesianBRNEPolicy(response_enabled=response, posterior_mean=mean)
    policy.configure(config, selected)
    return policy


def _min_clearance(env) -> float:
    if not env.humans:
        return float("inf")
    return min(
        float(np.linalg.norm(human.pos - env.robot_pos) - human.radius - env.robot_radius)
        for human in env.humans
    )


def run_episode(method: str, artifact: CausalResponseArtifact, brne_root: str, *,
                scenario: str, split: str, suite_seed: int, episode_index: int,
                max_steps: int, smoke: bool = False) -> dict:
    episode_seed = suite_seed * 100000 + episode_index
    env = make_scenario(scenario, split, np.random.default_rng(episode_seed), artifact.dt)
    policy = None
    controller = None
    if method.startswith("causal_"):
        policy = _policy(method, artifact, brne_root, smoke=smoke)
        policy.reset(episode_seed)
    else:
        controller = RobotControllerState(
            controller_type=method, brne_root=brne_root,
            brne_rng=np.random.default_rng(episode_seed + 17),
        )
    minimum_clearance = _min_clearance(env)
    fallback_count = 0
    elapsed = []
    outcome = "timeout"
    steps = max_steps
    for step in range(max_steps):
        started = time.perf_counter()
        if policy is not None:
            action = policy.predict(_observation(env, step, episode_seed))
            robot_action = np.array([action.vx, action.vy])
        else:
            robot_action, fallback = compute_robot_action(env, controller)
            fallback_count += int(fallback is not None)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        env.step(robot_action)
        minimum_clearance = min(minimum_clearance, _min_clearance(env))
        if minimum_clearance < 0.0:
            outcome, steps = "collision", step + 1
            break
        if float(np.linalg.norm(env.robot_goal - env.robot_pos)) <= env.robot_radius:
            outcome, steps = "success", step + 1
            break
    return {
        "method": method, "scenario": scenario, "split": split,
        "suite_seed": suite_seed, "episode_index": episode_index,
        "episode_seed": episode_seed, "outcome": outcome, "steps": steps,
        "minimum_clearance": minimum_clearance, "fallback_count": fallback_count,
        "mean_decision_ms": float(np.mean(elapsed)),
        "p95_decision_ms": float(np.quantile(elapsed, 0.95)),
    }


def _summarize(records: list) -> dict:
    result = {}
    groups = sorted(set((row["method"], row["scenario"], row["split"]) for row in records))
    for method, scenario, split in groups:
        rows = [row for row in records if (row["method"], row["scenario"], row["split"]) ==
                (method, scenario, split)]
        result[f"{method}/{scenario}/{split}"] = {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["outcome"] == "success" for row in rows])),
            "collision_rate": float(np.mean([row["outcome"] == "collision" for row in rows])),
            "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_minimum_clearance": float(np.mean([row["minimum_clearance"] for row in rows])),
            "mean_decision_ms": float(np.mean([row["mean_decision_ms"] for row in rows])),
        }
    return result


def _write_records(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _run_suite(output: Path, artifact: CausalResponseArtifact, brne_root: str, *,
               scenarios: Iterable[str], splits: Iterable[str], suite_seeds: Iterable[int],
               episodes: int, methods: Iterable[str], max_steps: int, smoke: bool,
               name: str) -> dict:
    records = []
    for split in splits:
        for scenario in scenarios:
            for suite_seed in suite_seeds:
                for episode_index in range(episodes):
                    for method in methods:
                        row = run_episode(
                            method, artifact, brne_root, scenario=scenario, split=split,
                            suite_seed=suite_seed, episode_index=episode_index,
                            max_steps=max_steps, smoke=smoke,
                        )
                        records.append(row)
                        print(
                            f"[EVAL] {name} {len(records)} method={method} scenario={scenario} "
                            f"split={split} seed={suite_seed} ep={episode_index} outcome={row['outcome']}",
                            flush=True,
                        )
    _write_records(output / "evaluation" / f"{name}_records.csv", records)
    summary = _summarize(records)
    (output / "evaluation" / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return {"status": "PASS", "name": name, "records": len(records), "summary": summary}


def run_evaluation_stage(stage: str, output: Path, registry: dict, brne_root: str) -> dict:
    artifact_path = output / "production" / "causal_response_model.npz"
    artifact = CausalResponseArtifact.load(artifact_path, require_production=True)
    if stage == "latency":
        return _run_suite(
            output, artifact, brne_root, scenarios=("baseline_circle", "dense_square"),
            splits=("test_heldout_interactive",), suite_seeds=(8691,), episodes=1,
            methods=("causal_full",), max_steps=10, smoke=True, name="latency",
        )
    if stage == "smoke":
        return _run_suite(
            output, artifact, brne_root, scenarios=SCENARIO_TABLE.keys(),
            splits=("test_nominal", "test_heldout_interactive"), suite_seeds=(8692,), episodes=1,
            methods=("causal_full", "causal_zero_response", "orca"), max_steps=8,
            smoke=True, name="smoke",
        )
    return _run_suite(
        output, artifact, brne_root, scenarios=SCENARIO_TABLE.keys(),
        splits=("test_nominal", "test_heldout_interactive"), suite_seeds=range(8701, 8711),
        episodes=100,
        methods=(
            "causal_full", "causal_zero_response", "causal_posterior_mean",
            "orca", "original_brne", "goal_directed",
        ),
        max_steps=int(round(35.0 / artifact.dt)), smoke=False, name="formal",
    )
