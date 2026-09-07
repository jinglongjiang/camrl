"""Closed-loop evaluation for the base-only multimodal BRNE policy."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from crowd_nav.bayesian_brne.causal_evaluate import _min_clearance, _observation
from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact
from crowd_nav.bayesian_brne.causal_runtime import CausalBayesianBRNEPolicy
from crowd_nav.bayesian_brne.config import PlannerConfig
from crowd_nav.bayesian_brne.interaction_protocol import (
    RobotControllerState,
    SCENARIO_TABLE,
    compute_robot_action,
    make_scenario,
)


def _planner(registry: dict, brne_root: str, *, smoke: bool) -> PlannerConfig:
    values = dict(registry["planner"])
    if smoke:
        values.update(horizon_steps=4, num_samples=8, max_outer_iterations=2, equilibrium_iterations=2)
    values["brne_root"] = brne_root
    return PlannerConfig(**values)


def _make_policy(method: str, selected: CausalResponseArtifact, k1: CausalResponseArtifact,
                 registry: dict, brne_root: str, smoke: bool) -> CausalBayesianBRNEPolicy:
    if method == "full_posterior_selected":
        artifact, posterior_mean = selected, False
    elif method == "posterior_mean_selected":
        artifact, posterior_mean = selected, True
    elif method == "full_posterior_k1":
        artifact, posterior_mean = k1, False
    else:
        raise ValueError(method)
    policy = CausalBayesianBRNEPolicy(response_enabled=False, posterior_mean=posterior_mean)
    policy.configure(_planner(registry, brne_root, smoke=smoke), artifact)
    return policy


def run_episode(method: str, selected: CausalResponseArtifact, k1: CausalResponseArtifact,
                registry: dict, brne_root: str, *, scenario: str, split: str,
                suite_seed: int, episode_index: int, max_steps: int, smoke: bool) -> dict:
    episode_seed = int(suite_seed) * 100000 + int(episode_index)
    env = make_scenario(scenario, split, np.random.default_rng(episode_seed), selected.dt)
    policy = None
    controller = None
    if method.startswith("full_") or method.startswith("posterior_"):
        policy = _make_policy(method, selected, k1, registry, brne_root, smoke)
        policy.reset(episode_seed)
    else:
        controller = RobotControllerState(
            controller_type=method,
            brne_root=brne_root,
            brne_rng=np.random.default_rng(episode_seed + 17),
        )
    minimum_clearance = _min_clearance(env)
    elapsed = []
    fallback_count = 0
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
        "method": method,
        "scenario": scenario,
        "split": split,
        "suite_seed": suite_seed,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "outcome": outcome,
        "steps": steps,
        "minimum_clearance": minimum_clearance,
        "fallback_count": fallback_count,
        "mean_decision_ms": float(np.mean(elapsed)),
        "p95_decision_ms": float(np.quantile(elapsed, 0.95)),
    }


def _write_records(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _summary(records: list) -> dict:
    output = {}
    keys = sorted({(row["method"], row["scenario"], row["split"]) for row in records})
    for method, scenario, split in keys:
        rows = [row for row in records if (row["method"], row["scenario"], row["split"]) == (method, scenario, split)]
        output[f"{method}/{scenario}/{split}"] = {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["outcome"] == "success" for row in rows])),
            "collision_rate": float(np.mean([row["outcome"] == "collision" for row in rows])),
            "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_minimum_clearance": float(np.mean([row["minimum_clearance"] for row in rows])),
            "mean_decision_ms": float(np.mean([row["mean_decision_ms"] for row in rows])),
        }
    return output


def _paired_seed_ci(records: list, comparator: str, registry: dict, offset: int) -> dict:
    full_rows = {
        (row["suite_seed"], row["episode_index"], row["split"]): row
        for row in records if row["method"] == "full_posterior_selected"
    }
    other_rows = {
        (row["suite_seed"], row["episode_index"], row["split"]): row
        for row in records if row["method"] == comparator
    }
    if set(full_rows) != set(other_rows):
        raise RuntimeError(f"paired episode identity mismatch for {comparator}")
    per_seed: Dict[int, dict] = {}
    for key in sorted(full_rows):
        seed = key[0]
        left, right = full_rows[key], other_rows[key]
        bucket = per_seed.setdefault(seed, {"sr": [], "cr": [], "tr": [], "clearance": [], "steps": []})
        bucket["sr"].append(float(left["outcome"] == "success") - float(right["outcome"] == "success"))
        bucket["cr"].append(float(right["outcome"] == "collision") - float(left["outcome"] == "collision"))
        bucket["tr"].append(float(right["outcome"] == "timeout") - float(left["outcome"] == "timeout"))
        bucket["clearance"].append(left["minimum_clearance"] - right["minimum_clearance"])
        if left["outcome"] == right["outcome"] == "success":
            bucket["steps"].append(right["steps"] - left["steps"])
    seed_metrics = {
        name: np.asarray([np.mean(values[name]) if values[name] else 0.0 for _, values in sorted(per_seed.items())])
        for name in ("sr", "cr", "tr", "clearance", "steps")
    }
    rng = np.random.default_rng(int(registry["gate"]["bootstrap_seed"]) + offset)
    replicates = int(registry["gate"]["bootstrap_replicates"])
    result = {}
    for name, values in seed_metrics.items():
        samples = values[rng.integers(0, len(values), (replicates, len(values)))].mean(axis=1)
        result[name] = {
            "point": float(values.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
        }
    return result


def _pilot_gate(records: list, registry: dict) -> dict:
    comparisons = {
        "posterior_mean_selected": _paired_seed_ci(records, "posterior_mean_selected", registry, 101),
        "full_posterior_k1": _paired_seed_ci(records, "full_posterior_k1", registry, 102),
    }
    no_harm = all(
        metrics["sr"]["point"] >= -0.02
        and metrics["cr"]["point"] >= 0.0
        and metrics["tr"]["point"] >= 0.0
        for metrics in comparisons.values()
    )
    nontrivial = any(
        metrics["clearance"]["ci_low"] > 0.0
        or metrics["cr"]["ci_low"] > 0.0
        or metrics["steps"]["ci_low"] > 0.0
        for metrics in comparisons.values()
    )
    return {
        "status": "PASS" if no_harm and nontrivial else "FAIL",
        "no_harm": no_harm,
        "nontrivial_improvement": nontrivial,
        "comparisons": comparisons,
    }


def _run(output: Path, selected: CausalResponseArtifact, k1: CausalResponseArtifact,
         registry: dict, brne_root: str, *, stage: str) -> dict:
    spec = registry["evaluation"][stage]
    records = []
    for split in spec["splits"]:
        for scenario in spec["scenarios"]:
            for suite_seed in spec["suite_seeds"]:
                for episode_index in range(int(spec["episodes_per_seed"])):
                    for method in spec["methods"]:
                        row = run_episode(
                            method, selected, k1, registry, brne_root,
                            scenario=scenario, split=split, suite_seed=int(suite_seed),
                            episode_index=episode_index, max_steps=int(spec["max_steps"]),
                            smoke=bool(spec.get("smoke", False)),
                        )
                        records.append(row)
                        print(
                            f"[MM-EVAL] {stage} n={len(records)} method={method} split={split} "
                            f"seed={suite_seed} ep={episode_index} outcome={row['outcome']}",
                            flush=True,
                        )
    _write_records(output / "evaluation" / f"{stage}_records.csv", records)
    result = {"status": "PASS", "records": len(records), "summary": _summary(records)}
    if stage == "pilot":
        gate = _pilot_gate(records, registry)
        result["gate"] = gate
        result["status"] = gate["status"]
    (output / "evaluation" / f"{stage}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def run_evaluation_stage(stage: str, output: Path, registry: dict, brne_root: str) -> dict:
    selected = CausalResponseArtifact.load(
        output / "production" / "base_motion_selected.npz", require_production=True,
    )
    k1 = CausalResponseArtifact.load(
        output / "production" / "base_motion_k1.npz", require_production=True,
    )
    if np.max(np.abs(selected.D)) > 1e-12 or np.max(np.abs(k1.D)) > 1e-12:
        raise RuntimeError("MM-S1 production artifact is not base-only")
    return _run(output, selected, k1, registry, brne_root, stage=stage)
