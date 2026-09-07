#!/usr/bin/env python3
"""Order R6/F7 engineering benchmark for SM-BRNE.

This command measures runtime and numerical health only.  It never changes
planner parameters based on SR/CR and never produces a paper performance
claim.  Use ``--smoke`` locally for a short check; use the same command with
more episodes/steps on the 4090 for the read-only formal F7 report.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Dict, List

import numpy as np

from crowd_nav.tools.evaluate_sm_brne import (
    ENV_CONFIG_PATH,
    POLICY_CONFIG_PATH,
    SCENARIO_TABLE,
    _build_crowdsim,
)

REPO_ROOT = Path("/home/abc/workspace/nav_data/mamba/camrl/CrowdNav")
DEFAULT_OUT = REPO_ROOT / "runs/bayesian_brne/f7_20260804"


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{q}": float(np.percentile(array, q)) for q in (50, 95, 99)}


def _load_policy():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact
    from crowd_nav.bayesian_brne.config import load_policy_config_file
    from crowd_nav.bayesian_brne.policy import BayesianBRNEPolicy

    model_config, planner_config, runtime_config = load_policy_config_file(str(POLICY_CONFIG_PATH))
    artifact = ARHMMArtifact.load(runtime_config.artifact_path, expect_tier=runtime_config.artifact_tier)
    policy = BayesianBRNEPolicy()
    policy.configure(planner_config, artifact)
    return policy, planner_config, runtime_config, artifact


def _observation(env, episode_seed: int):
    from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation

    return PolicyObservation(
        robot_px=env.robot.px, robot_py=env.robot.py,
        robot_vx=env.robot.vx, robot_vy=env.robot.vy,
        robot_radius=env.robot.radius, robot_gx=env.robot.gx, robot_gy=env.robot.gy,
        robot_v_pref=env.robot.v_pref, timestamp=env.global_time, time_step=env.time_step,
        humans=[TrackObservation(
            track_id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius,
            timestamp=env.global_time,
        ) for i, h in enumerate(env.humans)],
        episode_seed=episode_seed,
    )


def _run_case(scenario: str, episodes: int, steps: int) -> List[dict]:
    policy, _planner, _runtime, _artifact = _load_policy()
    rows = []
    for episode_index in range(episodes):
        episode_seed = 2407 * 100000 + episode_index
        env = _build_crowdsim(scenario)
        env.reset(seed=episode_seed, options={"test_case": episode_index})
        policy.reset(episode_seed)
        for step in range(steps):
            observation = _observation(env, episode_seed)
            start = time.perf_counter()
            action = policy.predict(observation)
            total_ms = (time.perf_counter() - start) * 1000.0
            diagnostics = policy.last_diagnostics or {}
            iteration_diags = diagnostics.get("iteration_diagnostics", [])
            rows.append({
                "scenario": scenario,
                "human_count": len(env.humans),
                "episode_index": episode_index,
                "step": step,
                "predict_total_ms": total_ms,
                "policy_predict_total_ms": diagnostics.get("timings_ms", {}).get("predict_total"),
                "belief_update_ms": diagnostics.get("timings_ms", {}).get("belief_update"),
                "robot_sampling_ms": diagnostics.get("timings_ms", {}).get("robot_sampling"),
                "outer_loop_ms": diagnostics.get("timings_ms", {}).get("outer_loop"),
                "action_selection_ms": diagnostics.get("timings_ms", {}).get("action_selection"),
                "sampling_ms": float(sum(d.get("sampling_ms", 0.0) for d in iteration_diags)),
                "solver_wall_ms": float(sum(d.get("solver_wall_ms", 0.0) for d in iteration_diags)),
                "outer_iterations": int(diagnostics.get("outer_iterations", 0)),
                "status": diagnostics.get("status"),
                "inner_fallback": bool(any(d.get("inner_numeric_fallback_used", False) for d in iteration_diags)),
                "action_finite": bool(np.isfinite([action.vx, action.vy]).all()),
            })
            _observation_after, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    return rows


def _numeric_stress() -> dict:
    """Small deterministic solver stress fixtures; no policy tuning."""
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver

    results = {}
    for name, offset, radius in (
        ("full_overlap", 0.0, 0.3),
        ("near_singular", 1e-9, 0.3),
        ("zero_velocity_like", 0.0, 0.3),
    ):
        trajectories = np.zeros((3, 8, 4, 2), dtype=np.float64)
        trajectories[1:, :, :, 0] = offset
        trajectories[2, :, :, 1] = offset
        solver = BRNESolver(
            solver_mode="stable",
            brne_root="/home/abc/temp/brne",
            safe_distance=0.20,
            cost_sigma=0.10,
            cost_scale=100.0,
        )
        result = solver.solve(
            trajectories,
            np.full(3, radius),
            edge_mask=None,
            equilibrium_iterations=10,
        )
        finite = bool(np.isfinite(result.weights).all())
        results[name] = {
            "finite": finite,
            "numeric_fallback_used": bool(result.numeric_fallback_used),
            "iterations": int(result.iterations),
        }
    return results


def run(*, scenarios, episodes: int, steps: int, out_dir: Path, smoke: bool) -> dict:
    rows = []
    for scenario in scenarios:
        rows.extend(_run_case(scenario, episodes, steps))
    metric_names = (
        "predict_total_ms", "belief_update_ms", "robot_sampling_ms",
        "outer_loop_ms", "action_selection_ms", "sampling_ms", "solver_wall_ms",
    )
    latency = {name: _quantiles([
        float(row[name]) for row in rows if row[name] is not None
    ]) for name in metric_names}
    report = {
        "order": "R6_F7",
        "date": "2026-08-04",
        "smoke": bool(smoke),
        "scenarios": list(scenarios),
        "episodes": episodes,
        "steps_requested": steps,
        "steps_observed": len(rows),
        "latency_ms": latency,
        "status_counts": {str(k): int(v) for k, v in zip(*np.unique([r["status"] for r in rows], return_counts=True))},
        "outer_nonconverged_count": int(sum(r["status"] not in ("converged", "no_tracks") for r in rows)),
        "inner_fallback_count": int(sum(r["inner_fallback"] for r in rows)),
        "nan_count": int(sum(not r["action_finite"] for r in rows)),
        "numeric_stress": _numeric_stress(),
        "memory_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "note": "Engineering latency/numeric report only; no SR/CR or paper claim.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f7_steps.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (out_dir / "f7_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIO_TABLE), default=["baseline_circle", "dense_square"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("episodes and steps must be >= 1")
    report = run(
        scenarios=args.scenarios, episodes=args.episodes, steps=args.steps,
        out_dir=args.out_dir, smoke=args.smoke,
    )
    print(json.dumps({
        "rows": report["steps_observed"],
        "latency_ms": report["latency_ms"]["predict_total_ms"],
        "nan_count": report["nan_count"],
        "inner_fallback_count": report["inner_fallback_count"],
        "out_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
