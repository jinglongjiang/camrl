#!/usr/bin/env python3
"""Paired SM-BRNE evaluator (Order R5).

This is intentionally independent of the legacy ``test.py``.  It evaluates
the same method-independent episode keys with four adapters:

``sm_brne``
    stable entity-clearance solver + full posterior.
``official_brne``
    pinned upstream BRNE solver + the same posterior samples.
``cv_brne``
    stable entity-clearance solver + constant-velocity pedestrian samples.
``orca``
    the project's unmodified RVO2 ORCA policy.

``nominal`` uses CrowdSim's standard ORCA humans.  ``heldout_interactive``
uses an action-conditioned subclass of the real CrowdSim backend.  The two
profiles are never pooled into one score.
"""

from __future__ import annotations

import argparse
import configparser
import csv
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/abc/workspace/nav_data/mamba/camrl/CrowdNav")
POLICY_CONFIG_PATH = REPO_ROOT / "crowd_nav/configs/policy_bayesian_brne.config"
ENV_CONFIG_PATH = REPO_ROOT / "crowd_nav/configs/env_bayesian_brne.config"
OUT_DIR = REPO_ROOT / "runs/bayesian_brne/r5_20260804"
METHODS = ("sm_brne", "official_brne", "cv_brne", "orca")
SCENARIOS = (
    "baseline_circle", "baseline_square", "dense_circle", "dense_square",
    "large_circle", "large_square",
)
SCENARIO_TABLE = {
    "baseline_circle": ("circle", 4.0, 5),
    "baseline_square": ("square", 10.0, 10),
    "dense_circle": ("circle", 4.0, 10),
    "dense_square": ("square", 10.0, 20),
    "large_circle": ("circle", 6.0, 12),
    "large_square": ("square", 14.0, 20),
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_json(value) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_crowdsim(scenario: str, env_class=None):
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot

    if env_class is None:
        env_class = CrowdSim

    shape, size, human_num = SCENARIO_TABLE[scenario]
    config = configparser.RawConfigParser()
    config.read(str(ENV_CONFIG_PATH))
    config.set("sim", "test_sim", f"{shape}_crossing")
    config.set("sim", "circle_radius", str(size) if shape == "circle" else config.get("sim", "circle_radius"))
    config.set("sim", "square_width", str(size) if shape == "square" else config.get("sim", "square_width"))
    config.set("sim", "human_num", str(human_num))
    env = env_class()
    env.configure(config)
    env.phase = "test"
    robot = Robot(config, "robot")
    robot.policy.multiagent_training = True
    robot.env = env
    env.set_robot(robot)
    return env


def _make_policy(method: str, artifact, model_config, planner_config, runtime_config):
    from crowd_nav.bayesian_brne.policy import BayesianBRNEPolicy

    if method == "sm_brne":
        config = replace(planner_config, solver_mode="stable_clearance", sampling_mode="full_posterior")
    elif method == "official_brne":
        config = replace(planner_config, solver_mode="official_exact", sampling_mode="full_posterior")
    elif method == "cv_brne":
        config = replace(planner_config, solver_mode="stable_clearance", sampling_mode="cv")
    else:
        return None
    policy = BayesianBRNEPolicy()
    policy.configure(config, artifact)
    return policy


def _snapshot_from_crowdsim(env) -> dict:
    return {
        "robot": env.robot.get_full_state().to_array().astype(np.float64),
        "humans": np.asarray([h.get_obs_array() for h in env.humans], dtype=np.float64),
        "track_ids": list(range(len(env.humans))),
    }


def _snapshot_from_synthetic(env) -> dict:
    return {
        "robot": env.robot_state_row().astype(np.float64),
        "humans": np.asarray(
            [[h.pos[0], h.pos[1], h.vel[0], h.vel[1], h.radius] for h in env.humans],
            dtype=np.float64,
        ),
        "track_ids": [int(h.track_id) for h in env.humans],
    }


def _method_independent_initial_hash(snapshot: dict, *, scenario: str, profile: str, suite_seed: int, episode_index: int) -> str:
    payload = {
        "scenario": scenario,
        "profile": profile,
        "suite_seed": int(suite_seed),
        "episode_index": int(episode_index),
        "robot": np.round(snapshot["robot"], 8),
        "humans": np.round(snapshot["humans"], 8),
        "track_ids": snapshot["track_ids"],
    }
    return _sha256_json(payload)


def _crowdsim_observation(env, identity_by_object: Dict[int, int], episode_seed: int):
    from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation

    return PolicyObservation(
        robot_px=env.robot.px, robot_py=env.robot.py, robot_vx=env.robot.vx, robot_vy=env.robot.vy,
        robot_radius=env.robot.radius, robot_gx=env.robot.gx, robot_gy=env.robot.gy,
        robot_v_pref=env.robot.v_pref, timestamp=env.global_time, time_step=env.time_step,
        humans=[
            TrackObservation(
                track_id=identity_by_object[id(h)], px=h.px, py=h.py, vx=h.vx, vy=h.vy,
                radius=h.radius, timestamp=env.global_time,
            )
            for h in env.humans
        ],
        episode_seed=episode_seed,
    )


def _synthetic_observation(env, episode_seed: int):
    from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation

    return PolicyObservation(
        robot_px=float(env.robot_pos[0]), robot_py=float(env.robot_pos[1]),
        robot_vx=float(env.robot_vel[0]), robot_vy=float(env.robot_vel[1]),
        robot_radius=float(env.robot_radius), robot_gx=float(env.robot_goal[0]),
        robot_gy=float(env.robot_goal[1]), robot_v_pref=float(env.robot_pref_speed),
        timestamp=float(getattr(env, "global_time", 0.0)), time_step=float(env.dt),
        humans=[
            TrackObservation(
                track_id=int(h.track_id), px=float(h.pos[0]), py=float(h.pos[1]),
                vx=float(h.vel[0]), vy=float(h.vel[1]), radius=float(h.radius),
                timestamp=float(getattr(env, "global_time", 0.0)),
            )
            for h in env.humans
        ],
        episode_seed=episode_seed,
    )


def _identity_reordered_ids(env) -> Tuple[List[int], List[int]]:
    """Return IDs in normal and reversed observation order.

    For CrowdSim, identity is assigned to object identity at reset and then
    survives list reordering.  For SyntheticEnv, track_id is already part of
    the state.  This function is used by the evaluator's adapter assertion,
    not by the policy's decision.
    """
    if hasattr(env.humans[0], "track_id"):
        normal = [int(h.track_id) for h in env.humans]
    else:
        identity = {id(h): i for i, h in enumerate(env.humans)}
        normal = [identity[id(h)] for h in env.humans]
    reversed_ids = list(reversed(normal))
    if sorted(normal) != sorted(reversed_ids) or len(set(reversed_ids)) != len(reversed_ids):
        raise AssertionError("track identity changed under list reordering")
    return normal, reversed_ids


def _orca_action(env, state):
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState

    policy = state["orca_policy"]
    if policy is None:
        policy = ORCA()
        policy.time_step = env.time_step if hasattr(env, "time_step") else env.dt
        policy.max_speed = env.robot.v_pref if hasattr(env, "robot") else env.robot_pref_speed
        state["orca_policy"] = policy
    if hasattr(env, "robot"):
        full = env.robot.get_full_state()
        humans = [h.get_observable_state() for h in env.humans]
        return policy.predict(JointState(full, humans))
    full = FullState(
        float(env.robot_pos[0]), float(env.robot_pos[1]), float(env.robot_vel[0]), float(env.robot_vel[1]),
        float(env.robot_radius), float(env.robot_goal[0]), float(env.robot_goal[1]),
        float(env.robot_pref_speed), float(np.arctan2(env.robot_vel[1], env.robot_vel[0])),
    )
    humans = [ObservableState(float(h.pos[0]), float(h.pos[1]), float(h.vel[0]), float(h.vel[1]), float(h.radius)) for h in env.humans]
    return policy.predict(JointState(full, humans))


def _run_crowdsim_episode(method: str, scenario: str, profile: str, episode_seed: int, suite_seed: int, episode_index: int, artifact, model_config, planner_config, runtime_config) -> dict:
    if profile == "heldout_interactive":
        from crowd_nav.bayesian_brne.interactive_crowdsim import HeldoutInteractiveCrowdSim
        env = _build_crowdsim(scenario, HeldoutInteractiveCrowdSim)
    elif profile == "nominal":
        env = _build_crowdsim(scenario)
    else:
        raise ValueError(f"unsupported CrowdSim profile {profile!r}")
    env.reset(seed=episode_seed, options={"test_case": episode_index})
    identity_by_object = {id(h): i for i, h in enumerate(env.humans)}
    snapshot = _snapshot_from_crowdsim(env)
    initial_hash = _method_independent_initial_hash(snapshot, scenario=scenario, profile=profile, suite_seed=suite_seed, episode_index=episode_index)
    _identity_reordered_ids(env)
    policy = _make_policy(method, artifact, model_config, planner_config, runtime_config)
    if policy is not None:
        policy.reset(episode_seed)
    orca_state = {"orca_policy": None}
    returned_action_mismatch_count = 0
    actions = []
    start = time.perf_counter()
    event = "did_not_terminate"
    info = {}
    max_steps = int(round(env.time_limit / env.time_step)) + 1
    for step in range(max_steps):
        if policy is not None:
            returned = policy.predict(_crowdsim_observation(env, identity_by_object, episode_seed))
        else:
            returned = _orca_action(env, orca_state)
        returned_v = np.array([float(returned.vx), float(returned.vy)], dtype=np.float64)
        actions.append(returned_v.tolist())
        _ob, _reward, terminated, truncated, info = env.step(returned)
        executed_v = np.array([float(env.robot.vx), float(env.robot.vy)], dtype=np.float64)
        if not np.allclose(returned_v, executed_v, atol=1e-12, rtol=0.0):
            returned_action_mismatch_count += 1
        if terminated or truncated:
            event = str(info.get("event", "unknown"))
            break
    return {
        "method": method, "scenario": scenario, "profile": profile,
        "suite_seed": int(suite_seed), "episode_seed": int(episode_seed), "episode_index": int(episode_index),
        "initial_state_hash": initial_hash, "event": event, "success": event == "reach_goal",
        "collision": event == "collision", "timeout": event == "timeout", "steps": len(actions),
        "returned_action_mismatch_count": returned_action_mismatch_count,
        "fallback_count": 0, "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        "actions": actions,
    }


def _run_synthetic_episode(method: str, scenario: str, episode_seed: int, suite_seed: int, episode_index: int, artifact, model_config, planner_config, runtime_config) -> dict:
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario

    env = make_scenario(scenario, "test_heldout_interactive", np.random.default_rng(episode_seed), dt=artifact.dt)
    env.global_time = 0.0
    identity_by_object = {id(h): int(h.track_id) for h in env.humans}
    snapshot = _snapshot_from_synthetic(env)
    initial_hash = _method_independent_initial_hash(snapshot, scenario=scenario, profile="heldout_interactive", suite_seed=suite_seed, episode_index=episode_index)
    _identity_reordered_ids(env)
    policy = _make_policy(method, artifact, model_config, planner_config, runtime_config)
    if policy is not None:
        policy.reset(episode_seed)
    orca_state = {"orca_policy": None}
    returned_action_mismatch_count = 0
    actions = []
    start = time.perf_counter()
    event = "did_not_terminate"
    max_steps = int(round(35.0 / env.dt))
    for step in range(max_steps):
        env.global_time = step * env.dt
        if policy is not None:
            returned = policy.predict(_synthetic_observation(env, episode_seed))
            returned_v = np.array([float(returned.vx), float(returned.vy)], dtype=np.float64)
        else:
            returned = _orca_action(env, orca_state)
            returned_v = np.array([float(returned.vx), float(returned.vy)], dtype=np.float64)
        actions.append(returned_v.tolist())
        env.step(returned_v)
        executed_v = np.asarray(env.robot_vel, dtype=np.float64)
        if not np.allclose(returned_v, executed_v, atol=1e-12, rtol=0.0):
            returned_action_mismatch_count += 1
        dmin = min(
            [float(np.hypot(h.pos[0] - env.robot_pos[0], h.pos[1] - env.robot_pos[1]) - h.radius - env.robot_radius) for h in env.humans]
            or [float("inf")]
        )
        if dmin < 0.0:
            event = "collision"
            break
        if float(np.hypot(*(env.robot_goal - env.robot_pos))) <= 0.25:
            event = "reach_goal"
            break
    if event == "did_not_terminate":
        event = "timeout"
    return {
        "method": method, "scenario": scenario, "profile": "heldout_interactive",
        "suite_seed": int(suite_seed), "episode_seed": int(episode_seed), "episode_index": int(episode_index),
        "initial_state_hash": initial_hash, "event": event, "success": event == "reach_goal",
        "collision": event == "collision", "timeout": event == "timeout", "steps": len(actions),
        "returned_action_mismatch_count": returned_action_mismatch_count,
        "fallback_count": 0, "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        "actions": actions,
    }


def _write_outputs(rows: List[dict], manifest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable_rows = [{k: v for k, v in row.items() if k != "actions"} for row in rows]
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in serializable_rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    fields = list(serializable_rows[0].keys())
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable_rows)
    (out_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")


def run_suite(*, methods: Iterable[str], scenarios: Iterable[str], profiles: Iterable[str], episodes: int, suite_seed: int, out_dir: Path) -> dict:
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact
    from crowd_nav.bayesian_brne.config import load_policy_config_file

    methods = tuple(methods)
    scenarios = tuple(scenarios)
    profiles = tuple(profiles)
    model_config, planner_config, runtime_config = load_policy_config_file(str(POLICY_CONFIG_PATH))
    artifact = ARHMMArtifact.load(runtime_config.artifact_path, expect_tier=runtime_config.artifact_tier)
    rows: List[dict] = []
    base_keys = {}
    for scenario in scenarios:
        for profile in profiles:
            for episode_index in range(episodes):
                episode_seed = int(suite_seed * 100000 + episode_index)
                for method in methods:
                    if method not in METHODS:
                        raise ValueError(f"unknown method {method!r}; expected {METHODS}")
                    if profile in ("nominal", "heldout_interactive"):
                        row = _run_crowdsim_episode(method, scenario, profile, episode_seed, suite_seed, episode_index, artifact, model_config, planner_config, runtime_config)
                    else:
                        raise ValueError("profile must be nominal or heldout_interactive")
                    key = (scenario, profile, suite_seed, episode_index)
                    if key in base_keys and base_keys[key] != row["initial_state_hash"]:
                        raise AssertionError(f"initial_state_hash mismatch for paired key {key}")
                    base_keys[key] = row["initial_state_hash"]
                    rows.append(row)
    if not rows:
        raise ValueError("no evaluation rows produced")
    keys_by_method = {(r["method"], r["scenario"], r["profile"], r["suite_seed"], r["episode_index"], r["initial_state_hash"]) for r in rows}
    for scenario in scenarios:
        for profile in profiles:
            for episode_index in range(episodes):
                hashes = {r["initial_state_hash"] for r in rows if r["scenario"] == scenario and r["profile"] == profile and r["episode_index"] == episode_index}
                if len(hashes) != 1 or len([r for r in rows if r["scenario"] == scenario and r["profile"] == profile and r["episode_index"] == episode_index]) != len(tuple(methods)):
                    raise AssertionError("paired method rows are incomplete or have mismatched initial hashes")
    config_hash = _sha256_json({"model": model_config, "planner": planner_config, "runtime": runtime_config})
    manifest = {
        "order": "R5_paired_evaluator",
        "date": "2026-08-04",
        "methods": list(methods), "scenarios": list(scenarios), "profiles": list(profiles),
        "episodes_per_key": episodes, "suite_seed": suite_seed,
        "policy_config_path": str(POLICY_CONFIG_PATH), "config_sha256": config_hash,
        "artifact_path": runtime_config.artifact_path, "artifact_tier": runtime_config.artifact_tier,
        "artifact_content_sha256": artifact.content_sha256(),
        "brne_root": runtime_config.brne_root, "brne_commit": runtime_config.brne_commit,
        "initial_state_hashes": {"keys": len(base_keys), "unique": len(set(base_keys.values()))},
        "returned_equals_executed": all(r["returned_action_mismatch_count"] == 0 for r in rows),
        "source_sha256": {p: _sha256_file(REPO_ROOT / p) for p in (
            "crowd_nav/tools/evaluate_sm_brne.py", "crowd_nav/bayesian_brne/policy.py",
            "crowd_nav/bayesian_brne/equilibrium_loop.py", "crowd_nav/bayesian_brne/brne_adapter.py",
            "crowd_nav/bayesian_brne/interactive_crowdsim.py",
        )},
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "heldout_backend": "HeldoutInteractiveCrowdSim",
        "note": "Paired engineering evaluator. Metrics are not a paper result until S2/S3 gates pass.",
    }
    _write_outputs(rows, manifest, out_dir)
    return {"rows": rows, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--scenarios", nargs="+", default=["baseline_circle"], choices=SCENARIOS)
    parser.add_argument("--profiles", nargs="+", default=["nominal", "heldout_interactive"], choices=("nominal", "heldout_interactive"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--suite-seed", type=int, default=2407)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="explicitly label the default small paired run")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >=1")
    result = run_suite(
        methods=args.methods, scenarios=args.scenarios, profiles=args.profiles,
        episodes=args.episodes, suite_seed=args.suite_seed, out_dir=args.out_dir,
    )
    rows = result["rows"]
    print(json.dumps({
        "rows": len(rows), "paired_keys": result["manifest"]["initial_state_hashes"],
        "returned_equals_executed": result["manifest"]["returned_equals_executed"],
        "out_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
