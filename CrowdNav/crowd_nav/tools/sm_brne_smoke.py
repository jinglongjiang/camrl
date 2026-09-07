#!/usr/bin/env python3
"""Order F6 (2026-08-03): independent SM-BRNE smoke/evaluation entry point.

Deliberately NOT a modification of crowd_nav/test.py (guide.md 4.7: "不得
大改旧test.py。新建独立的SM-BRNE smoke/evaluation入口, 避免再次受旧草稿覆盖
问题影响"). This script drives crowd_sim.envs.CrowdSim directly:

- Registers/uses ``policy_factory['bayesian_brne']`` (Order F6's own
  registration in crowd_nav/policy/policy_factory.py).
- Solves the track-ID bridging gap noted in Order F5's policy.py docstring:
  CrowdSim's native ``JointState``/``ObservableState`` carry no per-human
  ID, but ``env.humans`` (confirmed by direct inspection of crowd_sim.py)
  is NEVER reordered/resized during an episode once ``reset()`` builds it --
  only mutated in place -- so this script assigns ``track_id = list index``
  ONCE per episode (right after ``reset()``) and reads ``env.humans``
  DIRECTLY every step (never crowd_nav/test.py's TTC-re-sorted view, which
  would silently break that index-based identity).
- Bypasses ``Robot.act()`` entirely (its policy-name dispatch whitelist
  does not include "BayesianBRNEPolicy", and does not need to -- this
  script computes the action itself every step and calls ``env.step()``
  directly, exactly as guide.md 4.6's ``predict()`` order specifies).
- Uses the Order F1 ``engineering_only`` artifact (never a
  ``tier='production'`` artifact, since none exists yet -- U3/Phase S1 has
  not produced a converged, formally-selected K). Every result this script
  produces is an ENGINEERING check (does the pipe not leak?), never a
  performance number.

Usage: python -m crowd_nav.tools.sm_brne_smoke
"""

from __future__ import annotations

import configparser
from dataclasses import asdict
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/abc/workspace/nav_data/mamba/camrl/CrowdNav")
POLICY_CONFIG_PATH = REPO_ROOT / "crowd_nav/configs/policy_bayesian_brne.config"
ENV_CONFIG_PATH = REPO_ROOT / "crowd_nav/configs/env_bayesian_brne.config"
OUT_DIR = REPO_ROOT / "runs/sm_brne_smoke_r4_20260804"

SCENARIO_TABLE = {
    "baseline_circle": ("circle", 4.0, 5),
    "baseline_square": ("square", 10.0, 10),
    "dense_circle": ("circle", 4.0, 10),
    "dense_square": ("square", 10.0, 20),
    "large_circle": ("circle", 6.0, 12),
    "large_square": ("square", 14.0, 20),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _build_env(shape: str, radius_or_width: float, human_num: int, seed_env_dt: float = 0.25):
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot

    env_config = configparser.RawConfigParser()
    env_config.read(str(ENV_CONFIG_PATH))
    env_config.set("sim", "test_sim", f"{shape}_crossing")
    if shape == "circle":
        env_config.set("sim", "circle_radius", str(radius_or_width))
    else:
        env_config.set("sim", "square_width", str(radius_or_width))
    env_config.set("sim", "human_num", str(human_num))

    env = CrowdSim()
    env.configure(env_config)
    env.phase = "test"

    robot = Robot(env_config, "robot")
    robot.policy.multiagent_training = True  # NonePolicy (policy=none) does not set this; crowd_sim.reset() reads it unconditionally
    robot.env = env
    env.set_robot(robot)
    return env


def _load_policy():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact
    from crowd_nav.bayesian_brne.config import load_policy_config_file
    from crowd_nav.bayesian_brne.policy import BayesianBRNEPolicy

    _model_config, planner_config, runtime_config = load_policy_config_file(str(POLICY_CONFIG_PATH))
    artifact = ARHMMArtifact.load(runtime_config.artifact_path, expect_tier=runtime_config.artifact_tier)
    policy = BayesianBRNEPolicy()
    policy.configure(planner_config, artifact)
    return policy, planner_config, runtime_config, artifact


def run_episode(env, policy, episode_seed: int, max_steps: int = 200) -> dict:
    from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation
    from crowd_nav.bayesian_brne.diagnostics import summarize_policy_diagnostics

    ob, _info = env.reset(seed=episode_seed, options={"test_case": episode_seed})
    # Order F6's track-ID bridge: list index at reset() time, valid for the
    # WHOLE episode since env.humans is never reordered/resized in place.
    policy.reset(episode_seed)

    actions = []
    nan_detected = False
    event = "did_not_terminate"
    step_count = 0
    diagnostic_steps = []
    for step_count in range(1, max_steps + 1):
        robot = env.robot
        humans = env.humans  # NEVER crowd_nav.test's TTC-resorted view
        observation = PolicyObservation(
            robot_px=robot.px, robot_py=robot.py, robot_vx=robot.vx, robot_vy=robot.vy,
            robot_radius=robot.radius, robot_gx=robot.gx, robot_gy=robot.gy,
            robot_v_pref=robot.v_pref, timestamp=env.global_time, time_step=env.time_step,
            humans=[
                TrackObservation(track_id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius, timestamp=env.global_time)
                for i, h in enumerate(humans)
            ],
            episode_seed=episode_seed,
        )
        if not np.isfinite(observation.robot_px + observation.robot_py + observation.robot_vx + observation.robot_vy):
            nan_detected = True
            break

        action = policy.predict(observation)
        diagnostic_steps.append(summarize_policy_diagnostics(policy.last_diagnostics, step_count))
        if not (np.isfinite(action.vx) and np.isfinite(action.vy)):
            nan_detected = True
            break
        actions.append((action.vx, action.vy))

        ob, reward, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(ob)):
            nan_detected = True
            break
        if terminated or truncated:
            event = info.get("event", "unknown")
            break

    return {
        "episode_seed": episode_seed,
        "event": event,
        "step_count": step_count,
        "nan_detected": nan_detected,
        "legal_terminal_state": (event in ("reach_goal", "collision", "timeout")) and not nan_detected,
        "actions": actions,
        "diagnostic_steps": diagnostic_steps,
        "n_active_tracks_final": len(policy.belief_bank.active_track_ids()) if policy.belief_bank else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        log_lines.append(line)

    log("[sm_brne_smoke] loading engineering_only artifact + policy ...")
    policy, planner_config, runtime_config, artifact = _load_policy()
    log(f"[sm_brne_smoke] resolved solver_mode={planner_config.solver_mode} "
        f"H={planner_config.horizon_steps} M={planner_config.num_samples} "
        f"artifact={runtime_config.artifact_path} tier={runtime_config.artifact_tier}")

    from crowd_nav.bayesian_brne.diagnostics import JsonlDiagnosticWriter

    diagnostic_writer = JsonlDiagnosticWriter(
        OUT_DIR / "diagnostics.jsonl",
        resolved_config=planner_config,
        artifact_sha256=artifact.content_sha256(),
        suite_seed=2407,
        overwrite=True,
    )

    def record_episode(ep: dict, scenario: str) -> None:
        diagnostic_writer.write_episode(
            episode_seed=ep["episode_seed"],
            scenario=scenario,
            profile="engineering_smoke",
            step_records=ep.get("diagnostic_steps", []),
            termination_event=ep["event"],
            elapsed_ms=float(ep.get("elapsed_s", 0.0)) * 1000.0,
        )

    results = {"single_episode": None, "six_scenarios": {}, "reproducibility_check": None}

    # 1. Single episode first (basic smoke).
    log("[sm_brne_smoke] running 1 solo episode (baseline_circle) ...")
    env = _build_env(*SCENARIO_TABLE["baseline_circle"])
    t0 = time.time()
    solo = run_episode(env, policy, episode_seed=0)
    solo["elapsed_s"] = time.time() - t0
    results["single_episode"] = solo
    record_episode(solo, "baseline_circle")
    log(f"  event={solo['event']} steps={solo['step_count']} nan={solo['nan_detected']} "
        f"legal={solo['legal_terminal_state']} elapsed={solo['elapsed_s']:.2f}s")

    # 2. Two episodes per each of the six scenarios (12 total, engineering
    # check only -- guide.md F6 explicitly: "这些数字只用于工程检查，不用于
    # 效果判断").
    all_legal = solo["legal_terminal_state"]
    for name, (shape, size, human_num) in SCENARIO_TABLE.items():
        log(f"[sm_brne_smoke] scenario={name} ({shape}, size={size}, human_num={human_num}) ...")
        env = _build_env(shape, size, human_num)
        episodes = []
        for seed in (100, 101):
            t0 = time.time()
            ep = run_episode(env, policy, episode_seed=seed)
            ep["elapsed_s"] = time.time() - t0
            episodes.append(ep)
            record_episode(ep, name)
            all_legal = all_legal and ep["legal_terminal_state"]
            log(f"    seed={seed} event={ep['event']} steps={ep['step_count']} "
                f"nan={ep['nan_detected']} legal={ep['legal_terminal_state']} elapsed={ep['elapsed_s']:.2f}s")
        results["six_scenarios"][name] = episodes

    n_smoke_episodes = 1 + 2 * len(SCENARIO_TABLE)
    log(f"[sm_brne_smoke] {n_smoke_episodes} smoke episodes total, all_legal_terminal_state={all_legal}")

    # 3. Reproducibility: rerun the SAME (scenario, seed) pair and require
    # byte-identical action sequences.
    log("[sm_brne_smoke] reproducibility check: baseline_circle seed=100, rerun twice ...")
    env_a = _build_env(*SCENARIO_TABLE["baseline_circle"])
    run_a = run_episode(env_a, policy, episode_seed=100)
    env_b = _build_env(*SCENARIO_TABLE["baseline_circle"])
    run_b = run_episode(env_b, policy, episode_seed=100)
    reproducible = (
        run_a["event"] == run_b["event"]
        and run_a["step_count"] == run_b["step_count"]
        and np.allclose(run_a["actions"], run_b["actions"])
    )
    results["reproducibility_check"] = {
        "reproducible": bool(reproducible),
        "run_a_event": run_a["event"], "run_b_event": run_b["event"],
        "run_a_steps": run_a["step_count"], "run_b_steps": run_b["step_count"],
    }
    log(f"  reproducible={reproducible} (run_a: {run_a['event']}/{run_a['step_count']} steps, "
        f"run_b: {run_b['event']}/{run_b['step_count']} steps)")

    manifest = {
        "order": "R4_diagnostics_engineering_smoke",
        "date": "2026-08-04",
        "policy_config_path": str(POLICY_CONFIG_PATH),
        "resolved_config": {
            "horizon_steps": planner_config.horizon_steps,
            "num_samples": planner_config.num_samples,
            "max_speed": planner_config.max_speed,
            "max_acceleration": planner_config.max_acceleration,
            "max_outer_iterations": planner_config.max_outer_iterations,
            "outer_tolerance": planner_config.outer_tolerance,
            "outer_damping": planner_config.outer_damping,
            "oscillation_tolerance": planner_config.oscillation_tolerance,
            "equilibrium_iterations": planner_config.equilibrium_iterations,
            "solver_mode": planner_config.solver_mode,
            "safe_distance": planner_config.safe_distance,
            "cost_sigma": planner_config.cost_sigma,
            "cost_scale": planner_config.cost_scale,
            "sparse_enabled": planner_config.sparse_enabled,
            "pruning_margin": planner_config.pruning_margin,
            "sampling_mode": planner_config.sampling_mode,
        },
        "artifact_path": runtime_config.artifact_path,
        "artifact_tier": runtime_config.artifact_tier,
        "artifact_content_sha256": artifact.content_sha256(),
        "brne_root": runtime_config.brne_root,
        "brne_commit": runtime_config.brne_commit,
        "git_head": _git_head(),
        "source_sha256": {
            p: _sha256_file(REPO_ROOT / p) for p in [
                "crowd_nav/bayesian_brne/policy.py",
                "crowd_nav/bayesian_brne/belief_tracker.py",
                "crowd_nav/bayesian_brne/robot_sampler.py",
                "crowd_nav/bayesian_brne/trajectory_sampler.py",
                "crowd_nav/bayesian_brne/equilibrium_loop.py",
                "crowd_nav/bayesian_brne/diagnostics.py",
                "crowd_nav/bayesian_brne/brne_adapter.py",
                "crowd_nav/bayesian_brne/config.py",
                "crowd_nav/configs/policy_bayesian_brne.config",
                "crowd_nav/policy/policy_factory.py",
                "crowd_nav/tools/sm_brne_smoke.py",
            ]
        },
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "diagnostics_jsonl": str(OUT_DIR / "diagnostics.jsonl"),
        "note": "Engineering smoke check only -- NOT a performance evaluation. "
                "Uses tier='engineering_only' artifact (not production; K unconverged/not formally selected).",
    }

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(OUT_DIR / "controller.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n[sm_brne_smoke] wrote manifest.json / results.json / controller.log to {OUT_DIR}")
    print(f"[sm_brne_smoke] FINAL: {n_smoke_episodes}/{n_smoke_episodes} legal_terminal_state={all_legal}, "
          f"reproducible={reproducible}")


if __name__ == "__main__":
    main()
