#!/usr/bin/env python3
"""Reproduce the MM-S1 ORCA execution-model mismatch diagnostic.

This is a read-only scientific audit: it does not fit a model, modify an
artifact, or alter the MM-S1 verdict.  It compares three execution models on
the exact suite seeds and episode indices registered for the MM-S1 pilot:

``current_wrapper``
    The current project ORCA wrapper chooses the robot action, while humans
    execute the separate behavior FSM in ``SyntheticEnv.step``.
``raw_robot_orca``
    The same setup without the wrapper's post-ORCA acceleration clipping.
``reciprocal_orca``
    A mechanism sanity control in which both robot and humans execute the
    velocities returned by one shared RVO2 simulator.

The output directory contains per-episode records, collision geometry,
aggregate summaries, and a provenance manifest with source hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import rvo2

from crowd_nav.bayesian_brne.causal_evaluate import _min_clearance
from crowd_nav.bayesian_brne.interaction_protocol import (
    RobotControllerState,
    SyntheticEnv,
    compute_robot_action,
    make_scenario,
)
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


VARIANTS = ("current_wrapper", "raw_robot_orca", "reciprocal_orca")
SOURCE_FILES = (
    "crowd_nav/tools/audit_orca_protocol_mismatch.py",
    "crowd_nav/bayesian_brne/interaction_protocol.py",
    "crowd_nav/bayesian_brne/multimodal_evaluate.py",
    "crowd_nav/bayesian_brne/causal_evaluate.py",
    "crowd_sim/envs/policy/orca.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nearest_geometry(env: SyntheticEnv) -> Dict[str, object]:
    if not env.humans:
        return {
            "track_id": None,
            "behavior_type": None,
            "center_distance": float("inf"),
            "radius_sum": float("nan"),
            "clearance": float("inf"),
        }
    center_distances = np.asarray([
        np.linalg.norm(human.pos - env.robot_pos) for human in env.humans
    ])
    radius_sums = np.asarray([
        human.radius + env.robot_radius for human in env.humans
    ])
    clearances = center_distances - radius_sums
    index = int(np.argmin(clearances))
    human = env.humans[index]
    behavior = getattr(human.behavior_state.behavior_type, "value", None)
    if behavior is None:
        behavior = str(human.behavior_state.behavior_type)
    return {
        "track_id": int(human.track_id),
        "behavior_type": behavior,
        "center_distance": float(center_distances[index]),
        "radius_sum": float(radius_sums[index]),
        "clearance": float(clearances[index]),
    }


def _joint_state(env: SyntheticEnv) -> JointState:
    theta = float(np.arctan2(env.robot_vel[1], env.robot_vel[0]))
    self_state = FullState(
        px=float(env.robot_pos[0]),
        py=float(env.robot_pos[1]),
        vx=float(env.robot_vel[0]),
        vy=float(env.robot_vel[1]),
        radius=float(env.robot_radius),
        gx=float(env.robot_goal[0]),
        gy=float(env.robot_goal[1]),
        v_pref=float(env.robot_pref_speed),
        theta=theta,
    )
    humans = [
        ObservableState(
            px=float(human.pos[0]),
            py=float(human.pos[1]),
            vx=float(human.vel[0]),
            vy=float(human.vel[1]),
            radius=float(human.radius),
        )
        for human in env.humans
    ]
    return JointState(self_state, humans)


class _CurrentWrapper:
    def __init__(self, env: SyntheticEnv) -> None:
        self.state = RobotControllerState(controller_type="orca")
        self.fallback_count = 0

    def step(self, env: SyntheticEnv) -> None:
        action, fallback = compute_robot_action(env, self.state)
        self.fallback_count += int(fallback is not None)
        env.step(action)


class _RawRobotOrca:
    def __init__(self, env: SyntheticEnv) -> None:
        self.policy = ORCA()
        self.policy.time_step = env.dt
        self.policy.max_speed = env.robot_pref_speed
        self.fallback_count = 0

    def step(self, env: SyntheticEnv) -> None:
        action = self.policy.predict(_joint_state(env))
        env.step(np.asarray([action.vx, action.vy], dtype=float))


class _ReciprocalOrca:
    def __init__(self, env: SyntheticEnv) -> None:
        self.sim = rvo2.PyRVOSimulator(
            env.dt, 10.0, 10, 5.0, 5.0, 0.3, 2.0,
        )
        self.sim.addAgent(
            tuple(env.robot_pos), 10.0, 10, 5.0, 5.0,
            env.robot_radius + 0.01, env.robot_pref_speed,
            tuple(env.robot_vel),
        )
        for human in env.humans:
            self.sim.addAgent(
                tuple(human.pos), 10.0, 10, 5.0, 5.0,
                human.radius + 0.01, human.pref_speed, tuple(human.vel),
            )
        self.fallback_count = 0

    @staticmethod
    def _goal_velocity(position: np.ndarray, goal: np.ndarray, speed: float) -> np.ndarray:
        displacement = goal - position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-8:
            return np.zeros(2)
        return displacement / distance * speed

    def step(self, env: SyntheticEnv) -> None:
        robot_pref = self._goal_velocity(
            env.robot_pos, env.robot_goal, env.robot_pref_speed,
        )
        self.sim.setAgentPrefVelocity(0, tuple(robot_pref))
        for index, human in enumerate(env.humans, start=1):
            human_pref = self._goal_velocity(
                human.pos, human.goal, human.pref_speed,
            )
            self.sim.setAgentPrefVelocity(index, tuple(human_pref))
        self.sim.doStep()

        env.robot_pos = np.asarray(self.sim.getAgentPosition(0), dtype=float)
        env.robot_vel = np.asarray(self.sim.getAgentVelocity(0), dtype=float)
        for index, human in enumerate(env.humans, start=1):
            human.pos = np.asarray(self.sim.getAgentPosition(index), dtype=float)
            human.vel = np.asarray(self.sim.getAgentVelocity(index), dtype=float)


CONTROLLERS: Dict[str, Callable[[SyntheticEnv], object]] = {
    "current_wrapper": _CurrentWrapper,
    "raw_robot_orca": _RawRobotOrca,
    "reciprocal_orca": _ReciprocalOrca,
}


def _run_episode(
    variant: str,
    split: str,
    scenario: str,
    suite_seed: int,
    episode_index: int,
    dt: float,
    max_steps: int,
) -> Tuple[dict, Optional[dict]]:
    episode_seed = int(suite_seed) * 100000 + int(episode_index)
    env = make_scenario(
        scenario, split, np.random.default_rng(episode_seed), dt,
    )
    controller = CONTROLLERS[variant](env)
    minimum_clearance = float(_min_clearance(env))
    outcome = "timeout"
    completed_steps = max_steps
    collision = None

    for step in range(max_steps):
        before = _nearest_geometry(env)
        controller.step(env)
        after = _nearest_geometry(env)
        minimum_clearance = min(minimum_clearance, float(after["clearance"]))
        if float(after["clearance"]) < 0.0:
            outcome = "collision"
            completed_steps = step + 1
            collision = {
                "variant": variant,
                "split": split,
                "scenario": scenario,
                "suite_seed": suite_seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "collision_step": step + 1,
                "before": before,
                "after": after,
            }
            break
        if float(np.linalg.norm(env.robot_goal - env.robot_pos)) <= env.robot_radius:
            outcome = "success"
            completed_steps = step + 1
            break

    record = {
        "variant": variant,
        "split": split,
        "scenario": scenario,
        "suite_seed": suite_seed,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "outcome": outcome,
        "steps": completed_steps,
        "minimum_clearance": minimum_clearance,
        "fallback_count": int(controller.fallback_count),
        "collision_step": collision["collision_step"] if collision else "",
        "collision_track_id": collision["after"]["track_id"] if collision else "",
        "collision_behavior_type": collision["after"]["behavior_type"] if collision else "",
        "pre_collision_center_distance": collision["before"]["center_distance"] if collision else "",
        "pre_collision_radius_sum": collision["before"]["radius_sum"] if collision else "",
        "pre_collision_clearance": collision["before"]["clearance"] if collision else "",
        "collision_center_distance": collision["after"]["center_distance"] if collision else "",
        "collision_radius_sum": collision["after"]["radius_sum"] if collision else "",
        "collision_clearance": collision["after"]["clearance"] if collision else "",
    }
    return record, collision


def _summarize(records: list) -> dict:
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for split in sorted({record["split"] for record in records}):
            rows = [
                record for record in records
                if record["variant"] == variant and record["split"] == split
            ]
            counts = {
                outcome: sum(row["outcome"] == outcome for row in rows)
                for outcome in ("success", "collision", "timeout")
            }
            n = len(rows)
            summary[variant][split] = {
                "episodes": n,
                "success_count": counts["success"],
                "collision_count": counts["collision"],
                "timeout_count": counts["timeout"],
                "success_rate": counts["success"] / n,
                "collision_rate": counts["collision"] / n,
                "timeout_rate": counts["timeout"] / n,
                "mean_minimum_clearance": float(np.mean([
                    row["minimum_clearance"] for row in rows
                ])),
                "fallback_count": int(sum(row["fallback_count"] for row in rows)),
            }
    return summary


def _compare_pilot_records(records: list, pilot_path: Path) -> dict:
    with pilot_path.open(newline="") as handle:
        reference_rows = [
            row for row in csv.DictReader(handle) if row["method"] == "orca"
        ]
    reference = {
        (
            row["split"], row["scenario"], int(row["suite_seed"]),
            int(row["episode_index"]),
        ): row
        for row in reference_rows
    }
    current = {
        (
            row["split"], row["scenario"], int(row["suite_seed"]),
            int(row["episode_index"]),
        ): row
        for row in records if row["variant"] == "current_wrapper"
    }
    missing_in_audit = sorted(set(reference) - set(current))
    missing_in_reference = sorted(set(current) - set(reference))
    common = sorted(set(reference) & set(current))
    outcome_mismatches = sum(
        reference[key]["outcome"] != current[key]["outcome"] for key in common
    )
    steps_mismatches = sum(
        int(reference[key]["steps"]) != int(current[key]["steps"])
        for key in common
    )
    clearance_errors = [
        abs(
            float(reference[key]["minimum_clearance"])
            - float(current[key]["minimum_clearance"])
        )
        for key in common
    ]
    exact = (
        not missing_in_audit
        and not missing_in_reference
        and outcome_mismatches == 0
        and steps_mismatches == 0
        and (not clearance_errors or max(clearance_errors) <= 1e-12)
    )
    return {
        "status": "PASS" if exact else "FAIL",
        "reference_path": str(pilot_path.resolve()),
        "reference_orca_rows": len(reference_rows),
        "audit_current_wrapper_rows": len(current),
        "common_rows": len(common),
        "missing_in_audit": len(missing_in_audit),
        "missing_in_reference": len(missing_in_reference),
        "outcome_mismatches": outcome_mismatches,
        "steps_mismatches": steps_mismatches,
        "max_abs_minimum_clearance_error": (
            max(clearance_errors) if clearance_errors else None
        ),
    }


def _write_csv(path: Path, records: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="crowd_nav/configs/multimodal_s1_registry.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pilot-records", default=None,
        help=(
            "Existing MM-S1 pilot CSV to reproduce. Defaults to the path "
            "derived from registry.experiment_id."
        ),
    )
    parser.add_argument(
        "--episodes-per-seed", type=int, default=None,
        help="Override only for a smoke run; formal audit uses registry value.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    registry_path = Path(args.registry).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    registry = json.loads(registry_path.read_text())
    pilot_records_path = Path(
        args.pilot_records
        or (
            f"runs/bayesian_brne/{registry['experiment_id']}"
            "/evaluation/pilot_records.csv"
        )
    ).resolve()
    if not pilot_records_path.is_file():
        raise FileNotFoundError(
            f"registered pilot records are required for reproduction: {pilot_records_path}"
        )
    pilot = registry["evaluation"]["pilot"]
    episodes_per_seed = (
        int(args.episodes_per_seed)
        if args.episodes_per_seed is not None
        else int(pilot["episodes_per_seed"])
    )
    splits = tuple(pilot["splits"])
    scenarios = tuple(pilot["scenarios"])
    suite_seeds = tuple(int(seed) for seed in pilot["suite_seeds"])
    max_steps = int(pilot["max_steps"])
    dt = float(registry["dt"])

    records = []
    collisions = []
    for variant in VARIANTS:
        for split in splits:
            for scenario in scenarios:
                for suite_seed in suite_seeds:
                    for episode_index in range(episodes_per_seed):
                        record, collision = _run_episode(
                            variant, split, scenario, suite_seed, episode_index,
                            dt, max_steps,
                        )
                        records.append(record)
                        if collision is not None:
                            collisions.append(collision)

    pilot_reproduction = _compare_pilot_records(records, pilot_records_path)
    if pilot_reproduction["status"] != "PASS":
        raise RuntimeError(
            "current_wrapper did not exactly reproduce registered MM-S1 pilot ORCA rows: "
            f"{pilot_reproduction}"
        )
    summary = {
        "audit": "orca_execution_model_mismatch",
        "status": "COMPLETED",
        "scientific_scope": (
            "mechanism diagnostic only; does not modify the MM-S1 verdict"
        ),
        "configuration": {
            "registry": str(registry_path),
            "scenario": list(scenarios),
            "splits": list(splits),
            "suite_seeds": list(suite_seeds),
            "episodes_per_seed": episodes_per_seed,
            "max_steps": max_steps,
            "dt": dt,
            "variants": list(VARIANTS),
        },
        "reference_pilot_reproduction": pilot_reproduction,
        "results": _summarize(records),
    }
    _write_csv(output / "episode_records.csv", records)
    (output / "collision_cases.json").write_text(
        json.dumps(collisions, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required provenance source missing: {path}")
        source_hashes[relative] = _sha256(path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(repo_root),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": _git_commit(repo_root),
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "pilot_records_path": str(pilot_records_path),
        "pilot_records_sha256": _sha256(pilot_records_path),
        "source_sha256": source_hashes,
        "outputs_sha256": {
            name: _sha256(output / name)
            for name in ("episode_records.csv", "collision_cases.json", "summary.json")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary["results"], indent=2, sort_keys=True))
    print(f"[ORCA-AUDIT] saved reproducible audit to {output}")


if __name__ == "__main__":
    main()
