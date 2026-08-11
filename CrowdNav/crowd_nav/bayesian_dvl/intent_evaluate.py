"""Formal evaluator for the goal-intent (V6) chain -- Order C3.2/C3.3.

Plan section 7 / 2.2 point 16: the previous evaluator kept every episode
in memory and printed only aggregate rates to stdout -- no per-episode
rows, no manifest, no hashes, no resume, and the three result KINDS
(paper-main / held-out junction stress / ablation) all went to the same
place. A run that died at episode 400 of 600 left nothing.

This module:
  * appends ONE row per episode IMMEDIATELY (crash-safe), reusing the
    existing belief-free ``statistics.EpisodeRecord`` schema and
    ``evaluate.write/read_episode_records_csv`` so the goal-intent chain
    reports the SAME metrics as every other method in this project;
  * records the full metric set plan section 7 requires (outcome, nav
    time, path length/ratio, min clearance, discomfort frequency, mean
    speed, smoothness) plus the initial-state hash;
  * writes a manifest with command, environment, and every provenance
    hash (checkpoint / config / code / action grid / scene registry);
  * RESUMES by episode identity -- already-completed episodes are skipped,
    never re-run and never duplicated;
  * keeps paper-main / stress / ablation in SEPARATE result directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from crowd_nav.bayesian_dvl.config import FROZEN_VALUES
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.evaluate import (
    deterministic_records_sha256, read_episode_records_csv, write_episode_records_csv,
)
from crowd_nav.bayesian_dvl.intent_policy import (
    build_intent_human_feature_batch, remaining_time_fraction, score_candidates_v5,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.intent_train import (
    FORMAL_SIX_SCENARIOS, _ScenarioEpisode, build_formal_scenario_env,
)
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
from crowd_nav.bayesian_dvl.scene_candidates import circle_scene, make_candidate_fn, square_scene
from crowd_nav.bayesian_dvl.statistics import EpisodeRecord
from crowd_sim.envs.utils.action import ActionXY


class IntentEvaluateError(ValueError):
    pass


# The three result KINDS must never share a directory (plan C3.3).
RESULT_KINDS = ("paper_main", "heldout_junction", "ablation")


def initial_state_hash(robot, humans) -> str:
    """Identity of an episode's STARTING configuration. Plan section 7:
    ablation arms must be shown to share the same initial conditions --
    not the same trajectories, which legitimately diverge once the closed-
    loop actions differ."""
    payload = [
        [round(float(robot.px), 9), round(float(robot.py), 9),
         round(float(robot.gx), 9), round(float(robot.gy), 9), round(float(robot.v_pref), 9)],
        [[round(float(h.px), 9), round(float(h.py), 9), round(float(h.gx), 9), round(float(h.gy), 9),
          round(float(h.radius), 9), round(float(h.v_pref), 9)] for h in humans],
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class EpisodeMetrics:
    outcome: str
    steps: int
    elapsed_time: float
    min_clearance: float
    path_length: float
    path_ratio: float
    discomfort_frequency: float
    mean_speed: float
    smoothness: float
    initial_goal_distance: float
    final_goal_distance: float
    initial_state_hash: str
    mean_decision_latency_ms: float


def _run_one_episode(
    env, robot, scene, model, action_table, belief_mode: str, planner_seed: int,
    advance_hidden_state=None, n_samples: int = 60, horizon: int = 8, device: str = "cpu",
) -> EpisodeMetrics:
    """Drive ONE episode with the model's own greedy policy and collect the
    full plan-section-7 metric set."""
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
    planner_rng = np.random.default_rng(planner_seed)
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    discomfort_dist = float(FROZEN_VALUES.get("discomfort_distance", 0.2) or 0.2)

    init_hash = initial_state_hash(robot, env.humans)
    start = np.array([robot.px, robot.py], dtype=np.float64)
    goal = np.array([robot.gx, robot.gy], dtype=np.float64)
    initial_goal_distance = float(np.linalg.norm(goal - start))

    outcome, step = None, 0
    path_length, min_clearance = 0.0, float("inf")
    discomfort_steps, speeds, headings, latencies = 0, [], [], []
    prev = start.copy()

    for step in range(max_steps):
        if advance_hidden_state is not None:
            advance_hidden_state()
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        bank.update({h.track_id: (h.px, h.py) for h in humans})
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        remaining = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])

        t0 = time.perf_counter()
        human_feats, human_mask = build_intent_human_feature_batch(
            bank, robot_obs, humans, mode=belief_mode, rng=planner_rng, horizon=horizon, n_samples=n_samples)
        results = score_candidates_v5(
            model, robot_obs, human_feats, human_mask, action_table, remaining, device=device)
        best = max(results, key=lambda r: r.q_mean)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        vx, vy = action_table[best.action_index]
        speeds.append(float(np.hypot(vx, vy)))
        headings.append(float(np.arctan2(vy, vx)))

        clearance = min(
            (float(np.hypot(robot_obs.px - h.px, robot_obs.py - h.py)) - robot_obs.radius - h.radius)
            for h in humans) if humans else float("inf")
        min_clearance = min(min_clearance, clearance)
        if clearance < discomfort_dist:
            discomfort_steps += 1

        _, _reward, terminated, truncated, info = env.step(ActionXY(float(vx), float(vy)))
        cur = np.array([env.robot.px, env.robot.py], dtype=np.float64)
        path_length += float(np.linalg.norm(cur - prev))
        prev = cur
        if terminated or truncated:
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(
                info.get("event"), "timeout")
            break
    if outcome is None:
        outcome = "timeout"

    n = step + 1
    final_goal_distance = float(np.linalg.norm(np.array([env.robot.gx, env.robot.gy]) - prev))
    # smoothness: mean absolute heading change per step (lower is smoother)
    smoothness = 0.0
    if len(headings) > 1:
        d = np.diff(np.unwrap(np.array(headings)))
        smoothness = float(np.mean(np.abs(d)))
    return EpisodeMetrics(
        outcome=outcome, steps=n, elapsed_time=n * float(FROZEN_VALUES["dt"]),
        min_clearance=float(min_clearance if np.isfinite(min_clearance) else 0.0),
        path_length=path_length,
        path_ratio=float(path_length / initial_goal_distance) if initial_goal_distance > 1e-9 else 0.0,
        discomfort_frequency=float(discomfort_steps) / n,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        smoothness=smoothness,
        initial_goal_distance=initial_goal_distance, final_goal_distance=final_goal_distance,
        initial_state_hash=init_hash,
        mean_decision_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
    )


EXTRA_FIELDS = ("path_ratio", "discomfort_frequency", "mean_speed", "smoothness", "initial_state_hash")


def _append_row(path: Path, record: EpisodeRecord, extra: Dict[str, object]) -> None:
    """Append ONE row immediately and flush+fsync. Plan section 7: a
    crashed run must keep every episode it already finished."""
    from crowd_nav.bayesian_dvl.evaluate import CSV_FIELDS
    fields = list(CSV_FIELDS) + list(EXTRA_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if is_new:
            writer.writeheader()
        row = asdict(record)
        row.update(extra)
        writer.writerow(row)
        fh.flush()
        import os
        os.fsync(fh.fileno())


def completed_identities(path: Path) -> set:
    """(scenario, episode_seed) pairs already on disk -- the resume key."""
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["scenario"], int(row["episode_seed"])))
    return done


def write_manifest(out_dir: Path, payload: Dict[str, object]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    payload = dict(payload)
    payload.update({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_persistent_evaluation(
    env_config_path: Path,
    model: DistributionalValueModel,
    action_table: np.ndarray,
    out_dir: Path,
    kind: str,
    jobs: Sequence[Tuple[str, int, bool]],
    belief_mode: str = "full",
    method: str = "intent_bdvl",
    profile: str = "nominal",
    suite_seed: int = 0,
    n_samples: int = 60,
    horizon: int = 8,
    device: str = "cpu",
    provenance: Optional[Dict[str, object]] = None,
    resume: bool = True,
) -> Path:
    """Run ``jobs`` = [(scenario, episode_seed, is_heldout), ...], appending
    one row per episode to ``out_dir/episodes.csv`` and writing a manifest.

    ``kind`` must be one of RESULT_KINDS and becomes part of the output
    path, so paper-main / stress / ablation results can never land in the
    same directory (plan C3.3).
    """
    if kind not in RESULT_KINDS:
        raise IntentEvaluateError(f"kind must be one of {RESULT_KINDS}, got {kind!r}")
    out_dir = Path(out_dir) / kind
    if kind == "ablation":
        # EVERY arm gets its own directory, full included -- otherwise the
        # full arm's rows sit in the parent next to the other arms' folders
        # and an aggregation script silently treats them differently.
        out_dir = out_dir / f"arm_{belief_mode}"
    elif belief_mode != "full":
        out_dir = out_dir / f"arm_{belief_mode}"
    csv_path = out_dir / "episodes.csv"
    done = completed_identities(csv_path) if resume else set()
    if not resume and csv_path.exists():
        csv_path.unlink()

    n_new = 0
    for scenario, episode_seed, is_heldout in jobs:
        if (scenario, episode_seed) in done:
            continue
        if scenario in FORMAL_SIX_SCENARIOS:
            env, robot, shape, size = build_formal_scenario_env(env_config_path, scenario)
            env.case_counter["test"] = episode_seed % (2**32 - 1)
            env.reset()
            scene = square_scene(width=size, n_rows=4) if shape == "square" else circle_scene(radius=size, n_sectors=8)
            advance = None
        else:
            ep = _ScenarioEpisode(env_config_path, scenario, episode_seed, is_heldout=is_heldout)
            env, robot, scene = ep.env, ep.robot, ep.scene
            advance = ep.advance_hidden_state

        m = _run_one_episode(
            env, robot, scene, model, action_table, belief_mode, planner_seed=5_000_000 + episode_seed,
            advance_hidden_state=advance, n_samples=n_samples, horizon=horizon, device=device)
        record = EpisodeRecord(
            method=f"{method}:{belief_mode}", scenario=scenario, profile=profile,
            suite_seed=suite_seed, episode_seed=episode_seed, outcome=m.outcome, steps=m.steps,
            elapsed_time=m.elapsed_time, min_clearance=m.min_clearance, path_length=m.path_length,
            mean_decision_latency_ms=m.mean_decision_latency_ms,
            initial_goal_distance=m.initial_goal_distance, final_goal_distance=m.final_goal_distance,
        )
        _append_row(csv_path, record, {
            "path_ratio": m.path_ratio, "discomfort_frequency": m.discomfort_frequency,
            "mean_speed": m.mean_speed, "smoothness": m.smoothness,
            "initial_state_hash": m.initial_state_hash,
        })
        n_new += 1

    payload = {
        "kind": kind, "belief_mode": belief_mode, "method": method, "profile": profile,
        "suite_seed": suite_seed, "n_jobs": len(jobs), "n_new_episodes": n_new,
        "n_total_rows": len(completed_identities(csv_path)),
        "episodes_csv_sha256": _sha256_file(csv_path) if csv_path.exists() else None,
    }
    if provenance:
        payload["provenance"] = provenance
    write_manifest(out_dir, payload)
    return csv_path


def summarize_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Per-scenario aggregate over a persisted episodes.csv."""
    rows: Dict[str, List[dict]] = {}
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.setdefault(row["scenario"], []).append(row)
    out = {}
    for scenario, rs in rows.items():
        n = len(rs)
        out[scenario] = {
            "n": float(n),
            "success_rate": sum(1 for r in rs if r["outcome"] == "success") / n,
            "collision_rate": sum(1 for r in rs if r["outcome"] == "collision") / n,
            "timeout_rate": sum(1 for r in rs if r["outcome"] == "timeout") / n,
            "mean_nav_time": sum(float(r["elapsed_time"]) for r in rs) / n,
            "mean_path_ratio": sum(float(r["path_ratio"]) for r in rs) / n,
            "mean_min_clearance": sum(float(r["min_clearance"]) for r in rs) / n,
            "mean_discomfort_frequency": sum(float(r["discomfort_frequency"]) for r in rs) / n,
            "mean_speed": sum(float(r["mean_speed"]) for r in rs) / n,
            "mean_smoothness": sum(float(r["smoothness"]) for r in rs) / n,
        }
    return out
