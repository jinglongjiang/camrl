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

import argparse

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

from crowd_nav.bayesian_dvl.intent_runtime_config import FROZEN_VALUES, TRACKER_DEFAULTS
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
RESULT_KINDS = ("paper_main", "heldout_junction", "ablation", "dev_standard", "selection_dev",
                "paper_junction")


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
    action_fn=None,
) -> EpisodeMetrics:
    """Drive ONE episode and collect the full plan-section-7 metric set.

    ``action_fn`` replaces ONLY the choice of action: given the env it
    returns the ActionXY to take. Everything downstream -- the simulator's
    swept ``dmin``, path length, discomfort, outcome mapping, path ratio,
    smoothness, the initial-state hash -- runs through this one function for
    every method. A baseline scored by a separate runner would drift on any
    of those definitions and the comparison would silently stop being a
    comparison, which is the whole reason a baseline exists.
    """
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
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
        if action_fn is None:
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
        else:
            # the baseline plans from the simulator state directly: it has no
            # belief bank and no action grid, so neither is built for it
            t0 = time.perf_counter()
            act = action_fn(env)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            vx, vy = float(act.vx), float(act.vy)

        speeds.append(float(np.hypot(vx, vy)))
        headings.append(float(np.arctan2(vy, vx)))

        _, _reward, terminated, truncated, info = env.step(ActionXY(float(vx), float(vy)))

        # Order 1R: clearance MUST come from the simulator's SWEPT separation
        # for the interval this action just covered, not from a snapshot
        # taken before the action.
        #
        # The previous version measured `robot_obs` against the humans'
        # pre-step positions and then `break`ed on termination, so the
        # interval in which a collision actually happened was never
        # measured. CrowdSim decides `collision` precisely when its swept
        # `dmin` (point_to_segment_dist over the relative motion, minus both
        # radii) goes negative -- so the old metric could not report a
        # negative clearance even for an episode that ended in a collision,
        # and every "zero negative clearance" reading was an artifact.
        #
        # Fails closed: silently falling back to the pre-action snapshot
        # would reintroduce exactly the bug this replaces.
        if "dmin" not in info:
            raise IntentEvaluateError(
                "env.step() returned no 'dmin'; the swept clearance is required and there is no safe "
                f"fallback (info keys: {sorted(info)})")
        clearance = float(info["dmin"])
        min_clearance = min(min_clearance, clearance)
        if clearance < discomfort_dist:
            discomfort_steps += 1

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
    action_fn=None,
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
            advance_hidden_state=advance, n_samples=n_samples, horizon=horizon, device=device,
            action_fn=action_fn)
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


# --------------------------------------------------------------------- #
# CLI. Order 7: validate / eval-* / ablate answer "how good is this
# checkpoint", a different job from producing one. Keeping them in the
# training entry point forced it to know every evaluation seed block.
# --------------------------------------------------------------------- #

def cmd_eval(args, which: str) -> int:
    _cli_env()
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    scene_hash = scene_registry_sha256(cfg)
    model = _load_eval_model(Path(args.checkpoint), cfg, device, grid.table_hash(), scene_hash)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "results"
    prov = _provenance(cfg, args, scene_hash, grid.table_hash())
    # belief_mode comes from the WEIGHTS, not from a flag: scoring a
    # mean/cv checkpoint with full features would measure an input mismatch
    # instead of the belief treatment.
    arm = checkpoint_arm(Path(args.checkpoint))
    prov["belief_mode"] = arm
    common = dict(belief_mode=arm, n_samples=cfg.future_n_samples, horizon=cfg.future_horizon,
                  device=str(device), provenance=prov, resume=not args.no_resume)

    if which == "validate":
        seeds = list(cfg.validation_seeds)[: args.episodes] if args.episodes else list(cfg.validation_seeds)
        jobs = [("standard", s, False) for s in seeds]
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "paper_main",
                                              jobs, method="intent_bdvl_validation", **common)
        _print_summary("validation (training health only -- never checkpoint selection)", csv_path)
        return 0

    if which == "eval-paper":
        # C4RF.5: the paper-main table must be episode-for-episode pairable
        # with the Mamba-VL / SARL / LSTM numbers, which come from test8.py.
        # That means test8's OWN identities -- base seed 42, 500 episodes
        # per scenario, seed = (42 + case_id*1_000_003 + ep) % (2**31-1) --
        # not a private held-out block of ours.
        n_ep = args.episodes or PAPER_MAIN_EPISODES_PER_SCENARIO
        jobs = paper_main_jobs(episodes_per_scenario=n_ep, base_seed=args.base_seed)
        prov.update({"protocol": "test8_paper_main", "base_seed": args.base_seed,
                     "episodes_per_scenario": n_ep,
                     "episode_seed_formula": "(base_seed + case_id*1_000_003 + ep) % (2**31-1)"})
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "paper_main",
                                              jobs, **common)
        _print_summary(f"paper-main: test8 protocol (base_seed={args.base_seed}, {n_ep} eps/scenario)", csv_path)
        return 0

    if which == "eval-stress":
        seeds = (list(JUNCTION_CROWD_HELDOUT_SEEDS)[: args.episodes] if args.episodes
                 else list(JUNCTION_CROWD_HELDOUT_SEEDS))
        jobs = [("junction_crowd", s, True) for s in seeds]
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "heldout_junction",
                                              jobs, **common)
        _print_summary("held-out junction-crowd stress (shifted speeds + wider fork)", csv_path)
        return 0

    if which == "eval-dev-standard":
        # Order 1: large-sample GREEDY diagnosis of the `standard` scenario.
        # DEVELOPMENT ONLY -- this exists to characterise an existing run,
        # never to select the paper's weights.
        seeds = (list(STANDARD_DEV_DIAGNOSTIC_SEEDS)[: args.episodes] if args.episodes
                 else list(STANDARD_DEV_DIAGNOSTIC_SEEDS))
        jobs = [("standard", s, False) for s in seeds]
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "dev_standard",
                                              jobs, **common)
        _print_summary("development standard-scenario diagnostic (greedy, DEV-ONLY seeds)", csv_path)
        return 0

    raise IntentCLIError(f"unknown eval kind {which!r}")


# The final development acceptance run. Deliberately the thinnest possible
# entry point: there is NO seed argument, NO scenario argument and NO
# episode-count argument, because every one of them is a lever that could
# turn a fixed pass/fail test into a search. The only thing the caller
# chooses is which run directory to read final_ema.pth from.
FINAL_DEV_CHECKPOINT_NAME = "final_ema.pth"


def checkpoint_arm(path: Path) -> str:
    """The belief arm a checkpoint was TRAINED under, read from the file.

    belief_mode selects how human features are built at evaluation time
    (build_intent_human_feature_batch's ``mode``). A mean- or cv-trained
    network scored with ``full`` features is being fed a different input
    distribution than it was fit on, so the ablation would measure that
    mismatch rather than the belief treatment. Deriving the mode from the
    weights themselves means the two can never disagree -- there is no flag
    to set wrong.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    arm = (payload.get("extra") or {}).get("training_arm")
    if arm not in ("full", "mean", "cv"):
        raise IntentCLIError(
            f"{Path(path).name} does not record which arm trained it "
            f"(extra.training_arm={arm!r}); refusing to guess the belief mode")
    return str(arm)


def final_dev_jobs() -> Dict[str, List[Tuple[str, int, bool]]]:
    """The two FROZEN development-acceptance blocks, one per scenario.

    junction runs with shifted=True: 2_400_000-2_400_099 was frozen as the
    HELD-OUT junction variant (shifted speeds, wider fork), which is harder
    than the junction_crowd distribution trained on. The SR >= 0.90 bar is
    defined against that harder variant, so running these seeds nominal
    would be a different -- easier -- test than the one being decided.
    """
    # imported here, not at module level: intent_train_cli imports this
    # module, so the seed blocks are reached the same lazy way the rest of
    # the CLI helpers are.
    from crowd_nav.bayesian_dvl.evaluation_protocol import STANDARD_SELECTION_DEV_SEEDS
    from crowd_nav.bayesian_dvl.junction_scenario import JUNCTION_CROWD_SELECTION_DEV_SEEDS
    return {
        "standard": [("standard", s, False) for s in STANDARD_SELECTION_DEV_SEEDS],
        "junction_crowd": [("junction_crowd", s, True)
                           for s in JUNCTION_CROWD_SELECTION_DEV_SEEDS],
    }


def cmd_eval_final_dev(args) -> int:
    """Score ONE final weight on both frozen development blocks.

    Refuses anything but final_ema.pth. Milestones exist for curves and
    diagnosis; scoring them here would make this a checkpoint search, which
    is exactly what this run is not allowed to be.
    """
    _cli_env()
    ck = Path(args.checkpoint)
    if ck.name != FINAL_DEV_CHECKPOINT_NAME:
        raise IntentCLIError(
            f"eval-final-dev scores the FINAL weight only, got {ck.name!r}; "
            f"milestones are for curves and diagnosis, not for selection")
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    scene_hash = scene_registry_sha256(cfg)
    model = _load_eval_model(ck, cfg, device, grid.table_hash(), scene_hash)
    out_dir = Path(args.out_dir) if args.out_dir else ck.parent / "final_eval_200"
    prov = _provenance(cfg, args, scene_hash, grid.table_hash())
    arm = checkpoint_arm(ck)
    prov.update({"protocol": "final_dev_acceptance", "belief_mode": arm,
                 "checkpoint_sha256": _sha256_file(ck),
                 "decision_rule": "both scenarios success_rate >= 0.90 (point estimate)"})

    jobs = final_dev_jobs()
    paths = {}
    for scenario, job in sorted(jobs.items()):
        # ONE directory per scenario: a shared one lets resume append the
        # second scenario's rows to the first file and average them together.
        csv_path = run_persistent_evaluation(
            args.env_config, model, action_table, out_dir / scenario, "selection_dev", job,
            belief_mode=arm, method="intent_bdvl_final_dev", n_samples=cfg.future_n_samples,
            horizon=cfg.future_horizon, device=str(device), provenance=prov,
            resume=not args.no_resume)
        rows = list(csv.DictReader(open(csv_path)))
        if len(rows) != len(job):
            raise IntentCLIError(f"{scenario}: {len(rows)} rows for {len(job)} jobs")
        seen = [int(r["episode_seed"]) for r in rows]
        if len(set(seen)) != len(job):
            raise IntentCLIError(f"{scenario}: duplicate episode seeds ({len(set(seen))} unique)")
        if set(seen) != {s for _, s, _ in job}:
            raise IntentCLIError(f"{scenario}: episode seeds do not match the frozen block")
        paths[scenario] = csv_path
        _print_summary(f"final dev acceptance -- {scenario}", csv_path)
    print(f"final dev acceptance CSVs -> {out_dir}")
    return 0


def paper_junction_jobs() -> List[Tuple[str, int, bool]]:
    """The junction core-ablation paper block: 2_500_000-2_500_499, shifted.

    Frozen and never touched by training or selection. shifted=True because
    ``paper_test`` is a HELDOUT_GEOMETRY_ROLE -- the same harder variant the
    development-acceptance block uses, not the trained-on distribution.
    """
    from crowd_nav.bayesian_dvl.junction_scenario import JUNCTION_CROWD_PAPER_TEST_SEEDS
    return [("junction_crowd", s, True) for s in JUNCTION_CROWD_PAPER_TEST_SEEDS]


def cmd_eval_paper_junction(args) -> int:
    """Score ONE final weight on the frozen junction paper-test block.

    ``eval-paper`` covers the Test8 negative control; nothing covered this
    block, which is the core ablation the paper's junction claim rests on.
    Same discipline as eval-final-dev: final weight only, no seed, scenario
    or episode-count argument.
    """
    _cli_env()
    ck = Path(args.checkpoint)
    if ck.name != FINAL_DEV_CHECKPOINT_NAME:
        raise IntentCLIError(
            f"the paper junction test scores the FINAL weight only, got {ck.name!r}")
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    scene_hash = scene_registry_sha256(cfg)
    model = _load_eval_model(ck, cfg, device, grid.table_hash(), scene_hash)
    out_dir = Path(args.out_dir) if args.out_dir else ck.parent / "paper_junction"
    prov = _provenance(cfg, args, scene_hash, grid.table_hash())
    arm = checkpoint_arm(ck)
    prov.update({"protocol": "paper_junction_core_ablation", "belief_mode": arm,
                 "checkpoint_sha256": _sha256_file(ck),
                 "seed_block": "2500000-2500499", "shifted": True})

    jobs = paper_junction_jobs()
    csv_path = run_persistent_evaluation(
        args.env_config, model, action_table, out_dir, "paper_junction", jobs,
        belief_mode=arm, method="intent_bdvl_paper_junction", n_samples=cfg.future_n_samples,
        horizon=cfg.future_horizon, device=str(device), provenance=prov,
        resume=not args.no_resume)
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) != len(jobs):
        raise IntentCLIError(f"{len(rows)} rows for {len(jobs)} jobs")
    seen = [int(r["episode_seed"]) for r in rows]
    if set(seen) != {s for _, s, _ in jobs} or len(set(seen)) != len(jobs):
        raise IntentCLIError("episode seeds do not match the frozen paper-test block")
    _print_summary("paper junction core ablation (shifted, 500 frozen seeds)", csv_path)
    return 0


BASELINE_METHODS = ("orca",)


def _orca_action_fn():
    """ORCA driving the robot from the simulator state.

    ``build_formal_scenario_env`` already attaches an ORCA policy to the
    robot -- the BDVL evaluator simply bypasses it and drives the robot
    itself. Here that same policy plans, so the baseline and the arms differ
    in the controller and in nothing else: identical scenario construction,
    identical episode seeds, identical initial states.
    """
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import JointState
    state = {}

    def act(env):
        policy = state.get("policy")
        if policy is None:
            policy = ORCA()
            policy.time_step = FROZEN_VALUES["dt"]
            policy.max_speed = float(env.robot.v_pref)
            policy.multiagent_training = True
            state["policy"] = policy
        return policy.predict(JointState(env.robot.get_full_state(),
                                         [h.get_observable_state() for h in env.humans]))
    return act


def cmd_eval_paper_baseline(args) -> int:
    """Score a classical baseline on the SAME frozen paper protocol.

    Without this the arms' numbers have no scale: whether dense_circle 0.626
    is good or bad is undefined until something else is measured on those
    exact episodes. The baseline has no weights, so it is run ONCE -- the
    episode identities come from the frozen base seed, not from a training
    seed, and the same rows pair with every arm of every seed.
    """
    _cli_env()
    if args.method not in BASELINE_METHODS:
        raise IntentCLIError(f"unknown baseline {args.method!r}, expected one of {list(BASELINE_METHODS)}")
    cfg = load_intent_training_config(args.config)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    scene_hash = scene_registry_sha256(cfg)
    out_dir = Path(args.out_dir)
    # _provenance() is checkpoint-shaped and reads the weight file; a
    # baseline has no weights, so the same fields are recorded WITHOUT
    # inventing a checkpoint for it.
    prov = {
        "protocol": "paper_baseline", "baseline_method": args.method,
        "checkpoint": None, "checkpoint_sha256": None,
        "config_path": cfg.source_path, "config_sha256": cfg.source_sha256,
        "config_content_hash": cfg.content_hash(), "code_hash": code_sha256(),
        "action_grid_hash": grid.table_hash(), "scene_registry_hash": scene_hash,
        "feature_schema": cfg.feature_schema,
        "training_contract_schema": cfg.training_contract_schema,
        "note": "no learned weights; identical scenarios/seeds/initial states as the arms",
    }
    action_fn = _orca_action_fn()

    # Test8 negative control, then the junction core-ablation block -- the
    # same two blocks, in the same protocol, that every arm is scored on.
    for kind, jobs in (("paper_main", paper_main_jobs(base_seed=PAPER_MAIN_BASE_SEED)),
                       ("paper_junction", paper_junction_jobs())):
        csv_path = run_persistent_evaluation(
            args.env_config, None, None, out_dir, kind, jobs,
            # "none", not the "full" default: this controller has no belief at
            # all, and a row labelled baseline_orca:full would read as though
            # it did. It also gives the baseline its own arm_none directory.
            belief_mode="none", method=f"baseline_{args.method}", n_samples=cfg.future_n_samples,
            horizon=cfg.future_horizon, device="cpu", provenance=prov,
            resume=not args.no_resume, action_fn=action_fn)
        rows = list(csv.DictReader(open(csv_path)))
        if len(rows) != len(jobs):
            raise IntentCLIError(f"{kind}: {len(rows)} rows for {len(jobs)} jobs")
        _print_summary(f"paper baseline {args.method} -- {kind}", csv_path)
    return 0


def cmd_ablate(args) -> int:
    _cli_env()
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    scene_hash = scene_registry_sha256(cfg)
    model = _load_eval_model(Path(args.checkpoint), cfg, device, grid.table_hash(), scene_hash)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "results"
    seeds = (list(JUNCTION_CROWD_HELDOUT_SEEDS)[: args.episodes] if args.episodes
             else list(JUNCTION_CROWD_HELDOUT_SEEDS))
    jobs = [("junction_crowd", s, True) for s in seeds]
    arms = (args.arm,) if args.arm else ("full", "mean", "cv", "uniform")
    print("NOTE: this is a FEATURE INTERVENTION on ONE checkpoint -- a mechanism diagnosis.")
    print("      The paper's main ablation requires SEPARATELY TRAINING full/mean/cv under an")
    print("      identical budget (plan section 3.2); 'uniform' is a supplementary control only.")
    for arm in arms:
        csv_path = run_persistent_evaluation(
            args.env_config, model, action_table, out_dir, "ablation", jobs, belief_mode=arm,
            n_samples=cfg.future_n_samples, horizon=cfg.future_horizon, device=str(device),
            provenance=_provenance(cfg, args, scene_hash, grid.table_hash()), resume=not args.no_resume)
        _print_summary(f"ablation arm: {arm}", csv_path)
    return 0
from crowd_nav.bayesian_dvl.intent_config import (  # noqa: E402
    IntentConfigError, load_intent_training_config,
)
from crowd_nav.bayesian_dvl.intent_policy import HUMAN_FEATURE_DIM_V6  # noqa: E402
from crowd_nav.bayesian_dvl.evaluation_protocol import PAPER_MAIN_BASE_SEED  # noqa: E402


def _cli_env():                                   # _LAZY_CLI_IMPORTS
    """Bind the training CLI's shared helpers into this module, lazily.

    intent_train_cli imports this module, so a top-level import here would
    close the cycle. The CLI bodies moved out of it still use its helpers
    (resolve_device, the run-directory conventions, the hashes), so they are
    bound on first CLI use rather than duplicated.
    """
    from crowd_nav.bayesian_dvl import intent_train_cli as t
    g = globals()
    for name in dir(t):
        if not name.startswith("__") and name not in g:
            g[name] = getattr(t, name)
    return t


def build_parser() -> argparse.ArgumentParser:
    t = _cli_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=t.DEFAULT_TRAINING_CONFIG)
    p.add_argument("--env-config", type=Path, default=t.DEFAULT_ENV_CONFIG)
    p.add_argument("--device", type=str, default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "eval-paper", "eval-stress", "eval-dev-standard"):
        e = sub.add_parser(name)
        e.add_argument("--checkpoint", type=Path, required=True)
        e.add_argument("--episodes", type=int, default=None)
        e.add_argument("--out-dir", type=Path, default=None)
        e.add_argument("--no-resume", action="store_true")
        e.add_argument("--base-seed", type=int, default=PAPER_MAIN_BASE_SEED,
                       help="eval-paper only: the frozen Test8 base seed")
    # No --episodes, no --seed, no --scenario: the blocks and their counts
    # are the test, not parameters of it.
    f = sub.add_parser("eval-final-dev")
    f.add_argument("--checkpoint", type=Path, required=True,
                   help=f"must be a {FINAL_DEV_CHECKPOINT_NAME}; milestones are refused")
    f.add_argument("--out-dir", type=Path, default=None)
    f.add_argument("--no-resume", action="store_true")
    b = sub.add_parser("eval-paper-baseline")
    b.add_argument("--method", type=str, required=True, choices=list(BASELINE_METHODS))
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--no-resume", action="store_true")

    j = sub.add_parser("eval-paper-junction")
    j.add_argument("--checkpoint", type=Path, required=True,
                   help=f"must be a {FINAL_DEV_CHECKPOINT_NAME}")
    j.add_argument("--out-dir", type=Path, default=None)
    j.add_argument("--no-resume", action="store_true")
    a = sub.add_parser("ablate")
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--arm", type=str, default=None, choices=["full", "mean", "cv", "uniform"])
    a.add_argument("--episodes", type=int, default=None)
    a.add_argument("--out-dir", type=Path, default=None)
    a.add_argument("--no-resume", action="store_true")
    return p


def main(argv=None) -> int:
    t = _cli_env()
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "ablate":
            return cmd_ablate(args)
        if args.cmd == "eval-final-dev":
            return cmd_eval_final_dev(args)
        if args.cmd == "eval-paper-junction":
            return cmd_eval_paper_junction(args)
        if args.cmd == "eval-paper-baseline":
            return cmd_eval_paper_baseline(args)
        return cmd_eval(args, args.cmd)
    except Exception as exc:
        if type(exc).__name__ not in ("IntentCLIError", "IntentConfigError", "CandidateAuditError"):
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
