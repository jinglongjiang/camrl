#!/usr/bin/env python3
"""D1-D4 candidate feasibility and decision-oracle audit.

The script uses a scripted reference trajectory only to expose common
snapshots.  Every candidate is then replayed from a deep-copied snapshot with
the real FSM transition kernel.  Ground-truth branch outcomes are written to
disk and are never passed to a model score or a deployment policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from crowd_nav.bayesian_brne.causal_evaluate import _observation
from crowd_nav.bayesian_brne.causal_runtime import CausalBeliefBank
from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact
from crowd_nav.bayesian_brne.decision_oracle import (
    _integrate_actions,
    choose_prediction,
    clone_env,
    make_candidate_sets,
    predict_scores,
    rollout_candidate,
    summarize_true_results,
    true_results,
    weighted_sequence,
)
from crowd_nav.bayesian_brne.interaction_protocol import (
    RobotControllerState,
    compute_robot_action,
    make_scenario,
)
from crowd_nav.bayesian_brne.multimodal_evaluate import _planner
from crowd_nav.bayesian_brne.nonreciprocal_policy import NonReciprocalBayesianPolicy


METHODS = (
    ("k4_full", False, False),
    ("k4_mean", True, False),
    ("k1_full", False, False),
    ("cv", False, True),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _make_policy(artifact, original_registry, brne_root, *, horizon, samples, posterior_mean):
    config = replace(
        _planner(original_registry, brne_root, smoke=False),
        horizon_steps=horizon,
        num_samples=samples,
    )
    policy = NonReciprocalBayesianPolicy(posterior_mean=posterior_mean)
    policy.configure(config, artifact)
    return policy


def _env_state(env) -> dict:
    return {
        "robot": {
            "px": float(env.robot_pos[0]), "py": float(env.robot_pos[1]),
            "vx": float(env.robot_vel[0]), "vy": float(env.robot_vel[1]),
            "gx": float(env.robot_goal[0]), "gy": float(env.robot_goal[1]),
            "radius": float(env.robot_radius), "v_pref": float(env.robot_pref_speed),
        },
        "humans": [
            {
                "track_id": int(human.track_id),
                "px": float(human.pos[0]), "py": float(human.pos[1]),
                "vx": float(human.vel[0]), "vy": float(human.vel[1]),
                "gx": float(human.goal[0]), "gy": float(human.goal[1]),
                "radius": float(human.radius), "pref_speed": float(human.pref_speed),
                "behavior_state": {
                    "behavior_type": human.behavior_state.behavior_type.value,
                    "stop_timer": int(human.behavior_state.stop_timer),
                    "goal_switched": bool(human.behavior_state.goal_switched),
                    "local_goal": None if human.behavior_state.local_goal is None else [
                        float(value) for value in human.behavior_state.local_goal
                    ],
                },
            }
            for human in env.humans
        ],
    }


def _true_choice(results: list) -> int:
    return min(
        range(len(results)),
        key=lambda index: (
            bool(results[index].collision),
            -float(results[index].min_clearance),
            -float(results[index].progress),
        ),
    )


def _row_regret(predicted: list[dict], truth: list, chosen: int, oracle: int) -> dict:
    return {
        "collision_regret": int(bool(truth[chosen].collision) and not truth[oracle].collision),
        "clearance_regret": float(
            max(0.0, float(truth[oracle].min_clearance) - float(truth[chosen].min_clearance))
        ),
        "safe_hit": int(not truth[chosen].collision),
        "chosen": int(chosen),
        "oracle": int(oracle),
        "predicted_collision_probability": float(predicted[chosen]["collision_probability"]),
        "predicted_clearance_q05": float(predicted[chosen]["clearance_q05"]),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scan_seed_conflicts(
    root: Path, requested: list[int], output: Path,
) -> list[dict]:
    """Scan persisted formal artifacts before D0 seeds are admitted.

    Explicitly marked ``smoke`` directories are engineering checks, not formal
    episode identities.  They intentionally reuse a fixed smoke seed and must
    not block the frozen formal queue.  All other persisted artifacts remain
    hard conflicts.
    """
    requested = {int(value) for value in requested}
    conflicts = []
    runs = root / "runs"
    if not runs.exists():
        return conflicts
    for path in runs.rglob("*"):
        if not path.is_file() or output in path.parents:
            continue
        if "smoke" in path.parts and "decision_oracle_sm_brne_20260805" in path.parts:
            continue
        if path.suffix not in {".json", ".csv", ".jsonl"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        found = []
        for seed in sorted(requested):
            if f'"suite_seed": {seed}' in text or f'"suite_seed":{seed}' in text or f"suite_seed,{seed}" in text:
                found.append(seed)
        if found:
            conflicts.append({"path": str(path), "suite_seeds": found})
    (output / "seed_conflict_report.json").write_text(
        json.dumps({"requested": sorted(requested), "conflicts": conflicts}, indent=2, sort_keys=True) + "\n"
    )
    if conflicts:
        raise RuntimeError(f"D0 seed conflict detected; see {output / 'seed_conflict_report.json'}")
    return conflicts


def _oracle_episode(env, *, horizon, num_legacy, max_steps, seed, safe_distance, brne_root):
    """Receding-horizon upper bound using true branch outcomes only."""
    minimum = float("inf")
    collision = False
    for step in range(max_steps):
        sets = make_candidate_sets(
            env, horizon=horizon, num_legacy=num_legacy, seed=seed + step, brne_root=brne_root,
        )
        branches = true_results( env, sets["structured_candidates"], safe_distance=safe_distance)
        selected = _true_choice(branches)
        action = sets["structured_candidates"].actions[selected, 0]
        env.step(action)
        after = min(
            [float(np.linalg.norm(human.pos - env.robot_pos) - human.radius - env.robot_radius)
             for human in env.humans] or [float("inf")]
        )
        minimum = min(minimum, after)
        if after < 0.0:
            collision = True
            return {"outcome": "collision", "steps": step + 1, "minimum_clearance": minimum}
        if float(np.linalg.norm(env.robot_goal - env.robot_pos)) <= env.robot_radius:
            return {"outcome": "success", "steps": step + 1, "minimum_clearance": minimum}
    return {"outcome": "timeout", "steps": max_steps, "minimum_clearance": minimum}


def _run_episode(
    selected, k1, original_registry, brne_root, *, split, scenario, suite_seed,
    episode_index, max_steps, max_audit_states, horizon, num_legacy, prediction_samples,
    safe_distance, include_oracle,
):
    episode_seed = int(suite_seed) * 100000 + int(episode_index)
    env = make_scenario(scenario, split, np.random.default_rng(episode_seed), selected.dt)
    trackers = {
        "k4_full": CausalBeliefBank(selected, max_missed_steps=5),
        "k4_mean": CausalBeliefBank(selected, max_missed_steps=5),
        "k1_full": CausalBeliefBank(k1, max_missed_steps=5),
    }
    for tracker in trackers.values():
        tracker.reset()
    reference = RobotControllerState(
        controller_type="scripted_probe", brne_root=brne_root,
        brne_rng=np.random.default_rng(episode_seed + 17),
    )
    state_rows = []
    candidate_rows = []
    conflict_count = 0
    last_action = np.zeros(2, dtype=np.float64)
    for step in range(max_steps):
        observation = _observation(env, step, episode_seed)
        reference_action, fallback = compute_robot_action(env, reference)
        if fallback is not None:
            raise RuntimeError(f"scripted_probe fallback at {episode_seed}/{step}: {fallback}")
        for tracker in trackers.values():
            tracker.update(observation, last_action)
        probe = clone_env(env)
        transition = probe.step(reference_action)
        conflict = bool(np.any(transition["conflicts"]))
        if conflict and conflict_count < max_audit_states:
            conflict_count += 1
            sets = make_candidate_sets(
                env, horizon=horizon, num_legacy=num_legacy, seed=episode_seed + step, brne_root=brne_root,
            )
            truth_by_set = {
                name: true_results(env, candidate_set, safe_distance=safe_distance)
                for name, candidate_set in sets.items()
            }
            summary = {
                name: summarize_true_results(values) for name, values in truth_by_set.items()
            }
            selected_set = sets["structured_candidates"]
            oracle_index = _true_choice(truth_by_set["structured_candidates"])
            state = {
                "experiment_state_id": f"{split}:{suite_seed}:{episode_index}:{step}",
                "split": split, "scenario": scenario, "suite_seed": suite_seed,
                "episode_index": episode_index, "episode_seed": episode_seed, "step": step,
                "fsm_conflict": int(conflict), "candidate_summary_json": json.dumps(summary, sort_keys=True),
                "snapshot_json": json.dumps(_env_state(env), sort_keys=True, separators=(",", ":")),
                "oracle_structured_index": oracle_index,
            }
            predicted_by_method = {}
            for method, posterior_mean, use_cv in METHODS:
                if use_cv:
                    predicted = predict_scores(
                        env, selected_set, selected, trackers["k4_full"],
                        episode_seed=episode_seed, step=step, num_samples=prediction_samples,
                        brne_root=brne_root, use_cv=True, safe_distance=safe_distance,
                    )
                else:
                    artifact = selected if method != "k1_full" else k1
                    predicted = predict_scores(
                        env, selected_set, artifact, trackers[method],
                        episode_seed=episode_seed, step=step, num_samples=prediction_samples,
                        brne_root=brne_root, posterior_mean=posterior_mean,
                        safe_distance=safe_distance,
                    )
                predicted_by_method[method] = predicted
                chosen = choose_prediction(predicted)
                regret = _row_regret(predicted, truth_by_set["structured_candidates"], chosen, oracle_index)
                for key, value in regret.items():
                    state[f"{method}_{key}"] = value
                state[f"{method}_candidate_count"] = len(predicted)
                for candidate_index, (pred, truth) in enumerate(zip(predicted, truth_by_set["structured_candidates"])):
                    candidate_rows.append({
                        "experiment_state_id": state["experiment_state_id"],
                        "candidate_set": "structured_candidates", "candidate_index": candidate_index,
                        "method": method, "true_collision": int(truth.collision),
                        "true_min_clearance": truth.min_clearance, "true_progress": truth.progress,
                        "predicted_collision_probability": pred["collision_probability"],
                        "predicted_clearance_q05": pred["clearance_q05"],
                        "predicted_clearance_mean": pred["clearance_mean"],
                        "predicted_progress": pred["progress"],
                    })
            legacy = sets["legacy_candidates"]
            # This is the same weighted-action failure mode under audit, but
            # scored from the persisted per-candidate K4 costs.  The full
            # deployment outer loop is intentionally not called at every
            # reference step; D1-D4 are an audit, not a second pilot.
            legacy_predicted = predict_scores(
                env, legacy, selected, trackers["k4_full"],
                episode_seed=episode_seed, step=step, num_samples=prediction_samples,
                brne_root=brne_root, safe_distance=safe_distance,
            )
            costs = np.asarray([
                float(item["comfort_collision_probability"])
                for item in legacy_predicted
            ])
            log_weights = -100.0 * costs
            log_weights -= float(np.max(log_weights))
            weights = np.exp(log_weights)
            weights /= max(float(weights.sum()), 1e-12)
            weighted = _integrate_actions(
                env, weighted_sequence(weights, legacy)[None, :, :], name="legacy_weighted_sequence"
            )
            weighted_truth = rollout_candidate(env, weighted.actions[0], safe_distance=safe_distance)
            state["legacy_weighted_collision"] = int(weighted_truth.collision)
            state["legacy_weighted_min_clearance"] = weighted_truth.min_clearance
            state["legacy_weighted_progress"] = weighted_truth.progress
            state["weighted_nearest_candidate_l2"] = float(np.min(
                np.linalg.norm(legacy.actions[:, 0] - weighted.actions[0, 0], axis=1)
            ))
            state["weighted_collision_when_safe_exists"] = int(
                weighted_truth.collision and summary["structured_candidates"]["safe_candidate_exists"]
            )
            state_rows.append(state)
        last_action = np.asarray(reference_action, dtype=float).copy()
        env.step(reference_action)
        if float(np.linalg.norm(env.robot_goal - env.robot_pos)) <= env.robot_radius:
            break
    oracle_row = None
    if include_oracle:
        oracle_env = make_scenario(scenario, split, np.random.default_rng(episode_seed), selected.dt)
        oracle_row = _oracle_episode(
        oracle_env, horizon=horizon, num_legacy=num_legacy, max_steps=max_steps,
            seed=episode_seed + 7901, safe_distance=safe_distance, brne_root=brne_root,
        )
    return state_rows, candidate_rows, oracle_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="crowd_nav/configs/decision_oracle_registry.json")
    parser.add_argument("--original-registry", default="crowd_nav/configs/multimodal_s1_registry.json")
    parser.add_argument("--brne-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--suite-seed", type=int, default=None,
        help="Run one frozen suite seed as an execution shard; merge shards before D4.",
    )
    parser.add_argument(
        "--split", choices=("test_nominal", "test_heldout_interactive"), default=None,
        help="Run one frozen split as an execution shard; merge all seed/split shards before D4.",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text())
    original_registry = json.loads(Path(args.original_registry).resolve().read_text())
    for relative_path, expected_hash in registry["source_sha256"].items():
        source_path = root / relative_path
        if _sha256(source_path) != expected_hash:
            raise RuntimeError(f"source hash mismatch for frozen D0 file: {relative_path}")
    selected_path = root / registry["frozen_artifacts"]["selected_path"]
    k1_path = root / registry["frozen_artifacts"]["k1_path"]
    if _sha256(selected_path) != registry["frozen_artifacts"]["selected_sha256"]:
        raise RuntimeError("selected artifact hash mismatch")
    if _sha256(k1_path) != registry["frozen_artifacts"]["k1_sha256"]:
        raise RuntimeError("K1 artifact hash mismatch")
    selected = CausalResponseArtifact.load(selected_path)
    k1 = CausalResponseArtifact.load(k1_path)
    spec = dict(registry["data_protocol"])
    if args.suite_seed is not None:
        requested_seed = int(args.suite_seed)
        frozen_seeds = [int(value) for value in spec["suite_seeds"]]
        if requested_seed not in frozen_seeds:
            raise ValueError(f"suite seed {requested_seed} is not in frozen registry: {frozen_seeds}")
        spec["suite_seeds"] = [requested_seed]
    if args.split is not None:
        if args.split not in spec["splits"]:
            raise ValueError(f"split {args.split} is not in frozen registry: {spec['splits']}")
        spec["splits"] = [args.split]
    if args.smoke:
        spec.update(suite_seeds=[8911], episodes_per_seed=1, max_steps=8, max_audit_states_per_episode=2,
                    horizon_steps=4, legacy_candidates=8, prediction_samples=4, bootstrap_replicates=100)
        if args.suite_seed is not None and int(args.suite_seed) != 8911:
            raise ValueError("--smoke only supports its fixed smoke seed 8911")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _scan_seed_conflicts(
        root, [int(value) for value in spec["suite_seeds"]], output,
    )
    all_states, all_candidates, all_oracle = [], [], []
    for split in spec["splits"]:
        for suite_seed in spec["suite_seeds"]:
            for episode_index in range(int(spec["episodes_per_seed"])):
                states, candidates, oracle = _run_episode(
                    selected, k1, original_registry, args.brne_root,
                    split=split, scenario=spec["scenario"], suite_seed=int(suite_seed), episode_index=episode_index,
                    max_steps=int(spec["max_steps"]), max_audit_states=int(spec["max_audit_states_per_episode"]),
                    horizon=int(spec["horizon_steps"]), num_legacy=int(spec["legacy_candidates"]),
                    prediction_samples=int(spec["prediction_samples"]),
                    safe_distance=float(registry["physics"]["safe_distance"]), include_oracle=True,
                )
                all_states.extend(states); all_candidates.extend(candidates)
                all_oracle.append({"split": split, "suite_seed": int(suite_seed), "episode_index": episode_index, **(oracle or {})})
                print(f"[DECISION-ORACLE] split={split} seed={suite_seed} episode={episode_index} states={len(states)}", flush=True)
    _write_csv(output / "state_records.csv", all_states)
    _write_csv(output / "candidate_records.csv", all_candidates)
    _write_csv(output / "oracle_episode_records.csv", all_oracle)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(root), "hostname": socket.gethostname(), "platform": platform.platform(),
        "git_commit": _git_commit(root), "registry_sha256": _sha256(registry_path),
        "selected_sha256": _sha256(selected_path), "k1_sha256": _sha256(k1_path),
        "outputs_sha256": {
            name: _sha256(output / name) for name in ("state_records.csv", "candidate_records.csv", "oracle_episode_records.csv")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "status.json").write_text(json.dumps({"status": "completed", "states": len(all_states), "candidates": len(all_candidates)}, indent=2) + "\n")
    print(json.dumps({"status": "completed", "states": len(all_states), "candidates": len(all_candidates), "oracle_episodes": len(all_oracle)}, indent=2))


if __name__ == "__main__":
    main()
