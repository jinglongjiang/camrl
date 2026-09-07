#!/usr/bin/env python3
"""Same-state action-sensitivity audit for frozen MM-S1 artifacts.

The three planners never control the audit environment.  A deterministic
``scripted_probe`` reference controller generates one common trajectory;
at every state, K*>1 full posterior, K=1 full posterior, and K*>1 posterior
mean receive the same observation history and the same actually executed
reference action.  This avoids confusing policy-induced state divergence
with a direct effect of the Bayesian representation on action selection.
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
from typing import Dict, Iterable, Optional

import numpy as np

from crowd_nav.bayesian_brne.causal_evaluate import _min_clearance, _observation
from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact
from crowd_nav.bayesian_brne.interaction_protocol import (
    RobotControllerState,
    compute_robot_action,
    make_scenario,
)
from crowd_nav.bayesian_brne.multimodal_evaluate import _make_policy
from crowd_nav.bayesian_brne.multimodal_evaluate import _planner
from crowd_nav.bayesian_brne.nonreciprocal_policy import NonReciprocalBayesianPolicy


METHODS = (
    "full_posterior_selected",
    "full_posterior_k1",
    "posterior_mean_selected",
)
PAIRS = (
    ("full_posterior_selected", "full_posterior_k1", "selected_vs_k1"),
    ("full_posterior_selected", "posterior_mean_selected", "full_vs_mean"),
)
SOURCE_FILES = (
    "crowd_nav/tools/audit_multimodal_action_sensitivity.py",
    "crowd_nav/bayesian_brne/causal_runtime.py",
    "crowd_nav/bayesian_brne/multimodal_evaluate.py",
    "crowd_nav/bayesian_brne/interaction_protocol.py",
    "crowd_nav/bayesian_brne/robot_sampler.py",
    "crowd_nav/bayesian_brne/brne_adapter.py",
    "crowd_nav/bayesian_brne/nonreciprocal_policy.py",
)


def _make_audit_policy(
    method: str,
    selected: CausalResponseArtifact,
    k1: CausalResponseArtifact,
    registry: dict,
    brne_root: str,
    responsibility_mode: str,
):
    if responsibility_mode == "joint_equilibrium":
        return _make_policy(method, selected, k1, registry, brne_root, False)
    if responsibility_mode != "robot_only":
        raise ValueError(f"unknown responsibility mode: {responsibility_mode}")
    if method == "full_posterior_selected":
        artifact, posterior_mean = selected, False
    elif method == "posterior_mean_selected":
        artifact, posterior_mean = selected, True
    elif method == "full_posterior_k1":
        artifact, posterior_mean = k1, False
    else:
        raise ValueError(method)
    policy = NonReciprocalBayesianPolicy(posterior_mean=posterior_mean)
    policy.configure(_planner(registry, brne_root, smoke=False), artifact)
    return policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _belief_diagnostics(policy) -> tuple[float, float]:
    distributions = [
        state.predictive for state in policy.beliefs.tracks.values()
    ]
    if not distributions:
        return 0.0, 1.0
    entropy = []
    max_probability = []
    for distribution in distributions:
        distribution = np.asarray(distribution, dtype=float)
        normalized = distribution / max(float(distribution.sum()), 1e-12)
        denominator = np.log(max(normalized.size, 2))
        entropy.append(
            float(-np.sum(normalized * np.log(np.maximum(normalized, 1e-300))) / denominator)
        )
        max_probability.append(float(np.max(normalized)))
    return float(np.mean(entropy)), float(np.mean(max_probability))


def _run_episode(
    selected: CausalResponseArtifact,
    k1: CausalResponseArtifact,
    registry: dict,
    brne_root: str,
    *,
    split: str,
    scenario: str,
    suite_seed: int,
    episode_index: int,
    max_steps: int,
    responsibility_mode: str,
) -> list[dict]:
    episode_seed = int(suite_seed) * 100000 + int(episode_index)
    env = make_scenario(
        scenario, split, np.random.default_rng(episode_seed), selected.dt,
    )
    reference = RobotControllerState(
        controller_type="scripted_probe",
        brne_root=brne_root,
        brne_rng=np.random.default_rng(episode_seed + 17),
    )
    policies = {
        method: _make_audit_policy(
            method, selected, k1, registry, brne_root, responsibility_mode,
        )
        for method in METHODS
    }
    for policy in policies.values():
        policy.reset(episode_seed)

    records = []
    episode_outcome = "timeout"
    for step in range(max_steps):
        observation = _observation(env, step, episode_seed)
        reference_action, fallback = compute_robot_action(env, reference)
        if fallback is not None:
            raise RuntimeError(
                f"scripted_probe fallback at seed={suite_seed} episode={episode_index} step={step}: {fallback}"
            )
        actions = {}
        diagnostics = {}
        for method, policy in policies.items():
            action = policy.predict(observation)
            actions[method] = np.asarray([action.vx, action.vy], dtype=float)
            diagnostics[method] = policy.last_diagnostics

        selected_entropy, selected_max_probability = _belief_diagnostics(
            policies["full_posterior_selected"]
        )
        pre_clearance = float(_min_clearance(env))
        transition = env.step(reference_action)
        post_clearance = float(_min_clearance(env))
        fsm_conflict = bool(np.any(transition["conflicts"]))

        row = {
            "split": split,
            "scenario": scenario,
            "suite_seed": suite_seed,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "step": step,
            "pre_clearance": pre_clearance,
            "post_clearance": post_clearance,
            "near_state": int(pre_clearance <= 1.0),
            "fsm_conflict": int(fsm_conflict),
            "selected_belief_normalized_entropy": selected_entropy,
            "selected_belief_mean_max_probability": selected_max_probability,
            "robot_px": float(env.robot_pos[0]),
            "robot_py": float(env.robot_pos[1]),
            "robot_vx": float(env.robot_vel[0]),
            "robot_vy": float(env.robot_vel[1]),
            "robot_gx": float(env.robot_goal[0]),
            "robot_gy": float(env.robot_goal[1]),
            "humans_json": json.dumps([
                {
                    "track_id": int(human.track_id),
                    "px": float(human.pos[0]),
                    "py": float(human.pos[1]),
                    "vx": float(human.vel[0]),
                    "vy": float(human.vel[1]),
                    "radius": float(human.radius),
                    "behavior_type": getattr(
                        human.behavior_state.behavior_type,
                        "value",
                        str(human.behavior_state.behavior_type),
                    ),
                }
                for human in env.humans
            ], separators=(",", ":"), sort_keys=True),
            "reference_vx": float(reference_action[0]),
            "reference_vy": float(reference_action[1]),
        }
        for method, action in actions.items():
            row[f"{method}_vx"] = float(action[0])
            row[f"{method}_vy"] = float(action[1])
            row[f"{method}_outer_iterations"] = int(
                diagnostics[method]["outer_iterations"]
            )
            row[f"{method}_outer_converged"] = int(
                diagnostics[method].get(
                    "converged", diagnostics[method].get("outer_converged", False)
                )
            )
            row[f"{method}_max_weight_residual"] = float(
                diagnostics[method]["max_weight_residual"]
            )
            row[f"{method}_solver_converged"] = int(
                bool(diagnostics[method].get("solver_converged", True))
            )
            row[f"{method}_elapsed_ms"] = float(
                diagnostics[method]["elapsed_ms"]
            )
        for left, right, label in PAIRS:
            row[f"{label}_action_l2"] = float(np.linalg.norm(actions[left] - actions[right]))
            left_weights = np.asarray(diagnostics[left]["robot_weights"], dtype=float)
            right_weights = np.asarray(diagnostics[right]["robot_weights"], dtype=float)
            row[f"{label}_weight_l1"] = float(np.sum(np.abs(left_weights - right_weights)))
        records.append(row)

        # The common observation history must contain the action that was
        # actually executed by the reference controller, not each planner's
        # unexecuted proposal.
        for policy in policies.values():
            policy.last_action = np.asarray(reference_action, dtype=float).copy()

        if post_clearance < 0.0:
            episode_outcome = "collision"
            break
        if float(np.linalg.norm(env.robot_goal - env.robot_pos)) <= env.robot_radius:
            episode_outcome = "success"
            break

    for row in records:
        row["reference_episode_outcome"] = episode_outcome
    return records


def _bootstrap_seed_means(
    rows: list[dict], field: str, suite_seeds: Iterable[int],
    *, replicates: int, seed: int,
) -> dict:
    suite_seeds = sorted(set(int(value) for value in suite_seeds))
    represented = sorted({int(row["suite_seed"]) for row in rows})
    if not represented:
        raise ValueError(f"cannot bootstrap empty rows for field {field}")
    per_seed = {
        suite_seed: float(np.mean([
            float(row[field]) for row in rows if int(row["suite_seed"]) == suite_seed
        ]))
        for suite_seed in represented
    }
    values = np.asarray([per_seed[value] for value in represented], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, values.size, size=(replicates, values.size))
    boot = values[sampled].mean(axis=1)
    return {
        "point": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "per_suite_seed": {str(key): value for key, value in per_seed.items()},
        "represented_suite_seeds": represented,
        "missing_suite_seeds": sorted(set(suite_seeds) - set(represented)),
    }


def _subset_summary(
    rows: list[dict], suite_seeds: list[int], *, replicates: int, seed: int,
) -> dict:
    output = {"states": len(rows)}
    for pair_index, (_, _, label) in enumerate(PAIRS):
        field = f"{label}_action_l2"
        weight_field = f"{label}_weight_l1"
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        if values.size == 0:
            output[label] = {"states": 0}
            continue
        metrics = {
            "mean_l2": _bootstrap_seed_means(
                rows, field, suite_seeds, replicates=replicates,
                seed=seed + pair_index * 101,
            ),
            "median_l2": float(np.median(values)),
            "p95_l2": float(np.quantile(values, 0.95)),
            "mean_weight_l1": _bootstrap_seed_means(
                rows, weight_field, suite_seeds, replicates=replicates,
                seed=seed + pair_index * 101 + 71,
            ),
            "median_weight_l1": float(np.median([
                float(row[weight_field]) for row in rows
            ])),
            "p95_weight_l1": float(np.quantile([
                float(row[weight_field]) for row in rows
            ], 0.95)),
        }
        for threshold in (0.001, 0.01, 0.05, 0.10):
            binary_field = f"_{label}_gt_{threshold}"
            enriched = []
            for row in rows:
                copy = dict(row)
                copy[binary_field] = float(row[field]) > threshold
                enriched.append(copy)
            metrics[f"fraction_gt_{threshold}"] = _bootstrap_seed_means(
                enriched, binary_field, suite_seeds, replicates=replicates,
                seed=seed + pair_index * 101 + int(threshold * 1000) + 1,
            )
        output[label] = metrics
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", default="crowd_nav/configs/multimodal_s1_registry.json",
    )
    parser.add_argument(
        "--experiment-root",
        default="runs/bayesian_brne/mm_s1_multimodal_brne_20260805",
    )
    parser.add_argument("--brne-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite-seeds", default="8781,8782,8783,8784,8785")
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--responsibility-mode",
        choices=("joint_equilibrium", "robot_only"),
        default="joint_equilibrium",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path.cwd().resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.registry).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    selected_path = experiment_root / "production" / "base_motion_selected.npz"
    k1_path = experiment_root / "production" / "base_motion_k1.npz"
    registry = json.loads(registry_path.read_text())
    selected = CausalResponseArtifact.load(selected_path)
    k1 = CausalResponseArtifact.load(k1_path)
    if selected.K <= 1 or k1.K != 1:
        raise ValueError(f"expected selected K>1 and baseline K=1, got {selected.K}, {k1.K}")
    suite_seeds = [int(value) for value in args.suite_seeds.split(",") if value]
    if len(suite_seeds) < 2 or len(set(suite_seeds)) != len(suite_seeds):
        raise ValueError("suite seeds must contain at least two unique values")

    records = []
    pilot = registry["evaluation"]["pilot"]
    for split in pilot["splits"]:
        for scenario in pilot["scenarios"]:
            for suite_seed in suite_seeds:
                for episode_index in range(args.episodes_per_seed):
                    records.extend(_run_episode(
                        selected, k1, registry, args.brne_root,
                        split=split, scenario=scenario, suite_seed=suite_seed,
                        episode_index=episode_index, max_steps=args.max_steps,
                        responsibility_mode=args.responsibility_mode,
                    ))
                    print(
                        f"[ACTION-AUDIT] split={split} seed={suite_seed} "
                        f"episode={episode_index} states={len(records)}",
                        flush=True,
                    )
    if not records:
        raise RuntimeError("action audit produced no states")

    with (output / "state_action_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    scopes = {"combined": records}
    scopes.update({
        split: [row for row in records if row["split"] == split]
        for split in pilot["splits"]
    })

    def summarize_scope(scope_rows: list[dict], scope_index: int) -> dict:
        subsets = {
            "all": scope_rows,
            "near": [row for row in scope_rows if row["near_state"]],
            "fsm_conflict": [row for row in scope_rows if row["fsm_conflict"]],
            "uncertain_belief": [
                row for row in scope_rows
                if float(row["selected_belief_normalized_entropy"]) >= 0.5
            ],
        }
        return {
            name: _subset_summary(
                subset_rows, suite_seeds, replicates=args.bootstrap_replicates,
                seed=8811 + scope_index * 10000 + subset_index * 1000,
            )
            for subset_index, (name, subset_rows) in enumerate(subsets.items())
        }

    summary = {
        "audit": "same_state_multimodal_action_sensitivity",
        "status": "COMPLETED",
        "selected_K": selected.K,
        "baseline_K": k1.K,
        "reference_controller": "scripted_probe",
        "responsibility_mode": args.responsibility_mode,
        "suite_seeds": suite_seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "max_steps": args.max_steps,
        "scopes": {
            name: summarize_scope(scope_rows, scope_index)
            for scope_index, (name, scope_rows) in enumerate(scopes.items())
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"provenance source missing: {path}")
        source_hashes[relative] = _sha256(path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(root),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": _git_commit(root),
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "selected_artifact_path": str(selected_path),
        "selected_artifact_sha256": _sha256(selected_path),
        "k1_artifact_path": str(k1_path),
        "k1_artifact_sha256": _sha256(k1_path),
        "source_sha256": source_hashes,
        "outputs_sha256": {
            name: _sha256(output / name)
            for name in ("state_action_records.csv", "summary.json")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["scopes"], indent=2, sort_keys=True))
    print(f"[ACTION-AUDIT] saved to {output}")


if __name__ == "__main__":
    main()
