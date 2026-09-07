#!/usr/bin/env python3
"""Paired repaired pilot using robot-responsible Bayesian planning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from crowd_nav.bayesian_brne.causal_evaluate import _min_clearance, _observation
from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact
from crowd_nav.bayesian_brne.multimodal_evaluate import _pilot_gate, _planner, _summary
from crowd_nav.bayesian_brne.nonreciprocal_policy import NonReciprocalBayesianPolicy


METHODS = (
    "robot_only_full_selected",
    "robot_only_posterior_mean_selected",
    "robot_only_full_k1",
)
SOURCE_FILES = (
    "crowd_nav/tools/run_multimodal_repaired_pilot.py",
    "crowd_nav/bayesian_brne/nonreciprocal_policy.py",
    "crowd_nav/bayesian_brne/causal_runtime.py",
    "crowd_nav/bayesian_brne/robot_sampler.py",
    "crowd_nav/bayesian_brne/interaction_protocol.py",
)


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


def _make_policy(
    method: str,
    selected: CausalResponseArtifact,
    k1: CausalResponseArtifact,
    original_registry: dict,
    brne_root: str,
) -> NonReciprocalBayesianPolicy:
    if method == "robot_only_full_selected":
        artifact, posterior_mean = selected, False
    elif method == "robot_only_posterior_mean_selected":
        artifact, posterior_mean = selected, True
    elif method == "robot_only_full_k1":
        artifact, posterior_mean = k1, False
    else:
        raise ValueError(f"unknown repaired-pilot method: {method}")
    policy = NonReciprocalBayesianPolicy(posterior_mean=posterior_mean)
    policy.configure(_planner(original_registry, brne_root, smoke=False), artifact)
    return policy


def _run_episode(
    method: str,
    selected: CausalResponseArtifact,
    k1: CausalResponseArtifact,
    original_registry: dict,
    brne_root: str,
    *,
    scenario: str,
    split: str,
    suite_seed: int,
    episode_index: int,
    max_steps: int,
) -> dict:
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario

    episode_seed = int(suite_seed) * 100000 + int(episode_index)
    env = make_scenario(
        scenario, split, np.random.default_rng(episode_seed), selected.dt,
    )
    policy = _make_policy(method, selected, k1, original_registry, brne_root)
    policy.reset(episode_seed)
    minimum_clearance = float(_min_clearance(env))
    elapsed = []
    outcome = "timeout"
    steps = max_steps
    for step in range(max_steps):
        started = time.perf_counter()
        action = policy.predict(_observation(env, step, episode_seed))
        elapsed.append((time.perf_counter() - started) * 1000.0)
        env.step(np.asarray([action.vx, action.vy], dtype=float))
        minimum_clearance = min(minimum_clearance, float(_min_clearance(env)))
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
        "fallback_count": 0,
        "mean_decision_ms": float(np.mean(elapsed)),
        "p95_decision_ms": float(np.quantile(elapsed, 0.95)),
    }


def _mapped_gate(records: list, repaired_registry: dict) -> dict:
    mapping = {
        "robot_only_full_selected": "full_posterior_selected",
        "robot_only_posterior_mean_selected": "posterior_mean_selected",
        "robot_only_full_k1": "full_posterior_k1",
    }
    mapped = []
    for row in records:
        copy = dict(row)
        copy["method"] = mapping[row["method"]]
        mapped.append(copy)
    gate_registry = {
        "gate": {
            "bootstrap_seed": 8841,
            "bootstrap_replicates": int(
                repaired_registry["repaired_pilot"]["bootstrap_replicates"]
            ),
        }
    }
    return _pilot_gate(mapped, gate_registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="crowd_nav/configs/multimodal_repaired_eval_registry.json",
    )
    parser.add_argument(
        "--original-registry",
        default="crowd_nav/configs/multimodal_s1_registry.json",
    )
    parser.add_argument(
        "--gate-result",
        default=(
            "runs/bayesian_brne/mm_s1_multimodal_brne_20260805/diagnostics/"
            "repaired_gate_result.json"
        ),
    )
    parser.add_argument("--brne-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.registry).resolve()
    original_registry_path = Path(args.original_registry).resolve()
    registry = json.loads(registry_path.read_text())
    original_registry = json.loads(original_registry_path.read_text())
    gate_path = Path(args.gate_result).resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "PASS" or not gate.get("eligible_for_repaired_pilot"):
        raise RuntimeError(f"repaired pilot is blocked by gate result: {gate_path}")

    selected_path = root / registry["frozen_artifacts"]["selected_path"]
    k1_path = root / registry["frozen_artifacts"]["k1_path"]
    for name, path in (("selected", selected_path), ("k1", k1_path)):
        expected = registry["frozen_artifacts"][f"{name}_sha256"]
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen {name} artifact hash mismatch: {path}")
    selected = CausalResponseArtifact.load(selected_path)
    k1 = CausalResponseArtifact.load(k1_path)

    spec = dict(registry["repaired_pilot"])
    if args.smoke:
        spec["suite_seeds"] = spec["suite_seeds"][:1]
        spec["episodes_per_seed"] = 1
        spec["max_steps"] = 4
    records = []
    for split in spec["splits"]:
        for suite_seed in spec["suite_seeds"]:
            for episode_index in range(int(spec["episodes_per_seed"])):
                for method in METHODS:
                    row = _run_episode(
                        method, selected, k1, original_registry, args.brne_root,
                        scenario=spec["scenario"], split=split,
                        suite_seed=int(suite_seed), episode_index=episode_index,
                        max_steps=int(spec["max_steps"]),
                    )
                    records.append(row)
                    print(
                        f"[REPAIRED-PILOT] n={len(records)} method={method} "
                        f"split={split} seed={suite_seed} episode={episode_index} "
                        f"outcome={row['outcome']}",
                        flush=True,
                    )
    with (output / "episode_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "status": "SMOKE" if args.smoke else "COMPLETED",
        "execution_contract": "nonreciprocal_robot_responsible",
        "records": len(records),
        "summary": _summary(records),
        "gate": None if args.smoke else _mapped_gate(records, registry),
    }
    if not args.smoke:
        result["status"] = result["gate"]["status"]
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    source_hashes = {relative: _sha256(root / relative) for relative in SOURCE_FILES}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(root),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(root),
        "registry_sha256": _sha256(registry_path),
        "original_registry_sha256": _sha256(original_registry_path),
        "gate_result_sha256": _sha256(gate_path),
        "selected_artifact_sha256": _sha256(selected_path),
        "k1_artifact_sha256": _sha256(k1_path),
        "source_sha256": source_hashes,
        "outputs_sha256": {
            name: _sha256(output / name)
            for name in ("episode_records.csv", "summary.json")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[REPAIRED-PILOT] saved to {output}")


if __name__ == "__main__":
    main()
