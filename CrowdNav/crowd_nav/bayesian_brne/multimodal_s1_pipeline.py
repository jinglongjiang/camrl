"""Strict base-motion multimodal gate for MM-S1.

This pipeline intentionally removes robot-action response from the model.
Only the reference branch is fitted; three identical copies are used so the
existing, audited AR-HMM optimizer can be reused without changing its MAP-EM
objective or artifact contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from crowd_nav.bayesian_brne.causal_pair_data import collect_role, load_role
from crowd_nav.bayesian_brne.causal_response_arhmm import (
    CausalFitConfig,
    CausalResponseArtifact,
    CausalSequence,
    expectation,
    fit,
    score_sequences,
    sequences_from_episodes,
)


STAGES = (
    "preflight",
    "collect_train_selection",
    "fit",
    "select_k",
    "collect_predictive_audit",
    "predictive_audit",
    "promote",
    "latency",
    "smoke",
    "pilot",
)


class ScientificStop(RuntimeError):
    pass


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _role_path(output: Path, role: str) -> Path:
    return output / "data" / role


def _artifact_path(output: Path, K: int) -> Path:
    return output / "fits" / f"base_motion_k{K}.npz"


def reference_only_sequence(sequence: CausalSequence) -> CausalSequence:
    """Return a sequence whose likelihood contains no action-response term."""
    value = copy.deepcopy(sequence)
    value.action_a = value.robot_velocity.copy()
    value.action_b = value.robot_velocity.copy()
    value.next_velocity_a = value.next_velocity_ref.copy()
    value.next_velocity_b = value.next_velocity_ref.copy()
    value.validate()
    return value


def reference_only_sequences(sequences: Sequence[CausalSequence]) -> List[CausalSequence]:
    return [reference_only_sequence(sequence) for sequence in sequences]


def _role_sequences(output: Path, registry: dict, role: str) -> List[CausalSequence]:
    spec = registry["roles"][role]
    episodes = load_role(
        _role_path(output, role), spec["suite_seeds"],
        nominal=spec["split"] == "test_nominal",
    )
    expected = len(spec["suite_seeds"]) * int(spec["episodes_per_seed"])
    if len(episodes) != expected:
        raise RuntimeError(f"{role}: expected {expected} episodes, got {len(episodes)}")
    return reference_only_sequences(sequences_from_episodes(episodes))


def _collect(output: Path, registry: dict, role: str, brne_root: str) -> dict:
    spec = registry["roles"][role]
    return collect_role(
        output_dir=_role_path(output, role),
        suite_seeds=spec["suite_seeds"],
        episodes_per_seed=int(spec["episodes_per_seed"]),
        split=spec["split"],
        scenario=registry["scenario"],
        horizon_steps=int(registry["horizon_steps"]),
        dt=float(registry["dt"]),
        brne_root=brne_root,
    )


def _config(registry: dict) -> CausalFitConfig:
    return CausalFitConfig(dt=float(registry["dt"]), **registry["fit"])


def _assert_base_only(artifact: CausalResponseArtifact, tolerance: float = 1e-12) -> None:
    maximum = float(np.max(np.abs(artifact.D)))
    if maximum > tolerance:
        raise RuntimeError(f"base-only artifact has nonzero response matrix: max_abs_D={maximum}")


def _per_suite_nll(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> Dict[int, float]:
    grouped: Dict[int, List[CausalSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.suite_seed, []).append(sequence)
    return {seed: score_sequences(group, artifact)["mean_nll"] for seed, group in grouped.items()}


def _block_ci(values: Dict[int, float], replicates: int, seed: int) -> dict:
    keys = sorted(values)
    array = np.asarray([values[key] for key in keys], dtype=np.float64)
    if len(array) < 2:
        raise RuntimeError("suite-seed block bootstrap needs at least two blocks")
    rng = np.random.default_rng(seed)
    sampled = np.mean(array[rng.integers(0, len(array), (replicates, len(array)))], axis=1)
    return {
        "point": float(array.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n_suite_seeds": len(array),
    }


def _occupancy(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> List[float]:
    result = expectation(sequences, artifact)
    counts = np.sum(np.concatenate(result["gammas"], axis=0), axis=0)
    return (counts / counts.sum()).tolist()


def preflight(output: Path, registry: dict, registry_path: Path, brne_root: str) -> dict:
    seeds = [seed for spec in registry["roles"].values() for seed in spec["suite_seeds"]]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("suite seeds overlap across MM-S1 roles")
    upstream = Path(brne_root) / "socnavbench" / "brne.py"
    if not upstream.exists() or _sha256(upstream) != registry["brne_sha256"]:
        raise RuntimeError("BRNE source hash mismatch")
    sources = [
        Path(__file__),
        Path(__file__).with_name("causal_pair_data.py"),
        Path(__file__).with_name("causal_response_arhmm.py"),
        Path(__file__).with_name("causal_runtime.py"),
        Path(__file__).with_name("multimodal_evaluate.py"),
    ]
    manifest = {
        "experiment_id": registry["experiment_id"],
        "created_at": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "registry_sha256": _sha256(registry_path),
        "source_sha256": {
            str(path.relative_to(_repo_root())): _sha256(path) for path in sources
        },
        "brne_commit": registry["brne_commit"],
        "brne_sha256": registry["brne_sha256"],
        "frozen_cr_s1_result": str(output.parent / "s1_causal_response_20260805"),
    }
    _write_json(output / "manifest.json", manifest)
    return {"status": "PASS", "manifest": manifest}


def fit_models(output: Path, registry: dict) -> dict:
    sequences = _role_sequences(output, registry, "train")
    config = _config(registry)
    records = {}
    for K in range(1, 5):
        path = _artifact_path(output, K)
        if path.exists():
            artifact = CausalResponseArtifact.load(path)
            _assert_base_only(artifact)
        else:
            artifact = fit(sequences, K, config)
            _assert_base_only(artifact)
            artifact.D[:] = 0.0
            artifact.model_card.update({
                "experiment_id": registry["experiment_id"],
                "model_role": "base_motion_only",
                "response_disabled": True,
            })
            artifact.save(path)
        records[str(K)] = {
            "converged": bool(artifact.model_card.get("fit_converged", False)),
            "iterations": int(artifact.model_card.get("em_iterations", 0)),
            "train_nll": score_sequences(sequences, artifact)["mean_nll"],
            "max_abs_D": float(np.max(np.abs(artifact.D))),
        }
        print(f"[MM-FIT] K={K} {records[str(K)]}", flush=True)
    _write_json(output / "fits" / "initial_fit_summary.json", records)
    return {"status": "PASS", "models": records}


def select_k(output: Path, registry: dict) -> dict:
    train = _role_sequences(output, registry, "train")
    selection = _role_sequences(output, registry, "selection")
    config = _config(registry)
    artifacts = {K: CausalResponseArtifact.load(_artifact_path(output, K)) for K in range(1, 5)}
    for artifact in artifacts.values():
        _assert_base_only(artifact)
    nll = {K: _per_suite_nll(selection, artifact) for K, artifact in artifacts.items()}
    boundary = _block_ci(
        {seed: nll[3][seed] - nll[4][seed] for seed in nll[3]},
        int(registry["gate"]["bootstrap_replicates"]),
        int(registry["gate"]["bootstrap_seed"]) + 10,
    )
    min_occupancy = float(registry["gate"]["min_mode_occupancy"])
    boundary_valid = all(
        artifacts[K].model_card.get("fit_converged", False)
        and min(_occupancy(selection, artifacts[K])) >= min_occupancy
        for K in (3, 4)
    )
    expanded = boundary_valid and boundary["ci_low"] > 0.0
    if expanded:
        for K in (5, 6):
            path = _artifact_path(output, K)
            if path.exists():
                artifact = CausalResponseArtifact.load(path)
                _assert_base_only(artifact)
            else:
                artifact = fit(train, K, config)
                _assert_base_only(artifact)
                artifact.D[:] = 0.0
                artifact.model_card.update({
                    "experiment_id": registry["experiment_id"],
                    "model_role": "base_motion_only",
                    "response_disabled": True,
                })
                artifact.save(path)
            artifacts[K] = artifact
            nll[K] = _per_suite_nll(selection, artifact)
            print(f"[MM-FIT-BOUNDARY] K={K}", flush=True)

    records = {}
    for K, artifact in artifacts.items():
        occupancy = _occupancy(selection, artifact)
        values = np.asarray(list(nll[K].values()))
        records[K] = {
            "mean_nll": float(values.mean()),
            "se_nll": float(values.std(ddof=1) / np.sqrt(len(values))),
            "occupancy": occupancy,
            "converged": bool(artifact.model_card.get("fit_converged", False)),
            "eligible": bool(
                artifact.model_card.get("fit_converged", False)
                and min(occupancy) >= min_occupancy
            ),
        }
    eligible = [K for K, record in records.items() if record["eligible"]]
    if not eligible:
        raise ScientificStop("no converged base-motion K satisfies occupancy gate")
    best = min(eligible, key=lambda K: records[K]["mean_nll"])
    threshold = records[best]["mean_nll"] + records[best]["se_nll"]
    selected = min(K for K in eligible if records[K]["mean_nll"] <= threshold)
    result = {
        "status": "PASS",
        "selected_k": selected,
        "best_k": best,
        "one_se_threshold": threshold,
        "boundary_k4_vs_k3": boundary,
        "expanded_to_k6": expanded,
        "models": {str(K): record for K, record in records.items()},
    }
    _write_json(output / "selection_result.json", result)
    return result


def predictive_audit(output: Path, registry: dict) -> dict:
    selected = int(_load_json(output / "selection_result.json")["selected_k"])
    full = CausalResponseArtifact.load(_artifact_path(output, selected))
    k1 = CausalResponseArtifact.load(_artifact_path(output, 1))
    _assert_base_only(full)
    _assert_base_only(k1)
    sequences = _role_sequences(output, registry, "predictive_audit")
    full_nll = _per_suite_nll(sequences, full)
    k1_nll = _per_suite_nll(sequences, k1)
    advantage = _block_ci(
        {seed: k1_nll[seed] - full_nll[seed] for seed in full_nll},
        int(registry["gate"]["bootstrap_replicates"]),
        int(registry["gate"]["bootstrap_seed"]) + 20,
    )
    passed = (
        selected > 1
        and full.model_card.get("fit_converged", False)
        and advantage["ci_low"] > 0.0
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "selected_k": selected,
        "selected_vs_k1_reference_nll": advantage,
        "max_abs_D": float(np.max(np.abs(full.D))),
    }
    _write_json(output / "predictive_audit_result.json", result)
    return result


def promote(output: Path, registry: dict) -> dict:
    audit = _load_json(output / "predictive_audit_result.json")
    if audit["status"] != "PASS":
        raise ScientificStop("MM-S1 promotion denied by predictive audit")
    selected = int(_load_json(output / "selection_result.json")["selected_k"])
    manifest = _load_json(output / "manifest.json")
    promoted = {}
    for name, K in (("selected", selected), ("k1", 1)):
        artifact = CausalResponseArtifact.load(_artifact_path(output, K))
        artifact.D[:] = 0.0
        _assert_base_only(artifact)
        artifact.model_card.update({
            "scientific_gate": "GO",
            "model_role": "base_motion_only",
            "response_disabled": True,
            "predictive_audit_sha256": _sha256(output / "predictive_audit_result.json"),
            "registry_sha256": manifest["registry_sha256"],
            "source_sha256": manifest["source_sha256"],
        })
        path = output / "production" / f"base_motion_{name}.npz"
        artifact.save(path, tier="production")
        promoted[name] = {"K": K, "path": str(path), "sha256": _sha256(path)}
    return {"status": "PASS", "artifacts": promoted}


def run_stage(stage: str, output: Path, registry: dict, registry_path: Path, brne_root: str) -> dict:
    if stage == "preflight":
        return preflight(output, registry, registry_path, brne_root)
    if stage == "collect_train_selection":
        return {
            "status": "PASS",
            "train": _collect(output, registry, "train", brne_root),
            "selection": _collect(output, registry, "selection", brne_root),
        }
    if stage == "fit":
        return fit_models(output, registry)
    if stage == "select_k":
        return select_k(output, registry)
    if stage == "collect_predictive_audit":
        return {"status": "PASS", "predictive_audit": _collect(output, registry, "predictive_audit", brne_root)}
    if stage == "predictive_audit":
        return predictive_audit(output, registry)
    if stage == "promote":
        return promote(output, registry)
    if stage in ("latency", "smoke", "pilot"):
        from crowd_nav.bayesian_brne.multimodal_evaluate import run_evaluation_stage
        return run_evaluation_stage(stage, output, registry, brne_root)
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brne-root", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    registry_path = Path(args.registry).resolve()
    registry = _load_json(registry_path)
    started = time.time()
    try:
        result = run_stage(args.stage, output, registry, registry_path, args.brne_root)
        marker = {"stage": args.stage, "elapsed_seconds": time.time() - started, **result}
        _write_json(output / "stages" / f"{args.stage}.json", marker)
        print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
        if result.get("status") == "FAIL":
            raise ScientificStop(f"stage {args.stage} failed scientific gate")
    except ScientificStop as error:
        print(f"[MM-S1-SCIENTIFIC-STOP] {error}", file=sys.stderr, flush=True)
        raise SystemExit(20)


if __name__ == "__main__":
    main()
