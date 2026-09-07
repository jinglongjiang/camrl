"""Resumable stages for the pre-registered CR-S1 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_brne.causal_pair_data import collect_role, load_role
from crowd_nav.bayesian_brne.causal_response_arhmm import (
    CausalFitConfig, CausalResponseArtifact, CausalSequence, expectation, fit,
    response_metrics, score_sequences, sequences_from_episodes,
)


STAGES = (
    "preflight", "collect_train_selection", "fit", "select_k",
    "collect_necessity", "necessity", "collect_audit", "audit", "promote",
    "latency", "smoke", "formal_eval",
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


def _config(registry: dict) -> CausalFitConfig:
    return CausalFitConfig(dt=float(registry["dt"]), **registry["fit"])


def _role_path(output: Path, role: str) -> Path:
    return output / "data" / role


def _artifact_path(output: Path, K: int) -> Path:
    return output / "fits" / f"causal_response_k{K}.npz"


def _role_sequences(output: Path, registry: dict, role: str) -> List[CausalSequence]:
    spec = registry["roles"][role]
    episodes = load_role(
        _role_path(output, role), spec["suite_seeds"], nominal=spec["split"] == "test_nominal",
    )
    expected = len(spec["suite_seeds"]) * int(spec["episodes_per_seed"])
    if len(episodes) != expected:
        raise RuntimeError(f"{role}: expected {expected} episodes, got {len(episodes)}")
    return sequences_from_episodes(episodes)


def _collect(output: Path, registry: dict, role: str, brne_root: str) -> dict:
    spec = registry["roles"][role]
    return collect_role(
        output_dir=_role_path(output, role), suite_seeds=spec["suite_seeds"],
        episodes_per_seed=int(spec["episodes_per_seed"]), split=spec["split"],
        scenario=registry["scenario"], horizon_steps=int(registry["horizon_steps"]),
        dt=float(registry["dt"]), brne_root=brne_root,
    )


def _per_suite_nll(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> Dict[int, float]:
    grouped: Dict[int, List[CausalSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.suite_seed, []).append(sequence)
    return {seed: score_sequences(values, artifact)["mean_nll"] for seed, values in grouped.items()}


def _block_ci(values: Dict[int, float], replicates: int, seed: int) -> dict:
    seeds = sorted(values)
    array = np.asarray([values[item] for item in seeds], dtype=np.float64)
    if len(array) < 2:
        raise RuntimeError("suite-seed bootstrap needs at least two blocks")
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(array[rng.integers(0, len(array), size=(replicates, len(array)))], axis=1)
    return {
        "point": float(array.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_suite_seeds": len(array),
    }


def _nll_advantage(sequences: Sequence[CausalSequence], full: CausalResponseArtifact,
                   comparator: CausalResponseArtifact, registry: dict, offset: int = 0) -> dict:
    full_values = _per_suite_nll(sequences, full)
    comparator_values = _per_suite_nll(sequences, comparator)
    difference = {seed: comparator_values[seed] - full_values[seed] for seed in full_values}
    return _block_ci(
        difference, int(registry["gate"]["bootstrap_replicates"]),
        int(registry["gate"]["bootstrap_seed"]) + offset,
    )


def _mse_advantage(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact,
                   registry: dict, offset: int = 0) -> dict:
    grouped: Dict[int, List[CausalSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.suite_seed, []).append(sequence)
    values = {}
    for seed, group in grouped.items():
        metrics = response_metrics(group, artifact)
        values[seed] = metrics["relative_mse_improvement"]
    return _block_ci(
        values, int(registry["gate"]["bootstrap_replicates"]),
        int(registry["gate"]["bootstrap_seed"]) + offset,
    )


def _response_magnitude(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact,
                        registry: dict, offset: int = 0) -> dict:
    grouped: Dict[int, List[CausalSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.suite_seed, []).append(sequence)
    values = {
        seed: response_metrics(group, artifact)["predicted_response_mean"]
        for seed, group in grouped.items()
    }
    return _block_ci(
        values, int(registry["gate"]["bootstrap_replicates"]),
        int(registry["gate"]["bootstrap_seed"]) + offset,
    )


def _shuffled_actions(sequences: Sequence[CausalSequence], seed: int) -> List[CausalSequence]:
    """Shuffle whole episode-time action rows inside each suite seed.

    All pedestrian tracks from one snapshot retain the same shuffled robot
    action. This destroys action/outcome pairing without creating an
    impossible state where different pedestrians saw different robot actions.
    """
    result = [copy.deepcopy(sequence) for sequence in sequences]
    grouped: Dict[int, List[int]] = {}
    for index, sequence in enumerate(result):
        grouped.setdefault(sequence.suite_seed, []).append(index)
    rng = np.random.default_rng(seed)
    for indices in grouped.values():
        by_episode: Dict[int, List[int]] = {}
        for index in indices:
            by_episode.setdefault(result[index].episode_seed, []).append(index)
        representative_indices = [items[0] for _, items in sorted(by_episode.items())]
        action_a = np.concatenate([result[index].action_a for index in representative_indices])
        action_b = np.concatenate([result[index].action_b for index in representative_indices])
        permutation = rng.permutation(len(action_a))
        action_a, action_b = action_a[permutation], action_b[permutation]
        offset = 0
        for episode_seed, items in sorted(by_episode.items()):
            length = len(result[items[0]].action_a)
            shuffled_a = action_a[offset:offset + length]
            shuffled_b = action_b[offset:offset + length]
            for index in items:
                result[index].action_a = shuffled_a.copy()
                result[index].action_b = shuffled_b.copy()
            offset += length
    return result


def _occupancy(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> List[float]:
    result = expectation(sequences, artifact)
    counts = np.sum(np.concatenate(result["gammas"], axis=0), axis=0)
    return (counts / counts.sum()).tolist()


def preflight(output: Path, registry: dict, registry_path: Path, brne_root: str) -> dict:
    all_seeds = []
    for spec in registry["roles"].values():
        all_seeds.extend(spec["suite_seeds"])
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("suite seeds overlap across CR-S1 roles")
    required = [
        Path(__file__), Path(__file__).with_name("causal_pair_data.py"),
        Path(__file__).with_name("causal_response_arhmm.py"),
        Path(__file__).with_name("causal_runtime.py"), registry_path,
    ]
    upstream = Path(brne_root) / "socnavbench" / "brne.py"
    if not upstream.exists():
        raise RuntimeError(f"BRNE upstream missing: {upstream}")
    upstream_hash = _sha256(upstream)
    if upstream_hash != registry["brne_sha256"]:
        raise RuntimeError(
            f"BRNE source hash mismatch: expected {registry['brne_sha256']}, got {upstream_hash}"
        )
    manifest = {
        "experiment_id": registry["experiment_id"],
        "created_at": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "registry_sha256": _sha256(registry_path),
        "source_sha256": {str(path.relative_to(_repo_root())): _sha256(path) for path in required[:-1]},
        "brne_commit": registry["brne_commit"],
        "brne_sha256": upstream_hash,
        "old_s1_immutable": str(output.parent / "s1_strict_20260804"),
    }
    _write_json(output / "manifest.json", manifest)
    return {"status": "PASS", "manifest": manifest}


def fit_initial(output: Path, registry: dict) -> dict:
    sequences = _role_sequences(output, registry, "train")
    config = _config(registry)
    results = {}
    for K in range(1, 5):
        path = _artifact_path(output, K)
        if path.exists():
            artifact = CausalResponseArtifact.load(path)
        else:
            artifact = fit(sequences, K, config)
            artifact.model_card["experiment_id"] = registry["experiment_id"]
            artifact.save(path)
        results[str(K)] = {
            "converged": artifact.model_card["fit_converged"],
            "iterations": artifact.model_card["em_iterations"],
            "train_nll": score_sequences(sequences, artifact)["mean_nll"],
        }
        print(f"[FIT] K={K} {results[str(K)]}", flush=True)
    _write_json(output / "fits" / "initial_fit_summary.json", results)
    return {"status": "PASS", "models": results}


def select_k(output: Path, registry: dict) -> dict:
    train = _role_sequences(output, registry, "train")
    selection = _role_sequences(output, registry, "selection")
    config = _config(registry)
    artifacts = {K: CausalResponseArtifact.load(_artifact_path(output, K)) for K in range(1, 5)}
    nll = {K: _per_suite_nll(selection, artifact) for K, artifact in artifacts.items()}
    boundary_values = {seed: nll[3][seed] - nll[4][seed] for seed in nll[3]}
    boundary = _block_ci(
        boundary_values, int(registry["gate"]["bootstrap_replicates"]),
        int(registry["gate"]["bootstrap_seed"]) + 100,
    )
    boundary_models_valid = all(
        artifacts[K].model_card.get("fit_converged", False)
        and min(_occupancy(selection, artifacts[K])) >= float(registry["gate"]["min_mode_occupancy"])
        for K in (3, 4)
    )
    expanded = boundary_models_valid and boundary["ci_low"] > 0.0
    if expanded:
        for K in (5, 6):
            path = _artifact_path(output, K)
            artifact = CausalResponseArtifact.load(path) if path.exists() else fit(train, K, config)
            artifact.model_card["experiment_id"] = registry["experiment_id"]
            if not path.exists():
                artifact.save(path)
            artifacts[K] = artifact
            nll[K] = _per_suite_nll(selection, artifact)
            print(f"[FIT-BOUNDARY] K={K} mean_nll={np.mean(list(nll[K].values())):.6f}", flush=True)

    records = {}
    min_occupancy = float(registry["gate"]["min_mode_occupancy"])
    for K, artifact in artifacts.items():
        occupancy = _occupancy(selection, artifact)
        records[K] = {
            "mean_nll": float(np.mean(list(nll[K].values()))),
            "se_nll": float(np.std(list(nll[K].values()), ddof=1) / np.sqrt(len(nll[K]))),
            "occupancy": occupancy,
            "converged": bool(artifact.model_card["fit_converged"]),
            "eligible": bool(artifact.model_card["fit_converged"] and min(occupancy) >= min_occupancy),
        }
    eligible = [K for K in records if records[K]["eligible"]]
    if not eligible:
        raise ScientificStop("no converged K satisfies occupancy gate")
    best = min(eligible, key=lambda item: records[item]["mean_nll"])
    threshold = records[best]["mean_nll"] + records[best]["se_nll"]
    selected = min(K for K in eligible if records[K]["mean_nll"] <= threshold)
    result = {
        "status": "PASS", "selected_k": selected, "best_k": best,
        "one_se_threshold": threshold, "boundary_k4_vs_k3": boundary,
        "expanded_to_k6": expanded, "models": {str(k): v for k, v in records.items()},
    }
    _write_json(output / "selection_result.json", result)
    return result


def _evaluate_necessity(output: Path, registry: dict, role: str, offset: int) -> dict:
    selected = _load_json(output / "selection_result.json")["selected_k"]
    full = CausalResponseArtifact.load(_artifact_path(output, selected))
    k1 = CausalResponseArtifact.load(_artifact_path(output, 1))
    sequences = _role_sequences(output, registry, role)
    shared_zero = full.zero_response()
    nll = _nll_advantage(sequences, full, shared_zero, registry, offset)
    mse = _mse_advantage(sequences, full, registry, offset + 1)
    shuffled = _shuffled_actions(sequences, int(registry["gate"]["bootstrap_seed"]) + offset + 2)
    shuffle_nll = _nll_advantage(shuffled, full, shared_zero, registry, offset + 2)
    k1_nll = _nll_advantage(sequences, full, k1, registry, offset + 3)
    passed = (
        nll["ci_low"] > 0.0
        and mse["ci_low"] > float(registry["gate"]["min_response_mse_improvement"])
        and shuffle_nll["ci_high"] <= float(registry["gate"]["nominal_nll_equivalence"])
        and selected > 1
        and k1_nll["ci_low"] > 0.0
    )
    return {
        "status": "PASS" if passed else "FAIL", "role": role, "selected_k": selected,
        "full_vs_shared_zero_nll": nll, "response_mse_improvement": mse,
        "shuffled_action_advantage": shuffle_nll, "selected_vs_k1_nll": k1_nll,
    }


def _evaluate_nominal(output: Path, registry: dict, role: str, offset: int) -> dict:
    selected = _load_json(output / "selection_result.json")["selected_k"]
    full = CausalResponseArtifact.load(_artifact_path(output, selected))
    sequences = _role_sequences(output, registry, role)
    nll = _nll_advantage(sequences, full, full.zero_response(), registry, offset)
    magnitude = _response_magnitude(sequences, full, registry, offset + 1)
    equivalence = float(registry["gate"]["nominal_nll_equivalence"])
    passed = (
        nll["ci_low"] >= -equivalence and nll["ci_high"] <= equivalence
        and magnitude["ci_high"] < float(registry["gate"]["nominal_response_upper"])
    )
    return {
        "status": "PASS" if passed else "FAIL", "role": role,
        "full_vs_shared_zero_nll": nll, "predicted_response_magnitude": magnitude,
    }


def promote(output: Path, registry: dict) -> dict:
    necessity_result = _load_json(output / "necessity_result.json")
    audit_result = _load_json(output / "audit_result.json")
    if necessity_result["status"] != "PASS" or audit_result["status"] != "PASS":
        raise ScientificStop("production promotion denied by scientific gate")
    selected = _load_json(output / "selection_result.json")["selected_k"]
    artifact = CausalResponseArtifact.load(_artifact_path(output, selected))
    if not artifact.model_card.get("fit_converged"):
        raise ScientificStop("production promotion denied: selected fit did not converge")
    artifact.model_card.update({
        "scientific_gate": "GO", "necessity_result_sha256": _sha256(output / "necessity_result.json"),
        "audit_result_sha256": _sha256(output / "audit_result.json"),
        "registry_sha256": _load_json(output / "manifest.json")["registry_sha256"],
        "source_sha256": _load_json(output / "manifest.json")["source_sha256"],
        "data_sha256": {
            role: hashlib.sha256("".join(
                _sha256(path) for path in sorted(_role_path(output, role).rglob("*.npz"))
            ).encode("ascii")).hexdigest()
            for role in registry["roles"]
        },
    })
    path = output / "production" / "causal_response_model.npz"
    artifact.save(path, tier="production")
    return {"status": "PASS", "artifact": str(path), "sha256": _sha256(path)}


def run_stage(stage: str, output: Path, registry: dict, registry_path: Path, brne_root: str) -> dict:
    if stage == "preflight":
        return preflight(output, registry, registry_path, brne_root)
    if stage == "collect_train_selection":
        return {"status": "PASS", "train": _collect(output, registry, "train", brne_root),
                "selection": _collect(output, registry, "selection", brne_root)}
    if stage == "fit":
        return fit_initial(output, registry)
    if stage == "select_k":
        return select_k(output, registry)
    if stage == "collect_necessity":
        return {"status": "PASS", "necessity": _collect(output, registry, "necessity", brne_root)}
    if stage == "necessity":
        result = _evaluate_necessity(output, registry, "necessity", 200)
        _write_json(output / "necessity_result.json", result)
        return result
    if stage == "collect_audit":
        return {
            "status": "PASS",
            "audit_interactive": _collect(output, registry, "audit_interactive", brne_root),
            "audit_nominal": _collect(output, registry, "audit_nominal", brne_root),
        }
    if stage == "audit":
        interactive = _evaluate_necessity(output, registry, "audit_interactive", 300)
        nominal = _evaluate_nominal(output, registry, "audit_nominal", 400)
        result = {
            "status": "PASS" if interactive["status"] == nominal["status"] == "PASS" else "FAIL",
            "interactive": interactive, "nominal": nominal,
        }
        _write_json(output / "audit_result.json", result)
        return result
    if stage == "promote":
        return promote(output, registry)
    if stage in ("latency", "smoke", "formal_eval"):
        from crowd_nav.bayesian_brne.causal_evaluate import run_evaluation_stage
        return run_evaluation_stage(stage, output, registry, brne_root)
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brne-root", required=True)
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    registry = _load_json(registry_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = run_stage(args.stage, output, registry, registry_path, args.brne_root)
        result["stage"] = args.stage
        result["elapsed_seconds"] = time.time() - started
        _write_json(output / "stages" / f"{args.stage}.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if result.get("status") != "PASS":
            print(f"[SCIENTIFIC-STOP] {args.stage} returned {result.get('status')}", flush=True)
            raise SystemExit(20)
    except ScientificStop as exc:
        result = {"stage": args.stage, "status": "SCIENTIFIC_STOP", "reason": str(exc),
                  "elapsed_seconds": time.time() - started}
        _write_json(output / "scientific_stop.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        raise SystemExit(20)


if __name__ == "__main__":
    main()
