"""Executable implementation of the frozen S1 SM-BRNE science gate.

This module owns stage orchestration and artifact I/O. Statistical rules
remain in :mod:`s1_protocol`; the CLI is a thin dispatcher. Formal stages
fail closed on identity, source, convergence, and resume mismatches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_brne import data_io
from crowd_nav.bayesian_brne import s1_protocol as sp
from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
    ARHMMArtifact,
    ARHMMConfig,
    CONTEXT_FEATURE_NAMES,
    e_step,
    extract_sequences,
    fit,
)
from crowd_nav.bayesian_brne.collect_dataset import collect


VARIANT_ACTION_CONDITIONED = "action_conditioned"
VARIANT_SELF_ONLY = "self_only"
VARIANT_SHUFFLE = "constrained_action_shuffle"
VARIANTS = (VARIANT_ACTION_CONDITIONED, VARIANT_SELF_ONLY, VARIANT_SHUFFLE)


class S1PipelineError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_root(registry: Dict[str, Any]) -> Path:
    return sp.repo_root() / "runs" / "bayesian_brne" / registry["experiment_id"]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else sp.repo_root() / candidate


def _role_dir(registry: Dict[str, Any], role: str) -> Path:
    spec = registry[role]
    return _resolve(spec.get("path", spec.get("output_root")))


def _role_paths(registry: Dict[str, Any], role: str) -> List[Path]:
    root = _role_dir(registry, role)
    return sorted(path for path in root.rglob("*.npz") if path.is_file()) if root.exists() else []


def load_role_episodes(registry: Dict[str, Any], role: str) -> Tuple[List[Dict[str, Any]], List[Path]]:
    paths = _role_paths(registry, role)
    episodes = [data_io.load_episode(str(path)) for path in paths]
    expected_seeds = set(int(seed) for seed in registry[role]["suite_seeds"])
    expected_count = int(registry[role]["episodes_per_seed"])
    counts = {seed: 0 for seed in expected_seeds}
    identities = set()
    for episode in episodes:
        seed = int(episode["suite_seed"])
        if seed not in expected_seeds:
            raise S1PipelineError(f"{role} contains unexpected suite_seed={seed}")
        identity = (seed, int(episode["episode_seed"]))
        if identity in identities:
            raise S1PipelineError(f"{role} contains duplicate episode identity {identity}")
        identities.add(identity)
        counts[seed] += 1
    bad = {seed: count for seed, count in counts.items() if count != expected_count}
    if bad:
        raise S1PipelineError(f"{role} count mismatch: {bad}, expected {expected_count} per suite seed")
    return episodes, paths


def _aggregate_file_hash(paths: Sequence[Path]) -> str:
    lines = [f"{path.as_posix()}\t{_sha256_file(path)}" for path in sorted(paths)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _validate_generated_role(registry: Dict[str, Any], role: str) -> Dict[str, Any]:
    episodes, paths = load_role_episodes(registry, role)
    spec = registry[role]
    expected_total = len(spec["suite_seeds"]) * int(spec["episodes_per_seed"])
    if len(episodes) != expected_total:
        raise S1PipelineError(f"{role} has {len(episodes)} episodes, expected {expected_total}")
    controller_counts = {name: 0 for name in registry["controllers"]}
    fallback_count = overlap_count = nonfinite_count = 0
    hashes = set()
    for episode in episodes:
        checks = {
            "scenario": registry["scenario"],
            "split": spec["environment_split"],
            "profile_name": spec["profile_name"],
        }
        for field, expected in checks.items():
            if episode[field] != expected:
                raise S1PipelineError(f"{role} {field}={episode[field]!r}, expected {expected!r}")
        if float(episode["dt"]) != float(registry["dt"]):
            raise S1PipelineError(f"{role} dt mismatch")
        if np.asarray(episode["humans"]).shape[1] != int(registry["n_humans"]):
            raise S1PipelineError(f"{role} human-count mismatch")
        if np.asarray(episode["robot"]).shape[0] != int(registry["horizon_steps"]):
            raise S1PipelineError(f"{role} horizon mismatch")
        controller = str(episode["controller_type"])
        if controller not in controller_counts:
            raise S1PipelineError(f"{role} unknown controller {controller!r}")
        controller_counts[controller] += 1
        fallback_count += sum(event.get("type") == "controller_fallback" for event in episode["events"])
        arrays = (episode["robot"], episode["humans"], episode["robot_actions"], episode["human_actions"])
        nonfinite_count += sum(not np.all(np.isfinite(np.asarray(array))) for array in arrays)
        robot0 = np.asarray(episode["robot"])[0]
        humans0 = np.asarray(episode["humans"])[0]
        valid0 = np.asarray(episode["valid_mask"])[0]
        for human in humans0[valid0]:
            distance = float(np.linalg.norm(human[:2] - robot0[:2]))
            if distance < float(human[4] + robot0[4]) - 1e-9:
                overlap_count += 1
        value = str(episode["initial_state_hash"])
        if value in hashes:
            raise S1PipelineError(f"{role} duplicate initial_state_hash={value}")
        hashes.add(value)
    expected_per_controller = expected_total // len(controller_counts)
    if any(value != expected_per_controller for value in controller_counts.values()):
        raise S1PipelineError(f"{role} controller imbalance: {controller_counts}")
    if fallback_count or overlap_count or nonfinite_count:
        raise S1PipelineError(
            f"{role} quality failure fallback={fallback_count} overlap={overlap_count} nonfinite={nonfinite_count}"
        )
    report = {
        "role": role,
        "n_episodes": len(episodes),
        "controller_counts": controller_counts,
        "fallback_count": fallback_count,
        "overlap_count": overlap_count,
        "nonfinite_count": nonfinite_count,
        "file_aggregate_sha256": _aggregate_file_hash(paths),
        "initial_state_hashes_sha256": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
        "registry_sha256": sp.registry_content_sha256(registry),
        "method_source_manifest_sha256": _sha256_file(output_root(registry) / "method_source_manifest.json"),
    }
    _assert_role_disjoint(registry, role, episodes)
    return report


def _assert_role_disjoint(
    registry: Dict[str, Any], role: str, episodes: Sequence[Dict[str, Any]],
) -> None:
    current_suite = {int(ep["suite_seed"]) for ep in episodes}
    current_episode = {int(ep["episode_seed"]) for ep in episodes}
    current_hash = {str(ep["initial_state_hash"]) for ep in episodes}
    compare_roles = ["train", "selection"]
    if role != "necessity_id":
        compare_roles.append("necessity_id")
    if role == "audit_nominal_negative_control":
        compare_roles.append("audit_interactive")
    for other in compare_roles:
        other_paths = _role_paths(registry, other)
        if not other_paths:
            continue
        other_episodes = [data_io.load_episode(str(path)) for path in other_paths]
        overlaps = {
            "suite_seed": current_suite & {int(ep["suite_seed"]) for ep in other_episodes},
            "episode_seed": current_episode & {int(ep["episode_seed"]) for ep in other_episodes},
            "initial_state_hash": current_hash & {str(ep["initial_state_hash"]) for ep in other_episodes},
        }
        bad = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
        if bad:
            raise S1PipelineError(f"data-role identity overlap {role} vs {other}: {bad}")


def collect_role(registry: Dict[str, Any], role: str) -> Dict[str, Any]:
    if role not in ("necessity_id", "audit_interactive", "audit_nominal_negative_control"):
        raise S1PipelineError(f"unsupported generated role {role}")
    out = output_root(registry)
    if role == "necessity_id":
        for audit_role in ("audit_interactive", "audit_nominal_negative_control"):
            if _role_dir(registry, audit_role).exists():
                raise S1PipelineError(f"AUDIT_CONTAMINATED: {_role_dir(registry, audit_role)} already exists")
    spec = registry[role]
    role_root = _role_dir(registry, role)
    role_root.mkdir(parents=True, exist_ok=True)
    for seed in spec["suite_seeds"]:
        expected_episode_seeds = {int(seed) * 100000 + index for index in range(int(spec["episodes_per_seed"]))}
        existing = []
        for path in role_root.rglob("*.npz"):
            episode = data_io.load_episode(str(path))
            if int(episode["suite_seed"]) == int(seed):
                existing.append((path, episode))
        if existing:
            found = {int(episode["episode_seed"]) for _, episode in existing}
            if found != expected_episode_seeds:
                raise S1PipelineError(
                    f"partial/inconsistent resume for {role} seed={seed}: found {len(found)}, expected {len(expected_episode_seeds)}"
                )
        else:
            collect(
                split=spec["environment_split"], scenario=registry["scenario"],
                episodes=int(spec["episodes_per_seed"]), seed=int(seed),
                profile_name=spec["profile_name"], output_dir=str(role_root),
                horizon_steps=int(registry["horizon_steps"]), dt=float(registry["dt"]),
            )
        seed_paths = []
        for path in role_root.rglob("*.npz"):
            episode = data_io.load_episode(str(path))
            if int(episode["suite_seed"]) == int(seed):
                seed_paths.append(path)
        _atomic_json(out / "data_manifests" / f"{role}_seed{seed}.json", {
            "role": role, "suite_seed": int(seed), "n_files": len(seed_paths),
            "file_aggregate_sha256": _aggregate_file_hash(seed_paths),
        })
    report = _validate_generated_role(registry, role)
    _atomic_json(out / "data_manifests" / f"{role}_aggregate.json", report)
    return report


def _variant_episodes(
    episodes: Sequence[Dict[str, Any]], variant: str, shuffle_map: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if variant not in VARIANTS:
        raise S1PipelineError(f"unknown variant {variant!r}")
    transformed = [dict(episode) for episode in episodes]
    if variant == VARIANT_ACTION_CONDITIONED:
        return transformed
    if variant == VARIANT_SELF_ONLY:
        for episode in transformed:
            episode["robot_actions"] = np.zeros_like(np.asarray(episode["robot_actions"], dtype=float))
        return transformed
    if shuffle_map is None:
        raise S1PipelineError("shuffle variant requires frozen shuffle_map")
    source = {sp._episode_identity(episode): episode for episode in episodes}
    assignments = {item["recipient_id"]: item["donor_id"] for item in shuffle_map["mapping"]}
    if set(assignments) != set(source) or set(assignments.values()) != set(source):
        raise S1PipelineError("shuffle map identities do not exactly match training episodes")
    for episode in transformed:
        recipient = sp._episode_identity(episode)
        episode["robot_actions"] = np.array(source[assignments[recipient]]["robot_actions"], copy=True)
    if sp._action_multiset_sha256(transformed) != shuffle_map["action_multiset_sha256"]:
        raise S1PipelineError("shuffle materialization changed the action multiset")
    return transformed


def build_shuffle(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    episodes, _ = load_role_episodes(registry, "train")
    result = sp.build_constrained_shuffle_map(episodes, int(registry["bootstrap_seed"]))
    path = out / "shuffle_map.json"
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise S1PipelineError("existing shuffle_map.json differs from deterministic recomputation")
    if not path.exists():
        path.write_text(encoded)
    return result


def _method_manifest(registry: Dict[str, Any]) -> Dict[str, Any]:
    path = output_root(registry) / "method_source_manifest.json"
    if not path.exists():
        raise S1PipelineError("method_source_manifest.json is missing; S1-BUILD is not locked")
    manifest = json.loads(path.read_text())
    sp.verify_source_manifest_unchanged(str(sp.repo_root()), manifest, soft_paths=frozenset())
    return manifest


def create_method_lock(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    source_manifest = sp.build_source_manifest(str(sp.repo_root()))
    if source_manifest["missing_files"]:
        raise S1PipelineError(f"method source manifest has missing files: {source_manifest['missing_files']}")
    manifest_path = out / "method_source_manifest.json"
    lock_path = out / "method_lock.json"
    if manifest_path.exists() or lock_path.exists():
        if not (manifest_path.exists() and lock_path.exists()):
            raise S1PipelineError("partial method lock exists")
        manifest = json.loads(manifest_path.read_text())
        sp.verify_source_manifest_unchanged(str(sp.repo_root()), manifest, soft_paths=frozenset())
        lock = json.loads(lock_path.read_text())
        if lock["method_source_manifest_sha256"] != _sha256_file(manifest_path):
            raise S1PipelineError("method lock hash does not match method source manifest")
        return lock
    _atomic_json(manifest_path, source_manifest)
    lock = {
        "schema_version": 1,
        "experiment_id": registry["experiment_id"],
        "registry_sha256": sp.registry_content_sha256(registry),
        "method_source_manifest_sha256": _sha256_file(manifest_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_json(lock_path, lock)
    return lock


def _model_provenance(registry: Dict[str, Any], paths: Sequence[Path], variant: str, K: int, restart: int) -> Dict[str, Any]:
    manifest = _method_manifest(registry)
    return {
        "feature_schema": {"name": "ARHMM_CONTEXT_FEATURES", "version": 1, "names": list(CONTEXT_FEATURE_NAMES)},
        "time_semantics": {
            "dt": float(registry["dt"]), "state": "state at t before action",
            "action": "robot_actions[t] applied during t->t+1", "target": "human velocity at t+1",
        },
        "training_config": {"variant": variant, "K": int(K), "restart_seed": int(restart), **{
            key: registry[key] for key in (
                "sticky_kappa", "dirichlet_alpha", "shrinkage_scale", "inverse_wishart_dof",
                "inverse_wishart_scale", "em_max_iters", "em_tol",
            )
        }},
        "data_aggregate_sha256": _aggregate_file_hash(paths),
        "source_sha256": {"method_source_manifest": _sha256_file(output_root(registry) / "method_source_manifest.json")},
        "human_physical_bounds": {"max_speed": 2.0, "max_acceleration": 2.0},
    }


def _predictive_similarity(artifact: ARHMMArtifact, sequences: Sequence[Any], max_rows: int = 4096) -> float:
    if artifact.K <= 1:
        return 0.0
    designs = []
    for sequence in sequences:
        for t in range(sequence.v_current.shape[0]):
            designs.append((sequence.v_current[t], sequence.u_robot[t], sequence.context[t]))
            if len(designs) >= max_rows:
                break
        if len(designs) >= max_rows:
            break
    if not designs:
        return float("nan")
    maximum = 0.0
    for i in range(artifact.K):
        for j in range(i + 1, artifact.K):
            sigma = 0.5 * (artifact.Q[i] + artifact.Q[j])
            inv_sigma = np.linalg.inv(sigma)
            determinant_factor = np.sqrt(
                np.sqrt(np.linalg.det(artifact.Q[i]) * np.linalg.det(artifact.Q[j])) / np.linalg.det(sigma)
            )
            values = []
            for v, u, context in designs:
                mean_i = artifact.A[i] @ v + artifact.B[i] @ u + artifact.C[i] @ context + artifact.d[i]
                mean_j = artifact.A[j] @ v + artifact.B[j] @ u + artifact.C[j] @ context + artifact.d[j]
                diff = mean_i - mean_j
                values.append(float(determinant_factor * np.exp(-0.125 * diff @ inv_sigma @ diff)))
            maximum = max(maximum, float(np.mean(values)))
    return maximum


def _fit_dir(registry: Dict[str, Any], variant: str, K: int, restart: int) -> Path:
    return output_root(registry) / "fits" / variant / f"K{K}" / f"restart_{restart}"


def fit_one(registry: Dict[str, Any], variant: str, K: int, restart: int) -> Dict[str, Any]:
    fit_dir = _fit_dir(registry, variant, K, restart)
    report_path = fit_dir / "fit_report.json"
    artifact_dir = fit_dir / "artifact"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("status") != "COMPLETED":
            raise S1PipelineError(f"incomplete fit report at {report_path}")
        artifact = ARHMMArtifact.load(str(artifact_dir), expect_tier="engineering_only")
        if artifact.content_sha256() != report["artifact_content_sha256"]:
            raise S1PipelineError(f"resume artifact hash mismatch at {fit_dir}")
        return report
    if fit_dir.exists() and any(fit_dir.iterdir()):
        raise S1PipelineError(f"partial fit directory exists without completed report: {fit_dir}")
    fit_dir.mkdir(parents=True, exist_ok=True)
    sp.update_status_atomic(
        str(output_root(registry)), experiment_id=registry["experiment_id"],
        registry_sha256=sp.registry_content_sha256(registry),
        active_fit={"variant": variant, "K": int(K), "restart_seed": int(restart), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    episodes, paths = load_role_episodes(registry, "train")
    shuffle_map = None
    if variant == VARIANT_SHUFFLE:
        shuffle_path = output_root(registry) / "shuffle_map.json"
        if not shuffle_path.exists():
            raise S1PipelineError("shuffle_map.json missing")
        shuffle_map = json.loads(shuffle_path.read_text())
    transformed = _variant_episodes(episodes, variant, shuffle_map)
    sequences = extract_sequences(transformed, float(registry["dt"]))
    config = ARHMMConfig(
        k_candidates=(int(K),), sticky_kappa=float(registry["sticky_kappa"]),
        dirichlet_alpha=float(registry["dirichlet_alpha"]),
        shrinkage_scale=float(registry["shrinkage_scale"]),
        inverse_wishart_dof=float(registry["inverse_wishart_dof"]),
        inverse_wishart_scale=float(registry["inverse_wishart_scale"]),
        em_max_iters=int(registry["em_max_iters"]), em_tol=float(registry["em_tol"]), seed=int(restart),
    )
    started = time.time()
    fit_status_path = fit_dir / "fit_status.json"
    _atomic_json(fit_status_path, {
        "status": "RUNNING", "pid": os.getpid(), "variant": variant, "K": int(K),
        "restart_seed": int(restart), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    def progress(payload: Dict[str, Any]) -> None:
        state = {
            "status": "RUNNING", "pid": os.getpid(), "variant": variant, "K": int(K),
            "restart_seed": int(restart), "started_at_epoch": started,
            "elapsed_seconds": float(time.time() - started), **payload,
        }
        _atomic_json(fit_status_path, state)
        print(
            f"[S1-FIT] variant={variant} K={K} restart={restart} "
            f"iter={payload['iteration']}/{registry['em_max_iters']} "
            f"objective={payload['penalized_objective']:.6f}",
            flush=True,
        )

    print(f"[S1-FIT] START variant={variant} K={K} restart={restart} rows={sum(s.v_current.shape[0] for s in sequences)}", flush=True)
    _best, results = fit(
        sequences, [], config, dt=float(registry["dt"]), progress_callback=progress,
    )
    result = results[int(K)]
    artifact = result["artifact"]
    stats = e_step(sequences, artifact)
    occupancy = np.sum(np.vstack(stats["gammas"]), axis=0)
    occupancy = occupancy / np.sum(occupancy)
    q_eigenvalues = [np.linalg.eigvalsh(q).tolist() for q in artifact.Q]
    all_arrays = [artifact.Pi, artifact.initial_distribution, *artifact.A, *artifact.B, *artifact.C, *artifact.d, *artifact.Q]
    finite = all(np.all(np.isfinite(array)) for array in all_arrays)
    normalized = bool(
        np.allclose(artifact.Pi.sum(axis=1), 1.0, atol=1e-8)
        and np.isclose(artifact.initial_distribution.sum(), 1.0, atol=1e-8)
    )
    similarity = _predictive_similarity(artifact, sequences)
    convergence = result["convergence"]
    self_b_max = max(float(np.max(np.abs(b))) for b in artifact.B)
    eligible = bool(
        convergence["converged"] and finite and normalized
        and min(float(value) for value in occupancy) >= float(registry["min_mode_fraction"])
        and similarity <= float(registry["max_predictive_similarity"])
        and min(min(values) for values in q_eigenvalues) > 0.0
        and (variant != VARIANT_SELF_ONLY or self_b_max <= 1e-10)
    )
    artifact.model_card["artifact_provenance"] = _model_provenance(registry, paths, variant, K, restart)
    artifact.model_card["s1_fit_identity"] = {
        "experiment_id": registry["experiment_id"], "variant": variant, "K": int(K),
        "restart_seed": int(restart), "registry_sha256": sp.registry_content_sha256(registry),
    }
    artifact.save(str(artifact_dir), tier="engineering_only", convergence=convergence)
    report = {
        "status": "COMPLETED", "variant": variant, "K": int(K), "restart_seed": int(restart),
        "eligible": eligible, "convergence": convergence,
        "train_final_objective": float(result["train_objective_history"][-1]),
        "train_objective_history": [float(value) for value in result["train_objective_history"]],
        "train_ll_history": [float(value) for value in result["train_ll_history"]],
        "mode_occupancy": occupancy.tolist(), "max_predictive_similarity": float(similarity),
        "q_eigenvalues": q_eigenvalues, "pi_diagonal": np.diag(artifact.Pi).tolist(),
        "self_only_b_max_abs": self_b_max, "finite": finite, "normalized": normalized,
        "elapsed_seconds": float(time.time() - started),
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "artifact_content_sha256": artifact.content_sha256(),
        "artifact_path": str(artifact_dir),
    }
    _atomic_json(report_path, report)
    _atomic_json(fit_status_path, {
        "status": "COMPLETED", "pid": None, "variant": variant, "K": int(K),
        "restart_seed": int(restart), "elapsed_seconds": report["elapsed_seconds"],
        "converged": bool(convergence["converged"]), "eligible": bool(eligible),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(
        f"[S1-FIT] DONE variant={variant} K={K} restart={restart} "
        f"converged={convergence['converged']} eligible={eligible} elapsed={report['elapsed_seconds']:.1f}s",
        flush=True,
    )
    sp.update_status_atomic(
        str(output_root(registry)), experiment_id=registry["experiment_id"],
        registry_sha256=sp.registry_content_sha256(registry), active_fit=None,
    )
    return report


def fit_grid(registry: Dict[str, Any], variant: str, ks: Iterable[int]) -> Dict[str, Any]:
    reports = []
    for K in ks:
        for restart in registry["restart_seeds"]:
            reports.append(fit_one(registry, variant, int(K), int(restart)))
    return {"variant": variant, "fits": reports}


def _selected_restart(registry: Dict[str, Any], variant: str, K: int) -> Dict[str, Any]:
    reports = []
    for restart in registry["restart_seeds"]:
        path = _fit_dir(registry, variant, K, int(restart)) / "fit_report.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
    return sp.select_best_restart_by_train_objective(reports)


def _score_artifact_by_seed(
    artifact: ARHMMArtifact, episodes: Sequence[Dict[str, Any]], variant_input: str = VARIANT_ACTION_CONDITIONED,
) -> Dict[int, Dict[str, Any]]:
    transformed = _variant_episodes(episodes, VARIANT_SELF_ONLY if variant_input == VARIANT_SELF_ONLY else VARIANT_ACTION_CONDITIONED)
    sequences = extract_sequences(transformed, artifact.dt)
    stats = e_step(sequences, artifact)
    grouped: Dict[int, Dict[str, Any]] = {}
    for sequence, ll in zip(sequences, stats["sequence_log_likelihoods"]):
        episode_index = int(sequence.track_key[0])
        episode = episodes[episode_index]
        seed = int(episode["suite_seed"])
        controller = str(episode["controller_type"])
        entry = grouped.setdefault(seed, {"log_likelihood": 0.0, "n_rows": 0, "controllers": {}})
        entry["log_likelihood"] += float(ll)
        entry["n_rows"] += int(sequence.v_current.shape[0])
        ctrl = entry["controllers"].setdefault(controller, {"log_likelihood": 0.0, "n_rows": 0})
        ctrl["log_likelihood"] += float(ll)
        ctrl["n_rows"] += int(sequence.v_current.shape[0])
    return grouped


def _comparison(
    registry: Dict[str, Any], candidate: ARHMMArtifact, baseline: ARHMMArtifact,
    episodes: Sequence[Dict[str, Any]], baseline_input: str = VARIANT_ACTION_CONDITIONED,
    bootstrap_seed_offset: int = 0,
) -> Dict[str, Any]:
    cand = _score_artifact_by_seed(candidate, episodes)
    base = _score_artifact_by_seed(baseline, episodes, baseline_input)
    seeds = sorted(set(cand) & set(base))
    blocks = [{
        "suite_seed": seed,
        "delta_log_likelihood": cand[seed]["log_likelihood"] - base[seed]["log_likelihood"],
        "n_rows": cand[seed]["n_rows"],
    } for seed in seeds]
    for seed in seeds:
        if cand[seed]["n_rows"] != base[seed]["n_rows"]:
            raise S1PipelineError(f"paired score row-count mismatch for suite_seed={seed}")
    overall = sp.block_bootstrap_suite_seed_ci(
        blocks, int(registry["bootstrap_resamples"]),
        int(registry["bootstrap_seed"]) + int(bootstrap_seed_offset), float(registry["confidence_level"]),
    )
    controllers = {}
    for controller_index, controller in enumerate(registry["controllers"]):
        ctrl_blocks = []
        for seed in seeds:
            a = cand[seed]["controllers"].get(controller)
            b = base[seed]["controllers"].get(controller)
            if a and b:
                ctrl_blocks.append({
                    "suite_seed": seed, "delta_log_likelihood": a["log_likelihood"] - b["log_likelihood"],
                    "n_rows": a["n_rows"],
                })
        controllers[controller] = sp.block_bootstrap_suite_seed_ci(
            ctrl_blocks, int(registry["bootstrap_resamples"]),
            int(registry["bootstrap_seed"]) + int(bootstrap_seed_offset) + 100 + controller_index,
            float(registry["confidence_level"]),
        )
    return {"overall": overall, "controllers": controllers, "blocks": blocks}


def _load_selected_artifact(registry: Dict[str, Any], variant: str, K: int) -> Tuple[ARHMMArtifact, Dict[str, Any]]:
    report = _selected_restart(registry, variant, K)
    return ARHMMArtifact.load(report["artifact_path"], expect_tier="engineering_only"), report


def _k_summary(registry: Dict[str, Any], K: int, selection_episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        artifact, report = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, K)
    except ValueError:
        return {"K": int(K), "eligible": False, "reason": "no_eligible_converged_restart", "per_seed_nll": []}
    scores = _score_artifact_by_seed(artifact, selection_episodes)
    return {
        "K": int(K), "eligible": True, "selected_restart": int(report["restart_seed"]),
        "per_seed_nll": [-scores[seed]["log_likelihood"] / scores[seed]["n_rows"] for seed in sorted(scores)],
    }


def select_models(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    selection_episodes, _ = load_role_episodes(registry, "selection")
    primary = [_k_summary(registry, int(K), selection_episodes) for K in registry["primary_k_candidates"]]
    selection = sp.select_k(primary, float(registry["confidence_level"]))
    boundary = False
    if selection.get("status") == "SELECTED" and selection.get("best_mean_k") == 4:
        k4, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, 4)
        k3, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, 3)
        k4_vs_k3 = _comparison(registry, k4, k3, selection_episodes, bootstrap_seed_offset=400)
        selection["k4_vs_k3"] = k4_vs_k3
        boundary = k4_vs_k3["overall"].get("ci_low", float("-inf")) > 0.0
    if boundary:
        fit_grid(registry, VARIANT_ACTION_CONDITIONED, registry["boundary_extension_k_candidates"])
        extended = primary + [
            _k_summary(registry, int(K), selection_episodes) for K in registry["boundary_extension_k_candidates"]
        ]
        selection = sp.select_k(extended, float(registry["confidence_level"]))
        selection["boundary_extended"] = True
        if selection.get("best_mean_k") == 6:
            k6, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, 6)
            k5, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, 5)
            k6_vs_k5 = _comparison(registry, k6, k5, selection_episodes, bootstrap_seed_offset=600)
            selection["k6_vs_k5"] = k6_vs_k5
            if k6_vs_k5["overall"].get("ci_low", float("-inf")) > 0.0:
                selection = {**selection, "status": "INCONCLUSIVE", "reason": "unresolved_k6_boundary"}
    if selection.get("status") == "SELECTED":
        K = int(selection["selected_k"])
        if K <= 1:
            selection = {**selection, "status": "NO_GO", "reason": "selected_k_is_one"}
        else:
            selected, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, K)
            k1, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, 1)
            comparison = _comparison(registry, selected, k1, selection_episodes, bootstrap_seed_offset=1000)
            selection["selected_k_vs_k1"] = comparison
            if comparison["overall"].get("ci_low", float("-inf")) <= 0.0:
                selection = {**selection, "status": "NO_GO", "reason": "multimodality_ci_not_positive"}
    selection["all_fit_reports"] = [
        json.loads(path.read_text()) for path in sorted((out / "fits" / VARIANT_ACTION_CONDITIONED).rglob("fit_report.json"))
    ]
    _atomic_json(out / "selection_report.json", selection)
    return selection


def _gate_comparisons(
    registry: Dict[str, Any], episodes: Sequence[Dict[str, Any]], selected_k: int,
) -> Dict[str, Any]:
    ac, ac_report = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, selected_k)
    self_model, self_report = _load_selected_artifact(registry, VARIANT_SELF_ONLY, selected_k)
    shuffle, shuffle_report = _load_selected_artifact(registry, VARIANT_SHUFFLE, selected_k)
    k1, k1_report = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, 1)
    comparisons = {
        "ac_vs_self": _comparison(registry, ac, self_model, episodes, VARIANT_SELF_ONLY, 2000),
        "ac_vs_shuffle": _comparison(registry, ac, shuffle, episodes, VARIANT_ACTION_CONDITIONED, 3000),
        "ac_vs_k1": _comparison(registry, ac, k1, episodes, VARIANT_ACTION_CONDITIONED, 4000),
    }
    restart_directions = {"ac_vs_self": [], "ac_vs_shuffle": []}
    for restart in registry["restart_seeds"]:
        try:
            fit_reports = {}
            for variant in VARIANTS:
                report_path = _fit_dir(registry, variant, selected_k, int(restart)) / "fit_report.json"
                fit_reports[variant] = json.loads(report_path.read_text())
                if not fit_reports[variant].get("eligible") or not fit_reports[variant]["convergence"].get("converged"):
                    raise S1PipelineError(f"{variant} restart={restart} is not eligible and converged")
            ac_r = ARHMMArtifact.load(str(_fit_dir(registry, VARIANT_ACTION_CONDITIONED, selected_k, int(restart)) / "artifact"), "engineering_only")
            self_r = ARHMMArtifact.load(str(_fit_dir(registry, VARIANT_SELF_ONLY, selected_k, int(restart)) / "artifact"), "engineering_only")
            shuf_r = ARHMMArtifact.load(str(_fit_dir(registry, VARIANT_SHUFFLE, selected_k, int(restart)) / "artifact"), "engineering_only")
            restart_directions["ac_vs_self"].append(_comparison(registry, ac_r, self_r, episodes, VARIANT_SELF_ONLY, 5000 + int(restart))["overall"])
            restart_directions["ac_vs_shuffle"].append(_comparison(registry, ac_r, shuf_r, episodes, VARIANT_ACTION_CONDITIONED, 6000 + int(restart))["overall"])
        except Exception as exc:
            restart_directions["ac_vs_self"].append({"status": "INCONCLUSIVE", "reason": str(exc), "restart": int(restart)})
            restart_directions["ac_vs_shuffle"].append({"status": "INCONCLUSIVE", "reason": str(exc), "restart": int(restart)})
    return {
        "selected_reports": {"ac": ac_report, "self": self_report, "shuffle": shuffle_report, "k1": k1_report},
        "comparisons": comparisons, "restart_directions": restart_directions,
        "secondary_diagnostics": {"brier": None, "ece": None, "coverage": None, "note": "NLL is the frozen primary gate"},
    }


def _necessity_status(gate: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons = []
    comparisons = gate["comparisons"]
    for name in ("ac_vs_self", "ac_vs_shuffle", "ac_vs_k1"):
        if comparisons[name]["overall"].get("status") != "OK":
            return "INCONCLUSIVE", [f"{name}_bootstrap_unavailable"]
        if comparisons[name]["overall"]["ci_low"] <= 0.0:
            reasons.append(f"{name}_ci_low_not_positive")
    for name in ("ac_vs_self", "ac_vs_shuffle"):
        controller_stats = comparisons[name]["controllers"]
        positive = sum(stats.get("point_estimate", float("-inf")) > 0.0 for stats in controller_stats.values())
        if positive < 3:
            reasons.append(f"{name}_fewer_than_3_positive_controllers")
        if any(stats.get("ci_high", float("inf")) < 0.0 for stats in controller_stats.values()):
            reasons.append(f"{name}_controller_harm")
        restart_positive = sum(
            result.get("point_estimate", float("-inf")) > 0.0
            for result in gate["restart_directions"][name]
        )
        if restart_positive < 2:
            reasons.append(f"{name}_fewer_than_2_positive_restarts")
    return ("PASS", []) if not reasons else ("FAIL", reasons)


def run_necessity(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    selection = json.loads((out / "selection_report.json").read_text())
    if selection.get("status") != "SELECTED":
        report = {"status": selection.get("status", "INCONCLUSIVE"), "reason": "selection_not_passed"}
        _atomic_json(out / "necessity_report.json", report)
        return report
    episodes, paths = load_role_episodes(registry, "necessity_id")
    gate = _gate_comparisons(registry, episodes, int(selection["selected_k"]))
    status, reasons = _necessity_status(gate)
    report = {
        "status": status, "reasons": reasons, "selected_k": int(selection["selected_k"]),
        "data_aggregate_sha256": _aggregate_file_hash(paths),
        "registry_sha256": sp.registry_content_sha256(registry), **gate,
    }
    _atomic_json(out / "necessity_report.json", report)
    return report


def unlock_and_collect_audit(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    necessity_path = out / "necessity_report.json"
    necessity = json.loads(necessity_path.read_text())
    if necessity.get("status") != "PASS":
        raise S1PipelineError("audit remains locked because necessity did not PASS")
    manifest_path = out / "method_source_manifest.json"
    stable_lock = {
        "necessity_report_sha256": _sha256_file(necessity_path),
        "registry_sha256": sp.registry_content_sha256(registry),
        "method_source_manifest_sha256": _sha256_file(manifest_path),
    }
    unlock_path = out / "audit_unlock.json"
    if unlock_path.exists():
        existing_lock = json.loads(unlock_path.read_text())
        if any(existing_lock.get(key) != value for key, value in stable_lock.items()):
            raise S1PipelineError("existing audit unlock does not match current immutable inputs")
    else:
        _atomic_json(unlock_path, {**stable_lock, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    reports = {
        "interactive": collect_role(registry, "audit_interactive"),
        "nominal": collect_role(registry, "audit_nominal_negative_control"),
    }
    return reports


def run_audit(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    report_path = out / "audit_report.json"
    if report_path.exists():
        raise S1PipelineError("audit_report.json already exists; audit is one-shot and sealed")
    unlock_path = out / "audit_unlock.json"
    if not unlock_path.exists():
        raise S1PipelineError("audit_unlock.json is missing")
    unlock = json.loads(unlock_path.read_text())
    if unlock["necessity_report_sha256"] != _sha256_file(out / "necessity_report.json"):
        raise S1PipelineError("necessity report changed after audit unlock")
    selection = json.loads((out / "selection_report.json").read_text())
    K = int(selection["selected_k"])
    interactive, interactive_paths = load_role_episodes(registry, "audit_interactive")
    nominal, nominal_paths = load_role_episodes(registry, "audit_nominal_negative_control")
    interactive_gate = _gate_comparisons(registry, interactive, K)
    interactive_status, interactive_reasons = _necessity_status(interactive_gate)
    ac, _ = _load_selected_artifact(registry, VARIANT_ACTION_CONDITIONED, K)
    self_model, _ = _load_selected_artifact(registry, VARIANT_SELF_ONLY, K)
    nominal_comparison = _comparison(registry, ac, self_model, nominal, VARIANT_SELF_ONLY, 9000)
    margin = float(registry["nominal_equivalence_margin_nats_per_row"])
    low = nominal_comparison["overall"].get("ci_low", float("nan"))
    high = nominal_comparison["overall"].get("ci_high", float("nan"))
    nominal_status = "PASS" if np.isfinite(low) and np.isfinite(high) and low >= -margin and high <= margin else "FAIL"
    report = {
        "status": "PASS" if interactive_status == "PASS" and nominal_status == "PASS" else "FAIL",
        "interactive": {"status": interactive_status, "reasons": interactive_reasons, **interactive_gate},
        "nominal": {"status": nominal_status, "ci_low": low, "ci_high": high, "comparison": nominal_comparison},
        "data_hashes": {
            "interactive": _aggregate_file_hash(interactive_paths), "nominal": _aggregate_file_hash(nominal_paths),
        },
    }
    _atomic_json(report_path, report)
    return report


def promote(registry: Dict[str, Any]) -> Dict[str, Any]:
    out = output_root(registry)
    selection = json.loads((out / "selection_report.json").read_text())
    necessity = json.loads((out / "necessity_report.json").read_text())
    audit = json.loads((out / "audit_report.json").read_text())
    final = sp.three_state_judgment(
        selection, necessity, audit.get("interactive"), audit.get("nominal"),
        float(registry["nominal_equivalence_margin_nats_per_row"]),
    )
    if final["verdict"] == "GO":
        artifact, fit_report = _load_selected_artifact(
            registry, VARIANT_ACTION_CONDITIONED, int(selection["selected_k"])
        )
        destination = out / "production_artifact"
        artifact.model_card["s1_final_judgment"] = final
        artifact.model_card["s1_reports"] = {
            "selection_sha256": _sha256_file(out / "selection_report.json"),
            "necessity_sha256": _sha256_file(out / "necessity_report.json"),
            "audit_sha256": _sha256_file(out / "audit_report.json"),
        }
        artifact.save(str(destination), tier="production", convergence=fit_report["convergence"])
        final["production_artifact"] = str(destination)
        final["artifact_content_sha256"] = artifact.content_sha256()
    _atomic_json(out / "final_verdict.json", final)
    return final
