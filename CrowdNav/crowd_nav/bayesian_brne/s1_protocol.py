"""s1_protocol.py: the ONE place S1's statistical/identity logic lives
(guide.md section 13, Order S1-0, 2026-08-04).

guide.md is explicit that "统计逻辑不得散落在CLI脚本里" -- every CLI entry
point (``tools/s1_strict_gate.py``, ``tools/run_s1_queue.py``) is a thin
dispatcher that calls into this module; it must never reimplement registry
parsing, identity checks, restart/K selection, constrained shuffling,
bootstrap CIs, or the final three-state judgment itself.

This module deliberately does NOT import ``tools/u3_necessity_gate.py``.
guide.md section 13 is explicit that the old U3 pilot gate and its GO
result are frozen, read-only history -- S1 must not depend on it, extend
it, or treat its conclusion as a prior. Any resemblance to its block-
bootstrap style is independent re-derivation against a stricter, more
constrained protocol (disjoint K-selection/necessity/audit splits, >=3
restarts per K, controller/context-constrained shuffle, boundary handling
capped at K=6) -- not shared code.

S1-BUILD status: registry/identity checks, constrained shuffle, train-only
restart selection, one-SE K selection, suite-seed block bootstrap, and the
final three-state judgment are implemented here. Stage orchestration and
artifact I/O live in ``s1_pipeline.py``; the two CLI files remain thin
dispatchers and never duplicate these statistical rules.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from crowd_nav.bayesian_brne import provenance
from crowd_nav.bayesian_brne.schemas import canonical_json_dumps

SCHEMA_VERSION = 1


def repo_root() -> Path:
    """The CrowdNav repository root, derived from THIS FILE's own location
    (``<repo_root>/crowd_nav/bayesian_brne/s1_protocol.py``) -- never a
    hardcoded host-specific path. Order S1-0R fix: the original CLIs
    hardcoded one developer machine's absolute checkout path, which does
    not exist at all on the 4090 training host (a different user account,
    a different directory tree) -- code synced there and run as-is would
    have silently read/written the wrong location."""
    return Path(__file__).resolve().parents[2]

REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version", "experiment_id", "scenario", "n_humans", "dt", "horizon_steps",
    "controllers", "train", "selection", "necessity_id", "audit_interactive",
    "audit_nominal_negative_control", "primary_k_candidates", "boundary_extension_k_candidates",
    "restart_seeds", "sticky_kappa", "dirichlet_alpha", "shrinkage_scale",
    "inverse_wishart_dof", "inverse_wishart_scale", "em_max_iters", "em_tol",
    "bootstrap_resamples", "bootstrap_seed", "confidence_level",
    "nominal_equivalence_margin_nats_per_row", "min_mode_fraction", "max_predictive_similarity",
)

REQUIRED_CONTROLLERS = ("goal_directed", "orca", "original_brne", "scripted_probe")

# name -> required keys for the five data-role sub-dicts.
_TRAIN_SELECTION_KEYS = ("path", "suite_seeds", "episodes_per_seed")
_GENERATED_SPLIT_KEYS = ("environment_split", "output_root", "suite_seeds", "episodes_per_seed", "profile_name")

# Source files whose SHA256 the source manifest must record (guide.md
# S1-0: "AR-HMM、data_io、interaction_protocol、s1_protocol、s1
# CLI、collector、registry"; S1-0R added run_s1_queue.py, which the
# original list omitted despite it being a real S1 CLI entry point).
# Committed inside the repo (guide.md 13/S1-0R-A A1 item 5) so the frozen
# S1 normative spec syncs to and is hard-verified on every host, including
# the 4090, which has no local guide.md at all to check against. See the
# "frozen_protocol_spec.md" section below for how this file is written/
# amended (never silently, and never as an ordinary-preflight side effect).
COMMITTED_PROTOCOL_SPEC_REL_PATH = "crowd_nav/configs/s1_frozen_protocol_spec.md"

SOURCE_MANIFEST_FILES = (
    "crowd_nav/bayesian_brne/action_conditioned_arhmm.py",
    "crowd_nav/bayesian_brne/data_io.py",
    "crowd_nav/bayesian_brne/interaction_protocol.py",
    "crowd_nav/bayesian_brne/s1_protocol.py",
    "crowd_nav/bayesian_brne/s1_pipeline.py",
    "crowd_nav/tools/s1_strict_gate.py",
    "crowd_nav/tools/run_s1_queue.py",
    "crowd_nav/bayesian_brne/collect_dataset.py",
    "crowd_nav/configs/s1_strict_registry.json",
    COMMITTED_PROTOCOL_SPEC_REL_PATH,
)

# Order S1-0R (R0R-4): of the tracked files above, these three are the S1
# CLI/protocol module ITSELF, which S1-1 through S1-BUILD must keep
# editing -- treating their hash drift as a hard preflight failure would
# make continued development impossible ("必须先修的阻断项...执行纪律自相
# 矛盾"). Until a real ``method_lock.json`` exists (post S1-BUILD), drift
# on exactly these three paths is reported informationally, never raised;
# every OTHER tracked file (the AR-HMM, data_io, interaction_protocol, and
# the registry itself) still hard-fails on any drift, since none of those
# are supposed to change while the S1 CLI is being built out.
SOURCE_MANIFEST_SOFT_PATHS_BEFORE_METHOD_LOCK = frozenset((
    "crowd_nav/bayesian_brne/s1_protocol.py",
    "crowd_nav/bayesian_brne/s1_pipeline.py",
    "crowd_nav/tools/s1_strict_gate.py",
    "crowd_nav/tools/run_s1_queue.py",
))

# The full SM-BRNE-relevant file set backed up into S1-0's rollback tar --
# deliberately scoped (never a bare `git add -A`/whole-workspace tar): this
# repo's `git status --short` carries enormous unrelated content (old
# checkpoints, dozens of `.zip` backups) that must never be swept into an
# SM-BRNE-labeled archive.
ROLLBACK_BAYESIAN_BRNE_GLOBS = ("crowd_nav/bayesian_brne/*.py",)
ROLLBACK_EXTRA_FILES = (
    "crowd_nav/bayesian_brne/README.md",
    "crowd_nav/policy/policy_factory.py",
    "crowd_nav/configs/s1_strict_registry.json",
    "crowd_nav/configs/env_bayesian_brne.config",
    "crowd_nav/configs/policy_bayesian_brne.config",
    "crowd_nav/configs/bayesian_brne_experiment_registry.yaml",
    COMMITTED_PROTOCOL_SPEC_REL_PATH,
)
ROLLBACK_TOOLS_FILES = (
    "crowd_nav/tools/audit_action_identifiability.py",
    "crowd_nav/tools/compare_bayesian_brne.py",
    "crowd_nav/tools/coverage_report.py",
    "crowd_nav/tools/evaluate_sm_brne.py",
    "crowd_nav/tools/f7_benchmark.py",
    "crowd_nav/tools/fit_bayesian_brne.py",
    "crowd_nav/tools/migrate_engineering_artifact_r3.py",
    "crowd_nav/tools/order9s_3b_feature_ablation.py",
    "crowd_nav/tools/order9s_independent_generator.py",
    "crowd_nav/tools/order9s_k_plateau_audit.py",
    "crowd_nav/tools/order9s_orca_negative_control.py",
    "crowd_nav/tools/order9s_shuffle_test.py",
    "crowd_nav/tools/run_bayesian_brne_suite.py",
    "crowd_nav/tools/sm_brne_smoke.py",
    "crowd_nav/tools/s1_strict_gate.py",
    "crowd_nav/tools/run_s1_queue.py",
    "crowd_nav/tools/u3_necessity_gate.py",
)


class RegistryError(ValueError):
    """Raised when the S1 registry is missing a field, has an invalid type/
    range, or its own declared data roles are not mutually seed-disjoint --
    fail closed rather than silently running with an ambiguous protocol."""


class SeedCollisionError(ValueError):
    """Raised when a registry-reserved seed (necessity_id/audit_interactive/
    audit_nominal_negative_control) collides with a suite_seed already used
    by an existing SM-BRNE episode found anywhere under the scanned roots --
    guide.md 13.2's seeds 31-60 must be genuinely unused, not merely assumed
    unused."""


class PreflightError(RuntimeError):
    """Raised when a preflight check itself cannot be completed (missing
    file, unreadable directory) -- distinct from RegistryError/
    SeedCollisionError, which are findings, not execution failures."""


class DataRoleIntegrityError(ValueError):
    """Raised by ``verify_formal_data_role`` (Order S1-0R/R0R-3) when the
    registry's declared ``train``/``selection`` directory does not contain
    EXACTLY what the frozen registry claims: wrong per-seed episode count,
    an unexpected suite_seed, a duplicate ``(suite_seed, episode_seed)``
    or ``initial_state_hash``, or metadata (scenario/split/profile_name/dt/
    n_humans/horizon_steps/controller_type) that disagrees with the
    registry -- fail closed rather than silently trusting "the count looks
    roughly right" or a raw ``np.load`` that bypasses schema validation."""


class StatusError(RuntimeError):
    """Raised by ``update_status_atomic``/``read_status_or_default`` when
    an existing ``status.json`` is corrupt, or its ``experiment_id``/
    ``registry_sha256`` identity does not match the caller's -- guide.md
    S1-0R (R0R-2): two different runs/registries must never silently share
    and overwrite one status file."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(v, int) and not isinstance(v, bool) for v in value)


def load_registry(path: str) -> Dict[str, Any]:
    """Parse and fully validate the S1 strict registry. Raises
    ``RegistryError`` on any missing field, wrong type, out-of-range value,
    or internal seed overlap between the registry's own five data roles --
    never returns a partially-valid registry for a caller to misuse."""
    registry_path = Path(path)
    if not registry_path.exists():
        raise RegistryError(f"registry not found: {path}")
    with open(registry_path, "r") as f:
        registry = json.load(f)

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        _require(field_name in registry, f"registry missing required field {field_name!r}")

    _require(registry["schema_version"] == SCHEMA_VERSION,
              f"registry schema_version={registry['schema_version']!r} does not match expected {SCHEMA_VERSION}")
    _require(isinstance(registry["experiment_id"], str) and registry["experiment_id"],
              "experiment_id must be a non-empty string")
    _require(isinstance(registry["scenario"], str) and registry["scenario"], "scenario must be a non-empty string")
    _require(isinstance(registry["n_humans"], int) and registry["n_humans"] > 0, "n_humans must be a positive int")
    _require(isinstance(registry["dt"], (int, float)) and registry["dt"] > 0, "dt must be > 0")
    _require(isinstance(registry["horizon_steps"], int) and registry["horizon_steps"] > 0,
              "horizon_steps must be a positive int")
    _require(
        isinstance(registry["controllers"], list) and tuple(registry["controllers"]) == REQUIRED_CONTROLLERS,
        f"controllers must be exactly {REQUIRED_CONTROLLERS!r}, got {registry['controllers']!r}",
    )

    for role in ("train", "selection"):
        block = registry[role]
        _require(isinstance(block, dict), f"{role} must be an object")
        for key in _TRAIN_SELECTION_KEYS:
            _require(key in block, f"{role} missing required field {key!r}")
        _require(isinstance(block["path"], str) and block["path"], f"{role}.path must be a non-empty string")
        _require(_is_int_list(block["suite_seeds"]), f"{role}.suite_seeds must be a non-empty list of ints")
        _require(len(set(block["suite_seeds"])) == len(block["suite_seeds"]), f"{role}.suite_seeds must not contain duplicates")
        _require(isinstance(block["episodes_per_seed"], int) and block["episodes_per_seed"] > 0,
                  f"{role}.episodes_per_seed must be a positive int")

    for role in ("necessity_id", "audit_interactive", "audit_nominal_negative_control"):
        block = registry[role]
        _require(isinstance(block, dict), f"{role} must be an object")
        for key in _GENERATED_SPLIT_KEYS:
            _require(key in block, f"{role} missing required field {key!r}")
        _require(isinstance(block["environment_split"], str) and block["environment_split"],
                  f"{role}.environment_split must be a non-empty string")
        _require(isinstance(block["output_root"], str) and block["output_root"], f"{role}.output_root must be a non-empty string")
        _require(_is_int_list(block["suite_seeds"]), f"{role}.suite_seeds must be a non-empty list of ints")
        _require(len(set(block["suite_seeds"])) == len(block["suite_seeds"]), f"{role}.suite_seeds must not contain duplicates")
        _require(isinstance(block["episodes_per_seed"], int) and block["episodes_per_seed"] > 0,
                  f"{role}.episodes_per_seed must be a positive int")
        _require(isinstance(block["profile_name"], str) and block["profile_name"], f"{role}.profile_name must be a non-empty string")

    _require(_is_int_list(registry["primary_k_candidates"]), "primary_k_candidates must be a non-empty list of ints")
    _require(_is_int_list(registry["boundary_extension_k_candidates"]), "boundary_extension_k_candidates must be a non-empty list of ints")
    _require(min(registry["primary_k_candidates"]) >= 1, "primary_k_candidates must all be >= 1")
    _require(1 in registry["primary_k_candidates"], "primary_k_candidates must include K=1 as the no-switching baseline")
    _require(
        set(registry["boundary_extension_k_candidates"]).isdisjoint(registry["primary_k_candidates"]),
        "boundary_extension_k_candidates must not overlap primary_k_candidates",
    )
    _require(max(registry["boundary_extension_k_candidates"]) <= 6, "boundary_extension_k_candidates must not exceed K=6 (guide.md 13.2)")

    _require(_is_int_list(registry["restart_seeds"]) and len(registry["restart_seeds"]) >= 3,
              "restart_seeds must list at least 3 distinct restart seeds")

    for key in ("sticky_kappa", "dirichlet_alpha", "shrinkage_scale", "inverse_wishart_dof", "inverse_wishart_scale", "em_tol"):
        _require(isinstance(registry[key], (int, float)) and registry[key] > 0, f"{key} must be > 0")
    _require(isinstance(registry["em_max_iters"], int) and registry["em_max_iters"] > 0, "em_max_iters must be a positive int")
    _require(isinstance(registry["bootstrap_resamples"], int) and registry["bootstrap_resamples"] > 0,
              "bootstrap_resamples must be a positive int")
    _require(isinstance(registry["bootstrap_seed"], int), "bootstrap_seed must be an int")
    _require(isinstance(registry["confidence_level"], (int, float)) and 0.0 < registry["confidence_level"] < 1.0,
              "confidence_level must be in (0, 1)")
    _require(isinstance(registry["nominal_equivalence_margin_nats_per_row"], (int, float))
              and registry["nominal_equivalence_margin_nats_per_row"] > 0,
              "nominal_equivalence_margin_nats_per_row must be > 0")
    _require(isinstance(registry["min_mode_fraction"], (int, float)) and 0.0 < registry["min_mode_fraction"] < 1.0,
              "min_mode_fraction must be in (0, 1)")
    _require(isinstance(registry["max_predictive_similarity"], (int, float)) and 0.0 < registry["max_predictive_similarity"] <= 1.0,
              "max_predictive_similarity must be in (0, 1]")

    # Internal seed-disjointness: the registry's own five data roles must
    # never share a suite_seed with each other -- this is checkable from
    # the registry's OWN content alone, before touching disk.
    role_seeds: Dict[str, Set[int]] = {
        role: set(registry[role]["suite_seeds"])
        for role in ("train", "selection", "necessity_id", "audit_interactive", "audit_nominal_negative_control")
    }
    roles = list(role_seeds.keys())
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            overlap = role_seeds[roles[i]] & role_seeds[roles[j]]
            _require(not overlap, f"registry suite_seeds overlap between {roles[i]!r} and {roles[j]!r}: {sorted(overlap)}")

    return registry


def new_reserved_seeds(registry: Dict[str, Any]) -> Dict[str, Set[int]]:
    """The three data roles this Order is about to generate FRESH data for
    -- necessity_id/audit_interactive/audit_nominal_negative_control --
    keyed by role name. ``train``/``selection`` already have data on disk
    (guide.md 13.2 notes train/selection reuse the existing formal
    collection) and are checked for self-consistency, not treated as "new"."""
    return {
        role: set(registry[role]["suite_seeds"])
        for role in ("necessity_id", "audit_interactive", "audit_nominal_negative_control")
    }


@dataclass
class EpisodeIdentityScan:
    # suite_seed -> set of episode_seed found under that suite_seed.
    episode_seeds_by_suite_seed: Dict[int, Set[int]]
    # initial_state_hash -> list of (path) for duplicate-hash diagnostics.
    paths_by_initial_state_hash: Dict[str, List[str]]
    n_files_scanned: int
    n_files_matched: int
    n_files_skipped_non_episode: int
    scanned_roots: List[str]


# Order S1-0R-A (A3): the ONLY allowlisted key signatures a non-episode
# npz under runs/ is permitted to have, to be skipped rather than raising.
# A file "not looking like an episode" because it happens to be MISSING
# one field is never grounds for a silent skip -- it must positively match
# one of these known-good shapes. Currently only ARHMMArtifact.save()'s
# ``arhmm.npz`` (K/A_k/B_k/C_k/d_k/Q_k/Pi/initial_distribution/dt).
_KNOWN_NON_EPISODE_KEY_SIGNATURES: Tuple[FrozenSet[str], ...] = (
    frozenset({"K", "Pi", "initial_distribution", "dt"}),
)


def _is_known_non_episode_artifact(keys: Set[str]) -> bool:
    return any(signature.issubset(keys) for signature in _KNOWN_NON_EPISODE_KEY_SIGNATURES)


def _is_well_formed_sha256_hex(value: str) -> bool:
    return bool(value) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def scan_existing_episode_identities(roots: Sequence[str]) -> EpisodeIdentityScan:
    """Scan every ``*.npz`` under ``roots`` and record each SM-BRNE
    episode's ``(suite_seed, episode_seed, initial_state_hash)``.

    Order S1-0R (R0R-3) fix: a file is only "skipped, not an episode" when
    it loads fine as an npz AND clearly lacks ``suite_seed``/``episode_seed``
    (e.g. an ``ARHMMArtifact``'s ``arhmm.npz``, whose keys are K/A/B/C/d/Q,
    nothing episode-shaped) -- that is a real, checkable "wrong file type"
    determination, not a swallowed exception. If the npz fails to open at
    all, OR it has those keys but reading them (or ``initial_state_hash``)
    raises, this now raises ``PreflightError`` instead of silently counting
    it as skipped: a corrupt file sitting where episodes are expected must
    block, not quietly vanish from the identity count guide.md 13.2's
    "3960 episodes, 0 collisions" claim depends on."""
    episode_seeds_by_suite_seed: Dict[int, Set[int]] = {}
    paths_by_initial_state_hash: Dict[str, List[str]] = {}
    n_scanned = 0
    n_matched = 0
    n_skipped = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for npz_path in sorted(root_path.rglob("*.npz")):
            n_scanned += 1
            try:
                with np.load(npz_path, allow_pickle=False) as npz:
                    keys = set(npz.files)
                    if "suite_seed" not in keys or "episode_seed" not in keys:
                        if _is_known_non_episode_artifact(keys):
                            n_skipped += 1
                            continue
                        raise PreflightError(
                            f"npz file has neither suite_seed/episode_seed NOR a recognized "
                            f"non-episode artifact key signature (not silently skipped): {npz_path} "
                            f"keys={sorted(keys)}"
                        )
                    suite_seed = int(npz["suite_seed"])
                    episode_seed = int(npz["episode_seed"])
                    # Order S1-0R-A (A3) fix: an earlier version defaulted
                    # to an empty string when "initial_state_hash" was
                    # absent, silently treating a suspected episode's
                    # missing/malformed third identity field as harmless --
                    # independently reproduced: a file with ONLY
                    # suite_seed/episode_seed (no initial_state_hash at
                    # all) was accepted as matched=1, skipped=0, no error.
                    # This function's own docstring promises a THREE-field
                    # identity contract; a file that clearly claims to be
                    # an episode (has suite_seed/episode_seed) but lacks a
                    # valid hash must fail closed, not silently degrade to
                    # a two-field identity.
                    if "initial_state_hash" not in keys:
                        raise PreflightError(
                            f"suspected SM-BRNE episode has suite_seed/episode_seed but no "
                            f"initial_state_hash at all: {npz_path}"
                        )
                    init_hash = str(npz["initial_state_hash"])
                    if not _is_well_formed_sha256_hex(init_hash):
                        raise PreflightError(
                            f"suspected SM-BRNE episode has a missing/malformed initial_state_hash "
                            f"(expected 64 lowercase hex chars, got {init_hash!r}): {npz_path}"
                        )
            except PreflightError:
                raise
            except Exception as exc:
                raise PreflightError(
                    f"suspected SM-BRNE episode file failed to read cleanly (not silently skipped): "
                    f"{npz_path}: {exc}"
                )
            n_matched += 1
            episode_seeds_by_suite_seed.setdefault(suite_seed, set()).add(episode_seed)
            paths_by_initial_state_hash.setdefault(init_hash, []).append(str(npz_path))
    return EpisodeIdentityScan(
        episode_seeds_by_suite_seed=episode_seeds_by_suite_seed,
        paths_by_initial_state_hash=paths_by_initial_state_hash,
        n_files_scanned=n_scanned,
        n_files_matched=n_matched,
        n_files_skipped_non_episode=n_skipped,
        scanned_roots=[str(r) for r in roots],
    )


def check_seed_disjoint(registry: Dict[str, Any], scan: EpisodeIdentityScan) -> Dict[str, Any]:
    """Raise ``SeedCollisionError`` if any registry-reserved suite_seed
    (necessity_id/audit_interactive/audit_nominal_negative_control) is
    already used by an episode found on disk anywhere under the scanned
    roots. Returns a report dict on success (no exception) so callers can
    still record what was checked."""
    reserved = new_reserved_seeds(registry)
    existing_suite_seeds = set(scan.episode_seeds_by_suite_seed.keys())
    collisions: Dict[str, List[int]] = {}
    for role, seeds in reserved.items():
        hit = sorted(seeds & existing_suite_seeds)
        if hit:
            collisions[role] = hit
    if collisions:
        raise SeedCollisionError(
            f"registry-reserved suite_seeds already used by existing episodes on disk: {collisions} "
            "-- guide.md 13.2 requires seeds 31-60 be genuinely unused; do not silently pick new seeds, stop and report"
        )
    return {
        "reserved_seeds": {role: sorted(seeds) for role, seeds in reserved.items()},
        "existing_suite_seeds_found": sorted(existing_suite_seeds),
        "n_files_scanned": scan.n_files_scanned,
        "n_files_matched": scan.n_files_matched,
        "n_files_skipped_non_episode": scan.n_files_skipped_non_episode,
        "collision": False,
    }


# role name (as it appears in the registry) -> the data_io/schemas.py
# ``split`` field its episodes are stamped with (guide.md 13.2's
# "selection" role reuses the "validation" environment split).
_ROLE_TO_EXPECTED_SPLIT = {"train": "train", "selection": "validation"}


def verify_formal_data_role(repo_root_path: str, role_name: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    """Strictly verify ``registry[role_name]``'s declared directory
    (``train`` or ``selection``) against the registry's own frozen counts
    and metadata -- Order S1-0R (R0R-3), enriched by Order S1-0R-A (A2).
    Every file is loaded via ``data_io.load_episode`` (full schema
    validation), never a raw ``np.load`` that could be fooled by a
    malformed/out-of-schema file. Raises ``DataRoleIntegrityError`` on: a
    per-seed count that isn't exactly ``episodes_per_seed``, any suite_seed
    outside the registry's declared set, a duplicate ``(suite_seed,
    episode_seed)`` within the role, a duplicate ``initial_state_hash``, an
    unbalanced/missing controller (guide.md's Order 6.1 equal allocation
    means the current frozen dataset must have EXACTLY
    ``total_episodes / len(controllers)`` per controller type), or any
    episode whose scenario/split/profile_name/dt/n_humans/horizon_steps/
    controller_type disagrees with the frozen registry."""
    from crowd_nav.bayesian_brne import data_io

    if role_name not in ("train", "selection"):
        raise ValueError(f"verify_formal_data_role only supports 'train'/'selection', got {role_name!r}")
    role_config = registry[role_name]
    controllers = list(registry["controllers"])
    expected_split = _ROLE_TO_EXPECTED_SPLIT[role_name]
    directory = Path(repo_root_path) / role_config["path"]
    if not directory.is_dir():
        raise DataRoleIntegrityError(f"{role_name}.path does not exist or is not a directory: {directory}")

    expected_seeds = set(role_config["suite_seeds"])
    seed_counts: Dict[int, int] = {s: 0 for s in expected_seeds}
    controller_counts: Dict[str, int] = {c: 0 for c in controllers}
    seen_episode_ids: Set[Tuple[int, int]] = set()
    seen_episode_seeds: Set[int] = set()
    initial_state_hashes: Set[str] = set()
    extra_seeds: Set[int] = set()

    files = sorted(directory.glob("*.npz"))
    if not files:
        raise DataRoleIntegrityError(f"{role_name}.path contains no .npz files: {directory}")

    for f in files:
        try:
            episode = data_io.load_episode(str(f))
        except Exception as exc:
            raise DataRoleIntegrityError(f"{role_name}: {f} failed data_io.load_episode schema validation: {exc}")

        if episode["scenario"] != registry["scenario"]:
            raise DataRoleIntegrityError(
                f"{role_name}: {f} scenario={episode['scenario']!r} does not match registry scenario={registry['scenario']!r}"
            )
        if episode["split"] != expected_split:
            raise DataRoleIntegrityError(
                f"{role_name}: {f} split={episode['split']!r} does not match expected split={expected_split!r}"
            )
        if episode["profile_name"] != "formal":
            raise DataRoleIntegrityError(f"{role_name}: {f} profile_name={episode['profile_name']!r}, expected 'formal'")
        if abs(float(episode["dt"]) - float(registry["dt"])) > 1e-12:
            raise DataRoleIntegrityError(f"{role_name}: {f} dt={episode['dt']} does not match registry dt={registry['dt']}")
        n_humans = np.asarray(episode["humans"]).shape[1]
        if n_humans != registry["n_humans"]:
            raise DataRoleIntegrityError(f"{role_name}: {f} has {n_humans} humans, registry requires {registry['n_humans']}")
        horizon = np.asarray(episode["humans"]).shape[0]
        if horizon != registry["horizon_steps"]:
            raise DataRoleIntegrityError(
                f"{role_name}: {f} has {horizon} timesteps, registry requires horizon_steps={registry['horizon_steps']}"
            )
        controller_type = episode["controller_type"]
        if controller_type not in controllers:
            raise DataRoleIntegrityError(
                f"{role_name}: {f} controller_type={controller_type!r} not in registry controllers={controllers}"
            )

        suite_seed = int(episode["suite_seed"])
        episode_seed = int(episode["episode_seed"])
        if suite_seed not in expected_seeds:
            extra_seeds.add(suite_seed)
            continue
        episode_id = (suite_seed, episode_seed)
        if episode_id in seen_episode_ids:
            raise DataRoleIntegrityError(f"{role_name}: duplicate (suite_seed, episode_seed)={episode_id} at {f}")
        if episode_seed in seen_episode_seeds:
            raise DataRoleIntegrityError(f"{role_name}: duplicate raw episode_seed={episode_seed} at {f}")
        seen_episode_ids.add(episode_id)
        seen_episode_seeds.add(episode_seed)
        init_hash = str(episode["initial_state_hash"])
        if init_hash in initial_state_hashes:
            raise DataRoleIntegrityError(f"{role_name}: duplicate initial_state_hash={init_hash} at {f}")
        initial_state_hashes.add(init_hash)
        seed_counts[suite_seed] += 1
        controller_counts[controller_type] += 1

    if extra_seeds:
        raise DataRoleIntegrityError(
            f"{role_name}: found suite_seed(s) not declared in registry: {sorted(extra_seeds)}"
        )
    for seed, count in seed_counts.items():
        if count != role_config["episodes_per_seed"]:
            raise DataRoleIntegrityError(
                f"{role_name}: suite_seed={seed} has {count} episodes, registry requires exactly "
                f"{role_config['episodes_per_seed']}"
            )

    total_episodes = sum(seed_counts.values())
    if total_episodes % len(controllers) != 0:
        raise DataRoleIntegrityError(
            f"{role_name}: total episode count {total_episodes} is not evenly divisible by "
            f"{len(controllers)} controllers -- cannot be exactly balanced"
        )
    expected_per_controller = total_episodes // len(controllers)
    unbalanced = {c: n for c, n in controller_counts.items() if n != expected_per_controller}
    if unbalanced:
        raise DataRoleIntegrityError(
            f"{role_name}: controller counts are not exactly balanced (expected "
            f"{expected_per_controller} each of {controllers}): {controller_counts}"
        )

    return {
        "role": role_name,
        "n_files": len(files),
        "seed_counts": {str(s): c for s, c in sorted(seed_counts.items())},
        "controller_counts": dict(sorted(controller_counts.items())),
        "n_unique_episode_id": len(seen_episode_ids),
        "n_unique_initial_state_hash": len(initial_state_hashes),
        "initial_state_hashes": sorted(initial_state_hashes),
        "episode_seeds": sorted(seen_episode_seeds),
        "episode_ids": sorted(seen_episode_ids),
        "file_list_sha256": _hash_string_set([f.name for f in files]),
        "episode_ids_sha256": _hash_string_set([f"{s},{e}" for s, e in seen_episode_ids]),
        "episode_seeds_sha256": _hash_string_set([str(e) for e in seen_episode_seeds]),
        "initial_state_hashes_sha256_of_set": _hash_string_set(initial_state_hashes),
    }


def _hash_string_set(values: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def verify_train_selection_cross_disjoint(train_report: Dict[str, Any], selection_report: Dict[str, Any]) -> None:
    """Raise ``DataRoleIntegrityError`` if train and selection share any
    ``initial_state_hash``, raw ``episode_seed``, or ``(suite_seed,
    episode_seed)`` identity. Order S1-0R-A (A2): suite_seed disjointness
    already follows structurally from the registry's own seed-disjointness
    check, and episode_seed/episode_id disjointness follows from suite_seed
    disjointness by the ``suite_seed*100000+ep`` construction -- but
    guide.md wants this checked EXPLICITLY against the actual data, not
    merely assumed from the seed-generation convention, in case a bug ever
    let an out-of-range episode_seed reach disk."""
    hash_overlap = set(train_report["initial_state_hashes"]) & set(selection_report["initial_state_hashes"])
    if hash_overlap:
        raise DataRoleIntegrityError(
            f"train and selection share {len(hash_overlap)} initial_state_hash value(s): "
            f"{sorted(hash_overlap)[:5]}{'...' if len(hash_overlap) > 5 else ''}"
        )
    seed_overlap = set(train_report["episode_seeds"]) & set(selection_report["episode_seeds"])
    if seed_overlap:
        raise DataRoleIntegrityError(
            f"train and selection share {len(seed_overlap)} raw episode_seed value(s): "
            f"{sorted(seed_overlap)[:5]}{'...' if len(seed_overlap) > 5 else ''}"
        )
    id_overlap = set(map(tuple, train_report["episode_ids"])) & set(map(tuple, selection_report["episode_ids"]))
    if id_overlap:
        raise DataRoleIntegrityError(
            f"train and selection share {len(id_overlap)} (suite_seed, episode_seed) pair(s): "
            f"{sorted(id_overlap)[:5]}{'...' if len(id_overlap) > 5 else ''}"
        )


def _git_status_short(repo_dir: Path, paths: Sequence[str]) -> str:
    """``git status --short`` scoped to ``paths`` only -- guide.md S1-0:
    "不得因全仓其他脏文件而失败，但相关文件在冻结后变化必须失败" -- the huge
    amount of unrelated dirty content elsewhere in this repo (old
    checkpoints, .zip backups) must never affect this check; only the files
    this manifest actually tracks matter."""
    try:
        out = subprocess.run(
            ["git", "status", "--short", "--"] + list(paths),
            cwd=str(repo_dir), capture_output=True, text=True, check=True,
        )
        return out.stdout
    except Exception as exc:
        return f"unavailable: {exc}"


def build_source_manifest(repo_root: str, extra_files: Sequence[str] = ()) -> Dict[str, Any]:
    """SHA256 of every file in ``SOURCE_MANIFEST_FILES`` (plus any
    ``extra_files``), plus interpreter/library versions, git HEAD, and a
    git-status scoped ONLY to the tracked files -- guide.md S1-0's
    ``source_manifest.json``."""
    root = Path(repo_root)
    tracked = list(SOURCE_MANIFEST_FILES) + list(extra_files)
    source_sha256: Dict[str, str] = {}
    missing: List[str] = []
    for rel_path in tracked:
        full = root / rel_path
        if not full.exists():
            missing.append(rel_path)
            continue
        source_sha256[rel_path] = provenance.sha256_file(full)
    return {
        "tracked_files": tracked,
        "missing_files": missing,
        "source_sha256": source_sha256,
        "git_head": provenance.git_head(root),
        "git_status_short_scoped": _git_status_short(root, tracked),
        "python_version": sys.version,
        "numpy_version": provenance.package_version("numpy"),
        "scipy_version": provenance.package_version("scipy"),
        "numba_version": provenance.package_version("numba"),
        "platform": platform.platform(),
    }


def verify_source_manifest_unchanged(
    repo_root: str, frozen_source_manifest: Dict[str, Any], soft_paths: FrozenSet[str] = frozenset(),
) -> Dict[str, List[str]]:
    """Recompute hashes for every file in a previously-frozen source
    manifest and raise if any HARD-tracked file changed -- "相关文件在冻结后
    变化必须失败" (S1-0). Order S1-0R (R0R-4): ``soft_paths`` (the S1 CLI/
    protocol module files themselves, which S1-1..S1-BUILD must keep
    editing) are reported as drifted in the returned dict but NEVER raise
    -- there is no real "frozen source" for code that is, by design, still
    being built; only files outside ``soft_paths`` are true scaffold-time
    invariants. Missing files at freeze time are not re-checked (already
    reported missing then); a file that exists now but was recorded
    missing before is itself a change and always hard-fails, even if it is
    one of the soft paths -- guide.md never anticipated a TRACKED file not
    existing at all."""
    root = Path(repo_root)
    hard_changed: List[str] = []
    soft_changed: List[str] = []
    for rel_path, frozen_hash in frozen_source_manifest["source_sha256"].items():
        full = root / rel_path
        if not full.exists():
            raise PreflightError(f"tracked file missing since freeze: {rel_path}")
        current_hash = provenance.sha256_file(full)
        if current_hash != frozen_hash:
            if rel_path in soft_paths:
                soft_changed.append(rel_path)
            else:
                hard_changed.append(rel_path)
    if hard_changed:
        raise PreflightError(
            f"tracked file(s) changed since freeze: {hard_changed} -- these are outside the "
            "soft (actively-being-built) path set, so drift here is a real regression, not "
            "expected development churn"
        )
    frozen_missing = set(frozen_source_manifest.get("missing_files", []))
    for rel_path in frozen_missing:
        full = root / rel_path
        if full.exists():
            raise PreflightError(f"tracked file appeared after being recorded missing at freeze: {rel_path}")
    return {"soft_changed": soft_changed, "hard_changed": hard_changed}


def registry_content_sha256(registry: Dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_dumps(registry).encode("utf-8")).hexdigest()


def build_rollback_archive(repo_root: str, output_dir: str, file_list: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """One-time S1-0 rollback point: tar every SM-BRNE-relevant source/
    config file (never a bare whole-workspace archive -- this repo's git
    status carries large unrelated content) into
    ``<output_dir>/sm_brne_s1_0_rollback.tar.gz``, plus a per-file SHA256
    manifest, so a later regression can be diffed against a known-good
    snapshot taken before any S1 data collection or fitting began.

    ``file_list``, if given, REPLACES the real (large) SM-BRNE file set --
    used only by selftest to exercise the tar/manifest/restore-verify
    machinery against a small synthetic file set without duplicating the
    full production archive on every test run."""
    import hashlib
    import tarfile

    root = Path(repo_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_list is not None:
        files = sorted(set(file_list))
    else:
        files = []
        for pattern in ROLLBACK_BAYESIAN_BRNE_GLOBS:
            files.extend(str(p.relative_to(root)) for p in sorted(root.glob(pattern)) if p.is_file())
        files.extend(ROLLBACK_EXTRA_FILES)
        files.extend(ROLLBACK_TOOLS_FILES)
        files = sorted(set(files))

    per_file_sha256: Dict[str, str] = {}
    missing: List[str] = []
    archive_path = out_dir / "sm_brne_s1_0_rollback.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for rel_path in files:
            full = root / rel_path
            if not full.exists():
                missing.append(rel_path)
                continue
            per_file_sha256[rel_path] = provenance.sha256_file(full)
            tar.add(full, arcname=rel_path)

    archive_sha256 = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            archive_sha256.update(chunk)

    try:
        archive_path_recorded = str(archive_path.resolve().relative_to(root.resolve()))
    except ValueError:
        archive_path_recorded = str(archive_path)

    manifest = {
        "archive_path": archive_path_recorded,
        "archive_sha256": archive_sha256.hexdigest(),
        "n_files": len(per_file_sha256),
        "missing_files": missing,
        "per_file_sha256": per_file_sha256,
    }
    with open(out_dir / "sm_brne_s1_0_rollback_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def verify_rollback_archive(repo_root: str, manifest: Dict[str, Any]) -> bool:
    """Verify the rollback archive itself, THEN extract and confirm every
    recorded file hash matches -- the same restore-verification discipline
    R0 used, so a claimed rollback point is provably restorable, not just
    an unverified tar.

    Order S1-0R (R0R-5) fix: the original version only checked per-file
    hashes AFTER extraction, never the tar's own ``archive_sha256`` --  a
    tampered/corrupted/truncated archive that still happened to extract
    something for every recorded path would not necessarily be caught.
    This now hashes the archive file itself FIRST and refuses to even
    attempt extraction if it disagrees with ``manifest['archive_sha256']``;
    a corrupt/unreadable tar is caught as a clean ``False``, never an
    uncaught exception that could be mistaken by a caller for "not run
    yet"."""
    import tarfile
    import tempfile

    archive_path = Path(repo_root) / manifest["archive_path"]
    if not archive_path.exists():
        return False
    if provenance.sha256_file(archive_path) != manifest["archive_sha256"]:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmp)
            for rel_path, expected_hash in manifest["per_file_sha256"].items():
                extracted = Path(tmp) / rel_path
                if not extracted.exists():
                    return False
                if provenance.sha256_file(extracted) != expected_hash:
                    return False
    except Exception:
        return False
    return True


# --------------------------------------------------------------------- #
# status.json: ONE schema, ONE atomic writer (Order S1-0R / R0R-2). Both
# CLIs (``s1_strict_gate.py``'s per-stage result, ``run_s1_queue.py``'s
# queue-level progress) call ``update_status_atomic`` exclusively -- never
# ``json.dump``/``write_text`` the whole file themselves, which is exactly
# what let the two CLIs silently clobber each other's status fields.
# --------------------------------------------------------------------- #

# Order S1-0R-A (A4) bumped this to v2: v1 had a single ambiguous ``pid``
# field that a direct (non-queue) stage invocation and the queue's own
# process both wrote to, plus a ``started_at`` the queue rewrote on every
# stage instead of once -- independently reproduced: after a queue run
# followed by one direct ``--stage preflight`` invocation, status.json's
# ``pid`` was the direct invocation's PID while ``queue_status`` still said
# ``COMPLETED_REQUESTED_STAGES`` and ``started_at`` was the OLD queue run's
# timestamp -- three fields from two unrelated executions, no reader could
# reconstruct what actually happened. v2 never has a bare ``pid``; queue
# and stage identity/timing are fully separate fields.
STATUS_SCHEMA_VERSION = 2
QUEUE_STATUS_FIELDS = ("queue_pid", "queue_status", "queue_started_at")
STAGE_STATUS_FIELDS = (
    "invocation_mode", "stage_pid", "current_stage", "stage_status",
    "stage_started_at", "stage_finished_at", "returncode", "completed_stages", "active_fit",
)
STATUS_FIELDS = QUEUE_STATUS_FIELDS + STAGE_STATUS_FIELDS
VALID_INVOCATION_MODES = ("queue", "direct")


def read_status_or_default(output_dir: str) -> Dict[str, Any]:
    """Read ``<output_dir>/status.json`` if present; otherwise return an
    empty-but-well-shaped default. Raises ``StatusError`` if the file
    exists but is not valid JSON -- a corrupt status file must never be
    silently treated as "no status yet"."""
    path = Path(output_dir) / "status.json"
    if not path.exists():
        return {"schema_version": STATUS_SCHEMA_VERSION, "completed_stages": []}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise StatusError(f"existing status.json is corrupt, refusing to treat as absent: {path}: {exc}")


def update_status_atomic(output_dir: str, *, experiment_id: str, registry_sha256: str, **fields: Any) -> Dict[str, Any]:
    """Merge ``fields`` (a subset of ``STATUS_FIELDS``) into
    ``<output_dir>/status.json``, identity-checked and written atomically
    (temp file + ``os.replace``) so a reader never observes a
    partially-written file. Any field NOT passed keeps its previous value
    (or ``None`` if never set) -- this is a merge, never a whole-file
    overwrite, which is the concrete fix for gate/queue clobbering each
    other. Raises ``StatusError`` if an existing status.json belongs to a
    DIFFERENT experiment_id/registry_sha256 (two different runs must never
    share one status file) or has an unrecognized schema_version (v1 files
    from before Order S1-0R-A must be deleted, not silently migrated --
    they describe pre-fix, ambiguous-identity runs)."""
    unknown = set(fields) - set(STATUS_FIELDS)
    if unknown:
        raise ValueError(f"update_status_atomic got unknown field(s) {sorted(unknown)}, expected subset of {STATUS_FIELDS}")
    if "invocation_mode" in fields and fields["invocation_mode"] not in VALID_INVOCATION_MODES:
        raise ValueError(f"invocation_mode must be one of {VALID_INVOCATION_MODES}, got {fields['invocation_mode']!r}")

    path = Path(output_dir) / "status.json"
    existing = read_status_or_default(output_dir)
    if existing.get("schema_version") not in (None, STATUS_SCHEMA_VERSION):
        raise StatusError(
            f"status.json schema_version={existing.get('schema_version')} does not match expected "
            f"{STATUS_SCHEMA_VERSION} -- delete the stale file rather than migrating it (a v1 file "
            "describes a run whose queue/stage identity was ambiguous by construction)"
        )
    if "experiment_id" in existing and existing["experiment_id"] not in (None, experiment_id):
        raise StatusError(
            f"status.json belongs to a different experiment_id (existing={existing['experiment_id']!r}, "
            f"this call={experiment_id!r}) -- refusing to let two different runs share one status file"
        )
    if "registry_sha256" in existing and existing["registry_sha256"] not in (None, registry_sha256):
        raise StatusError(
            f"status.json belongs to a different registry_sha256 (existing={existing['registry_sha256']!r}, "
            f"this call={registry_sha256!r}) -- refusing to let a changed registry silently share one status file"
        )

    merged = dict(existing)
    merged["schema_version"] = STATUS_SCHEMA_VERSION
    merged["experiment_id"] = experiment_id
    merged["registry_sha256"] = registry_sha256
    for key in STATUS_FIELDS:
        if key in fields:
            merged[key] = fields[key]
        elif key not in merged:
            merged[key] = None
    merged["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)
    return merged


def _pid_cmdline(pid: int) -> Optional[str]:
    """Best-effort ``/proc/<pid>/cmdline`` read; ``None`` if the process
    does not exist or ``/proc`` is unavailable (never raises)."""
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        return None
    return raw.decode("utf-8", errors="replace").replace("\x00", " ")


def is_recorded_stage_process_alive(status: Dict[str, Any], registry_path: str) -> bool:
    """True only if ``status['stage_pid']`` names a PROCESS THAT IS STILL
    THIS RUN's stage invocation -- Order S1-0R-A (A4): "仅PID数字相同不算
    同一任务" (a matching PID number alone does not prove it's the same
    task; PIDs get reused by unrelated processes over time). Requires the
    live process's cmdline to contain BOTH the ``s1_strict_gate`` module
    name and the exact ``--registry`` path this status file was written
    for -- a coincidentally-reused PID running an unrelated command fails
    both checks."""
    pid = status.get("stage_pid")
    if not pid:
        return False
    cmdline = _pid_cmdline(pid)
    if cmdline is None:
        return False
    return "s1_strict_gate" in cmdline and str(registry_path) in cmdline


def is_recorded_queue_process_alive(status: Dict[str, Any], registry_path: str) -> bool:
    """The queue-level analogue of ``is_recorded_stage_process_alive``."""
    pid = status.get("queue_pid")
    if not pid:
        return False
    cmdline = _pid_cmdline(pid)
    if cmdline is None:
        return False
    return "run_s1_queue" in cmdline and str(registry_path) in cmdline


# --------------------------------------------------------------------- #
# frozen_protocol_spec.md (Order S1-0R / R0R-4, hardened by S1-0R-A / A1):
# guide.md's normative S1 content claims to be frozen. This makes that
# claim checkable WITHOUT any hardcoded path to guide.md in this module --
# guide.md lives outside the repository and is not something 4090 needs to
# read, so the caller (a local-only, optional CLI flag) supplies its text;
# extraction is pure and testable on a plain string.
#
# Order S1-0R-A (A1) fix: the original extractor used a single heading
# ("## 13") to a single stop condition ("the next `## `") -- independently
# reproduced to capture ONLY the original 13.1-13.3 (340 lines) and miss
# every amendment written as its OWN top-level "## Order ..." heading
# after section 13 (R0R-1..7, "新增S1-BUILD", and S1-0R-A's own A1..A6).
# This version instead scans for EXPLICIT
# ``<!-- S1_PROTOCOL_FREEZE_START -->``/``<!-- S1_PROTOCOL_FREEZE_END -->``
# marker pairs (guide.md now wraps THREE such regions: 13.1-13.3, the
# R0R-1..7/S1-BUILD amendment, and the A1..A6/verification-command
# amendment) and concatenates all of them, in document order -- never a
# heading-heuristic. Execution reports (CC's/Codex's "Order ... 执行报告"
# narrative, evidence, and verdict prose) are deliberately NOT inside any
# marker pair, so they can never leak into the frozen spec.
# --------------------------------------------------------------------- #

PROTOCOL_SPEC_FREEZE_START_MARKER = "<!-- S1_PROTOCOL_FREEZE_START -->"
PROTOCOL_SPEC_FREEZE_END_MARKER = "<!-- S1_PROTOCOL_FREEZE_END -->"

# COMMITTED_PROTOCOL_SPEC_REL_PATH is defined near SOURCE_MANIFEST_FILES
# above (it is one of the tracked/hard-verified files) -- tracked there,
# not redefined here, so its hash is ALWAYS checked by ordinary preflight
# (build_source_manifest/verify_source_manifest_unchanged), everywhere,
# with no dependency on --protocol-spec-path being passed (guide.md
# 13/S1-0R-A A1 item 5: "4090即使没有本地guide.md，也必须同步并校验仓库内
# 已冻结的protocol spec/hash，不能跳过就算通过").

# Strings that must NEVER appear inside the extracted spec -- a stray
# execution-report heading leaking in would mean the marker placement (or
# a future guide.md edit) broke the normative/narrative boundary again.
_FORBIDDEN_IN_FROZEN_SPEC = ("Order S1-0 执行报告", "Order S1-0R 执行报告", "独立验收与阻断修补", "独立复验与最终收口")

# Guide.md's own explicit self-check strings (guide.md 13/S1-0R-A A1):
# the frozen spec must contain evidence of the R0R and A-series amendments
# actually being present, not just the original 13.1-13.3.
_REQUIRED_IN_FROZEN_SPEC = ("R0R-1", "新增S1-BUILD", "A1", "A6")


def extract_frozen_protocol_spec(guide_md_text: str) -> str:
    """Concatenate every ``S1_PROTOCOL_FREEZE_START``/``_END`` marker-pair
    region in ``guide_md_text``, in document order. Raises
    ``PreflightError`` if there are zero pairs, an unequal number of start
    vs. end markers, an end before its matching start, or if the
    concatenated result contains a forbidden execution-report substring or
    is missing a required normative substring -- this function refuses to
    return a spec it cannot itself prove is complete and report-free."""
    lines = guide_md_text.splitlines(keepends=True)
    regions: List[str] = []
    i = 0
    n_starts = 0
    n_ends = 0
    while i < len(lines):
        if lines[i].startswith(PROTOCOL_SPEC_FREEZE_START_MARKER):
            n_starts += 1
            start = i + 1
            end = None
            for j in range(start, len(lines)):
                if lines[j].startswith(PROTOCOL_SPEC_FREEZE_END_MARKER):
                    end = j
                    n_ends += 1
                    break
                if lines[j].startswith(PROTOCOL_SPEC_FREEZE_START_MARKER):
                    raise PreflightError(
                        f"nested {PROTOCOL_SPEC_FREEZE_START_MARKER} at line {j} before its predecessor "
                        f"(line {i}) was closed -- marker pairs must not nest or overlap"
                    )
            if end is None:
                raise PreflightError(f"{PROTOCOL_SPEC_FREEZE_START_MARKER} at line {i} has no matching END marker")
            regions.append("".join(lines[start:end]))
            i = end + 1
        else:
            i += 1

    if n_starts == 0:
        raise PreflightError(
            f"no {PROTOCOL_SPEC_FREEZE_START_MARKER!r} marker found in provided guide.md text -- "
            "the normative S1 spec must be explicitly marked, never inferred from headings"
        )
    if n_starts != n_ends:
        raise PreflightError(f"unbalanced protocol-freeze markers: {n_starts} START vs {n_ends} END")

    extracted = "\n".join(regions)
    # A HEADING-LEVEL leak (an actual "## Order ... 执行报告" section
    # accidentally included between markers) is the real contamination
    # this guards against -- a narrative MENTION of that heading's name in
    # running prose (e.g. an audit section quoting "in `## Order S1-0
    # 执行报告` we found...") is legitimate normative content describing
    # the bug being fixed, not a leaked report, so only match actual
    # heading lines, never a bare substring search over the whole text.
    forbidden_hits = [
        s for s in _FORBIDDEN_IN_FROZEN_SPEC
        if any(line.strip().startswith("#") and s in line for line in extracted.splitlines())
    ]
    if forbidden_hits:
        raise PreflightError(
            f"extracted protocol spec contains an execution-report HEADING that must never be frozen "
            f"as normative: {forbidden_hits} -- check marker placement in guide.md"
        )
    missing_required = [s for s in _REQUIRED_IN_FROZEN_SPEC if s not in extracted]
    if missing_required:
        raise PreflightError(
            f"extracted protocol spec is missing required normative content: {missing_required} -- "
            "this usually means a marker pair was removed or misplaced, silently shrinking the frozen spec"
        )
    return extracted


def freeze_protocol_spec(output_root: str, guide_md_text: str) -> Dict[str, Any]:
    """Write (first call) or verify (later calls) ``<output_root>/
    frozen_protocol_spec.md`` -- a runs/-local AUDIT-TRAIL copy only.
    Raises ``PreflightError`` if the extracted content has changed since
    it was frozen at THIS output root. This is NOT the cross-host
    verification gate; see ``sync_committed_protocol_spec`` for the
    repo-committed copy every host (including 4090) actually checks."""
    import hashlib

    spec_text = extract_frozen_protocol_spec(guide_md_text)
    spec_hash = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()
    path = Path(output_root) / "frozen_protocol_spec.md"
    if path.exists():
        existing_text = path.read_text()
        existing_hash = hashlib.sha256(existing_text.encode("utf-8")).hexdigest()
        if existing_hash != spec_hash:
            raise PreflightError(
                f"frozen_protocol_spec.md content changed: frozen_sha256={existing_hash[:16]}... "
                f"current_sha256={spec_hash[:16]}... -- guide.md's normative S1 content claims to be frozen"
            )
        return {"path": str(path), "sha256": existing_hash, "newly_frozen": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec_text)
    return {"path": str(path), "sha256": spec_hash, "newly_frozen": True}


def verify_committed_protocol_spec_matches_live_guide(repo_root_path: str, guide_md_text: str) -> None:
    """Local-only consistency check (never run on the 4090, which has no
    ``guide_md_text`` to pass): raise ``PreflightError`` if the LIVE
    guide.md's currently-marked normative content no longer matches what
    is committed at ``COMMITTED_PROTOCOL_SPEC_REL_PATH`` -- i.e. guide.md
    changed and nobody ran the explicit amendment yet. Never
    auto-corrects; that is exclusively ``amend_committed_protocol_spec``'s
    job, invoked deliberately, never as a side effect of an ordinary
    preflight run."""
    committed_path = Path(repo_root_path) / COMMITTED_PROTOCOL_SPEC_REL_PATH
    if not committed_path.exists():
        raise PreflightError(
            f"{COMMITTED_PROTOCOL_SPEC_REL_PATH} does not exist yet -- run the explicit "
            "amend_committed_protocol_spec one-time correction before relying on --protocol-spec-path"
        )
    live_extracted = extract_frozen_protocol_spec(guide_md_text)
    committed_text = committed_path.read_text()
    if live_extracted != committed_text:
        raise PreflightError(
            f"live guide.md's marked normative content no longer matches the committed "
            f"{COMMITTED_PROTOCOL_SPEC_REL_PATH} -- guide.md changed since the last recorded amendment; "
            "run the explicit amendment procedure (never silently re-freeze)"
        )


def amend_committed_protocol_spec(repo_root_path: str, guide_md_text: str, reason: str, registry_sha256: str, output_root: str) -> Dict[str, Any]:
    """The ONLY sanctioned way to update ``COMMITTED_PROTOCOL_SPEC_REL_PATH``
    -- a deliberate, explicitly-invoked, fully-recorded correction (guide.md
    13/S1-0R-A A1 item 3: "允许做一次有记录的协议冻结修正...不得静默覆盖").
    Writes ``<output_root>/protocol_amendment.json`` with the OLD content's
    hash (empty string if none existed yet), the NEW hash, ``reason``,
    timestamp, and ``registry_sha256``, THEN overwrites the committed file.
    Never called automatically by ``preflight``."""
    import hashlib

    new_text = extract_frozen_protocol_spec(guide_md_text)
    new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    committed_path = Path(repo_root_path) / COMMITTED_PROTOCOL_SPEC_REL_PATH
    old_hash = ""
    if committed_path.exists():
        old_hash = hashlib.sha256(committed_path.read_text().encode("utf-8")).hexdigest()

    amendment = {
        "old_sha256": old_hash,
        "new_sha256": new_hash,
        "reason": reason,
        "registry_sha256": registry_sha256,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    amendment_path = out_dir / "protocol_amendment.json"
    history = []
    if amendment_path.exists():
        history = json.loads(amendment_path.read_text())
        if not isinstance(history, list):
            history = [history]
    history.append(amendment)
    amendment_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")

    committed_path.parent.mkdir(parents=True, exist_ok=True)
    committed_path.write_text(new_text)
    return {"amendment_path": str(amendment_path), "committed_path": str(committed_path), **amendment}


# --------------------------------------------------------------------- #
# Frozen S1 statistical rules. guide.md: "统计逻辑不得散落在CLI脚本里".
# --------------------------------------------------------------------- #

def _episode_identity(episode: Dict[str, Any]) -> str:
    return f"{int(episode['suite_seed'])}:{int(episode['episode_seed'])}"


def _effective_action_length(episode: Dict[str, Any]) -> int:
    actions = np.asarray(episode["robot_actions"])
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise DataRoleIntegrityError(f"robot_actions must have shape [T,2], got {actions.shape}")
    if "valid_mask" not in episode:
        return int(actions.shape[0])
    valid = np.asarray(episode["valid_mask"])
    return int(min(actions.shape[0], valid.shape[0]))


def _initial_observable_context(episode: Dict[str, Any]) -> np.ndarray:
    """Frozen S1-1 matching summary, using deployment-observable values only."""
    robot = np.asarray(episode["robot"], dtype=float)
    humans = np.asarray(episode["humans"], dtype=float)
    valid = np.asarray(episode.get("valid_mask", np.ones(humans.shape[:2], dtype=bool)), dtype=bool)
    if robot.ndim != 2 or robot.shape[1] != 9 or humans.ndim != 3 or humans.shape[2] != 5:
        raise DataRoleIntegrityError("invalid robot/human arrays while building constrained shuffle")
    active = humans[0, valid[0]]
    if active.size == 0:
        raise DataRoleIntegrityError("episode has no initially visible human")
    # Canonical robot layout is px,py,vx,vy,radius,gx,gy,v_pref,theta.
    rel_pos = active[:, :2] - robot[0, :2]
    rel_vel = active[:, 2:4] - robot[0, 2:4]
    min_distance = float(np.min(np.linalg.norm(rel_pos, axis=1)))
    mean_relative_speed = float(np.mean(np.linalg.norm(rel_vel, axis=1)))
    p = episode.get("profile_params", {})
    required = (
        "speed_lo", "speed_hi", "yield_ttc_threshold",
        "goal_switch_ttc_threshold", "assertive_probability",
    )
    missing = [name for name in required if name not in p]
    if missing:
        raise DataRoleIntegrityError(f"profile_params missing constrained-shuffle fields: {missing}")
    return np.asarray([
        min_distance,
        mean_relative_speed,
        float(p["speed_lo"]),
        float(p["speed_hi"]),
        float(p["yield_ttc_threshold"]),
        float(p["goal_switch_ttc_threshold"]),
        float(p["assertive_probability"]),
    ], dtype=float)


def _action_multiset_sha256(episodes: Sequence[Dict[str, Any]]) -> str:
    import hashlib

    members = []
    for episode in episodes:
        actions = np.ascontiguousarray(np.asarray(episode["robot_actions"], dtype=np.float64))
        members.append(hashlib.sha256(actions.tobytes()).hexdigest())
    return hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()


def build_constrained_shuffle_map(episodes: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """Build S1-1's deterministic nearest-context donor derangement.

    Recipients and donors are matched one-to-one inside exact
    ``(controller, action shape, 5-step effective-length bin)`` blocks.
    Same-episode and same-suite-seed edges are forbidden. A Hungarian
    assignment minimizes standardized initial-context distance; therefore
    every donor is used exactly once and the action multiset is preserved.
    """
    import hashlib
    from scipy.optimize import linear_sum_assignment

    if not episodes:
        raise DataRoleIntegrityError("cannot shuffle an empty episode collection")
    identities = [_episode_identity(ep) for ep in episodes]
    if len(set(identities)) != len(identities):
        raise DataRoleIntegrityError("duplicate episode identity in constrained shuffle input")
    contexts = np.vstack([_initial_observable_context(ep) for ep in episodes])
    scale = np.std(contexts, axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    standardized = (contexts - np.mean(contexts, axis=0)) / scale

    blocks: Dict[Tuple[Any, ...], List[int]] = {}
    for i, episode in enumerate(episodes):
        actions = np.asarray(episode["robot_actions"])
        length = _effective_action_length(episode)
        key = (str(episode["controller_type"]), tuple(actions.shape), int(length // 5))
        blocks.setdefault(key, []).append(i)

    mapping: List[Dict[str, Any]] = []
    forbidden_cost = 1e12
    for block_key in sorted(blocks, key=str):
        indices = sorted(blocks[block_key], key=lambda i: identities[i])
        if len(indices) < 2:
            raise DataRoleIntegrityError(f"shuffle block {block_key} has fewer than two episodes")
        cost = np.zeros((len(indices), len(indices)), dtype=float)
        for row, recipient in enumerate(indices):
            for col, donor in enumerate(indices):
                invalid = (
                    recipient == donor
                    or int(episodes[recipient]["suite_seed"]) == int(episodes[donor]["suite_seed"])
                )
                if invalid:
                    cost[row, col] = forbidden_cost
                    continue
                base = float(np.linalg.norm(standardized[recipient] - standardized[donor]))
                tie_payload = f"{seed}|{identities[recipient]}|{identities[donor]}".encode("utf-8")
                tie = int(hashlib.sha256(tie_payload).hexdigest()[:12], 16) / float(16 ** 12)
                cost[row, col] = base + tie * 1e-9
        rows, cols = linear_sum_assignment(cost)
        if len(rows) != len(indices) or np.any(cost[rows, cols] >= forbidden_cost):
            raise DataRoleIntegrityError(
                f"shuffle block {block_key} has no complete cross-suite-seed derangement"
            )
        for row, col in zip(rows.tolist(), cols.tolist()):
            recipient, donor = indices[row], indices[col]
            changed = not np.array_equal(
                np.asarray(episodes[recipient]["robot_actions"]),
                np.asarray(episodes[donor]["robot_actions"]),
            )
            mapping.append({
                "recipient_id": identities[recipient],
                "donor_id": identities[donor],
                "recipient_suite_seed": int(episodes[recipient]["suite_seed"]),
                "donor_suite_seed": int(episodes[donor]["suite_seed"]),
                "controller_type": str(episodes[recipient]["controller_type"]),
                "action_shape": list(np.asarray(episodes[recipient]["robot_actions"]).shape),
                "effective_length": _effective_action_length(episodes[recipient]),
                "context_distance": float(np.linalg.norm(standardized[recipient] - standardized[donor])),
                "actions_changed": bool(changed),
            })
    mapping.sort(key=lambda item: item["recipient_id"])
    changed_fraction = float(np.mean([item["actions_changed"] for item in mapping]))
    if changed_fraction < 0.95:
        raise DataRoleIntegrityError(
            f"constrained shuffle changed only {changed_fraction:.3%} of action sequences; required >=95%"
        )
    return {
        "schema_version": 1,
        "seed": int(seed),
        "n_episodes": len(episodes),
        "context_fields": [
            "min_initial_robot_human_distance", "mean_initial_relative_speed",
            "speed_lo", "speed_hi", "yield_ttc_threshold",
            "goal_switch_ttc_threshold", "assertive_probability",
        ],
        "action_multiset_sha256": _action_multiset_sha256(episodes),
        "changed_fraction": changed_fraction,
        "mapping": mapping,
    }


def select_best_restart_by_train_objective(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Select one eligible converged restart using train objective only."""
    eligible = [
        report for report in reports
        if bool(report.get("eligible"))
        and bool(report.get("convergence", {}).get("converged"))
        and np.isfinite(float(report.get("train_final_objective", float("nan"))))
    ]
    if not eligible:
        raise ValueError("no eligible converged restart is available")
    return max(eligible, key=lambda report: (float(report["train_final_objective"]), -int(report["restart_seed"])))


def select_k(k_summaries: Sequence[Dict[str, Any]], confidence_level: float = 0.95) -> Dict[str, Any]:
    """Apply the pre-registered smallest-K one-standard-error rule.

    Each summary must contain ``K``, ``eligible`` and ``per_seed_nll``.
    The minimum-mean-NLL model defines the one-SE threshold; the smallest
    eligible K within that threshold is selected. Boundary extension is a
    separate orchestration decision and is reported explicitly here.
    """
    del confidence_level  # Frozen rule is one standard error, not a z-CI.
    prepared = []
    for raw in k_summaries:
        values = np.asarray(raw.get("per_seed_nll", []), dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            continue
        prepared.append({
            **raw,
            "mean_nll": float(np.mean(values)),
            "se_nll": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("inf"),
        })
    eligible = [item for item in prepared if bool(item.get("eligible"))]
    if not eligible:
        return {"status": "INCONCLUSIVE", "reason": "no_eligible_k", "candidates": prepared}
    best = min(eligible, key=lambda item: (item["mean_nll"], int(item["K"])))
    threshold = best["mean_nll"] + best["se_nll"]
    selected = min(
        (item for item in eligible if item["mean_nll"] <= threshold),
        key=lambda item: int(item["K"]),
    )
    return {
        "status": "SELECTED",
        "selected_k": int(selected["K"]),
        "best_mean_k": int(best["K"]),
        "best_mean_nll": float(best["mean_nll"]),
        "best_mean_se": float(best["se_nll"]),
        "one_se_threshold": float(threshold),
        "boundary_candidate": int(best["K"]) == max(int(item["K"]) for item in eligible),
        "candidates": prepared,
    }


def block_bootstrap_suite_seed_ci(
    blocks: Sequence[Dict[str, Any]],
    resamples: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Any]:
    """Paired suite-seed block bootstrap of NLL/row improvement.

    A block contains ``delta_log_likelihood`` (candidate minus baseline)
    and ``n_rows``. Positive output means the candidate has lower NLL.
    Entire suite seeds are sampled, never individual episodes or rows.
    """
    if resamples < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap configuration")
    if len(blocks) < 2:
        return {"status": "INCONCLUSIVE", "reason": "fewer_than_two_suite_seed_blocks"}
    delta = np.asarray([float(block["delta_log_likelihood"]) for block in blocks], dtype=float)
    rows = np.asarray([float(block["n_rows"]) for block in blocks], dtype=float)
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(rows)) or np.any(rows <= 0):
        raise ValueError("bootstrap blocks contain non-finite values or non-positive row counts")
    point = float(np.sum(delta) / np.sum(rows))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=float)
    for i in range(int(resamples)):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        draws[i] = float(np.sum(delta[sampled]) / np.sum(rows[sampled]))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "status": "OK",
        "point_estimate": point,
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence_level": float(confidence_level),
        "n_suite_seed_blocks": len(blocks),
        "resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def three_state_judgment(
    selection: Dict[str, Any],
    necessity: Dict[str, Any],
    interactive_audit: Optional[Dict[str, Any]] = None,
    nominal_audit: Optional[Dict[str, Any]] = None,
    nominal_margin: float = 0.01,
) -> Dict[str, Any]:
    """Return the frozen GO/NO_GO/INCONCLUSIVE scientific verdict."""
    if selection.get("status") == "INCONCLUSIVE":
        return {"verdict": "INCONCLUSIVE", "reason": "k_selection_inconclusive"}
    if selection.get("status") not in ("SELECTED", "PASS"):
        return {"verdict": "NO_GO", "reason": "k_selection_failed"}
    if necessity.get("status") == "INCONCLUSIVE":
        return {"verdict": "INCONCLUSIVE", "reason": "necessity_inconclusive"}
    if necessity.get("status") != "PASS":
        return {"verdict": "NO_GO", "reason": "necessity_failed"}
    if interactive_audit is None or nominal_audit is None:
        return {"verdict": "INCONCLUSIVE", "reason": "audit_not_run"}
    if interactive_audit.get("status") == "INCONCLUSIVE" or nominal_audit.get("status") == "INCONCLUSIVE":
        return {"verdict": "INCONCLUSIVE", "reason": "audit_inconclusive"}
    if interactive_audit.get("status") != "PASS":
        return {"verdict": "NO_GO", "reason": "interactive_audit_failed"}
    low = float(nominal_audit.get("ci_low", float("nan")))
    high = float(nominal_audit.get("ci_high", float("nan")))
    if not np.isfinite(low) or not np.isfinite(high):
        return {"verdict": "INCONCLUSIVE", "reason": "nominal_ci_missing"}
    if low < -float(nominal_margin) or high > float(nominal_margin):
        return {"verdict": "NO_GO", "reason": "nominal_negative_control_not_equivalent"}
    return {"verdict": "GO", "reason": "all_pre_registered_gates_passed"}
