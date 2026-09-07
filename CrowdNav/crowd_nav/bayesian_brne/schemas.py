"""Typed dataclasses and shape-validation helpers. No algorithm logic here.

``TrackObservation``/``PolicyObservation`` exist so that pedestrian IDENTITY
(``track_id``) is threaded explicitly from the environment/perception layer
all the way to ``BeliefBank`` -- guide.md 5.6 is explicit that ``JointState``
itself carries no IDs, so any code that reconstructs identity from "current
distance order" is the exact bug this project is not allowed to repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


class ShapeError(ValueError):
    """Raised when an array does not match its required shape/dtype."""


# Canonical, NAMED layout for the "robot [T, 9]" arrays this package passes
# between collect_dataset.py / data_io.py / mode_model.py / belief_tracker.py.
#
# This project has TWO functions in crowd_sim that both produce a 9-float
# robot array with DIFFERENT field orders:
#   FullState.to_array()   -> [px, py, vx, vy, radius, gx, gy, v_pref, theta]
#   Robot.get_obs_array()  -> [px, py, gx, gy, vx, vy, radius, v_pref, theta]
# Indices 2:4 mean VELOCITY under the first and GOAL POSITION under the
# second -- silently picking the wrong one would feed goal coordinates into
# every relative-velocity/TTC feature this package computes, corrupting
# fitting and belief-tracking without raising any error (exactly the field-
# misalignment bug class that has recurred earlier in this project).
#
# This package's internal contract is FullState.to_array()'s order --
# collect_dataset.py MUST build its ``robot`` arrays from
# ``robot.get_full_state().to_array()`` (or an explicitly-reordered
# equivalent), never from ``Robot.get_obs_array()``. Every access in this
# package goes through the named indices below, never a bare ``[2:4]``.
ROBOT_STATE_FIELDS = ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta")
ROBOT_PX, ROBOT_PY, ROBOT_VX, ROBOT_VY, ROBOT_RADIUS, ROBOT_GX, ROBOT_GY, ROBOT_VPREF, ROBOT_THETA = range(9)

# Data-file schema contract (guide.md 5.5, fixes B8's remaining PARTIAL: named
# accessors above protect internal package code from misreading a correctly-
# laid-out array, but cannot detect a collector that constructs the raw
# array from the WRONG source function in the first place, since a bare
# ndarray carries no self-describing metadata. Every episode saved by
# data_io.py/collect_dataset.py MUST carry both fields below; every episode
# loaded MUST have them checked before any array inside is trusted.
#
# Bumped 1 -> 2 for Order 5d (2026-08-03): added split/controller_type/
# profile_name/profile_params/initial_state_hash/generator_version/
# behavior_type_map as required fields. This is a genuine incompatible
# schema change (old files lack these fields entirely) -- bumping the
# version means every Step-4 smoke .npz written under version 1 now fails
# ``validate_episode_metadata`` outright instead of silently being read as
# if it were formal data (guide.md 5d.3's explicit requirement).
DATA_SCHEMA_VERSION = 2

REQUIRED_METADATA_SCALAR_FIELDS = ("split", "controller_type", "profile_name", "generator_version", "initial_state_hash")
REQUIRED_METADATA_DICT_FIELDS = ("profile_params", "behavior_type_map")


class SchemaVersionError(ValueError):
    """Raised when a loaded episode's schema_version or robot_state_layout
    does not match this package's canonical contract (B8, guide.md 5.5)."""


class CanonicalJSONError(ValueError):
    """Raised when a value cannot be canonically JSON-serialized (Order 5d:
    only basic numeric/string/bool/list/dict/None types are allowed, so a
    hash computed from the serialization is reproducible across processes
    and Python versions)."""


def _validate_canonical_json_value(value, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_canonical_json_value(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise CanonicalJSONError(f"{path}: dict keys must be str, got {type(k)}")
            _validate_canonical_json_value(v, f"{path}.{k}")
        return
    raise CanonicalJSONError(f"{path}: value of type {type(value)} is not canonically serializable")


def canonical_json_dumps(obj) -> str:
    """Deterministic JSON serialization (Order 5d.4): sorted keys, no
    whitespace ambiguity, and only basic types accepted (rejects anything
    that would make the same logical value serialize differently across
    runs/processes, e.g. numpy scalars, dict key order, sets)."""
    import json

    _validate_canonical_json_value(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate_episode_metadata(episode: dict) -> None:
    """Refuse to trust an episode dict's arrays unless it explicitly
    declares the schema version and robot field order this package's
    readers assume. Never falls back to a default -- an episode missing
    either field is exactly the "collector saved Robot.get_obs_array()
    without telling anyone" failure mode B8 flagged, and must be a hard
    error, not a silent assumption. Also enforces Order 5d's traceability
    fields (split/controller_type/profile_name/profile_params/
    initial_state_hash/generator_version/behavior_type_map)."""
    if "schema_version" not in episode:
        raise SchemaVersionError("episode missing required 'schema_version' field (guide.md 5.5/B8)")
    if int(episode["schema_version"]) != DATA_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"episode schema_version={episode['schema_version']!r} does not match "
            f"this package's expected {DATA_SCHEMA_VERSION!r}"
        )
    if "robot_state_layout" not in episode:
        raise SchemaVersionError("episode missing required 'robot_state_layout' field (guide.md 5.5/B8)")
    layout = tuple(str(f) for f in episode["robot_state_layout"])
    if layout != ROBOT_STATE_FIELDS:
        raise SchemaVersionError(
            f"episode robot_state_layout={layout!r} does not match this package's canonical "
            f"{ROBOT_STATE_FIELDS!r} -- refusing to load data whose robot field order is unverified"
        )
    for field in REQUIRED_METADATA_SCALAR_FIELDS:
        if field not in episode or episode[field] is None or episode[field] == "":
            raise SchemaVersionError(f"episode missing required Order 5d metadata field {field!r} (guide.md 5d.1)")
    for field in REQUIRED_METADATA_DICT_FIELDS:
        if field not in episode or not isinstance(episode[field], dict):
            raise SchemaVersionError(f"episode missing required Order 5d dict field {field!r} (guide.md 5d.1)")
        _validate_canonical_json_value(episode[field])


def robot_velocity(robot_row: np.ndarray) -> np.ndarray:
    """``robot_row`` is one timestep's 9-float robot array (canonical
    FullState.to_array() order). Returns ``[vx, vy]`` by NAME, never a bare
    slice, so a caller that accidentally passes a ``Robot.get_obs_array()``-
    ordered row cannot silently have this return goal coordinates instead."""
    require_shape(np.asarray(robot_row), (9,), "robot_row")
    return np.asarray(robot_row)[[ROBOT_VX, ROBOT_VY]]


def robot_position(robot_row: np.ndarray) -> np.ndarray:
    require_shape(np.asarray(robot_row), (9,), "robot_row")
    return np.asarray(robot_row)[[ROBOT_PX, ROBOT_PY]]


def robot_goal(robot_row: np.ndarray) -> np.ndarray:
    require_shape(np.asarray(robot_row), (9,), "robot_row")
    return np.asarray(robot_row)[[ROBOT_GX, ROBOT_GY]]


def build_robot_state_row(px: float, py: float, vx: float, vy: float, radius: float,
                           gx: float, gy: float, v_pref: float, theta: float) -> np.ndarray:
    """The ONLY sanctioned way to construct a canonical-order robot row from
    named scalars (guide.md B8 fix, recommendation 1). Collectors should
    call this rather than hand-assembling a bare ``np.array([...])`` in some
    remembered order -- a positional literal is exactly how a
    ``Robot.get_obs_array()``-ordered array gets constructed by accident."""
    return np.array([px, py, vx, vy, radius, gx, gy, v_pref, theta], dtype=np.float64)


# The OTHER real field order this project's code produces (Robot.get_obs_array()).
# Kept here, named, so a caller that genuinely has get-obs-ordered data can
# declare it explicitly to data_io.save_episode instead of silently guessing.
GET_OBS_ARRAY_LAYOUT = ("px", "py", "gx", "gy", "vx", "vy", "radius", "v_pref", "theta")


def reorder_robot_array_to_canonical(robot: np.ndarray, source_layout: tuple) -> np.ndarray:
    """Permute the last axis of ``robot`` (``[..., 9]``) from
    ``source_layout`` order into ``ROBOT_STATE_FIELDS`` canonical order.

    This is the actual enforcement mechanism behind B8's writer-side fix: a
    caller MUST explicitly state what order its own array is in (there is no
    default), and this function does the reordering by field NAME, never by
    position. ``source_layout`` must contain exactly the 9 canonical field
    names as a permutation -- anything else (wrong length, unknown field
    name, duplicate) is rejected outright rather than silently truncated or
    padded.
    """
    source_layout = tuple(source_layout)
    if sorted(source_layout) != sorted(ROBOT_STATE_FIELDS):
        raise SchemaVersionError(
            f"source_robot_state_layout {source_layout!r} is not a permutation of the canonical "
            f"9 robot fields {ROBOT_STATE_FIELDS!r} -- refusing to guess a mapping"
        )
    robot = np.asarray(robot)
    if robot.shape[-1] != 9:
        raise ShapeError(f"robot array's last axis must be 9, got shape {robot.shape}")
    if source_layout == ROBOT_STATE_FIELDS:
        return robot.copy()
    index_in_source = {name: i for i, name in enumerate(source_layout)}
    perm = [index_in_source[name] for name in ROBOT_STATE_FIELDS]
    return robot[..., perm]


@dataclass(frozen=True)
class TrackObservation:
    """One pedestrian's observed state at one timestep, with a stable ID."""

    track_id: int
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    timestamp: float


@dataclass(frozen=True)
class PolicyObservation:
    """Everything BayesianBRNEPolicy.predict() needs for one control step."""

    robot_px: float
    robot_py: float
    robot_vx: float
    robot_vy: float
    robot_radius: float
    robot_gx: float
    robot_gy: float
    robot_v_pref: float
    timestamp: float
    time_step: float
    humans: List[TrackObservation]
    episode_seed: int


@dataclass
class EquilibriumResult:
    """Output of BRNESolver.solve() -- see guide.md 7.1."""

    weights: np.ndarray  # [A, M]
    iterations: int
    converged: bool
    max_weight_change: float
    pair_cost_summary: dict
    elapsed_ms: float
    numeric_fallback_used: bool


def require_shape(array: np.ndarray, shape: tuple, name: str) -> None:
    """Raise ShapeError unless ``array.shape`` matches ``shape`` (``None``
    entries in ``shape`` are wildcards, e.g. ``(None, 2)`` accepts any first
    dimension)."""
    if array.ndim != len(shape):
        raise ShapeError(f"{name}: expected {len(shape)} dims, got shape {array.shape}")
    for actual, expected in zip(array.shape, shape):
        if expected is not None and actual != expected:
            raise ShapeError(f"{name}: expected shape {shape}, got {array.shape}")


def require_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ShapeError(f"{name}: contains NaN/Inf")


def require_dtype_float(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.floating):
        raise ShapeError(f"{name}: expected floating dtype, got {array.dtype}")
