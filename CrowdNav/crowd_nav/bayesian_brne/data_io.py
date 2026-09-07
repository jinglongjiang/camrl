"""Schema-versioned NPZ read/save for SM-BRNE episode data (guide.md 5.5).

Single responsibility: load/save the episode schema (schema_version,
suite_seed, episode_seed, scenario, profile, dt, robot/humans/
human_track_ids/robot_actions/human_actions/events/latent_behavior_labels/
valid_mask). No collection logic (collect_dataset.py) and no fitting logic
(mode_model.py) belongs here.

Every episode written here is stamped with ``schema_version`` and the
CANONICAL ``robot_state_layout`` (schemas.DATA_SCHEMA_VERSION /
schemas.ROBOT_STATE_FIELDS); every episode read here validates both before
returning any array to a caller.

C1 fix (2026-08-03 second Step 4 audit): an earlier version of
``save_episode`` stamped ``schema_version``/``robot_state_layout``
UNCONDITIONALLY, regardless of what order the caller's ``robot`` array was
actually in -- so a caller that (by bug) built its array in
``Robot.get_obs_array()`` order still got it saved, loaded, and trusted as
canonical. The writer must never self-certify data it did not itself
produce. ``save_episode`` now REQUIRES an explicit
``source_robot_state_layout`` argument (no default) describing the actual
column order of ``episode['robot']`` as the caller built it, and reorders
via ``schemas.reorder_robot_array_to_canonical`` before saving -- an
unknown/missing/malformed layout is rejected outright rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from crowd_nav.bayesian_brne.schemas import (
    DATA_SCHEMA_VERSION,
    REQUIRED_METADATA_DICT_FIELDS,
    REQUIRED_METADATA_SCALAR_FIELDS,
    ROBOT_STATE_FIELDS,
    canonical_json_dumps,
    reorder_robot_array_to_canonical,
    validate_episode_metadata,
)

STEP = 4
IMPLEMENTED = True

REQUIRED_ARRAY_FIELDS = (
    "robot", "humans", "human_track_ids", "robot_actions",
    "human_actions", "latent_behavior_labels", "valid_mask",
)
REQUIRED_SCALAR_FIELDS = ("suite_seed", "episode_seed", "scenario", "profile", "dt") + REQUIRED_METADATA_SCALAR_FIELDS
REQUIRED_DICT_FIELDS = REQUIRED_METADATA_DICT_FIELDS


class EpisodeValidationError(ValueError):
    """Raised when an episode dict is missing a required field or an array
    has an inconsistent shape, independent of the schema_version/layout
    checks in schemas.validate_episode_metadata."""


def _check_episode_shapes(episode: Dict[str, Any]) -> None:
    for field in REQUIRED_ARRAY_FIELDS:
        if field not in episode:
            raise EpisodeValidationError(f"episode missing required array field {field!r}")
    for field in REQUIRED_SCALAR_FIELDS:
        if field not in episode:
            raise EpisodeValidationError(f"episode missing required scalar field {field!r}")
    for field in REQUIRED_DICT_FIELDS:
        if field not in episode or not isinstance(episode[field], dict):
            raise EpisodeValidationError(f"episode missing required dict field {field!r}")

    robot = np.asarray(episode["robot"])
    humans = np.asarray(episode["humans"])
    track_ids = np.asarray(episode["human_track_ids"])
    valid_mask = np.asarray(episode["valid_mask"])
    robot_actions = np.asarray(episode["robot_actions"])
    human_actions = np.asarray(episode["human_actions"])
    latent_labels = np.asarray(episode["latent_behavior_labels"])

    if robot.ndim != 2 or robot.shape[1] != 9:
        raise EpisodeValidationError(f"robot must be [T, 9], got {robot.shape}")
    T = robot.shape[0]
    if humans.ndim != 3 or humans.shape[0] != T or humans.shape[2] != 5:
        raise EpisodeValidationError(f"humans must be [T={T}, N, 5], got {humans.shape}")
    N = humans.shape[1]
    if track_ids.shape != (T, N):
        raise EpisodeValidationError(f"human_track_ids must be [T={T}, N={N}], got {track_ids.shape}")
    if valid_mask.shape != (T, N):
        raise EpisodeValidationError(f"valid_mask must be [T={T}, N={N}], got {valid_mask.shape}")
    if robot_actions.ndim != 2 or robot_actions.shape[0] != T or robot_actions.shape[1] != 2:
        raise EpisodeValidationError(f"robot_actions must be [T={T}, 2], got {robot_actions.shape}")
    if human_actions.shape != (T, N, 2):
        raise EpisodeValidationError(f"human_actions must be [T={T}, N={N}, 2], got {human_actions.shape}")
    if latent_labels.shape != (T, N):
        raise EpisodeValidationError(f"latent_behavior_labels must be [T={T}, N={N}], got {latent_labels.shape}")


def save_episode(path: str, episode: Dict[str, Any], source_robot_state_layout: Sequence[str]) -> None:
    """Reorder ``episode['robot']`` from ``source_robot_state_layout`` into
    canonical order, stamp schema_version/robot_state_layout, and write to
    ``path`` (a ``.npz`` file).

    ``source_robot_state_layout`` is REQUIRED and has no default -- it must
    be the actual column order the caller built ``episode['robot']`` in
    (e.g. ``schemas.ROBOT_STATE_FIELDS`` if built via
    ``schemas.build_robot_state_row``, or ``schemas.GET_OBS_ARRAY_LAYOUT`` if
    built from ``Robot.get_obs_array()``). This is the actual enforcement
    behind B8: a writer that accepted no such argument could not tell a
    correctly-built canonical array from a misordered one, and would end up
    self-certifying data it never checked (the exact bug the 2026-08-03
    audit demonstrated with a live reproduction). ``episode`` must also
    contain every field in REQUIRED_ARRAY_FIELDS/REQUIRED_SCALAR_FIELDS plus
    ``events`` (a list[dict], JSON-serialized here since npz has no native
    list-of-dict support)."""
    _check_episode_shapes(episode)
    events = episode.get("events", [])
    canonical_robot = reorder_robot_array_to_canonical(np.asarray(episode["robot"], dtype=np.float64), source_robot_state_layout)

    payload = {
        "schema_version": np.array(DATA_SCHEMA_VERSION),
        "robot_state_layout": np.array(ROBOT_STATE_FIELDS),
        "suite_seed": np.array(episode["suite_seed"]),
        "episode_seed": np.array(episode["episode_seed"]),
        "scenario": np.array(episode["scenario"]),
        "profile": np.array(episode["profile"]),
        "dt": np.array(episode["dt"], dtype=np.float64),
        "split": np.array(episode["split"]),
        "controller_type": np.array(episode["controller_type"]),
        "profile_name": np.array(episode["profile_name"]),
        "generator_version": np.array(episode["generator_version"]),
        "initial_state_hash": np.array(episode["initial_state_hash"]),
        "profile_params_json": np.array(canonical_json_dumps(episode["profile_params"])),
        "behavior_type_map_json": np.array(canonical_json_dumps(episode["behavior_type_map"])),
        "robot": canonical_robot,
        "humans": np.asarray(episode["humans"], dtype=np.float64),
        "human_track_ids": np.asarray(episode["human_track_ids"]),
        "robot_actions": np.asarray(episode["robot_actions"], dtype=np.float64),
        "human_actions": np.asarray(episode["human_actions"], dtype=np.float64),
        "latent_behavior_labels": np.asarray(episode["latent_behavior_labels"]),
        "valid_mask": np.asarray(episode["valid_mask"], dtype=bool),
        "events_json": np.array(json.dumps(events)),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


def load_episode(path: str) -> Dict[str, Any]:
    """Load ``path`` and validate schema_version/robot_state_layout before
    returning. Raises SchemaVersionError (schemas.py) if the file's stamped
    metadata does not match this package's current contract -- callers must
    never bypass this by reading the npz directly."""
    with np.load(path, allow_pickle=False) as npz:
        episode: Dict[str, Any] = {
            "schema_version": int(npz["schema_version"]),
            "robot_state_layout": tuple(str(f) for f in npz["robot_state_layout"]),
            "suite_seed": int(npz["suite_seed"]),
            "episode_seed": int(npz["episode_seed"]),
            "scenario": str(npz["scenario"]),
            "profile": str(npz["profile"]),
            "dt": float(npz["dt"]),
            "split": str(npz["split"]),
            "controller_type": str(npz["controller_type"]),
            "profile_name": str(npz["profile_name"]),
            "generator_version": str(npz["generator_version"]),
            "initial_state_hash": str(npz["initial_state_hash"]),
            "profile_params": json.loads(str(npz["profile_params_json"])),
            "behavior_type_map": json.loads(str(npz["behavior_type_map_json"])),
            "robot": npz["robot"],
            "humans": npz["humans"],
            "human_track_ids": npz["human_track_ids"],
            "robot_actions": npz["robot_actions"],
            "human_actions": npz["human_actions"],
            "latent_behavior_labels": npz["latent_behavior_labels"],
            "valid_mask": npz["valid_mask"],
            "events": json.loads(str(npz["events_json"])),
        }
    validate_episode_metadata(episode)
    _check_episode_shapes(episode)
    return episode


def load_episodes(paths: List[str]) -> List[Dict[str, Any]]:
    return [load_episode(p) for p in paths]


def compute_initial_state_hash(
    robot_row0: np.ndarray,
    humans_row0: np.ndarray,
    behavior_type_map: Dict[str, str],
    controller_type: str,
    profile_name: str,
    profile_params: Dict[str, Any],
) -> str:
    """SHA256 of a canonical JSON snapshot of everything that identifies
    this episode's GENERATION (Order 5d's ``initial_state_hash``).

    Deliberately includes ``controller_type``/``profile_name``/
    ``profile_params`` alongside the raw t=0 physical state: guide.md 5d's
    acceptance criterion is "改变初始状态/controller/profile时 hash必须改
    变" -- the raw (px,py,vx,vy,...) at t=0 is IDENTICAL across different
    controllers for the same seed (the controller only acts from t=0
    onward), so a hash of physical state alone would not satisfy that
    requirement. Floats are rounded to 8 decimals before hashing so the
    hash is stable across platforms/reruns of the exact same generation,
    without being sensitive to insignificant floating-point noise.
    """
    import hashlib

    payload = {
        "robot": [round(float(x), 8) for x in np.asarray(robot_row0).reshape(-1).tolist()],
        "humans": [[round(float(x), 8) for x in row] for row in np.asarray(humans_row0).tolist()],
        "behavior_type_map": behavior_type_map,
        "controller_type": controller_type,
        "profile_name": profile_name,
        "profile_params": profile_params,
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()
