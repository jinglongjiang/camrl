"""Paired robot-action interventions for causal SM-BRNE.

The canonical environment follows a natural controller. At every canonical
state two deep-copied snapshots receive randomized, physically feasible robot
velocity actions. Only their one-step pedestrian responses are recorded; the
branches never alter the canonical trajectory.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
    CONTEXT_FEATURE_NAMES,
    compute_context_features,
)
from crowd_nav.bayesian_brne.interaction_protocol import (
    CONTROLLER_TYPES,
    RobotControllerState,
    compute_robot_action,
    make_scenario,
)


PAIR_SCHEMA_VERSION = 2
GENERATOR_VERSION = "causal_triplet_v2"
ACTION_DIRECTIONS = 16
MIN_ACTION_SEPARATION = 0.20


class CausalPairDataError(ValueError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_hash(env) -> str:
    payload = [env.robot_state_row().astype(np.float64).tobytes()]
    for human in env.humans:
        payload.append(np.asarray([
            human.track_id, human.pos[0], human.pos[1], human.vel[0], human.vel[1],
            human.radius, human.pref_speed,
        ], dtype=np.float64).tobytes())
        payload.append(human.behavior_state.behavior_type.value.encode("ascii"))
    return _sha256_bytes(b"".join(payload))


def _clip_velocity(value: np.ndarray, max_speed: float) -> np.ndarray:
    speed = float(np.linalg.norm(value))
    if speed > max_speed:
        return value * (max_speed / speed)
    return value


def intervention_action_pair(
    robot_velocity: np.ndarray,
    *,
    dt: float,
    max_speed: float,
    max_acceleration: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Draw two actions from a fixed acceleration lattice without replacement."""
    angles = 2.0 * np.pi * np.arange(ACTION_DIRECTIONS) / ACTION_DIRECTIONS
    deltas = max_acceleration * dt * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    candidates = np.stack([_clip_velocity(robot_velocity + delta, max_speed) for delta in deltas])
    order = rng.permutation(ACTION_DIRECTIONS)
    for i in order:
        for j in order:
            if i == j:
                continue
            if float(np.linalg.norm(candidates[i] - candidates[j])) >= MIN_ACTION_SEPARATION:
                return candidates[i].copy(), candidates[j].copy(), (int(i), int(j))
    raise CausalPairDataError("could not construct two sufficiently separated intervention actions")


def collect_pair_episode(
    *,
    suite_seed: int,
    episode_index: int,
    split: str,
    scenario: str,
    controller_type: str,
    horizon_steps: int,
    dt: float,
    brne_root: str,
) -> Dict[str, object]:
    episode_seed = int(suite_seed) * 100000 + int(episode_index)
    env = make_scenario(scenario, split, np.random.default_rng(episode_seed), dt)
    controller = RobotControllerState(
        controller_type=controller_type,
        brne_root=brne_root,
        brne_rng=np.random.default_rng(episode_seed + 17),
    )
    n_humans = len(env.humans)
    shape = (horizon_steps, n_humans)
    v_current = np.zeros(shape + (2,), dtype=np.float64)
    context = np.zeros(shape + (len(CONTEXT_FEATURE_NAMES),), dtype=np.float64)
    robot_velocity = np.zeros((horizon_steps, 2), dtype=np.float64)
    action_ref = np.zeros((horizon_steps, 2), dtype=np.float64)
    action_a = np.zeros((horizon_steps, 2), dtype=np.float64)
    action_b = np.zeros((horizon_steps, 2), dtype=np.float64)
    action_indices = np.zeros((horizon_steps, 2), dtype=np.int64)
    next_velocity_ref = np.zeros(shape + (2,), dtype=np.float64)
    next_velocity_a = np.zeros(shape + (2,), dtype=np.float64)
    next_velocity_b = np.zeros(shape + (2,), dtype=np.float64)
    track_ids = np.zeros(shape, dtype=np.int64)
    snapshot_hashes: List[str] = []
    fallback_events: List[dict] = []

    for t in range(horizon_steps):
        snapshot_hashes.append(_state_hash(env))
        robot_velocity[t] = env.robot_vel
        for n, human in enumerate(env.humans):
            v_current[t, n] = human.vel
            context[t, n] = compute_context_features(
                human.pos, human.vel, env.robot_pos, env.robot_vel,
            )
            track_ids[t, n] = human.track_id

        intervention_rng = np.random.default_rng([suite_seed, episode_index, t, 0xCA55])
        u_a, u_b, indices = intervention_action_pair(
            env.robot_vel,
            dt=env.dt,
            max_speed=env.config.max_human_speed,
            max_acceleration=env.config.max_human_acceleration,
            rng=intervention_rng,
        )
        # u_ref means zero robot acceleration. It anchors the action-free
        # base dynamics, while the two randomized branches identify only
        # the causal response around that anchor.
        u_ref = env.robot_vel.copy()
        action_ref[t], action_a[t], action_b[t], action_indices[t] = u_ref, u_a, u_b, indices
        branch_ref, branch_a, branch_b = copy.deepcopy(env), copy.deepcopy(env), copy.deepcopy(env)
        branch_ref.step(u_ref)
        branch_a.step(u_a)
        branch_b.step(u_b)
        next_velocity_ref[t] = np.stack([human.vel for human in branch_ref.humans])
        next_velocity_a[t] = np.stack([human.vel for human in branch_a.humans])
        next_velocity_b[t] = np.stack([human.vel for human in branch_b.humans])

        natural_action, fallback = compute_robot_action(env, controller)
        if fallback is not None:
            fallback_events.append({"t": t, **fallback})
        env.step(natural_action)

    behavior_map = {str(h.track_id): h.behavior_state.behavior_type.value for h in env.humans}
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "suite_seed": int(suite_seed),
        "episode_seed": episode_seed,
        "episode_index": int(episode_index),
        "split": split,
        "scenario": scenario,
        "controller_type": controller_type,
        "dt": float(dt),
        "behavior_type_map": behavior_map,
        "snapshot_hashes": snapshot_hashes,
        "fallback_events": fallback_events,
        "v_current": v_current,
        "context": context,
        "robot_velocity": robot_velocity,
        "action_ref": action_ref,
        "action_a": action_a,
        "action_b": action_b,
        "action_indices": action_indices,
        "next_velocity_ref": next_velocity_ref,
        "next_velocity_a": next_velocity_a,
        "next_velocity_b": next_velocity_b,
        "track_ids": track_ids,
    }


def validate_pair_episode(episode: Dict[str, object], *, nominal: bool = False) -> None:
    if int(episode.get("schema_version", -1)) != PAIR_SCHEMA_VERSION:
        raise CausalPairDataError("causal-pair schema version mismatch")
    arrays = {name: np.asarray(episode[name]) for name in (
        "v_current", "context", "robot_velocity", "action_ref", "action_a", "action_b",
        "action_indices", "next_velocity_ref", "next_velocity_a", "next_velocity_b", "track_ids",
    )}
    t, n, d = arrays["v_current"].shape
    if d != 2 or arrays["context"].shape != (t, n, len(CONTEXT_FEATURE_NAMES)):
        raise CausalPairDataError("invalid state/context shape")
    for name in ("next_velocity_ref", "next_velocity_a", "next_velocity_b"):
        if arrays[name].shape != (t, n, 2):
            raise CausalPairDataError(f"invalid {name} shape")
    for name in ("robot_velocity", "action_ref", "action_a", "action_b", "action_indices"):
        if arrays[name].shape != (t, 2):
            raise CausalPairDataError(f"invalid {name} shape")
    if arrays["track_ids"].shape != (t, n):
        raise CausalPairDataError("invalid track_ids shape")
    if not all(np.all(np.isfinite(value)) for key, value in arrays.items() if key != "action_indices"):
        raise CausalPairDataError("causal pair contains NaN/Inf")
    separation = np.linalg.norm(arrays["action_a"] - arrays["action_b"], axis=1)
    if np.any(separation < MIN_ACTION_SEPARATION - 1e-9):
        raise CausalPairDataError("intervention actions are not sufficiently separated")
    if not np.array_equal(arrays["action_ref"], arrays["robot_velocity"]):
        raise CausalPairDataError("reference action is not the zero-acceleration anchor")
    if len(set(episode["snapshot_hashes"])) != t:
        raise CausalPairDataError("snapshot hashes are missing or repeated inside an episode")
    if episode.get("fallback_events"):
        raise CausalPairDataError("natural controller fallback occurred")
    if nominal:
        values = np.stack([
            arrays["next_velocity_ref"], arrays["next_velocity_a"], arrays["next_velocity_b"]
        ])
        if not np.array_equal(values[0], values[1]) or not np.array_equal(values[0], values[2]):
            difference = float(np.max(np.abs(values - values[0:1])))
            raise CausalPairDataError(
                f"nominal causal branch changed pedestrian future (max_abs={difference})"
            )


def save_pair_episode(path: Path, episode: Dict[str, object]) -> None:
    validate_pair_episode(episode, nominal=episode["split"] == "test_nominal")
    metadata = {
        key: episode[key] for key in (
            "schema_version", "generator_version", "suite_seed", "episode_seed",
            "episode_index", "split", "scenario", "controller_type", "dt",
            "behavior_type_map", "snapshot_hashes", "fallback_events",
        )
    }
    payload = {"metadata_json": np.array(json.dumps(metadata, sort_keys=True))}
    for name in (
        "v_current", "context", "robot_velocity", "action_ref", "action_a", "action_b",
        "action_indices", "next_velocity_ref", "next_velocity_a", "next_velocity_b", "track_ids",
    ):
        payload[name] = np.asarray(episode[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_pair_episode(path: Path, *, nominal: bool = False) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        episode = json.loads(str(data["metadata_json"]))
        for name in (
            "v_current", "context", "robot_velocity", "action_ref", "action_a", "action_b",
            "action_indices", "next_velocity_ref", "next_velocity_a", "next_velocity_b", "track_ids",
        ):
            episode[name] = data[name]
    validate_pair_episode(episode, nominal=nominal)
    return episode


def collect_role(
    *,
    output_dir: Path,
    suite_seeds: Sequence[int],
    episodes_per_seed: int,
    split: str,
    scenario: str,
    horizon_steps: int,
    dt: float,
    brne_root: str,
) -> Dict[str, object]:
    counts = {controller: 0 for controller in CONTROLLER_TYPES}
    hashes = set()
    for suite_seed in suite_seeds:
        for episode_index in range(episodes_per_seed):
            controller = CONTROLLER_TYPES[episode_index % len(CONTROLLER_TYPES)]
            path = output_dir / f"suite_{suite_seed}" / f"episode_{episode_index:04d}.npz"
            if path.exists():
                episode = load_pair_episode(path, nominal=split == "test_nominal")
            else:
                episode = collect_pair_episode(
                    suite_seed=suite_seed,
                    episode_index=episode_index,
                    split=split,
                    scenario=scenario,
                    controller_type=controller,
                    horizon_steps=horizon_steps,
                    dt=dt,
                    brne_root=brne_root,
                )
                save_pair_episode(path, episode)
            if int(episode["suite_seed"]) != suite_seed or int(episode["episode_index"]) != episode_index:
                raise CausalPairDataError(f"resume identity mismatch at {path}")
            if episode["controller_type"] != controller or episode["split"] != split or episode["scenario"] != scenario:
                raise CausalPairDataError(f"resume protocol mismatch at {path}")
            counts[controller] += 1
            for value in episode["snapshot_hashes"]:
                if value in hashes:
                    raise CausalPairDataError(f"duplicate snapshot hash {value}")
                hashes.add(value)
    expected = len(suite_seeds) * episodes_per_seed // len(CONTROLLER_TYPES)
    if any(value != expected for value in counts.values()):
        raise CausalPairDataError(f"controller imbalance: {counts}, expected={expected}")
    return {
        "status": "PASS",
        "n_episodes": len(suite_seeds) * episodes_per_seed,
        "controller_counts": counts,
        "n_snapshots": len(hashes),
    }


def load_role(root: Path, suite_seeds: Iterable[int], *, nominal: bool = False) -> List[Dict[str, object]]:
    episodes = []
    expected = set(int(seed) for seed in suite_seeds)
    for path in sorted(root.rglob("*.npz")):
        episode = load_pair_episode(path, nominal=nominal)
        if int(episode["suite_seed"]) in expected:
            episodes.append(episode)
    return episodes
