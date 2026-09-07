"""Decision-oracle primitives for auditing the frozen SM-BRNE controller.

This module is deliberately policy-free.  It creates physically executable
candidate sequences, replays each sequence from the same environment
snapshot, and computes model scores without using the replayed ground truth.
It is the D1-D3 audit layer; a deployable policy belongs in a later, gated
module and must not import oracle labels.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from crowd_nav.bayesian_brne.causal_runtime import CausalBeliefBank, sample_causal_trajectories
from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact
from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates
from crowd_nav.bayesian_brne.trajectory_sampler import sample_cv, _clip_delta_v_and_speed


@dataclass(frozen=True)
class CandidateSet:
    """A physically executable batch of robot candidates."""

    name: str
    actions: np.ndarray
    state_positions: np.ndarray
    state_velocities: np.ndarray
    future_positions: np.ndarray

    def validate(self, *, dt: float, max_speed: float, max_acceleration: float) -> None:
        arrays = (self.actions, self.state_positions, self.state_velocities, self.future_positions)
        if any(value.ndim != 3 or value.shape[-1] != 2 for value in arrays):
            raise ValueError(f"invalid candidate array ranks/shapes for {self.name}")
        shapes = {value.shape for value in arrays}
        if len(shapes) != 1:
            raise ValueError(f"candidate arrays have inconsistent shapes: {shapes}")
        if not np.all(np.isfinite(self.actions)):
            raise ValueError(f"non-finite actions in {self.name}")
        if np.max(np.linalg.norm(self.actions, axis=-1)) > max_speed + 1e-8:
            raise ValueError(f"speed cap violated in {self.name}")
        if self.actions.shape[1] > 1:
            delta = np.diff(np.concatenate([self.state_velocities[:, :1], self.actions], axis=1), axis=1)
            if np.max(np.linalg.norm(delta, axis=-1)) > max_acceleration * dt + 1e-8:
                raise ValueError(f"acceleration cap violated in {self.name}")


@dataclass(frozen=True)
class BranchResult:
    collision: bool
    min_clearance: float
    min_comfort_clearance: float
    progress: float
    first_collision_step: int


def clone_env(env):
    """Return an independent snapshot; dataclass numpy fields are copied."""
    return copy.deepcopy(env)


def _min_clearance(env) -> float:
    if not env.humans:
        return float("inf")
    return min(
        float(np.linalg.norm(human.pos - env.robot_pos) - human.radius - env.robot_radius)
        for human in env.humans
    )


def _min_comfort_clearance(env, safe_distance: float) -> float:
    return _min_clearance(env) - float(safe_distance)


def _integrate_actions(
    env,
    raw_actions: np.ndarray,
    *,
    name: str,
) -> CandidateSet:
    actions = np.asarray(raw_actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(f"raw actions must be [M,H,2], got {actions.shape}")
    state_positions = np.zeros_like(actions)
    state_velocities = np.zeros_like(actions)
    future_positions = np.zeros_like(actions)
    for m in range(actions.shape[0]):
        position = np.array([env.robot_pos[0], env.robot_pos[1]], dtype=np.float64)
        velocity = np.array([env.robot_vel[0], env.robot_vel[1]], dtype=np.float64)
        for h in range(actions.shape[1]):
            state_positions[m, h] = position
            state_velocities[m, h] = velocity
            velocity = _clip_delta_v_and_speed(
                velocity,
                actions[m, h] - velocity,
                env.dt,
                env.config.max_human_speed,
                env.config.max_human_acceleration,
            )
            actions[m, h] = velocity
            position = position + env.dt * velocity
            future_positions[m, h] = position
    result = CandidateSet(name, actions, state_positions, state_velocities, future_positions)
    result.validate(
        dt=env.dt,
        max_speed=env.config.max_human_speed,
        max_acceleration=env.config.max_human_acceleration,
    )
    return result


def _goal_direction(env) -> np.ndarray:
    vector = np.asarray(env.robot_goal, dtype=np.float64) - np.asarray(env.robot_pos, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def structured_candidates(env, *, horizon: int, legacy: CandidateSet) -> CandidateSet:
    """Add deterministic brake/hold/side-step families to legacy samples."""
    direction = _goal_direction(env)
    lateral = np.array([-direction[1], direction[0]], dtype=np.float64)
    speed = float(min(env.robot_pref_speed, env.config.max_human_speed))
    current = np.asarray(env.robot_vel, dtype=np.float64).copy()
    sequences = [legacy.actions]

    controls = []
    controls.append(np.tile(current, (horizon, 1)))
    brake = np.zeros((horizon, 2), dtype=np.float64)
    brake[0] = current
    for h in range(1, horizon):
        brake[h] = np.zeros(2)
    controls.append(brake)
    for fraction in (0.25, 0.5, 0.75):
        controls.append(np.tile(direction * speed * fraction, (horizon, 1)))
    for side in (-1.0, 1.0):
        for lateral_fraction in (0.35, 0.7, 1.0):
            target = direction * speed * 0.65 + side * lateral * speed * lateral_fraction
            controls.append(np.tile(target, (horizon, 1)))
    for side in (-1.0, 1.0):
        target = direction * speed * 0.35 + side * lateral * speed * 0.7
        sequence = np.tile(target, (horizon, 1))
        sequence[: max(1, horizon // 3)] = 0.0
        controls.append(sequence)
    sequences.append(np.stack(controls, axis=0))
    return _integrate_actions(env, np.concatenate(sequences, axis=0), name="structured_candidates")


def dense_oracle_candidates(env, *, horizon: int, legacy: CandidateSet) -> CandidateSet:
    """A fixed, denser lateral grid used only to estimate candidate coverage."""
    direction = _goal_direction(env)
    lateral = np.array([-direction[1], direction[0]], dtype=np.float64)
    speed = float(min(env.robot_pref_speed, env.config.max_human_speed))
    controls = []
    for forward in (0.0, 0.35, 0.65, 1.0):
        for side in np.linspace(-1.0, 1.0, 9):
            controls.append(np.tile(direction * speed * forward + lateral * speed * 0.85 * side, (horizon, 1)))
    return _integrate_actions(
        env, np.concatenate([legacy.actions, np.stack(controls, axis=0)], axis=0), name="dense_oracle_candidates"
    )


def make_candidate_sets(
    env, *, horizon: int, num_legacy: int, seed: int, brne_root: str | None = None,
) -> dict[str, CandidateSet]:
    legacy_arrays = sample_robot_candidates(
        robot_px=float(env.robot_pos[0]), robot_py=float(env.robot_pos[1]),
        robot_vx=float(env.robot_vel[0]), robot_vy=float(env.robot_vel[1]),
        goal_gx=float(env.robot_goal[0]), goal_gy=float(env.robot_goal[1]),
        horizon=horizon, num_candidates=num_legacy, dt=env.dt,
        rng=np.random.default_rng(seed), v_pref=float(env.robot_pref_speed),
        max_speed=float(env.config.max_human_speed),
        max_acceleration=float(env.config.max_human_acceleration),
        brne_root=brne_root,
    )
    legacy = CandidateSet("legacy_candidates", *legacy_arrays)
    legacy.validate(dt=env.dt, max_speed=env.config.max_human_speed, max_acceleration=env.config.max_human_acceleration)
    structured = structured_candidates(env, horizon=horizon, legacy=legacy)
    dense = dense_oracle_candidates(env, horizon=horizon, legacy=legacy)
    return {"legacy_candidates": legacy, "structured_candidates": structured, "dense_oracle_candidates": dense}


def rollout_candidate(env, actions: np.ndarray, *, safe_distance: float) -> BranchResult:
    branch = clone_env(env)
    minimum = _min_clearance(branch)
    comfort = _min_comfort_clearance(branch, safe_distance)
    first_collision = -1
    initial_goal_distance = float(np.linalg.norm(branch.robot_goal - branch.robot_pos))
    for step, action in enumerate(np.asarray(actions, dtype=np.float64)):
        branch.step(action)
        minimum = min(minimum, _min_clearance(branch))
        comfort = min(comfort, _min_comfort_clearance(branch, safe_distance))
        if first_collision < 0 and minimum < 0.0:
            first_collision = step
    final_goal_distance = float(np.linalg.norm(branch.robot_goal - branch.robot_pos))
    return BranchResult(
        collision=first_collision >= 0,
        min_clearance=float(minimum),
        min_comfort_clearance=float(comfort),
        progress=float(initial_goal_distance - final_goal_distance),
        first_collision_step=int(first_collision),
    )


def true_results(env, candidates: CandidateSet, *, safe_distance: float) -> list[BranchResult]:
    return [rollout_candidate(env, actions, safe_distance=safe_distance) for actions in candidates.actions]


def _prediction_key(collision_probability: float, clearance_q05: float, progress: float) -> tuple[float, float, float]:
    return (float(collision_probability), -float(clearance_q05), -float(progress))


def choose_prediction(scores: list[dict]) -> int:
    return min(range(len(scores)), key=lambda index: _prediction_key(
        scores[index]["collision_probability"], scores[index]["clearance_q05"], scores[index]["progress"]
    ))


def _score_trajectories(robot: CandidateSet, humans: np.ndarray, env, *, safe_distance: float) -> list[dict]:
    # humans: [M,P,N,H,2]
    robot_future = robot.future_positions
    displacement = robot_future[:, None, None, :, :] - humans
    distances = np.linalg.norm(displacement, axis=-1)
    radii = np.asarray([human.radius for human in env.humans], dtype=np.float64)
    boundary = float(env.robot_radius) + radii[None, :, None, None]
    clearance = distances - boundary
    minimum = np.min(clearance, axis=-1)  # [M,P,N]
    return [
        {
            "collision_probability": float(np.mean(minimum[m] < 0.0)),
            "comfort_collision_probability": float(np.mean(minimum[m] < safe_distance)),
            "clearance_q05": float(np.quantile(minimum[m], 0.05)),
            "clearance_mean": float(np.mean(minimum[m])),
            "progress": float(np.linalg.norm(env.robot_goal - env.robot_pos)
                               - np.linalg.norm(env.robot_goal - robot_future[m, -1])),
        }
        for m in range(robot_future.shape[0])
    ]


def predict_scores(
    env,
    candidates: CandidateSet,
    artifact: CausalResponseArtifact,
    beliefs: CausalBeliefBank,
    *,
    episode_seed: int,
    step: int,
    num_samples: int,
    brne_root: str,
    posterior_mean: bool = False,
    use_cv: bool = False,
    safe_distance: float = 0.2,
) -> list[dict]:
    """Score each candidate using model samples, never true branch outcomes."""
    if not env.humans:
        return [
            {
                "collision_probability": 0.0,
                "comfort_collision_probability": 0.0,
                "clearance_q05": float("inf"),
                "clearance_mean": float("inf"),
                "progress": float(np.linalg.norm(env.robot_goal - env.robot_pos)
                                   - np.linalg.norm(env.robot_goal - candidate[-1])),
            }
            for candidate in candidates.future_positions
        ]
    if use_cv:
        human_samples = []
        for human in env.humans:
            state0 = np.array([human.pos[0], human.pos[1], human.vel[0], human.vel[1]], dtype=np.float64)
            human_samples.append(sample_cv(
                state0, candidates.actions.shape[1], num_samples, env.dt, brne_root,
                np.random.default_rng([episode_seed, step, human.track_id, 991]),
            ))
        by_candidate = np.broadcast_to(
            np.stack(human_samples, axis=0)[None, :, :, :, :],
            (candidates.actions.shape[0], len(env.humans), num_samples, candidates.actions.shape[1], 2),
        )
        return _score_trajectories(candidates, by_candidate, env, safe_distance=safe_distance)

    by_candidate = []
    for candidate_index in range(candidates.actions.shape[0]):
        human_samples = []
        for human in env.humans:
            posterior = beliefs.predictive(human.track_id)
            state0 = np.array([human.pos[0], human.pos[1], human.vel[0], human.vel[1]], dtype=np.float64)
            human_samples.append(sample_causal_trajectories(
                posterior, state0, artifact,
                candidates.state_positions[candidate_index], candidates.state_velocities[candidate_index],
                candidates.actions[candidate_index], num_samples=num_samples,
                rng=np.random.default_rng([episode_seed, step, human.track_id, candidate_index, artifact.K]),
                response_enabled=False, posterior_mean=posterior_mean,
                max_speed=env.config.max_human_speed, max_acceleration=env.config.max_human_acceleration,
            ))
        by_candidate.append(np.stack(human_samples, axis=0))
    return _score_trajectories(candidates, np.stack(by_candidate, axis=0), env, safe_distance=safe_distance)


def weighted_sequence(weights: np.ndarray, candidates: CandidateSet) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (candidates.actions.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid candidate weights")
    normalized = weights / max(float(weights.sum()), 1e-12)
    return np.tensordot(normalized, candidates.actions, axes=(0, 0))


def summarize_true_results(results: Iterable[BranchResult]) -> dict:
    values = list(results)
    if not values:
        raise ValueError("empty true result list")
    return {
        "safe_candidate_exists": bool(any(not item.collision for item in values)),
        "collision_free_fraction": float(np.mean([not item.collision for item in values])),
        "best_min_clearance": float(max(item.min_clearance for item in values)),
        "best_progress": float(max(item.progress for item in values)),
    }
