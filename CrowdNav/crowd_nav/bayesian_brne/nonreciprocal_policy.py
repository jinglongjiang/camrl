"""Robot-responsible Bayesian planner for non-reciprocal FSM crowds.

Unlike BRNE's joint equilibrium update, this policy never reweights human
trajectory samples as if pedestrians would accept part of the avoidance
burden.  Human posterior samples remain fixed; only robot candidate weights
are updated from expected radius-aware collision cost.  The policy is used
only in the repaired non-reciprocal evaluation protocol.
"""

from __future__ import annotations

from typing import Optional
import time

import numpy as np

from crowd_nav.bayesian_brne.brne_adapter import weighted_first_control
from crowd_nav.bayesian_brne.causal_runtime import (
    CausalBeliefBank,
    CausalResponseArtifact,
    CausalRuntimeError,
    sample_causal_trajectories,
)
from crowd_nav.bayesian_brne.config import PlannerConfig
from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates
from crowd_nav.bayesian_brne.trajectory_sampler import _clip_delta_v_and_speed


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0.0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def robot_only_expected_cost_weights(
    robot_future_positions: np.ndarray,
    human_trajectories_by_candidate: np.ndarray,
    robot_radius: float,
    human_radii: np.ndarray,
    *,
    safe_distance: float,
    cost_sigma: float,
    cost_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized robot weights and expected costs.

    ``robot_future_positions`` is ``[M,H,2]`` and pedestrian trajectories are
    ``[P,M,N,H,2]``: pedestrian, robot candidate, posterior sample, horizon,
    xy.  Human samples are integrated with fixed uniform weights.  Only the
    robot's candidate distribution is updated, assigning all collision
    avoidance responsibility to the robot.
    """
    robot = np.asarray(robot_future_positions, dtype=np.float64)
    humans = np.asarray(human_trajectories_by_candidate, dtype=np.float64)
    radii = np.asarray(human_radii, dtype=np.float64)
    if robot.ndim != 3 or robot.shape[-1] != 2:
        raise ValueError(f"robot trajectories must be [M,H,2], got {robot.shape}")
    if humans.ndim != 5 or humans.shape[-1] != 2:
        raise ValueError(f"human trajectories must be [P,M,N,H,2], got {humans.shape}")
    pedestrians, candidates, samples, horizon, _ = humans.shape
    if robot.shape != (candidates, horizon, 2):
        raise ValueError(f"robot/human candidate-horizon mismatch: {robot.shape} vs {humans.shape}")
    if radii.shape != (pedestrians,):
        raise ValueError(f"human_radii must have shape ({pedestrians},), got {radii.shape}")
    if pedestrians == 0 or samples == 0:
        return np.full(candidates, 1.0 / candidates), np.zeros(candidates)
    if not np.all(np.isfinite(robot)) or not np.all(np.isfinite(humans)):
        raise ValueError("trajectories must be finite")
    if robot_radius <= 0.0 or np.any(radii <= 0.0):
        raise ValueError("all radii must be positive")
    if safe_distance < 0.0 or cost_sigma <= 0.0 or cost_scale <= 0.0:
        raise ValueError("invalid clearance-cost parameters")

    displacement = (
        robot[None, :, None, :, :] - humans
    )  # [P,M,N,H,2]
    distance = np.linalg.norm(displacement, axis=-1)
    boundary = robot_radius + radii[:, None, None, None] + safe_distance
    clearance = distance - boundary
    per_step = cost_scale * _stable_sigmoid(-clearance / cost_sigma)
    trajectory_cost = np.max(per_step, axis=-1)  # [P,M,N]
    expected_cost = trajectory_cost.mean(axis=(0, 2))  # [M]
    log_weight = -expected_cost
    log_weight -= float(np.max(log_weight))
    weights = np.exp(log_weight)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("robot-only expected-risk weights are non-finite")
    weights /= total
    return weights, expected_cost


class NonReciprocalBayesianPolicy:
    """Frozen-artifact policy for the non-reciprocal evaluation protocol."""

    def __init__(self, *, posterior_mean: bool = False):
        self.posterior_mean = posterior_mean
        self.config: Optional[PlannerConfig] = None
        self.artifact: Optional[CausalResponseArtifact] = None
        self.beliefs: Optional[CausalBeliefBank] = None
        self.episode_seed: Optional[int] = None
        self.step_index = 0
        self.last_action = np.zeros(2)
        self.last_timestamp: Optional[float] = None
        self.last_diagnostics: dict = {}

    def configure(self, config: PlannerConfig, artifact: CausalResponseArtifact) -> None:
        artifact.validate()
        if config.solver_mode != "stable_clearance":
            raise CausalRuntimeError(
                "non-reciprocal policy requires radius-aware stable_clearance parameters"
            )
        self.config = config
        self.artifact = artifact
        self.beliefs = CausalBeliefBank(artifact, config.max_missed_steps)

    def reset(self, episode_seed: int) -> None:
        if self.beliefs is None:
            raise CausalRuntimeError("configure before reset")
        self.beliefs.reset()
        self.episode_seed = int(episode_seed)
        self.step_index = 0
        self.last_action = np.zeros(2)
        self.last_timestamp = None
        self.last_diagnostics = {}

    def predict(self, observation):
        from crowd_sim.envs.utils.action import ActionXY

        started = time.perf_counter()

        if self.artifact is None or self.config is None or self.beliefs is None:
            raise CausalRuntimeError("policy is not configured")
        if self.episode_seed is None:
            raise CausalRuntimeError("policy is not reset")
        if abs(observation.time_step - self.artifact.dt) > 1e-9:
            raise CausalRuntimeError("environment/artifact dt mismatch")
        if self.last_timestamp is not None and abs(
            observation.timestamp - self.last_timestamp - observation.time_step
        ) > 1e-8:
            raise CausalRuntimeError("non-consecutive policy timestamp")

        self.beliefs.update(observation, self.last_action)
        rng = np.random.default_rng([self.episode_seed, self.step_index])
        actions, state_positions, state_velocities, future_positions = sample_robot_candidates(
            robot_px=observation.robot_px,
            robot_py=observation.robot_py,
            robot_vx=observation.robot_vx,
            robot_vy=observation.robot_vy,
            goal_gx=observation.robot_gx,
            goal_gy=observation.robot_gy,
            horizon=self.config.horizon_steps,
            num_candidates=self.config.num_samples,
            dt=observation.time_step,
            rng=rng,
            v_pref=observation.robot_v_pref,
            max_speed=self.config.max_speed,
            max_acceleration=self.config.max_acceleration,
            brne_root=self.config.brne_root,
        )

        humans = sorted(observation.humans, key=lambda human: human.track_id)
        if humans:
            weights = np.full(
                self.config.num_samples, 1.0 / self.config.num_samples,
            )
            residual_history = []
            expected_cost = np.zeros(self.config.num_samples)
            for _iteration in range(self.config.max_outer_iterations):
                normalized = weights / max(float(weights.sum()), 1e-12)
                weighted_positions = np.tensordot(
                    normalized, state_positions, axes=(0, 0),
                )
                weighted_velocities = np.tensordot(
                    normalized, state_velocities, axes=(0, 0),
                )
                weighted_actions = np.tensordot(
                    normalized, actions, axes=(0, 0),
                )
                sampled_by_human = []
                for human in humans:
                    state0 = np.array([human.px, human.py, human.vx, human.vy])
                    sampled_by_human.append(sample_causal_trajectories(
                        self.beliefs.predictive(human.track_id),
                        state0,
                        self.artifact,
                        weighted_positions,
                        weighted_velocities,
                        weighted_actions,
                        num_samples=self.config.num_samples,
                        rng=np.random.default_rng([
                            self.episode_seed, self.step_index, human.track_id,
                        ]),
                        response_enabled=False,
                        posterior_mean=self.posterior_mean,
                        max_speed=self.config.max_speed,
                        max_acceleration=self.config.max_acceleration,
                    ))
                common_samples = np.stack(sampled_by_human, axis=0)  # [P,N,H,2]
                candidate_view = np.broadcast_to(
                    common_samples[:, None, :, :, :],
                    (
                        len(humans), self.config.num_samples,
                        self.config.num_samples, self.config.horizon_steps, 2,
                    ),
                )
                proposed, expected_cost = robot_only_expected_cost_weights(
                    future_positions,
                    candidate_view,
                    observation.robot_radius,
                    np.asarray([human.radius for human in humans]),
                    safe_distance=self.config.safe_distance,
                    cost_sigma=self.config.cost_sigma,
                    cost_scale=self.config.cost_scale,
                )
                proposed = (
                    (1.0 - self.config.outer_damping) * weights
                    + self.config.outer_damping * proposed
                )
                proposed /= max(float(proposed.sum()), 1e-12)
                residual = float(np.max(np.abs(proposed - weights)))
                residual_history.append(residual)
                weights = proposed
                if residual <= self.config.outer_tolerance:
                    break
        else:
            weights = np.zeros(self.config.num_samples)
            weights[0] = 1.0
            expected_cost = np.zeros(self.config.num_samples)
            residual_history = []

        control = weighted_first_control(weights, actions)
        current_velocity = np.array([observation.robot_vx, observation.robot_vy])
        control = _clip_delta_v_and_speed(
            current_velocity,
            control - current_velocity,
            observation.time_step,
            self.config.max_speed,
            self.config.max_acceleration,
        )
        self.last_action = control.copy()
        self.last_timestamp = observation.timestamp
        self.step_index += 1
        self.last_diagnostics = {
            "responsibility_mode": "robot_only",
            "posterior_mean": self.posterior_mean,
            "robot_weights": weights.copy(),
            "expected_cost": expected_cost.copy(),
            "effective_sample_size": float(1.0 / np.sum(weights * weights)),
            "outer_iterations": len(residual_history),
            "outer_converged": bool(
                not residual_history
                or residual_history[-1] <= self.config.outer_tolerance
            ),
            "max_weight_residual": (
                residual_history[-1] if residual_history else 0.0
            ),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        return ActionXY(float(control[0]), float(control[1]))
