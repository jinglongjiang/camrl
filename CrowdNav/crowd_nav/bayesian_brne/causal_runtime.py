"""Online filtering, rollout, and BRNE policy for causal-response artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import compute_context_features
from crowd_nav.bayesian_brne.brne_adapter import BRNESolver, weighted_first_control
from crowd_nav.bayesian_brne.causal_response_arhmm import CausalResponseArtifact, _log_gaussian
from crowd_nav.bayesian_brne.config import PlannerConfig
from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates
from crowd_nav.bayesian_brne.schemas import PolicyObservation
from crowd_nav.bayesian_brne.trajectory_sampler import _clip_delta_v_and_speed


class CausalRuntimeError(ValueError):
    pass


@dataclass
class _TrackState:
    predictive: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    robot_position: np.ndarray
    robot_velocity: np.ndarray
    timestamp: float
    missed: int = 0


class CausalBeliefBank:
    """Exact finite-state filter using the artifact's causal emission."""

    def __init__(self, artifact: CausalResponseArtifact, max_missed_steps: int = 5):
        artifact.validate()
        self.artifact = artifact
        self.max_missed_steps = max_missed_steps
        self.tracks: Dict[int, _TrackState] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, observation: PolicyObservation, last_action: np.ndarray) -> None:
        seen = set()
        for human in observation.humans:
            seen.add(human.track_id)
            position = np.array([human.px, human.py], dtype=np.float64)
            velocity = np.array([human.vx, human.vy], dtype=np.float64)
            current = self.tracks.get(human.track_id)
            if current is None:
                predictive = self.artifact.initial_distribution.copy()
            else:
                gap = (observation.timestamp - current.timestamp) / self.artifact.dt
                if abs(gap - 1.0) > 1e-6:
                    raise CausalRuntimeError(
                        f"track {human.track_id} timestamp gap is {gap}, expected one step"
                    )
                context = compute_context_features(
                    current.position, current.velocity, current.robot_position, current.robot_velocity,
                )
                log_likelihood = np.array([
                    _log_gaussian(
                        velocity[None, :],
                        self.artifact.mean(
                            k, current.velocity, context, current.robot_velocity, last_action,
                        )[None, :],
                        self.artifact.Q[k],
                    )[0]
                    for k in range(self.artifact.K)
                ])
                log_posterior = np.log(np.maximum(current.predictive, 1e-300)) + log_likelihood
                posterior = np.exp(log_posterior - np.max(log_posterior))
                posterior /= posterior.sum()
                predictive = posterior @ self.artifact.Pi
            self.tracks[human.track_id] = _TrackState(
                predictive=predictive,
                position=position,
                velocity=velocity,
                robot_position=np.array([observation.robot_px, observation.robot_py]),
                robot_velocity=np.array([observation.robot_vx, observation.robot_vy]),
                timestamp=observation.timestamp,
            )
        for track_id in list(self.tracks):
            if track_id in seen:
                continue
            state = self.tracks[track_id]
            state.missed += 1
            state.predictive = state.predictive @ self.artifact.Pi
            if state.missed > self.max_missed_steps:
                del self.tracks[track_id]

    def predictive(self, track_id: int) -> np.ndarray:
        if track_id not in self.tracks:
            raise CausalRuntimeError(f"unknown track {track_id}")
        return self.tracks[track_id].predictive.copy()


def sample_causal_trajectories(
    posterior: np.ndarray,
    state0: np.ndarray,
    artifact: CausalResponseArtifact,
    robot_state_positions: np.ndarray,
    robot_state_velocities: np.ndarray,
    robot_actions: np.ndarray,
    *,
    num_samples: int,
    rng: np.random.Generator,
    response_enabled: bool = True,
    posterior_mean: bool = False,
    max_speed: float = 2.0,
    max_acceleration: float = 2.0,
) -> np.ndarray:
    horizon = robot_actions.shape[0]
    output = np.zeros((num_samples, horizon, 2))
    posterior = np.asarray(posterior, dtype=np.float64)
    for sample in range(num_samples):
        position = np.asarray(state0[:2], dtype=np.float64).copy()
        velocity = np.asarray(state0[2:], dtype=np.float64).copy()
        mode = int(rng.choice(artifact.K, p=posterior))
        distribution = posterior.copy()
        for step in range(horizon):
            context = compute_context_features(
                position, velocity, robot_state_positions[step], robot_state_velocities[step],
            )
            if posterior_mean:
                means = np.stack([
                    artifact.mean(
                        k, velocity, context, robot_state_velocities[step], robot_actions[step],
                        response_enabled=response_enabled,
                    )
                    for k in range(artifact.K)
                ])
                mean = np.sum(distribution[:, None] * means, axis=0)
                centered = means - mean
                covariance = sum(
                    distribution[k] * (artifact.Q[k] + np.outer(centered[k], centered[k]))
                    for k in range(artifact.K)
                )
                next_velocity = rng.multivariate_normal(mean, covariance)
                distribution = distribution @ artifact.Pi
            else:
                mean = artifact.mean(
                    mode, velocity, context, robot_state_velocities[step], robot_actions[step],
                    response_enabled=response_enabled,
                )
                next_velocity = rng.multivariate_normal(mean, artifact.Q[mode])
                if step + 1 < horizon:
                    mode = int(rng.choice(artifact.K, p=artifact.Pi[mode]))
            next_velocity = _clip_delta_v_and_speed(
                velocity, next_velocity - velocity, artifact.dt, max_speed, max_acceleration,
            )
            position = position + artifact.dt * next_velocity
            velocity = next_velocity
            output[sample, step] = position
    return output


class CausalBayesianBRNEPolicy:
    """Pure Bayesian planner; no learned navigation policy is loaded."""

    def __init__(self, *, response_enabled: bool = True, posterior_mean: bool = False):
        self.response_enabled = response_enabled
        self.posterior_mean = posterior_mean
        self.config: Optional[PlannerConfig] = None
        self.artifact: Optional[CausalResponseArtifact] = None
        self.beliefs: Optional[CausalBeliefBank] = None
        self.solver: Optional[BRNESolver] = None
        self.episode_seed: Optional[int] = None
        self.step_index = 0
        self.last_action = np.zeros(2)
        self.last_timestamp: Optional[float] = None
        self.last_diagnostics: dict = {}

    def configure(self, config: PlannerConfig, artifact: CausalResponseArtifact) -> None:
        artifact.validate()
        self.config = config
        self.artifact = artifact
        self.beliefs = CausalBeliefBank(artifact, config.max_missed_steps)
        kwargs = {"solver_mode": config.solver_mode, "brne_root": config.brne_root}
        if config.solver_mode == "stable_clearance":
            kwargs.update(safe_distance=config.safe_distance, cost_sigma=config.cost_sigma,
                          cost_scale=config.cost_scale)
        self.solver = BRNESolver(**kwargs)

    def reset(self, episode_seed: int) -> None:
        if self.beliefs is None:
            raise CausalRuntimeError("configure before reset")
        self.beliefs.reset()
        self.episode_seed = int(episode_seed)
        self.step_index = 0
        self.last_action = np.zeros(2)
        self.last_timestamp = None

    def predict(self, observation: PolicyObservation):
        from crowd_sim.envs.utils.action import ActionXY

        if self.artifact is None or self.config is None or self.beliefs is None or self.solver is None:
            raise CausalRuntimeError("policy is not configured")
        if self.episode_seed is None:
            raise CausalRuntimeError("policy is not reset")
        if abs(observation.time_step - self.artifact.dt) > 1e-9:
            raise CausalRuntimeError("environment/artifact dt mismatch")
        if self.last_timestamp is not None and abs(
            observation.timestamp - self.last_timestamp - observation.time_step
        ) > 1e-8:
            raise CausalRuntimeError("non-consecutive policy timestamp")
        started = time.perf_counter()
        self.beliefs.update(observation, self.last_action)
        rng = np.random.default_rng([self.episode_seed, self.step_index])
        actions, state_positions, state_velocities, future_positions = sample_robot_candidates(
            robot_px=observation.robot_px, robot_py=observation.robot_py,
            robot_vx=observation.robot_vx, robot_vy=observation.robot_vy,
            goal_gx=observation.robot_gx, goal_gy=observation.robot_gy,
            horizon=self.config.horizon_steps, num_candidates=self.config.num_samples,
            dt=observation.time_step, rng=rng, v_pref=observation.robot_v_pref,
            max_speed=self.config.max_speed, max_acceleration=self.config.max_acceleration,
            brne_root=self.config.brne_root,
        )
        weights = np.full(self.config.num_samples, 1.0 / self.config.num_samples)
        residual_history = []
        final = None
        track_ids = sorted(h.track_id for h in observation.humans)
        humans = {h.track_id: h for h in observation.humans}
        radii = np.array([observation.robot_radius] + [humans[t].radius for t in track_ids])
        for iteration in range(self.config.max_outer_iterations):
            normalized = weights / weights.sum()
            weighted_positions = np.tensordot(normalized, state_positions, axes=(0, 0))
            weighted_velocities = np.tensordot(normalized, state_velocities, axes=(0, 0))
            weighted_actions = np.tensordot(normalized, actions, axes=(0, 0))
            trajectories = [future_positions]
            for track_id in track_ids:
                human = humans[track_id]
                trajectories.append(sample_causal_trajectories(
                    self.beliefs.predictive(track_id),
                    np.array([human.px, human.py, human.vx, human.vy]),
                    self.artifact, weighted_positions, weighted_velocities, weighted_actions,
                    num_samples=self.config.num_samples,
                    rng=np.random.default_rng([self.episode_seed, self.step_index, track_id]),
                    response_enabled=self.response_enabled, posterior_mean=self.posterior_mean,
                    max_speed=self.config.max_speed, max_acceleration=self.config.max_acceleration,
                ))
            final = self.solver.solve(
                np.stack(trajectories), radii, None,
                equilibrium_iterations=self.config.equilibrium_iterations,
            )
            proposed = np.asarray(final.weights[0], dtype=np.float64)
            proposed /= max(proposed.sum(), 1e-12)
            proposed = (1.0 - self.config.outer_damping) * weights + self.config.outer_damping * proposed
            residual = float(np.max(np.abs(proposed - weights)))
            residual_history.append(residual)
            weights = proposed
            if residual <= self.config.outer_tolerance:
                break
        control = weighted_first_control(weights, actions)
        current_velocity = np.array([observation.robot_vx, observation.robot_vy])
        control = _clip_delta_v_and_speed(
            current_velocity, control - current_velocity, observation.time_step,
            self.config.max_speed, self.config.max_acceleration,
        )
        self.last_action = control.copy()
        self.last_timestamp = observation.timestamp
        self.step_index += 1
        self.last_diagnostics = {
            "outer_iterations": len(residual_history),
            "converged": bool(residual_history and residual_history[-1] <= self.config.outer_tolerance),
            "max_weight_residual": residual_history[-1] if residual_history else 0.0,
            "response_enabled": self.response_enabled,
            "posterior_mean": self.posterior_mean,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "solver_converged": None if final is None else final.converged,
            # Diagnostic only: exposing the final robot distribution lets
            # same-state audits distinguish "belief changed the equilibrium
            # but weighted controls cancelled" from "belief never changed
            # the equilibrium". This value is recorded after action
            # selection and is never read back into control.
            "robot_weights": weights.copy(),
        }
        return ActionXY(float(control[0]), float(control[1]))
