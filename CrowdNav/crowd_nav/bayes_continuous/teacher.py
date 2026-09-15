"""Existing unicycle CEM controller adapted to the full-observation IL teacher.
Core classes copied without algorithm changes from the user's bayes_occ_mpc:
continuous_mpc_gate.py SHA256 5b5489cb115984f5356c81d09deda5e77af452f47efd8b718e5c173fbbd804df
unicycle_mpc_gate.py SHA256 f00867a5c654d6b176160469af9521e28ded17ca2cc20807e1a743763715a164
Only the adapter below reads current observed states. No goals/futures of humans.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2

@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.25
    horizon: int = 16
    population: int = 768
    iterations: int = 4
    elite_fraction: float = 0.08
    v_max: float = 1.0
    a_max: float = 2.0
    human_margin: float = 0.16
    unknown_margin: float = 0.05
    goal_stage_weight: float = 0.28
    goal_terminal_weight: float = 9.0
    smooth_weight: float = 0.35
    effort_weight: float = 0.02
    collision_weight: float = 20000.0
    discomfort_weight: float = 500.0
    unknown_weight: float = 6500.0
    probability_weight: float = 2500.0
    chance_limit: float = 0.10
    risk_gamma: float = 2.0
    near_chance_limit: float = 0.10
    occupancy_chance_limit: float = 0.15
    fixed_uncertainty_radius: float = 0.35
    stagnation_weight: float = 250.0
    min_progress: float = 0.15
    init_std: float = 0.65
    min_std: float = 0.08
    acceleration_std: float = 0.55
    # Ablation: pin the existence probability used by the risk term while
    # leaving the track set, the means and the covariances untouched.  This
    # separates "is this person still there" from "where exactly are they".
    existence_override: Optional[float] = None
    conformal_visible_radii: Tuple[float, ...] = ()
    conformal_hidden_radii: Tuple[float, ...] = ()


@dataclass
class UnknownField:
    clearance: np.ndarray
    x0: float
    y0: float
    resolution: float

    def sample(self, positions: np.ndarray) -> np.ndarray:
        cols = np.floor((positions[..., 0] - self.x0) / self.resolution).astype(int)
        rows = np.floor((positions[..., 1] - self.y0) / self.resolution).astype(int)
        valid = (
            (rows >= 0)
            & (rows < self.clearance.shape[0])
            & (cols >= 0)
            & (cols < self.clearance.shape[1])
        )
        out = np.zeros(rows.shape, dtype=np.float64)
        out[valid] = self.clearance[rows[valid], cols[valid]]
        return out


@dataclass
class ProbabilityField:
    probability: np.ndarray
    x0: float
    y0: float
    resolution: float

    def sample(self, positions: np.ndarray) -> np.ndarray:
        cols = np.floor((positions[..., 0] - self.x0) / self.resolution).astype(int)
        rows = np.floor((positions[..., 1] - self.y0) / self.resolution).astype(int)
        valid = (
            (rows >= 0)
            & (rows < self.probability.shape[0])
            & (cols >= 0)
            & (cols < self.probability.shape[1])
        )
        out = np.ones(rows.shape, dtype=np.float64)
        out[valid] = self.probability[rows[valid], cols[valid]]
        return out


@dataclass
class PlannerObservation:
    robot_xy: np.ndarray
    robot_velocity: np.ndarray
    robot_radius: float
    goal_xy: np.ndarray
    entities: np.ndarray
    human_segment_start: Optional[np.ndarray]
    human_segment_end: Optional[np.ndarray]
    human_uncertainty_buffer: Optional[np.ndarray]
    human_position_covariance: Optional[np.ndarray]
    human_existence: Optional[np.ndarray]
    human_visible: Optional[np.ndarray]
    unknown: Optional[UnknownField]
    occupancy_probability: Optional[ProbabilityField]
    provenance: str
    # Only the unicycle planner reads this; the holonomic path never touches it.
    robot_heading: Optional[float] = None


class ContinuousCEMMPC:
    """Continuous trajectory optimizer with temporally correlated CEM samples."""

    def __init__(self, config: MPCConfig):
        self.cfg = config
        self._previous_mean: Optional[np.ndarray] = None
        self.last_controls: Optional[np.ndarray] = None
        self.last_diagnostics: Dict[str, float] = {}
        if config.population < 32:
            raise ValueError("route search needs at least 32 samples")

    def reset(self) -> None:
        self._previous_mean = None
        self.last_controls = None

    def _belief_risk_limits(self) -> np.ndarray:
        """Risk budget is strict near execution and relaxed far in the horizon."""
        if self.cfg.horizon <= 1:
            return np.asarray([self.cfg.near_chance_limit], dtype=np.float64)
        phase = np.linspace(0.0, 1.0, self.cfg.horizon, dtype=np.float64)
        limits = (
            self.cfg.near_chance_limit
            + (self.cfg.chance_limit - self.cfg.near_chance_limit)
            * np.power(phase, self.cfg.risk_gamma)
        )
        return limits

    def _belief_hazard_limits(self) -> np.ndarray:
        return -np.log1p(-np.clip(self._belief_risk_limits(), 0.0, 1.0 - 1e-12))

    @staticmethod
    def _goal_velocity(obs: PlannerObservation, speed: float) -> np.ndarray:
        delta = obs.goal_xy - obs.robot_xy
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros(2, dtype=np.float64)
        return delta / distance * min(speed, distance / 0.25)

    @staticmethod
    def _active_until_goal(
        positions: np.ndarray, obs: PlannerObservation
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return task-active steps and goal distances under terminal semantics.

        CrowdSim checks collision on the step entering the goal and then ends
        the episode.  That entry step remains active; only later, nonexistent
        trajectory steps are masked out.
        """
        center_distance = np.linalg.norm(
            positions - obs.goal_xy[None, None, :], axis=2
        )
        reached = center_distance < obs.robot_radius
        reached_before = np.concatenate(
            (
                np.zeros((positions.shape[0], 1), dtype=bool),
                np.maximum.accumulate(reached[:, :-1], axis=1),
            ),
            axis=1,
        )
        return ~reached_before, reached, center_distance

    def _initial_mean(self, obs: PlannerObservation) -> np.ndarray:
        target = self._goal_velocity(obs, self.cfg.v_max)
        if self._previous_mean is None:
            return np.repeat(target[None, :], self.cfg.horizon, axis=0)
        shifted = np.vstack((self._previous_mean[1:], target[None, :]))
        return shifted

    def _seed_trajectories(self, obs: PlannerObservation) -> np.ndarray:
        """Deterministic topology seeds; never average their final actions."""
        goal = self._goal_velocity(obs, self.cfg.v_max)
        goal_norm = max(float(np.linalg.norm(goal)), 1e-12)
        forward = goal / goal_norm
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        targets = [
            goal,
            np.zeros(2, dtype=np.float64),
            -self.cfg.v_max * forward,
            self.cfg.v_max * left,
            -self.cfg.v_max * left,
            self.cfg.v_max * (0.70 * forward + 0.70 * left),
            self.cfg.v_max * (0.70 * forward - 0.70 * left),
        ]
        raw = np.stack(
            [np.repeat(target[None, :], self.cfg.horizon, axis=0) for target in targets]
        )
        return self._project_controls(raw, obs.robot_velocity)

    def _rollout(self, samples: np.ndarray, obs: PlannerObservation):
        """Sampled parameters to (executable parameters, XY velocities, positions).

        The holonomic model is parameterised directly by its velocities, so the
        first two returns are the same array.  A different kinematic model
        overrides this and leaves every cost and clearance term untouched: they
        only ever see XY velocities and the positions they produce.
        """
        controls = self._project_controls(samples, obs.robot_velocity)
        positions = obs.robot_xy[None, None, :] + np.cumsum(
            controls * self.cfg.dt, axis=1
        )
        return controls, controls, positions

    def _project_controls(
        self, controls: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        controls = controls.copy()
        previous = np.broadcast_to(initial_velocity, (controls.shape[0], 2)).copy()
        max_delta = self.cfg.a_max * self.cfg.dt
        for k in range(self.cfg.horizon):
            delta = controls[:, k] - previous
            delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-12))
            controls[:, k] = previous + delta * scale
            speed = np.linalg.norm(controls[:, k], axis=1, keepdims=True)
            controls[:, k] *= np.minimum(
                1.0, self.cfg.v_max / np.maximum(speed, 1e-12)
            )
            previous = controls[:, k]
        return controls

    def _route_seeds(self, obs: PlannerObservation) -> np.ndarray:
        """Left/right bends and wait-then-go, returning toward the actual goal."""
        forward = self._goal_velocity(obs, 1.0)
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        left = np.array([-forward[1], forward[0]])
        controls = np.zeros((3, self.cfg.horizon, 2))
        positions = np.repeat(obs.robot_xy[None, :], 3, axis=0)
        previous = np.repeat(obs.robot_velocity[None, :], 3, axis=0)
        bend_steps = max(1, self.cfg.horizon // 3)
        for step in range(self.cfg.horizon):
            target = obs.goal_xy[None, :] - positions
            distance = np.linalg.norm(target, axis=1, keepdims=True)
            target *= np.minimum(self.cfg.v_max / np.maximum(distance, 1e-12),
                                 1.0 / self.cfg.dt)
            if step < bend_steps:
                target[0] = self.cfg.v_max * (0.5 * forward + np.sqrt(0.75) * left)
                target[1] = self.cfg.v_max * (0.5 * forward - np.sqrt(0.75) * left)
            if step < max(1, self.cfg.horizon // 4):
                target[2] = 0.0
            delta = target - previous
            delta *= np.minimum(1.0, self.cfg.a_max * self.cfg.dt /
                                np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12))
            previous = previous + delta
            controls[:, step] = previous
            positions += self.cfg.dt * previous
        return controls

    @staticmethod
    def _elite_indices(costs, full_clearance, first_clearance, count):
        # Fit each route to feasible samples first, not a mix across obstacles.
        feasible = full_clearance >= 0.0
        first_feasible = first_clearance >= 0.0
        category = np.where(feasible, 0, np.where(first_feasible, 1, 2))
        violation = np.where(feasible, 0.0,
                             np.where(first_feasible, -full_clearance, -first_clearance))
        return np.lexsort((costs, violation, category))[:count]

    def _human_clearance(
        self, controls: np.ndarray, obs: PlannerObservation
    ) -> np.ndarray:
        """Swept clearance for every population/human/horizon segment."""
        if not obs.entities.size:
            return np.empty((controls.shape[0], 0, self.cfg.horizon))
        cfg = self.cfg
        robot_end = obs.robot_xy[None, None, :] + np.cumsum(
            controls * cfg.dt, axis=1
        )
        robot_start = np.concatenate(
            (
                np.broadcast_to(obs.robot_xy, (controls.shape[0], 1, 2)),
                robot_end[:, :-1],
            ),
            axis=1,
        )
        if obs.human_segment_start is not None:
            human_start = obs.human_segment_start
            human_end = obs.human_segment_end
        else:
            times = cfg.dt * np.arange(1, cfg.horizon + 1, dtype=np.float64)
            human_end = (
                obs.entities[:, None, 0:2]
                + obs.entities[:, None, 2:4] * times[None, :, None]
            )
            human_start = np.concatenate(
                (obs.entities[:, None, 0:2], human_end[:, :-1]), axis=1
            )
        rel_start = human_start[None, :, :, :] - robot_start[:, None, :, :]
        rel_end = human_end[None, :, :, :] - robot_end[:, None, :, :]
        segment = rel_end - rel_start
        denominator = np.square(segment).sum(axis=3)
        fraction = np.clip(
            -np.sum(rel_start * segment, axis=3)
            / np.maximum(denominator, 1e-12),
            0.0,
            1.0,
        )
        closest = rel_start + fraction[:, :, :, None] * segment
        radii = obs.robot_radius + obs.entities[None, :, None, 4]
        return np.linalg.norm(closest, axis=3) - radii

    def _cost(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: np.ndarray,
        human_clearance: Optional[np.ndarray],
        occupancy_probability: Optional[np.ndarray],
        belief_hazard: Optional[np.ndarray],
    ) -> np.ndarray:
        cfg = self.cfg
        active, reached, center_distance = self._active_until_goal(positions, obs)
        # CrowdSim terminates when the robot enters its goal-radius disc.  The
        # optimizer must target that same set instead of an unnecessarily
        # stricter (and sometimes occupied) zero-radius point.
        goal_dist = np.maximum(center_distance - obs.robot_radius, 0.0)
        terminal_goal_dist = np.where(
            reached.any(axis=1), 0.0, goal_dist[:, -1]
        )
        cost = (
            cfg.goal_stage_weight * (goal_dist * active).sum(axis=1)
            + cfg.goal_terminal_weight * terminal_goal_dist
            + cfg.effort_weight
            * (np.square(controls) * active[:, :, None]).sum(axis=(1, 2))
        )
        previous = np.concatenate(
            (
                np.broadcast_to(obs.robot_velocity, (controls.shape[0], 1, 2)),
                controls[:, :-1],
            ),
            axis=1,
        )
        cost += cfg.smooth_weight * (
            np.square(controls - previous) * active[:, :, None]
        ).sum(axis=(1, 2))

        initial_distance = max(
            float(np.linalg.norm(obs.robot_xy - obs.goal_xy)) - obs.robot_radius,
            0.0,
        )
        progress = initial_distance - terminal_goal_dist
        shortfall = np.maximum(cfg.min_progress - progress, 0.0)
        cost += cfg.stagnation_weight * np.square(shortfall)

        if obs.entities.size:
            clearance = human_clearance
            penetration = np.maximum(-clearance, 0.0)
            margin = cfg.human_margin
            if obs.human_uncertainty_buffer is not None:
                margin = margin + obs.human_uncertainty_buffer[None, :, :]
            discomfort = np.maximum(margin - clearance, 0.0)
            human_active = active[:, None, :]
            cost += cfg.collision_weight * (
                np.square(penetration) * human_active
            ).sum(axis=(1, 2))
            cost += cfg.discomfort_weight * (
                np.square(discomfort) * human_active
            ).sum(axis=(1, 2))
            cost += 1e5 * (
                np.where(human_active, penetration, 0.0).max(axis=(1, 2)) > 0.02
            )

        if obs.unknown is not None:
            clearance = obs.unknown.sample(positions)
            required = obs.robot_radius + cfg.unknown_margin
            penetration = np.maximum(required - clearance, 0.0)
            cost += cfg.unknown_weight * (np.square(penetration) * active).sum(axis=1)
            cost += 1e5 * (
                np.where(active, penetration, 0.0).max(axis=1) > 0.05
            )

        if obs.occupancy_probability is not None:
            probability = occupancy_probability
            cost += cfg.probability_weight * (
                -np.log1p(-np.clip(probability, 0.0, 1.0 - 1e-9)) * active
            ).sum(axis=1)
            cost += 1e5 * (
                np.where(active, probability, 0.0).max(axis=1)
                > cfg.occupancy_chance_limit
            )

        if belief_hazard is not None:
            cost += cfg.probability_weight * (belief_hazard * active).sum(axis=1)
            limits = self._belief_hazard_limits()[None, :]
            cost += 1e5 * (
                np.where(active, belief_hazard - limits, -math.inf).max(axis=1)
                > 0.0
            )

        return cost

    def _belief_collision_hazard(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Bernoulli-Gaussian collision hazard per step.

        The current filter's position covariance is isotropic, so squared
        radial distance follows a non-central chi-square distribution.  Its
        CDF gives the exact probability mass inside the collision disc.  The
        additive negative log-survival form preserves ordering when union
        probabilities round to one in dense crowds.
        """
        if (
            obs.human_position_covariance is None
            or obs.human_existence is None
            or not obs.entities.size
        ):
            return None
        if positions is None:
            positions = obs.robot_xy[None, None, :] + np.cumsum(
                controls * self.cfg.dt, axis=1
            )
        human_positions = obs.human_segment_end
        relative = positions[:, None, :, :] - human_positions[None, :, :, :]
        distance = np.linalg.norm(relative, axis=3)
        variance = 0.5 * np.trace(
            obs.human_position_covariance, axis1=2, axis2=3
        )
        collision_radius = (
            obs.robot_radius
            + obs.entities[None, :, None, 4]
            + self.cfg.human_margin
        )
        scaled_radius = np.square(collision_radius) / np.maximum(
            variance[None, :, :], 1e-9
        )
        noncentrality = np.square(distance) / np.maximum(
            variance[None, :, :], 1e-9
        )
        # A collision disc lies inside its near-side tangent half-plane.
        # Beyond 8 sigma use the upper bound Phi(-8), never drop a person.
        # Per-step hazard error is at most N * 6.23e-16 (roundoff aside).
        far = (distance - collision_radius) >= 8.0 * np.sqrt(variance)[None, :, :]
        conditional = np.full(distance.shape, ndtr(-8.0))
        near = ~far
        conditional[near] = ncx2.cdf(
            np.broadcast_to(scaled_radius, distance.shape)[near],
            2.0, noncentrality[near],
        )
        deterministic = variance < 1e-9
        if np.any(deterministic):
            conditional = np.where(
                deterministic[None, :, :],
                distance <= collision_radius,
                conditional,
            )
        existence = obs.human_existence
        if self.cfg.existence_override is not None:
            existence = np.full_like(existence, self.cfg.existence_override)
        component = conditional * existence[None, :, None]
        return -np.log1p(-np.clip(component, 0.0, 1.0 - 1e-12)).sum(axis=1)

    def _belief_collision_probability(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        hazard = self._belief_collision_hazard(controls, obs, positions)
        return None if hazard is None else -np.expm1(-hazard)

    def _combined_clearance(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: np.ndarray,
        human_clearance: Optional[np.ndarray],
        occupancy_probability: Optional[np.ndarray],
        belief_hazard: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Minimum robust clearance over the horizon and over the first step."""
        population = controls.shape[0]
        active, _, _ = self._active_until_goal(positions, obs)
        full = np.full(population, math.inf, dtype=np.float64)
        first = np.full(population, math.inf, dtype=np.float64)
        if obs.entities.size:
            margin = self.cfg.human_margin
            if obs.human_uncertainty_buffer is not None:
                margin = margin + obs.human_uncertainty_buffer[None, :, :]
            human = np.where(
                active[:, None, :], human_clearance - margin, math.inf
            )
            full = np.minimum(full, human.min(axis=(1, 2)))
            first = np.minimum(first, human[:, :, 0].min(axis=1))
        if obs.unknown is not None:
            unknown = np.where(
                active,
                obs.unknown.sample(positions)
                - (obs.robot_radius + self.cfg.unknown_margin),
                math.inf,
            )
            full = np.minimum(full, unknown.min(axis=1))
            first = np.minimum(first, unknown[:, 0])
        if obs.occupancy_probability is not None:
            probability = occupancy_probability
            slack = np.where(
                active, self.cfg.occupancy_chance_limit - probability, math.inf
            )
            full = np.minimum(full, slack.min(axis=1))
            first = np.minimum(first, slack[:, 0])
        if belief_hazard is not None:
            slack = np.where(
                active,
                self._belief_hazard_limits()[None, :] - belief_hazard,
                math.inf,
            )
            full = np.minimum(full, slack.min(axis=1))
            first = np.minimum(first, slack[:, 0])
        return full, first

    def _physical_clearance(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: np.ndarray,
        human_clearance: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Geometric clearance used when every chance constraint is infeasible."""
        population = controls.shape[0]
        active, _, _ = self._active_until_goal(positions, obs)
        full = np.full(population, math.inf, dtype=np.float64)
        first = np.full(population, math.inf, dtype=np.float64)
        if obs.entities.size:
            margin = self.cfg.human_margin
            if obs.human_uncertainty_buffer is not None:
                margin = margin + obs.human_uncertainty_buffer[None, :, :]
            human = np.where(
                active[:, None, :], human_clearance - margin, math.inf
            )
            full = np.minimum(full, human.min(axis=(1, 2)))
            first = np.minimum(first, human[:, :, 0].min(axis=1))
        if obs.unknown is not None:
            unknown = np.where(
                active,
                obs.unknown.sample(positions)
                - (obs.robot_radius + self.cfg.unknown_margin),
                math.inf,
            )
            full = np.minimum(full, unknown.min(axis=1))
            first = np.minimum(first, unknown[:, 0])
        return full, first

    def _degraded_choice(self, costs, first_physical, params, obs) -> int:
        """Which candidate to execute when nothing is feasible.

        The registered policy keeps moving: among the candidates that hold the
        largest physical clearance at the first step, take the cheapest.  The
        modern comparators instead brake on an infeasible solve, and whether that
        difference matters is an experimental question, so subclasses override
        this rather than the whole optimiser.
        """
        safest = float(first_physical.max())
        near_safest = np.flatnonzero(first_physical >= safest - 0.02)
        return int(near_safest[np.argmin(costs[near_safest])])

    def plan(self, obs: PlannerObservation, seed: int) -> Tuple[np.ndarray, float]:
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        mean = self._initial_mean(obs)
        start = time.perf_counter()
        seeds = self._seed_trajectories(obs)
        route_seeds = self._route_seeds(obs)
        means = np.concatenate((mean[None, :, :], route_seeds), axis=0)
        deviations = np.full_like(means, cfg.init_std)
        # Retain half the budget around the warm start. Fit alternatives
        # independently so left/right turns cannot cancel out.
        # The mode structure follows the number of route seeds the planner
        # produced.  With the default three routes this is the original
        # half-to-the-warm-start, thirds-to-the-routes split; with zero routes it
        # collapses to one distribution over the whole population, which is what
        # the route-separation ablation needs -- same budget, same seeds, same
        # execution rule, only the grouping changes.
        n_routes = int(route_seeds.shape[0])
        if n_routes:
            remaining = cfg.population - cfg.population // 2
            counts = [cfg.population // 2] + [remaining // n_routes] * n_routes
            for i in range(remaining % n_routes):
                counts[i + 1] += 1
        else:
            counts = [cfg.population]
        bounds = np.cumsum([0] + counts)
        mode_ids = np.repeat(np.arange(len(counts)), counts)
        best_controls = seeds[0].copy()
        best_cost = math.inf
        best_class = 3
        best_first_clearance = -math.inf
        best_full_clearance = -math.inf
        best_first_physical = -math.inf

        for _ in range(cfg.iterations):
            noise = rng.standard_normal((cfg.population, cfg.horizon, 2))
            # Low-pass noise gives coherent curves instead of jittering controls.
            noise[:, 1:] = 0.68 * noise[:, :-1] + 0.32 * noise[:, 1:]
            samples = means[mode_ids] + deviations[mode_ids] * noise
            samples[:len(seeds)] = seeds
            samples[len(seeds)] = mean
            for route in range(1, len(counts)):
                samples[bounds[route]] = means[route]
                samples[bounds[route] + 1] = route_seeds[route - 1]
            params, controls, positions = self._rollout(samples, obs)
            human_clearance = (
                self._human_clearance(controls, obs) if obs.entities.size else None
            )
            occupancy_probability = (
                obs.occupancy_probability.sample(positions)
                if obs.occupancy_probability is not None else None
            )
            belief_hazard = self._belief_collision_hazard(
                controls, obs, positions
            )
            costs = self._cost(
                controls,
                obs,
                positions,
                human_clearance,
                occupancy_probability,
                belief_hazard,
            )
            full_clearance, first_clearance = self._combined_clearance(
                controls,
                obs,
                positions,
                human_clearance,
                occupancy_probability,
                belief_hazard,
            )
            _, first_physical = self._physical_clearance(
                controls, obs, positions, human_clearance
            )
            # Clearances already include the configured human/unknown buffers.
            full_feasible = full_clearance >= 0.0
            first_feasible = first_clearance >= 0.0
            if full_feasible.any():
                pool = np.flatnonzero(full_feasible)
                iteration_best = int(pool[np.argmin(costs[pool])])
                candidate_class = 0
            elif first_feasible.any():
                pool = np.flatnonzero(first_feasible)
                safest = float(full_clearance[pool].max())
                near_safest = pool[full_clearance[pool] >= safest - 0.02]
                iteration_best = int(near_safest[np.argmin(costs[near_safest])])
                candidate_class = 1
            else:
                iteration_best = self._degraded_choice(
                    costs, first_physical, params, obs)
                candidate_class = 2
            candidate_cost = float(costs[iteration_best])
            candidate_first = float(first_clearance[iteration_best])
            replace = candidate_class < best_class
            if candidate_class == best_class == 2:
                candidate_physical = float(first_physical[iteration_best])
                replace = (
                    candidate_physical > best_first_physical + 0.02
                    or (
                        candidate_physical >= best_first_physical - 0.02
                        and candidate_cost < best_cost
                    )
                )
            elif candidate_class == best_class == 1:
                replace = (
                    float(full_clearance[iteration_best]) > best_full_clearance + 0.02
                    or (
                        float(full_clearance[iteration_best])
                        >= best_full_clearance - 0.02
                        and candidate_cost < best_cost
                    )
                )
            elif candidate_class == best_class:
                replace = candidate_cost < best_cost
            if replace:
                best_class = candidate_class
                best_cost = candidate_cost
                best_first_clearance = candidate_first
                best_full_clearance = float(full_clearance[iteration_best])
                best_first_physical = float(first_physical[iteration_best])
                best_controls = params[iteration_best].copy()
            for route, count in enumerate(counts):
                begin, end = bounds[route:route + 2]
                indices = self._elite_indices(
                    costs[begin:end], full_clearance[begin:end],
                    first_clearance[begin:end], max(4, round(count * cfg.elite_fraction)),
                ) + begin
                elite = params[indices]
                means[route] = 0.22 * means[route] + 0.78 * elite.mean(axis=0)
                deviations[route] = np.maximum(cfg.min_std, elite.std(axis=0))

        # The average of trajectories passing on opposite sides can pass through
        # the pedestrian.  Execute the best optimized trajectory, not that mean.
        action = best_controls[0]
        self._previous_mean = best_controls
        # Read-only diagnostic snapshot.  Keeping the selected trajectory makes
        # failure attribution possible without changing the optimizer contract.
        self.last_controls = best_controls.copy()
        self.last_diagnostics = {
            "feasibility_class": float(best_class),
            "first_clearance": best_first_clearance,
            "horizon_clearance": best_full_clearance,
            "first_physical_clearance": best_first_physical,
            "cost": best_cost,
        }
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return action, elapsed_ms


@dataclass(frozen=True)
class UnicycleConfig(MPCConfig):
    """MPCConfig plus the two limits a unicycle needs.

    Registered before any development run and not tuned per arm:
    `omega_max` is the turn rate, `reverse` states whether negative speeds are
    allowed.  Both must be identical across every method in a matched
    comparison; they are recorded in the result metadata.
    """

    # 0.8 rad/s: the turn rate the modern MPC comparators' shipped model
    # supports.  Taking the tighter of the two keeps every method inside its
    # own design range rather than giving one an advantage it cannot match.
    omega_max: float = 0.8          # rad/s
    reverse: bool = False           # unicycle comparators do not drive backwards


class UnicycleCEMMPC(ContinuousCEMMPC):
    """ContinuousCEMMPC with a second-order-free unicycle rollout."""

    def __init__(self, config: MPCConfig):
        if not isinstance(config, UnicycleConfig):
            # Accept a plain MPCConfig so callers can share one config factory.
            config = UnicycleConfig(
                **{f.name: getattr(config, f.name) for f in fields(MPCConfig)}
            )
        super().__init__(config)

    # ---------------------------------------------------------------- helpers

    def _heading(self, obs: PlannerObservation) -> float:
        if obs.robot_heading is not None:
            return float(obs.robot_heading)
        # Falling back to the velocity direction would silently invent a heading
        # for a stopped robot, so this is an error rather than a guess.
        raise ValueError(
            "unicycle planner requires observation.robot_heading; the adapter "
            "did not provide it"
        )

    def _speed(self, obs: PlannerObservation) -> float:
        return float(np.linalg.norm(obs.robot_velocity))

    def _turn_to(self, heading: float, target_angle: np.ndarray) -> np.ndarray:
        """Shortest signed turn from heading to each target angle."""
        return np.arctan2(np.sin(target_angle - heading), np.cos(target_angle - heading))

    # ------------------------------------------------- parameterisation hooks

    def _project_controls(
        self, controls: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        """Clip (speed, heading increment) sequences to the registered limits.

        `initial_velocity` carries the current speed only; the heading enters
        through the rollout, not through the limits, because the turn bound is
        on the increment itself.
        """
        cfg = self.cfg
        controls = np.array(controls, dtype=np.float64, copy=True)
        max_turn = cfg.omega_max * cfg.dt
        max_speed_delta = cfg.a_max * cfg.dt
        speed_floor = -cfg.v_max if cfg.reverse else 0.0

        previous_speed = np.full(
            controls.shape[0], float(np.linalg.norm(initial_velocity)), dtype=np.float64
        )
        for k in range(cfg.horizon):
            speed = controls[:, k, 0]
            speed = np.clip(
                speed,
                np.maximum(previous_speed - max_speed_delta, speed_floor),
                np.minimum(previous_speed + max_speed_delta, cfg.v_max),
            )
            controls[:, k, 0] = speed
            controls[:, k, 1] = np.clip(controls[:, k, 1], -max_turn, max_turn)
            previous_speed = speed
        return controls

    def _rollout(self, samples: np.ndarray, obs: PlannerObservation):
        """Integrate the unicycle exactly as CrowdSim does, then hand the rest
        of the planner the XY velocities it expects."""
        cfg = self.cfg
        controls = self._project_controls(samples, obs.robot_velocity)

        heading = np.cumsum(controls[:, :, 1], axis=1) + self._heading(obs)
        velocities = np.stack(
            (controls[:, :, 0] * np.cos(heading),
             controls[:, :, 0] * np.sin(heading)),
            axis=2,
        )
        positions = obs.robot_xy[None, None, :] + np.cumsum(velocities * cfg.dt, axis=1)
        return controls, velocities, positions

    # ---------------------------------------------------------------- seeding

    def _initial_mean(self, obs: PlannerObservation) -> np.ndarray:
        """Warm start: last plan shifted forward, ending with a step that aims
        at the goal at full speed."""
        cfg = self.cfg
        heading = self._heading(obs)
        to_goal = obs.goal_xy - obs.robot_xy
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0]))
        turn = float(np.clip(self._turn_to(heading, np.asarray(goal_angle)),
                             -cfg.omega_max * cfg.dt, cfg.omega_max * cfg.dt))
        target = np.array([cfg.v_max, turn], dtype=np.float64)
        if self._previous_mean is None:
            return np.repeat(target[None, :], cfg.horizon, axis=0)
        return np.vstack((self._previous_mean[1:], target[None, :]))

    def _seed_trajectories(self, obs: PlannerObservation) -> np.ndarray:
        """The same seven topologies the holonomic planner uses, expressed as
        turn-and-go sequences: straight at the goal, stop, turn hard either way,
        and two intermediate bends."""
        cfg = self.cfg
        heading = self._heading(obs)
        to_goal = obs.goal_xy - obs.robot_xy
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0]))
        goal_turn = float(self._turn_to(heading, np.asarray(goal_angle)))
        max_turn = cfg.omega_max * cfg.dt

        def sequence(total_turn: float, speed: float) -> np.ndarray:
            """Spread a total heading change over the first steps, then hold."""
            per_step = np.clip(total_turn, -max_turn * cfg.horizon,
                               max_turn * cfg.horizon) / max(1, cfg.horizon // 2)
            per_step = float(np.clip(per_step, -max_turn, max_turn))
            turns = np.zeros(cfg.horizon, dtype=np.float64)
            turns[: max(1, cfg.horizon // 2)] = per_step
            return np.stack((np.full(cfg.horizon, speed), turns), axis=1)

        seeds = [
            sequence(goal_turn, cfg.v_max),                 # straight at the goal
            sequence(0.0, 0.0),                             # hold still
            sequence(np.pi, cfg.v_max * 0.5),               # turn around
            sequence(max_turn * cfg.horizon, cfg.v_max),    # hard left
            sequence(-max_turn * cfg.horizon, cfg.v_max),   # hard right
            sequence(goal_turn + max_turn * 4, cfg.v_max),  # bend left of the goal
            sequence(goal_turn - max_turn * 4, cfg.v_max),  # bend right of the goal
        ]
        return self._project_controls(np.stack(seeds), obs.robot_velocity)

    def _route_seeds(self, obs: PlannerObservation) -> np.ndarray:
        """Left bend, right bend, and wait-then-go, each returning to the goal
        heading afterwards.  These are the three independent CEM modes."""
        cfg = self.cfg
        heading = self._heading(obs)
        max_turn = cfg.omega_max * cfg.dt
        bend_steps = max(1, cfg.horizon // 3)
        wait_steps = max(1, cfg.horizon // 4)

        controls = np.zeros((3, cfg.horizon, 2), dtype=np.float64)
        headings = np.full(3, heading, dtype=np.float64)
        positions = np.repeat(obs.robot_xy[None, :], 3, axis=0)

        for step in range(cfg.horizon):
            to_goal = obs.goal_xy[None, :] - positions
            goal_angle = np.arctan2(to_goal[:, 1], to_goal[:, 0])
            turn = self._turn_to(headings, goal_angle)

            # Route 0 bends left first, route 1 bends right first, route 2 waits.
            if step < bend_steps:
                turn[0] = max_turn
                turn[1] = -max_turn
            turn = np.clip(turn, -max_turn, max_turn)

            speed = np.full(3, cfg.v_max, dtype=np.float64)
            if step < wait_steps:
                speed[2] = 0.0

            controls[:, step, 0] = speed
            controls[:, step, 1] = turn
            headings = headings + turn
            positions = positions + cfg.dt * np.stack(
                (speed * np.cos(headings), speed * np.sin(headings)), axis=1
            )

        return self._project_controls(controls, obs.robot_velocity)


def gdbn_teacher_moments(entities, snapshot, model, horizon, dt):
    """Exact first/second moments of the fitted switching-linear prediction.

    Current geometry is observed exactly. Conditional mode moments are propagated
    separately; only the existing CEM risk interface uses an isotropic projection.
    Mode entropy is not an existence probability.
    """
    if dt != .25 or snapshot.coordinate_frame != 'world_xy_velocity':
        raise ValueError('GDBN teacher requires the fitted .25s world-frame model')
    n=len(entities); k=snapshot.mode_count
    valid=np.asarray(snapshot.valid_mask, bool)
    if valid.sum()!=n or not valid[:n].all():
        raise ValueError('Teacher requires aligned valid entity slots')
    weights=np.asarray(snapshot.features,float)[:n,:k].copy()
    transition=np.asarray(model.Pi,float)
    if not np.allclose(weights.sum(1),1.,atol=1e-6) or not np.allclose(transition.sum(1),1.):
        raise ValueError('Unnormalized mode probabilities')
    weights/=weights.sum(1,keepdims=True)
    means=np.repeat(entities[:,None,:4],k,axis=1)
    covs=np.zeros((n,k,4,4))
    ends=[]; positions=[]
    for _ in range(horizon):
        joint=weights[:,:,None]*transition[None,:,:]
        next_weights=joint.sum(1)
        conditional=joint/np.maximum(next_weights[:,None,:],1e-300)
        mixed_mean=np.einsum('nij,nid->njd',conditional,means)
        second=covs+means[:,:,:,None]*means[:,:,None,:]
        mixed_second=np.einsum('nij,nide->njde',conditional,second)
        mixed_cov=mixed_second-mixed_mean[:,:,:,None]*mixed_mean[:,:,None,:]
        for mode in range(k):
            a=np.asarray(model.A[mode]); q=np.asarray(model.Q[mode])
            means[:,mode]=mixed_mean[:,mode]@a.T
            covs[:,mode]=a[None]@mixed_cov[:,mode]@a.T[None]+q[None]
        weights=next_weights
        mean=np.einsum('nk,nkd->nd',weights,means)
        delta=means-mean[:,None,:]
        covariance=np.einsum('nk,nkij->nij',weights,covs+delta[:,:,:,None]*delta[:,:,None,:])
        covariance=(covariance+covariance.swapaxes(-1,-2))/2
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() < -1e-7:
            raise ValueError('Invalid GDBN predictive covariance')
        ends.append(mean[:,:2].copy()); positions.append(covariance[:,:2,:2].copy())
    end=np.stack(ends,axis=1); covariance=np.stack(positions,axis=1)
    start=np.concatenate([entities[:,None,:2],end[:,:-1]],axis=1)
    return start,end,covariance


def unicycle_teacher(world, belief_snapshot=None, dynamics=None):
    robot, dt = world.robot, world.env.time_step
    if getattr(world, "_cem_teacher", None) is None:
        world._cem_teacher = UnicycleCEMMPC(UnicycleConfig(
            dt=dt, population=512, iterations=4, omega_max=1.2, human_margin=.50))
    planner = world._cem_teacher
    # Keep the qualified .50m geometric buffer, but do not call intrusion into
    # that buffer a physical collision event in the probability calculation.
    planner.cfg = replace(planner.cfg, human_margin=0.0 if belief_snapshot is not None else .50)
    entities = np.asarray([[h.px, h.py, h.vx, h.vy, h.radius]
                           for h in world.env.humans], dtype=np.float64).reshape(-1, 5)
    observation = PlannerObservation(
        robot_xy=np.array([robot.px, robot.py]),
        robot_velocity=np.array([robot.vx, robot.vy]), robot_radius=robot.radius,
        goal_xy=np.array([robot.gx, robot.gy]), entities=entities,
        human_segment_start=None, human_segment_end=None,
        human_uncertainty_buffer=None, human_position_covariance=None,
        human_existence=None, human_visible=None, unknown=None,
        occupancy_probability=None, provenance="full_observation_cv_teacher",
        robot_heading=robot.theta)
    if belief_snapshot is not None:
        if dynamics is None:
            raise ValueError('Belief teacher requires dynamics')
        start,end,cov=gdbn_teacher_moments(entities,belief_snapshot,dynamics,planner.cfg.horizon,dt)
        observation.human_segment_start=start
        observation.human_segment_end=end
        observation.human_position_covariance=cov
        observation.human_uncertainty_buffer=np.full((len(entities),planner.cfg.horizon),.50)
        observation.human_existence=np.ones(len(entities))
        observation.human_visible=np.ones(len(entities),bool)
        observation.provenance='full_observation_gdbn_moment_teacher'
    command, elapsed = planner.plan(observation, seed=2407+round(world.env.global_time/dt))
    world.teacher_diagnostics = dict(planner.last_diagnostics, elapsed_ms=elapsed,
        provenance=observation.provenance, covariance_projection='trace/2 isotropic',
        full_multimodal_risk=False, geometric_margin=.50,
        probability_event='physical_overlap' if belief_snapshot is not None else 'disabled')
    return np.asarray([command[0], command[1]/dt], dtype=np.float32)
