"""Finite L-corner model with an exact discrete hidden state.

The hidden mode is sampled once per episode.  It records whether one pedestrian
exists and, if so, its initial distance and speed in the side corridor.  This
is intentionally *not* called a Poisson arrival model: Phase 2 has at most one
pedestrian and ``p_exist`` is an episode-level prior.

Only a newly deeper rightward peek yields an observation.  Consequently an
episode contains at most ``max_peek`` informative observations, keeping the
finite-horizon belief tree exact rather than silently quantising beliefs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple


ACTIONS: Tuple[str, ...] = (
    "proceed",
    "slow",
    "wait",
    "peek_right",
    "peek_left",
    "detour",
)


@dataclass(frozen=True)
class HiddenMode:
    name: str
    distance: float = 0.0
    speed: float = 0.0

    @property
    def exists(self) -> bool:
        return self.name != "none"


@dataclass(frozen=True)
class RobotState:
    x: int = 0
    y: int = 0
    max_peek: int = 0
    t: int = 0


@dataclass(frozen=True)
class Transition:
    next_state: Optional[RobotState]
    outcome: Optional[str]
    elapsed: int


@dataclass(frozen=True)
class ModelParams:
    goal_x: int = 12
    intersection_x: int = 6
    horizon: int = 9
    distances: Tuple[float, ...] = (3.0, 4.0, 5.0, 6.0)
    speeds: Tuple[float, float] = (1.0, 2.0)
    occupancy_steps: int = 2
    max_peek: int = 2
    peek_steps: int = 1
    visibility_depths: Tuple[int, int] = (3, 6)
    detour_steps: int = 11
    p_miss: float = 0.05
    p_false_alarm: float = 0.02
    success_reward: float = 1.0
    collision_cost: float = 2.0
    timeout_cost: float = 1.0
    step_cost: float = 0.02

    def __post_init__(self) -> None:
        if self.goal_x <= self.intersection_x:
            raise ValueError("goal_x must lie beyond the intersection")
        if self.horizon <= 0 or self.peek_steps <= 0:
            raise ValueError("horizon and peek_steps must be positive")
        if len(self.speeds) != 2 or any(v <= 0 for v in self.speeds):
            raise ValueError("exactly two positive pedestrian speeds are required")
        if len(self.visibility_depths) != self.max_peek:
            raise ValueError("one visibility depth is required per peek level")
        if tuple(sorted(self.visibility_depths)) != self.visibility_depths:
            raise ValueError("visibility depths must be nondecreasing")
        if not (0.0 <= self.p_miss < 1.0 and 0.0 <= self.p_false_alarm < 1.0):
            raise ValueError("sensor error probabilities must be in [0,1)")


class LCornerModel:
    """Exact transition, observation and reward model."""

    def __init__(self, params: ModelParams):
        self.params = params
        modes = [HiddenMode("none")]
        for distance in params.distances:
            for speed_idx, speed in enumerate(params.speeds):
                modes.append(HiddenMode(f"d{distance:g}_v{speed_idx}", distance, speed))
        self.modes: Tuple[HiddenMode, ...] = tuple(modes)

    @property
    def initial_state(self) -> RobotState:
        return RobotState()

    def prior(self, p_exist: float) -> Tuple[float, ...]:
        if not 0.0 <= p_exist <= 1.0:
            raise ValueError("p_exist must be in [0,1]")
        each = p_exist / (len(self.modes) - 1)
        return (1.0 - p_exist,) + (each,) * (len(self.modes) - 1)

    def available_actions(self, state: RobotState) -> Tuple[str, ...]:
        if state.t >= self.params.horizon or state.x >= self.params.goal_x:
            return ()
        actions = ["proceed", "slow", "wait"]
        if state.y < self.params.max_peek and state.max_peek < self.params.max_peek:
            actions.append("peek_right")
        if state.y > 0:
            actions.append("peek_left")
        if state.x < self.params.intersection_x:
            actions.append("detour")
        return tuple(actions)

    def crossing_step(self, mode_idx: int) -> Optional[int]:
        mode = self.modes[mode_idx]
        if not mode.exists:
            return None
        return max(1, int(math.ceil(mode.distance / mode.speed - 1e-12)))

    def pedestrian_distance(self, mode_idx: int, t: int) -> Optional[float]:
        mode = self.modes[mode_idx]
        if not mode.exists:
            return None
        return mode.distance - mode.speed * t

    def pedestrian_at_intersection(self, mode_idx: int, t: int) -> bool:
        crossing = self.crossing_step(mode_idx)
        return crossing is not None and crossing <= t < crossing + self.params.occupancy_steps

    def transition(self, state: RobotState, action: str, mode_idx: int) -> Transition:
        if action not in self.available_actions(state):
            raise ValueError(f"action {action!r} unavailable at {state}")

        if action == "detour":
            elapsed = self.params.detour_steps
            finish = state.t + elapsed
            outcome = "success" if finish <= self.params.horizon else "timeout"
            return Transition(None, outcome, min(elapsed, self.params.horizon - state.t))

        elapsed = self.params.peek_steps if action == "peek_right" else 1
        next_t = state.t + elapsed
        x, y, max_peek = state.x, state.y, state.max_peek
        if action == "proceed":
            x = min(self.params.goal_x, x + 2)
        elif action == "slow":
            x = min(self.params.goal_x, x + 1)
        elif action == "peek_right":
            y += 1
            max_peek = max(max_peek, y)
        elif action == "peek_left":
            y -= 1

        # A peek consumes time while the robot is stationary.  Collision is
        # checked only when the robot actually crosses the junction.
        crosses = state.x < self.params.intersection_x <= x
        if crosses and self.pedestrian_at_intersection(mode_idx, next_t):
            return Transition(None, "collision", elapsed)
        if x >= self.params.goal_x:
            return Transition(None, "success", elapsed)
        if next_t >= self.params.horizon:
            return Transition(None, "timeout", min(elapsed, self.params.horizon - state.t))
        return Transition(RobotState(x=x, y=y, max_peek=max_peek, t=next_t), None, elapsed)

    def is_informative(self, state: RobotState, action: str, next_state: RobotState) -> bool:
        return action == "peek_right" and next_state.max_peek > state.max_peek

    def visibility_depth(self, state: RobotState) -> int:
        if state.max_peek <= 0:
            return 0
        return self.params.visibility_depths[state.max_peek - 1]

    def observation_alphabet(self, state: RobotState, action: str,
                             next_state: RobotState) -> Tuple[int, ...]:
        if not self.is_informative(state, action, next_state):
            return (0,)
        return tuple(range(self.visibility_depth(next_state) + 1))

    def observation_likelihood(self, next_state: RobotState, mode_idx: int,
                               observation: int) -> float:
        """Probability of ``0=no detection`` or a detected distance cell."""
        depth = self.visibility_depth(next_state)
        if depth <= 0:
            return 1.0 if observation == 0 else 0.0
        distance = self.pedestrian_distance(mode_idx, next_state.t)
        visible = distance is not None and 0.0 < distance <= depth
        if visible:
            cell = min(depth, max(1, int(math.ceil(distance - 1e-12))))
            if observation == cell:
                return 1.0 - self.params.p_miss
            if observation == 0:
                return self.params.p_miss
            return 0.0
        if observation == 0:
            return 1.0 - self.params.p_false_alarm
        if 1 <= observation <= depth:
            return self.params.p_false_alarm / depth
        return 0.0

    def transition_reward(self, transition: Transition) -> float:
        reward = -self.params.step_cost * transition.elapsed
        if transition.outcome == "success":
            reward += self.params.success_reward
        elif transition.outcome == "collision":
            reward -= self.params.collision_cost
        elif transition.outcome == "timeout":
            reward -= self.params.timeout_cost
        return reward

    def observation_distribution(self, state: RobotState, action: str,
                                 next_state: RobotState,
                                 weights: Sequence[float]) -> Tuple[Tuple[int, float], ...]:
        if not self.is_informative(state, action, next_state):
            return ((0, float(sum(weights))),)
        result = []
        for observation in self.observation_alphabet(state, action, next_state):
            probability = sum(
                weights[i] * self.observation_likelihood(next_state, i, observation)
                for i in range(len(self.modes))
            )
            if probability > 0.0:
                result.append((observation, probability))
        return tuple(result)


def normalise(weights: Iterable[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    total = sum(values)
    if total <= 0.0:
        raise ValueError("cannot normalise zero probability mass")
    return tuple(value / total for value in values)


def project_existence(belief: Sequence[float], p_exist: float) -> Tuple[float, ...]:
    """Reset total existence confidence while preserving conditional mode mass."""
    if not 0.0 <= p_exist <= 1.0:
        raise ValueError("p_exist must be in [0,1]")
    existing = sum(belief[1:])
    if existing <= 0.0:
        conditional = (1.0 / (len(belief) - 1),) * (len(belief) - 1)
    else:
        conditional = tuple(value / existing for value in belief[1:])
    return (1.0 - p_exist,) + tuple(p_exist * value for value in conditional)
