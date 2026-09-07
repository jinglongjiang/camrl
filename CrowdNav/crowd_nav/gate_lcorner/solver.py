"""Exact finite-horizon solvers and exact policy evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple

from .model import LCornerModel, RobotState, normalise


Belief = Tuple[float, ...]
Projector = Callable[[Sequence[float]], Belief]


def identity_projector(belief: Sequence[float]) -> Belief:
    return tuple(float(value) for value in belief)


def _continuation(model: LCornerModel, state: RobotState, action: str,
                  belief: Sequence[float]):
    terminal_value = 0.0
    continuing = [0.0] * len(model.modes)
    next_state = None
    elapsed = None
    for mode_idx, probability in enumerate(belief):
        if probability <= 0.0:
            continue
        transition = model.transition(state, action, mode_idx)
        if transition.outcome is not None:
            terminal_value += probability * model.transition_reward(transition)
        else:
            continuing[mode_idx] = probability
            if next_state is None:
                next_state = transition.next_state
                elapsed = transition.elapsed
            elif next_state != transition.next_state or elapsed != transition.elapsed:
                raise AssertionError("observable continuation must not depend on hidden mode")
    return terminal_value, tuple(continuing), next_state, elapsed


def posterior(model: LCornerModel, state: RobotState, action: str,
              next_state: RobotState, belief: Sequence[float],
              observation: int, projector: Projector = identity_projector) -> Belief:
    informative = model.is_informative(state, action, next_state)
    joint = []
    for mode_idx, probability in enumerate(belief):
        transition = model.transition(state, action, mode_idx)
        if transition.outcome is not None:
            joint.append(0.0)
        else:
            likelihood = (model.observation_likelihood(next_state, mode_idx, observation)
                          if informative else float(observation == 0))
            joint.append(probability * likelihood)
    return projector(normalise(joint))


class ExactBeliefSolver:
    """Bellman recursion over every reachable, unrounded posterior."""

    def __init__(self, model: LCornerModel, *, future_observations: bool = True,
                 projector: Projector = identity_projector):
        self.model = model
        self.future_observations = future_observations
        self.projector = projector

    @lru_cache(maxsize=None)
    def _value(self, state: RobotState, belief: Belief) -> Tuple[float, str]:
        belief = self.projector(belief)
        best_value = float("-inf")
        best_action = ""
        for action in self.model.available_actions(state):
            value = self._q_value(state, belief, action)
            if value > best_value + 1e-12:
                best_value, best_action = value, action
        if not best_action:
            raise AssertionError(f"no action at nonterminal state {state}")
        return best_value, best_action

    def _q_value(self, state: RobotState, belief: Belief, action: str) -> float:
        terminal, continuing, next_state, elapsed = _continuation(
            self.model, state, action, belief
        )
        continuing_mass = sum(continuing)
        if continuing_mass <= 0.0:
            return terminal
        assert next_state is not None and elapsed is not None
        step_reward = -self.model.params.step_cost * elapsed

        if self.future_observations and self.model.is_informative(state, action, next_state):
            value = terminal
            for observation in self.model.observation_alphabet(state, action, next_state):
                joint = tuple(
                    continuing[i]
                    * self.model.observation_likelihood(next_state, i, observation)
                    for i in range(len(self.model.modes))
                )
                probability = sum(joint)
                if probability <= 0.0:
                    continue
                updated = self.projector(normalise(joint))
                value += probability * (step_reward + self._value(next_state, updated)[0])
            return value

        updated = self.projector(normalise(continuing))
        return terminal + continuing_mass * (step_reward + self._value(next_state, updated)[0])

    def value(self, state: RobotState, belief: Sequence[float]) -> float:
        return self._value(state, self.projector(normalise(belief)))[0]

    def action(self, state: RobotState, belief: Sequence[float]) -> str:
        return self._value(state, self.projector(normalise(belief)))[1]

    @property
    def cached_states(self) -> int:
        return self._value.cache_info().currsize


class OracleSolver:
    def __init__(self, model: LCornerModel):
        self.model = model

    @lru_cache(maxsize=None)
    def _value(self, state: RobotState, mode_idx: int) -> Tuple[float, str]:
        best_value = float("-inf")
        best_action = ""
        for action in self.model.available_actions(state):
            transition = self.model.transition(state, action, mode_idx)
            value = self.model.transition_reward(transition)
            if transition.outcome is None:
                assert transition.next_state is not None
                value += self._value(transition.next_state, mode_idx)[0]
            if value > best_value + 1e-12:
                best_value, best_action = value, action
        if not best_action:
            raise AssertionError(f"no action at nonterminal state {state}")
        return best_value, best_action

    def value(self, state: RobotState, mode_idx: int) -> float:
        return self._value(state, mode_idx)[0]

    def action(self, state: RobotState, mode_idx: int) -> str:
        return self._value(state, mode_idx)[1]


class BeliefPolicy(Protocol):
    name: str
    projector: Projector

    def action(self, state: RobotState, belief: Sequence[float]) -> str:
        ...


class SolverPolicy:
    def __init__(self, name: str, solver: ExactBeliefSolver):
        self.name = name
        self.solver = solver
        self.projector = solver.projector

    def action(self, state: RobotState, belief: Sequence[float]) -> str:
        return self.solver.action(state, belief)


class MapAdaptivePolicy:
    name = "map_adaptive"
    projector = staticmethod(identity_projector)

    def __init__(self, oracle: OracleSolver):
        self.oracle = oracle

    def action(self, state: RobotState, belief: Sequence[float]) -> str:
        mode_idx = max(range(len(belief)), key=lambda idx: belief[idx])
        return self.oracle.action(state, mode_idx)


class FixedSequencePolicy:
    projector = staticmethod(identity_projector)

    def __init__(self, pattern: str, wait_steps: int = 0):
        self.pattern = pattern
        self.wait_steps = wait_steps
        self.name = pattern if pattern != "wait_then_go" else f"wait_{wait_steps}_then_go"

    def action(self, state: RobotState, belief: Sequence[float]) -> str:
        del belief
        if self.pattern == "always_proceed":
            return "proceed"
        if self.pattern == "always_slow":
            return "slow"
        if self.pattern == "always_detour":
            return "detour" if state.x == 0 else "proceed"
        if self.pattern == "wait_then_go":
            return "wait" if state.x == 0 and state.t < self.wait_steps else "proceed"
        raise ValueError(self.pattern)


@dataclass(frozen=True)
class Metrics:
    expected_reward: float = 0.0
    success: float = 0.0
    collision: float = 0.0
    timeout: float = 0.0
    duration: float = 0.0
    success_duration: float = 0.0

    def scaled(self, weight: float) -> "Metrics":
        return Metrics(*(weight * value for value in self.__dict__.values()))

    def plus(self, other: "Metrics") -> "Metrics":
        return Metrics(*(a + b for a, b in zip(self.__dict__.values(), other.__dict__.values())))

    @property
    def conditional_success_time(self) -> Optional[float]:
        return self.success_duration / self.success if self.success > 0.0 else None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "expected_reward": self.expected_reward,
            "SR": self.success,
            "CR": self.collision,
            "TR": self.timeout,
            "expected_duration": self.duration,
            "conditional_success_time": self.conditional_success_time,
        }


def _terminal_metrics(model: LCornerModel, transition) -> Metrics:
    kwargs = {
        "expected_reward": model.transition_reward(transition),
        "duration": float(transition.elapsed),
    }
    kwargs[transition.outcome] = 1.0
    if transition.outcome == "success":
        kwargs["success_duration"] = float(transition.elapsed)
    return Metrics(**kwargs)


class ExactEvaluator:
    def __init__(self, model: LCornerModel, policy: BeliefPolicy):
        self.model = model
        self.policy = policy

    @lru_cache(maxsize=None)
    def _evaluate(self, state: RobotState, actual: Belief, internal: Belief) -> Metrics:
        internal = self.policy.projector(internal)
        action = self.policy.action(state, internal)
        total = Metrics()
        continuing = [0.0] * len(self.model.modes)
        next_state = None
        elapsed = None
        for mode_idx, probability in enumerate(actual):
            if probability <= 0.0:
                continue
            transition = self.model.transition(state, action, mode_idx)
            if transition.outcome is not None:
                terminal = _terminal_metrics(self.model, transition)
                # Terminal duration is absolute episode time, not just this action.
                absolute = state.t + transition.elapsed
                terminal = Metrics(
                    expected_reward=terminal.expected_reward,
                    success=terminal.success,
                    collision=terminal.collision,
                    timeout=terminal.timeout,
                    duration=float(absolute),
                    success_duration=float(absolute) if terminal.success else 0.0,
                )
                total = total.plus(terminal.scaled(probability))
            else:
                continuing[mode_idx] = probability
                if next_state is None:
                    next_state, elapsed = transition.next_state, transition.elapsed
                elif next_state != transition.next_state or elapsed != transition.elapsed:
                    raise AssertionError("observable continuation depends on mode")

        continuing_mass = sum(continuing)
        if continuing_mass <= 0.0:
            return total
        assert next_state is not None and elapsed is not None
        step_reward = -self.model.params.step_cost * elapsed
        internal_continuing = []
        for mode_idx, probability in enumerate(internal):
            transition = self.model.transition(state, action, mode_idx)
            internal_continuing.append(probability if transition.outcome is None else 0.0)

        informative = self.model.is_informative(state, action, next_state)
        observations = self.model.observation_alphabet(state, action, next_state)
        for observation in observations:
            actual_joint = tuple(
                continuing[i]
                * (self.model.observation_likelihood(next_state, i, observation)
                   if informative else float(observation == 0))
                for i in range(len(self.model.modes))
            )
            probability = sum(actual_joint)
            if probability <= 0.0:
                continue
            actual_updated = normalise(actual_joint)
            internal_joint = tuple(
                internal_continuing[i]
                * (self.model.observation_likelihood(next_state, i, observation)
                   if informative else float(observation == 0))
                for i in range(len(self.model.modes))
            )
            if sum(internal_joint) <= 0.0:
                raise AssertionError("actual observation has zero probability under internal model")
            internal_updated = self.policy.projector(normalise(internal_joint))
            branch = self._evaluate(next_state, actual_updated, internal_updated)
            branch = Metrics(
                expected_reward=step_reward + branch.expected_reward,
                success=branch.success,
                collision=branch.collision,
                timeout=branch.timeout,
                duration=branch.duration,
                success_duration=branch.success_duration,
            )
            total = total.plus(branch.scaled(probability))
        return total

    def evaluate(self, actual: Sequence[float], internal: Sequence[float]) -> Metrics:
        return self._evaluate(
            self.model.initial_state,
            normalise(actual),
            self.policy.projector(normalise(internal)),
        )


def evaluate_oracle(model: LCornerModel, prior: Sequence[float],
                    oracle: Optional[OracleSolver] = None) -> Metrics:
    oracle = oracle or OracleSolver(model)
    total = Metrics()
    for mode_idx, probability in enumerate(normalise(prior)):
        if probability <= 0.0:
            continue
        state = model.initial_state
        reward = 0.0
        while True:
            action = oracle.action(state, mode_idx)
            transition = model.transition(state, action, mode_idx)
            reward += model.transition_reward(transition)
            if transition.outcome is not None:
                absolute = state.t + transition.elapsed
                terminal = Metrics(
                    expected_reward=reward,
                    success=float(transition.outcome == "success"),
                    collision=float(transition.outcome == "collision"),
                    timeout=float(transition.outcome == "timeout"),
                    duration=float(absolute),
                    success_duration=float(absolute) if transition.outcome == "success" else 0.0,
                )
                total = total.plus(terminal.scaled(probability))
                break
            assert transition.next_state is not None
            state = transition.next_state
    return total


def fixed_candidates(model: LCornerModel):
    max_wait = max(0, model.params.horizon - math_ceil_div(model.params.goal_x, 2))
    candidates = [
        FixedSequencePolicy("always_proceed"),
        FixedSequencePolicy("always_slow"),
        FixedSequencePolicy("always_detour"),
    ]
    candidates.extend(FixedSequencePolicy("wait_then_go", wait) for wait in range(1, max_wait + 1))
    return tuple(candidates)


def math_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def select_best_fixed(results: Dict[str, Metrics]) -> Tuple[str, Metrics]:
    """Oracle envelope: maximise SR, then CR, TR and successful travel time."""
    def key(item):
        name, metrics = item
        success_time = metrics.conditional_success_time
        return (
            metrics.success,
            -metrics.collision,
            -metrics.timeout,
            -(success_time if success_time is not None else float("inf")),
            name,
        )
    return max(results.items(), key=key)
