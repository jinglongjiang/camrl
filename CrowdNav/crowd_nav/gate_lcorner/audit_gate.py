"""Independent red-team checks for the registered L-corner gate.

This module deliberately does not call ``ExactBeliefSolver._value`` when it
constructs the history-tree optimum.  It keeps probability mass over hidden
worlds, branches on raw observation histories, and exhaustively evaluates each
available root action.  The implementation is intentionally redundant with the
production solver: agreement is useful only if the two code paths can disagree.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import LCornerModel, ModelParams, RobotState, normalise
from .run_gate import evaluate_cell, load_registry, params_from_cell
from .solver import ExactBeliefSolver, ExactEvaluator, SolverPolicy


Belief = Tuple[float, ...]


@dataclass(frozen=True)
class AuditMetrics:
    expected_reward: float = 0.0
    success: float = 0.0
    collision: float = 0.0
    timeout: float = 0.0
    duration: float = 0.0
    success_duration: float = 0.0

    def scaled(self, weight: float) -> "AuditMetrics":
        return AuditMetrics(*(weight * value for value in asdict(self).values()))

    def plus(self, other: "AuditMetrics") -> "AuditMetrics":
        return AuditMetrics(*(
            left + right
            for left, right in zip(asdict(self).values(), asdict(other).values())
        ))

    def as_dict(self) -> dict:
        return {
            "expected_reward": self.expected_reward,
            "SR": self.success,
            "CR": self.collision,
            "TR": self.timeout,
            "expected_duration": self.duration,
            "conditional_success_time": (
                self.success_duration / self.success if self.success > 0.0 else None
            ),
        }


class HistoryTreeEnumerator:
    """Exhaustive history-policy tree enumerator using raw world mass.

    No production posterior, continuation helper, Bellman cache or policy
    evaluator is used.  The only shared object is the environment model whose
    transitions and sensor probabilities are themselves audited by traces.
    """

    def __init__(self, model: LCornerModel, *, future_observations: bool = True):
        self.model = model
        self.future_observations = future_observations
        self.expanded_nodes = 0

    def action_values(self, state: RobotState, belief: Sequence[float]) -> Dict[str, float]:
        weights = normalise(belief)
        return {
            action: self._action_value(state, weights, action)
            for action in self.model.available_actions(state)
        }

    def value_action(self, state: RobotState, belief: Sequence[float]) -> Tuple[float, str]:
        self.expanded_nodes += 1
        values = self.action_values(state, belief)
        best_action = ""
        best_value = float("-inf")
        for action in self.model.available_actions(state):
            value = values[action]
            if value > best_value + 1e-12:
                best_value, best_action = value, action
        if not best_action:
            raise AssertionError(f"no action at nonterminal state {state}")
        return best_value, best_action

    def _action_value(self, state: RobotState, weights: Belief, action: str) -> float:
        terminal_value = 0.0
        continuing = [0.0] * len(self.model.modes)
        next_state: Optional[RobotState] = None
        elapsed: Optional[int] = None

        for mode_idx, world_mass in enumerate(weights):
            if world_mass <= 0.0:
                continue
            transition = self.model.transition(state, action, mode_idx)
            if transition.outcome is not None:
                terminal_value += world_mass * self.model.transition_reward(transition)
                continue
            continuing[mode_idx] = world_mass
            if next_state is None:
                next_state, elapsed = transition.next_state, transition.elapsed
            elif next_state != transition.next_state or elapsed != transition.elapsed:
                raise AssertionError("observable state depends on the hidden mode")

        alive_mass = sum(continuing)
        if alive_mass <= 0.0:
            return terminal_value
        assert next_state is not None and elapsed is not None
        step_reward = -self.model.params.step_cost * elapsed

        informative = (
            self.future_observations
            and self.model.is_informative(state, action, next_state)
        )
        observations: Iterable[int]
        if informative:
            observations = self.model.observation_alphabet(state, action, next_state)
        else:
            observations = (0,)

        total = terminal_value
        for observation in observations:
            branch = []
            for mode_idx, world_mass in enumerate(continuing):
                likelihood = (
                    self.model.observation_likelihood(next_state, mode_idx, observation)
                    if informative else float(observation == 0)
                )
                branch.append(world_mass * likelihood)
            branch_mass = sum(branch)
            if branch_mass <= 0.0:
                continue
            child_value, _ = self.value_action(next_state, normalise(branch))
            total += branch_mass * (step_reward + child_value)
        return total


def _tiny_models() -> Tuple[Tuple[str, ModelParams], ...]:
    base = ModelParams(
        goal_x=4,
        intersection_x=2,
        horizon=4,
        distances=(1.5,),
        speeds=(0.5, 1.0),
        occupancy_steps=1,
        max_peek=1,
        peek_steps=1,
        visibility_depths=(3,),
        detour_steps=5,
        p_miss=0.10,
        p_false_alarm=0.05,
        success_reward=1.0,
        collision_cost=2.0,
        timeout_cost=1.0,
        step_cost=0.02,
    )
    return (
        ("noisy", base),
        ("perfect_sensor", replace(base, p_miss=0.0, p_false_alarm=0.0)),
        ("two_step_peek", replace(base, horizon=5, peek_steps=2, detour_steps=6)),
    )


def run_bruteforce(output: Path) -> dict:
    cases = []
    for model_name, params in _tiny_models():
        for p_exist in (0.2, 0.6, 1.0):
            model = LCornerModel(params)
            prior = model.prior(p_exist)
            for future_observations in (True, False):
                label = "dual" if future_observations else "openloop"
                brute = HistoryTreeEnumerator(
                    model, future_observations=future_observations
                )
                brute_q = brute.action_values(model.initial_state, prior)
                brute_value, brute_action = brute.value_action(model.initial_state, prior)
                solver = ExactBeliefSolver(
                    model, future_observations=future_observations
                )
                solver_q = {
                    action: solver._q_value(model.initial_state, prior, action)
                    for action in model.available_actions(model.initial_state)
                }
                solver_value = solver.value(model.initial_state, prior)
                solver_action = solver.action(model.initial_state, prior)

                q_error = max(abs(brute_q[action] - solver_q[action]) for action in brute_q)
                value_error = abs(brute_value - solver_value)
                passed = (
                    q_error <= 1e-11
                    and value_error <= 1e-11
                    and brute_action == solver_action
                )
                case = {
                    "model": model_name,
                    "p_exist": p_exist,
                    "planner": label,
                    "bruteforce_q": brute_q,
                    "solver_q": solver_q,
                    "bruteforce_value": brute_value,
                    "solver_value": solver_value,
                    "bruteforce_action": brute_action,
                    "solver_action": solver_action,
                    "max_root_q_abs_error": q_error,
                    "value_abs_error": value_error,
                    "history_nodes_expanded_without_cache": brute.expanded_nodes,
                    "pass": passed,
                }
                cases.append(case)
                print(
                    f"BRUTE {model_name:14s} p={p_exist:.1f} {label:8s} "
                    f"action={solver_action:10s} qerr={q_error:.3e} "
                    f"verr={value_error:.3e} {'PASS' if passed else 'FAIL'}",
                    flush=True,
                )

    report = {
        "method": "raw-history exhaustive action/observation tree; no solver recursion",
        "cases": cases,
        "max_root_q_abs_error": max(case["max_root_q_abs_error"] for case in cases),
        "max_value_abs_error": max(case["value_abs_error"] for case in cases),
        "pass": all(case["pass"] for case in cases),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["pass"]:
        raise SystemExit(2)
    return report


def _posterior_summary(model: LCornerModel, belief: Sequence[float]) -> dict:
    ranked = sorted(
        ((probability, model.modes[idx].name) for idx, probability in enumerate(belief)),
        reverse=True,
    )
    return {
        "p_exist": sum(belief[1:]),
        "top_modes": [
            {"mode": name, "probability": probability}
            for probability, name in ranked[:3]
        ],
    }


def _terminal_audit_metrics(model: LCornerModel, state: RobotState, transition) -> AuditMetrics:
    absolute_time = state.t + transition.elapsed
    return AuditMetrics(
        expected_reward=model.transition_reward(transition),
        success=float(transition.outcome == "success"),
        collision=float(transition.outcome == "collision"),
        timeout=float(transition.outcome == "timeout"),
        duration=float(absolute_time),
        success_duration=float(absolute_time) if transition.outcome == "success" else 0.0,
    )


def evaluate_policy_from_histories(
    model: LCornerModel,
    action_fn: Callable[[RobotState, Sequence[float]], str],
    state: RobotState,
    belief: Sequence[float],
) -> AuditMetrics:
    """Evaluate a policy by explicitly enumerating hidden worlds and observations."""
    belief = normalise(belief)
    action = action_fn(state, belief)
    total = AuditMetrics()
    continuing = [0.0] * len(model.modes)
    next_state: Optional[RobotState] = None
    elapsed: Optional[int] = None

    for mode_idx, probability in enumerate(belief):
        if probability <= 0.0:
            continue
        transition = model.transition(state, action, mode_idx)
        if transition.outcome is not None:
            total = total.plus(
                _terminal_audit_metrics(model, state, transition).scaled(probability)
            )
        else:
            continuing[mode_idx] = probability
            if next_state is None:
                next_state, elapsed = transition.next_state, transition.elapsed
            elif next_state != transition.next_state or elapsed != transition.elapsed:
                raise AssertionError("observable state depends on hidden mode")

    alive_mass = sum(continuing)
    if alive_mass <= 0.0:
        return total
    assert next_state is not None and elapsed is not None
    informative = model.is_informative(state, action, next_state)
    observations = (
        model.observation_alphabet(state, action, next_state) if informative else (0,)
    )
    step_reward = -model.params.step_cost * elapsed
    for observation in observations:
        branch = tuple(
            probability
            * (model.observation_likelihood(next_state, mode_idx, observation)
               if informative else float(observation == 0))
            for mode_idx, probability in enumerate(continuing)
        )
        branch_mass = sum(branch)
        if branch_mass <= 0.0:
            continue
        child = evaluate_policy_from_histories(
            model, action_fn, next_state, normalise(branch)
        )
        child = AuditMetrics(
            expected_reward=step_reward + child.expected_reward,
            success=child.success,
            collision=child.collision,
            timeout=child.timeout,
            duration=child.duration,
            success_duration=child.success_duration,
        )
        total = total.plus(child.scaled(branch_mass))
    return total


def _trace_tree(
    model: LCornerModel,
    action_fn: Callable[[RobotState, Sequence[float]], str],
    state: RobotState,
    belief: Sequence[float],
) -> dict:
    belief = normalise(belief)
    action = action_fn(state, belief)
    node = {
        "state_before_action": asdict(state),
        "belief_before_action": _posterior_summary(model, belief),
        "action": action,
        "terminal_before_observation": {},
        "observation_after_surviving_transition": [],
    }
    continuing = [0.0] * len(model.modes)
    terminal: Dict[str, float] = {}
    next_state: Optional[RobotState] = None
    for mode_idx, probability in enumerate(belief):
        if probability <= 0.0:
            continue
        transition = model.transition(state, action, mode_idx)
        if transition.outcome is not None:
            terminal[transition.outcome] = terminal.get(transition.outcome, 0.0) + probability
        else:
            continuing[mode_idx] = probability
            if next_state is None:
                next_state = transition.next_state
            elif next_state != transition.next_state:
                raise AssertionError("observable state depends on hidden mode")
    node["terminal_before_observation"] = terminal
    if sum(continuing) <= 0.0:
        return node
    assert next_state is not None
    informative = model.is_informative(state, action, next_state)
    observations = (
        model.observation_alphabet(state, action, next_state) if informative else (0,)
    )
    for observation in observations:
        joint = tuple(
            probability
            * (model.observation_likelihood(next_state, mode_idx, observation)
               if informative else float(observation == 0))
            for mode_idx, probability in enumerate(continuing)
        )
        probability = sum(joint)
        if probability <= 0.0:
            continue
        updated = normalise(joint)
        node["observation_after_surviving_transition"].append({
            "observation": observation,
            "unconditional_probability_at_node": probability,
            "state_after_action": asdict(next_state),
            "posterior": _posterior_summary(model, updated),
            "child": _trace_tree(model, action_fn, next_state, updated),
        })
    return node


def _representative_cells(registry: dict) -> Tuple[Tuple[str, dict], ...]:
    return (
        ("default", {
            "p_exist": 0.6, "speed_name": "nominal", "fov_name": "medium", "peek_steps": 1,
        }),
        ("low_prior", {
            "p_exist": 0.2, "speed_name": "nominal", "fov_name": "medium", "peek_steps": 1,
        }),
        ("fast_human", {
            "p_exist": 0.6, "speed_name": "fast", "fov_name": "medium", "peek_steps": 1,
        }),
        ("expensive_peek", {
            "p_exist": 0.6, "speed_name": "nominal", "fov_name": "medium", "peek_steps": 3,
        }),
    )


def run_traces(registry_path: Path, output: Path) -> dict:
    registry = load_registry(registry_path)
    reports = []
    max_metric_error = 0.0
    for label, spec in _representative_cells(registry):
        params = params_from_cell(registry, **spec)
        model = LCornerModel(params)
        prior = model.prior(spec["p_exist"])
        solver = ExactBeliefSolver(model, future_observations=True)
        policy = SolverPolicy("bayes_dual", solver)
        production = ExactEvaluator(model, policy).evaluate(prior, prior).as_dict()
        independent = evaluate_policy_from_histories(
            model, solver.action, model.initial_state, prior
        ).as_dict()
        metric_errors = {
            key: abs(independent[key] - production[key])
            for key in ("expected_reward", "SR", "CR", "TR", "expected_duration")
        }
        cell_max_error = max(metric_errors.values())
        max_metric_error = max(max_metric_error, cell_max_error)
        result = evaluate_cell(
            params,
            p_true=spec["p_exist"],
            p_assumed=spec["p_exist"],
            fixed_confidence=registry["fixed_confidence"],
        )
        report = {
            "label": label,
            "spec": spec,
            "root_action": solver.action(model.initial_state, prior),
            "production_metrics": production,
            "independent_history_metrics": independent,
            "metric_abs_errors": metric_errors,
            "best_fixed_policy": result["best_fixed_policy"],
            "all_arms": result["arms"],
            "event_order": (
                "action -> transition -> terminal/collision check -> "
                "observation only if alive -> posterior"
            ),
            "trace": _trace_tree(model, solver.action, model.initial_state, prior),
            "pass": cell_max_error <= 1e-11,
        }
        reports.append(report)
        print(
            f"TRACE {label:14s} action={report['root_action']:10s} "
            f"best_fixed={report['best_fixed_policy']:18s} "
            f"metric_error={cell_max_error:.3e} "
            f"{'PASS' if report['pass'] else 'FAIL'}",
            flush=True,
        )

    payload = {
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "max_metric_abs_error": max_metric_error,
        "pass": all(report["pass"] for report in reports),
        "representative_cells": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["pass"]:
        raise SystemExit(2)
    return payload


def _without_wall_seconds(value):
    if isinstance(value, dict):
        return {
            key: _without_wall_seconds(child)
            for key, child in value.items()
            if key != "wall_seconds"
        }
    if isinstance(value, list):
        return [_without_wall_seconds(child) for child in value]
    return value


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        _without_wall_seconds(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_difference(left, right, path: str = "root") -> Optional[str]:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(set(left) ^ set(right))} differ"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def compare_sweeps(reference_path: Path, candidate_path: Path, output: Path) -> dict:
    reference = _without_wall_seconds(json.loads(reference_path.read_text(encoding="utf-8")))
    candidate = _without_wall_seconds(json.loads(candidate_path.read_text(encoding="utf-8")))
    difference = _first_difference(reference, candidate)
    report = {
        "ignored_fields": ["wall_seconds"],
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_canonical_sha256": _canonical_digest(reference),
        "candidate_canonical_sha256": _canonical_digest(candidate),
        "first_difference": difference,
        "pass": difference is None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if difference is not None:
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    brute = subparsers.add_parser("bruteforce")
    brute.add_argument("--output", type=Path, required=True)

    traces = subparsers.add_parser("traces")
    traces.add_argument("--registry", type=Path, required=True)
    traces.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.stage == "bruteforce":
        run_bruteforce(args.output)
    elif args.stage == "traces":
        run_traces(args.registry, args.output)
    else:
        compare_sweeps(args.reference, args.candidate, args.output)


if __name__ == "__main__":
    main()
