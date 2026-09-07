"""CLI for Phase 2A/2B/2C of the L-corner falsification gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import time
from typing import Dict

from .model import LCornerModel, ModelParams, project_existence
from .solver import (
    ExactBeliefSolver,
    ExactEvaluator,
    FixedSequencePolicy,
    MapAdaptivePolicy,
    Metrics,
    OracleSolver,
    SolverPolicy,
    evaluate_oracle,
    fixed_candidates,
    select_best_fixed,
)


HERE = Path(__file__).resolve().parent


def load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("status") != "frozen_before_2b":
        raise ValueError("registered grid must be frozen before running 2B")
    return registry


def params_from_cell(registry: dict, *, p_exist: float, speed_name: str,
                     fov_name: str, peek_steps: int) -> ModelParams:
    base = registry["base_model"]
    del p_exist  # prior parameter, not a ModelParams field
    return ModelParams(
        goal_x=base["goal_x"],
        intersection_x=base["intersection_x"],
        horizon=base["horizon"],
        distances=tuple(base["distances"]),
        speeds=tuple(registry["speed_pairs"][speed_name]),
        occupancy_steps=base["occupancy_steps"],
        max_peek=base["max_peek"],
        peek_steps=peek_steps,
        visibility_depths=tuple(registry["fov_depths"][fov_name]),
        detour_steps=base["detour_steps"],
        p_miss=base["p_miss"],
        p_false_alarm=base["p_false_alarm"],
        success_reward=base["success_reward"],
        collision_cost=base["collision_cost"],
        timeout_cost=base["timeout_cost"],
        step_cost=base["step_cost"],
    )


def evaluate_cell(params: ModelParams, p_true: float, p_assumed: float,
                  fixed_confidence: float) -> dict:
    started = time.perf_counter()
    model = LCornerModel(params)
    actual_prior = model.prior(p_true)
    internal_prior = model.prior(p_assumed)
    oracle_solver = OracleSolver(model)

    dual_solver = ExactBeliefSolver(model, future_observations=True)
    open_solver = ExactBeliefSolver(model, future_observations=False)
    fixed_projector = lambda belief: project_existence(belief, fixed_confidence)
    fixed_solver = ExactBeliefSolver(
        model, future_observations=True, projector=fixed_projector
    )

    policies = {
        "bayes_dual": SolverPolicy("bayes_dual", dual_solver),
        "bayes_openloop": SolverPolicy("bayes_openloop", open_solver),
        "map_adaptive": MapAdaptivePolicy(oracle_solver),
        "fixed_conf": SolverPolicy("fixed_conf", fixed_solver),
    }
    metrics: Dict[str, Metrics] = {
        name: ExactEvaluator(model, policy).evaluate(actual_prior, internal_prior)
        for name, policy in policies.items()
    }
    metrics["oracle"] = evaluate_oracle(model, actual_prior, oracle_solver)

    fixed_results = {}
    for policy in fixed_candidates(model):
        fixed_results[policy.name] = ExactEvaluator(model, policy).evaluate(
            actual_prior, internal_prior
        )
    best_fixed_name, best_fixed_metrics = select_best_fixed(fixed_results)
    metrics["best_fixed"] = best_fixed_metrics

    # Under a matched prior, exact policy evaluation must reproduce Bellman V.
    if abs(p_true - p_assumed) < 1e-15:
        bellman = dual_solver.value(model.initial_state, internal_prior)
        if abs(bellman - metrics["bayes_dual"].expected_reward) > 1e-9:
            raise AssertionError((bellman, metrics["bayes_dual"].expected_reward))

    return {
        "params": asdict(params),
        "p_true": p_true,
        "p_assumed": p_assumed,
        "fixed_confidence": fixed_confidence,
        "best_fixed_policy": best_fixed_name,
        "arms": {name: value.as_dict() for name, value in metrics.items()},
        "fixed_candidates": {name: value.as_dict() for name, value in fixed_results.items()},
        "solver": {
            "dual_cached_beliefs": dual_solver.cached_states,
            "openloop_cached_beliefs": open_solver.cached_states,
            "fixed_cached_beliefs": fixed_solver.cached_states,
            "wall_seconds": time.perf_counter() - started,
        },
    }


def print_cell(result: dict) -> None:
    print(
        f"p_true={result['p_true']:.3f} p_assumed={result['p_assumed']:.3f} "
        f"best_fixed={result['best_fixed_policy']} wall={result['solver']['wall_seconds']:.3f}s"
    )
    for arm in ("oracle", "bayes_dual", "bayes_openloop", "map_adaptive", "fixed_conf", "best_fixed"):
        value = result["arms"][arm]
        print(
            f"  {arm:16s} SR={value['SR']:.4f} CR={value['CR']:.4f} "
            f"TR={value['TR']:.4f} Tsucc={value['conditional_success_time']} "
            f"J={value['expected_reward']:.4f}"
        )


def single_gate(result: dict) -> dict:
    arms = result["arms"]
    oracle_gap = arms["oracle"]["SR"] - arms["best_fixed"]["SR"]
    return {
        "oracle_minus_best_fixed_SR": oracle_gap,
        "pass_2b_oracle_gap": oracle_gap >= 0.10 - 1e-12,
        "dual_minus_openloop_SR": arms["bayes_dual"]["SR"] - arms["bayes_openloop"]["SR"],
        "dual_minus_map_SR": arms["bayes_dual"]["SR"] - arms["map_adaptive"]["SR"],
    }


def run_single(registry: dict, output: Path) -> int:
    default = registry["default_cell"]
    params = params_from_cell(registry, **default)
    result = evaluate_cell(
        params,
        p_true=default["p_exist"],
        p_assumed=default["p_exist"],
        fixed_confidence=registry["fixed_confidence"],
    )
    result["gate_2b"] = single_gate(result)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "single_cell.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print_cell(result)
    print("GATE_2B=" + ("PASS" if result["gate_2b"]["pass_2b_oracle_gap"] else "FAIL"))
    return 0 if result["gate_2b"]["pass_2b_oracle_gap"] else 2


def connected_fraction(passing, shape) -> float:
    if not passing:
        return 0.0
    unseen = set(passing)
    largest = 0
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for axis in range(len(shape)):
                for delta in (-1, 1):
                    other = list(node)
                    other[axis] += delta
                    other = tuple(other)
                    if 0 <= other[axis] < shape[axis] and other in unseen:
                        unseen.remove(other)
                        stack.append(other)
        largest = max(largest, size)
    return largest / math_prod(shape)


def math_prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def run_sweep(registry: dict, output: Path) -> int:
    single_path = output / "single_cell.json"
    if not single_path.exists():
        raise SystemExit("run --stage single first")
    with single_path.open(encoding="utf-8") as handle:
        if not json.load(handle)["gate_2b"]["pass_2b_oracle_gap"]:
            raise SystemExit("2B failed; registered sweep is forbidden")

    axes = registry["grid"]
    speed_names = list(registry["speed_pairs"])
    fov_names = list(registry["fov_depths"])
    cells = []
    passing_g2 = set()
    shape = (len(axes["p_exist"]), len(speed_names), len(fov_names), len(axes["peek_steps"]))
    for idx, values in enumerate(itertools.product(
        enumerate(axes["p_exist"]),
        enumerate(speed_names),
        enumerate(fov_names),
        enumerate(axes["peek_steps"]),
    )):
        (ip, p_exist), (iv, speed_name), (ifov, fov_name), (ic, peek_steps) = values
        params = params_from_cell(
            registry,
            p_exist=p_exist,
            speed_name=speed_name,
            fov_name=fov_name,
            peek_steps=peek_steps,
        )
        result = evaluate_cell(
            params, p_true=p_exist, p_assumed=p_exist,
            fixed_confidence=registry["fixed_confidence"]
        )
        arms = result["arms"]
        result["index"] = [ip, iv, ifov, ic]
        result["labels"] = {
            "speed": speed_name, "fov": fov_name, "peek_steps": peek_steps
        }
        result["gaps"] = {
            "oracle_best_fixed": arms["oracle"]["SR"] - arms["best_fixed"]["SR"],
            "dual_openloop": arms["bayes_dual"]["SR"] - arms["bayes_openloop"]["SR"],
            "dual_map": arms["bayes_dual"]["SR"] - arms["map_adaptive"]["SR"],
        }
        result["passes"] = {
            "g1_cell": result["gaps"]["oracle_best_fixed"] >= 0.10 - 1e-12,
            "g2_cell": result["gaps"]["dual_openloop"] >= 0.03 - 1e-12
                       and result["gaps"]["dual_map"] >= 0.03 - 1e-12,
            "g3_cell": arms["bayes_dual"]["SR"] > arms["best_fixed"]["SR"] + 1e-12
                       and arms["bayes_dual"]["CR"] <= arms["best_fixed"]["CR"] + 1e-12
                       and arms["bayes_dual"]["TR"] <= arms["best_fixed"]["TR"] + 1e-12,
        }
        if result["passes"]["g2_cell"]:
            passing_g2.add((ip, iv, ifov, ic))
        cells.append(result)
        print(f"CELL {idx + 1:03d}/{math_prod(shape)}", flush=True)

    g1_fraction = sum(cell["passes"]["g1_cell"] for cell in cells) / len(cells)
    g2_fraction = sum(cell["passes"]["g2_cell"] for cell in cells) / len(cells)
    g3_fraction = sum(cell["passes"]["g3_cell"] for cell in cells) / len(cells)
    connected = connected_fraction(passing_g2, shape)
    summary = {
        "cells": len(cells),
        "G1": {"fraction": g1_fraction, "pass": g1_fraction >= 0.60},
        "G2": {"fraction": g2_fraction},
        "G3": {"fraction": g3_fraction},
        "G4": {"largest_connected_fraction": connected, "pass": connected >= 0.40},
        "G5": {"pass": None, "reason": "prior-mismatch stage not run yet"},
        "G6": {"pass": None, "reason": "computed after G1-G4 audit"},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "registered_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "cells": cells}, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("single", "sweep"), required=True)
    parser.add_argument("--registry", type=Path, default=HERE / "registered_grid.json")
    parser.add_argument("--output", type=Path, default=Path("/home/abc/temp/lcorner_gate"))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if args.stage == "single":
        raise SystemExit(run_single(registry, args.output))
    raise SystemExit(run_sweep(registry, args.output))


if __name__ == "__main__":
    main()

