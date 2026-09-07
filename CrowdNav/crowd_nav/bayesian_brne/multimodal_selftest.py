"""Focused tests for the MM-S1 base-only experiment."""

from __future__ import annotations

import argparse

import numpy as np

from crowd_nav.bayesian_brne.causal_response_arhmm import CausalFitConfig, fit
from crowd_nav.bayesian_brne.causal_selftest import _synthetic_sequences
from crowd_nav.bayesian_brne.multimodal_evaluate import _pilot_gate
from crowd_nav.bayesian_brne.multimodal_s1_pipeline import (
    _assert_base_only,
    reference_only_sequences,
)
from crowd_nav.bayesian_brne.nonreciprocal_policy import (
    robot_only_expected_cost_weights,
)


def run_tests() -> None:
    assertions = 0
    original = _synthetic_sequences(response=True, n_sequences=6, length=12)
    base = reference_only_sequences(original)
    for source, transformed in zip(original, base):
        assert np.array_equal(transformed.next_velocity_a, source.next_velocity_ref)
        assert np.array_equal(transformed.next_velocity_b, source.next_velocity_ref)
        assert np.array_equal(transformed.action_a, source.robot_velocity)
        assert np.array_equal(transformed.action_b, source.robot_velocity)
        assert not np.shares_memory(transformed.next_velocity_ref, source.next_velocity_ref)
        assertions += 5

    artifact = fit(
        base,
        2,
        CausalFitConfig(em_max_iters=12, restarts=1, response_ridge=0.05, seed=29),
    )
    _assert_base_only(artifact)
    assert float(np.max(np.abs(artifact.D))) <= 1e-12
    assertions += 2

    records = []
    for seed in range(5):
        for episode in range(3):
            for method, clearance, steps in (
                ("full_posterior_selected", 0.8, 40),
                ("posterior_mean_selected", 0.5, 45),
                ("full_posterior_k1", 0.4, 48),
            ):
                records.append({
                    "method": method,
                    "suite_seed": seed,
                    "episode_index": episode,
                    "split": "test_heldout_interactive",
                    "outcome": "success",
                    "minimum_clearance": clearance,
                    "steps": steps,
                })
    registry = {"gate": {"bootstrap_seed": 31, "bootstrap_replicates": 500}}
    assert _pilot_gate(records, registry)["status"] == "PASS"
    damaged = [dict(row) for row in records]
    for row in damaged:
        if row["method"] == "full_posterior_selected":
            row["outcome"] = "collision"
    assert _pilot_gate(damaged, registry)["status"] == "FAIL"
    assertions += 2

    robot = np.array([
        [[0.5, 0.0], [1.0, 0.0]],
        [[0.5, 1.0], [1.0, 1.0]],
    ])
    humans = np.zeros((1, 2, 4, 2, 2))
    humans[0, :, :, :, 0] = np.array([0.5, 1.0])[None, None, :]
    weights, costs = robot_only_expected_cost_weights(
        robot, humans, 0.3, np.array([0.3]),
        safe_distance=0.2, cost_sigma=0.1, cost_scale=100.0,
    )
    assert weights.shape == (2,)
    assert costs.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert costs[0] > costs[1]
    assert weights[1] > weights[0]
    assertions += 5
    print(f"MM-S1 selftest PASS: {assertions} assertions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_tests()


if __name__ == "__main__":
    main()
