"""Focused D1-D3 tests for the decision-oracle audit layer."""

from __future__ import annotations

import os

import numpy as np

from crowd_nav.bayesian_brne.decision_oracle import (
    _integrate_actions,
    clone_env,
    make_candidate_sets,
    rollout_candidate,
)
from crowd_nav.bayesian_brne.interaction_protocol import make_scenario


def run() -> None:
    brne_root = os.environ.get("SM_BRNE_UPSTREAM_ROOT", "/home/abc/temp/brne")
    env = make_scenario("baseline_circle", "test_heldout_interactive", np.random.default_rng(8911), 0.25)
    snapshot = clone_env(env)
    sets_a = make_candidate_sets(env, horizon=4, num_legacy=8, seed=9911, brne_root=brne_root)
    sets_b = make_candidate_sets(snapshot, horizon=4, num_legacy=8, seed=9911, brne_root=brne_root)
    assert np.array_equal(sets_a["legacy_candidates"].actions, sets_b["legacy_candidates"].actions)
    assert np.array_equal(sets_a["structured_candidates"].actions, sets_b["structured_candidates"].actions)
    assert len(sets_a["structured_candidates"].actions) > len(sets_a["legacy_candidates"].actions)
    assert len(sets_a["dense_oracle_candidates"].actions) > len(sets_a["structured_candidates"].actions)

    structured = sets_a["structured_candidates"]
    structured.validate(dt=env.dt, max_speed=env.config.max_human_speed, max_acceleration=env.config.max_human_acceleration)
    branch_a = rollout_candidate(env, structured.actions[0], safe_distance=0.2)
    branch_b = rollout_candidate(snapshot, structured.actions[0], safe_distance=0.2)
    assert branch_a == branch_b

    trap_actions = np.array([[[0.8, 0.0], [0.8, 0.0]], [[0.0, 0.8], [0.0, 0.8]]], dtype=float)
    trap = _integrate_actions(env, trap_actions, name="trap")
    assert trap.actions.shape == (2, 2, 2)
    assert np.all(np.isfinite(trap.future_positions))
    assert np.allclose(trap.state_positions[:, 0], env.robot_pos)
    print("DECISION-ORACLE selftest PASS: 10 assertions")


if __name__ == "__main__":
    run()
