from __future__ import annotations

import unittest

from .model import LCornerModel, ModelParams, RobotState, normalise
from .solver import (
    ExactBeliefSolver,
    ExactEvaluator,
    OracleSolver,
    SolverPolicy,
    evaluate_oracle,
)


class ExactGateTests(unittest.TestCase):
    def setUp(self):
        self.model = LCornerModel(ModelParams())
        self.prior = self.model.prior(0.6)

    def test_prior_and_sensor_likelihood_normalise(self):
        self.assertAlmostEqual(sum(self.prior), 1.0)
        state = self.model.initial_state
        transition = self.model.transition(state, "peek_right", 0)
        self.assertIsNotNone(transition.next_state)
        for mode_idx in range(len(self.model.modes)):
            total = sum(
                self.model.observation_likelihood(transition.next_state, mode_idx, obs)
                for obs in self.model.observation_alphabet(state, "peek_right", transition.next_state)
            )
            self.assertAlmostEqual(total, 1.0)

    def test_observation_depends_on_action(self):
        state = self.model.initial_state
        proceed = self.model.transition(state, "proceed", 1)
        peek = self.model.transition(state, "peek_right", 1)
        self.assertEqual(self.model.observation_alphabet(state, "proceed", proceed.next_state), (0,))
        self.assertGreater(len(self.model.observation_alphabet(state, "peek_right", peek.next_state)), 1)

    def test_bellman_value_matches_exact_evaluator(self):
        solver = ExactBeliefSolver(self.model, future_observations=True)
        policy = SolverPolicy("bayes_dual", solver)
        evaluated = ExactEvaluator(self.model, policy).evaluate(self.prior, self.prior)
        self.assertAlmostEqual(
            solver.value(self.model.initial_state, self.prior),
            evaluated.expected_reward,
            places=10,
        )
        self.assertAlmostEqual(evaluated.success + evaluated.collision + evaluated.timeout, 1.0)

    def test_information_cannot_reduce_optimal_planning_value(self):
        dual = ExactBeliefSolver(self.model, future_observations=True)
        openloop = ExactBeliefSolver(self.model, future_observations=False)
        self.assertGreaterEqual(
            dual.value(self.model.initial_state, self.prior) + 1e-12,
            openloop.value(self.model.initial_state, self.prior),
        )

    def test_oracle_upper_bounds_belief_value(self):
        dual = ExactBeliefSolver(self.model, future_observations=True)
        oracle = OracleSolver(self.model)
        oracle_value = sum(
            probability * oracle.value(self.model.initial_state, mode_idx)
            for mode_idx, probability in enumerate(self.prior)
        )
        self.assertGreaterEqual(
            oracle_value + 1e-12,
            dual.value(self.model.initial_state, self.prior),
        )
        evaluated = evaluate_oracle(self.model, self.prior, oracle)
        self.assertAlmostEqual(oracle_value, evaluated.expected_reward)
        self.assertAlmostEqual(evaluated.success + evaluated.collision + evaluated.timeout, 1.0)

    def test_normalise_rejects_zero_mass(self):
        with self.assertRaises(ValueError):
            normalise([0.0] * len(self.model.modes))


if __name__ == "__main__":
    unittest.main()

