from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from .audit_gate import HistoryTreeEnumerator, compare_sweeps
from .model import LCornerModel
from .audit_gate import _tiny_models
from .solver import ExactBeliefSolver


class IndependentHistoryTreeTests(unittest.TestCase):
    def test_tiny_history_trees_match_every_root_q(self):
        for model_name, params in _tiny_models():
            model = LCornerModel(params)
            for p_exist in (0.2, 0.6, 1.0):
                prior = model.prior(p_exist)
                for future_observations in (True, False):
                    with self.subTest(
                        model=model_name,
                        p_exist=p_exist,
                        future_observations=future_observations,
                    ):
                        brute = HistoryTreeEnumerator(
                            model, future_observations=future_observations
                        )
                        brute_q = brute.action_values(model.initial_state, prior)
                        solver = ExactBeliefSolver(
                            model, future_observations=future_observations
                        )
                        for action, brute_value in brute_q.items():
                            self.assertAlmostEqual(
                                brute_value,
                                solver._q_value(model.initial_state, prior, action),
                                places=11,
                            )
                        brute_value, brute_action = brute.value_action(
                            model.initial_state, prior
                        )
                        self.assertAlmostEqual(
                            brute_value,
                            solver.value(model.initial_state, prior),
                            places=11,
                        )
                        self.assertEqual(
                            brute_action, solver.action(model.initial_state, prior)
                        )

    def test_sweep_comparison_ignores_only_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            output = root / "report.json"
            reference.write_text(
                '{"solver":{"wall_seconds":1.0,"cached":2},"SR":0.5}',
                encoding="utf-8",
            )
            candidate.write_text(
                '{"solver":{"wall_seconds":9.0,"cached":2},"SR":0.5}',
                encoding="utf-8",
            )
            report = compare_sweeps(reference, candidate, output)
            self.assertTrue(report["pass"])


if __name__ == "__main__":
    unittest.main()
