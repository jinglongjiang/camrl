"""No training: masks, causal geometry, adjacent-history control and density wiring."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from crowd_nav.gdbn import GDBNIntegration
from crowd_nav.belief_mdp.runtime import BeliefMDPFeatureEngine, FullCrowdNavigationEnvironment, SIX_SCENARIOS
from crowd_nav.belief_mdp.model import BeliefMDPQNetwork
from crowd_nav.belief_mdp.train import build_parser
from crowd_nav.belief_mdp.evaluate import resolve_params, PARAMS_FROM_CHECKPOINT
from crowd_nav.belief_mdp.protocol import FEATURE_CONTRACT, validate_teacher_receipt
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.test_belief_contract_repair import config
from crowd_nav.belief_mdp.runtime import sort_humans_by_ttc
from crowd_sim.envs.utils.state import JointState

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / 'repair_results/params'


class DensityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        cls.policy = MambaRLPolicy(config(), 'cpu').eval()

    def engine(self, count, mode='recursive'):
        return BeliefMDPFeatureEngine(self.policy, str(PARAMS), torch.device('cpu'),
                                     num_humans=count, belief_mode=mode, n_particles=12, seed=31)

    def test_padding_does_not_enter_summary(self):
        engine = self.engine(20)
        belief = np.tile([1/3, 1/3, 1/3, 1., 0.], (20, 1)).astype(np.float32)
        belief[:5] = [1., 0., 0., 0., .5]
        mask = np.arange(20) < 5
        result = engine._global_belief(belief, mask)
        np.testing.assert_allclose(result, [1, 0, 0, 1, 0, 0, 0, 0, .5, .5, .25])
        belief[5:] = 99
        np.testing.assert_array_equal(result, engine._global_belief(belief, mask))
        np.testing.assert_array_equal(engine._global_belief(belief, mask & False), np.zeros(11))

    def test_all_scenarios_same_network_no_truncation(self):
        network = None
        for scenario, (_, _, count) in SIX_SCENARIOS.items():
            environment = FullCrowdNavigationEnvironment(
                str(ROOT / 'crowd_nav/configs/env_belief_mdp.config'), scenario, robot_visible=False)
            robot, humans = environment.reset(seed=72031, profile='nominal', test_case=31)
            self.assertEqual(len(humans), count)
            engine = self.engine(count)
            features = engine.encode(robot, humans)
            self.assertEqual(sum(engine.filter.get_belief_snapshot().valid_mask), count)
            self.assertEqual(features.belief.shape, (11,))
            # Same weights, not one network per population.
            if network is None:
                network = BeliefMDPQNetwork(features.context.size, 11, 12)
            with torch.no_grad():
                values = network(torch.tensor(features.context)[None],
                                 torch.tensor(features.belief)[None],
                                 torch.tensor(features.candidates)[None])
            self.assertEqual(tuple(values.shape), (1, 80))
            self.assertTrue(torch.isfinite(values).all())
            if count > 5:
                with self.assertRaises(ValueError):
                    self.engine(5).encode(robot, humans)
        padded = self.engine(20)
        padded.encode(robot, humans[:5])
        self.assertEqual(sum(padded.filter.get_belief_snapshot().valid_mask), 5)

    def test_zero_action_dynamics_keeps_robot_geometry(self):
        engine = self.engine(1)
        self.assertTrue(all(np.count_nonzero(b) == 0 for b in engine.filter.B_action))
        state = np.array([0, 0, 0, 0, .3, 5, 0, 1, 0, 1, 0, 0, 0, .3], dtype=np.float32)
        engine.filter.update(state, [True])
        belief = np.asarray(engine.filter.get_belief_snapshot().features)
        risk = engine._risk_features(state, np.array([[1., 0.], [-1., 0.]]), belief)
        self.assertGreater(float(np.max(np.abs(risk[0] - risk[1]))), 1e-6)

    def test_mlp_teacher_score_wiring_not_mamba_certification(self):
        policy = MambaRLPolicy(config(), 'cpu').eval()
        policy.set_phase('test')
        policy.build_action_space(1.)
        policy.capture_lookahead_scores = True
        policy.test_action_smoothing = 0.
        policy.test_min_clearance = .2
        policy.test_risk_lambda = .1
        engine = BeliefMDPFeatureEngine(policy, str(PARAMS), torch.device('cpu'), num_humans=5)
        env = FullCrowdNavigationEnvironment(str(ROOT / 'crowd_nav/configs/env_belief_mdp.config'),
                                            'baseline_circle', robot_visible=False)
        robot, humans = env.reset(seed=72039, profile='nominal', test_case=39)
        for _ in range(8):
            engine.encode(robot, humans)
            expected = engine.teacher_scores(robot, humans)
            state = JointState(robot.get_full_state(),
                               [h.get_observable_state() for h in sort_humans_by_ttc(robot, humans)[:5]])
            action = policy.predict_sarl_style(state)
            np.testing.assert_allclose(expected, policy._last_lookahead_scores.cpu().numpy(), atol=1e-4, rtol=1e-6)
            result = env.step(action)
            if result.done:
                break
            robot, humans = env.robot, list(env.env.humans)

    def test_frame_only_forgets_earlier_particles_and_missing_breaks_history(self):
        a = GDBNIntegration(params_dir=str(PARAMS), max_peds=1, n_particles=12, history_mode='frame_only')
        b = GDBNIntegration(params_dir=str(PARAMS), max_peds=1, n_particles=12, history_mode='frame_only')
        def obs(x):
            return np.array([0, 0, 0, 0, .3, 5, 0, 1, 0, x, 1, .2, 0, .3], dtype=float)
        for x in [-100, -50, -25]:
            a.update(obs(x), [True])
        for x in [100, 50, 25]:
            b.update(obs(x), [True])
        for x in [1., 1.1]:
            a.update(obs(x), [True])
            b.update(obs(x), [True])
        np.testing.assert_array_equal(a.get_per_ped_belief_vec(), b.get_per_ped_belief_vec())
        a.update(obs(1.2), [False])
        self.assertFalse(a._previous_observations)
        a.update(obs(1.3), [True])
        np.testing.assert_allclose(a.get_per_ped_belief_vec()[0, :3], np.full(3, 1/3), atol=1e-6)
        a.reset()
        self.assertFalse(a._previous_observations)

    def test_training_fixed_five_and_three_arms(self):
        parser = build_parser()
        for mode in ('no_belief', 'frame_only', 'recursive'):
            args = parser.parse_args(['--teacher_verification', 'receipt.json', '--belief_mode', mode])
            self.assertEqual(args.num_humans, 5)
        with self.assertRaises(SystemExit):
            parser.parse_args(['--teacher_verification', 'receipt.json', '--num_humans', '20'])

    def test_eval_population_not_checkpoint_feature(self):
        args = SimpleNamespace(**{k: None for k in PARAMS_FROM_CHECKPOINT}, num_humans=None)
        saved = dict(feature_contract=FEATURE_CONTRACT, train_num_humans=5, num_humans=5)
        self.assertEqual(resolve_params(args, saved, False)['train_num_humans'], 5)
        self.assertNotIn('num_humans', PARAMS_FROM_CHECKPOINT)
        with self.assertRaises(SystemExit):
            resolve_params(args, {}, False)

    def test_teacher_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            receipt = dict(passed=True, states=100, tensor_mismatches=0, value_mismatches=0, score_mismatches=0,
                           top1_mismatches=0, fingerprint={'test': 'new'})
            with patch('crowd_nav.belief_mdp.protocol.teacher_fingerprint', return_value={'test': 'new'}):
                path.write_text(json.dumps(receipt))
                validate_teacher_receipt(path, None)
                for key, value in [('states', 99), ('passed', False), ('fingerprint', {'test': 'old'}),
                                   ('value_mismatches', 1)]:
                    invalid = dict(receipt, **{key: value})
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        validate_teacher_receipt(path, None)


if __name__ == '__main__':
    unittest.main()
