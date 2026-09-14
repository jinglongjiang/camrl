"""Regression and causal-contract tests; no navigation-performance claims."""
import configparser
import dataclasses
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from crowd_nav.gdbn import GDBN, GDBNIntegration, PedestrianBeliefTracker
from crowd_nav.contracts import entities_to_tokens
from crowd_nav.policy.mamba_rl import (EnhancedSpatialEncoder, MambaRLPolicy,
    occlusion_checkpoint_meta, assert_checkpoint_compatible)
from crowd_nav.belief_mdp.model import BeliefMDPQNetwork
from crowd_nav.belief_mdp.hashing import sha256_dir
from crowd_sim.envs.occlusion_belief import OcclusionBelief
from crowd_sim.envs.utils.state import FullState, JointState

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def tracker(n=4, seed=11):
    model = GDBN(K=2)
    model.A = [np.eye(4), np.eye(4)]
    model.Q = [np.zeros((4, 4)), np.zeros((4, 4))]
    model.R = np.eye(4)
    model.Pi = np.eye(2)
    t = PedestrianBeliefTracker(model, n, np.random.default_rng(seed))
    t.reset(np.zeros(4))
    t.particles_x[:] = 0
    t.particles_s[:] = np.arange(n) >= n // 2
    return t


def config():
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    for name in ('env.config', 'policy.config', 'train.config'):
        cfg.read(str(ROOT / 'configs' / name))
    cfg.set('mamba', 'temporal_backbone', 'mlp')
    return cfg


class RepairTests(unittest.TestCase):
    def test_equal_likelihood_preserves_nonuniform_weight(self):
        t = tracker()
        t.weights[:] = [.4, .3, .2, .1]
        t.step(np.zeros(4))
        np.testing.assert_allclose(t.weights, [.4, .3, .2, .1], atol=1e-12)
        np.testing.assert_allclose(t.get_mode_distribution(), [.7, .3], atol=1e-9)

    def test_two_evidence_steps_multiply_odds(self):
        t = tracker()
        t.particles_x[2:, 0] = 1.
        for step in (1, 2):
            t.step(np.zeros(4))
            p = t.get_mode_distribution()
            self.assertAlmostEqual(p[0] / p[1], np.exp(.5 * step), places=10)

    def test_transition_is_not_frozen_by_uninformative_observation(self):
        t = tracker()
        t.weights[:] = [.4, .3, .2, .1]
        t.gdbn.Pi = np.array([[0., 1.], [1., 0.]])
        t.step(np.zeros(4))
        np.testing.assert_allclose(t.get_mode_distribution(), [.3, .7], atol=1e-9)

    def test_resampling_is_unbiased_in_ensemble(self):
        estimates = []
        weights = np.full(100, .2 / 99)
        weights[0] = .8
        expected = weights[:50].sum()
        for seed in range(200):
            t = tracker(100, seed)
            t.weights[:] = weights
            t.step(np.zeros(4))
            np.testing.assert_allclose(t.weights, .01)
            estimates.append(t.get_mode_distribution()[0])
        self.assertLess(abs(np.mean(estimates) - expected), .002)

    def test_coordinates_and_legacy_contract(self):
        robot = FullState(2., 3., 0., 0., .3, 5., 7., 1., 0.)
        entity = dict(px=3., py=3., vx=0., vy=0., radius=.3,
                      visible=1., hidden=0., p_exist=1., uncertainty=0.)
        token = entities_to_tokens(robot, [entity], token_contract='belief_v3')
        for contract, expected in [('relative_v2', [1., 0.]), ('legacy_v1', [-1., -3.])]:
            model = EnhancedSpatialEncoder(64, contract).eval()
            captured = []
            hook = model.human_encoder.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().clone()))
            model(torch.from_numpy(token)[None, None])
            hook.remove()
            np.testing.assert_allclose(captured[0][0, 13:15].numpy(), expected)

    def test_random_and_greedy_both_append_once(self):
        policy = MambaRLPolicy(config(), 'cpu')
        policy.set_phase('train')
        state = JointState(FullState(0., 0., 0., 0., .3, 0., 4., 1., 0.), [])
        for i, epsilon in enumerate([1., 1., 0., 1., 0., 0.]):
            policy.epsilon = epsilon
            policy.predict_sarl_style(state)
            self.assertEqual(len(policy._history), i + 1)

    def test_snapshot_read_is_immutable_and_idempotent(self):
        filt = OcclusionBelief(mode='bayes')
        robot = FullState(0., 0., 0., 0., .3, 0., 4., 1., 0.)
        # Controlled sensor input isolates update vs read, without hidden truth.
        def sensor(robot_xy, humans):
            filt._mesh = (np.array([[1.]]), np.array([[0.]]))
            return np.zeros((1, 1)), np.zeros((1, 1)), np.full((1, 1), .5), [], []
        filt._label_and_sensor = sensor
        filt.logodds = np.array([[np.log(.8 / .2)]])
        filt._prev_modes = [[.75, 0., .8, .25]]
        filt.update(robot, [])
        snapshot = filt.get_belief_snapshot()
        self.assertEqual(snapshot.entities[0].vx, 1.)
        def state_hash():
            data = {k: v for k, v in filt.__dict__.items() if not callable(v)}
            return hashlib.sha256(pickle.dumps(data)).hexdigest()
        before = state_hash()
        for _ in range(10):
            self.assertEqual(filt.get_belief_snapshot(), snapshot)
            entities = filt.policy_entities()
            self.assertEqual(entities[0]['vx'], 1.)
            entities[0]['vx'] = 999.
        self.assertEqual(state_hash(), before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.entities[0].vx = 99.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.frame_index = 99

    def test_missing_and_corrupt_parameters_fail_fast(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = str(Path(temp) / 'missing')
            with self.assertRaises(FileNotFoundError):
                GDBNIntegration(params_dir=missing)
            with self.assertRaises(FileNotFoundError):
                sha256_dir(missing)
            with self.assertRaises(ValueError):
                sha256_dir(temp)
            with self.assertRaises(FileNotFoundError):
                GDBNIntegration(params_dir=temp)
            f = GDBNIntegration(params_dir=missing, allow_unfitted=True)
            for call in (lambda: f.update(np.zeros(34)), f.get_belief_features,
                         f.get_klda_risk, f.get_all_klda,
                         f.get_belief_snapshot, lambda: f.predict_action_rollout_batch(
                             np.zeros((1, 34)), np.zeros((1, 2)))):
                with self.assertRaises(RuntimeError):
                    call()

    def test_old_checkpoint_metadata_is_rejected(self):
        cfg = config()
        meta = occlusion_checkpoint_meta(cfg, 'bayes', 'mlp')
        old = {k: v for k, v in meta.items() if k not in (
            'coordinate_contract', 'belief_read_contract', 'history_contract')}
        with self.assertRaises(ValueError):
            assert_checkpoint_compatible(old, 'bayes', 'mlp', meta)
        assert_checkpoint_compatible(meta, 'bayes', 'mlp', meta)

    def test_belief_can_affect_only_the_existing_risk_head(self):
        torch.manual_seed(23)
        net = BeliefMDPQNetwork(8, 11, 12)
        # Test connectivity after a nonconstant risk head; initialization is constant by design.
        torch.nn.init.normal_(net.q_c_head[2].weight, std=.2)
        context = torch.randn(1, 8)
        candidate = torch.randn(1, 4, 12)
        belief = torch.tensor([[.7, .3, 0., .7, .3, 0., .4, .4, .1, .1, 1.]])
        changed = belief.clone()
        changed[:, :2] = torch.tensor([.3, .7])
        a, aa = net(context, belief, candidate, True)
        b, bb = net(context, changed, candidate, True)
        self.assertTrue(torch.equal(aa['q_r'], bb['q_r']))
        self.assertGreater(float((a - b).abs().max()), 1e-7)
        self.assertGreater(float((aa['q_c'] - bb['q_c']).abs().max()), 1e-7)


if __name__ == '__main__':
    unittest.main()
