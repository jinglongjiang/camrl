import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from crowd_nav.bayes_continuous.environment import BeliefEnv, project_orca, transform_observation
from crowd_nav.bayes_continuous.network import SetEncoder

PARAMS = Path(__file__).resolve().parents[2] / 'repair_results/params'


class Contracts(unittest.TestCase):
    def test_no_mamba_import(self):
        self.assertFalse(any(name.startswith('crowd_nav.policy.mamba') for name in sys.modules))

    def test_gym_and_continuous_executor(self):
        env = BeliefEnv(PARAMS)
        check_env(env, warn=True)
        obs, _ = env.reset(seed=77)
        self.assertEqual(env.world.robot.kinematics, 'unicycle')
        self.assertTrue(all(h.kinematics == 'holonomic' for h in env.world.env.humans))
        self.assertTrue(all(np.count_nonzero(b) == 0 for b in env.filter.B_action))
        heading = env.world.robot.theta
        env.step(np.array([.2, .731], np.float32))
        expected = (heading+.731*env.world.env.time_step) % (2*np.pi)
        self.assertAlmostEqual(env.world.robot.theta, expected, places=6)
        with self.assertRaises(ValueError):
            env.step(np.array([2., 0.], np.float32))

    def test_projection_and_cached_teacher(self):
        np.testing.assert_allclose(project_orca([1, 0], 0., .25), [1, 0])
        self.assertAlmostEqual(float(project_orca([0, 1], 0., .25)[1]), 1.2, places=6)
        env = BeliefEnv(PARAMS)
        env.reset(seed=81)
        before = env.expert_action()
        memory = env.world.teacher._last_pref_vel.copy()
        np.testing.assert_array_equal(before, env.expert_action())
        np.testing.assert_array_equal(memory, env.world.teacher._last_pref_vel)

    def test_set_permutation_padding_and_empty(self):
        torch.manual_seed(77)
        env = BeliefEnv(PARAMS)
        obs, _ = env.reset(seed=77)
        net = SetEncoder(env.observation_space)
        batch = {k:torch.as_tensor(v)[None] for k,v in obs.items()}
        reference = net(batch)
        order = torch.randperm(20)
        permuted = dict(batch, humans=batch['humans'][:,order], mask=batch['mask'][:,order])
        torch.testing.assert_close(net(permuted), reference, atol=1e-6, rtol=1e-6)
        padded = {k:v.clone() for k,v in batch.items()}
        padded['humans'][:,5:] = 99
        torch.testing.assert_close(net(padded), reference)
        padded['mask'][:] = 0
        self.assertTrue(torch.isfinite(net(padded)).all())

    def test_information_arms_and_density(self):
        encoder = None
        for scenario, count in [('baseline_circle',5), ('dense_circle',10), ('dense_square',20)]:
            env = BeliefEnv(PARAMS, scenario=scenario, training=False)
            obs, _ = env.reset(seed=79)
            self.assertEqual(int(obs['mask'].sum()), count)
            full = obs['humans'][:count,5:8]
            np.testing.assert_allclose(full.sum(1), 1, atol=1e-6)
            mapped = transform_observation(obs, 'map')
            self.assertTrue(np.all(np.count_nonzero(mapped['humans'][:count,5:8], axis=1) == 1))
            removed = transform_observation(obs, 'no_belief')
            self.assertEqual(np.count_nonzero(removed['humans'][:,5:]), 0)
            if encoder is None:
                encoder = SetEncoder(env.observation_space)
            self.assertEqual(tuple(encoder({k:torch.tensor(v)[None] for k,v in obs.items()}).shape), (1,96))
        with self.assertRaises(ValueError):
            BeliefEnv(PARAMS, scenario='dense_square', training=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
