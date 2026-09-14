import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from crowd_nav.bayes_continuous.environment import BeliefEnv, project_orca, transform_observation
from crowd_nav.bayes_continuous.network import SetEncoder
from crowd_nav.bayes_continuous.algorithm import BayesSetTD3, CostReplay
from crowd_nav.belief_mdp.runtime import StepResult

PARAMS = Path(__file__).resolve().parents[2] / 'repair_results/params'


class Contracts(unittest.TestCase):
    def test_parallel_dagger_matches_serial(self):
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory, dagger_rollout_worker, collect_dagger_parallel
        torch.set_num_threads(1)
        env = ActionHistory(BeliefEnv(PARAMS,arm='no_belief'),route=True)
        model = BayesSetTD3('MultiInputPolicy',env,buffer_size=100,device='cpu',
            policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=96),net_arch=[32,32]))
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.zero_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'student.zip'
            model.save(path)
            serial, a = dagger_rollout_worker((PARAMS,path,2,838000))
            parallel, b = collect_dagger_parallel(PARAMS,path,2,838000,workers=2,chunk_size=1)
        self.assertEqual([(r['layout_sha256'],r['outcome'],r['steps']) for r in serial],
                         [(r['layout_sha256'],r['outcome'],r['steps']) for r in parallel])
        for left,right in zip(a,b):
            self.assertEqual(len(left),len(right))
            for x,y in zip(left,right):
                for key in ('action','teacher_action'):
                    np.testing.assert_allclose(x[key],y[key],atol=1e-6,rtol=0)
                for key in x['observation']:
                    np.testing.assert_array_equal(x['observation'][key],y['observation'][key])
        env.close()

    def test_route_uses_execution_not_expert_label(self):
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory, add_route_history
        env = ActionHistory(BeliefEnv(PARAMS,arm='no_belief'),route=True)
        obs,_ = env.reset(seed=42)
        self.assertEqual(obs['robot'].shape,(10,))
        self.assertEqual(obs['robot'][-1],0.)
        nxt,*_ = env.step(np.array([.2,.6],np.float32))
        self.assertAlmostEqual(float(nxt['robot'][-1]),.15)
        rows = [[dict(observation=dict(obs,robot=obs['robot'][:9]),
            next_observation=dict(nxt,robot=nxt['robot'][:9]),action=np.array([.2,.6],np.float32),
            teacher_action=np.array([1.,-1.2],np.float32))]]
        add_route_history(rows)
        np.testing.assert_array_equal(rows[0][0]['next_observation']['robot'],nxt['robot'])
        last,*_ = env.step(np.array([.2,-.6],np.float32))
        self.assertAlmostEqual(float(last['robot'][-1]),-.045)
        env.close()

    def test_dagger_query_does_not_drive_robot(self):
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory, episodes
        env = ActionHistory(BeliefEnv(PARAMS,arm='no_belief'),route=True)
        class Student:
            observation_space = env.observation_space
            def predict(self, obs, deterministic=True):
                return np.array([.3,.1],np.float32),None
        ordinary,rows,_ = episodes(PARAMS,1,91234,Student(),collect=True,case_offset=91324,arm='no_belief',diagnostics=True)
        labelled,annotated,_ = episodes(PARAMS,1,91234,Student(),collect=True,case_offset=91324,arm='no_belief',diagnostics=True,query_teacher=True)
        self.assertEqual(ordinary[0]['outcome'],labelled[0]['outcome'])
        self.assertEqual(len(rows[0]),len(annotated[0]))
        for a,b in zip(rows[0],annotated[0]):
            np.testing.assert_array_equal(a['action'],b['action'])
            for key in a['next_observation']:
                np.testing.assert_array_equal(a['next_observation'][key],b['next_observation'][key])
            self.assertIn('teacher_action',b)
        self.assertTrue(any(not np.array_equal(r['action'],r['teacher_action']) for r in annotated[0]))
        env.close()

    def test_executed_action_history_and_collection_alignment(self):
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory, augment_collection
        env = ActionHistory(BeliefEnv(PARAMS, arm='no_belief'))
        obs, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs['robot'][-2:], [0.,0.])
        nxt, _, _, _, _ = env.step(np.array([.2,.6], np.float32))
        np.testing.assert_allclose(nxt['robot'][-2:], [.2,.5])
        rows = [[dict(observation=dict(obs,robot=obs['robot'][:7]),
                      next_observation=dict(nxt,robot=nxt['robot'][:7]), action=np.array([.2,.6],np.float32))]]
        augment_collection(rows)
        np.testing.assert_array_equal(rows[0][0]['observation']['robot'], obs['robot'])
        np.testing.assert_array_equal(rows[0][0]['next_observation']['robot'], nxt['robot'])
        obs, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs['robot'][-2:], [0.,0.])
        env.close()

    def test_cost_probability_and_empty_safety_gate(self):
        from crowd_nav.bayes_continuous.train_smoke import supervised_warmup
        from types import SimpleNamespace
        values = BayesSetTD3.collision_probability((torch.tensor([[-100.], [0.]]),
                                                    torch.tensor([[100.], [-100.]])))
        torch.testing.assert_close(values, torch.tensor([[1.], [.5]]))
        with self.assertRaises(ValueError):
            supervised_warmup(SimpleNamespace(), [])
        negatives = [[{'collision_cost':0.}] for _ in range(100)]
        with self.assertRaises(ValueError):
            supervised_warmup(SimpleNamespace(), [], risk_train=negatives, risk_valid=negatives)

    def test_actual_motion_collision_cannot_be_hidden_by_native_check(self):
        env = BeliefEnv(PARAMS)
        env.reset(seed=91)
        r, h = env.world.robot, env.world.env.humans[0]
        h.px, h.py = r.px+.8, r.py
        def motion(_):
            h.px, h.py = r.px+.4, r.py
            return StepResult('running', 0., False, .2)
        with patch.object(env.world, 'step', side_effect=motion):
            _, reward, done, _, info = env.step(np.zeros(2, np.float32))
        self.assertTrue(done)
        self.assertEqual(info['outcome'], 'collision')
        self.assertEqual(info['native_outcome'], 'running')
        self.assertLess(info['actual_clearance'], 0)
        self.assertEqual(reward, env.world.env.collision_penalty)

    def test_teacher_uses_observations_not_human_goals(self):
        env = BeliefEnv(PARAMS)
        env.reset(seed=91)
        expected = env.expert_action()
        env._teacher_cache = None
        env.world._cem_teacher = None
        for h in env.world.env.humans:
            h.gx, h.gy = 10000., -10000.
        np.testing.assert_array_equal(expected, env.expert_action())

    def test_layout_split_uses_case_not_filter_seed(self):
        env = BeliefEnv(PARAMS)
        def layout(seed, case):
            env.reset(seed=seed, options={'test_case':case})
            return np.array([[h.px,h.py,h.gx,h.gy] for h in env.world.env.humans])
        first = layout(1, 81)
        np.testing.assert_array_equal(first, layout(2, 81))
        self.assertFalse(np.array_equal(first, layout(1, 82)))

    def test_explicit_critic_and_stage_contract(self):
        env = BeliefEnv(PARAMS)
        model = BayesSetTD3('MultiInputPolicy', env, replay_buffer_class=CostReplay,
            buffer_size=100, device='cpu', policy_kwargs=dict(features_extractor_class=SetEncoder,
            net_arch=[32], share_features_extractor=False), seed=2)
        model.belief_arm = 'full'
        model.check_arm('full')
        with self.assertRaises(ValueError):
            model.check_arm('no_belief')
        obs, _ = env.reset(seed=9)
        for i in range(8):
            model.replay_buffer.add({k:v[None] for k,v in obs.items()},
                {k:v[None] for k,v in obs.items()}, np.zeros((1,2), np.float32),
                np.array([-.5]), np.array([True]), [{'collision_cost':float(i%2)}])
        before = {k:v.clone() for k,v in model.actor.state_dict().items()}
        model.train(3, 8)
        self.assertTrue(all(torch.equal(before[k], v) for k,v in model.actor.state_dict().items()))
        with self.assertRaises(RuntimeError):
            model.enable_actor(dict(episodes=100, success_rate=1., collision_rate=0.))
        # Synthetic receipt only exercises the gated optimizer, not navigation qualification.
        model.warmup_updates = 1000
        model.critic_validation = {'passed': True}
        model.enable_actor(dict(episodes=100, success_rate=1., collision_rate=0.))
        model.demo_observations = {k:np.repeat(v[None], 8, axis=0) for k,v in obs.items()}
        model.demo_actions = np.zeros((8,2), np.float32)
        model.train(2, 8)
        self.assertIn('actor_cost_grad_norm', model.loss_history[-2])
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder)/'checkpoint')
            restored = BayesSetTD3.load(Path(folder)/'checkpoint', device='cpu')
            self.assertEqual(restored.warmup_updates, 1000)
            restored.check_arm('full')
            self.assertFalse(restored.actor_enabled)
            self.assertTrue(restored.critic_validation['passed'])
            np.testing.assert_array_equal(model.predict(obs)[0], restored.predict(obs)[0])
            for a,b in zip(model.cost_critic.parameters(), restored.cost_critic.parameters()):
                torch.testing.assert_close(a, b)

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
        position = env.world.robot.px
        np.testing.assert_array_equal(before, env.expert_action())
        self.assertEqual(position, env.world.robot.px)

    def test_seed_and_uniform_prior(self):
        env = BeliefEnv(PARAMS)
        def rollout():
            observations = [env.reset(seed=123)[0]]
            for _ in range(3):
                observations.append(env.step(np.array([.1, .2], np.float32))[0])
            return observations
        first, second = rollout(), rollout()
        np.testing.assert_allclose(first[0]['humans'][:5, 5:8], 1./3, atol=1e-7)
        for a, b in zip(first, second):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_terminal_step(self):
        env = BeliefEnv(PARAMS)
        env.reset(seed=123)
        env.world.env.humans = []
        env.world.env.global_time = env.world.env.time_limit - .25
        _, _, done, _, info = env.step(np.zeros(2, np.float32))
        self.assertTrue(done)
        self.assertEqual(info['outcome'], 'timeout')
        self.assertEqual(env.world.env.global_time, env.world.env.time_limit)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(2, np.float32))

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
