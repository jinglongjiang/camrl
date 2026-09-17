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
    def test_mc_return_excludes_partial_episodes(self):
        from crowd_nav.bayes_continuous.train_smoke import complete_mc_rows
        rows = [dict(reward=.1,done=False),dict(reward=1.,done=True,outcome='success'),
                dict(reward=99.,done=False),dict(reward=-.5,done=True,episode_start=True,outcome='collision'),
                dict(reward=99.,done=False)]
        complete,excluded = complete_mc_rows(rows)
        self.assertEqual(excluded,2)
        self.assertEqual(len(complete),3)
        self.assertAlmostEqual(complete[1]['mc_return'],1.09)
        self.assertEqual(complete[2]['mc_outcome'],'collision')

    def test_finetune_learning_rates_survive_native_train(self):
        from stable_baselines3 import TD3
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory, FineTuneTD3
        env = ActionHistory(BeliefEnv(PARAMS,arm='no_belief'),route=True)
        model = FineTuneTD3('MultiInputPolicy',env,learning_rate=3e-4,buffer_size=100,
            batch_size=2,learning_starts=0,train_freq=(1,'step'),gradient_steps=1,
            policy_delay=10,device='cpu',policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=96),net_arch=[32,32],share_features_extractor=False))
        model.learn(10)
        self.assertIs(FineTuneTD3.train,TD3.train)
        self.assertEqual(model._n_updates,10)
        self.assertTrue(all(g['lr']==3e-5 for g in model.actor.optimizer.param_groups))
        self.assertTrue(all(g['lr']==3e-4 for g in model.critic.optimizer.param_groups))
        self.assertFalse(hasattr(model,'cost_critic'))
        env.close()

    def test_native_reward_warmup_does_not_update_actor(self):
        from stable_baselines3 import TD3
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory, reward_critic_warmup
        env = ActionHistory(BeliefEnv(PARAMS,arm='no_belief'),route=True)
        model = TD3('MultiInputPolicy',env,buffer_size=100,batch_size=2,device='cpu',
            policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=96),net_arch=[32,32],
                share_features_extractor=False))
        obs,_ = env.reset(seed=2407)
        action = np.array([.2,.1],np.float32)
        nxt,reward,done,_,_ = env.step(action)
        model.replay_buffer.add({k:v[None] for k,v in obs.items()},
            {k:v[None] for k,v in nxt.items()},model.policy.scale_action(action[None]),
            np.array([reward]),np.array([done]),[{}])
        actor = {k:v.clone() for k,v in model.actor.state_dict().items()}
        critic = {k:v.clone() for k,v in model.critic.state_dict().items()}
        result = reward_critic_warmup(model,updates=2)
        self.assertTrue(result['actor_unchanged'])
        self.assertTrue(all(torch.equal(actor[k],v) for k,v in model.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(critic[k],v) for k,v in model.critic.state_dict().items()))
        self.assertFalse(hasattr(model,'cost_critic'))
        self.assertEqual(model._n_updates,0)
        env.close()

    def test_directed_safety_suffix_and_quotas(self):
        from types import SimpleNamespace
        from crowd_nav.bayes_continuous.train_smoke import collect_safety_replay
        class Student:
            observation_space = {'robot': SimpleNamespace(shape=(10,))}
            def check_arm(self, arm):
                assert arm == 'no_belief'
            def predict(self, obs, deterministic=True):
                return np.array([.2, 0.],np.float32), None
        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.world = SimpleNamespace(robot=SimpleNamespace(px=0.,py=0.,theta=0.,radius=.3),
                    env=SimpleNamespace(time_step=.25,humans=[]))
            def observation(self):
                return {'robot':np.array([self.t,0,0,0,0,0,0,0,0,self.route],np.float32)}
            def reset(self, options):
                self.t, self.route, self.danger = 0, .0, False
                self.world.env.humans = [SimpleNamespace(px=2.,py=.2,gx=options['test_case'],
                    gy=0.,radius=.3,v_pref=1.)]
                return self.observation(), {}
            def step(self, action):
                self.danger = self.danger or action[0] > .9
                self.t += 1
                self.route = .7*self.route+.3*float(action[1])/1.2
                if self.t == 2:
                    self.world.env.humans[0].px = 1.
                done = self.t == 5
                info = dict(action=action,collision_cost=int(done and self.danger))
                if done:
                    info['episode_result'] = dict(outcome='collision' if self.danger else 'success')
                return self.observation(), 0., done, False, info
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp, \
                patch('crowd_nav.bayes_continuous.train_smoke.BeliefEnv',return_value=FakeEnv()), \
                patch('crowd_nav.bayes_continuous.train_smoke.ActionHistory',side_effect=lambda env,route:env):
            out = Path(tmp)/'safety'
            result = collect_safety_replay(PARAMS,out,'no_belief',actor=Student())
            fingerprints = []
            for split in ('train','validation'):
                self.assertEqual(result[split]['positive_trajectories'],30)
                self.assertEqual(result[split]['negative_trajectories'],30)
                self.assertEqual(result[split]['attempts'],60)
                data = torch.load(out/(split+'.pt'),weights_only=False)
                for rec,rows in zip(data['records'],data['trajectories']):
                    fingerprints.append(rec['layout_sha256'])
                    self.assertEqual(rows[0]['observation']['robot'].shape,(10,))
                    if rec['safety_suffix_only']:
                        self.assertEqual([r['source_step'] for r in rows],[2,3,4])
                        self.assertTrue(all(r['safety_intervention'] for r in rows))
                        self.assertEqual(rows[-1]['collision_cost'],1)
                    else:
                        self.assertEqual(len(rows),5)
                        self.assertFalse(any(r['safety_intervention'] for r in rows))
            self.assertEqual(len(set(fingerprints)),120)

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


class GaussianTransferTests(unittest.TestCase):
    def test_type_oracle_value_requires_conflicting_optima(self):
        from crowd_nav.bayes_continuous.audit_saved import type_oracle_value
        collision=np.zeros((2,2))
        same=type_oracle_value([[1.,0.],[.5,0.]],collision)
        self.assertAlmostEqual(same['gain'],0.)
        conflict=type_oracle_value([[1.,0.],[0.,1.]],collision)
        self.assertAlmostEqual(conflict['gain'],.5)
        self.assertEqual(conflict['known_choices'],[0,1])
        self.assertEqual(conflict['unknown_choice'],0)
        identical=type_oracle_value([[1.,1.],[1.,1.]],collision)
        self.assertEqual(identical['known_choices'],[0,0])
        self.assertAlmostEqual(identical['gain'],0.)

    def test_reciprocity_physical_input_excludes_truth_and_future(self):
        from crowd_nav.bayes_continuous.audit_saved import reciprocity_physical
        rng=np.random.default_rng(91)
        frames=[dict(robot=rng.normal(size=10),humans=rng.normal(size=(5,5)),
                     truth=rng.normal(size=(5,4))) for _ in range(8)]
        original=reciprocity_physical(frames,3,2)
        for frame in frames:frame['truth'][:]=999
        np.testing.assert_array_equal(original,reciprocity_physical(frames[:4],3,2))
        np.testing.assert_array_equal(original[10:15],frames[3]['humans'][2])

    def test_reciprocity_guard_and_persistent_type(self):
        from crowd_nav.bayes_continuous.audit_saved import reciprocity_reset
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory
        from crowd_sim.envs.utils.state import JointState
        env=ActionHistory(BeliefEnv(PARAMS,arm='no_belief'),route=True)
        env.unwrapped.world.robot.visible=True
        flags=[True,False,True,False,False]
        reciprocity_reset(env,250001,flags)
        world=env.unwrapped.world
        h=world.env.humans[0]
        without_robot=[other.get_observable_state() for other in world.env.humans[1:]]
        with self.assertRaises(AssertionError):
            world.policies[0].predict(JointState(h.get_full_state(),without_robot))
        for _ in range(3):env.step(np.array([.2,0.],np.float32))
        self.assertEqual([p.is_non_reciprocal for p in world.policies],flags)
        self.assertTrue(all(p.mode=='nominal' for p in world.policies))
        env.close()

    def test_information_probe_features_are_causal_and_masked(self):
        from crowd_nav.bayes_continuous.audit_saved import information_features,information_arm
        rng=np.random.default_rng(5)
        frames=[dict(robot=rng.normal(size=10).astype(np.float32),
            humans=rng.normal(size=(5,9)).astype(np.float32),
            oracle=rng.normal(size=(5,8)).astype(np.float32)) for _ in range(8)]
        x,o=information_features(frames,3,2)
        causal,_=information_features(frames[:4],3,2)
        np.testing.assert_array_equal(x,causal)
        np.testing.assert_array_equal(x[10:15],frames[3]['humans'][2,:5])
        for arm in ['current','history','map','full','oracle']:
            z=information_arm(x[None],o[None],arm)
            self.assertEqual(z.shape,(1,180))
            np.testing.assert_array_equal(z[0,:35],x[:35])
            if arm!='history':self.assertFalse(z[:,35:140].any())
            if arm!='oracle':self.assertFalse(z[:,160:].any())
            if arm=='current':self.assertFalse(z[:,35:].any())

    def test_standalone_actor_has_no_critic_and_can_learn(self):
        from crowd_nav.bayes_continuous.network import ContinuousSetActor
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory
        env=ActionHistory(BeliefEnv(PARAMS,arm='full'),route=True)
        obs,_=env.reset(seed=2407)
        actor=ContinuousSetActor(env.observation_space)
        self.assertFalse(hasattr(actor,'critic'))
        self.assertFalse(hasattr(actor,'cost_critic'))
        self.assertTrue(env.action_space.contains(actor.predict(obs)[0]))
        batch={k:torch.as_tensor(v)[None] for k,v in obs.items()}
        loss=actor(batch).square().sum();loss.backward()
        self.assertTrue(all(p.grad is not None for p in actor.parameters()))
        env.close()

    def test_failed_teacher_gate_blocks_student_creation(self):
        import tempfile,json,sys
        from unittest.mock import patch
        from pathlib import Path
        from crowd_nav.bayes_continuous.train_smoke import bayes_il_main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'status.json').write_text(json.dumps(dict(passed=False,completed=200)))
            (root/'protocol.json').write_text(json.dumps(dict(teacher='gdbn',profile='train_nonstationary')))
            with patch.object(sys,'argv',['test','--teacher-gate',str(root),'--out',str(root/'student')]):
                with self.assertRaises(ValueError):bayes_il_main()
            self.assertFalse((root/'student').exists())

    def test_teacher_margin_transfer_preserves_geometry(self):
        from dataclasses import fields, replace
        from crowd_nav.bayes_continuous.teacher import PlannerObservation,UnicycleCEMMPC,UnicycleConfig
        values={f.name:None for f in fields(PlannerObservation)}
        values.update(robot_xy=np.array([0.,0.]),robot_velocity=np.array([.5,0.]),
            robot_radius=.3,goal_xy=np.array([4.,0.]),entities=np.array([[2.,1.,0.,-.2,.3]]),
            robot_heading=0.,provenance='test')
        old=PlannerObservation(**values)
        new=replace(old,human_uncertainty_buffer=np.full((1,16),.5))
        p=UnicycleCEMMPC(UnicycleConfig(population=32,human_margin=.5))
        q=UnicycleCEMMPC(replace(p.cfg,human_margin=0.))
        controls,velocities,positions=p._rollout(p._seed_trajectories(old),old)
        clearance=p._human_clearance(velocities,old)
        np.testing.assert_allclose(p._cost(velocities,old,positions,clearance,None,None),
            q._cost(velocities,new,positions,clearance,None,None))
        np.testing.assert_allclose(p._combined_clearance(velocities,old,positions,clearance,None,None),
            q._combined_clearance(velocities,new,positions,clearance,None,None))
        np.testing.assert_allclose(p._physical_clearance(velocities,old,positions,clearance),
            q._physical_clearance(velocities,new,positions,clearance))

    def test_bayes_teacher_moment_contract(self):
        from types import SimpleNamespace
        from crowd_nav.gdbn import ModeBeliefSnapshot
        from crowd_nav.bayes_continuous.teacher import gdbn_teacher_moments
        a=np.eye(4);a[0,2]=.25;a[1,3]=.25
        model=SimpleNamespace(Pi=np.eye(2),A=[a,a],Q=[np.eye(4)*.01,np.eye(4)*.01])
        snapshot=ModeBeliefSnapshot(0,((.7,.3,0.,0.),),2,(True,))
        entities=np.array([[2.,3.,1.,0.,.3]])
        start,end,cov=gdbn_teacher_moments(entities,snapshot,model,2,.25)
        np.testing.assert_allclose(end,[[[2.25,3.],[2.5,3.]]])
        np.testing.assert_allclose(cov[0,0],np.eye(2)*.01)
        np.testing.assert_allclose(cov[0,1],np.eye(2)*.020625)
        np.testing.assert_array_equal(start[0,0],entities[0,:2])
        env=BeliefEnv(PARAMS,arm='no_belief');env.teacher_mode='gdbn'
        env.reset(seed=2407); prior=env.filter.get_belief_snapshot()
        first=env.expert_action();second=env.expert_action()
        np.testing.assert_array_equal(first,second)
        self.assertEqual(prior,env.filter.get_belief_snapshot())
        self.assertEqual(env.world.teacher_diagnostics['provenance'],'full_observation_gdbn_moment_teacher')
        env.close()

    def test_physical_mean_and_gaussian_log_probability(self):
        from stable_baselines3 import PPO
        from crowd_nav.bayes_continuous.network import DaggerGaussianPolicy, PhysicalActionMean
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory
        env = ActionHistory(BeliefEnv(PARAMS, arm='no_belief'), route=True)
        model = PPO(DaggerGaussianPolicy, env, n_steps=8, batch_size=8, device='cpu',
            policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=192),
                net_arch=dict(pi=[256,256],vf=[96,96]), activation_fn=torch.nn.ReLU,
                share_features_extractor=False))
        mapping = PhysicalActionMean(env.action_space.low, env.action_space.high)
        torch.testing.assert_close(mapping(torch.zeros(1,2)), torch.tensor([[.5,0.]]))
        obs,_ = env.reset(seed=11)
        batch={k:torch.as_tensor(v)[None] for k,v in obs.items()}
        distribution=model.policy.get_distribution(batch)
        mean=distribution.distribution.mean
        np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],mean.detach().numpy()[0])
        action=distribution.get_actions()
        _, log_prob, _=model.policy.evaluate_actions(batch,action)
        torch.testing.assert_close(log_prob,distribution.log_prob(action))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertIsNot(model.policy.pi_features_extractor,model.policy.vf_features_extractor)
        env.close()

    def test_ppo_belief_columns_can_learn_from_zero_initialization(self):
        from stable_baselines3 import PPO
        from crowd_nav.bayes_continuous.network import DaggerGaussianPolicy
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory
        env=ActionHistory(BeliefEnv(PARAMS,arm='full'),route=True)
        model=PPO(DaggerGaussianPolicy,env,n_steps=8,batch_size=8,device='cpu',
            policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=192),
                net_arch=dict(pi=[256,256],vf=[96,96]),activation_fn=torch.nn.ReLU,
                share_features_extractor=False))
        obs,_=env.reset(seed=2407)
        batch={k:torch.as_tensor(v)[None] for k,v in obs.items()}
        first=model.policy.pi_features_extractor.human[0]
        with torch.no_grad(): first.weight[:,5:9].zero_()
        no_belief={k:v.clone() for k,v in batch.items()}
        no_belief['humans'][:,:,5:9]=0
        mean=model.policy.get_distribution(batch).distribution.mean
        torch.testing.assert_close(mean,model.policy.get_distribution(no_belief).distribution.mean)
        mean.sum().backward()
        self.assertGreater(float(first.weight.grad[:,5:9].abs().max()),0.)
        self.assertFalse(hasattr(model,'cost_critic'))
        self.assertFalse(hasattr(model,'replay_buffer'))
        env.close()

    def test_ppo_training_seed_and_profile_contract(self):
        from crowd_nav.bayes_continuous.train_smoke import BeliefPPOEnv, ActionHistory
        env=ActionHistory(BeliefPPOEnv(PARAMS,'full','train_nonstationary'),route=True)
        a,ia=env.reset(seed=2407)
        env.step(np.array([.5,.2]))
        b,ib=env.reset(seed=2407)
        self.assertEqual(ia,ib)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue(80000000<=ia['layout_seed']<81000000)
        self.assertEqual(env.unwrapped.train_profile,'train_nonstationary')
        self.assertEqual(len(env.unwrapped.world.env.humans),5)
        env.close()
        with self.assertRaises(ValueError): BeliefPPOEnv(PARAMS,'full','heldout_nonstationary')


class RiskGeneralizationContracts(unittest.TestCase):
    def test_physical_layout_split_uses_distinct_case_ranges(self):
        from crowd_nav.bayes_continuous.risk_generalization import env_for, options
        env=env_for('base')
        env.reset(options=options(0,210000000,620000))
        teacher=env.unwrapped.layout_hash
        env.reset(options=options(0,230000000,620000))
        self.assertEqual(teacher,env.unwrapped.layout_hash)
        env.reset(options=options(0,230000000,625000))
        self.assertNotEqual(teacher,env.unwrapped.layout_hash)
        env.close()

    def test_teacher_queried_once_and_executed_label_matches(self):
        from types import SimpleNamespace
        from crowd_nav.bayes_continuous.risk_generalization import rollout
        class FakeEnv:
            def __init__(self):
                self.unwrapped=self;self.calls=0;self.all_risks=np.zeros((4,20))
                self.layout_hash='test'
                self.world=SimpleNamespace(env=SimpleNamespace(humans=[None]*5,test_sim='circle_crossing'))
            def reset(self,options): return {'robot':np.zeros(10)},{}
            def expert_action(self):
                self.calls+=1
                return np.array([.5,self.calls*.1])
            def step(self,action):
                self.executed=action.copy()
                return {},0.,True,False,{'episode_result':{'outcome':'success'}}
        env=FakeEnv()
        _,rows=rollout(env,None,{'profile':'nominal'},collect=True)
        self.assertEqual(env.calls,1)
        np.testing.assert_array_equal(rows[0][2],env.executed)

    def test_shared_start_and_active_risk_mask(self):
        from crowd_nav.bayes_continuous.risk_generalization import env_for, make_model, options, physical_mean
        torch.set_num_threads(1)
        env=env_for('bayes')
        obs,_=env.reset(options=options(0,260100000,670000))
        model=make_model(env,2407)
        batch={k:torch.as_tensor(v[None]) for k,v in obs.items()}
        altered={k:v.clone() for k,v in batch.items()}
        altered['risk']=torch.linspace(0,1,20)[None]
        with torch.no_grad():
            np.testing.assert_array_equal(physical_mean(model.policy,batch),physical_mean(model.policy,altered))
            model.policy.pi_features_extractor.risk_gain.fill_(1.)
            encoder=model.policy.pi_features_extractor
            permutation=torch.randperm(20)
            permuted={k:v.clone() for k,v in batch.items()}
            for k in ['humans','mask','risk']:permuted[k]=permuted[k][:,permutation]
            torch.testing.assert_close(encoder(batch),encoder(permuted),atol=1e-6,rtol=1e-6)
            padded={k:v.clone() for k,v in batch.items()}
            padded['humans'][:,5:]=1000
            padded['risk'][:,5:]=1
            torch.testing.assert_close(encoder(batch),encoder(padded),atol=0,rtol=0)
        self.assertEqual(len(env.unwrapped.world.env.humans),5)
        self.assertTrue(np.isfinite(env.unwrapped.all_risks).all())
        env.close()

    def test_counts_and_deterministic_risk_history(self):
        from crowd_nav.bayes_continuous.risk_generalization import env_for, options
        for scene,count in [('baseline_circle',5),('dense_circle',10),('large_circle',12),('dense_square',20)]:
            env=env_for('bayes',False,scene)
            opts=dict(layout_seed=260100001,test_case=670001,profile='nominal')
            a,_=env.reset(options=opts)
            self.assertEqual(int(a['mask'].sum()),count)
            env.step(np.array([.5,.1],np.float32))
            risk=env.unwrapped.all_risks.copy()
            b,_=env.reset(options=opts)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            env.step(np.array([.5,.1],np.float32))
            np.testing.assert_array_equal(risk,env.unwrapped.all_risks)
            self.assertTrue(np.all(risk>=0) and np.all(risk<=1))
            env.close()


class LocalRiskContracts(unittest.TestCase):
    def test_inverse_wishart_predictive_projection(self):
        from scipy.stats import invwishart
        from scipy.special import ndtr, stdtr
        nu=9
        psi=np.array([[.4,.12],[.12,.2]])
        direction=np.array([.6,.8])
        samples=invwishart.rvs(df=nu,scale=psi,size=40000,random_state=2407)
        projected=np.einsum('i,nij,j->n',direction,samples,direction)
        analytic=stdtr(nu-1,.15/np.sqrt(direction@psi@direction/(nu-1)))
        estimate=ndtr(.15/np.sqrt(projected)).mean()
        self.assertLess(abs(analytic-estimate),.005)

    def test_accumulation_can_distinguish_matched_covariance(self):
        from scipy.special import ndtr, stdtr
        z=np.array([[-2.,-8.,-8.,-8.],[-2.8,-2.8,-2.8,-2.8]])
        gaussian=ndtr(z)
        student=stdtr(4,z*np.sqrt(2.))
        self.assertEqual(gaussian.max(1).argmax(),student.max(1).argmax())
        self.assertNotEqual(gaussian.sum(1).argmax(),student.sum(1).argmax())

    def test_calibration_uses_disjoint_five_person_episodes(self):
        from crowd_nav.bayes_continuous.risk_generalization import local_calibration
        result = local_calibration()
        self.assertFalse(set(result['train_cases']) & set(result['validation_cases']))
        self.assertGreater(result['selected']['q'],0.)
        self.assertTrue(all(np.isfinite(r['nll']) for r in result['candidates']))

    def test_identity_causality_and_no_entity_deletion(self):
        from crowd_nav.bayes_continuous.risk_generalization import LocalRiskActor, LOCAL_ARMS
        from crowd_nav.bayes_continuous.stage_audit import make_env, FrozenActor, SOURCE
        config=dict(window=4,nu0=5,q=.006)
        for scene,count in [('baseline_circle',5),('dense_square',20)]:
            env=make_env(scene,'no_belief')
            obs,_=env.reset(options=dict(layout_seed=290000002,test_case=709902,profile='nominal'))
            actor=FrozenActor(SOURCE/'2407_no_belief/attempt0_20480.zip',env.observation_space)
            before={k:v.copy() for k,v in obs.items()}
            for arm in LOCAL_ARMS:
                routed=LocalRiskActor(actor,env,arm,config)
                action=routed.predict(obs)
                np.testing.assert_array_equal(action,routed.predict(obs))
                self.assertEqual(len(routed.residuals),0)
                self.assertTrue(env.action_space.contains(action))
                if count==5:
                    np.testing.assert_array_equal(action,actor.predict(obs))
                self.assertEqual(len(env.unwrapped.world.env.humans),count)
                for key in obs:
                    np.testing.assert_array_equal(obs[key],before[key])
            routed=LocalRiskActor(actor,env,'bayes_local',config)
            action=routed.predict(obs)
            obs,_,_,_,_=env.step(action)
            routed.predict(obs)
            self.assertEqual(len(routed.residuals),1)
            routed.predict(obs)
            self.assertEqual(len(routed.residuals),1)
            env.close()


class ModelPolicySearchContracts(unittest.TestCase):
    def test_ars_update_and_zero_signal(self):
        from crowd_nav.bayes_continuous.risk_generalization import ars_step
        theta=np.zeros(4)
        directions=np.eye(4)
        returns=np.array([[1.,-1.],[.5,-.5],[0.,0.],[0.,0.]])
        updated=ars_step(theta,directions,returns)
        self.assertGreater(updated[0],updated[1])
        self.assertGreater(updated[1],0.)
        np.testing.assert_array_equal(updated[2:],np.zeros(2))
        np.testing.assert_array_equal(ars_step(theta,directions,np.ones((4,2))),theta)

    def test_predictive_policy_does_not_read_private_human_goals(self):
        from crowd_nav.bayes_continuous.risk_generalization import ModelSearchActor
        from crowd_nav.bayes_continuous.stage_audit import make_env,FrozenActor,INITIAL
        env=make_env('baseline_circle','no_belief')
        obs,_=env.reset(options=dict(layout_seed=310749001,test_case=749001,profile='nominal'))
        actor=FrozenActor(INITIAL,env.observation_space)
        config=dict(window=4,nu0=5,q=.006)
        for arm in ('cv','gaussian','bayes'):
            before=ModelSearchActor(actor,env,arm,np.zeros(4),config).predict(obs)
            humans=env.unwrapped.world.env.humans
            old=[(h.gx,h.gy) for h in humans]
            for h in humans:
                h.gx,h.gy=999.,-999.
            after=ModelSearchActor(actor,env,arm,np.zeros(4),config).predict(obs)
            for h,goal in zip(humans,old):
                h.gx,h.gy=goal
            np.testing.assert_array_equal(before,after)
            self.assertTrue(env.action_space.contains(after))
        env.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
