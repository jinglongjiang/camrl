"""Frozen five-human development stages; no high-density model selection."""
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import obs_as_tensor
from crowd_nav.bayes_continuous.environment import BeliefEnv, ARMS
from crowd_nav.bayes_continuous.algorithm import BayesSetTD3, CostReplay
from crowd_nav.bayes_continuous.network import SetEncoder
from crowd_nav.gdbn import GNG, GDBN


def stack_obs(observations):
    return {k:np.stack([o[k] for o in observations]) for k in observations[0]}


class ActionHistory(gym.Wrapper):
    """Executed action, not the teacher's hidden future plan."""
    def __init__(self, env, route=False):
        super().__init__(env)
        self.route_enabled = route
        self.route = 0.
        self.observation_space = gym.spaces.Dict(dict(env.observation_space.spaces,
            robot=gym.spaces.Box(-np.inf, np.inf, (10 if route else 9,), np.float32)))
        self.previous = np.zeros(2, np.float32)

    def augment(self, obs):
        parts = [obs['robot'], self.previous/[1.,1.2]]
        if self.route_enabled:
            parts.append([self.route])
        return dict(obs, robot=np.concatenate(parts).astype(np.float32))

    def reset(self, **kwargs):
        self.previous[:] = 0
        self.route = 0.
        obs, info = self.env.reset(**kwargs)
        return self.augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.previous = np.asarray(info['action'], np.float32).copy()
        self.route = .7*self.route + .3*float(self.previous[1])/1.2
        return self.augment(obs), reward, terminated, truncated, info


def augment_collection(trajectories):
    for episode in trajectories:
        previous = np.zeros(2, np.float32)
        for row in episode:
            if row['observation']['robot'].shape != (7,):
                raise ValueError('Expected original seven-field observations')
            row['observation'] = dict(row['observation'], robot=np.concatenate([
                row['observation']['robot'], previous/[1.,1.2]]).astype(np.float32))
            previous = np.asarray(row['action'], np.float32)
            row['next_observation'] = dict(row['next_observation'], robot=np.concatenate([
                row['next_observation']['robot'], previous/[1.,1.2]]).astype(np.float32))


def neighbor_conflicts(trajectories, records):
    from scipy.spatial import cKDTree
    rows, episode_ids = [], []
    for i, (episode, record) in enumerate(zip(trajectories, records)):
        if record['outcome'] == 'success':
            rows.extend(episode)
            episode_ids.extend([i]*len(episode))
    features = []
    for row in rows:
        obs = row['observation']
        people = obs['humans'][obs['mask']>.5,:5]
        people = people[np.argsort(np.linalg.norm(people[:,:2], axis=1), kind='stable')]
        features.append(np.concatenate([obs['robot'][:7], people.ravel()]))
    x = np.asarray(features)
    prev = np.stack([r['observation']['robot'][7:] for r in rows])
    omega = np.array([r['action'][1] for r in rows])
    ids = np.asarray(episode_ids)
    def nearest(features):
        distances, indices = cKDTree(features).query(features, k=min(128,len(rows)))
        eligible = ids[indices] != ids[:,None]
        found = eligible.any(1)
        first = eligible.argmax(1)
        return distances[np.arange(len(rows)),first], indices[np.arange(len(rows)),first], found
    d, j, found = nearest(x)
    close = found & (d <= np.quantile(d[found], .1))
    _, augmented, augmented_found = nearest(np.column_stack([x,prev]))
    def conflict(index):
        return (omega*omega[index] < 0) & (np.abs(omega)>.2) & (np.abs(omega[index])>.2)
    selected = close & augmented_found
    return dict(samples=len(rows), selected=int(selected.sum()),
        current_only_conflict=float(conflict(j)[selected].mean()),
        previous_action_conflict=float(conflict(augmented)[selected].mean()),
        distance_cutoff=float(np.quantile(d[found],.1)),
        augmented_neighbors_current_distance_mean=float(np.linalg.norm(x[selected]-x[augmented[selected]],axis=1).mean()),
        current_neighbors_distance_mean=float(d[selected].mean()),
        scope='Cross-episode nearest neighbors, closest 10% current states; sorted geometry; diagnostic, not a causal proof')


def fit_metrics(model, observations, physical_actions):
    predictions = np.concatenate([model.predict({k:v[i:i+256] for k,v in observations.items()},
        deterministic=True)[0] for i in range(0,len(physical_actions),256)])
    rmse = np.sqrt(np.mean((predictions-physical_actions)**2,axis=0))
    std = predictions[:,1].std()
    teacher_std = physical_actions[:,1].std()
    return dict(speed_rmse=float(rmse[0]), omega_rmse=float(rmse[1]),
        omega_std=float(std), teacher_omega_std=float(teacher_std),
        passed=bool(rmse[1]<=.15 and std >= .8*teacher_std))


def supervised_warmup(model, trajectories, updates=1000, risk_train=None, risk_valid=None):
    model.critic_validation = {'passed': False}
    for name, split in [('train', risk_train), ('validation', risk_valid)]:
        labels = [any(r['collision_cost'] for r in episode) for episode in (split or [])]
        if sum(labels) < 20 or len(labels)-sum(labels) < 20:
            raise ValueError('Independent safety '+name+' requires >=20 positive and >=20 negative trajectories')
    # Splits must be made by layout before extracting transition rows.
    def fingerprint(episode):
        return hashlib.sha256(b''.join(np.asarray(episode[0]['observation'][k]).tobytes()
            for k in sorted(episode[0]['observation']))).hexdigest()
    train_fingerprints = {fingerprint(x) for x in risk_train}
    valid_fingerprints = {fingerprint(x) for x in risk_valid}
    if len(train_fingerprints) != len(risk_train) or len(valid_fingerprints) != len(risk_valid):
        raise ValueError('Repeated safety layouts cannot count as independent trajectories')
    if train_fingerprints & valid_fingerprints:
        raise ValueError('Safety train/validation trajectories overlap')
    def dataset(episodes):
        rows, returns, costs = [], [], []
        for episode in episodes:
            reward, cost = 0., 0.
            episode_returns, episode_costs = [], []
            for row in reversed(episode):
                reward = row['reward'] + model.gamma*reward
                cost = max(cost, row['collision_cost'])
                episode_returns.append(reward)
                episode_costs.append(cost)
            rows.extend(episode)
            returns.extend(reversed(episode_returns))
            costs.extend(reversed(episode_costs))
        return ({k:np.stack([r['observation'][k] for r in rows]) for k in rows[0]['observation']},
                model.policy.scale_action(np.stack([r['action'] for r in rows])),
                np.asarray(returns, np.float32)[:,None], np.asarray(costs, np.float32)[:,None])
    train, valid = dataset(trajectories[:-20]), dataset(trajectories[-20:])
    safety_train, safety_valid = dataset(risk_train), dataset(risk_valid)
    rng = np.random.default_rng(2407)
    before = {k:v.clone() for k,v in model.actor.state_dict().items()}
    def losses(data, indices):
        obs, act, returns, costs = data
        obs = obs_as_tensor({k:v[indices] for k,v in obs.items()}, model.device)
        act, returns, costs = [torch.as_tensor(v[indices], device=model.device) for v in (act, returns, costs)]
        qr = model.critic(obs, act)
        qc = model.cost_critic(obs, act)
        return (sum(torch.nn.functional.smooth_l1_loss(q, returns) for q in qr),
                sum(torch.nn.functional.mse_loss(torch.sigmoid(q), costs) for q in qc))
    with torch.no_grad():
        initial = [float(losses(valid, np.arange(len(valid[1])))[0]),
                   float(losses(safety_valid, np.arange(len(safety_valid[1])))[1])]
    for _ in range(updates):
        reward_loss = losses(train, rng.integers(len(train[1]), size=128))[0]
        cost_loss = losses(safety_train, rng.integers(len(safety_train[1]), size=128))[1]
        for optimizer, loss in [(model.critic.optimizer, reward_loss), (model.cost_optimizer, cost_loss)]:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        final = [float(losses(valid, np.arange(len(valid[1])))[0]),
                 float(losses(safety_valid, np.arange(len(safety_valid[1])))[1])]
        obs, act, _, labels = safety_valid
        pred = model.collision_probability(model.cost_critic(obs_as_tensor(obs, model.device),
            torch.as_tensor(act, device=model.device))).cpu().numpy().ravel()
        labels = labels.ravel().astype(bool)
        from scipy.stats import rankdata
        ranks = rankdata(pred)
        npos, nneg = labels.sum(), (~labels).sum()
        auc = float((ranks[labels].sum()-npos*(npos+1)/2)/(npos*nneg))
        metrics = dict(positive_mse=float(np.mean((pred[labels]-1)**2)),
                       negative_mse=float(np.mean(pred[~labels]**2)),
                       brier=float(np.mean((pred-labels)**2)), auroc=auc,
                       positive_mean=float(pred[labels].mean()), negative_mean=float(pred[~labels].mean()),
                       scope='transition metrics on separate, class-enriched trajectory validation; not deployment calibration')
    assert all(torch.equal(before[k], v) for k,v in model.actor.state_dict().items())
    model.critic_target.load_state_dict(model.critic.state_dict())
    model.cost_target.load_state_dict(model.cost_critic.state_dict())
    model.warmup_updates = updates
    model.critic_validation = dict(before=initial, after=final, heldout_episodes=20,
                                  actor_unchanged=True, safety_metrics=metrics,
                                  safety_episode_counts={name:dict(
                                      positive=sum(any(r['collision_cost'] for r in ep) for ep in split),
                                      negative=sum(not any(r['collision_cost'] for r in ep) for ep in split))
                                      for name, split in [('train',risk_train), ('validation',risk_valid)]},
                                  passed=bool(final[0] <= .2 and final[1] <= .04 and
                                              metrics['positive_mse'] <= .04 and metrics['negative_mse'] <= .04 and
                                              auc > .5 and metrics['positive_mean'] > metrics['negative_mean'] and
                                              final[0] <= initial[0] and final[1] <= initial[1]))
    return model.critic_validation



def dagger_online_smoke(args, stages):
    """Authorized diagnostic only; does not waive the independent safety gate."""
    if (args.arm != 'no_belief' or args.safety is None or args.steps != 2000 or
            args.nonstationary_probability != 0.):
        raise ValueError('DAgger smoke requires no_belief, --safety, 2000 steps and nominal')
    args.out.mkdir(exist_ok=False)
    report = dict(stage='initializing', arm=args.arm, seed=2407, robot_fields=10,
                  train_humans=5, profile='nominal', requested_steps=2000,
                  diagnostic_only=True, independent_safety_gate_waived=False,
                  formal_training_authorized=False)
    def save(stage):
        report['stage'] = stage
        (args.out/'status.json').write_text(json.dumps(report, indent=2))
        print('TD3_SMOKE', stage, flush=True)
    save('initializing')
    env = None
    try:
        for name, key in [('teacher.py','teacher_source_sha256'),
                          ('environment.py','environment_source_sha256')]:
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != stages[key]:
                raise ValueError('Frozen source mismatch: '+name)
        checkpoint = args.stages/'round16.zip'
        report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if report['checkpoint_sha256'] != '634e8106713c0c316afc77bf345615d3a6a101789f82f182853356ab57dbc086':
            raise ValueError('Not the approved round16 checkpoint')
        torch.set_num_threads(1)
        np.random.seed(2407)
        torch.manual_seed(2407)
        env = ActionHistory(BeliefEnv(args.params, arm=args.arm, seed=2407), route=True)
        model = BayesSetTD3.load(checkpoint, env=env, device='cuda')
        model.check_arm(args.arm)
        if model.observation_space['robot'].shape != (10,) or model.learning_starts != 0:
            raise ValueError('Wrong observation or random warm-up contract')
        if model.num_timesteps != 0 or model._n_updates != 0:
            raise ValueError('Expected a BC-only checkpoint')
        model.action_noise = NormalActionNoise(np.zeros(2), .1*np.ones(2))
        model.actor_enabled = False
        report['execution'] = dict(learning_starts=model.learning_starts,
            train_freq=[model.train_freq.frequency, model.train_freq.unit.value], gradient_steps=model.gradient_steps,
            policy_delay=model.policy_delay, tau=model.tau, gamma=model.gamma,
            normalized_action_noise_std=.1)
        if (model.gradient_steps, model.policy_delay, model.tau, model.gamma) != (1, 2, .005, .99):
            raise ValueError('Unexpected TD3 schedule')
        save('safety_collection')
        safety_path = args.safety
        if safety_path.exists():
            counts = json.loads((safety_path/'receipt.json').read_text())
        else:
            counts = collect_safety_replay(args.params, safety_path, args.arm, actor=model)
        if counts.get('protocol') != 'directed_suffix_v1':
            raise ValueError('Refusing legacy random-perturbation safety data')
        report['safety_path'], report['safety_counts'] = str(safety_path), counts
        save('safety_collection')
        if not all(counts[s]['quota_complete'] for s in ('train','validation')):
            save('blocked_insufficient_safety_classes')
            return
        safety = {s:torch.load(safety_path/(s+'.pt'), weights_only=False) for s in ('train','validation')}
        collection = torch.load(args.demos/'collection.pt', weights_only=False)
        if json.loads((args.demos/'results.json').read_text()).get('arm') != args.arm:
            raise ValueError('Demo arm mismatch')
        collection['arm'] = args.arm
        add_route_history(collection['trajectories'])
        for data in [collection, *safety.values()]:
            if data.get('arm') != args.arm:
                raise ValueError('Replay arm mismatch')
            for episode in data['trajectories']:
                for row in episode:
                    for key in ('observation','next_observation'):
                        if row[key]['robot'].shape != (10,):
                            raise ValueError('Replay is not route-aware 10D')
        save('critic_warmup_1000_actor_frozen')
        model.actor.requires_grad_(False)
        report['critic'] = supervised_warmup(model, collection['trajectories'], updates=1000,
            risk_train=safety['train']['trajectories'], risk_valid=safety['validation']['trajectories'])
        model.save(args.out/'critic_warmup')
        save('critic_warmup_complete')
        if not report['critic']['passed']:
            save('blocked_critic_gate')
            return
        # Use the known five-person development queue, not the 500-case confirmation set.
        def evaluate():
            numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all()
            try:
                records, _, _ = episodes(args.params, 100, 530000, actor=model,
                    case_offset=30000, arm=args.arm)
                return records
            finally:
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                torch.cuda.set_rng_state_all(cuda_state)
        save('baseline_development_evaluation')
        baseline = evaluate()
        report['baseline'] = receipt(baseline)
        (args.out/'baseline_episodes.json').write_text(json.dumps(baseline))
        model.enable_actor(report['baseline'])
        model.actor.requires_grad_(True)
        good = [r for rec, rows in zip(collection['records'], collection['trajectories'])
                if rec['outcome']=='success' for r in rows]
        model.demo_observations = {k:np.stack([r['observation'][k] for r in good])
                                   for k in good[0]['observation']}
        model.demo_actions = model.policy.scale_action(np.stack([r['action'] for r in good])).astype(np.float32)
        # Validation trajectories never enter replay; perturbed actions are never BC labels.
        for rows in collection['trajectories'][:-20]:
            for r in rows:
                model.replay_buffer.add({k:v[None] for k,v in r['observation'].items()},
                    {k:v[None] for k,v in r['next_observation'].items()},
                    model.policy.scale_action(r['action'][None]), np.array([r['reward']]),
                    np.array([r['done']]), [{'collision_cost':r['collision_cost']}])
        report['evaluations'] = []
        for step in (1000, 2000):
            save('td3_to_'+str(step))
            model.learn(total_timesteps=1000, reset_num_timesteps=(step==1000))
            model.save(args.out/('rl_'+str(step)))
            records = evaluate()
            result = dict(steps=model.num_timesteps, **receipt(records))
            report['evaluations'].append(result)
            (args.out/('evaluation_'+str(step)+'.json')).write_text(json.dumps(records))
            (args.out/'losses.json').write_text(json.dumps(model.loss_history))
            (args.out/'episodes.json').write_text(json.dumps(env.unwrapped.episode_records))
            save('evaluated_'+str(step))
            if result['success_rate'] < .8:
                save('stopped_degradation')
                return
        report['retained_90_percent'] = bool(report['evaluations'][-1]['success_rate'] >= .9)
        save('completed_diagnostic_only')
    except Exception as exc:
        report['error'] = repr(exc)
        save('failed')
        raise
    finally:
        if env is not None:
            env.close()


def online_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--safety', type=Path)
    parser.add_argument('--params', type=Path, default=Path('repair_results/params'))
    parser.add_argument('--demos', type=Path, default=Path('repair_results/student_bc_history_continued_20260914'))
    parser.add_argument('--nonstationary-probability', type=float, default=0.)
    args = parser.parse_args()
    results = json.loads((args.stages/'results.json').read_text())
    if results.get('arm') != args.arm:
        raise ValueError('Stage and online arms differ')
    if results.get('stage') == 'dagger':
        return dagger_online_smoke(args, results)
    if (results['teacher']['episodes'] < 100 or results['teacher']['success_rate'] < .9 or
            results['teacher']['collision_rate'] > .02):
        raise RuntimeError('Unqualified teacher: RL is forbidden')
    for name, key in [('teacher.py','teacher_source_sha256'), ('environment.py','environment_source_sha256')]:
        current = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        if results.get(key) != current:
            raise RuntimeError('Stage receipt does not match current teacher/environment')
    if not 0 <= args.nonstationary_probability <= 1:
        raise ValueError('Invalid training profile probability')
    torch.set_num_threads(1)
    env = BeliefEnv(results['params'], arm=args.arm)
    env.nonstationary_probability = args.nonstationary_probability
    model = BayesSetTD3.load(args.stages/'supervised_warmup.zip', env=env)
    model.check_arm(args.arm)
    model.enable_actor(results['bc'])
    args.out.mkdir(exist_ok=False)
    collection = torch.load(args.stages/'collection.pt', weights_only=False)
    good = []
    for record, rows in zip(collection['records'], collection['trajectories']):
        if record['outcome'] == 'success':
            good.extend(rows)
        for r in rows:
            model.replay_buffer.add({k:v[None] for k,v in r['observation'].items()},
                {k:v[None] for k,v in r['next_observation'].items()},
                model.policy.scale_action(r['action'][None]), np.array([r['reward']]),
                np.array([r['done']]), [{'collision_cost':r['collision_cost']}])
    model.demo_observations = {k:np.stack([r['observation'][k] for r in good]) for k in good[0]['observation']}
    model.demo_actions = model.policy.scale_action(np.stack([r['action'] for r in good])).astype(np.float32)
    model.learn(total_timesteps=args.steps)
    model.save(args.out/'final')
    (args.out/'episodes.json').write_text(json.dumps(env.episode_records, indent=2))
    (args.out/'losses.json').write_text(json.dumps(model.loss_history))


def episodes(params, count, offset, actor=None, collect=False, case_offset=0, diagnostics=False,
             arm='full', perturbation_seed=None, query_teacher=False, route_history=False):
    env = BeliefEnv(params, seed=2407, arm=arm)
    width = actor.observation_space['robot'].shape[0] if actor is not None else (10 if route_history else 7)
    policy_env = ActionHistory(env, route=width==10) if width in (9,10) else env
    if query_teacher and (actor is None or not collect):
        raise ValueError('DAgger requires student execution and label collection')
    records, trajectories, raw = [], [], []
    perturbations = np.random.default_rng(perturbation_seed)
    for episode in range(count):
        obs, _ = policy_env.reset(options={'layout_seed': offset+episode, 'test_case': case_offset+episode,
                                    'profile': 'nominal'})
        rows, states, trace = [], [], []
        layout_hash = hashlib.sha256(np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref]
            for h in env.world.env.humans], dtype=np.float64).tobytes()).hexdigest()
        for _ in range(140):
            states.append(np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in env.world.env.humans]))
            action = env.expert_action() if actor is None else actor.predict(obs, deterministic=True)[0]
            label, teacher_info = None, None
            if query_teacher:
                physical = np.array([env.world.robot.px,env.world.robot.py,env.world.robot.theta,
                                     env.world.robot.vx,env.world.robot.vy,env.world.env.global_time])
                planner = getattr(env.world,'_cem_teacher',None)
                label = env.expert_action().copy()
                teacher_info = dict(env.world.teacher_diagnostics)
                assert planner is None or planner is env.world._cem_teacher
                np.testing.assert_array_equal(physical, [env.world.robot.px,env.world.robot.py,
                    env.world.robot.theta,env.world.robot.vx,env.world.robot.vy,env.world.env.global_time])
                np.testing.assert_array_equal(states[-1], [[h.px,h.py,h.vx,h.vy,h.radius] for h in env.world.env.humans])
            if perturbation_seed is not None:
                # Safety data only: never use these actions as BC demonstrations.
                action = np.clip(action + perturbations.normal(0., [.3, .8]),
                                 env.action_space.low, env.action_space.high).astype(np.float32)
            before = [env.world.robot.px, env.world.robot.py, env.world.robot.theta]
            nxt, reward, done, _, info = policy_env.step(action)
            if diagnostics:
                trace.append(dict(robot=before, humans=states[-1].tolist(), action=action.tolist(),
                                  clearance=float(info['dmin']),
                                  actual_clearance=float(info['actual_clearance']), native_outcome=info['native_outcome'],
                                  planner=getattr(env.world, 'teacher_diagnostics', {})))
            if collect:
                rows.append(dict(observation=obs, next_observation=nxt, action=action,
                                 reward=reward, done=done, collision_cost=info['collision_cost']))
                if query_teacher:
                    np.testing.assert_array_equal(action, info['action'])
                    rows[-1].update(teacher_action=label, teacher_diagnostics=teacher_info)
            obs = nxt
            if done:
                info['episode_result']['layout_sha256'] = layout_hash
                if diagnostics:
                    info['episode_result']['trace'] = trace
                records.append(info['episode_result'])
                break
        else:
            raise AssertionError('140-step terminal contract failed')
        trajectories.append(rows)
        raw.append(np.asarray(states))
        if episode % 20 == 19:
            print('episodes', offset, episode+1, flush=True)
    env.close()
    return records, trajectories, raw


def collect_safety_replay(params, out, arm, actor=None):
    """30/30 outcome-stratified episodes per split, at most 300 attempts each.

    Positive labels describe a collision-seeking continuation, not calibrated
    collision probabilities of the student. Only its intervention suffix is kept.
    Simulator geometry is used solely by the safety-data intervention.
    """
    if arm != 'no_belief':
        raise ValueError('Directed safety collection is restricted to no_belief')
    env = ActionHistory(BeliefEnv(params, arm=arm, seed=2407), route=True)
    if actor is None:
        actor = BayesSetTD3.load(Path('repair_results/student_dagger_coverage_20260915/round16.zip'),
                                env=env, device='cuda')
    actor.check_arm(arm)
    if actor.observation_space['robot'].shape != (10,):
        raise ValueError('Safety student must have the 10D route contract')
    out.mkdir(exist_ok=False)
    result = dict(protocol='directed_suffix_v1', target_per_class=30, max_attempts_per_split=300,
                  trigger_surface_clearance=1., intervention_steps=15,
                  use='cost_critic_only_not_BC_DAgger_or_reward_replay',
                  scope='Outcome-stratified intervention data, not deployment calibration')
    hashes = set()
    try:
        for split, base in [('train', 40000), ('validation', 50000)]:
            records, trajectories, attempts = [], [], []
            positive = negative = 0
            for trial in range(300):
                if positive == negative == 30:
                    break
                # Alternate while both quotas are open, then finish the missing class.
                seek = negative == 30 or (positive < 30 and trial % 2 == 1)
                case = base + trial
                obs, _ = env.reset(options=dict(layout_seed=540000+case, test_case=case, profile='nominal'))
                world = env.unwrapped.world
                layout_hash = hashlib.sha256(np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref]
                    for h in world.env.humans], dtype=np.float64).tobytes()).hexdigest()
                if layout_hash in hashes:
                    raise ValueError('Repeated physical layout in safety collection')
                hashes.add(layout_hash)
                rows, switched, outcome, terminal = [], None, None, None
                for step in range(140):
                    robot = world.robot
                    closest = min(world.env.humans, key=lambda h:
                        np.hypot(h.px-robot.px,h.py-robot.py)-h.radius-robot.radius)
                    clearance = np.hypot(closest.px-robot.px,closest.py-robot.py)-closest.radius-robot.radius
                    if seek and switched is None and clearance < 1.:
                        switched = step
                        rows = []
                    intervention = switched is not None
                    if intervention:
                        target = np.arctan2(closest.py-robot.py, closest.px-robot.px)
                        error = (target-robot.theta+np.pi) % (2*np.pi)-np.pi
                        action = np.array([1.,np.clip(error/world.env.time_step,-1.2,1.2)],dtype=np.float32)
                    else:
                        action = actor.predict(obs, deterministic=True)[0]
                    nxt, reward, done, truncated, info = env.step(action)
                    if not seek or intervention:
                        rows.append(dict(observation=obs, next_observation=nxt,
                            action=np.asarray(info['action'],dtype=np.float32).copy(),
                            reward=reward, done=done or truncated, collision_cost=info['collision_cost'],
                            safety_intervention=intervention, source_step=step))
                    obs = nxt
                    if done or truncated:
                        terminal = dict(info['episode_result'])
                        outcome = terminal['outcome']
                        break
                    if intervention and len(rows) == 15:
                        break
                accepted = (seek and outcome=='collision' and switched is not None) or (
                    not seek and outcome in ('success','timeout'))
                attempts.append(dict(test_case=case,layout_seed=540000+case,layout_sha256=layout_hash,
                    seek_collision=seek,switch_step=switched,outcome=outcome,accepted=accepted,
                    stored_steps=len(rows) if accepted else 0))
                if accepted:
                    if seek:
                        assert 1 <= len(rows) <= 15 and all(r['safety_intervention'] for r in rows)
                        assert rows[0]['source_step'] == switched and rows[-1]['collision_cost'] == 1
                        positive += 1
                    else:
                        assert not any(r['collision_cost'] or r['safety_intervention'] for r in rows)
                        negative += 1
                    terminal.update(layout_sha256=layout_hash,safety_suffix_only=seek,
                                    switch_step=switched,stored_steps=len(rows))
                    records.append(terminal)
                    trajectories.append(rows)
                result[split] = dict(positive_trajectories=positive,negative_trajectories=negative,
                    attempts=trial+1,adequate=positive>=20 and negative>=20,
                    quota_complete=positive==negative==30)
                (out/'receipt.json').write_text(json.dumps(result,indent=2))
                if accepted or trial % 20 == 19:
                    print('SAFETY_DIRECTED',split,trial+1,positive,negative,flush=True)
            torch.save(dict(arm=arm,robot_fields=10,route_history=True,records=records,
                trajectories=trajectories,use=result['use'],protocol=result['protocol']),out/(split+'.pt'))
            (out/(split+'_attempts.json')).write_text(json.dumps(attempts,indent=2))
    finally:
        env.close()
    return result


def receipt(records):
    return dict(episodes=len(records), success_rate=np.mean([r['outcome']=='success' for r in records]),
                collision_rate=np.mean([r['outcome']=='collision' for r in records]))


def add_route_history(trajectories):
    for episode in trajectories:
        route = 0.
        for row in episode:
            if row['observation']['robot'].shape != (9,):
                raise ValueError('Expected previous-action demonstration contract')
            row['observation'] = dict(row['observation'], robot=np.append(row['observation']['robot'],route).astype(np.float32))
            route = .7*route + .3*float(row['action'][1])/1.2
            row['next_observation'] = dict(row['next_observation'], robot=np.append(row['next_observation']['robot'],route).astype(np.float32))


def migrate_actor(old, new):
    state = old.actor.state_dict()
    target = new.actor.state_dict()
    for key, value in state.items():
        if value.shape != target[key].shape:
            if not key.endswith('robot.0.weight') or target[key].shape != (value.shape[0],value.shape[1]+1):
                raise ValueError('Unexpected actor migration: '+key)
            target[key].zero_()
            target[key][:,:-1].copy_(value)
        else:
            target[key].copy_(value)
    new.actor.load_state_dict(target)
    new.actor_target.load_state_dict(target)


def dagger_artifact(folder, name):
    path = folder/name
    if path.exists():
        return path
    result = json.loads((folder/'results.json').read_text())
    parent = Path(result['extension_parent'])
    if hashlib.sha256((parent/'results.json').read_bytes()).hexdigest() != result['extension_parent_sha256']:
        raise ValueError('DAgger parent changed')
    return dagger_artifact(parent,name)


def dagger_rollout_worker(task):
    params, checkpoint, count, offset = task
    torch.set_num_threads(1)
    actor = BayesSetTD3.load(checkpoint, device='cpu')
    records, trajectories, _ = episodes(params,count,offset,actor,collect=True,
        case_offset=offset,arm='no_belief',query_teacher=True)
    return records, trajectories


def collect_dagger_parallel(params, checkpoint, count, offset, workers=4, chunk_size=25):
    tasks = [(params,checkpoint,min(chunk_size,count-i),offset+i) for i in range(0,count,chunk_size)]
    records, trajectories = [], []
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for rec, rows in pool.imap(dagger_rollout_worker,tasks):
            records.extend(rec)
            trajectories.extend(rows)
            print('DAgger collection',len(records),'/',count,flush=True)
    return records, trajectories


def dagger_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3, choices=[3,5,10,15,16])
    parser.add_argument('--resume-dagger', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if (args.rounds > 3) != (args.resume_dagger is not None):
        parser.error('Extensions require a completed previous DAgger queue')
    if args.epochs != 20:
        parser.error('Frozen first DAgger queue uses 20 epochs per round')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.2)
    args.out.mkdir(exist_ok=False)
    baseline = json.loads((args.stages/'results.json').read_text())
    if baseline['arm'] != 'no_belief' or not baseline['train_fit']['passed']:
        raise ValueError('DAgger requires the qualified-fit No-Belief BC starting checkpoint')
    for key,name in [('teacher_source_sha256','teacher.py'),('environment_source_sha256','environment.py')]:
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != baseline[key]:
            raise ValueError('Frozen teacher/environment changed')
    env = ActionHistory(BeliefEnv(args.params, arm='no_belief'), route=True)
    old = BayesSetTD3.load(args.stages/'bc_only.zip', device='cuda')
    model = BayesSetTD3('MultiInputPolicy',env,replay_buffer_class=CostReplay, buffer_size=50000,
        learning_rate=3e-4,batch_size=128,seed=2407,device='cuda',learning_starts=0,
        train_freq=(1,'step'),gradient_steps=1,policy_delay=2,tau=.005,gamma=.99,
        action_noise=NormalActionNoise(np.zeros(2),.1*np.ones(2)),
        policy_kwargs=dict(features_extractor_class=SetEncoder,features_extractor_kwargs=dict(features_dim=192),
                           net_arch=dict(pi=[256,256],qf=[96,96]),share_features_extractor=False))
    migrate_actor(old,model)
    model.belief_arm = 'no_belief'
    model.stage_metadata = dict(arm='no_belief',stage='dagger',robot_fields=10,route_decay=.7,train_humans=5)
    collection = torch.load(args.stages/'collection.pt', weights_only=False)
    add_route_history(collection['trajectories'])
    permanent = [row for rec,ep in zip(collection['records'],collection['trajectories']) if rec['outcome']=='success' for row in ep]
    migration_error = 0.
    for start in range(0,len(permanent),256):
        obs = stack_obs([r['observation'] for r in permanent[start:start+256]])
        old_obs = dict(obs,robot=obs['robot'][:,:9])
        migration_error = max(migration_error,float(np.max(np.abs(old.predict(old_obs,deterministic=True)[0]-model.predict(obs,deterministic=True)[0]))))
    if migration_error > 1e-5:
        raise AssertionError('Migration changed initial policy')
    report = dict(arm='no_belief',stage='dagger',rounds=[],rl_started=False,teacher_frozen=True,
        base_checkpoint_sha256=hashlib.sha256((args.stages/'bc_only.zip').read_bytes()).hexdigest(),
        teacher_source_sha256=baseline['teacher_source_sha256'], environment_source_sha256=baseline['environment_source_sha256'],
        protocol=dict(rounds=3,rollouts_per_round=100,epochs_per_round=20,seed=2407,
            validation_cases=[30000,30099],student_execution_probability=1.,route_decay=.7,
            optimizer='BC Adam reset once after input expansion; no critic updates',sampler='equal left/straight/right, omega threshold .2',
            loss='normalized Lv+2*Lomega',scope='five-human nominal development, not independent confirmation'),
        migration_max_action_difference=migration_error,original_demo_steps=len(permanent))
    def save():
        (args.out/'results.json').write_text(json.dumps(report,indent=2))
    save()
    del old
    start_round = 1
    if args.resume_dagger:
        parent_path = args.resume_dagger/'results.json'
        parent = json.loads(parent_path.read_text())
        completed = len(parent['rounds'])
        if (completed,args.rounds) not in ((3,5),(5,10),(10,15),(15,16)) or parent['base_checkpoint_sha256'] != report['base_checkpoint_sha256']:
            raise ValueError('Unexpected DAgger parent')
        for key in ('teacher_source_sha256','environment_source_sha256'):
            if parent[key] != report[key]:
                raise ValueError('DAgger parent uses different teacher/environment')
        model = BayesSetTD3.load(dagger_artifact(args.resume_dagger,f'round{completed}.zip'),env=env,device='cuda')
        model.check_arm('no_belief')
        for i in range(1,completed+1):
            prior = torch.load(dagger_artifact(args.resume_dagger,f'round{i}_collection.pt'),weights_only=False)
            permanent.extend(r for ep in prior['trajectories'] for r in ep)
        report = parent
        report.setdefault('initial_protocol',dict(parent['protocol']))
        report['protocol'] = dict(parent['protocol'],rounds=args.rounds,extension=f'{args.rounds-completed} additional rounds, same settings, no outcome filtering')
        report['extension_parent'] = str(args.resume_dagger.resolve())
        report['extension_parent_sha256'] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        report['extension_checkpoint_sha256'] = hashlib.sha256(dagger_artifact(args.resume_dagger,f'round{completed}.zip').read_bytes()).hexdigest()
        report['qualified'] = False
        report['decision'] = 'DAgger extension in progress'
        zero = report['round0']['records']
        start_round = completed+1
    else:
        zero,_,_ = episodes(args.params,100,530000,model,case_offset=30000,arm='no_belief',diagnostics=True)
        report['round0'] = dict(metrics=receipt(zero),records=zero)
    save()
    print('ROUND 0',receipt(zero),flush=True)
    validation_hashes = {r['layout_sha256'] for r in zero}
    seen = {r['layout_sha256'] for r in collection['records']} | validation_hashes
    for prior in report['rounds']:
        seen.update(r['layout_sha256'] for r in prior['rollout_records'])
    if args.rounds==16:
        excluded = set()
        ancestor = args.resume_dagger
        while ancestor is not None:
            if (ancestor/'independent_confirmation.json').exists():
                confirmation = json.loads((ancestor/'independent_confirmation.json').read_text())
                excluded.update(r['layout_sha256'] for r in confirmation['records'])
            previous = json.loads((ancestor/'results.json').read_text())
            ancestor = Path(previous['extension_parent']) if 'extension_parent' in previous else None
        if excluded & seen:
            raise AssertionError('Earlier confirmation overlaps training/development layouts')
        seen |= excluded
        report['excluded_confirmation_layouts'] = len(excluded)
    rng = np.random.default_rng(2407)
    if args.resume_dagger:
        # The sampling stream is part of the supervised optimizer checkpoint.
        if hasattr(model,'dagger_rng_state'):
            rng.bit_generator.state = model.dagger_rng_state
        else:
            for prior in report['rounds']:
                targets = np.stack([r.get('teacher_action',r['action']) for r in permanent[:prior['permanent_steps']]])
                groups = [np.flatnonzero(targets[:,1]<-.2),np.flatnonzero(np.abs(targets[:,1])<=.2),np.flatnonzero(targets[:,1]>.2)]
                for _ in range(prior['supervised_updates']):
                    idx = np.concatenate([rng.choice(g,n,replace=True) for g,n in zip(groups,[43,42,43])])
                    rng.shuffle(idx)
            report['sampler_resume'] = 'Reconstructed exact original seed/strata/update sequence'
    for round_id in range(start_round,args.rounds+1):
        rollout_count = 1000 if round_id==16 else 100
        if round_id==16:
            report['protocol']['large_coverage_round'] = dict(round=16,rollouts=1000,workers=4,
                inference_device='cpu',epochs=20,other_settings_unchanged=True)
            records,trajectories = collect_dagger_parallel(args.params,
                dagger_artifact(args.resume_dagger,'round15.zip'),rollout_count,826000)
        else:
            records,trajectories,_ = episodes(args.params,rollout_count,810000+round_id*1000,model,
                collect=True,case_offset=810000+round_id*1000,arm='no_belief',query_teacher=True)
        hashes = {r['layout_sha256'] for r in records}
        if len(hashes)!=rollout_count or hashes & seen:
            raise AssertionError('DAgger layouts overlap existing training/development data')
        seen |= hashes
        added = [r for ep in trajectories for r in ep]
        torch.save(dict(records=records,trajectories=trajectories,arm='no_belief',
                        contract='action=student execution; teacher_action=supervision only'),args.out/f'round{round_id}_collection.pt')
        permanent.extend(added)
        obs = stack_obs([r['observation'] for r in permanent])
        if np.any(obs['humans'][:,:,5:]):
            raise AssertionError('Belief entered No-Belief input')
        labels = np.stack([r.get('teacher_action',r['action']) for r in permanent])
        actions = model.policy.scale_action(labels).astype(np.float32)
        groups = [np.flatnonzero(labels[:,1]<-.2),np.flatnonzero(np.abs(labels[:,1])<=.2),np.flatnonzero(labels[:,1]>.2)]
        updates = args.epochs*int(np.ceil(len(permanent)/128))
        losses = []
        model.actor.train()
        for update in range(updates):
            idx = np.concatenate([rng.choice(g,n,replace=True) for g,n in zip(groups,[43,42,43])])
            rng.shuffle(idx)
            batch = obs_as_tensor({k:v[idx] for k,v in obs.items()},model.device)
            residual = (model.actor(batch)-torch.as_tensor(actions[idx],device=model.device)).square()
            loss = residual[:,0].mean()+2*residual[:,1].mean()
            model.actor.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(),10.)
            model.actor.optimizer.step()
            if (update+1)%500==0:
                losses.append(dict(update=update+1,weighted_loss=float(loss.detach())))
                print('DAGGER BC',round_id,losses[-1],flush=True)
        model.actor_target.load_state_dict(model.actor.state_dict())
        model.dagger_rng_state = rng.bit_generator.state
        model.save(args.out/f'round{round_id}')
        validation,_,_ = episodes(args.params,100,530000,model,case_offset=30000,arm='no_belief',diagnostics=True)
        if {r['layout_sha256'] for r in validation} != validation_hashes:
            raise AssertionError('Development layouts changed')
        report['rounds'].append(dict(round=round_id,rollout_count=rollout_count,rollout=receipt(records),rollout_records=records,
            added_steps=len(added),permanent_steps=len(permanent),original_steps_retained=report['original_demo_steps'],
            supervised_updates=updates,losses=losses,metrics=receipt(validation),records=validation,
            train_fit_diagnostic=fit_metrics(model,obs,labels),
            label_execution_difference_fraction=float(np.mean([np.max(np.abs(r['action']-r['teacher_action']))>1e-5 for r in added]))))
        save()
        print('ROUND',round_id,receipt(validation),flush=True)
    report['qualified'] = bool(report['rounds'][-1]['metrics']['success_rate']>=.9 and report['rounds'][-1]['metrics']['collision_rate']<=.02)
    report['decision'] = 'Fixed DAgger queue complete; TD3 and Bayes unchanged'
    save()
    env.close()


def fit_world_models(raw, out):
    # Both mode discovery and linear dynamics use absolute world coordinates.
    features = np.concatenate([x.reshape(-1, 5) for x in raw])
    np.random.seed(2407)
    gng = GNG(k_target=3)
    gng._feat_mean = features.mean(0)
    gng._feat_std = features.std(0)+1e-6
    standardized = (features-gng._feat_mean)/gng._feat_std
    rng = np.random.default_rng(2407)
    gng.nodes = standardized[rng.choice(len(features), 2, replace=False)].copy()
    gng.errors = np.zeros(2)
    for _ in range(3):
        for i in rng.permutation(len(features)):
            gng._gng_step(standardized[i])
    if gng.n_modes != 3:
        raise RuntimeError('Mode fitting did not produce three supported modes')
    pairs = [[] for _ in range(3)]
    counts = np.ones((3, 3))
    for trajectory in raw:
        for t in range(len(trajectory)-1):
            for i in range(5):
                x, y = trajectory[t, i], trajectory[t+1, i]
                k, j = gng.assign_mode(x), gng.assign_mode(y)
                pairs[k].append((x[:4], y[:4]))
                counts[k, j] += 1
    model = GDBN(3)
    model.Pi = counts/counts.sum(1, keepdims=True)
    for k in range(3):
        if len(pairs[k]) < 100:
            raise RuntimeError('Insufficient five-human training support')
        x, y = np.asarray(pairs[k]).transpose(1, 0, 2)
        model.A[k] = np.linalg.solve(x.T@x+1e-3*np.eye(4), x.T@y).T
        residual = y-x@model.A[k].T
        model.Q[k] = residual.T@residual/len(x)+1e-4*np.eye(4)
    out.mkdir()
    gng.save(str(out/'gng.npz'))
    model.save(str(out/'gdbn.npz'))
    (out/'provenance.json').write_text(json.dumps(dict(
        scenario='baseline_circle', humans=5, profile='nominal', layout_offset=520000,
        mode_features='world px,py,vx,vy,radius', dynamics='world px,py,vx,vy',
        coordinate_contract='No robot-relative feature assignment on this inference path',
        mode_support=[len(x) for x in pairs], model_R='fixed 0.01 I; not calibrated sensor noise',
        raw_sha256=hashlib.sha256(b''.join(x.tobytes() for x in raw)).hexdigest()), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--teacher-only', action='store_true')
    parser.add_argument('--arm', choices=ARMS, default='no_belief')
    parser.add_argument('--teacher-receipt', type=Path)
    parser.add_argument('--collection-dir', type=Path)
    parser.add_argument('--bc-updates', type=int, default=12000)
    parser.add_argument('--resume-bc', type=Path)
    parser.add_argument('--case-offset', type=int, default=0)
    parser.add_argument('--evaluation-episodes', type=int, default=100)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.resume_bc and not args.collection_dir:
        parser.error('--resume-bc requires the frozen --collection-dir')
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.2)
    result = {'protocol': 'Five-human development only, fixed stages, seed 2407',
              'gate': {'episodes':100, 'min_success':.9, 'max_collision':.02},
              'bc_updates':args.bc_updates, 'rl_started':False, 'arm':args.arm, 'stage':'bc_only_prev_action'}
    def save():
        (args.out/'results.json').write_text(json.dumps(result, indent=2))
    result['teacher_source_sha256'] = hashlib.sha256((Path(__file__).parent/'teacher.py').read_bytes()).hexdigest()
    result['environment_source_sha256'] = hashlib.sha256((Path(__file__).parent/'environment.py').read_bytes()).hexdigest()
    result['collision_rule'] = 'union of native detection and actual swept overlap'
    result['case_offset'] = args.case_offset
    if args.teacher_only:
        records, _, _ = episodes(args.params, args.evaluation_episodes, 510000,
                                 case_offset=args.case_offset, diagnostics=True)
        result['teacher_records'] = records
        result['teacher'] = receipt(records)
    else:
        if args.teacher_receipt is None:
            raise ValueError('BC requires frozen teacher qualification receipt')
        qualification = json.loads(args.teacher_receipt.read_text())
        qualified = qualification['runs']['teacher_cem_margin50_confirm200']
        if not qualification['final_qualified']:
            raise ValueError('Teacher not qualified')
        for key in ('teacher_source_sha256', 'environment_source_sha256'):
            if qualified[key] != result[key]:
                raise ValueError('Frozen teacher/environment hash mismatch')
        result['teacher'] = dict(episodes=qualified['episodes'], success_rate=qualified['success']/qualified['episodes'],
                                 collision_rate=qualified['collision']/qualified['episodes'])
        result['teacher_receipt_sha256'] = hashlib.sha256(args.teacher_receipt.read_bytes()).hexdigest()
    save()
    if args.teacher_only:
        return
    # Always retain the finite development collection, even if teacher qualification fails.
    params = args.params.resolve()
    if args.arm != 'no_belief':
        records, _, raw = episodes(params, 200, 520000, case_offset=20000)
        fit_world_models(raw, args.out/'fitted_params')
        torch.save(raw, args.out/'fit_trajectories.pt')
        params = (args.out/'fitted_params').resolve()
    result['params'] = str(params)
    result['bayes_fitted'] = args.arm != 'no_belief'
    if args.collection_dir:
        old = json.loads((args.collection_dir/'results.json').read_text())
        if old['arm'] != args.arm or args.arm != 'no_belief':
            raise ValueError('Reuse is restricted to matching No-Belief demonstrations')
        for key in ('teacher_source_sha256','environment_source_sha256'):
            if old[key] != result[key]:
                raise ValueError('Collection source mismatch')
        source = args.collection_dir/'collection.pt'
        collection = torch.load(source, weights_only=False)
        records, trajectories = collection['records'], collection['trajectories']
        result['collection_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        records, trajectories, _ = episodes(params, 200, 520000, collect=True, case_offset=20000, arm=args.arm)
    augment_collection(trajectories)
    result['neighbor_diagnostic'] = neighbor_conflicts(trajectories, records)
    print('neighbors',result['neighbor_diagnostic'],flush=True)
    torch.save(dict(records=records, trajectories=trajectories), args.out/'collection.pt')
    good = [row for record, rows in zip(records, trajectories) if record['outcome']=='success' for row in rows]
    result['accepted_bc_episodes'] = sum(r['outcome']=='success' for r in records)
    result['rejected_bc_episodes'] = 200-result['accepted_bc_episodes']
    result['collection_records'] = records
    env = ActionHistory(BeliefEnv(params, arm=args.arm))
    model = BayesSetTD3('MultiInputPolicy', env, replay_buffer_class=CostReplay,
        buffer_size=50000, learning_rate=3e-4, batch_size=128, seed=2407, device='cuda',
        learning_starts=0, train_freq=(1, 'step'), gradient_steps=1, policy_delay=2, tau=.005, gamma=.99,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=.1*np.ones(2)),
        policy_kwargs=dict(features_extractor_class=SetEncoder, features_extractor_kwargs=dict(features_dim=192),
                           net_arch=dict(pi=[256,256],qf=[96,96]), share_features_extractor=False))
    model.belief_arm = args.arm
    model.stage_metadata = dict(arm=args.arm, train_humans=5, stage='bc_only_prev_action', seed=2407,
                               previous_action_scale=[1.,1.2])
    completed = 0
    if args.resume_bc:
        previous_result = json.loads((args.resume_bc/'results.json').read_text())
        if previous_result.get('collection_source_sha256') != result.get('collection_source_sha256'):
            raise ValueError('Resume demonstration source changed')
        model = BayesSetTD3.load(args.resume_bc/'bc_only.zip', env=env, device='cuda')
        model.check_arm(args.arm)
        if model.stage_metadata.get('stage') != 'bc_only_prev_action' or model.actor_enabled:
            raise ValueError('Only previous-action BC checkpoints can resume')
        completed = previous_result['actual_bc_updates']
        result['resume_checkpoint_sha256'] = hashlib.sha256((args.resume_bc/'bc_only.zip').read_bytes()).hexdigest()
        result['resume_updates'] = completed
    result['td3_config'] = dict(learning_starts=0, train_freq=[1,'step'], gradient_steps=1,
        policy_delay=2, tau=.005, gamma=.99, normalized_action_noise_std=.1)
    observations = stack_obs([r['observation'] for r in good])
    if args.arm == 'no_belief' and np.any(observations['humans'][:,:,5:] != 0):
        raise AssertionError('Belief leaked into No-Belief BC')
    actions = model.policy.scale_action(np.stack([r['action'] for r in good])).astype(np.float32)
    physical_actions = np.stack([r['action'] for r in good])
    bins = [np.flatnonzero(physical_actions[:,1]<-.2),
            np.flatnonzero(np.abs(physical_actions[:,1])<=.2),np.flatnonzero(physical_actions[:,1]>.2)]
    if any(len(group)==0 for group in bins):
        raise ValueError('Missing turn stratum')
    result['sampling'] = dict(omega_threshold=.2, counts=[len(g) for g in bins], batch=[43,42,43], omega_loss_weight=2.)
    rng = np.random.default_rng(2407)
    result['bc_loss'] = []
    for _ in range(completed):
        indices = np.concatenate([rng.choice(group,size=n,replace=True) for group,n in zip(bins,[43,42,43])])
        rng.shuffle(indices)
    if completed >= args.bc_updates:
        raise ValueError('BC update limit must exceed completed updates')
    for update in range(completed,args.bc_updates):
        idx = np.concatenate([rng.choice(group,size=n,replace=True) for group,n in zip(bins,[43,42,43])])
        rng.shuffle(idx)
        batch = obs_as_tensor({k:v[idx] for k,v in observations.items()}, model.device)
        residual = (model.actor(batch)-torch.as_tensor(actions[idx], device=model.device)).square()
        loss = residual[:,0].mean()+2*residual[:,1].mean()
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.)
        model.actor.optimizer.step()
        if (update+1) % 500 == 0:
            result['bc_loss'].append(dict(update=update+1, mse=float(loss.detach())))
            print('BC', result['bc_loss'][-1], flush=True)
        if (update+1)%1000 == 0 or update+1 == args.bc_updates:
            result['train_fit'] = fit_metrics(model, observations, physical_actions)
            result['actual_bc_updates'] = update+1
            print('fit',update+1,result['train_fit'],flush=True)
            save()
            if result['train_fit']['passed']:
                break
    model.actor_target.load_state_dict(model.actor.state_dict())
    model.save(args.out/'bc_only')
    if not result['train_fit']['passed']:
        result['decision'] = 'Training fit gate failed; closed-loop evaluation and TD3 blocked'
        save()
        env.close()
        return
    restored = BayesSetTD3.load(args.out/'bc_only.zip', env=env)
    restored.check_arm(args.arm)
    assert np.array_equal(model.predict(good[0]['observation'], deterministic=True)[0],
                          restored.predict(good[0]['observation'], deterministic=True)[0])
    result['checkpoint_reload_exact'] = True
    records, _, _ = episodes(params, 100, 530000, restored, case_offset=30000, arm=args.arm, diagnostics=True)
    result['bc_records'], result['bc'] = records, receipt(records)
    train_hashes = {r['layout_sha256'] for r in result['collection_records']}
    val_hashes = {r['layout_sha256'] for r in records}
    assert len(train_hashes)==200 and len(val_hashes)==100 and not train_hashes & val_hashes
    result['split_audit'] = dict(train_unique=200, validation_unique=100, overlap=0)
    result['bc_passed'] = bool(result['bc']['success_rate'] >= .9 and result['bc']['collision_rate'] <= .02)
    result['decision'] = 'BC-only complete; no critic warm-up or TD3 started'
    save()
    print(result['teacher'], result['bc'], result['decision'], flush=True)
    env.close()


if __name__ == '__main__':
    import sys
    if '--dagger' in sys.argv:
        sys.argv.remove('--dagger')
        dagger_main()
    elif '--online' in sys.argv:
        sys.argv.remove('--online')
        online_main()
    else:
        main()
