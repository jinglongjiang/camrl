"""Frozen five-human development stages; no high-density model selection."""
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import TD3
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
    pos_idx = np.flatnonzero(safety_train[3].ravel() > .5)
    neg_idx = np.flatnonzero(safety_train[3].ravel() < .5)
    if not len(pos_idx) or not len(neg_idx):
        raise ValueError('Balanced cost warm-up requires both transition classes')
    for _ in range(updates):
        reward_loss = losses(train, rng.integers(len(train[1]), size=128))[0]
        indices = np.concatenate([rng.choice(pos_idx, 64, replace=True),
                                  rng.choice(neg_idx, 64, replace=True)])
        rng.shuffle(indices)
        obs, act, _, costs = safety_train
        batch_obs = obs_as_tensor({k:v[indices] for k,v in obs.items()}, model.device)
        batch_act = torch.as_tensor(act[indices], device=model.device)
        batch_costs = torch.as_tensor(costs[indices], device=model.device)
        cost_loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(q, batch_costs)
                        for q in model.cost_critic(batch_obs, batch_act))
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
                                  cost_training='64 positive + 64 negative with replacement; twin BCE logits',
                                  cost_validation='unchanged full validation transitions; twin sigmoid MSE sum',
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


def reward_critic_warmup(model, updates=1000):
    """Fixed-actor Bellman warm-up for native TD3, without cost or BC losses."""
    from stable_baselines3.common.utils import polyak_update
    if type(model) is not TD3 or hasattr(model, 'cost_critic'):
        raise TypeError('Reward warm-up requires native TD3')
    before = {k:v.clone() for k,v in model.actor.state_dict().items()}
    model.actor.requires_grad_(False)
    model.critic.set_training_mode(True)
    losses = []
    for step in range(updates):
        data = model.replay_buffer.sample(model.batch_size)
        with torch.no_grad():
            noise = torch.randn_like(data.actions)*model.target_policy_noise
            actions = (model.actor_target(data.next_observations)+noise.clamp(
                -model.target_noise_clip,model.target_noise_clip)).clamp(-1,1)
            next_q = torch.cat(model.critic_target(data.next_observations,actions),1).min(1,keepdim=True).values
            target = data.rewards+(1-data.dones)*model.gamma*next_q
        loss = sum(torch.nn.functional.mse_loss(q,target) for q in model.critic(data.observations,data.actions))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite reward critic warm-up loss')
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        polyak_update(model.critic.parameters(),model.critic_target.parameters(),model.tau)
        losses.append(float(loss.detach()))
        if (step+1)%200 == 0:
            print('REWARD_WARMUP',step+1,losses[-1],flush=True)
    assert all(torch.equal(before[k],v) for k,v in model.actor.state_dict().items())
    model.actor.requires_grad_(True)
    return dict(updates=updates,actor_unchanged=True,losses=losses,
                objective='native TD3 twin MSE Bellman target, frozen actor, target smoothing, Polyak .005')


class FineTuneTD3(TD3):
    """Native TD3 updates, with separate fixed Actor/Critic learning rates."""
    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        for group in self.actor.optimizer.param_groups:
            group['lr'] = 3e-5
        for group in self.critic.optimizer.param_groups:
            group['lr'] = 3e-4


def local_td3_finetune(args):
    import random
    from stable_baselines3.common.utils import polyak_update
    args.out.mkdir(exist_ok=False)
    report = dict(protocol='empty_replay_local_finetuning',seed=2407,arm='no_belief',
        train_humans=5,profile='nominal',robot_fields=10,buffer_size=10000,old_demo_steps=0,
        frozen_steps=3000,actor_lr=3e-5,critic_lr=3e-4,policy_delay=10,noise_std=.1,
        gamma=.99,tau=.005,cost_critic=False,bc_loss=False,reward_changed=False)
    def save(stage):
        report['stage'] = stage
        (args.out/'status.json').write_text(json.dumps(report,indent=2))
        print('LOCAL_TD3',stage,flush=True)
    env = None
    try:
        save('initializing')
        torch.set_num_threads(1)
        checkpoint = args.stages/'round16.zip'
        report['source_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if report['source_sha256'] != '634e8106713c0c316afc77bf345615d3a6a101789f82f182853356ab57dbc086':
            raise ValueError('Wrong source actor')
        env = ActionHistory(BeliefEnv(args.params,arm='no_belief',seed=2407),route=True)
        source = BayesSetTD3.load(checkpoint,device='cuda')
        model = FineTuneTD3('MultiInputPolicy',env,learning_rate=3e-4,buffer_size=10000,batch_size=128,
            learning_starts=0,train_freq=(1,'step'),gradient_steps=1,policy_delay=10,tau=.005,gamma=.99,
            action_noise=NormalActionNoise(np.zeros(2),.1*np.ones(2)),device='cuda',seed=2407,
            policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=192),net_arch=dict(pi=[256,256],qf=[96,96]),
                share_features_extractor=False))
        model.actor.load_state_dict(source.actor.state_dict())
        model.actor_target.load_state_dict(source.actor.state_dict())
        del source
        assert FineTuneTD3.train is TD3.train and not hasattr(model,'cost_critic')
        assert model.replay_buffer.size() == 0
        actor_before = {k:v.clone() for k,v in model.actor.state_dict().items()}
        model.actor.requires_grad_(False)
        obs,_ = env.reset(seed=2407)
        noise_rng = np.random.default_rng(2407)
        losses,rows = [],[]
        save('frozen_actor_local_collection_3000')
        for step in range(3000):
            physical = model.predict(obs,deterministic=True)[0]
            normalized = np.clip(model.policy.scale_action(physical)+noise_rng.normal(0,.1,2),-1,1).astype(np.float32)
            action = model.policy.unscale_action(normalized)
            nxt,reward,done,truncated,info = env.step(action)
            model.replay_buffer.add({k:v[None] for k,v in obs.items()},
                {k:v[None] for k,v in nxt.items()},normalized[None],np.array([reward]),np.array([done or truncated]),[{}])
            rows.append(dict(observation=obs,next_observation=nxt,action=info['action'],
                             reward=reward,done=done or truncated))
            data = model.replay_buffer.sample(128)
            with torch.no_grad():
                noise = (torch.randn_like(data.actions)*model.target_policy_noise).clamp(
                    -model.target_noise_clip,model.target_noise_clip)
                next_actions = (model.actor_target(data.next_observations)+noise).clamp(-1,1)
                next_q = torch.cat(model.critic_target(data.next_observations,next_actions),1).min(1,keepdim=True).values
                target = data.rewards+(1-data.dones)*model.gamma*next_q
            loss = sum(torch.nn.functional.mse_loss(q,target) for q in model.critic(data.observations,data.actions))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite local critic loss')
            model.critic.optimizer.zero_grad(); loss.backward(); model.critic.optimizer.step()
            polyak_update(model.critic.parameters(),model.critic_target.parameters(),model.tau)
            losses.append(float(loss.detach()))
            obs = nxt
            if done or truncated:
                obs,_ = env.reset()
            if (step+1)%500 == 0:
                print('LOCAL_FROZEN',step+1,losses[-1],flush=True)
        assert all(torch.equal(actor_before[k],v) for k,v in model.actor.state_dict().items())
        report['frozen_actor_unchanged'] = True
        report['frozen_critic_losses'] = losses
        report['frozen_episode_records'] = list(env.unwrapped.episode_records)
        report['replay_after_frozen'] = model.replay_buffer.size()
        torch.save(rows,args.out/'frozen_transitions.pt')
        model.actor.requires_grad_(True)
        # Save a 500-state, speed-only Q slice without modifying any training RNG.
        ids = np.random.default_rng(9507).choice(len(rows),500,replace=False)
        probe_obs = stack_obs([rows[i]['observation'] for i in ids])
        acts = model.predict(probe_obs,deterministic=True)[0]
        values = []
        with torch.no_grad():
            tensor_obs = obs_as_tensor(probe_obs,model.device)
            for speed in np.linspace(0,1,11):
                candidate = acts.copy(); candidate[:,0] = speed
                tensor_act = torch.as_tensor(model.policy.scale_action(candidate),device=model.device)
                values.append(torch.cat(model.critic(tensor_obs,tensor_act),1).min(1).values.cpu().numpy())
        values = np.stack(values,1)
        winners = values.argmax(1)
        report['speed_probe'] = dict(states=500,argmax_counts=np.bincount(winners,minlength=11).tolist(),
            full_speed_fraction=float((winners==10).mean()),
            scope='Critic preference only, not counterfactual return or proof of gradient error')
        (args.out/'speed_probe.json').write_text(json.dumps(dict(row_indices=ids.tolist(),
            actor_actions=acts.tolist(),speeds=np.linspace(0,1,11).tolist(),q_min=values.tolist())))
        del rows
        model.save(args.out/'frozen_actor_3000')
        def evaluate():
            states = random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()
            try:
                rec,_,_ = episodes(args.params,100,530000,actor=model,case_offset=30000,arm='no_belief')
                return rec
            finally:
                random.setstate(states[0]); np.random.set_state(states[1])
                torch.set_rng_state(states[2]); torch.cuda.set_rng_state_all(states[3])
        save('frozen_baseline_evaluation')
        baseline = evaluate()
        report['baseline'] = receipt(baseline)
        (args.out/'baseline_episodes.json').write_text(json.dumps(baseline))
        report['evaluations'] = []
        for step in (1000,2000):
            save('finetune_to_'+str(step))
            model.learn(total_timesteps=1000,reset_num_timesteps=step==1000)
            assert all(g['lr']==3e-5 for g in model.actor.optimizer.param_groups)
            assert all(g['lr']==3e-4 for g in model.critic.optimizer.param_groups)
            model.save(args.out/('rl_'+str(step)))
            rec = evaluate()
            report['evaluations'].append(dict(finetune_steps=model.num_timesteps,
                total_env_steps=3000+model.num_timesteps,**receipt(rec)))
            (args.out/('evaluation_'+str(step)+'.json')).write_text(json.dumps(rec))
            (args.out/'training_episodes.json').write_text(json.dumps(env.unwrapped.episode_records))
            if report['evaluations'][-1]['success_rate'] < .8:
                save('stopped_below_80_percent')
                return
        save('completed_2000_finetuning')
    except Exception as exc:
        report['error'] = repr(exc); save('failed'); raise
    finally:
        if env is not None:
            env.close()


def native_td3_main():
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',type=Path,default=Path('repair_results/params'))
    parser.add_argument('--stages',type=Path,default=Path('repair_results/student_dagger_coverage_20260915'))
    parser.add_argument('--demos',type=Path,default=Path('repair_results/student_bc_history_continued_20260914'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--finetune',action='store_true')
    args = parser.parse_args()
    if args.finetune:
        return local_td3_finetune(args)
    args.out.mkdir(exist_ok=False)
    report = dict(algorithm='stable_baselines3.TD3',arm='no_belief',seed=2407,train_humans=5,
                  profile='nominal',robot_fields=10,cost_critic=False,bc_loss=False,
                  reward_changed=False,requested_steps=2000,diagnostic_only=True)
    def save(stage):
        report['stage'] = stage
        (args.out/'status.json').write_text(json.dumps(report,indent=2))
        print('NATIVE_TD3',stage,flush=True)
    env = None
    try:
        save('initializing')
        torch.set_num_threads(1)
        checkpoint = args.stages/'round16.zip'
        report['actor_source_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if report['actor_source_sha256'] != '634e8106713c0c316afc77bf345615d3a6a101789f82f182853356ab57dbc086':
            raise ValueError('Unexpected source actor')
        env = ActionHistory(BeliefEnv(args.params,arm='no_belief',seed=2407),route=True)
        source = BayesSetTD3.load(checkpoint,device='cuda')
        model = TD3('MultiInputPolicy',env,learning_rate=3e-4,buffer_size=50000,batch_size=128,
            learning_starts=0,train_freq=(1,'step'),gradient_steps=1,policy_delay=2,tau=.005,gamma=.99,
            action_noise=NormalActionNoise(np.zeros(2),.1*np.ones(2)),device='cuda',seed=2407,
            policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=192),
                net_arch=dict(pi=[256,256],qf=[96,96]),share_features_extractor=False))
        model.actor.load_state_dict(source.actor.state_dict())
        model.actor_target.load_state_dict(source.actor.state_dict())
        obs,_ = env.reset(seed=2407)
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],source.predict(obs,deterministic=True)[0])
        assert all(torch.equal(v,source.actor.state_dict()[k]) for k,v in model.actor.state_dict().items())
        assert not hasattr(model,'cost_critic') and type(model) is TD3
        report['actor_transfer_exact'] = True
        del source
        teacher = torch.load(args.demos/'collection.pt',weights_only=False)
        add_route_history(teacher['trajectories'])
        student = torch.load(args.stages/'round16_collection.pt',weights_only=False)
        if len(student['trajectories']) != 1000 or student['arm'] != 'no_belief':
            raise ValueError('Expected round16 1000 student rollouts')
        teacher_rows = [r for rec,ep in zip(teacher['records'],teacher['trajectories'])
                        if rec['outcome']=='success' for r in ep]
        student_rows = [r for ep in student['trajectories'] for r in ep]
        all_rows = teacher_rows+student_rows
        # Randomize insertion so a finite replay does not discard only the teacher prefix.
        order = np.random.default_rng(2407).permutation(len(all_rows))
        for i in order:
            row = all_rows[i]
            if row['observation']['robot'].shape != (10,) or row['next_observation']['robot'].shape != (10,):
                raise ValueError('Replay observation mismatch')
            action = np.asarray(row['action'],np.float32)
            if not env.action_space.contains(action):
                raise ValueError('Invalid executed replay action')
            model.replay_buffer.add({k:v[None] for k,v in row['observation'].items()},
                {k:v[None] for k,v in row['next_observation'].items()},
                model.policy.scale_action(action[None]),np.array([row['reward']]),
                np.array([row['done']]),[{}])
        retained = order[-50000:]
        report['replay'] = dict(teacher_success_steps=len(teacher_rows),student_actual_steps=len(student_rows),
            size=model.replay_buffer.size(),retained_teacher_steps=int((retained<len(teacher_rows)).sum()),
            retained_student_steps=int((retained>=len(teacher_rows)).sum()),action_field='action, never teacher_action',
            capacity=50000,source_student_episodes=1000)
        del teacher,student,teacher_rows,student_rows,all_rows
        save('reward_warmup_1000')
        report['warmup'] = reward_critic_warmup(model,1000)
        model.save(args.out/'reward_warmup')
        def evaluate():
            states = random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()
            try:
                rec,_,_ = episodes(args.params,100,530000,actor=model,case_offset=30000,arm='no_belief')
                return rec
            finally:
                random.setstate(states[0]); np.random.set_state(states[1])
                torch.set_rng_state(states[2]); torch.cuda.set_rng_state_all(states[3])
        save('baseline_evaluation')
        baseline = evaluate()
        report['baseline'] = receipt(baseline)
        (args.out/'baseline_episodes.json').write_text(json.dumps(baseline))
        report['evaluations'] = []
        for step in (1000,2000):
            save('td3_to_'+str(step))
            model.learn(total_timesteps=1000,reset_num_timesteps=step==1000)
            model.save(args.out/('rl_'+str(step)))
            rec = evaluate()
            report['evaluations'].append(dict(steps=model.num_timesteps,**receipt(rec)))
            (args.out/('evaluation_'+str(step)+'.json')).write_text(json.dumps(rec))
            (args.out/'training_episodes.json').write_text(json.dumps(env.unwrapped.episode_records))
            if report['evaluations'][-1]['success_rate'] < .8:
                save('stopped_below_80_percent')
                return
        save('completed_2000')
    except Exception as exc:
        report['error'] = repr(exc)
        save('failed')
        raise
    finally:
        if env is not None:
            env.close()


def audit_physical_state(env):
    world = env.unwrapped.world
    fields = ('px','py','vx','vy','theta','gx','gy','radius','v_pref')
    return np.asarray([[float(getattr(a,k,0.)) for k in fields]
        for a in [world.robot,*world.env.humans]] +
        [[world.env.global_time,env.unwrapped.elapsed,env.route,*env.previous,0,0,0,0]],dtype=np.float64)


def critic_counterfactual_worker(task):
    """Fixed reference policy; reset and replay every candidate, without snapshot shortcuts."""
    params,checkpoint,case = task
    torch.set_num_threads(1)
    actor = TD3.load(checkpoint,device='cpu')
    env = ActionHistory(BeliefEnv(params,arm='no_belief',seed=2407),route=True)
    options = dict(layout_seed=700000+case,test_case=case,profile='nominal')
    obs,_ = env.reset(options=options)
    history = []
    for _ in range(140):
        action = actor.predict(obs,deterministic=True)[0]
        history.append(dict(observation=obs,physical=audit_physical_state(env),action=action.copy()))
        obs,_,done,truncated,_ = env.step(action)
        if done or truncated:
            break
    selected = sorted(set([len(history)//3,2*len(history)//3]))
    outputs = []
    for t in selected:
        node = history[t]
        prefix = [x['action'] for x in history[:t]]
        a0 = node['action']
        candidates = [np.clip(a0+np.array([dv,dw]),env.action_space.low,env.action_space.high).astype(np.float32)
                      for dv in (-.15,0,.15) for dw in (-.3,0,.3)]
        candidates.append(np.array([1.,a0[1]],np.float32))
        returns,outcomes,lengths = [],[],[]
        cache = {}
        for candidate in candidates:
            key = candidate.tobytes()
            if key not in cache:
                current,_ = env.reset(options=options)
                for a in prefix:
                    current,_,done,truncated,_ = env.step(a)
                    assert not done and not truncated
                np.testing.assert_array_equal(audit_physical_state(env),node['physical'])
                for k in current:
                    np.testing.assert_array_equal(current[k],node['observation'][k])
                total,discount = 0.,1.
                action = candidate
                for n in range(140-t):
                    current,reward,done,truncated,info = env.step(action)
                    total += discount*reward; discount *= .99
                    if done or truncated:
                        cache[key] = (total,info['episode_result']['outcome'],n+1)
                        break
                    action = actor.predict(current,deterministic=True)[0]
                else:
                    raise AssertionError('Counterfactual failed to terminate')
            g,o,n = cache[key]; returns.append(g);outcomes.append(o);lengths.append(n)
        outputs.append(dict(case=case,layout_seed=options['layout_seed'],step=t,
            prefix_actions=prefix,observation=node['observation'],physical=node['physical'],
            candidates=np.stack(candidates),returns=np.asarray(returns),outcomes=outcomes,
            lengths=lengths,replay_exact=True))
    env.close()
    return outputs


def complete_mc_rows(rows):
    episodes_out,episode,excluded = [],[],0
    for row in rows:
        if row.get('episode_start') and episode:
            excluded += len(episode)
            episode = []
        episode.append(row)
        if row['done']:
            value = 0.
            for remaining,r in enumerate(reversed(episode),1):
                value = float(r['reward'])+.99*value
                r = dict(r,mc_return=value,remaining=remaining,
                         mc_outcome=episode[-1].get('outcome','success' if episode[-1]['reward'] > 0 else 'failure'))
                episodes_out.append(r)
            episode = []
    return episodes_out,excluded+len(episode)


def truth_critic_train(model,rows,updates,mode='td'):
    from stable_baselines3.common.utils import polyak_update
    selected = complete_mc_rows(rows)[0] if mode=='mc' else rows
    obs = stack_obs([r['observation'] for r in selected])
    nxt = stack_obs([r['next_observation'] for r in selected])
    actions = model.policy.scale_action(np.stack([r['action'] for r in selected]))
    rewards = np.asarray([r['reward'] for r in selected],np.float32)[:,None]
    dones = np.asarray([r['done'] for r in selected],np.float32)[:,None]
    returns = np.asarray([r.get('mc_return',0.) for r in selected],np.float32)[:,None]
    before = {k:v.clone() for k,v in model.actor.state_dict().items()}
    model.actor.requires_grad_(False)
    model.critic.set_training_mode(True)
    losses = []
    for step in range(updates):
        ids = np.random.randint(len(selected),size=128)
        x = obs_as_tensor({k:v[ids] for k,v in obs.items()},model.device)
        a = torch.as_tensor(actions[ids],device=model.device)
        with torch.no_grad():
            if mode=='mc':
                target = torch.as_tensor(returns[ids],device=model.device)
            else:
                xp = obs_as_tensor({k:v[ids] for k,v in nxt.items()},model.device)
                noise = (torch.randn_like(a)*.2).clamp(-.5,.5)
                ap = (model.actor_target(xp)+noise).clamp(-1,1)
                q = torch.cat(model.critic_target(xp,ap),1).min(1,keepdim=True).values
                target = torch.as_tensor(rewards[ids],device=model.device)+(1-torch.as_tensor(dones[ids],device=model.device))*.99*q
        loss = sum(torch.nn.functional.mse_loss(q,target) for q in model.critic(x,a))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite critic-only loss')
        model.critic.optimizer.zero_grad(); loss.backward(); model.critic.optimizer.step()
        polyak_update(model.critic.parameters(),model.critic_target.parameters(),.005)
        if step%500==0:
            losses.append(float(loss.detach()));print('CRITIC_ONLY',mode,step,float(loss.detach()),flush=True)
    if mode=='mc':
        model.critic_target.load_state_dict(model.critic.state_dict())
    assert all(torch.equal(before[k],v) for k,v in model.actor.state_dict().items())
    return dict(mode=mode,updates=updates,steps_available=len(selected),loss_samples=losses,actor_unchanged=True)


def collect_local_coverage(model,env,count,noise_rng,start_case):
    rows,records = [],[]
    case = start_case
    obs,_ = env.reset(options=dict(test_case=case,layout_seed=900000+case,profile='nominal'))
    first = True
    for _ in range(count):
        action = model.predict(obs,deterministic=True)[0]
        sigma = noise_rng.choice([.05,.15,.30],p=[.5,.3,.2])
        noisy = np.clip(model.policy.scale_action(action)+noise_rng.normal(0,sigma,2),-1,1).astype(np.float32)
        actual = model.policy.unscale_action(noisy)
        nxt,reward,done,truncated,info = env.step(actual)
        rows.append(dict(observation=obs,next_observation=nxt,action=info['action'],reward=reward,
            done=done or truncated,episode_start=first,noise_sigma=float(sigma),test_case=case,
            outcome=info['episode_result']['outcome'] if done or truncated else None))
        obs = nxt; first=False
        if done or truncated:
            records.append(info['episode_result']); case+=1
            obs,_ = env.reset(options=dict(test_case=case,layout_seed=900000+case,profile='nominal'))
            first=True
    return rows,records,case+1


def evaluate_critic_truth(model,rows,counterfactuals):
    from scipy.stats import pearsonr,spearmanr
    complete,partial = complete_mc_rows(rows)
    def q_values(obs,acts):
        outputs = []
        with torch.no_grad():
            for i in range(0,len(acts),256):
                x = obs_as_tensor({k:v[i:i+256] for k,v in obs.items()},model.device)
                a = torch.as_tensor(model.policy.scale_action(acts[i:i+256]),device=model.device)
                outputs.append(torch.cat(model.critic(x,a),1).min(1).values.cpu().numpy())
        return np.concatenate(outputs)
    def corr(x,y,fn):
        return float(fn(x,y).statistic) if len(x)>1 and np.std(x)>1e-10 and np.std(y)>1e-10 else None
    obs = stack_obs([r['observation'] for r in complete])
    q = q_values(obs,np.stack([r['action'] for r in complete]))
    g = np.asarray([r['mc_return'] for r in complete])
    def stats(mask):
        x,y = q[mask],g[mask]
        return dict(n=len(x),mae=float(np.abs(x-y).mean()),rmse=float(np.sqrt(np.mean((x-y)**2))),
            bias=float((x-y).mean()),pearson=corr(x,y,pearsonr),spearman=corr(x,y,spearmanr)) if len(x) else dict(n=0)
    remaining = np.asarray([r['remaining'] for r in complete])
    a = dict(all=stats(np.ones(len(g),bool)),by_outcome={k:stats(np.asarray([r['mc_outcome']==k for r in complete]))
        for k in ('success','collision','timeout','failure')},by_remaining={str((lo,hi)):stats((remaining>=lo)&(remaining<=hi))
        for lo,hi in ((1,10),(11,40),(41,80),(81,140))},excluded_partial_steps=partial,
        scope='Behavior-policy realized return, not exact deterministic-reference Q; in-sample diagnostic')
    results = []
    for node in counterfactuals:
        values = q_values({k:np.repeat(v[None],10,axis=0) for k,v in node['observation'].items()},node['candidates'])
        truth = node['returns']
        pairs = [(i,j) for i in range(10) for j in range(i+1,10)
                 if not np.array_equal(node['candidates'][i],node['candidates'][j]) and abs(truth[i]-truth[j])>1e-6]
        correct = sum(1. if (values[i]-values[j])*(truth[i]-truth[j])>0 else
                      (.5 if abs(values[i]-values[j])<=1e-8 else 0.) for i,j in pairs)
        pick = int(values.argmax())
        rho = corr(values,truth,spearmanr)
        results.append(dict(case=node['case'],step=node['step'],q=values.tolist(),returns=truth.tolist(),
            correct=correct,pairs=len(pairs),spearman=rho,regret=float(truth.max()-truth[pick]),
            selected_minus_actor=float(truth[pick]-truth[4]),
            top1=bool(truth[pick]>=truth.max()-1e-6),q_full_speed=bool(values[9]>=values.max()-1e-8),
            true_full_speed=bool(truth[9]>=truth.max()-1e-6)))
    valid_rho = [x['spearman'] for x in results if x['spearman'] is not None]
    npairs = sum(x['pairs'] for x in results)
    b = dict(states=len(results),informative_pairs=npairs,
        pairwise_accuracy=sum(x['correct'] for x in results)/npairs if npairs else 0.,
        spearman=float(np.mean(valid_rho)) if valid_rho else -1.,
        defined_spearman_states=len(valid_rho),mean_regret=float(np.mean([x['regret'] for x in results])),
        mean_selected_minus_actor=float(np.mean([x['selected_minus_actor'] for x in results])),
        top1_agreement=float(np.mean([x['top1'] for x in results])),
        q_full_speed_fraction=float(np.mean([x['q_full_speed'] for x in results])),
        true_full_speed_fraction=float(np.mean([x['true_full_speed'] for x in results])))
    passed = a['all']['mae']<=.2 and a['all']['rmse']<=.3 and b['pairwise_accuracy']>=.7 and b['spearman']>=.5 and b['mean_selected_minus_actor']>=-.05
    return dict(A=a,B=b,passed=bool(passed),per_state=results)


def critic_audit_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',type=Path,default=Path('repair_results/params'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not args.resume:
        args.out.mkdir(exist_ok=False)
    report = dict(protocol='critic_truth_finite_tree',seed=2407,actor_fixed=True,
        A_limits=dict(mae=.2,rmse=.3),B_limits=dict(pairwise=.7,spearman=.5,selected_minus_actor=-.05),
        counterfactual_cases=[61000,61049],selection='one-third and two-thirds of frozen reference episode',
        branch_limits=dict(C1_updates=5000,C2_max_steps=20000,C3_mc=5000,C3_td=1000,C4_repeat_cap=20000),
        continuation='fixed round16, deterministic, original reward',stages=[])
    if args.resume:
        report = json.loads((args.out/'status.json').read_text())
    def save(stage):
        report['stage'] = stage
        (args.out/'status.json').write_text(json.dumps(report,indent=2))
        print('CRITIC_AUDIT',stage,flush=True)
    save('protocol_frozen')
    torch.set_num_threads(1)
    source = Path('repair_results/no_belief_local_finetune/frozen_actor_3000.zip')
    model = TD3.load(source,device='cuda')
    rows = torch.load('repair_results/no_belief_local_finetune/frozen_transitions.pt',weights_only=False)
    old_receipt = json.loads(Path('repair_results/no_belief_local_finetune/status.json').read_text())
    terminal_records = iter(old_receipt['frozen_episode_records'])
    for row in rows:
        if row['done']:
            row['outcome'] = next(terminal_records)['outcome']
    if args.resume:
        cf = torch.load(args.out/'counterfactuals.pt',weights_only=False)
    else:
        save('counterfactual_rollouts')
        tasks = [(args.params,source,case) for case in range(61000,61050)]
        cf = []
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for nodes in pool.imap(critic_counterfactual_worker,tasks):
                cf.extend(nodes)
                print('COUNTERFACTUAL_STATES',len(cf),flush=True)
    assert len(cf)==100
    torch.save(cf,args.out/'counterfactuals.pt')
    def audit(name):
        result = evaluate_critic_truth(model,rows,cf)
        (args.out/(name+'.json')).write_text(json.dumps(result,indent=2))
        report['stages'].append(dict(name=name,passed=result['passed'],A=result['A']['all'],B=result['B']))
        save(name)
        return result['passed']
    if audit('A_B_initial'):
        save('branch_D_ready')
        return
    save('branch_C_required')
    audit_rows = rows
    reference = {k:v.clone() for k,v in model.actor.state_dict().items()}
    env = ActionHistory(BeliefEnv(args.params,arm='no_belief',seed=2407),route=True)
    def fresh(wide=False):
        m = TD3('MultiInputPolicy',env,learning_rate=1e-4 if wide else 3e-4,
            buffer_size=100,batch_size=128,device='cuda',seed=2407,
            policy_kwargs=dict(features_extractor_class=SetEncoder,features_extractor_kwargs=dict(features_dim=192),
                net_arch=dict(pi=[256,256],qf=[256,256] if wide else [96,96]),share_features_extractor=False))
        m.actor.load_state_dict(reference);m.actor_target.load_state_dict(reference)
        m.critic.features_extractor.load_state_dict(m.actor.features_extractor.state_dict())
        m.critic_target.load_state_dict(m.critic.state_dict())
        assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(m.actor.features_extractor.parameters(),m.critic.features_extractor.parameters()))
        return m
    def assess(name,train_result):
        report.setdefault('training',[]).append(dict(name=name,**train_result))
        # The A audit remains fixed on the original 3000-step data at every stage.
        result = evaluate_critic_truth(model,audit_rows,cf)
        (args.out/(name+'.json')).write_text(json.dumps(result,indent=2))
        torch.save(model.critic.state_dict(),args.out/(name+'_critic.pt'))
        report['stages'].append(dict(name=name,passed=result['passed'],A=result['A']['all'],B=result['B']))
        save(name)
        if result['passed']:
            model.save(args.out/'qualified_critic')
            save('branch_D_ready')
        return result['passed']
    model = fresh()
    if assess('C1_encoder_copy_5000',truth_critic_train(model,rows,5000)):
        env.close();return
    def coverage(prefix,start):
        data = list(audit_rows)
        records = []
        rng = np.random.default_rng(start)
        case = start
        for steps in (5000,10000,15000,20000):
            save(prefix+'_collect_'+str(steps))
            more,rec,case = collect_local_coverage(model,env,5000,rng,case)
            data.extend(more); records.extend(rec)
            report.setdefault('coverage',{})[prefix] = dict(steps=steps,completed=len(records),
                failures=sum(r['outcome']!='success' for r in records),records=records,
                training='fixed actor; collect 5000 then 5000 critic-only updates; no policy change between')
            if assess(prefix+'_'+str(steps),truth_critic_train(model,data,5000)):
                return data,True
            if steps>=10000 and len(records)>=200 and sum(r['outcome']!='success' for r in records)>=20:
                break
        return data,False
    data,passed = coverage('C2_coverage',70000)
    if passed:
        env.close();return
    save('C3_mc_initialization')
    mc = truth_critic_train(model,data,5000,mode='mc')
    td = truth_critic_train(model,data,1000)
    if assess('C3_mc5000_td1000',dict(mc=mc,td=td)):
        env.close();return
    model = fresh(wide=True)
    data,passed = coverage('C4_wide_coverage',80000)
    if passed:
        env.close();return
    save('C4_mc_initialization')
    mc = truth_critic_train(model,data,5000,mode='mc')
    td = truth_critic_train(model,data,1000)
    passed = assess('C4_wide_mc5000_td1000',dict(mc=mc,td=td))
    if not passed:
        save('exit3_critic_audit_failed_at_finite_cap')
    env.close()


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


def scratch_baseline_main():
    """Independent standard RL baselines; no transfer or auxiliary losses."""
    import random
    import time
    import traceback
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['td3', 'ppo'], required=True)
    parser.add_argument('--params', type=Path, default=Path('repair_results/params'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--eval-count', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.time()
    report = dict(algorithm=args.algo, seed=2407, arm='no_belief', humans=5,
        profile='nominal', robot_fields=10, requested_steps=args.steps,
        reward_changed=False, pretrained_weights=False, old_replay=False,
        auxiliary_losses=False, device=args.device, sb3_version=sb3.__version__,
        evaluations=[], status='initializing', single_seed_development_only=True,
        ppo_budget_note='Completes the last 2048-step rollout: 100000 requested gives 100352 actual steps.')
    def save():
        report['elapsed_seconds'] = time.time()-started
        tmp = args.out/'status.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.out/'status.json')
    class NominalEnv(BeliefEnv):
        def reset(self, *, seed=None, options=None):
            options = dict(options or {})
            options['profile'] = 'nominal'
            # Disjoint layout seeds from the fixed development evaluation.
            if 'layout_seed' not in options:
                options['layout_seed'] = 20000000 + int(self.rng.integers(10000000))
            return super().reset(seed=seed, options=options)
    env = ActionHistory(NominalEnv(args.params, arm='no_belief', seed=2407), route=True)
    model = None
    try:
        common = dict(features_extractor_class=SetEncoder,
            features_extractor_kwargs=dict(features_dim=192))
        if args.algo == 'td3':
            model = sb3.TD3('MultiInputPolicy', env, learning_rate=3e-4,
                buffer_size=100000, learning_starts=5000, batch_size=256,
                gamma=.99, tau=.005, policy_delay=2, train_freq=(1, 'step'),
                gradient_steps=1, action_noise=NormalActionNoise(np.zeros(2), .1*np.ones(2)),
                policy_kwargs=dict(common, net_arch=dict(pi=[256,256], qf=[96,96]),
                    share_features_extractor=False), seed=2407, device=args.device)
        else:
            model = sb3.PPO('MultiInputPolicy', env, learning_rate=3e-4,
                n_steps=2048, batch_size=256, n_epochs=10, gamma=.99,
                gae_lambda=.95, clip_range=.2,
                policy_kwargs=dict(common, net_arch=dict(pi=[256,256], vf=[96,96]),
                    share_features_extractor=False), seed=2407, device=args.device)
        assert model.observation_space['robot'].shape == (10,)
        assert not hasattr(model, 'cost_critic')
        report['policy_kwargs'] = str(model.policy_kwargs)
        report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        model.set_logger(configure(str(args.out), ['csv']))
        def evaluate(step):
            # A separate evaluation environment must not change training RNG streams.
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            training_mode = model.policy.training
            try:
                records, _, _ = episodes(args.params, args.eval_count, 910000,
                    actor=model, case_offset=81000, arm='no_belief')
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state_all(rng[3])
                model.policy.set_training_mode(training_mode)
            result = dict(steps=step, records=records,
                success=sum(r['outcome']=='success' for r in records),
                collision=sum(r['outcome']=='collision' for r in records),
                timeout=sum(r['outcome']=='timeout' for r in records))
            (args.out/f'eval_{step}.json').write_text(json.dumps(result, indent=2))
            report['evaluations'].append({k:v for k,v in result.items() if k!='records'})
            model.save(args.out/f'checkpoint_{step}')
            save()
            print('EVAL', report['evaluations'][-1], flush=True)
        class Progress(BaseCallback):
            def _on_step(self):
                step = self.num_timesteps
                for info in self.locals.get('infos', []):
                    if 'episode_result' in info:
                        with (args.out/'train_episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(info['episode_result'], global_step=step))+'\n')
                if step % 1000 == 0:
                    report.update(status='training', actual_steps=step)
                    save()
                    print('PROGRESS', args.algo, step, round(time.time()-started, 1), flush=True)
                if step % args.eval_every == 0:
                    evaluate(step)
                return True
        save()
        model.learn(total_timesteps=args.steps, callback=Progress(), log_interval=10)
        if not report['evaluations'] or report['evaluations'][-1]['steps'] != model.num_timesteps:
            evaluate(model.num_timesteps)
        report.update(status='complete', actual_steps=model.num_timesteps)
        save()
    except Exception:
        report.update(status='error', error=traceback.format_exc())
        save()
        raise
    finally:
        env.close()


def dagger_ppo_confirmation_worker(task):
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    params, checkpoint, destination = task
    model = PPO.load(checkpoint, device='cpu')
    records, _, _ = episodes(Path(params), 500, 73000000, actor=model,
                             case_offset=85000, arm='no_belief')
    Path(destination).write_text(json.dumps(dict(checkpoint=str(checkpoint), records=records), indent=2))
    return records


def dagger_ppo_confirmation_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, default=Path('repair_results/params'))
    parser.add_argument('--stages', type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.stages/'status.json').read_text())
    if status['status'] != 'complete_stable':
        raise ValueError('Only a completed fixed-budget candidate can enter confirmation')
    folder = args.stages/'confirmation500'
    folder.mkdir(exist_ok=False)
    protocol = dict(count=500, layout_seed_start=73000000, test_case_start=85000,
        candidate=status['candidate'], selection='last budget checkpoint, not best development score',
        no_retraining=True, paired=True)
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2))
    tasks = [(str(args.params), str(args.stages/(name+'.zip')), str(folder/(tag+'.json')))
             for name, tag in [('initial','initial'),(status['candidate'],'final')]]
    with multiprocessing.get_context('spawn').Pool(2) as pool:
        initial, final = pool.map(dagger_ppo_confirmation_worker, tasks)
    assert all(a['layout_sha256']==b['layout_sha256'] and a['test_case']==b['test_case']
               for a,b in zip(initial,final))
    protocol['outcomes'] = {tag:dict(success=sum(r['outcome']=='success' for r in rows),
        collision=sum(r['outcome']=='collision' for r in rows),timeout=sum(r['outcome']=='timeout' for r in rows),
        mean_return=float(np.mean([r['reward'] for r in rows]))) for tag,rows in [('initial',initial),('final',final)]}
    protocol['status']='complete'
    (folder/'result.json').write_text(json.dumps(protocol,indent=2))
    print('CONFIRMATION',protocol,flush=True)


def dagger_ppo_main():
    """Finite nominal PPO transfer test with exact physical-mean conversion."""
    import copy
    import random
    import time
    import traceback
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from crowd_nav.bayes_continuous.network import DaggerGaussianPolicy
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, default=Path('repair_results/params'))
    parser.add_argument('--source', type=Path, default=Path('repair_results/student_dagger_coverage_20260915/round16.zip'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.time()
    report = dict(status='initializing', arm='no_belief', profile='nominal', seed=2407,
        requested_steps_per_attempt=10240, max_attempts=2, reward_changed=False,
        initialization='exact deterministic function conversion, no new BC fit',
        distribution='physical Gaussian, tanh-bounded mean, ordinary SB3 action clipping',
        initial_std=[.03,.06], evaluations=[], single_seed_development_only=True)
    def save():
        report['elapsed_seconds'] = time.time()-started
        tmp=args.out/'status.tmp'
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(args.out/'status.json')
        print('STATUS',report['status'],round(report['elapsed_seconds'],1),flush=True)
    def make_env(seed):
        env=ActionHistory(BeliefEnv(args.params,arm='no_belief',seed=seed),route=True)
        assert env.unwrapped.nonstationary_probability == 0
        return env
    env=make_env(2407)
    try:
        report['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
        if report['source_sha256']!='634e8106713c0c316afc77bf345615d3a6a101789f82f182853356ab57dbc086':
            raise ValueError('Unexpected DAgger source')
        source=BayesSetTD3.load(args.source,device='cpu')
        model=PPO(DaggerGaussianPolicy,env,learning_rate=3e-5,n_steps=2048,batch_size=256,
            n_epochs=3,gamma=.99,gae_lambda=.95,clip_range=.1,ent_coef=0,target_kl=.01,
            seed=2407,device='cpu',policy_kwargs=dict(features_extractor_class=SetEncoder,
                features_extractor_kwargs=dict(features_dim=192),net_arch=dict(pi=[256,256],vf=[96,96]),
                activation_fn=torch.nn.ReLU,share_features_extractor=False))
        policy=model.policy
        policy.pi_features_extractor.load_state_dict(source.actor.features_extractor.state_dict())
        policy.mlp_extractor.policy_net.load_state_dict(source.actor.mu[:-2].state_dict())
        policy.action_net[0].load_state_dict(source.actor.mu[-2].state_dict())
        # Independent value encoder initialization; no action-Q or optimizer transfer.
        policy.vf_features_extractor.load_state_dict(source.actor.features_extractor.state_dict())
        with torch.no_grad():
            policy.log_std.copy_(torch.tensor(np.log([.03,.06]),dtype=torch.float32))
        max_error=0.
        for case in range(10):
            obs,_=env.reset(options=dict(layout_seed=920000+case,test_case=82000+case,profile='nominal'))
            for _ in range(100):
                a=source.predict(obs,deterministic=True)[0]
                b=model.predict(obs,deterministic=True)[0]
                max_error=max(max_error,float(np.max(np.abs(a-b))))
                np.testing.assert_allclose(a,b,rtol=0,atol=2e-6)
                obs,_,done,_,_=env.step(a)
                if done: break
        report['mean_conversion_max_error']=max_error
        model.save(args.out/'initial')
        restored=PPO.load(args.out/'initial.zip',device='cpu')
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],
                                      restored.predict(obs,deterministic=True)[0])
        del restored,source
        def evaluate(tag, deterministic):
            states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            testing=make_env(2407)
            records=[]
            try:
                torch.manual_seed(4807)
                for case in range(100):
                    obs,_=testing.reset(options=dict(layout_seed=910000+case,test_case=81000+case,profile='nominal'))
                    for _ in range(140):
                        action=model.predict(obs,deterministic=deterministic)[0]
                        obs,_,done,_,info=testing.step(action)
                        if done:
                            records.append(info['episode_result']); break
                    else: raise AssertionError('Terminal contract')
                    if case%25==24: print('EVAL',tag,case+1,flush=True)
            finally:
                testing.close(); random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            summary=dict(tag=tag,deterministic=deterministic,success=sum(r['outcome']=='success' for r in records),
                collision=sum(r['outcome']=='collision' for r in records),timeout=sum(r['outcome']=='timeout' for r in records),
                mean_return=float(np.mean([r['reward'] for r in records])))
            (args.out/(tag+'.json')).write_text(json.dumps(dict(summary=summary,records=records),indent=2))
            report['evaluations'].append(summary);save()
            return summary
        report['status']='initial_validation';save()
        deterministic=evaluate('initial_deterministic',True)
        stochastic=evaluate('initial_stochastic',False)
        if min(deterministic['success'],stochastic['success'])<90:
            report['status']='stopped_initial_policy_gate';save();return
        for attempt in range(2):
            if attempt:
                env.close();env=make_env(2407)
                model=PPO.load(args.out/'initial.zip',env=env,device='cpu')
                model.learning_rate=1e-5
                model.lr_schedule=lambda _:1e-5
                for parameter in model.policy.pi_features_extractor.parameters(): parameter.requires_grad_(False)
                report['fallback']='initial checkpoint, frozen actor SetEncoder, lr1e-5'
            model.set_logger(configure(str(args.out/f'attempt{attempt}'),['csv']))
            report['status']=f'fine_tuning_attempt{attempt}';save()
            failed=False
            for stage in range(1,6):
                model.learn(total_timesteps=2048,reset_num_timesteps=False,log_interval=1)
                tag=f'attempt{attempt}_{stage*2048}'
                model.save(args.out/tag)
                result=evaluate(tag,True)
                if result['success']<90:
                    failed=True
                    report['rejected_checkpoint']=tag
                    break
                model.save(args.out/'last_accepted')
            if not failed:
                final_stochastic=evaluate(f'attempt{attempt}_final_stochastic',False)
                report['status']='complete_stable' if final_stochastic['success']>=90 else 'stopped_final_sampling_gate'
                report['candidate']=tag
                save();return
        report['status']='stopped_both_finetune_attempts'
        report['retained_policy']='initial.zip; original DAgger unchanged'
        save()
    except Exception:
        report.update(status='error',error=traceback.format_exc());save();raise
    finally: env.close()


if __name__ == '__main__':
    import sys
    if '--dagger-ppo-confirm' in sys.argv:
        sys.argv.remove('--dagger-ppo-confirm')
        dagger_ppo_confirmation_main()
    elif '--dagger-ppo' in sys.argv:
        sys.argv.remove('--dagger-ppo')
        dagger_ppo_main()
    elif '--scratch-baseline' in sys.argv:
        sys.argv.remove('--scratch-baseline')
        scratch_baseline_main()
    elif '--critic-audit' in sys.argv:
        sys.argv.remove('--critic-audit')
        critic_audit_main()
    elif '--native-td3' in sys.argv:
        sys.argv.remove('--native-td3')
        native_td3_main()
    elif '--dagger' in sys.argv:
        sys.argv.remove('--dagger')
        dagger_main()
    elif '--online' in sys.argv:
        sys.argv.remove('--online')
        online_main()
    else:
        main()
