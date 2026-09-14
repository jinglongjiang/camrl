"""Frozen five-human development stages; no high-density model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor
from crowd_nav.bayes_continuous.environment import BeliefEnv
from crowd_nav.bayes_continuous.algorithm import BayesSetTD3, CostReplay
from crowd_nav.bayes_continuous.network import SetEncoder
from crowd_nav.gdbn import GNG, GDBN


def stack_obs(observations):
    return {k:np.stack([o[k] for o in observations]) for k in observations[0]}


def supervised_warmup(model, trajectories, updates=1000):
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
    rng = np.random.default_rng(2407)
    before = {k:v.clone() for k,v in model.actor.state_dict().items()}
    def losses(data, indices):
        obs, act, returns, costs = data
        obs = obs_as_tensor({k:v[indices] for k,v in obs.items()}, model.device)
        act, returns, costs = [torch.as_tensor(v[indices], device=model.device) for v in (act, returns, costs)]
        qr = model.critic(obs, act)
        qc = model.cost_critic(obs, act)
        return (sum(torch.nn.functional.smooth_l1_loss(q, returns) for q in qr),
                sum(torch.nn.functional.mse_loss(q, costs) for q in qc))
    with torch.no_grad():
        initial = [float(x) for x in losses(valid, np.arange(len(valid[1])))]
    for _ in range(updates):
        reward_loss, cost_loss = losses(train, rng.integers(len(train[1]), size=128))
        for optimizer, loss in [(model.critic.optimizer, reward_loss), (model.cost_optimizer, cost_loss)]:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        final = [float(x) for x in losses(valid, np.arange(len(valid[1])))]
    assert all(torch.equal(before[k], v) for k,v in model.actor.state_dict().items())
    model.critic_target.load_state_dict(model.critic.state_dict())
    model.cost_target.load_state_dict(model.cost_critic.state_dict())
    model.warmup_updates = updates
    model.critic_validation = dict(before=initial, after=final, heldout_episodes=20,
                                  actor_unchanged=True,
                                  passed=bool(final[0] <= .2 and final[1] <= .04 and
                                              final[0] <= initial[0] and final[1] <= initial[1]))
    return model.critic_validation



def online_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--nonstationary-probability', type=float, default=0.)
    args = parser.parse_args()
    results = json.loads((args.stages/'results.json').read_text())
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
    env = BeliefEnv(args.stages/'fitted_params')
    env.nonstationary_probability = args.nonstationary_probability
    model = BayesSetTD3.load(args.stages/'supervised_warmup.zip', env=env)
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


def episodes(params, count, offset, actor=None, collect=False, case_offset=0, diagnostics=False):
    env = BeliefEnv(params, seed=2407)
    records, trajectories, raw = [], [], []
    for episode in range(count):
        obs, _ = env.reset(options={'layout_seed': offset+episode, 'test_case': case_offset+episode,
                                    'profile': 'nominal'})
        rows, states, trace = [], [], []
        layout_hash = hashlib.sha256(np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref]
            for h in env.world.env.humans], dtype=np.float64).tobytes()).hexdigest()
        for _ in range(140):
            states.append(np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in env.world.env.humans]))
            action = env.expert_action() if actor is None else actor.predict(obs, deterministic=True)[0]
            before = [env.world.robot.px, env.world.robot.py, env.world.robot.theta]
            nxt, reward, done, _, info = env.step(action)
            if diagnostics:
                trace.append(dict(robot=before, humans=states[-1].tolist(), action=action.tolist(),
                                  clearance=float(info['dmin']),
                                  actual_clearance=float(info['actual_clearance']), native_outcome=info['native_outcome'],
                                  planner=getattr(env.world, 'teacher_diagnostics', {})))
            if collect:
                rows.append(dict(observation=obs, next_observation=nxt, action=action,
                                 reward=reward, done=done, collision_cost=info['collision_cost']))
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


def receipt(records):
    return dict(episodes=len(records), success_rate=np.mean([r['outcome']=='success' for r in records]),
                collision_rate=np.mean([r['outcome']=='collision' for r in records]))


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
    parser.add_argument('--case-offset', type=int, default=0)
    parser.add_argument('--evaluation-episodes', type=int, default=100)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.2)
    result = {'protocol': 'Five-human development only, fixed stages, seed 2407',
              'gate': {'episodes':100, 'min_success':.9, 'max_collision':.02},
              'bc_updates':3000, 'rl_started':False}
    def save():
        (args.out/'results.json').write_text(json.dumps(result, indent=2))
    result['teacher_source_sha256'] = hashlib.sha256((Path(__file__).parent/'teacher.py').read_bytes()).hexdigest()
    result['environment_source_sha256'] = hashlib.sha256((Path(__file__).parent/'environment.py').read_bytes()).hexdigest()
    result['collision_rule'] = 'union of native detection and actual swept overlap'
    result['case_offset'] = args.case_offset
    records, _, _ = episodes(args.params, args.evaluation_episodes, 510000,
                             case_offset=args.case_offset, diagnostics=True)
    result['teacher_records'] = records
    result['teacher'] = receipt(records)
    save()
    if args.teacher_only:
        return
    # Always retain the finite development collection, even if teacher qualification fails.
    records, _, raw = episodes(args.params, 200, 520000, case_offset=20000)
    fit_world_models(raw, args.out/'fitted_params')
    torch.save(raw, args.out/'fit_trajectories.pt')
    params = args.out/'fitted_params'
    # Replay the fixed training layouts with the newly fitted filter, not stale beliefs.
    records, trajectories, _ = episodes(params, 200, 520000, collect=True, case_offset=20000)
    torch.save(dict(records=records, trajectories=trajectories), args.out/'collection.pt')
    good = [row for record, rows in zip(records, trajectories) if record['outcome']=='success' for row in rows]
    result['accepted_bc_episodes'] = sum(r['outcome']=='success' for r in records)
    result['rejected_bc_episodes'] = 200-result['accepted_bc_episodes']
    result['collection_records'] = records
    env = BeliefEnv(params)
    model = BayesSetTD3('MultiInputPolicy', env, replay_buffer_class=CostReplay,
        buffer_size=50000, learning_rate=3e-4, batch_size=128, seed=2407, device='cuda',
        policy_kwargs=dict(features_extractor_class=SetEncoder, net_arch=[96,96], share_features_extractor=False))
    observations = stack_obs([r['observation'] for r in good])
    actions = model.policy.scale_action(np.stack([r['action'] for r in good])).astype(np.float32)
    rng = np.random.default_rng(2407)
    for _ in range(3000):
        idx = rng.integers(len(good), size=128)
        batch = obs_as_tensor({k:v[idx] for k,v in observations.items()}, model.device)
        loss = torch.nn.functional.mse_loss(model.actor(batch), torch.as_tensor(actions[idx], device=model.device))
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.)
        model.actor.optimizer.step()
    model.actor_target.load_state_dict(model.actor.state_dict())
    model.save(args.out/'bc_only')
    records, _, _ = episodes(params, 100, 530000, model, case_offset=30000)
    result['bc_records'], result['bc'] = records, receipt(records)
    save()
    # Critic pretraining is allowed for diagnosis, but actor remains frozen.
    for rows in trajectories:
        for row in rows:
            model.replay_buffer.add({k:v[None] for k,v in row['observation'].items()},
                {k:v[None] for k,v in row['next_observation'].items()},
                model.policy.scale_action(row['action'][None]), np.array([row['reward']]),
                np.array([row['done']]), [{'collision_cost':row['collision_cost']}])
    before = {k:v.clone() for k,v in model.actor.state_dict().items()}
    result['critic_validation'] = supervised_warmup(model, trajectories)
    assert all(torch.equal(before[k], v) for k,v in model.actor.state_dict().items())
    result['warmup_actor_unchanged'] = True
    result['warmup_updates'] = model.warmup_updates
    (args.out/'losses.json').write_text(json.dumps(model.loss_history))
    model.save(args.out/'supervised_warmup')
    try:
        if result['teacher']['success_rate'] < .9 or result['teacher']['collision_rate'] > .02:
            raise RuntimeError('Teacher gate failed')
        model.enable_actor(result['bc'])
    except RuntimeError as exc:
        result['decision'] = 'RL blocked: '+str(exc)
    else:
        # Separate explicit learner entry; never silently launch a long run here.
        result['decision'] = 'Teacher/BC/held-out critic gates passed; ready for bounded RL run'
    save()
    print(result['teacher'], result['bc'], result['decision'], flush=True)


if __name__ == '__main__':
    import sys
    if '--online' in sys.argv:
        sys.argv.remove('--online')
        online_main()
    else:
        main()
