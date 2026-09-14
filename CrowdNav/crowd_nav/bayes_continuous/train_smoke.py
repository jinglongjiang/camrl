"""BC + SB3 TD3 with a decaying online BC regularizer; smoke evidence only."""
import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import obs_as_tensor

from crowd_nav.bayes_continuous.environment import BeliefEnv, ARMS, transform_observation
from crowd_nav.bayes_continuous.network import SetEncoder


def stack_obs(observations):
    return {key: np.stack([o[key] for o in observations]) for key in observations[0]}


def collect(params, episodes):
    env = BeliefEnv(params, seed=2407)
    rows, records = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=4000+episode,
                           options={'layout_seed': 40000+episode, 'test_case': 100+episode})
        for _ in range(160):
            action = env.expert_action()
            nxt, reward, done, _, info = env.step(action)
            rows.append(dict(observation=obs, action=action.copy(), reward=reward,
                             next_observation=nxt, done=done, episode=episode))
            obs = nxt
            if done:
                records.append(info['episode_result'])
                break
        else:
            raise AssertionError('Environment failed to terminate at its time limit')
    env.close()
    return rows, records


def mse(model, observations, actions):
    with torch.no_grad():
        predicted = model.actor(obs_as_tensor(observations, model.device))
        return float(F.mse_loss(predicted, torch.as_tensor(actions, device=model.device)))


def evaluate(model, params, arm):
    records = []
    for index, scenario in enumerate(('baseline_circle', 'dense_circle', 'dense_square')):
        env = BeliefEnv(params, arm, scenario, training=False, seed=8107)
        obs, _ = env.reset(options={'layout_seed': 810000+index, 'test_case': 800+index})
        for _ in range(160):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            if done:
                records.append(dict(scenario=scenario, humans=len(env.world.env.humans), **info['episode_result']))
                break
        else:
            raise AssertionError('Evaluation did not terminate')
        env.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bc-updates', type=int, default=80)
    parser.add_argument('--rl-steps', type=int, default=256)
    parser.add_argument('--demo-episodes', type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh output directory; never overwrite an attempt')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This runner is for the requested real GPU training smoke')
    # Do not consume the memory reserved by the user's other GPU job.
    torch.cuda.set_per_process_memory_fraction(.20)
    assert not any(name.startswith('crowd_nav.policy.mamba') for name in sys.modules)
    start = time.time()
    rows, teacher_records = collect(args.params, args.demo_episodes)
    torch.save(rows, args.out / 'shared_demonstrations.pt')
    demo_hash = hashlib.sha256((args.out / 'shared_demonstrations.pt').read_bytes()).hexdigest()
    train_rows = [r for r in rows if r['episode'] < args.demo_episodes-1]
    held_rows = [r for r in rows if r['episode'] == args.demo_episodes-1]
    if not train_rows or not held_rows:
        raise AssertionError('Need separate BC train and held-out episodes')
    results = dict(scope='Minimal training/wiring only; one seed, not a performance comparison',
                   args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                   demonstrations_sha256=demo_hash, demonstrations=len(train_rows),
                   heldout_demonstrations=len(held_rows), teacher_episodes=teacher_records, arms={})
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    for arm in ARMS:
        print('START', arm, flush=True)
        rng = np.random.default_rng(2407)
        env = BeliefEnv(args.params, arm, seed=2407)
        model = TD3('MultiInputPolicy', env, learning_rate=3e-4, buffer_size=5000,
                    learning_starts=32, batch_size=32, train_freq=1, gradient_steps=1,
                    policy_delay=2, tau=.005, gamma=.99, seed=2407, device='cuda',
                    action_noise=NormalActionNoise(np.zeros(2), np.full(2, .1)),
                    policy_kwargs=dict(features_extractor_class=SetEncoder, net_arch=[96, 96],
                                       share_features_extractor=False), verbose=0)
        observations = stack_obs([transform_observation(r['observation'], arm) for r in train_rows])
        actions = model.policy.scale_action(np.stack([r['action'] for r in train_rows])).astype(np.float32)
        held_obs = stack_obs([transform_observation(r['observation'], arm) for r in held_rows])
        held_actions = model.policy.scale_action(np.stack([r['action'] for r in held_rows])).astype(np.float32)
        before = mse(model, held_obs, held_actions)
        model.policy.set_training_mode(True)
        for _ in range(args.bc_updates):
            indices = rng.integers(len(train_rows), size=64)
            batch = obs_as_tensor({k:v[indices] for k,v in observations.items()}, model.device)
            loss = F.mse_loss(model.actor(batch), torch.as_tensor(actions[indices], device=model.device))
            model.actor.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.)
            model.actor.optimizer.step()
        model.actor_target.load_state_dict(model.actor.state_dict())
        after = mse(model, held_obs, held_actions)
        actor_before_rl = {k:v.detach().clone() for k,v in model.actor.state_dict().items()}
        for r in train_rows:
            obs = {k:v[None] for k,v in transform_observation(r['observation'], arm).items()}
            nxt = {k:v[None] for k,v in transform_observation(r['next_observation'], arm).items()}
            model.replay_buffer.add(obs, nxt, model.policy.scale_action(r['action'][None]),
                                    np.array([r['reward']]), np.array([r['done']]),
                                    [{'TimeLimit.truncated': False}])
        bc_steps = []
        def regularize(optimizer, positional, keyword):
            # Called after TD3's actor backward and before its optimizer/target updates.
            weight = .1 * max(float(model._current_progress_remaining), 0.)
            indices = rng.integers(len(train_rows), size=32)
            batch = obs_as_tensor({k:v[indices] for k,v in observations.items()}, model.device)
            loss = weight * F.mse_loss(model.actor(batch), torch.as_tensor(actions[indices], device=model.device))
            loss.backward()
            bc_steps.append(float(loss.detach()))
        hook = model.actor.optimizer.register_step_pre_hook(regularize)
        model.learn(total_timesteps=args.rl_steps)
        hook.remove()
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise FloatingPointError('Nonfinite model parameters')
        change = max(float((v-actor_before_rl[k]).abs().max()) for k,v in model.actor.state_dict().items())
        if change <= 0 or model._n_updates < 1 or not bc_steps:
            raise AssertionError('RL failed to update actor/critics')
        model.save(args.out / arm)
        replay_size = model.replay_buffer.size()
        model.policy.set_training_mode(False)
        evaluation = evaluate(model, args.params, arm)
        results['arms'][arm] = dict(bc_heldout_mse_before=before, bc_heldout_mse_after=after,
            critic_updates=model._n_updates, actor_bc_updates=len(bc_steps), actor_weight_change=change,
            replay_size=replay_size, rl_episodes=env.episode_records, evaluation=evaluation,
            finite_parameters=True, continuous_action_dim=2)
        results['elapsed_seconds'] = time.time()-start
        (args.out / 'results.json').write_text(json.dumps(results, indent=2))
        print('DONE', arm, results['arms'][arm], flush=True)
        env.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    results['complete'] = True
    results['mamba_imported'] = any(name.startswith('crowd_nav.policy.mamba') for name in sys.modules)
    assert not results['mamba_imported']
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    print('ALL THREE ARMS COMPLETED', flush=True)


if __name__ == '__main__':
    main()
