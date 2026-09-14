"""Archive contract tests and a short integration run, without formal training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def integration(params):
    from crowd_nav.belief_mdp.runtime import BeliefMDPFeatureEngine, FullCrowdNavigationEnvironment
    from crowd_nav.belief_mdp.model import BeliefMDPQNetwork, ReplayBuffer
    from crowd_nav.belief_mdp.train import optimize
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.test_belief_contract_repair import config
    from crowd_sim.envs.utils.action import ActionXY
    torch.manual_seed(71)
    np.random.seed(71)
    cfg = config()
    policy = MambaRLPolicy(cfg, 'cpu').eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    engine = BeliefMDPFeatureEngine(policy, str(params), torch.device('cpu'),
        num_humans=5, seed=71)
    env = FullCrowdNavigationEnvironment(str(ROOT / 'crowd_nav/configs/env_belief_mdp.config'),
                                        'baseline_circle', robot_visible=False)
    robot, humans = env.reset(seed=971071, profile='nominal', test_case=71)
    engine.reset()
    features = engine.encode(robot, humans)
    network = BeliefMDPQNetwork(len(features.context), len(features.belief), features.candidates.shape[1])
    replay = ReplayBuffer(32, len(features.context), len(features.belief), len(engine.actions),
                          features.candidates.shape[1])
    transitions = []
    max_read_error = 0.
    for i in range(12):
        snap = engine.filter.get_belief_snapshot()
        before = copy.deepcopy(engine.filter._trackers[0].weights)
        for _ in range(10):
            assert engine.filter.get_belief_snapshot() == snap
        max_read_error = max(max_read_error, float(np.max(np.abs(before-engine.filter._trackers[0].weights))))
        a = (i * 7) % len(engine.actions)
        result = env.step(ActionXY(*map(float, engine.actions[a])))
        next_features = engine.encode(env.robot, list(env.env.humans))
        transitions.append((features, a, result, next_features))
        features = next_features
        if result.done:
            break
    # Truncated smoke trajectories are not assigned fabricated MC targets.
    # Populate zero auxiliary returns only for exercising optimizer wiring.
    for f, a, result, nxt in transitions:
        replay.add(f, a, 0, np.array([0, 1, 2]), result.reward,
                   float(result.done and result.outcome == 'collision'), 0., 0., nxt,
                   result.done, False, False, np.zeros(len(engine.actions)))
    target = copy.deepcopy(network)
    optim = torch.optim.Adam(network.parameters(), lr=1e-4)
    weights_before = network.q_r_head[-1].weight.detach().clone()
    losses = []
    for _ in range(3):
        losses.append(optimize(network, target, optim, {'smoke': replay}, {'smoke': 1.}, 4, .99, 0.,
                               np.random.default_rng(71), torch.device('cpu')))
    assert not torch.equal(weights_before, network.q_r_head[-1].weight)
    assert all(np.isfinite(v) for row in losses for v in row.values() if isinstance(v, (float, int)))
    return dict(steps=len(transitions), replay_size=replay.size, optimizer_updates=3,
                max_snapshot_read_weight_change=max_read_error, parameters_loaded=True,
                decision_shape=list(features.candidates.shape), losses=losses,
                scope='Random frozen MLP context, real GDBN params and CrowdSim; wiring only, not Mamba teacher parity or navigation value.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'repair_results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = {}
    commands = {
        'regression': ['-m', 'unittest', 'crowd_nav.test_belief_contract_repair', '-v'],
        'dual_head_selftest': ['crowd_nav/belief_mdp/selftest.py'],
        'occlusion_unit': ['crowd_nav/test_occlusion_belief.py', '--unit'],
        'leakage': ['crowd_nav/test_occlusion_belief.py', '--leakage'],
        'smoke_full': ['crowd_nav/test_occlusion_belief.py', '--smoke', '--modes', 'bayes',
                       '--backbones', 'mlp,gru', '--episodes', '1'],
        'smoke_fixed': ['crowd_nav/test_occlusion_belief.py', '--smoke', '--modes', 'bayes',
                        '--backbones', 'mlp,gru', '--episodes', '1', '--belief-features', 'fixed_confidence'],
    }
    for name, cmd in commands.items():
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        run = subprocess.run([sys.executable] + cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        (args.out / (name + '.log')).write_text(run.stdout + run.stderr)
        results[name] = dict(returncode=run.returncode)
        print(name, run.returncode, flush=True)
        if run.returncode:
            print(run.stdout + run.stderr)
            raise SystemExit(1)
    results['integration'] = integration(args.params.resolve())
    results['parameter_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.params.glob('*.npz')}
    results['source_base'] = '775ae6b3116a5eb84061e08f6baac27bbbadb10b'
    results['formal_retraining'] = False
    results['mamba_teacher_equivalence'] = 'not run: mamba_ssm unavailable'
    results['passed'] = True
    (args.out / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
    print('INTEGRATION PASSED', flush=True)


if __name__ == '__main__':
    main()
