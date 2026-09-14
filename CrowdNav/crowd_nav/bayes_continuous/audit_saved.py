"""Audit saved smoke models without training or selecting checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from crowd_nav.bayes_continuous.environment import BeliefEnv, ARMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {'python': platform.python_version(), 'versions': {}, 'models': {}}
    for package in ('torch', 'numpy', 'scipy', 'stable-baselines3', 'gymnasium'):
        report['versions'][package] = importlib.metadata.version(package)
    report['gpu'] = torch.cuda.get_device_name(0)
    for arm in ARMS:
        env = BeliefEnv(args.params, arm)
        obs, _ = env.reset(options={'layout_seed': 910001, 'test_case': 700})
        path = args.results / (arm + '.zip')
        first = TD3.load(path, device='cpu')
        second = TD3.load(path, device='cpu')
        action, _ = first.predict(obs, deterministic=True)
        repeated, _ = second.predict(obs, deterministic=True)
        assert np.array_equal(action, repeated)
        assert env.action_space.contains(action)
        changed = {key: value.copy() for key, value in obs.items()}
        changed['humans'][:, 5:8] = changed['humans'][:, 5:8][:, ::-1]
        perturbed, _ = first.predict(changed, deterministic=True)
        report['models'][arm] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'repeat_load_equal': True, 'action': action.tolist(),
            'mode_permutation_action_delta': (perturbed-action).tolist()}
        env.close()
    report['module_paths'] = {name: getattr(sys.modules[name], '__file__', None)
                              for name in ('stable_baselines3', 'crowd_nav.gdbn',
                                           'crowd_sim.envs.crowd_sim')}
    report['mamba_imported'] = any(name.startswith('crowd_nav.policy.mamba') for name in sys.modules)
    assert not report['mamba_imported']
    source_root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob('*.py')) + [
        source_root / 'crowd_nav/belief_mdp/runtime.py',
        source_root / 'crowd_sim/envs/crowd_sim.py']
    report['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files}
    (args.results / 'saved_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
