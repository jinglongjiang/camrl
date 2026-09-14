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
from crowd_nav.bayes_continuous.algorithm import BayesSetTD3


def audit_bc(folder, params):
    from unittest.mock import patch
    result = json.loads((folder/'results.json').read_text())
    arm = result['arm']
    env = BeliefEnv(params, arm)
    model = BayesSetTD3.load(folder/'bc_only.zip', env=env, device='cpu')
    model.check_arm(arm)
    assert model.learning_starts == 0 and model.gradient_steps == 1
    assert model.policy_delay == 2 and model.tau == .005 and model.gamma == .99
    assert model.train_freq.frequency == 1 and model.train_freq.unit.value == 'step'
    np.testing.assert_allclose(model.action_noise._sigma, [.1, .1])
    assert model.stage_metadata['arm'] == arm and not model.actor_enabled
    obs, _ = env.reset(seed=2407)
    model._last_obs = {k:v[None] for k,v in obs.items()}
    with patch.object(model.action_space, 'sample', side_effect=AssertionError('Random warmup used')):
        action, _ = model._sample_action(model.learning_starts, action_noise=None)
    np.testing.assert_array_equal(action[0], model.predict(obs, deterministic=True)[0])
    collection = torch.load(folder/'collection.pt', weights_only=False)
    train_hashes = {r['layout_sha256'] for r in collection['records']}
    val_hashes = {r['layout_sha256'] for r in result['bc_records']}
    assert len(train_hashes)==200 and len(val_hashes)==100 and not train_hashes & val_hashes
    rows = [x for rec, ep in zip(collection['records'], collection['trajectories'])
            if rec['outcome']=='success' for x in ep]
    squared, predictions = [], []
    for start in range(0, len(rows), 256):
        batch = rows[start:start+256]
        inputs = {k:np.stack([r['observation'][k] for r in batch]) for k in batch[0]['observation']}
        if arm == 'no_belief':
            assert not np.any(inputs['humans'][:,:,5:])
        pred, _ = model.predict(inputs, deterministic=True)
        target = np.stack([r['action'] for r in batch])
        squared.extend((pred-target)**2)
        predictions.extend(pred)
    records = result['bc_records']
    outcomes = {key:sum(r['outcome']==key for r in records) for key in ('success','collision','timeout')}
    actions = np.array([step['action'] for r in records for step in r['trace']])
    audit = dict(arm=arm, outcomes=outcomes, training_outcomes={key:sum(r['outcome']==key
        for r in collection['records']) for key in outcomes}, training_steps=len(rows),
        training_action_mse_physical=np.mean(squared, axis=0).tolist(),
        training_teacher_action_std=np.std([r['action'] for r in rows], axis=0).tolist(),
        training_prediction_std=np.std(predictions, axis=0).tolist(),
        validation_action_std=actions.std(0).tolist(),
        validation_slow_step_fraction=float(np.mean(actions[:,0]<.1)),
        validation_actual_overlap_episodes=sum(any(s['actual_clearance']<0 for s in r['trace']) for r in records),
        zero_random_warmup_verified=True, checkpoint_arm_verified=True,
        checkpoint_sha256=hashlib.sha256((folder/'bc_only.zip').read_bytes()).hexdigest(),
        train_unique=200, validation_unique=100, layout_overlap=0,
        actor_enabled=model.actor_enabled, warmup_updates=model.warmup_updates,
        rl_started=result['rl_started'], gpu=torch.cuda.get_device_name(0),
        module_paths={name:getattr(sys.modules[name], '__file__', None) for name in
            ('stable_baselines3', 'crowd_nav.gdbn', 'crowd_sim.envs.crowd_sim')},
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','stable-baselines3','gymnasium')})
    assert actions[:,0].min()>=0 and actions[:,0].max()<=1+1e-6 and np.abs(actions[:,1]).max()<=1.2+1e-6
    assert not any(name.startswith('crowd_nav.policy.mamba') for name in sys.modules)
    assert not any(name == 'mamba_ssm' or name.startswith('mamba_ssm.') for name in sys.modules)
    (folder/'saved_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    env.close()


def audit_teachers(paths, destination):
    report = {'scope':'five-human nominal full observation; empirical qualification, not a safety guarantee',
              'runs':{}, 'final_layout_overlap_with_previous':0}
    previous = set()
    for path in paths:
        data = json.loads(path.read_text())
        records = data['teacher_records']
        layouts = {r['layout_sha256'] for r in records}
        if len(layouts) != len(records):
            raise AssertionError('Duplicate physical layouts within an evaluation')
        actions = np.asarray([s['action'] for r in records for s in r['trace']])
        timings = [s['planner']['elapsed_ms'] for r in records for s in r['trace']]
        if actions[:,0].min() < -1e-6 or actions[:,0].max() > 1.+1e-6 or np.abs(actions[:,1]).max() > 1.2+1e-6:
            raise AssertionError('Teacher exceeded actuator limits')
        max_accel = max(float(np.abs(np.diff([0.]+[s['action'][0] for s in r['trace']])).max()/.25) for r in records)
        if max_accel > 2.+1e-5:
            raise AssertionError('Teacher exceeded its acceleration limit')
        summary = {event:sum(r['outcome']==event for r in records) for event in ('success','collision','timeout')}
        summary.update(episodes=len(records), case_min=min(r['test_case'] for r in records),
            case_max=max(r['test_case'] for r in records), unique_layouts=len(layouts),
            max_speed=float(actions[:,0].max()), max_abs_omega=float(np.abs(actions[:,1]).max()),
            max_acceleration=max_accel, planner_p50_ms=float(np.median(timings)),
            planner_p95_ms=float(np.percentile(timings,95)),
            teacher_source_sha256=data['teacher_source_sha256'],
            environment_source_sha256=data.get('environment_source_sha256'),
            collision_rule=data.get('collision_rule','native only'),
            results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        summary['point_gate_passed'] = (len(records)>=100 and summary['success']/len(records)>=.9
                                        and summary['collision']/len(records)<=.02)
        if 'actual_clearance' in records[0]['trace'][0]:
            summary['actual_overlap_episodes'] = sum(any(s['actual_clearance'] < 0 for s in r['trace']) for r in records)
            summary['native_collision_episodes'] = sum(r['trace'][-1]['native_outcome']=='collision' for r in records)
        report['runs'][path.parent.name] = summary
        overlap = len(previous & layouts)
        previous.update(layouts)
    report['final_layout_overlap_with_previous'] = overlap
    report['final_code_matches_current'] = all(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==summary[key]
        for name,key in [('teacher.py','teacher_source_sha256'), ('environment.py','environment_source_sha256')])
    report['final_qualified'] = (summary['point_gate_passed'] and overlap == 0 and
                                 'actual' in summary['collision_rule'] and report['final_code_matches_current'])
    destination.mkdir(exist_ok=True)
    output = destination/'teacher_summary.json'
    if output.exists():
        raise ValueError('Never overwrite a qualification report')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--params', type=Path)
    parser.add_argument('--teacher-runs', type=Path, nargs='+')
    args = parser.parse_args()
    if args.teacher_runs:
        audit_teachers(args.teacher_runs, args.results)
        return
    if args.params is None:
        parser.error('--params is required for checkpoint auditing')
    torch.set_num_threads(1)
    if (args.results/'bc_only.zip').exists():
        audit_bc(args.results, args.params)
        return
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
