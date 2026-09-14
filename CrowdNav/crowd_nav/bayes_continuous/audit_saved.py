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


def audit_dagger(folder, params):
    from crowd_nav.bayes_continuous.train_smoke import ActionHistory, dagger_artifact
    result = json.loads((folder/'results.json').read_text())
    if len(result['rounds']) != result['protocol']['rounds']:
        raise ValueError('Incomplete fixed DAgger queue')
    env = ActionHistory(BeliefEnv(params,arm='no_belief'),route=True)
    validation_hashes = {r['layout_sha256'] for r in result['round0']['records']}
    seen = set(validation_hashes)
    summaries, critic_hashes = [], []
    cumulative = 9037
    def source_path(name):
        return dagger_artifact(folder,name)
    for item in result['rounds']:
        i = item['round']
        data = torch.load(source_path(f'round{i}_collection.pt'),weights_only=False)
        layouts = {r['layout_sha256'] for r in data['records']}
        assert len(layouts)==item.get('rollout_count',100) and not layouts & seen
        seen |= layouts
        steps, disagreements = 0,0
        feasibility = {0:0,1:0,2:0}
        for ep in data['trajectories']:
            prev,route = np.zeros(2),0.
            for row in ep:
                obs,nxt = row['observation'],row['next_observation']
                np.testing.assert_allclose(obs['robot'][-3:-1],prev/[1.,1.2],atol=1e-6)
                np.testing.assert_allclose(obs['robot'][-1],route,atol=1e-6)
                assert not np.any(obs['humans'][:,5:])
                assert env.action_space.contains(row['action']) and env.action_space.contains(row['teacher_action'])
                route = .7*route+.3*float(row['action'][1])/1.2
                np.testing.assert_allclose(nxt['robot'][-1],route,atol=1e-6)
                np.testing.assert_allclose(nxt['robot'][-3:-1],row['action']/[1.,1.2],atol=1e-6)
                prev = row['action']
                steps += 1
                disagreements += int(np.max(np.abs(row['action']-row['teacher_action']))>1e-5)
                feasibility[int(row['teacher_diagnostics']['feasibility_class'])] += 1
        assert steps == item['added_steps'] and item['original_steps_retained']==9037
        cumulative += steps
        assert cumulative == item['permanent_steps']
        assert {r['layout_sha256'] for r in item['records']} == validation_hashes
        model = BayesSetTD3.load(source_path(f'round{i}.zip'),env=env,device='cpu')
        model.check_arm('no_belief')
        assert model.num_timesteps==0 and model._n_updates==0 and model.warmup_updates==0
        def digest(module):
            return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in module.state_dict().values())).hexdigest()
        critic_hashes.append((digest(model.critic),digest(model.cost_critic)))
        counts = {k:sum(r['outcome']==k for r in item['records']) for k in ('success','collision','timeout')}
        summaries.append(dict(round=i,**counts,student_steps=steps,label_action_disagreements=disagreements,
            teacher_feasibility_class_counts=feasibility,
            actual_overlap_episodes=sum(any(s['actual_clearance']<0 for s in r['trace']) for r in item['records']),
            checkpoint_sha256=hashlib.sha256(source_path(f'round{i}.zip').read_bytes()).hexdigest()))
    assert len(set(critic_hashes))==1
    for key,name in [('teacher_source_sha256','teacher.py'),('environment_source_sha256','environment.py')]:
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==result[key]
    report = dict(rounds=summaries,route_tracks_student_not_teacher=True,original_9037_steps_retained=True,
        unique_new_layouts=len(seen)-len(validation_hashes),development_overlap=0,
        reward_and_cost_critics_unchanged_between_rounds=True,rl_steps=0,
        scope='five-human nominal development; not evidence of Bayesian value or independent generalization')
    (folder/'saved_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    env.close()


def confirm_dagger(folder, params, original_collection):
    from crowd_nav.bayes_continuous.train_smoke import episodes, receipt, dagger_artifact
    result = json.loads((folder/'results.json').read_text())
    final_round = result['protocol']['rounds']
    if final_round not in (10,15,16) or len(result['rounds']) != final_round:
        raise ValueError('Confirmation uses the final declared round, not a selected checkpoint')
    output = folder/'independent_confirmation.json'
    if output.exists():
        raise ValueError('Never overwrite independent confirmation')
    original = torch.load(original_collection, weights_only=False)
    seen = {r['layout_sha256'] for r in original['records']}
    seen.update(r['layout_sha256'] for r in result['round0']['records'])
    for item in result['rounds']:
        seen.update(r['layout_sha256'] for r in item['rollout_records'])
    parent_result = result
    while 'extension_parent' in parent_result:
        parent_folder = Path(parent_result['extension_parent'])
        if (parent_folder/'independent_confirmation.json').exists():
            parent_confirmation = json.loads((parent_folder/'independent_confirmation.json').read_text())
            seen.update(r['layout_sha256'] for r in parent_confirmation['records'])
        parent_result = json.loads((parent_folder/'results.json').read_text())
    offset = {10:940000, 15:950000, 16:960000}[final_round]
    count = 500 if final_round==16 else 100
    path = dagger_artifact(folder, f'round{final_round}.zip')
    model = BayesSetTD3.load(path, device='cpu')
    model.check_arm('no_belief')
    before = {k:v.detach().clone() for k,v in model.actor.state_dict().items()}
    records, _, _ = episodes(params, count, offset, model, case_offset=offset,
                             arm='no_belief', diagnostics=True)
    layouts = {r['layout_sha256'] for r in records}
    assert len(layouts)==count and not layouts & seen
    assert all(torch.equal(v, model.actor.state_dict()[k]) for k,v in before.items())
    report = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_round=final_round, cases=[offset,offset+count-1], layout_seed_offset=offset,
        selection='Final round fixed before confirmation; no checkpoint selection',
        scope='One training seed; five-human nominal full-observation No-Belief student',
        metrics=receipt(records), records=records, unique_layouts=count, previous_layout_overlap=0,
        actor_unchanged=True, teacher_queried=False, rl_started=False)
    output.write_text(json.dumps(report, indent=2))
    print('INDEPENDENT CONFIRMATION', report['metrics'], flush=True)


def diagnose_dagger(folder, params):
    from crowd_nav.bayes_continuous.train_smoke import episodes, dagger_artifact
    confirmation = json.loads((folder/'independent_confirmation.json').read_text())
    path = dagger_artifact(folder, f"round{confirmation['checkpoint_round']}.zip")
    model = BayesSetTD3.load(path, device='cpu')
    output = folder/'failure_diagnosis.json'
    if output.exists():
        raise ValueError('Do not overwrite failure diagnosis')
    cases = []
    for original in confirmation['records']:
        if original['outcome'] == 'success':
            continue
        replay, trajectories, _ = episodes(params, 1, original['layout_seed'], model,
            collect=True, case_offset=original['test_case'], diagnostics=True,
            arm='no_belief', query_teacher=True)
        assert replay[0]['layout_sha256'] == original['layout_sha256']
        assert replay[0]['outcome'] == original['outcome']
        assert len(replay[0]['trace']) == len(original['trace'])
        for actual, expected in zip(replay[0]['trace'], original['trace']):
            for key in ('robot','humans','action','actual_clearance'):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=0)
        teacher, _, _ = episodes(params, 1, original['layout_seed'],
            case_offset=original['test_case'], diagnostics=True, arm='no_belief')
        assert teacher[0]['layout_sha256'] == original['layout_sha256']
        rows = trajectories[0]
        tail = [dict(step=j, student_action=row['action'].tolist(),
                     teacher_action=row['teacher_action'].tolist(),
                     teacher_diagnostics=row['teacher_diagnostics'],
                     actual_clearance=replay[0]['trace'][j]['actual_clearance'])
                for j,row in enumerate(rows) if j>=len(rows)-16]
        item = dict(test_case=original['test_case'], student_outcome=original['outcome'],
                    teacher_outcome=teacher[0]['outcome'], replay_exact=True,
                    final_16_steps=tail, teacher_record=teacher[0])
        cases.append(item)
        print('DIAGNOSE',item['test_case'],item['student_outcome'],item['teacher_outcome'],flush=True)
    output.write_text(json.dumps(dict(cases=cases, training_updates=0,
        excluded_from_training=True, scope='Selected failed confirmation cases; not an overall teacher success-rate estimate'),indent=2))


def audit_bc(folder, params):
    from unittest.mock import patch
    result = json.loads((folder/'results.json').read_text())
    arm = result['arm']
    env = BeliefEnv(params, arm)
    if result.get('stage') == 'bc_only_prev_action':
        from crowd_nav.bayes_continuous.train_smoke import ActionHistory
        env = ActionHistory(env)
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
    parser.add_argument('--dagger-confirm', action='store_true')
    parser.add_argument('--dagger-diagnose', action='store_true')
    parser.add_argument('--original-collection', type=Path)
    args = parser.parse_args()
    if args.teacher_runs:
        audit_teachers(args.teacher_runs, args.results)
        return
    if args.params is None:
        parser.error('--params is required for checkpoint auditing')
    torch.set_num_threads(1)
    if args.dagger_diagnose:
        diagnose_dagger(args.results,args.params)
        return
    if args.dagger_confirm:
        if args.original_collection is None:
            parser.error('--dagger-confirm requires --original-collection')
        confirm_dagger(args.results,args.params,args.original_collection)
        return
    if (args.results/'results.json').exists() and json.loads((args.results/'results.json').read_text()).get('stage')=='dagger':
        audit_dagger(args.results,args.params)
        return
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
