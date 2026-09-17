"""Frozen teacher / IL / PPO stage audit; no training or checkpoint selection."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import time
import types
import zipfile

import gymnasium as gym
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from crowd_nav.bayes_continuous.environment import BeliefEnv

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'repair_results/final_belief_ppo_qualification_20260915'
INITIAL = ROOT/'repair_results/dagger_ppo_nominal_20260915/initial.zip'
PARAMS = ROOT/'repair_results/params'
SCENES = ('baseline_circle', 'dense_circle', 'baseline_square', 'dense_square')
PROFILES = ('nominal', 'heldout_nonstationary')
SEEDS = (2407, 4807, 7207)
ARMS = ('no_belief', 'map', 'full')


class FeatureBase(nn.Module):
    def __init__(self, observation_space, features_dim):
        super().__init__()
        self._features_dim = features_dim


def production_class(file, name, namespace):
    tree = ast.parse(file.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(file), 'exec'), namespace)
    return namespace[name]


# Use the actual encoder/history classes without importing SB3 training machinery.
ActionHistory = production_class(HERE/'train_smoke.py', 'ActionHistory', dict(gym=gym, np=np))
SetEncoder = production_class(HERE/'network.py', 'SetEncoder',
                             dict(torch=torch, nn=nn, BaseFeaturesExtractor=FeatureBase))


class FrozenActor:
    def __init__(self, checkpoint, space, td3=False):
        with zipfile.ZipFile(checkpoint) as z:
            self.weights = torch.load(z.open('policy.pth'), map_location='cpu', weights_only=True)
        self.td3 = td3
        prefix = 'actor.features_extractor.' if td3 else 'pi_features_extractor.'
        self.encoder = SetEncoder(space, 192).eval()
        self.encoder.load_state_dict({k[len(prefix):]: v for k,v in self.weights.items()
                                      if k.startswith(prefix)}, strict=True)

    def linear(self, x, key):
        return F.linear(x, self.weights[key+'.weight'], self.weights[key+'.bias'])

    def predict(self, obs):
        with torch.no_grad():
            batch = {k: torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
            state = self.encoder(batch)
            prefix = 'actor.mu.' if self.td3 else 'mlp_extractor.policy_net.'
            state = F.relu(self.linear(state, prefix+'0'))
            state = F.relu(self.linear(state, prefix+'2'))
            if self.td3:
                action = torch.tensor([.5, 0.]) + torch.tensor([.5, 1.2])*torch.tanh(self.linear(state,'actor.mu.4'))
            else:
                action = self.weights['action_net.1.center'] + self.weights['action_net.1.scale']*torch.tanh(self.linear(state,'action_net.0'))
            return np.clip(action[0].numpy(), [0., -1.2], [1., 1.2]).astype(np.float32)


def sequential_spawn(self, human_num, rule):
    assert rule in ('circle_crossing', 'square_crossing')
    self.humans = []
    generate = self.generate_circle_crossing_human if rule == 'circle_crossing' else self.generate_square_crossing_human
    for _ in range(human_num):
        self.humans.append(generate())


def make_env(scene, arm, corrected=True):
    env = ActionHistory(BeliefEnv(PARAMS, arm=arm, training=False, scenario=scene, seed=2407), route=True)
    if corrected:
        world = env.unwrapped.world.env
        world.generate_random_human_position = types.MethodType(sequential_spawn, world)
    assert env.unwrapped.teacher_mode == 'cv'
    return env


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode(env, actor, layout, case, profile, corrected=True, compare_actor=None):
    obs, _ = env.reset(options=dict(layout_seed=layout, test_case=case, profile=profile))
    world = env.unwrapped.world.env
    robot, humans = world.robot, world.humans
    assert not robot.visible and robot.kinematics == 'unicycle'
    assert obs['robot'].shape == (10,) and int(obs['mask'].sum()) == len(humans)
    state = np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref] for h in humans], np.float64)
    rhash = np.asarray([robot.px,robot.py,robot.gx,robot.gy,robot.theta,robot.radius],np.float64)
    layout_hash = hashlib.sha256(state.tobytes()+rhash.tobytes()).hexdigest()
    human_hash = hashlib.sha256(state.tobytes()).hexdigest()
    clearance = min(np.hypot(a.px-b.px,a.py-b.py)-a.radius-b.radius
                    for i,a in enumerate([robot]+humans) for b in ([robot]+humans)[i+1:])
    if corrected:
        assert clearance >= world.discomfort_dist-1e-9, clearance
    actions, clearances, distances, teacher_ms = [], [], [], []
    conversion_error = 0.
    trace = hashlib.sha256()
    start = time.monotonic()
    for step in range(140):
        action = env.unwrapped.expert_action() if actor is None else actor.predict(obs)
        if actor is None:
            diagnostics = env.unwrapped.world.teacher_diagnostics
            assert diagnostics['provenance'] == 'full_observation_cv_teacher'
            teacher_ms.append(diagnostics['elapsed_ms'])
        if compare_actor is not None:
            conversion_error = max(conversion_error, float(np.max(np.abs(action-compare_actor.predict(obs)))))
        actions.append(action.tolist())
        obs, reward, done, truncated, info = env.step(action)
        clearances.append(info['actual_clearance'])
        distances.append(float(np.hypot(robot.px-robot.gx,robot.py-robot.gy)))
        trace.update(np.asarray([*action,robot.px,robot.py,robot.theta,reward,info['actual_clearance']],np.float64).tobytes())
        if done or truncated:
            result = dict(info['episode_result'], profile=profile, humans=len(humans),
                layout_sha256=layout_hash, human_layout_sha256=human_hash,
                initial_min_clearance=float(clearance), actual_min_clearance=float(min(clearances)),
                mean_speed=float(np.mean(np.asarray(actions)[:,0])),
                mean_abs_omega=float(np.mean(np.abs(np.asarray(actions)[:,1]))),
                stopped_fraction=float(np.mean(np.asarray(actions)[:,0]<.05)),
                final_goal_distance=distances[-1], elapsed_seconds=time.monotonic()-start,
                conversion_error=conversion_error, executed_trace_sha256=trace.hexdigest(),
                actions=actions, goal_distances=distances,
                teacher_mean_plan_ms=float(np.mean(teacher_ms)) if teacher_ms else None)
            assert result['bound_violations'] == 0
            return result
    raise AssertionError('Terminal contract exceeds 140 steps')


def preflight(out):
    torch.set_num_threads(1)
    env = make_env('baseline_circle','no_belief',corrected=False)
    actor = FrozenActor(INITIAL, env.observation_space)
    original = FrozenActor(ROOT/'repair_results/student_dagger_coverage_20260915/round16.zip',env.observation_space,td3=True)
    expected = json.loads((INITIAL.parent/'initial_deterministic.json').read_text())['records'][:3]
    receipts=[]
    for e in expected:
        got=episode(env,actor,e['layout_seed'],e['test_case'],'nominal',corrected=False,compare_actor=original)
        assert got['outcome']==e['outcome'] and got['steps']==e['steps']
        assert abs(got['reward']-e['reward'])<5e-5
        assert got['conversion_error']<1e-6
        receipts.append(dict(case=e['test_case'],conversion_error=got['conversion_error'],reward_error=got['reward']-e['reward']))
    env.close()
    env=make_env('baseline_circle','full',corrected=False)
    cp=SOURCE/'2407_full/attempt0_20480.zip'
    actor=FrozenActor(cp,env.observation_space)
    expected=json.loads((cp.parent/'attempt0_20480_nonstationary.json').read_text())['records'][:3]
    for e in expected:
        got=episode(env,actor,e['layout_seed'],e['test_case'],'train_nonstationary',corrected=False)
        assert got['outcome']==e['outcome'] and got['steps']==e['steps']
        assert abs(got['reward']-e['reward'])<5e-5
        receipts.append(dict(case=e['test_case'],reward_error=got['reward']-e['reward']))
    env.close()
    # The actual CV teacher kernel must match the teacher used to generate DAgger.
    old=subprocess.check_output(['git','show','fa432a4:CrowdNav/crowd_nav/bayes_continuous/teacher.py'],cwd=ROOT.parent,text=True)
    current=(HERE/'teacher.py').read_text()
    def classes(s):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.ClassDef)}
    assert classes(old)==classes(current)
    result=dict(passed=True,reference_episode_receipts=receipts,teacher_classes_match_fa432a4=True,
                runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__))
    (out/'preflight.json').write_text(json.dumps(result,indent=2))
    print('PREFLIGHT',json.dumps(result),flush=True)


def worker(task):
    out,scene,profile,method,seed,first,count=task
    torch.set_num_threads(1)
    arm=method if method in ARMS else 'no_belief'
    env=make_env(scene,arm)
    checkpoint=INITIAL if method=='il' else SOURCE/f'{seed}_{method}/attempt0_20480.zip'
    actor=None if method=='teacher' else FrozenActor(checkpoint,env.observation_space)
    records=[]
    index=SCENES.index(scene)
    for i in range(first,first+count):
        records.append(episode(env,actor,170000000+index*10000+i,510000+index*1000+i,profile))
    env.close()
    result=dict(scene=scene,profile=profile,method=method,seed=seed,records=records,
                checkpoint_sha256=None if actor is None else sha(checkpoint))
    dest=Path(out)/'episodes'/f'{scene}_{profile}_{method}_{seed}_{first:03d}.json'
    dest.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('DONE',scene,profile,method,seed,first, [sum(r['outcome']==o for r in records) for o in ('success','collision','timeout')],flush=True)
    return str(dest)


def register(out,count):
    files=list(HERE.glob('*.py'))+[ROOT/'crowd_nav/belief_mdp/runtime.py',
        ROOT/'crowd_nav/bayesian_pilot/protocol.py',ROOT/'crowd_sim/envs/crowd_sim.py',
        ROOT/'crowd_nav/configs/env_belief_mdp.config']+sorted(PARAMS.glob('*'))
    files=[p for p in files if p.is_file()]
    checkpoints=[INITIAL]+[SOURCE/f'{seed}_{arm}/attempt0_20480.zip' for seed in SEEDS for arm in ARMS]
    p=dict(count_per_cell=count,scenes=SCENES,profiles=PROFILES,seeds=SEEDS,arms=ARMS,
           total_episodes=count*len(SCENES)*len(PROFILES)*11,unique_physical_layouts=count*len(SCENES),
           layout_seeds='170000000 + scene_index*10000 + case_index',
           case_ids='510000 + scene_index*1000 + case_index',
           spawn='sequential insertion, no initial overlaps; all methods identical',
           teacher='CV unicycle CEM, unchanged qualified kernel; not archived GDBN teacher',
           profile_pairing='same layout and case seeds; profiles change behavior distribution',
           design='frozen diagnostic, no training, no model selection; pointwise paired CIs, not a new method claim',
           controls='same observation, unicycle action, reward, timeout, physical collision audit',
           code_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
           checkpoint_sha256={str(f.relative_to(ROOT)):sha(f) for f in checkpoints},
           base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip())
    file=out/'protocol.json'
    if file.exists():
        assert json.loads(file.read_text())==json.loads(json.dumps(p)), 'Frozen protocol changed'
    else:
        file.write_text(json.dumps(p,indent=2))
    return p


def summarize(out):
    protocol=json.loads((out/'protocol.json').read_text()); n=protocol['count_per_cell']
    groups={}; records=[]
    for f in sorted((out/'episodes').glob('*.json')):
        cell=json.loads(f.read_text())
        key=(cell['scene'],cell['profile'],cell['method'],cell['seed'])
        groups.setdefault(key,[]).extend(cell['records'])
        records.extend(dict(scene=cell['scene'],method=cell['method'],seed=cell['seed'],**r) for r in cell['records'])
    assert len(records)==protocol['total_episodes']
    assert len(groups)==len(SCENES)*len(PROFILES)*11
    for key,rs in groups.items():
        rs.sort(key=lambda r:r['layout_seed'])
        assert len(rs)==n and len({r['layout_seed'] for r in rs})==n
    cells=[]; comparisons=[]
    rng=np.random.default_rng(240717)
    for scene in SCENES:
        for profile in PROFILES:
            teacher=groups[(scene,profile,'teacher',0)]
            for key,rs in groups.items():
                if key[:2]==(scene,profile):
                    assert [r['layout_sha256'] for r in rs]==[r['layout_sha256'] for r in teacher]
            def values(method):
                ss=SEEDS if method in ARMS else (0,)
                return np.asarray([[[r['outcome']=='success',r['outcome']=='collision',r['reward']]
                    for r in groups[(scene,profile,method,seed)]] for seed in ss],float)
            for method in ('teacher','il')+ARMS:
                v=values(method)
                cells.append(dict(scene=scene,profile=profile,method=method,n_layouts=n,n_policy_seeds=len(v),
                    success=float(v[:,:,0].mean()),collision=float(v[:,:,1].mean()),
                    timeout=float(1-v[:,:,0].mean()-v[:,:,1].mean()),mean_return=float(v[:,:,2].mean()),
                    per_seed_counts=[[int(x[:,0].sum()),int(x[:,1].sum()),int(n-x[:,0].sum()-x[:,1].sum())] for x in v]))
            for a,b in [('teacher','il'),('no_belief','il'),('map','il'),('full','il'),('full','no_belief'),('full','map')]:
                diff=values(a)-values(b)
                boot=[]
                for _ in range(5000):
                    seeds=rng.integers(len(diff),size=len(diff)); idx=rng.integers(n,size=n)
                    boot.append(diff[seeds][:,idx].mean(axis=(0,1)))
                comparisons.append(dict(scene=scene,profile=profile,contrast=f'{a}-{b}',
                    metrics=['success','collision','return'],difference=diff.mean(axis=(0,1)).tolist(),
                    ci95=np.quantile(boot,[.025,.975],axis=0).tolist()))
    result=dict(protocol=protocol,cells=cells,comparisons=comparisons,
                episode_count=len(records),bound_violations=sum(r['bound_violations'] for r in records),
                min_initial_clearance=min(r['initial_min_clearance'] for r in records),
                unique_layouts=len({r['layout_sha256'] for r in records}),
                inference='5000 paired layout bootstrap draws, also resampling three PPO seeds; pointwise exploratory intervals')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('SUMMARY',len(records),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=50);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'episodes').mkdir(exist_ok=True)
    if args.summarize_only:
        summarize(args.out);return
    register(args.out,args.count);preflight(args.out)
    if args.preflight_only:return
    tasks=[]
    for scene in SCENES:
        for profile in PROFILES:
            for method in ('teacher','il')+ARMS:
                for seed in (SEEDS if method in ARMS else (0,)):
                    for start in range(0,args.count,5):
                        dest=args.out/'episodes'/f'{scene}_{profile}_{method}_{seed}_{start:03d}.json'
                        if not dest.exists():
                            tasks.append((str(args.out),scene,profile,method,seed,start,min(5,args.count-start)))
    # Schedule teacher batches first so slower MPC calls do not form a long tail.
    tasks.sort(key=lambda t:t[3]!='teacher')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for f in as_completed(futures):f.result()
    register(args.out,args.count)
    summarize(args.out)


if __name__=='__main__':
    main()
