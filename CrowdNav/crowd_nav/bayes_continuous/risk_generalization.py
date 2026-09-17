"""Bounded five-human IL/PPO experiment; never select on larger crowds."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import types
import zipfile

import gymnasium as gym
import numpy as np
import scipy
from scipy.stats import ncx2
import torch
from torch import nn
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from crowd_nav.bayes_continuous.environment import BeliefEnv
from crowd_nav.bayes_continuous.network import SetEncoder, DaggerGaussianPolicy
from crowd_nav.bayes_continuous.stage_audit import ActionHistory, sequential_spawn, FrozenActor
from crowd_nav.bayes_continuous.teacher import gdbn_teacher_moments

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT/'repair_results/params'
SOURCE = ROOT/'repair_results/dagger_ppo_nominal_20260915/initial.zip'
ARMS = ('base', 'cv', 'ewma', 'bayes')
SEEDS = (2407, 4807, 7207)
TESTS = ('baseline_circle', 'dense_circle', 'large_circle',
         'baseline_square', 'dense_square', 'large_square')


class RiskEnv(BeliefEnv):
    def __init__(self, arm, training=True, scenario='baseline_circle', seed=2407):
        self.risk_arm = arm
        self.previous_velocity = None
        self.residual_variance = None
        self.train_sequence = 0
        self.experiment_seed = seed
        super().__init__(PARAMS, arm='no_belief', training=training, scenario=scenario, seed=seed)
        self.observation_space = gym.spaces.Dict(dict(self.observation_space.spaces,
            risk=gym.spaces.Box(0., 1., (20,), np.float32)))
        world = self.world.env
        world.generate_random_human_position = types.MethodType(sequential_spawn, world)

    def reset(self, *, seed=None, options=None):
        self.previous_velocity = None
        self.residual_variance = None
        if options is None:
            i = self.train_sequence
            self.train_sequence += 1
            options = dict(layout_seed=220000000+self.experiment_seed*1000+i,
                test_case=610000+i,
                profile='nominal' if i % 4 < 2 else 'train_nonstationary',
                shape='circle' if i % 2 == 0 else 'square')
        options = dict(options)
        if 'shape' in options:
            self.world.env.test_sim = options.pop('shape')+'_crossing'
        obs, info = super().reset(seed=seed, options=options)
        world = self.world.env
        if self.training:
            assert len(world.humans) == 5
        entities = [world.robot]+world.humans
        clearance = min(np.hypot(a.px-b.px,a.py-b.py)-a.radius-b.radius
            for i,a in enumerate(entities) for b in entities[i+1:])
        assert clearance >= world.discomfort_dist-1e-9
        self.layout_hash = hashlib.sha256(np.asarray(
            [[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref] for h in world.humans],
            np.float64).tobytes()).hexdigest()
        return obs, info

    def _observation(self):
        obs = super()._observation()
        humans, robot = self.world.env.humans, self.world.robot
        entities = np.asarray([[h.px,h.py,h.vx,h.vy,h.radius] for h in humans])
        n = len(entities)
        velocity = entities[:,2:4]
        if self.previous_velocity is None:
            self.residual_variance = np.full(n,.01)
        else:
            residual = np.mean((velocity-self.previous_velocity)**2,axis=1)
            self.residual_variance = .85*self.residual_variance+.15*residual
        self.previous_velocity = velocity.copy()
        t = np.arange(1,9)*.25
        cv = entities[:,None,:2]+velocity[:,None]*t[None,:,None]
        _, mean, covariance = gdbn_teacher_moments(entities,
            self.filter.get_belief_snapshot(),self.filter.gdbn,8,.25)
        variance = np.maximum(.01,np.trace(covariance,axis1=-2,axis2=-1)/2)
        paths=[]
        for v,w in [(0.,0.)]+[(v,w) for v in (.5,1.) for w in (-.6,0.,.6)]:
            headings=robot.theta+w*t
            paths.append(np.asarray([robot.px,robot.py])+np.cumsum(
                v*.25*np.stack([np.cos(headings),np.sin(headings)],axis=-1),axis=0))
        paths=np.asarray(paths)
        radius2=(entities[:,4]+robot.radius)**2
        def risk(mu,var):
            var=np.broadcast_to(var,(n,8))
            d2=((paths[:,None]-mu[None])**2).sum(-1)
            probability=ncx2.cdf(radius2[None,:,None]/var[None],2,d2/var[None])
            return probability.max(axis=(0,2))
        self.all_risks=np.zeros((4,20),np.float32)
        self.all_risks[1,:n]=risk(cv,(.1+.2*t[None])**2)
        self.all_risks[2,:n]=risk(cv,.01+self.residual_variance[:,None]*t[None]**3/(3*.25))
        self.all_risks[3,:n]=risk(mean,variance)
        assert np.isfinite(self.all_risks).all()
        obs['risk']=self.all_risks[ARMS.index(self.risk_arm)].copy()
        return obs


def env_for(arm, training=True, scenario='baseline_circle', seed=2407):
    return ActionHistory(RiskEnv(arm,training,scenario,seed),route=True)


def options(i, start, case_start, profile=None):
    return dict(layout_seed=start+i,test_case=case_start+i,
        shape='circle' if i%2==0 else 'square',
        profile=profile or ('nominal' if i%4<2 else 'train_nonstationary'))


def make_model(env, seed):
    model=PPO(DaggerGaussianPolicy,env,learning_rate=1e-5,n_steps=2048,
        batch_size=256,n_epochs=3,gamma=.99,gae_lambda=.95,clip_range=.1,
        ent_coef=0,target_kl=None,seed=seed,device='cpu',verbose=0,
        policy_kwargs=dict(features_extractor_class=SetEncoder,
            features_extractor_kwargs=dict(features_dim=192,risk_guided=True),
            net_arch=dict(pi=[256,256],vf=[96,96]),
            activation_fn=nn.ReLU,share_features_extractor=False))
    with zipfile.ZipFile(SOURCE) as archive:
        weights=torch.load(archive.open('policy.pth'),map_location='cpu',weights_only=True)
    policy=model.policy
    for encoder in (policy.pi_features_extractor,policy.vf_features_extractor):
        prefix='pi_features_extractor.'
        state={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
        missing,extra=encoder.load_state_dict(state,strict=False)
        assert missing==['risk_gain'] and not extra
        with torch.no_grad():
            encoder.human[0].weight[:,5:].zero_()
    for module,prefix in [(policy.mlp_extractor.policy_net,'mlp_extractor.policy_net.'),
                          (policy.action_net,'action_net.')]:
        module.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)})
    with torch.no_grad():
        policy.log_std.copy_(torch.tensor(np.log([.03,.06]),dtype=torch.float32))
    model.set_logger(configure(None,[]))
    return model


def physical_mean(policy, obs):
    features=policy.pi_features_extractor(obs)
    return policy.action_net(policy.mlp_extractor.policy_net(features))


def rollout(env, actor, opt, collect=False):
    obs,_=env.reset(options=opt)
    rows=[]
    for _ in range(140):
        if actor is None:
            action=env.unwrapped.expert_action()
        else:
            action=actor(obs)
        if collect:
            label=env.unwrapped.expert_action()
            rows.append((copy.deepcopy(obs),env.unwrapped.all_risks.copy(),label.copy()))
        obs,_,done,_,info=env.step(action)
        if done:
            r=dict(info['episode_result'],layout_sha256=env.unwrapped.layout_hash,
                humans=len(env.unwrapped.world.env.humans),
                profile=opt['profile'],shape=env.unwrapped.world.env.test_sim)
            return r,rows
    raise AssertionError('Missing terminal')


def collection_worker(task):
    destination,first,count,checkpoint=task
    torch.set_num_threads(1)
    env=env_for('base')
    actor=None
    if checkpoint:
        model=PPO.load(checkpoint,device='cpu')
        arm=Path(checkpoint).parent.name.split('_',1)[1]
        env.close();env=env_for(arm)
        actor=lambda o:model.predict(o,deterministic=True)[0]
    records=[]; rows=[]
    for i in range(first,first+count):
        start=210000000 if checkpoint is None else 230000000
        r,rr=rollout(env,actor,options(i,start,620000),collect=True)
        records.append(r)
        if checkpoint is not None or r['outcome']=='success':
            rows.extend(rr)
    env.close()
    torch.save(dict(records=records,rows=rows),destination)
    print('COLLECT',first,count,len(rows),flush=True)
    return destination


def merge_data(files,destination):
    rows=[]; records=[]
    for f in files:
        data=torch.load(f,map_location='cpu',weights_only=False)
        rows.extend(data['rows']);records.extend(data['records'])
    data=dict(obs={k:np.stack([r[0][k] for r in rows]) for k in rows[0][0]},
        risks=np.stack([r[1] for r in rows]),actions=np.stack([r[2] for r in rows]),records=records)
    torch.save(data,destination)
    return len(rows)


def development(model,arm):
    env=env_for(arm)
    result={}
    for profile in ('nominal','train_nonstationary'):
        records=[]
        for i in range(100):
            r,_=rollout(env,lambda o:model.predict(o,deterministic=True)[0],
                options(i,240000000,630000,profile))
            records.append(r)
        result[profile]=dict(records=records,success=sum(r['outcome']=='success' for r in records),
            collision=sum(r['outcome']=='collision' for r in records))
    env.close()
    return result


def fit_worker(task):
    out,seed,arm,stage,updates=task
    out=Path(out);folder=out/f'{seed}_{arm}';folder.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed)
    env=env_for(arm,seed=seed)
    model=make_model(env,seed) if stage=='bc' else PPO.load(folder/'bc.zip',env=env,device='cpu')
    datasets=[torch.load(out/'demos.pt',weights_only=False)]
    if stage=='dagger':
        datasets.append(torch.load(out/'dagger.pt',weights_only=False))
    obs={k:np.concatenate([d['obs'][k] for d in datasets]) for k in datasets[0]['obs']}
    obs['risk']=np.concatenate([d['risks'][:,ARMS.index(arm)] for d in datasets])
    actions=np.concatenate([d['actions'] for d in datasets])
    strata=[np.flatnonzero(actions[:,1]<-.2),np.flatnonzero(abs(actions[:,1])<=.2),
            np.flatnonzero(actions[:,1]>.2)]
    assert all(len(s) for s in strata)
    params=list(model.policy.pi_features_extractor.parameters())+list(
        model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    optimizer=torch.optim.Adam(params,lr=3e-4)
    rng=np.random.default_rng(seed)
    for update in range(updates):
        idx=np.concatenate([rng.choice(s,86,replace=True) for s in strata]);rng.shuffle(idx)
        batch={k:torch.as_tensor(v[idx]) for k,v in obs.items()}
        target=torch.as_tensor(actions[idx]);pred=physical_mean(model.policy,batch)
        loss=((pred[:,0]-target[:,0])/.5).square().mean()+2*((pred[:,1]-target[:,1])/1.2).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    model.save(folder/stage)
    dev=development(model,arm)
    receipt=dict(stage=stage,seed=seed,arm=arm,updates=updates,transitions=len(actions),
        final_loss=float(loss),risk_gain=float(model.policy.pi_features_extractor.risk_gain),
        development=dev)
    (folder/(stage+'.json')).write_text(json.dumps(receipt,indent=2))
    env.close();print('FIT',seed,arm,stage,{k:v['success'] for k,v in dev.items()},flush=True)
    return receipt


def ppo_worker(task):
    out,seed,arm,start=task
    out=Path(out);folder=out/f'{seed}_{arm}'
    torch.set_num_threads(1);env=env_for(arm,seed=seed)
    model=PPO.load(folder/(start+'.zip'),env=env,device='cpu')
    model.set_logger(configure(None,[]))
    model.policy.vf_features_extractor.load_state_dict(model.policy.pi_features_extractor.state_dict())
    # IL optimizer is separate; PPO starts with a fresh optimizer for every arm.
    model.policy.optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-5,eps=1e-5)
    model.learn(20480)
    steps=sorted({int(s['step']) for s in model.policy.optimizer.state.values() if 'step' in s})
    assert steps==[240],steps
    model.save(folder/'ppo')
    dev=development(model,arm)
    receipt=dict(seed=seed,arm=arm,environment_steps=model.num_timesteps,adam_steps=steps,
        risk_gain=float(model.policy.pi_features_extractor.risk_gain),development=dev)
    (folder/'ppo.json').write_text(json.dumps(receipt,indent=2));env.close()
    print('PPO',seed,arm,{k:v['success'] for k,v in dev.items()},flush=True)
    return receipt


def evaluation_worker(task):
    out,seed,arm,scene,profile,first=task
    out=Path(out);torch.set_num_threads(1)
    model=PPO.load(out/f'{seed}_{arm}/ppo.zip',device='cpu')
    env=env_for(arm,False,scene)
    records=[]
    for i in range(first,first+10):
        j=TESTS.index(scene)
        r,_=rollout(env,lambda o:model.predict(o,deterministic=True)[0],
            dict(layout_seed=250000000+j*10000+i,test_case=640000+j*1000+i,profile=profile))
        records.append(r)
    env.close()
    dest=out/'evaluation'/f'{seed}_{arm}_{scene}_{profile}_{first}.json'
    dest.write_text(json.dumps(dict(seed=seed,arm=arm,scene=scene,profile=profile,records=records),indent=2))
    print('EVAL',seed,arm,scene,profile,first,flush=True)


def map_tasks(worker,tasks,workers):
    with mp.get_context('spawn').Pool(workers) as pool:
        return list(pool.imap_unordered(worker,tasks))


def gate(receipts):
    # Gate only on five-human development; all seeds and arms must pass.
    return all(r['development']['nominal']['success']>=90 and
        r['development']['train_nonstationary']['success']>=80 for r in receipts)


def run(out,workers):
    out.mkdir(parents=True,exist_ok=False)
    code_files=[Path(__file__),Path(__file__).with_name('network.py'),
        Path(__file__).with_name('environment.py'),Path(__file__).with_name('teacher.py')]
    protocol=dict(arms=ARMS,seeds=SEEDS,train_humans=5,teacher_attempts=240,
        bc_updates=3000,dagger_fallback='once, 30 episodes per arm/seed pooled, 3000 updates',
        ppo_steps=20480,ppo_adam_steps=240,dev_gate='each run: nominal >=90/100, train_nonstationary >=80/100',
        test_scenes=TESTS,test_profiles=['nominal','heldout_nonstationary'],test_layouts_per_cell=50,
        test_before_gate=False,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(),sb3=sb3.__version__,torch=torch.__version__,
            numpy=np.__version__,scipy=scipy.__version__),
        params_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PARAMS.glob('*') if p.is_file()},
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
        mechanism='learned zero-initialized log-risk attention bias; physical tokens identical',
        cv_variance='(.1+.2*t)^2',ewma_variance='.01+EWMA(dv^2)*t^3/(3*dt), alpha=.15',
        bayes='existing GDBN predictive mean and trace/2 covariance, floor=.01; B_action=0',
        risk='maximum Gaussian-disc overlap over 7 fixed robot arcs and 8 steps; no claim of union probability',
        statistical_rule='Primary: mean OOD SR, 5 non-baseline configurations x 2 profiles equally weighted; paired seed/layout bootstrap with shared profile pairs; 98.333% CI for each of 3 Bayes-control contrasts (Bonferroni).',
        benefit_gate='Bayes >=3pp over each of base/CV/EWMA on primary OOD metric, all CI lower bounds >0; no >2pp nominal 5-human loss against base.',
        strong_generalization_gate='Separately report whether 20-human nominal SR reaches 80%; a relative gain alone does not establish strong generalization.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    files=map_tasks(collection_worker,[(str(out/f'demo_{i}.pt'),i,10,None) for i in range(0,240,10)],workers)
    merge_data(files,out/'demos.pt')
    receipts=map_tasks(fit_worker,[(str(out),s,a,'bc',3000) for s in SEEDS for a in ARMS],workers)
    start='bc'
    if not gate(receipts):
        tasks=[]
        for j,(s,a) in enumerate((s,a) for s in SEEDS for a in ARMS):
            tasks.append((str(out/f'dagger_{j}.pt'),j*30,30,str(out/f'{s}_{a}/bc.zip')))
        files=map_tasks(collection_worker,tasks,workers)
        merge_data(files,out/'dagger.pt')
        receipts=map_tasks(fit_worker,[(str(out),s,a,'dagger',3000) for s in SEEDS for a in ARMS],workers)
        start='dagger'
    (out/'il_gate.json').write_text(json.dumps(dict(passed=gate(receipts),stage=start),indent=2))
    if not gate(receipts):
        (out/'verdict.json').write_text(json.dumps(dict(status='IL_GATE_FAILED',
            conclusion='Cannot judge Bayesian generalization; no high-density selection or test was performed.'),indent=2))
        return
    receipts=map_tasks(ppo_worker,[(str(out),s,a,start) for s in SEEDS for a in ARMS],workers)
    (out/'ppo_gate.json').write_text(json.dumps(dict(passed=gate(receipts)),indent=2))
    if not gate(receipts):
        (out/'verdict.json').write_text(json.dumps(dict(status='PPO_GATE_FAILED',
            conclusion='Training qualification failed; no high-density test.'),indent=2))
        return
    (out/'evaluation').mkdir()
    map_tasks(evaluation_worker,[(str(out),s,a,scene,p,i) for s in SEEDS for a in ARMS
        for scene in TESTS for p in ('nominal','heldout_nonstationary') for i in range(0,50,10)],workers)
    (out/'verdict.json').write_text(json.dumps(dict(status='EVALUATION_COMPLETE',
        episodes=12*6*2*50),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    run(args.out,args.workers)
