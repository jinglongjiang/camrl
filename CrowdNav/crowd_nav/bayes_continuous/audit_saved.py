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


def information_features(frames, t, person):
    """Causal, target-first ordering, shared across all four history frames."""
    now=frames[t]; xy=now['humans'][:,:2]
    others=[i for i in np.argsort(np.linalg.norm(xy-xy[person],axis=1)) if i!=person]
    order=[person]+others
    def physical(frame):
        return np.concatenate([frame['robot'],frame['humans'][order,:5].ravel()])
    x=np.zeros(180,np.float32)
    x[:35]=physical(now)
    x[35:140]=np.concatenate([physical(frames[t-j]) for j in (1,2,3)])
    x[140:160]=now['humans'][order,5:9].ravel()
    oracle=now['oracle'][order].ravel()
    return x,oracle


def information_arm(x, oracle, arm):
    out=np.zeros_like(x);out[:,:35]=x[:,:35]
    if arm=='history':out[:,35:140]=x[:,35:140]
    elif arm=='full':out[:,140:160]=x[:,140:160]
    elif arm=='map':
        p=x[:,140:160].reshape(-1,5,4)
        q=np.zeros_like(p);q[:,:,:3]=np.eye(3,dtype=np.float32)[p[:,:,:3].argmax(-1)]
        out[:,140:160]=q.reshape(-1,20)
    elif arm=='oracle':out[:,140:180]=oracle
    elif arm!='current':raise ValueError(arm)
    return out


def information_episode(task):
    from stable_baselines3 import PPO
    from crowd_nav.bayes_continuous.train_smoke import ActionHistory
    from crowd_nav.bayes_continuous.environment import transform_observation
    from crowd_nav.bayesian_pilot.protocol import MODE_TO_ID
    params,source,split,index,layout=task
    torch.set_num_threads(1)
    env=ActionHistory(BeliefEnv(params,arm='full'),route=True)
    policy=PPO.load(source,device='cpu')
    case_start=dict(train=110000,validation=120000,audit=130000,smoke=150000)[split]
    obs,_=env.reset(options=dict(layout_seed=layout,test_case=case_start+index,profile='train_nonstationary'))
    world=env.unwrapped.world
    layout_hash=hashlib.sha256(np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref]
        for h in world.env.humans]+[[world.robot.px,world.robot.py,world.robot.gx,
        world.robot.gy,world.robot.radius,world.robot.theta]],dtype=np.float64).tobytes()).hexdigest()
    frames=[]
    for step in range(141):
        humans=env.unwrapped.world.env.humans
        truth=np.array([[h.px,h.py,h.vx,h.vy] for h in humans],np.float64)
        oracle=np.zeros((5,8),np.float32)
        events=env.unwrapped.world.scheduler.events
        for i,e in enumerate(events):
            oracle[i,MODE_TO_ID[e.mode]]=1
            oracle[i,5:]=[e.remaining/7.,e.scale,e.turn_radians]
        frames.append(dict(robot=obs['robot'].copy(),humans=obs['humans'][:5].copy(),
            truth=truth,theta=env.unwrapped.world.robot.theta,oracle=oracle,
            klda=np.array(env.unwrapped.filter.get_all_klda()[:5],np.float32)))
        if step and done:break
        action=policy.predict(transform_observation(obs,'no_belief'),deterministic=True)[0]
        obs,_,done,_,info=env.step(action)
    else:raise AssertionError('Episode exceeds legal horizon')
    rows=[]
    for t in range(3,len(frames)-1):
        c,s=np.cos(frames[t]['theta']),np.sin(frames[t]['theta'])
        rotation=np.array([[c,s],[-s,c]])
        for person in range(5):
            x,oracle=information_features(frames,t,person)
            y=np.zeros((4,4),np.float32);valid=np.zeros(4,np.float32)
            for j,h in enumerate((1,2,4,8)):
                if t+h>=len(frames):continue
                current=frames[t]['truth'][person];future=frames[t+h]['truth'][person]
                y[j,:2]=rotation@((future[:2]-current[:2])/(h*.25)-current[2:])/2
                y[j,2:]=rotation@(future[2:]-current[2:])/2
                valid[j]=1
            rows.append((x,oracle,y.ravel(),valid,t,person))
    result=dict(info['episode_result'],split=split,layout_sha256=layout_hash,frames=len(frames),
                events=dict(env.unwrapped.world.scheduler.counts))
    env.close()
    return result,frames,rows


def information_audit(args):
    """Frozen representation audit; only diagnostic predictors are trained."""
    import copy,multiprocessing,time
    root=args.results;root.mkdir(parents=True,exist_ok=False)
    source=Path('repair_results/dagger_ppo_nominal_20260915/initial.zip')
    protocol=dict(counts=dict(train=400,validation=100,audit=200),
        seed_starts=dict(train=82000000,validation=83000000,audit=84000000),
        case_starts=dict(train=110000,validation=120000,audit=130000),
        profile='train_nonstationary',humans=5,horizons_steps=[1,2,4,8],dt=.25,
        arms=['current','history','map','full','oracle'],
        neural_seeds=[2407,4807,7207],updates=6000,batch=512,lr=.001,
        predictor='180-128-128-32: diagonal Gaussian residual mean/log-std',
        checkpoint_selection='lowest validation masked NLL, every 1000 updates',
        primary='mean episode Gaussian NLL, horizons 4/8, averaged across three predictor seeds',
        gate='FULL gain >= .02 nats/dimension and simultaneous 98.333% bootstrap lower bound > 0 vs Current/History/MAP; >=2/3 seed directions positive',
        positive_control='Oracle current event type/remaining/strength, never available to policy',
        terminal_censoring='per-horizon mask; no post-terminal invented labels',
        initialization_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_params={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.params.glob('*.npz')},
        no_ppo_training=True,no_input_change=True,no_formal_heldout=True)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started=time.time();data={};seen_layouts=set()
    for split,count in protocol['counts'].items():
        records=[];xs=[];os=[];ys=[];ms=[];ids=[];steps=[];persons=[]
        tasks=[(str(args.params),str(source),split,i,protocol['seed_starts'][split]+i) for i in range(count)]
        with multiprocessing.get_context('spawn').Pool(6) as pool:
            for record,frames,rows in pool.imap(information_episode,tasks):
                if record['layout_sha256'] in seen_layouts:
                    raise AssertionError('Physical layout duplication: abort before predictor training')
                seen_layouts.add(record['layout_sha256'])
                ep=len(records);records.append(record)
                for x,o,y,m,t,person in rows:
                    xs.append(x);os.append(o);ys.append(y);ms.append(m);ids.append(ep);steps.append(t);persons.append(person)
                # Record event truth only for audit, separate from legal feature tensors.
                np.savez_compressed(root/f'{split}_episode_{ep:03d}.npz',
                    truth=np.stack([f['truth'] for f in frames]),
                    posterior=np.stack([f['humans'][:,5:9] for f in frames]),
                    oracle=np.stack([f['oracle'] for f in frames]),
                    klda=np.stack([f['klda'] for f in frames]))
                if len(records)%25==0:print('COLLECT',split,len(records),flush=True)
        data[split]=dict(x=np.stack(xs),oracle=np.stack(os),y=np.stack(ys),mask=np.stack(ms),
                         episode=np.array(ids),step=np.array(steps),person=np.array(persons))
        np.savez_compressed(root/f'{split}_features.npz',**data[split])
        (root/f'{split}_episodes.json').write_text(json.dumps(records,indent=2))
    print('COLLECTION_COMPLETE',round(time.time()-started,1),flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    ymean=np.zeros(16,np.float32);yscale=np.ones(16,np.float32)
    for h in range(4):
        v=data['train']['y'][data['train']['mask'][:,h]>0,h*4:h*4+4]
        ymean[h*4:h*4+4]=v.mean(0);yscale[h*4:h*4+4]=np.maximum(v.std(0),.02)
    def score(model,bundle):
        model.eval();parts=[];preds=[]
        with torch.no_grad():
            for start in range(0,len(bundle['x']),4096):
                x,y,m=(bundle[k][start:start+4096] for k in ('x','y','mask'))
                out=model(x);mu,ls=out[:,:16],out[:,16:].clamp(-5,3)
                nll=.5*((y-mu)*(-ls).exp()).square()+ls+.5*np.log(2*np.pi)
                parts.append(nll.cpu().numpy().reshape(-1,4,4).mean(-1));preds.append(mu.cpu().numpy())
        model.train()
        return np.concatenate(parts),np.concatenate(preds)
    results={};episode_scores={}
    for arm in protocol['arms']:
        raw={s:information_arm(d['x'],d['oracle'],arm) for s,d in data.items()}
        mean=raw['train'].mean(0);scale=np.maximum(raw['train'].std(0),.01)
        tensors={s:dict(x=torch.as_tensor((raw[s]-mean)/scale,device=device),
            y=torch.as_tensor((d['y']-ymean)/yscale,device=device),
            mask=torch.as_tensor(d['mask'],device=device)) for s,d in data.items()}
        results[arm]=[];episode_scores[arm]=[]
        for seed in protocol['neural_seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=torch.nn.Sequential(torch.nn.Linear(180,128),torch.nn.ReLU(),
                torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,32)).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');best_state=None
            curve=[]
            for update in range(6000):
                idx=rng.integers(len(raw['train']),size=512)
                x,y,m=(tensors['train'][k][idx] for k in ('x','y','mask'))
                out=model(x);mu,ls=out[:,:16],out[:,16:].clamp(-5,3)
                loss=((.5*((y-mu)*(-ls).exp()).square()+ls).reshape(-1,4,4).mean(-1)*m).sum()/m.sum()
                if not torch.isfinite(loss):raise AssertionError('Nonfinite predictor training')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
                if (update+1)%1000==0:
                    nll,_=score(model,tensors['validation']);mask=data['validation']['mask']
                    val=float((nll*mask).sum()/mask.sum());curve.append([update+1,val])
                    if val<best:best=val;best_state=copy.deepcopy(model.state_dict())
                    print('FIT',arm,seed,update+1,round(val,4),flush=True)
            model.load_state_dict(best_state);nll,pred=score(model,tensors['audit'])
            pred=pred*yscale+ymean;errors=(pred-data['audit']['y']).reshape(-1,4,4)
            position=np.linalg.norm(errors[:,:,:2],axis=-1)*np.array([.5,1.,2.,4.])
            velocity=np.linalg.norm(errors[:,:,2:],axis=-1)*2
            mask=data['audit']['mask'];eps=data['audit']['episode']
            scores=np.stack([np.divide((nll[eps==e]*mask[eps==e]).sum(0),mask[eps==e].sum(0),
                out=np.full(4,np.nan),where=mask[eps==e].sum(0)>0) for e in range(200)])
            episode_scores[arm].append(scores)
            results[arm].append(dict(seed=seed,validation_curve=curve,best_validation_nll=best,
                audit_nll=(nll*mask).sum(0).tolist(),audit_counts=mask.sum(0).tolist(),
                position_error=((position*mask).sum(0)/mask.sum(0)).tolist(),
                velocity_error=((velocity*mask).sum(0)/mask.sum(0)).tolist()))
            torch.save(dict(state=best_state,xmean=mean,xscale=scale,ymean=ymean,yscale=yscale),root/f'{arm}_{seed}.pt')
            np.savez_compressed(root/f'{arm}_{seed}_audit.npz',nll=nll,position_error=position,
                                velocity_error=velocity,episode_scores=scores)
        (root/'predictor_results.json').write_text(json.dumps(results,indent=2))
    rng=np.random.default_rng(9017);comparisons={}
    full=np.nanmean(np.asarray(episode_scores['full'])[:,:,2:],axis=-1)
    eligible=np.isfinite(full).all(0);full=full[:,eligible]
    idx=rng.integers(eligible.sum(),size=(20000,int(eligible.sum())))
    for arm in ['current','history','map','oracle']:
        competitor=np.nanmean(np.asarray(episode_scores[arm])[:,:,2:],axis=-1)[:,eligible]
        differences=competitor-full;d=differences.mean(0)
        interval=np.quantile(d[idx].mean(1),[.0083333333,.9916666667]).tolist()
        comparisons[arm]=dict(full_nll_gain=float(d.mean()),simultaneous_interval=interval,
            positive_seeds=int((differences.mean(1)>0).sum()),
            passed=bool(d.mean()>=.02 and interval[0]>0 and (differences.mean(1)>0).sum()>=2))
    passed=all(comparisons[a]['passed'] for a in ['current','history','map'])
    summary=dict(stage='complete',prediction_gate=passed,comparisons=comparisons,
        eligible_audit_episodes=int(eligible.sum()),
        elapsed_seconds=time.time()-started,decision_test_started=False,
        verdict='eligible_for_separate_decision_protocol' if passed else 'no_approval_for_long_PPO_on_this_evidence',
        limitation='Finite diagonal-Gaussian probe; a failure is not proof of information-theoretic equivalence.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


def information_ridge_audit(root):
    """Secondary fixed linear mean probe; does not replace the frozen NLL gate."""
    protocol=dict(reason='Cross-check Gaussian variance overfitting, not rescue a failed primary gate',
        arms=['current','history','map','full','oracle'],
        alpha=[.0001,.001,.01,.1,1.],selection='minimum validation standardized masked MSE',
        model='same 180 inputs plus intercept, linear residual predictor',
        predictor_training='train split only',variance_calibration='validation split only',
        original_gate_unchanged=True)
    if (root/'ridge_results.json').exists():raise ValueError('Do not overwrite a completed secondary audit')
    (root/'ridge_protocol.json').write_text(json.dumps(protocol,indent=2))
    data={s:dict(np.load(root/f'{s}_features.npz')) for s in ['train','validation','audit']}
    ym=np.zeros(16);ys=np.ones(16)
    for h in range(4):
        y=data['train']['y'][data['train']['mask'][:,h]>0,h*4:h*4+4]
        ym[h*4:h*4+4]=y.mean(0);ys[h*4:h*4+4]=np.maximum(y.std(0),.02)
    results={}
    for arm in protocol['arms']:
        raw={s:information_arm(d['x'],d['oracle'],arm).astype(np.float64) for s,d in data.items()}
        mean=raw['train'].mean(0);scale=np.maximum(raw['train'].std(0),.01)
        x={s:np.column_stack([(v-mean)/scale,np.ones(len(v))]) for s,v in raw.items()}
        y={s:(d['y']-ym)/ys for s,d in data.items()}
        grams=[];cross=[]
        for h in range(4):
            mask=data['train']['mask'][:,h]>0;a=x['train'][mask];b=y['train'][mask,h*4:h*4+4]
            grams.append(a.T@a/len(a));cross.append(a.T@b/len(a))
        best=float('inf');best_beta=None;curve=[]
        for alpha in protocol['alpha']:
            penalty=np.eye(181)*alpha;penalty[-1,-1]=0
            beta=np.concatenate([np.linalg.solve(g+penalty,c) for g,c in zip(grams,cross)],axis=1)
            error=(x['validation']@beta-y['validation']).reshape(-1,4,4)
            mask=data['validation']['mask'];loss=float((np.mean(error**2,axis=-1)*mask).sum()/mask.sum())
            curve.append([alpha,loss])
            if loss<best:best=loss;best_beta=beta;chosen=alpha
        calibration=(x['validation']@best_beta-y['validation']).reshape(-1,4,4)
        vmask=data['validation']['mask'][:,:,None]
        variance=np.maximum((calibration**2*vmask).sum(0)/vmask.sum(0),1e-4)
        normalized_error=(x['audit']@best_beta-y['audit']).reshape(-1,4,4)
        nll=(.5*normalized_error**2/variance+.5*np.log(2*np.pi*variance)).mean(-1)
        error=normalized_error*ys.reshape(1,4,4)
        position=np.linalg.norm(error[:,:,:2],axis=-1)*[.5,1.,2.,4.]
        mask=data['audit']['mask'];episode=data['audit']['episode']
        ep_position=np.stack([(position[episode==e]*mask[episode==e]).sum(0)/mask[episode==e].sum(0) for e in range(200)])
        results[arm]=dict(alpha=chosen,validation_curve=curve,
            position_error=((position*mask).sum(0)/mask.sum(0)).tolist(),
            nll=((nll*mask).sum(0)/mask.sum(0)).tolist())
        np.savez_compressed(root/f'ridge_{arm}.npz',beta=best_beta,xmean=mean,xscale=scale,
            ymean=ym,yscale=ys,variance=variance,episode_position=ep_position,nll=nll)
        print('RIDGE',arm,results[arm],flush=True)
    (root/'ridge_results.json').write_text(json.dumps(results,indent=2))


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
    if final_round==16:
        baseline_path = original_collection.parent/'bc_only.zip'
        baseline = BayesSetTD3.load(baseline_path,device='cpu')
        baseline.check_arm('no_belief')
        old_records,_,_ = episodes(params,count,offset,baseline,case_offset=offset,
                                   arm='no_belief',diagnostics=True)
        assert [r['layout_sha256'] for r in old_records]==[r['layout_sha256'] for r in records]
        report['baseline'] = dict(checkpoint_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            metrics=receipt(old_records),records=old_records)
        report['paired_outcome_transitions'] = {
            f'{a}->{b}':sum(x['outcome']==a and y['outcome']==b for x,y in zip(old_records,records))
            for a in ('success','collision','timeout') for b in ('success','collision','timeout')}
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


def reciprocity_world(params, source):
    from stable_baselines3 import PPO
    from crowd_nav.bayes_continuous.train_smoke import ActionHistory
    env = ActionHistory(BeliefEnv(params, arm='no_belief'), route=True)
    env.unwrapped.world.robot.visible = True
    return env, PPO.load(source, device='cpu')


def reciprocity_reset(env, case, flags):
    from types import MethodType
    obs, _ = env.reset(options=dict(layout_seed=91000000+case,
                                    test_case=case, profile='nominal'))
    world = env.unwrapped.world
    assert world.robot.visible and len(world.env.humans) == 5
    for i, (human, policy) in enumerate(zip(world.env.humans, world.policies)):
        policy.set_reciprocity(bool(flags[i]))
        original = policy.predict
        def checked(self, state, original=original):
            expected = world.robot.get_observable_state()
            assert world.robot.visible and len(state.human_states) == 5
            np.testing.assert_array_equal(state.human_states[-1].to_array(), expected.to_array())
            return original(state)
        policy.predict = MethodType(checked, policy)
    return obs


def reciprocity_state(env, obs):
    world = env.unwrapped.world
    truth = np.array([[h.px,h.py,h.vx,h.vy] for h in world.env.humans],np.float64)
    # Private goals are used ONLY to verify identical replay, never as model input.
    audit = np.array([h.get_full_state().to_array() for h in world.env.humans] +
                     [world.robot.get_full_state().to_array()],np.float64)
    memory = [None if p.base._last_pref_vel is None else
              np.asarray(p.base._last_pref_vel).tolist() for p in world.policies]
    digest = hashlib.sha256(audit.tobytes()+obs['robot'].tobytes()+
                           json.dumps(memory).encode()).hexdigest()
    return dict(robot=obs['robot'].copy(), humans=obs['humans'][:5,:5].copy(),
                truth=truth, theta=float(world.robot.theta), state_hash=digest)


def reciprocity_physical(frames, t, person):
    order = [person]+[i for i in range(5) if i != person]
    return np.concatenate([frames[t]['robot'], frames[t]['humans'][order].ravel()])


def reciprocity_collect(task):
    params, source, case, gate = task
    torch.set_num_threads(1)
    env, actor = reciprocity_world(params, source)
    flags = np.random.default_rng(92000000+case).integers(0,2,5)
    obs = reciprocity_reset(env,case,flags)
    initial = reciprocity_state(env,obs)
    frames = [initial]; actions = []; anchors = []
    done = False
    for t in range(140):
        action = actor.predict(obs,deterministic=True)[0].astype(np.float32)
        if t in (16,32,48):
            anchors.append(dict(t=t,action=action.copy(),state_hash=frames[-1]['state_hash']))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(reciprocity_state(env,obs));done=terminated or truncated
        if done:break
    assert done
    branches=[]
    for anchor in anchors:
        t=anchor['t'];base=anchor['action']
        candidates = np.unique(np.array([[np.clip(base[0]+dv,0,1),
            np.clip(base[1]+dw,-1.2,1.2)] for dv in (-.2,0,.2)
            for dw in (-.4,0,.4)],np.float32),axis=0)
        if gate:candidates=np.array([[.3,-.8],[1.,0.],[.3,.8]],np.float32)
        for flip in ((False,True) if gate else (False,)):
            for j, action in enumerate(candidates):
                obs=reciprocity_reset(env,case,flags)
                for prefix in actions[:t]:
                    obs,_,term,trunc,_=env.step(prefix)
                    assert not (term or trunc)
                assert reciprocity_state(env,obs)['state_hash']==anchor['state_hash']
                if flip:
                    env.unwrapped.world.policies[0].set_reciprocity(not bool(flags[0]))
                future=[];ret=0.;outcome='running'
                for h in range(8):
                    obs,reward,term,trunc,info=env.step(action)
                    future.append(reciprocity_state(env,obs)['truth'])
                    ret += .99**h*reward
                    if term or trunc:
                        outcome=info['episode_result'].get('outcome','terminal');break
                branches.append(dict(t=t,candidate=j,action=action,future=np.array(future),
                                     flipped=flip,return8=ret,outcome=outcome))
    env.close()
    return dict(case=case,layout_hash=initial['state_hash'],flags=flags,
                frames=frames,actions=actions,branches=branches)


def reciprocity_audit(args):
    """Frozen diagnostic only; never updates a navigation policy or old GDBN."""
    import multiprocessing, time, copy
    from scipy.special import expit
    start=time.time();root=args.results
    resume=getattr(args,'reciprocity_resume',False)
    root.mkdir(parents=True,exist_ok=resume)
    source='repair_results/dagger_ppo_nominal_20260915/initial.zip'
    protocol=dict(stage='frozen_before_collection',robot_visible=True,
        intervention_profile='nominal',persistent_type_prior=.5,
        counts=dict(environment=30,train=180,validation=60,audit=120),
        case_starts=dict(environment=210000,train=220000,validation=230000,audit=240000),
        anchors=[16,32,48],future_steps=[4,8],candidate_hold_steps=8,
        candidate_offsets=dict(v=[-.2,0,.2],omega=[-.4,0,.4]),
        environment_gate='At least 10% matched branches change target position by >0.01m at 8 steps; at least 10% anchors respond to candidate change >0.01m; exact first-step action-delay check',
        posterior='Two conditional Gaussian ridge velocity-residual likelihoods fitted with TRAIN type labels only; fixed type, prior 0.5, sequential log-odds, no private goals; validation variance calibration; no type labels at inference',
        likelihood_ridge=1.,likelihood_variance_floor=.0001,
        predictor='148 -> 128 -> 128 -> 8; MSE on training-standardized CV residuals',
        predictor_seeds=[2407,4807,7207],updates=3000,batch=256,lr=.001,
        selection='lowest validation MSE each 500 updates',
        information_gate='FULL reduces 1/2-second position MSE by >=2% vs EACH Current/History/MAP; paired episode 98.333% bootstrap lower bound >0; positive direction in >=2/3 seeds',
        bootstrap_replicates=10000,bootstrap_seed=9017,
        decision_gate='Only after information pass: same candidates held8 then fixed source Actor to terminal, replay exact prefix; equal-capacity return/collision probes on same splits; FULL regret at least .01 below every comparator with 98.333% lower CI>0 and no higher collision selection rate',
        no_extra_sampling=True,no_PPO=True,source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Known simulator type labels allowed ONLY for training likelihood models',
                    'Fixed behavior policy, binary simulator types, not real-human evidence',
                    'Terminal branches masked; no invented post-terminal motion',
                    'Finite diagnostic regressors; failure is not information-theoretic equivalence'])
    if resume:
        protocol=json.loads((root/'protocol.json').read_text())
        assert protocol['source_sha256']==hashlib.sha256(Path(source).read_bytes()).hexdigest()
        (root/'implementation_repair.json').write_text(json.dumps(dict(
            reason='Zero valid future targets for some terminated episodes caused division by zero; apply one shared mask across every arm',
            no_new_data=True,no_gate_change=True,
            resumed_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    else:
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def collect(split,gate=False):
        if resume:
            return torch.load(root/(split+'_raw.pt'),weights_only=False)
        tasks=[(str(args.params),source,protocol['case_starts'][split]+i,gate)
               for i in range(protocol['counts'][split])]
        with multiprocessing.get_context('spawn').Pool(6) as pool:
            data=[]
            for row in pool.imap(reciprocity_collect,tasks):
                data.append(row)
                if len(data)%10==0:print(split,len(data),flush=True)
        torch.save(data,root/(split+'_raw.pt'))
        return data
    envdata=collect('environment',True)
    differences=[];action_effect=[];delay_errors=[]
    for ep in envdata:
        for t in (16,32,48):
            rows=[b for b in ep['branches'] if b['t']==t]
            for j in range(3):
                a=[b for b in rows if b['candidate']==j and not b['flipped']]
                b=[b for b in rows if b['candidate']==j and b['flipped']]
                if a and b and len(a[0]['future'])==len(b[0]['future'])==8:
                    differences.append(float(np.linalg.norm(a[0]['future'][-1,0,:2]-b[0]['future'][-1,0,:2])))
            for flip in (False,True):
                bs=[b for b in rows if b['flipped']==flip]
                if len(bs)==3:
                    delay_errors.extend(float(np.max(np.abs(b['future'][0]-bs[0]['future'][0]))) for b in bs)
                    if all(len(b['future'])==8 for b in bs):
                        action_effect.append(max(float(np.linalg.norm(b['future'][-1,:,:2]-bs[0]['future'][-1,:,:2],axis=1).max()) for b in bs))
    envresult=dict(matched_branches=len(differences),type_position_change_rate=float(np.mean(np.array(differences)>.01)),
        action_position_change_rate=float(np.mean(np.array(action_effect)>.01)),
        max_first_step_action_effect=max(delay_errors,default=0.),
        passed=bool(differences and np.mean(np.array(differences)>.01)>=.1 and
                    action_effect and np.mean(np.array(action_effect)>.01)>=.1 and max(delay_errors,default=0.)<1e-9))
    (root/'environment_gate.json').write_text(json.dumps(envresult,indent=2));print(envresult,flush=True)
    if not envresult['passed']:
        (root/'summary.json').write_text(json.dumps(dict(stage='stopped_environment_gate',environment=envresult,no_PPO=True),indent=2));return
    raw={split:collect(split) for split in ('train','validation','audit')}
    hashes=[e['layout_hash'] for data in raw.values() for e in data]
    assert len(set(hashes))==len(hashes)
    # Fit emission models on actual executed transitions, never branch outcomes.
    emissions={}
    for split in ('train','validation'):
        xs=[];ys=[];labels=[]
        for ep in raw[split]:
            fs=ep['frames']
            for t in range(len(fs)-1):
                c,s=np.cos(fs[t]['theta']),np.sin(fs[t]['theta']);rot=np.array([[c,s],[-s,c]])
                for p in range(5):
                    xs.append(reciprocity_physical(fs,t,p))
                    ys.append(rot@(fs[t+1]['truth'][p,2:]-fs[t]['truth'][p,2:]))
                    labels.append(ep['flags'][p])
        emissions[split]=(np.array(xs),np.array(ys),np.array(labels))
    ex,ey,et=emissions['train'];emu=ex.mean(0);esd=np.maximum(ex.std(0),.01)
    def design(x):return np.column_stack([(x-emu)/esd,np.ones(len(x))])
    weights=[];variances=[]
    for kind in (0,1):
        x=design(ex[et==kind]);y=ey[et==kind]
        reg=np.eye(x.shape[1]);reg[-1,-1]=0
        w=np.linalg.solve(x.T@x+reg,x.T@y);weights.append(w)
        vx,vy,vt=emissions['validation'];err=vy[vt==kind]-design(vx[vt==kind])@w
        variances.append(np.maximum(np.mean(err**2,axis=0),.0001))
    np.savez_compressed(root/'likelihood.npz',mean=emu,std=esd,weights=weights,variance=variances)
    datasets={};poststats={}
    for split,episodes in raw.items():
        xs=[];ys=[];masks=[];ids=[];ps=[];truths=[];records=[]
        for ei,ep in enumerate(episodes):
            fs=ep['frames'];belief=np.full((len(fs),5),.5);logodds=np.zeros(5)
            for t in range(1,len(fs)):
                c,s=np.cos(fs[t-1]['theta']),np.sin(fs[t-1]['theta']);rot=np.array([[c,s],[-s,c]])
                x=design(np.array([reciprocity_physical(fs,t-1,p) for p in range(5)]))
                y=(fs[t]['truth'][:,2:]-fs[t-1]['truth'][:,2:])@rot.T
                ll=[-.5*np.sum((y-x@weights[k])**2/variances[k]+np.log(variances[k]),axis=1) for k in (0,1)]
                logodds+=ll[1]-ll[0];belief[t]=expit(logodds)
            ps.extend(belief[4:].ravel());truths.extend(np.tile(ep['flags'],len(fs)-4))
            for bi,b in enumerate(ep['branches']):
                t=b['t'];c,s=np.cos(fs[t]['theta']),np.sin(fs[t]['theta']);rot=np.array([[c,s],[-s,c]])
                for p in range(5):
                    x=np.zeros(148,np.float32)
                    x[:35]=reciprocity_physical(fs,t,p);x[35:37]=b['action']/[1,1.2]
                    x[37:142]=np.concatenate([reciprocity_physical(fs,t-j,p) for j in (1,2,3)])
                    order=[p]+[i for i in range(5) if i!=p];x[142:147]=belief[t,order]
                    y=np.zeros((2,4),np.float32);mask=np.zeros((2,4),np.float32)
                    for j,h in enumerate((4,8)):
                        if len(b['future'])<h:continue
                        now=fs[t]['truth'][p];future=b['future'][h-1,p]
                        y[j,:2]=rot@(future[:2]-now[:2]-now[2:]*h*.25)
                        y[j,2:]=rot@(future[2:]-now[2:]);mask[j]=1
                    xs.append(x);ys.append(y.ravel());masks.append(mask.ravel());ids.append(ei);records.append((ei,bi,p))
        datasets[split]=(np.array(xs),np.array(ys),np.array(masks),np.array(ids))
        ps=np.array(ps);truths=np.array(truths)
        poststats[split]=dict(brier=float(np.mean((ps-truths)**2)),accuracy=float(np.mean((ps>.5)==truths)),
            ambiguity_fraction=float(np.mean((ps>.1)&(ps<.9))))
        np.savez_compressed(root/(split+'_features.npz'),x=xs,y=ys,mask=masks,episode=ids,record=records)
    (root/'posterior_diagnostics.json').write_text(json.dumps(poststats,indent=2))
    train=datasets['train'];xmu=train[0].mean(0);xsd=np.maximum(train[0].std(0),.01)
    ymu=np.sum(train[1]*train[2],0)/train[2].sum(0)
    ysd=np.maximum(np.sqrt(np.sum((train[1]-ymu)**2*train[2],0)/train[2].sum(0)),.01)
    audit_ids=datasets['audit'][3];audit_mask=datasets['audit'][2]
    eligible=np.array([i for i in range(len(raw['audit']))
                       if audit_mask[audit_ids==i][:,[0,1,4,5]].sum()>0])
    (root/'target_availability.json').write_text(json.dumps(dict(
        total_audit_episodes=len(raw['audit']),eligible_episodes=len(eligible),
        no_valid_future_cases=[e['case'] for i,e in enumerate(raw['audit']) if i not in eligible],
        reason='No observed 1/2-second target; same inclusion for all arms, no outcome-based model selection'),indent=2))
    results={};episode_scores={}
    for arm in ('current','history','map','full'):
        tensors={}
        for split,(x,y,m,ids) in datasets.items():
            x=x.copy()
            if arm!='history':x[:,37:142]=0
            if arm in ('current','history'):x[:,142:]=0
            if arm=='map':x[:,142:147]=(x[:,142:147]>.5).astype(float)
            # Mask AFTER standardization, so absent blocks are exactly zero.
            z=(x-xmu)/xsd
            if arm!='history':z[:,37:142]=0
            if arm in ('current','history'):z[:,142:]=0
            tensors[split]=tuple(torch.as_tensor(a,device='cuda',dtype=torch.float32)
                                for a in (z,(y-ymu)/ysd,m))
        results[arm]=[];episode_scores[arm]=[]
        for seed in protocol['predictor_seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=torch.nn.Sequential(torch.nn.Linear(148,128),torch.nn.ReLU(),
                torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8)).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=.001);best=None;bestloss=float('inf')
            for update in range(1,3001):
                idx=rng.integers(len(train[0]),size=256);x,y,m=tensors['train']
                loss=(((model(x[idx])-y[idx])**2)*m[idx]).sum()/m[idx].sum()
                optimizer.zero_grad();loss.backward();optimizer.step()
                if update%500==0:
                    with torch.no_grad():
                        x,y,m=tensors['validation'];vl=float((((model(x)-y)**2)*m).sum()/m.sum())
                    if vl<bestloss:bestloss=vl;best=copy.deepcopy(model.state_dict());beststep=update
            model.load_state_dict(best)
            with torch.no_grad():pred=model(tensors['audit'][0]).cpu().numpy()*ysd+ymu
            y=datasets['audit'][1];m=datasets['audit'][2];ids=datasets['audit'][3]
            cols=np.array([0,1,4,5]);sq=(pred[:,cols]-y[:,cols])**2;mm=m[:,cols]
            scores=np.array([(sq[ids==i]*mm[ids==i]).sum()/mm[ids==i].sum()
                             for i in eligible])
            assert np.isfinite(scores).all()
            episode_scores[arm].append(scores)
            metrics=dict(seed=seed,validation_loss=bestloss,selected_update=beststep,
                         position_mse=float(scores.mean()))
            for j,h in enumerate((4,8)):
                valid=m[:,4*j]>0
                metrics['position_error_'+str(h)]=float(np.linalg.norm(pred[valid,4*j:4*j+2]-y[valid,4*j:4*j+2],axis=1).mean())
            results[arm].append(metrics);print(arm,metrics,flush=True)
            torch.save(model.state_dict(),root/(arm+'_'+str(seed)+'.pt'))
            (root/'predictor_results.json').write_text(json.dumps(results,indent=2))
    comparisons={};rng=np.random.default_rng(9017)
    full=np.mean(episode_scores['full'],axis=0)
    for arm in ('current','history','map'):
        other=np.mean(episode_scores[arm],axis=0);delta=other-full
        draws=rng.integers(len(delta),size=(10000,len(delta)))
        lo,hi=np.quantile(delta[draws].mean(1),[.008333333,.991666667])
        gain=float(delta.mean()/other.mean());positive=int(np.sum(np.mean(episode_scores[arm],axis=1)>np.mean(episode_scores['full'],axis=1)))
        comparisons[arm]=dict(relative_mse_gain=gain,absolute_ci=[float(lo),float(hi)],
                              positive_seeds=positive,passed=bool(gain>=.02 and lo>0 and positive>=2))
    passed=all(x['passed'] for x in comparisons.values())
    summary=dict(stage='information_complete',environment=envresult,information_gate=passed,
        comparisons=comparisons,posterior=poststats,elapsed_seconds=time.time()-start,
        decision_gate='pending' if passed else 'not_started_information_failed',PPO_started=False)
    np.savez_compressed(root/'episode_scores.npz',**{k:np.array(v) for k,v in episode_scores.items()})
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


def type_oracle_value(returns, collisions):
    """Perfect-information value within one shared finite candidate set."""
    returns=np.asarray(returns,dtype=float);collisions=np.asarray(collisions,dtype=float)
    assert returns.shape==collisions.shape and returns.shape[0]==2
    def best(values):return int(np.flatnonzero(values>=values.max()-1e-8)[0])
    unknown=best(returns.mean(0));known=[best(row) for row in returns]
    informed=float(np.mean([returns[k,known[k]] for k in range(2)]))
    blind=float(returns[:,unknown].mean())
    return dict(gain=informed-blind,known_return=informed,unknown_return=blind,
        known_choices=known,unknown_choice=unknown,
        known_collision=float(np.mean([collisions[k,known[k]] for k in range(2)])),
        unknown_collision=float(collisions[:,unknown].mean()),
        baseline_return=float(returns[:,0].mean()),baseline_collision=float(collisions[:,0].mean()),
        risk_only_value=float(collisions.mean(0).min()-collisions.min(1).mean()))


def type_oracle_episode(task):
    params,source,case=task;torch.set_num_threads(1)
    env,actor=reciprocity_world(params,source)
    flags=np.random.default_rng(92000000+case).integers(0,2,5)
    obs=reciprocity_reset(env,case,flags)
    initial=reciprocity_state(env,obs);prefix=[];anchors=[]
    for t in range(140):
        action=actor.predict(obs,deterministic=True)[0].astype(np.float32)
        if t in (16,32,48):
            world=env.unwrapped.world
            clearances=np.array([np.hypot(h.px-world.robot.px,h.py-world.robot.py)-
                h.radius-world.robot.radius for h in world.env.humans])
            anchors.append(dict(step=t,target=int(clearances.argmin()),
                clearance=float(clearances.min()),action=action.copy(),
                state_hash=reciprocity_state(env,obs)['state_hash']))
        prefix.append(action)
        obs,_,term,trunc,_=env.step(action)
        if term or trunc:break
    else:raise AssertionError('Source episode failed to terminate')
    if not anchors:
        env.close();return dict(case=case,eligible=False,layout_hash=initial['state_hash'])
    anchor=min(anchors,key=lambda a:(a['clearance'],a['step']))
    base=anchor['action'];t=anchor['step'];target=anchor['target']
    offsets=np.unique(np.array([[np.clip(base[0]+dv,0,1),np.clip(base[1]+dw,-1.2,1.2)]
        for dv in (-.2,0,.2) for dw in (-.4,0,.4)]+[[0.,0.]],np.float32),axis=0)
    candidates=[None]+list(offsets)  # The unchanged source policy is a feasible common option.
    returns=np.zeros((2,len(candidates)));collisions=np.zeros_like(returns);records=[]
    first_actions=[]
    for kind in (0,1):
        for j,candidate in enumerate(candidates):
            obs=reciprocity_reset(env,case,flags)
            for action in prefix[:t]:
                obs,_,term,trunc,_=env.step(action);assert not(term or trunc)
            assert reciprocity_state(env,obs)['state_hash']==anchor['state_hash']
            env.unwrapped.world.policies[target].set_reciprocity(bool(kind))
            rewards=[];executed=[];clearance=float('inf')
            for h in range(140-t):
                action=(candidate if candidate is not None and h<8 else
                        actor.predict(obs,deterministic=True)[0]).astype(np.float32)
                obs,reward,term,trunc,info=env.step(action)
                rewards.append(float(reward));executed.append(action.tolist())
                clearance=min(clearance,float(info['actual_clearance']))
                if term or trunc:break
            else:raise AssertionError('Counterfactual did not terminate')
            returns[kind,j]=sum(.99**h*r for h,r in enumerate(rewards))
            collisions[kind,j]=info['outcome']=='collision'
            records.append(dict(kind=kind,candidate=j,rewards=rewards,executed=executed,
                outcome=info['outcome'],steps=len(rewards),minimum_clearance=clearance))
            if j==0:first_actions.append(executed[0])
    np.testing.assert_array_equal(first_actions[0],first_actions[1])
    value=type_oracle_value(returns,collisions)
    assert value['gain']>=-1e-7
    env.close()
    return dict(case=case,eligible=True,layout_hash=initial['state_hash'],anchor=anchor,
        flags=flags,prefix=prefix[:t],candidates=candidates,returns=returns,
        collisions=collisions,records=records,value=value)


def type_oracle_prediction(root, data_root):
    import copy
    raw={s:torch.load(data_root/(s+'_raw.pt'),weights_only=False)
         for s in ('train','validation','audit')}
    data={s:dict(np.load(data_root/(s+'_features.npz'))) for s in raw}
    for split,d in data.items():
        d['x']=d['x'].copy();d['x'][:,37:]=0
        for row,(ei,bi,p) in enumerate(d['record']):
            order=[p]+[i for i in range(5) if i!=p]
            d['x'][row,142:147]=raw[split][ei]['flags'][order]
    xmu=data['train']['x'].mean(0);xsd=np.maximum(data['train']['x'].std(0),.01)
    y=data['train']['y'];mask=data['train']['mask'];ymu=(y*mask).sum(0)/mask.sum(0)
    ysd=np.maximum(np.sqrt(((y-ymu)**2*mask).sum(0)/mask.sum(0)),.01)
    ids=data['audit']['episode'];am=data['audit']['mask']
    eligible=[i for i in range(len(raw['audit'])) if am[ids==i][:,[0,1,4,5]].sum()>0]
    results={};scores={}
    for arm in ('current','oracle_type'):
        tensors={}
        for split,d in data.items():
            x=(d['x']-xmu)/xsd;x[:,37:142]=0
            if arm=='current':x[:,142:]=0
            tensors[split]=tuple(torch.as_tensor(a,device='cuda',dtype=torch.float32)
                for a in (x,(d['y']-ymu)/ysd,d['mask']))
        results[arm]=[];scores[arm]=[]
        for seed in (2407,4807,7207):
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=torch.nn.Sequential(torch.nn.Linear(148,128),torch.nn.ReLU(),
                torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8)).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=.001);bestloss=float('inf')
            for step in range(1,3001):
                idx=rng.integers(len(y),size=256);x,yy,m=tensors['train']
                loss=(((model(x[idx])-yy[idx])**2)*m[idx]).sum()/m[idx].sum()
                optimizer.zero_grad();loss.backward();optimizer.step()
                if step%500==0:
                    with torch.no_grad():
                        x,yy,m=tensors['validation'];v=float((((model(x)-yy)**2)*m).sum()/m.sum())
                    if v<bestloss:bestloss=v;best=copy.deepcopy(model.state_dict());beststep=step
            model.load_state_dict(best)
            with torch.no_grad():pred=model(tensors['audit'][0]).cpu().numpy()*ysd+ymu
            yy=data['audit']['y'];cols=[0,1,4,5];sq=(pred[:,cols]-yy[:,cols])**2;mm=am[:,cols]
            sc=np.array([(sq[ids==i]*mm[ids==i]).sum()/mm[ids==i].sum() for i in eligible])
            assert np.isfinite(sc).all();scores[arm].append(sc)
            r=dict(seed=seed,selected_update=beststep,validation_loss=bestloss,position_mse=float(sc.mean()))
            for j,h in enumerate((4,8)):
                valid=am[:,4*j]>0
                r['position_error_'+str(h)]=float(np.linalg.norm(pred[valid,4*j:4*j+2]-yy[valid,4*j:4*j+2],axis=1).mean())
            results[arm].append(r);print('PREDICTION_ORACLE',arm,r,flush=True)
            torch.save(model.state_dict(),root/(arm+'_'+str(seed)+'.pt'))
    delta=np.mean(scores['current'],0)-np.mean(scores['oracle_type'],0)
    rng=np.random.default_rng(9017);bs=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    ci=np.quantile(bs,[.025,.975]);gain=float(delta.mean()/np.mean(scores['current']))
    positives=int(np.sum(np.mean(scores['current'],1)>np.mean(scores['oracle_type'],1)))
    report=dict(results=results,relative_mse_gain=gain,absolute_ci=ci.tolist(),
        positive_seeds=positives,passed=bool(gain>=.02 and ci[0]>0 and positives>=2),
        limitation='Finite learned predictor: failure cannot reject value of true type')
    (root/'prediction_oracle.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(root/'prediction_episode_scores.npz',**{k:np.array(v) for k,v in scores.items()})
    return report


def type_oracle_audit(args):
    import multiprocessing,time
    start=time.time();root=args.results;root.mkdir(parents=True,exist_ok=False)
    data_root=Path('repair_results/reciprocity_diagnostic_20260915')
    source='repair_results/dagger_ppo_nominal_20260915/initial.zip'
    protocol=dict(frozen_before_run=True,prediction_data=str(data_root),
        prediction_data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for split in ('train','validation','audit')
            for p in (data_root/(split+'_raw.pt'),data_root/(split+'_features.npz'))},
        prediction_arms=['current','oracle_type'],predictor='148->128->128->8 same parameter count',
        seeds=[2407,4807,7207],updates=3000,batch=256,lr=.001,
        validation_selection_every=500,prediction_gate='MSE improvement >=2%, paired 95% lower CI>0, >=2/3 seeds',
        prediction_failure_does_not_stop_decision=True,decision_cases=list(range(270000,270080)),
        decision_anchor='One of steps16/32/48 with minimum current surface clearance; nearest person target; no outcome-based selection',
        candidates='Source feedback policy plus clipped 3x3 offsets dv[-.2,0,.2],dw[-.4,0,.4] and stop; non-feedback candidates held8 steps then source Actor to terminal',
        type_worlds='Target reciprocal vs non-reciprocal after identical prefix; all other state, goals, ORCA memory and types held fixed',
        unknown_type_prior=[.5,.5],gamma=.99,
        value='0.5*max_a R(type0,a)+0.5*max_a R(type1,a)-max_a[0.5*R(type0,a)+0.5*R(type1,a)]',
        decision_gate='Mean value >=0.01, paired episode bootstrap95% lower>0, >=10% states value>0.01, known-type selected collision rate <= unknown-type selected collision rate',
        bootstrap_replicates=10000,bootstrap_seed=9017,no_critic=True,no_policy_training=True,
        limitations=['Finite candidate set, fixed continuation policy and nearest-person type only; not global POMDP upper bound',
                    'Counterfactual 50/50 worlds do not prove natural histories remain ambiguous or that legal type inference is possible',
                    'Proximity-enriched states: rates are not natural deployment prevalence',
                    'Prediction Oracle is a learned finite probe, not an exact oracle'],
        actor_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    prediction=type_oracle_prediction(root,data_root)
    tasks=[(str(args.params),source,c) for c in protocol['decision_cases']]
    rows=[]
    with multiprocessing.get_context('spawn').Pool(6) as pool:
        for row in pool.imap(type_oracle_episode,tasks):
            rows.append(row)
            if len(rows)%10==0:print('DECISION_ORACLE',len(rows),flush=True)
    torch.save(rows,root/'decision_raw.pt')
    assert len({r['layout_hash'] for r in rows})==len(rows)
    valid=[r for r in rows if r['eligible']];values=np.array([r['value']['gain'] for r in valid])
    assert len(valid)>0
    rng=np.random.default_rng(9017);draws=rng.integers(len(valid),size=(10000,len(valid)))
    ci=np.quantile(values[draws].mean(1),[.025,.975])
    def average(key):return float(np.mean([r['value'][key] for r in valid]))
    known=average('known_collision');unknown=average('unknown_collision')
    fraction=float(np.mean(values>.01))
    result=dict(eligible_cases=len(valid),ineligible_cases=[r['case'] for r in rows if not r['eligible']],
        branches=sum(len(r['records']) for r in valid),mean_value=float(values.mean()),
        median_value=float(np.median(values)),maximum_value=float(values.max()),value_ci=ci.tolist(),
        meaningful_fraction=fraction,known_collision=known,unknown_collision=unknown,
        known_return=average('known_return'),unknown_return=average('unknown_return'),
        baseline_return=average('baseline_return'),baseline_collision=average('baseline_collision'),
        risk_only_value=average('risk_only_value'),
        passed=bool(values.mean()>=.01 and ci[0]>0 and fraction>=.1 and known<=unknown+1e-12))
    (root/'decision_oracle.json').write_text(json.dumps(result,indent=2))
    (root/'decision_cases.json').write_text(json.dumps([dict(case=r['case'],target=r['anchor']['target'],
        step=r['anchor']['step'],clearance=r['anchor']['clearance'],**r['value']) for r in valid],indent=2))
    summary=dict(stage='complete',prediction_gate=prediction['passed'],decision=result,
        elapsed_seconds=time.time()-start,PPO_started=False,Bayes_rebuilt=False,
        verdict='approve_likelihood_research_not_PPO' if result['passed'] else 'no_approval_under_this_local_oracle_protocol')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


def failure_evidence_case(task):
    """Paired invisible-human counterfactuals; no learned weights are updated."""
    from stable_baselines3 import PPO
    from crowd_nav.bayes_continuous.train_smoke import ActionHistory
    from crowd_nav.bayes_continuous.teacher import PlannerObservation, UnicycleConfig, UnicycleCEMMPC
    params, source, spec = task
    torch.set_num_threads(1)
    if spec['profile']=='nominal':
        from crowd_nav.bayes_continuous.train_smoke import dagger_artifact
        model=BayesSetTD3.load(dagger_artifact(Path('repair_results/student_dagger_coverage_20260915'),'round16.zip'),device='cpu')
    else:
        model = PPO.load(source, device='cpu')
    env = ActionHistory(BeliefEnv(params, arm='no_belief'), route=True)
    def reset():
        observation, _ = env.reset(options=dict(layout_seed=spec['layout'], test_case=spec['case'], profile=spec['profile']))
        assert not env.unwrapped.world.robot.visible
        return observation
    def humans():
        return np.asarray([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.unwrapped.world.env.humans], np.float64)
    obs = reset(); frames=[]; actions=[]; hashes=[]
    for t in range(140):
        frames.append(humans()); hashes.append(reciprocity_state(env,obs)['state_hash'])
        action = model.predict(obs, deterministic=True)[0]
        if spec.get('trace'):
            expected=spec['trace'][t]
            np.testing.assert_allclose(humans(),expected['humans'],atol=2e-5,rtol=0)
            np.testing.assert_allclose(action,expected['action'],atol=2e-5,rtol=0)
        if spec.get('truth_file'):
            # These files are audit-only and never enter either legal selector.
            if t==0: recorded=np.load(spec['truth_file'])['truth']
            np.testing.assert_allclose(humans()[:,:4],recorded[t],atol=2e-5,rtol=0)
        actions.append(action.copy())
        obs,_,done,truncated,info=env.step(action)
        if done or truncated: break
    assert not spec.get('expected') or info['outcome']==spec['expected']
    baseline=info['outcome']; length=len(actions)
    if spec['fresh']:
        available=[t for t in (16,32) if t<length]
        # Fixed time, independent of future outcomes, for the confirmation set.
        anchors=[available[0]] if available else []
    else:
        anchors=sorted(set(max(3,length-k) for k in (8,16))) if baseline=='collision' else [64,112]
        anchors=[t for t in anchors if t<length]
    def replay(t):
        o=reset()
        for a in actions[:t]:
            o,_,d,tr,_=env.step(a)
            assert not (d or tr)
        assert reciprocity_state(env,o)['state_hash']==hashes[t]
        return o
    records=[]
    planner=UnicycleCEMMPC(UnicycleConfig(horizon=8,omega_max=1.2,human_margin=.50))
    for t in anchors:
        obs=replay(t); world=env.unwrapped.world; robot=world.robot; current=humans()
        po=PlannerObservation(np.array([robot.px,robot.py]),np.array([robot.vx,robot.vy]),robot.radius,
            np.array([robot.gx,robot.gy]),current,None,None,None,None,None,None,None,None,'diagnostic',robot.theta)
        initial=actions[t]
        bank=[np.clip(initial+np.array([dv,dw]),[0,-1.2],[1,1.2]) for dv in (-.2,0,.2) for dw in (-.4,0,.4)]
        bank.append(np.array([0.,0.]))
        samples=np.repeat(np.asarray(bank)[:,None,:],8,axis=1); samples[:,:,1]*=.25
        controls,velocities,positions=planner._rollout(samples,po)
        executable=controls.copy(); executable[:,:,1]/=.25
        times=np.arange(1,9)*.25
        cv=current[:,None,:2]+current[:,None,2:4]*times[None,:,None]
        acceleration=(frames[t][:,2:4]-frames[t-3][:,2:4])/.75
        norms=np.linalg.norm(acceleration,axis=1,keepdims=True)
        acceleration*=np.minimum(1.,1./np.maximum(norms,1e-12))
        # Fixed one-second acceleration decay, rather than unbounded extrapolation.
        history=cv+acceleration[:,None,:]*(times-1+np.exp(-times))[None,:,None]
        future=[]
        for k in range(8):
            world.scheduler.advance(world.policies)
            people=world.env.humans
            ha=[h.act([other.get_observable_state() for other in people if other is not h]) for h in people]
            for h,a in zip(people,ha): h.step(a)
            world.env.global_time+=.25
            future.append(humans()[:,:2].copy())
        oracle=np.stack(future,axis=1)
        selections={}; forecasts={}
        for name,pred in [('current',cv),('history',history),('oracle',oracle)]:
            po.human_segment_start=np.concatenate([current[:,None,:2],pred[:,:-1]],axis=1)
            po.human_segment_end=pred
            clearance=planner._human_clearance(velocities,po)
            costs=planner._cost(velocities,po,positions,clearance,None,None)
            selections[name]=int(np.argmin(costs))
            forecasts[name]=dict(costs=costs.tolist(),min_clearance=clearance.min(axis=(1,2)).tolist(),
                endpoint_error=float(np.linalg.norm(pred[:,-1]-oracle[:,-1],axis=1).mean()))
        branches=[]
        for j in range(11):
            o=replay(t); ret=0.; minimum=100.; steps=0
            for k in range(140-t):
                a=executable[j,k].astype(np.float32) if j<10 and k<8 else model.predict(o,deterministic=True)[0]
                o,r,d,tr,inf=env.step(a); ret+=(.99**k)*r; steps+=1
                minimum=min(minimum,float(inf['actual_clearance']))
                if k<8: np.testing.assert_allclose(humans()[:,:2],oracle[:,k],atol=2e-6,rtol=0)
                if d or tr: break
            assert d or tr
            branches.append(dict(candidate=j,return_value=float(ret),outcome=inf['outcome'],steps=steps,min_clearance=minimum))
        assert branches[10]['outcome']==baseline
        best=int(np.argmax([b['return_value'] for b in branches]))
        records.append(dict(step=t,selections=selections,forecasts=forecasts,branches=branches,best_return_candidate=best,
                            controls=executable.tolist(),state_hash=hashes[t]))
    env.close()
    return dict(case=spec['case'],layout=spec['layout'],profile=spec['profile'],fresh=spec['fresh'],baseline=baseline,
                baseline_steps=length,replay_exact=True,human_future_verified=True,anchors=records)


def failure_evidence_audit(args):
    import multiprocessing
    root=args.results;root.mkdir(parents=True,exist_ok=False)
    source=Path('repair_results/dagger_ppo_nominal_20260915/initial.zip')
    original=Path('repair_results/student_dagger_coverage_20260915/independent_confirmation.json')
    nonstationary=Path('repair_results/incremental_information_disjoint_20260915')
    protocol=dict(purpose='Locate missing-information evidence, not train or design Bayes',
        discovery='all failures in original 500 nominal and 200 nonstationary audit episodes',
        anchors='collision: 8/16 steps before termination; timeout:64/112',
        candidates='3x3 actor-local dv +/- .2 and omega +/- .4, plus stop; projected by existing unicycle planner; execute 8 steps then frozen actor',
        selectors='same existing cost, margin .50: CV, four-frame acceleration clipped 1m/s2 and decayed 1s, exact human future',
        retrospective_upper_bound='best realized return in same finite bank plus feedback baseline; not a global upper bound',
        fresh_confirmation='50 nominal +50 train_nonstationary, layout=case=971000..971099; fixed step16 if reached',
        history_evidence_gate='fresh paired History-Current return improvement 95% episode bootstrap lower bound >0 and no excess collisions',
        limitation='Oracle future may include unpredictable scheduler draws; privileged value alone is not learnable information or Bayesian necessity',
        training_updates=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actors='nominal uses original round16; nonstationary uses original PPO-initial checkpoint; no cross-profile actor comparison',
        actor_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    specs=[]
    for r in json.loads(original.read_text())['records']:
        if r['outcome']!='success': specs.append(dict(case=r['test_case'],layout=r['layout_seed'],profile='nominal',fresh=False,expected=r['outcome'],trace=r['trace']))
    for i,r in enumerate(json.loads((nonstationary/'audit_episodes.json').read_text())):
        if r['outcome']!='success': specs.append(dict(case=130000+i,layout=84000000+i,profile='train_nonstationary',fresh=False,expected=r['outcome'],truth_file=str(nonstationary/f'audit_episode_{i:03d}.npz')))
    specs.extend(dict(case=971000+i,layout=971000+i,profile='nominal' if i<50 else 'train_nonstationary',fresh=True) for i in range(100))
    all_results=[]
    with multiprocessing.get_context('spawn').Pool(4) as pool:
        for result in pool.imap(failure_evidence_case,[(str(args.params),str(source),s) for s in specs]):
            all_results.append(result)
            (root/f"case_{result['case']}.json").write_text(json.dumps(result,indent=2))
            print('EVIDENCE',len(all_results),len(specs),result['case'],result['baseline'],flush=True)
    summary={}
    rng=np.random.default_rng(2407)
    for split in ('discovery','fresh'):
        cases=[r for r in all_results if r['fresh']==(split=='fresh')]
        out=dict(cases=len(cases),eligible=sum(bool(r['anchors']) for r in cases))
        for arm in ('current','history','oracle','best'):
            picked=[];deltas=[]
            for r in cases:
                for a in r['anchors']:
                    b=a['branches'][a['best_return_candidate'] if arm=='best' else a['selections'][arm]]
                    picked.append(b)
                    deltas.append(b['return_value']-a['branches'][10]['return_value'])
            out[arm]=dict(anchor_count=len(picked),success=sum(b['outcome']=='success' for b in picked),
                collision=sum(b['outcome']=='collision' for b in picked),timeout=sum(b['outcome']=='timeout' for b in picked),
                mean_return_gain_vs_actor=float(np.mean(deltas)))
        differences=[];excess=0
        for r in cases:
            ds=[]
            for a in r['anchors']:
                c=a['branches'][a['selections']['current']];h=a['branches'][a['selections']['history']]
                ds.append(h['return_value']-c['return_value'])
                excess+=int(h['outcome']=='collision')-int(c['outcome']=='collision')
            if ds:differences.append(np.mean(ds))
        boot=np.mean(rng.choice(differences,(10000,len(differences))),axis=1)
        ci=np.quantile(boot,[.025,.975]).tolist()
        out['history_vs_current']=dict(mean=float(np.mean(differences)),ci95=ci,excess_collision_anchors=excess,passed=bool(ci[0]>0 and excess<=0))
        summary[split]=out
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def failure_evidence_integrity(root, params):
    """Post-run provenance and physical-layout audit, without changing selectors."""
    from collections import Counter
    from crowd_nav.bayes_continuous.train_smoke import ActionHistory, dagger_artifact
    rows=[json.loads(p.read_text()) for p in sorted(root.glob('case_*.json'))]
    assert len(rows)==129
    old=json.loads(Path('repair_results/student_dagger_coverage_20260915/independent_confirmation.json').read_text())['records']
    previous={r['layout_sha256'] for r in old}
    for split in ('train','validation','audit'):
        records=json.loads((Path('repair_results/incremental_information_disjoint_20260915')/(split+'_episodes.json')).read_text())
        previous.update(r['layout_sha256'] for r in records)
    env=ActionHistory(BeliefEnv(params,arm='no_belief'),route=True)
    hashes=[]
    for r in rows:
        if not r['fresh']:continue
        env.reset(options=dict(layout_seed=r['layout'],test_case=r['case'],profile=r['profile']))
        w=env.unwrapped.world
        human=np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref] for h in w.env.humans],np.float64)
        joint=np.concatenate([human,np.asarray([[w.robot.px,w.robot.py,w.robot.gx,w.robot.gy,w.robot.radius,w.robot.theta]])])
        hh=hashlib.sha256(human.tobytes()).hexdigest();jh=hashlib.sha256(joint.tobytes()).hexdigest()
        assert hh not in previous and jh not in previous
        hashes.append(dict(case=r['case'],human_hash=hh,joint_hash=jh))
    assert len({x['human_hash'] for x in hashes})==100
    env.close()
    report=dict(total_cases=len(rows),anchors=sum(len(r['anchors']) for r in rows),
        branches=sum(len(a['branches']) for r in rows for a in r['anchors']),
        branch_environment_steps=sum(b['steps'] for r in rows for a in r['anchors'] for b in a['branches']),
        independent_fresh_layouts=100,overlap_with_previous_1200=0,layout_hashes=hashes,
        actor_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            dagger_artifact(Path('repair_results/student_dagger_coverage_20260915'),'round16.zip'),
            Path('repair_results/dagger_ppo_nominal_20260915/initial.zip')]},
        frozen_mainline={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path('crowd_nav/bayes_continuous/train_smoke.py'),Path('crowd_nav/bayes_continuous/network.py')]},
        ineligible=[dict(case=r['case'],outcome=r['baseline'],steps=r['baseline_steps']) for r in rows if not r['anchors']],
        strata={})
    for fresh in (False,True):
        for profile in ('nominal','train_nonstationary'):
            subset=[r for r in rows if r['fresh']==fresh and r['profile']==profile]
            entry=dict(cases=len(subset),baseline=dict(Counter(r['baseline'] for r in subset)),arms={})
            for arm in ('current','history','oracle','best'):
                outcomes=[];gains=[];rescued=[]
                for r in subset:
                    success=False
                    if not r['anchors']:outcomes.append(r['baseline']);continue
                    for a in r['anchors']:
                        b=a['branches'][a['best_return_candidate'] if arm=='best' else a['selections'][arm]]
                        outcomes.append(b['outcome']);gains.append(b['return_value']-a['branches'][10]['return_value'])
                        success|=b['outcome']=='success'
                    if success and r['baseline']!='success':rescued.append(r['case'])
                entry['arms'][arm]=dict(outcomes=dict(Counter(outcomes)),mean_return_gain=float(np.mean(gains)),rescued_cases=rescued)
            report['strata'][f'{fresh}_{profile}']=entry
    (root/'integrity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='layout_hashes'},indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--params', type=Path)
    parser.add_argument('--teacher-runs', type=Path, nargs='+')
    parser.add_argument('--dagger-confirm', action='store_true')
    parser.add_argument('--dagger-diagnose', action='store_true')
    parser.add_argument('--original-collection', type=Path)
    parser.add_argument('--information-audit', action='store_true')
    parser.add_argument('--information-ridge', action='store_true')
    parser.add_argument('--reciprocity-audit', action='store_true')
    parser.add_argument('--reciprocity-resume', action='store_true')
    parser.add_argument('--type-oracle', action='store_true')
    parser.add_argument('--failure-evidence', action='store_true')
    args = parser.parse_args()
    if args.information_ridge:
        information_ridge_audit(args.results)
        return
    if args.teacher_runs:
        audit_teachers(args.teacher_runs, args.results)
        return
    if args.params is None:
        parser.error('--params is required for checkpoint auditing')
    torch.set_num_threads(1)
    if args.failure_evidence:
        failure_evidence_audit(args)
        return
    if args.type_oracle:
        type_oracle_audit(args)
        return
    if args.reciprocity_audit or args.reciprocity_resume:
        reciprocity_audit(args)
        return
    if args.information_audit:
        information_audit(args)
        return
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
