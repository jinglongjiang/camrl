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
