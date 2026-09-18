"""Invisible-robot local predictive IL -> discrete policy-gradient training.

This is the existing training entry point, with the previous sequence backbone
removed. No imports of legacy trainers, recurrent policies, or simulator queries.
Run --self-test before collecting data. Smoke tests are not navigation evidence.
"""
from __future__ import annotations
import argparse
import configparser
import copy
import hashlib
import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from crowd_nav.bayesian import (SCHEMA, ARMS, COMPOSITIONS, RISK_COMPOSITIONS, MotionBelief as InteractionBelief,
                                LocalValueNetwork as BeliefQNetwork,
                                action_table, batch_observations)
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState, JointState


def configuration(path):
    config = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
    if not config.read(str(path)):
        raise FileNotFoundError(path)
    if config.get('action_space','kinematics') != 'holonomic':
        raise ValueError('This discrete protocol requires holonomic actions')
    if config.getboolean('robot','visible') or config.getboolean('interaction','enabled'):
        raise ValueError('Invisible robot, no reciprocity types: required by motion protocol')
    return config


def environment(config, humans=5, phase='train', scene='circle_crossing'):
    if phase in ('train','val') and humans != 5:
        raise ValueError('Training and model selection are strictly five-human')
    config = copy.deepcopy(config)
    config.set('sim','human_num',str(humans))
    config.set('sim','train_val_sim',scene)
    config.set('sim','test_sim',scene)
    env = CrowdSim()
    env.configure(config)
    robot = Robot(config,'robot')
    robot.policy.multiagent_training = True
    env.set_robot(robot)
    env.phase = phase
    return env


def expert(config, raw):
    """ORCA expert gets robot goal but only observable human states."""
    teacher = ORCA()
    teacher.configure(config)
    teacher.safety_space = config.getfloat('policy','teacher_safety_space',
                                           fallback=teacher.safety_space)
    r = raw[:9]
    state = FullState(r[0],r[1],r[4],r[5],r[6],r[2],r[3],r[7],r[8])
    humans = [ObservableState(*h) for h in raw[9:].reshape(-1,5)]
    action = teacher.predict(JointState(state,humans))
    return np.array([action.vx,action.vy])


def episode(env, model, config, case, arm, rng, epsilon=0., teacher=False,
            max_steps=None, step_callback=None,temperature=0.,record_raw=False):
    raw,_ = env.reset(options={'test_case':int(case)})
    belief = InteractionBelief(config)
    obs,rotation = belief.observe(raw,0,1.,arm)
    table = model.actions.detach().cpu().numpy()
    device = next(model.parameters()).device
    transitions = []
    reward_sum = 0.
    clearance = float('inf')
    outcome = 'incomplete'
    limit = int(np.ceil(env.time_limit/env.time_step))
    if max_steps is not None:
        limit = min(limit,max_steps)
    for step in range(limit):
        if teacher:
            target = rotation @ expert(config,raw)
            action_index = int(np.square(table-target).sum(1).argmin())
        elif rng.random() < epsilon:
            action_index = int(rng.integers(len(table)))
        else:
            with torch.no_grad():
                scores = model(batch_observations([obs],device))
                action_index = int(torch.distributions.Categorical(logits=scores/temperature).sample().item()
                                   if temperature > 0 else scores.argmax(1).item())
        velocity = rotation.T @ table[action_index]
        next_raw,reward,terminated,truncated,info = env.step(ActionXY(*velocity))
        clearance = min(clearance,float(info.get('dmin',float('inf'))))
        remaining = max(0.,1-env.global_time/env.time_limit)
        nxt,next_rotation = belief.observe(next_raw,step+1,remaining,arm)
        done = terminated or truncated
        row = dict(obs=obs,action=action_index,reward=reward,next_obs=nxt,done=done)
        if record_raw:
            row.update(raw=raw.copy(),next_raw=next_raw.copy())
        transitions.append(row)
        if step_callback is not None:
            step_callback(row)
        reward_sum += reward
        obs,rotation,raw = nxt,next_rotation,next_raw
        if done:
            outcome = info['event']
            break
    return transitions,dict(case=case,humans=len(env.humans),outcome=outcome,
        steps=len(transitions),reward=reward_sum,elapsed_time=len(transitions)*env.time_step,
        min_clearance=clearance if np.isfinite(clearance) else None)


def risk_examples(rows, actions, dt, horizon=2.):
    """Training labels only: constant robot action against recorded human future.

    Exact swept segments for this counterfactual, justified by invisible robot.
    This is a fixed-horizon event, not an episode return or deployed lookahead.
    """
    steps = int(round(horizon/dt))
    if steps < 1 or not np.isclose(steps*dt,horizon):
        raise ValueError('Risk horizon must contain an integer number of steps')
    examples = []
    for index in range(len(rows)-steps+1):
        raw = rows[index]['raw'];r=raw[:9];h=raw[9:].reshape(-1,5)
        future = np.stack([h[:,:2]]+[rows[t]['next_raw'][9:].reshape(-1,5)[:,:2]
                                    for t in range(index,index+steps)])
        angle = np.arctan2(r[3]-r[1],r[2]-r[0]);c,s=np.cos(angle),np.sin(angle)
        world = actions@np.array([[c,s],[-s,c]])
        times = np.arange(steps+1)*dt
        relative = future[None]-r[None,None,None,:2]-world[:,None,None,:]*times[None,:,None,None]
        segment = np.diff(relative,axis=1);start=relative[:,:-1]
        fraction = (-(start*segment).sum(-1)/np.maximum((segment**2).sum(-1),1e-12)).clip(0.,1.)
        clearance = np.linalg.norm(start+fraction[...,None]*segment,axis=-1)-r[6]-h[None,None,:,4]
        rel = h[None,:,:2]-r[None,None,:2];v=h[None,:,2:4]-world[:,None,:]
        t=(-(rel*v).sum(-1)/np.maximum((v*v).sum(-1),1e-12)).clip(0.,horizon)
        cv = np.linalg.norm(rel+t[...,None]*v,axis=-1)-r[6]-h[None,:,4]
        examples.append(dict(obs=rows[index]['obs'],risk_target=(clearance.min(1)<0).astype(np.float32),cv_target=(cv<0).astype(np.float32)))
    return examples


def risk_fit(model, train, validation, args):
    optimizer = torch.optim.Adam(model.local.parameters(),lr=args.il_lr)
    rng = np.random.default_rng(args.seed+101)
    for update in range(args.risk_updates):
        sample=[train[i] for i in rng.integers(len(train),size=args.batch_size)]
        predicted=model(batch_observations([r['obs'] for r in sample],args.device),return_risk=True)
        target=torch.as_tensor(np.stack([r['risk_target'] for r in sample]),device=args.device)
        loss=F.binary_cross_entropy(predicted.clamp(1e-6,1.-1e-6),target)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.local.parameters(),1.);optimizer.step()
        if (update+1)%500==0:
            print('RISK_LOSS',update+1,float(loss),flush=True)
    metrics=dict(count=0,brier=0.,nll=0.,cv_brier=0.,positives=0.,prediction_sum=0.)
    with torch.no_grad():
        for start in range(0,len(validation),args.batch_size):
            sample=validation[start:start+args.batch_size]
            p=model(batch_observations([r['obs'] for r in sample],args.device),return_risk=True).cpu().numpy()
            y=np.stack([r['risk_target'] for r in sample]);cv=np.stack([r['cv_target'] for r in sample]);clipped=p.clip(1e-6,1.-1e-6)
            metrics['count']+=y.size;metrics['brier']+=float(((p-y)**2).sum());metrics['cv_brier']+=float(((cv-y)**2).sum())
            metrics['nll']+=float((-y*np.log(clipped)-(1-y)*np.log1p(-clipped)).sum())
            metrics['positives']+=float(y.sum());metrics['prediction_sum']+=float(p.sum())
    for key in ('brier','cv_brier','nll','positives','prediction_sum'):
        metrics[key]/=max(1,metrics['count'])
    return metrics


def q_update(model,target,optimizer,rows,gamma):
    device = next(model.parameters()).device
    obs = batch_observations([r['obs'] for r in rows],device)
    nxt = batch_observations([r['next_obs'] for r in rows],device)
    actions = torch.tensor([r['action'] for r in rows],device=device)
    rewards = torch.tensor([r['reward'] for r in rows],device=device,dtype=torch.float32)
    done = torch.tensor([r['done'] for r in rows],device=device,dtype=torch.float32)
    with torch.no_grad():
        selected = model(nxt).argmax(1,keepdim=True)
        backup = rewards + gamma*(1-done)*target(nxt).gather(1,selected).squeeze(1)
    loss = F.smooth_l1_loss(model(obs).gather(1,actions[:,None]).squeeze(1),backup)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return float(loss.detach())


def il_update(model,optimizer,rows,margin=.2,objective='margin'):
    device = next(model.parameters()).device
    q = model(batch_observations([r['obs'] for r in rows],device))
    actions = torch.tensor([r['action'] for r in rows],device=device)
    selected = q.gather(1,actions[:,None]).squeeze(1)
    penalties = torch.full_like(q,margin).scatter_(1,actions[:,None],0.)
    if objective == 'softmax':
        imitation = F.cross_entropy(q/.1,actions)
    elif objective == 'margin':
        imitation = ((q+penalties).max(1).values-selected).mean()
    else:
        raise ValueError('Unknown IL objective')
    returns = torch.tensor([r['return'] for r in rows],device=device,dtype=torch.float32)
    loss = imitation + F.smooth_l1_loss(selected,returns)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return float(loss.detach())


def evaluate(model,config,arm,count,start,humans=5,phase='val',scene='circle_crossing'):
    env = environment(config,humans,phase,scene)
    rng = np.random.default_rng(0)
    state = np.random.get_state()
    try:
        records = [episode(env,model,config,start+i,arm,rng)[1] for i in range(count)]
    finally:
        np.random.set_state(state)
    return dict(humans=humans,scene=scene,episodes=count,
        success=sum(r['outcome']=='reach_goal' for r in records),
        collision=sum(r['outcome']=='collision' for r in records),
        timeout=sum(r['outcome']=='timeout' for r in records),
        mean_return=float(np.mean([r['reward'] for r in records])),
        mean_success_time=(float(np.mean([r['elapsed_time'] for r in records if r['outcome']=='reach_goal']))
                           if any(r['outcome']=='reach_goal' for r in records) else None),records=records)


def save_checkpoint(path,model,config,args,stage):
    payload = dict(schema=SCHEMA,stage=stage,arm=args.arm,seed=args.seed,train_num_humans=5,
        composition=model.composition,
        risk_supervised=getattr(model,'risk_supervised',False),
        rl_algorithm=args.rl_algorithm,
        state=model.state_dict(),config={s:dict(config[s]) for s in config.sections()},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__).resolve(),ROOT/'crowd_nav/bayesian.py',
                                 ROOT/'crowd_sim/envs/crowd_sim.py',ROOT/'crowd_sim/envs/policy/orca.py')})
    torch.save(payload,path)


def reinforce_update(model,optimizer,rows,temperature,kl_limit,batch_size):
    """Self-critical REINFORCE, with backtracking on measured old-policy KL.

    The baseline is the frozen IL policy's separate rollout of the same case.
    No demonstration loss, action-Q target, critic or PPO clipping is used.
    Scores are action preferences in this phase, not calibrated Q estimates.
    """
    device = next(model.parameters()).device
    chunks = [rows[i:i+batch_size] for i in range(0,len(rows),batch_size)]
    if not rows or not any(row['advantage'] != 0. for row in rows):
        return dict(accepted=False,parameter_changed=False,kl=0.,surrogate_gain=0.,
                    scale=0.,gradient_norm=0.,samples=len(rows))
    episodes = len({row.get('trajectory_id',0) for row in rows})
    old_logits = []
    with torch.no_grad():
        for chunk in chunks:
            logits = model(batch_observations([r['obs'] for r in chunk],device))/temperature
            old_logits.append(logits.log_softmax(-1).detach().cpu())
    optimizer.zero_grad()
    for chunk in chunks:
        logits = model(batch_observations([r['obs'] for r in chunk],device))/temperature
        logp = logits.log_softmax(-1)
        actions = torch.tensor([r['action'] for r in chunk],device=device)
        advantage = torch.tensor([r['advantage'] for r in chunk],device=device,dtype=torch.float32)
        loss = -(logp.gather(1,actions[:,None]).squeeze(1)*advantage).sum()/episodes
        loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    before = [p.detach().clone() for p in model.parameters()]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    proposed = [p.detach().clone() for p in model.parameters()]
    accepted,accepted_kl,accepted_gain,scale = False,0.,0.,0.
    for factor in (1.,.5,.25,.125,.0625,.03125,.015625):
        with torch.no_grad():
            for p,a,b in zip(model.parameters(),before,proposed):
                p.copy_(a+factor*(b-a))
            kl,gain = 0.,0.
            for chunk,old in zip(chunks,old_logits):
                old = old.to(device)
                new = (model(batch_observations([r['obs'] for r in chunk],device))/temperature).log_softmax(-1)
                actions = torch.tensor([r['action'] for r in chunk],device=device)
                advantage = torch.tensor([r['advantage'] for r in chunk],device=device,dtype=torch.float32)
                ratio = (new-old).gather(1,actions[:,None]).squeeze(1).exp()
                kl += float((old.exp()*(old-new)).sum())/len(rows)
                gain += float(((ratio-1)*advantage).sum())/episodes
            if np.isfinite(kl) and np.isfinite(gain) and kl <= kl_limit and gain >= -1e-8:
                accepted,accepted_kl,accepted_gain,scale = True,kl,gain,factor
                break
    if not accepted:
        with torch.no_grad():
            for p,a in zip(model.parameters(),before):
                p.copy_(a)
        optimizer.load_state_dict(optimizer_before)
    changed = any(not torch.equal(p,a) for p,a in zip(model.parameters(),before))
    return dict(accepted=accepted,parameter_changed=changed,kl=accepted_kl,surrogate_gain=accepted_gain,
                scale=scale,gradient_norm=norm,samples=len(rows))


def policy_gradient_training(args,config,model,env,rng,results,persist):
    reference = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    baseline = results['il_validation']
    rank = lambda v:(v['success'],-v['collision'],sum(r['reward'] for r in v['records']))
    best_rank = rank(baseline)
    save_checkpoint(args.out/'best.pt',model,config,args,'il')
    results.update(selected_rl_step=0,reference_environment_steps=0,policy_updates=[],rl_algorithm='self_critical_reinforce')
    budget = args.rl_steps if args.rl_steps is not None else float('inf')
    steps,index,updates,next_evaluation = 0,0,0,args.eval_steps
    pending = []
    stopped = False
    def returns(rows):
        values = []
        total = 0.
        for row in reversed(rows):
            total = row['reward']+args.gamma*total
            values.append(total)
        return values[::-1]
    while steps < budget and (args.rl_steps is not None or index < args.rl_episodes):
        case = 30000+index
        reference_rows,reference_record = episode(env,reference,config,case,args.arm,rng)
        rows,record = episode(env,model,config,case,args.arm,rng,temperature=args.temperature,
                              max_steps=budget-steps)
        results['reference_environment_steps'] += len(reference_rows)
        steps += len(rows)
        index += 1
        results['rl'].append(dict(reference=reference_record,**record))
        # Never pretend a budget-truncated trajectory has a zero terminal return.
        if record['outcome'] != 'incomplete':
            values,reference_values = returns(rows),returns(reference_rows)
            for t,row in enumerate(rows):
                control = reference_values[t] if t < len(reference_values) else 0.
                row['advantage'] = args.gamma**t*(values[t]-control)
                row['trajectory_id'] = case
            pending.extend(rows)
        last = steps == budget or (args.rl_steps is None and index == args.rl_episodes)
        if pending and (index%args.pg_batch_episodes == 0 or steps >= next_evaluation or last):
            result = reinforce_update(model,optimizer,pending,args.temperature,args.kl_limit,args.batch_size)
            result.update(step=steps,episode=index)
            results['policy_updates'].append(result)
            updates += int(result['accepted'] and result['parameter_changed'])
            print('PG_UPDATE',steps,result,flush=True)
            pending = []
        if steps >= next_evaluation or last:
            validation = evaluate(model,config,args.arm,args.eval_episodes,args.validation_start)
            results['validation'].append(dict(episode=index,step=steps,**validation))
            save_checkpoint(args.out/'last.pt',model,config,args,'rl')
            if rank(validation) > best_rank:
                best_rank = rank(validation)
                results['selected_rl_step'] = steps
                save_checkpoint(args.out/'best.pt',model,config,args,'rl')
            results['rl_environment_steps'] = steps
            results['rl_optimizer_updates'] = updates
            persist()
            print('RL_VALIDATION',steps,validation['success'],validation['collision'],flush=True)
            next_evaluation = (steps//args.eval_steps+1)*args.eval_steps
            if args.rollback_below > 0 and validation['success']/args.eval_episodes < args.rollback_below:
                stopped = True
                break
    best = torch.load(args.out/'best.pt',map_location=args.device)
    model.load_state_dict(best['state'])
    save_checkpoint(args.out/'final.pt',model,config,args,best['stage'])
    results['status'] = 'RL_ROLLED_BACK' if stopped else 'COMPLETED'
    results['rl_environment_steps'] = steps
    results['rl_optimizer_updates'] = updates
    persist()


def training(args,config):
    if args.out is None:
        raise ValueError('--out is required for train')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty run directory')
    args.out.mkdir(parents=True,exist_ok=True)
    model = BeliefQNetwork(action_table(config),composition=args.composition).to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location=args.device)
        model.risk_supervised=checkpoint.get('risk_supervised',False)
        arm_transfer = (args.initialize_arm and checkpoint.get('risk_supervised',False)
                        and checkpoint['stage']=='risk_pretrain' and args.composition in RISK_COMPOSITIONS)
        if checkpoint['schema'] != SCHEMA or (checkpoint['arm'] != args.arm and not arm_transfer):
            raise ValueError('Initialization checkpoint schema/arm mismatch')
        saved = configparser.ConfigParser()
        saved.read_dict(checkpoint['config'])
        if {s:dict(config[s]) for s in config.sections()} != checkpoint['config']:
            raise ValueError('Training initialization requires identical saved configuration')
        saved_composition = checkpoint.get('composition','mixture')
        if saved_composition != args.composition:
            risk_transfer = saved_composition in RISK_COMPOSITIONS and args.composition in RISK_COMPOSITIONS
            if not args.initialize_composition or (saved_composition != 'mixture' and not risk_transfer):
                raise ValueError('Composition change requires explicit mixture initialization')
            state = dict(checkpoint['state'])
            if args.composition == 'conflict':
                state['conflict_gain'] = torch.zeros((),device=args.device)
            if args.composition in ('moments','ordered'):
                state.update({k:v for k,v in model.state_dict().items() if k.startswith('combination.')})
            if args.composition == 'attention':
                state.update({k:v for k,v in model.state_dict().items() if k.startswith('attention.')})
            if args.composition in RISK_COMPOSITIONS and not risk_transfer:
                state['risk_gain'] = model.risk_gain.detach().clone()
            model.load_state_dict(state,strict=True)
        else:
            model.load_state_dict(checkpoint['state'],strict=True)
    if args.composition in RISK_COMPOSITIONS and args.risk_updates == 0 and not getattr(model,'risk_supervised',False):
        raise ValueError('Risk composition requires event supervision or a supervised risk checkpoint')
    if args.freeze_local_il and args.composition not in RISK_COMPOSITIONS:
        if not hasattr(model,'combination') or not args.checkpoint:
            raise ValueError('Frozen-local IL requires a combination head and an IL checkpoint')
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.combination.parameters():
            p.requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.il_lr)
    env = environment(config)
    rng = np.random.default_rng(args.seed)
    replay = deque(maxlen=args.buffer_size)
    demos,collection,risk_data = [],[],[]
    demo_count = 0 if args.checkpoint and args.il_updates == 0 and args.rl_algorithm == 'reinforce' else args.demo_episodes
    for i in range(demo_count):
        rows,record = episode(env,model,config,10000+i,args.arm,rng,teacher=True,record_raw=args.risk_updates>0)
        if args.risk_updates > 0:
            risk_data.extend(risk_examples(rows,model.actions.cpu().numpy(),env.time_step))
        collection.append(record)
        replay.extend(rows)
        if record['outcome']=='reach_goal':
            ret = 0.
            for row in reversed(rows):
                ret = row['reward']+args.gamma*ret
                row['return'] = ret
            demos.extend(rows)
        if (i+1)%50 == 0:
            print('DEMO',i+1,'SUCCESS',sum(r['outcome']=='reach_goal' for r in collection),flush=True)
    if not demos and not (args.checkpoint and args.il_updates == 0):
        raise RuntimeError('No successful expert episodes; IL cannot start')
    results = dict(collection=collection,il_validation=None,il_curve=[],rl=[],validation=[],
        schema=SCHEMA,seed=args.seed,arm=args.arm,smoke_only=args.il_gate < .9,
        arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        il_optimizer_updates=0,rl_optimizer_updates=0,rl_environment_steps=0,
        teacher_success=sum(r['outcome']=='reach_goal' for r in collection),
        demo_transitions=len(demos))
    if args.risk_updates > 0:
        validation=[]
        for case in range(25000,25020):
            rows,_=episode(env,model,config,case,args.arm,rng,teacher=True,record_raw=True)
            validation.extend(risk_examples(rows,model.actions.cpu().numpy(),env.time_step))
        if not risk_data or not validation:
            raise RuntimeError('Missing fixed-horizon risk supervision')
        results['risk_validation']=risk_fit(model,risk_data,validation,args)
        model.risk_supervised=True
        save_checkpoint(args.out/'risk.pt',model,config,args,'risk_pretrain')
        results['risk_training_states']=len(risk_data)
        print('RISK_VALIDATION',results['risk_validation'],flush=True)
    if args.composition in RISK_COMPOSITIONS:
        # Preserve the supervised event semantics during both IL and RL.
        for p in model.local.parameters():
            p.requires_grad_(False)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.il_lr)
    hard = []
    if args.il_hard_fraction > 0:
        if not args.checkpoint:
            raise ValueError('Hard-example sampling requires a fixed initial IL model')
        with torch.no_grad():
            for start in range(0,len(demos),args.batch_size):
                chunk = demos[start:start+args.batch_size]
                prediction = model(batch_observations([row['obs'] for row in chunk],args.device)).argmax(1).cpu().numpy()
                hard.extend(start+i for i,(p,row) in enumerate(zip(prediction,chunk)) if p != row['action'])
    results['hard_demo_count'] = len(hard)
    results['hard_demo_indices_sha256'] = hashlib.sha256(np.asarray(hard,dtype=np.int64).tobytes()).hexdigest()
    def persist():
        temporary = args.out/'results.tmp'
        temporary.write_text(json.dumps(results,indent=2))
        temporary.replace(args.out/'results.json')
    def rank(value):
        return (value['success'],-value['collision'],
                sum(row['reward'] for row in value['records']))
    best_rank = None
    # All checkpoint selection uses the same five-human development layouts.
    for update in range(args.il_updates+1):
        if update:
            count = int(args.batch_size*args.il_hard_fraction) if hard else 0
            indices = list(rng.integers(len(demos),size=args.batch_size-count))
            if count:
                indices += [hard[i] for i in rng.integers(len(hard),size=count)]
            batch = [demos[i] for i in indices]
            loss = il_update(model,optimizer,batch,objective=args.il_objective)
            results['il_optimizer_updates'] = update
            if update%100 == 0:
                print('IL_LOSS',update,loss,flush=True)
        if (update and update%args.il_eval_every == 0) or update == args.il_updates or (args.checkpoint and update == 0):
            accuracy = None
            if demos:
                sample = [demos[i] for i in rng.integers(len(demos),size=args.batch_size)]
                with torch.no_grad():
                    prediction = model(batch_observations([row['obs'] for row in sample],args.device)).argmax(1).cpu().numpy()
                accuracy = float(np.mean(prediction == [row['action'] for row in sample]))
            validation = evaluate(model,config,args.arm,args.eval_episodes,args.validation_start)
            results['il_curve'].append(dict(update=update,training_action_accuracy=accuracy,**validation))
            eligible = not (args.select_trained_il and update == 0)
            if eligible and (best_rank is None or rank(validation) > best_rank):
                best_rank = rank(validation)
                results['il_validation'] = validation
                results['selected_il_update'] = update
                save_checkpoint(args.out/'il.pt',model,config,args,'il')
            persist()
            print('IL_VALIDATION',update,validation['success'],validation['collision'],'ACCURACY',accuracy,flush=True)
    baseline = results['il_validation']
    model.load_state_dict(torch.load(args.out/'il.pt',map_location=args.device)['state'])
    for p in model.parameters():
        p.requires_grad_(True)
    if args.composition in RISK_COMPOSITIONS:
        for p in model.local.parameters():
            p.requires_grad_(False)
    if baseline['success']/args.eval_episodes < args.il_gate:
        results['status'] = 'IL_GATE_FAILED'
        persist()
        print('IL_GATE_FAILED: no RL',flush=True)
        return
    if args.rl_steps == 0 or (args.rl_steps is None and args.rl_episodes == 0):
        results['status'] = 'IL_QUALIFIED_NO_RL_REQUESTED'
        save_checkpoint(args.out/'final.pt',model,config,args,'il')
        persist()
        return
    if args.rl_algorithm == 'reinforce':
        policy_gradient_training(args,config,model,env,rng,results,persist)
        return
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    best_rank = rank(baseline)
    save_checkpoint(args.out/'best.pt',model,config,args,'il')
    results['selected_rl_step'] = 0
    steps,updates = 0,0
    def online_update(row):
        nonlocal steps,updates
        replay.append(row)
        steps += 1
        if len(replay) >= args.batch_size:
            batch = random.sample(list(replay),args.batch_size)
            q_update(model,target,optimizer,batch,args.gamma)
            updates += 1
        if steps%args.target_steps == 0:
            target.load_state_dict(model.state_dict())
    index = 0
    stopped = False
    while (steps < args.rl_steps if args.rl_steps is not None else index < args.rl_episodes):
        progress = steps/max(args.rl_steps,1) if args.rl_steps is not None else index/max(args.rl_episodes,1)
        epsilon = args.epsilon_final+(args.epsilon_initial-args.epsilon_final)*max(0.,1-progress)
        # Cut collection exactly at the environment-step budget and evaluation boundaries.
        remaining = None
        if args.rl_steps is not None:
            remaining = min(args.rl_steps-steps,args.eval_steps-steps%args.eval_steps)
        _,record = episode(env,model,config,30000+index,args.arm,rng,epsilon=epsilon,
                           max_steps=remaining,step_callback=online_update)
        results['rl'].append(record)
        index += 1
        should_eval = ((steps%args.eval_steps == 0 or steps == args.rl_steps)
                       if args.rl_steps is not None
                       else (index%args.eval_every == 0 or index == args.rl_episodes))
        if should_eval:
            validation = evaluate(model,config,args.arm,args.eval_episodes,args.validation_start)
            results['validation'].append(dict(episode=index,step=steps,**validation))
            save_checkpoint(args.out/'last.pt',model,config,args,'rl')
            if rank(validation) > best_rank:
                best_rank = rank(validation)
                results['selected_rl_step'] = steps
                save_checkpoint(args.out/'best.pt',model,config,args,'rl')
            results['rl_environment_steps'] = steps
            results['rl_optimizer_updates'] = updates
            persist()
            print('RL_VALIDATION',steps,validation['success'],validation['collision'],flush=True)
            if args.rollback_below > 0 and validation['success']/args.eval_episodes < args.rollback_below:
                stopped = True
                break
    best = torch.load(args.out/'best.pt',map_location=args.device)
    model.load_state_dict(best['state'])
    save_checkpoint(args.out/'final.pt',model,config,args,best['stage'])
    results['status'] = 'RL_ROLLED_BACK' if stopped else 'COMPLETED'
    results['rl_environment_steps'] = steps
    results['rl_optimizer_updates'] = updates
    persist()


def smoke(config,device):
    model = BeliefQNetwork(action_table(config)).to(device)
    observations = []
    for n in (0,1,5,10,12,20):
        raw = np.array([0,0,0,4,0,0,.3,1,0]+[2,2,.1,.2,.3]*n,dtype=float)
        belief = InteractionBelief(config)
        obs,_ = belief.observe(raw,0,1.)
        duplicate,_ = belief.observe(raw,0,1.)
        for key in obs:
            np.testing.assert_array_equal(obs[key],duplicate[key])
        observations.append(obs)
    batch = batch_observations(observations,device)
    with torch.no_grad():
        q = model(batch)
        assert q.shape == (6,80) and torch.isfinite(q).all()
        permuted = {k:v.clone() for k,v in batch.items()}
        for key in ('humans','modes','conditional','mask'):
            permuted[key] = permuted[key].flip(1)
        torch.testing.assert_close(q,model(permuted),rtol=1e-5,atol=1e-6)
        for i,obs in enumerate(observations):
            torch.testing.assert_close(q[i],model(batch_observations([obs],device))[0],rtol=1e-5,atol=1e-6)
        for composition in COMPOSITIONS:
            variant = BeliefQNetwork(action_table(config),composition=composition).to(device)
            state = dict(model.state_dict())
            if composition == 'conflict':
                state['conflict_gain'] = torch.zeros((),device=device)
            if composition in ('moments','ordered'):
                state.update({k:v for k,v in variant.state_dict().items() if k.startswith('combination.')})
            if composition == 'attention':
                state.update({k:v for k,v in variant.state_dict().items() if k.startswith('attention.')})
            if composition in RISK_COMPOSITIONS:
                state['risk_gain']=variant.risk_gain.detach().clone()
            variant.load_state_dict(state,strict=True)
            values = variant(batch)
            assert torch.isfinite(values).all()
            torch.testing.assert_close(values,variant(permuted),rtol=1e-5,atol=1e-6)
            for i,obs in enumerate(observations):
                torch.testing.assert_close(values[i],variant(batch_observations([obs],device))[0],rtol=1e-5,atol=1e-6)
            if composition == 'conflict':
                torch.testing.assert_close(q,values,rtol=0.,atol=0.)
                variant.conflict_gain.fill_(.5)
                changed = variant(batch)
                torch.testing.assert_close(changed,variant(permuted),rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(changed[:2],q[:2],rtol=0.,atol=0.)
            if composition in ('moments','ordered'):
                torch.testing.assert_close(q,values,rtol=0.,atol=0.)
                variant.combination[-1].weight.fill_(.01)
                changed = variant(batch)
                torch.testing.assert_close(changed,variant(permuted),rtol=1e-5,atol=1e-6)
                for i,obs in enumerate(observations):
                    torch.testing.assert_close(changed[i],variant(batch_observations([obs],device))[0],rtol=1e-5,atol=1e-6)
        costs = torch.tensor([[[2.,3.]]],device=device)
        local = torch.rand(7,80,20,device=device)
        offsets = torch.rand(7,1,20,device=device)*5.
        weights = torch.tensor([.4,.6],device=device)
        ordinary = model.regret_cost(local,weights)
        shifted = model.regret_cost(local+offsets,weights)
        torch.testing.assert_close(ordinary-ordinary[:,:1],shifted-shifted[:,:1],rtol=1e-5,atol=1e-5)
        assert torch.equal(ordinary.argmin(1),shifted.argmin(1))
        torch.testing.assert_close(model.regret_cost(local[:,:,:1],weights),local[:,:,0])
        times = torch.zeros_like(costs)
        opposite = torch.tensor([[[[1.,0.],[-1.,0.]]]],device=device)
        torch.testing.assert_close(model.conflict_cost(costs,times,opposite),costs[...,:1].squeeze(-1))
        assert model.conflict_cost(costs,times,opposite.abs()).item() == 0.
        times[...,1] = 2.
        assert model.conflict_cost(costs,times,opposite).item() < 2.
    env = environment(config)
    rows,record = episode(env,model,config,17,'full',np.random.default_rng(17),teacher=True,max_steps=8)
    for row in rows:
        row['return'] = row['reward']
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    before = [p.detach().clone() for p in model.parameters()]
    il_loss = il_update(model,optimizer,rows)
    assert any(not torch.equal(p,q) for p,q in zip(before,model.parameters()))
    target = copy.deepcopy(model)
    frozen = [p.detach().clone() for p in target.parameters()]
    td_loss = q_update(model,target,optimizer,rows,.99)
    assert all(torch.equal(p,q) for p,q in zip(frozen,target.parameters()))
    terminal = [dict(row,done=True) for row in rows]
    with torch.no_grad():
        scores = model(batch_observations([row['obs'] for row in terminal],device))
        selected = scores.gather(1,torch.tensor([row['action'] for row in terminal],device=device)[:,None]).squeeze(1)
        expected = F.smooth_l1_loss(selected,torch.tensor([row['reward'] for row in terminal],device=device)).item()
    actual = q_update(model,target,torch.optim.SGD(model.parameters(),lr=0.),terminal,.99)
    assert abs(actual-expected) < 1e-6
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(),buffer)
    buffer.seek(0)
    restored = BeliefQNetwork(action_table(config)).to(device)
    restored.load_state_dict(torch.load(buffer,map_location=device),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(batch),restored(batch),rtol=0,atol=0)
    # With an invisible robot, human paths must be unchanged by robot actions.
    human_paths = []
    for action in (ActionXY(0.,0.),ActionXY(.2,0.)):
        env.reset(options={'test_case':170})
        path = []
        for _ in range(4):
            env.step(action)
            path.append([h.get_obs_array() for h in env.humans])
        human_paths.append(path)
    np.testing.assert_allclose(human_paths[0],human_paths[1],atol=0,rtol=0)
    # Analytic one-observation conjugate update, and no evidence on duplicate frames.
    raw = np.array([0,0,0,4,0,0,.3,1,0, 2,2,0,0,.3],dtype=float)
    belief = InteractionBelief(config)
    belief.observe(raw,0,1.)
    raw[11] = .2
    posterior,_ = belief.observe(raw,1,1.)
    _,k,mu,alpha,beta = belief.tracks[0]
    assert k == 2. and alpha == 2.5
    np.testing.assert_allclose(mu,[.1,0.])
    np.testing.assert_allclose(beta,[belief.scale**2/2.+.01,belief.scale**2/2.])
    belief.observe(raw,1,1.)
    assert belief.tracks[0][1] == 2.
    predictive = posterior['modes'][0]*2.
    expected_variance = beta*3./(2.*1.5)
    # Goal points north, so rotation exchanges x/y variances.
    np.testing.assert_allclose(predictive.var(axis=0),expected_variance[::-1],rtol=1e-5)
    fresh = InteractionBelief(config)
    initial_full,_ = fresh.observe(raw,0,1.,'full')
    initial_prior,_ = InteractionBelief(config).observe(raw,0,1.,'prior')
    for key in initial_full:
        np.testing.assert_array_equal(initial_full[key],initial_prior[key])
    window = InteractionBelief(config)
    window.observe(raw,0,1.,'history')
    for frame in range(1,7):
        raw[11] += frame*.01
        history,_ = window.observe(raw,frame,1.,'history')
    assert len(window.recent[0]) == 4
    np.testing.assert_allclose(np.asarray(window.recent[0])[:,0],[.03,.04,.05,.06])
    assert np.isfinite(model(batch_observations([history],device)).detach().cpu().numpy()).all()
    try:
        belief.observe(raw,3,1.)
        raise AssertionError('Skipped frame accepted')
    except ValueError:
        pass
    # Far-away people have zero local support, so they do not dilute near costs.
    one = copy.deepcopy(observations[1])
    two = copy.deepcopy(one)
    for key in ('humans','modes','conditional','mask'):
        two[key] = np.concatenate((one[key],one[key]),axis=0)
    two['humans'][1,:2] = 100.
    with torch.no_grad():
        torch.testing.assert_close(model(batch_observations([one],device)),
                                   model(batch_observations([two],device)),rtol=1e-5,atol=1e-6)
    # Physical collision check uses the velocity actually executed this step.
    import types
    env.reset(options={'test_case':99})
    for i,h in enumerate(env.humans):
        h.set(100+i*3,100,100+i*3,100,0,0,0)
        h.act = types.MethodType(lambda self,ob:ActionXY(0.,0.),h)
    env.time_limit = 1.
    for i in range(4):
        _,_,terminated,truncated,info = env.step(ActionXY(0.,0.))
        assert not terminated and truncated == (i==3)
    assert info['event'] == 'timeout'
    env.reset(options={'test_case':99})
    for i,h in enumerate(env.humans):
        h.set(100+i*3,100,100+i*3,100,0,0,0)
        h.act = types.MethodType(lambda self,ob:ActionXY(0.,0.),h)
    h = env.humans[0]
    h.set(env.robot.px+1.,env.robot.py,10,10,0,0,0)
    h.act = types.MethodType(lambda self,ob:ActionXY(-4.,0.),h)
    _,_,terminated,_,info = env.step(ActionXY(0.,0.))
    assert terminated and info['event'] == 'collision'
    print(json.dumps(dict(schema=SCHEMA,checks='cardinality/permutation/padding, NIG analytic and quadrature, duplicate/skipped frames, distant-human invariance, IL/TD gradients, target frozen, invisible robot, swept collision, exact timeout',il_loss=il_loss,td_loss=td_loss,rollout=record)))


def study(args,config):
    """Bounded, sequential runs; seal all models before opening the test matrix."""
    import gc
    import shutil
    import time
    import tarfile
    if args.out is None or (args.out.exists() and any(args.out.iterdir())):
        raise ValueError('Study requires a new --out directory')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        initial = torch.load(args.checkpoint,map_location='cpu')
        if (not args.initialize_composition or args.study_arms != [initial['arm']]
                or args.study_seeds != [initial['seed']] or initial['stage'] != 'il'):
            raise ValueError('Checkpoint study requires one matching arm/seed and an IL checkpoint')
    start = time.monotonic()
    source_files = [Path(__file__).resolve(),ROOT/'crowd_nav/bayesian.py',
                    ROOT/'crowd_sim/envs/crowd_sim.py',ROOT/'crowd_sim/envs/policy/orca.py',args.config.resolve()]
    with tarfile.open(args.out/'source_snapshot.tgz','w:gz') as archive:
        for path in source_files:
            archive.add(path,arcname=str(path.relative_to(ROOT)))
    report = dict(schema=SCHEMA,status='TRAINING',arms=args.study_arms,seeds=args.study_seeds,
        compositions=args.study_compositions,
        initialization_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        train_humans=5,train_scene='circle_crossing',development_cases=[20000,20000+args.eval_episodes-1],
        test_cases=[args.case_start,args.case_start+args.study_eval_episodes-1],
        config={s:dict(config[s]) for s in config.sections()},runs=[],evaluations=[],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    def persist():
        report['elapsed_seconds'] = time.monotonic()-start
        temporary = args.out/'study.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False))
        temporary.replace(args.out/'study.json')
    persist()
    for seed in args.study_seeds:
        for arm,composition in [(a,c) for a in args.study_arms for c in args.study_compositions]:
            if shutil.disk_usage(args.out).free < 128*1024**2:
                raise RuntimeError('Study stopped before disk exhaustion')
            run = copy.copy(args)
            run.seed,run.arm,run.mode = seed,arm,'train'
            run.composition = composition
            suffix = '' if args.study_compositions == ['mixture'] else '_'+composition
            run.out = args.out/f'{arm}_{seed}{suffix}'
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print('STUDY_TRAIN',arm,seed,composition,flush=True)
            training(run,config)
            result = json.loads((run.out/'results.json').read_text())
            checkpoint = run.out/'final.pt'
            if not checkpoint.exists():
                checkpoint = run.out/'il.pt'
            report['runs'].append(dict(arm=arm,seed=seed,status=result['status'],
                composition=composition,
                il_success=result['il_validation']['success'],
                il_collision=result['il_validation']['collision'],
                selected_il_update=result['selected_il_update'],
                rl_steps=result.get('rl_environment_steps',0),
                reference_steps=result.get('reference_environment_steps',0),
                rl_updates=result.get('rl_optimizer_updates',0),
                selected_rl_step=result.get('selected_rl_step',0),checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            persist()
            gc.collect()
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
    report['status'] = 'FROZEN_EVALUATION'
    persist()
    for run in report['runs']:
        paths = [('selected',Path(run['checkpoint']))]
        if (run['arm']=='full' or args.checkpoint) and run['selected_rl_step'] > 0:
            paths.append(('il',Path(run['checkpoint']).parent/'il.pt'))
        for label,path in paths:
            payload = torch.load(path,map_location=args.device)
            model = BeliefQNetwork(action_table(config),composition=payload.get('composition','mixture')).to(args.device)
            model.load_state_dict(payload['state'],strict=True)
            model.eval()
            for scene in ('circle_crossing','square_crossing'):
                for n in (5,10,12,20):
                    result = evaluate(model,config,run['arm'],args.study_eval_episodes,args.case_start,n,'test',scene)
                    result.update(arm=run['arm'],seed=run['seed'],selection=label,
                                  composition=run['composition'],
                                  stage=payload['stage'],training_status=run['status'])
                    report['evaluations'].append(result)
                    persist()
                    print('STUDY_TEST',run['arm'],run['seed'],label,scene,n,result['success'],result['collision'],flush=True)
            del model
    report['status'] = 'COMPLETED'
    persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'crowd_nav/configs/env.config')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--mode',choices=('smoke','train','evaluate','study'),default='smoke')
    parser.add_argument('--arm',choices=ARMS,default='full')
    parser.add_argument('--composition',choices=COMPOSITIONS,default='mixture')
    parser.add_argument('--initialize-composition',action='store_true',
                        help='Explicitly initialize a new composition from a mixture IL checkpoint')
    parser.add_argument('--initialize-arm',action='store_true',
                        help='Explicitly share a supervised, frozen risk predictor across observation arms')
    parser.add_argument('--seed',type=int,default=2407)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--humans',type=int,default=5)
    parser.add_argument('--scene',choices=('circle_crossing','square_crossing'),default='circle_crossing')
    parser.add_argument('--case-start',type=int,default=100000)
    parser.add_argument('--validation-start',type=int,default=20000)
    parser.add_argument('--matrix',action='store_true',help='Frozen evaluation: 5/10/12/20, circle and square')
    parser.add_argument('--demo-episodes',type=int,default=200)
    parser.add_argument('--il-updates',type=int,default=3000)
    parser.add_argument('--il-eval-every',type=int,default=1000)
    parser.add_argument('--il-lr',type=float,default=1e-4)
    parser.add_argument('--freeze-local-il',action='store_true')
    parser.add_argument('--il-hard-fraction',type=float,default=0.)
    parser.add_argument('--risk-updates',type=int,default=0)
    parser.add_argument('--select-trained-il',action='store_true',
                        help='Select among trained IL checkpoints; initial policy is a separately reported reference')
    parser.add_argument('--il-objective',choices=('margin','softmax'),default='softmax')
    parser.add_argument('--rl-episodes',type=int,default=2000)
    parser.add_argument('--rl-steps',type=int)
    parser.add_argument('--rl-algorithm',choices=('ddqn','reinforce'),default='reinforce')
    parser.add_argument('--temperature',type=float,default=.05)
    parser.add_argument('--kl-limit',type=float,default=.002)
    parser.add_argument('--pg-batch-episodes',type=int,default=4)
    parser.add_argument('--eval-steps',type=int,default=1000)
    parser.add_argument('--rollback-below',type=float,default=.9)
    parser.add_argument('--epsilon-initial',type=float,default=.05)
    parser.add_argument('--epsilon-final',type=float,default=.01)
    parser.add_argument('--eval-episodes',type=int,default=100)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--il-gate',type=float,default=.9)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--buffer-size',type=int,default=50000)
    parser.add_argument('--target-steps',type=int,default=1000)
    parser.add_argument('--gpu-memory-fraction',type=float,default=.15)
    parser.add_argument('--study-arms',nargs='+',choices=ARMS,default=['full','prior','map','history'])
    parser.add_argument('--study-compositions',nargs='+',choices=COMPOSITIONS,default=['mixture'])
    parser.add_argument('--study-seeds',nargs='+',type=int,default=[2407,4807,7207])
    parser.add_argument('--study-eval-episodes',type=int,default=50)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    config = configuration(args.config)
    if min(args.batch_size,args.target_steps,args.eval_episodes,args.eval_every,args.il_eval_every,args.eval_steps) < 1:
        parser.error('Batch, target and evaluation intervals must be positive')
    if not 0 <= args.il_gate <= 1 or not 0 <= args.gamma <= 1:
        parser.error('Invalid gate/gamma')
    if not 0 <= args.il_hard_fraction <= 1:
        parser.error('Invalid hard-example fraction')
    if args.select_trained_il and args.il_updates < 1:
        parser.error('Selecting trained IL requires positive IL updates')
    if args.risk_updates < 0 or (args.risk_updates > 0 and args.composition not in RISK_COMPOSITIONS):
        parser.error('Risk supervision requires a risk composition and nonnegative update count')
    if min(args.demo_episodes,args.il_updates,args.rl_episodes) < 0 or (args.rl_steps is not None and args.rl_steps < 0):
        parser.error('Negative training budget')
    if not 0 <= args.rollback_below <= 1 or not 0 <= args.epsilon_final <= args.epsilon_initial <= 1:
        parser.error('Invalid rollback/exploration probability')
    if min(args.temperature,args.kl_limit,args.pg_batch_episodes) <= 0:
        parser.error('Invalid policy-gradient configuration')
    if args.self_test or args.mode == 'smoke':
        smoke(config,args.device)
    elif args.mode in ('train','study'):
        if args.humans != 5:
            parser.error('Training requires --humans 5')
        if args.scene != 'circle_crossing':
            parser.error('Training protocol is five-human circle only')
        if args.mode == 'study':
            if args.rl_steps is None or args.study_eval_episodes < 1:
                parser.error('Study requires an explicit RL-step budget and positive test count')
            if len(set(args.study_arms)) != len(args.study_arms) or len(set(args.study_seeds)) != len(args.study_seeds):
                parser.error('Duplicate study arms/seeds')
            study(args,config)
        else:
            training(args,config)
    else:
        if args.checkpoint is None:
            parser.error('--checkpoint required')
        checkpoint = torch.load(args.checkpoint,map_location=args.device)
        if checkpoint.get('schema') != SCHEMA or checkpoint['arm'] != args.arm:
            raise ValueError('Checkpoint schema/arm mismatch')
        saved = configparser.ConfigParser()
        saved.read_dict(checkpoint['config'])
        model = BeliefQNetwork(action_table(saved),composition=checkpoint.get('composition','mixture')).to(args.device)
        model.load_state_dict(checkpoint['state'],strict=True)
        if args.matrix:
            records = []
            for scene in ('circle_crossing','square_crossing'):
                for n in (5,10,12,20):
                    result = evaluate(model,saved,args.arm,args.eval_episodes,args.case_start,n,'test',scene)
                    records.append(result)
                    print('MATRIX',args.arm,scene,n,result['success'],result['collision'],file=sys.stderr,flush=True)
            result = dict(arm=args.arm,seed=checkpoint['seed'],stage=checkpoint['stage'],
                schema=SCHEMA,checkpoint=str(args.checkpoint),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),results=records)
        else:
            result = evaluate(model,saved,args.arm,args.eval_episodes,args.case_start,args.humans,'test',args.scene)
        print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
