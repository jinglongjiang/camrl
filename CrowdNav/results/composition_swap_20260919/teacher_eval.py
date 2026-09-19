"""ORCA teacher (projected onto the 80-action grid) as the ceiling for this action space."""
import os, sys, json, configparser, io, contextlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
STAGE='/tmp/claude-1000/-home-abc/f8a2c1a1-47f8-41fe-adc9-f2b70b538149/scratchpad/stage/CrowdNav'
sys.path.insert(0, STAGE)
import numpy as np, torch

def job(task):
    humans, scene = task
    torch.set_num_threads(1); os.chdir(STAGE)
    from crowd_nav.bayesian import LocalValueNetwork as Net, action_table
    from crowd_nav import train as T
    ck = torch.load('/home/abc/temp/compose_runs_20260919/sum_2407/il.pt', map_location='cpu', weights_only=False)
    cfg = configparser.ConfigParser(); cfg.read_dict(ck['config'])
    model = Net(action_table(cfg), composition='mixture')
    env = T.environment(cfg, humans, 'test', scene)
    rng = np.random.default_rng(0)
    rec=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(50):
            rec.append(T.episode(env, model, cfg, 9900000+i, 'prior', rng, teacher=True)[1])
    return dict(humans=humans, scene=scene, episodes=len(rec),
                success=sum(r['outcome']=='reach_goal' for r in rec),
                collision=sum(r['outcome']=='collision' for r in rec),
                timeout=sum(r['outcome']=='timeout' for r in rec),
                outcomes=[r['outcome'] for r in rec])

if __name__ == '__main__':
    import multiprocessing as mp
    tasks=[(h,s) for h in (5,10,12,15,20,25,30) for s in ('circle_crossing','square_crossing')]
    with mp.get_context('spawn').Pool(14) as pool, open(sys.argv[1],'a') as fh:
        for r in pool.imap_unordered(job, tasks):
            fh.write(json.dumps(r)+'\n'); fh.flush(); print('done', r['humans'], r['scene'], r['success'], flush=True)
