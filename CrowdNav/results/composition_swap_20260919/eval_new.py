"""Evaluate remote-trained IL checkpoints locally across crowd sizes."""
import os, sys, json, configparser, io, contextlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
ROOT='/home/abc/workspace/nav_data/mamba/camrl/CrowdNav'
STAGE='/tmp/claude-1000/-home-abc/f8a2c1a1-47f8-41fe-adc9-f2b70b538149/scratchpad/stage/CrowdNav'
sys.path.insert(0, STAGE)          # 用带 mean/lse 的暂存副本
import numpy as np, torch
CKPT_DIR = os.environ.get('CKPT_DIR','/home/abc/temp/compose_runs_20260919')

def job(task):
    name, label, humans, scene = task
    torch.set_num_threads(1); os.chdir(STAGE)
    from crowd_nav.bayesian import LocalValueNetwork as Net, action_table
    from crowd_nav import train as T
    ck = torch.load(f'{CKPT_DIR}/{name}/{label}.pt', map_location='cpu', weights_only=False)
    cfg = configparser.ConfigParser(); cfg.read_dict(ck['config'])
    model = Net(action_table(cfg), composition=ck.get('composition','mixture'))
    model.load_state_dict(ck['state'], strict=True); model.eval()
    with contextlib.redirect_stdout(io.StringIO()):
        out = T.evaluate(model, cfg, ck['arm'], 50, 9900000, humans, 'test', scene)
    return dict(run=name, label=label, composition=ck.get('composition'), stage=ck.get('stage'),
                humans=humans, scene=scene, episodes=out['episodes'], success=out['success'],
                collision=out['collision'], timeout=out['timeout'],
                outcomes=[r['outcome'] for r in out['records']])

if __name__ == '__main__':
    import multiprocessing as mp
    names = sys.argv[1].split(','); out_path = sys.argv[2]; workers = int(sys.argv[3])
    dens = tuple(int(x) for x in os.environ.get('DENS','5,10,12,20').split(','))
    tasks = [(n,'il',h,s) for n in names for h in dens for s in ('circle_crossing','square_crossing')]
    with mp.get_context('spawn').Pool(workers) as pool, open(out_path,'a') as fh:
        for r in pool.imap_unordered(job, tasks):
            fh.write(json.dumps(r)+'\n'); fh.flush()
            print('done', r['run'], r['humans'], r['scene'], r['success'], flush=True)
