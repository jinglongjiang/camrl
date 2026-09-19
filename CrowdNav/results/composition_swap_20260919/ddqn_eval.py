"""Finish the unevaluated Double-DQN checkpoints (training already completed)."""
import os, sys, json, configparser, io, contextlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
ROOT='/home/abc/workspace/nav_data/mamba/camrl/CrowdNav'
sys.path.insert(0, ROOT)
RUNS='/home/abc/temp/local_event_results_20260918/matched_ddqn_runs'
import torch

def job(task):
    run, label, humans, scene = task
    torch.set_num_threads(1); os.chdir(ROOT)
    from crowd_nav.bayesian import LocalValueNetwork as Net, action_table
    from crowd_nav import train as T
    ckpt = torch.load(f'{RUNS}/{run}/{label}.pt', map_location='cpu', weights_only=False)
    cfg = configparser.ConfigParser(); cfg.read_dict(ckpt['config'])
    model = Net(action_table(cfg), composition=ckpt.get('composition','mixture'))
    model.load_state_dict(ckpt['state'], strict=True); model.eval()
    with contextlib.redirect_stdout(io.StringIO()):
        out = T.evaluate(model, cfg, ckpt['arm'], 50, 9900000, humans, 'test', scene)
    return dict(run=run, label=label, arm=ckpt['arm'], stage=ckpt.get('stage'),
                humans=humans, scene=scene, episodes=out['episodes'], success=out['success'],
                collision=out['collision'], timeout=out['timeout'],
                outcomes=[r['outcome'] for r in out['records']])

if __name__ == '__main__':
    import multiprocessing as mp
    tasks = [(r, l, h, s) for r in ('prior_7207','full_7207') for l in ('final','last')
             for h in (5,10,12,20) for s in ('circle_crossing','square_crossing')]
    with mp.get_context('spawn').Pool(int(sys.argv[2])) as pool, open(sys.argv[1],'a') as fh:
        for r in pool.imap_unordered(job, tasks):
            fh.write(json.dumps(r)+'\n'); fh.flush()
            print('done', r['run'], r['label'], r['humans'], r['scene'], r['success'], flush=True)
