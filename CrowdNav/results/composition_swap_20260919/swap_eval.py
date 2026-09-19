"""Zero-training control: same trained weights, swap the composition operator at execution."""
import os, sys, json, configparser, io, contextlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
ROOT='/home/abc/workspace/nav_data/mamba/camrl/CrowdNav'
sys.path.insert(0, ROOT)
RUNS='/home/abc/temp/local_event_results_20260918/matched_attention_runs'
import numpy as np, torch

def job(task):
    seed, comp, humans, scene, count, start = task
    torch.set_num_threads(1)
    os.chdir(ROOT)
    from crowd_nav.bayesian import LocalValueNetwork as BeliefQNetwork, action_table
    from crowd_nav import train as T
    ckpt = torch.load(f'{RUNS}/mixture_prior_{seed}/final.pt', map_location='cpu', weights_only=False)
    cfg = configparser.ConfigParser(); cfg.read_dict(ckpt['config'])
    model = BeliefQNetwork(action_table(cfg), composition='mixture')
    model.load_state_dict(ckpt['state'], strict=True)
    model.composition = comp          # only the execution-time operator changes
    model.eval()
    with contextlib.redirect_stdout(io.StringIO()):
        out = T.evaluate(model, cfg, ckpt['arm'], count, start, humans, 'test', scene)
    return dict(seed=seed, composition=comp, humans=humans, scene=scene,
                episodes=out['episodes'], success=out['success'],
                collision=out['collision'], timeout=out['timeout'],
                mean_return=out['mean_return'],
                outcomes=[r['outcome'] for r in out['records']])

DENS = tuple(int(x) for x in os.environ.get('DENS','5,10,12,20').split(','))

if __name__ == '__main__':
    import multiprocessing as mp
    comps = sys.argv[1].split(',')
    seeds = [int(s) for s in sys.argv[2].split(',')]
    count = int(sys.argv[3]); out_path = sys.argv[4]; workers = int(sys.argv[5])
    tasks = [(s, c, h, sc, count, 9900000)
             for s in seeds for c in comps
             for h in DENS for sc in ('circle_crossing', 'square_crossing')]
    with mp.get_context('spawn').Pool(workers) as pool, open(out_path, 'a') as fh:
        for r in pool.imap_unordered(job, tasks):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print('done', r['seed'], r['composition'], r['humans'], r['scene'], r['success'], flush=True)
