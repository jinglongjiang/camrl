#!/usr/bin/env python3
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return True


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        try:
            devnull = open(os.devnull, 'w')
            sys.stdout = devnull
            sys.stderr = devnull
        except Exception:
            pass


def wait_for_lock(out_path, poll_seconds):
    lock_path = Path(str(out_path) + '.lock')
    while lock_path.exists():
        raw = lock_path.read_text(encoding='utf-8', errors='ignore').strip()
        if raw and pid_alive(raw):
            safe_print(f"[EVAL-QUEUE] waiting for active lock: {lock_path} pid={raw}", flush=True)
            time.sleep(max(5, int(poll_seconds)))
            continue
        safe_print(f"[EVAL-QUEUE] removing stale lock: {lock_path}", flush=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        break


def active_lock_pid(out_path):
    lock_path = Path(str(out_path) + '.lock')
    if not lock_path.exists():
        return None
    raw = lock_path.read_text(encoding='utf-8', errors='ignore').strip()
    if raw and pid_alive(raw):
        return raw
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return None


def wait_for_existing_preset_locks(tasks, poll_seconds):
    while True:
        active = []
        for item in tasks:
            pid = active_lock_pid(item['out'])
            if pid is not None:
                active.append((item['name'], pid))
        if not active:
            return
        label = ', '.join(f"{name}:pid={pid}" for name, pid in active)
        safe_print(f"[EVAL-QUEUE] waiting for existing active preset run(s): {label}", flush=True)
        time.sleep(max(5, int(poll_seconds)))


def task(name, script, out, log_dir, extra, seeds=None):
    return {
        'name': name,
        'script': script,
        'out': out,
        'log_dir': log_dir,
        'extra': extra,
        'seeds': seeds,
    }


def eval35_remaining_tasks():
    common = [
        '--gpu',
        '--weights', 'rl_model_pathY_champion_82_8',
        '--episodes', '100',
        '--legacy_cases',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    stress_common = [
        '--gpu',
        '--episodes', '100',
        '--test_sims', 'square_crossing',
        '--human_nums', '5,8,10,15,20,25,30',
        '--square_width', '10',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    return [
        task(
            'ours_no_vl_s1_10',
            'test.py',
            'runs/eval35/ours_no_vl_s1_10.csv',
            'runs/eval35/ours_no_vl_s1_10_logs',
            common + ['--ablation_profile', 'no_vl'],
        ),
        task(
            'ours_nearest_s1_10',
            'test.py',
            'runs/eval35/ours_nearest_s1_10.csv',
            'runs/eval35/ours_nearest_s1_10_logs',
            common + ['--ablation_profile', 'nearest'],
        ),
        task(
            'ours_veto_025_s1_10',
            'test.py',
            'runs/eval35/ours_veto_025_s1_10.csv',
            'runs/eval35/ours_veto_025_s1_10_logs',
            common + ['--ablation_profile', 'veto025'],
        ),
        task(
            'ours_veto_045_s1_10',
            'test.py',
            'runs/eval35/ours_veto_045_s1_10.csv',
            'runs/eval35/ours_veto_045_s1_10_logs',
            common + ['--ablation_profile', 'veto045'],
        ),
        task(
            'ours_stress_square_s1_10',
            'test.py',
            'runs/eval35/ours_stress_square_s1_10.csv',
            'runs/eval35/ours_stress_square_s1_10_logs',
            ['--weights', 'rl_model_pathY_champion_82_8'] + stress_common,
        ),
        task(
            'lstm_stress_square_s1_10',
            'test3.py',
            'runs/eval35/lstm_stress_square_s1_10.csv',
            'runs/eval35/lstm_stress_square_s1_10_logs',
            stress_common,
        ),
        task(
            'sarl_stress_square_s1_10',
            'test2.py',
            'runs/eval35/sarl_stress_square_s1_10.csv',
            'runs/eval35/sarl_stress_square_s1_10_logs',
            stress_common,
        ),
    ]


def eval35_baseline_adapt_tasks():
    baseline_common = [
        '--gpu',
        '--episodes', '100',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    stress_common = [
        '--gpu',
        '--episodes', '100',
        '--test_sims', 'square_crossing',
        '--human_nums', '5,8,10,15,20,25,30',
        '--square_width', '10',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    return [
        task(
            'sarl_stress_square_s1_10',
            'test2.py',
            'runs/eval35/sarl_stress_square_s1_10.csv',
            'runs/eval35/sarl_stress_square_s1_10_logs',
            stress_common,
        ),
        task(
            'orca_adapt_s1_10',
            'test5.py',
            'runs/eval35/orca_adapt_s1_10.csv',
            'runs/eval35/orca_adapt_s1_10_logs',
            baseline_common,
        ),
        task(
            'dsrnn_example_adapt_s1_10',
            'test7.py',
            'runs/eval35/dsrnn_example_adapt_s1_10.csv',
            'runs/eval35/dsrnn_example_adapt_s1_10_logs',
            baseline_common + [
                '--model_dir', 'runs/dsrnn',
                '--weights', 'dsrnn_example_27776.pt',
                '--dsrnn_selection_mode', 'static_nearest',
            ],
        ),
    ]


def eval35_polish_tasks():
    ours_common = [
        '--gpu',
        '--weights', 'rl_model_pathY_champion_82_8',
        '--episodes', '100',
        '--legacy_cases',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    baseline_common = [
        '--gpu',
        '--episodes', '100',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    return [
        task(
            'ours_no_gdbn_veto_s1_10',
            'test.py',
            'runs/eval35/ours_no_gdbn_veto_s1_10.csv',
            'runs/eval35/ours_no_gdbn_veto_s1_10_logs',
            ours_common + ['--disable_gdbn_veto'],
        ),
        task(
            'cadrl_adapt_s1_10',
            'test4.py',
            'runs/eval35/cadrl_adapt_s1_10.csv',
            'runs/eval35/cadrl_adapt_s1_10_logs',
            baseline_common,
        ),
    ]


def eval35_outdoor_tasks():
    stress_common = [
        '--gpu',
        '--weights', 'rl_model_pathY_champion_82_8',
        '--episodes', '100',
        '--test_sims', 'square_crossing',
        '--human_nums', '5,8,10,15,20,25,30',
        '--square_width', '10',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    ours_common = [
        '--gpu',
        '--weights', 'rl_model_pathY_champion_82_8',
        '--episodes', '100',
        '--legacy_cases',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    baseline_common = [
        '--gpu',
        '--episodes', '100',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    return [
        task(
            'cadrl_adapt_s1_10',
            'test4.py',
            'runs/eval35/cadrl_adapt_s1_10.csv',
            'runs/eval35/cadrl_adapt_s1_10_logs',
            baseline_common,
            seeds='1-10',
        ),
        task(
            'ours_stress_no_gdbn_veto_s1_3',
            'test.py',
            'runs/eval35/ours_stress_no_gdbn_veto_s1_3.csv',
            'runs/eval35/ours_stress_no_gdbn_veto_s1_3_logs',
            stress_common + ['--disable_gdbn_veto'],
            seeds='1-3',
        ),
        task(
            'ours_stress_no_vl_s1_3',
            'test.py',
            'runs/eval35/ours_stress_no_vl_s1_3.csv',
            'runs/eval35/ours_stress_no_vl_s1_3_logs',
            stress_common + ['--ablation_profile', 'no_vl'],
            seeds='1-3',
        ),
        task(
            'ours_stress_nearest_s1_3',
            'test.py',
            'runs/eval35/ours_stress_nearest_s1_3.csv',
            'runs/eval35/ours_stress_nearest_s1_3_logs',
            stress_common + ['--ablation_profile', 'nearest'],
            seeds='1-3',
        ),
        task(
            'ours_noise010_s1_3',
            'test.py',
            'runs/eval35/ours_noise010_s1_3.csv',
            'runs/eval35/ours_noise010_s1_3_logs',
            ours_common + ['--obs_noise_std', '0.10'],
            seeds='1-3',
        ),
        task(
            'ours_noise010_no_gdbn_veto_s1_3',
            'test.py',
            'runs/eval35/ours_noise010_no_gdbn_veto_s1_3.csv',
            'runs/eval35/ours_noise010_no_gdbn_veto_s1_3_logs',
            ours_common + ['--obs_noise_std', '0.10', '--disable_gdbn_veto'],
            seeds='1-3',
        ),
    ]


def gdbn_decisive_tasks():
    common = [
        '--gpu',
        '--weights', 'rl_model_pathY_champion_82_8',
        '--episodes', '100',
        '--legacy_cases',
        '--time_limit', '35',
        '--progress_interval', '100',
    ]
    root = 'runs/eval35_decisive'
    return [
        task(
            'base_vl_only_s1_10',
            'test_gdbn.py',
            f'{root}/base_vl_only_s1_10.csv',
            f'{root}/base_vl_only_s1_10_logs',
            common + [
                '--ablation_profile', 'vl_only',
                '--disable_belief_tokens',
            ],
        ),
        task(
            'cv_governor_s1_10',
            'test_gdbn.py',
            f'{root}/cv_governor_s1_10.csv',
            f'{root}/cv_governor_s1_10_logs',
            common + [
                '--risk_model', 'cv',
                '--disable_belief_tokens',
            ],
        ),
        task(
            'k1_governor_s1_10',
            'test_gdbn.py',
            f'{root}/k1_governor_s1_10.csv',
            f'{root}/k1_governor_s1_10_logs',
            common + [
                '--risk_model', 'gdbn',
                '--gdbn_params', 'gdbn_params_k1',
                '--gdbn_K', '1',
                '--disable_belief_tokens',
            ],
        ),
        task(
            'k4_belief_off_s1_10',
            'test_gdbn.py',
            f'{root}/k4_belief_off_s1_10.csv',
            f'{root}/k4_belief_off_s1_10_logs',
            common + [
                '--risk_model', 'gdbn',
                '--gdbn_params', 'gdbn_params_k4',
                '--gdbn_K', '4',
                '--disable_belief_tokens',
            ],
        ),
    ]


def fullcrowd_formal_tasks():
    common = [
        '--gpu',
        '--episodes', '100',
        '--time-limit', '35',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
    ]
    root = 'runs/eval35_fullcrowd'
    return [
        task(
            'bayesian_fullcrowd_s1_10',
            'test.py',
            f'{root}/bayesian_fullcrowd_s1_10.csv',
            f'{root}/bayesian_fullcrowd_s1_10_logs',
            common + [
                '--policy', 'bayesian_fullcrowd_risk_value',
                '--model_dir', 'runs/bayesian_distributional',
                '--weights',
                'model_fullcrowd_directrisk_lambda1_clean.pth',
                '--policy_config',
                'configs/policy_bayesian_distributional.config',
            ],
        ),
        task(
            'mamba_value_base_s1_10',
            'test.py',
            f'{root}/mamba_value_base_s1_10.csv',
            f'{root}/mamba_value_base_s1_10_logs',
            common + [
                '--policy', 'mamba',
                '--model_dir', 'runs/mamba_vl',
                '--weights', 'rl_model_ep10000_T24.pth',
                '--policy_config', 'configs/policy.config',
            ],
        ),
    ]


def bayesian_model_average_formal_tasks():
    """Paired model-selection study on nonstationary and nominal crowds."""
    common = [
        '--gpu',
        '--time-limit', '35',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
        '--test-cases', '0,3',
        '--behavior-seed-offset', '104729',
    ]
    risk_checkpoint = [
        '--model_dir', 'runs/bayesian_distributional',
        '--weights', 'model_fullcrowd_directrisk_lambda1_clean.pth',
    ]
    conditions = [
        (
            'bma',
            'bayesian_model_average_risk_value',
            'configs/policy_bayesian_model_average.config',
            risk_checkpoint,
        ),
        (
            'gdbn',
            'bayesian_fullcrowd_risk_value',
            'configs/policy_bayesian_distributional.config',
            risk_checkpoint,
        ),
        (
            'cv',
            'cv_fullcrowd_risk_value',
            'configs/policy_bayesian_distributional.config',
            risk_checkpoint,
        ),
        (
            'base',
            'mamba',
            'configs/policy.config',
            [
                '--model_dir', 'runs/mamba_vl',
                '--weights', 'rl_model_ep10000_T24.pth',
            ],
        ),
        (
            'fixed50',
            'fixed_model_average_risk_value',
            'configs/policy_fixed_model_average.config',
            risk_checkpoint,
        ),
    ]
    root = 'runs/eval35_bayesian_model_average'
    tasks = []
    for label, policy_name, policy_config, checkpoint_args in conditions:
        name = f'{label}_heldout_nonstationary_s1_10'
        tasks.append(
            task(
                name,
                'test.py',
                f'{root}/{name}.csv',
                f'{root}/{name}_logs',
                common + [
                    '--episodes', '100',
                    '--behavior-profile', 'heldout_nonstationary',
                    '--policy', policy_name,
                    '--policy_config', policy_config,
                ] + checkpoint_args,
                seeds='1-10',
            )
        )
    for label, policy_name, policy_config, checkpoint_args in conditions:
        name = f'{label}_nominal_s1_3'
        tasks.append(
            task(
                name,
                'test.py',
                f'{root}/{name}.csv',
                f'{root}/{name}_logs',
                common + [
                    '--episodes', '100',
                    '--behavior-profile', 'nominal',
                    '--policy', policy_name,
                    '--policy_config', policy_config,
                ] + checkpoint_args,
                seeds='1-3',
            )
        )
    return tasks


def fullcrowd_supplement_tasks():
    common = [
        '--gpu',
        '--episodes', '100',
        '--time-limit', '35',
        '--test-size', '1000',
        '--case-block-by-seed',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
    ]
    root = 'runs/eval35_fullcrowd_supplement'
    return [
        task(
            'bayesian_fullcrowd_cases500_999',
            'test.py',
            f'{root}/bayesian_fullcrowd_cases500_999.csv',
            f'{root}/bayesian_fullcrowd_cases500_999_logs',
            common + [
                '--policy', 'bayesian_fullcrowd_risk_value',
                '--model_dir', 'runs/bayesian_distributional',
                '--weights',
                'model_fullcrowd_directrisk_lambda1_clean.pth',
                '--policy_config',
                'configs/policy_bayesian_distributional.config',
            ],
            seeds='5-9',
        ),
        task(
            'mamba_value_base_cases500_999',
            'test.py',
            f'{root}/mamba_value_base_cases500_999.csv',
            f'{root}/mamba_value_base_cases500_999_logs',
            common + [
                '--policy', 'mamba',
                '--model_dir', 'runs/mamba_vl',
                '--weights', 'rl_model_ep10000_T24.pth',
                '--policy_config', 'configs/policy.config',
            ],
            seeds='5-9',
        ),
    ]


def fullcrowd_cv_control_tasks():
    common = [
        '--gpu',
        '--episodes', '100',
        '--time-limit', '35',
        '--test-size', '1000',
        '--case-block-by-seed',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
        '--model_dir', 'runs/bayesian_distributional',
        '--weights', 'model_fullcrowd_directrisk_lambda1_clean.pth',
        '--policy_config', 'configs/policy_bayesian_distributional.config',
    ]
    root = 'runs/eval35_fullcrowd_cv_control'
    tasks = []
    for case_id, case_name in ((3, 'dense_square'), (5, 'large_square')):
        for policy_name, output_name in (
            ('bayesian_fullcrowd_risk_value', 'gdbn'),
            ('cv_fullcrowd_risk_value', 'cv'),
        ):
            name = f'{output_name}_{case_name}_s1_3'
            tasks.append(
                task(
                    name,
                    'test.py',
                    f'{root}/{name}.csv',
                    f'{root}/{name}_logs',
                    common + [
                        '--policy', policy_name,
                        '--test_case', str(case_id),
                    ],
                    seeds='1-3',
                )
            )
    return tasks


def fullcrowd_tail_diagnostic_tasks():
    common = [
        '--gpu',
        '--episodes', '100',
        '--time-limit', '35',
        '--test-size', '1000',
        '--case-block-by-seed',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
        '--model_dir', 'runs/bayesian_distributional',
        '--weights', 'model_fullcrowd_directrisk_lambda1_clean.pth',
        '--policy_config', 'configs/policy_bayesian_fullcrowd_tail.config',
        '--policy', 'bayesian_fullcrowd_risk_value',
    ]
    root = 'runs/eval35_fullcrowd_tail_diagnostic'
    tasks = []
    for case_id, case_name in ((3, 'dense_square'), (5, 'large_square')):
        name = f'gdbn_tail_{case_name}_s1_3'
        tasks.append(
            task(
                name,
                'test.py',
                f'{root}/{name}.csv',
                f'{root}/{name}_logs',
                common + ['--test_case', str(case_id)],
                seeds='1-3',
            )
        )
    return tasks


def fullcrowd_compact_control_tasks():
    common = [
        '--gpu',
        '--episodes', '100',
        '--time-limit', '35',
        '--test-size', '1000',
        '--case-block-by-seed',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
        '--model_dir', 'runs/bayesian_distributional',
        '--weights', 'model_fullcrowd_directrisk_lambda1_clean.pth',
        '--policy_config', 'configs/policy_bayesian_fullcrowd_compact.config',
        '--policy', 'bayesian_fullcrowd_risk_value',
    ]
    root = 'runs/eval35_fullcrowd_compact_control'
    tasks = []
    for case_id, case_name in ((3, 'dense_square'), (5, 'large_square')):
        name = f'gdbn_compact5_{case_name}_s1_3'
        tasks.append(
            task(
                name,
                'test.py',
                f'{root}/{name}.csv',
                f'{root}/{name}_logs',
                common + ['--test_case', str(case_id)],
                seeds='1-3',
            )
        )
    return tasks


def fullcrowd_noise_control_tasks():
    common = [
        '--gpu',
        '--episodes', '100',
        '--time-limit', '35',
        '--test-size', '1000',
        '--case-block-by-seed',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
        '--model_dir', 'runs/bayesian_distributional',
        '--weights', 'model_fullcrowd_directrisk_lambda1_clean.pth',
        '--policy_config', 'configs/policy_bayesian_distributional.config',
        '--obs-noise-std', '0.10',
    ]
    root = 'runs/eval35_fullcrowd_noise_control'
    tasks = []
    for case_id, case_name in ((3, 'dense_square'), (5, 'large_square')):
        for policy_name, output_name in (
            ('bayesian_fullcrowd_risk_value', 'gdbn'),
            ('cv_fullcrowd_risk_value', 'cv'),
        ):
            name = f'{output_name}_noise010_{case_name}_s1_3'
            tasks.append(
                task(
                    name,
                    'test.py',
                    f'{root}/{name}.csv',
                    f'{root}/{name}_logs',
                    common + [
                        '--policy', policy_name,
                        '--test_case', str(case_id),
                    ],
                    seeds='1-3',
                )
            )
    return tasks


def fullcrowd_latency_tasks():
    common = [
        '--gpu',
        '--episodes', '20',
        '--time-limit', '35',
        '--test-size', '1000',
        '--case-block-by-seed',
        '--seq_len', '24',
        '--temporal-backbone', 'mamba',
        '--measure-latency',
    ]
    root = 'runs/eval35_fullcrowd_latency'
    tasks = []
    for case_id, case_name in (
        (0, 'circle5'),
        (1, 'square10'),
        (3, 'square20'),
    ):
        name = f'gdbn_{case_name}_s1'
        tasks.append(
            task(
                name,
                'test.py',
                f'{root}/{name}.csv',
                f'{root}/{name}_logs',
                common + [
                    '--policy', 'bayesian_fullcrowd_risk_value',
                    '--model_dir', 'runs/bayesian_distributional',
                    '--weights',
                    'model_fullcrowd_directrisk_lambda1_clean.pth',
                    '--policy_config',
                    'configs/policy_bayesian_distributional.config',
                    '--test_case', str(case_id),
                ],
                seeds='1',
            )
        )
    name = 'mamba_base_square20_s1'
    tasks.append(
        task(
            name,
            'test.py',
            f'{root}/{name}.csv',
            f'{root}/{name}_logs',
            common + [
                '--policy', 'mamba',
                '--model_dir', 'runs/mamba_vl',
                '--weights', 'rl_model_ep10000_T24.pth',
                '--policy_config', 'configs/policy.config',
                '--test_case', '3',
            ],
            seeds='1',
        )
    )
    return tasks


def ensure_k1_params():
    model_dir = Path('runs/mamba_vl/gdbn_params_k1')
    required = [model_dir / name for name in ('gng.npz', 'gdbn.npz', 'action_model.npz')]
    if all(path.is_file() for path in required):
        safe_print(f"[EVAL-QUEUE] using existing K=1 parameters: {model_dir}", flush=True)
        return

    fitter = Path(__file__).resolve().with_name('fit_k1_gdbn.py')
    cmd = [
        sys.executable, str(fitter),
        '--demo', '../orca_demos_seq.npz',
        '--output', str(model_dir),
    ]
    safe_print('[EVAL-QUEUE] fitting K=1 parameters: ' + ' '.join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)


def detach_self(args):
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(Path(__file__).resolve())]
    for item in sys.argv[1:]:
        if item == '--detach':
            continue
        argv.append(item)
    with log_path.open('ab', buffering=0) as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    safe_print(f"[EVAL-QUEUE] detached pid={proc.pid}")
    safe_print(f"[EVAL-QUEUE] log={log_path}")
    safe_print(f"[EVAL-QUEUE] status: tail -f {shlex.quote(str(log_path))}")


def main():
    if hasattr(signal, 'SIGHUP'):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description='Run eval35 seed suites sequentially with resume.')
    parser.add_argument(
        '--preset',
        choices=[
            'eval35_remaining',
            'eval35_baseline_adapt',
            'eval35_polish',
            'eval35_outdoor',
            'gdbn_decisive',
            'fullcrowd_formal',
            'fullcrowd_supplement',
            'fullcrowd_cv_control',
            'fullcrowd_tail_diagnostic',
            'fullcrowd_compact_control',
            'fullcrowd_noise_control',
            'fullcrowd_latency',
            'bayesian_model_average_formal',
        ],
        default='eval35_remaining',
    )
    parser.add_argument('--seeds', default='1-10')
    parser.add_argument('--child_threads', type=int, default=1)
    parser.add_argument('--poll_seconds', type=int, default=60)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--log', default='runs/eval35/eval35_queue.controller.log')
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='Stream child output and retain per-episode progress bars',
    )
    parser.add_argument('--allow_parallel_existing', action='store_true',
                        help='Do not wait for active locks from other tasks in this preset before starting')
    args = parser.parse_args()

    if args.detach:
        detach_self(args)
        return

    if args.preset == 'eval35_remaining':
        tasks = eval35_remaining_tasks()
    elif args.preset == 'eval35_baseline_adapt':
        tasks = eval35_baseline_adapt_tasks()
    elif args.preset == 'eval35_polish':
        tasks = eval35_polish_tasks()
    elif args.preset == 'eval35_outdoor':
        tasks = eval35_outdoor_tasks()
    elif args.preset == 'gdbn_decisive':
        ensure_k1_params()
        tasks = gdbn_decisive_tasks()
    elif args.preset == 'fullcrowd_formal':
        tasks = fullcrowd_formal_tasks()
    elif args.preset == 'fullcrowd_supplement':
        tasks = fullcrowd_supplement_tasks()
    elif args.preset == 'fullcrowd_cv_control':
        tasks = fullcrowd_cv_control_tasks()
    elif args.preset == 'fullcrowd_tail_diagnostic':
        tasks = fullcrowd_tail_diagnostic_tasks()
    elif args.preset == 'fullcrowd_compact_control':
        tasks = fullcrowd_compact_control_tasks()
    elif args.preset == 'fullcrowd_noise_control':
        tasks = fullcrowd_noise_control_tasks()
    elif args.preset == 'fullcrowd_latency':
        tasks = fullcrowd_latency_tasks()
    elif args.preset == 'bayesian_model_average_formal':
        tasks = bayesian_model_average_formal_tasks()
    else:
        raise SystemExit(f"Unknown preset: {args.preset}")

    if not args.allow_parallel_existing:
        wait_for_existing_preset_locks(tasks, args.poll_seconds)

    runner = Path(__file__).resolve().with_name('run_seed_suite.py')
    for idx, item in enumerate(tasks, start=1):
        safe_print(f"[EVAL-QUEUE] task {idx}/{len(tasks)} start: {item['name']}", flush=True)
        wait_for_lock(item['out'], args.poll_seconds)
        cmd = [
            sys.executable, str(runner),
            '--child_threads', str(max(1, int(args.child_threads))),
            '--script', item['script'],
            '--seeds', str(item.get('seeds') or args.seeds),
            '--out', item['out'],
            '--log_dir', item['log_dir'],
        ]
        if args.show_progress:
            cmd.append('--show_progress')
        else:
            cmd.append('--quiet_child')
        cmd.extend(['--'] + item['extra'])
        safe_print('[EVAL-QUEUE] command=' + ' '.join(shlex.quote(x) for x in cmd), flush=True)
        code = subprocess.call(cmd)
        if code != 0:
            raise SystemExit(f"[EVAL-QUEUE] task failed: {item['name']} exit={code}")
        safe_print(f"[EVAL-QUEUE] task {idx}/{len(tasks)} done: {item['name']}", flush=True)

    if args.preset == 'fullcrowd_supplement':
        combiner = Path(__file__).resolve().with_name(
            'combine_fullcrowd_eval.py'
        )
        safe_print(
            '[EVAL-QUEUE] combining cases 0-499 with cases 500-999',
            flush=True,
        )
        subprocess.check_call([sys.executable, str(combiner)])

    safe_print('[EVAL-QUEUE] all tasks complete', flush=True)


if __name__ == '__main__':
    main()
