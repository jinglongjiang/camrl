#!/usr/bin/env python3
import argparse
import atexit
import codecs
import csv
import errno
import fcntl
import os
import pty
import re
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import termios
from pathlib import Path


RESULT_RE = re.compile(r"^Results for (.+):\s*$")
COUNT_RE = re.compile(r"^\s*(SUCCESS|COLLISION|TIMEOUT):\s+(\d+)/(\d+)\s+\(([\d.]+)%\)")


def parse_seeds(value):
    seeds = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    seen = set()
    unique = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            unique.append(seed)
    return unique


def parse_results(text, seed):
    rows = []
    current = None
    stats = {}
    for line in text.splitlines():
        match = RESULT_RE.match(line)
        if match:
            if current and stats:
                rows.append(build_row(seed, current, stats))
            current = match.group(1).strip()
            stats = {}
            continue

        match = COUNT_RE.match(line)
        if match and current:
            key = match.group(1).lower()
            stats[key] = {
                'count': int(match.group(2)),
                'total': int(match.group(3)),
                'rate': float(match.group(4)),
            }

    if current and stats:
        rows.append(build_row(seed, current, stats))
    return rows


def build_row(seed, case, stats):
    total = next((item['total'] for item in stats.values()), 0)
    return {
        'seed': seed,
        'case': case,
        'episodes': total,
        'success': stats.get('success', {}).get('count', 0),
        'collision': stats.get('collision', {}).get('count', 0),
        'timeout': stats.get('timeout', {}).get('count', 0),
        'success_rate': stats.get('success', {}).get('rate', 0.0),
        'collision_rate': stats.get('collision', {}).get('rate', 0.0),
        'timeout_rate': stats.get('timeout', {}).get('rate', 0.0),
    }


def mean(values):
    return statistics.fmean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case'], []).append(row)

    lines = []
    all_seed_means = {}
    for row in rows:
        all_seed_means.setdefault(row['seed'], []).append(float(row['success_rate']))

    lines.append("case,mean_SR,std_SR,mean_CR,mean_TR,n_seeds")
    for case in sorted(by_case):
        case_rows = by_case[case]
        sr = [float(r['success_rate']) for r in case_rows]
        cr = [float(r['collision_rate']) for r in case_rows]
        tr = [float(r['timeout_rate']) for r in case_rows]
        lines.append(
            f"{case},{mean(sr):.2f},{stdev(sr):.2f},{mean(cr):.2f},{mean(tr):.2f},{len(case_rows)}"
        )

    seed_means = [mean(v) for v in all_seed_means.values()]
    lines.append("")
    lines.append(
        f"overall_mean_SR,{mean(seed_means):.2f},overall_std_across_seeds,{stdev(seed_means):.2f},n_seeds,{len(seed_means)}"
    )
    return "\n".join(lines)


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


def load_existing_rows(path):
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def completed_seeds(rows):
    out = set()
    for row in rows:
        try:
            out.add(int(row.get('seed')))
        except Exception:
            continue
    return out


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return True


def acquire_lock(lock_path):
    lock_path = Path(lock_path)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(str(os.getpid()))
            break
        except FileExistsError:
            try:
                raw = lock_path.read_text(encoding='utf-8').strip()
                if raw and pid_alive(int(raw)):
                    raise SystemExit(f"Another run appears active for this output: {lock_path} pid={raw}")
            except ValueError:
                pass
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _cleanup():
        try:
            if lock_path.read_text(encoding='utf-8').strip() == str(os.getpid()):
                lock_path.unlink()
        except Exception:
            pass
    atexit.register(_cleanup)


def run_with_pty(cmd, child_env, log_file):
    """Stream a child through a pseudo-terminal so tqdm can overwrite in place."""
    master_fd, slave_fd = pty.openpty()
    try:
        if sys.stdout.isatty():
            size = os.get_terminal_size(sys.stdout.fileno())
            winsize = struct.pack("HHHH", size.lines, size.columns, 0, 0)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            env=child_env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    output_parts = []
    try:
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            text = decoder.decode(data)
            output_parts.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
            if log_file:
                log_file.write(text)
                log_file.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            output_parts.append(tail)
            sys.stdout.write(tail)
            sys.stdout.flush()
            if log_file:
                log_file.write(tail)
                log_file.flush()
        return proc.wait(), "".join(output_parts)
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        raise
    finally:
        os.close(master_fd)


def detach_self(args):
    out_path = Path(args.out)
    if args.log_dir:
        controller_dir = Path(args.log_dir)
    else:
        controller_dir = out_path.parent
    controller_dir.mkdir(parents=True, exist_ok=True)
    controller_log = controller_dir / (out_path.stem + '.controller.log')

    argv = [sys.executable, str(Path(__file__).resolve())]
    skip_next = False
    for item in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if item == '--detach':
            continue
        if item == '--python':
            argv.extend(['--python', args.python])
            skip_next = True
            continue
        argv.append(item)
    if '--quiet_child' not in argv:
        argv.insert(2, '--quiet_child')

    env = os.environ.copy()
    with controller_log.open('ab', buffering=0) as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )

    cmd_line = ' '.join(shlex.quote(x) for x in argv)
    safe_print(f"[SEED-SUITE] detached pid={proc.pid}")
    safe_print(f"[SEED-SUITE] controller_log={controller_log}")
    safe_print(f"[SEED-SUITE] command={cmd_line}")
    safe_print(f"[SEED-SUITE] status: tail -f {shlex.quote(str(controller_log))}")


def main():
    if hasattr(signal, 'SIGHUP'):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Run one test script over a fixed seed suite and aggregate SR/CR/TR."
    )
    parser.add_argument('--script', required=True, help='Test script, e.g. test.py or test2.py')
    parser.add_argument('--seeds', default='1-100', help='Seed list/range, e.g. 1-20 or 1,2,3,42')
    parser.add_argument('--out', required=True, help='CSV output path')
    parser.add_argument('--log_dir', default=None, help='Optional directory for raw per-seed logs')
    parser.add_argument('--python', default=sys.executable, help='Python executable')
    parser.add_argument('--show_progress', action='store_true', help='Keep child tqdm progress bars enabled')
    parser.add_argument('--quiet_child', action='store_true',
                        help='Do not stream child process output to the terminal; logs are still saved when --log_dir is set')
    parser.add_argument('--detach', action='store_true',
                        help='Start this seed suite in a detached session and return immediately')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not reuse complete seeds already present in --out')
    parser.add_argument('--child_threads', type=int, default=1,
                        help='Default BLAS/OpenMP/Torch thread count for each child process')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Arguments passed to the test script after --')
    args = parser.parse_args()
    if args.detach:
        detach_self(args)
        return

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == '--':
        extra_args = extra_args[1:]
    if not args.show_progress and '--no_progress' not in extra_args:
        extra_args.append('--no_progress')

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds parsed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acquire_lock(str(out_path) + '.lock')
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    rows = [] if args.no_resume else load_existing_rows(out_path)
    done_seeds = completed_seeds(rows)
    if done_seeds:
        safe_print(f"[SEED-SUITE] resume: loaded {len(rows)} rows for seeds={sorted(done_seeds)}", flush=True)

    child_env = os.environ.copy()
    thread_count = str(max(1, int(args.child_threads)))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        child_env[key] = thread_count
    child_env['TORCH_NUM_THREADS'] = thread_count

    script = str(Path(args.script))
    for idx, seed in enumerate(seeds, start=1):
        if int(seed) in done_seeds:
            safe_print(f"[SEED-SUITE] {idx}/{len(seeds)} seed={seed}: already complete, skipped", flush=True)
            continue
        cmd = [args.python, script] + extra_args + ['--seed', str(seed)]
        safe_print(f"[SEED-SUITE] {idx}/{len(seeds)} seed={seed}: {' '.join(cmd)}", flush=True)
        log_file = None
        try:
            if args.log_dir:
                log_path = Path(args.log_dir) / f"seed_{seed}.log"
                log_file = log_path.open('w', encoding='utf-8', buffering=1024 * 1024)
            if args.show_progress:
                return_code, output = run_with_pty(cmd, child_env, log_file)
            else:
                proc = subprocess.Popen(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    env=child_env,
                    start_new_session=True,
                )
                output_parts = []
                line_count = 0
                assert proc.stdout is not None
                try:
                    for line in proc.stdout:
                        if not args.quiet_child:
                            safe_print(line, end='')
                        output_parts.append(line)
                        if log_file:
                            log_file.write(line)
                            line_count += 1
                            if line_count % 200 == 0:
                                log_file.flush()
                    return_code = proc.wait()
                except KeyboardInterrupt:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                    raise
                output = ''.join(output_parts)
        finally:
            if log_file:
                log_file.close()

        if return_code != 0:
            raise SystemExit(f"Command failed for seed={seed} with exit code {return_code}")
        seed_rows = parse_results(output, seed)
        if not seed_rows:
            raise SystemExit(f"No result rows parsed for seed={seed}")
        rows.extend(seed_rows)
        done_seeds.add(int(seed))

        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'seed', 'case', 'episodes',
                    'success', 'collision', 'timeout',
                    'success_rate', 'collision_rate', 'timeout_rate',
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        summary_path = out_path.with_suffix('.summary.txt')
        summary_path.write_text(summarize(rows) + "\n", encoding='utf-8')
        safe_print(summarize(rows), flush=True)

    safe_print(f"[SEED-SUITE] saved: {out_path}")
    safe_print(f"[SEED-SUITE] summary: {out_path.with_suffix('.summary.txt')}")


if __name__ == '__main__':
    main()
