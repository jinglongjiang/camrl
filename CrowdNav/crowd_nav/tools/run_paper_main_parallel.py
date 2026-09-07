#!/usr/bin/env python3
"""Run evaluate_bdvl_paper_main.py's 6 scenarios as separate CONCURRENT
subprocesses (one per scenario) instead of sequentially, then merge their
CSVs/manifests into one combined result.

Why subprocesses and not threads/async: each scenario needs its own
CrowdSim + BDVL policy + CUDA context; subprocesses keep those fully
isolated (no shared-state risk) and each gets its own CUDA context,
which is fine on a 12GB card running six tiny IQN networks concurrently.

episode_seed is unaffected by parallelism -- evaluate_bdvl_paper_main.py
always computes case_id from the scenario's fixed index in
SCENARIO_ORDER, never from its position in a --scenarios subset, so the
merged result is bit-identical to running everything in one process.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import BDVL_PRODUCTION_SOURCES, FORMAL_SCENARIOS  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402

SCENARIO_ORDER = tuple(FORMAL_SCENARIOS.keys())


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--episodes-per-scenario", type=int, default=500)
    parser.add_argument("--decision-budget", choices=["validation", "formal"], default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="bayesian_dvl_risk_neutral")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The FINAL merged result is the one thing this script protects --
    # if it already exists, a prior run genuinely completed and must not
    # be silently clobbered.
    if Path(str(output_path) + ".manifest.json").exists():
        raise SystemExit(
            f"refusing to overwrite existing completed result: {output_path}.manifest.json "
            "(use a different --output, or remove it first if you really mean to redo this run)"
        )
    per_scenario_dir = output_path.parent / (output_path.name + "_per_scenario")
    # 2026-08-08 fix: per_scenario_dir is purely this launcher's OWN
    # internal scratch state, never itself a protected final result --
    # unlike the merged output above, leftovers here are always either
    # (a) from a run that was killed/crashed mid-way (no --manifest.json
    # for the FINAL merge exists, confirmed above) or (b) genuinely
    # stale. Previously this raised "refusing to overwrite" on ANY
    # leftover file and left cleanup to the operator, which produced the
    # exact same failure on every retry after any kill/crash. Wipe it
    # automatically instead -- the real protection is the check above.
    if per_scenario_dir.exists():
        import shutil
        print(f"CLEANING_STALE_SCRATCH_DIR {per_scenario_dir} (leftover from a prior incomplete run)", flush=True)
        shutil.rmtree(per_scenario_dir)
    per_scenario_dir.mkdir(parents=True, exist_ok=True)

    procs = {}
    for scenario_key in SCENARIO_ORDER:
        scenario_output = per_scenario_dir / scenario_key
        log_path = per_scenario_dir / f"{scenario_key}.log"
        cmd = [
            sys.executable, "-m", "crowd_nav.tools.evaluate_bdvl_paper_main",
            "--env-config", args.env_config, "--registry", args.registry,
            "--artifact-path", args.artifact_path, "--checkpoint", args.checkpoint,
            "--base-seed", str(args.base_seed), "--episodes-per-scenario", str(args.episodes_per_scenario),
            "--decision-budget", args.decision_budget, "--device", args.device, "--method", args.method,
            "--scenarios", scenario_key, "--output", str(scenario_output),
        ]
        log_handle = log_path.open("w")
        proc = subprocess.Popen(cmd, cwd=str(PACKAGE_ROOT), stdout=log_handle, stderr=subprocess.STDOUT)
        procs[scenario_key] = (proc, log_handle, scenario_output, log_path)
        print(f"LAUNCHED {scenario_key} pid={proc.pid} log={log_path}", flush=True)

    # 2026-08-08 fix: the original version just called proc.wait() on each
    # subprocess in turn, printing NOTHING until the whole run finished --
    # each scenario's own progress bar only exists inside its own log
    # file, so the launcher's own terminal looked dead for however long
    # the run takes (tens of minutes), even though every subprocess was
    # actively working. Poll all six scenarios' progress.jsonl line
    # counts every few seconds and print one aggregate line instead.
    total_target = len(SCENARIO_ORDER) * args.episodes_per_scenario
    last_report = 0.0
    while True:
        all_done = all(proc.poll() is not None for proc, *_rest in procs.values())
        now = time.time()
        if now - last_report >= 3.0 or all_done:
            last_report = now
            per_scenario = {}
            for scenario_key, (_proc, _log_handle, scenario_output, _log_path) in procs.items():
                jsonl_path = Path(str(scenario_output) + ".progress.jsonl")
                n = 0
                if jsonl_path.exists():
                    with jsonl_path.open() as handle:
                        n = sum(1 for _ in handle)
                per_scenario[scenario_key] = n
            completed = sum(per_scenario.values())
            detail = " ".join(f"{k}={v}/{args.episodes_per_scenario}" for k, v in per_scenario.items())
            sys.stdout.write(f"\rPARALLEL_PROGRESS {completed}/{total_target} [{detail}]   ")
            sys.stdout.flush()
        if all_done:
            print()
            break
        time.sleep(0.5)

    failed = []
    for scenario_key, (proc, log_handle, scenario_output, log_path) in procs.items():
        log_handle.close()
        if proc.returncode != 0:
            failed.append((scenario_key, proc.returncode, log_path))
        else:
            print(f"SCENARIO_SUBPROCESS_DONE {scenario_key} returncode=0", flush=True)
    if failed:
        for scenario_key, returncode, log_path in failed:
            print(f"SCENARIO_SUBPROCESS_FAILED {scenario_key} returncode={returncode} see={log_path}")
        raise SystemExit(f"{len(failed)} scenario subprocess(es) failed; see logs above")

    # Merge the six per-scenario CSVs into one combined CSV.
    all_rows = []
    fieldnames = None
    for scenario_key in SCENARIO_ORDER:
        _, _, scenario_output, _ = procs[scenario_key]
        csv_path = Path(str(scenario_output) + ".csv")
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            all_rows.extend(reader)

    merged_csv_path = Path(str(output_path) + ".csv")
    if merged_csv_path.exists():
        raise SystemExit(f"refusing to overwrite existing merged CSV: {merged_csv_path}")
    with merged_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
    for row in all_rows:
        outcome_counts[row["outcome"]] += 1
    n = len(all_rows)
    expected_n = len(SCENARIO_ORDER) * args.episodes_per_scenario
    if n != expected_n:
        raise RuntimeError(f"merged CSV has {n} rows, expected {expected_n}")
    seen_seeds = {(row["scenario"], row["episode_seed"]) for row in all_rows}
    if len(seen_seeds) != n:
        raise RuntimeError(f"duplicate (scenario, episode_seed) pairs in merged output: {n} rows but {len(seen_seeds)} distinct")

    manifest = build_run_manifest(
        repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES,
        extra={
            "protocol": "paper_main_nominal_only_parallel", "method": args.method,
            "base_seed": args.base_seed, "episodes_per_scenario": args.episodes_per_scenario,
            "decision_budget": args.decision_budget, "scenarios": list(SCENARIO_ORDER),
            "n_episodes": n, "overall_sr": outcome_counts["success"] / n,
            "overall_cr": outcome_counts["collision"] / n, "overall_tr": outcome_counts["timeout"] / n,
            "outcomes": outcome_counts,
            "per_scenario_manifest_sha256": {
                scenario_key: sha256_of_file(str(procs[scenario_key][2]) + ".manifest.json")
                for scenario_key in SCENARIO_ORDER
            },
            "merged_csv_sha256": sha256_of_file(str(merged_csv_path)),
        },
    )
    atomic_write_json(str(output_path) + ".manifest.json", manifest)
    print(
        f"RUN_PAPER_MAIN_PARALLEL_OVERALL n_episodes={n} "
        f"sr={outcome_counts['success'] / n:.4f} cr={outcome_counts['collision'] / n:.4f} "
        f"tr={outcome_counts['timeout'] / n:.4f} outcomes={outcome_counts}"
    )
    print(f"RUN_PAPER_MAIN_PARALLEL_DONE n_episodes={n} merged_csv={merged_csv_path}")


if __name__ == "__main__":
    main()
