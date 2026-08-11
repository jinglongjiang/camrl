#!/usr/bin/env python3
"""BDVL paper-main-table evaluator: 6 scenarios x N nominal-only episodes,
using the SAME deterministic episode_seed formula as test8.py (the one
verified to actually honor --seed, unlike test2.py-test7.py which ignore
it and reseed from wall-clock time on every run).

This is a DIFFERENT protocol from evaluate_bdvl.py's --phase
validation/formal (which require both nominal+nonstationary profiles and
BDVL-specific suite/formal seed roles) -- it exists specifically so
Mamba-VL/SARL/LSTM (run through test8.py) and BDVL (run through this
script) can be scored on the EXACT SAME episode set for a fair paper
comparison table. It is NOT a replacement for the formal
heldout_nonstationary run; that remains the real generalization/
robustness result and should be reported separately.

episode_seed formula (bit-identical to test8.py):
    episode_seed = (base_seed + case_id * 1_000_003 + ep) % (2**31 - 1)
case_id follows FORMAL_SCENARIOS' own iteration order, which is already
baseline_circle=0, baseline_square=1, dense_circle=2, dense_square=3,
large_circle=4, large_square=5 -- identical to test8.py's hardcoded list.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_sim.envs.utils.robot import Robot  # noqa: E402

from crowd_nav.bayesian_dvl.config import ActionGridSpec, BDVL_PRODUCTION_SOURCES, FORMAL_SCENARIOS, load_and_validate_registry  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402
from crowd_nav.bayesian_dvl.evaluate import write_episode_records_csv  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402
from crowd_nav.tools.evaluate_bdvl import _make_env, _make_policy, run_episode  # noqa: E402

SCENARIO_ORDER = tuple(FORMAL_SCENARIOS.keys())
assert SCENARIO_ORDER == (
    "baseline_circle", "baseline_square", "dense_circle",
    "dense_square", "large_circle", "large_square",
), f"FORMAL_SCENARIOS order changed, no longer matches test8.py's hardcoded case_id list: {SCENARIO_ORDER}"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-seed", type=int, default=42, help="matches test8.py's --seed default")
    parser.add_argument("--episodes-per-scenario", type=int, default=500)
    parser.add_argument("--decision-budget", choices=["validation", "formal"], default="validation",
                         help="world_samples/iqn_quantiles budget; validation is faster, formal is the full protocol cost")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method", default="bayesian_dvl_risk_neutral")
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIO_ORDER), default=None,
        help="run only a subset of the 6 scenarios (used by the parallel launcher); "
             "case_id in the episode_seed formula always follows SCENARIO_ORDER's fixed "
             "index, not the position within this subset, so seeds stay identical either way",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scenarios_to_run = tuple(args.scenarios) if args.scenarios else SCENARIO_ORDER

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_config_path = _resolve_path(args.env_config)
    registry_path = _resolve_path(args.registry)
    registry = load_and_validate_registry(str(registry_path), env_config_path=str(env_config_path))
    action_table = ActionGridSpec.from_env_config(str(env_config_path)).build_action_table()
    artifact = SBKHMMArtifact.load(str(_resolve_path(args.artifact_path)), expect_tier="production")
    checkpoint_path = str(_resolve_path(args.checkpoint))

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(output_path) + ".progress.jsonl")
    if progress_path.exists():
        raise SystemExit(f"refusing to append to existing progress file: {progress_path}")

    import json
    import os

    def _append_progress(record) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "scenario": record.scenario, "episode_seed": record.episode_seed,
                "outcome": record.outcome, "steps": record.steps,
            }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    total_episodes = len(scenarios_to_run) * args.episodes_per_scenario
    all_records = []
    start_time = time.time()
    last_draw = [0.0]

    def _draw_bar(scenario_key: str, force: bool = False) -> None:
        completed = len(all_records)
        now = time.time()
        if not force and completed < total_episodes and (now - last_draw[0]) < 2.0:
            return
        last_draw[0] = now
        fraction = completed / total_episodes
        bar_width = 30
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = now - start_time
        eta = (elapsed / completed * (total_episodes - completed)) if completed > 0 else 0.0
        sys.stdout.write(
            f"\r[{bar}] {fraction * 100:5.1f}% ({completed}/{total_episodes}) "
            f"{scenario_key} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m   "
        )
        sys.stdout.flush()

    for scenario_key in scenarios_to_run:
        # case_id is always the scenario's fixed index within the FULL
        # SCENARIO_ORDER, never its position within scenarios_to_run --
        # otherwise running scenarios as separate parallel subprocesses
        # (each given a 1-scenario subset) would compute different
        # episode_seeds than a single-process full run, breaking the
        # test8.py seed-formula match this whole script exists for.
        case_id = SCENARIO_ORDER.index(scenario_key)
        env, env_config_for_robot = _make_env(env_config_path, FORMAL_SCENARIOS[scenario_key]["humans"], scenario_key)
        policy = _make_policy(
            action_table, artifact, checkpoint_path, registry, False, args.base_seed,
            args.decision_budget, device, risk_neutral=True, collect_calibration=False,
        )
        robot = Robot(env_config_for_robot, "robot")
        robot.set_policy(policy); robot.visible = True; robot.time_step = registry["frozen_values"]["dt"]
        policy.time_step = registry["frozen_values"]["dt"]; robot.env = env; env.set_robot(robot)
        scenario_outcomes = {"success": 0, "collision": 0, "timeout": 0}
        for ep in range(args.episodes_per_scenario):
            episode_seed = (args.base_seed + case_id * 1_000_003 + ep) % (2**31 - 1)
            record = run_episode(
                policy, env, args.base_seed, ep, args.method, scenario_key,
                "nominal", registry["frozen_values"]["dt"], None,
                episode_seed=episode_seed,
            )
            all_records.append(record)
            scenario_outcomes[record.outcome] += 1
            _append_progress(record)
            _draw_bar(scenario_key, force=(len(all_records) in (1, total_episodes)))
        n = sum(scenario_outcomes.values())
        print(
            f"\nSCENARIO_DONE {scenario_key} episodes={n} "
            f"sr={scenario_outcomes['success'] / n:.4f} "
            f"cr={scenario_outcomes['collision'] / n:.4f} "
            f"tr={scenario_outcomes['timeout'] / n:.4f} "
            f"outcomes={scenario_outcomes}", flush=True,
        )

    # BDVL's _assert_complete() assumes the suite_seed*100000+ep formula
    # used by validation/formal; this protocol uses a different one, so
    # completeness/uniqueness is checked directly against episode_seed
    # instead of reusing that helper.
    if len(all_records) != total_episodes:
        raise RuntimeError(f"expected {total_episodes} episodes, got {len(all_records)}")
    seen_seeds = {r.episode_seed for r in all_records}
    if len(seen_seeds) != total_episodes:
        raise RuntimeError(f"episode_seed collisions detected: {total_episodes} episodes but only {len(seen_seeds)} distinct seeds")

    write_episode_records_csv(all_records, str(output_path) + ".csv")
    outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
    for r in all_records:
        outcome_counts[r.outcome] += 1
    n = len(all_records)
    manifest = build_run_manifest(
        repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES,
        extra={
            "protocol": "paper_main_nominal_only", "method": args.method,
            "base_seed": args.base_seed, "episodes_per_scenario": args.episodes_per_scenario,
            "decision_budget": args.decision_budget, "scenarios": list(SCENARIO_ORDER),
            "episode_seed_formula": "(base_seed + case_id*1_000_003 + ep) % (2**31-1), matches test8.py",
            "n_episodes": n, "overall_sr": outcome_counts["success"] / n,
            "overall_cr": outcome_counts["collision"] / n, "overall_tr": outcome_counts["timeout"] / n,
            "outcomes": outcome_counts, "registry_sha256": sha256_of_file(str(registry_path)),
            "env_config_sha256": sha256_of_file(str(env_config_path)),
            "artifact_sha256": artifact.content_sha256(),
            "checkpoint_sha256": sha256_of_file(checkpoint_path),
            "csv_sha256": sha256_of_file(str(output_path) + ".csv"),
        },
    )
    atomic_write_json(str(output_path) + ".manifest.json", manifest)
    print(
        f"EVALUATE_BDVL_PAPER_MAIN_OVERALL n_episodes={n} "
        f"sr={outcome_counts['success'] / n:.4f} cr={outcome_counts['collision'] / n:.4f} "
        f"tr={outcome_counts['timeout'] / n:.4f} outcomes={outcome_counts}"
    )
    print(f"EVALUATE_BDVL_PAPER_MAIN_DONE n_episodes={n}")


if __name__ == "__main__":
    main()
