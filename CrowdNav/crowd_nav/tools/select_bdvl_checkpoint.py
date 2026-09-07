#!/usr/bin/env python3
"""Select a BDVL checkpoint using only frozen training-domain validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import ActionGridSpec, BDVL_PRODUCTION_SOURCES, FROZEN_VALUES, load_and_validate_registry  # noqa: E402
from crowd_nav.bayesian_dvl.evaluate import assert_role_allowed_for_checkpoint_selection  # noqa: E402
from crowd_nav.bayesian_dvl.statistics import check_r2_gate  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402
from crowd_nav.tools.evaluate_bdvl import _assert_complete, _make_env, _make_policy, run_episode  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402


def _score_checkpoint(checkpoint_path, artifact, registry, action_table, env_config_path, suite_seeds, profiles, episodes_per_seed, device, decision_rule):
    values = []
    for profile in profiles:
        for suite_seed in suite_seeds:
            env, env_config = _make_env(env_config_path, human_num=5, scenario_key="baseline_circle")
            policy = _make_policy(
                action_table, artifact, str(checkpoint_path), registry, False,
                suite_seed, "validation", device,
                risk_neutral=(decision_rule == "risk_neutral"),
            )
            robot = Robot(env_config, "robot")
            robot.set_policy(policy); robot.visible = True; robot.time_step = FROZEN_VALUES["dt"]; policy.time_step = FROZEN_VALUES["dt"]; robot.env = env; env.set_robot(robot)
            for episode_index in range(episodes_per_seed):
                values.append(run_episode(policy, env, suite_seed, episode_index, "bayesian_dvl", "baseline_circle", profile, FROZEN_VALUES["dt"]))
    n = len(values)
    _assert_complete(values, ["baseline_circle"], profiles, suite_seeds, episodes_per_seed)
    sr = sum(r.outcome == "success" for r in values) / n
    cr = sum(r.outcome == "collision" for r in values) / n
    tr = sum(r.outcome == "timeout" for r in values) / n
    score_bound_violations = sum(r.score_bound_violations for r in values)
    gate = check_r2_gate(sr, cr, tr, score_bound_violations, n)
    return {
        "success_rate": float(sr), "collision_rate": float(cr), "timeout_rate": float(tr), "n": n,
        "score_bound_violations": int(score_bound_violations),
        "mean_negative_alignment_fraction": float(np.mean([r.negative_alignment_fraction for r in values])),
        "mean_max_goal_distance_over_initial": float(np.mean([
            r.max_goal_distance / r.initial_goal_distance if r.initial_goal_distance > 0 else 1.0 for r in values
        ])),
        # R3R-4 fix (2026-08-07, point 3): guide.md 2225 requires path
        # ratio (actual path length / straight-line initial-goal distance,
        # the standard directness metric -- 1.0 is a perfectly straight
        # path) to be recorded per validation and used as the selector's
        # FINAL tiebreak, after SR/TR/CR/alignment all agree.
        "mean_path_ratio": float(np.mean([
            r.path_length / r.initial_goal_distance if r.initial_goal_distance > 0 else 1.0 for r in values
        ])),
        "r2_gate": gate,
    }


# R3-1 fix (2026-08-07, independent audit of bdvl_r2_final_20260807): the
# previous key=(collision_rate, timeout_rate, -success_rate) put absolute
# priority on collision_rate, so it chose a 0%-collision/86.5%-SR
# checkpoint over a 0.5%-collision/99.5%-SR one -- a single extra
# collision out of 200 episodes, not a meaningful safety margin. The R2
# gate (CR<=5%) already enforces the safety floor; among candidates that
# already cleared it, maximize navigation capability first. This is the
# ONE production ranking rule -- selftest imports it directly rather than
# re-deriving an "equivalent" comparison.
# R3R-4 fix (2026-08-07, point 3): path_ratio_asc appended as the FINAL
# tiebreak (after SR/TR/CR/alignment all agree) -- guide.md 3045 "selector
# 末级加入path ratio". A lower path_length/initial_goal_distance means a
# more direct route among otherwise-equal candidates.
SELECTION_KEY = "success_rate_desc,timeout_rate_asc,collision_rate_asc,negative_alignment_asc,path_ratio_asc"


def selection_rank_key(result: dict):
    return (
        -result["success_rate"], result["timeout_rate"],
        result["collision_rate"], result["mean_negative_alignment_fraction"],
        result["mean_path_ratio"],
    )


def select_best_eligible(eligible):
    if not eligible:
        raise ValueError("eligible must be non-empty")
    return min(eligible, key=selection_rank_key)


# R3R-4 fix (2026-08-07, point 4): guide.md 3047 "新增稳定区间选择：按train
# seed和raw/EMA类别分组，只允许来自至少两个连续合格checkpoint的候选；孤立
# 峰值不得晋升." run_bdvl_queue.py's train_one_seed() writes checkpoints at
# base/seed_<seed>/checkpoints/checkpoint_ep<N>[_ema].pth -- that convention
# is the only source of "which seed/category/episode is this" available to
# the selector, so it is parsed rather than re-derived some other way.
#
# Engineering-gap fix (2026-08-08): the original pattern required a path
# COMPONENT equal to exactly "seed_<N>", which broke on a real R3-8
# single-seed validation run launched directly via train_bdvl.py (not
# through run_bdvl_queue.py), whose --checkpoint-dir was named
# "bdvl_r3r8_seed93001_20260807_fix2" -- "seed" and the digits are
# adjacent with no underscore, and not their own path component. The
# selector crashed at its FINAL step, after already spending real GPU
# time validating all 12 checkpoints (guide.md's fail-closed philosophy
# meant it errored loudly rather than silently mis-grouping, but a
# rigid convention this brittle is still a real gap). The pattern below
# searches each path component for "seed" immediately followed by
# digits (underscore optional), still requiring an UNAMBIGUOUS single
# match across the whole path -- multiple conflicting matches still
# fail closed rather than silently picking one.
_CHECKPOINT_FILENAME_PATTERN = re.compile(r"^checkpoint_ep(\d+)(_ema)?\.pth$")
_SEED_TOKEN_PATTERN = re.compile(r"seed_?(\d+)")


def parse_checkpoint_identity(path) -> Tuple[int, str, int]:
    """(seed, category, episode) from a checkpoint path whose filename is
    checkpoint_ep<N>[_ema].pth and which has exactly one path component
    containing a seed<N> or seed_<N> token. Raises ValueError if the
    filename doesn't match, or if zero or more-than-one DISTINCT seed
    values are found -- the stability gate fails closed rather than
    silently guessing."""
    path = Path(path)
    name_match = _CHECKPOINT_FILENAME_PATTERN.match(path.name)
    if not name_match:
        raise ValueError(f"checkpoint filename does not match checkpoint_ep<N>[_ema].pth: {path.name}")
    episode = int(name_match.group(1))
    category = "ema" if name_match.group(2) else "raw"
    found_seeds = set()
    for part in path.parts:
        for seed_match in _SEED_TOKEN_PATTERN.finditer(part):
            found_seeds.add(int(seed_match.group(1)))
    if len(found_seeds) == 0:
        raise ValueError(f"could not find a seed_<N>/seed<N> token in checkpoint path: {path}")
    if len(found_seeds) > 1:
        raise ValueError(f"found multiple conflicting seed tokens {sorted(found_seeds)} in checkpoint path: {path}")
    return found_seeds.pop(), category, episode


def apply_stability_gate(results: List[dict]) -> List[dict]:
    """Keep only R2-gate-passing candidates that also have at least one
    IMMEDIATELY ADJACENT (by episode, within the same train seed and
    raw/EMA category) checkpoint that also passed. An isolated passing
    checkpoint surrounded by failing neighbors is an unstable peak, not a
    reproducible improvement, and must not reach final selection."""
    groups: Dict[Tuple[int, str], List[dict]] = {}
    for result in results:
        seed, category, episode = parse_checkpoint_identity(result["checkpoint"])
        groups.setdefault((seed, category), []).append({"result": result, "episode": episode})
    interval = int(FROZEN_VALUES["stability_checkpoint_interval"])
    max_sr_delta = float(FROZEN_VALUES["stability_success_rate_delta_max"])
    stable_paths = set()
    for group in groups.values():
        group.sort(key=lambda entry: entry["episode"])
        for left, right in zip(group, group[1:]):
            left_result = left["result"]
            right_result = right["result"]
            if right["episode"] - left["episode"] != interval:
                continue
            if not (left_result["r2_gate"]["passed"] and right_result["r2_gate"]["passed"]):
                continue
            if abs(float(left_result["success_rate"]) - float(right_result["success_rate"])) > max_sr_delta:
                continue
            stable_paths.add(left_result["checkpoint"])
            stable_paths.add(right_result["checkpoint"])
    return [result for result in results if result["checkpoint"] in stable_paths]


def _append_progress(path: Path, record: dict) -> None:
    """Persist one candidate result immediately so long selection runs are observable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-role", default="checkpoint-validation")
    parser.add_argument("--suite-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--profiles", nargs="+", choices=["nominal", "train_nonstationary"], default=["nominal", "train_nonstationary"])
    parser.add_argument("--decision-rule", choices=["risk_neutral", "cvar"], default="risk_neutral")
    parser.add_argument("--allow-short-run", action="store_true", help="diagnostic-only selection; not an R2 gate result")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-output", default=None, help="JSONL file updated after every candidate")
    args = parser.parse_args()
    assert_role_allowed_for_checkpoint_selection(args.data_role)
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    env_config_path = PACKAGE_ROOT / args.env_config
    if env_config_path.name != "env_bayesian_dvl.config":
        raise SystemExit("BDVL checkpoint selection must use env_bayesian_dvl.config")
    registry = load_and_validate_registry(str(PACKAGE_ROOT / args.registry), env_config_path=str(env_config_path))
    seeds = args.suite_seeds or list(registry["seed_roles"]["checkpoint_validation_seeds"])
    if set(seeds) & set(registry["seed_roles"]["formal_test_suite_seeds"]):
        raise SystemExit("checkpoint selector cannot read formal-test seeds")
    artifact = SBKHMMArtifact.load(str(PACKAGE_ROOT / args.artifact_path), expect_tier="production")
    if not args.allow_short_run:
        expected_seeds = set(registry["seed_roles"]["checkpoint_validation_seeds"])
        if set(seeds) != expected_seeds or args.episodes_per_seed != 20:
            raise SystemExit(
                "R2 checkpoint selection requires all five checkpoint-validation seeds "
                "and exactly 20 episodes per seed (200 paired episodes total); "
                "use --allow-short-run only for diagnostics"
            )
    action_table = ActionGridSpec.from_env_config(str(env_config_path)).build_action_table()
    selected_path = Path(args.selected_checkpoint) if Path(args.selected_checkpoint).is_absolute() else PACKAGE_ROOT / args.selected_checkpoint
    output_path = Path(args.output) if Path(args.output).is_absolute() else PACKAGE_ROOT / args.output
    progress_path = Path(args.progress_output) if args.progress_output and Path(args.progress_output).is_absolute() else PACKAGE_ROOT / (args.progress_output or (str(args.output) + ".progress.jsonl"))
    if progress_path.exists():
        raise SystemExit(f"refusing to append to existing selector progress file: {progress_path}")
    results = []
    total = len(args.checkpoints)
    for candidate_index, raw_path in enumerate(args.checkpoints, start=1):
        path = Path(raw_path) if Path(raw_path).is_absolute() else PACKAGE_ROOT / raw_path
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")
        print(f"SELECT_PROGRESS START {candidate_index}/{total} checkpoint={path}", flush=True)
        score = _score_checkpoint(path, artifact, registry, action_table, env_config_path, seeds, args.profiles, args.episodes_per_seed, device, args.decision_rule)
        results.append({"checkpoint": str(path), **score})
        progress_record = {
            "candidate_index": candidate_index,
            "candidate_total": total,
            "checkpoint": str(path),
            "suite_seeds": seeds,
            "profiles": args.profiles,
            "episodes_per_seed": args.episodes_per_seed,
            **score,
        }
        _append_progress(progress_path, progress_record)
        print(f"SELECT_PROGRESS DONE {candidate_index}/{total} SR={score['success_rate']:.3f} CR={score['collision_rate']:.3f} TR={score['timeout_rate']:.3f}", flush=True)
    gate_passed = [result for result in results if bool(result["r2_gate"]["passed"])]
    # Diagnostic selection may produce a best-effort handoff even when the
    # tiny run cannot satisfy a formal safety gate.  The result is explicitly
    # marked diagnostic_only below; formal selection still requires the R2
    # gate and the stability interval.
    if args.allow_short_run:
        eligible = results
        stability_gate_applied = False
    else:
        eligible = apply_stability_gate(results)
        stability_gate_applied = True
    if not eligible:
        if not gate_passed:
            reason = "no checkpoint passed the R2 gate"
        else:
            reason = (
                f"{len(gate_passed)} checkpoint(s) passed the R2 gate but none had a passing "
                "consecutive neighbor within the same train seed/category (isolated peak(s) only); "
                "use --allow-short-run for diagnostics or train/evaluate additional nearby checkpoints"
            )
        payload = {
            "data_role": args.data_role,
            "decision_rule": args.decision_rule,
            "suite_seeds": seeds,
            "profiles": args.profiles,
            "episodes_per_seed": args.episodes_per_seed,
            "results": results,
            "selected_checkpoint": None,
            "selected_checkpoint_sha256": None,
            "selected_r2_gate": {"passed": False, "reasons": [reason]},
            "stability_gate_applied": stability_gate_applied,
            "diagnostic_only": bool(args.allow_short_run),
            "gate_passed_count": len(gate_passed),
        }
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite selection report: {output_path}")
        atomic_write_json(str(output_path), payload)
        manifest = build_run_manifest(
            repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv),
            source_files=BDVL_PRODUCTION_SOURCES,
            extra={"registry_sha256": sha256_of_file(str(PACKAGE_ROOT / args.registry)), "artifact_sha256": artifact.content_sha256(), "selected_checkpoint_sha256": None, "selected_r2_gate": payload["selected_r2_gate"], "diagnostic_only": bool(args.allow_short_run)},
        )
        atomic_write_json(str(output_path) + ".manifest.json", manifest)
        print(f"SELECT_BDVL_CHECKPOINT_FAIL {reason}", flush=True)
        raise SystemExit(2)

    best = select_best_eligible(eligible)
    selection_key = SELECTION_KEY
    source_sha256 = sha256_of_file(best["checkpoint"])
    if selected_path.exists():
        raise SystemExit(f"refusing to overwrite selected checkpoint: {selected_path}")
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["checkpoint"], selected_path)
    copied_sha256 = sha256_of_file(str(selected_path))
    if copied_sha256 != source_sha256:
        # guide.md R3-1: "复制后重新计算hash，必须与源hash一致；否则删除
        # 目标并失败" -- a mismatch here means the copy itself is corrupt;
        # never leave a silently-wrong file at the canonical selected path.
        selected_path.unlink()
        raise SystemExit(
            f"selected checkpoint copy hash mismatch: source={source_sha256} copied={copied_sha256}; "
            f"deleted the corrupt copy at {selected_path}"
        )
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite selection report: {output_path}")
    selected_source_metrics = {k: v for k, v in best.items() if k != "checkpoint"}
    payload = {
        "data_role": args.data_role, "decision_rule": args.decision_rule, "suite_seeds": seeds,
        "profiles": args.profiles, "episodes_per_seed": args.episodes_per_seed,
        "results": results, "selected_checkpoint": str(selected_path), "selected_checkpoint_sha256": copied_sha256,
        "selected_r2_gate": best["r2_gate"],
        "selected_source_checkpoint": best["checkpoint"],
        "selected_source_sha256": source_sha256,
        "selected_source_metrics": selected_source_metrics,
        "selection_key": selection_key,
        "eligible_candidate_count": len(eligible),
        "candidate_count": len(results),
        "stability_gate_applied": stability_gate_applied,
        "diagnostic_only": bool(args.allow_short_run),
        "gate_passed_count": len(gate_passed),
    }
    atomic_write_json(str(output_path), payload)
    manifest = build_run_manifest(repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES, extra={"registry_sha256": sha256_of_file(str(PACKAGE_ROOT / args.registry)), "artifact_sha256": artifact.content_sha256(), "selected_checkpoint_sha256": payload["selected_checkpoint_sha256"], "selected_source_checkpoint": best["checkpoint"], "selected_source_sha256": source_sha256, "selection_key": selection_key, "selected_r2_gate": best["r2_gate"], "diagnostic_only": bool(args.allow_short_run)})
    atomic_write_json(str(output_path) + ".manifest.json", manifest)
    # R2-5/R3-1: this print is the one place a human/orchestrator scanning
    # logs sees whether the selected checkpoint actually cleared the
    # minimum engineering gate, which source file it came from, and by
    # what rule it was picked -- guide.md is explicit this gate is NOT
    # enforced automatically by run_bdvl_queue.py to block later stages.
    print(
        f"SELECT_BDVL_CHECKPOINT_DONE selected={selected_path} source={best['checkpoint']} "
        f"selection_key={selection_key} eligible={len(eligible)}/{len(results)} "
        f"SR={best['success_rate']:.3f} CR={best['collision_rate']:.3f} TR={best['timeout_rate']:.3f} "
        f"r2_gate_passed={best['r2_gate']['passed']} reasons={best['r2_gate']['reasons']}"
    )


if __name__ == "__main__":
    main()
