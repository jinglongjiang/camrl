#!/usr/bin/env python3
"""Fail-closed serial BDVL pipeline.

collect(world train/validation) -> fit -> train(each seed) -> select ->
evaluate. Every stage gets its own log and status record; an existing output
root is rejected instead of silently overwriting an experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import SEED_ROLES  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json  # noqa: E402


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, payload: dict) -> None:
    atomic_write_json(str(path), payload)


def _run(cmd, base: Path, status_path: Path, history: list, stage_name: str) -> None:
    # R3R-4 fix (2026-08-07, point 3): status was written as schema_version=2
    # (missing validation_complete/formal_complete) for every RUNNING/PASS/
    # FAIL update and only bumped to schema_version=3 at the very end -- a
    # crash or external read mid-run would see an inconsistent, older
    # schema. schema_version=3 and validation_complete/formal_complete are
    # now present on EVERY status write; the two completion flags are only
    # ever True in the final COMPLETED_REQUESTED_STAGES record written by
    # main(), so False here truthfully means "not yet determined."
    log_path = base / "logs" / f"{len(history):03d}_{stage_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"stage": stage_name, "command": cmd, "log": str(log_path), "started_at": _now(), "status": "RUNNING"}
    history.append(entry)
    _write_status(status_path, {"schema_version": 3, "status": "RUNNING", "current_stage": stage_name, "history": history, "validation_complete": False, "formal_complete": False})
    with log_path.open("w") as handle:
        result = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), stdout=handle, stderr=subprocess.STDOUT)
    entry.update({"finished_at": _now(), "returncode": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL"})
    _write_status(status_path, {"schema_version": 3, "status": entry["status"], "current_stage": stage_name, "history": history, "validation_complete": False, "formal_complete": False})
    if result.returncode != 0:
        raise SystemExit(f"stage {stage_name!r} failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--train-config", default="crowd_nav/configs/train_bayesian_dvl.config")
    # R4-1R-2 fix (2026-08-10, guide.md "R4-1R -- 冻结 V4 registry"): the
    # select and evaluate stages previously either relied on
    # select_bdvl_checkpoint.py's own hardcoded default or a hardcoded
    # literal path here -- two different code paths that could (and did,
    # for R4) silently diverge from whatever train_bdvl.py actually
    # trained against. ONE explicit --registry threaded through every
    # stage this queue launches; no stage may fall back to its own default.
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", default=None, help="existing frozen production artifact; skips refitting when supplied")
    parser.add_argument("--world-train-seeds", type=int, nargs="+", default=SEED_ROLES["world_train_suite_seeds"])
    parser.add_argument("--world-validation-seeds", type=int, nargs="+", default=SEED_ROLES["world_validation_suite_seeds"])
    parser.add_argument("--train-seeds", type=int, nargs="+", default=SEED_ROLES["rl_training_seeds"])
    parser.add_argument("--collect-episodes-per-seed", type=int, default=20)
    parser.add_argument("--il-episodes", type=int, default=5000)
    parser.add_argument("--rl-episodes", type=int, default=3000)
    parser.add_argument("--allow-short-run", action="store_true", help="diagnostic-only short queue; never a formal result")
    parser.add_argument("--profiles", nargs="+", choices=["nominal", "train_nonstationary"], default=["nominal", "train_nonstationary"])
    parser.add_argument("--evaluate-phase", choices=["validation", "formal"], default="validation")
    parser.add_argument("--evaluate-suite-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--evaluate-episodes-per-seed", type=int, default=None)
    parser.add_argument("--select-suite-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--select-episodes-per-seed", type=int, default=None)
    parser.add_argument("--stages", nargs="+", default=["collect", "fit", "train", "select", "evaluate"], choices=["collect", "fit", "train", "select", "evaluate"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    base = PACKAGE_ROOT / args.base
    if base.exists() and any(base.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty queue output root: {base}")
    base.mkdir(parents=True, exist_ok=True)
    status_path = base / "queue_status.json"
    history = []
    py = sys.executable
    train_files = [base / f"world_train_{profile}.json" for profile in args.profiles]
    validation_files = [base / f"world_validation_{profile}.json" for profile in args.profiles]
    artifact_path = base / "production_artifact.json"
    if args.artifact_path:
        source_artifact = Path(args.artifact_path)
        if not source_artifact.is_absolute():
            source_artifact = PACKAGE_ROOT / source_artifact
        if not source_artifact.is_file():
            raise SystemExit(f"frozen production artifact not found: {source_artifact}")
        shutil.copy2(source_artifact, artifact_path)
        source_manifest = Path(str(source_artifact) + ".manifest.json")
        if source_manifest.is_file():
            shutil.copy2(source_manifest, Path(str(artifact_path) + ".manifest.json"))
    if args.artifact_path and "fit" in args.stages:
        raise SystemExit("--artifact-path is a frozen artifact mode; omit 'fit' from --stages")
    checkpoint_paths = []
    train_seed_order = list(args.train_seeds)
    if "train" in args.stages and len(train_seed_order) > 1 and "select" not in args.stages and not args.allow_short_run:
        raise SystemExit("BDVL queue requires the select gate before training more than one RL seed")

    def train_one_seed(seed: int) -> None:
        final_path = base / f"seed_{seed}" / "final.pth"
        checkpoint_dir = base / f"seed_{seed}" / "checkpoints"
        train_cmd = [
            py, "-m", "crowd_nav.tools.train_bdvl", "--env-config", args.env_config,
            "--train-config", args.train_config, "--registry", args.registry,
            "--artifact-path", str(artifact_path),
            "--seed", str(seed), "--profiles", *args.profiles, "--device", args.device,
            "--il-episodes", str(args.il_episodes), "--rl-episodes", str(args.rl_episodes),
            "--checkpoint-dir", str(checkpoint_dir), "--output", str(final_path),
        ]
        if args.allow_short_run:
            train_cmd.append("--allow-short-run")
        _run(train_cmd, base, status_path, history, f"train_seed_{seed}")
        # `final.pth` is a full resume checkpoint, not a selector candidate.
        # It has no checkpoint_ep<N> identity and must never enter the
        # stability gate. The periodic policy-only checkpoints below are
        # the complete selector candidate set.
        checkpoint_paths.extend(sorted(checkpoint_dir.glob("checkpoint_ep*.pth")))

    if "collect" in args.stages:
        for profile, output in zip(args.profiles, train_files):
            _run([py, "-m", "crowd_nav.tools.collect_bdvl_data", "--env-config", args.env_config, "--profile", profile, "--role", "world-train", "--suite-seeds", *map(str, args.world_train_seeds), "--episodes-per-seed", str(args.collect_episodes_per_seed), "--output", str(output)], base, status_path, history, f"collect_train_{profile}")
        for profile, output in zip(args.profiles, validation_files):
            _run([py, "-m", "crowd_nav.tools.collect_bdvl_data", "--env-config", args.env_config, "--profile", profile, "--role", "world-validation", "--suite-seeds", *map(str, args.world_validation_seeds), "--episodes-per-seed", str(args.collect_episodes_per_seed), "--output", str(output)], base, status_path, history, f"collect_validation_{profile}")

    if "fit" in args.stages:
        _run([py, "-m", "crowd_nav.tools.fit_bdvl_world_model", "--data", *map(str, train_files), "--validation-data", *map(str, validation_files), "--output", str(artifact_path)], base, status_path, history, "fit")

    if "train" in args.stages:
        first_batch = train_seed_order[:1] if "select" in args.stages else train_seed_order
        for seed in first_batch:
            train_one_seed(seed)

    selected = base / "selected_model.pth"
    if "select" in args.stages:
        if not checkpoint_paths:
            checkpoint_paths = sorted(base.glob("seed_*/checkpoints/checkpoint_ep*.pth"))
        if not checkpoint_paths:
            raise SystemExit("select requested but no checkpoint candidates exist")
        run_additional_seeds = "train" in args.stages and len(train_seed_order) > 1 and not args.allow_short_run
        primary_selected = base / "selected_primary_model.pth" if run_additional_seeds else selected
        primary_result = base / "selection_primary_result.json" if run_additional_seeds else base / "selection_result.json"
        select_seeds = args.select_suite_seeds
        select_episodes = args.select_episodes_per_seed
        if args.allow_short_run:
            select_seeds = select_seeds or [SEED_ROLES["checkpoint_validation_seeds"][0]]
            select_episodes = select_episodes or 1
        select_cmd = [py, "-m", "crowd_nav.tools.select_bdvl_checkpoint", "--env-config", args.env_config, "--registry", args.registry, "--artifact-path", str(artifact_path), "--checkpoints", *map(str, checkpoint_paths), "--profiles", "nominal", "train_nonstationary", "--decision-rule", "risk_neutral", "--selected-checkpoint", str(primary_selected), "--output", str(primary_result)]
        if args.allow_short_run:
            select_cmd.append("--allow-short-run")
        if select_seeds:
            select_cmd += ["--suite-seeds", *map(str, select_seeds)]
        if select_episodes:
            select_cmd += ["--episodes-per-seed", str(select_episodes)]
        _run(select_cmd, base, status_path, history, "select")

        if run_additional_seeds:
            for seed in train_seed_order[1:]:
                train_one_seed(seed)
            final_select_cmd = [
                py, "-m", "crowd_nav.tools.select_bdvl_checkpoint", "--env-config", args.env_config,
                "--artifact-path", str(artifact_path), "--checkpoints", *map(str, checkpoint_paths),
                "--profiles", "nominal", "train_nonstationary", "--decision-rule", "risk_neutral",
                "--selected-checkpoint", str(selected), "--output", str(base / "selection_result.json"),
            ]
            _run(final_select_cmd, base, status_path, history, "select_all_seeds")

    if "evaluate" in args.stages:
        if not selected.is_file():
            raise SystemExit("evaluate requires selected_model.pth from select stage")
        eval_seeds = args.evaluate_suite_seeds or (SEED_ROLES["checkpoint_validation_seeds"] if args.evaluate_phase == "validation" else SEED_ROLES["formal_test_suite_seeds"])
        eval_episodes = args.evaluate_episodes_per_seed or (20 if args.evaluate_phase == "validation" else 100)
        eval_profiles = ["nominal", "train_nonstationary"] if args.evaluate_phase == "validation" else ["nominal", "heldout_nonstationary"]
        _run([py, "-m", "crowd_nav.tools.evaluate_bdvl", "--env-config", args.env_config, "--registry", args.registry, "--artifact-path", str(artifact_path), "--checkpoint", str(selected), "--phase", args.evaluate_phase, "--suite-seeds", *map(str, eval_seeds), "--episodes-per-seed", str(eval_episodes), "--profiles", *eval_profiles, "--decision-rule", "risk_neutral", "--device", args.device, "--output", str(base / "evaluation" / "result")], base, status_path, history, "evaluate")
    # R3-1 fix (guide.md P8): "COMPLETED_REQUESTED_STAGES" alone let a
    # validation-phase-only run (this queue's default) be read as if the
    # six-scenario formal protocol had finished. Split the two so a
    # validation run can never be mistaken for a formal result.
    evaluate_entry = next((h for h in reversed(history) if h["stage"] in ("evaluate",)), None)
    evaluate_passed = bool(evaluate_entry and evaluate_entry.get("status") == "PASS")
    validation_complete = evaluate_passed and args.evaluate_phase == "validation"
    formal_complete = evaluate_passed and args.evaluate_phase == "formal"
    _write_status(status_path, {
        "schema_version": 3, "status": "COMPLETED_REQUESTED_STAGES", "history": history,
        "completed_stages": [x["stage"] for x in history],
        "validation_complete": validation_complete,
        "formal_complete": formal_complete,
        "evaluate_phase": args.evaluate_phase if evaluate_entry else None,
    })
    print(f"RUN_BDVL_QUEUE_DONE validation_complete={validation_complete} formal_complete={formal_complete}")


if __name__ == "__main__":
    main()
