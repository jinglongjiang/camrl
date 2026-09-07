#!/usr/bin/env python3
"""BDVL A13 acceptance (guide.md R3R-6 point 2): run the REAL
train_bdvl.py CLI through direct and split/resume paths with the same seed,
then compare canonical checkpoint content. The PyTorch archive byte stream
is not used as the equality criterion because archive metadata/object
memoization can differ while every tensor, replay item, optimizer state and
RNG state is identical. Logs/manifests persist under crowd_nav/runs/.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import FROZEN_VALUES  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import Track, fit_sbk_hmm, promote_to_production  # noqa: E402
from crowd_nav.tools.audit_bdvl_a12_resume_equivalence import _canonical_bytes  # noqa: E402

import numpy as np  # noqa: E402

RUN_ROOT = PACKAGE_ROOT / "crowd_nav" / "runs" / "bayesian_dvl" / "a13_e2e_determinism"
SEED = 81001


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_content_sha256(path: Path) -> str:
    payload = __import__("torch").load(path, map_location="cpu", weights_only=False)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _build_and_save_fixture_artifact(path: Path) -> None:
    dt = FROZEN_VALUES["dt"]

    def track(speed0, accel, omega):
        pos = [np.array([0.0, 0.0])]
        speed, heading = speed0, 0.0
        for _ in range(30):
            speed = max(speed + accel * dt, 0.0)
            heading += omega * dt
            vel = speed * np.array([np.cos(heading), np.sin(heading)])
            pos.append(pos[-1] + vel * dt)
        return Track(positions=np.array(pos), dt=dt)

    tracks = [track(1.0, 0.0, 0.0), track(0.3, 0.8, 0.0), track(1.5, -0.8, 0.0), track(1.0, 0.0, 1.0), track(1.0, 0.0, -1.0)]
    artifact = promote_to_production(fit_sbk_hmm(tracks, train_data_sha256="a13_e2e_determinism_fixture", max_iterations=15))
    if artifact.tier != "production":
        raise AssertionError(f"A13 fixture promotion did not produce production tier: {artifact.tier!r}")
    artifact.save(str(path))


def _run_train(run_dir: Path, artifact_path: Path, rl_episodes: int, resume: Path | None = None, stop_after: int | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "final.pth"
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "train.log"
    cmd = [
        sys.executable, "-m", "crowd_nav.tools.train_bdvl",
        "--artifact-path", str(artifact_path),
        "--il-episodes", "2", "--rl-episodes", str(rl_episodes), "--allow-short-run",
        "--seed", str(SEED), "--profiles", "nominal", "--device", "cpu",
        "--checkpoint-dir", str(checkpoint_dir), "--output", str(output_path),
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    if stop_after is not None:
        cmd += ["--stop-after-rl-episodes", str(stop_after)]
    with log_path.open("w") as handle:
        result = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"train_bdvl.py subprocess failed (see {log_path}); returncode={result.returncode}")
    return output_path


def _run_selector_and_evaluator(run_dir: Path, artifact_path: Path, checkpoint_path: Path) -> None:
    """Exercise the actual selector -> evaluator production handoff."""
    candidate_dir = run_dir / "seed_81001" / "checkpoints"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate = candidate_dir / "checkpoint_ep500_ema.pth"
    candidate.write_bytes((run_dir / "final_ema.pth").read_bytes())
    selected = run_dir / "selected_model.pth"
    selection = run_dir / "selection_result.json"
    selector_cmd = [
        sys.executable, "-m", "crowd_nav.tools.select_bdvl_checkpoint",
        "--env-config", "crowd_nav/configs/env_bayesian_dvl.config",
        "--artifact-path", str(artifact_path), "--checkpoints", str(candidate),
        "--allow-short-run", "--suite-seeds", "94001", "--episodes-per-seed", "1",
        "--profiles", "nominal", "train_nonstationary", "--device", "cpu",
        "--selected-checkpoint", str(selected), "--output", str(selection),
    ]
    selector_log = run_dir / "selector.log"
    with selector_log.open("w") as handle:
        result = subprocess.run(selector_cmd, cwd=str(PACKAGE_ROOT), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"selector subprocess failed (see {selector_log})")
    eval_prefix = run_dir / "evaluation" / "result"
    eval_cmd = [
        sys.executable, "-m", "crowd_nav.tools.evaluate_bdvl",
        "--env-config", "crowd_nav/configs/env_bayesian_dvl.config",
        "--registry", "crowd_nav/configs/bayesian_dvl_registry_r4.json",
        "--artifact-path", str(artifact_path), "--checkpoint", str(selected),
        "--phase", "validation", "--suite-seeds", "94001", "--episodes-per-seed", "1",
        "--profiles", "nominal", "train_nonstationary", "--scenarios", "baseline_circle",
        "--device", "cpu", "--output", str(eval_prefix),
    ]
    eval_log = run_dir / "evaluator.log"
    with eval_log.open("w") as handle:
        result = subprocess.run(eval_cmd, cwd=str(PACKAGE_ROOT), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"evaluator subprocess failed (see {eval_log})")


if __name__ == "__main__":
    # The directory is owned exclusively by this audit.  Start clean so
    # selector/evaluator fail-closed output files from an interrupted prior
    # run cannot contaminate the next acceptance result.
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path = RUN_ROOT / "fixture_artifact.json"
    # This audit owns its fixture directory.  Always regenerate the fixture
    # so a previous interrupted run cannot leave an engineering-only artifact
    # that makes the production selector fail for the wrong reason.
    _build_and_save_fixture_artifact(artifact_path)

    output_a = _run_train(RUN_ROOT / "run_a", artifact_path, rl_episodes=4)
    split_output = _run_train(RUN_ROOT / "run_b", artifact_path, rl_episodes=4, stop_after=2)
    output_c = _run_train(RUN_ROOT / "run_c", artifact_path, rl_episodes=4, resume=split_output)
    _run_selector_and_evaluator(RUN_ROOT / "run_a", artifact_path, output_a)

    sha_a, sha_c = _sha256_file(output_a), _sha256_file(output_c)
    content_sha_a, content_sha_c = _checkpoint_content_sha256(output_a), _checkpoint_content_sha256(output_c)
    manifest_a = json.loads(Path(str(output_a) + ".manifest.json").read_text())
    manifest_c = json.loads(Path(str(output_c) + ".manifest.json").read_text())
    # command/timestamps legitimately differ across the two invocations
    # (different output paths, different wall-clock start times); the
    # TRAINING RESULT fields (losses, sizes, hashes, schema) must not.
    manifest_split = json.loads(Path(str(split_output) + ".manifest.json").read_text())
    assert manifest_a["il_losses"] == manifest_split["il_losses"], "resume path changed the shared IL prefix"
    combined_rl = manifest_split["rl_losses"] + manifest_c["rl_losses"]
    assert manifest_a["rl_losses"] == combined_rl, "resume path changed the full RL loss trajectory"
    for key in ("demo_size", "online_size", "artifact_sha256", "world_train_data_sha256",
                "feature_schema", "reward_schema", "return_bounds"):
        va, vc = manifest_a[key], manifest_c[key]
        assert va == vc, f"manifest field {key!r} diverged between direct/resume runs: {va!r} != {vc!r}"

    summary = {
        "seed": SEED, "checkpoint_file_sha256_match": sha_a == sha_c,
        "checkpoint_file_sha256_direct": sha_a, "checkpoint_file_sha256_resumed": sha_c,
        "checkpoint_content_sha256_match": content_sha_a == content_sha_c,
        "checkpoint_content_sha256_direct": content_sha_a, "checkpoint_content_sha256_resumed": content_sha_c,
        "run_direct": str(output_a), "run_split": str(split_output), "run_resumed": str(output_c),
        "coverage": "CLI direct 4 episodes vs CLI 2 episodes + final.pth resume to 4 episodes",
    }
    (RUN_ROOT / "determinism_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    assert content_sha_a == content_sha_c, f"final checkpoint content diverged between direct/resume CLI runs: {content_sha_a} != {content_sha_c}"
    print(f"A13_E2E_DETERMINISM_PASS checkpoint_content_sha256={content_sha_a} file_sha_equal={sha_a == sha_c} run_root={RUN_ROOT}")
