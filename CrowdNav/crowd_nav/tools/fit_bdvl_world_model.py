#!/usr/bin/env python3
"""Fit an SBK-HMM artifact from collected track data and gate it
through ``promote_to_production`` before saving (guide.md 10 S0, B4).
"""

from __future__ import annotations

import argparse
import json
import sys
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

from crowd_nav.bayesian_dvl.world_model import Track, fit_sbk_hmm, promote_to_production, WorldModelError  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file, sha256_of_obj  # noqa: E402


def _require_balanced_profiles(payloads, expected_profiles, label: str) -> None:
    profiles = {data.get("profile") for data in payloads}
    if profiles != set(expected_profiles):
        raise SystemExit(f"{label} must contain exactly profiles={sorted(expected_profiles)}, got {sorted(profiles)}")
    for profile in expected_profiles:
        count = sum(1 for data in payloads if data.get("profile") == profile for _ in data.get("episodes", []))
        if count <= 0:
            raise SystemExit(f"{label} profile {profile!r} has no episodes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="one or more world-train data files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-data", nargs="+", default=None)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--allow-unconverged", action="store_true", help="save even if EM did not converge (diagnostic only, NOT for production)")
    args = parser.parse_args()

    data_paths = [Path(value) if Path(value).is_absolute() else PACKAGE_ROOT / value for value in args.data]
    train_payloads = [json.loads(path.read_text()) for path in data_paths]
    if any(data.get("schema_version") != 2 or not data.get("episodes") for data in train_payloads):
        raise SystemExit("fit inputs must be schema_version=2 episode-structured data")
    if any(data.get("role") not in {"world-train", "IL-train", "RL-train"} for data in train_payloads):
        raise SystemExit("fit inputs must all be training-role data")
    if not args.allow_unconverged:
        _require_balanced_profiles(train_payloads, ("nominal", "train_nonstationary"), "world-train data")
    dt = train_payloads[0]["dt"]
    if any(abs(float(data["dt"]) - float(dt)) > 1e-12 for data in train_payloads):
        raise SystemExit("mixed dt across fit inputs")
    tracks = [Track(positions=np.array(positions), dt=dt) for data in train_payloads for positions in data["tracks"]]
    train_file_hashes = {str(path): sha256_of_file(str(path)) for path in data_paths}
    # The artifact identity must survive copying the same data between
    # local storage and 4090.  Absolute file names are provenance metadata,
    # not model identity; only the ordered content hashes enter the fit hash.
    train_hash = sha256_of_obj({"role": "world-train", "file_sha256": sorted(train_file_hashes.values())})

    artifact = fit_sbk_hmm(tracks, train_data_sha256=train_hash, max_iterations=args.max_iterations)

    if not args.allow_unconverged and not args.validation_data:
        raise SystemExit("production fit requires --validation-data; use --allow-unconverged only for diagnostics")
    validation_summary = None
    if args.validation_data:
        validation_paths = [Path(value) if Path(value).is_absolute() else PACKAGE_ROOT / value for value in args.validation_data]
        validations = [json.loads(path.read_text()) for path in validation_paths]
        if any(data.get("schema_version") != 2 or data.get("role") != "world-validation" for data in validations):
            raise SystemExit("validation data must be schema_version=2 with role=world-validation")
        if not args.allow_unconverged:
            _require_balanced_profiles(validations, ("nominal", "train_nonstationary"), "world-validation data")
        validation_dt = validations[0]["dt"]
        if any(abs(float(data["dt"]) - float(validation_dt)) > 1e-12 for data in validations):
            raise SystemExit("mixed dt across validation inputs")
        if abs(float(validation_dt) - float(dt)) > 1e-12:
            raise SystemExit("world-train/world-validation dt mismatch")
        train_ids = {
            (episode.get("suite_seed"), episode.get("episode_seed"))
            for data in train_payloads for episode in data.get("episodes", [])
        }
        validation_ids = {
            (episode.get("suite_seed"), episode.get("episode_seed"))
            for data in validations for episode in data.get("episodes", [])
        }
        if not train_ids or not validation_ids or train_ids & validation_ids:
            raise SystemExit("world-train/world-validation episode identities must be non-empty and disjoint")
        validation_tracks = [Track(positions=np.array(pos), dt=data["dt"])
                             for data in validations for episode in data["episodes"] for pos in episode["tracks"]]
        values = [float(np.max(artifact.emission_log_prob(row)))
                  for track in validation_tracks for run in track.feature_runs() for row in run]
        if not values or not np.isfinite(values).all():
            raise SystemExit("validation data produced no finite predictive scores")
        validation_summary = {"n_frames": len(values), "mean_max_log_predictive": float(np.mean(values))}

    if not args.allow_unconverged:
        try:
            promote_to_production(artifact)
        except WorldModelError as exc:
            print(f"FIT_BDVL_WORLD_MODEL_FAIL: {exc}")
            raise SystemExit(1)

    output_path = PACKAGE_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content_hash = artifact.save(str(output_path))

    manifest = build_run_manifest(
        repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv),
        source_files=["crowd_nav/tools/fit_bdvl_world_model.py", "crowd_nav/bayesian_dvl/world_model.py"],
        extra={
            "data_path": [str(path) for path in data_paths], "data_sha256": train_file_hashes, "combined_train_data_sha256": train_hash,
            "artifact_content_sha256": content_hash, "n_iterations": artifact.n_iterations,
            "converged": artifact.converged,
            "tier": artifact.tier,
            "validation_data": [str(path) for path in validation_paths] if args.validation_data else None,
            "validation_summary": validation_summary,
        },
    )
    atomic_write_json(str(output_path) + ".manifest.json", manifest)
    print(f"FIT_BDVL_WORLD_MODEL_DONE converged={artifact.converged} n_iterations={artifact.n_iterations} content_sha256={content_hash}")


if __name__ == "__main__":
    main()
