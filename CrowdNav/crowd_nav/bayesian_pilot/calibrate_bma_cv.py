#!/usr/bin/env python3
"""Fit the CV evidence covariance on the disjoint pilot training split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
TOOLS_DIR = CROWD_NAV_DIR / "tools"
REPO_ROOT = CROWD_NAV_DIR.parent
for path in (str(REPO_ROOT), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fit_behavior_gdbn_modes import (  # noqa: E402
    _cv_covariance,
    _extract_transitions,
    _logpdf_gaussian,
    _logsumexp,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gdbn(params_dir: Path):
    dynamics = np.load(params_dir / "gdbn.npz", allow_pickle=False)
    action = np.load(params_dir / "action_model.npz", allow_pickle=False)
    mode_count = int(np.asarray(dynamics["K"]).item())
    return {
        "K": mode_count,
        "Pi": np.asarray(dynamics["Pi"], dtype=np.float64),
        "A": [
            np.asarray(dynamics[f"A_{index}"], dtype=np.float64)
            for index in range(mode_count)
        ],
        "Q": [
            np.asarray(dynamics[f"Q_{index}"], dtype=np.float64)
            for index in range(mode_count)
        ],
        "B": [
            np.asarray(action[f"B_{index}"], dtype=np.float64)
            for index in range(mode_count)
        ],
    }


def evidence_calibration(data, model, cv_covariance, dt: float):
    cv_matrix = np.eye(4, dtype=np.float64)
    cv_matrix[0, 2] = float(dt)
    cv_matrix[1, 3] = float(dt)
    all_factors = []
    nominal_factors = []
    event_factors = []

    for obs_seq, act_seq, mode_seq in zip(
        data["obs"],
        data["act"],
        data["modes"],
    ):
        for ped_index in range(min(5, mode_seq.shape[1])):
            posterior = np.ones(model["K"], dtype=np.float64) / model["K"]
            start = 9 + 5 * ped_index
            for step in range(min(len(obs_seq) - 1, len(act_seq), len(mode_seq))):
                previous = np.asarray(
                    obs_seq[step, start:start + 4],
                    dtype=np.float64,
                )
                current = np.asarray(
                    obs_seq[step + 1, start:start + 4],
                    dtype=np.float64,
                )
                if np.all(previous == 0.0) or np.all(current == 0.0):
                    posterior.fill(1.0 / model["K"])
                    continue
                action = np.asarray(act_seq[step, :2], dtype=np.float64)
                prior = np.maximum(posterior @ model["Pi"], 1e-12)
                prior /= prior.sum()
                terms = np.asarray(
                    [
                        np.log(prior[index])
                        + _logpdf_gaussian(
                            current,
                            model["A"][index] @ previous
                            + model["B"][index] @ action,
                            model["Q"][index],
                        )
                        for index in range(model["K"])
                    ],
                    dtype=np.float64,
                )
                gdbn_log_likelihood = _logsumexp(terms)
                cv_log_likelihood = _logpdf_gaussian(
                    current,
                    cv_matrix @ previous,
                    cv_covariance,
                )
                factor = float(gdbn_log_likelihood - cv_log_likelihood)
                all_factors.append(factor)
                if int(mode_seq[step, ped_index]) == 0:
                    nominal_factors.append(factor)
                else:
                    event_factors.append(factor)
                posterior = np.exp(terms - gdbn_log_likelihood)
                posterior /= max(float(posterior.sum()), 1e-12)

    nominal = np.asarray(nominal_factors, dtype=np.float64)
    center = float(np.median(nominal))
    median_absolute_deviation = float(
        np.median(np.abs(nominal - center))
    )
    robust_scale = max(1.4826 * median_absolute_deviation, 1e-3)
    return {
        "center": center,
        "scale": robust_scale,
        "all_count": len(all_factors),
        "nominal_count": len(nominal_factors),
        "event_count": len(event_factors),
        "nominal_mean": float(np.mean(nominal)),
        "event_mean": (
            float(np.mean(event_factors))
            if event_factors
            else float("nan")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--max_peds", type=int, default=5)
    parser.add_argument("--params_dir", required=True)
    args = parser.parse_args()

    train_path = Path(args.train).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    data = np.load(train_path, allow_pickle=True)
    rows = _extract_transitions(
        data["obs"],
        data["act"],
        max_peds=args.max_peds,
        dt=args.dt,
    )
    covariance = _cv_covariance(rows, args.dt)
    params_dir = Path(args.params_dir).expanduser().resolve()
    evidence = evidence_calibration(
        data,
        load_gdbn(params_dir),
        covariance,
        args.dt,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        cv_covariance=covariance,
        dt=np.asarray(args.dt),
        transition_count=np.asarray(len(rows)),
        source_sha256=np.asarray(sha256(train_path)),
        log_bayes_factor_center=np.asarray(evidence["center"]),
        log_bayes_factor_scale=np.asarray(evidence["scale"]),
    )
    manifest = {
        "source": str(train_path),
        "source_sha256": sha256(train_path),
        "output": str(output_path),
        "dt": args.dt,
        "max_peds": args.max_peds,
        "params_dir": str(params_dir),
        "transition_count": len(rows),
        "cv_covariance": covariance.tolist(),
        "evidence_calibration": evidence,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
