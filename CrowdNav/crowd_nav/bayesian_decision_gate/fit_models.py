#!/usr/bin/env python3
"""Build the four models compared by the decision gate.

1. ``cv``          -- ``ConstantVelocityRiskModel`` (risk_models.py), no fitting.
2. ``frozen_k3``    -- the paper's already-fitted K=3 GDBN
                       (``runs/bayesian_distributional/gdbn_params_cv_residual_k3``),
                       loaded READ-ONLY. Never refit, never overwritten.
3. ``refit_k{2,3,4}`` -- freshly refit on THIS experiment's own
                       ``train_nonstationary`` split, then one is selected
                       (by NLL, subject to the same eligibility rule as
                       ``bayesian_pilot/evaluate_prediction_gate.py``: minimum
                       per-mode sample count and minimum pairwise dynamics
                       distance) using the disjoint ``validation_nonstationary``
                       split -- never the heldout/nominal test data.

The refitting itself reuses ``tools/fit_behavior_gdbn_modes.py``'s dynamics
fit, transition-matrix fit, k-means clustering, and NLL/ADE evaluation
functions verbatim (they operate on a `rows` list of per-transition dicts and
are agnostic to how many pedestrians a scenario has). The only new code here
is transition EXTRACTION generalized to variable pedestrian counts -- the
original function hardcodes a 34-column (5-pedestrian) observation array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
TOOLS_DIR = THIS_DIR.parent / "tools"
for path in (str(REPO_ROOT), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fit_behavior_gdbn_modes import (  # noqa: E402
    _cv_A,
    _cv_covariance,
    _evaluate,
    _fit_dynamics,
    _fit_transition_matrix,
    _kmeans_pp,
    _matrix_diag,
    _save_params,
    _standardize,
)
from crowd_nav.gdbn import GDBNIntegration  # noqa: E402
from crowd_nav.risk_models import ConstantVelocityRiskModel  # noqa: E402

FROZEN_K3_PARAMS_DIR = "runs/bayesian_distributional/gdbn_params_cv_residual_k3"
MIN_MODE_COUNT = 100
MIN_MODE_DISTANCE = 0.08


def _angle_wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _safe_heading(vx, vy):
    speed = np.hypot(vx, vy)
    return np.where(speed > 1e-4, np.arctan2(vy, vx), 0.0)


def extract_transitions_variable(dataset_paths: List[Path], dt: float) -> List[Dict]:
    """Generalizes ``fit_behavior_gdbn_modes._extract_transitions`` to
    variable pedestrian counts (``obs`` width ``9 + 5 * num_humans`` per
    episode, not a fixed 34)."""
    rows = []
    seq_counter = 0
    for path in dataset_paths:
        data = np.load(path, allow_pickle=True)
        obs_list = data["obs"]
        act_list = data["act"]
        num_humans_list = data["num_humans"]
        for obs_seq, act_seq, num_humans in zip(obs_list, act_list, num_humans_list):
            num_humans = int(num_humans)
            expected_width = 9 + 5 * num_humans
            if not (
                isinstance(obs_seq, np.ndarray)
                and obs_seq.ndim == 2
                and obs_seq.shape[1] == expected_width
            ):
                continue
            T = len(obs_seq)
            robot = obs_seq[:, :9]
            for p in range(num_humans):
                s = 9 + p * 5
                ped = obs_seq[:, s:s + 5]
                valid_idx = np.where(~np.all(ped == 0, axis=1))[0]
                if len(valid_idx) < 2:
                    continue
                for ii in range(len(valid_idx) - 1):
                    t, t1 = int(valid_idx[ii]), int(valid_idx[ii + 1])
                    if t >= T - 1 or t1 >= T or t1 - t > 2:
                        continue
                    step_dt = max(float(dt) * (t1 - t), 1e-6)
                    x = ped[t, :4].astype(np.float64)
                    y = ped[t1, :4].astype(np.float64)
                    action = (
                        act_seq[t, :2].astype(np.float64)
                        if t < len(act_seq)
                        else np.zeros(2)
                    )
                    r = robot[t]
                    rel = x[:2] - r[:2]
                    rel_v = x[2:4] - r[2:4]
                    dist = float(np.hypot(rel[0], rel[1]) + 1e-6)
                    speed = float(np.hypot(x[2], x[3]))
                    next_speed = float(np.hypot(y[2], y[3]))
                    acc = (y[2:4] - x[2:4]) / step_dt
                    cv_pos = x[:2] + x[2:4] * step_dt
                    cv_res = (y[:2] - cv_pos) / step_dt
                    heading = float(_safe_heading(x[2], x[3]))
                    next_heading = float(_safe_heading(y[2], y[3]))
                    d_heading = float(_angle_wrap(next_heading - heading) / step_dt)
                    unit = x[2:4] / max(speed, 1e-4)
                    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
                    cv_long = float(np.dot(cv_res, unit))
                    cv_lat = float(np.dot(cv_res, normal))
                    acc_long = float(np.dot(acc, unit))
                    acc_lat = float(np.dot(acc, normal))
                    closing = float(-np.dot(rel, rel_v) / dist)
                    crossing = float((rel[0] * rel_v[1] - rel[1] * rel_v[0]) / dist)
                    static_feat = np.array(
                        [rel[0], rel[1], dist, rel_v[0], rel_v[1]], dtype=np.float64
                    )
                    behavior_feat = np.array(
                        [
                            speed,
                            (next_speed - speed) / step_dt,
                            acc_long,
                            acc_lat,
                            d_heading,
                            cv_long,
                            cv_lat,
                        ],
                        dtype=np.float64,
                    )
                    rows.append(
                        {
                            "seq": seq_counter,
                            "ped": p,
                            "t": t,
                            "x": x,
                            "y": y,
                            "action": action,
                            "static": static_feat,
                            "behavior": behavior_feat,
                        }
                    )
            seq_counter += 1
    if not rows:
        raise ValueError(f"no valid transitions found in {dataset_paths}")
    return rows


def fit_one_k(rows: List[Dict], K: int, dt: float, seed: int, ridge: float, cv_prior: float):
    behavior = np.asarray([r["behavior"] for r in rows], dtype=np.float64)
    features, _, _ = _standardize(behavior)
    labels, _ = _kmeans_pp(features, K, seed=seed)
    A_list, B_list, Q_list, C_list = _fit_dynamics(rows, labels, K, dt, ridge, cv_prior)
    Pi = _fit_transition_matrix(rows, labels, K)
    return labels, A_list, B_list, Q_list, C_list, Pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="runs/bayesian_decision_gate/data")
    parser.add_argument("--output_dir", default="runs/bayesian_decision_gate/models")
    parser.add_argument("--Ks", default="2,3,4")
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cv_prior", type=float, default=10.0)
    parser.add_argument("--frozen_k3_params", default=FROZEN_K3_PARAMS_DIR)
    parser.add_argument("--max_peds_runtime", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = sorted(data_dir.glob("train_nonstationary_*.npz"))
    validation_paths = sorted(data_dir.glob("validation_nonstationary_*.npz"))
    if not train_paths:
        raise SystemExit(f"[FIT] no train_nonstationary_*.npz found under {data_dir}")
    if not validation_paths:
        raise SystemExit(f"[FIT] no validation_nonstationary_*.npz found under {data_dir}")

    train_rows = extract_transitions_variable(train_paths, args.dt)
    validation_rows = extract_transitions_variable(validation_paths, args.dt)
    print(f"[FIT] train transitions: {len(train_rows)}, validation transitions: {len(validation_rows)}")

    cv_cov_train = _cv_covariance(train_rows, args.dt)
    candidates = []
    for K in [int(v) for v in args.Ks.split(",") if v.strip()]:
        labels, A_list, B_list, Q_list, C_list, Pi = fit_one_k(
            train_rows, K, args.dt, args.seed, args.ridge, args.cv_prior
        )
        metrics = _evaluate(validation_rows, A_list, B_list, Q_list, Pi, args.dt, cv_cov_train)
        diagnostics = _matrix_diag(A_list, labels, Pi)
        distances = [item["fro"] for item in diagnostics["pairwise_A_fro"]]
        eligible = (
            min(diagnostics["mode_counts"], default=0) >= MIN_MODE_COUNT
            and min(distances, default=0.0) >= MIN_MODE_DISTANCE
        )
        candidates.append(
            {
                "K": K,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "eligible": eligible,
                "model": (labels, A_list, B_list, Q_list, C_list, Pi),
            }
        )
        print(
            f"[FIT] K={K} eligible={eligible} validation_nll={metrics['gdbn_nll']:.4f} "
            f"(cv_nll={metrics['cv_nll']:.4f}) min_mode_count={min(diagnostics['mode_counts'], default=0)} "
            f"min_A_distance={min(distances, default=0.0):.4f}"
        )

    eligible_candidates = [c for c in candidates if c["eligible"]]
    pool = eligible_candidates if eligible_candidates else candidates
    selected = min(pool, key=lambda c: c["metrics"]["gdbn_nll"])
    selected_k = int(selected["K"])
    print(f"[FIT] selected K={selected_k} (eligible_pool={bool(eligible_candidates)})")

    # Refit the selected K on train+validation combined for the final model
    # (matches evaluate_prediction_gate.py's "refit on all training data"
    # pattern) -- heldout/nominal are never touched here.
    final_rows = train_rows + validation_rows
    _, A_final, B_final, Q_final, C_final, Pi_final = fit_one_k(
        final_rows, selected_k, args.dt, args.seed, args.ridge, args.cv_prior
    )
    behavior_final = np.asarray([r["behavior"] for r in final_rows], dtype=np.float64)
    features_final, _, _ = _standardize(behavior_final)
    final_labels, _ = _kmeans_pp(features_final, selected_k, seed=args.seed)

    refit_dir = output_dir / f"refit_k{selected_k}"
    _save_params(refit_dir, selected_k, final_labels, final_rows, A_final, B_final, Q_final, C_final, Pi_final)

    summary = {
        "selected_K": selected_k,
        "candidates": [
            {
                "K": c["K"],
                "eligible": c["eligible"],
                "metrics": c["metrics"],
                "diagnostics": c["diagnostics"],
            }
            for c in candidates
        ],
        "train_transitions": len(train_rows),
        "validation_transitions": len(validation_rows),
        "refit_dir": str(refit_dir),
        "frozen_k3_params": args.frozen_k3_params,
    }
    (output_dir / "fit_models_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    # Sanity-load every model exactly the way evaluate_action_ranking.py will.
    frozen = GDBNIntegration(
        K=3, params_dir=args.frozen_k3_params, max_peds=args.max_peds_runtime, random_seed=args.seed
    )
    refit = GDBNIntegration(
        K=selected_k, params_dir=str(refit_dir), max_peds=args.max_peds_runtime, random_seed=args.seed
    )
    cv_model = ConstantVelocityRiskModel(max_peds=args.max_peds_runtime, modeled_peds=args.max_peds_runtime)
    assert frozen.is_fitted, "frozen K3 GDBN failed to load"
    assert refit.is_fitted, "refit GDBN failed to load"
    assert cv_model.is_fitted
    print("[FIT] all three model families load successfully via GDBNIntegration/ConstantVelocityRiskModel")
    print(f"[FIT] wrote {output_dir / 'fit_models_summary.json'}")


if __name__ == "__main__":
    main()
