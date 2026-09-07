#!/usr/bin/env python3
"""Held-out prediction validation for CV and GDBN pedestrian predictors.

This script deliberately stays below the navigation layer.  It answers whether
the GDBN motion model itself improves over a constant-velocity predictor before
we spend more time wiring it into value lookahead or safety governance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fit_behavior_gdbn_modes import (  # noqa: E402
    _cv_A,
    _cv_covariance,
    _extract_transitions,
    _fit_dynamics,
    _fit_transition_matrix,
    _kmeans_pp,
    _load_demo,
    _logpdf_gaussian,
    _logsumexp,
    _standardize,
)


def _clearance_risk(clearance: float, safe_distance: float) -> float:
    margin = max(float(safe_distance), 1e-6)
    if clearance <= 0.0:
        return 1.0
    return float(np.clip(math.exp(-clearance / margin), 0.0, 1.0))


def _ece(probabilities, labels, n_bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.size == 0:
        return float('nan')
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(len(p))
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        out += (float(mask.sum()) / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out)


def _brier(probabilities, labels) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2)) if p.size else float('nan')


def _mean(values):
    return float(np.mean(values)) if values else float('nan')


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values else float('nan')


def _fit_labels(
    rows,
    feature_key: str,
    K: int,
    seed: int,
    kmeans_restarts: int,
    kmeans_max_iter: int,
) -> np.ndarray:
    if K == 1:
        return np.zeros(len(rows), dtype=np.int64)
    features = np.asarray([r[feature_key] for r in rows], dtype=np.float64)
    features, _, _ = _standardize(features)
    labels, _ = _kmeans_pp(
        features,
        K,
        seed=seed,
        restarts=int(kmeans_restarts),
        max_iter=int(kmeans_max_iter),
    )
    return labels.astype(np.int64)


def _fit_model(
    rows,
    feature_key: str,
    K: int,
    dt: float,
    seed: int,
    ridge: float,
    kmeans_restarts: int,
    kmeans_max_iter: int,
):
    labels = _fit_labels(rows, feature_key, K, seed, kmeans_restarts, kmeans_max_iter)
    A_list, B_list, Q_list, C_list = _fit_dynamics(rows, labels, K, dt, ridge)
    Pi = _fit_transition_matrix(rows, labels, K)
    return labels, A_list, B_list, Q_list, C_list, Pi


def _mode_diagnostics(A_list, labels, Pi):
    distances = []
    values = []
    for i in range(len(A_list)):
        for j in range(i + 1, len(A_list)):
            d = float(np.linalg.norm(A_list[i] - A_list[j]))
            values.append(d)
            distances.append({'i': int(i), 'j': int(j), 'fro': d})
    counts = np.bincount(labels, minlength=len(A_list)).astype(int)
    row_entropy = -np.sum(Pi * np.log(np.maximum(Pi, 1e-10)), axis=1)
    return {
        'mode_counts': counts.tolist(),
        'mode_fraction': (counts / max(1, counts.sum())).round(6).tolist(),
        'pairwise_A_fro': distances,
        'min_A_fro': float(np.min(values)) if values else 0.0,
        'mean_A_fro': float(np.mean(values)) if values else 0.0,
        'max_A_fro': float(np.max(values)) if values else 0.0,
        'Pi_row_entropy': float(np.mean(row_entropy)),
    }


def _track_rows(rows):
    by_track = {}
    for row in rows:
        by_track.setdefault((int(row['seq']), int(row['ped'])), []).append(row)
    return {key: sorted(items, key=lambda r: int(r['t'])) for key, items in by_track.items()}


def _evaluate_cv(rows, dt: float, cv_cov: np.ndarray, safe_distance: float, horizon: int):
    cvA = _cv_A(dt)
    pos = []
    vel = []
    nll = []
    h_ade = []
    h_fde = []
    unsafe_prob = []
    unsafe_label = []
    collision_prob = []
    collision_label = []

    for items in _track_rows(rows).values():
        for idx, row in enumerate(items):
            x = row['x']
            y = row['y']
            pred = cvA @ x
            pos.append(float(np.linalg.norm(pred[:2] - y[:2])))
            vel.append(float(np.linalg.norm(pred[2:4] - y[2:4])))
            nll.append(-_logpdf_gaussian(y, pred, cv_cov))

            rel = row['static'][:2]
            robot_xy = x[:2] - rel
            robot_future = robot_xy + row['action'][:2] * float(dt)
            clearance_pred = float(np.linalg.norm(pred[:2] - robot_future) - 0.6)
            clearance_true = float(np.linalg.norm(y[:2] - robot_future) - 0.6)
            unsafe_prob.append(_clearance_risk(clearance_pred, safe_distance))
            unsafe_label.append(float(clearance_true <= safe_distance))
            collision_prob.append(_clearance_risk(clearance_pred, 1e-3))
            collision_label.append(float(clearance_true <= 0.0))

            max_h = min(int(horizon), len(items) - idx)
            if max_h >= 2:
                state = y.copy()
                errs = []
                for h in range(1, max_h):
                    fut_y = items[idx + h]['y']
                    state = cvA @ state
                    errs.append(float(np.linalg.norm(state[:2] - fut_y[:2])))
                if errs:
                    h_ade.append(float(np.mean(errs)))
                    h_fde.append(float(errs[-1]))

    return {
        'pairs': int(len(pos)),
        'pos_ade': _mean(pos),
        'vel_rmse': _rmse(vel),
        'nll': _mean(nll),
        f'h{horizon}_ade': _mean(h_ade),
        f'h{horizon}_fde': _mean(h_fde),
        'unsafe_event_rate': _mean(unsafe_label),
        'unsafe_brier': _brier(unsafe_prob, unsafe_label),
        'unsafe_ece': _ece(unsafe_prob, unsafe_label),
        'collision_event_rate': _mean(collision_label),
        'collision_brier': _brier(collision_prob, collision_label),
        'collision_ece': _ece(collision_prob, collision_label),
        'pred_unsafe_mean': _mean(unsafe_prob),
    }


def _evaluate_model(rows, A_list, B_list, Q_list, Pi, dt: float, safe_distance: float, horizon: int):
    K = len(A_list)
    pos = []
    vel = []
    nll = []
    h_ade = []
    h_fde = []
    unsafe_prob = []
    unsafe_label = []
    collision_prob = []
    collision_label = []

    for items in _track_rows(rows).values():
        pi = np.ones(K, dtype=np.float64) / K
        for idx, row in enumerate(items):
            x = row['x']
            y = row['y']
            u = row['action']
            prior = pi @ Pi
            prior = np.maximum(prior, 1e-10)
            prior /= prior.sum()
            pred_modes = np.stack([A_list[k] @ x + B_list[k] @ u for k in range(K)], axis=0)
            pred = (prior[:, None] * pred_modes).sum(axis=0)
            pos.append(float(np.linalg.norm(pred[:2] - y[:2])))
            vel.append(float(np.linalg.norm(pred[2:4] - y[2:4])))

            log_terms = [
                math.log(float(prior[k]) + 1e-12) + _logpdf_gaussian(y, pred_modes[k], Q_list[k])
                for k in range(K)
            ]
            log_norm = _logsumexp(log_terms)
            nll.append(-log_norm)
            posterior = np.exp(np.asarray(log_terms) - log_norm)
            posterior = np.maximum(posterior, 1e-10)
            pi = posterior / posterior.sum()

            rel = row['static'][:2]
            robot_xy = x[:2] - rel
            robot_future = robot_xy + u[:2] * float(dt)
            clearances = np.linalg.norm(pred_modes[:, :2] - robot_future[None, :], axis=1) - 0.6
            unsafe_by_mode = np.asarray([
                _clearance_risk(float(c), safe_distance) for c in clearances
            ], dtype=np.float64)
            collision_by_mode = np.asarray([
                _clearance_risk(float(c), 1e-3) for c in clearances
            ], dtype=np.float64)
            clearance_true = float(np.linalg.norm(y[:2] - robot_future) - 0.6)
            unsafe_prob.append(float(np.sum(prior * unsafe_by_mode)))
            unsafe_label.append(float(clearance_true <= safe_distance))
            collision_prob.append(float(np.sum(prior * collision_by_mode)))
            collision_label.append(float(clearance_true <= 0.0))

            max_h = min(int(horizon), len(items) - idx)
            if max_h >= 2:
                mode_dist = pi.copy()
                state = y.copy()
                errs = []
                for h in range(1, max_h):
                    fut = items[idx + h]
                    alpha = mode_dist @ Pi
                    alpha = np.maximum(alpha, 1e-10)
                    alpha /= alpha.sum()
                    fut_modes = np.stack(
                        [A_list[k] @ state + B_list[k] @ fut['action'] for k in range(K)],
                        axis=0,
                    )
                    state = (alpha[:, None] * fut_modes).sum(axis=0)
                    errs.append(float(np.linalg.norm(state[:2] - fut['y'][:2])))
                    mode_dist = alpha
                if errs:
                    h_ade.append(float(np.mean(errs)))
                    h_fde.append(float(errs[-1]))

    return {
        'pairs': int(len(pos)),
        'pos_ade': _mean(pos),
        'vel_rmse': _rmse(vel),
        'nll': _mean(nll),
        f'h{horizon}_ade': _mean(h_ade),
        f'h{horizon}_fde': _mean(h_fde),
        'unsafe_event_rate': _mean(unsafe_label),
        'unsafe_brier': _brier(unsafe_prob, unsafe_label),
        'unsafe_ece': _ece(unsafe_prob, unsafe_label),
        'collision_event_rate': _mean(collision_label),
        'collision_brier': _brier(collision_prob, collision_label),
        'collision_ece': _ece(collision_prob, collision_label),
        'pred_unsafe_mean': _mean(unsafe_prob),
    }


def _split_by_sequence(rows, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    seqs = np.unique([int(r['seq']) for r in rows])
    rng.shuffle(seqs)
    split = max(1, min(len(seqs) - 1, int(round(len(seqs) * float(train_ratio)))))
    train = set(int(x) for x in seqs[:split])
    train_rows = [r for r in rows if int(r['seq']) in train]
    valid_rows = [r for r in rows if int(r['seq']) not in train]
    return train_rows, valid_rows


def _flatten_row(method: str, feature_mode: str, K: int, metrics: dict, diag: dict | None = None):
    row = {
        'method': method,
        'feature_mode': feature_mode,
        'K': K,
    }
    for key, value in metrics.items():
        row[key] = value
    if diag:
        row['min_A_fro'] = diag.get('min_A_fro', 0.0)
        row['mean_A_fro'] = diag.get('mean_A_fro', 0.0)
        row['max_A_fro'] = diag.get('max_A_fro', 0.0)
        row['Pi_row_entropy'] = diag.get('Pi_row_entropy', 0.0)
        row['mode_counts'] = json.dumps(diag.get('mode_counts', []), separators=(',', ':'))
    else:
        row['min_A_fro'] = 0.0
        row['mean_A_fro'] = 0.0
        row['max_A_fro'] = 0.0
        row['Pi_row_entropy'] = 0.0
        row['mode_counts'] = '[]'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', default='../orca_demos_seq.npz')
    parser.add_argument('--output_dir', default='runs/prediction_validation')
    parser.add_argument('--max_peds', type=int, default=5)
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument('--safe_distance', type=float, default=0.2)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--Ks', default='1,2,3,4')
    parser.add_argument('--feature_modes', default='behavior')
    parser.add_argument('--kmeans_restarts', type=int, default=4)
    parser.add_argument('--kmeans_max_iter', type=int, default=60)
    args = parser.parse_args()

    demo = Path(args.demo).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    obs, act = _load_demo(demo)
    rows = _extract_transitions(obs, act, max_peds=args.max_peds, dt=args.dt)
    train_rows, valid_rows = _split_by_sequence(rows, args.train_ratio, args.seed)
    cv_cov = _cv_covariance(train_rows, args.dt)

    feature_modes = [x.strip() for x in args.feature_modes.split(',') if x.strip()]
    Ks = [int(x.strip()) for x in args.Ks.split(',') if x.strip()]

    summary = {
        'demo': str(demo),
        'n_rows': len(rows),
        'train_rows': len(train_rows),
        'valid_rows': len(valid_rows),
        'seed': int(args.seed),
        'train_ratio': float(args.train_ratio),
        'dt': float(args.dt),
        'safe_distance': float(args.safe_distance),
        'horizon': int(args.horizon),
        'results': [],
    }

    cv_metrics = _evaluate_cv(valid_rows, args.dt, cv_cov, args.safe_distance, args.horizon)
    summary['results'].append(_flatten_row('CV', 'constant_velocity', 1, cv_metrics))

    for feature_mode in feature_modes:
        if feature_mode not in ('static', 'behavior'):
            raise ValueError(f'unknown feature mode: {feature_mode}')
        feature_key = 'static' if feature_mode == 'static' else 'behavior'
        for K in Ks:
            labels, A_list, B_list, Q_list, _C_list, Pi = _fit_model(
                train_rows,
                feature_key,
                K,
                args.dt,
                args.seed,
                args.ridge,
                args.kmeans_restarts,
                args.kmeans_max_iter,
            )
            metrics = _evaluate_model(
                valid_rows, A_list, B_list, Q_list, Pi, args.dt,
                args.safe_distance, args.horizon,
            )
            diag = _mode_diagnostics(A_list, labels, Pi)
            summary['results'].append(
                _flatten_row(f'GDBN-{feature_mode}-K{K}', feature_mode, K, metrics, diag)
            )

    json_path = output / 'gdbn_prediction_validation.json'
    csv_path = output / 'gdbn_prediction_validation.csv'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    fieldnames = list(summary['results'][0].keys())
    for row in summary['results'][1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary['results'])

    print(json.dumps(summary, indent=2))
    print(f"[PRED-VALIDATION] wrote {json_path}")
    print(f"[PRED-VALIDATION] wrote {csv_path}")


if __name__ == '__main__':
    main()
