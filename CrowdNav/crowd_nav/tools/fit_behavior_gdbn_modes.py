#!/usr/bin/env python3
"""Fit CV-anchored residual behavior modes from ORCA demonstrations."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _load_demo(path: Path):
    payload = np.load(path, allow_pickle=True)
    obs = list(payload['obs'])
    act = list(payload['act']) if 'act' in payload.files else None
    return obs, act


def _write_dummy_gng(path: Path, static_features: np.ndarray, labels: np.ndarray, K: int):
    mean = static_features.mean(axis=0)
    std = static_features.std(axis=0) + 1e-6
    features_n = (static_features - mean) / std
    nodes = np.zeros((K, static_features.shape[1]), dtype=np.float32)
    for k in range(K):
        idx = labels == k
        if idx.any():
            nodes[k] = features_n[idx].mean(axis=0)
    np.savez(
        path,
        nodes=nodes.astype(np.float32),
        errors=np.zeros(K, dtype=np.float32),
        feat_mean=mean.astype(np.float32),
        feat_std=std.astype(np.float32),
    )


def _angle_wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _safe_heading(vx, vy):
    speed = np.hypot(vx, vy)
    return np.where(speed > 1e-4, np.arctan2(vy, vx), 0.0)


def _extract_transitions(obs_list, act_list, max_peds: int, dt: float):
    rows = []
    for seq_id, obs_seq in enumerate(obs_list):
        if not (
            isinstance(obs_seq, np.ndarray)
            and obs_seq.ndim == 2
            and obs_seq.shape[1] == 34
        ):
            continue
        T = len(obs_seq)
        act_seq = None if act_list is None else act_list[seq_id]
        if act_seq is None or not isinstance(act_seq, np.ndarray) or act_seq.ndim != 2:
            act_seq = np.zeros((T, 2), dtype=np.float64)
        robot = obs_seq[:, :9]
        for p in range(min(max_peds, 5)):
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
                action = act_seq[t, :2].astype(np.float64) if t < len(act_seq) else np.zeros(2)

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
                    [rel[0], rel[1], dist, rel_v[0], rel_v[1]],
                    dtype=np.float64,
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
                rows.append({
                    'seq': seq_id,
                    'ped': p,
                    't': t,
                    'x': x,
                    'y': y,
                    'action': action,
                    'static': static_feat,
                    'behavior': behavior_feat,
                })
    if not rows:
        raise ValueError('no valid transitions found')
    return rows


def _standardize(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    return (features - mean) / std, mean, std


def _kmeans_pp(features, K: int, seed: int, restarts: int = 12, max_iter: int = 100):
    rng = np.random.default_rng(seed)
    n = len(features)
    best_labels = None
    best_centers = None
    best_score = float('inf')
    for _ in range(max(1, restarts)):
        centers = np.empty((K, features.shape[1]), dtype=np.float64)
        centers[0] = features[rng.integers(n)]
        dist2 = np.sum((features - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            probs = dist2 / max(float(dist2.sum()), 1e-12)
            centers[k] = features[rng.choice(n, p=probs)]
            dist2 = np.minimum(dist2, np.sum((features - centers[k]) ** 2, axis=1))

        labels = np.zeros(n, dtype=np.int64)
        for _it in range(max_iter):
            d = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d, axis=1)
            new_centers = centers.copy()
            for k in range(K):
                idx = new_labels == k
                if idx.any():
                    new_centers[k] = features[idx].mean(axis=0)
                else:
                    farthest = int(np.argmax(np.min(d, axis=1)))
                    new_centers[k] = features[farthest]
                    new_labels[farthest] = k
            shift = float(np.linalg.norm(new_centers - centers))
            labels = new_labels
            centers = new_centers
            if shift < 1e-5:
                break

        d = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(d, axis=1)))
        counts = np.bincount(labels, minlength=K)
        balance_penalty = float(np.sum((counts < max(20, n // (K * 40))) * n))
        score = inertia + balance_penalty
        if score < best_score:
            best_score = score
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def _cv_A(dt):
    A = np.eye(4, dtype=np.float64)
    A[0, 2] = float(dt)
    A[1, 3] = float(dt)
    return A


def _fit_dynamics(
    rows,
    labels,
    K: int,
    dt: float,
    ridge: float,
    cv_prior: float,
):
    A_list = []
    B_list = []
    Q_list = []
    C_list = []
    cv = _cv_A(dt)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 20:
            A_list.append(cv.copy())
            B_list.append(np.zeros((4, 2), dtype=np.float64))
            Q_list.append(0.05 * np.eye(4, dtype=np.float64))
            C_list.append(0.05 * np.eye(4, dtype=np.float64))
            continue
        X = np.asarray([rows[i]['x'] for i in idx], dtype=np.float64)
        U = np.asarray([rows[i]['action'] for i in idx], dtype=np.float64)
        Y = np.asarray([rows[i]['y'] for i in idx], dtype=np.float64)
        D = np.concatenate([X, U], axis=1)
        target_residual = Y - (cv @ X.T).T
        regularization = np.diag(
            [float(cv_prior)] * X.shape[1] + [float(ridge)] * U.shape[1]
        )
        theta = np.linalg.solve(
            D.T @ D + regularization,
            D.T @ target_residual,
        )
        A = cv + theta[:4, :].T
        B = theta[4:, :].T
        u, s, vt = np.linalg.svd(A)
        A = u @ np.diag(np.clip(s, 0.0, 1.5)) @ vt
        pred = (A @ X.T).T + (B @ U.T).T
        residual = Y - pred
        cov = (residual.T @ residual) / max(1, len(residual)) + 1e-4 * np.eye(4)
        A_list.append(A)
        B_list.append(B)
        Q_list.append(cov)
        C_list.append(cov)
    return A_list, B_list, Q_list, C_list


def _fit_transition_matrix(rows, labels, K: int):
    counts = np.zeros((K, K), dtype=np.float64)
    by_track = {}
    for i, row in enumerate(rows):
        by_track.setdefault((row['seq'], row['ped']), []).append((row['t'], int(labels[i])))
    for items in by_track.values():
        items = sorted(items)
        for (_, a), (_, b) in zip(items[:-1], items[1:]):
            counts[a, b] += 1.0
    return (counts + 0.1) / (counts.sum(axis=1, keepdims=True) + 0.1 * K)


def _logpdf_gaussian(x, mean, cov):
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    try:
        chol = np.linalg.cholesky(cov)
        diff = x - mean
        sol = np.linalg.solve(chol, diff)
        return float(-0.5 * (np.dot(sol, sol) + 2.0 * np.log(np.diag(chol)).sum() + len(x) * np.log(2.0 * np.pi)))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        return float(-0.5 * ((x - mean) @ inv @ (x - mean) + logdet + len(x) * np.log(2.0 * np.pi)))


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    m = float(values.max())
    return m + float(np.log(np.exp(values - m).sum()))


def _evaluate(rows, A_list, B_list, Q_list, Pi, dt: float, cv_cov: np.ndarray):
    K = len(A_list)
    by_track = {}
    for row in rows:
        by_track.setdefault((row['seq'], row['ped']), []).append(row)

    g_pos = []
    c_pos = []
    g_vel = []
    c_vel = []
    g_nll = []
    c_nll = []
    h4_g_ade = []
    h4_c_ade = []
    h4_g_fde = []
    h4_c_fde = []

    cvA = _cv_A(dt)
    for items in by_track.values():
        items = sorted(items, key=lambda r: r['t'])
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
            cv_pred = cvA @ x
            g_pos.append(float(np.linalg.norm(pred[:2] - y[:2])))
            c_pos.append(float(np.linalg.norm(cv_pred[:2] - y[:2])))
            g_vel.append(float(np.linalg.norm(pred[2:4] - y[2:4])))
            c_vel.append(float(np.linalg.norm(cv_pred[2:4] - y[2:4])))

            log_terms = [
                np.log(prior[k] + 1e-12) + _logpdf_gaussian(y, pred_modes[k], Q_list[k])
                for k in range(K)
            ]
            g_nll.append(-_logsumexp(log_terms))
            c_nll.append(-_logpdf_gaussian(y, cv_pred, cv_cov))

            # Update mode posterior with the observed next state.
            likelihood = np.exp(np.asarray(log_terms) - _logsumexp(log_terms))
            pi = likelihood / max(float(likelihood.sum()), 1e-12)

            max_h = min(4, len(items) - idx)
            if max_h >= 2:
                mode_dist = pi.copy()
                g_state = y.copy()
                c_state = y.copy()
                g_errs = []
                c_errs = []
                for h in range(1, max_h):
                    fut = items[idx + h]
                    fut_u = fut['action']
                    fut_y = fut['y']
                    alpha = mode_dist @ Pi
                    alpha = np.maximum(alpha, 1e-10)
                    alpha /= alpha.sum()
                    fut_modes = np.stack([A_list[k] @ g_state + B_list[k] @ fut_u for k in range(K)], axis=0)
                    g_state = (alpha[:, None] * fut_modes).sum(axis=0)
                    c_state = cvA @ c_state
                    g_errs.append(float(np.linalg.norm(g_state[:2] - fut_y[:2])))
                    c_errs.append(float(np.linalg.norm(c_state[:2] - fut_y[:2])))
                    mode_dist = alpha
                if g_errs:
                    h4_g_ade.append(float(np.mean(g_errs)))
                    h4_c_ade.append(float(np.mean(c_errs)))
                    h4_g_fde.append(float(g_errs[-1]))
                    h4_c_fde.append(float(c_errs[-1]))

    def mean(values):
        return float(np.mean(values)) if values else float('nan')

    return {
        'pairs': int(len(g_pos)),
        'gdbn_pos_ade': mean(g_pos),
        'cv_pos_ade': mean(c_pos),
        'gdbn_vel_rmse': mean(g_vel),
        'cv_vel_rmse': mean(c_vel),
        'gdbn_nll': mean(g_nll),
        'cv_nll': mean(c_nll),
        'gdbn_h4_ade': mean(h4_g_ade),
        'cv_h4_ade': mean(h4_c_ade),
        'gdbn_h4_fde': mean(h4_g_fde),
        'cv_h4_fde': mean(h4_c_fde),
    }


def _save_params(output: Path, K: int, labels, rows, A_list, B_list, Q_list, C_list, Pi):
    output.mkdir(parents=True, exist_ok=True)
    static = np.asarray([r['static'] for r in rows], dtype=np.float64)
    _write_dummy_gng(output / 'gng.npz', static, labels, K)
    payload = {'K': np.asarray(K), 'Pi': Pi, 'R': 0.01 * np.eye(4)}
    for k in range(K):
        payload[f'A_{k}'] = A_list[k]
        payload[f'Q_{k}'] = Q_list[k]
    np.savez(output / 'gdbn.npz', **payload)
    action_payload = {'K': np.asarray(K)}
    for k in range(K):
        action_payload[f'B_{k}'] = B_list[k]
        action_payload[f'C_{k}'] = C_list[k]
    np.savez(output / 'action_model.npz', **action_payload)


def _matrix_diag(A_list, labels, Pi):
    distances = []
    for i in range(len(A_list)):
        for j in range(i + 1, len(A_list)):
            distances.append({'i': i, 'j': j, 'fro': float(np.linalg.norm(A_list[i] - A_list[j]))})
    counts = np.bincount(labels, minlength=len(A_list))
    return {
        'mode_counts': counts.astype(int).tolist(),
        'mode_fraction': (counts / max(1, counts.sum())).round(6).tolist(),
        'pairwise_A_fro': distances,
        'Pi_row_entropy': float(-np.mean(np.sum(Pi * np.log(np.maximum(Pi, 1e-10)), axis=1))),
    }


def _fit_from_rows(
    rows,
    K: int,
    dt: float,
    seed: int,
    ridge: float,
    cv_prior: float,
):
    behavior = np.asarray([r['behavior'] for r in rows], dtype=np.float64)
    features, _, _ = _standardize(behavior)
    labels, _ = _kmeans_pp(features, K, seed=seed)
    A_list, B_list, Q_list, C_list = _fit_dynamics(
        rows,
        labels,
        K,
        dt,
        ridge,
        cv_prior,
    )
    Pi = _fit_transition_matrix(rows, labels, K)
    return labels, A_list, B_list, Q_list, C_list, Pi


def _cv_covariance(rows, dt: float):
    cvA = _cv_A(dt)
    residuals = np.asarray([r['y'] - cvA @ r['x'] for r in rows], dtype=np.float64)
    return (residuals.T @ residuals) / max(1, len(residuals)) + 1e-4 * np.eye(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--demo', default='../orca_demos_seq.npz')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_peds', type=int, default=5)
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument(
        '--cv_prior',
        type=float,
        default=10.0,
        help='Shrink mode dynamics toward the constant-velocity transition.',
    )
    parser.add_argument(
        '--min_mode_distance',
        type=float,
        default=0.08,
        help='Fail when fitted dynamics contain redundant modes.',
    )
    args = parser.parse_args()

    demo = Path(args.demo).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    obs, act = _load_demo(demo)
    all_rows = _extract_transitions(obs, act, max_peds=args.max_peds, dt=args.dt)

    rng = np.random.default_rng(args.seed)
    seqs = np.unique([r['seq'] for r in all_rows])
    rng.shuffle(seqs)
    split = max(1, min(len(seqs) - 1, int(round(len(seqs) * args.train_ratio))))
    train_seqs = set(int(x) for x in seqs[:split])
    train_rows = [r for r in all_rows if int(r['seq']) in train_seqs]
    valid_rows = [r for r in all_rows if int(r['seq']) not in train_seqs]

    split_labels, split_A, split_B, split_Q, split_C, split_Pi = _fit_from_rows(
        train_rows,
        args.K,
        args.dt,
        args.seed,
        args.ridge,
        args.cv_prior,
    )
    cv_cov = _cv_covariance(train_rows, args.dt)
    validation = _evaluate(valid_rows, split_A, split_B, split_Q, split_Pi, args.dt, cv_cov)
    validation_matrix = _matrix_diag(split_A, split_labels, split_Pi)

    if output.exists():
        shutil.rmtree(output)
    full_labels, full_A, full_B, full_Q, full_C, full_Pi = _fit_from_rows(
        all_rows,
        args.K,
        args.dt,
        args.seed,
        args.ridge,
        args.cv_prior,
    )
    _save_params(output, args.K, full_labels, all_rows, full_A, full_B, full_Q, full_C, full_Pi)
    full_eval = _evaluate(all_rows, full_A, full_B, full_Q, full_Pi, args.dt, _cv_covariance(all_rows, args.dt))
    full_matrix = _matrix_diag(full_A, full_labels, full_Pi)

    summary = {
        'K': int(args.K),
        'demo': str(demo),
        'output': str(output),
        'feature_mode': 'cv_residual_kinematics',
        'dynamics_model': 'cv_anchored_residual',
        'cv_prior': float(args.cv_prior),
        'validation': validation,
        'validation_matrix': validation_matrix,
        'full': full_eval,
        'full_matrix': full_matrix,
    }
    (output / 'diagnostics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    minimum_distance = min(
        (
            item['fro']
            for item in validation_matrix['pairwise_A_fro']
        ),
        default=float('inf'),
    )
    if minimum_distance < args.min_mode_distance:
        raise RuntimeError(
            f"Redundant modes: minimum validation A distance={minimum_distance:.6f} "
            f"< required {args.min_mode_distance:.6f}"
        )
    print(f"[BEHAVIOR-GDBN] wrote {output / 'diagnostics.json'}")


if __name__ == '__main__':
    main()
