#!/usr/bin/env python3
"""Select a GDBN on training data and evaluate it against matched CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
TOOLS_DIR = CROWD_NAV_DIR / "tools"
REPO_ROOT = CROWD_NAV_DIR.parent
for path in (str(REPO_ROOT), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fit_behavior_gdbn_modes import (  # noqa: E402
    _cv_A,
    _cv_covariance,
    _evaluate,
    _extract_transitions,
    _fit_from_rows,
    _logpdf_gaussian,
    _logsumexp,
    _matrix_diag,
    _save_params,
)


def load_rows(path: Path):
    data = np.load(path, allow_pickle=True)
    return _extract_transitions(data["obs"], data["act"], max_peds=5, dt=0.25)


def protocol_mode_statistics(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    counts = np.zeros(5, dtype=np.int64)
    for sequence in data["modes"]:
        counts += np.bincount(
            np.asarray(sequence, dtype=np.int64).ravel(),
            minlength=5,
        )
    fractions = counts / max(int(counts.sum()), 1)
    return {
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "nominal_fraction": float(fractions[0]),
        "nonstationary_fraction": float(1.0 - fractions[0]),
    }


def split_rows_by_sequence(rows, train_fraction: float, seed: int):
    sequences = np.unique([int(row["seq"]) for row in rows])
    rng = np.random.default_rng(seed)
    rng.shuffle(sequences)
    split = max(1, min(len(sequences) - 1, int(len(sequences) * train_fraction)))
    selected = set(int(value) for value in sequences[:split])
    return (
        [row for row in rows if int(row["seq"]) in selected],
        [row for row in rows if int(row["seq"]) not in selected],
    )


def rank_auc(labels: Iterable[float], scores: Iterable[float]) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[positive].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def ece(labels, scores, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(scores[mask].mean()) - float(labels[mask].mean())
            )
    return result


def gaussian_circle_probability(mean, covariance, center, radius, samples):
    covariance = covariance[:2, :2] + 1e-8 * np.eye(2)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(covariance)
        chol = vectors @ np.diag(np.sqrt(np.maximum(values, 1e-8)))
    points = mean[:2] + samples @ chol.T
    return float(np.mean(np.linalg.norm(points - center[:2], axis=1) <= radius))


def risk_calibration(
    rows,
    A_list,
    B_list,
    Q_list,
    Pi,
    cv_cov,
    seed: int,
    risk_radius: float,
):
    by_track = {}
    for row in rows:
        by_track.setdefault((int(row["seq"]), int(row["ped"])), []).append(row)
    base_samples = np.random.default_rng(seed).standard_normal((64, 2))
    standard_samples = np.concatenate([base_samples, -base_samples], axis=0)
    cv_a = _cv_A(0.25)
    labels = []
    cv_scores = []
    gdbn_scores = []
    radius = float(risk_radius)

    for items in by_track.values():
        items = sorted(items, key=lambda item: int(item["t"]))
        posterior = np.ones(len(A_list), dtype=np.float64) / len(A_list)
        for row in items:
            state = row["x"]
            target = row["y"]
            action = row["action"]
            prior = np.maximum(posterior @ Pi, 1e-12)
            prior /= prior.sum()
            mode_means = np.stack(
                [
                    A_list[index] @ state + B_list[index] @ action
                    for index in range(len(A_list))
                ]
            )
            robot_now = state[:2] - row["static"][:2]
            robot_next = robot_now + action[:2] * 0.25
            labels.append(float(np.linalg.norm(target[:2] - robot_next) <= radius))
            cv_scores.append(
                gaussian_circle_probability(
                    cv_a @ state,
                    cv_cov,
                    robot_next,
                    radius,
                    standard_samples,
                )
            )
            gdbn_scores.append(
                sum(
                    prior[index]
                    * gaussian_circle_probability(
                        mode_means[index],
                        Q_list[index],
                        robot_next,
                        radius,
                        standard_samples,
                    )
                    for index in range(len(A_list))
                )
            )
            terms = np.asarray(
                [
                    np.log(prior[index] + 1e-12)
                    + _logpdf_gaussian(target, mode_means[index], Q_list[index])
                    for index in range(len(A_list))
                ],
                dtype=np.float64,
            )
            posterior = np.exp(terms - _logsumexp(terms))
            posterior /= max(float(posterior.sum()), 1e-12)

    labels = np.asarray(labels, dtype=np.float64)
    cv_scores = np.asarray(cv_scores, dtype=np.float64)
    gdbn_scores = np.asarray(gdbn_scores, dtype=np.float64)
    return {
        "pairs": int(len(labels)),
        "unsafe_events": int(labels.sum()),
        "unsafe_rate": float(labels.mean()),
        "risk_radius": radius,
        "cv_auc": rank_auc(labels, cv_scores),
        "gdbn_auc": rank_auc(labels, gdbn_scores),
        "cv_brier": float(np.mean((cv_scores - labels) ** 2)),
        "gdbn_brier": float(np.mean((gdbn_scores - labels) ** 2)),
        "cv_ece": ece(labels, cv_scores),
        "gdbn_ece": ece(labels, gdbn_scores),
    }


def evaluate_dataset(rows, model, cv_cov, seed: int, risk_radius: float):
    _, A_list, B_list, Q_list, _, Pi = model
    prediction = _evaluate(rows, A_list, B_list, Q_list, Pi, 0.25, cv_cov)
    risk = risk_calibration(
        rows,
        A_list,
        B_list,
        Q_list,
        Pi,
        cv_cov,
        seed,
        risk_radius,
    )
    return {"prediction": prediction, "risk": risk}


def finite(value) -> bool:
    return bool(np.isfinite(float(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="runs/bayesian_belief_pilot/data")
    parser.add_argument("--output_dir", default="runs/bayesian_belief_pilot/gate")
    parser.add_argument("--Ks", default="2,3,4")
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cv_prior", type=float, default=10.0)
    parser.add_argument(
        "--risk_radius",
        type=float,
        default=0.9,
        help="Center distance for a fixed 0.30 m surface-clearance caution zone.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_rows = load_rows(data_dir / "train.npz")
    protocol_modes = protocol_mode_statistics(data_dir / "train.npz")
    fit_rows, selection_rows = split_rows_by_sequence(
        all_train_rows,
        train_fraction=0.8,
        seed=args.seed,
    )
    selection_cv_cov = _cv_covariance(fit_rows, 0.25)
    candidates = []
    for K in [int(value) for value in args.Ks.split(",") if value.strip()]:
        model = _fit_from_rows(
            fit_rows,
            K,
            0.25,
            args.seed,
            args.ridge,
            args.cv_prior,
        )
        metrics = _evaluate(
            selection_rows,
            model[1],
            model[2],
            model[3],
            model[5],
            0.25,
            selection_cv_cov,
        )
        diagnostics = _matrix_diag(model[1], model[0], model[5])
        distances = [
            item["fro"] for item in diagnostics["pairwise_A_fro"]
        ]
        candidates.append(
            {
                "K": K,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "eligible": (
                    min(diagnostics["mode_counts"]) >= 100
                    and min(distances, default=0.0) >= 0.08
                ),
            }
        )

    eligible = [item for item in candidates if item["eligible"]]
    pool = eligible if eligible else candidates
    selected = min(pool, key=lambda item: item["metrics"]["gdbn_nll"])
    selected_k = int(selected["K"])
    final_model = _fit_from_rows(
        all_train_rows,
        selected_k,
        0.25,
        args.seed,
        args.ridge,
        args.cv_prior,
    )
    final_cv_cov = _cv_covariance(all_train_rows, 0.25)
    nominal_rows = load_rows(data_dir / "test_nominal.npz")
    nonstationary_rows = load_rows(data_dir / "test_nonstationary.npz")
    nominal = evaluate_dataset(
        nominal_rows,
        final_model,
        final_cv_cov,
        args.seed,
        args.risk_radius,
    )
    nonstationary = evaluate_dataset(
        nonstationary_rows,
        final_model,
        final_cv_cov,
        args.seed + 1,
        args.risk_radius,
    )
    final_diagnostics = _matrix_diag(
        final_model[1],
        final_model[0],
        final_model[5],
    )
    distances = [
        item["fro"] for item in final_diagnostics["pairwise_A_fro"]
    ]

    ns_prediction = nonstationary["prediction"]
    ns_risk = nonstationary["risk"]
    nominal_prediction = nominal["prediction"]
    checks = {
        "nonstationary_nll_delta_ge_0_25": (
            ns_prediction["cv_nll"] - ns_prediction["gdbn_nll"] >= 0.25
        ),
        "nonstationary_brier_improvement_ge_10pct": (
            ns_risk["gdbn_brier"] <= 0.90 * ns_risk["cv_brier"]
        ),
        "nonstationary_risk_auc_noninferior_0_01": (
            finite(ns_risk["gdbn_auc"])
            and finite(ns_risk["cv_auc"])
            and ns_risk["gdbn_auc"] >= ns_risk["cv_auc"] - 0.01
        ),
        "nonstationary_unsafe_events_ge_50": ns_risk["unsafe_events"] >= 50,
        "nominal_nll_not_worse_than_cv": (
            nominal_prediction["gdbn_nll"] <= nominal_prediction["cv_nll"]
        ),
        "mode_dominance_matches_protocol": (
            max(final_diagnostics["mode_fraction"])
            <= protocol_modes["nominal_fraction"] + 0.02
        ),
        "mode_min_count_ge_500": min(final_diagnostics["mode_counts"]) >= 500,
        "mode_min_distance_ge_0_08": min(distances, default=0.0) >= 0.08,
    }
    passed = all(checks.values())
    summary = {
        "selected_K": selected_k,
        "selection_candidates": candidates,
        "protocol_mode_statistics": protocol_modes,
        "final_diagnostics": final_diagnostics,
        "nominal": nominal,
        "nonstationary": nonstationary,
        "risk_definition": {
            "center_distance": float(args.risk_radius),
            "combined_radii": 0.6,
            "caution_clearance": float(args.risk_radius - 0.6),
        },
        "gate_checks": checks,
        "passed": passed,
        "decision": (
            "Proceed to minimal belief-space RL."
            if passed
            else "Stop before RL; predictor gate did not pass."
        ),
    }
    (output_dir / "prediction_gate.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _save_params(
        output_dir / "selected_gdbn",
        selected_k,
        final_model[0],
        all_train_rows,
        final_model[1],
        final_model[2],
        final_model[3],
        final_model[4],
        final_model[5],
    )

    lines = [
        "# Bayesian Belief Pilot: Prediction Gate",
        "",
        f"- Selected K: {selected_k}",
        f"- Gate passed: **{passed}**",
        f"- Decision: {summary['decision']}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if value else ' '}] `{name}`"
        for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## Held-out Nonstationary",
            "",
            f"- GDBN/CV NLL: {ns_prediction['gdbn_nll']:.6f} / "
            f"{ns_prediction['cv_nll']:.6f}",
            f"- GDBN/CV H4 FDE: {ns_prediction['gdbn_h4_fde']:.6f} / "
            f"{ns_prediction['cv_h4_fde']:.6f}",
            f"- GDBN/CV risk AUC: {ns_risk['gdbn_auc']:.6f} / "
            f"{ns_risk['cv_auc']:.6f}",
            f"- GDBN/CV risk Brier: {ns_risk['gdbn_brier']:.6f} / "
            f"{ns_risk['cv_brier']:.6f}",
            f"- Unsafe events: {ns_risk['unsafe_events']} "
            f"({ns_risk['unsafe_rate']:.6f})",
        ]
    )
    (output_dir / "prediction_gate.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"[PILOT-GATE] wrote {output_dir / 'prediction_gate.json'}")
    print(f"[PILOT-GATE] passed={passed}")


if __name__ == "__main__":
    main()
