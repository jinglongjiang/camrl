#!/usr/bin/env python3
"""CLI: fit SM-BRNE's mode model on formally-collected 5-person data and
apply guide.md Order 9's Bayesian scientific gate.

Only reads Order 8 FORMAL data (``profile_name == "formal"``,
``split in {train, validation}``) -- smoke/pilot data must never reach
fitting (guide.md 10.4/Order 7). ``latent_behavior_labels`` is explicitly
dropped before any row is built (guide.md 5d.5: it is audit-only).
Physical speed/acceleration caps are frozen from TRAIN split statistics
only, never from validation. Runs K=1..6 with ``require_multimodal``
semantics, reports held-out NLL/Brier/ECE-coverage/mode occupancy/
predictive similarity/Pi/Q eigenvalues/fit+inference latency/suite-seed
bootstrap CI for EVERY K, and prints the RAW PASS/FAIL gate result.

PASS condition (guide.md Order 9): at least one K>1 candidate survives
mode_model.py's occupancy/collapse/numeric checks AND beats K=1 by the
pre-registered ``min_nll_improvement`` with a suite-seed bootstrap CI that
does not include zero/negative improvement.

FAIL: no K>1 candidate qualifies. This script does NOT fall back to
reporting a K=1 artifact as "the SM-BRNE artifact" on FAIL -- it reports
FAIL and stops. Modifying heldout/profile ranges or the rejection
thresholds to force a PASS after seeing this result is exactly what
guide.md Order 9 forbids.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2

from crowd_nav.bayesian_brne import data_io
from crowd_nav.bayesian_brne.config import BayesianModelConfig, load_model_config
from crowd_nav.bayesian_brne.mode_model import (
    FEATURE_NAMES,
    TransitionRow,
    _logpdf_gaussian,
    extract_transitions,
    fit_and_select,
)

NLL_BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2407
COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def load_formal_episodes(data_dir: str, split: str, scenario: str = "baseline_circle") -> Tuple[List[dict], List[int]]:
    """Load every episode under ``data_dir/split/scenario/*.npz``, requiring
    ``profile_name == 'formal'`` and the file's own ``split`` field to match
    ``split`` -- refuses to silently mix in smoke/pilot data or a
    mislabeled split. Returns ``(episode_dicts_for_extract_transitions,
    suite_seed_per_episode_index)``; ``latent_behavior_labels`` is dropped
    from the dict passed onward (guide.md 5d.5 -- audit-only field, never a
    fitting input)."""
    paths = sorted(glob.glob(f"{data_dir}/{split}/{scenario}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episodes found under {data_dir}/{split}/{scenario}/")
    episodes = []
    suite_seeds = []
    for p in paths:
        ep = data_io.load_episode(p)
        if ep["profile_name"] != "formal":
            raise ValueError(f"{p}: profile_name={ep['profile_name']!r}, expected 'formal' -- refusing to fit on non-formal data")
        if ep["split"] != split:
            raise ValueError(f"{p}: split={ep['split']!r}, expected {split!r}")
        episodes.append({
            "humans": ep["humans"], "human_track_ids": ep["human_track_ids"],
            "robot": ep["robot"], "valid_mask": ep["valid_mask"],
            # latent_behavior_labels deliberately NOT included: audit-only (guide.md 5d.5).
        })
        suite_seeds.append(ep["suite_seed"])
    return episodes, suite_seeds


def freeze_physical_caps(train_rows: List[TransitionRow], quantile: float = 0.99) -> Tuple[float, float]:
    """Order 9: physical speed/acceleration caps must be frozen from TRAIN
    split statistics only, before validation participates in anything.
    Returns ``(max_human_speed, max_human_acceleration)`` at the given
    quantile of observed train-split speed/acceleration magnitudes."""
    speeds = np.array([r.phi[0] for r in train_rows])
    accels = np.array([float(np.hypot(*r.delta_v_next)) / max(r.step_dt, 1e-6) for r in train_rows])
    return float(np.quantile(speeds, quantile)), float(np.quantile(accels, quantile))


def _rows_by_suite_seed(rows: List[TransitionRow], suite_seed_per_episode: List[int]) -> Dict[int, List[TransitionRow]]:
    grouped: Dict[int, List[TransitionRow]] = {}
    for row in rows:
        episode_index, _ = row.track_key
        seed = suite_seed_per_episode[episode_index]
        grouped.setdefault(seed, []).append(row)
    return grouped


# NOTE (Order 9S self-audit, 2026-08-03): an earlier version of this file had
# a bootstrap_nll_ci() that resampled suite seeds by PHYSICALLY
# concatenating each drawn seed's rows and re-running
# mode_model._held_out_nll on the concatenation. With only 5 validation
# suite seeds, ~95% of 5-draws-with-replacement resamples contain at least
# one repeated seed -- and concatenating the SAME episode's rows twice under
# the SAME (episode_index, track_id) key makes _held_out_nll's per-track
# recursion walk through a track whose timeline contains each real timestep
# TWICE adjacently, which is not "this suite seed counts twice in the
# resample", it is a corrupted, artificially-extended sequence the
# recursive posterior filter was never meant to see (independently verified:
# concatenating one seed's rows with itself changed the NLL, -0.6085 ->
# -0.6390, rather than leaving it unchanged as a correct double-count
# should). That function has been REMOVED. ``per_track_nlls`` +
# ``fast_bootstrap_nll_ci`` below are the correct replacement: each track's
# per-row NLL sequence is computed ONCE (never duplicated within a
# recursion), and a suite seed drawn twice in a resample simply re-uses that
# already-correctly-computed sequence's values twice when averaging -- this
# is also ~20,000x faster (0.2s vs ~4546s for 2000 resamples on the full
# formal validation set), since resampling no longer re-runs any Gaussian
# likelihood evaluation.


def per_track_nlls(rows: List[TransitionRow], F_list, Q_list, Pi: np.ndarray) -> Dict[Tuple[int, int], List[float]]:
    """Order 9S speedup: the SAME per-row recursive computation
    ``mode_model._held_out_nll`` does, but returns each TRACK's own list of
    per-row NLL contributions instead of collapsing to one mean. A track's
    sequence is independent of every other track (the posterior recursion
    only carries over WITHIN a track), so this only needs to be computed
    ONCE per (K, dataset) -- suite-seed block bootstrap can then resample
    which tracks' precomputed lists to average, with NO repeated Gaussian
    evaluation per resample. This is what makes running dozens of bootstrap
    comparisons (Order 9S needs many more than Order 9's single one)
    tractable instead of requiring hours per comparison."""
    K = len(F_list)
    by_track: Dict[Tuple[int, int], List[TransitionRow]] = {}
    for row in rows:
        by_track.setdefault(row.track_key, []).append(row)

    result: Dict[Tuple[int, int], List[float]] = {}
    for track_key, items in by_track.items():
        items = sorted(items, key=lambda r: r.t)
        posterior = np.ones(K) / K
        nlls = []
        for row in items:
            prior = np.maximum(posterior @ Pi, 1e-12)
            prior /= prior.sum()
            means = [row.v + F_list[k] @ row.phi for k in range(K)]
            outcome = row.delta_v_next + row.v
            log_terms = np.array([np.log(prior[k]) + _logpdf_gaussian(outcome, means[k], Q_list[k]) for k in range(K)])
            m = log_terms.max()
            nlls.append(-(m + np.log(np.exp(log_terms - m).sum())))
            likelihood = np.exp(log_terms - log_terms.max())
            posterior = likelihood / max(float(likelihood.sum()), 1e-12)
        result[track_key] = nlls
    return result


def track_nlls_by_suite_seed(
    per_track: Dict[Tuple[int, int], List[float]], suite_seed_per_episode: List[int]
) -> Dict[int, np.ndarray]:
    grouped: Dict[int, List[float]] = {}
    for (episode_index, _track_id), nlls in per_track.items():
        seed = suite_seed_per_episode[episode_index]
        grouped.setdefault(seed, []).extend(nlls)
    return {s: np.array(v) for s, v in grouped.items()}


def fast_bootstrap_nll_ci(
    nlls_by_seed_target: Dict[int, np.ndarray],
    nlls_by_seed_baseline: Dict[int, np.ndarray],
    n_resamples: int = NLL_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Correct suite-seed block bootstrap: resamples PRECOMPUTED per-track
    NLL arrays (see ``per_track_nlls``) instead of re-running the recursive
    filter on physically-concatenated resampled rows -- both far faster AND
    correct where the earlier removed ``bootstrap_nll_ci`` was not (see the
    NOTE above ``per_track_nlls``)."""
    seeds = sorted(nlls_by_seed_target.keys())
    rng = np.random.default_rng(seed)
    nlls_target = np.zeros(n_resamples)
    improvements = np.zeros(n_resamples)
    for i in range(n_resamples):
        resampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        t_vals = np.concatenate([nlls_by_seed_target[s] for s in resampled_seeds])
        b_vals = np.concatenate([nlls_by_seed_baseline[s] for s in resampled_seeds])
        nlls_target[i] = t_vals.mean()
        improvements[i] = b_vals.mean() - t_vals.mean()
    return {
        "nll_p2_5": float(np.percentile(nlls_target, 2.5)),
        "nll_p50": float(np.percentile(nlls_target, 50)),
        "nll_p97_5": float(np.percentile(nlls_target, 97.5)),
        "improvement_p2_5": float(np.percentile(improvements, 2.5)),
        "improvement_p50": float(np.percentile(improvements, 50)),
        "improvement_p97_5": float(np.percentile(improvements, 97.5)),
        "ci_supports_real_improvement": bool(np.percentile(improvements, 2.5) > 0.0),
    }


def evaluate_calibration(rows: List[TransitionRow], F_list, Q_list, Pi: np.ndarray) -> Dict[str, float]:
    """Brier score + coverage-based ECE for one K's fitted (F, Q, Pi),
    scored on ``rows`` (guide.md Order 9: "Brier、ECE/coverage").

    Brier: the model's PRIOR mixture weight (before seeing this row's
    outcome -- never the posterior after updating on it, which would leak
    the answer) scored against a hard oracle pseudo-label: whichever mode's
    own Gaussian gives the observed outcome the highest individual
    likelihood. This measures whether the model's own belief about "which
    mode is active" tracks which mode's dynamics actually best explain what
    happened next.

    ECE/coverage: the moment-matched single-Gaussian mixture predictive's
    Mahalanobis-distance chi2(df=2) coverage at COVERAGE_LEVELS, averaged
    |empirical - nominal| deviation across those levels -- standard
    calibration reporting for a continuous (not classification) predictive
    distribution.
    """
    K = len(F_list)
    by_track: Dict[Tuple[int, int], List[TransitionRow]] = {}
    for row in rows:
        by_track.setdefault(row.track_key, []).append(row)

    brier_terms = []
    mahalanobis_sq = []
    for items in by_track.values():
        items = sorted(items, key=lambda r: r.t)
        posterior = np.ones(K) / K
        for row in items:
            prior = np.maximum(posterior @ Pi, 1e-12)
            prior /= prior.sum()

            means = [row.v + F_list[k] @ row.phi for k in range(K)]
            outcome = row.delta_v_next + row.v
            log_liks = np.array([_logpdf_gaussian(outcome, means[k], Q_list[k]) for k in range(K)])
            winning_mode = int(np.argmax(log_liks))

            indicator = np.zeros(K)
            indicator[winning_mode] = 1.0
            brier_terms.append(float(np.sum((prior - indicator) ** 2)))

            mix_mean = sum(prior[k] * means[k] for k in range(K))
            mix_cov = sum(
                prior[k] * (Q_list[k] + np.outer(means[k] - mix_mean, means[k] - mix_mean))
                for k in range(K)
            )
            diff = outcome - mix_mean
            mix_cov_reg = mix_cov + 1e-6 * np.eye(2)
            maha_sq = float(diff @ np.linalg.inv(mix_cov_reg) @ diff)
            mahalanobis_sq.append(maha_sq)

            m = log_liks.max()
            likelihood = np.exp(log_liks - m)
            posterior = likelihood / max(float(likelihood.sum()), 1e-12)

    mahalanobis_sq = np.array(mahalanobis_sq)
    ece_terms = []
    empirical_coverage = {}
    for level in COVERAGE_LEVELS:
        threshold = chi2.ppf(level, df=2)
        empirical = float(np.mean(mahalanobis_sq <= threshold))
        empirical_coverage[level] = empirical
        ece_terms.append(abs(empirical - level))

    return {
        "brier_score": float(np.mean(brier_terms)) if brier_terms else float("nan"),
        "ece": float(np.mean(ece_terms)) if ece_terms else float("nan"),
        "empirical_coverage": empirical_coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--k-candidates", default="1,2,3,4,5,6")
    parser.add_argument("--scenario", default="baseline_circle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--min-nll-improvement", type=float, default=0.01)
    parser.add_argument(
        "--policy-config", default="crowd_nav/configs/policy_bayesian_brne.config",
        help="base [bayesian_model] section; max_human_speed/acceleration are OVERRIDDEN by train-split-frozen values regardless of what this file says",
    )
    args = parser.parse_args()

    k_candidates = tuple(int(k) for k in args.k_candidates.split(","))

    print(f"[fit_bayesian_brne] loading TRAIN episodes from {args.train_data}/train/{args.scenario}/ ...")
    train_episodes, train_suite_seeds = load_formal_episodes(args.train_data, "train", args.scenario)
    print(f"[fit_bayesian_brne] loading VALIDATION episodes from {args.validation_data}/validation/{args.scenario}/ ...")
    val_episodes, val_suite_seeds = load_formal_episodes(args.validation_data, "validation", args.scenario)
    print(f"[fit_bayesian_brne] {len(train_episodes)} train episodes ({len(set(train_suite_seeds))} suite seeds), "
          f"{len(val_episodes)} validation episodes ({len(set(val_suite_seeds))} suite seeds)")

    train_rows = extract_transitions(train_episodes, dt=0.25)
    val_rows = extract_transitions(val_episodes, dt=0.25)
    print(f"[fit_bayesian_brne] {len(train_rows)} train transitions, {len(val_rows)} validation transitions")

    max_speed, max_accel = freeze_physical_caps(train_rows)
    print(f"[fit_bayesian_brne] frozen from TRAIN split only: max_human_speed={max_speed:.4f}, max_human_acceleration={max_accel:.4f}")

    base_config = configparser.RawConfigParser()
    if Path(args.policy_config).exists():
        base_config.read(args.policy_config)
    model_config = load_model_config(base_config)
    config = BayesianModelConfig(
        k_candidates=k_candidates,
        n_particles=model_config.n_particles,
        min_mode_fraction=model_config.min_mode_fraction,
        max_mode_similarity=model_config.max_mode_similarity,
        covariance_floor=model_config.covariance_floor,
        dt=model_config.dt,
        max_human_speed=max_speed,
        max_human_acceleration=max_accel,
    )

    t_fit_start = time.time()
    artifact_relaxed, reports, fitted = fit_and_select(
        train_rows, val_rows, config, seed=args.seed,
        require_multimodal=False, min_nll_improvement=args.min_nll_improvement, return_all_fits=True,
    )
    fit_latency_s = time.time() - t_fit_start

    k1_report = next(r for r in reports if r.K == 1)
    multimodal_eligible = [r for r in reports if r.eligible and r.K > 1]
    best_multimodal_k = min(multimodal_eligible, key=lambda r: r.held_out_nll).K if multimodal_eligible else None

    # per_track_nlls (Order 9S fix) makes bootstrap ~20,000x cheaper than
    # the removed bootstrap_nll_ci, so it is now affordable to compute a
    # bootstrap CI for EVERY K against K=1, not only the eventual winner --
    # useful for the K-order plateau diagnostic (Order 9S.1), not just the
    # single PASS/FAIL decision.
    per_track_by_k = {r.K: per_track_nlls(val_rows, *fitted[r.K][:3]) for r in reports}
    by_seed_by_k = {k: track_nlls_by_suite_seed(pt, val_suite_seeds) for k, pt in per_track_by_k.items()}

    per_k_report = {}
    for r in reports:
        F_list, Q_list, Pi, centers = fitted[r.K]
        q_eigs = [np.linalg.eigvalsh(Q).tolist() for Q in Q_list]

        t_inf_start = time.time()
        calib = evaluate_calibration(val_rows, F_list, Q_list, Pi)
        inference_latency_s = (time.time() - t_inf_start) / max(len(val_rows), 1)

        bootstrap = None
        if r.K > 1:
            t_boot_start = time.time()
            bootstrap = fast_bootstrap_nll_ci(by_seed_by_k[r.K], by_seed_by_k[1], n_resamples=NLL_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
            bootstrap["bootstrap_wall_time_s"] = time.time() - t_boot_start

        per_k_report[r.K] = {
            "eligible": r.eligible,
            "rejection_reasons": r.rejection_reasons,
            "held_out_nll": r.held_out_nll,
            "mode_counts": r.mode_counts,
            "max_pairwise_similarity_f_cosine_diagnostic_only": r.max_pairwise_similarity,
            "max_predictive_similarity": r.max_predictive_similarity,
            "pi_row_entropy": r.pi_row_entropy,
            "pi": Pi.tolist(),
            "q_eigenvalues": q_eigs,
            "brier_score": calib["brier_score"],
            "ece": calib["ece"],
            "empirical_coverage": calib["empirical_coverage"],
            "inference_latency_s_per_transition": inference_latency_s,
            "bootstrap": bootstrap,
        }

    gate_pass = best_multimodal_k is not None
    selected_k = None
    ci_supports = None
    if gate_pass:
        selected_k = best_multimodal_k
        ci_supports = per_k_report[selected_k]["bootstrap"]["ci_supports_real_improvement"]
        gate_pass = gate_pass and bool(ci_supports)

    result = {
        "gate": "PASS" if gate_pass else "FAIL",
        "selected_k": selected_k if gate_pass else None,
        "k1_held_out_nll": k1_report.held_out_nll,
        "fit_latency_s": fit_latency_s,
        "frozen_max_human_speed": max_speed,
        "frozen_max_human_acceleration": max_accel,
        "n_train_transitions": len(train_rows),
        "n_validation_transitions": len(val_rows),
        "n_train_suite_seeds": len(set(train_suite_seeds)),
        "n_validation_suite_seeds": len(set(val_suite_seeds)),
        "feature_names": list(FEATURE_NAMES),
        "per_k": per_k_report,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print()
    print("=" * 70)
    print(f"ORDER 9 GATE RESULT: {result['gate']}")
    print("=" * 70)
    for k in sorted(per_k_report):
        r = per_k_report[k]
        boot_str = ""
        if r["bootstrap"] is not None:
            b = r["bootstrap"]
            boot_str = (
                f" | improvement_over_K1 CI=[{b['improvement_p2_5']:.4f}, {b['improvement_p97_5']:.4f}] "
                f"nats, ci_supports_real_improvement={b['ci_supports_real_improvement']}"
            )
        print(
            f"K={k}: eligible={r['eligible']} held_out_nll={r['held_out_nll']:.4f} "
            f"mode_counts={r['mode_counts']} brier={r['brier_score']:.4f} ece={r['ece']:.4f}{boot_str}"
        )
        if r["rejection_reasons"]:
            for reason in r["rejection_reasons"]:
                print(f"    rejected: {reason}")
    print()
    if gate_pass:
        print(f"Selected K={selected_k}. artifact NOT saved by this script's gate report -- "
              "freezing the artifact for policy use is a separate, explicit step after PASS is confirmed.")
    else:
        print("NO K>1 candidate passed. Per guide.md Order 9: STOP. Do not revert to K=1 and continue "
              "calling it a switching-mode Bayesian method; do not modify heldout/profile to force a pass.")
    print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
