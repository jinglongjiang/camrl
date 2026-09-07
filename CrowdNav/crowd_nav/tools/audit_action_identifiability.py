#!/usr/bin/env python3
"""Permanent, reproducible action-identifiability diagnostic (2026-08-03
fourth audit round item 2, corrected 2026-08-03 per fifth audit round).

This replaces an earlier one-off, non-reproducible check (R^2 ~ 0.4-0.54,
condition number ~78.3) that the audit flagged as methodologically
imprecise for two reasons:

1. The condition number was computed on the RAW (unstandardized)
   ``[v_current, u_robot, context, 1]`` design matrix. Raw feature-scale
   differences (e.g. ``relative_px`` in meters vs ``ttc_clipped`` in
   seconds vs a constant 1) can dominate a raw condition number without
   reflecting genuine collinearity between the robot action and the
   context features. This script standardizes every non-intercept column
   (mean/std from TRAIN only) before computing condition number/effective
   rank.

2. A bare ``R^2(u_t ~ context_t)`` only shows that the action is PARTIALLY
   explainable by context -- it does not show whether the UNEXPLAINED part
   of the action (its residual after regressing out context) has genuine
   INCREMENTAL predictive power for ``v_next``. This script adds that
   comparison directly: held-out R^2/NLL of "context-only -> v_next" vs
   "context + action-residual -> v_next", where action-residual is
   ``u_robot - predict(context)`` using a regression fit on TRAIN ONLY and
   applied to validation without refitting.

Fifth audit round (2026-08-03 17:34 KST) found two further problems in the
first version of this script, both fixed here:

3. ``covariance_fit_split`` leakage: the first version estimated each
   model's scoring Gaussian's mean/covariance FROM THE VALIDATION
   RESIDUALS THEMSELVES, then scored NLL on those same residuals -- this
   lets each model fit its own most-favorable validation-set nuisance
   parameters, a form of test-set leakage that can inflate or deflate the
   apparent gap between models. Fixed: mean/covariance for BOTH models'
   Gaussian scoring distribution are now estimated from TRAIN residuals
   ONLY, frozen, and applied unchanged when scoring validation (see
   ``_fit_gaussian_from_train`` / ``_gaussian_nll_frozen``, and the
   ``covariance_fit_split: "train"`` field in each result).

4. No suite-seed block bootstrap: a raw point-estimate improvement of
   ~0.002-0.006 nats could easily be noise given only 5 validation suite
   seeds. Fixed: every per-row quantity needed for R^2/NLL is precomputed
   once, grouped by the VALIDATION suite seed of its source episode, then
   block-bootstrap resampled (same discipline as
   ``fit_bayesian_brne.fast_bootstrap_nll_ci`` -- resample precomputed
   per-unit values, never re-run any recursive/refitting computation per
   resample) to get a percentile CI on the incremental R^2/NLL gain.

Uses the real formal data (``runs/bayesian_brne/data_formal``, train suite
seeds 11-15, validation suite seeds 21-25), broken down per controller
(goal_directed/orca/original_brne/scripted_probe) plus a combined "all"
row, since the four controllers embed very different action policies and
a pooled number alone could hide controller-specific structure.

This script does not gate anything by itself -- it is a diagnostic. It
writes a full-provenance JSON report (source hashes, data manifest hash,
environment info) so results are independently reproducible without
re-deriving them from chat history.
"""

from __future__ import annotations

import glob
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from crowd_nav.bayesian_brne import data_io
from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
    CONTEXT_FEATURE_NAMES,
    ARHMMSequence,
    _design_dims,
    extract_sequences,
)

DATA_DIR = "runs/bayesian_brne/data_formal"
SCENARIO = "baseline_circle"
TRAIN_SEEDS = [11, 12, 13, 14, 15]
VALIDATION_SEEDS = [21, 22, 23, 24, 25]
CONTROLLERS = ["goal_directed", "orca", "original_brne", "scripted_probe"]
OUT_PATH = "runs/bayesian_brne/models/audit_action_identifiability.json"
SOURCE_FILES = [
    "crowd_nav/tools/audit_action_identifiability.py",
    "crowd_nav/bayesian_brne/action_conditioned_arhmm.py",
    "crowd_nav/bayesian_brne/mode_model.py",
    "crowd_nav/bayesian_brne/data_io.py",
]
N_BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2407


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_split_episodes(split: str, suite_seeds: List[int]) -> Tuple[Dict[str, List[dict]], Dict[str, List[int]]]:
    """Load every formal episode for ``split``, grouped by controller_type.
    Unlike ``fit_bayesian_brne.load_formal_episodes`` this keeps
    ``robot_actions`` and ``controller_type`` (both required here) and
    hard-asserts the file set matches the expected suite seeds exactly, so
    a stale/partial data_formal directory fails loudly instead of silently
    producing a diagnostic over the wrong data. Also returns each
    controller's per-episode suite_seed list (SAME order as the episode
    list), so a sequence's ``track_key[0]`` (its index into that
    controller's episode list) can be mapped back to a suite seed for
    block bootstrap grouping."""
    paths = sorted(glob.glob(f"{DATA_DIR}/{split}/{SCENARIO}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episodes found under {DATA_DIR}/{split}/{SCENARIO}/")
    by_controller: Dict[str, List[dict]] = {c: [] for c in CONTROLLERS}
    seeds_by_controller: Dict[str, List[int]] = {c: [] for c in CONTROLLERS}
    seen_seeds = set()
    for p in paths:
        ep = data_io.load_episode(p)
        if ep["profile_name"] != "formal":
            raise ValueError(f"{p}: profile_name={ep['profile_name']!r}, expected 'formal'")
        if ep["split"] != split:
            raise ValueError(f"{p}: split={ep['split']!r}, expected {split!r}")
        if ep["controller_type"] not in by_controller:
            raise ValueError(f"{p}: unexpected controller_type={ep['controller_type']!r}")
        seen_seeds.add(int(ep["suite_seed"]))
        by_controller[ep["controller_type"]].append({
            "humans": ep["humans"], "human_track_ids": ep["human_track_ids"],
            "robot": ep["robot"], "valid_mask": ep["valid_mask"],
            "robot_actions": ep["robot_actions"],
        })
        seeds_by_controller[ep["controller_type"]].append(int(ep["suite_seed"]))
    if seen_seeds != set(suite_seeds):
        raise ValueError(f"{split}: expected suite seeds {sorted(suite_seeds)}, found {sorted(seen_seeds)}")
    return by_controller, seeds_by_controller


def _design_rows_with_seed(
    sequences: List[ARHMMSequence], suite_seed_per_episode: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(v_current, u_robot, context, v_next, row_suite_seed)``
    stacked across all sequences' rows, where ``row_suite_seed[i]`` is the
    suite seed of the episode that produced row ``i`` (looked up via each
    sequence's ``track_key[0]``, i.e. its index into ``suite_seed_per_episode``)."""
    if not sequences:
        d_v, d_u, d_c, _ = _design_dims()
        z = lambda w: np.zeros((0, w))
        return z(d_v), z(d_u), z(d_c), z(d_v), np.zeros((0,), dtype=int)
    v = np.concatenate([s.v_current for s in sequences], axis=0)
    u = np.concatenate([s.u_robot for s in sequences], axis=0)
    c = np.concatenate([s.context for s in sequences], axis=0)
    y = np.concatenate([s.v_next for s in sequences], axis=0)
    row_seed = np.concatenate([
        np.full(s.v_current.shape[0], suite_seed_per_episode[s.track_key[0]], dtype=int) for s in sequences
    ])
    return v, u, c, y, row_seed


def _ridge_fit(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Least-squares ``Y = X @ W.T`` fit with a tiny ridge for numerical
    stability only (not a modeling choice) -- returns ``W`` of shape
    ``[Y.shape[1], X.shape[1]]``."""
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY).T
    return W


def _fit_gaussian_from_train(residuals_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate (mean, covariance) of a model's scoring Gaussian from TRAIN
    residuals ONLY, then freeze -- these are used unchanged to score
    validation, closing the leakage the fifth audit round found (the first
    version of this script re-estimated mean/covariance from validation
    residuals themselves before scoring them). Mean is NOT hardcoded to 0:
    it is computed from train residuals directly, since a ridge-regularized
    fit's train residual mean is not guaranteed exactly zero."""
    mean = residuals_train.mean(axis=0)
    centered = residuals_train - mean
    n, d = residuals_train.shape
    cov = (centered.T @ centered) / max(n, 1) + 1e-9 * np.eye(d)
    return mean, cov


def _gaussian_nll_frozen(residuals: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-row negative log-likelihood of ``residuals`` under a FIXED
    (train-fit, frozen) Gaussian -- returns one value per row so it can be
    grouped by suite seed and block-bootstrap resampled without
    re-estimating anything per resample."""
    d = residuals.shape[1]
    centered = residuals - mean
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    quad = np.einsum("ij,jk,ik->i", centered, inv, centered)
    return 0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def _standardize_with_stats(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / np.where(std < 1e-9, 1.0, std)


def _block_bootstrap_ci(
    per_seed: Dict[int, Dict[str, np.ndarray]],
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Suite-seed block bootstrap over precomputed per-row quantities.
    ``per_seed[s]`` holds, for validation suite seed ``s``, the arrays
    ``sq_resid_a``, ``sq_resid_b``, ``sq_dev`` (all per-row squared terms
    for R^2's numerator/denominator) and ``nll_a``, ``nll_b`` (per-row NLL
    under each model's FROZEN train-fit Gaussian). Each resample draws
    suite seeds with replacement and recomputes the AGGREGATE R^2/NLL from
    the concatenated precomputed values -- no refitting, no recursive
    computation, matching the discipline in
    ``fit_bayesian_brne.fast_bootstrap_nll_ci``."""
    seeds = sorted(per_seed.keys())
    rng = np.random.default_rng(seed)
    r2_gain = np.zeros(n_resamples)
    nll_reduction = np.zeros(n_resamples)
    for i in range(n_resamples):
        picked = rng.choice(seeds, size=len(seeds), replace=True)
        sq_resid_a = np.concatenate([per_seed[s]["sq_resid_a"] for s in picked])
        sq_resid_b = np.concatenate([per_seed[s]["sq_resid_b"] for s in picked])
        sq_dev = np.concatenate([per_seed[s]["sq_dev"] for s in picked])
        nll_a = np.concatenate([per_seed[s]["nll_a"] for s in picked])
        nll_b = np.concatenate([per_seed[s]["nll_b"] for s in picked])
        denom = sq_dev.sum()
        r2_a = 1.0 - sq_resid_a.sum() / denom if denom > 0 else float("nan")
        r2_b = 1.0 - sq_resid_b.sum() / denom if denom > 0 else float("nan")
        r2_gain[i] = r2_b - r2_a
        nll_reduction[i] = nll_a.mean() - nll_b.mean()
    return {
        "r2_gain_p2_5": float(np.percentile(r2_gain, 2.5)),
        "r2_gain_p50": float(np.percentile(r2_gain, 50)),
        "r2_gain_p97_5": float(np.percentile(r2_gain, 97.5)),
        "r2_gain_ci_excludes_zero": bool(np.percentile(r2_gain, 2.5) > 0),
        "nll_reduction_p2_5": float(np.percentile(nll_reduction, 2.5)),
        "nll_reduction_p50": float(np.percentile(nll_reduction, 50)),
        "nll_reduction_p97_5": float(np.percentile(nll_reduction, 97.5)),
        "nll_reduction_ci_excludes_zero": bool(np.percentile(nll_reduction, 2.5) > 0),
        "n_suite_seeds": len(seeds),
        "n_resamples": n_resamples,
    }


def _analyze_group(
    train_sequences: List[ARHMMSequence],
    val_sequences: List[ARHMMSequence],
    train_suite_seed_per_episode: List[int],
    val_suite_seed_per_episode: List[int],
    label: str,
) -> Dict[str, object]:
    v_tr, u_tr, c_tr, y_tr, _seed_tr = _design_rows_with_seed(train_sequences, train_suite_seed_per_episode)
    v_va, u_va, c_va, y_va, seed_va = _design_rows_with_seed(val_sequences, val_suite_seed_per_episode)
    n_train, n_val = v_tr.shape[0], v_va.shape[0]
    if n_train < 50 or n_val < 20:
        return {"label": label, "n_train_rows": n_train, "n_val_rows": n_val, "skipped": "too few rows"}

    ones_tr = np.ones((n_train, 1))
    ones_va = np.ones((n_val, 1))

    # --- standardized design matrix condition number / effective rank ---
    # [v_current, u_robot, context, 1] -- the SAME columns m_step's design
    # vector uses, standardized (train mean/std) so raw unit-scale
    # differences cannot dominate the number.
    raw_design_tr = np.concatenate([v_tr, u_tr, c_tr], axis=1)
    design_mean = raw_design_tr.mean(axis=0)
    design_std = raw_design_tr.std(axis=0)
    std_design_tr = np.concatenate([_standardize_with_stats(raw_design_tr, design_mean, design_std), ones_tr], axis=1)
    singular_values = np.linalg.svd(std_design_tr, compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    effective_rank = int(np.sum(singular_values > 1e-8 * singular_values[0]))

    # --- u_t ~ context_t regression (train-only) + residual covariance ---
    context_design_tr = np.concatenate([c_tr, ones_tr], axis=1)
    W_u_given_c = _ridge_fit(context_design_tr, u_tr)
    u_resid_tr = u_tr - context_design_tr @ W_u_given_c.T
    resid_cov = (u_resid_tr.T @ u_resid_tr) / max(n_train, 1)
    resid_min_eig = float(np.min(np.linalg.eigvalsh(resid_cov)))
    r2_u_given_context_train = 1.0 - float(np.sum(u_resid_tr ** 2)) / max(float(np.sum((u_tr - u_tr.mean(axis=0)) ** 2)), 1e-12)

    # apply the SAME train-fit context->action model to validation (no refit)
    context_design_va = np.concatenate([c_va, ones_va], axis=1)
    u_resid_va = u_va - context_design_va @ W_u_given_c.T

    # --- incremental predictive power: v_next ~ [v_current, context, 1] ---
    #     vs v_next ~ [v_current, context, u_action_residual, 1] ---
    # Both models fit on TRAIN only. R^2's baseline (sq_dev) and each
    # model's scoring Gaussian (mean/cov) are ALSO frozen from TRAIN only --
    # validation is scored, never used to estimate any nuisance parameter.
    X_a_tr = np.concatenate([v_tr, c_tr, ones_tr], axis=1)
    X_a_va = np.concatenate([v_va, c_va, ones_va], axis=1)
    W_a = _ridge_fit(X_a_tr, y_tr)
    resid_a_tr = y_tr - X_a_tr @ W_a.T
    resid_a_va = y_va - X_a_va @ W_a.T
    mean_a, cov_a = _fit_gaussian_from_train(resid_a_tr)

    X_b_tr = np.concatenate([v_tr, c_tr, u_resid_tr, ones_tr], axis=1)
    X_b_va = np.concatenate([v_va, c_va, u_resid_va, ones_va], axis=1)
    W_b = _ridge_fit(X_b_tr, y_tr)
    resid_b_tr = y_tr - X_b_tr @ W_b.T
    resid_b_va = y_va - X_b_va @ W_b.T
    mean_b, cov_b = _fit_gaussian_from_train(resid_b_tr)

    y_train_mean = y_tr.mean(axis=0)  # frozen R^2 baseline, NOT validation's own mean
    sq_dev_va = np.sum((y_va - y_train_mean) ** 2, axis=1)
    sq_resid_a_va = np.sum(resid_a_va ** 2, axis=1)
    sq_resid_b_va = np.sum(resid_b_va ** 2, axis=1)
    nll_a_va = _gaussian_nll_frozen(resid_a_va, mean_a, cov_a)
    nll_b_va = _gaussian_nll_frozen(resid_b_va, mean_b, cov_b)

    r2_context_only = 1.0 - float(sq_resid_a_va.sum()) / float(sq_dev_va.sum())
    r2_context_plus_action_residual = 1.0 - float(sq_resid_b_va.sum()) / float(sq_dev_va.sum())
    nll_context_only = float(nll_a_va.mean())
    nll_context_plus_action_residual = float(nll_b_va.mean())

    per_seed: Dict[int, Dict[str, np.ndarray]] = {}
    for s in np.unique(seed_va):
        mask = seed_va == s
        per_seed[int(s)] = {
            "sq_resid_a": sq_resid_a_va[mask], "sq_resid_b": sq_resid_b_va[mask],
            "sq_dev": sq_dev_va[mask], "nll_a": nll_a_va[mask], "nll_b": nll_b_va[mask],
        }
    bootstrap = _block_bootstrap_ci(per_seed) if len(per_seed) >= 2 else None

    return {
        "label": label,
        "n_train_rows": n_train,
        "n_val_rows": n_val,
        "standardized_design_condition_number": condition_number,
        "standardized_design_effective_rank": effective_rank,
        "standardized_design_full_rank": std_design_tr.shape[1],
        "r2_u_given_context_train": r2_u_given_context_train,
        "action_residual_covariance_min_eigenvalue": resid_min_eig,
        "covariance_fit_split": "train",
        "r2_context_only": r2_context_only,
        "r2_context_plus_action_residual": r2_context_plus_action_residual,
        "incremental_r2_gain": r2_context_plus_action_residual - r2_context_only,
        "nll_context_only": nll_context_only,
        "nll_context_plus_action_residual": nll_context_plus_action_residual,
        # NLL is a NEGATIVE log-likelihood: lower is better, so a genuine
        # incremental gain from adding the action residual shows up as a
        # POSITIVE "nll_reduction" (context_only_nll - context_plus_action_nll).
        "incremental_nll_reduction": nll_context_only - nll_context_plus_action_residual,
        "suite_seed_block_bootstrap": bootstrap,
    }


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    print(f"[audit_action_identifiability] context features: {CONTEXT_FEATURE_NAMES}")
    print(f"[audit_action_identifiability] loading train (seeds {TRAIN_SEEDS}) and validation (seeds {VALIDATION_SEEDS}) ...")
    train_by_controller, train_seeds_by_controller = _load_split_episodes("train", TRAIN_SEEDS)
    val_by_controller, val_seeds_by_controller = _load_split_episodes("validation", VALIDATION_SEEDS)

    dt = 0.25
    results: Dict[str, object] = {}
    all_train_eps: List[dict] = []
    all_val_eps: List[dict] = []
    all_train_seeds: List[int] = []
    all_val_seeds: List[int] = []
    for controller in CONTROLLERS:
        train_eps = train_by_controller[controller]
        val_eps = val_by_controller[controller]
        train_seeds = train_seeds_by_controller[controller]
        val_seeds = val_seeds_by_controller[controller]
        all_train_eps.extend(train_eps)
        all_val_eps.extend(val_eps)
        all_train_seeds.extend(train_seeds)
        all_val_seeds.extend(val_seeds)
        train_seqs = extract_sequences(train_eps, dt=dt)
        val_seqs = extract_sequences(val_eps, dt=dt)
        print(f"[audit_action_identifiability] controller={controller}: "
              f"{len(train_eps)} train episodes / {sum(s.v_current.shape[0] for s in train_seqs)} train rows, "
              f"{len(val_eps)} validation episodes / {sum(s.v_current.shape[0] for s in val_seqs)} validation rows")
        results[controller] = _analyze_group(train_seqs, val_seqs, train_seeds, val_seeds, controller)

    all_train_seqs = extract_sequences(all_train_eps, dt=dt)
    all_val_seqs = extract_sequences(all_val_eps, dt=dt)
    results["all_controllers_combined"] = _analyze_group(
        all_train_seqs, all_val_seqs, all_train_seeds, all_val_seeds, "all_controllers_combined"
    )

    print()
    print("=" * 70)
    print("ACTION IDENTIFIABILITY AUDIT (standardized design + incremental predictive power, train-frozen scoring)")
    print("=" * 70)
    for key, r in results.items():
        if r.get("skipped"):
            print(f"{key}: SKIPPED ({r['skipped']})")
            continue
        print(f"{key}: n_train={r['n_train_rows']} n_val={r['n_val_rows']} "
              f"cond_number(std)={r['standardized_design_condition_number']:.2f} "
              f"eff_rank={r['standardized_design_effective_rank']}/{r['standardized_design_full_rank']} "
              f"R2(u|context,train)={r['r2_u_given_context_train']:.4f} "
              f"resid_cov_min_eig={r['action_residual_covariance_min_eigenvalue']:.6f}")
        print(f"    R2: context_only={r['r2_context_only']:.4f} "
              f"context+action_resid={r['r2_context_plus_action_residual']:.4f} "
              f"incremental_gain={r['incremental_r2_gain']:+.4f}")
        print(f"    NLL: context_only={r['nll_context_only']:.4f} "
              f"context+action_resid={r['nll_context_plus_action_residual']:.4f} "
              f"incremental_reduction={r['incremental_nll_reduction']:+.4f}")
        b = r.get("suite_seed_block_bootstrap")
        if b:
            print(f"    bootstrap({b['n_resamples']} resamples over {b['n_suite_seeds']} suite seeds): "
                  f"r2_gain CI=[{b['r2_gain_p2_5']:.5f},{b['r2_gain_p97_5']:.5f}] excludes_zero={b['r2_gain_ci_excludes_zero']} | "
                  f"nll_reduction CI=[{b['nll_reduction_p2_5']:.5f},{b['nll_reduction_p97_5']:.5f}] excludes_zero={b['nll_reduction_ci_excludes_zero']}")

    manifest_path = f"{DATA_DIR}/manifest_post_collection.json"
    with open(manifest_path) as f:
        data_manifest_hash = json.load(f)["aggregate_sha256"]

    report = {
        "audit": "action_identifiability_permanent_script",
        "date": "2026-08-03",
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "results": results,
        "provenance": {
            "source_sha256": {p: _sha256_file(p) for p in SOURCE_FILES},
            "data_manifest_aggregate_sha256": data_manifest_hash,
            "data_manifest_path": manifest_path,
            "git_head": _git_head(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "note": (
            "This is a diagnostic script, not a pass/fail gate. It does not "
            "by itself decide whether the action-conditioned model is "
            "necessary -- that is Upgrade U3's job (same-K/same-budget "
            "comparison of K=1 / self-only / action-conditioned / "
            "action-shuffled with held-out NLL confidence intervals)."
        ),
    }
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
