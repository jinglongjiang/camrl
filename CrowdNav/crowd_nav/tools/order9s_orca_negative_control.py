#!/usr/bin/env python3
"""Order 9S.2: ORCA negative control.

``orca_demos_seq.npz`` (project root) is a pre-existing dataset of 1000
ORCA-driven episodes recorded as ``obs`` ([T,34] flattened JointState:
robot FullState(9) + 5 humans' ObservableState(5) each) and ``act``
([T,2], the robot's own action -- not used here). ORCA-driven humans are
near-single-mode: each reacts to the crowd via the SAME reciprocal
avoidance law, with no scripted behavioral diversity. This is NOT a
positive validation of SM-BRNE (it is not real human data), and per
guide.md Order 9S.2 must not be used as one -- its only job is to check
whether the SAME K-selection pipeline that picked K=6 on the hand-scripted
6-behavior-type synthetic data ALSO spuriously splits this near-unimodal
data into many clusters. If it does, that is evidence the mode-collapse
gate is not doing its job (over-splitting), independent of the synthetic
data's known 6-mode structure.

Track identity here is taken from the array's fixed slot order (human i is
always in position i within a given episode) -- a legacy recording
convention specific to this pre-existing file, not a pattern any new
collector in this package is allowed to rely on (guide.md 5.6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from crowd_nav.bayesian_brne.config import BayesianModelConfig
from crowd_nav.bayesian_brne.mode_model import extract_transitions, fit_and_select
from crowd_nav.tools.fit_bayesian_brne import (
    evaluate_calibration,
    fast_bootstrap_nll_ci,
    freeze_physical_caps,
    per_track_nlls,
    track_nlls_by_suite_seed,
)

N_ROBOT_FIELDS = 9
N_HUMAN_FIELDS = 5
N_HUMANS = 5


def load_orca_demo_episodes(path: str) -> List[dict]:
    """Convert orca_demos_seq.npz's [T,34]-flattened-JointState format into
    this package's episode dict format (humans[T,N,5], human_track_ids[T,N],
    robot[T,9], valid_mask[T,N])."""
    data = np.load(path, allow_pickle=True)
    obs_all = data["obs"]
    episodes = []
    for obs in obs_all:
        obs = np.asarray(obs, dtype=np.float64)
        T = obs.shape[0]
        robot = obs[:, :N_ROBOT_FIELDS]
        humans = obs[:, N_ROBOT_FIELDS:].reshape(T, N_HUMANS, N_HUMAN_FIELDS)
        track_ids = np.tile(np.arange(N_HUMANS), (T, 1))
        valid_mask = np.ones((T, N_HUMANS), dtype=bool)
        episodes.append({"humans": humans, "human_track_ids": track_ids, "robot": robot, "valid_mask": valid_mask})
    return episodes


def main() -> None:
    path = "orca_demos_seq.npz"
    print(f"[order9s_orca] loading {path} ...")
    episodes = load_orca_demo_episodes(path)
    print(f"[order9s_orca] {len(episodes)} episodes")

    # No suite-seed structure exists in this legacy file; split by episode
    # index into a train/validation partition ONLY for this negative-control
    # sanity check (not a claim of suite-seed-block rigor -- this check's
    # only purpose is "does K stay low on near-unimodal data").
    rng = np.random.default_rng(2407)
    order = rng.permutation(len(episodes))
    n_train = int(0.7 * len(episodes))
    train_eps = [episodes[i] for i in order[:n_train]]
    val_eps = [episodes[i] for i in order[n_train:]]
    # Pseudo "suite seeds": chunk validation episodes into 5 blocks so the
    # bootstrap machinery (which resamples by seed) has something to grip.
    val_pseudo_seeds = [i % 5 for i in range(len(val_eps))]

    train_rows = extract_transitions(train_eps, dt=0.25)
    val_rows = extract_transitions(val_eps, dt=0.25)
    print(f"[order9s_orca] {len(train_rows)} train transitions, {len(val_rows)} validation transitions")

    max_speed, max_accel = freeze_physical_caps(train_rows)
    print(f"[order9s_orca] frozen from ORCA train split: max_human_speed={max_speed:.4f}, max_human_acceleration={max_accel:.4f}")

    config = BayesianModelConfig(k_candidates=tuple(range(1, 11)), max_human_speed=max_speed, max_human_acceleration=max_accel)
    artifact, reports, fitted = fit_and_select(
        train_rows, val_rows, config, seed=2407, require_multimodal=False, min_nll_improvement=0.01, return_all_fits=True,
    )

    per_track_by_k = {r.K: per_track_nlls(val_rows, *fitted[r.K][:3]) for r in reports}
    by_seed_by_k = {k: track_nlls_by_suite_seed(pt, val_pseudo_seeds) for k, pt in per_track_by_k.items()}

    result = {"n_episodes": len(episodes), "max_human_speed": max_speed, "max_human_acceleration": max_accel, "per_k": {}}
    print()
    print("=" * 70)
    print("ORDER 9S.2 -- ORCA NEGATIVE CONTROL")
    print("=" * 70)
    for r in reports:
        F_list, Q_list, Pi, _ = fitted[r.K]
        calib = evaluate_calibration(val_rows, F_list, Q_list, Pi)
        boot = None
        if r.K > 1:
            boot = fast_bootstrap_nll_ci(by_seed_by_k[r.K], by_seed_by_k[1], n_resamples=2000, seed=2407)
        result["per_k"][r.K] = {
            "eligible": r.eligible, "rejection_reasons": r.rejection_reasons,
            "held_out_nll": r.held_out_nll, "mode_counts": r.mode_counts,
            "max_predictive_similarity": r.max_predictive_similarity,
            "brier_score": calib["brier_score"], "ece": calib["ece"],
            "bootstrap": boot,
        }
        boot_str = ""
        if boot is not None:
            boot_str = f" | improvement_over_K1 CI=[{boot['improvement_p2_5']:.4f},{boot['improvement_p97_5']:.4f}], ci_supports={boot['ci_supports_real_improvement']}"
        print(f"K={r.K}: eligible={r.eligible} nll={r.held_out_nll:.4f} mode_counts={r.mode_counts}{boot_str}")
        if r.rejection_reasons:
            for reason in r.rejection_reasons:
                print(f"    rejected: {reason}")

    multimodal_eligible = [r for r in reports if r.eligible and r.K > 1]
    print()
    if not multimodal_eligible:
        print("RESULT: no K>1 eligible on ORCA data -- negative control PASSES (model correctly stays near-unimodal).")
        result["negative_control_result"] = "PASS_STAYS_UNIMODAL"
    else:
        best_k = min(multimodal_eligible, key=lambda r: r.held_out_nll).K
        print(f"RESULT: K={best_k} eligible on ORCA (near-unimodal) data -- negative control indicates possible over-splitting.")
        result["negative_control_result"] = f"CONCERN_SELECTED_K_{best_k}"

    out_path = "runs/bayesian_brne/models/order9s_orca_negative_control.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
