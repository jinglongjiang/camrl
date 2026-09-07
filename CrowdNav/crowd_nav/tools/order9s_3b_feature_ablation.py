#!/usr/bin/env python3
"""Order 9S.3b: frozen-model feature-dependence ablation (NO refit).

The 2026-08-03 audit's diagnosis: mode_model.py fits K-means on the
10-dim phi THEN fits per-cluster linear dynamics -- K behaves like "number
of piecewise-linear regression regions" over whatever features happen to
vary smoothly, not necessarily a genuine latent intent/interaction mode.
Order 9S.3 (refit on shuffled robot pairing) already showed the advantage
survives refitting on corrupted robot info, but refitting itself could in
principle re-discover clusters from self-kinematics alone regardless of
what the ORIGINAL model actually used. This script removes that
confound: fit ONCE on correctly-paired data, freeze (F, Q, Pi), and only
at SCORING time (never refitting) compare held-out NLL under:

  full       -- the real, correctly-paired phi (10 dims)
  self_only  -- phi with the 6 robot-relative dims (relative_px/py/vx/vy,
                ttc_clipped, passing_side) replaced by their TRAIN-set mean
                (removes robot information, keeps self-kinematics real)
  robot_only -- phi with the 4 self-kinematic dims (speed, delta_speed,
                heading_change, lateral_acceleration) replaced by their
                TRAIN-set mean (removes self-motion information, keeps
                robot-relative features real)
  shuffled_robot -- the ACTUAL robot-relative values from a mismatched
                (derangement-shuffled) episode, not just blanked to the
                mean -- tests whether genuinely wrong (not just absent)
                robot info still scores well under the frozen model.

If ``self_only`` recovers nearly all of ``full``'s NLL improvement over
K=1 while ``robot_only`` recovers almost none, that is strong evidence the
frozen model's discriminative power lives almost entirely in self-motion
features, independent of any refitting-on-corrupted-data confound.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

from crowd_nav.bayesian_brne.config import BayesianModelConfig
from crowd_nav.bayesian_brne.mode_model import FEATURE_NAMES, TransitionRow, extract_transitions, fit_and_select
from crowd_nav.tools.fit_bayesian_brne import (
    fast_bootstrap_nll_ci,
    freeze_physical_caps,
    load_formal_episodes,
    per_track_nlls,
    track_nlls_by_suite_seed,
)
from crowd_nav.tools.order9s_shuffle_test import shuffle_robot_pairing

ROBOT_DIMS = (4, 5, 6, 7, 8, 9)  # relative_px, relative_py, relative_vx, relative_vy, ttc_clipped, passing_side
SELF_DIMS = (0, 1, 2, 3)  # speed, delta_speed, heading_change, lateral_acceleration


def _mask_rows(rows: List[TransitionRow], dims_to_replace: tuple, replacement_values: np.ndarray) -> List[TransitionRow]:
    """Returns NEW TransitionRow objects whose ``phi`` has ``dims_to_replace``
    overwritten by ``replacement_values`` (per-dim constants). Never mutates
    the fitted model or the original rows."""
    out = []
    for row in rows:
        phi = row.phi.copy()
        for d in dims_to_replace:
            phi[d] = replacement_values[d]
        out.append(replace(row, phi=phi))
    return out


def main() -> None:
    print("[order9s_3b] loading Order 8 formal episodes (correctly paired) ...")
    train_eps, train_seeds = load_formal_episodes("runs/bayesian_brne/data_formal", "train", "baseline_circle")
    val_eps, val_seeds = load_formal_episodes("runs/bayesian_brne/data_formal", "validation", "baseline_circle")

    train_rows = extract_transitions(train_eps, dt=0.25)
    val_rows_full = extract_transitions(val_eps, dt=0.25)
    print(f"[order9s_3b] {len(train_rows)} train / {len(val_rows_full)} validation transitions")

    max_speed, max_accel = freeze_physical_caps(train_rows)
    config = BayesianModelConfig(k_candidates=(1, 6), max_human_speed=max_speed, max_human_acceleration=max_accel)

    print("[order9s_3b] fitting ONCE on correctly-paired data (K=1 and K=6, frozen thereafter) ...")
    artifact, reports, fitted = fit_and_select(
        train_rows, val_rows_full, config, seed=2407, require_multimodal=False, min_nll_improvement=0.01, return_all_fits=True,
    )
    F1, Q1, Pi1, _ = fitted[1]
    F6, Q6, Pi6, _ = fitted[6]

    train_feature_means = np.array([r.phi for r in train_rows]).mean(axis=0)

    val_eps_shuffled = shuffle_robot_pairing(val_eps, val_seeds, seed=2409)
    val_rows_shuffled = extract_transitions(val_eps_shuffled, dt=0.25)
    # extract_transitions may drop/keep different row counts if valid_mask
    # differs; guard the ablation on the row count actually produced.

    variants = {
        "full": val_rows_full,
        "self_only": _mask_rows(val_rows_full, ROBOT_DIMS, train_feature_means),
        "robot_only": _mask_rows(val_rows_full, SELF_DIMS, train_feature_means),
        "shuffled_robot": val_rows_shuffled,
    }

    print()
    print("=" * 70)
    print("ORDER 9S.3b -- FROZEN-MODEL FEATURE-DEPENDENCE ABLATION (no refit)")
    print("=" * 70)
    print(f"FEATURE_NAMES={FEATURE_NAMES}")
    print(f"ROBOT_DIMS={[FEATURE_NAMES[d] for d in ROBOT_DIMS]}")
    print(f"SELF_DIMS={[FEATURE_NAMES[d] for d in SELF_DIMS]}")
    print()

    result = {"variants": {}}
    for name, rows in variants.items():
        pt6 = per_track_nlls(rows, F6, Q6, Pi6)
        pt1 = per_track_nlls(rows, F1, Q1, Pi1)
        seeds_for_rows = val_seeds if name != "shuffled_robot" else val_seeds  # same episode set, same seed list
        by_seed6 = track_nlls_by_suite_seed(pt6, seeds_for_rows)
        by_seed1 = track_nlls_by_suite_seed(pt1, seeds_for_rows)
        boot = fast_bootstrap_nll_ci(by_seed6, by_seed1, n_resamples=2000, seed=2407)
        nll6 = float(np.mean([v for vals in pt6.values() for v in vals]))
        nll1 = float(np.mean([v for vals in pt1.values() for v in vals]))
        improvement = nll1 - nll6
        result["variants"][name] = {
            "n_rows": len(rows), "nll_k1": nll1, "nll_k6": nll6, "improvement_k6_over_k1": improvement,
            "improvement_ci": [boot["improvement_p2_5"], boot["improvement_p97_5"]],
            "ci_supports_real_improvement": boot["ci_supports_real_improvement"],
        }
        print(f"{name:16s}: nll_K1={nll1:.4f} nll_K6={nll6:.4f} improvement={improvement:.4f} "
              f"CI=[{boot['improvement_p2_5']:.4f},{boot['improvement_p97_5']:.4f}] ci_supports={boot['ci_supports_real_improvement']}")

    full_improvement = result["variants"]["full"]["improvement_k6_over_k1"]
    self_only_improvement = result["variants"]["self_only"]["improvement_k6_over_k1"]
    robot_only_improvement = result["variants"]["robot_only"]["improvement_k6_over_k1"]
    self_fraction = self_only_improvement / full_improvement if full_improvement else float("nan")
    robot_fraction = robot_only_improvement / full_improvement if full_improvement else float("nan")

    print()
    print(f"self_only recovers {self_fraction*100:.1f}% of the full model's K6-over-K1 improvement")
    print(f"robot_only recovers {robot_fraction*100:.1f}% of the full model's K6-over-K1 improvement")
    result["self_only_fraction_of_full_improvement"] = self_fraction
    result["robot_only_fraction_of_full_improvement"] = robot_fraction

    out_path = "runs/bayesian_brne/models/order9s_3b_feature_ablation.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
