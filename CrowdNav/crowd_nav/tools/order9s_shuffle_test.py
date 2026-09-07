#!/usr/bin/env python3
"""Order 9S.3: necessity-breaking (shuffle) test.

Takes the SAME Order 8 formal episodes, but pairs each episode's humans
with a DIFFERENT, randomly chosen episode's robot trajectory before
extracting transitions. This corrupts exactly the robot-relative phi
features (relative_px/py/vx/vy, ttc_clipped, passing_side -- 6 of 10) while
leaving each human's own self-kinematic features (speed, delta_speed,
heading_change, lateral_acceleration) untouched, since those depend only
on the human's own true trajectory.

If the K>1-over-K=1 NLL improvement found on correctly-paired data
persists at similar magnitude even with FAKE robot pairing, that is
evidence the "improvement" is coming from marginal self-motion-style
clustering (which behavior type a human has, recoverable from its own
kinematics alone), not from genuine robot-action-conditioned interaction
structure -- guide.md Order 9S.3's exact concern. If the improvement
collapses toward zero, that supports the model actually using the
robot-conditioned features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crowd_nav.bayesian_brne.config import BayesianModelConfig
from crowd_nav.bayesian_brne.mode_model import extract_transitions, fit_and_select
from crowd_nav.tools.fit_bayesian_brne import (
    fast_bootstrap_nll_ci,
    freeze_physical_caps,
    load_formal_episodes,
    per_track_nlls,
    track_nlls_by_suite_seed,
)


def shuffle_robot_pairing(episodes: list, suite_seeds: list, seed: int) -> list:
    """Returns a NEW list of episode dicts where every episode's ``robot``
    array has been swapped for a DIFFERENT episode's ``robot`` array (a
    derangement -- no episode keeps its own true robot trajectory),
    humans/track_ids/valid_mask left untouched."""
    rng = np.random.default_rng(seed)
    n = len(episodes)
    perm = rng.permutation(n)
    # Ensure a derangement (no fixed points) -- extremely unlikely for n=2000
    # but cheap to guarantee.
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    shuffled = []
    for i in range(n):
        donor = episodes[perm[i]]
        ep = dict(episodes[i])
        ep["robot"] = donor["robot"]
        shuffled.append(ep)
    return shuffled


def main() -> None:
    print("[order9s_shuffle] loading Order 8 formal episodes ...")
    train_eps, train_seeds = load_formal_episodes("runs/bayesian_brne/data_formal", "train", "baseline_circle")
    val_eps, val_seeds = load_formal_episodes("runs/bayesian_brne/data_formal", "validation", "baseline_circle")

    print("[order9s_shuffle] shuffling robot<->human episode pairing (derangement, seed=2407) ...")
    train_eps_shuffled = shuffle_robot_pairing(train_eps, train_seeds, seed=2407)
    val_eps_shuffled = shuffle_robot_pairing(val_eps, val_seeds, seed=2408)

    train_rows = extract_transitions(train_eps_shuffled, dt=0.25)
    val_rows = extract_transitions(val_eps_shuffled, dt=0.25)
    print(f"[order9s_shuffle] {len(train_rows)} train / {len(val_rows)} validation transitions (shuffled pairing)")

    max_speed, max_accel = freeze_physical_caps(train_rows)
    config = BayesianModelConfig(k_candidates=(1, 2, 3, 4, 5, 6), max_human_speed=max_speed, max_human_acceleration=max_accel)

    artifact, reports, fitted = fit_and_select(
        train_rows, val_rows, config, seed=2407, require_multimodal=False, min_nll_improvement=0.01, return_all_fits=True,
    )

    per_track_by_k = {r.K: per_track_nlls(val_rows, *fitted[r.K][:3]) for r in reports}
    by_seed_by_k = {k: track_nlls_by_suite_seed(pt, val_seeds) for k, pt in per_track_by_k.items()}

    print()
    print("=" * 70)
    print("ORDER 9S.3 -- SHUFFLED ROBOT-PAIRING NECESSITY TEST")
    print("=" * 70)
    result = {"per_k": {}}
    for r in reports:
        boot = None
        if r.K > 1:
            boot = fast_bootstrap_nll_ci(by_seed_by_k[r.K], by_seed_by_k[1], n_resamples=2000, seed=2407)
        result["per_k"][r.K] = {
            "eligible": r.eligible, "rejection_reasons": r.rejection_reasons,
            "held_out_nll": r.held_out_nll, "mode_counts": r.mode_counts,
            "max_predictive_similarity": r.max_predictive_similarity, "bootstrap": boot,
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
        print("RESULT: no K>1 eligible with SHUFFLED robot pairing -- advantage VANISHED as required.")
        result["shuffle_test_result"] = "ADVANTAGE_VANISHED_AS_EXPECTED"
    else:
        best_k = min(multimodal_eligible, key=lambda r: r.held_out_nll).K
        improvement = next(r for r in reports if r.K == 1).held_out_nll - next(r for r in reports if r.K == best_k).held_out_nll
        print(f"RESULT: K={best_k} STILL eligible with SHUFFLED robot pairing (improvement={improvement:.4f} nats) -- "
              "advantage did NOT require correct robot conditioning. This is a FAIL for the interaction-necessity check.")
        result["shuffle_test_result"] = f"CONCERN_ADVANTAGE_PERSISTS_K_{best_k}"

    out_path = "runs/bayesian_brne/models/order9s_shuffle_test.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
