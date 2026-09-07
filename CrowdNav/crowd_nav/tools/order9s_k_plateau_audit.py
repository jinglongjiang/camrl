#!/usr/bin/env python3
"""Order 9S.1: model-order (K) plateau audit on fresh, never-viewed audit
suite seeds.

Order 9's original K=1..6 search hit its own upper bound (K=6) with NLL
still improving -- consistent with either (a) real structure beyond 6
modes, or (b) the fitting code simply recovering the 6 hand-scripted
behavior types built into the synthetic generator, with no informative
upper bound ever tested. This script collects NEW data with suite seeds
never used in Order 7/8/9 (pilot: 1001/1002/2001/2002; formal train:
11-15; formal validation: 21-25), fits K=1..10, and reports whether NLL
plateaus (one-standard-error / minimal-sufficient-K rule) or keeps
climbing to the new boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from crowd_nav.bayesian_brne import collect_dataset
from crowd_nav.bayesian_brne.config import BayesianModelConfig
from crowd_nav.bayesian_brne.mode_model import extract_transitions, fit_and_select
from crowd_nav.tools.fit_bayesian_brne import (
    fast_bootstrap_nll_ci,
    freeze_physical_caps,
    load_formal_episodes,
    per_track_nlls,
    track_nlls_by_suite_seed,
)

AUDIT_TRAIN_SEEDS = [301, 302, 303]
AUDIT_VAL_SEEDS = [401, 402, 403]
EPISODES_PER_SEED_TRAIN = 300
EPISODES_PER_SEED_VAL = 80
DATA_DIR = "runs/bayesian_brne/data_audit"


def collect_audit_data() -> None:
    print(f"[order9s_k_plateau] collecting FRESH audit data (seeds never used before: "
          f"train={AUDIT_TRAIN_SEEDS}, validation={AUDIT_VAL_SEEDS}) ...")
    for seed in AUDIT_TRAIN_SEEDS:
        collect_dataset.collect(
            split="train", scenario="baseline_circle", episodes=EPISODES_PER_SEED_TRAIN, seed=seed,
            profile_name="formal", output_dir=DATA_DIR, horizon_steps=40, dt=0.25,
        )
    for seed in AUDIT_VAL_SEEDS:
        collect_dataset.collect(
            split="validation", scenario="baseline_circle", episodes=EPISODES_PER_SEED_VAL, seed=seed,
            profile_name="formal", output_dir=DATA_DIR, horizon_steps=40, dt=0.25,
        )


def main() -> None:
    if not Path(DATA_DIR).exists():
        collect_audit_data()
    else:
        print(f"[order9s_k_plateau] reusing existing {DATA_DIR}")

    train_eps, train_seeds = load_formal_episodes(DATA_DIR, "train", "baseline_circle")
    val_eps, val_seeds = load_formal_episodes(DATA_DIR, "validation", "baseline_circle")
    print(f"[order9s_k_plateau] {len(train_eps)} train episodes ({len(set(train_seeds))} suite seeds), "
          f"{len(val_eps)} validation episodes ({len(set(val_seeds))} suite seeds)")
    assert set(train_seeds) == set(AUDIT_TRAIN_SEEDS), "train seeds must be exactly the fresh audit seeds"
    assert set(val_seeds) == set(AUDIT_VAL_SEEDS), "validation seeds must be exactly the fresh audit seeds"
    assert not (set(train_seeds) & set(val_seeds)), "audit train/validation suite seeds must not overlap"

    train_rows = extract_transitions(train_eps, dt=0.25)
    val_rows = extract_transitions(val_eps, dt=0.25)
    print(f"[order9s_k_plateau] {len(train_rows)} train / {len(val_rows)} validation transitions")

    max_speed, max_accel = freeze_physical_caps(train_rows)
    k_candidates = tuple(range(1, 11))
    config = BayesianModelConfig(k_candidates=k_candidates, max_human_speed=max_speed, max_human_acceleration=max_accel)

    print(f"[order9s_k_plateau] fitting K={k_candidates} ...")
    artifact, reports, fitted = fit_and_select(
        train_rows, val_rows, config, seed=2407, require_multimodal=False, min_nll_improvement=0.01, return_all_fits=True,
    )

    per_track_by_k = {r.K: per_track_nlls(val_rows, *fitted[r.K][:3]) for r in reports}
    by_seed_by_k = {k: track_nlls_by_suite_seed(pt, val_seeds) for k, pt in per_track_by_k.items()}

    print()
    print("=" * 70)
    print("ORDER 9S.1 -- K-ORDER PLATEAU AUDIT (fresh, never-viewed seeds, K=1..10)")
    print("=" * 70)
    result = {"per_k": {}, "audit_train_seeds": AUDIT_TRAIN_SEEDS, "audit_validation_seeds": AUDIT_VAL_SEEDS}
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

    # One-standard-error / minimal-sufficient-K: find the SMALLEST eligible K
    # whose pairwise improvement CI over the NEXT larger eligible K does NOT
    # exclude zero (i.e. going bigger buys no statistically real gain).
    eligible_ks = sorted(r.K for r in reports if r.eligible and r.K > 1)
    plateau_k = None
    for i, k in enumerate(eligible_ks):
        larger_ks = eligible_ks[i + 1:]
        if not larger_ks:
            plateau_k = k
            break
        no_further_gain = True
        for bigger_k in larger_ks:
            pt_big = per_track_nlls(val_rows, *fitted[bigger_k][:3])
            by_seed_big = track_nlls_by_suite_seed(pt_big, val_seeds)
            comp = fast_bootstrap_nll_ci(by_seed_big, by_seed_by_k[k], n_resamples=2000, seed=2407)
            if comp["ci_supports_real_improvement"]:
                no_further_gain = False
                break
        if no_further_gain:
            plateau_k = k
            break

    print()
    if plateau_k is not None:
        print(f"PLATEAU FOUND at K={plateau_k}: no larger eligible K gives a statistically real further NLL improvement.")
        result["plateau_result"] = f"PLATEAU_AT_K_{plateau_k}"
    elif eligible_ks and eligible_ks[-1] == k_candidates[-1]:
        print(f"NO PLATEAU: NLL keeps improving all the way to the tested boundary K={k_candidates[-1]}.")
        result["plateau_result"] = f"NO_PLATEAU_HIT_BOUNDARY_K_{k_candidates[-1]}"
    else:
        result["plateau_result"] = "NO_ELIGIBLE_K_GT_1"
        print("No K>1 candidate was eligible at all on the audit data.")

    out_path = "runs/bayesian_brne/models/order9s_k_plateau_audit.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
