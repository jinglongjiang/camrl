#!/usr/bin/env python3
"""Independent, high-power re-audit of a single Stage 1/DAgger gate decision (Round 13).

Why this exists: seed 3407's Round 12 run missed the Stage 1 gate at its
final DAgger round by a small margin (stress SR 96.67% vs the required
>=97%, stress TR 3.33% vs the required <=2% -- roughly 4 extra timeouts out
of 120 validation episodes). 120 episodes is not enough statistical power to
tell a real regression apart from sampling noise at a boundary that close.
This script re-evaluates a single checkpoint (any `checkpoint_dagger_round*
.pth`, `checkpoint_ep*.pth`, or other Stage 1-era checkpoint -- read-only,
never modifies weights) against the frozen Mamba-VL baseline on a much
larger, independent sample (>=500 episodes/profile by default), reusing the
same paired-seed-block bootstrap machinery evaluate.py's formal six-scenario
test uses.

Restricted to `runtime.VALIDATION_SCENARIOS` (baseline_circle, the 5-person
scenario training itself uses) and `AUDIT_PROFILES` (nominal,
train_nonstationary) -- never the six-scenario battery or
heldout_nonstationary, which are reserved entirely for evaluate.py's
one-time formal test of the frozen, final, selected model. Using the formal
distribution here to re-litigate a Stage 1 gate decision would be exactly
the kind of test-distribution leakage Round 10 already fixed once (see
train.py's module docstring).

This script decides nothing about which checkpoint to keep -- it only
answers "was the original 120-episode gate call, on this specific
checkpoint, likely correct." Run once per checkpoint under audit (e.g. once
for each of the three seeds' final DAgger-round checkpoint):

    python3 -u belief_mdp/stage1_gate_audit.py \\
        --checkpoint runs/<run_dir>/checkpoint_dagger_round5.pth \\
        --seeds 5 --episodes_per_seed 100 \\
        --output runs/<run_dir>/stage1_gate_audit.json \\
        --episode_records_output runs/<run_dir>/stage1_gate_audit_records.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_mdp.evaluate import (  # noqa: E402
    RECORD_FIELDS,
    block_bootstrap_ci,
    build_network_from_checkpoint,
    resolve_params,
    run_belief_mdp_episode,
    run_mamba_baseline_episode,
    verify_artifact_hashes,
)
from crowd_nav.belief_mdp.hashing import sha256_file  # noqa: E402
from crowd_nav.belief_mdp.runtime import (  # noqa: E402
    BeliefMDPFeatureEngine,
    DEFAULT_K1_GDBN_PARAMS,
    FullCrowdNavigationEnvironment,
    VALIDATION_SCENARIOS,
)
from crowd_nav.belief_space_rl.runtime import build_frozen_mamba, merged_policy_config  # noqa: E402

# Never heldout_nonstationary (reserved for the one-time formal test) and
# never any of the other five SIX_SCENARIOS entries (10-20 people) -- see
# the module docstring.
AUDIT_PROFILES = ("nominal", "train_nonstationary")
# Disjoint from train.py's quick-eval seeds (810000/910000 offsets),
# select_checkpoint.py's VALIDATION_SEED_BASE (2,000,000), and evaluate.py's
# formal test seed base (5,000,000).
AUDIT_SEED_BASE_DEFAULT = 3_000_000
MIN_EPISODES_PER_PROFILE = 500


def evaluate_profile_audit(
    network, mamba, make_engine, args, device, profile, seeds, episodes_per_seed, max_steps,
    record_writer,
):
    """Same paired candidate-vs-frozen-baseline design as evaluate.py's
    evaluate_profile (same seed, same test_case, for both methods every
    episode), restricted to VALIDATION_SCENARIOS instead of the six-scenario
    battery -- this is what makes it a Stage 1 gate audit rather than a
    (partial, single-profile) formal test.
    """
    candidate_outcomes = []
    baseline_outcomes = []
    seed_diff_blocks = {"success": [], "collision": [], "timeout": []}

    for seed_index in range(seeds):
        seed = args.audit_seed_base + seed_index * 1_000_000
        seed_candidate = []
        seed_baseline = []
        for scenario in VALIDATION_SCENARIOS:
            for episode_index in range(episodes_per_seed):
                test_case = (seed_index * episodes_per_seed + episode_index) % 9000

                engine = make_engine()
                environment = FullCrowdNavigationEnvironment(args.env_config, scenario, robot_visible=False)
                outcome, steps, min_dmin = run_belief_mdp_episode(
                    network, engine, environment, seed, profile, test_case, device, max_steps
                )
                candidate_outcomes.append(outcome)
                seed_candidate.append(outcome)
                if record_writer is not None:
                    record_writer.writerow({
                        "checkpoint": args.checkpoint, "method": "belief_mdp",
                        "scenario": scenario, "profile": profile, "seed": seed,
                        "test_case": test_case, "outcome": outcome, "steps": steps,
                        "min_dmin": min_dmin,
                    })

                baseline_environment = FullCrowdNavigationEnvironment(
                    args.env_config, scenario, robot_visible=False
                )
                outcome_b, steps_b, min_dmin_b = run_mamba_baseline_episode(
                    mamba, baseline_environment, seed, profile, test_case, max_steps
                )
                baseline_outcomes.append(outcome_b)
                seed_baseline.append(outcome_b)
                if record_writer is not None:
                    record_writer.writerow({
                        "checkpoint": args.checkpoint, "method": "frozen_mamba",
                        "scenario": scenario, "profile": profile, "seed": seed,
                        "test_case": test_case, "outcome": outcome_b, "steps": steps_b,
                        "min_dmin": min_dmin_b,
                    })

        for metric in ("success", "collision", "timeout"):
            candidate_hits = np.array([1.0 if o == metric else 0.0 for o in seed_candidate])
            baseline_hits = np.array([1.0 if o == metric else 0.0 for o in seed_baseline])
            seed_diff_blocks[metric].append(candidate_hits - baseline_hits)

    def rates(outcomes):
        total = max(len(outcomes), 1)
        return {
            "SR": outcomes.count("success") / total,
            "CR": outcomes.count("collision") / total,
            "TR": outcomes.count("timeout") / total,
            "episodes": total,
        }

    return rates(candidate_outcomes), rates(baseline_outcomes), seed_diff_blocks


def non_inferior(candidate: dict, baseline: dict, sr_slack: float, cr_slack: float, tr_slack: float) -> bool:
    """Same paired non-inferiority rule train.py's Stage 1 gate uses
    (run_stage1_gate_check's non_inferior helper) -- the audit re-checks
    the *same* criterion at higher statistical power, it does not loosen it."""
    return (
        candidate["SR"] >= baseline["SR"] - sr_slack
        and candidate["CR"] <= baseline["CR"] + cr_slack
        and candidate["TR"] <= baseline["TR"] + tr_slack
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    # All of these default to None: None means "use the checkpoint's own
    # training-time value." An explicit, different value is a hard error
    # unless --allow_feature_override is also passed -- see resolve_params.
    parser.add_argument("--policy_config", default=None)
    parser.add_argument("--base_env_config", default=None)
    parser.add_argument("--env_config", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--gdbn_params", default=None)
    parser.add_argument("--k1_gdbn_params", default=None)
    parser.add_argument(
        "--belief_mode", default=None,
        choices=(None, "action_conditioned", "cv", "state_only", "corrupted", "no_belief", "k1_belief"),
    )
    parser.add_argument("--gdbn_K", type=int, default=None)
    parser.add_argument("--num_humans", type=int, default=None)
    parser.add_argument("--risk_horizon", type=int, default=None)
    parser.add_argument("--safe_distance", type=float, default=None)
    parser.add_argument("--cvar_alpha", type=float, default=None)
    parser.add_argument("--pedestrian_aggregation", default=None)
    parser.add_argument(
        "--allow_feature_override", action="store_true",
        help="Permit CLI values / changed file content that disagree with the "
        "checkpoint's own training-time record. Without this flag, any "
        "disagreement is a hard error.",
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument(
        "--audit_seed_base", type=int, default=AUDIT_SEED_BASE_DEFAULT,
        help="Disjoint from train.py's quick-eval, select_checkpoint.py's "
        "VALIDATION_SEED_BASE, and evaluate.py's formal test seed base.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes_per_seed", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--bootstrap_replicates", type=int, default=100_000)
    parser.add_argument(
        "--sr_slack", type=float, default=0.03,
        help="Same non-inferiority slack as train.py's --gate_sr_slack -- the audit "
        "re-checks the same bar at higher power, it does not loosen it.",
    )
    parser.add_argument("--cr_slack", type=float, default=0.02)
    parser.add_argument("--tr_slack", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--episode_records_output", default=None,
        help="CSV path to append per-episode records.",
    )
    args = parser.parse_args()

    total_episodes = args.seeds * args.episodes_per_seed
    if total_episodes < MIN_EPISODES_PER_PROFILE:
        raise SystemExit(
            f"[AUDIT] refusing to run: --seeds x --episodes_per_seed = {total_episodes} "
            f"< {MIN_EPISODES_PER_PROFILE} required episodes per profile. This audit exists "
            f"specifically because 120 episodes was not enough statistical power -- running "
            f"it with another too-small sample would not answer the question."
        )

    checkpoint_blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint_blob.get("args", {})
    saved_hashes = checkpoint_blob.get("artifact_hashes", {})
    if not saved_hashes:
        print(
            "[AUDIT] WARNING: checkpoint has no artifact_hashes recorded "
            "(trained before this check existed) -- hash verification skipped.",
            flush=True,
        )
    for name, key in (
        ("train.py", "train_py_sha256"), ("model.py", "model_py_sha256"),
        ("runtime.py", "runtime_py_sha256"), ("evaluate.py", "evaluate_py_sha256"),
    ):
        saved_source_hash = saved_hashes.get(key)
        if saved_source_hash is None:
            continue
        current_source_hash = sha256_file(str(THIS_DIR / name))
        if current_source_hash != saved_source_hash:
            print(
                f"[AUDIT] WARNING: {name} has changed since this checkpoint was trained "
                f"(trained hash {saved_source_hash[:12]}..., current hash {current_source_hash[:12]}...) "
                "-- results reflect the checkpoint's weights under the *current* code.",
                flush=True,
            )

    resolved = resolve_params(args, saved_args, args.allow_feature_override)
    print(f"[AUDIT] resolved params: {resolved}", flush=True)
    verify_artifact_hashes(resolved, saved_hashes, args.allow_feature_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = merged_policy_config(resolved["policy_config"], resolved["base_env_config"])
    mamba = build_frozen_mamba(config, resolved["base_checkpoint"], device)
    belief_mode = resolved["belief_mode"]

    def make_engine():
        return BeliefMDPFeatureEngine(
            mamba,
            resolved["gdbn_params"],
            device,
            belief_mode=belief_mode,
            K=resolved["gdbn_K"],
            n_particles=resolved["particles"],
            num_humans=resolved["num_humans"],
            risk_horizon=resolved["risk_horizon"],
            safe_distance=resolved["safe_distance"],
            cvar_alpha=resolved["cvar_alpha"],
            pedestrian_aggregation=resolved["pedestrian_aggregation"],
            seed=args.seed,
            k1_gdbn_params=resolved["k1_gdbn_params"] or DEFAULT_K1_GDBN_PARAMS,
        )

    args.env_config = resolved["env_config"]

    probe = make_engine()
    network = build_network_from_checkpoint(checkpoint_blob, probe, device)
    print(
        f"[AUDIT] checkpoint={args.checkpoint} belief_mode={belief_mode} "
        f"beta={saved_args.get('beta')} scenarios={VALIDATION_SCENARIOS} "
        f"profiles={AUDIT_PROFILES} episodes_per_profile={total_episodes} "
        f"slack=SR-{args.sr_slack:.0%}/CR+{args.cr_slack:.0%}/TR+{args.tr_slack:.0%}",
        flush=True,
    )

    record_file = None
    record_writer = None
    if args.episode_records_output:
        record_path = Path(args.episode_records_output)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not record_path.exists()
        record_file = record_path.open("a", newline="", encoding="utf-8")
        record_writer = csv.DictWriter(record_file, fieldnames=RECORD_FIELDS)
        if is_new:
            record_writer.writeheader()

    rng = np.random.default_rng(args.seed + 91)
    profile_summaries = {}
    overall_pass = True
    try:
        for profile in AUDIT_PROFILES:
            candidate_rates, baseline_rates, seed_diff_blocks = evaluate_profile_audit(
                network, mamba, make_engine, args, device, profile,
                args.seeds, args.episodes_per_seed, args.max_steps, record_writer,
            )
            bootstrap = {}
            for metric in ("success", "collision", "timeout"):
                mean_diff, low, high = block_bootstrap_ci(
                    seed_diff_blocks[metric], args.bootstrap_replicates, rng
                )
                bootstrap[metric] = {"mean_diff": mean_diff, "ci_low": low, "ci_high": high}
            passed = non_inferior(candidate_rates, baseline_rates, args.sr_slack, args.cr_slack, args.tr_slack)
            overall_pass = overall_pass and passed
            profile_summaries[profile] = {
                "candidate": candidate_rates,
                "baseline": baseline_rates,
                "paired_seed_block_bootstrap_diff": bootstrap,
                "non_inferior": passed,
            }
            print(
                f"[AUDIT] profile={profile} {'PASS' if passed else 'FAIL'} "
                f"candidate={candidate_rates['SR']:.2%}/{candidate_rates['CR']:.2%}/{candidate_rates['TR']:.2%} "
                f"baseline={baseline_rates['SR']:.2%}/{baseline_rates['CR']:.2%}/{baseline_rates['TR']:.2%} "
                f"SR diff CI=[{bootstrap['success']['ci_low']:+.3f}, {bootstrap['success']['ci_high']:+.3f}] "
                f"CR diff CI=[{bootstrap['collision']['ci_low']:+.3f}, {bootstrap['collision']['ci_high']:+.3f}] "
                f"TR diff CI=[{bootstrap['timeout']['ci_low']:+.3f}, {bootstrap['timeout']['ci_high']:+.3f}]",
                flush=True,
            )
    finally:
        if record_file is not None:
            record_file.close()

    summary = {
        "checkpoint": args.checkpoint,
        "belief_mode": belief_mode,
        "beta": saved_args.get("beta"),
        "scenarios": list(VALIDATION_SCENARIOS),
        "profiles": AUDIT_PROFILES,
        "episodes_per_profile": total_episodes,
        "sr_slack": args.sr_slack, "cr_slack": args.cr_slack, "tr_slack": args.tr_slack,
        "gate_passed": overall_pass,
        "profile_results": profile_summaries,
    }
    print(f"\n[AUDIT] OVERALL: {'PASS' if overall_pass else 'FAIL'} (gate_passed={overall_pass})", flush=True)
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
