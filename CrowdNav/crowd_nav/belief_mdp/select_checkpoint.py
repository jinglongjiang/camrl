#!/usr/bin/env python3
"""Post-hoc model selection over a training run's periodic checkpoints.

train.py no longer picks a "best" checkpoint during training -- the old
mechanism scored each candidate on a noisy 30-episode quick-eval, which is
prone to picking a checkpoint that got lucky rather than one that is
actually good (confirmed: the first full run's picked "best" episode 1900
was the peak of a run that later degraded, not necessarily a stable point).

This script evaluates every Stage 2 RL checkpoint (`checkpoint_rl_ep*.pth`)
in an output directory on independent *validation* seeds -- distinct from
both train.py's own quick-eval seeds (offsets 810000/910000) and
evaluate.py's formal *test* seeds (default base 5,000,000) -- with at least
20 episodes per profile,
and selects the checkpoint with the best combined SR/CR that is not wildly
imbalanced between the nominal and train_nonstationary profiles. The
formal test seeds AND the six-scenario battery AND the heldout_nonstationary
profile in evaluate.py must never be used for this selection step (Round
10: an earlier version of this script validated on all six scenarios x
heldout_nonstationary, which is exactly the distribution the formal test
reports zero-shot generalization on -- selecting a checkpoint using
performance on that same distribution invalidates the generalization claim
regardless of the final numbers). Validation here is restricted to
`runtime.VALIDATION_SCENARIOS` (the 5-person scenario training itself uses)
and `("nominal", "train_nonstationary")`; the six-scenario/heldout_nonstationary
formal seeds are spent exactly once, on whichever checkpoint this script
selects.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_mdp.evaluate import (  # noqa: E402
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
from crowd_nav.belief_space_rl.runtime import (  # noqa: E402
    build_frozen_mamba,
    merged_policy_config,
)

VALIDATION_SEED_BASE = 2_000_000  # disjoint from train.py's quick-eval and evaluate.py's test seeds
# Never heldout_nonstationary and never the other five SIX_SCENARIOS entries
# (10-20 people) -- see the module docstring's Round 10 note.
VALIDATION_PROFILES = ("nominal", "train_nonstationary")
# Round 13 fix: this must match ONLY the Stage 2 periodic RL checkpoints
# (train.py's `checkpoint_rl_ep{episode}.pth`). The previous pattern,
# `checkpoint_ep(\d+)\.pth`, was written before the 5-way-buffer/DAgger
# redesign and only ever matches `checkpoint_ep{demo_episodes}.pth` (the
# single Stage 1b gate checkpoint, saved *before* DAgger or any RL
# training) -- it never matches `checkpoint_dagger_round*.pth` or
# `checkpoint_rl_ep*.pth` at all, because glob("checkpoint_ep*.pth") does
# not match filenames like "checkpoint_rl_ep200.pth" or
# "checkpoint_dagger_round1.pth" (the literal prefix differs). Running
# selection against a real Round 12-style run directory found exactly one
# "checkpoint" this way and silently selected it as "best" -- the Stage 1b
# checkpoint that hadn't even passed its own gate yet, not any of the 12
# real Stage 2 candidates.
CHECKPOINT_PATTERN = re.compile(r"checkpoint_rl_ep(\d+)\.pth$")


def find_checkpoints(run_dir: Path):
    checkpoints = []
    for path in run_dir.glob("checkpoint_rl_ep*.pth"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def evaluate_checkpoint(path, args, device):
    checkpoint_blob = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint_blob.get("args", {})
    saved_hashes = checkpoint_blob.get("artifact_hashes", {})

    resolved = resolve_params(args, saved_args, allow_override=False)
    verify_artifact_hashes(resolved, saved_hashes, allow_override=False)

    config = merged_policy_config(resolved["policy_config"], resolved["base_env_config"])
    mamba = build_frozen_mamba(config, resolved["base_checkpoint"], device)

    def make_engine():
        return BeliefMDPFeatureEngine(
            mamba,
            resolved["gdbn_params"],
            device,
            belief_mode=resolved["belief_mode"],
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

    probe = make_engine()
    network = build_network_from_checkpoint(checkpoint_blob, probe, device)

    profile_rates = {}
    for profile in VALIDATION_PROFILES:
        outcomes = []
        for scenario in VALIDATION_SCENARIOS:
            for episode_index in range(args.episodes_per_scenario):
                seed = VALIDATION_SEED_BASE + episode_index * 1000
                test_case = episode_index % 9000
                engine = make_engine()
                environment = FullCrowdNavigationEnvironment(
                    resolved["env_config"], scenario, robot_visible=False
                )
                outcome, _, _ = run_belief_mdp_episode(
                    network, engine, environment, seed, profile, test_case, device, args.max_steps
                )
                outcomes.append(outcome)
        total = max(len(outcomes), 1)
        profile_rates[profile] = {
            "SR": outcomes.count("success") / total,
            "CR": outcomes.count("collision") / total,
            "TR": outcomes.count("timeout") / total,
            "episodes": total,
        }
    return profile_rates


def evaluate_baseline(args, device, resolved):
    """Frozen Mamba-VL baseline evaluated on the exact same validation
    seeds/scenarios/profiles as every candidate checkpoint, so eligibility
    can be defined relative to it rather than an arbitrary fixed cap."""
    config = merged_policy_config(resolved["policy_config"], resolved["base_env_config"])
    mamba = build_frozen_mamba(config, resolved["base_checkpoint"], device)
    profile_rates = {}
    for profile in VALIDATION_PROFILES:
        outcomes = []
        for scenario in VALIDATION_SCENARIOS:
            for episode_index in range(args.episodes_per_scenario):
                seed = VALIDATION_SEED_BASE + episode_index * 1000
                test_case = episode_index % 9000
                environment = FullCrowdNavigationEnvironment(
                    resolved["env_config"], scenario, robot_visible=False
                )
                outcome, _, _ = run_mamba_baseline_episode(
                    mamba, environment, seed, profile, test_case, args.max_steps
                )
                outcomes.append(outcome)
        total = max(len(outcomes), 1)
        profile_rates[profile] = {
            "SR": outcomes.count("success") / total,
            "CR": outcomes.count("collision") / total,
            "TR": outcomes.count("timeout") / total,
            "episodes": total,
        }
    return profile_rates


def is_eligible(profile_rates: dict, baseline_rates: dict, sr_slack: float, cr_slack: float, tr_slack: float) -> bool:
    """Constrained filter relative to the frozen Mamba-VL baseline on the
    same validation episodes -- a fixed CR/TR cap has no way to know
    whether e.g. 15% CR is close to or far from what the existing strong
    policy achieves in the same scenarios. A checkpoint with a great
    composite number but degraded CR or TR relative to baseline on either
    profile is not a real candidate no matter how high its SR is. Selection
    among eligible checkpoints is by SR alone (see main()) -- a single
    weighted score can quietly trade safety for success in a way this
    filter does not allow.
    """
    for key in ("nominal", "train_nonstationary"):
        candidate, baseline = profile_rates[key], baseline_rates[key]
        if candidate["SR"] < baseline["SR"] - sr_slack:
            return False
        if candidate["CR"] > baseline["CR"] + cr_slack:
            return False
        if candidate["TR"] > baseline["TR"] + tr_slack:
            return False
    return True


def mean_sr(profile_rates: dict) -> float:
    nominal, stress = profile_rates["nominal"], profile_rates["train_nonstationary"]
    return 0.5 * (nominal["SR"] + stress["SR"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Directory containing checkpoint_ep*.pth files")
    parser.add_argument("--episodes_per_scenario", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2407)
    # Same PARAMS_FROM_CHECKPOINT-style resolution as evaluate.py; all None
    # by default so every checkpoint's own recorded config is used untouched.
    parser.add_argument("--belief_mode", default=None)
    parser.add_argument("--gdbn_params", default=None)
    parser.add_argument("--k1_gdbn_params", default=None)
    parser.add_argument("--gdbn_K", type=int, default=None)
    parser.add_argument("--num_humans", type=int, default=None)
    parser.add_argument("--risk_horizon", type=int, default=None)
    parser.add_argument("--safe_distance", type=float, default=None)
    parser.add_argument("--cvar_alpha", type=float, default=None)
    parser.add_argument("--pedestrian_aggregation", default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--policy_config", default=None)
    parser.add_argument("--base_env_config", default=None)
    parser.add_argument("--env_config", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--sr_slack", type=float, default=0.02,
        help="Eligible if candidate SR >= baseline SR - sr_slack, per profile.",
    )
    parser.add_argument(
        "--cr_slack", type=float, default=0.01,
        help="Eligible if candidate CR <= baseline CR + cr_slack, per profile.",
    )
    parser.add_argument(
        "--tr_slack", type=float, default=0.02,
        help="Eligible if candidate TR <= baseline TR + tr_slack, per profile.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoint_ep*.pth files found under {run_dir}")

    print(
        f"[SELECT] {len(checkpoints)} checkpoints found; validating each on "
        f"{args.episodes_per_scenario} episodes x {len(VALIDATION_SCENARIOS)} scenarios x "
        f"{len(VALIDATION_PROFILES)} profiles (validation seeds, never the formal test seeds)",
        flush=True,
    )

    first_blob = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    resolved = resolve_params(args, first_blob.get("args", {}), allow_override=False)
    verify_artifact_hashes(resolved, first_blob.get("artifact_hashes", {}), allow_override=False)
    print("[SELECT] evaluating frozen Mamba-VL baseline on the same validation episodes...", flush=True)
    baseline_rates = evaluate_baseline(args, device, resolved)
    baseline_nominal, baseline_stress = baseline_rates["nominal"], baseline_rates["train_nonstationary"]
    print(
        f"[SELECT] baseline: nominal={baseline_nominal['SR']:.1%}/{baseline_nominal['CR']:.1%}/"
        f"{baseline_nominal['TR']:.1%} stress={baseline_stress['SR']:.1%}/{baseline_stress['CR']:.1%}/"
        f"{baseline_stress['TR']:.1%}",
        flush=True,
    )

    results = []
    for episode, path in checkpoints:
        profile_rates = evaluate_checkpoint(path, args, device)
        eligible = is_eligible(profile_rates, baseline_rates, args.sr_slack, args.cr_slack, args.tr_slack)
        sr = mean_sr(profile_rates)
        results.append({
            "episode": episode, "path": str(path), "sha256": sha256_file(path),
            "eligible": eligible, "mean_sr": sr, "profiles": profile_rates,
        })
        nominal, stress = profile_rates["nominal"], profile_rates["train_nonstationary"]
        print(
            f"[SELECT] ep{episode}: eligible={eligible} mean_sr={sr:.1%} "
            f"nominal={nominal['SR']:.1%}/{nominal['CR']:.1%}/{nominal['TR']:.1%} "
            f"stress={stress['SR']:.1%}/{stress['CR']:.1%}/{stress['TR']:.1%}",
            flush=True,
        )

    eligible_results = [r for r in results if r["eligible"]]
    summary = {
        "run_dir": str(run_dir), "episodes_per_scenario": args.episodes_per_scenario,
        "sr_slack": args.sr_slack, "cr_slack": args.cr_slack, "tr_slack": args.tr_slack,
        "baseline": baseline_rates, "results": results,
    }
    output_path = Path(args.output) if args.output else run_dir / "checkpoint_selection.json"
    if not eligible_results:
        summary["selected"] = None
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit(
            f"[SELECT] FAILED: no checkpoint is within sr_slack={args.sr_slack:.0%}/"
            f"cr_slack={args.cr_slack:.0%}/tr_slack={args.tr_slack:.0%} of the frozen "
            f"Mamba-VL baseline on both profiles. Not writing selected_model.pth -- "
            f"there is no checkpoint from this run that should be used for the paper-"
            f"level ablation battery. Report written to {output_path}."
        )
    best = max(eligible_results, key=lambda r: r["mean_sr"])
    print(
        f"\n[SELECT] best: episode {best['episode']} (eligible={best['eligible']}, "
        f"mean_sr={best['mean_sr']:.1%}) -> {best['path']} (sha256 {best['sha256'][:16]}...)",
        flush=True,
    )

    selected_path = run_dir / "selected_model.pth"
    selected_path.write_bytes(Path(best["path"]).read_bytes())
    copied_hash = sha256_file(selected_path)
    if copied_hash != best["sha256"]:
        raise SystemExit(
            f"[SELECT] FAILED: {selected_path} content hash ({copied_hash[:16]}...) does not match "
            f"the source checkpoint's hash ({best['sha256'][:16]}...) after copying -- refusing to "
            "leave a corrupted selected_model.pth in place."
        )
    print(f"[SELECT] copied to {selected_path} (sha256 verified identical to source)", flush=True)

    summary["selected"] = best
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SELECT] report written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
