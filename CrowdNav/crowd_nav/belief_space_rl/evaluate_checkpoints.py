#!/usr/bin/env python3
"""Paired, fixed-case evaluation of baseline and belief-space checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_space_rl.model import BeliefSpaceQNetwork  # noqa: E402
from crowd_nav.belief_space_rl.runtime import (  # noqa: E402
    BeliefFeatureEngine,
    build_frozen_mamba,
    merged_policy_config,
)
from crowd_nav.belief_space_rl.train import (  # noqa: E402
    evaluate,
    evaluate_mamba_baseline,
)


def paired_counts(baseline, candidate):
    baseline_outcomes = baseline["episode_outcomes"]
    candidate_outcomes = candidate["episode_outcomes"]
    return {
        "both_success": sum(
            left == "success" and right == "success"
            for left, right in zip(baseline_outcomes, candidate_outcomes)
        ),
        "candidate_only_success": sum(
            left != "success" and right == "success"
            for left, right in zip(baseline_outcomes, candidate_outcomes)
        ),
        "baseline_only_success": sum(
            left == "success" and right != "success"
            for left, right in zip(baseline_outcomes, candidate_outcomes)
        ),
        "candidate_avoided_baseline_collision": sum(
            left == "collision" and right != "collision"
            for left, right in zip(baseline_outcomes, candidate_outcomes)
        ),
        "candidate_added_collision": sum(
            left != "collision" and right == "collision"
            for left, right in zip(baseline_outcomes, candidate_outcomes)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        default="runs/belief_space_rl_pilot_20260730_full",
    )
    parser.add_argument(
        "--output",
        default=(
            "runs/belief_space_rl_pilot_20260730_full/"
            "paired_evaluation.json"
        ),
    )
    parser.add_argument("--policy_config", default="configs/policy.config")
    parser.add_argument("--base_env_config", default="configs/env.config")
    parser.add_argument("--env_config", default="configs/env_gdbn.config")
    parser.add_argument(
        "--base_checkpoint",
        default="runs/mamba_vl/rl_model_ep10000_T24.pth",
    )
    parser.add_argument(
        "--gdbn_params",
        default=(
            "runs/bayesian_belief_pilot_20260730_full/"
            "gate_protocol_aware/selected_gdbn"
        ),
    )
    parser.add_argument("--eval_profile", default="decision_stress")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--nominal_seed_offset", type=int, default=810000)
    parser.add_argument("--stress_seed_offset", type=int, default=910000)
    parser.add_argument(
        "--checkpoints",
        default="best_model.pth,final_model.pth",
        help="Comma-separated checkpoint filenames within run_dir.",
    )
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--risk_samples", type=int, default=24)
    parser.add_argument("--risk_horizon", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = merged_policy_config(
        args.policy_config,
        args.base_env_config,
    )
    mamba = build_frozen_mamba(
        config,
        args.base_checkpoint,
        device,
    )
    probe = BeliefFeatureEngine(
        mamba,
        args.gdbn_params,
        device,
        particles=args.particles,
        risk_samples=args.risk_samples,
        risk_horizon=args.risk_horizon,
        seed=args.seed,
    )
    fixed_offsets = {
        "nominal": args.nominal_seed_offset,
        "stress": args.stress_seed_offset,
    }
    baseline = {
        "nominal": evaluate_mamba_baseline(
            mamba,
            args,
            "nominal",
            args.episodes,
            fixed_offsets["nominal"],
        ),
        "stress": evaluate_mamba_baseline(
            mamba,
            args,
            args.eval_profile,
            args.episodes,
            fixed_offsets["stress"],
        ),
    }
    result = {
        "episodes_per_protocol": args.episodes,
        "fixed_seed_offsets": fixed_offsets,
        "baseline": baseline,
        "checkpoints": {},
    }
    checkpoint_names = [
        name.strip()
        for name in args.checkpoints.split(",")
        if name.strip()
    ]
    for name in checkpoint_names:
        checkpoint_path = Path(args.run_dir) / name
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        network = BeliefSpaceQNetwork(
            context_dim=256,
            belief_dim=probe.belief_dim,
            candidate_dim=probe.candidate_dim,
        ).to(device)
        network.load_state_dict(checkpoint["network"])
        candidate = {
            "nominal": evaluate(
                network,
                mamba,
                args,
                device,
                "nominal",
                args.episodes,
                fixed_offsets["nominal"],
            ),
            "stress": evaluate(
                network,
                mamba,
                args,
                device,
                args.eval_profile,
                args.episodes,
                fixed_offsets["stress"],
            ),
        }
        result["checkpoints"][name] = {
            **candidate,
            "paired_nominal": paired_counts(
                baseline["nominal"],
                candidate["nominal"],
            ),
            "paired_stress": paired_counts(
                baseline["stress"],
                candidate["stress"],
            ),
        }
        print(
            f"[PAIRED] {name}: "
            f"nominal={candidate['nominal']['SR']:.1%}/"
            f"{candidate['nominal']['CR']:.1%}/"
            f"{candidate['nominal']['TR']:.1%}, "
            f"stress={candidate['stress']['SR']:.1%}/"
            f"{candidate['stress']['CR']:.1%}/"
            f"{candidate['stress']['TR']:.1%}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[PAIRED] wrote {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
