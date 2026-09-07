#!/usr/bin/env python3
"""Initialize full-crowd Bayesian risk calibration from a Mamba value model."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import sys

import torch


THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.policy.bayesian_fullcrowd_risk_value import (
    BayesianFullCrowdRiskValuePolicy,
)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    state = checkpoint.get(
        "policy_state",
        checkpoint.get(
            "model_state_dict",
            checkpoint.get("value_state", checkpoint.get("model", checkpoint)),
        ),
    )
    return {
        (
            key.replace("_orig_mod.", "", 1)
            if key.startswith("_orig_mod.")
            else key
        ): value
        for key, value in state.items()
    }


def load_config(args):
    config = configparser.RawConfigParser()
    loaded = config.read(
        [args.env_config, args.policy_config, args.train_config]
    )
    if len(loaded) != 3:
        raise FileNotFoundError(
            f"Expected three config files, loaded={loaded}"
        )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="runs/mamba_vl/rl_model_ep10000_T24.pth",
    )
    parser.add_argument("--env_config", default="configs/env.config")
    parser.add_argument(
        "--policy_config",
        default="configs/policy_bayesian_distributional.config",
    )
    parser.add_argument("--train_config", default="configs/train.config")
    parser.add_argument(
        "--output",
        default=(
            "runs/bayesian_distributional/"
            "model_fullcrowd_directrisk_lambda1_clean.pth"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "runs/bayesian_distributional/"
            "fullcrowd_risk_initialization_report.json"
        ),
    )
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )
    policy = BayesianFullCrowdRiskValuePolicy(
        config=load_config(args),
        device=device,
    )
    source_state = extract_state_dict(torch_load(args.source))
    result = policy.load_state_dict(source_state, strict=False)
    expected_missing = sorted(
        ["collision_penalty_scale", "penalty_cap"]
    )
    if sorted(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "Source checkpoint is not an exact Mamba base-policy match: "
            f"missing={sorted(result.missing_keys)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": {
                key: value.detach().cpu()
                for key, value in policy.state_dict().items()
            },
            "algo": "sarl",
            "stage": "bayesian_fullcrowd_risk_calibration",
            "source": str(Path(args.source).resolve()),
            "collision_penalty_scale": float(
                policy.collision_penalty_scale
            ),
            "penalty_cap": float(policy.penalty_cap),
        },
        output,
    )

    report = {
        "source": str(Path(args.source).resolve()),
        "output": str(output.resolve()),
        "base_keys": len(source_state),
        "initialized_buffers": expected_missing,
        "collision_penalty_scale": float(
            policy.collision_penalty_scale
        ),
        "penalty_cap": float(policy.penalty_cap),
        "bayesian_pedestrians": policy.belief_num_humans,
        "pedestrian_aggregation": policy.action_pedestrian_aggregation,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"[INIT] base checkpoint matched ({len(source_state)} keys)")
    print(f"[SAVE] {output}")
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
