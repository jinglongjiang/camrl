#!/usr/bin/env python3
"""Head-to-head paired comparison between two belief-MDP checkpoints.

evaluate.py only ever compares a checkpoint against the frozen Mamba-VL
baseline; that supports "does belief_mdp beat the frozen policy" but not
"does action_conditioned beat cv/state_only/corrupted/no_belief" -- the
actual claim the ablation battery needs. This script reads the per-episode
CSV records both runs wrote via `evaluate.py --episode_records_output`,
joins them on (scenario, profile, seed, test_case) -- which is identical
across runs as long as both used the same --seeds/--episodes_per_seed/
--eval_seed_base -- and computes the direct paired difference with the same
seed-level block bootstrap evaluate.py uses.

Usage:
    python3 -u belief_mdp/compare_checkpoints.py \\
        --records runs/belief_mdp_eval_records.csv \\
        --checkpoint_a runs/belief_mdp_action_conditioned/best_model.pth \\
        --checkpoint_b runs/belief_mdp_cv/best_model.pth
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_records(path: str, checkpoint: str):
    rows = defaultdict(dict)
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["checkpoint"] != checkpoint or row["method"] != "belief_mdp":
                continue
            key = (row["scenario"], row["profile"], row["seed"], row["test_case"])
            rows[key] = row
    return rows


def block_bootstrap_ci(seed_blocks: list, replicates: int, rng: np.random.Generator):
    n_seeds = len(seed_blocks)
    if n_seeds == 0:
        return 0.0, 0.0, 0.0
    observed = np.concatenate(seed_blocks)
    means = np.empty(replicates, dtype=np.float64)
    seed_indices = rng.integers(0, n_seeds, size=(replicates, n_seeds))
    for replicate in range(replicates):
        resampled = [seed_blocks[i] for i in seed_indices[replicate]]
        means[replicate] = np.concatenate(resampled).mean()
    return (
        float(observed.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="CSV written by evaluate.py --episode_records_output")
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows_a = load_records(args.records, args.checkpoint_a)
    rows_b = load_records(args.records, args.checkpoint_b)
    common_keys = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    missing_a = set(rows_b.keys()) - set(rows_a.keys())
    missing_b = set(rows_a.keys()) - set(rows_b.keys())
    if not common_keys:
        raise SystemExit(
            f"No overlapping (scenario, profile, seed, test_case) episodes between "
            f"{args.checkpoint_a} ({len(rows_a)} rows) and {args.checkpoint_b} "
            f"({len(rows_b)} rows) in {args.records} -- were both evaluated with "
            f"identical --seeds/--episodes_per_seed/--eval_seed_base?"
        )
    if missing_a or missing_b:
        print(
            f"[COMPARE] warning: {len(missing_a)} episodes present only for "
            f"checkpoint_b, {len(missing_b)} present only for checkpoint_a -- "
            f"these are dropped from the paired comparison.",
            flush=True,
        )

    by_profile_seed = defaultdict(lambda: defaultdict(list))
    for scenario, profile, seed, test_case in common_keys:
        by_profile_seed[profile][seed].append((scenario, test_case))

    rng = np.random.default_rng(args.seed)
    results = {}
    for profile, seeds in by_profile_seed.items():
        seed_blocks = {"success": [], "collision": [], "timeout": []}
        pooled_a, pooled_b = [], []
        for seed, keys in seeds.items():
            hits_a = {"success": [], "collision": [], "timeout": []}
            hits_b = {"success": [], "collision": [], "timeout": []}
            for scenario, test_case in keys:
                key = (scenario, profile, seed, test_case)
                outcome_a = rows_a[key]["outcome"]
                outcome_b = rows_b[key]["outcome"]
                pooled_a.append(outcome_a)
                pooled_b.append(outcome_b)
                for metric in ("success", "collision", "timeout"):
                    hits_a[metric].append(1.0 if outcome_a == metric else 0.0)
                    hits_b[metric].append(1.0 if outcome_b == metric else 0.0)
            for metric in ("success", "collision", "timeout"):
                seed_blocks[metric].append(
                    np.array(hits_a[metric]) - np.array(hits_b[metric])
                )

        bootstrap = {}
        for metric in ("success", "collision", "timeout"):
            mean_diff, low, high = block_bootstrap_ci(
                seed_blocks[metric], args.bootstrap_replicates, rng
            )
            bootstrap[metric] = {"mean_diff": mean_diff, "ci_low": low, "ci_high": high}

        total = max(len(pooled_a), 1)
        results[profile] = {
            "episodes": total,
            "checkpoint_a_rates": {
                "SR": pooled_a.count("success") / total,
                "CR": pooled_a.count("collision") / total,
                "TR": pooled_a.count("timeout") / total,
            },
            "checkpoint_b_rates": {
                "SR": pooled_b.count("success") / total,
                "CR": pooled_b.count("collision") / total,
                "TR": pooled_b.count("timeout") / total,
            },
            "paired_seed_block_bootstrap_diff_a_minus_b": bootstrap,
        }
        print(
            f"[COMPARE] profile={profile} episodes={total} "
            f"A: SR={results[profile]['checkpoint_a_rates']['SR']:.1%} "
            f"CR={results[profile]['checkpoint_a_rates']['CR']:.1%} | "
            f"B: SR={results[profile]['checkpoint_b_rates']['SR']:.1%} "
            f"CR={results[profile]['checkpoint_b_rates']['CR']:.1%} | "
            f"SR diff CI=[{bootstrap['success']['ci_low']:+.3f}, "
            f"{bootstrap['success']['ci_high']:+.3f}]",
            flush=True,
        )

    summary = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "records": args.records,
        "profiles": results,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
