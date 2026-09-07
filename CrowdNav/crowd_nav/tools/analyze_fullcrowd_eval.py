#!/usr/bin/env python3
"""Analyze paired full-crowd evaluation blocks for paper reporting."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


METRICS = ("success", "collision", "timeout")


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (int(row["seed"]), row["case"]): row
            for row in csv.DictReader(stream)
        }


def read_many(value):
    rows = {}
    for item in str(value).split(","):
        path = Path(item.strip())
        if not item.strip():
            continue
        current = read_rows(path)
        overlap = rows.keys() & current.keys()
        if overlap:
            raise RuntimeError(
                f"Duplicate seed/scenario rows across input CSVs: {overlap}"
            )
        rows.update(current)
    return rows


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values, samples, seed):
    rng = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(rng.choice(values) for _ in range(count))
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize(proposed, baseline, bootstrap_samples, bootstrap_seed):
    if proposed.keys() != baseline.keys():
        raise RuntimeError("Proposed and baseline CSV keys do not match")

    keys = sorted(proposed)
    scenarios = sorted({case for _, case in keys})
    blocks = sorted({seed for seed, _ in keys})
    result = {
        "paired_blocks": len(blocks),
        "scenarios": len(scenarios),
        "episodes_per_method": sum(
            int(proposed[key]["episodes"]) for key in keys
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "overall": {},
        "per_scenario": {},
    }

    for metric_index, metric in enumerate(METRICS):
        proposed_total = sum(int(proposed[key][metric]) for key in keys)
        baseline_total = sum(int(baseline[key][metric]) for key in keys)
        episodes = result["episodes_per_method"]
        block_differences = []
        for block in blocks:
            block_keys = [key for key in keys if key[0] == block]
            block_episodes = sum(
                int(proposed[key]["episodes"]) for key in block_keys
            )
            difference = sum(
                int(proposed[key][metric]) - int(baseline[key][metric])
                for key in block_keys
            )
            block_differences.append(100.0 * difference / block_episodes)
        ci_low, ci_high = bootstrap_mean_ci(
            block_differences,
            bootstrap_samples,
            bootstrap_seed + metric_index,
        )
        proposed_rate = 100.0 * proposed_total / episodes
        baseline_rate = 100.0 * baseline_total / episodes
        result["overall"][metric] = {
            "proposed_count": proposed_total,
            "baseline_count": baseline_total,
            "proposed_rate_percent": proposed_rate,
            "baseline_rate_percent": baseline_rate,
            "paired_difference_pp": proposed_rate - baseline_rate,
            "paired_bootstrap_95_ci_pp": [ci_low, ci_high],
            "improved_blocks": sum(
                difference > 0.0
                if metric == "success"
                else difference < 0.0
                for difference in block_differences
            ),
        }
        if metric == "collision" and baseline_total:
            result["overall"][metric]["relative_reduction_percent"] = (
                100.0 * (baseline_total - proposed_total) / baseline_total
            )

    for scenario_index, scenario in enumerate(scenarios):
        scenario_keys = [key for key in keys if key[1] == scenario]
        scenario_result = {}
        for metric_index, metric in enumerate(METRICS):
            proposed_values = [
                int(proposed[key][metric]) for key in scenario_keys
            ]
            baseline_values = [
                int(baseline[key][metric]) for key in scenario_keys
            ]
            differences = [
                float(left - right)
                for left, right in zip(proposed_values, baseline_values)
            ]
            ci_low, ci_high = bootstrap_mean_ci(
                differences,
                bootstrap_samples,
                bootstrap_seed + 10 + scenario_index * 3 + metric_index,
            )
            scenario_result[metric] = {
                "proposed_rate_percent": statistics.fmean(proposed_values),
                "baseline_rate_percent": statistics.fmean(baseline_values),
                "paired_difference_pp": statistics.fmean(differences),
                "paired_bootstrap_95_ci_pp": [ci_low, ci_high],
            }
        result["per_scenario"][scenario] = scenario_result
    return result


def render_text(result):
    lines = [
        "Full-crowd paired evaluation",
        (
            f"blocks={result['paired_blocks']} "
            f"scenarios={result['scenarios']} "
            f"episodes_per_method={result['episodes_per_method']}"
        ),
        "",
        "Overall:",
    ]
    for metric in METRICS:
        item = result["overall"][metric]
        low, high = item["paired_bootstrap_95_ci_pp"]
        suffix = ""
        if metric == "collision":
            suffix = (
                f", relative reduction="
                f"{item['relative_reduction_percent']:.2f}%"
            )
        lines.append(
            f"  {metric}: proposed={item['proposed_rate_percent']:.2f}% "
            f"({item['proposed_count']}), "
            f"baseline={item['baseline_rate_percent']:.2f}% "
            f"({item['baseline_count']}), "
            f"delta={item['paired_difference_pp']:+.2f} pp, "
            f"paired bootstrap 95% CI=[{low:+.2f}, {high:+.2f}] pp"
            f"{suffix}"
        )
    lines.extend(["", "Per scenario (proposed / baseline / delta pp):"])
    for scenario, metrics in result["per_scenario"].items():
        values = []
        for metric in METRICS:
            item = metrics[metric]
            values.append(
                f"{metric}={item['proposed_rate_percent']:.2f}/"
                f"{item['baseline_rate_percent']:.2f}/"
                f"{item['paired_difference_pp']:+.2f}"
            )
        lines.append(f"  {scenario}: " + ", ".join(values))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposed",
        default=(
            "runs/eval35_fullcrowd_independent/"
            "bayesian_fullcrowd_independent_s1_10.csv"
        ),
    )
    parser.add_argument(
        "--baseline",
        default=(
            "runs/eval35_fullcrowd_independent/"
            "mamba_value_base_independent_s1_10.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="runs/eval35_fullcrowd_independent/statistical_report",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = summarize(
        read_many(args.proposed),
        read_many(args.baseline),
        args.bootstrap_samples,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    text = render_text(result)
    output.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
