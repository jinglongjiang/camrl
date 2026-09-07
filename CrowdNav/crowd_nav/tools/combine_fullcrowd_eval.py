#!/usr/bin/env python3
"""Combine two disjoint 500-case evaluations into ten independent blocks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


CASES = (
    "baseline_circle",
    "baseline_square",
    "dense_circle",
    "dense_square",
    "large_circle",
    "large_square",
)

METHODS = (
    (
        "bayesian_fullcrowd",
        "bayesian_fullcrowd_s1_10.csv",
        "bayesian_fullcrowd_cases500_999.csv",
    ),
    (
        "mamba_value_base",
        "mamba_value_base_s1_10.csv",
        "mamba_value_base_cases500_999.csv",
    ),
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_rows(rows, seed_mapping):
    selected = []
    for row in rows:
        source_seed = int(row["seed"])
        if source_seed not in seed_mapping:
            continue
        output = dict(row)
        output["seed"] = seed_mapping[source_seed]
        selected.append(output)
    return selected


def validate(rows):
    keys = [(int(row["seed"]), row["case"]) for row in rows]
    expected = [
        (seed, case)
        for seed in range(1, 11)
        for case in CASES
    ]
    if sorted(keys) != sorted(expected):
        raise RuntimeError(
            "Combined evaluation does not contain ten complete blocks"
        )
    if any(int(row["episodes"]) != 100 for row in rows):
        raise RuntimeError("Expected 100 episodes in every block and scenario")


def mean(values):
    return statistics.fmean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows):
    lines = ["case,mean_SR,std_SR,mean_CR,mean_TR,n_blocks"]
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        sr = [float(row["success_rate"]) for row in case_rows]
        cr = [float(row["collision_rate"]) for row in case_rows]
        tr = [float(row["timeout_rate"]) for row in case_rows]
        lines.append(
            f"{case},{mean(sr):.2f},{stdev(sr):.2f},"
            f"{mean(cr):.2f},{mean(tr):.2f},{len(case_rows)}"
        )

    seed_rates = []
    for seed in range(1, 11):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        seed_rates.append(
            mean([float(row["success_rate"]) for row in seed_rows])
        )
    lines.extend(
        [
            "",
            (
                f"overall_mean_SR,{mean(seed_rates):.2f},"
                f"overall_std_across_blocks,{stdev(seed_rates):.2f},"
                f"n_blocks,{len(seed_rates)}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-root",
        default="runs/eval35_fullcrowd",
    )
    parser.add_argument(
        "--supplement-root",
        default="runs/eval35_fullcrowd_supplement",
    )
    parser.add_argument(
        "--output-root",
        default="runs/eval35_fullcrowd_independent",
    )
    args = parser.parse_args()

    initial_root = Path(args.initial_root)
    supplement_root = Path(args.supplement_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # The original 500-case pool produced blocks in the order
    # 100-199, 200-299, 300-399, 400-499, and 0-99.
    initial_mapping = {5: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    # With test_size=1000, seeds 5-9 produce cases 500-999.
    supplement_mapping = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10}

    for method, initial_name, supplement_name in METHODS:
        initial = select_rows(
            read_rows(initial_root / initial_name),
            initial_mapping,
        )
        supplement = select_rows(
            read_rows(supplement_root / supplement_name),
            supplement_mapping,
        )
        rows = initial + supplement
        validate(rows)
        rows.sort(
            key=lambda row: (
                int(row["seed"]),
                CASES.index(row["case"]),
            )
        )

        output = output_root / f"{method}_independent_s1_10.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)
        summary = output.with_suffix(".summary.txt")
        summary.write_text(summarize(rows), encoding="utf-8")
        print(f"[COMBINE] {method}: {len(rows)} rows -> {output}")
        print(summary.read_text(encoding="utf-8"), end="")

    provenance = output_root / "PROVENANCE.txt"
    provenance.write_text(
        "Independent cases 0-499 are retained from the first formal run. "
        "The exact duplicate blocks from its seeds 6-10 are discarded. "
        "Cases 500-999 come from the supplemental run with test_size=1000. "
        "Each final method therefore contains 1000 distinct cases per "
        "scenario, represented as ten 100-case blocks.\n",
        encoding="utf-8",
    )
    print(f"[COMBINE] provenance -> {provenance}")


if __name__ == "__main__":
    main()
