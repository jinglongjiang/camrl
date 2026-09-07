#!/usr/bin/env python3
"""Summarize read-only top-5 occlusion slot audit CSV files."""
import argparse
import csv
import math
from collections import defaultdict


COUNT_FIELDS = (
    'n_hidden_modes_uncapped',
    'n_hidden_upstream_dropped',
    'n_hidden_candidates',
    'n_selected_hidden',
    'n_hidden_dropped',
    'n_near_hidden_candidates',
    'n_near_hidden_modes_uncapped',
    'n_near_hidden_upstream_dropped',
    'n_near_hidden_end_to_end_dropped',
    'n_near_hidden_selected',
    'n_near_hidden_dropped',
    'n_true_occluded',
    'n_near_true_occluded',
)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def fmt(value):
    return 'n/a' if not math.isfinite(value) else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='+')
    args = parser.parse_args()

    groups = defaultdict(lambda: {
        'steps': 0,
        'hidden_steps': 0,
        'near_hidden_steps': 0,
        'hidden_no_slot_steps': 0,
        'near_hidden_no_slot_steps': 0,
        'all_visible_steps': 0,
        'upstream_saturated_steps': 0,
        **{field: 0 for field in COUNT_FIELDS},
    })
    for path in args.csv_files:
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                density = int(row['density'])
                group = groups[density]
                group['steps'] += 1
                for field in COUNT_FIELDS:
                    group[field] += int(row[field])
                hidden = int(row['n_hidden_candidates'])
                near_hidden = int(row['n_near_hidden_candidates'])
                group['hidden_steps'] += int(hidden > 0)
                group['near_hidden_steps'] += int(near_hidden > 0)
                group['hidden_no_slot_steps'] += int(
                    row['hidden_available_but_no_slot'])
                group['near_hidden_no_slot_steps'] += int(
                    row['near_hidden_available_but_no_slot'])
                group['all_visible_steps'] += int(row['selected_all_visible'])
                group['upstream_saturated_steps'] += int(
                    row['upstream_cap_saturated'])

    if not groups:
        raise SystemExit('no audit rows found')

    print(
        'density steps cap-sat upstream-drop token-retain hidden-no-slot '
        'near-e2e-miss near-no-slot all-visible true-hidden/step')
    for density in sorted(groups):
        group = groups[density]
        print(
            f"{density:>7} {group['steps']:>6} "
            f"{fmt(ratio(group['upstream_saturated_steps'], group['steps'])):>7} "
            f"{fmt(ratio(group['n_hidden_upstream_dropped'], group['n_hidden_modes_uncapped'])):>13} "
            f"{fmt(ratio(group['n_selected_hidden'], group['n_hidden_candidates'])):>13} "
            f"{fmt(ratio(group['hidden_no_slot_steps'], group['hidden_steps'])):>14} "
            f"{fmt(ratio(group['n_near_hidden_end_to_end_dropped'], group['n_near_hidden_modes_uncapped'])):>13} "
            f"{fmt(ratio(group['near_hidden_no_slot_steps'], group['near_hidden_steps'])):>12} "
            f"{fmt(ratio(group['all_visible_steps'], group['steps'])):>11} "
            f"{fmt(ratio(group['n_true_occluded'], group['steps'])):>16}"
        )


if __name__ == '__main__':
    main()
