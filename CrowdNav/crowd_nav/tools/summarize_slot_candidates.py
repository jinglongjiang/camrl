#!/usr/bin/env python3
"""Summarize closing/TTC behavior from candidate-level slot audit rows."""
import argparse
import csv
import math
import statistics
from collections import defaultdict


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def fmt(value):
    return 'n/a' if not math.isfinite(value) else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='+')
    args = parser.parse_args()

    groups = defaultdict(lambda: {
        'count': 0,
        'retained': 0,
        'selected': 0,
        'fallback': 0,
        'near': 0,
        'near_selected': 0,
        'closings': [],
        'ttcs': [],
    })
    for path in args.csv_files:
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                key = (int(row['density']), row['source'])
                group = groups[key]
                group['count'] += 1
                group['retained'] += int(row['upstream_retained'])
                group['selected'] += int(row['token_selected'])
                group['fallback'] += int(row['ttc_branch'] == 'distance_x10')
                near = float(row['clearance_m']) <= 2.0
                group['near'] += int(near)
                group['near_selected'] += int(
                    near and int(row['token_selected']))
                group['closings'].append(float(row['closing_mps']))
                group['ttcs'].append(float(row['ttc_score']))

    if not groups:
        raise SystemExit('no candidate rows found')

    print(
        'density source count upstream-retain selected fallback '
        'near-selected median-closing median-ttc')
    for (density, source), group in sorted(groups.items()):
        print(
            f'{density:>7} {source:>7} {group["count"]:>7} '
            f'{fmt(ratio(group["retained"], group["count"])):>15} '
            f'{fmt(ratio(group["selected"], group["count"])):>8} '
            f'{fmt(ratio(group["fallback"], group["count"])):>8} '
            f'{fmt(ratio(group["near_selected"], group["near"])):>13} '
            f'{fmt(statistics.median(group["closings"])):>14} '
            f'{fmt(statistics.median(group["ttcs"])):>10}'
        )


if __name__ == '__main__':
    main()
