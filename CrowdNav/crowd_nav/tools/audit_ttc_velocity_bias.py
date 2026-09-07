#!/usr/bin/env python3
"""Policy-independent audit of TTC ranking under hidden-velocity error."""
import argparse
import csv
import itertools
import math


DEFAULT_SPEEDS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_DISTANCES = (0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_ESTIMATE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def ttc_score(distance, robot_speed, pedestrian_toward_speed):
    # Entity lies on +x. Robot moves +x and pedestrian moves -x.
    rel_x, rel_y = float(distance), 0.0
    rel_vx = -float(pedestrian_toward_speed) - float(robot_speed)
    rel_vy = 0.0
    dist = math.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)
    closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
    score = (
        dist / (closing + 1e-6)
        if closing > 0.1 else dist * 10.0
    )
    return closing, score, closing > 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    rows = []
    dangerous_pairs = 0
    inversions = 0
    branch_mismatches = 0
    hidden_cases = 0
    demotion_factors = []
    grid = itertools.product(
        DEFAULT_SPEEDS,
        DEFAULT_DISTANCES,
        DEFAULT_SPEEDS,
        DEFAULT_ESTIMATE_FRACTIONS,
        DEFAULT_DISTANCES,
        DEFAULT_SPEEDS,
    )
    for robot_speed, hidden_distance, hidden_speed, estimate_fraction, \
            visible_distance, visible_speed in grid:
        true_closing, true_ttc, true_branch = ttc_score(
            hidden_distance, robot_speed, hidden_speed)
        estimated_closing, estimated_ttc, estimated_branch = ttc_score(
            hidden_distance, robot_speed, hidden_speed * estimate_fraction)
        visible_closing, visible_ttc, _ = ttc_score(
            visible_distance, robot_speed, visible_speed)
        hidden_cases += 1
        branch_mismatches += int(true_branch and not estimated_branch)
        if true_ttc > 0.0:
            demotion_factors.append(estimated_ttc / true_ttc)
        truly_more_dangerous = true_ttc < visible_ttc
        rank_inversion = truly_more_dangerous and estimated_ttc > visible_ttc
        dangerous_pairs += int(truly_more_dangerous)
        inversions += int(rank_inversion)
        if args.csv:
            rows.append({
                'robot_speed': robot_speed,
                'hidden_distance': hidden_distance,
                'hidden_true_toward_speed': hidden_speed,
                'hidden_estimate_fraction': estimate_fraction,
                'visible_distance': visible_distance,
                'visible_toward_speed': visible_speed,
                'hidden_true_closing': true_closing,
                'hidden_estimated_closing': estimated_closing,
                'visible_closing': visible_closing,
                'hidden_true_ttc': true_ttc,
                'hidden_estimated_ttc': estimated_ttc,
                'visible_ttc': visible_ttc,
                'branch_mismatch': int(true_branch and not estimated_branch),
                'truly_more_dangerous': int(truly_more_dangerous),
                'rank_inversion': int(rank_inversion),
            })

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    demotion_factors.sort()
    median_demotion = demotion_factors[len(demotion_factors) // 2]
    inversion_rate = inversions / dangerous_pairs if dangerous_pairs else 0.0
    mismatch_rate = branch_mismatches / hidden_cases if hidden_cases else 0.0
    verdict = (
        'STRONG_BIAS' if inversion_rate >= 0.25
        else 'BIAS_CONFIRMED' if inversion_rate >= 0.10
        else 'NO_MATERIAL_ANALYTIC_BIAS'
    )
    print('TTC_VELOCITY_BIAS_AUDIT')
    print(f'grid_cases={hidden_cases}')
    print(f'truly_more_dangerous_pairs={dangerous_pairs}')
    print(f'rank_inversions={inversions}')
    print(f'rank_inversion_rate={inversion_rate:.6f}')
    print(f'closing_branch_mismatch_rate={mismatch_rate:.6f}')
    print(f'median_ttc_demotion_factor={median_demotion:.6f}')
    print(f'VERDICT={verdict}')


if __name__ == '__main__':
    main()
