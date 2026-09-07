#!/usr/bin/env python3
"""Offline, truth-matched audit of candidate-to-token selection contracts.

Contract selection uses development episodes only. Test episodes are opened
once for confirmation after the development winner has been fixed.
"""
import argparse
import csv
import glob
import json
import math
from collections import defaultdict


KEY_FIELDS = (
    'scenario_index', 'episode', 'seed', 'step', 'density',
)
PRIMARY_MATCH_RADIUS_M = 0.75
SENSITIVITY_RADII_M = (0.50, 0.75, 1.00)
NEAR_CLEARANCE_M = 2.0
VISIBLE_TTC_S = 5.0
MIN_CONDITIONAL_RECALL = 0.90
MAX_VISIBLE_RETENTION_DROP = 0.01
MAX_HIDDEN_PRECISION_DROP = 0.05


def _key(row):
    return tuple(int(row[field]) for field in KEY_FIELDS)


def _load(paths, converters):
    groups = defaultdict(list)
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        for path in matches:
            with open(path, newline='', encoding='utf-8') as handle:
                for raw in csv.DictReader(handle):
                    row = dict(raw)
                    for field, convert in converters.items():
                        row[field] = convert(row[field])
                    groups[_key(row)].append(row)
    return groups


def _distance(left, right):
    return math.hypot(left['px'] - right['px'], left['py'] - right['py'])


def _maximum_match_count(truths, candidates, radius):
    """Maximum-cardinality bipartite match under a metric distance gate."""
    adjacency = []
    for truth in truths:
        edges = [
            (index, _distance(truth, candidate))
            for index, candidate in enumerate(candidates)
            if _distance(truth, candidate) <= radius
        ]
        adjacency.append([index for index, _ in sorted(edges, key=lambda x: x[1])])
    candidate_match = {}

    def augment(truth_index, seen):
        for candidate_index in adjacency[truth_index]:
            if candidate_index in seen:
                continue
            seen.add(candidate_index)
            previous = candidate_match.get(candidate_index)
            if previous is None or augment(previous, seen):
                candidate_match[candidate_index] = truth_index
                return True
        return False

    matched = 0
    for truth_index in sorted(range(len(truths)), key=lambda i: len(adjacency[i])):
        matched += int(augment(truth_index, set()))
    return matched


def _rank(rows, rule):
    if rule == 'unified_ttc':
        return sorted(rows, key=lambda row: (row['ttc_score'], row['candidate_index']))
    if rule == 'unified_clearance':
        return sorted(rows, key=lambda row: (row['clearance_m'], row['candidate_index']))
    raise ValueError(rule)


def _fill_separate(visible, hidden, slots, visible_budget,
                   hidden_rule='clearance'):
    visible = sorted(visible, key=lambda row: (row['ttc_score'], row['candidate_index']))
    if hidden_rule == 'clearance':
        hidden = sorted(
            hidden, key=lambda row: (row['clearance_m'], row['candidate_index']))
    elif hidden_rule == 'confidence':
        hidden = sorted(
            hidden, key=lambda row: (-row['p_exist'], row['clearance_m'],
                                     row['candidate_index']))
    else:
        raise ValueError(hidden_rule)
    visible_budget = min(slots, visible_budget)
    hidden_budget = slots - visible_budget
    visible_take = min(len(visible), visible_budget)
    hidden_take = min(
        len(hidden), hidden_budget + (visible_budget - visible_take))
    selected = visible[:visible_take] + hidden[:hidden_take]
    if len(selected) < slots:
        selected.extend(visible[visible_take:visible_take + slots - len(selected)])
    return selected


def _select(candidates, cap, slots, rule):
    visible = [row for row in candidates if row['source'] == 'visible']
    hidden = [row for row in candidates if row['source'] == 'hidden']
    if cap is not None:
        hidden = [row for row in hidden if row['source_rank'] < cap]
    combined = visible + hidden
    if rule in ('unified_ttc', 'unified_clearance'):
        return _rank(combined, rule)[:slots]
    if rule == 'separate_equal':
        return _fill_separate(visible, hidden, slots, (slots + 1) // 2)
    if rule == 'visible5_plus_hidden':
        return _fill_separate(visible, hidden, slots, min(5, slots))
    if rule == 'visible5_hidden_confidence':
        return _fill_separate(
            visible, hidden, slots, min(5, slots), hidden_rule='confidence')
    raise ValueError(rule)


def _contract_name(cap, slots, rule):
    return f"cap_{'all' if cap is None else cap}__slots_{slots}__{rule}"


def _new_counts():
    return defaultdict(float)


def _ratio(num, den):
    return num / den if den else float('nan')


def _update(counts, candidates, truths, selected, radius):
    hidden_truth = [row for row in truths if row['visibility'] == 'hidden']
    near_hidden_truth = [
        row for row in hidden_truth if row['clearance_m'] <= NEAR_CLEARANCE_M
    ]
    visible = [row for row in candidates if row['source'] == 'visible']
    all_hidden = [row for row in candidates if row['source'] == 'hidden']
    selected_ids = {row['candidate_index'] for row in selected}
    selected_hidden = [row for row in selected if row['source'] == 'hidden']

    representable = _maximum_match_count(near_hidden_truth, all_hidden, radius)
    selected_near = _maximum_match_count(near_hidden_truth, selected_hidden, radius)
    selected_hidden_matches = _maximum_match_count(hidden_truth, selected_hidden, radius)

    visible_near = [row for row in visible if row['clearance_m'] <= NEAR_CLEARANCE_M]
    visible_ttc = [row for row in visible if row['ttc_score'] <= VISIBLE_TTC_S]
    counts['steps'] += 1
    counts['tokens'] += len(selected)
    counts['near_hidden_truth'] += len(near_hidden_truth)
    counts['representable_near_hidden_truth'] += representable
    counts['selected_near_hidden_truth'] += selected_near
    counts['selected_hidden'] += len(selected_hidden)
    counts['selected_hidden_matches'] += selected_hidden_matches
    counts['visible_near'] += len(visible_near)
    counts['selected_visible_near'] += sum(
        row['candidate_index'] in selected_ids for row in visible_near)
    counts['visible_ttc'] += len(visible_ttc)
    counts['selected_visible_ttc'] += sum(
        row['candidate_index'] in selected_ids for row in visible_ttc)


def _summarize(counts):
    return {
        'steps': int(counts['steps']),
        'avg_tokens': _ratio(counts['tokens'], counts['steps']),
        'belief_near_coverage': _ratio(
            counts['representable_near_hidden_truth'], counts['near_hidden_truth']),
        'selected_near_absolute_recall': _ratio(
            counts['selected_near_hidden_truth'], counts['near_hidden_truth']),
        'selected_near_conditional_recall': _ratio(
            counts['selected_near_hidden_truth'],
            counts['representable_near_hidden_truth']),
        'selected_hidden_precision': _ratio(
            counts['selected_hidden_matches'], counts['selected_hidden']),
        'visible_near_retention': _ratio(
            counts['selected_visible_near'], counts['visible_near']),
        'visible_ttc_retention': _ratio(
            counts['selected_visible_ttc'], counts['visible_ttc']),
        'near_hidden_truth_count': int(counts['near_hidden_truth']),
        'representable_near_hidden_truth_count': int(
            counts['representable_near_hidden_truth']),
    }


def _evaluate(candidate_groups, truth_groups, contracts, split, radius):
    totals = defaultdict(_new_counts)
    candidate_keys = set(candidate_groups)
    truth_keys = set(truth_groups)
    if candidate_keys - truth_keys:
        missing_truth = len(candidate_keys - truth_keys)
        raise RuntimeError(
            f'candidate rows without truth rows: {missing_truth=}')
    for key in sorted(truth_keys):
        episode = key[1]
        if split != 'all':
            is_dev = episode < 15
            if (split == 'dev') != is_dev:
                continue
        # A fully occluded step can legitimately have no visible or belief
        # candidates. It still belongs in recall and token-retention metrics.
        candidates = candidate_groups.get(key, [])
        truths = truth_groups[key]
        density = key[-1]
        for cap, slots, rule in contracts:
            name = _contract_name(cap, slots, rule)
            selected = _select(candidates, cap, slots, rule)
            _update(totals[(name, density)], candidates, truths, selected, radius)
    return {
        key: _summarize(value) for key, value in totals.items()
    }


def _baseline_by_density(metrics):
    name = _contract_name(5, 5, 'unified_ttc')
    return {
        density: values
        for (contract, density), values in metrics.items()
        if contract == name
    }


def _is_feasible(name, metrics, baseline):
    reasons = []
    densities = sorted(baseline)
    for density in densities:
        values = metrics[(name, density)]
        base = baseline[density]
        conditional = values['selected_near_conditional_recall']
        if not math.isfinite(conditional) or conditional < MIN_CONDITIONAL_RECALL:
            reasons.append(f'n{density}:conditional={conditional:.3f}')
        for field in ('visible_near_retention', 'visible_ttc_retention'):
            if values[field] + MAX_VISIBLE_RETENTION_DROP < base[field]:
                reasons.append(
                    f'n{density}:{field}={values[field]:.3f}<base={base[field]:.3f}')
    density = max(densities)
    precision = metrics[(name, density)]['selected_hidden_precision']
    base_precision = baseline[density]['selected_hidden_precision']
    if precision + MAX_HIDDEN_PRECISION_DROP < base_precision:
        reasons.append(
            f'n{density}:hidden_precision={precision:.3f}<base={base_precision:.3f}')
    return not reasons, reasons


def _contract_tuple(name):
    cap_text, slots_text, rule = name.split('__')
    cap_value = cap_text.split('_', 1)[1]
    cap = 10 ** 9 if cap_value == 'all' else int(cap_value)
    slots = int(slots_text.split('_', 1)[1])
    return cap, slots, rule


def _write_metrics(path, split, radius, metrics, feasibility=None):
    fields = [
        'split', 'match_radius_m', 'contract', 'density', 'feasible',
        'failure_reasons', 'steps', 'avg_tokens', 'belief_near_coverage',
        'selected_near_absolute_recall', 'selected_near_conditional_recall',
        'selected_hidden_precision', 'visible_near_retention',
        'visible_ttc_retention', 'near_hidden_truth_count',
        'representable_near_hidden_truth_count',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (name, density), values in sorted(metrics.items()):
            feasible, reasons = (feasibility or {}).get(name, ('', []))
            writer.writerow({
                'split': split,
                'match_radius_m': radius,
                'contract': name,
                'density': density,
                'feasible': feasible,
                'failure_reasons': '|'.join(reasons),
                **values,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--truth', nargs='+', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument(
        '--confirm-contract', default=None,
        help='Evaluate one preselected contract on every supplied episode')
    args = parser.parse_args()

    candidate_groups = _load(args.candidates, {
        'scenario_index': int, 'episode': int, 'seed': int, 'step': int,
        'density': int, 'candidate_index': int, 'source_rank': int,
        'entity_id': int, 'upstream_retained': int, 'token_selected': int,
        'px': float, 'py': float, 'vx': float, 'vy': float,
        'distance_m': float, 'clearance_m': float, 'closing_mps': float,
        'ttc_score': float, 'p_exist': float, 'uncertainty': float,
    })
    truth_groups = _load(args.truth, {
        'scenario_index': int, 'episode': int, 'seed': int, 'step': int,
        'density': int, 'true_id': int, 'px': float, 'py': float,
        'vx': float, 'vy': float, 'radius': float, 'distance_m': float,
        'clearance_m': float, 'closing_mps': float, 'ttc_score': float,
        'near_2m': int,
    })
    contracts = []
    for cap in (5, 8, 10, 15, None):
        for slots in (5, 8, 10, 12, 15):
            for rule in (
                    'unified_ttc', 'unified_clearance',
                    'separate_equal', 'visible5_plus_hidden',
                    'visible5_hidden_confidence'):
                contracts.append((cap, slots, rule))

    if args.confirm_contract:
        parsed = _contract_tuple(args.confirm_contract)
        selected_tuple = (
            None if parsed[0] == 10 ** 9 else parsed[0], parsed[1], parsed[2])
        confirm_contracts = [(5, 5, 'unified_ttc'), selected_tuple]
        confirm = _evaluate(
            candidate_groups, truth_groups, confirm_contracts, 'all',
            PRIMARY_MATCH_RADIUS_M)
        baseline = _baseline_by_density(confirm)
        feasible = {
            args.confirm_contract: _is_feasible(
                args.confirm_contract, confirm, baseline)
        }
        _write_metrics(
            args.output_prefix + '_confirm.csv', 'all',
            PRIMARY_MATCH_RADIUS_M, confirm, feasible)
        sensitivity = {}
        for radius in SENSITIVITY_RADII_M:
            values = _evaluate(
                candidate_groups, truth_groups, [selected_tuple], 'all', radius)
            sensitivity[str(radius)] = {
                str(density): metrics
                for (name, density), metrics in values.items()
            }
        passed, reasons = feasible[args.confirm_contract]
        decision = {
            'confirm_contract': args.confirm_contract,
            'primary_match_radius_m': PRIMARY_MATCH_RADIUS_M,
            'test_pass': passed,
            'test_failure_reasons': reasons,
            'thresholds': {
                'min_conditional_recall': MIN_CONDITIONAL_RECALL,
                'max_visible_retention_drop': MAX_VISIBLE_RETENTION_DROP,
                'max_hidden_precision_drop': MAX_HIDDEN_PRECISION_DROP,
            },
            'sensitivity': sensitivity,
        }
        with open(
                args.output_prefix + '_decision.json', 'w',
                encoding='utf-8') as handle:
            json.dump(decision, handle, indent=2, sort_keys=True)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return

    dev = _evaluate(
        candidate_groups, truth_groups, contracts, 'dev', PRIMARY_MATCH_RADIUS_M)
    dev_baseline = _baseline_by_density(dev)
    names = sorted({name for name, _ in dev})
    dev_feasibility = {
        name: _is_feasible(name, dev, dev_baseline) for name in names
    }
    feasible_names = [name for name in names if dev_feasibility[name][0]]
    if feasible_names:
        # Prefer the smallest network input, then the highest test-independent
        # hidden precision on development density 20, then the smallest cap.
        feasible_names.sort(key=lambda name: (
            _contract_tuple(name)[1],
            -dev[(name, max(dev_baseline))]['selected_hidden_precision'],
            _contract_tuple(name)[0],
            name,
        ))
        winner = feasible_names[0]
    else:
        winner = None

    test = _evaluate(
        candidate_groups, truth_groups, contracts, 'test', PRIMARY_MATCH_RADIUS_M)
    test_baseline = _baseline_by_density(test)
    test_feasibility = {
        name: _is_feasible(name, test, test_baseline) for name in names
    }
    _write_metrics(
        args.output_prefix + '_dev.csv', 'dev', PRIMARY_MATCH_RADIUS_M,
        dev, dev_feasibility)
    _write_metrics(
        args.output_prefix + '_test.csv', 'test', PRIMARY_MATCH_RADIUS_M,
        test, test_feasibility)

    sensitivity = {}
    if winner is not None:
        winner_tuple = [_contract_tuple(winner)]
        winner_tuple = [(
            None if cap == 10 ** 9 else cap, slots, rule
        ) for cap, slots, rule in winner_tuple]
        for radius in SENSITIVITY_RADII_M:
            values = _evaluate(
                candidate_groups, truth_groups, winner_tuple, 'test', radius)
            sensitivity[str(radius)] = {
                str(density): metrics
                for (name, density), metrics in values.items()
            }

    decision = {
        'primary_match_radius_m': PRIMARY_MATCH_RADIUS_M,
        'near_clearance_m': NEAR_CLEARANCE_M,
        'visible_ttc_s': VISIBLE_TTC_S,
        'thresholds': {
            'min_conditional_recall': MIN_CONDITIONAL_RECALL,
            'max_visible_retention_drop': MAX_VISIBLE_RETENTION_DROP,
            'max_hidden_precision_drop': MAX_HIDDEN_PRECISION_DROP,
        },
        'dev_feasible_contracts': feasible_names,
        'selected_contract': winner,
        'test_pass': bool(winner and test_feasibility[winner][0]),
        'test_failure_reasons': (
            test_feasibility[winner][1] if winner else ['no development-feasible contract']),
        'sensitivity': sensitivity,
    }
    with open(args.output_prefix + '_decision.json', 'w', encoding='utf-8') as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
