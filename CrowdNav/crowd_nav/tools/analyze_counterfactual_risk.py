#!/usr/bin/env python3
"""Measure how Bayesian action risk aligns with simulator outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def binary_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "runs/bayesian_distributional/"
            "counterfactual_fullcrowd_v2.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "runs/bayesian_distributional/"
            "counterfactual_fullcrowd_v2_risk_analysis.json"
        ),
    )
    args = parser.parse_args()

    payload = torch_load(args.dataset)
    samples = payload["samples"]
    all_expected = []
    all_tail = []
    all_collision = []
    within_expected = []
    within_tail = []
    colliding_teacher_states = 0
    safe_lower_risk_states = 0

    for sample in samples:
        features = np.asarray(sample["action_features"], dtype=np.float64)
        collision = np.asarray(
            [bool(label["collision"]) for label in sample["labels"]],
            dtype=np.bool_,
        )
        expected = features[:, 0]
        tail = features[:, 1]
        all_expected.extend(expected.tolist())
        all_tail.extend(tail.tolist())
        all_collision.extend(collision.tolist())

        expected_auc = binary_auc(expected, collision)
        tail_auc = binary_auc(tail, collision)
        if expected_auc is not None:
            within_expected.append(expected_auc)
        if tail_auc is not None:
            within_tail.append(tail_auc)

        teacher_top = int(torch.argmax(sample["teacher_scores"]).item())
        if collision[teacher_top]:
            colliding_teacher_states += 1
            safe = ~collision
            if np.any(safe & (expected < expected[teacher_top])):
                safe_lower_risk_states += 1

    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "states": len(samples),
        "candidates": len(all_collision),
        "collision_rate": float(np.mean(all_collision)),
        "global_auc": {
            "expected_risk": binary_auc(all_expected, all_collision),
            "tail_risk": binary_auc(all_tail, all_collision),
        },
        "mean_within_state_auc": {
            "expected_risk": float(np.mean(within_expected)),
            "tail_risk": float(np.mean(within_tail)),
            "eligible_states": len(within_expected),
        },
        "teacher_top_collision_states": colliding_teacher_states,
        "safe_lower_expected_risk_states": safe_lower_risk_states,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[SAVE] {output}")


if __name__ == "__main__":
    main()
