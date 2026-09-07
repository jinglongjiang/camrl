#!/usr/bin/env python3
"""Build a supervised doom-head dataset from candidate_diag JSONL files."""

import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch


def _iter_paths(patterns):
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if path not in seen:
                seen.add(path)
                yield path


def _load_episodes(paths, horizon, include_timeout_negatives):
    episodes = []
    missing_features = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                outcome = str(payload.get("outcome", "")).lower()
                if outcome == "timeout" and not include_timeout_negatives:
                    continue
                records = payload.get("records") or []
                steps = int(payload.get("steps", len(records)) or len(records))
                samples = []
                for rec in records:
                    feature = rec.get("selected_next_feature")
                    if not isinstance(feature, list) or not feature:
                        missing_features += 1
                        continue
                    step = int(rec.get("step", 0))
                    if outcome == "collision":
                        steps_to_event = max(0, steps - 1 - step)
                        label = 1.0 if steps_to_event <= horizon else 0.0
                    else:
                        steps_to_event = None
                        label = 0.0
                    samples.append({
                        "feature": feature,
                        "label": label,
                        "case": payload.get("case"),
                        "outcome": outcome,
                        "episode_seed": payload.get("episode_seed"),
                        "step": step,
                        "steps_to_event": steps_to_event,
                        "source": f"{os.path.basename(path)}:{line_no}",
                    })
                if samples:
                    episodes.append(samples)
    return episodes, missing_features


def _balance(samples, neg_ratio, rng):
    positives = [s for s in samples if float(s["label"]) >= 0.5]
    negatives = [s for s in samples if float(s["label"]) < 0.5]
    if positives and neg_ratio > 0:
        max_neg = min(len(negatives), int(round(len(positives) * float(neg_ratio))))
        negatives = rng.sample(negatives, max_neg) if len(negatives) > max_neg else negatives
    balanced = positives + negatives
    rng.shuffle(balanced)
    return balanced


def _pack(samples):
    if not samples:
        return torch.empty(0, 256), torch.empty(0)
    features = torch.tensor([s["feature"] for s in samples], dtype=torch.float32)
    labels = torch.tensor([float(s["label"]) for s in samples], dtype=torch.float32)
    return features, labels


def _count_by(samples, key):
    counts = defaultdict(int)
    for sample in samples:
        counts[str(sample.get(key))] += 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", help="candidate_diag JSONL files or glob patterns")
    parser.add_argument("--out", default="data/doom_dataset.pth")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_timeout_negatives", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    paths = list(_iter_paths(args.jsonl))
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {args.jsonl}")

    episodes, missing_features = _load_episodes(paths, args.horizon, args.include_timeout_negatives)
    if not episodes:
        raise RuntimeError("No usable records found. Re-run test.py with --candidate_diag_features.")

    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * float(args.val_ratio)))) if len(episodes) > 1 else 0
    val_episodes = episodes[:n_val]
    train_episodes = episodes[n_val:]

    train_samples_raw = [s for ep in train_episodes for s in ep]
    val_samples_raw = [s for ep in val_episodes for s in ep]
    train_samples = _balance(train_samples_raw, args.neg_ratio, rng)
    val_samples = _balance(val_samples_raw, args.neg_ratio, rng)

    train_features, train_labels = _pack(train_samples)
    val_features, val_labels = _pack(val_samples)
    all_features, all_labels = _pack(train_samples + val_samples)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
        "features": all_features,
        "labels": all_labels,
        "meta": {
            "source_files": paths,
            "horizon": int(args.horizon),
            "val_ratio": float(args.val_ratio),
            "neg_ratio": float(args.neg_ratio),
            "seed": int(args.seed),
            "include_timeout_negatives": bool(args.include_timeout_negatives),
            "episodes_total": len(episodes),
            "episodes_train": len(train_episodes),
            "episodes_val": len(val_episodes),
            "missing_feature_records": int(missing_features),
            "train_counts_by_outcome": _count_by(train_samples, "outcome"),
            "val_counts_by_outcome": _count_by(val_samples, "outcome"),
            "train_positive": int(train_labels.sum().item()),
            "train_negative": int((train_labels < 0.5).sum().item()),
            "val_positive": int(val_labels.sum().item()),
            "val_negative": int((val_labels < 0.5).sum().item()),
            "feature_dim": int(all_features.shape[1]) if all_features.numel() else 0,
        },
    }
    torch.save(payload, args.out)
    print(f"[DOOM-DATA] saved {args.out}")
    print(f"[DOOM-DATA] files={len(paths)} episodes={len(episodes)} missing_features={missing_features}")
    print(
        "[DOOM-DATA] train=%d pos=%d neg=%d | val=%d pos=%d neg=%d | dim=%d" % (
            len(train_labels), int(train_labels.sum().item()), int((train_labels < 0.5).sum().item()),
            len(val_labels), int(val_labels.sum().item()), int((val_labels < 0.5).sum().item()),
            int(all_features.shape[1]) if all_features.numel() else 0,
        )
    )


if __name__ == "__main__":
    main()
