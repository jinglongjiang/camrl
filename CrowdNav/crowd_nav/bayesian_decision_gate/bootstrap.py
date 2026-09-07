"""Block-bootstrap statistics shared by the prediction and decision gates.

Every CI in this experiment resamples whole SEED or EPISODE blocks, never
individual timesteps -- timesteps within one episode (and episodes within one
seed's rollout) are correlated, so per-timestep bootstrap would understate
variance and make small real effects look significant. Mirrors
``belief_mdp/evaluate.py``'s ``block_bootstrap_ci`` (seed-block resampling),
generalized to accept episode-level blocks too.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def block_bootstrap_ci(
    blocks: Sequence[np.ndarray],
    replicates: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Resample whole blocks (seeds or episodes) with replacement.

    Returns (observed_mean, ci_lo_2.5, ci_hi_97.5). Empty blocks list ->
    (nan, nan, nan).
    """
    rng = rng or np.random.default_rng(0)
    blocks = [np.asarray(b, dtype=np.float64).ravel() for b in blocks if len(b) > 0]
    n_blocks = len(blocks)
    if n_blocks == 0:
        return float("nan"), float("nan"), float("nan")
    observed = np.concatenate(blocks)
    means = np.empty(replicates, dtype=np.float64)
    block_indices = rng.integers(0, n_blocks, size=(replicates, n_blocks))
    for replicate in range(replicates):
        resampled = [blocks[i] for i in block_indices[replicate]]
        means[replicate] = np.concatenate(resampled).mean()
    return (
        float(observed.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_block_bootstrap_diff(
    blocks_a: Sequence[np.ndarray],
    blocks_b: Sequence[np.ndarray],
    replicates: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) using the SAME block resampling
    draw for both (paired by seed/episode index) -- required whenever a and b
    are two models' metrics measured on the identical episodes."""
    rng = rng or np.random.default_rng(0)
    blocks_a = [np.asarray(b, dtype=np.float64).ravel() for b in blocks_a]
    blocks_b = [np.asarray(b, dtype=np.float64).ravel() for b in blocks_b]
    n_blocks = len(blocks_a)
    if n_blocks == 0 or len(blocks_b) != n_blocks:
        return float("nan"), float("nan"), float("nan")
    observed = np.concatenate(blocks_a).mean() - np.concatenate(blocks_b).mean()
    diffs = np.empty(replicates, dtype=np.float64)
    block_indices = rng.integers(0, n_blocks, size=(replicates, n_blocks))
    for replicate in range(replicates):
        idx = block_indices[replicate]
        mean_a = np.concatenate([blocks_a[i] for i in idx]).mean()
        mean_b = np.concatenate([blocks_b[i] for i in idx]).mean()
        diffs[replicate] = mean_a - mean_b
    return (
        float(observed),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (1-indexed), matching
    bayesian_pilot/evaluate_prediction_gate.py's rank_auc tie-breaking."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, numpy-only (no scipy dependency).
    Returns nan if either input has zero variance or fewer than 2 points."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def top_k_hit_rate(true_rank_of_predicted_best: Sequence[int], k: int) -> float:
    """Fraction of decisions where the model's chosen (predicted-safest)
    action is within the top-``k`` of the TRUE risk ranking (rank 0 = safest).
    """
    ranks = np.asarray(true_rank_of_predicted_best, dtype=np.int64)
    if len(ranks) == 0:
        return float("nan")
    return float(np.mean(ranks < k))
