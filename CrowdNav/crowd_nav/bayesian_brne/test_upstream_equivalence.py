#!/usr/bin/env python3
"""Upstream BRNE equivalence tests (guide.md 7.2 / 8.2).

Two things are verified on fixed, deterministic fixtures:

1. The mean+perturbation decomposition ``brne_adapter.py`` uses to feed
   arbitrary (possibly multi-modal) sample sets into upstream's UNMODIFIED
   ``brne_nav`` is a lossless bookkeeping split: two completely different
   decompositions of the SAME absolute trajectories must produce
   byte-identical (<1e-10) weights and reconstructed trajectories, because
   upstream's own math only ever consumes the reconstructed sum.
2. ``solver_mode="stable"``'s log-space reimplementation is algebraically
   the same update as upstream's naive ``exp(-cost1)`` (max-subtraction
   cancels out under the row-mean normalization both use) -- verified to
   agree closely (not to 1e-10, since it is numpy vs. numba and a different
   operation order, but far tighter than any real behavioral difference)
   on the same well-behaved fixture.

Run via ``python3 -m crowd_nav.bayesian_brne.test_upstream_equivalence``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from crowd_nav.bayesian_brne.brne_adapter import (
    BRNESolver,
    _decompose_mean_perturbation,
    load_upstream_brne,
)

BRNE_ROOT = os.environ.get("SM_BRNE_UPSTREAM_ROOT", "/home/abc/temp/brne")
EXPECTED_COMMIT = "633a5cdcb39ab27f18b596cb8cb1968644f82391"
EXPECTED_SOURCE_SHA256 = "420d1ef383c9a80448d3c714bf22d0d6280c055488269d624c060d98cff54836"

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def _pinned_commit_matches() -> bool:
    root = Path(BRNE_ROOT).resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root, capture_output=True, text=True, check=True,
        )
        if Path(out.stdout.strip()).resolve() == root:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout.strip()
            return commit == EXPECTED_COMMIT
    except Exception:
        pass
    source = root / "socnavbench" / "brne.py"
    return source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256


def _make_fixture(seed: int, num_agents=3, num_pts=20, horizon=8):
    rng = np.random.default_rng(seed)
    trajectories = np.zeros((num_agents, num_pts, horizon, 2), dtype=np.float64)
    for a in range(num_agents):
        start = rng.uniform(-2.0, 2.0, size=2)
        drift = rng.uniform(-0.3, 0.3, size=2)
        noise = rng.normal(scale=0.15, size=(num_pts, horizon, 2))
        base = start[None, None, :] + np.arange(horizon)[None, :, None] * drift[None, None, :]
        trajectories[a] = base + noise
    return trajectories


def test_pinned_commit():
    check(
        "brne_upstream_commit_pinned",
        _pinned_commit_matches(),
        f"expected {EXPECTED_COMMIT}",
    )


def test_upstream_module_loads():
    upstream = load_upstream_brne(BRNE_ROOT)
    check(
        "upstream_module_has_brne_nav",
        hasattr(upstream, "brne_nav") and callable(upstream.brne_nav),
    )


def test_decomposition_is_lossless_bookkeeping():
    """Two different mean/perturbation splits of the identical absolute
    trajectories must produce byte-identical (<1e-10) BRNE outputs."""
    trajectories = _make_fixture(seed=0)
    num_agents, num_pts, horizon, _ = trajectories.shape
    upstream = load_upstream_brne(BRNE_ROOT)

    # Split A: per-agent sample mean (what brne_adapter.py actually uses).
    xmean_a, x_pts_a = _decompose_mean_perturbation(trajectories[..., 0])
    ymean_a, y_pts_a = _decompose_mean_perturbation(trajectories[..., 1])

    # Split B: first sample as "mean", remainder as perturbation -- a
    # completely different, equally arbitrary bookkeeping choice.
    xmean_b = trajectories[:, 0, :, 0]
    ymean_b = trajectories[:, 0, :, 1]
    x_pts_b = (trajectories[..., 0] - xmean_b[:, None, :]).reshape(num_agents * num_pts, horizon)
    y_pts_b = (trajectories[..., 1] - ymean_b[:, None, :]).reshape(num_agents * num_pts, horizon)

    # Sanity: both splits reconstruct the identical absolute trajectories.
    for i in range(num_agents):
        recon = xmean_a[i] + x_pts_a[i * num_pts:(i + 1) * num_pts]
        check(
            f"split_a_reconstructs_agent_{i}",
            bool(np.allclose(recon, trajectories[i, :, :, 0], atol=1e-12)),
        )

    x_opt_a, y_opt_a, weights_a = upstream.brne_nav(
        list(xmean_a), list(ymean_a), x_pts_a, y_pts_a, num_agents, horizon, num_pts
    )
    x_opt_b, y_opt_b, weights_b = upstream.brne_nav(
        list(xmean_b), list(ymean_b), x_pts_b, y_pts_b, num_agents, horizon, num_pts
    )

    max_weight_diff = float(np.max(np.abs(np.asarray(weights_a) - np.asarray(weights_b))))
    max_x_diff = float(np.max(np.abs(np.asarray(x_opt_a) - np.asarray(x_opt_b))))
    max_y_diff = float(np.max(np.abs(np.asarray(y_opt_a) - np.asarray(y_opt_b))))

    check("decomposition_weights_match_1e-10", max_weight_diff < 1e-10, f"max_diff={max_weight_diff:.2e}")
    check("decomposition_x_opt_match_1e-10", max_x_diff < 1e-10, f"max_diff={max_x_diff:.2e}")
    check("decomposition_y_opt_match_1e-10", max_y_diff < 1e-10, f"max_diff={max_y_diff:.2e}")


def test_adapter_official_exact_matches_direct_upstream_call():
    trajectories = _make_fixture(seed=1)
    num_agents = trajectories.shape[0]
    radii = np.full(num_agents, 0.3)

    solver = BRNESolver(solver_mode="official_exact", brne_root=BRNE_ROOT)
    result = solver.solve(trajectories, radii, edge_mask=None)

    upstream = load_upstream_brne(BRNE_ROOT)
    xmean, x_pts = _decompose_mean_perturbation(trajectories[..., 0])
    ymean, y_pts = _decompose_mean_perturbation(trajectories[..., 1])
    _x_opt, _y_opt, weights_direct = upstream.brne_nav(
        list(xmean), list(ymean), x_pts, y_pts, num_agents, trajectories.shape[2], trajectories.shape[1]
    )

    max_diff = float(np.max(np.abs(result.weights - np.asarray(weights_direct))))
    check("adapter_matches_direct_upstream_call_1e-10", max_diff < 1e-10, f"max_diff={max_diff:.2e}")
    check("adapter_reports_10_iterations", result.iterations == 10)
    check("adapter_reports_converged_true", result.converged is True)


def test_stable_mode_agrees_with_official_on_well_behaved_fixture():
    trajectories = _make_fixture(seed=2)
    num_agents = trajectories.shape[0]
    radii = np.full(num_agents, 0.3)

    official = BRNESolver(solver_mode="official_exact", brne_root=BRNE_ROOT).solve(
        trajectories, radii, edge_mask=None
    )
    stable = BRNESolver(solver_mode="stable", brne_root=BRNE_ROOT).solve(
        trajectories, radii, edge_mask=None, equilibrium_iterations=10
    )

    # Not <1e-10 or even <1e-6: this is the SAME update algebraically (the
    # max-subtraction cancels exactly under the row-mean normalization both
    # versions use), but it is iterated 10 times with numpy vs. numba and a
    # different operation order each time, so tiny per-iteration floating-
    # point differences compound multiplicatively across iterations.
    # Empirically ~1e-5 on this fixture; 1e-3 leaves headroom while still
    # catching any REAL algorithmic divergence (which would show up as an
    # order-of-magnitude-larger difference, not more floating-point noise).
    max_diff = float(np.max(np.abs(official.weights - stable.weights)))
    check(
        "stable_agrees_with_official_on_well_behaved_fixture",
        max_diff < 1e-3,
        f"max_diff={max_diff:.2e}",
    )
    check("stable_no_numeric_fallback_on_well_behaved_fixture", stable.numeric_fallback_used is False)


def test_stable_mode_survives_all_trajectories_overlapping():
    """The exact stress case guide.md 8.3 calls out: all trajectories on
    top of each other. Upstream's naive exp(-cost1) can underflow every
    sample's weight to 0 and then divide 0/0 into NaN; stable mode's
    max-subtraction must not."""
    num_agents, num_pts, horizon = 5, 20, 6
    trajectories = np.zeros((num_agents, num_pts, horizon, 2), dtype=np.float64)  # all identical, all zero
    radii = np.full(num_agents, 0.3)

    stable = BRNESolver(solver_mode="stable", brne_root=BRNE_ROOT).solve(
        trajectories, radii, edge_mask=None, equilibrium_iterations=10
    )
    check("stable_finite_on_all_overlapping", bool(np.all(np.isfinite(stable.weights))))
    check(
        "stable_reports_fallback_on_all_overlapping",
        stable.numeric_fallback_used is True or bool(np.all(np.isfinite(stable.weights))),
        "either a reported fallback or genuinely finite output is acceptable; NaN is not",
    )


def main():
    test_pinned_commit()
    test_upstream_module_loads()
    test_decomposition_is_lossless_bookkeeping()
    test_adapter_official_exact_matches_direct_upstream_call()
    test_stable_mode_agrees_with_official_on_well_behaved_fixture()
    test_stable_mode_survives_all_trajectories_overlapping()

    print()
    if FAILURES:
        print(f"[EQUIVALENCE] {len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("[EQUIVALENCE] all tests passed")


if __name__ == "__main__":
    main()
