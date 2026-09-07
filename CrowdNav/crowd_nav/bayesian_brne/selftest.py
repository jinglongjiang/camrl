#!/usr/bin/env python3
"""Pure-CPU selftest suite for SM-BRNE (guide.md 8.1's 24 items).

Tests are added incrementally as each build step lands (Step 2 adds the
``solver`` group, Step 3 adds ``belief``/``sampling``, Step 4 adds
``protocol``, etc.) -- this file is NOT written once at the end. Any failing
test blocks the next build step; see README.md for the step order.

Usage:
    python3 -m crowd_nav.bayesian_brne.selftest              # run everything registered so far
    python3 -m crowd_nav.bayesian_brne.selftest --group solver
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

FAILURES: List[str] = []

# name -> (group, test_function). Populated by register() calls below as
# each module's tests are added; never a single 2000-line file.
_REGISTRY: Dict[str, Tuple[str, Callable[[], None]]] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def register(name: str, group: str):
    def _decorator(fn: Callable[[], None]):
        _REGISTRY[name] = (group, fn)
        return fn

    return _decorator


# --------------------------------------------------------------------- #
# group="solver" (Step 2: brne_adapter.py) -- guide.md 8.1 items 18, 19, 22.
# --------------------------------------------------------------------- #

_BRNE_ROOT = os.environ.get("SM_BRNE_UPSTREAM_ROOT", "/home/abc/temp/brne")


@register("solver_symmetric_corridor_weights", "solver")
def _test_solver_symmetric_corridor_weights():
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver

    num_pts, horizon = 16, 6
    rng = np.random.default_rng(7)
    # Agent 0 walks +x with samples spread laterally in y; agent 1 is agent
    # 0's exact mirror image (walks -x, mirrored y). The joint cost
    # landscape is symmetric under swapping the two agents, so their
    # post-equilibrium weight profiles must match.
    lateral = rng.normal(scale=0.3, size=num_pts)
    base_forward = np.linspace(0.0, 1.5, horizon)

    traj0 = np.zeros((num_pts, horizon, 2))
    traj1 = np.zeros((num_pts, horizon, 2))
    for m in range(num_pts):
        traj0[m, :, 0] = -2.0 + base_forward
        traj0[m, :, 1] = lateral[m]
        traj1[m, :, 0] = 2.0 - base_forward
        traj1[m, :, 1] = -lateral[m]

    trajectories = np.stack([traj0, traj1], axis=0)
    radii = np.full(2, 0.3)
    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)
    result = solver.solve(trajectories, radii, edge_mask=None, equilibrium_iterations=10)

    # Agent 1's weight profile should match agent 0's under the sign-flip
    # symmetry of the lateral sample ordering (same lateral offsets, just
    # negated) -- compare sorted weight distributions rather than assuming
    # sample index correspondence.
    diff = float(np.max(np.abs(np.sort(result.weights[0]) - np.sort(result.weights[1]))))
    check("solver_symmetric_corridor_weights", diff < 1e-6, f"max_diff={diff:.2e}")


@register("solver_collision_trajectories_get_lower_weight", "solver")
def _test_solver_collision_trajectories_get_lower_weight():
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver

    horizon = 6
    # Agent 0 has two candidate samples: one drives straight through agent
    # 1's position (collision), one detours around it (safe). Agent 1 is a
    # single stationary trajectory sitting in agent 0's path.
    collide = np.zeros((horizon, 2))
    collide[:, 0] = np.linspace(-1.0, 1.0, horizon)  # passes through x=0 where agent 1 sits
    safe = np.zeros((horizon, 2))
    safe[:, 0] = np.linspace(-1.0, 1.0, horizon)
    safe[:, 1] = 2.0  # 2m lateral offset -- clearly safe

    traj0 = np.stack([collide, safe], axis=0)  # [2 samples, horizon, 2]
    traj1 = np.zeros((2, horizon, 2))  # agent 1: both samples stationary at origin

    trajectories = np.stack([traj0, traj1], axis=0)  # [2 agents, 2 samples, horizon, 2]
    radii = np.full(2, 0.3)
    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)
    result = solver.solve(trajectories, radii, edge_mask=None, equilibrium_iterations=10)

    weight_collide, weight_safe = result.weights[0, 0], result.weights[0, 1]
    check(
        "solver_collision_trajectories_get_lower_weight",
        weight_safe > weight_collide,
        f"weight_safe={weight_safe:.4f} weight_collide={weight_collide:.4f}",
    )


@register("solver_numeric_fallback_counted_no_nan", "solver")
def _test_solver_numeric_fallback_counted_no_nan():
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver

    num_agents, num_pts, horizon = 6, 24, 6
    trajectories = np.zeros((num_agents, num_pts, horizon, 2), dtype=np.float64)
    radii = np.full(num_agents, 0.3)
    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)
    result = solver.solve(trajectories, radii, edge_mask=None, equilibrium_iterations=10)

    check("solver_no_nan_on_degenerate_overlap", bool(np.all(np.isfinite(result.weights))))
    check(
        "solver_numeric_fallback_field_is_bool",
        isinstance(result.numeric_fallback_used, bool),
        f"got {type(result.numeric_fallback_used)}",
    )


@register("solver_clearance_cost_is_radius_aware", "solver")
def _test_solver_clearance_cost_is_radius_aware():
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver

    horizon = 6
    collide = np.zeros((horizon, 2))
    collide[:, 0] = np.linspace(-1.0, 1.0, horizon)
    safe = np.zeros((horizon, 2))
    safe[:, 0] = np.linspace(-1.0, 1.0, horizon)
    safe[:, 1] = 2.0
    trajectories = np.stack([np.stack([collide, safe]), np.zeros((2, horizon, 2))], axis=0)

    small = BRNESolver(solver_mode="stable_clearance", brne_root=_BRNE_ROOT, safe_distance=0.2, cost_sigma=0.1)
    large = BRNESolver(solver_mode="stable_clearance", brne_root=_BRNE_ROOT, safe_distance=0.2, cost_sigma=0.1)
    result_small = small.solve(trajectories, np.array([0.01, 0.3]), None, equilibrium_iterations=10)
    result_large = large.solve(trajectories, np.array([0.8, 0.3]), None, equilibrium_iterations=10)
    check("solver_clearance_safe_candidate_beats_collision", result_small.weights[0, 1] > result_small.weights[0, 0],
          f"small-radius weights={result_small.weights[0]}")
    check("solver_clearance_larger_radius_changes_weights",
          not np.array_equal(result_small.weights[0], result_large.weights[0]),
          f"small={result_small.weights[0]} large={result_large.weights[0]}")


@register("solver_clearance_cost_monotonic_and_finite", "solver")
def _test_solver_clearance_cost_monotonic_and_finite():
    from crowd_nav.bayesian_brne.brne_adapter import _costs_clearance

    traj_x = np.array([[0.0], [1.0]], dtype=np.float64)
    traj_y = np.array([[0.0], [0.0]], dtype=np.float64)
    low = _costs_clearance(traj_x, traj_y, np.array([0.1, 0.1]), 2, 1, 0.1, 0.1, 100.0)
    high = _costs_clearance(traj_x, traj_y, np.array([0.6, 0.1]), 2, 1, 0.1, 0.1, 100.0)
    overlap = _costs_clearance(
        np.zeros((8, 3)), np.zeros((8, 3)), np.full(2, 0.3), 2, 4, 0.2, 1e-6, 100.0,
    )
    check("solver_clearance_larger_radius_increases_pair_cost", high[0, 1] > low[0, 1],
          f"low={low[0,1]:.6f} high={high[0,1]:.6f}")
    check("solver_clearance_diagonal_is_zero", bool(np.allclose(np.diag(high), 0.0)))
    check("solver_clearance_extreme_overlap_is_finite", bool(np.all(np.isfinite(overlap))))


@register("solver_clearance_and_official_modes_are_explicit", "solver")
def _test_solver_clearance_and_official_modes_are_explicit():
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver

    rejected_official = False
    try:
        BRNESolver(solver_mode="official_exact", safe_distance=0.2)
    except ValueError:
        rejected_official = True
    check("official_exact_rejects_radius_parameters", rejected_official)
    bad_values = []
    for kwargs in (
        {"safe_distance": -0.1}, {"cost_sigma": 0.0}, {"cost_scale": 0.0},
    ):
        try:
            BRNESolver(solver_mode="stable_clearance", **kwargs)
        except ValueError:
            bad_values.append(True)
    check("stable_clearance_rejects_invalid_parameters", len(bad_values) == 3, str(bad_values))


# --------------------------------------------------------------------- #
# group="belief" (Step 3: belief_tracker.py) -- guide.md 8.1 items 3-7.
# --------------------------------------------------------------------- #


def _make_synthetic_artifact(K=2):
    """Revised for B3 (2026-08-03 audit): F magnitudes chosen so that, even
    with NO acceleration/speed clipping engaged (i.e. the fixture itself is
    physically reasonable, not just saved by the clip), a 10-step / 2.5s
    rollout produces a final lateral divergence of roughly 0.5-2.0m between
    the two modes -- a real pedestrian-scale bifurcation, not the previous
    7.55m fixture that only "worked" because nothing bounded it."""
    from crowd_nav.bayesian_brne.mode_model import ModeModelArtifact

    F = [np.zeros((2, 10)) for _ in range(K)]
    Q = [1e-3 * np.eye(2) for _ in range(K)]
    if K == 2:
        F[0][1, 0] = 0.09   # mode 0: at speed=phi[0]=0.8, adds +0.072 m/s of lateral velocity per 0.25s step
        F[1][1, 0] = -0.09  # mode 1: symmetric, -0.072 m/s per step
    Pi = np.full((K, K), 0.05 / max(1, K - 1))
    np.fill_diagonal(Pi, 0.95)
    feature_mean = np.zeros(10)
    feature_std = np.ones(10)
    return ModeModelArtifact(
        K=K, F=F, Q=Q, Pi=Pi, feature_mean=feature_mean, feature_std=feature_std,
        cluster_centers=np.zeros((K, 10)), dt=0.25, model_card={},
    )


# --------------------------------------------------------------------- #
# group="mode" (Step 3: mode_model.py) -- 2026-08-03 audit items B5, B6, B8
# plus the K-selection/roundtrip regressions the audit noted were only ever
# verified by a one-off shell script, not committed to the repo (D6).
# --------------------------------------------------------------------- #


def _make_bimodal_episodes(rng, n_episodes=30, T=40, N=2, dt=0.25):
    episodes = []
    for i in range(n_episodes):
        mode = "decel" if i % 2 == 0 else "turn"
        humans = np.zeros((T, N, 5))
        track_ids = np.tile(np.arange(N), (T, 1))
        robot = np.zeros((T, 9))
        robot[:, 6:8] = [5.0, 0.0]  # gx, gy -- deliberately far from vx,vy=0 so a
        # field-order bug (B8) would be obvious, not accidentally benign.
        valid = np.ones((T, N), dtype=bool)
        for n in range(N):
            px, py, vx, vy = 0.0, float(n), 0.8, 0.0
            for t in range(T):
                humans[t, n] = [px, py, vx, vy, 0.3]
                if mode == "decel":
                    vx = max(vx - 0.05, 0.05)
                else:
                    theta = np.arctan2(vy, vx) + 0.05
                    speed = np.hypot(vx, vy)
                    vx, vy = speed * np.cos(theta), speed * np.sin(theta)
                px += vx * dt
                py += vy * dt
        episodes.append({"humans": humans, "human_track_ids": track_ids, "robot": robot, "valid_mask": valid})
    return episodes


def _make_unimodal_episodes(n_episodes=30, T=40, N=2, dt=0.25):
    episodes = []
    for i in range(n_episodes):
        humans = np.zeros((T, N, 5))
        track_ids = np.tile(np.arange(N), (T, 1))
        robot = np.zeros((T, 9))
        robot[:, 6:8] = [5.0, 0.0]
        valid = np.ones((T, N), dtype=bool)
        for n in range(N):
            px, py, vx, vy = 0.0, float(n), 0.7, 0.0  # everyone behaves identically: constant velocity
            for t in range(T):
                humans[t, n] = [px, py, vx, vy, 0.3]
                px += vx * dt
                py += vy * dt
        episodes.append({"humans": humans, "human_track_ids": track_ids, "robot": robot, "valid_mask": valid})
    return episodes


@register("mode_selects_k2_on_bimodal_fixture", "mode")
def _test_mode_selects_k2_on_bimodal_fixture():
    from crowd_nav.bayesian_brne.config import BayesianModelConfig
    from crowd_nav.bayesian_brne.mode_model import extract_transitions, fit_and_select

    rng = np.random.default_rng(0)
    train_eps = _make_bimodal_episodes(rng, n_episodes=30)
    val_eps = _make_bimodal_episodes(rng, n_episodes=10)
    train_rows = extract_transitions(train_eps, dt=0.25)
    val_rows = extract_transitions(val_eps, dt=0.25)

    config = BayesianModelConfig(k_candidates=(1, 2, 3))
    artifact, reports = fit_and_select(train_rows, val_rows, config, seed=2407, require_multimodal=True)
    check("mode_bimodal_fixture_selects_k2", artifact.K == 2, f"got K={artifact.K}")
    for r in reports:
        print(f"    K={r.K} eligible={r.eligible} nll={r.held_out_nll:.4f} predictive_sim={r.max_predictive_similarity:.3f} reasons={r.rejection_reasons}")


@register("mode_main_fit_rejects_unimodal_fixture", "mode")
def _test_mode_main_fit_rejects_unimodal_fixture():
    """B5: fitting on data with NO real behavioral bimodality, with
    require_multimodal=True (the only mode the real artifact may use), must
    raise -- not silently return K=1 and let the caller believe a
    switching-mode artifact was produced."""
    from crowd_nav.bayesian_brne.config import BayesianModelConfig
    from crowd_nav.bayesian_brne.mode_model import extract_transitions, fit_and_select

    train_rows = extract_transitions(_make_unimodal_episodes(30), dt=0.25)
    val_rows = extract_transitions(_make_unimodal_episodes(10), dt=0.25)
    config = BayesianModelConfig(k_candidates=(1, 2, 3))

    raised = False
    try:
        fit_and_select(train_rows, val_rows, config, seed=2407, require_multimodal=True)
    except RuntimeError:
        raised = True
    check("mode_unimodal_fixture_raises_with_require_multimodal", raised)

    # allow_unimodal path (require_multimodal=False) must still succeed,
    # for the explicit K=1/legacy baseline use case.
    artifact, _ = fit_and_select(train_rows, val_rows, config, seed=2407, require_multimodal=False)
    check("mode_unimodal_fixture_allows_k1_when_not_required_multimodal", artifact.K == 1, f"got K={artifact.K}")


@register("mode_artifact_roundtrip_exact", "mode")
def _test_mode_artifact_roundtrip_exact():
    import tempfile
    from pathlib import Path
    from crowd_nav.bayesian_brne.config import BayesianModelConfig
    from crowd_nav.bayesian_brne.mode_model import ModeModelArtifact, extract_transitions, fit_and_select

    rng = np.random.default_rng(1)
    rows = extract_transitions(_make_bimodal_episodes(rng, n_episodes=20), dt=0.25)
    config = BayesianModelConfig(k_candidates=(1, 2))
    artifact, _ = fit_and_select(rows, rows, config, seed=7, require_multimodal=True)
    with tempfile.TemporaryDirectory() as d:
        artifact.save(Path(d))
        loaded = ModeModelArtifact.load(Path(d))
    check("mode_roundtrip_k_matches", loaded.K == artifact.K)
    check("mode_roundtrip_pi_exact", bool(np.array_equal(loaded.Pi, artifact.Pi)))
    check("mode_roundtrip_f_exact", all(np.array_equal(loaded.F[k], artifact.F[k]) for k in range(artifact.K)))
    check("mode_roundtrip_q_exact", all(np.array_equal(loaded.Q[k], artifact.Q[k]) for k in range(artifact.K)))


@register("mode_similarity_uses_predictive_distribution", "mode")
def _test_mode_similarity_uses_predictive_distribution():
    """B6: two modes with IDENTICAL F but very different Q are genuinely
    different uncertainty regimes and must NOT be flagged as near-duplicate
    just because F-cosine similarity is 1.0. Two modes with different F but
    (on the actual validation feature range) nearly identical predicted
    distributions SHOULD be flagged."""
    from crowd_nav.bayesian_brne.mode_model import _bhattacharyya_coefficient, _mode_similarity

    F_a = np.zeros((2, 10)); F_a[1, 0] = 0.1
    F_b = F_a.copy()  # identical F
    Q_low = 1e-4 * np.eye(2)
    Q_high = 1.0 * np.eye(2)  # very different uncertainty

    cosine_sim = _mode_similarity(F_a, F_b)
    check("mode_similarity_identical_f_cosine_is_1", abs(cosine_sim - 1.0) < 1e-9)

    mu = F_a @ np.array([0.8] + [0.0] * 9)
    predictive_sim_different_q = _bhattacharyya_coefficient(mu, Q_low, mu, Q_high)
    check(
        "mode_predictive_similarity_low_despite_identical_f",
        predictive_sim_different_q < 0.5,
        f"got {predictive_sim_different_q:.4f} (same F, very different Q should NOT look like duplicate modes)",
    )

    # Different F, but at this particular phi the predicted means coincide
    # and Q is identical -> should be flagged as a near-duplicate at THIS phi.
    F_c = np.zeros((2, 10)); F_c[1, 5] = 100.0  # depends on a feature that's 0 in our probe phi
    mu_c = F_c @ np.array([0.8] + [0.0] * 9)  # == 0 vector since phi[5]=0
    predictive_sim_same_pred = _bhattacharyya_coefficient(mu, Q_low, mu_c, Q_low) if np.allclose(mu, mu_c) else None
    if predictive_sim_same_pred is not None:
        check("mode_predictive_similarity_high_when_predictions_coincide", predictive_sim_same_pred > 0.9)


@register("robot_schema_velocity_goal_not_swapped", "mode")
def _test_robot_schema_velocity_goal_not_swapped():
    """B8: extract_transitions must read vx,vy from indices 2:4 of the
    canonical FullState.to_array() layout, never accidentally reading
    Robot.get_obs_array()'s gx,gy (which sit at the SAME indices under that
    OTHER function's different field order). Constructs a robot row where
    gx,gy are wildly different from vx,vy so any accidental swap is
    impossible to miss."""
    from crowd_nav.bayesian_brne.mode_model import extract_transitions

    T, N, dt = 5, 1, 0.25
    humans = np.zeros((T, N, 5))
    track_ids = np.zeros((T, N), dtype=np.int64)
    valid = np.ones((T, N), dtype=bool)
    robot = np.zeros((T, 9))
    for t in range(T):
        humans[t, 0] = [float(t) * 0.2, 0.0, 0.8, 0.3, 0.3]  # human moving with vx=0.8, vy=0.3
        # Canonical layout: px,py,vx,vy,radius,gx,gy,v_pref,theta.
        # Robot velocity is tiny (0.01, 0.01); goal is huge (999, 999) --
        # if extract_transitions ever reads indices [2:4] from a
        # get_obs_array()-ordered input it would see (999, 999) instead.
        robot[t] = [0.0, 0.0, 0.01, 0.01, 0.3, 999.0, 999.0, 1.0, 0.0]

    rows = extract_transitions([{"humans": humans, "human_track_ids": track_ids, "robot": robot, "valid_mask": valid}], dt=dt)
    check("robot_schema_rows_extracted", len(rows) > 0)
    for row in rows:
        # relative_vx/vy = human_v - robot_v; with robot_v=(0.01,0.01), this
        # should be close to (0.79, 0.29), NOT close to (0.8-999, 0.3-999).
        rel_vx, rel_vy = row.phi[6], row.phi[7]
        check(
            "robot_schema_relative_velocity_not_polluted_by_goal",
            abs(rel_vx - 0.79) < 0.05 and abs(rel_vy - 0.29) < 0.05,
            f"got rel_v=({rel_vx:.4f}, {rel_vy:.4f}), expected ~(0.79, 0.29) -- "
            f"if this is instead close to (0.8-999, 0.3-999) the goal/velocity fields were swapped",
        )


@register("mode_extract_transitions_rejects_non_consecutive_gap", "mode")
def _test_mode_extract_transitions_rejects_non_consecutive_gap():
    """D3 (2026-08-03 audit): an earlier version accepted tracks with a gap
    of up to 2 raw steps and still fed the raw ``v_t1 - v_t`` straight into
    fitting as if it were a one-``dt`` delta -- biasing ``F_k`` upward since
    a 2-step gap's real velocity change is roughly double a genuine 1-step
    delta for the same phi. Builds one track with a skipped frame (t=2
    missing) and one fully consecutive track with an IDENTICAL constant
    acceleration, and requires: (a) the gapped track contributes strictly
    fewer transition rows than the consecutive one (the t=1->t=3 gap must be
    dropped, not silently kept), and (b) every emitted row's ``step_dt``
    equals the base ``dt`` exactly (never ``2*dt``)."""
    from crowd_nav.bayesian_brne.mode_model import extract_transitions

    dt = 0.25
    T, accel = 6, 0.1

    def _build(skip_index):
        humans = np.zeros((T, 1, 5))
        track_ids = np.zeros((T, 1), dtype=np.int64)
        valid = np.ones((T, 1), dtype=bool)
        robot = np.zeros((T, 9))
        px, py, vx, vy = 0.0, 0.0, 0.5, 0.0
        for t in range(T):
            humans[t, 0] = [px, py, vx, vy, 0.3]
            vx += accel * dt
            px += vx * dt
        if skip_index is not None:
            valid[skip_index, 0] = False
        return {"humans": humans, "human_track_ids": track_ids, "robot": robot, "valid_mask": valid}

    rows_consecutive = extract_transitions([_build(None)], dt=dt)
    rows_gapped = extract_transitions([_build(2)], dt=dt)

    check(
        "mode_gapped_track_yields_fewer_rows_than_consecutive",
        len(rows_gapped) < len(rows_consecutive),
        f"consecutive={len(rows_consecutive)} rows, gapped(skip t=2)={len(rows_gapped)} rows -- "
        "the t=1->t=3 gap transition must be dropped entirely, not kept with a doubled implicit dt",
    )
    check(
        "mode_all_emitted_step_dt_equal_base_dt",
        all(abs(row.step_dt - dt) < 1e-9 for row in rows_consecutive + rows_gapped),
        f"step_dts consecutive={[r.step_dt for r in rows_consecutive]} gapped={[r.step_dt for r in rows_gapped]} "
        f"-- every emitted row must be a strictly single-dt={dt} transition",
    )


def _make_synthetic_arhmm_artifact(K=2):
    """Order F2 (2026-08-03): hand-built ``ARHMMArtifact`` (not fit via EM)
    for belief_tracker tests. ``A_k`` is a mild, shared autoregression;
    ``B_k`` is STRONG and OPPOSITE-SIGN per mode, so the robot's action has
    a real, mode-discriminating causal effect on the predicted velocity
    change -- this is what makes
    ``arhmm_belief_action_misalignment_degrades_recovery`` meaningful: if
    the tracker did not actually use ``u_r`` (or used it with the wrong
    timing), that test would show no difference between correct and
    misaligned action feeding. ``C_k=0``/``d_k=0`` since context is not the
    focus of these tests; ``Q_k`` small and isotropic."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact, CONTEXT_FEATURE_NAMES

    d_c = len(CONTEXT_FEATURE_NAMES)
    A = [0.3 * np.eye(2) for _ in range(K)]
    if K == 2:
        B = [0.6 * np.eye(2), -0.6 * np.eye(2)]
    else:
        B = [0.6 * np.eye(2) for _ in range(K)]
    C = [np.zeros((2, d_c)) for _ in range(K)]
    d = [np.zeros(2) for _ in range(K)]
    Q = [0.01 * np.eye(2) for _ in range(K)]
    Pi = np.full((K, K), 0.05 / max(1, K - 1))
    np.fill_diagonal(Pi, 0.95)
    initial_distribution = np.full(K, 1.0 / K)
    return ARHMMArtifact(K=K, A=A, B=B, C=C, d=d, Q=Q, Pi=Pi,
                          initial_distribution=initial_distribution, dt=0.25, model_card={})


@register("belief_converges_on_clean_mode_evidence", "belief")
def _test_belief_converges_on_clean_mode_evidence():
    """The convergence property behind every other belief test in this
    file, made an explicit, standalone, committed regression test (D6 from
    the 2026-08-03 audit: this exact number was previously only ever
    verified via a one-off shell script, not the repo's own test suite).
    Order F2 rewrite: drives a FIXED robot action through mode 0's own
    (A, B) dynamics, noise-free, so the tracker sees unambiguous
    mode-0-consistent evidence at every step."""
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    bank = BeliefBank(artifact, seed=1)
    bank.reset(episode_seed=1)
    v = np.zeros(2)
    u_r = np.array([0.5, 0.0])
    px, py = 0.0, 0.0
    for t in range(30):
        obs = TrackObservation(track_id=1, px=px, py=py, vx=v[0], vy=v[1], radius=0.3, timestamp=t * 0.25)
        bank.update([obs], robot_action=tuple(u_r), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)
        # Evidence consistent with mode 0's own prediction at every step
        # (not a static/symmetric state, which -- as
        # belief_posterior_valid_and_reset deliberately demonstrates --
        # gives uninformative evidence between two symmetric modes).
        v = artifact.A[0] @ v + artifact.B[0] @ u_r
        px += v[0] * 0.25
        py += v[1] * 0.25
    posterior = bank.posterior(1)
    check(
        "belief_converges_to_confident_correct_mode",
        bool(posterior[0] >= 0.94),
        f"got {posterior} after 30 steps of clean mode-0-consistent evidence",
    )


@register("belief_pi_q_valid", "belief")
def _test_belief_pi_q_valid():
    artifact = _make_synthetic_arhmm_artifact(K=2)
    check("belief_pi_rows_normalized", bool(np.allclose(artifact.Pi.sum(axis=1), 1.0)))
    check("belief_q_positive_definite", all(np.all(np.linalg.eigvalsh(q) > 0) for q in artifact.Q))


@register("belief_posterior_valid_and_reset", "belief")
def _test_belief_posterior_valid_and_reset():
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    bank = BeliefBank(artifact, seed=1)
    bank.reset(episode_seed=1)
    for t in range(10):
        obs = TrackObservation(track_id=1, px=0.0, py=0.0, vx=0.8, vy=0.0, radius=0.3, timestamp=t * 0.25)
        bank.update([obs], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)
        p = bank.posterior(1)
        check(f"belief_posterior_valid_t{t}", bool(np.all(p >= 0) and np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-9), str(p))

    bank.reset(episode_seed=2)
    check("belief_reset_clears_tracks", bank.active_track_ids() == [])
    check(
        "belief_reset_returns_artifact_initial_for_unknown_track",
        bool(np.allclose(bank.posterior(1), artifact.initial_distribution)),
    )


@register("belief_track_identity_order_independent", "belief")
def _test_belief_track_identity_order_independent():
    """Feeding the same two tracks' observations in different LIST ORDER
    must not change either track's resulting posterior -- identity is keyed
    by track_id, never by position in the observations list (guide.md 5.6)."""
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    bank_a = BeliefBank(artifact, seed=1)
    bank_b = BeliefBank(artifact, seed=1)
    bank_a.reset(1)
    bank_b.reset(1)

    for t in range(8):
        obs1 = TrackObservation(track_id=1, px=0.0, py=0.0, vx=0.8, vy=0.1, radius=0.3, timestamp=t * 0.25)
        obs2 = TrackObservation(track_id=2, px=1.0, py=1.0, vx=-0.5, vy=0.2, radius=0.3, timestamp=t * 0.25)
        bank_a.update([obs1, obs2], robot_action=(0.2, -0.1), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)
        bank_b.update([obs2, obs1], robot_action=(0.2, -0.1), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)  # reversed order

    check("belief_order_independent_track1", bool(np.allclose(bank_a.posterior(1), bank_b.posterior(1))))
    check("belief_order_independent_track2", bool(np.allclose(bank_a.posterior(2), bank_b.posterior(2))))


@register("belief_fresh_deleted_track_uses_artifact_initial", "r1a")
def _test_belief_fresh_deleted_track_uses_artifact_initial():
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    artifact.initial_distribution = np.array([0.91, 0.09])
    bank = BeliefBank(artifact, max_missed_steps=0, seed=1)
    bank.reset(1)
    obs = TrackObservation(track_id=7, px=0.0, py=0.0, vx=0.2, vy=0.0, radius=0.3, timestamp=0.0)
    bank.update([obs], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0,
                robot_vx=0.0, robot_vy=0.0, current_timestamp=0.0)
    first = bank.posterior(7)
    check("r1a_fresh_track_initial_distribution_exact", bool(np.allclose(first, artifact.initial_distribution)), str(first))
    bank.update([], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0,
                robot_vx=0.0, robot_vy=0.0, current_timestamp=0.25)
    reappeared = TrackObservation(track_id=7, px=0.5, py=0.0, vx=0.2, vy=0.0, radius=0.3, timestamp=0.5)
    bank.update([reappeared], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0,
                robot_vx=0.0, robot_vy=0.0, current_timestamp=0.5)
    check("r1a_deleted_track_recreation_uses_artifact_initial", bool(np.allclose(bank.posterior(7), artifact.initial_distribution)), str(bank.posterior(7)))
    check("r1a_unknown_track_uses_artifact_initial", bool(np.allclose(bank.posterior(99), artifact.initial_distribution)))


@register("belief_bayes_sequence_matches_hand_calculation", "r1a")
def _test_belief_bayes_sequence_matches_hand_calculation():
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank, _emission_log_prob, _build_context
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    artifact.initial_distribution = np.array([0.8, 0.2])
    bank = BeliefBank(artifact, seed=1)
    bank.reset(1)
    obs0 = TrackObservation(track_id=1, px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, timestamp=0.0)
    obs1 = TrackObservation(track_id=1, px=0.02, py=0.0, vx=0.3, vy=0.0, radius=0.3, timestamp=0.25)
    action = np.array([0.2, 0.0])
    bank.update([obs0], robot_action=tuple(action), robot_px=0.0, robot_py=0.0,
                robot_vx=0.0, robot_vy=0.0, current_timestamp=0.0)
    context0 = _build_context(obs0, 0.0, 0.0, 0.0, 0.0)
    log_terms = np.array([
        np.log(artifact.initial_distribution[k]) + _emission_log_prob(
            np.array([obs1.vx, obs1.vy]), np.array([obs0.vx, obs0.vy]), action, context0,
            artifact.A[k], artifact.B[k], artifact.C[k], artifact.d[k], artifact.Q[k],
        ) for k in range(artifact.K)
    ])
    scaled = np.exp(log_terms - log_terms.max())
    filtered = scaled / scaled.sum()
    expected_next = filtered @ artifact.Pi
    bank.update([obs1], robot_action=tuple(action), robot_px=0.0, robot_py=0.0,
                robot_vx=0.0, robot_vy=0.0, current_timestamp=0.25)
    check("r1a_hand_bayes_next_prior_matches", bool(np.allclose(bank.posterior(1), expected_next, atol=1e-12)),
          f"got={bank.posterior(1)} expected={expected_next}")


@register("belief_global_timestamp_fail_closed", "r1a")
def _test_belief_global_timestamp_fail_closed():
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank, BeliefTimestampError
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    obs = TrackObservation(track_id=1, px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, timestamp=0.0)

    def raises(second_timestamp):
        bank = BeliefBank(artifact, seed=1)
        bank.reset(1)
        bank.update([obs], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0,
                    robot_vx=0.0, robot_vy=0.0, current_timestamp=0.0)
        try:
            bank.update([], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0,
                        robot_vx=0.0, robot_vy=0.0, current_timestamp=second_timestamp)
        except BeliefTimestampError:
            return True
        return False

    check("r1a_duplicate_timestamp_rejected", raises(0.0))
    check("r1a_backwards_timestamp_rejected", raises(-0.25))
    check("r1a_skipped_global_timestamp_rejected", raises(0.5))


@register("sampling_public_inputs_fail_closed", "r1a")
def _test_sampling_public_inputs_fail_closed():
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior

    artifact = _make_synthetic_arhmm_artifact(K=2)
    state0 = np.zeros(4)
    posterior = np.array([0.5, 0.5])
    pos, vel, act = _stationary_robot_sequences(3)

    def rejects(posterior_value=posterior, position_value=pos, action_value=act):
        try:
            sample_full_posterior(
                posterior_value, state0, artifact, position_value, vel, action_value,
                horizon=3, num_samples=2, rng=np.random.default_rng(1),
            )
        except ValueError:
            return True
        return False

    bad_horizon_pos = np.zeros((2, 2))
    check("r1a_sampler_rejects_time_length_mismatch", rejects(position_value=bad_horizon_pos))
    bad_action = act.copy()
    bad_action[0, 0] = np.nan
    check("r1a_sampler_rejects_nonfinite_time_array", rejects(action_value=bad_action))
    check("r1a_sampler_rejects_unnormalized_posterior", rejects(posterior_value=np.array([0.6, 0.6])))


@register("belief_missing_advances_once", "belief")
def _test_belief_missing_advances_once():
    """B7: a track missing for one real timestep must be advanced by the
    Bayes prior EXACTLY once for that timestep -- not once by ``update([])``
    and then again by a separately-documented "predict" call (the old public
    ``predict_only`` no longer exists; ``update()`` is the sole entry point)."""
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    bank = BeliefBank(artifact, max_missed_steps=5, seed=1)
    bank.reset(1)
    # Drive posterior to a confident, NON-UNIFORM, NON-FIXED-POINT state
    # first. [0.5, 0.5] is a fixed point of this fixture's symmetric Pi
    # (0.5*0.95 + 0.5*0.05 == 0.5) -- one vs. two Pi applications are
    # IDENTICAL there, so a test that never leaves that fixed point cannot
    # actually distinguish "advanced once" from "advanced twice".
    v = np.zeros(2)
    u_r = np.array([0.5, 0.0])
    px, py = 0.0, 0.0
    for t in range(20):
        obs = TrackObservation(track_id=1, px=px, py=py, vx=v[0], vy=v[1], radius=0.3, timestamp=t * 0.25)
        bank.update([obs], robot_action=tuple(u_r), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)
        v = artifact.A[0] @ v + artifact.B[0] @ u_r
        px += v[0] * 0.25
        py += v[1] * 0.25
    posterior_before = bank.posterior(1).copy()
    check(
        "belief_missing_test_setup_is_not_at_fixed_point",
        bool(abs(posterior_before[0] - 0.5) > 0.1),
        f"posterior_before={posterior_before} -- must be measurably away from the symmetric "
        "[0.5,0.5] fixed point, otherwise this test cannot distinguish one vs. two Pi applications",
    )

    expected_after_one_miss = posterior_before @ artifact.Pi
    expected_after_one_miss = np.maximum(expected_after_one_miss, 1e-12)
    expected_after_one_miss /= expected_after_one_miss.sum()

    bank.update([], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=20 * 0.25)  # exactly one missing timestep, sole call
    posterior_after_one_miss = bank.posterior(1)

    check(
        "belief_single_miss_advances_by_exactly_one_pi_step",
        bool(np.allclose(posterior_after_one_miss, expected_after_one_miss, atol=1e-9)),
        f"got {posterior_after_one_miss}, expected {expected_after_one_miss} (one Pi application)",
    )

    # The OLD bug: calling update([]) then ALSO calling a separate predict
    # step for the same timestep would apply Pi twice. Verify two calls to
    # update([]) (two REAL timesteps) equal exactly two Pi applications --
    # i.e. confirms there is no hidden extra advancement per call.
    bank.update([], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=21 * 0.25)
    expected_after_two_misses = expected_after_one_miss @ artifact.Pi
    expected_after_two_misses = np.maximum(expected_after_two_misses, 1e-12)
    expected_after_two_misses /= expected_after_two_misses.sum()
    check(
        "belief_two_misses_advance_by_exactly_two_pi_steps",
        bool(np.allclose(bank.posterior(1), expected_after_two_misses, atol=1e-9)),
    )


@register("belief_reappearance_drops_stale_likelihood", "belief")
def _test_belief_reappearance_drops_stale_likelihood():
    """B7: a track reappearing after being missing for 2 steps must NOT have
    a likelihood term applied on its first reappearing frame (the gap length
    is unknown/unmodeled, so treating it as a fake one-step transition would
    apply Q_k -- calibrated for ONE real timestep -- to what was actually a
    multi-step, larger displacement, silently overconfident). Its posterior
    on that first reappearing frame must equal exactly what repeated
    Bayes-prior advancement during the gap already produced -- i.e.
    unchanged by the reappearance observation itself."""
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    artifact = _make_synthetic_arhmm_artifact(K=2)
    bank = BeliefBank(artifact, max_missed_steps=5, seed=1)
    bank.reset(1)
    v = np.array([0.8, 0.0])
    u_r = np.array([0.5, 0.0])
    px, py = 0.0, 0.0
    for t in range(20):
        obs = TrackObservation(track_id=1, px=px, py=py, vx=v[0], vy=v[1], radius=0.3, timestamp=t * 0.25)
        bank.update([obs], robot_action=tuple(u_r), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)
        v = artifact.A[0] @ v + artifact.B[0] @ u_r
        px += v[0] * 0.25

    bank.update([], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=20 * 0.25)  # miss 1
    bank.update([], robot_action=(0.0, 0.0), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=21 * 0.25)  # miss 2
    posterior_before_reappear = bank.posterior(1).copy()

    # Reappear with a velocity that would look like a HUGE, mode-diagnostic
    # jump if (incorrectly) treated as a genuine one-step transition from
    # the stale pre-miss velocity.
    reappear_obs = TrackObservation(track_id=1, px=px + 5.0, py=py, vx=5.0, vy=5.0, radius=0.3, timestamp=22 * 0.25)
    bank.update([reappear_obs], robot_action=tuple(u_r), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=22 * 0.25)
    posterior_after_reappear = bank.posterior(1)

    expected_after_reappear = posterior_before_reappear @ artifact.Pi
    expected_after_reappear = np.maximum(expected_after_reappear, 1e-12)
    expected_after_reappear /= expected_after_reappear.sum()
    check(
        "belief_reappearance_advances_one_transition_without_likelihood",
        bool(np.allclose(expected_after_reappear, posterior_after_reappear, atol=1e-9)),
        f"before={posterior_before_reappear} after={posterior_after_reappear} expected={expected_after_reappear} "
        "(reappearance advances one real timestep but must not apply a stale-gap likelihood)",
    )

    # The NEXT (genuinely consecutive) frame must resume normal likelihood
    # updating. Construct its velocity to clearly match mode 0's prediction
    # given the STORED (v=[5,5], u_r) pair from the reappearing frame (not
    # a symmetric jump that would equally support both modes).
    v_reappear = np.array([5.0, 5.0])
    predicted_next = artifact.A[0] @ v_reappear + artifact.B[0] @ u_r
    next_obs = TrackObservation(
        track_id=1, px=px + 5.5, py=py, vx=predicted_next[0], vy=predicted_next[1],
        radius=0.3, timestamp=23 * 0.25,
    )
    bank.update([next_obs], robot_action=tuple(u_r), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=23 * 0.25)
    posterior_next = bank.posterior(1)
    check(
        "belief_resumes_likelihood_update_on_next_consecutive_frame",
        bool(posterior_next[0] > 0.6),
        f"posterior should shift toward mode 0 once a genuinely consecutive, mode-0-consistent "
        f"pair is available: {posterior_next}",
    )


@register("belief_action_misalignment_degrades_recovery", "belief")
def _test_belief_action_misalignment_degrades_recovery():
    """Order F2 acceptance criterion (guide.md 4.2/Order F2): feeding the
    robot's action with a ONE-FRAME timing misalignment (the exact class of
    bug guide.md forbids -- using a candidate/stale action to explain a
    transition it did not actually cause) must measurably DEGRADE mode
    recovery, proving the tracker is genuinely sensitive to getting u_r's
    timing right (if it weren't, this manipulation would have no effect).
    Uses an ALTERNATING true action sequence (not a constant one) so that
    an off-by-one shift actually changes what "explains" each transition --
    a constant action would make correct and misaligned feeding
    indistinguishable."""
    from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
    from crowd_nav.bayesian_brne.schemas import TrackObservation

    def run_trial(misaligned: bool) -> float:
        artifact = _make_synthetic_arhmm_artifact(K=2)
        bank = BeliefBank(artifact, seed=1)
        bank.reset(1)
        true_mode = 0
        v = np.zeros(2)
        px, py = 0.0, 0.0
        last_executed_action = np.zeros(2)
        stale_action = np.zeros(2)  # one-step-old action, simulates the "off by one" bug
        for t in range(40):
            obs = TrackObservation(track_id=1, px=px, py=py, vx=v[0], vy=v[1], radius=0.3, timestamp=t * 0.25)
            fed_action = stale_action if misaligned else last_executed_action
            bank.update([obs], robot_action=tuple(fed_action), robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, current_timestamp=t * 0.25)

            action_now = np.array([1.0, 0.0]) if t % 2 == 0 else np.array([-1.0, 0.0])
            v = artifact.A[true_mode] @ v + artifact.B[true_mode] @ action_now
            px += v[0] * 0.25
            py += v[1] * 0.25
            stale_action = last_executed_action
            last_executed_action = action_now
        return float(bank.posterior(1)[true_mode])

    correct_confidence = run_trial(misaligned=False)
    misaligned_confidence = run_trial(misaligned=True)
    check(
        "belief_correct_action_timing_recovers_true_mode_confidently",
        correct_confidence > 0.9,
        f"correct_confidence={correct_confidence:.4f}",
    )
    check(
        "belief_action_misalignment_degrades_recovery",
        misaligned_confidence < correct_confidence - 0.2,
        f"correct={correct_confidence:.4f} misaligned={misaligned_confidence:.4f} -- "
        "a one-frame action-timing shift must measurably degrade recovery, proving the "
        "tracker actually depends on correct u_r[t-1] timing",
    )


# --------------------------------------------------------------------- #
# group="sampling" (Step 3: trajectory_sampler.py) -- guide.md 8.1 items 8-13,
# and the mean-collapse-trap test from guide.md 6.6.
# --------------------------------------------------------------------- #


def _stationary_robot_sequences(horizon, robot_velocity=(0.0, 0.0), robot_position=(0.0, 0.0)):
    """Order F3 (2026-08-03): the new rollout signature requires an explicit
    robot position/velocity sequence at every horizon step (to recompute
    context per-step -- no more frozen-future-relative-feature
    approximation). Most trajectory_sampler tests don't care about the
    SPECIFIC robot trajectory (context is 0-weighted in
    ``_make_synthetic_arhmm_artifact``'s fixture, C_k=0), so this builds a
    constant one for convenience.

    Order R1 (2026-08-03): the rollout now takes THREE separate robot
    arrays (state position / state velocity / action), never one standing
    in for two roles. A constant-velocity robot never accelerates, so its
    action at every step is trivially equal to its (constant) velocity --
    this helper returns all three, with action_sequence == velocity_sequence
    by construction, not by coincidence."""
    position_sequence = np.tile(np.asarray(robot_position, dtype=np.float64), (horizon, 1))
    velocity_sequence = np.tile(np.asarray(robot_velocity, dtype=np.float64), (horizon, 1))
    action_sequence = velocity_sequence.copy()
    return position_sequence, velocity_sequence, action_sequence


@register("sampling_mode_frequency_matches_posterior", "sampling")
def _test_sampling_mode_frequency_matches_posterior():
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior_with_modes

    artifact = _make_synthetic_arhmm_artifact(K=2)
    posterior = np.array([0.3, 0.7])
    state0 = np.array([0.0, 0.0, 0.8, 0.0])
    horizon = 5
    robot_pos_seq, robot_vel_seq, robot_act_seq = _stationary_robot_sequences(horizon)
    rng = np.random.default_rng(0)
    _trajs, z0_choices, mode_paths = sample_full_posterior_with_modes(
        posterior, state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon=horizon, num_samples=4000, rng=rng
    )
    empirical = np.bincount(z0_choices, minlength=2) / len(z0_choices)
    check(
        "sampling_z0_frequency_within_tolerance",
        bool(np.max(np.abs(empirical - posterior)) < 0.03),
        f"empirical={empirical} target={posterior}",
    )
    check(
        "sampling_first_emitted_mode_equals_z0_choice",
        bool(np.array_equal(mode_paths[:, 0], z0_choices)),
        "the first emitted transition must use the supplied b_next draw directly",
    )

    # Pi transition frequency: among steps where mode was 0, how often does
    # it stay 0 vs move to 1 -- should match Pi[0] within tolerance.
    from_mode0 = mode_paths[:, :-1] == 0
    to_mode0_given_from0 = mode_paths[:, 1:][from_mode0] == 0
    if from_mode0.sum() > 200:
        empirical_stay = float(to_mode0_given_from0.mean())
        check(
            "sampling_pi_transition_frequency_within_tolerance",
            abs(empirical_stay - artifact.Pi[0, 0]) < 0.03,
            f"empirical={empirical_stay:.3f} target={artifact.Pi[0, 0]:.3f}",
        )


@register("sampling_k1_zero_noise_deterministic", "sampling")
def _test_sampling_k1_zero_noise_deterministic():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact, CONTEXT_FEATURE_NAMES
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior

    d_c = len(CONTEXT_FEATURE_NAMES)
    artifact = ARHMMArtifact(
        K=1, A=[np.eye(2)], B=[np.zeros((2, 2))], C=[np.zeros((2, d_c))], d=[np.zeros(2)],
        Q=[np.zeros((2, 2))], Pi=np.ones((1, 1)), initial_distribution=np.ones(1),
        dt=0.25, model_card={},
    )
    state0 = np.array([0.0, 0.0, 0.5, 0.0])
    horizon = 5
    robot_pos_seq, robot_vel_seq, robot_act_seq = _stationary_robot_sequences(horizon)
    rng = np.random.default_rng(0)
    trajs = sample_full_posterior(np.ones(1), state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon=horizon, num_samples=10, rng=rng)
    check(
        "sampling_k1_zero_noise_all_samples_identical",
        bool(np.allclose(trajs, trajs[0:1])),
    )
    expected_final_x = state0[0] + horizon * artifact.dt * state0[2]
    check(
        "sampling_k1_zero_noise_matches_constant_velocity",
        bool(np.isclose(trajs[0, -1, 0], expected_final_x)),
        f"got {trajs[0, -1, 0]:.4f} expected {expected_final_x:.4f}",
    )


@register("sampling_cv_matches_hand_computed_mean", "sampling")
def _test_sampling_cv_matches_hand_computed():
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_cv

    state0 = np.array([0.0, 0.0, 0.6, -0.2])
    horizon, num_samples, dt = 8, 500, 0.25
    rng = np.random.default_rng(0)
    trajs = sample_cv(state0, horizon, num_samples, dt, brne_root=_BRNE_ROOT, rng=rng)
    tlist = np.arange(horizon) * dt
    expected_x = state0[0] + tlist * state0[2]
    expected_y = state0[1] + tlist * state0[3]
    mean_x = trajs[:, :, 0].mean(axis=0)
    mean_y = trajs[:, :, 1].mean(axis=0)
    check("sampling_cv_mean_x_matches", bool(np.max(np.abs(mean_x - expected_x)) < 0.1), f"max_diff={np.max(np.abs(mean_x - expected_x)):.4f}")
    check("sampling_cv_mean_y_matches", bool(np.max(np.abs(mean_y - expected_y)) < 0.1), f"max_diff={np.max(np.abs(mean_y - expected_y)):.4f}")


@register("sampling_shuffle_deterministic_per_seed", "sampling")
def _test_sampling_shuffle_deterministic_per_seed():
    from crowd_nav.bayesian_brne.trajectory_sampler import shuffle_track_beliefs

    beliefs = {1: np.array([0.9, 0.1]), 2: np.array([0.1, 0.9]), 3: np.array([0.5, 0.5])}
    perm_a = shuffle_track_beliefs([1, 2, 3], beliefs, np.random.default_rng(42))
    perm_b = shuffle_track_beliefs([1, 2, 3], beliefs, np.random.default_rng(42))
    check(
        "sampling_shuffle_same_seed_same_result",
        all(np.allclose(perm_a[k], perm_b[k]) for k in (1, 2, 3)),
    )


@register("sampling_moment_matched_mean_has_same_first_two_moments", "sampling")
def _test_sampling_moment_matched_mean_matches_moments():
    """B1: sample_posterior_mean's single-step distribution must match the
    TRUE mixture's mean AND covariance (within + between mode variance),
    not just the within-mode term. On this fixture the analytically correct
    total y-variance is `within (Q_y) + between (mode-mean spread)`.
    Order F3 rewrite: mode means now come from A@v_prev + B@u_r + C@context
    + d (this fixture's C=0, so context is irrelevant here; B differs in
    SIGN per mode, so a nonzero u_r is what creates the between-mode
    spread, playing the role phi[0]=speed played in the old F@phi model)."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import CONTEXT_FEATURE_NAMES
    from crowd_nav.bayesian_brne.trajectory_sampler import _moment_matched_mixture_gaussian

    artifact = _make_synthetic_arhmm_artifact(K=2)
    prior = np.array([0.5, 0.5])
    # B_k is ISOTROPIC (0.6*I / -0.6*I), so it only creates divergence along
    # whichever axis u_r itself has signal in -- this test checks the Y
    # component (index 1), so u_r (and v_prev, for a nonzero A@v_prev
    # contribution too) must carry their signal on the Y axis, not X.
    v_prev = np.array([0.0, 0.8])
    u_r = np.array([0.0, 0.5])
    context = np.zeros(len(CONTEXT_FEATURE_NAMES))

    mean, cov = _moment_matched_mixture_gaussian(prior, v_prev, u_r, context, artifact)
    mode_means = np.array([artifact.A[k] @ v_prev + artifact.B[k] @ u_r + artifact.d[k] for k in range(2)])  # [2, 2]
    mode_means_y = mode_means[:, 1]
    within_var_y = float(prior[0] * artifact.Q[0][1, 1] + prior[1] * artifact.Q[1][1, 1])
    mean_y = float(mean[1])
    between_var_y = float(prior[0] * (mode_means_y[0] - mean_y) ** 2 + prior[1] * (mode_means_y[1] - mean_y) ** 2)
    expected_total_var_y = within_var_y + between_var_y
    expected_mean = prior[0] * mode_means[0] + prior[1] * mode_means[1]

    check(
        "moment_matched_mean_matches_hand_computed_weighted_average",
        bool(np.allclose(mean, expected_mean, atol=1e-9)),
        f"got {mean}, expected {expected_mean}",
    )
    check(
        "moment_matched_covariance_includes_between_mode_term",
        abs(float(cov[1, 1]) - expected_total_var_y) < 1e-9,
        f"got cov_yy={cov[1, 1]:.6f}, expected {expected_total_var_y:.6f} "
        f"(within={within_var_y:.6f} + between={between_var_y:.6f})",
    )
    check(
        "moment_matched_between_term_is_dominant_and_nonzero",
        between_var_y > 2 * within_var_y,
        f"between={between_var_y:.6f} within={within_var_y:.6f} "
        "(an earlier version omitted this term entirely; the exact ratio depends on the "
        "fixture's A/B/Q magnitudes, but it must not be negligible relative to within-mode variance)",
    )


@register("sampling_human_speed_and_acceleration_bounded", "sampling")
def _test_sampling_human_speed_and_acceleration_bounded():
    """B3: no rollout, under any sampling mode, may exceed the configured
    max_speed or max_acceleration -- an earlier version let velocity
    accumulate unbounded across a 10-step rollout, reaching an unphysical
    7.55m lateral spread in 2.5s."""
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior, sample_posterior_mean

    artifact = _make_synthetic_arhmm_artifact(K=2)
    posterior = np.array([0.5, 0.5])
    state0 = np.array([0.0, 0.0, 0.8, 0.0])
    horizon, num_samples = 20, 100
    max_speed, max_acceleration = 2.0, 2.0
    robot_pos_seq, robot_vel_seq, robot_act_seq = _stationary_robot_sequences(horizon, robot_velocity=(0.5, 0.0))

    for name, fn in (("full_posterior", sample_full_posterior), ("posterior_mean", sample_posterior_mean)):
        trajs = fn(
            posterior, state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon, num_samples, np.random.default_rng(0),
            max_speed, max_acceleration,
        )
        velocities = np.diff(trajs, axis=1) / artifact.dt
        speeds = np.linalg.norm(velocities, axis=-1)
        check(
            f"sampling_{name}_speed_bounded",
            bool(np.all(speeds <= max_speed + 1e-6)),
            f"max observed speed={speeds.max():.4f}, cap={max_speed}",
        )
        accelerations = np.diff(velocities, axis=1) / artifact.dt
        accel_mag = np.linalg.norm(accelerations, axis=-1)
        check(
            f"sampling_{name}_acceleration_bounded",
            bool(np.all(accel_mag <= max_acceleration + 1e-3)),
            f"max observed accel={accel_mag.max():.4f}, cap={max_acceleration}",
        )


@register("sampling_cv_local_rng_reproducible", "sampling")
def _test_sampling_cv_local_rng_reproducible():
    """B4: sample_cv must use ITS OWN rng argument, not upstream's module-
    global one -- same local-seed calls must be byte-identical, different
    seeds must differ, and calling it must not perturb another call's
    independent rng stream."""
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_cv

    state0 = np.array([0.0, 0.0, 0.6, -0.2])
    horizon, num_samples, dt = 8, 50, 0.25

    trajs_a = sample_cv(state0, horizon, num_samples, dt, brne_root=_BRNE_ROOT, rng=np.random.default_rng(123))
    trajs_b = sample_cv(state0, horizon, num_samples, dt, brne_root=_BRNE_ROOT, rng=np.random.default_rng(123))
    check("sampling_cv_same_seed_byte_identical", bool(np.array_equal(trajs_a, trajs_b)))

    trajs_c = sample_cv(state0, horizon, num_samples, dt, brne_root=_BRNE_ROOT, rng=np.random.default_rng(456))
    check("sampling_cv_different_seed_differs", not bool(np.array_equal(trajs_a, trajs_c)))

    # A call using an UNRELATED rng instance in between must not change the
    # result of a call using the original seed again (i.e. no shared global
    # state is being silently consumed).
    _ = sample_cv(state0, horizon, num_samples, dt, brne_root=_BRNE_ROOT, rng=np.random.default_rng(999))
    trajs_a_again = sample_cv(state0, horizon, num_samples, dt, brne_root=_BRNE_ROOT, rng=np.random.default_rng(123))
    check("sampling_cv_independent_of_other_calls", bool(np.array_equal(trajs_a, trajs_a_again)))


@register("sampling_full_is_bimodal_while_mean_is_unimodal", "sampling")
def _test_sampling_full_is_bimodal_while_mean_is_unimodal():
    """Revised mean-trap fixture test (B3: realistic physical range, not
    7.55m). Checks distributional SHAPE (bimodal vs. unimodal), matched
    first/second moments between full and mean, and defers the
    "BRNE decision changes" claim to the dedicated weighted-first-control
    test below (B2) rather than conflating the two. Order F3: a nonzero
    constant robot_velocity feeds u_r every step, which is what makes the
    two modes' opposite-signed B_k terms actually diverge (this fixture's
    C_k=0, so it is u_r -- not context -- that drives the bifurcation, the
    same role phi[0]=speed played in the old F@phi model)."""
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior, sample_posterior_mean

    artifact = _make_synthetic_arhmm_artifact(K=2)
    posterior = np.array([0.5, 0.5])
    state0 = np.array([0.0, 0.0, 0.8, 0.0])
    horizon, num_samples = 10, 400
    # B_k is ISOTROPIC, so it only creates divergence along whichever axis
    # u_r carries signal in -- this test checks final_y (index 1) for
    # bimodality, so robot_velocity's signal must be on the Y axis.
    robot_pos_seq, robot_vel_seq, robot_act_seq = _stationary_robot_sequences(horizon, robot_velocity=(0.0, 0.5))

    full = sample_full_posterior(posterior, state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon, num_samples, np.random.default_rng(0))
    mean = sample_posterior_mean(posterior, state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon, num_samples, np.random.default_rng(0))

    final_y_full = full[:, -1, 1]
    final_y_mean = mean[:, -1, 1]

    check(
        "sampling_full_final_y_in_realistic_range",
        bool(0.2 < np.abs(final_y_full).max() < 3.0),
        f"max|y|={np.abs(final_y_full).max():.3f}m over {horizon * artifact.dt:.1f}s (expect roughly 0.5-2.0m, not 7.55m)",
    )
    frac_positive = float((final_y_full > 0.15).mean())
    frac_negative = float((final_y_full < -0.15).mean())
    check(
        "sampling_full_posterior_forms_two_populated_clusters",
        frac_positive > 0.3 and frac_negative > 0.3,
        f"frac_positive={frac_positive:.2f} frac_negative={frac_negative:.2f}",
    )
    # Loose sanity check only: exact moment-matching is guaranteed and
    # separately verified at the SINGLE-step level
    # (test_sampling_moment_matched_mean_matches_moments). Over a full
    # 10-step rollout the two accumulate differently BY DESIGN -- Pi's 0.95
    # self-persistence means most full_posterior samples commit to one mode
    # for nearly the whole horizon (compounding drift in one direction),
    # while posterior_mean re-draws a fresh single-Gaussian blend every
    # step (no persistence to compound) -- so a same-order-of-magnitude
    # bound is the right check here, not near-equality.
    check(
        "sampling_full_and_mean_same_order_of_magnitude_spread",
        abs(final_y_full.std() - final_y_mean.std()) < 3.0 * max(final_y_full.std(), final_y_mean.std()),
        f"full_std={final_y_full.std():.4f} mean_std={final_y_mean.std():.4f}",
    )

    # Bimodality check: full's histogram should have a trough near 0 (the
    # midpoint) with mass concentrated away from it; mean's histogram should
    # peak AT the midpoint.
    hist_full, edges = np.histogram(final_y_full, bins=20)
    hist_mean, _ = np.histogram(final_y_mean, bins=edges)
    mid_bin = len(hist_full) // 2
    check(
        "sampling_full_posterior_trough_near_center",
        hist_full[mid_bin] <= 0.6 * hist_full.max(),
        f"center bin count={hist_full[mid_bin]}, max bin count={hist_full.max()}",
    )
    check(
        "sampling_posterior_mean_peaks_near_center",
        hist_mean[mid_bin] >= 0.6 * hist_mean.max(),
        f"center bin count={hist_mean[mid_bin]}, max bin count={hist_mean.max()}",
    )


@register("mean_trap_changes_weighted_first_control_not_only_sorted_weights", "sampling")
def _test_mean_trap_changes_weighted_first_control():
    """The corrected core test from guide.md 6.6 (fixes B2): comparing
    SORTED weight arrays discards the correspondence between a weight and
    the robot action it belongs to -- two different weight profiles could
    still average out to the SAME executed control. This test instead
    computes the actual ``weighted_first_control`` (guide.md 2.4) under
    full_posterior vs. moment-matched posterior_mean pedestrian
    representations and requires the EXECUTED controls to differ.

    Hardened per the 2026-08-03 second-pass audit's B2 PARTIAL finding: an
    earlier version of this fixture built ``robot_traj`` (constant lateral
    offset for the whole horizon) and ``robot_controls`` (offset written
    only at the first timestep) as two INDEPENDENTLY parameterized arrays --
    they were not one integrated from the other, so passing only proved
    BRNE's weight-driven control AGGREGATE differed, not that two genuinely
    different EXECUTABLE control forks from the SAME start state produce
    different weighted outcomes. Here ``robot_traj`` is built strictly as
    the forward kinematic integral of ``robot_controls`` (holonomic:
    p[t] = p[t-1] + control[t]*dt from a shared start position), and the
    test additionally reports clearance and passing side alongside the
    control difference, as guide.md's B2 fix requires."""
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver, weighted_first_control
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior, sample_posterior_mean

    artifact = _make_synthetic_arhmm_artifact(K=2)
    posterior = np.array([0.5, 0.5])
    state0 = np.array([0.0, 0.0, 0.8, 0.0])
    horizon, num_samples = 10, 200
    dt = artifact.dt
    robot_pos_seq, robot_vel_seq, robot_act_seq = _stationary_robot_sequences(horizon, robot_velocity=(0.5, 0.0))

    full = sample_full_posterior(posterior, state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon, num_samples, np.random.default_rng(0))
    mean = sample_posterior_mean(posterior, state0, artifact, robot_pos_seq, robot_vel_seq, robot_act_seq, horizon, num_samples, np.random.default_rng(0))

    # Robot candidates: a genuinely different, per-sample HOLONOMIC VELOCITY
    # CONTROL held constant across the whole horizon (not just step 0), and
    # robot_traj is the exact forward kinematic integral of robot_controls
    # from a shared start position -- so the two arrays are, by
    # construction, one executable control fork each, not independently
    # invented bookkeeping.
    forward_speed = 0.8
    lateral_vel = np.linspace(-0.3, 0.3, num_samples)
    robot_controls = np.zeros((num_samples, horizon, 2))
    robot_controls[:, :, 0] = forward_speed
    robot_controls[:, :, 1] = lateral_vel[:, None]
    start_pos = np.zeros(2)
    robot_traj = np.cumsum(robot_controls * dt, axis=1) + start_pos[None, None, :]
    radii = np.full(2, 0.3)

    trajectories_full = np.stack([robot_traj, full], axis=0)
    trajectories_mean = np.stack([robot_traj, mean], axis=0)

    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)
    result_full = solver.solve(trajectories_full, radii, edge_mask=None, equilibrium_iterations=10)
    result_mean = solver.solve(trajectories_mean, radii, edge_mask=None, equilibrium_iterations=10)

    u_full = weighted_first_control(result_full.weights[0], robot_controls)
    u_mean = weighted_first_control(result_mean.weights[0], robot_controls)
    control_diff = float(np.linalg.norm(u_full - u_mean))

    check(
        "mean_trap_weighted_first_control_differs",
        control_diff > 1e-3,
        f"u_full={u_full}, u_mean={u_mean}, ||diff||={control_diff:.4e} "
        "(comparing the ACTUAL executed control, not just sorted weight arrays, "
        "on a robot_traj that is the literal kinematic integral of robot_controls)",
    )

    def _clearance_and_side(weights, human_samples):
        w = weights / weights.sum()
        weighted_traj = np.tensordot(w, robot_traj, axes=(0, 0))  # [H, 2]
        human_repr = human_samples.mean(axis=0)  # [H, 2] representative human path
        rel = weighted_traj - human_repr
        dist = np.linalg.norm(rel, axis=-1)
        idx = int(np.argmin(dist))
        side = "left" if rel[idx, 1] > 0 else "right"
        return float(dist[idx]), side, weighted_traj

    clearance_full, side_full, _ = _clearance_and_side(result_full.weights[0], full)
    clearance_mean, side_mean, _ = _clearance_and_side(result_mean.weights[0], mean)

    check(
        "mean_trap_clearance_and_passing_side_reported",
        clearance_full > 0.0 and clearance_mean > 0.0,
        f"full: clearance={clearance_full:.3f}m side={side_full} | "
        f"mean: clearance={clearance_mean:.3f}m side={side_mean} "
        "(reported per guide.md B2 fix; not required to differ, but must be well-defined "
        "for both the full-posterior and moment-matched-mean forks)",
    )


# --------------------------------------------------------------------- #
# group="robot_sampler" (Order F3, 2026-08-03) -- guide.md 4.3's acceptance
# criteria: velocity/acceleration never exceed physical limits, and (the
# load-bearing criterion this whole framework rewrite is FOR) changing the
# candidate robot trajectory must change an action-sensitive-mode
# pedestrian's rollout while a self-only (B=0) rollout stays unchanged.
# --------------------------------------------------------------------- #


@register("robot_sampler_shapes_and_reproducibility", "robot_sampler")
def _test_robot_sampler_shapes_and_reproducibility():
    from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates

    horizon, num_candidates = 8, 20
    actions, state_positions, state_velocities, future_positions = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=5.0, goal_gy=0.0, horizon=horizon, num_candidates=num_candidates, dt=0.25,
        rng=np.random.default_rng(1), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    check("robot_sampler_actions_shape", actions.shape == (num_candidates, horizon, 2), str(actions.shape))
    check("robot_sampler_state_positions_shape", state_positions.shape == (num_candidates, horizon, 2), str(state_positions.shape))
    check("robot_sampler_state_velocities_shape", state_velocities.shape == (num_candidates, horizon, 2), str(state_velocities.shape))
    check("robot_sampler_future_positions_shape", future_positions.shape == (num_candidates, horizon, 2), str(future_positions.shape))
    check(
        "robot_sampler_finite",
        bool(np.all(np.isfinite(actions)) and np.all(np.isfinite(state_positions))
             and np.all(np.isfinite(state_velocities)) and np.all(np.isfinite(future_positions))),
    )
    # Order R1's defining invariant: every candidate's step-0 state must be
    # exactly the robot's actual current position/velocity (the state
    # BEFORE any action is applied), never something already advanced.
    check(
        "robot_sampler_state_position_step0_matches_current_position",
        bool(np.allclose(state_positions[:, 0, :], np.array([0.0, 0.0]))),
        f"state_positions[:,0]={state_positions[:, 0, :]}",
    )
    check(
        "robot_sampler_state_velocity_step0_matches_current_velocity",
        bool(np.allclose(state_velocities[:, 0, :], np.array([0.0, 0.0]))),
        f"state_velocities[:,0]={state_velocities[:, 0, :]}",
    )
    # future_positions[h] must be the forward kinematic integral of
    # actions[h] applied to state_positions[h] -- and state_positions[h+1]
    # must equal future_positions[h] (the next state's "before" is this
    # step's "after").
    check(
        "robot_sampler_future_position_is_kinematic_integral_of_action",
        bool(np.allclose(future_positions, state_positions + 0.25 * actions, atol=1e-9)),
    )
    check(
        "robot_sampler_state_chains_to_previous_future",
        bool(np.allclose(state_positions[:, 1:, :], future_positions[:, :-1, :], atol=1e-9)),
    )

    actions_again, state_positions_again, state_velocities_again, future_positions_again = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=5.0, goal_gy=0.0, horizon=horizon, num_candidates=num_candidates, dt=0.25,
        rng=np.random.default_rng(1), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    check(
        "robot_sampler_reproducible_given_same_seed",
        bool(np.array_equal(actions, actions_again) and np.array_equal(future_positions, future_positions_again)
             and np.array_equal(state_positions, state_positions_again)
             and np.array_equal(state_velocities, state_velocities_again)),
    )

    # R3 invariant: candidate 0 is a strict noise-free nominal and must not
    # change when the RNG stream used for candidates 1..M-1 changes.
    actions_other_rng, *_ = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=5.0, goal_gy=0.0, horizon=horizon, num_candidates=num_candidates, dt=0.25,
        rng=np.random.default_rng(999), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    nominal_single, *_ = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=5.0, goal_gy=0.0, horizon=horizon, num_candidates=1, dt=0.25,
        rng=np.random.default_rng(12345), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    check(
        "robot_sampler_candidate_zero_is_noise_free_nominal",
        bool(np.array_equal(actions[0], actions_other_rng[0])
             and np.array_equal(actions[0], nominal_single[0])),
        "candidate 0 must be independent of the RNG and equal the one-candidate nominal",
    )

    # Goal-directed: mean final position across candidates should trend
    # toward the goal (positive x, roughly zero y, for a goal straight ahead).
    mean_final = future_positions[:, -1, :].mean(axis=0)
    check(
        "robot_sampler_trends_toward_goal",
        bool(mean_final[0] > 0.5 and abs(mean_final[1]) < 1.0),
        f"mean_final_position={mean_final}",
    )


@register("robot_sampler_respects_physical_limits", "robot_sampler")
def _test_robot_sampler_respects_physical_limits():
    from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates

    horizon, num_candidates, dt = 20, 50, 0.25
    max_speed, max_acceleration = 2.0, 2.0
    actions, _state_positions, _state_velocities, _future_positions = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=1.5, robot_vy=1.5,  # start near the speed cap
        goal_gx=-5.0, goal_gy=5.0, horizon=horizon, num_candidates=num_candidates, dt=dt,
        rng=np.random.default_rng(2), v_pref=1.0, max_speed=max_speed, max_acceleration=max_acceleration,
        brne_root=_BRNE_ROOT,
    )
    speeds = np.linalg.norm(actions, axis=-1)
    check(
        "robot_sampler_speed_bounded",
        bool(np.all(speeds <= max_speed + 1e-6)),
        f"max observed speed={speeds.max():.4f}, cap={max_speed}",
    )
    accel = np.linalg.norm(np.diff(actions, axis=1), axis=-1) / dt
    check(
        "robot_sampler_acceleration_bounded",
        bool(np.all(accel <= max_acceleration + 1e-3)),
        f"max observed accel={accel.max():.4f}, cap={max_acceleration}",
    )


@register("robot_action_changes_action_sensitive_trajectory_not_self_only", "robot_sampler")
def _test_robot_action_changes_action_sensitive_trajectory_not_self_only():
    """Order F3's load-bearing acceptance criterion (guide.md 4.3): changing
    the CANDIDATE robot trajectory fed into ``trajectory_sampler`` must
    change an action-sensitive-mode pedestrian's rollout (this fixture's
    ``B_k != 0``), while a self-only control (B forced to 0, mirroring how
    U3's necessity gate built its self_only variant by zeroing the action
    rather than editing the model) leaves the rollout unchanged regardless
    of which candidate robot trajectory is fed in. This is the online-
    rollout analogue of what U3's necessity gate proved offline."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact
    from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_full_posterior

    action_conditioned = _make_synthetic_arhmm_artifact(K=2)
    self_only = ARHMMArtifact(
        K=2, A=action_conditioned.A, B=[np.zeros((2, 2)), np.zeros((2, 2))],
        C=action_conditioned.C, d=action_conditioned.d, Q=action_conditioned.Q,
        Pi=action_conditioned.Pi, initial_distribution=action_conditioned.initial_distribution,
        dt=action_conditioned.dt, model_card={},
    )
    posterior = np.array([0.5, 0.5])
    state0 = np.array([0.0, 0.0, 0.8, 0.0])
    horizon, num_samples = 10, 300

    actions_a, state_pos_a, state_vel_a, _future_a = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=5.0, goal_gy=0.0, horizon=horizon, num_candidates=1, dt=0.25,
        rng=np.random.default_rng(10), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    actions_b, state_pos_b, state_vel_b, _future_b = sample_robot_candidates(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=-5.0, goal_gy=5.0, horizon=horizon, num_candidates=1, dt=0.25,
        rng=np.random.default_rng(11), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    robot_pos_a, robot_vel_a, robot_act_a = state_pos_a[0], state_vel_a[0], actions_a[0]
    robot_pos_b, robot_vel_b, robot_act_b = state_pos_b[0], state_vel_b[0], actions_b[0]
    check(
        "robot_action_test_setup_two_genuinely_different_candidates",
        not bool(np.allclose(robot_act_a, robot_act_b)),
        "the two candidate robot trajectories must actually differ for this test to mean anything",
    )

    traj_action_conditioned_a = sample_full_posterior(posterior, state0, action_conditioned, robot_pos_a, robot_vel_a, robot_act_a, horizon, num_samples, np.random.default_rng(0))
    traj_action_conditioned_b = sample_full_posterior(posterior, state0, action_conditioned, robot_pos_b, robot_vel_b, robot_act_b, horizon, num_samples, np.random.default_rng(0))
    traj_self_only_a = sample_full_posterior(posterior, state0, self_only, robot_pos_a, robot_vel_a, robot_act_a, horizon, num_samples, np.random.default_rng(0))
    traj_self_only_b = sample_full_posterior(posterior, state0, self_only, robot_pos_b, robot_vel_b, robot_act_b, horizon, num_samples, np.random.default_rng(0))

    check(
        "robot_action_changes_action_conditioned_trajectory",
        not bool(np.allclose(traj_action_conditioned_a, traj_action_conditioned_b)),
        "action-conditioned (B != 0) rollout must change when the candidate robot trajectory changes",
    )
    check(
        "robot_action_leaves_self_only_trajectory_unchanged",
        bool(np.allclose(traj_self_only_a, traj_self_only_b)),
        "self-only (B forced to 0) rollout must NOT depend on which candidate robot trajectory is fed in "
        "-- same rng, same everything else, only the (unused) robot inputs differ",
    )


# --------------------------------------------------------------------- #
# group="equilibrium_loop" (Order F4, 2026-08-03) -- guide.md 4.5/F4's
# load-bearing acceptance criterion: in at least one synthetic crossing
# scenario, changing the robot action changes the human prior/trajectory,
# which changes BRNE's robot equilibrium weights -- proving this module
# actually closes the loop rather than just wiring pieces together inertly.
# --------------------------------------------------------------------- #


def _crossing_scenario_fixture(B):
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact, CONTEXT_FEATURE_NAMES

    d_c = len(CONTEXT_FEATURE_NAMES)
    K = 2
    A = [0.3 * np.eye(2) for _ in range(K)]
    C = [np.zeros((2, d_c)) for _ in range(K)]
    d = [np.zeros(2) for _ in range(K)]
    Q = [0.01 * np.eye(2) for _ in range(K)]
    Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
    initial_distribution = np.array([0.5, 0.5])
    return ARHMMArtifact(K=K, A=A, B=B, C=C, d=d, Q=Q, Pi=Pi,
                          initial_distribution=initial_distribution, dt=0.25, model_card={})


@register("equilibrium_loop_converges_and_returns_valid_weights", "equilibrium_loop")
def _test_equilibrium_loop_converges_and_returns_valid_weights():
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver
    from crowd_nav.bayesian_brne.equilibrium_loop import run_outer_equilibrium_loop
    from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates

    artifact = _crossing_scenario_fixture(B=[0.6 * np.eye(2), -0.6 * np.eye(2)])
    horizon = 8
    robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions = sample_robot_candidates(
        robot_px=-2.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=3.0, goal_gy=0.0, horizon=horizon, num_candidates=30, dt=0.25,
        rng=np.random.default_rng(1), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    posteriors = {1: np.array([0.5, 0.5])}
    state0 = {1: np.array([0.0, -2.0, 0.0, 0.8])}
    radii = {1: 0.3}
    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)

    result = run_outer_equilibrium_loop(
        posteriors, state0, artifact,
        robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions, radii,
        robot_radius=0.3, horizon=horizon, solver=solver,
        max_outer_iterations=10, weight_change_threshold=1e-3, seed=2407,
    )
    check("equilibrium_loop_iterations_within_budget", 1 <= result.outer_iterations <= 10, str(result.outer_iterations))
    check(
        "equilibrium_loop_converged_flag_matches_residual",
        result.converged == (result.max_weight_residual < 1e-3),
        f"converged={result.converged} residual={result.max_weight_residual}",
    )
    check("equilibrium_loop_robot_weights_nonnegative", bool(np.all(result.robot_weights >= 0)))
    check("equilibrium_loop_robot_weights_normalized", abs(float(result.robot_weights.sum()) - 1.0) < 1e-9, str(result.robot_weights.sum()))
    check("equilibrium_loop_weighted_robot_position_shape", result.weighted_robot_state_position_sequence.shape == (horizon, 2))
    check("equilibrium_loop_human_weights_present_for_every_track", set(result.human_weights_by_track.keys()) == {1})


@register("equilibrium_loop_action_conditioned_vs_self_only_robot_weights_differ", "equilibrium_loop")
def _test_equilibrium_loop_action_conditioned_vs_self_only_differ():
    """Order F4's load-bearing acceptance criterion (guide.md 4.5): in this
    synthetic pedestrian-crossing scenario, using an action-CONDITIONED
    model (B != 0, so the pedestrian's simulated response genuinely depends
    on the weighted robot trajectory each outer iteration) versus a
    self-only model (B forced to 0, so the pedestrian's response never
    depends on the robot at all) must produce DIFFERENT final robot
    equilibrium weights -- proving the outer loop actually propagates
    robot-action-dependence through to BRNE's decision, not just wiring the
    pieces together without the causal chain actually mattering."""
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver
    from crowd_nav.bayesian_brne.equilibrium_loop import run_outer_equilibrium_loop
    from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates

    horizon = 8
    robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions = sample_robot_candidates(
        robot_px=-2.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=3.0, goal_gy=0.0, horizon=horizon, num_candidates=30, dt=0.25,
        rng=np.random.default_rng(1), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    posteriors = {1: np.array([0.5, 0.5])}
    state0 = {1: np.array([0.0, -2.0, 0.0, 0.8])}
    radii = {1: 0.3}
    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)

    action_conditioned = _crossing_scenario_fixture(B=[0.6 * np.eye(2), -0.6 * np.eye(2)])
    self_only = _crossing_scenario_fixture(B=[np.zeros((2, 2)), np.zeros((2, 2))])

    result_action = run_outer_equilibrium_loop(
        posteriors, state0, action_conditioned,
        robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions, radii,
        robot_radius=0.3, horizon=horizon, solver=solver, seed=2407,
    )
    result_self = run_outer_equilibrium_loop(
        posteriors, state0, self_only,
        robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions, radii,
        robot_radius=0.3, horizon=horizon, solver=solver, seed=2407,
    )
    diff = float(np.max(np.abs(result_action.robot_weights - result_self.robot_weights)))
    check(
        "equilibrium_loop_action_conditioned_vs_self_only_robot_weights_differ",
        diff > 1e-3,
        f"max robot_weights diff (action-conditioned vs self-only) = {diff:.6f} -- the robot's own "
        "equilibrium decision must depend on whether the pedestrian model actually reacts to it",
    )


@register("equilibrium_loop_reproducible_given_same_seed", "equilibrium_loop")
def _test_equilibrium_loop_reproducible_given_same_seed():
    """Common random numbers (guide.md 4.5 item 5): the SAME inputs and
    SAME seed must produce a BYTE-IDENTICAL outer-loop result -- this is
    what makes it valid to attribute any weight DIFFERENCE across two runs
    to a genuine input change, never to resampling noise."""
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver
    from crowd_nav.bayesian_brne.equilibrium_loop import run_outer_equilibrium_loop
    from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates

    artifact = _crossing_scenario_fixture(B=[0.6 * np.eye(2), -0.6 * np.eye(2)])
    horizon = 8
    robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions = sample_robot_candidates(
        robot_px=-2.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0,
        goal_gx=3.0, goal_gy=0.0, horizon=horizon, num_candidates=30, dt=0.25,
        rng=np.random.default_rng(1), v_pref=1.0, brne_root=_BRNE_ROOT,
    )
    posteriors = {1: np.array([0.5, 0.5])}
    state0 = {1: np.array([0.0, -2.0, 0.0, 0.8])}
    radii = {1: 0.3}
    solver = BRNESolver(solver_mode="stable", brne_root=_BRNE_ROOT)

    result_a = run_outer_equilibrium_loop(
        posteriors, state0, artifact,
        robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions, radii,
        robot_radius=0.3, horizon=horizon, solver=solver, seed=2407,
    )
    result_b = run_outer_equilibrium_loop(
        posteriors, state0, artifact,
        robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions, radii,
        robot_radius=0.3, horizon=horizon, solver=solver, seed=2407,
    )
    check(
        "equilibrium_loop_same_seed_byte_identical_weights",
        bool(np.array_equal(result_a.robot_weights, result_b.robot_weights)),
    )
    check(
        "equilibrium_loop_same_seed_same_iteration_count",
        result_a.outer_iterations == result_b.outer_iterations,
        f"{result_a.outer_iterations} vs {result_b.outer_iterations}",
    )


@register("equilibrium_loop_status_classifier", "equilibrium_loop")
def _test_equilibrium_loop_status_classifier():
    from crowd_nav.bayesian_brne.equilibrium_loop import _classify_outer_status

    check(
        "outer_status_converged",
        _classify_outer_status([0.4, 0.08, 0.0005], converged=True,
                                max_outer_iterations=10, oscillation_tolerance=1e-6) == "converged",
    )
    check(
        "outer_status_max_iterations",
        _classify_outer_status([0.4, 0.2, 0.1], converged=False,
                                max_outer_iterations=3, oscillation_tolerance=1e-6) == "max_iterations",
    )
    check(
        "outer_status_oscillating",
        _classify_outer_status([0.2, 0.3, 0.2, 0.3], converged=False,
                                max_outer_iterations=10, oscillation_tolerance=1e-12) == "oscillating",
    )


@register("diagnostic_writer_schema_and_config_hash", "diagnostics")
def _test_diagnostic_writer_schema_and_config_hash():
    import tempfile
    from pathlib import Path
    from crowd_nav.bayesian_brne.diagnostics import JsonlDiagnosticWriter, canonical_sha256

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "diagnostics.jsonl"
        writer = JsonlDiagnosticWriter(
            path, resolved_config={"horizon_steps": 12, "num_samples": 4},
            artifact_sha256="artifact-hash", suite_seed=2407,
        )
        record = writer.write_episode(
            episode_seed=7, scenario="baseline_circle", profile="engineering_smoke",
            step_records=[{"step": 1, "status": "converged"}],
            termination_event="reach_goal", elapsed_ms=12.5,
        )
        loaded = __import__("json").loads(path.read_text().strip())
        check("diagnostic_record_schema_version", loaded["schema_version"] == 1)
        check("diagnostic_record_required_identity", loaded["scenario"] == "baseline_circle" and loaded["episode_seed"] == 7)
        check("diagnostic_config_hash_stable", loaded["config_sha256"] == canonical_sha256({"horizon_steps": 12, "num_samples": 4}))
        check("diagnostic_writer_return_matches_disk", record["termination_event"] == loaded["termination_event"])


@register("r5_paired_evaluator_identity_and_hash_contract", "evaluator")
def _test_r5_paired_evaluator_identity_and_hash_contract():
    from types import SimpleNamespace
    from crowd_nav.tools.evaluate_sm_brne import _identity_reordered_ids, _method_independent_initial_hash

    env = SimpleNamespace(humans=[SimpleNamespace(track_id=8), SimpleNamespace(track_id=3), SimpleNamespace(track_id=12)])
    normal, reversed_ids = _identity_reordered_ids(env)
    check("r5_track_ids_survive_list_reordering", normal == [8, 3, 12] and reversed_ids == [12, 3, 8])
    snapshot = {
        "robot": np.zeros(9), "humans": np.zeros((3, 5)), "track_ids": [8, 3, 12],
    }
    hash_a = _method_independent_initial_hash(snapshot, scenario="baseline_circle", profile="nominal", suite_seed=2407, episode_index=0)
    hash_b = _method_independent_initial_hash(snapshot, scenario="baseline_circle", profile="nominal", suite_seed=2407, episode_index=0)
    snapshot["robot"][0] = 1.0
    hash_c = _method_independent_initial_hash(snapshot, scenario="baseline_circle", profile="nominal", suite_seed=2407, episode_index=0)
    check("r5_initial_hash_reproducible", hash_a == hash_b and len(hash_a) == 64)
    check("r5_initial_hash_changes_with_state", hash_a != hash_c)


@register("r5_heldout_interactive_crowdsim_branches_on_robot_action", "evaluator")
def _test_r5_heldout_interactive_crowdsim_branches_on_robot_action():
    """The held-out evaluator must use real CrowdSim and stage the robot
    action before CrowdSim asks each human policy for its next action."""
    import configparser
    from crowd_nav.bayesian_brne.interaction_protocol import (
        BehaviorType, HumanBehaviorState, compute_human_response,
    )
    from crowd_nav.bayesian_brne.interactive_crowdsim import HeldoutInteractiveCrowdSim
    from crowd_nav.tools.evaluate_sm_brne import ENV_CONFIG_PATH
    from crowd_sim.envs.utils.robot import Robot

    config = configparser.RawConfigParser()
    config.read(str(ENV_CONFIG_PATH))
    config.set("sim", "test_sim", "circle_crossing")
    config.set("sim", "human_num", "5")
    env = HeldoutInteractiveCrowdSim()
    env.configure(config)
    env.phase = "test"
    robot = Robot(config, "robot")
    robot.policy.multiagent_training = True
    robot.env = env
    env.set_robot(robot)
    env.reset(seed=2407, options={"test_case": 0})

    state = HumanBehaviorState(BehaviorType.YIELD)
    common = dict(
        human_pos=np.array([0.0, 1.0]), human_vel=np.zeros(2),
        human_pref_speed=1.0, goal=np.array([0.0, -2.0]),
        robot_pos=np.array([0.0, 0.0]), combined_radius=0.6,
        state=state, config=env.protocol_config, dt=env.time_step,
    )
    velocity_left, *_ = compute_human_response(
        robot_vel=np.array([-1.0, 1.0]), **common,
    )
    velocity_right, *_ = compute_human_response(
        robot_vel=np.array([1.0, 1.0]), **common,
    )
    check(
        "r5_real_crowdsim_action_conditioned_protocol_differs",
        not np.allclose(velocity_left, velocity_right),
        f"left={velocity_left.tolist()} right={velocity_right.tolist()}",
    )
    action = env.humans[0].act([])
    check(
        "r5_real_crowdsim_human_policy_returns_actionxy",
        hasattr(action, "vx") and hasattr(action, "vy"),
    )


# --------------------------------------------------------------------- #
# group="policy" (Order F5, 2026-08-03) -- guide.md 4.6/F5's acceptance
# criterion: a real predict() call returns a finite, valid ActionXY with
# zero neural-network weights loaded, following the fixed 7-step order.
# --------------------------------------------------------------------- #


def _make_policy_fixture():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact, CONTEXT_FEATURE_NAMES
    from crowd_nav.bayesian_brne.policy import BayesianBRNEPolicy, BayesianBRNEPolicyConfig

    d_c = len(CONTEXT_FEATURE_NAMES)
    K = 2
    A = [0.3 * np.eye(2) for _ in range(K)]
    B = [0.6 * np.eye(2), -0.6 * np.eye(2)]
    C = [np.zeros((2, d_c)) for _ in range(K)]
    d = [np.zeros(2) for _ in range(K)]
    Q = [0.01 * np.eye(2) for _ in range(K)]
    Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
    initial_distribution = np.array([0.5, 0.5])
    artifact = ARHMMArtifact(K=K, A=A, B=B, C=C, d=d, Q=Q, Pi=Pi,
                              initial_distribution=initial_distribution, dt=0.25, model_card={})
    policy = BayesianBRNEPolicy()
    config = BayesianBRNEPolicyConfig(horizon_steps=6, num_samples=15, brne_root=_BRNE_ROOT)
    policy.configure(config, artifact)
    return policy


def _run_crossing_episode(policy, episode_seed, n_steps=10):
    from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation

    policy.reset(episode_seed)
    robot = [-2.0, 0.0, 0.0, 0.0]
    human = [0.0, -2.0, 0.0, 0.8]
    actions = []
    for t in range(n_steps):
        obs = PolicyObservation(
            robot_px=robot[0], robot_py=robot[1], robot_vx=robot[2], robot_vy=robot[3],
            robot_radius=0.3, robot_gx=3.0, robot_gy=0.0, robot_v_pref=1.0, timestamp=t * 0.25, time_step=0.25,
            humans=[TrackObservation(track_id=1, px=human[0], py=human[1], vx=human[2], vy=human[3], radius=0.3, timestamp=t * 0.25)],
            episode_seed=episode_seed,
        )
        action = policy.predict(obs)
        actions.append((action.vx, action.vy))
        robot[0] += action.vx * 0.25
        robot[1] += action.vy * 0.25
        robot[2], robot[3] = action.vx, action.vy
        human[0] += human[2] * 0.25
        human[1] += human[3] * 0.25
    return actions


@register("policy_predict_returns_finite_valid_action_every_step", "policy")
def _test_policy_predict_returns_finite_valid_action_every_step():
    policy = _make_policy_fixture()
    actions = _run_crossing_episode(policy, episode_seed=42, n_steps=12)
    check("policy_ran_all_steps", len(actions) == 12)
    all_finite = all(np.isfinite(vx) and np.isfinite(vy) for vx, vy in actions)
    check("policy_all_actions_finite", all_finite, str(actions))
    max_speed = policy.config.max_speed
    all_bounded = all(np.hypot(vx, vy) <= max_speed + 1e-6 for vx, vy in actions)
    check("policy_all_actions_speed_bounded", all_bounded, f"max_speed={max_speed}, actions={actions}")


@register("policy_reproducible_given_same_episode_seed", "policy")
def _test_policy_reproducible_given_same_episode_seed():
    policy_a = _make_policy_fixture()
    policy_b = _make_policy_fixture()
    actions_a = _run_crossing_episode(policy_a, episode_seed=7, n_steps=10)
    actions_b = _run_crossing_episode(policy_b, episode_seed=7, n_steps=10)
    check(
        "policy_same_episode_seed_byte_identical_actions",
        bool(np.allclose(actions_a, actions_b)),
        f"{actions_a} vs {actions_b}",
    )


@register("policy_validation_fails_closed", "policy")
def _test_policy_validation_fails_closed():
    from crowd_nav.bayesian_brne.policy import PolicyValidationError
    from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation

    policy = _make_policy_fixture()

    raised_before_configure = False
    fresh_policy_module_state = type(policy)()
    try:
        fresh_policy_module_state.reset(1)
    except PolicyValidationError:
        raised_before_configure = True
    check("policy_reset_before_configure_fails_closed", raised_before_configure)

    policy.reset(1)
    dup_obs = PolicyObservation(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, robot_radius=0.3,
        robot_gx=1.0, robot_gy=0.0, robot_v_pref=1.0, timestamp=0.0, time_step=0.25,
        humans=[
            TrackObservation(track_id=1, px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, timestamp=0.0),
            TrackObservation(track_id=1, px=1.0, py=1.0, vx=0.0, vy=0.0, radius=0.3, timestamp=0.0),
        ],
        episode_seed=1,
    )
    raised_duplicate = False
    try:
        policy.predict(dup_obs)
    except PolicyValidationError:
        raised_duplicate = True
    check("policy_duplicate_track_id_fails_closed", raised_duplicate)

    nan_obs = PolicyObservation(
        robot_px=float("nan"), robot_py=0.0, robot_vx=0.0, robot_vy=0.0, robot_radius=0.3,
        robot_gx=1.0, robot_gy=0.0, robot_v_pref=1.0, timestamp=0.0, time_step=0.25, humans=[], episode_seed=1,
    )
    raised_nan = False
    try:
        policy.predict(nan_obs)
    except PolicyValidationError:
        raised_nan = True
    check("policy_non_finite_observation_fails_closed", raised_nan)

    timestamp_policy = _make_policy_fixture()
    timestamp_policy.reset(2)
    base_obs = PolicyObservation(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, robot_radius=0.3,
        robot_gx=1.0, robot_gy=0.0, robot_v_pref=1.0, timestamp=0.0, time_step=0.25,
        humans=[], episode_seed=2,
    )
    timestamp_policy.predict(base_obs)
    skipped_obs = PolicyObservation(
        robot_px=0.0, robot_py=0.0, robot_vx=0.0, robot_vy=0.0, robot_radius=0.3,
        robot_gx=1.0, robot_gy=0.0, robot_v_pref=1.0, timestamp=0.5, time_step=0.25,
        humans=[], episode_seed=2,
    )
    raised_skipped_timestamp = False
    try:
        timestamp_policy.predict(skipped_obs)
    except PolicyValidationError:
        raised_skipped_timestamp = True
    check("policy_skipped_timestamp_fails_closed", raised_skipped_timestamp)


@register("policy_belief_actually_updates_across_steps", "policy")
def _test_policy_belief_actually_updates_across_steps():
    """Confirms predict() is really calling BeliefBank.update() with the
    right data every step (not silently skipping it) -- the posterior for
    the tracked human must be something other than the initial uniform
    prior after several steps of real evidence."""
    policy = _make_policy_fixture()
    _run_crossing_episode(policy, episode_seed=3, n_steps=10)
    posterior = policy.belief_bank.posterior(1)
    check(
        "policy_belief_posterior_moved_from_uniform_prior",
        bool(np.max(np.abs(posterior - 0.5)) > 0.05),
        f"posterior={posterior} (must have moved from [0.5,0.5] after 10 steps of real evidence)",
    )


@register("policy_never_imports_neural_network_frameworks", "policy")
def _test_policy_never_imports_neural_network_frameworks():
    """Guide.md 4.6's explicit requirement: the main policy must not load
    any .pth/Mamba/SARL/RL checkpoint. Checks actual IMPORT STATEMENTS via
    AST (not a raw text/substring search, which false-positives on this
    exact requirement being described in a docstring/comment -- e.g.
    policy.py's own docstring saying "No Mamba/SARL/... is ever loaded
    here" contains the word "SARL") across policy.py and every
    crowd_nav.bayesian_brne module it transitively imports."""
    import ast
    from pathlib import Path

    forbidden_modules = ("torch", "mamba_ssm")
    package_dir = Path(__file__).resolve().parent
    module_names = {"policy", "belief_tracker", "robot_sampler", "trajectory_sampler",
                     "equilibrium_loop", "brne_adapter", "action_conditioned_arhmm", "schemas"}
    offending = []
    for name in module_names:
        path = package_dir / f"{name}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module] if node.module else []
            else:
                continue
            for mod in imported:
                if mod and any(mod == forbidden or mod.startswith(forbidden + ".") for forbidden in forbidden_modules):
                    offending.append((name, mod))
    check(
        "policy_dependency_closure_never_imports_torch_or_mamba_ssm",
        len(offending) == 0,
        f"found forbidden imports: {offending}" if offending else "clean",
    )


# --------------------------------------------------------------------- #
# group="protocol" (Step 4: interaction_protocol.py / data_io.py /
# collect_dataset.py) -- guide.md 5.2's branching test is the load-bearing
# acceptance criterion: SAME snapshot, two different robot actions, yield/
# turn/stop_go/goal_switch responses must differ; continue/assertive must
# not. If this fails the environment is "non-stationary scripted", not
# genuinely closed-loop (guide.md's own words).
# --------------------------------------------------------------------- #


@register("protocol_config_rejects_invalid_values", "protocol")
def _test_protocol_config_rejects_invalid_values():
    from crowd_nav.bayesian_brne.interaction_protocol import ProtocolConfig, ProtocolConfigError

    for kwargs in [
        {"ttc_threshold": 0.0}, {"clearance_threshold": -1.0}, {"conflict_horizon_steps": 0},
        {"yield_gain": 0.0}, {"yield_gain": 1.5}, {"turn_gain": 0.0},
        {"stop_release_steps": 0}, {"max_human_speed": 0.0}, {"max_human_acceleration": 0.0},
    ]:
        raised = False
        try:
            ProtocolConfig(**kwargs)
        except ProtocolConfigError:
            raised = True
        check(f"protocol_config_rejects_{list(kwargs)[0]}", raised, f"kwargs={kwargs}")


@register("protocol_conflict_detected_when_paths_cross", "protocol")
def _test_protocol_conflict_detected_when_paths_cross():
    from crowd_nav.bayesian_brne.interaction_protocol import ProtocolConfig, compute_conflict

    config = ProtocolConfig()
    human_pos, human_vel = np.array([0.0, -3.0]), np.array([0.0, 1.0])

    # Branch B: robot sitting stationary directly ahead on the human's path.
    robot_pos_conflict, robot_vel_conflict = np.array([0.0, -0.5]), np.array([0.0, 0.0])
    in_conflict, ttc, clearance = compute_conflict(robot_pos_conflict, robot_vel_conflict, human_pos, human_vel, 0.6, config, dt=0.25)
    check("protocol_conflict_true_when_paths_cross", in_conflict, f"ttc={ttc:.2f} clearance={clearance:.2f}")

    # Branch A: robot far away, moving away -- no plausible conflict.
    robot_pos_clear, robot_vel_clear = np.array([50.0, 50.0]), np.array([0.0, 0.0])
    in_conflict2, ttc2, clearance2 = compute_conflict(robot_pos_clear, robot_vel_clear, human_pos, human_vel, 0.6, config, dt=0.25)
    check("protocol_conflict_false_when_robot_far_away", not in_conflict2, f"ttc={ttc2:.2f} clearance={clearance2:.2f}")


@register("protocol_branching_reactive_types_differ_stable_types_dont", "protocol")
def _test_protocol_branching_reactive_types_differ_stable_types_dont():
    """The core guide.md 5.2 acceptance test: from the SAME simulator
    snapshot (SAME human state AND SAME robot POSITION), branch into two
    different robot ACTIONS (velocities) and require yield/turn/stop_go/
    goal_switch to react differently while continue/assertive's
    preferred-velocity output stays stable. This is what distinguishes a
    genuinely closed-loop protocol from a "non-stationary scripted" one
    (guide.md's own phrase for the failure mode).

    Fixed per the 2026-08-03 second Step 4 audit: an earlier version of
    this test varied BOTH the robot's position AND its velocity between the
    two branches (``[0,-0.5]`` vs ``[50,50]``), which is not a true
    same-snapshot branch -- the audit's own independent repro instead holds
    robot_pos FIXED at ``[0,-0.5]`` and only changes robot_vel from
    ``[0,1]`` (moving alongside the human, zero closing speed -> no
    conflict) to ``[0,0]`` (stationary -> conflict). That is the fixture
    used here, and it reproduces the exact diffs the audit found
    independently: 0.8485/0.6013/1.0/0.2444 for yield/turn/stop_go/
    goal_switch, 0.0 for continue/assertive."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        BehaviorType, HumanBehaviorState, ProtocolConfig, compute_human_response,
    )

    config = ProtocolConfig()
    human_pos, human_vel, human_pref_speed = np.array([0.0, -3.0]), np.array([0.0, 1.0]), 1.0
    goal = np.array([0.0, 3.0])
    combined_radius = 0.6
    dt = 0.25

    robot_pos = np.array([0.0, -0.5])  # SAME position in both branches.
    robot_vel_clear = np.array([0.0, 1.0])  # moving alongside the human: zero closing speed, no conflict.
    robot_vel_conflict = np.array([0.0, 0.0])  # stationary: closes on the human, triggers conflict.

    reactive_types = [BehaviorType.YIELD, BehaviorType.TURN, BehaviorType.STOP_GO, BehaviorType.GOAL_SWITCH]
    stable_types = [BehaviorType.CONTINUE, BehaviorType.ASSERTIVE]
    expected_diff = {
        BehaviorType.YIELD: 0.8485, BehaviorType.TURN: 0.6013,
        BehaviorType.STOP_GO: 1.0, BehaviorType.GOAL_SWITCH: 0.2444,
    }

    for bt in reactive_types:
        state_a = HumanBehaviorState(behavior_type=bt)
        state_b = HumanBehaviorState(behavior_type=bt)
        pref_a, _, conflict_a, _ = compute_human_response(
            human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel_clear, combined_radius, state_a, config, dt,
        )
        pref_b, _, conflict_b, _ = compute_human_response(
            human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel_conflict, combined_radius, state_b, config, dt,
        )
        diff = float(np.linalg.norm(pref_a - pref_b))
        check(
            f"protocol_{bt.value}_reacts_to_different_robot_action_same_position",
            (not conflict_a) and conflict_b and abs(diff - expected_diff[bt]) < 0.01,
            f"conflict_a={conflict_a} conflict_b={conflict_b} pref_a={pref_a} pref_b={pref_b} "
            f"diff={diff:.4f} expected~={expected_diff[bt]}",
        )

    for bt in stable_types:
        state_a = HumanBehaviorState(behavior_type=bt)
        state_b = HumanBehaviorState(behavior_type=bt)
        pref_a, _, _, _ = compute_human_response(
            human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel_clear, combined_radius, state_a, config, dt,
        )
        pref_b, _, _, _ = compute_human_response(
            human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel_conflict, combined_radius, state_b, config, dt,
        )
        diff = float(np.linalg.norm(pref_a - pref_b))
        check(
            f"protocol_{bt.value}_stays_stable_across_robot_actions_same_position",
            diff < 1e-9,
            f"pref_a={pref_a} pref_b={pref_b} diff={diff:.4e} (must be bit-identical: this type "
            "is conflict-blind by design, guide.md 5.2)",
        )


@register("protocol_assertive_differs_from_continue_via_repulsion_scale", "protocol")
def _test_protocol_assertive_differs_from_continue_via_repulsion_scale():
    """D-item from the 2026-08-03 audit: CONTINUE and ASSERTIVE produced
    bit-identical output in every state, so they could not honestly be
    reported as two distinguishable latent modes. ASSERTIVE now reduces its
    human-human repulsion_scale while in robot-conflict (still conflict-
    BLIND toward the robot itself, per guide.md 5.2); CONTINUE's
    repulsion_scale never depends on conflict. This test requires: (a)
    preferred_velocity stays identical between the two types (both remain
    conflict-blind toward the robot), but (b) repulsion_scale differs once
    in conflict."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        BehaviorType, HumanBehaviorState, ProtocolConfig, compute_human_response,
    )

    config = ProtocolConfig()
    human_pos, human_vel, human_pref_speed = np.array([0.0, -3.0]), np.array([0.0, 1.0]), 1.0
    goal = np.array([0.0, 3.0])
    robot_pos, robot_vel = np.array([0.0, -0.5]), np.array([0.0, 0.0])  # triggers conflict

    pref_c, _, conflict_c, scale_c = compute_human_response(
        human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel, 0.6,
        HumanBehaviorState(behavior_type=BehaviorType.CONTINUE), config, 0.25,
    )
    pref_a, _, conflict_a, scale_a = compute_human_response(
        human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel, 0.6,
        HumanBehaviorState(behavior_type=BehaviorType.ASSERTIVE), config, 0.25,
    )
    check("protocol_assertive_conflict_detected_for_repulsion_test", conflict_c and conflict_a)
    check(
        "protocol_assertive_preferred_velocity_matches_continue",
        np.array_equal(pref_c, pref_a),
        f"pref_continue={pref_c} pref_assertive={pref_a} (both must stay conflict-blind toward the robot)",
    )
    check(
        "protocol_assertive_repulsion_scale_differs_from_continue_in_conflict",
        scale_c == 1.0 and scale_a < 1.0 and scale_a == config.assertive_repulsion_scale,
        f"scale_continue={scale_c} scale_assertive={scale_a} "
        "(this is the only channel that makes assertive/continue distinguishable latent modes)",
    )


@register("protocol_same_snapshot_same_action_is_deterministic", "protocol")
def _test_protocol_same_snapshot_same_action_is_deterministic():
    """Calling compute_human_response twice with an IDENTICAL snapshot and
    robot action must give bit-identical output -- guide.md 5.2's "相同
    robot action 得到相同响应" half of the branching requirement."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        BehaviorType, HumanBehaviorState, ProtocolConfig, compute_human_response,
    )

    config = ProtocolConfig()
    human_pos, human_vel, human_pref_speed = np.array([0.0, -3.0]), np.array([0.0, 1.0]), 1.0
    goal = np.array([0.0, 3.0])
    robot_pos, robot_vel = np.array([0.0, -0.5]), np.array([0.0, 0.0])

    for bt in BehaviorType:
        pref1, _, _, _ = compute_human_response(
            human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel, 0.6,
            HumanBehaviorState(behavior_type=bt), config, 0.25,
        )
        pref2, _, _, _ = compute_human_response(
            human_pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel, 0.6,
            HumanBehaviorState(behavior_type=bt), config, 0.25,
        )
        check(f"protocol_{bt.value}_deterministic_given_same_snapshot", np.array_equal(pref1, pref2), f"pref1={pref1} pref2={pref2}")


@register("protocol_stop_go_releases_after_configured_steps", "protocol")
def _test_protocol_stop_go_releases_after_configured_steps():
    """A stop_go human sees a conflict for the first few steps (robot
    stationary directly ahead), then the robot moves away (as it would in
    any real rollout) so the conflict clears. Requires: (a) the human stops
    immediately once conflict is detected, (b) it does NOT resume the
    instant the conflict clears -- the countdown started at trigger time
    must finish first, and (c) it does resume within stop_release_steps of
    the conflict clearing."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        BehaviorType, HumanBehaviorState, ProtocolConfig, compute_human_response,
    )

    config = ProtocolConfig(stop_release_steps=4)
    human_pos, human_vel, human_pref_speed = np.array([0.0, -3.0]), np.array([0.0, 1.0]), 1.0
    goal = np.array([0.0, 3.0])
    robot_pos_conflict, robot_pos_clear = np.array([0.0, -2.5]), np.array([50.0, 50.0])
    robot_vel = np.array([0.0, 0.0])
    conflict_steps = 3

    state = HumanBehaviorState(behavior_type=BehaviorType.STOP_GO)
    pos = human_pos.copy()
    speeds = []
    for i in range(10):
        robot_pos = robot_pos_conflict if i < conflict_steps else robot_pos_clear
        pref, state, _, _ = compute_human_response(pos, human_vel, human_pref_speed, goal, robot_pos, robot_vel, 0.6, state, config, 0.25)
        speeds.append(float(np.hypot(pref[0], pref[1])))
        pos = pos + pref * 0.25

    check("protocol_stop_go_initially_stops", speeds[0] == 0.0, f"speeds={speeds}")
    check(
        "protocol_stop_go_stays_stopped_through_countdown",
        all(s == 0.0 for s in speeds[:conflict_steps]),
        f"speeds={speeds} (must stay stopped for the full conflict duration, not resume early)",
    )
    check(
        "protocol_stop_go_eventually_resumes",
        any(s > 0.0 for s in speeds[conflict_steps:conflict_steps + config.stop_release_steps + 1]),
        f"speeds={speeds} (must resume within stop_release_steps={config.stop_release_steps} steps of the conflict clearing)",
    )


@register("protocol_synthetic_env_produces_valid_schema_episode", "protocol")
def _test_protocol_synthetic_env_produces_valid_schema_episode():
    from crowd_nav.bayesian_brne import data_io
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario, run_episode
    from crowd_nav.bayesian_brne.schemas import ROBOT_STATE_FIELDS

    rng = np.random.default_rng(7)
    env = make_scenario("baseline_circle", "train", rng, dt=0.25)
    episode = run_episode(
        env, horizon_steps=20, controller="goal_directed", scenario="baseline_circle",
        profile="smoke", split="train", profile_name="smoke",
    )
    episode.update(suite_seed=7, episode_seed=7)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/ep.npz"
        data_io.save_episode(path, episode, source_robot_state_layout=ROBOT_STATE_FIELDS)
        loaded = data_io.load_episode(path)

    check("protocol_episode_robot_shape_ok", loaded["robot"].shape == (20, 9), f"got {loaded['robot'].shape}")
    check("protocol_episode_humans_shape_ok", loaded["humans"].shape == (20, 5, 5), f"got {loaded['humans'].shape}")
    check("protocol_episode_schema_version_stamped", loaded["schema_version"] == 2, f"got {loaded['schema_version']}")
    check("protocol_episode_split_recorded", loaded["split"] == "train")
    check("protocol_episode_profile_name_recorded", loaded["profile_name"] == "smoke")
    check("protocol_episode_controller_type_recorded", loaded["controller_type"] == "goal_directed")
    check("protocol_episode_generator_version_recorded", len(loaded["generator_version"]) > 0)
    check("protocol_episode_initial_state_hash_recorded", len(loaded["initial_state_hash"]) == 64, f"got {loaded['initial_state_hash']!r}")
    check("protocol_episode_profile_params_recorded", isinstance(loaded["profile_params"], dict) and "stop_release_steps" in loaded["profile_params"])
    check("protocol_episode_behavior_type_map_recorded", isinstance(loaded["behavior_type_map"], dict) and len(loaded["behavior_type_map"]) == 5)
    check(
        "protocol_episode_robot_state_layout_stamped",
        loaded["robot_state_layout"] == ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta"),
        f"got {loaded['robot_state_layout']}",
    )


@register("protocol_data_io_rejects_wrong_schema_version", "protocol")
def _test_protocol_data_io_rejects_wrong_schema_version():
    """B8 loader-side check: a file claiming a schema_version this package
    does not recognize (or a robot_state_layout that doesn't match the
    canonical FullState.to_array() order) must be rejected outright by the
    loader, not silently trusted. (This alone is NOT B8's full fix -- see
    ``protocol_data_io_writer_requires_explicit_source_layout`` below for
    the writer-side C1 fix the 2026-08-03 audit demanded.)"""
    import tempfile

    import numpy as _np

    from crowd_nav.bayesian_brne import data_io
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario, run_episode
    from crowd_nav.bayesian_brne.schemas import ROBOT_STATE_FIELDS, SchemaVersionError

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/ep.npz"
        rng = np.random.default_rng(3)
        env = make_scenario("baseline_circle", "train", rng, dt=0.25)
        episode = run_episode(env, horizon_steps=5, controller="goal_directed", scenario="baseline_circle", profile="smoke")
        episode.update(suite_seed=3, episode_seed=3)
        data_io.save_episode(path, episode, source_robot_state_layout=ROBOT_STATE_FIELDS)

        # Corrupt the on-disk schema_version in place.
        with _np.load(path, allow_pickle=False) as npz:
            payload = {k: npz[k] for k in npz.files}
        payload["schema_version"] = _np.array(999)
        _np.savez(path, **payload)

        raised = False
        try:
            data_io.load_episode(path)
        except SchemaVersionError:
            raised = True
        check("protocol_data_io_rejects_wrong_schema_version", raised)

        # Corrupt robot_state_layout instead (wrong field order).
        payload["schema_version"] = _np.array(data_io.DATA_SCHEMA_VERSION)
        payload["robot_state_layout"] = _np.array(["px", "py", "gx", "gy", "vx", "vy", "radius", "v_pref", "theta"])
        _np.savez(path, **payload)
        raised_layout = False
        try:
            data_io.load_episode(path)
        except SchemaVersionError:
            raised_layout = True
        check("protocol_data_io_rejects_wrong_robot_state_layout", raised_layout)


@register("protocol_data_io_writer_requires_explicit_source_layout", "protocol")
def _test_protocol_data_io_writer_requires_explicit_source_layout():
    """C1 fix (2026-08-03 second Step 4 audit): ``save_episode()`` used to
    stamp schema_version/robot_state_layout UNCONDITIONALLY regardless of
    what order the caller's ``robot`` array was actually in -- the audit
    reproduced a live example where a ``Robot.get_obs_array()``-ordered
    array was saved, loaded, and self-certified as canonical (index 2:4
    silently became goal coordinates instead of velocity). ``save_episode``
    now REQUIRES an explicit ``source_robot_state_layout`` with NO default,
    and reorders via ``schemas.reorder_robot_array_to_canonical``. This
    test reproduces exactly the audit's counter-example and requires it to
    now be handled correctly: (a) omitting the argument is a hard TypeError
    (Python's own call semantics, not a runtime guess), (b) a bogus
    (non-permutation) layout is rejected, (c) a genuinely
    get_obs_array()-ordered array, correctly DECLARED as such, is reordered
    to canonical and round-trips with velocity/goal NOT swapped."""
    import inspect
    import tempfile

    from crowd_nav.bayesian_brne import data_io
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario, run_episode
    from crowd_nav.bayesian_brne.schemas import (
        GET_OBS_ARRAY_LAYOUT, ROBOT_STATE_FIELDS, SchemaVersionError, robot_goal, robot_velocity,
    )

    sig = inspect.signature(data_io.save_episode)
    check(
        "protocol_data_io_save_episode_source_layout_has_no_default",
        sig.parameters["source_robot_state_layout"].default is inspect.Parameter.empty,
        f"signature={sig}",
    )

    rng = np.random.default_rng(11)
    env = make_scenario("baseline_circle", "train", rng, dt=0.25)
    episode = run_episode(env, horizon_steps=6, controller="goal_directed", scenario="baseline_circle", profile="smoke")
    episode.update(suite_seed=11, episode_seed=11)
    canonical_robot = episode["robot"].copy()

    with tempfile.TemporaryDirectory() as tmp:
        raised_missing = False
        try:
            data_io.save_episode(f"{tmp}/missing.npz", episode)  # type: ignore[call-arg]
        except TypeError:
            raised_missing = True
        check("protocol_data_io_missing_source_layout_rejected", raised_missing)

        raised_bogus = False
        try:
            data_io.save_episode(f"{tmp}/bogus.npz", episode, source_robot_state_layout=("a", "b", "c"))
        except SchemaVersionError:
            raised_bogus = True
        check("protocol_data_io_bogus_source_layout_rejected", raised_bogus)

        # THE REVERSE EXAMPLE the audit demanded: build a genuinely
        # get_obs_array()-ordered robot array (permute canonical -> get_obs
        # order), declare it correctly, and require save_episode to reorder
        # it back to canonical -- not just trust a stamped label blindly.
        canonical_index = {name: i for i, name in enumerate(ROBOT_STATE_FIELDS)}
        perm_to_get_obs = [canonical_index[name] for name in GET_OBS_ARRAY_LAYOUT]
        get_obs_ordered_robot = canonical_robot[:, perm_to_get_obs]

        episode_get_obs = dict(episode)
        episode_get_obs["robot"] = get_obs_ordered_robot
        path = f"{tmp}/get_obs.npz"
        data_io.save_episode(path, episode_get_obs, source_robot_state_layout=GET_OBS_ARRAY_LAYOUT)
        loaded = data_io.load_episode(path)

        check(
            "protocol_data_io_reorders_get_obs_layout_to_canonical",
            np.allclose(loaded["robot"], canonical_robot),
            "loaded robot array must match the ORIGINAL canonical array after being saved from get_obs order",
        )
        # Check the LAST timestep, not the first -- the robot starts at
        # rest (velocity exactly [0,0] at t=0), which would make a
        # swapped-field bug invisible; by the last step velocity is
        # genuinely nonzero and clearly distinct from the goal coordinates.
        last = -1
        vel_ok = np.allclose(robot_velocity(loaded["robot"][last]), robot_velocity(canonical_robot[last]))
        goal_ok = np.allclose(robot_goal(loaded["robot"][last]), robot_goal(canonical_robot[last]))
        loaded_vel = robot_velocity(loaded["robot"][last])
        check(
            "protocol_data_io_reordered_velocity_and_goal_not_swapped",
            vel_ok and goal_ok and float(np.hypot(*loaded_vel)) > 0.1,
            f"loaded vel={loaded_vel} goal={robot_goal(loaded['robot'][last])} vs "
            f"canonical vel={robot_velocity(canonical_robot[last])} goal={robot_goal(canonical_robot[last])}",
        )


@register("protocol_collect_dataset_smoke_end_to_end", "protocol")
def _test_protocol_collect_dataset_smoke_end_to_end():
    import tempfile

    from crowd_nav.bayesian_brne import collect_dataset, data_io

    with tempfile.TemporaryDirectory() as tmp:
        paths = collect_dataset.collect(
            split="train", scenario="baseline_circle", episodes=3, seed=2407,
            profile_name="smoke", output_dir=tmp, horizon_steps=10, dt=0.25, controller="goal_directed",
        )
        check("protocol_collect_dataset_writes_expected_count", len(paths) == 3, f"got {len(paths)}")
        for p in paths:
            episode = data_io.load_episode(p)
            check(f"protocol_collect_dataset_episode_loads_{Path(p).stem}", episode["robot"].shape[0] == 10)
            check(f"protocol_collect_dataset_episode_profile_name_{Path(p).stem}", episode["profile_name"] == "smoke")


@register("order6_allocate_controller_type_equal_and_deterministic", "protocol")
def _test_order6_allocate_controller_type_equal_and_deterministic():
    """Order 6.1: "按 episode index 和 suite seed 确定性等量分配，不能看结
    果后改比例". Requires exactly equal counts over a multiple-of-4 episode
    range, and requires the mapping to be a pure function of episode_index
    (same index -> same controller, every call)."""
    from crowd_nav.bayesian_brne.interaction_protocol import CONTROLLER_TYPES, allocate_controller_type

    n = 40
    counts = {}
    for i in range(n):
        c = allocate_controller_type(i)
        counts[c] = counts.get(c, 0) + 1
    check(
        "order6_allocation_is_exactly_equal",
        all(counts.get(c, 0) == n // len(CONTROLLER_TYPES) for c in CONTROLLER_TYPES),
        f"counts over {n} episodes: {counts}",
    )
    check(
        "order6_allocation_is_deterministic",
        all(allocate_controller_type(i) == allocate_controller_type(i) for i in range(n)),
    )


@register("order6_all_four_controllers_produce_valid_episodes", "protocol")
def _test_order6_all_four_controllers_produce_valid_episodes():
    """Order 6: every controller must actually run end-to-end and produce
    a schema-valid episode with kinematically legal robot speeds."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        CONTROLLER_TYPES, ProtocolConfig, make_scenario, run_episode,
    )

    for controller in CONTROLLER_TYPES:
        rng = np.random.default_rng(100)
        env = make_scenario("baseline_circle", "train", rng, dt=0.25)
        episode = run_episode(
            env, horizon_steps=12, controller=controller, scenario="baseline_circle",
            profile="pilot", split="train", profile_name="pilot",
        )
        robot = episode["robot"]
        speeds = np.hypot(robot[:, 2], robot[:, 3])
        max_speed_cap = ProtocolConfig().max_human_speed + 1e-6
        check(
            f"order6_{controller}_episode_shape_ok",
            robot.shape == (12, 9),
            f"got {robot.shape}",
        )
        check(
            f"order6_{controller}_robot_speed_kinematically_legal",
            bool(np.all(speeds <= max_speed_cap)),
            f"max observed speed={float(speeds.max()):.4f}, cap={max_speed_cap:.4f}",
        )
        check(
            f"order6_{controller}_controller_type_recorded",
            episode["controller_type"] == controller,
        )


@register("order6_orca_and_goal_directed_produce_different_actions", "protocol")
def _test_order6_orca_and_goal_directed_produce_different_actions():
    """Order 6's own required test: different controllers must give
    genuinely different action coverage from the SAME scenario/seed --
    ORCA reacts to the crowd, goal_directed does not."""
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario, run_episode

    rng1 = np.random.default_rng(7)
    env1 = make_scenario("dense_circle", "test_nominal", rng1, dt=0.25)
    ep_goal = run_episode(env1, horizon_steps=20, controller="goal_directed", scenario="dense_circle", profile="pilot", split="test_nominal", profile_name="pilot")

    rng2 = np.random.default_rng(7)
    env2 = make_scenario("dense_circle", "test_nominal", rng2, dt=0.25)
    ep_orca = run_episode(env2, horizon_steps=20, controller="orca", scenario="dense_circle", profile="pilot", split="test_nominal", profile_name="pilot")

    diff = float(np.abs(ep_goal["robot_actions"] - ep_orca["robot_actions"]).max())
    check(
        "order6_orca_diverges_from_goal_directed_in_dense_crowd",
        diff > 1e-3,
        f"max abs action diff over the episode={diff:.4f} (dense_circle has 10 humans, "
        "ORCA should visibly react while goal_directed ignores them)",
    )


@register("order6_original_brne_uses_locked_upstream_solver", "protocol")
def _test_order6_original_brne_uses_locked_upstream_solver():
    """Order 6.3: original_brne must use the LOCKED upstream BRNE solver
    with a fixed CV/GP prior, never SM-BRNE's own posterior machinery.
    Verifies the controller runs and its action differs from pure
    goal_directed nominal when humans are nearby (i.e. it is actually
    equilibrating, not silently falling back)."""
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario, run_episode

    rng = np.random.default_rng(3)
    env = make_scenario("baseline_circle", "train", rng, dt=0.25)
    episode = run_episode(env, horizon_steps=10, controller="original_brne", scenario="baseline_circle", profile="pilot", split="train", profile_name="pilot")
    fallback_events = [e for e in episode["events"] if e.get("type") == "controller_fallback"]
    check(
        "order6_original_brne_ran_without_fallback_on_normal_scenario",
        len(fallback_events) == 0,
        f"fallback_events={fallback_events} (a normal 5-person baseline_circle scenario should not trigger the BRNE solve to fail)",
    )
    check("order6_original_brne_actions_finite", bool(np.all(np.isfinite(episode["robot_actions"]))))


@register("order6_scripted_probe_executes_left_right_stop_cycle", "protocol")
def _test_order6_scripted_probe_executes_left_right_stop_cycle():
    """Order 6.4: scripted_probe must execute a reproducible, bounded
    left/right offset + brief stop within conflict windows. Uses a
    dense_circle scenario (more crossing conflicts likely) over enough
    steps to trigger at least one probe phase, and requires: (a) it
    reproduces byte-identically given the same seed, (b) at least one
    non-nominal probe phase actually fires over the episode."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        RobotControllerState, compute_robot_action, make_scenario,
    )

    def _run(seed):
        rng = np.random.default_rng(seed)
        env = make_scenario("dense_circle", "test_nominal", rng, dt=0.25)
        ctrl_state = RobotControllerState(controller_type="scripted_probe")
        phases = []
        actions = []
        for _ in range(30):
            action, _ = compute_robot_action(env, ctrl_state)
            actions.append(action.copy())
            phases.append(ctrl_state.scripted_probe_state.probe_phase)
            env.step(action)
        return np.array(actions), phases

    actions1, phases1 = _run(42)
    actions2, phases2 = _run(42)
    check("order6_scripted_probe_byte_identical_same_seed", np.array_equal(actions1, actions2))
    check(
        "order6_scripted_probe_at_least_one_probe_phase_fires",
        any(p is not None for p in phases1),
        f"phases over 30 steps: {phases1}",
    )


@register("order6_controller_fallback_is_counted_and_labeled", "protocol")
def _test_order6_controller_fallback_is_counted_and_labeled():
    """Order 6.6: "controller 失败必须计数并写 event，不能静默换 controller
    后保留原标签". Forces a real failure (a bogus BRNE root, so
    original_brne's solve cannot load upstream) and requires the fallback
    to be visible in the saved events, counted, and the episode to still
    produce valid (goal_directed-fallback) data rather than crashing."""
    from crowd_nav.bayesian_brne.interaction_protocol import make_scenario, run_episode

    rng = np.random.default_rng(5)
    env = make_scenario("baseline_circle", "train", rng, dt=0.25)
    episode = run_episode(
        env, horizon_steps=8, controller="original_brne", scenario="baseline_circle",
        profile="pilot", split="train", profile_name="pilot", brne_root="/nonexistent/brne/root",
    )
    fallback_events = [e for e in episode["events"] if e.get("type") == "controller_fallback"]
    check(
        "order6_forced_failure_produces_fallback_events",
        len(fallback_events) > 0,
        f"got {len(fallback_events)} fallback events out of {episode['robot'].shape[0]} steps",
    )
    check(
        "order6_fallback_events_labeled_with_failing_controller",
        all(e.get("controller") == "original_brne" for e in fallback_events),
        f"fallback_events={fallback_events[:2]}",
    )
    check("order6_episode_still_produces_finite_data_despite_fallback", bool(np.all(np.isfinite(episode["robot"]))))


@register("order6_controller_type_never_a_model_feature", "protocol")
def _test_order6_controller_type_never_a_model_feature():
    """Order 6.5: "controller_type 不进入 Bayesian 模型特征". Structural
    check: mode_model.py's feature vector must not name controller_type,
    and extract_transitions must not accept it as an argument."""
    import inspect

    from crowd_nav.bayesian_brne.mode_model import FEATURE_NAMES, extract_transitions

    check("order6_feature_names_excludes_controller_type", "controller_type" not in FEATURE_NAMES, f"FEATURE_NAMES={FEATURE_NAMES}")
    sig = inspect.signature(extract_transitions)
    check("order6_extract_transitions_signature_excludes_controller_type", "controller_type" not in sig.parameters, f"signature={sig}")


@register("protocol_unknown_scenario_and_split_fail_closed", "protocol")
def _test_protocol_unknown_scenario_and_split_fail_closed():
    """C2/C3 fix (2026-08-03 audit): an earlier version accepted all six
    scenario NAMES but silently routed every one through the circle
    generator regardless of name -- requesting 'baseline_square' produced
    circle-shaped data still labeled 'baseline_square'. ``make_scenario``
    must now raise on any name/split it does not recognize, never fall
    back to a default geometry."""
    from crowd_nav.bayesian_brne.interaction_protocol import ScenarioConfigError, make_scenario

    rng = np.random.default_rng(0)
    raised_scenario = False
    try:
        make_scenario("totally_unknown_scenario", "train", rng, dt=0.25)
    except ScenarioConfigError:
        raised_scenario = True
    check("protocol_unknown_scenario_name_raises", raised_scenario)

    raised_split = False
    try:
        make_scenario("baseline_circle", "totally_unknown_split", rng, dt=0.25)
    except ScenarioConfigError:
        raised_split = True
    check("protocol_unknown_split_raises", raised_split)


@register("protocol_circle_and_square_scenarios_are_genuinely_different_geometry", "protocol")
def _test_protocol_circle_and_square_scenarios_are_genuinely_different_geometry():
    """C2/C3 fix continued: 'circle' and 'square' scenarios must actually
    produce different spatial layouts, not the same generator under two
    labels. Circle: every human/robot position is equidistant from the
    origin (``hypot(x,y) == radius`` for all agents). Square: positions sit
    on a square perimeter (``max(|x|,|y|) == half-side`` for all agents,
    and NOT all equidistant from the origin -- corners are farther from
    the origin than edge midpoints)."""
    from crowd_nav.bayesian_brne.interaction_protocol import SCENARIO_TABLE, make_scenario

    rng_circle = np.random.default_rng(0)
    env_circle = make_scenario("baseline_circle", "train", rng_circle, dt=0.25)
    _, circle_radius, _ = SCENARIO_TABLE["baseline_circle"]
    circle_dists = [float(np.hypot(*h.pos)) for h in env_circle.humans] + [float(np.hypot(*env_circle.robot_pos))]
    check(
        "protocol_circle_scenario_all_agents_equidistant_from_origin",
        all(abs(d - circle_radius) < 0.05 for d in circle_dists),
        f"dists={circle_dists}, expected all ~{circle_radius}",
    )

    rng_square = np.random.default_rng(0)
    env_square = make_scenario("baseline_square", "test_nominal", rng_square, dt=0.25)  # C4: non-circle scenarios are test-only
    _, square_side, _ = SCENARIO_TABLE["baseline_square"]
    half = square_side / 2.0
    square_positions = [h.pos for h in env_square.humans] + [env_square.robot_pos]
    on_perimeter = [abs(max(abs(p[0]), abs(p[1])) - half) < 0.05 for p in square_positions]
    check("protocol_square_scenario_all_agents_on_square_perimeter", all(on_perimeter), f"max|x,y| per agent vs half={half}: {[max(abs(p[0]), abs(p[1])) for p in square_positions]}")

    square_dists = [float(np.hypot(p[0], p[1])) for p in square_positions]
    check(
        "protocol_square_scenario_not_all_equidistant_from_origin",
        max(square_dists) - min(square_dists) > 0.5,
        f"dists={square_dists} (a square perimeter must have corners farther from the "
        "center than edge midpoints -- if this is flat, it's still a circle in disguise)",
    )

    check(
        "protocol_square_scenario_uses_correct_n_humans",
        len(env_square.humans) == SCENARIO_TABLE["baseline_square"][2],
        f"got {len(env_square.humans)} humans, expected {SCENARIO_TABLE['baseline_square'][2]}",
    )


@register("protocol_train_validation_and_heldout_ranges_are_disjoint", "protocol")
def _test_protocol_train_validation_and_heldout_ranges_are_disjoint():
    """guide.md 5.3: train/validation and test_heldout_interactive must
    have non-overlapping ranges across at least stop duration, turn angle/
    rate, yield threshold, assertiveness probability, speed range, and
    goal-switch location/time. ``assert_profiles_disjoint`` already runs at
    IMPORT time (a hard invariant, not just a test) -- this test makes that
    guarantee a regression-checkable, reportable fact, and additionally
    verifies sampled params for the two splits never numerically collide
    across many seeds."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        HELDOUT_INTERACTIVE_PROFILE, TRAIN_VALIDATION_PROFILE, sample_profile_params,
    )

    check("protocol_profiles_disjoint_at_import_time_did_not_raise", True)

    rng = np.random.default_rng(42)
    for _ in range(200):
        train_params = sample_profile_params(rng, TRAIN_VALIDATION_PROFILE)
        heldout_params = sample_profile_params(rng, HELDOUT_INTERACTIVE_PROFILE)
        checks = [
            train_params.stop_release_steps < heldout_params.stop_release_steps,
            train_params.turn_gain < heldout_params.turn_gain,
            train_params.yield_ttc_threshold < heldout_params.yield_ttc_threshold,
            train_params.speed_hi < heldout_params.speed_lo,
            train_params.goal_switch_ttc_threshold < heldout_params.goal_switch_ttc_threshold,
        ]
        if not all(checks):
            check(
                "protocol_sampled_train_and_heldout_params_never_collide",
                False,
                f"train={train_params} heldout={heldout_params}",
            )
            return
    check("protocol_sampled_train_and_heldout_params_never_collide", True, "200 samples, all disjoint")


def _stable_seed(*parts: str) -> int:
    """Deterministic, cross-process-stable seed (D-item fix: an earlier
    version used Python's process-randomized ``hash()`` for test seeds,
    which the 2026-08-03 audit flagged as not byte-for-byte reproducible
    across processes/runs). Uses zlib.crc32, which is stable by spec."""
    import zlib

    return zlib.crc32(":".join(parts).encode("utf-8")) % (2**31)


@register("protocol_scenario_split_matrix_matches_guide_5_3", "protocol")
def _test_protocol_scenario_split_matrix_matches_guide_5_3():
    """C4 fix (2026-08-03 third Step 4 audit): guide.md 5.3 requires
    Train/Validation to use ONLY 5-person baseline_circle; all six
    scenarios are for test_nominal/test_heldout_interactive only. The valid
    (scenario, split) matrix is 2 (baseline_circle x {train, validation}) +
    12 (six scenarios x two test splits) = 14 -- NOT the full 6x4=24
    cartesian product an earlier version required to all succeed. This test
    requires: valid combinations construct with the right human count;
    invalid combinations (any non-baseline_circle scenario under train/
    validation) raise ScenarioConfigError."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        SCENARIO_TABLE, TRAIN_VALIDATION_SCENARIO, VALID_SPLITS, ScenarioConfigError, make_scenario,
    )

    test_splits = [s for s in VALID_SPLITS if s not in ("train", "validation")]
    n_valid_ok = 0
    n_invalid_rejected = 0
    expected_valid = len(SCENARIO_TABLE) * len(test_splits) + 2  # +2 for baseline_circle x {train, validation}

    for name, (geometry, size, n_humans) in SCENARIO_TABLE.items():
        for split in VALID_SPLITS:
            rng = np.random.default_rng(_stable_seed(name, split))
            is_valid_combo = (split in test_splits) or (name == TRAIN_VALIDATION_SCENARIO)
            if is_valid_combo:
                env = make_scenario(name, split, rng, dt=0.25)
                if len(env.humans) == n_humans:
                    n_valid_ok += 1
                else:
                    check(f"protocol_scenario_split_construct_{name}_{split}", False, f"got {len(env.humans)} humans, expected {n_humans}")
            else:
                raised = False
                try:
                    make_scenario(name, split, rng, dt=0.25)
                except ScenarioConfigError:
                    raised = True
                if raised:
                    n_invalid_rejected += 1
                else:
                    check(f"protocol_invalid_combo_rejected_{name}_{split}", False, "expected ScenarioConfigError, none raised")

    check(
        "protocol_valid_scenario_split_combos_all_construct",
        n_valid_ok == expected_valid,
        f"{n_valid_ok}/{expected_valid} valid (scenario, split) combinations constructed successfully",
    )
    n_expected_invalid = len(SCENARIO_TABLE) * len(VALID_SPLITS) - expected_valid
    check(
        "protocol_invalid_scenario_split_combos_all_rejected",
        n_invalid_rejected == n_expected_invalid,
        f"{n_invalid_rejected}/{n_expected_invalid} invalid (scenario, split) combinations correctly raised ScenarioConfigError",
    )


@register("protocol_six_scenarios_zero_overlap_across_many_seeds", "protocol")
def _test_protocol_six_scenarios_zero_overlap_across_many_seeds():
    """C3 fix (2026-08-03 third Step 4 audit): independent audit measured
    25.6%-55.8% initial robot-human/human-human overlap across the six
    scenarios before rejection sampling was added. Requires >=1000
    constructions per scenario with ZERO pairwise center-distance
    violations (below combined radius + INITIAL_CLEARANCE_MARGIN), and
    reports the minimum-clearance 5th percentile per scenario (same metric
    the audit used) as a regression baseline."""
    from crowd_nav.bayesian_brne.interaction_protocol import (
        INITIAL_CLEARANCE_MARGIN, SCENARIO_TABLE, make_scenario,
    )

    n_episodes = 1000
    for name in SCENARIO_TABLE:
        min_clearances = []
        violations = 0
        for i in range(n_episodes):
            rng = np.random.default_rng(_stable_seed(name, "overlap_check", str(i)))
            env = make_scenario(name, "test_nominal", rng, dt=0.25)
            positions = [h.pos for h in env.humans] + [env.robot_pos]
            radii = [h.radius for h in env.humans] + [env.robot_radius]
            min_clear = float("inf")
            for a in range(len(positions)):
                for b in range(a + 1, len(positions)):
                    dist = float(np.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]))
                    required = radii[a] + radii[b] + INITIAL_CLEARANCE_MARGIN
                    clearance = dist - required
                    min_clear = min(min_clear, clearance)
                    if clearance < -1e-9:
                        violations += 1
            min_clearances.append(min_clear)
        p5 = float(np.percentile(min_clearances, 5))
        check(
            f"protocol_{name}_zero_initial_overlap_over_{n_episodes}_episodes",
            violations == 0,
            f"{violations} pairwise clearance violations over {n_episodes} episodes; min-clearance 5th percentile={p5:.4f}m",
        )


@register("upgrade_u0_robot_actions_equals_shifted_velocity", "protocol")
def _test_upgrade_u0_robot_actions_equals_shifted_velocity():
    """Upgrade U0 (2026-08-03 core-model-upgrade decision): the new
    action-conditioned switching model's ``B_k @ u_R,t`` term requires
    ``u_R,t`` to be the robot's ACTUALLY EXECUTED action, not merely a
    velocity feature correlated with it (guide.md: "机器人状态中的vx,vy只
    有在它与执行命令逐步等价并有自动测试时才能作为动作"). Verifies, for
    EVERY controller type (not just goal_directed), that
    ``robot[t+1, 2:4] == robot_actions[t]`` exactly -- i.e. ``robot_actions``
    IS the executed action and the next frame's recorded velocity is
    nothing more than that action having been applied, with zero
    additional noise/dynamics in between. This is what licenses using
    ``robot_actions`` directly as ``u_R,t`` without recollecting data."""
    from crowd_nav.bayesian_brne.interaction_protocol import CONTROLLER_TYPES, make_scenario, run_episode
    from crowd_nav.bayesian_brne.schemas import robot_velocity

    for controller in CONTROLLER_TYPES:
        rng = np.random.default_rng(77)
        env = make_scenario("baseline_circle", "train", rng, dt=0.25)
        episode = run_episode(
            env, horizon_steps=15, controller=controller, scenario="baseline_circle",
            profile="pilot", split="train", profile_name="pilot",
        )
        robot = episode["robot"]
        actions = episode["robot_actions"]
        max_diff = 0.0
        for t in range(robot.shape[0] - 1):
            diff = float(np.max(np.abs(robot_velocity(robot[t + 1]) - actions[t])))
            max_diff = max(max_diff, diff)
        check(
            f"upgrade_u0_{controller}_robot_actions_is_executed_action",
            max_diff < 1e-9,
            f"max |robot[t+1,vx:vy] - robot_actions[t]| over episode = {max_diff:.2e} (must be exactly 0)",
        )


# --------------------------------------------------------------------- #
# group="arhmm" (Upgrade U1+: action_conditioned_arhmm.py) -- grows as EM
# is implemented in Upgrade U2. Only config validation exists so far.
# --------------------------------------------------------------------- #


@register("arhmm_config_rejects_invalid_values", "arhmm")
def _test_arhmm_config_rejects_invalid_values():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, ARHMMConfigError

    for kwargs in [
        {"k_candidates": ()}, {"k_candidates": (1, 1)}, {"sticky_kappa": -1.0},
        {"dirichlet_alpha": 0.0}, {"shrinkage_scale": 0.0}, {"inverse_wishart_dof": 0.0},
        {"inverse_wishart_scale": 0.0}, {"em_max_iters": 0}, {"em_tol": 0.0},
    ]:
        raised = False
        try:
            ARHMMConfig(**kwargs)
        except ARHMMConfigError:
            raised = True
        check(f"arhmm_config_rejects_{list(kwargs)[0]}", raised, f"kwargs={kwargs}")


@register("arhmm_context_features_exclude_self_kinematics", "arhmm")
def _test_arhmm_context_features_exclude_self_kinematics():
    """Structural guard for the design decision explained in the module
    docstring: c_t must never include self-kinematic features (those live
    in the A_k @ v_t term), or the model could silently reconstruct the
    legacy method's self-only redundancy through c_t's back door."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import CONTEXT_FEATURE_NAMES
    from crowd_nav.bayesian_brne.mode_model import FEATURE_NAMES

    self_kinematic = {"speed", "delta_speed", "heading_change", "lateral_acceleration"}
    check(
        "arhmm_context_features_are_subset_of_legacy_feature_names",
        set(CONTEXT_FEATURE_NAMES).issubset(set(FEATURE_NAMES)),
        f"CONTEXT_FEATURE_NAMES={CONTEXT_FEATURE_NAMES}",
    )
    check(
        "arhmm_context_features_exclude_self_kinematic_names",
        not (set(CONTEXT_FEATURE_NAMES) & self_kinematic),
        f"CONTEXT_FEATURE_NAMES={CONTEXT_FEATURE_NAMES} must not intersect {self_kinematic}",
    )


def _make_synthetic_arhmm_sequences(rng, n_seq=20, T=30, shuffle_action=False, anisotropic_q=False):
    """Genuinely action-dependent 2-mode synthetic data: mode 0 responds
    WEAKLY to the robot's action, mode 1 responds STRONGLY (and in the
    opposite direction). ``u`` is sampled independently of ``z`` (its
    EFFECT on v_{t+1} depends on z, but u itself is not caused by z) --
    the same "informative but not confounded" structure the real Order 6
    controllers/FSM are supposed to provide. ``anisotropic_q=True`` uses a
    deliberately NON-spherical true noise covariance (2026-08-03 second
    audit round item: the earlier Q-decoupled M-step formula was only
    exactly correct for isotropic Q, so monotonicity must be stress-tested
    on data whose true Q is NOT a multiple of the identity)."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMSequence

    true_B = {0: np.eye(2) * 0.05, 1: np.eye(2) * -0.6}
    if anisotropic_q:
        Q_true = np.array([[0.03, 0.015], [0.015, 0.02]])  # strongly non-spherical, correlated, unequal variances (PD: det=0.000375>0)
    else:
        Q_true = np.eye(2) * 0.01
    Pi_true = np.array([[0.95, 0.05], [0.05, 0.95]])

    sequences = []
    true_z_all = []
    for si in range(n_seq):
        z = int(rng.integers(0, 2))
        v = np.zeros((T, 2))
        v[0] = rng.normal(0, 0.3, size=2)
        u = rng.normal(0, 1.0, size=(T, 2))
        c = rng.normal(0, 1.0, size=(T, 6))
        zs = [z]
        for t in range(T - 1):
            v[t + 1] = v[t] + true_B[z] @ u[t] + rng.multivariate_normal([0, 0], Q_true)
            if rng.random() > Pi_true[z, z]:
                z = 1 - z
            zs.append(z)
        if shuffle_action:
            u = rng.permutation(u)
        sequences.append(ARHMMSequence(
            track_key=(si, 0), v_current=v[:-1], v_next=v[1:], u_robot=u[:-1], context=c[:-1],
        ))
        true_z_all.append(np.array(zs[:-1]))
    return sequences, true_z_all, true_B, Pi_true


@register("arhmm_emission_log_probs_matches_looped_reference", "arhmm")
def _test_arhmm_emission_log_probs_matches_looped_reference():
    """2026-08-03 U3 performance pass: ``_emission_log_probs`` was rewritten
    from a per-(t,k) Python loop to a vectorized-over-t computation, since
    profiling showed the ORIGINAL per-row loop dominated a real-data fit's
    wall time (~700s for a 390,000-row/K=1,2 fit, ~100x more than m_step).
    This test recomputes the SAME quantity via an independent per-(t,k)
    reference loop (not the code path under test) on a randomly initialized
    multi-mode artifact and requires the two to match to floating-point
    noise.

    2026-08-03 (U3 monotonicity fix): the reference here is written
    INLINE, NOT via ``mode_model._logpdf_gaussian`` (which this test
    originally called) -- that function adds its own ``+1e-6*I`` floor to
    the covariance, appropriate for the legacy K-means model's much
    larger-scale covariances but WRONG for this model, where ``Q_k`` is
    guaranteed strictly positive-definite by construction (see
    ``_emission_log_probs``'s docstring) and where a real, floor-scale
    ``Q_k`` shrinkage was found to silently break the E-step/M-step
    correspondence EM's monotonicity guarantee depends on. Using
    ``_logpdf_gaussian`` here would make this test enforce the WRONG
    (floored) behavior instead of catching a regression back to it."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMArtifact, _emission_log_probs, _design_dims,
    )

    rng = np.random.default_rng(31)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=5, T=20, anisotropic_q=True)
    d_v, d_u, d_c, _ = _design_dims()
    K = 3
    A = [rng.normal(size=(d_v, d_v)) * 0.3 for _ in range(K)]
    B = [rng.normal(size=(d_v, d_u)) * 0.3 for _ in range(K)]
    C = [rng.normal(size=(d_v, d_c)) * 0.3 for _ in range(K)]
    d_vec = [rng.normal(size=(d_v,)) * 0.1 for _ in range(K)]
    Q = []
    for _ in range(K):
        m = rng.normal(size=(d_v, d_v))
        Q.append(m @ m.T + 0.05 * np.eye(d_v))
    Pi = np.full((K, K), 1.0 / K)
    initial_distribution = np.full(K, 1.0 / K)
    artifact = ARHMMArtifact(K=K, A=A, B=B, C=C, d=d_vec, Q=Q, Pi=Pi,
                              initial_distribution=initial_distribution, dt=0.25, model_card={})

    def _reference_logpdf_no_floor(y, mean, cov):
        sign, logdet = np.linalg.slogdet(cov)
        diff = y - mean
        inv = np.linalg.inv(cov)
        return float(-0.5 * (diff @ inv @ diff + logdet + len(y) * np.log(2.0 * np.pi)))

    max_diff = 0.0
    for seq in sequences:
        vectorized = _emission_log_probs(seq, artifact)
        L = seq.v_current.shape[0]
        reference = np.zeros((L, K))
        for t in range(L):
            y = seq.v_next[t]
            for k in range(K):
                mean = (
                    artifact.A[k] @ seq.v_current[t] + artifact.B[k] @ seq.u_robot[t]
                    + artifact.C[k] @ seq.context[t] + artifact.d[k]
                )
                reference[t, k] = _reference_logpdf_no_floor(y, mean, artifact.Q[k])
        max_diff = max(max_diff, float(np.max(np.abs(vectorized - reference))))
    check(
        "arhmm_emission_log_probs_matches_looped_reference",
        max_diff < 1e-8,
        f"max abs diff between vectorized and looped-reference emission log-probs = {max_diff:.2e}",
    )


@register("arhmm_batched_e_step_matches_looped_reference", "arhmm")
def _test_arhmm_batched_e_step_matches_looped_reference():
    """2026-08-03 U3 performance pass: ``e_step`` was rewritten to batch the
    forward-backward recursion across all sequences sharing the same
    length L (real data has every non-gapped track at the same length, so
    this turns ~2000 per-sequence recursions into one batched recursion).
    This is the single most safety-critical check for that rewrite: it
    builds sequences of THREE DIFFERENT lengths (10, 15, 20) -- deliberately
    exercising more than one length-group in the same e_step call, unlike
    real (uniform-length) data would -- and cross-checks the batched
    e_step's ``log_likelihood``/``gammas``/``xis`` against an independent
    per-sequence computation using ``forward_backward`` (kept unchanged as
    the single-sequence reference implementation) plus the ORIGINAL
    (pre-rewrite) gamma/xi formula, reproduced here rather than imported so
    this test does not silently pass if both were broken the same way."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMArtifact, e_step, forward_backward, _design_dims,
    )

    rng = np.random.default_rng(41)
    d_v, d_u, d_c, _ = _design_dims()
    K = 3
    A = [rng.normal(size=(d_v, d_v)) * 0.3 for _ in range(K)]
    B = [rng.normal(size=(d_v, d_u)) * 0.3 for _ in range(K)]
    C = [rng.normal(size=(d_v, d_c)) * 0.3 for _ in range(K)]
    d_vec = [rng.normal(size=(d_v,)) * 0.1 for _ in range(K)]
    Q = []
    for _ in range(K):
        m = rng.normal(size=(d_v, d_v))
        Q.append(m @ m.T + 0.05 * np.eye(d_v))
    Pi = rng.dirichlet(np.ones(K), size=K)
    initial_distribution = rng.dirichlet(np.ones(K))
    artifact = ARHMMArtifact(K=K, A=A, B=B, C=C, d=d_vec, Q=Q, Pi=Pi,
                              initial_distribution=initial_distribution, dt=0.25, model_card={})

    sequences = []
    for T in (10, 15, 20, 15, 10, 20, 10):  # deliberately mixed + repeated lengths
        seqs, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=1, T=T + 1)
        sequences.append(seqs[0])

    e_result = e_step(sequences, artifact)

    from scipy.special import logsumexp as _logsumexp
    ref_total_ll = 0.0
    max_gamma_diff = 0.0
    max_xi_diff = 0.0
    for i, seq in enumerate(sequences):
        log_alpha, log_beta, seq_ll = forward_backward(seq, artifact)
        ref_total_ll += seq_ll
        gamma_ref = np.exp(np.clip(log_alpha + log_beta - seq_ll, -700, 0))
        gamma_ref = gamma_ref / np.maximum(gamma_ref.sum(axis=1, keepdims=True), 1e-300)
        max_gamma_diff = max(max_gamma_diff, float(np.max(np.abs(gamma_ref - e_result["gammas"][i]))))

        from crowd_nav.bayesian_brne.action_conditioned_arhmm import _emission_log_probs
        log_b = _emission_log_probs(seq, artifact)
        log_Pi = np.log(np.maximum(artifact.Pi, 1e-300))
        T_minus_1 = log_alpha.shape[0]
        if T_minus_1 >= 2:
            xi_ref = np.zeros((T_minus_1 - 1, K, K))
            for t in range(T_minus_1 - 1):
                mat = log_alpha[t][:, None] + log_Pi + log_b[t + 1][None, :] + log_beta[t + 1][None, :] - seq_ll
                xi_ref[t] = np.exp(np.clip(mat, -700, 0))
                s = xi_ref[t].sum()
                if s > 1e-300:
                    xi_ref[t] /= s
            max_xi_diff = max(max_xi_diff, float(np.max(np.abs(xi_ref - e_result["xis"][i]))))

    ll_diff = abs(ref_total_ll - e_result["log_likelihood"])
    check(
        "arhmm_batched_e_step_log_likelihood_matches_reference",
        ll_diff < 1e-6,
        f"batched total_ll={e_result['log_likelihood']:.6f} vs looped reference={ref_total_ll:.6f}, diff={ll_diff:.2e}",
    )
    per_seq_ll_diff = max(abs(e_result["sequence_log_likelihoods"][i] - float(forward_backward(seq, artifact)[2])) for i, seq in enumerate(sequences))
    check(
        "arhmm_batched_e_step_sequence_log_likelihoods_match_reference",
        per_seq_ll_diff < 1e-6,
        f"max abs per-sequence log-likelihood diff across {len(sequences)} mixed-length sequences = {per_seq_ll_diff:.2e}",
    )
    check(
        "arhmm_batched_e_step_gammas_match_reference",
        max_gamma_diff < 1e-8,
        f"max abs gamma diff across {len(sequences)} mixed-length sequences = {max_gamma_diff:.2e}",
    )
    check(
        "arhmm_batched_e_step_xis_match_reference",
        max_xi_diff < 1e-8,
        f"max abs xi diff across {len(sequences)} mixed-length sequences = {max_xi_diff:.2e}",
    )


@register("arhmm_gamma_and_xi_normalize_to_valid_distributions", "arhmm")
def _test_arhmm_gamma_and_xi_normalize_to_valid_distributions():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, e_step, fit

    rng = np.random.default_rng(1)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=6, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=5, seed=1)
    artifact, results = fit(sequences, sequences, config)
    e_result = e_step(sequences, artifact)

    gamma_ok = all(np.allclose(g.sum(axis=1), 1.0, atol=1e-6) and np.all(g >= -1e-9) for g in e_result["gammas"])
    check("arhmm_gamma_rows_sum_to_one_and_nonnegative", gamma_ok)

    xi_ok = True
    for xi in e_result["xis"]:
        for t in range(xi.shape[0]):
            s = xi[t].sum()
            if abs(s - 1.0) > 1e-6:
                xi_ok = False
    check("arhmm_xi_slices_sum_to_one", xi_ok)


@register("arhmm_penalized_objective_nondecreasing", "arhmm")
def _test_arhmm_penalized_objective_nondecreasing():
    """2026-08-03 audit item 2: this is MAP-EM (an explicit sticky-Dirichlet
    + matrix-normal/inverse-Wishart prior enters every M-step), so the
    quantity EM actually guarantees non-decreasing is the PENALIZED
    objective (``log_likelihood + log_prior_density``), not the bare data
    log-likelihood alone -- checking the wrong quantity would be
    overclaiming what this algorithm promises. See
    ``action_conditioned_arhmm.log_prior_density``'s docstring."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, fit

    rng = np.random.default_rng(2)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=15, T=25)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=25, em_tol=1e-6, seed=2)
    artifact, results = fit(sequences, sequences, config)
    history = results[2]["train_objective_history"]
    violations = [
        (i, history[i - 1], history[i])
        for i in range(1, len(history))
        if history[i] < history[i - 1] - 1e-5  # matches fit()'s own EMMonotonicityError tolerance
    ]
    check(
        "arhmm_penalized_objective_nondecreasing_within_tolerance",
        len(violations) == 0,
        f"history={[round(x,3) for x in history]} violations={violations}",
    )


@register("arhmm_monotonicity_holds_across_seeds_and_anisotropic_q", "arhmm")
def _test_arhmm_monotonicity_holds_across_seeds_and_anisotropic_q():
    """2026-08-03 second audit round: the earlier Q-decoupled M-step's W
    formula was only exactly correct when the true noise covariance Q was
    ISOTROPIC (a multiple of the identity) -- which is exactly what the
    single fixture used elsewhere in this file happened to have, so its
    passing monotonicity check proved nothing about the general case. This
    test fits on data with a deliberately ANISOTROPIC (correlated, unequal
    per-axis variance) true Q, across several different random seeds, and
    requires ``fit`` to complete without raising ``EMMonotonicityError``
    for every one of them (fit() itself raises immediately on any real
    violation, so "did not raise" is the pass condition here)."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, EMMonotonicityError, fit

    failures = []
    for seed in (10, 11, 12, 13, 14):
        rng = np.random.default_rng(seed)
        sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=15, T=25, anisotropic_q=True)
        config = ARHMMConfig(k_candidates=(2,), em_max_iters=25, em_tol=1e-6, seed=seed)
        try:
            fit(sequences, sequences, config)
        except EMMonotonicityError as exc:
            failures.append((seed, str(exc)))
    check(
        "arhmm_monotonicity_holds_across_seeds_and_anisotropic_q",
        len(failures) == 0,
        f"seeds tested=[10,11,12,13,14], failures={failures}" if failures else "all 5 seeds converged without any EMMonotonicityError",
    )


@register("arhmm_em_max_iters_exhaustion_still_verifies_final_artifact", "arhmm")
def _test_arhmm_em_max_iters_exhaustion_still_verifies_final_artifact():
    """2026-08-03 fourth audit round: with ``em_max_iters=1`` the training
    loop's convergence break can never fire (there is no second iteration to
    compare against), so it always falls into the exhausted-without-break
    path. Before the fix, the artifact returned in this path was the output
    of the ONE m_step call, whose own objective had never been computed or
    checked against ``EMMonotonicityError`` -- an unverified artifact could
    silently be returned. This test requires: (1) fit() with em_max_iters=1
    does not raise, (2) train_objective_history has length >= 2 (the
    pre-m_step evaluation AND the post-m_step final-artifact evaluation from
    the for/else branch), proving the final artifact really was evaluated."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, fit

    rng = np.random.default_rng(7)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=15, T=25)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=1, em_tol=1e-6, seed=7)
    artifact, results = fit(sequences, sequences, config)
    history = results[2]["train_objective_history"]
    check(
        "arhmm_em_max_iters_1_records_final_artifact_objective",
        len(history) >= 2,
        f"expected >=2 recorded objectives (pre-m_step + post-m_step final-artifact check), got {len(history)}: {history}",
    )
    check(
        "arhmm_em_max_iters_1_final_objective_not_worse_within_tolerance",
        history[-1] >= history[-2] - 1e-5,
        f"final artifact's objective ({history[-1]}) regressed vs prior ({history[-2]}) beyond tolerance",
    )


@register("arhmm_synthetic_identifiability", "arhmm")
def _test_arhmm_synthetic_identifiability():
    """The single most important U2 check: on data with a KNOWN, genuinely
    action-dependent 2-mode switching structure, EM (with NO access to the
    true z) must recover (a) a per-mode B matrix pointing in the correct
    direction/magnitude for each true mode, and (b) a z-sequence matching
    the true one at high accuracy (up to label permutation). If this fails,
    the whole model is not trustworthy regardless of what it finds on real
    data."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, e_step, fit

    rng = np.random.default_rng(0)
    sequences, true_z_all, true_B, true_Pi = _make_synthetic_arhmm_sequences(rng, n_seq=20, T=30)
    config = ARHMMConfig(k_candidates=(1, 2), em_max_iters=40, seed=1)
    artifact, results = fit(sequences, sequences, config)

    check("arhmm_identifiability_k2_selected_over_k1", artifact.K == 2, f"selected K={artifact.K}")
    check(
        "arhmm_identifiability_k2_beats_k1_in_ll",
        results[2]["train_ll_history"][-1] > results[1]["train_ll_history"][-1],
        f"K=1 ll={results[1]['train_ll_history'][-1]:.2f} K=2 ll={results[2]['train_ll_history'][-1]:.2f}",
    )

    e_result = e_step(sequences, results[2]["artifact"])
    correct_direct, correct_swapped, total = 0, 0, 0
    for gamma, true_z in zip(e_result["gammas"], true_z_all):
        pred_z = gamma.argmax(axis=1)
        correct_direct += int((pred_z == true_z).sum())
        correct_swapped += int((pred_z == (1 - true_z)).sum())
        total += len(true_z)
    accuracy = max(correct_direct, correct_swapped) / total
    check(
        "arhmm_identifiability_z_recovery_accuracy_high",
        accuracy > 0.9,
        f"best-permutation z recovery accuracy={accuracy:.4f} (over {total} steps)",
    )

    swapped = correct_swapped > correct_direct
    fitted_B = results[2]["artifact"].B
    b0_match = fitted_B[1 if swapped else 0][0, 0]
    b1_match = fitted_B[0 if swapped else 1][0, 0]
    check(
        "arhmm_identifiability_recovers_correct_b_magnitudes_and_signs",
        abs(b0_match - true_B[0][0, 0]) < 0.05 and abs(b1_match - true_B[1][0, 0]) < 0.1,
        f"recovered B_mode0[0,0]={b0_match:.4f} (true {true_B[0][0,0]}), "
        f"B_mode1[0,0]={b1_match:.4f} (true {true_B[1][0,0]})",
    )


@register("arhmm_q_positive_definite_and_pi_row_normalized", "arhmm")
def _test_arhmm_q_positive_definite_and_pi_row_normalized():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, fit

    rng = np.random.default_rng(3)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=15, T=20)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=20, seed=3)
    artifact, _ = fit(sequences, sequences, config)

    q_ok = all(np.all(np.linalg.eigvalsh(Q) > 0) for Q in artifact.Q)
    check("arhmm_all_q_matrices_positive_definite", q_ok)
    check(
        "arhmm_pi_rows_sum_to_one",
        bool(np.allclose(artifact.Pi.sum(axis=1), 1.0, atol=1e-9)),
        f"Pi row sums={artifact.Pi.sum(axis=1)}",
    )


@register("arhmm_q_mode_formula_valid_at_dof_boundary", "arhmm")
def _test_arhmm_q_mode_formula_valid_at_dof_boundary():
    """2026-08-03 audit items 3+4: the Q_k update uses the inverse-Wishart
    MODE (``(Psi0+S)/(nu0+n_eff+d+1)``), well-defined for any nu0+n_eff>0 --
    but the PRIOR ITSELF is only a proper distribution for
    ``inverse_wishart_dof > d-1 = 1`` (d=2 here), which
    ``ARHMMConfig.__post_init__`` now enforces (fixed from an earlier,
    too-permissive ``>0`` check). This fits right at that boundary (just
    above 1, the smallest legal value) and checks Q is still finite and
    positive-definite; separately confirms the config rejects values at or
    below the boundary."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, ARHMMConfigError, fit

    for bad_dof in (1.0, 0.5, 0.0, -1.0):
        raised = False
        try:
            ARHMMConfig(inverse_wishart_dof=bad_dof)
        except ARHMMConfigError:
            raised = True
        check(f"arhmm_config_rejects_improper_iw_dof_{bad_dof}", raised, f"dof={bad_dof}")

    rng = np.random.default_rng(9)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=10, T=15)
    config = ARHMMConfig(k_candidates=(2,), inverse_wishart_dof=1.01, em_max_iters=10, seed=9)
    artifact, _ = fit(sequences, sequences, config)
    ok = all(np.all(np.isfinite(Q)) and np.all(np.linalg.eigvalsh(Q) > 0) for Q in artifact.Q)
    check("arhmm_q_finite_and_pd_at_dof_just_above_proper_boundary", ok, f"Q={artifact.Q}")


@register("arhmm_sticky_prior_boosts_diagonal", "arhmm")
def _test_arhmm_sticky_prior_boosts_diagonal():
    """A larger sticky_kappa must push Pi's diagonal higher (more mode
    persistence, suppressing spurious rapid switching) -- the legacy
    method had no such regularization at all."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, fit

    rng = np.random.default_rng(4)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=15, T=20)

    config_low = ARHMMConfig(k_candidates=(2,), sticky_kappa=0.0, em_max_iters=15, seed=4)
    artifact_low, _ = fit(sequences, sequences, config_low)
    config_high = ARHMMConfig(k_candidates=(2,), sticky_kappa=50.0, em_max_iters=15, seed=4)
    artifact_high, _ = fit(sequences, sequences, config_high)

    diag_low = np.diag(artifact_low.Pi).mean()
    diag_high = np.diag(artifact_high.Pi).mean()
    check(
        "arhmm_higher_sticky_kappa_yields_higher_diagonal",
        diag_high >= diag_low,
        f"mean diag Pi: sticky_kappa=0 -> {diag_low:.4f}, sticky_kappa=50 -> {diag_high:.4f}",
    )


@register("arhmm_k1_degenerates_to_single_linear_gaussian_fit", "arhmm")
def _test_arhmm_k1_degenerates_to_single_linear_gaussian_fit():
    """2026-08-03 audit item 5: with an explicit matrix-normal shrinkage
    prior in play, K=1 should NOT be required to match a plain
    unregularized OLS fit -- it must match the closed-form MAP solution of
    "the SAME design matrix [v_t, u_t, c_t, 1], under the SAME shrinkage
    prior" (a single-mode Bayesian linear-Gaussian regression), which is
    exactly what the M-step's ridge (``1/shrinkage_scale``) already is.
    This test's "direct" comparison below uses that identical ridge, not
    an unregularized normal-equations solve -- if it used plain OLS
    instead, that would be testing the wrong target."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, fit

    rng = np.random.default_rng(5)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=10, T=20)
    config = ARHMMConfig(k_candidates=(1,), em_max_iters=10, seed=5)
    artifact, _ = fit(sequences, sequences, config)

    xs, ys = [], []
    for seq in sequences:
        L = seq.v_current.shape[0]
        for t in range(L):
            xs.append(np.concatenate([seq.v_current[t], seq.u_robot[t], seq.context[t], [1.0]]))
            ys.append(seq.v_next[t])
    xs, ys = np.array(xs), np.array(ys)
    ridge = 1.0 / config.shrinkage_scale
    W_direct = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ ys).T

    W_fitted = np.concatenate([artifact.A[0], artifact.B[0], artifact.C[0], artifact.d[0][:, None]], axis=1)
    check(
        "arhmm_k1_matches_direct_ridge_regression",
        bool(np.allclose(W_fitted, W_direct, atol=1e-6)),
        f"max abs diff={np.max(np.abs(W_fitted - W_direct)):.2e}",
    )


@register("arhmm_missing_track_frame_splits_sequence", "arhmm")
def _test_arhmm_missing_track_frame_splits_sequence():
    """extract_sequences must split a track at any non-consecutive gap
    (same D3 discipline as mode_model.extract_transitions) -- a single
    skipped frame must produce two shorter sequences, never one sequence
    whose 'transition' silently spans two real dt's."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import extract_sequences

    T, N, dt = 8, 1, 0.25
    humans = np.zeros((T, N, 5))
    track_ids = np.zeros((T, N), dtype=np.int64)
    valid = np.ones((T, N), dtype=bool)
    valid[3, 0] = False  # drop frame t=3
    robot = np.zeros((T, 9))
    robot_actions = np.zeros((T, 2))
    px, py, vx, vy = 0.0, 0.0, 0.5, 0.0
    for t in range(T):
        humans[t, 0] = [px, py, vx, vy, 0.3]
        vx += 0.05
        px += vx * dt
    episode = {
        "humans": humans, "human_track_ids": track_ids, "robot": robot,
        "robot_actions": robot_actions, "valid_mask": valid,
    }
    sequences = extract_sequences([episode], dt=dt)
    lengths = sorted(seq.v_current.shape[0] for seq in sequences)
    check(
        "arhmm_gap_splits_into_two_shorter_sequences",
        lengths == [2, 3],
        f"raw runs t=0..2 (3 frames -> 2 supervised transitions) and t=4..7 "
        f"(4 frames -> 3 supervised transitions), expected L=[2,3], got {lengths}",
    )


@register("arhmm_target_and_action_alignment_exact", "arhmm")
def _test_arhmm_target_and_action_alignment_exact():
    """2026-08-03 audit item 1's explicit permanent-test requirement: for
    EVERY supervised row, ``v_next[i]`` must be the SAME track's velocity
    one raw frame later than ``v_current[i]``, and ``u_robot[i]`` must be
    ``episode['robot_actions']`` at that SAME current frame -- not merely
    "the fields happen to have compatible shapes." Builds an episode with a
    distinctive, monotonically-identifiable velocity/action signature per
    frame (frame t's velocity encodes t itself) so any off-by-one or
    cross-track mixup is impossible to miss."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import extract_sequences

    T, N, dt = 6, 1, 0.25
    humans = np.zeros((T, N, 5))
    track_ids = np.zeros((T, N), dtype=np.int64)
    valid = np.ones((T, N), dtype=bool)
    robot = np.zeros((T, 9))
    robot_actions = np.zeros((T, 2))
    for t in range(T):
        humans[t, 0] = [0.0, 0.0, float(t), float(t) + 0.5, 0.3]  # velocity encodes t: (t, t+0.5)
        robot_actions[t] = [float(t) + 100.0, float(t) + 100.5]   # action encodes t distinctly (+100 offset)
    episode = {
        "humans": humans, "human_track_ids": track_ids, "robot": robot,
        "robot_actions": robot_actions, "valid_mask": valid,
    }
    sequences = extract_sequences([episode], dt=dt)
    check("arhmm_alignment_produces_exactly_one_sequence", len(sequences) == 1, f"got {len(sequences)}")
    seq = sequences[0]
    ok = True
    details = []
    for i in range(seq.v_current.shape[0]):
        expected_v_current = np.array([float(i), float(i) + 0.5])
        expected_v_next = np.array([float(i + 1), float(i + 1) + 0.5])
        expected_action = np.array([float(i) + 100.0, float(i) + 100.5])
        row_ok = (
            np.allclose(seq.v_current[i], expected_v_current)
            and np.allclose(seq.v_next[i], expected_v_next)
            and np.allclose(seq.u_robot[i], expected_action)
        )
        ok = ok and row_ok
        if not row_ok:
            details.append((i, seq.v_current[i], seq.v_next[i], seq.u_robot[i]))
    check(
        "arhmm_every_row_target_is_next_frame_action_is_current_frame",
        ok,
        f"mismatches (row, v_current, v_next, u_robot)={details}" if details else "all rows aligned exactly",
    )


# --------------------------------------------------------------------- #
# group="r1_timing" (Order R1, 2026-08-03) -- guide.md section 10 R1's
# explicit permanent-test requirement: prove the ONLINE rollout's
# per-step context construction (trajectory_sampler._rollout_one, via
# robot_sampler's state/action arrays) reproduces OFFLINE training's
# per-row context (action_conditioned_arhmm._build_sequence) EXACTLY for
# the same real transition, and that the OLD (pre-R1) conflated-sequence
# convention -- pairing POST-action position with a single array serving
# double duty as both state-velocity and action -- provably does NOT,
# whenever the robot actually accelerates between steps. Both tests
# operate directly on ``compute_context_features`` (the single canonical
# formula both the offline and online paths call), constructing inputs
# by hand under each convention -- this isolates exactly the pairing/
# indexing bug R1 fixed, independent of any noise/stochastic rollout
# machinery.
# --------------------------------------------------------------------- #


def _real_robot_trajectory_fixture():
    """A single, self-consistent real robot trajectory recorded the way
    real episode data is: ``robot[t]`` is the robot's ACTUAL state at t
    (position/velocity BEFORE ``robot_actions[t]`` is applied), and
    ``robot_actions[t]`` is the action applied at t that produces
    ``robot[t+1]`` (holonomic: velocity changes instantly to the
    commanded value, position integrates by ``dt*robot_actions[t]`` --
    exactly ``upgrade_u0_*_robot_actions_is_executed_action``'s own
    invariant, and exactly ``robot_sampler.sample_robot_candidates``'s
    per-step convention). Deliberately includes BOTH a nonzero-
    acceleration step (h=1, h=3: action differs from the current
    velocity) and a zero-acceleration step (h=2: action equals the
    current velocity) so the two tests below can tell the two bug
    components (position offset vs. velocity/action conflation) apart."""
    dt = 0.25
    v_r = [np.array([0.3, 0.0])]  # v_r[0]: initial velocity, BEFORE any action
    actions = [np.array([0.5, 0.0]), np.array([0.9, 0.1]), np.array([0.9, 0.1]), np.array([-0.2, 0.4])]
    for a in actions:
        v_r.append(a.copy())  # v_r[h+1] = actions[h] (holonomic instant velocity)
    p_r = [np.array([0.0, 0.0])]
    for h in range(4):
        p_r.append(p_r[h] + dt * actions[h])
    return dt, v_r, p_r, actions  # v_r/p_r have length 5 (t=0..4), actions length 4 (h=0..3)


@register("arhmm_offline_online_context_equivalence", "r1_timing")
def _test_arhmm_offline_online_context_equivalence():
    """guide.md R1's acceptance test: take a real episode's transition data,
    compare offline ``_build_sequence``'s per-row context against the
    online rollout convention's per-step context for the SAME transition
    -- must agree to 1e-12. This is the RED/GREEN reference: this same
    test, run against the pre-R1 conflated ``(robot_position_sequence,
    robot_velocity_sequence)`` construction, would fail (see the
    adversarial companion test below, which demonstrates exactly that
    failure directly)."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import extract_sequences
    from crowd_nav.bayesian_brne.schemas import build_robot_state_row
    from crowd_nav.bayesian_brne.trajectory_sampler import build_rollout_step_inputs

    dt, v_r, p_r, actions = _real_robot_trajectory_fixture()
    T, N = 5, 1
    humans = np.zeros((T, N, 5))
    human_vel = np.array([0.6, -0.2])
    for t in range(T):
        humans[t, 0] = [1.0 + 0.15 * t, 2.0 - 0.05 * t, human_vel[0], human_vel[1], 0.3]
    robot = np.zeros((T, 9))
    for t in range(T):
        robot[t] = build_robot_state_row(
            px=p_r[t][0], py=p_r[t][1], vx=v_r[t][0], vy=v_r[t][1],
            radius=0.3, gx=5.0, gy=5.0, v_pref=1.0, theta=0.0,
        )
    robot_actions = np.zeros((T, 2))
    robot_actions[:4] = actions
    episode = {
        "humans": humans, "human_track_ids": np.zeros((T, N), dtype=np.int64), "robot": robot,
        "robot_actions": robot_actions, "valid_mask": np.ones((T, N), dtype=bool),
    }
    sequences = extract_sequences([episode], dt=dt)
    check("r1_timing_equivalence_produces_one_sequence", len(sequences) == 1, f"got {len(sequences)}")
    seq = sequences[0]

    # ONLINE convention (Order R1-correct): robot_state_position_sequence[h]/
    # robot_state_velocity_sequence[h] are the state BEFORE robot_action_sequence[h]
    # is applied -- exactly p_r[h]/v_r[h] as this fixture defines them.
    max_diff = 0.0
    for h in range(4):
        pedestrian_state = np.array([humans[h, 0, 0], humans[h, 0, 1], humans[h, 0, 2], humans[h, 0, 3]])
        online_v_current, online_action, online_context = build_rollout_step_inputs(
            pedestrian_state, p_r[h], v_r[h], actions[h],
        )
        max_diff = max(max_diff, float(np.max(np.abs(online_context - seq.context[h]))))
        max_diff = max(max_diff, float(np.max(np.abs(online_v_current - seq.v_current[h]))))
        max_diff = max(max_diff, float(np.max(np.abs(online_action - seq.u_robot[h]))))
    check(
        "r1_timing_online_context_matches_offline_to_1e-12",
        max_diff < 1e-12,
        f"max abs diff between online-convention context and offline seq.context = {max_diff:.2e}",
    )


@register("arhmm_rollout_fails_on_old_conflated_sequence_when_accel_nonzero", "r1_timing")
def _test_arhmm_rollout_fails_on_old_conflated_sequence():
    """Adversarial companion to the equivalence test above: reconstructs
    what the PRE-R1 buggy convention would have fed into
    ``compute_context_features`` -- a single ``robot_velocity_sequence``
    array serving DOUBLE DUTY as both the state velocity AND the action
    (``u_r``), paired with ``robot_position_sequence[h]`` being the
    position AFTER that step's action (i.e. ``p_r[h+1]``, not ``p_r[h]``)
    -- and shows it does NOT match offline training's context whenever the
    robot actually accelerates between steps (h=1, h=3 in this fixture;
    h=2 is a zero-acceleration control step included specifically to show
    the position-offset component of the bug alone, distinct from the
    velocity/action conflation component)."""
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import compute_context_features, extract_sequences
    from crowd_nav.bayesian_brne.schemas import build_robot_state_row

    dt, v_r, p_r, actions = _real_robot_trajectory_fixture()
    T, N = 5, 1
    humans = np.zeros((T, N, 5))
    human_vel = np.array([0.6, -0.2])
    for t in range(T):
        humans[t, 0] = [1.0 + 0.15 * t, 2.0 - 0.05 * t, human_vel[0], human_vel[1], 0.3]
    robot = np.zeros((T, 9))
    for t in range(T):
        robot[t] = build_robot_state_row(
            px=p_r[t][0], py=p_r[t][1], vx=v_r[t][0], vy=v_r[t][1],
            radius=0.3, gx=5.0, gy=5.0, v_pref=1.0, theta=0.0,
        )
    robot_actions = np.zeros((T, 2))
    robot_actions[:4] = actions
    episode = {
        "humans": humans, "human_track_ids": np.zeros((T, N), dtype=np.int64), "robot": robot,
        "robot_actions": robot_actions, "valid_mask": np.ones((T, N), dtype=bool),
    }
    sequences = extract_sequences([episode], dt=dt)
    seq = sequences[0]

    # OLD (pre-R1) buggy convention: "robot_position_sequence[h]" was
    # actually the POST-action position (p_r[h+1]), and
    # "robot_velocity_sequence[h]" served as BOTH the state velocity AND
    # u_r -- i.e. it was actually ``actions[h]``, never the true
    # pre-action velocity ``v_r[h]``.
    accel_nonzero_diffs = []
    for h in range(4):
        human_pos = humans[h, 0, :2]
        human_vel_h = humans[h, 0, 2:4]
        old_buggy_position = p_r[h + 1]     # BUG: post-action, should be p_r[h]
        old_buggy_velocity = actions[h]     # BUG: the action itself, should be v_r[h]
        old_context = compute_context_features(human_pos, human_vel_h, old_buggy_position, old_buggy_velocity)
        diff = float(np.max(np.abs(old_context - seq.context[h])))
        accel_nonzero = not np.allclose(v_r[h], actions[h])
        if accel_nonzero:
            accel_nonzero_diffs.append(diff)

    check(
        "r1_timing_accel_nonzero_steps_exist_in_fixture",
        len(accel_nonzero_diffs) >= 1,
        "fixture must include at least one step where the robot actually accelerates for this test to mean anything",
    )
    check(
        "r1_timing_old_conflated_convention_disagrees_with_offline_when_accel_nonzero",
        bool(min(accel_nonzero_diffs) > 1e-3),
        f"per-step diffs at accelerating steps={accel_nonzero_diffs} -- the OLD conflated "
        "(post-action-position, action-as-velocity) convention must measurably disagree with "
        "offline training's context whenever the robot actually accelerates between steps, "
        "proving R1's fix was necessary, not cosmetic",
    )


@register("arhmm_artifact_roundtrip_exact", "arhmm")
def _test_arhmm_artifact_roundtrip_exact():
    import tempfile

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, fit

    rng = np.random.default_rng(6)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=10, seed=6)
    artifact, _ = fit(sequences, sequences, config)

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact

    with tempfile.TemporaryDirectory() as tmp:
        artifact.save(tmp, tier="engineering_only", convergence={"converged": True})
        loaded = ARHMMArtifact.load(tmp)

    ok = (
        loaded.K == artifact.K
        and np.allclose(loaded.Pi, artifact.Pi)
        and np.allclose(loaded.initial_distribution, artifact.initial_distribution)
        and all(np.allclose(loaded.A[k], artifact.A[k]) for k in range(artifact.K))
        and all(np.allclose(loaded.B[k], artifact.B[k]) for k in range(artifact.K))
        and all(np.allclose(loaded.C[k], artifact.C[k]) for k in range(artifact.K))
        and all(np.allclose(loaded.d[k], artifact.d[k]) for k in range(artifact.K))
        and all(np.allclose(loaded.Q[k], artifact.Q[k]) for k in range(artifact.K))
    )
    check("arhmm_artifact_roundtrip_exact", ok)


@register("r3_artifact_provenance_is_required_and_hashed", "arhmm")
def _test_r3_artifact_provenance_is_required_and_hashed():
    """R3: provenance is part of the content hash, not decorative JSON."""
    import json
    import tempfile
    from pathlib import Path

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMConfig, ARHMMArtifact, ArtifactIntegrityError, fit,
    )

    rng = np.random.default_rng(31)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    artifact, _ = fit(sequences, sequences, ARHMMConfig(k_candidates=(2,), em_max_iters=8, seed=31))
    with tempfile.TemporaryDirectory() as tmp:
        artifact.save(tmp, tier="engineering_only", convergence={"converged": True})
        loaded = ARHMMArtifact.load(tmp, expect_tier="engineering_only")
        card_path = Path(tmp) / "model_card.json"
        card = json.loads(card_path.read_text())
        card["artifact_provenance"]["time_semantics"]["action"] = "tampered"
        card_path.write_text(json.dumps(card))
        tampered_rejected = False
        try:
            ARHMMArtifact.load(tmp)
        except ArtifactIntegrityError:
            tampered_rejected = True
    check(
        "r3_artifact_provenance_is_required_and_hashed",
        loaded.model_card["artifact_provenance"]["feature_schema"]["names"]
        and tampered_rejected,
        "feature/time/training/data/source/bounds provenance is stored and metadata tampering breaks the hash",
    )


@register("r3_production_artifact_rejects_placeholder_provenance", "arhmm")
def _test_r3_production_artifact_rejects_placeholder_provenance():
    from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMConfig, ArtifactIntegrityError, fit

    rng = np.random.default_rng(32)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    artifact, _ = fit(sequences, sequences, ARHMMConfig(k_candidates=(2,), em_max_iters=8, seed=32))
    rejected = False
    with __import__("tempfile").TemporaryDirectory() as tmp:
        try:
            artifact.save(tmp, tier="production", convergence={"converged": True})
        except ArtifactIntegrityError:
            rejected = True
    check("r3_production_artifact_rejects_placeholder_provenance", rejected)


@register("r3_policy_file_is_single_config_source", "config")
def _test_r3_policy_file_is_single_config_source():
    from pathlib import Path
    from crowd_nav.bayesian_brne.config import ConfigError, load_policy_config_file

    config_path = Path(__file__).resolve().parents[1] / "configs" / "policy_bayesian_brne.config"
    _model, planner, runtime = load_policy_config_file(str(config_path))
    check(
        "r3_policy_file_is_single_config_source",
        planner.solver_mode == "stable_clearance"
        and planner.sparse_enabled is False
        and planner.pruning_margin == 0.0
        and runtime.artifact_tier == "engineering_only"
        and runtime.brne_root == "/home/abc/temp/brne",
        f"resolved solver={planner.solver_mode} sparse={planner.sparse_enabled} artifact={runtime.artifact_path}",
    )
    bad = config_path.read_text().replace("sparse_enabled = false", "sparse_enabled = true")
    import configparser
    parser = configparser.RawConfigParser()
    parser.read_string(bad)
    rejected = False
    try:
        from crowd_nav.bayesian_brne.config import load_planner_config
        load_planner_config(parser)
    except ConfigError:
        rejected = True
    check("r3_unimplemented_sparse_flag_fails_closed", rejected)


@register("arhmm_artifact_refuses_production_tier_when_unconverged", "arhmm")
def _test_arhmm_artifact_refuses_production_tier_when_unconverged():
    """Order F1 (2026-08-03): ``ARHMMArtifact.save(tier='production', ...)``
    must REFUSE an unconverged fit (e.g. a K=5/6-style fit that only hit
    ``em_max_iters`` without meeting ``em_tol``, like the U3 pilot's
    K=5/K=6) -- this is an enforced constraint, not a documentation note.
    ``tier='engineering_only'`` on the SAME unconverged convergence dict
    must be accepted (interface wiring is allowed on unconverged fits;
    calling it 'production' is not)."""
    import tempfile

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMConfig, ArtifactIntegrityError, fit,
    )

    rng = np.random.default_rng(9)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=10, seed=9)
    artifact, results = fit(sequences, sequences, config)
    convergence = results[2]["convergence"]

    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            artifact.save(tmp, tier="production", convergence={**convergence, "converged": False})
        except ArtifactIntegrityError:
            raised = True
    check(
        "arhmm_artifact_production_tier_refused_when_not_converged",
        raised,
        "save(tier='production', convergence={'converged': False}) must raise ArtifactIntegrityError",
    )

    accepted = True
    with tempfile.TemporaryDirectory() as tmp:
        try:
            artifact.save(tmp, tier="engineering_only", convergence={"converged": False, **convergence})
        except ArtifactIntegrityError:
            accepted = False
    check(
        "arhmm_artifact_engineering_only_tier_accepted_when_not_converged",
        accepted,
        "save(tier='engineering_only', convergence={'converged': False}) must NOT raise",
    )


@register("arhmm_artifact_rejects_invalid_tier", "arhmm")
def _test_arhmm_artifact_rejects_invalid_tier():
    import tempfile

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMConfig, ArtifactIntegrityError, fit,
    )

    rng = np.random.default_rng(10)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=10, seed=10)
    artifact, _ = fit(sequences, sequences, config)

    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            artifact.save(tmp, tier="not_a_real_tier", convergence={"converged": True})
        except ArtifactIntegrityError:
            raised = True
    check("arhmm_artifact_rejects_invalid_tier", raised, "save(tier='not_a_real_tier') must raise ArtifactIntegrityError")


@register("arhmm_artifact_load_fails_closed_on_schema_mismatch", "arhmm")
def _test_arhmm_artifact_load_fails_closed_on_schema_mismatch():
    """Order F1 (2026-08-03): a saved artifact whose ``model_card.json``
    has been hand-edited to a DIFFERENT ``schema_version`` than the code
    currently expects must fail to load, not silently proceed."""
    import json
    import tempfile
    from pathlib import Path

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMConfig, ARHMMArtifact, ArtifactIntegrityError, fit,
    )

    rng = np.random.default_rng(11)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=10, seed=11)
    artifact, _ = fit(sequences, sequences, config)

    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        artifact.save(tmp, tier="engineering_only", convergence={"converged": True})
        card_path = Path(tmp) / "model_card.json"
        card = json.loads(card_path.read_text())
        card["schema_version"] = 999999
        card_path.write_text(json.dumps(card))
        try:
            ARHMMArtifact.load(tmp)
        except ArtifactIntegrityError:
            raised = True
    check(
        "arhmm_artifact_load_fails_closed_on_schema_mismatch",
        raised,
        "load() with a mismatched schema_version must raise ArtifactIntegrityError, not silently proceed",
    )


@register("arhmm_artifact_load_fails_closed_on_content_hash_mismatch", "arhmm")
def _test_arhmm_artifact_load_fails_closed_on_content_hash_mismatch():
    """Order F1 (2026-08-03): if the recorded content_sha256 no longer
    matches the actual arrays (corruption, or someone hand-editing
    arhmm.npz without updating the hash), load() must fail closed."""
    import json
    import tempfile
    from pathlib import Path

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMConfig, ARHMMArtifact, ArtifactIntegrityError, fit,
    )

    rng = np.random.default_rng(12)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=10, seed=12)
    artifact, _ = fit(sequences, sequences, config)

    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        artifact.save(tmp, tier="engineering_only", convergence={"converged": True})
        card_path = Path(tmp) / "model_card.json"
        card = json.loads(card_path.read_text())
        card["content_sha256"] = "0" * 64
        card_path.write_text(json.dumps(card))
        try:
            ARHMMArtifact.load(tmp)
        except ArtifactIntegrityError:
            raised = True
    check(
        "arhmm_artifact_load_fails_closed_on_content_hash_mismatch",
        raised,
        "load() with a mismatched content_sha256 must raise ArtifactIntegrityError, not silently proceed",
    )


@register("arhmm_artifact_load_expect_tier_fails_closed", "arhmm")
def _test_arhmm_artifact_load_expect_tier_fails_closed():
    """Order F1 (2026-08-03): a caller that requires
    ``load(expect_tier='production')`` must never silently receive an
    ``engineering_only`` artifact instead."""
    import tempfile

    from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
        ARHMMConfig, ARHMMArtifact, ArtifactIntegrityError, fit,
    )

    rng = np.random.default_rng(13)
    sequences, _, _, _ = _make_synthetic_arhmm_sequences(rng, n_seq=8, T=15)
    config = ARHMMConfig(k_candidates=(2,), em_max_iters=10, seed=13)
    artifact, _ = fit(sequences, sequences, config)

    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        artifact.save(tmp, tier="engineering_only", convergence={"converged": True})
        try:
            ARHMMArtifact.load(tmp, expect_tier="production")
        except ArtifactIntegrityError:
            raised = True
    check(
        "arhmm_artifact_load_expect_tier_fails_closed",
        raised,
        "load(expect_tier='production') on an engineering_only artifact must raise ArtifactIntegrityError",
    )


# --------------------------------------------------------------------- #
# group="s1" (Order S1-0, 2026-08-04) -- guide.md section 13's frozen S1
# protocol. S1-0 scope: registry parsing/validation, existing-episode
# identity scanning/seed-collision detection, source-hash drift detection,
# and rollback-archive restore verification must all fail closed. These
# tests exercise crowd_nav.bayesian_brne.s1_protocol directly -- never a
# reimplementation of its logic (guide.md: "统计逻辑不得散落在CLI脚本里").
# --------------------------------------------------------------------- #

# Order S1-0R (R0R-6): resolved from this file's own location, exactly
# like ``s1_protocol.repo_root()``, so this test module works regardless
# of the shell's current working directory when selftest.py is invoked.
_S1_REPO_ROOT = Path(__file__).resolve().parents[2]
_S1_REGISTRY_PATH = str(_S1_REPO_ROOT / "crowd_nav" / "configs" / "s1_strict_registry.json")


def _load_real_s1_registry_dict():
    import json

    with open(_S1_REGISTRY_PATH) as f:
        return json.load(f)


@register("s1_registry_loads_and_validates", "s1")
def _test_s1_registry_loads_and_validates():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = sp.load_registry(_S1_REGISTRY_PATH)
    check("s1_registry_schema_version", registry["schema_version"] == 1, str(registry["schema_version"]))
    check("s1_registry_controllers_exact", tuple(registry["controllers"]) == sp.REQUIRED_CONTROLLERS, str(registry["controllers"]))
    check(
        "s1_registry_five_data_roles_present",
        all(role in registry for role in ("train", "selection", "necessity_id", "audit_interactive", "audit_nominal_negative_control")),
    )
    check(
        "s1_registry_necessity_audit_seeds_are_31_through_60",
        set(registry["necessity_id"]["suite_seeds"]) | set(registry["audit_interactive"]["suite_seeds"]) | set(registry["audit_nominal_negative_control"]["suite_seeds"])
        == set(range(31, 61)),
        f"got {sorted(set(registry['necessity_id']['suite_seeds']) | set(registry['audit_interactive']['suite_seeds']) | set(registry['audit_nominal_negative_control']['suite_seeds']))}",
    )


@register("s1_registry_missing_field_fails_closed", "s1")
def _test_s1_registry_missing_field_fails_closed():
    import json
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = _load_real_s1_registry_dict()
    del registry["em_tol"]
    raised = False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(registry, f)
        path = f.name
    try:
        sp.load_registry(path)
    except sp.RegistryError:
        raised = True
    check("s1_registry_missing_field_fails_closed", raised, "a registry missing a required field must raise RegistryError")


@register("s1_registry_cross_role_seed_overlap_fails_closed", "s1")
def _test_s1_registry_cross_role_seed_overlap_fails_closed():
    import json
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = _load_real_s1_registry_dict()
    # Corrupt necessity_id to share a suite_seed with audit_interactive.
    registry["necessity_id"]["suite_seeds"] = list(registry["necessity_id"]["suite_seeds"])
    registry["necessity_id"]["suite_seeds"][0] = registry["audit_interactive"]["suite_seeds"][0]
    raised = False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(registry, f)
        path = f.name
    try:
        sp.load_registry(path)
    except sp.RegistryError:
        raised = True
    check(
        "s1_registry_cross_role_seed_overlap_fails_closed", raised,
        "necessity_id and audit_interactive sharing a suite_seed must raise RegistryError",
    )


@register("s1_registry_boundary_k_overlap_fails_closed", "s1")
def _test_s1_registry_boundary_k_overlap_fails_closed():
    import json
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = _load_real_s1_registry_dict()
    registry["boundary_extension_k_candidates"] = [4, 5]  # 4 already in primary_k_candidates
    raised = False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(registry, f)
        path = f.name
    try:
        sp.load_registry(path)
    except sp.RegistryError:
        raised = True
    check(
        "s1_registry_boundary_k_overlap_fails_closed", raised,
        "boundary_extension_k_candidates overlapping primary_k_candidates must raise RegistryError",
    )


@register("s1_registry_content_hash_stable_across_loads", "s1")
def _test_s1_registry_content_hash_stable():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    a = sp.load_registry(_S1_REGISTRY_PATH)
    b = sp.load_registry(_S1_REGISTRY_PATH)
    check(
        "s1_registry_content_hash_stable_across_loads",
        sp.registry_content_sha256(a) == sp.registry_content_sha256(b),
    )


def _write_synthetic_episode_npz(path, suite_seed, episode_seed, initial_state_hash="a" * 64):
    import numpy as _np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _np.savez(
        str(path),
        suite_seed=_np.array(suite_seed), episode_seed=_np.array(episode_seed),
        initial_state_hash=_np.array(initial_state_hash),
    )


@register("s1_seed_scan_detects_collision_with_reserved_seed", "s1")
def _test_s1_seed_scan_detects_collision():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = sp.load_registry(_S1_REGISTRY_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        # suite_seed=31 collides with necessity_id's reserved seeds.
        _write_synthetic_episode_npz(Path(tmp) / "fake_split" / "ep00000_seed3100000.npz", 31, 3100000)
        scan = sp.scan_existing_episode_identities([tmp])
        check("s1_seed_scan_finds_synthetic_episode", 31 in scan.episode_seeds_by_suite_seed)
        raised = False
        try:
            sp.check_seed_disjoint(registry, scan)
        except sp.SeedCollisionError:
            raised = True
        check(
            "s1_seed_scan_detects_collision_with_reserved_seed", raised,
            "an existing episode using suite_seed=31 (reserved for necessity_id) must raise SeedCollisionError",
        )


@register("s1_seed_scan_no_collision_on_disjoint_seeds", "s1")
def _test_s1_seed_scan_no_collision_on_disjoint_seeds():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = sp.load_registry(_S1_REGISTRY_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        # suite_seed=11 matches an EXISTING train seed, not a reserved one.
        _write_synthetic_episode_npz(Path(tmp) / "fake_split" / "ep00000_seed1100000.npz", 11, 1100000)
        scan = sp.scan_existing_episode_identities([tmp])
        report = sp.check_seed_disjoint(registry, scan)
        check("s1_seed_scan_no_collision_on_disjoint_seeds", report["collision"] is False, str(report))


@register("s1_seed_scan_skips_non_episode_npz", "s1")
def _test_s1_seed_scan_skips_non_episode_npz():
    """Order S1-0R-A (A3): a non-episode npz is ONLY safely skipped when it
    positively matches a known-good artifact key signature (e.g. an
    ARHMMArtifact's K/Pi/initial_distribution/dt) -- an arbitrary npz that
    merely lacks suite_seed/episode_seed, with no recognized signature, is
    NOT assumed harmless; it must raise instead."""
    import tempfile

    import numpy as _np

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        # A recognized non-episode artifact signature (ARHMMArtifact-shaped).
        _np.savez(
            str(Path(tmp) / "arhmm.npz"),
            K=_np.array(1), Pi=_np.eye(1), initial_distribution=_np.ones(1), dt=_np.array(0.25),
        )
        _write_synthetic_episode_npz(Path(tmp) / "ep00000_seed1100000.npz", 11, 1100000)
        scan = sp.scan_existing_episode_identities([tmp])
        check("s1_seed_scan_scanned_both_files", scan.n_files_scanned == 2, str(scan.n_files_scanned))
        check("s1_seed_scan_matched_only_the_real_episode", scan.n_files_matched == 1, str(scan.n_files_matched))
        check("s1_seed_scan_skipped_the_known_artifact_signature", scan.n_files_skipped_non_episode == 1, str(scan.n_files_skipped_non_episode))


@register("s1_seed_scan_rejects_unrecognized_non_episode_npz", "s1")
def _test_s1_seed_scan_rejects_unrecognized_non_episode_npz():
    """The companion red case: an npz with NEITHER an episode identity NOR
    a recognized artifact signature must raise, never be silently skipped
    just because it is "missing a field" -- the original A3 bug generalized
    this into "any missing field is safe to skip", which is exactly wrong."""
    import tempfile

    import numpy as _np

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _np.savez(str(Path(tmp) / "not_an_episode.npz"), some_array=_np.zeros(4))
        raised = False
        try:
            sp.scan_existing_episode_identities([tmp])
        except sp.PreflightError:
            raised = True
        check("s1_seed_scan_rejects_unrecognized_non_episode_npz", raised)


@register("s1_source_manifest_change_detection_fails_closed", "s1")
def _test_s1_source_manifest_change_detection():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        tracked_rel = "tracked_file.py"
        tracked_path = Path(tmp) / tracked_rel
        tracked_path.write_text("x = 1\n")
        # build_source_manifest ALSO tracks the real module-level
        # SOURCE_MANIFEST_FILES, which do not exist under this synthetic
        # temp repo -- only assert on our own tracked_rel file, not the
        # full missing_files list.
        manifest = sp.build_source_manifest(tmp, extra_files=[tracked_rel])
        check("s1_source_manifest_no_missing_files_initially", tracked_rel not in manifest["missing_files"])

        # Unchanged: must NOT raise.
        raised_unchanged = False
        try:
            sp.verify_source_manifest_unchanged(tmp, manifest)
        except sp.PreflightError:
            raised_unchanged = True
        check("s1_source_manifest_unchanged_does_not_raise", not raised_unchanged)

        # Mutate the tracked file's content -- must raise.
        tracked_path.write_text("x = 2\n")
        raised_changed = False
        try:
            sp.verify_source_manifest_unchanged(tmp, manifest)
        except sp.PreflightError:
            raised_changed = True
        check(
            "s1_source_manifest_change_detection_fails_closed", raised_changed,
            "a tracked file changed after freeze must raise PreflightError",
        )


@register("s1_rollback_archive_restores_and_detects_tamper", "s1")
def _test_s1_rollback_archive_restores_and_detects_tamper():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as out_tmp:
        rel = "tiny_source_file.py"
        (Path(repo_tmp) / rel).write_text("y = 42\n")
        manifest = sp.build_rollback_archive(repo_tmp, out_tmp, file_list=[rel])
        check("s1_rollback_archive_records_one_file", manifest["n_files"] == 1, str(manifest["n_files"]))
        check("s1_rollback_archive_restores_correctly", sp.verify_rollback_archive(repo_tmp, manifest) is True)

        tampered = dict(manifest)
        tampered["per_file_sha256"] = dict(manifest["per_file_sha256"])
        tampered["per_file_sha256"][rel] = "0" * 64
        check(
            "s1_rollback_archive_detects_tampered_manifest_hash",
            sp.verify_rollback_archive(repo_tmp, tampered) is False,
        )


# --------------------------------------------------------------------- #
# group="s1" continued (Order S1-0R, 2026-08-04) -- independent audit
# ("CONDITIONAL PASS / FORMAL RUN BLOCKED") found 5 real gaps in S1-0:
# hardcoded host paths (R0R-1), gate/queue clobbering one status.json
# (R0R-2), preflight not strictly verifying train/selection (R0R-3), a
# source-freeze lifecycle contradiction (R0R-4), and rollback verification
# never checking the tar's own SHA256 (R0R-5). These tests cover the fixes.
# --------------------------------------------------------------------- #


@register("s1_repo_root_resolves_to_real_checkout", "s1")
def _test_s1_repo_root_resolves_to_real_checkout():
    """R0R-1: repo_root() must be derived from this file's own location,
    never a literal host path -- verified by confirming it points at a
    real, present file this checkout is known to have."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    root = sp.repo_root()
    check("s1_repo_root_is_a_directory", root.is_dir(), str(root))
    check(
        "s1_repo_root_contains_known_package_file",
        (root / "crowd_nav" / "bayesian_brne" / "__init__.py").exists(),
        f"root={root}",
    )
    check("s1_repo_root_matches_selftest_own_computation", root == _S1_REPO_ROOT, f"{root} vs {_S1_REPO_ROOT}")


@register("s1_cli_files_contain_no_hardcoded_host_path", "s1")
def _test_s1_cli_files_contain_no_hardcoded_host_path():
    """R0R-1: static assertion that none of the S1 CLI/protocol files
    contain a literal developer-machine absolute path -- the exact class
    of bug that would silently point the 4090 run at the wrong
    (nonexistent) directory tree. Covers s1_protocol.py itself too, not
    just the two CLIs -- a first pass at this test only checked the CLIs
    and missed that repo_root()'s own docstring in s1_protocol.py still
    quoted the old hardcoded paths as prose."""
    forbidden_substrings = ("/home/abc", "/root/workspace")
    for rel in (
        "crowd_nav/tools/s1_strict_gate.py", "crowd_nav/tools/run_s1_queue.py",
        "crowd_nav/bayesian_brne/s1_protocol.py",
    ):
        text = (_S1_REPO_ROOT / rel).read_text()
        hits = [s for s in forbidden_substrings if s in text]
        check(f"s1_cli_no_hardcoded_path_in_{Path(rel).name}", not hits, f"found forbidden substrings {hits} in {rel}")


@register("s1_status_atomic_merges_across_gate_then_queue_order", "s1")
def _test_s1_status_atomic_merges_gate_then_queue():
    """R0R-2: a gate-style write (stage_status) followed by a queue-style
    write (queue_status) must NOT lose the first write's fields -- the
    exact clobbering bug the independent audit found. v2 (A4): stage and
    queue identity are fully separate fields (``stage_pid``/``queue_pid``),
    never one shared ``pid``."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        sp.update_status_atomic(
            tmp, experiment_id="exp1", registry_sha256="abc", invocation_mode="direct",
            stage_pid=111, current_stage="preflight", stage_status="PASS", returncode=0,
        )
        sp.update_status_atomic(tmp, experiment_id="exp1", registry_sha256="abc", queue_pid=999, queue_status="RUNNING", current_stage="collect_necessity")
        status = sp.read_status_or_default(tmp)
        check("s1_status_gate_field_survives_queue_write", status["stage_status"] == "PASS", str(status))
        check("s1_status_queue_field_present_after_queue_write", status["queue_status"] == "RUNNING", str(status))
        check("s1_status_current_stage_updated_by_queue_write", status["current_stage"] == "collect_necessity", str(status))
        check("s1_status_stage_pid_survives_queue_write", status["stage_pid"] == 111, str(status))
        check("s1_status_queue_pid_independent_of_stage_pid", status["queue_pid"] == 999, str(status))


@register("s1_status_atomic_merges_across_queue_then_gate_order", "s1")
def _test_s1_status_atomic_merges_queue_then_gate():
    """R0R-2: the reverse ordering (queue writes first, then gate) must
    also merge, not clobber."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        sp.update_status_atomic(tmp, experiment_id="exp1", registry_sha256="abc", queue_pid=222, queue_status="RUNNING", current_stage="preflight")
        sp.update_status_atomic(tmp, experiment_id="exp1", registry_sha256="abc", stage_pid=333, stage_status="PASS", returncode=0)
        status = sp.read_status_or_default(tmp)
        check("s1_status_queue_field_survives_gate_write", status["queue_status"] == "RUNNING", str(status))
        check("s1_status_gate_field_present_after_gate_write", status["stage_status"] == "PASS", str(status))
        check("s1_status_queue_pid_survives_gate_write", status["queue_pid"] == 222, str(status))
        check("s1_status_stage_pid_set_by_gate_write", status["stage_pid"] == 333, str(status))


@register("s1_status_atomic_rejects_corrupt_json", "s1")
def _test_s1_status_atomic_rejects_corrupt_json():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "status.json").write_text("{not valid json")
        raised = False
        try:
            sp.read_status_or_default(tmp)
        except sp.StatusError:
            raised = True
        check("s1_status_atomic_rejects_corrupt_json", raised, "a corrupt status.json must raise StatusError, never be treated as absent")


@register("s1_status_atomic_rejects_v1_schema", "s1")
def _test_s1_status_atomic_rejects_v1_schema():
    """A4: a pre-fix v1 status.json (single ambiguous 'pid' field) must be
    rejected outright, never silently migrated -- it describes a run whose
    queue/stage identity was ambiguous by construction."""
    import json
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "status.json").write_text(json.dumps({"schema_version": 1, "pid": 42, "stage_status": "PASS"}))
        raised = False
        try:
            sp.update_status_atomic(tmp, experiment_id="exp1", registry_sha256="abc", stage_status="PASS")
        except sp.StatusError:
            raised = True
        check("s1_status_atomic_rejects_v1_schema", raised)


@register("s1_status_atomic_rejects_identity_mismatch", "s1")
def _test_s1_status_atomic_rejects_identity_mismatch():
    """R0R-2: two different experiment_id/registry_sha256 values must never
    silently share one status.json."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        sp.update_status_atomic(tmp, experiment_id="exp1", registry_sha256="abc", stage_pid=1)
        raised_exp = False
        try:
            sp.update_status_atomic(tmp, experiment_id="exp2", registry_sha256="abc", stage_pid=2)
        except sp.StatusError:
            raised_exp = True
        check("s1_status_atomic_rejects_experiment_id_mismatch", raised_exp)

        raised_hash = False
        try:
            sp.update_status_atomic(tmp, experiment_id="exp1", registry_sha256="different", stage_pid=3)
        except sp.StatusError:
            raised_hash = True
        check("s1_status_atomic_rejects_registry_hash_mismatch", raised_hash)


@register("s1_status_recovery_pid_alive_check_rejects_reused_pid", "s1")
def _test_s1_status_recovery_pid_alive_check():
    """A4: '仅PID数字相同不算同一任务' -- a matching PID number that belongs
    to a real, currently-running process, but whose cmdline does NOT
    mention this invocation's registry path, must NOT be treated as this
    run's own live stage process (a fake-alive-PID/wrong-cmdline case)."""
    import os as _os

    from crowd_nav.bayesian_brne import s1_protocol as sp

    real_pid = _os.getpid()  # genuinely alive right now, but not an s1_strict_gate invocation
    status_fake = {"stage_pid": real_pid}
    check(
        "s1_status_recovery_rejects_alive_pid_with_wrong_cmdline",
        sp.is_recorded_stage_process_alive(status_fake, "crowd_nav/configs/s1_strict_registry.json") is False,
    )
    check(
        "s1_status_recovery_rejects_missing_pid",
        sp.is_recorded_stage_process_alive({"stage_pid": None}, "crowd_nav/configs/s1_strict_registry.json") is False,
    )
    # An almost-certainly-unused PID number -- must not raise, just report not-alive.
    check(
        "s1_status_recovery_handles_nonexistent_pid_gracefully",
        sp.is_recorded_stage_process_alive({"stage_pid": 2**30 - 1}, "anything") is False,
    )


@register("s1_status_recovery_queue_then_direct_preflight_distinguishable", "s1")
def _test_s1_status_recovery_queue_then_direct():
    """A4 scenario 1: a completed queue run followed by a later DIRECT
    preflight invocation must leave both executions' identity
    reconstructible -- queue_status/queue_started_at from the queue run,
    stage_status/invocation_mode='direct' from the later direct call,
    never merged into one ambiguous record."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sp.update_status_atomic(
            tmp, experiment_id="exp1", registry_sha256="abc", queue_pid=100, queue_status="COMPLETED_REQUESTED_STAGES",
            queue_started_at="2026-08-04 10:00:00", current_stage="preflight", invocation_mode="queue",
            stage_pid=None, stage_status="PASS", stage_started_at="2026-08-04 10:00:01", stage_finished_at="2026-08-04 10:00:02",
        )
        sp.update_status_atomic(
            tmp, experiment_id="exp1", registry_sha256="abc", invocation_mode="direct",
            stage_pid=None, current_stage="preflight", stage_status="PASS",
            stage_started_at="2026-08-04 11:00:00", stage_finished_at="2026-08-04 11:00:01",
        )
        status = sp.read_status_or_default(tmp)
        check("s1_status_recovery_queue_then_direct_keeps_queue_status", status["queue_status"] == "COMPLETED_REQUESTED_STAGES", str(status))
        check("s1_status_recovery_queue_then_direct_keeps_queue_started_at", status["queue_started_at"] == "2026-08-04 10:00:00", str(status))
        check("s1_status_recovery_queue_then_direct_updates_invocation_mode", status["invocation_mode"] == "direct", str(status))
        check("s1_status_recovery_queue_then_direct_updates_stage_started_at", status["stage_started_at"] == "2026-08-04 11:00:00", str(status))


@register("s1_status_recovery_direct_then_queue_distinguishable", "s1")
def _test_s1_status_recovery_direct_then_queue():
    """A4 scenario 2: the reverse ordering -- a direct preflight run
    followed by a queue invocation -- must also keep both distinguishable."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sp.update_status_atomic(
            tmp, experiment_id="exp1", registry_sha256="abc", invocation_mode="direct",
            stage_pid=None, current_stage="preflight", stage_status="PASS",
            stage_started_at="2026-08-04 09:00:00", stage_finished_at="2026-08-04 09:00:01",
        )
        sp.update_status_atomic(
            tmp, experiment_id="exp1", registry_sha256="abc", queue_pid=200, queue_status="RUNNING",
            queue_started_at="2026-08-04 12:00:00", current_stage="preflight",
        )
        status = sp.read_status_or_default(tmp)
        check("s1_status_recovery_direct_then_queue_keeps_prior_stage_result", status["stage_status"] == "PASS", str(status))
        check("s1_status_recovery_direct_then_queue_sets_queue_fields", status["queue_status"] == "RUNNING" and status["queue_pid"] == 200, str(status))


@register("s1_status_recovery_interrupted_stage_leaves_running_with_no_finish_time", "s1")
def _test_s1_status_recovery_interrupted_stage():
    """A4 scenario 4: a stage that started but never reached a final write
    (simulating a kill -9 mid-stage) must be recoverable-as-ambiguous: it
    shows stage_status='RUNNING' with a stage_pid but no stage_finished_at
    -- a recovery process can then apply is_recorded_stage_process_alive
    to decide whether to treat it as orphaned."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sp.update_status_atomic(
            tmp, experiment_id="exp1", registry_sha256="abc", invocation_mode="direct",
            stage_pid=12345, current_stage="preflight", stage_status="RUNNING",
            stage_started_at="2026-08-04 13:00:00", stage_finished_at=None,
        )
        status = sp.read_status_or_default(tmp)
        check("s1_status_recovery_interrupted_stage_status_is_running", status["stage_status"] == "RUNNING", str(status))
        check("s1_status_recovery_interrupted_stage_has_no_finish_time", status["stage_finished_at"] is None, str(status))
        check(
            "s1_status_recovery_interrupted_stage_pid_correctly_reported_not_alive",
            sp.is_recorded_stage_process_alive(status, "some/registry/path.json") is False,
            "stage_pid=12345 almost certainly does not belong to a live s1_strict_gate process in this test environment",
        )


def _build_tiny_formal_role_dir(tmp_dir, split, suite_seed, n_episodes, horizon_steps=6, scenario="baseline_circle", controller_fn=None):
    """Build ``n_episodes`` real, fully schema-valid episodes (via the same
    ``interaction_protocol``/``data_io`` path real formal data collection
    uses, not a hand-rolled fixture) under ``tmp_dir``, for exercising
    ``verify_formal_data_role``'s red/green cases without touching the real
    2000/500-episode ``data_formal`` directory. ``controller_fn(ep) ->
    str``, if given, REPLACES the normal round-robin ``allocate_controller_type``
    -- used only to construct deliberately unbalanced/single-controller
    fixtures for the controller-balance red-case tests."""
    from crowd_nav.bayesian_brne import data_io
    from crowd_nav.bayesian_brne.interaction_protocol import allocate_controller_type, make_scenario, run_episode
    from crowd_nav.bayesian_brne.schemas import ROBOT_STATE_FIELDS

    controller_fn = controller_fn or allocate_controller_type
    paths = []
    for ep in range(n_episodes):
        episode_seed = suite_seed * 100000 + ep
        rng = np.random.default_rng(episode_seed)
        env = make_scenario(scenario, split, rng, dt=0.25)
        controller = controller_fn(ep)
        episode = run_episode(
            env, horizon_steps=horizon_steps, controller=controller, scenario=scenario,
            profile="formal", split=split, profile_name="formal",
        )
        episode.update(suite_seed=suite_seed, episode_seed=episode_seed)
        path = Path(tmp_dir) / f"ep{ep:05d}_seed{episode_seed}.npz"
        data_io.save_episode(str(path), episode, source_robot_state_layout=ROBOT_STATE_FIELDS)
        paths.append(path)
    return paths


def _tiny_registry_for_role(role_name, path, suite_seeds, episodes_per_seed, horizon_steps=6):
    registry = _load_real_s1_registry_dict()
    registry[role_name] = dict(registry[role_name])
    registry[role_name]["path"] = str(path)
    registry[role_name]["suite_seeds"] = list(suite_seeds)
    registry[role_name]["episodes_per_seed"] = episodes_per_seed
    registry["horizon_steps"] = horizon_steps
    return registry


@register("s1_data_role_accepts_exact_match", "s1")
def _test_s1_data_role_accepts_exact_match():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        # n_episodes=4 -- one full round-robin cycle over the 4 controllers
        # (allocate_controller_type is a pure episode_index % 4), so this
        # happy-path fixture is naturally controller-balanced too.
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=901, n_episodes=4, horizon_steps=6)
        registry = _tiny_registry_for_role("train", tmp, [901], 4, horizon_steps=6)
        report = sp.verify_formal_data_role("", "train", registry)
        check("s1_data_role_accepts_exact_match", report["seed_counts"] == {"901": 4}, str(report["seed_counts"]))
        check(
            "s1_data_role_accepts_exact_match_controller_balanced",
            report["controller_counts"] == {c: 1 for c in registry["controllers"]},
            str(report["controller_counts"]),
        )


@register("s1_data_role_rejects_missing_episode", "s1")
def _test_s1_data_role_rejects_missing_episode():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=902, n_episodes=2, horizon_steps=6)
        registry = _tiny_registry_for_role("train", tmp, [902], 3, horizon_steps=6)  # registry claims 3, only 2 exist
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_missing_episode", raised, "1-short episode count must raise DataRoleIntegrityError")


@register("s1_data_role_rejects_extra_episode", "s1")
def _test_s1_data_role_rejects_extra_episode():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=903, n_episodes=4, horizon_steps=6)
        registry = _tiny_registry_for_role("train", tmp, [903], 3, horizon_steps=6)  # registry claims 3, 4 exist
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_extra_episode", raised, "1-over episode count must raise DataRoleIntegrityError")


@register("s1_data_role_rejects_extra_seed", "s1")
def _test_s1_data_role_rejects_extra_seed():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=904, n_episodes=2, horizon_steps=6)
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=905, n_episodes=2, horizon_steps=6)  # not declared below
        registry = _tiny_registry_for_role("train", tmp, [904], 2, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_extra_seed", raised, "an undeclared suite_seed present on disk must raise DataRoleIntegrityError")


@register("s1_data_role_rejects_duplicate_episode_id", "s1")
def _test_s1_data_role_rejects_duplicate_episode_id():
    import shutil
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        paths = _build_tiny_formal_role_dir(tmp, "train", suite_seed=906, n_episodes=2, horizon_steps=6)
        # Duplicate an existing file verbatim under a new filename -- same
        # (suite_seed, episode_seed) content, different path on disk.
        shutil.copy(str(paths[0]), str(Path(tmp) / "duplicate_copy.npz"))
        registry = _tiny_registry_for_role("train", tmp, [906], 2, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_duplicate_episode_id", raised, "a duplicated (suite_seed, episode_seed) must raise DataRoleIntegrityError")


@register("s1_data_role_rejects_duplicate_initial_state_hash", "s1")
def _test_s1_data_role_rejects_duplicate_initial_state_hash():
    """A different episode_seed carrying the SAME initial_state_hash (e.g.
    an accidental re-run overwriting metadata but not physical state) must
    be rejected -- this is a DIFFERENT failure mode than a duplicate
    (suite_seed, episode_seed), which is covered separately above."""
    import tempfile

    from crowd_nav.bayesian_brne import data_io, s1_protocol as sp
    from crowd_nav.bayesian_brne.schemas import ROBOT_STATE_FIELDS

    with tempfile.TemporaryDirectory() as tmp:
        paths = _build_tiny_formal_role_dir(tmp, "train", suite_seed=907, n_episodes=2, horizon_steps=6)
        episode = data_io.load_episode(str(paths[0]))
        episode["episode_seed"] = 90700099  # new episode_seed, SAME initial_state_hash (untouched)
        data_io.save_episode(str(Path(tmp) / "cloned_hash.npz"), episode, source_robot_state_layout=ROBOT_STATE_FIELDS)
        registry = _tiny_registry_for_role("train", tmp, [907], 3, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check(
            "s1_data_role_rejects_duplicate_initial_state_hash", raised,
            "a repeated initial_state_hash under a new episode_seed must raise DataRoleIntegrityError",
        )


@register("s1_data_role_rejects_wrong_split", "s1")
def _test_s1_data_role_rejects_wrong_split():
    """An episode collected under split='validation' living inside the
    'train' role's directory must be rejected, not silently accepted just
    because the file COUNT happens to match."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "validation", suite_seed=908, n_episodes=2, horizon_steps=6)  # wrong split for "train" role
        registry = _tiny_registry_for_role("train", tmp, [908], 2, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_wrong_split", raised, "split='validation' data under the 'train' role must raise DataRoleIntegrityError")


@register("s1_data_role_rejects_wrong_scenario_expectation", "s1")
def _test_s1_data_role_rejects_wrong_scenario_expectation():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=909, n_episodes=4, horizon_steps=6, scenario="baseline_circle")
        registry = _tiny_registry_for_role("train", tmp, [909], 4, horizon_steps=6)
        registry["scenario"] = "baseline_square"  # real episodes are baseline_circle
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_wrong_scenario_expectation", raised)


@register("s1_data_role_rejects_wrong_dt_expectation", "s1")
def _test_s1_data_role_rejects_wrong_dt_expectation():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=910, n_episodes=4, horizon_steps=6)
        registry = _tiny_registry_for_role("train", tmp, [910], 4, horizon_steps=6)
        registry["dt"] = 0.5  # real episodes were built with dt=0.25
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_wrong_dt_expectation", raised)


@register("s1_data_role_rejects_wrong_horizon_expectation", "s1")
def _test_s1_data_role_rejects_wrong_horizon_expectation():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=911, n_episodes=4, horizon_steps=6)
        registry = _tiny_registry_for_role("train", tmp, [911], 4, horizon_steps=8)  # episodes are 6 steps, registry claims 8
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_wrong_horizon_expectation", raised)


@register("s1_data_role_rejects_wrong_n_humans_expectation", "s1")
def _test_s1_data_role_rejects_wrong_n_humans_expectation():
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        _build_tiny_formal_role_dir(tmp, "train", suite_seed=912, n_episodes=4, horizon_steps=6, scenario="baseline_circle")  # 5 humans
        registry = _tiny_registry_for_role("train", tmp, [912], 4, horizon_steps=6)
        registry["n_humans"] = 10  # baseline_circle is actually 5 humans
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_wrong_n_humans_expectation", raised)


@register("s1_data_role_rejects_unknown_controller_type", "s1")
def _test_s1_data_role_rejects_unknown_controller_type():
    """A controller_type outside the registry's frozen list must be
    rejected -- constructed by mutating a real, otherwise-valid episode's
    metadata (real generation code only ever emits the 4 real controllers,
    so this red case cannot arise from normal generation)."""
    import tempfile

    from crowd_nav.bayesian_brne import data_io, s1_protocol as sp
    from crowd_nav.bayesian_brne.schemas import ROBOT_STATE_FIELDS

    with tempfile.TemporaryDirectory() as tmp:
        paths = _build_tiny_formal_role_dir(tmp, "train", suite_seed=913, n_episodes=4, horizon_steps=6)
        episode = data_io.load_episode(str(paths[0]))
        episode["controller_type"] = "not_a_real_controller"
        data_io.save_episode(str(paths[0]), episode, source_robot_state_layout=ROBOT_STATE_FIELDS)
        registry = _tiny_registry_for_role("train", tmp, [913], 4, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_unknown_controller_type", raised)


@register("s1_data_role_rejects_controller_imbalance", "s1")
def _test_s1_data_role_rejects_controller_imbalance():
    """guide.md Order 6.1's equal-allocation guarantee means the frozen
    formal dataset must have EXACTLY total/4 episodes per controller; a
    skewed distribution (even with the right total count) must be
    rejected, not silently accepted just because the seed-count matches."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp
    from crowd_nav.bayesian_brne.interaction_protocol import CONTROLLER_TYPES

    with tempfile.TemporaryDirectory() as tmp:
        # All 4 episodes use the SAME controller instead of one each.
        _build_tiny_formal_role_dir(
            tmp, "train", suite_seed=914, n_episodes=4, horizon_steps=6,
            controller_fn=lambda ep: CONTROLLER_TYPES[0],
        )
        registry = _tiny_registry_for_role("train", tmp, [914], 4, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_controller_imbalance", raised)


@register("s1_data_role_rejects_missing_controller", "s1")
def _test_s1_data_role_rejects_missing_controller():
    """A controller that never appears AT ALL (count=0) is a special case
    of imbalance worth its own explicit red case, per guide.md's wording
    ("controller缺失")."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp
    from crowd_nav.bayesian_brne.interaction_protocol import CONTROLLER_TYPES

    with tempfile.TemporaryDirectory() as tmp:
        # 4 episodes cycling only the FIRST 3 controllers -- the 4th never appears.
        _build_tiny_formal_role_dir(
            tmp, "train", suite_seed=915, n_episodes=4, horizon_steps=6,
            controller_fn=lambda ep: CONTROLLER_TYPES[ep % 3],
        )
        registry = _tiny_registry_for_role("train", tmp, [915], 4, horizon_steps=6)
        raised = False
        try:
            sp.verify_formal_data_role("", "train", registry)
        except sp.DataRoleIntegrityError:
            raised = True
        check("s1_data_role_rejects_missing_controller", raised)


def _minimal_role_report(initial_state_hashes, episode_seeds, episode_ids):
    return {"initial_state_hashes": initial_state_hashes, "episode_seeds": episode_seeds, "episode_ids": episode_ids}


@register("s1_data_role_cross_disjoint_detects_shared_initial_hash", "s1")
def _test_s1_data_role_cross_disjoint_detects_shared_hash():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    train_report = _minimal_role_report(["h1", "h2"], [1100000, 1100001], [(11, 1100000), (11, 1100001)])
    selection_report_ok = _minimal_role_report(["h3", "h4"], [2100000, 2100001], [(21, 2100000), (21, 2100001)])
    selection_report_bad_hash = _minimal_role_report(["h2", "h5"], [2100002, 2100003], [(21, 2100002), (21, 2100003)])
    raised = False
    try:
        sp.verify_train_selection_cross_disjoint(train_report, selection_report_ok)
    except sp.DataRoleIntegrityError:
        raised = True
    check("s1_data_role_cross_disjoint_accepts_disjoint_hashes", not raised)
    raised = False
    try:
        sp.verify_train_selection_cross_disjoint(train_report, selection_report_bad_hash)
    except sp.DataRoleIntegrityError:
        raised = True
    check("s1_data_role_cross_disjoint_detects_shared_initial_hash", raised)


@register("s1_data_role_cross_disjoint_detects_shared_episode_seed", "s1")
def _test_s1_data_role_cross_disjoint_detects_shared_episode_seed():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    train_report = _minimal_role_report(["h1"], [1100000], [(11, 1100000)])
    selection_report_bad_seed = _minimal_role_report(["h9"], [1100000], [(21, 1100000)])
    raised = False
    try:
        sp.verify_train_selection_cross_disjoint(train_report, selection_report_bad_seed)
    except sp.DataRoleIntegrityError:
        raised = True
    check(
        "s1_data_role_cross_disjoint_detects_shared_episode_seed", raised,
        "a shared raw episode_seed across roles must raise even if initial_state_hash and episode_id both differ",
    )


@register("s1_source_manifest_soft_paths_do_not_raise", "s1")
def _test_s1_source_manifest_soft_paths_do_not_raise():
    """R0R-4: a tracked file in ``soft_paths`` (an S1 CLI/protocol file
    still under active development) must be reported as drifted, never
    raised -- otherwise continued S1-1..S1-BUILD development would make
    every subsequent preflight run PRECHECK_FAIL on its own progress."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        soft_rel = "soft_file.py"
        hard_rel = "hard_file.py"
        (Path(tmp) / soft_rel).write_text("a = 1\n")
        (Path(tmp) / hard_rel).write_text("b = 1\n")
        manifest = sp.build_source_manifest(tmp, extra_files=[soft_rel, hard_rel])

        (Path(tmp) / soft_rel).write_text("a = 2\n")  # drift on a SOFT path
        report = sp.verify_source_manifest_unchanged(tmp, manifest, soft_paths=frozenset([soft_rel]))
        check("s1_source_manifest_soft_drift_reported_not_raised", report["soft_changed"] == [soft_rel], str(report))
        check("s1_source_manifest_soft_drift_not_in_hard_changed", report["hard_changed"] == [], str(report))


@register("s1_source_manifest_hard_paths_still_raise", "s1")
def _test_s1_source_manifest_hard_paths_still_raise():
    """R0R-4: drift on a HARD (not actively-developed) tracked path must
    still raise, even when other soft paths are also drifting."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as tmp:
        soft_rel = "soft_file.py"
        hard_rel = "hard_file.py"
        (Path(tmp) / soft_rel).write_text("a = 1\n")
        (Path(tmp) / hard_rel).write_text("b = 1\n")
        manifest = sp.build_source_manifest(tmp, extra_files=[soft_rel, hard_rel])

        (Path(tmp) / hard_rel).write_text("b = 2\n")  # drift on a HARD path
        raised = False
        try:
            sp.verify_source_manifest_unchanged(tmp, manifest, soft_paths=frozenset([soft_rel]))
        except sp.PreflightError:
            raised = True
        check("s1_source_manifest_hard_drift_still_raises", raised)


def _fake_guide_with_all_required_regions(region2_body="R0R-1 body\n新增S1-BUILD body\n", region3_body="A1 body\nA6 body\n"):
    """A synthetic guide.md with THREE marker-pair regions (mirroring the
    real document's 13.1-13.3 / R0R-1..7+S1-BUILD / A1..A6 structure) plus
    narrative execution-report sections OUTSIDE any marker, and a
    legitimate in-region PROSE MENTION of an execution-report heading name
    (to prove that narrative quoting, not just an actual leaked heading,
    is what must be tolerated)."""
    return (
        "# Title\n\n"
        "<!-- S1_PROTOCOL_FREEZE_START -->\n"
        "## 13. S1 protocol\nprotocol body line 1\nprotocol body line 2\n"
        "<!-- S1_PROTOCOL_FREEZE_END -->\n\n"
        "## Order S1-0 执行报告（CC）\nnarrative report, must be excluded\n\n"
        "<!-- S1_PROTOCOL_FREEZE_START -->\n"
        f"### fixes\n{region2_body}"
        "<!-- S1_PROTOCOL_FREEZE_END -->\n\n"
        "## Order S1-0R 执行报告（CC）\nmore narrative, must be excluded\n\n"
        "<!-- S1_PROTOCOL_FREEZE_START -->\n"
        f"a note explaining that an earlier bug stopped at `## Order S1-0 执行报告` (quoted in prose, not a real heading here)\n{region3_body}"
        "<!-- S1_PROTOCOL_FREEZE_END -->\n\n"
        "因此当前唯一有效状态总结是：stale, must be excluded\n"
    )


@register("s1_frozen_protocol_spec_extracts_and_freezes", "s1")
def _test_s1_frozen_protocol_spec_extracts_and_freezes():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    fake_guide = _fake_guide_with_all_required_regions()
    extracted = sp.extract_frozen_protocol_spec(fake_guide)
    check("s1_frozen_protocol_spec_extraction_includes_all_three_regions",
          "protocol body line 1" in extracted and "R0R-1 body" in extracted and "A6 body" in extracted)
    check("s1_frozen_protocol_spec_extraction_excludes_cc_report", "narrative report, must be excluded" not in extracted)
    check("s1_frozen_protocol_spec_extraction_excludes_codex_report", "more narrative, must be excluded" not in extracted)
    check("s1_frozen_protocol_spec_extraction_excludes_trailing_summary", "stale, must be excluded" not in extracted)
    check(
        "s1_frozen_protocol_spec_extraction_tolerates_prose_mention_of_report_heading",
        "quoted in prose, not a real heading here" in extracted,
        "a narrative MENTION of an execution-report heading's name (in backticks, not a real '## ' line) "
        "must be preserved as legitimate normative content describing the bug, not treated as a leak",
    )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result_a = sp.freeze_protocol_spec(tmp, fake_guide)
        check("s1_frozen_protocol_spec_first_freeze_is_new", result_a["newly_frozen"] is True)
        result_b = sp.freeze_protocol_spec(tmp, fake_guide)
        check("s1_frozen_protocol_spec_second_call_not_new", result_b["newly_frozen"] is False)
        check("s1_frozen_protocol_spec_hash_stable", result_a["sha256"] == result_b["sha256"])


@register("s1_frozen_protocol_spec_detects_drift", "s1")
def _test_s1_frozen_protocol_spec_detects_drift():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    guide_v1 = _fake_guide_with_all_required_regions()
    guide_v2 = _fake_guide_with_all_required_regions(region3_body="A1 body CHANGED\nA6 body\n")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sp.freeze_protocol_spec(tmp, guide_v1)
        raised = False
        try:
            sp.freeze_protocol_spec(tmp, guide_v2)
        except sp.PreflightError:
            raised = True
        check("s1_frozen_protocol_spec_detects_drift", raised, "spec text changing after freeze must raise PreflightError")


@register("s1_frozen_protocol_spec_rejects_no_markers", "s1")
def _test_s1_frozen_protocol_spec_rejects_no_markers():
    """A1: heading-based extraction is exactly the bug that silently
    truncated the real spec at 340 lines -- there must be no fallback to
    heading heuristics when markers are entirely absent."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    raised = False
    try:
        sp.extract_frozen_protocol_spec("## 13. S1 protocol\nbody with no markers at all\n")
    except sp.PreflightError:
        raised = True
    check("s1_frozen_protocol_spec_rejects_no_markers", raised)


@register("s1_frozen_protocol_spec_rejects_unbalanced_markers", "s1")
def _test_s1_frozen_protocol_spec_rejects_unbalanced_markers():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    raised = False
    try:
        sp.extract_frozen_protocol_spec("<!-- S1_PROTOCOL_FREEZE_START -->\nbody, never closed\n")
    except sp.PreflightError:
        raised = True
    check("s1_frozen_protocol_spec_rejects_unbalanced_markers", raised)


@register("s1_frozen_protocol_spec_rejects_nested_markers", "s1")
def _test_s1_frozen_protocol_spec_rejects_nested_markers():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    raised = False
    try:
        sp.extract_frozen_protocol_spec(
            "<!-- S1_PROTOCOL_FREEZE_START -->\nouter\n<!-- S1_PROTOCOL_FREEZE_START -->\ninner\n"
            "<!-- S1_PROTOCOL_FREEZE_END -->\n<!-- S1_PROTOCOL_FREEZE_END -->\n"
        )
    except sp.PreflightError:
        raised = True
    check("s1_frozen_protocol_spec_rejects_nested_markers", raised)


@register("s1_frozen_protocol_spec_rejects_missing_required_content", "s1")
def _test_s1_frozen_protocol_spec_rejects_missing_required_content():
    """A1 item 4: the extracted spec must contain the three key strings
    (R0R-1/新增S1-BUILD/A1..A6) -- if a future edit removes a marker pair
    and silently shrinks the frozen spec back down, this must fail loudly
    rather than accept a truncated result."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    incomplete = "<!-- S1_PROTOCOL_FREEZE_START -->\n## 13. S1 protocol\nonly the original section\n<!-- S1_PROTOCOL_FREEZE_END -->\n"
    raised = False
    try:
        sp.extract_frozen_protocol_spec(incomplete)
    except sp.PreflightError:
        raised = True
    check("s1_frozen_protocol_spec_rejects_missing_required_content", raised)


@register("s1_frozen_protocol_spec_rejects_leaked_report_heading", "s1")
def _test_s1_frozen_protocol_spec_rejects_leaked_report_heading():
    """The true-positive companion to the prose-mention-tolerance check
    above: an ACTUAL execution-report heading line accidentally captured
    inside a marker pair (a real marker-placement mistake) must still be
    rejected."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    region2 = "R0R-1 body\n新增S1-BUILD body\n"
    region3_with_leak = "A1 body\n## Order S1-0 执行报告（accidentally captured）\nA6 body\n"
    leaky_guide = _fake_guide_with_all_required_regions(region2_body=region2, region3_body=region3_with_leak)
    raised = False
    try:
        sp.extract_frozen_protocol_spec(leaky_guide)
    except sp.PreflightError:
        raised = True
    check("s1_frozen_protocol_spec_rejects_leaked_report_heading", raised)


@register("s1_committed_protocol_spec_is_hard_tracked", "s1")
def _test_s1_committed_protocol_spec_is_hard_tracked():
    """A1 item 5: the repo-committed spec must be a normal SOURCE_MANIFEST_FILES
    entry (never in the soft-paths set) so its hash is verified
    unconditionally on every host, including the 4090, which never passes
    --protocol-spec-path at all."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    check(
        "s1_committed_protocol_spec_in_source_manifest_files",
        sp.COMMITTED_PROTOCOL_SPEC_REL_PATH in sp.SOURCE_MANIFEST_FILES,
    )
    check(
        "s1_committed_protocol_spec_not_a_soft_path",
        sp.COMMITTED_PROTOCOL_SPEC_REL_PATH not in sp.SOURCE_MANIFEST_SOFT_PATHS_BEFORE_METHOD_LOCK,
    )


@register("s1_amend_committed_protocol_spec_records_and_writes", "s1")
def _test_s1_amend_committed_protocol_spec():
    """A1 item 3: amendment must be explicit, recorded (old/new hash,
    reason, registry hash, timestamp), and never silent."""
    import json
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    guide_v1 = _fake_guide_with_all_required_regions()
    guide_v2 = _fake_guide_with_all_required_regions(region3_body="A1 body v2\nA6 body\n")
    with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as out_tmp:
        result_1 = sp.amend_committed_protocol_spec(repo_tmp, guide_v1, reason="initial commit", registry_sha256="rh1", output_root=out_tmp)
        check("s1_amend_first_call_old_hash_empty", result_1["old_sha256"] == "")
        check("s1_amend_writes_committed_file", Path(result_1["committed_path"]).exists())

        # Without amendment, ordinary consistency verification must now
        # fail once guide.md changes (guide_v2 disagrees with what's committed).
        raised = False
        try:
            sp.verify_committed_protocol_spec_matches_live_guide(repo_tmp, guide_v2)
        except sp.PreflightError:
            raised = True
        check("s1_amend_unamended_drift_detected_by_consistency_check", raised)

        result_2 = sp.amend_committed_protocol_spec(repo_tmp, guide_v2, reason="A1 correction round 2", registry_sha256="rh2", output_root=out_tmp)
        check("s1_amend_second_call_records_old_hash", result_2["old_sha256"] == result_1["new_sha256"])
        check("s1_amend_second_call_new_hash_differs", result_2["new_sha256"] != result_1["new_sha256"])

        amendment_history = json.loads(Path(result_2["amendment_path"]).read_text())
        check("s1_amend_history_has_two_records", len(amendment_history) == 2, str(amendment_history))
        check("s1_amend_history_records_reason", amendment_history[1]["reason"] == "A1 correction round 2")

        # Now verification against guide_v2 must pass -- it's been recorded.
        raised_after = False
        try:
            sp.verify_committed_protocol_spec_matches_live_guide(repo_tmp, guide_v2)
        except sp.PreflightError:
            raised_after = True
        check("s1_amend_consistency_check_passes_after_recorded_amendment", not raised_after)


@register("s1_rollback_archive_detects_tar_byte_tamper", "s1")
def _test_s1_rollback_archive_detects_tar_byte_tamper():
    """R0R-5: corrupting the tar file's own bytes (not the manifest) must
    be caught -- the original verify only checked per-file hashes AFTER
    extraction, never the archive's own recorded SHA256."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as out_tmp:
        rel = "tiny_source_file.py"
        (Path(repo_tmp) / rel).write_text("z = 99\n")
        manifest = sp.build_rollback_archive(repo_tmp, out_tmp, file_list=[rel])
        archive_path = Path(repo_tmp) / manifest["archive_path"] if not Path(manifest["archive_path"]).is_absolute() else Path(manifest["archive_path"])
        if not archive_path.exists():
            archive_path = Path(out_tmp) / "sm_brne_s1_0_rollback.tar.gz"
        raw = bytearray(archive_path.read_bytes())
        raw[-1] ^= 0xFF  # flip the last byte
        archive_path.write_bytes(bytes(raw))
        check(
            "s1_rollback_archive_detects_tar_byte_tamper",
            sp.verify_rollback_archive(repo_tmp, manifest) is False,
        )


@register("s1_rollback_archive_detects_archive_hash_only_tamper", "s1")
def _test_s1_rollback_archive_detects_archive_hash_only_tamper():
    """R0R-5: tampering ONLY ``manifest['archive_sha256']`` (leaving the
    real tar and per-file hashes untouched) must also be caught -- proving
    the archive-level check is real, not a no-op that always defers to the
    per-file check."""
    import tempfile

    from crowd_nav.bayesian_brne import s1_protocol as sp

    with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as out_tmp:
        rel = "tiny_source_file.py"
        (Path(repo_tmp) / rel).write_text("w = 7\n")
        manifest = sp.build_rollback_archive(repo_tmp, out_tmp, file_list=[rel])
        tampered = dict(manifest)
        tampered["archive_sha256"] = "0" * 64
        check(
            "s1_rollback_archive_detects_archive_hash_only_tamper",
            sp.verify_rollback_archive(repo_tmp, tampered) is False,
        )


@register("s1_registry_exact_frozen_values", "s1")
def _test_s1_registry_exact_frozen_values():
    """R0R-6: pin every constant to its EXACT guide.md 13.2 value -- proves
    the validator rejects "another legal combination" that still passes
    generic type/range checks but silently disagrees with the frozen spec."""
    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = sp.load_registry(_S1_REGISTRY_PATH)
    expected = {
        "experiment_id": "s1_strict_20260804", "scenario": "baseline_circle", "n_humans": 5,
        "dt": 0.25, "horizon_steps": 40,
        "primary_k_candidates": [1, 2, 3, 4], "boundary_extension_k_candidates": [5, 6],
        "restart_seeds": [2407, 3407, 4407], "sticky_kappa": 10.0, "dirichlet_alpha": 2.0,
        "shrinkage_scale": 1.0, "inverse_wishart_dof": 6.0, "inverse_wishart_scale": 0.01,
        "em_max_iters": 200, "em_tol": 0.0001, "bootstrap_resamples": 10000, "bootstrap_seed": 72407,
        "confidence_level": 0.95, "nominal_equivalence_margin_nats_per_row": 0.01,
        "min_mode_fraction": 0.03, "max_predictive_similarity": 0.98,
    }
    mismatches = {k: (registry[k], v) for k, v in expected.items() if registry[k] != v}
    check("s1_registry_exact_frozen_values", not mismatches, f"mismatches (actual, expected)={mismatches}")

    expected_seeds = {
        "train": [11, 12, 13, 14, 15], "selection": [21, 22, 23, 24, 25],
        "necessity_id": list(range(31, 41)), "audit_interactive": list(range(41, 51)),
        "audit_nominal_negative_control": list(range(51, 61)),
    }
    seed_mismatches = {
        role: (registry[role]["suite_seeds"], expected_role_seeds)
        for role, expected_role_seeds in expected_seeds.items()
        if registry[role]["suite_seeds"] != expected_role_seeds
    }
    check("s1_registry_exact_frozen_seed_lists", not seed_mismatches, f"mismatches={seed_mismatches}")
    check("s1_registry_exact_train_episodes_per_seed", registry["train"]["episodes_per_seed"] == 400)
    check("s1_registry_exact_selection_episodes_per_seed", registry["selection"]["episodes_per_seed"] == 100)


# Order S1-0R-A (A5): the frozen registry's canonical-JSON content hash,
# pinned to a fixed expected value -- this catches drift in EVERY field,
# including the five data-role sub-dicts' path/environment_split/
# output_root/profile_name values that the field-by-field
# ``s1_registry_exact_frozen_values`` test above never touched (the
# independent audit's exact finding: that test declared "full 28-field
# coverage" but never compared the data-role blocks in full).
_S1_REGISTRY_EXPECTED_CONTENT_SHA256 = "056d92a47ce53614297f5b04165bfc2cdce4d920d7f1ac77f6a9a0d22e279525"


@register("s1_registry_full_content_hash_pinned", "s1")
def _test_s1_registry_full_content_hash_pinned():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    registry = sp.load_registry(_S1_REGISTRY_PATH)
    actual = sp.registry_content_sha256(registry)
    check(
        "s1_registry_full_content_hash_pinned", actual == _S1_REGISTRY_EXPECTED_CONTENT_SHA256,
        f"registry content_sha256 drifted: expected={_S1_REGISTRY_EXPECTED_CONTENT_SHA256} actual={actual} "
        "-- this covers every field byte-for-byte, including the five data-role sub-dicts",
    )


def _mutated_registry_json_path(mutate_fn):
    import json
    import os
    import tempfile

    registry = _load_real_s1_registry_dict()
    mutate_fn(registry)
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(registry, f)
    return path, registry


@register("s1_registry_mutation_output_root_changes_frozen_hash", "s1")
def _test_s1_registry_mutation_output_root_changes_hash():
    """A1/A5: guide.md's independent audit specifically calls out that
    'legal type, wrong value' mutations to output_root/profile_name/
    environment_split pass the generic field validator -- proving the
    FROZEN CONTENT HASH (not the validator) is what must catch them."""
    import os

    from crowd_nav.bayesian_brne import s1_protocol as sp

    def mutate(registry):
        registry["necessity_id"]["output_root"] = "runs/bayesian_brne/some_other_experiment/data/necessity_id"

    path, mutated = _mutated_registry_json_path(mutate)
    try:
        loaded = sp.load_registry(path)  # still passes generic validation -- legal type, non-empty string
        check("s1_registry_mutation_output_root_passes_generic_validation", loaded is not None)
        check(
            "s1_registry_mutation_output_root_changes_frozen_hash",
            sp.registry_content_sha256(loaded) != _S1_REGISTRY_EXPECTED_CONTENT_SHA256,
        )
    finally:
        os.remove(path)


@register("s1_registry_mutation_profile_name_changes_frozen_hash", "s1")
def _test_s1_registry_mutation_profile_name_changes_hash():
    import os

    from crowd_nav.bayesian_brne import s1_protocol as sp

    def mutate(registry):
        registry["audit_interactive"]["profile_name"] = "s1_strict_necessity"  # a DIFFERENT role's legal profile_name

    path, mutated = _mutated_registry_json_path(mutate)
    try:
        loaded = sp.load_registry(path)
        check("s1_registry_mutation_profile_name_passes_generic_validation", loaded is not None)
        check(
            "s1_registry_mutation_profile_name_changes_frozen_hash",
            sp.registry_content_sha256(loaded) != _S1_REGISTRY_EXPECTED_CONTENT_SHA256,
        )
    finally:
        os.remove(path)


@register("s1_registry_mutation_environment_split_changes_frozen_hash", "s1")
def _test_s1_registry_mutation_environment_split_changes_hash():
    import os

    from crowd_nav.bayesian_brne import s1_protocol as sp

    def mutate(registry):
        registry["audit_nominal_negative_control"]["environment_split"] = "test_heldout_interactive"  # legal split, wrong role

    path, mutated = _mutated_registry_json_path(mutate)
    try:
        loaded = sp.load_registry(path)
        check("s1_registry_mutation_environment_split_passes_generic_validation", loaded is not None)
        check(
            "s1_registry_mutation_environment_split_changes_frozen_hash",
            sp.registry_content_sha256(loaded) != _S1_REGISTRY_EXPECTED_CONTENT_SHA256,
        )
    finally:
        os.remove(path)


def _s1_shuffle_episode(seed: int, episode_seed: int, controller: str, offset: float) -> dict:
    robot = np.zeros((10, 9), dtype=float)
    robot[:, 0] = offset
    robot[:, 4] = 0.3
    humans = np.zeros((10, 2, 5), dtype=float)
    humans[:, 0, 0] = 1.0 + offset
    humans[:, 1, 1] = 1.5
    humans[:, :, 4] = 0.3
    actions = np.column_stack((np.full(10, offset + seed), np.linspace(0.0, 1.0, 10)))
    return {
        "suite_seed": seed, "episode_seed": episode_seed, "controller_type": controller,
        "robot": robot, "humans": humans, "valid_mask": np.ones((10, 2), dtype=bool),
        "robot_actions": actions,
        "profile_params": {
            "speed_lo": 0.8, "speed_hi": 1.2, "yield_ttc_threshold": 2.5,
            "goal_switch_ttc_threshold": 2.7, "assertive_probability": 0.3,
        },
    }


@register("s1_build_constrained_shuffle_is_deterministic_derangement", "s1")
def _test_s1_build_constrained_shuffle_is_deterministic_derangement():
    import json
    from crowd_nav.bayesian_brne import s1_protocol as sp

    episodes = []
    for controller_index, controller in enumerate(("orca", "goal_directed")):
        for seed in (1, 2):
            for index in range(3):
                episodes.append(_s1_shuffle_episode(
                    seed, seed * 1000 + controller_index * 100 + index, controller, index * 0.1,
                ))
    first = sp.build_constrained_shuffle_map(episodes, 72407)
    second = sp.build_constrained_shuffle_map(episodes, 72407)
    check("s1_build_shuffle_byte_deterministic", json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True))
    check("s1_build_shuffle_no_self", all(x["recipient_id"] != x["donor_id"] for x in first["mapping"]))
    check("s1_build_shuffle_cross_seed", all(x["recipient_suite_seed"] != x["donor_suite_seed"] for x in first["mapping"]))
    check("s1_build_shuffle_unique_donors", len({x["donor_id"] for x in first["mapping"]}) == len(episodes))
    check("s1_build_shuffle_changes_actions", first["changed_fraction"] >= 0.95)


@register("s1_build_constrained_shuffle_impossible_block_fails", "s1")
def _test_s1_build_constrained_shuffle_impossible_block_fails():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    episodes = [_s1_shuffle_episode(1, 100 + i, "orca", i * 0.1) for i in range(3)]
    raised = False
    try:
        sp.build_constrained_shuffle_map(episodes, 72407)
    except sp.DataRoleIntegrityError:
        raised = True
    check("s1_build_shuffle_impossible_block_fails", raised)


@register("s1_build_restart_selection_uses_train_objective_only", "s1")
def _test_s1_build_restart_selection_uses_train_objective_only():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    reports = [
        {"restart_seed": 1, "eligible": True, "convergence": {"converged": True}, "train_final_objective": 10.0, "selection_nll": -999},
        {"restart_seed": 2, "eligible": True, "convergence": {"converged": True}, "train_final_objective": 11.0, "selection_nll": 999},
        {"restart_seed": 3, "eligible": True, "convergence": {"converged": False}, "train_final_objective": 100.0},
    ]
    selected = sp.select_best_restart_by_train_objective(reports)
    check("s1_build_restart_selected_train_best", selected["restart_seed"] == 2)


@register("s1_build_one_se_selects_smallest_sufficient_k", "s1")
def _test_s1_build_one_se_selects_smallest_sufficient_k():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    result = sp.select_k([
        {"K": 1, "eligible": True, "per_seed_nll": [1.10, 1.11, 1.09, 1.10, 1.10]},
        {"K": 2, "eligible": True, "per_seed_nll": [1.008, 1.018, 0.998, 1.008, 1.008]},
        {"K": 3, "eligible": True, "per_seed_nll": [1.00, 1.02, 0.99, 1.01, 1.00]},
    ])
    check("s1_build_one_se_selects_k2", result["selected_k"] == 2, str(result))


@register("s1_build_suite_seed_bootstrap_is_deterministic", "s1")
def _test_s1_build_suite_seed_bootstrap_is_deterministic():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    blocks = [{"delta_log_likelihood": value, "n_rows": 100} for value in (4, 5, 6, 7, 8)]
    a = sp.block_bootstrap_suite_seed_ci(blocks, 1000, 42, 0.95)
    b = sp.block_bootstrap_suite_seed_ci(blocks, 1000, 42, 0.95)
    check("s1_build_bootstrap_deterministic", a == b)
    check("s1_build_bootstrap_positive", a["ci_low"] > 0.0, str(a))


@register("s1_build_three_state_judgment_covers_all_states", "s1")
def _test_s1_build_three_state_judgment_covers_all_states():
    from crowd_nav.bayesian_brne import s1_protocol as sp

    go = sp.three_state_judgment(
        {"status": "SELECTED"}, {"status": "PASS"}, {"status": "PASS"},
        {"status": "PASS", "ci_low": -0.005, "ci_high": 0.006}, 0.01,
    )
    no_go = sp.three_state_judgment({"status": "NO_GO"}, {"status": "PASS"})
    inconclusive = sp.three_state_judgment({"status": "INCONCLUSIVE"}, {"status": "PASS"})
    check("s1_build_three_state_go", go["verdict"] == "GO")
    check("s1_build_three_state_no_go", no_go["verdict"] == "NO_GO")
    check("s1_build_three_state_inconclusive", inconclusive["verdict"] == "INCONCLUSIVE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None, help="Only run tests registered under this group.")
    args = parser.parse_args()

    if not _REGISTRY:
        print("[SELFTEST] no tests registered yet -- this is expected before Step 2 lands any solver tests.")
        return

    ran = 0
    for name, (group, fn) in sorted(_REGISTRY.items()):
        if args.group is not None and group != args.group:
            continue
        fn()
        ran += 1

    print()
    if ran == 0:
        print(f"[SELFTEST] no tests matched group={args.group!r}")
        raise SystemExit(1)
    if FAILURES:
        print(f"[SELFTEST] {len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print(f"[SELFTEST] all {ran} tests passed")


if __name__ == "__main__":
    main()
