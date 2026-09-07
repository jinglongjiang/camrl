"""equilibrium_loop.py: Order F4's outer action-conditioned-prior / BRNE
fixed-point coupling loop (2026-08-03 -- guide.md section 4.5).

Upstream BRNE assumes each agent already has a FIXED set of trajectory
samples before it ever runs its weight equilibration -- it has no concept
that a pedestrian's trajectory distribution should itself depend on which
robot trajectory the equilibrium ultimately favors. This module is the
explicit outer loop that couples the two:

    1. From the CURRENT robot candidate weights, form weighted
       representative robot state-position/state-velocity/action
       sequences.
    2. Regenerate every pedestrian's action-conditioned posterior
       trajectory samples CONDITIONED ON those weighted sequences (via
       ``trajectory_sampler.sample_pedestrian_trajectories``, which by
       Order F3 recomputes context at every rollout step -- no frozen
       future-relative-feature approximation -- and by Order R1 takes
       state and action as separate arrays, never one conflated with the
       other).
    3. Run ONE BRNE weight-equilibration pass (``BRNESolver.solve``,
       unmodified -- this loop is a NEW outer layer, never a change to
       upstream's own math) on the resulting FIXED trajectory samples
       (robot's own fixed FUTURE positions -- where it will actually be,
       for collision cost -- plus the just-regenerated pedestrian
       samples).
    4. Take the robot's own new equilibrium weights and go back to step 1.
    5. Common random numbers: pedestrian resampling at every outer
       iteration reseeds each track's RNG from the SAME per-track seed, so
       regenerating samples under a DIFFERENT weighted robot trajectory
       changes results ONLY because the conditioning input changed, never
       because of fresh independent resampling noise -- otherwise noise
       could masquerade as "the equilibrium moved."
    6. Stop once the robot's own weight vector changes by less than
       ``weight_change_threshold``, or after ``max_outer_iterations``.

``outer_iterations``, the final max weight residual, and whether the loop
actually converged (vs. exhausted its iteration budget) are always returned
in ``OuterLoopResult`` -- never silently dropped.

Order R1 (2026-08-03) fix: an earlier version took ``robot_controls``/
``robot_positions`` (two arrays, where ``robot_positions[m,h]`` was
actually the position AFTER ``robot_controls[m,h]`` was applied) and
weighted+passed BOTH into ``sample_pedestrian_trajectories`` as if they
were "the robot's state at h" -- conflating state with action. This
version consumes ``robot_sampler.sample_robot_candidates``'s FOUR-array
contract (``actions``/``state_positions``/``state_velocities``/
``future_positions``) and keeps them separate throughout: the pedestrian
conditioning uses the weighted STATE sequences + weighted ACTION sequence
(matching the offline training convention exactly), while BRNE's own
collision cost uses the (unweighted, per-candidate) FUTURE positions --
never state positions, which are NOT where the robot will actually be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact
from crowd_nav.bayesian_brne.brne_adapter import BRNESolver
from crowd_nav.bayesian_brne.schemas import EquilibriumResult
from crowd_nav.bayesian_brne.trajectory_sampler import sample_pedestrian_trajectories


@dataclass
class OuterLoopResult:
    robot_weights: np.ndarray                       # [M]
    human_weights_by_track: Dict[int, np.ndarray]   # track_id -> [M]
    weighted_robot_state_position_sequence: np.ndarray  # [H, 2]
    weighted_robot_state_velocity_sequence: np.ndarray  # [H, 2]
    weighted_robot_action_sequence: np.ndarray          # [H, 2]
    outer_iterations: int
    converged: bool
    max_weight_residual: float
    final_equilibrium: EquilibriumResult
    status: str
    outer_residual_history: List[float]
    iteration_diagnostics: List[dict]
    final_consistency_residual: float
    final_conditioning_robot_state_position_sequence: np.ndarray
    final_conditioning_robot_state_velocity_sequence: np.ndarray
    final_conditioning_robot_action_sequence: np.ndarray
    final_pedestrian_trajectories_by_track: Dict[int, np.ndarray]


def _weighted_trajectory(weights: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """``weights``:[M], ``candidates``:[M,H,2] -> [H,2] weighted average."""
    normalized = weights / max(float(weights.sum()), 1e-12)
    return np.tensordot(normalized, candidates, axes=(0, 0))


def _classify_outer_status(
    residual_history: List[float],
    *,
    converged: bool,
    max_outer_iterations: int,
    oscillation_tolerance: float,
) -> str:
    """Return a machine-readable reason for the outer loop stopping.

    A two-cycle is identified from two repeated adjacent residual pairs.  It
    is deliberately a diagnostic classification only: it never changes the
    action or silently increases the iteration budget.
    """
    if converged:
        return "converged"
    if len(residual_history) >= 4:
        previous_pair = np.asarray(residual_history[-4:-2], dtype=np.float64)
        latest_pair = np.asarray(residual_history[-2:], dtype=np.float64)
        if (
            np.max(np.abs(previous_pair - latest_pair)) <= oscillation_tolerance
            and float(np.max(latest_pair)) > oscillation_tolerance
        ):
            return "oscillating"
    if len(residual_history) >= max_outer_iterations:
        return "max_iterations"
    return "not_converged"


def run_outer_equilibrium_loop(
    posteriors_by_track: Dict[int, np.ndarray],
    human_state0_by_track: Dict[int, np.ndarray],
    artifact: ARHMMArtifact,
    robot_actions: np.ndarray,           # [M, H, 2] FIXED candidate set (from robot_sampler, computed once)
    robot_state_positions: np.ndarray,   # [M, H, 2] PRE-action state, matches robot_actions index-for-index
    robot_state_velocities: np.ndarray,  # [M, H, 2] PRE-action state
    robot_future_positions: np.ndarray,  # [M, H, 2] POST-action position -- what BRNE's collision cost uses
    radii_by_track: Dict[int, float],
    robot_radius: float,
    horizon: int,
    solver: BRNESolver,
    sampling_mode: str = "full_posterior",
    max_outer_iterations: int = 10,
    weight_change_threshold: float = 1e-3,
    outer_damping: float = 1.0,
    oscillation_tolerance: float = 1e-6,
    edge_mask: Optional[np.ndarray] = None,
    equilibrium_iterations: int = 10,
    seed: int = 2407,
    brne_root: Optional[str] = None,
) -> OuterLoopResult:
    """``posteriors_by_track``/``human_state0_by_track``/``radii_by_track``
    must share the same track_id keys. ``robot_actions``/
    ``robot_state_positions``/``robot_state_velocities``/
    ``robot_future_positions`` are the FIXED, already-sampled candidate set
    (``robot_sampler.sample_robot_candidates``'s output) -- this function
    only re-weights and re-conditions on them, it never re-samples the
    robot's own candidates (those are fixed for the whole outer loop,
    matching guide.md 4.5's own framing: "由当前robot weights得到加权机器人
    控制/轨迹序列", not "resample the robot candidates"). Every pedestrian
    sample batch uses ``num_samples = robot_actions.shape[0]`` so the
    ``[A, M, H, 2]`` tensor ``BRNESolver.solve`` expects has one consistent
    ``M`` across all agents.
    """
    if not (0.0 < outer_damping <= 1.0):
        raise ValueError(f"outer_damping must be in (0, 1], got {outer_damping}")
    if oscillation_tolerance <= 0.0:
        raise ValueError(f"oscillation_tolerance must be > 0, got {oscillation_tolerance}")

    track_ids = sorted(posteriors_by_track.keys())
    num_pedestrian_samples = robot_actions.shape[0]
    radii = np.array([robot_radius] + [radii_by_track[tid] for tid in track_ids])

    robot_weights = np.full(num_pedestrian_samples, 1.0 / num_pedestrian_samples)
    outer_residual_history: List[float] = []
    iteration_diagnostics: List[dict] = []
    converged = False

    def _solve_once(current_weights: np.ndarray, phase: str, iteration_number: int):
        iteration_start = time.perf_counter()
        weighted_state_pos = _weighted_trajectory(current_weights, robot_state_positions)
        weighted_state_vel = _weighted_trajectory(current_weights, robot_state_velocities)
        weighted_action = _weighted_trajectory(current_weights, robot_actions)

        pedestrian_trajs_by_track: Dict[int, np.ndarray] = {}
        sampling_start = time.perf_counter()
        for tid in track_ids:
            # Common random numbers (guide.md 4.5 item 5): the SAME
            # per-track seed is used at EVERY outer iteration, so a change
            # in the resulting samples can only be attributed to the
            # weighted robot trajectory changing, never to fresh independent
            # resampling noise.
            rng = np.random.default_rng(seed + tid)
            pedestrian_trajs_by_track[tid] = sample_pedestrian_trajectories(
                sampling_mode,
                posterior=posteriors_by_track[tid],
                state0=human_state0_by_track[tid],
                artifact=artifact,
                robot_state_position_sequence=weighted_state_pos,
                robot_state_velocity_sequence=weighted_state_vel,
                robot_action_sequence=weighted_action,
                horizon=horizon,
                num_samples=num_pedestrian_samples,
                rng=rng,
                dt=artifact.dt,
                brne_root=brne_root,
            )
        sampling_ms = (time.perf_counter() - sampling_start) * 1000.0

        # BRNE's collision cost needs where the robot will ACTUALLY be
        # (future_positions), never the pre-action state_positions.
        pedestrian_trajs = [pedestrian_trajs_by_track[tid] for tid in track_ids]
        trajectories = np.stack([robot_future_positions] + pedestrian_trajs, axis=0)  # [A, M, H, 2]
        solver_start = time.perf_counter()
        equilibrium = solver.solve(
            trajectories, radii, edge_mask, equilibrium_iterations=equilibrium_iterations
        )
        solver_wall_ms = (time.perf_counter() - solver_start) * 1000.0

        raw_weights = np.asarray(equilibrium.weights[0], dtype=np.float64)
        raw_weights = raw_weights / max(float(raw_weights.sum()), 1e-12)
        return (
            raw_weights,
            equilibrium,
            weighted_state_pos,
            weighted_state_vel,
            weighted_action,
            pedestrian_trajs_by_track,
            sampling_ms,
            solver_wall_ms,
            (time.perf_counter() - iteration_start) * 1000.0,
        )

    last_main_samples = None
    for iteration in range(max_outer_iterations):
        (
            raw_new_weights,
            result,
            weighted_state_pos,
            weighted_state_vel,
            weighted_action,
            pedestrian_trajs_by_track,
            sampling_ms,
            solver_wall_ms,
            iteration_ms,
        ) = _solve_once(robot_weights, "outer", iteration + 1)
        raw_residual = float(np.max(np.abs(raw_new_weights - robot_weights)))
        updated_weights = (1.0 - outer_damping) * robot_weights + outer_damping * raw_new_weights
        update_residual = float(np.max(np.abs(updated_weights - robot_weights)))
        outer_residual_history.append(update_residual)
        iteration_diagnostics.append({
            "phase": "outer",
            "iteration": iteration + 1,
            "raw_weight_residual": raw_residual,
            "updated_weight_residual": update_residual,
            "inner_iterations": int(result.iterations),
            "inner_converged": bool(result.converged),
            "inner_max_weight_change": float(result.max_weight_change),
            "inner_numeric_fallback_used": bool(result.numeric_fallback_used),
            "inner_elapsed_ms": float(result.elapsed_ms),
            "solver_wall_ms": solver_wall_ms,
            "sampling_ms": sampling_ms,
            "outer_iteration_ms": iteration_ms,
            "num_tracks": len(track_ids),
            "num_candidates": num_pedestrian_samples,
        })
        robot_weights = updated_weights / max(float(updated_weights.sum()), 1e-12)
        last_main_samples = (
            weighted_state_pos, weighted_state_vel, weighted_action, pedestrian_trajs_by_track
        )
        if update_residual < weight_change_threshold:
            converged = True
            break

    status = _classify_outer_status(
        outer_residual_history,
        converged=converged,
        max_outer_iterations=max_outer_iterations,
        oscillation_tolerance=oscillation_tolerance,
    )

    # Final consistency solve: the returned equilibrium weights and human
    # weights must come from the same final trajectory batch. This solve is
    # recorded separately and does not silently extend the pre-registered
    # outer iteration budget.
    (
        final_raw_weights,
        final_result,
        final_conditioning_state_pos,
        final_conditioning_state_vel,
        final_conditioning_action,
        final_pedestrian_trajs_by_track,
        sampling_ms,
        solver_wall_ms,
        iteration_ms,
    ) = _solve_once(robot_weights, "final_consistency", len(outer_residual_history) + 1)
    final_consistency_residual = float(np.max(np.abs(final_raw_weights - robot_weights)))
    robot_weights = final_raw_weights
    final_converged = bool(converged and final_consistency_residual < weight_change_threshold)
    if final_converged:
        status = "converged"
    elif status == "converged":
        status = "not_converged_final_consistency"
    iteration_diagnostics.append({
        "phase": "final_consistency",
        "iteration": len(outer_residual_history) + 1,
        "raw_weight_residual": final_consistency_residual,
        "updated_weight_residual": final_consistency_residual,
        "inner_iterations": int(final_result.iterations),
        "inner_converged": bool(final_result.converged),
        "inner_max_weight_change": float(final_result.max_weight_change),
        "inner_numeric_fallback_used": bool(final_result.numeric_fallback_used),
        "inner_elapsed_ms": float(final_result.elapsed_ms),
        "solver_wall_ms": solver_wall_ms,
        "sampling_ms": sampling_ms,
        "outer_iteration_ms": iteration_ms,
        "num_tracks": len(track_ids),
        "num_candidates": num_pedestrian_samples,
    })

    weighted_state_pos_final = _weighted_trajectory(robot_weights, robot_state_positions)
    weighted_state_vel_final = _weighted_trajectory(robot_weights, robot_state_velocities)
    weighted_action_final = _weighted_trajectory(robot_weights, robot_actions)
    human_weights_by_track = {
        tid: np.asarray(final_result.weights[1 + i], dtype=np.float64)
        for i, tid in enumerate(track_ids)
    }

    return OuterLoopResult(
        robot_weights=robot_weights,
        human_weights_by_track=human_weights_by_track,
        weighted_robot_state_position_sequence=weighted_state_pos_final,
        weighted_robot_state_velocity_sequence=weighted_state_vel_final,
        weighted_robot_action_sequence=weighted_action_final,
        outer_iterations=len(outer_residual_history),
        converged=final_converged,
        max_weight_residual=final_consistency_residual,
        final_equilibrium=final_result,
        status=status,
        outer_residual_history=outer_residual_history,
        iteration_diagnostics=iteration_diagnostics,
        final_consistency_residual=final_consistency_residual,
        final_conditioning_robot_state_position_sequence=final_conditioning_state_pos,
        final_conditioning_robot_state_velocity_sequence=final_conditioning_state_vel,
        final_conditioning_robot_action_sequence=final_conditioning_action,
        final_pedestrian_trajectories_by_track=final_pedestrian_trajs_by_track,
    )
