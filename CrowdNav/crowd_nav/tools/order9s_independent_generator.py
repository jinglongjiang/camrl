#!/usr/bin/env python3
"""Order 9S.4: independent (non-FSM) interaction generator cross-check.

guide.md Order 9S.4: since real robot-human interaction data is not
available, use "至少一套不共享六类 FSM 方程的独立 CrowdSim/RVO2/
Social-Force 交互生成器". This module drives EVERY agent (all humans AND
the robot) directly through the raw ``rvo2`` RVO2 simulator -- each agent
has its OWN real goal and preferred velocity, computed by RVO2's reciprocal
avoidance, never touching ``interaction_protocol.py``'s BehaviorType/
compute_human_response FSM at all. This is a genuinely different code path
from Order 6-8's data generator.

REWRITTEN per the 2026-08-03 audit's item 4: an earlier version fit fresh
K-means clusters directly on this generator's own data (bimodal structure
injected via a small-vs-large RVO2 safety-space regime per human) -- which
risks the same "write in N modes, refit and discover N modes" circularity
already flagged for the synthetic FSM result. This version instead FREEZES
the model already fit on guide.md Order 8's FSM-driven formal data and only
SCORES (never refits) it on this independent RVO2 generator's data -- a
genuine transfer/generalization test. Even a positive transfer result here
only supports a cross-simulation-mechanism generalization claim, never a
claim about real humans (guide.md is explicit about this scope limit).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import rvo2

from crowd_nav.bayesian_brne.config import BayesianModelConfig
from crowd_nav.bayesian_brne.mode_model import extract_transitions, fit_and_select
from crowd_nav.bayesian_brne.schemas import build_robot_state_row
from crowd_nav.tools.fit_bayesian_brne import (
    fast_bootstrap_nll_ci,
    freeze_physical_caps,
    load_formal_episodes,
    per_track_nlls,
    track_nlls_by_suite_seed,
)

N_HUMANS = 5
RADIUS = 4.0
AGENT_RADIUS = 0.3
DT = 0.25
HORIZON_STEPS = 40
SMALL_SAFETY_SPACE = 0.0
LARGE_SAFETY_SPACE = 0.6
MAX_SPEED = 1.2


def run_one_episode(rng: np.random.Generator) -> dict:
    """N humans + 1 robot on a circle (same geometry convention as
    baseline_circle, for a fair comparison), ALL driven by raw RVO2 with
    real per-agent goals -- no FSM, no behavior types, no
    interaction_protocol.py import."""
    n_agents = N_HUMANS + 1  # index 0 = robot, 1..N = humans
    angles = 2.0 * np.pi * np.arange(n_agents) / n_agents + rng.uniform(-0.1, 0.1, size=n_agents)
    positions = RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    goals = -positions

    safety_space_mode = rng.integers(0, 2, size=N_HUMANS)  # 0=small, 1=large; robot excluded
    safety_spaces = np.where(safety_space_mode == 1, LARGE_SAFETY_SPACE, SMALL_SAFETY_SPACE)

    sim = rvo2.PyRVOSimulator(DT, 4.5, 12, 5.0, 5.0, AGENT_RADIUS, MAX_SPEED)
    for i in range(n_agents):
        space = 0.0 if i == 0 else float(safety_spaces[i - 1])
        sim.addAgent(tuple(positions[i]), 4.5, 12, 5.0, 5.0, AGENT_RADIUS + space, MAX_SPEED, (0.0, 0.0))

    robot_arr = np.zeros((HORIZON_STEPS, 9))
    humans_arr = np.zeros((HORIZON_STEPS, N_HUMANS, 5))
    track_ids_arr = np.tile(np.arange(N_HUMANS), (HORIZON_STEPS, 1))
    valid_mask = np.ones((HORIZON_STEPS, N_HUMANS), dtype=bool)

    for t in range(HORIZON_STEPS):
        for i in range(n_agents):
            pos = np.array(sim.getAgentPosition(i))
            to_goal = goals[i] - pos
            dist = float(np.hypot(to_goal[0], to_goal[1]))
            direction = to_goal / dist if dist > 1e-6 else np.zeros(2)
            pref = direction * MAX_SPEED
            sim.setAgentPrefVelocity(i, tuple(pref))

        robot_pos = np.array(sim.getAgentPosition(0))
        robot_vel = np.array(sim.getAgentVelocity(0))
        theta = float(np.arctan2(robot_vel[1], robot_vel[0]))
        robot_arr[t] = build_robot_state_row(
            px=robot_pos[0], py=robot_pos[1], vx=robot_vel[0], vy=robot_vel[1],
            radius=AGENT_RADIUS, gx=goals[0][0], gy=goals[0][1], v_pref=MAX_SPEED, theta=theta,
        )
        for h in range(N_HUMANS):
            hp = np.array(sim.getAgentPosition(h + 1))
            hv = np.array(sim.getAgentVelocity(h + 1))
            humans_arr[t, h] = [hp[0], hp[1], hv[0], hv[1], AGENT_RADIUS]

        sim.doStep()

    return {
        "humans": humans_arr, "human_track_ids": track_ids_arr,
        "robot": robot_arr, "valid_mask": valid_mask,
        "_true_safety_space_mode": safety_space_mode.tolist(),  # audit-only, never fed to fitting
    }


def collect(n_episodes: int, seed_offset: int) -> List[dict]:
    episodes = []
    for i in range(n_episodes):
        rng = np.random.default_rng(seed_offset + i)
        episodes.append(run_one_episode(rng))
    return episodes


def main() -> None:
    """REWRITTEN per the 2026-08-03 audit's item 4: the original version of
    this script fit_and_select'd fresh K-means clusters ON the independent
    generator's own data -- which risks the same "write in 2 modes, refit
    and discover 2 modes" circularity already flagged for the synthetic FSM
    result, just with a different injection mechanism (safety-space
    bimodality instead of 6 behavior types). This version instead FREEZES
    the model already fit on the ORIGINAL FSM-driven formal train/
    validation data (guide.md Order 8) and only SCORES (never refits) it on
    the independent RVO2 generator's data -- a genuine transfer/
    generalization test, not a fresh discovery task with a fresh chance to
    match whatever was injected."""
    print("[order9s_independent] fitting the FROZEN model ONCE on the ORIGINAL FSM-driven formal data ...")
    train_eps, train_seeds = load_formal_episodes("runs/bayesian_brne/data_formal", "train", "baseline_circle")
    val_eps, val_seeds = load_formal_episodes("runs/bayesian_brne/data_formal", "validation", "baseline_circle")
    fsm_train_rows = extract_transitions(train_eps, dt=0.25)
    fsm_val_rows = extract_transitions(val_eps, dt=0.25)
    max_speed, max_accel = freeze_physical_caps(fsm_train_rows)
    config = BayesianModelConfig(k_candidates=(1, 6), max_human_speed=max_speed, max_human_acceleration=max_accel)
    artifact, reports, fitted = fit_and_select(
        fsm_train_rows, fsm_val_rows, config, seed=2407, require_multimodal=False, min_nll_improvement=0.01, return_all_fits=True,
    )
    F1, Q1, Pi1, _ = fitted[1]
    F6, Q6, Pi6, _ = fitted[6]
    print("[order9s_independent] frozen model ready (F/Q/Pi for K=1 and K=6). Will NOT be refit below.")

    print("[order9s_independent] generating INDEPENDENT episodes via raw RVO2 (no FSM import) ...")
    val_eps_independent = collect(n_episodes=100, seed_offset=91000)
    val_pseudo_seeds = [i % 5 for i in range(len(val_eps_independent))]
    val_rows_independent = extract_transitions(
        [{k: v for k, v in ep.items() if not k.startswith("_")} for ep in val_eps_independent], dt=DT,
    )
    print(f"[order9s_independent] {len(val_rows_independent)} transitions from the independent generator")

    print()
    print("=" * 70)
    print("ORDER 9S.4 (rewritten) -- FROZEN FSM-TRAINED MODEL SCORED (NO REFIT) ON INDEPENDENT RVO2 GENERATOR")
    print("=" * 70)

    pt6 = per_track_nlls(val_rows_independent, F6, Q6, Pi6)
    pt1 = per_track_nlls(val_rows_independent, F1, Q1, Pi1)
    by_seed6 = track_nlls_by_suite_seed(pt6, val_pseudo_seeds)
    by_seed1 = track_nlls_by_suite_seed(pt1, val_pseudo_seeds)
    boot = fast_bootstrap_nll_ci(by_seed6, by_seed1, n_resamples=2000, seed=2407)
    nll6 = float(np.mean([v for vals in pt6.values() for v in vals]))
    nll1 = float(np.mean([v for vals in pt1.values() for v in vals]))
    transfer_improvement = nll1 - nll6

    # Reference: the SAME frozen model's improvement on its OWN (FSM)
    # validation data, for comparison against the transfer result.
    pt6_native = per_track_nlls(fsm_val_rows, F6, Q6, Pi6)
    pt1_native = per_track_nlls(fsm_val_rows, F1, Q1, Pi1)
    nll6_native = float(np.mean([v for vals in pt6_native.values() for v in vals]))
    nll1_native = float(np.mean([v for vals in pt1_native.values() for v in vals]))
    native_improvement = nll1_native - nll6_native

    print(f"native (FSM validation) improvement K6-over-K1: {native_improvement:.4f} nats")
    print(f"transfer (independent RVO2 generator) improvement K6-over-K1: {transfer_improvement:.4f} nats, "
          f"CI=[{boot['improvement_p2_5']:.4f},{boot['improvement_p97_5']:.4f}], ci_supports={boot['ci_supports_real_improvement']}")

    result = {
        "native_improvement_k6_over_k1": native_improvement,
        "transfer_improvement_k6_over_k1": transfer_improvement,
        "transfer_improvement_ci": [boot["improvement_p2_5"], boot["improvement_p97_5"]],
        "transfer_ci_supports_real_improvement": boot["ci_supports_real_improvement"],
    }
    print()
    if boot["ci_supports_real_improvement"] and transfer_improvement > 0:
        print("RESULT: the FSM-trained model's K=6-over-K=1 advantage TRANSFERS to the independent RVO2 generator's data.")
        result["independent_generator_result"] = "TRANSFERS"
    else:
        print("RESULT: the FSM-trained model's K=6-over-K=1 advantage does NOT transfer to the independent "
              "RVO2 generator's data -- consistent with the modes being FSM-specific artifacts, not generalizable structure.")
        result["independent_generator_result"] = "DOES_NOT_TRANSFER"

    out_path = "runs/bayesian_brne/models/order9s_independent_generator.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
