"""Immutable configuration dataclasses and registry validation for BDVL.

All frozen numeric values come from guide.md section 11 ("配置冻结值").
Nothing here may be silently overridden by a stale env.config: callers
must go through ``ActionGridSpec.from_env_config`` and
``load_and_validate_registry`` so a wrong grid or a missing registry
field fails loudly instead of falling back to a different default.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEMANTIC_MODES: Tuple[str, ...] = ("CV", "ACC", "DECEL", "TURN_L", "TURN_R")

# R2 fix (2026-08-07): the v1 feature schema had no way to distinguish
# "5 seconds into the episode" from "34 seconds into the episode" at the
# same geometry -- the bootstrapped value function could not represent
# "circling further costs you the episode". v2 adds a 7th robot feature
# dim (remaining_fraction) and is therefore INCOMPATIBLE with v1
# checkpoints (different nn.Linear input shape); loading a v1 checkpoint
# under v2 code must fail closed, never silently reinterpret weights.
FEATURE_SCHEMA_V1 = "bdvl_z_state_v1"
FEATURE_SCHEMA_V2 = "bdvl_z_state_time_v2"
# R3-2 fix (2026-08-07): raw features spanned wildly different scales
# (belief/remaining_fraction in [0,1], velocities ~0-2, positions up to
# ~20m, TTC up to 35, track_age up to 141, predictive covariance often
# << 1) feeding one shared MLP with no normalization -- large-magnitude
# dims can dominate gradients and increase seed sensitivity. v3 adds a
# single frozen ``FeatureNormalizer`` (normalization.py) applied
# identically everywhere a feature is built; the robot/human dims
# themselves are unchanged in COUNT, only in scale, but the checkpoint
# is still incompatible with v2 (a v2-trained encoder's weights were
# fit to the old, unnormalized scale) and must fail closed.
FEATURE_SCHEMA_V3 = "bdvl_z_state_normalized_v3"

# R4-1 fix (2026-08-10, guide.md "R4 -- Belief-Bypass Remediation Plan"): v3's
# IQNValueNetwork scored a bare state_embedding (the candidate action only
# entered indirectly, via the one-step-forward-simulated successor state fed
# into SetEncoder) -- V(s',b'), not an explicit Q(s,b,a). The R4-0 audit
# confirmed the continuation value bypasses belief specifically while still
# reacting to raw human kinematics, and guide.md's fix requires the network to
# receive an explicit, action-identifying input that cannot be inferred solely
# from the successor state. v4 adds a 5-dim ActionFeature (vx, vy, speed,
# goal_alignment, turn_cost -- see policy.compute_action_features_array) fed
# through a new ActionEncoder and fused with the SetEncoder output before the
# IQN head. Checkpoint-incompatible with v3 (IQNValueNetwork gained a fusion
# layer with a different input shape) and must fail closed like v1/v2 did.
FEATURE_SCHEMA_V4 = "bdvl_z_state_action_conditioned_v4"
ACTION_FEATURE_DIM = 5

# BDVL reset consolidation plan v1 (2026-08-11, /home/abc/temp/
# bdvl_consolidation_plan.md): v1-v4's human feature belief block
# (belief[5]/pred_mean[2]/pred_cov[3]) was SBK-HMM-specific -- coupled to
# the 5 fixed semantic modes (CV/ACC/DECEL/TURN_L/TURN_R) and their NIW
# predictive moments. Real evidence (R4-4/R4-4R2, guide.md) found this
# belief provides no measurable decision value on CrowdNav's circle-
# crossing scenario, and a follow-up reset investigation
# (reset_exp/step2-step4b) found the root cause: the reactive kinematic
# tracker cannot represent PREDICTIVE multimodal intent (it only detects
# a turn after it starts). v5 replaces the belief block with a
# cardinality-general GOAL-INTENT posterior + posterior-future summary
# (intent_features.py, built from intent_tracker.py's goal-conditioned
# Bayesian posterior over PUBLIC candidate destinations -- never SBK-HMM,
# never a hidden human.gx/gy). Checkpoint-incompatible with v1-v4 (a
# different human-feature dimensionality) and must fail closed like
# those did; this is a genuinely different network, trained from scratch.
FEATURE_SCHEMA_V5 = "bdvl_z_state_goal_intent_v5"

# R4-2R-1 fix (2026-08-10, guide.md "R4-2R-1 -- 训练契约与checkpoint必须
# 升版"): FEATURE_SCHEMA describes the NETWORK's input/output shape
# contract (unchanged by R4-2 -- R4-1's Q(s,b,a) architecture is still
# exactly what's loaded). TRAINING_CONTRACT_SCHEMA is a genuinely
# SEPARATE axis: it describes the DATA/LOSS semantics a checkpoint's
# weights were fit under -- R4-1's checkpoints were trained against a
# placeholder MC-loss target (trainer._partial_action_features_from_table's
# crude features / expert_action_indices[0] standing in for the executed
# action), R4-2's are trained against the frozen executed-action
# posterior-MC formula (policy.compute_executed_action_quantile_target).
# Both produce checkpoints that pass FEATURE_SCHEMA_V4's shape checks
# identically -- shape alone cannot tell them apart -- so a checkpoint
# trained under the retired R4-1 loss must be rejected by this SEPARATE,
# explicit check, not silently accepted into R4-2 selector/evaluator.
TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC = "bdvl_training_contract_executed_action_posterior_mc_v1"

# C0.5 (plan section 5, 2026-08-11): the GOAL-INTENT chain's own training
# contract, on the same "data/loss semantics" axis as the V1 constant
# above but for the V5+ network. Bumped to V2 because the objective
# CHANGED in a way no shape check can detect: V5-era checkpoints were fit
# with the expert ranking loss applied to EVERY replay sample including
# ONLINE ones (so epsilon-random actions were trained to outrank the other
# 79 -- a real objective-function bug), and with a "MAP one-hot + weighted
# mean future" hybrid `mean` ablation arm. Weights fit under those
# semantics are not comparable to, and must never be silently loaded by,
# code implementing the corrected contract:
#   demo sample:   L = L_MC + lambda_rank * L_rank(expert equivalence set)
#   online sample: L = L_MC only
#   mean arm:      no per-goal posterior vector, posterior-weighted mean
#                  future only, spread == 0
TRAINING_CONTRACT_V2_DEMO_RANK_ONLINE_MC = "bdvl_intent_training_contract_demo_rank_online_mc_v2"

# R4-1R-3/R4-2 fix (2026-08-10, guide.md "R4-1R -- 在 R4-2 完成前禁止误启动
# 训练" / "R4-1R-4 -- 冻结 R4-2 的精确数学目标"): flipped to True now that
# R4-2 actually landed in this same change -- RankingDemoSample/
# MCReturnSample both store a genuinely recorded (not inferred)
# executed_action_index/executed_action_features/posterior_seed_key,
# separate from the ranking-only expert_action_indices; L_executed_MC in
# stage1_train_step/stage2_train_step reuses the SAME candidate builder
# (_score_all_candidates restricted to the one executed action) and
# shared posterior worlds deployment uses, aggregating over worlds
# exactly once (compute_executed_action_quantile_target) before
# comparing to the single realized G_t -- see policy.py's
# compute_executed_action_quantile_target docstring for the exact
# formula. All 7 of guide.md R4-2's minimum permanent tests
# (test_r4_2_*) pass, plus a real (non-synthetic) IL+online-RL smoke run
# and a real train-CLI->select->evaluate chain were verified before this
# flip -- see guide.md's R4-2 completion section for the exact commands
# and results. Do not flip this back to False as a shortcut past a
# future regression; fix the regression instead.
R4_2_REPLAY_CONTRACT_COMPLETE = True

# R3R-4 fix (2026-08-07, independent audit): progress_reward changed
# 0.01 -> 0.05 (R3-5) but every checkpoint still recorded the reward
# schema as "bdvl_reward_v1" -- a checkpoint trained under the old
# reward scale could be silently loaded and mixed with new-scale data.
# The schema string is bumped whenever the REWARD FUNCTION or any of
# its frozen coefficients changes; "v1" is retired.
REWARD_SCHEMA_V1 = "bdvl_reward_v1"
REWARD_SCHEMA_V2 = "bdvl_reward_v2_progress005"

# R3R-0 fix (2026-08-07): every CLI (train/evaluate/select/queue) was
# hand-maintaining its own "source_files" list for provenance manifests,
# and none of them covered every file that actually determines the
# result -- normalization.py, ranking.py, belief.py, rollout.py,
# world_model.py and transition.py were all missing from at least one
# manifest. ONE frozen list, imported everywhere, so "what determined
# this artifact" can never silently under-cover a real dependency.
BDVL_PRODUCTION_SOURCES: Tuple[str, ...] = (
    "crowd_nav/bayesian_dvl/config.py",
    "crowd_nav/bayesian_dvl/normalization.py",
    "crowd_nav/bayesian_dvl/ranking.py",
    "crowd_nav/bayesian_dvl/iqn.py",
    "crowd_nav/bayesian_dvl/set_encoder.py",
    "crowd_nav/bayesian_dvl/model.py",
    "crowd_nav/bayesian_dvl/belief.py",
    "crowd_nav/bayesian_dvl/geometry_features.py",
    "crowd_nav/bayesian_dvl/intent_tracker.py",
    "crowd_nav/bayesian_dvl/scene_candidates.py",
    "crowd_nav/bayesian_dvl/intent_policy.py",
    "crowd_nav/bayesian_dvl/junction_scenario.py",
    "crowd_nav/bayesian_dvl/intent_train.py",
    "crowd_nav/bayesian_dvl/intent_train_cli.py",
    "crowd_nav/bayesian_dvl/rollout.py",
    "crowd_nav/bayesian_dvl/counterfactual.py",
    "crowd_nav/bayesian_dvl/world_model.py",
    "crowd_nav/bayesian_dvl/transition.py",
    "crowd_nav/bayesian_dvl/contracts.py",
    "crowd_nav/bayesian_dvl/policy.py",
    "crowd_nav/bayesian_dvl/replay.py",
    "crowd_nav/bayesian_dvl/trainer.py",
    "crowd_nav/bayesian_dvl/statistics.py",
    "crowd_nav/bayesian_dvl/evaluate.py",
    "crowd_nav/bayesian_dvl/data_coverage.py",
    "crowd_nav/bayesian_dvl/oracle_regret.py",
    "crowd_nav/tools/train_bdvl.py",
    "crowd_nav/tools/evaluate_bdvl.py",
    "crowd_nav/tools/select_bdvl_checkpoint.py",
    "crowd_nav/tools/collect_bdvl_r4_4_data.py",
    "crowd_nav/tools/run_bdvl_queue.py",
    "crowd_nav/tools/promote_bdvl_cvar.py",
    "crowd_nav/tools/audit_bdvl_r3r2_gradient.py",
    "crowd_nav/configs/env_bayesian_dvl.config",
    "crowd_nav/configs/train_bayesian_dvl.config",
    # External sources that determine data generation, termination,
    # rewards, robot/action dispatch, and the human controller.
    "crowd_nav/bayesian_pilot/protocol.py",
    "crowd_sim/envs/crowd_sim.py",
    "crowd_sim/envs/policy/orca.py",
    "crowd_sim/envs/utils/robot.py",
    "crowd_sim/envs/utils/action.py",
    "crowd_sim/envs/utils/human.py",
)

FROZEN_VALUES: Dict[str, object] = {
    "dt": 0.25,
    "time_limit": 35.0,
    "max_humans": 20,
    "robot_radius": 0.30,
    "human_radius_default": 0.30,
    "max_human_speed": 2.0,
    "success_distance": 0.30,
    "success_reward": 1.0,
    "collision_penalty": -0.5,
    "timeout_penalty": -0.5,
    # R3-5 fix (2026-08-07): guide.md's R3-5 order specifies a
    # dimensionless normalized_progress = clip((d_t-d_t1)/(v_pref*dt),
    # -1, 1) formula. That formula is NOT implemented here: the real
    # crowd_sim.py (crowd_sim/envs/crowd_sim.py:432, `reward +=
    # self.progress_reward * progress` with `progress` in raw meters) is
    # outside BDVL's permitted edit scope, and transition.py's whole
    # purpose is staying bit-equivalent to it (guide.md A2, verified by
    # audit_bdvl_a2_transition_equivalence.py at 10000/10000 checks) --
    # silently diverging the HYPOTHETICAL candidate-scoring formula from
    # the REAL reward the network is actually trained on would be worse
    # than the problem it's meant to fix. Instead this keeps the
    # ORIGINAL raw-meters formula and only raises the coefficient (0.01
    # -> 0.05) so the single-step forward-vs-backward reward gap
    # (2*progress_reward*dt = 0.025 at v_pref=1.0/dt=0.25) matches the
    # maximum discomfort penalty per step (discomfort_penalty_factor*
    # discomfort_distance*dt = 0.5*0.20*0.25 = 0.025) -- a magnitude
    # this exact reward function is already known to produce learnable
    # behavior at (collision rate stayed low in every R2 experiment),
    # rather than an unverified guess.
    "progress_reward": 0.05,
    "time_penalty": -0.003,
    "stand_penalty": 0.0,
    "stand_speed_threshold": 0.05,
    "discomfort_distance": 0.20,
    "discomfort_penalty_factor": 0.5,
    "semantic_modes": list(SEMANTIC_MODES),
    "action_grid": {"n_speeds": 5, "n_headings": 16, "sampling": "exponential", "include_stop": False},
    "gamma": 0.99,
    "iqn_cosines": 64,
    "iqn_train_quantiles": 16,
    "world_samples_train": 8,
    "world_samples_validation": 16,
    "world_samples_formal": 32,
    "iqn_quantiles_validation": 32,
    "iqn_quantiles_formal": 64,
    "cvar_alpha": 0.20,
    "replay_capacity": 200000,
    "demo_sample_ratio": 0.20,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "epsilon_start": 0.30,
    "epsilon_end": 0.05,
    "updates_per_episode": 1,
    "il_episodes": 5000,
    "rl_min_episodes": 3000,
    "rl_max_episodes": 10000,
    # R3-0/R3-3/R3-4 protocol registry: locked BEFORE any R3 validation
    # run, per guide.md's requirement that these never be tuned against
    # validation SR after the fact.
    #   ranking_margin: softplus-hinge margin in expert_ranking_loss.
    #     Chosen as a modest fraction of the derived return range so a
    #     "barely ahead" expert set still gets pushed further apart, but
    #     the margin itself is not the dominant loss term once roughly
    #     satisfied.
    #   lambda_rank: NOT equal-weighted -- a real gradient-scale audit
    #     (6 trials, random fresh-init encoder/IQN, varied robot/human
    #     geometry and expert index) measured the L2 gradient norm each
    #     loss term contributes to the SAME shared parameters at
    #     lambda_rank=1. At lambda_rank=1 the ranking term is numerically
    #     negligible and Stage 1 is pure MC regression in everything but
    #     name -- guide.md's own required check ("通过训练loss/梯度尺度
    #     审计确认不会完全压倒MC目标；不得扫描多个lambda后选validation最
    #     好者") is satisfied by this gradient-scale measurement, not by
    #     scanning validation SR.
    #   R3R-1/R3R-2 fix (2026-08-07): the FIRST audit (median ratio 201)
    #     was run against a ranking loss with a real bug -- logsumexp
    #     over all 79 non-expert actions carries a systematic +log(79)
    #     bias unrelated to any actual score gap (equal-score loss was
    #     4.4808, not ~0; even the theoretical best case floored at
    #     0.8479, never reaching 0). R3R-1 replaced it with a max/max
    #     hardest-negative margin loss (exactly 0 once separated,
    #     provably invariant to expert-set size). R3R-2 re-ran the
    #     gradient audit against the FIXED loss using REAL ORCA
    #     demonstration data (10 trials on samples drawn from 8 real
    #     collected episodes, discarding trials where the margin was
    #     already satisfied at that random init and therefore had zero
    #     true gradient to measure): median ratio 255.8, range 8.9-330.
    #     lambda_rank=250 (this second audit's median, rounded) replaces
    #     the earlier synthetic-data/buggy-loss value.
    #   grad_clip_norm: global L2 norm clip for Stage 2 updates.
    #   ema_decay: exponential moving average for the deployment/
    #     validation candidate weights (guide.md R3-4 point 4).
    "ranking_margin": 0.10,
    "lambda_rank": 250.0,
    "grad_clip_norm": 10.0,
    # R3RF-0: stability is evaluated on the frozen checkpoint cadence,
    # not on incidental list adjacency. 0.05 means five percentage points.
    "stability_checkpoint_interval": 500,
    "stability_success_rate_delta_max": 0.05,
    "rank_gradient_ratio_min": 0.05,
    "rank_gradient_ratio_max": 50.0,
    "gradient_audit_batches": 128,
    # A single batch can have a large ratio when the MC gradient is near
    # convergence.  The training gate therefore treats out-of-range ratios
    # as diagnostic events and aborts only after this many consecutive active
    # updates remain out of range.  The window equals the frozen audit block.
    "gradient_ratio_sustained_updates": 128,
    # R3R-2: MC batch continues at batch_size=256 (all demo+online
    # samples get L_MC); the expensive 80-action/8-world ranking score
    # is computed for only the first `ranking_batch_size` demo samples
    # per update (guide.md R3R-2 point 1), bounding the extra Stage 1/2
    # compute instead of letting it scale with the full demo share.
    "ranking_batch_size": 16,
    "ema_decay": 0.995,
}

SEED_ROLES: Dict[str, List[int]] = {
    "world_train_suite_seeds": list(range(91001, 91011)),
    "world_validation_suite_seeds": list(range(91101, 91106)),
    "il_collection_seed_base": [92001],
    "rl_training_seeds": [93001, 93002, 93003],
    "checkpoint_validation_seeds": list(range(94001, 94006)),
    "formal_test_suite_seeds": list(range(95001, 95011)),
}

# B7 fix (independent audit, 2026-08-06): `train_nonstationary` /
# `heldout_nonstationary` are NOT undefined -- they already exist as
# `crowd_nav.bayesian_pilot.protocol.PROFILES`. Values copied here
# verbatim (not imported at runtime) so production BDVL code never
# depends on the legacy `bayesian_pilot` package; data-collection
# TOOLING (not the BDVL policy itself) may still import
# `bayesian_pilot.protocol.BehaviorScheduler`/`InterventionORCA`
# directly to actually apply these profiles, since re-deriving that
# RNG-driven scheduler independently would risk a silent mismatch.
NONSTATIONARY_PROFILES: Dict[str, Dict[str, object]] = {
    "nominal": {
        "event_rate": 0.0, "duration_steps": (0, 0), "turn_degrees": (0.0, 0.0),
        "slow_scale": (1.0, 1.0), "event_weights": (0.25, 0.25, 0.25, 0.25),
    },
    "train_nonstationary": {
        "event_rate": 0.025, "duration_steps": (2, 6), "turn_degrees": (20.0, 50.0),
        "slow_scale": (0.35, 0.70), "event_weights": (0.30, 0.25, 0.225, 0.225),
    },
    "heldout_nonstationary": {
        "event_rate": 0.035, "duration_steps": (4, 9), "turn_degrees": (55.0, 85.0),
        "slow_scale": (0.15, 0.45), "event_weights": (0.35, 0.20, 0.225, 0.225),
    },
}


def nonstationary_protocol_source_sha256() -> str:
    """Hash of the legacy source file these values were copied from,
    so registry generation can fail closed if that file ever drifts
    out of sync with the frozen copy above."""
    import hashlib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "crowd_nav" / "bayesian_pilot" / "protocol.py"
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalizer_source_sha256() -> str:
    """Hash of normalization.py's actual code, distinct from the derived
    NORMALIZATION_CONSTANTS *values* folded into the registry hash --
    this additionally catches a formula change that happens to produce
    the same derived numbers (guide.md R3R-0 point 2's "normalizer内容
    hash")."""
    import hashlib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "crowd_nav" / "bayesian_dvl" / "normalization.py"
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_audit_spec() -> Dict[str, object]:
    """Return the frozen provenance contract for the R3R-2 audit.

    The audit result is intentionally not folded into the registry: changing
    a measured result must never mutate the experiment definition.  The
    protocol, bounds, and audit implementation hashes are frozen here so a
    report can prove which contract it ran against without a circular
    registry/result dependency.
    """
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = {
        "audit_script": repo_root / "crowd_nav" / "tools" / "audit_bdvl_r3r2_gradient.py",
        "trainer": repo_root / "crowd_nav" / "bayesian_dvl" / "trainer.py",
        "data_collector": repo_root / "crowd_nav" / "tools" / "train_bdvl.py",
    }
    return {
        "required_batches": int(FROZEN_VALUES["gradient_audit_batches"]),
        "ratio_min": float(FROZEN_VALUES["rank_gradient_ratio_min"]),
        "ratio_max": float(FROZEN_VALUES["rank_gradient_ratio_max"]),
        "sustained_outside_updates": int(FROZEN_VALUES["gradient_ratio_sustained_updates"]),
        "ranking_batch_sizes_to_compare": [8, 16, 32],
        "data_role": "world-train ORCA demonstrations only",
        "source_sha256": {name: _sha256_of_file(path) for name, path in source_paths.items()},
    }


FORMAL_SCENARIOS: Dict[str, Dict[str, object]] = {
    "baseline_circle": {"layout": "circle", "radius": 4.0, "humans": 5},
    "baseline_square": {"layout": "square", "width": 10.0, "humans": 10},
    "dense_circle": {"layout": "circle", "radius": 4.0, "humans": 10},
    "dense_square": {"layout": "square", "width": 10.0, "humans": 20},
    "large_circle": {"layout": "circle", "radius": 6.0, "humans": 12},
    "large_square": {"layout": "square", "width": 14.0, "humans": 20},
}

TRAIN_SCENARIO = "baseline_circle"


def _max_scenario_extent(scenarios: Dict[str, Dict[str, object]]) -> float:
    """Largest reachable point-to-point distance across all frozen formal
    scenarios (circle: diameter: 2r; square: diagonal: side*sqrt(2)).
    Used as the position-normalization bound so it never depends on
    which scenario a given run happens to use (guide.md R3-2: normalizer
    constants must be frozen physical values, not read from
    validation/formal statistics)."""
    extents = []
    for spec in scenarios.values():
        if spec["layout"] == "circle":
            extents.append(2.0 * float(spec["radius"]))
        else:
            extents.append(float(spec["width"]) * float(np.sqrt(2.0)))
    return max(extents)


def _build_normalization_constants() -> Dict[str, float]:
    dt = float(FROZEN_VALUES["dt"])
    robot_max_speed = 1.0  # must match ActionGridSpec's frozen v_max
    max_human_speed = float(FROZEN_VALUES["max_human_speed"])
    return {
        # Position: bounded by the largest reachable extent across all six
        # frozen formal scenarios (large_square: 14*sqrt(2) = 19.80m).
        "max_position_distance": _max_scenario_extent(FORMAL_SCENARIOS),
        "robot_max_speed": robot_max_speed,
        "max_human_speed": max_human_speed,
        # Relative velocity (human - robot) can reach the sum of both
        # agents' own maximum speeds in the worst case (head-on).
        "max_relative_speed": robot_max_speed + max_human_speed,
        # No agent's radius is ever configured above this in the frozen
        # env/formal protocol (robot/human default to 0.30m); this bound
        # is deliberately looser than the actual configured value so the
        # normalized radius is not degenerately constant if a future
        # ablation varies it.
        "max_agent_radius": 1.0,
        "n_semantic_modes": float(len(SEMANTIC_MODES)),
        # Track age counts BeliefTracker.update() calls, capped at the
        # confirmed real max episode step count (config.derive_return_bounds
        # documents why this is 141, not 140, at dt=0.25/time_limit=35).
        "max_track_age_steps": float(int(round(float(FROZEN_VALUES["time_limit"]) / dt)) + 1),
        # World-model predictive [a_parallel, omega] physical bounds:
        # neither is directly frozen elsewhere, so both are derived from
        # already-frozen kinematic limits rather than guessed. Worst-case
        # acceleration: reaching max_human_speed from a standstill in one
        # dt. Worst-case turn rate: a full half-turn (pi rad) in one dt.
        "max_acceleration": max_human_speed / dt,
        "max_turn_rate": float(np.pi) / dt,
    }


NORMALIZATION_CONSTANTS: Dict[str, float] = _build_normalization_constants()


# R3R-4 fix (2026-08-07, point 1/2): the raw-meters reward formula
# (progress_reward * progress, progress in meters) and the dimensionless
# normalized_progress formula guide.md's R3-5 order originally specified
# (k_progress * clip(progress / (v_pref*dt), -1, 1)) are ALGEBRAICALLY
# the same function whenever |progress| <= v_pref*dt (true by construction:
# no action in the frozen 80-action grid exceeds robot_max_speed=v_pref,
# so one step can never cover more than v_pref*dt meters and the clip
# never binds). Setting k_progress = progress_reward * v_pref * dt makes
# k_progress * (progress / (v_pref*dt)) == progress_reward * progress
# identically. This is NOT a second implementation to keep in sync by
# hand -- it exists only so a future change to v_pref or dt has a single
# derived number to check against, per guide.md's explicit requirement
# that such a change "必须触发registry/schema变化，而不是静默沿用."
# audit_bdvl_a2_transition_equivalence.py and
# test_progress_reward_raw_meters_matches_normalized_formula (selftest.py)
# both check this equivalence numerically at the frozen v_pref=1.0/dt=0.25.
PROGRESS_REWARD_NORMALIZED_K: float = float(FROZEN_VALUES["progress_reward"]) * NORMALIZATION_CONSTANTS["robot_max_speed"] * float(FROZEN_VALUES["dt"])


def derive_return_bounds(
    frozen: Dict[str, object] = FROZEN_VALUES,
    max_action_speed: float = 1.0,
    initial_goal_distance: float = None,
) -> Tuple[float, float]:
    """Derive a conservative discounted-return envelope from frozen values.

    The previous derivation mixed an undiscounted telescoping argument with
    the discounted MC target. R2 instead bounds every possible terminal step
    with discounted per-step reward envelopes and the terminal reward at that
    step. ``initial_goal_distance`` remains only for source compatibility;
    the resulting envelope is valid for every frozen evaluation geometry.
    """
    dt = float(frozen["dt"])
    time_limit = float(frozen["time_limit"])
    if dt <= 0 or time_limit <= 0:
        raise ValueError(f"dt and time_limit must both be positive, got dt={dt}, time_limit={time_limit}")
    n_steps_max = int(round(time_limit / dt)) + 1  # empirically confirmed 141 at dt=0.25/time_limit=35
    _ = initial_goal_distance
    gamma = float(frozen["gamma"])
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0,1], got {gamma}")
    progress_reward = float(frozen["progress_reward"])
    max_progress = max_action_speed * dt
    nonterminal_min = (
        float(frozen["time_penalty"])
        - progress_reward * max_progress
        - float(frozen["discomfort_penalty_factor"]) * float(frozen["discomfort_distance"]) * dt
        + min(0.0, float(frozen["stand_penalty"]))
    )
    nonterminal_max = (
        float(frozen["time_penalty"])
        + progress_reward * max_progress
        + max(0.0, float(frozen["stand_penalty"]))
    )
    terminal_min = min(float(frozen["collision_penalty"]), float(frozen["timeout_penalty"]))
    terminal_max = max(float(frozen["success_reward"]), float(frozen["collision_penalty"]), float(frozen["timeout_penalty"]))

    def discounted_prefix(n: int) -> float:
        return sum(gamma ** index for index in range(n))

    lower_candidates = [
        nonterminal_min * discounted_prefix(k) + terminal_min * (gamma ** k)
        for k in range(n_steps_max)
    ]
    upper_candidates = [
        nonterminal_max * discounted_prefix(k) + terminal_max * (gamma ** k)
        for k in range(n_steps_max)
    ]
    v_min = min(lower_candidates)
    v_max = max(upper_candidates)

    return float(v_min), float(v_max)


class RegistryError(ValueError):
    """Raised when a registry, config file, or action grid fails validation."""


def _sha256_of_obj(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionGridSpec:
    """Immutable 80-action discrete grid: 5 speeds x 16 headings, exponential, no stop."""

    n_speeds: int
    n_headings: int
    v_min: float
    v_max: float
    sampling: str
    include_stop: bool

    def __post_init__(self) -> None:
        if self.n_speeds != 5 or self.n_headings != 16:
            raise RegistryError(
                f"BDVL action grid must be 5x16=80 actions, got "
                f"n_speeds={self.n_speeds}, n_headings={self.n_headings}"
            )
        if self.sampling != "exponential":
            raise RegistryError(f"BDVL action grid must use exponential sampling, got {self.sampling!r}")
        if self.include_stop:
            raise RegistryError("BDVL action grid must have include_stop=False")

    @property
    def n_actions(self) -> int:
        n = self.n_speeds * self.n_headings
        if self.include_stop:
            n += 1
        return n

    @classmethod
    def from_env_config(cls, config_path: str) -> "ActionGridSpec":
        """Load the grid from an env.config [policy] section, failing loudly.

        This does NOT use crowd_nav.contracts._load_grid_from_config's
        silent multi-path fallback (guide.md flags that fallback as a
        real footgun: a missing file there silently yields
        n_speeds=6/sampling=even instead of erroring). Here a missing
        file or missing [policy] section is a hard error.
        """
        path = Path(config_path)
        if not path.is_file():
            raise RegistryError(f"env.config not found at explicit path: {path}")
        # RawConfigParser + inline_comment_prefixes matches this project's
        # own convention (see crowd_nav/train.py) -- plain ConfigParser
        # chokes on values like "0.10  # ABLATION: ..." used elsewhere
        # in these config files.
        parser = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
        parser.read(str(path))
        if not parser.has_section("policy"):
            raise RegistryError(f"{path} has no [policy] section")
        section = parser["policy"]
        try:
            spec = cls(
                n_speeds=section.getint("n_speeds"),
                n_headings=section.getint("n_headings"),
                v_min=section.getfloat("v_min", fallback=0.0),
                v_max=section.getfloat("v_max", fallback=1.0),
                sampling=section.get("sampling"),
                include_stop=section.getboolean("include_stop"),
            )
        except (configparser.NoOptionError, ValueError) as exc:
            raise RegistryError(f"{path} [policy] section missing/invalid required grid field: {exc}") from exc
        return spec

    def build_action_table(self) -> List[Tuple[float, float]]:
        """Return the frozen (vx, vy) list, index-for-index reproducible."""
        actions: List[Tuple[float, float]] = []
        for heading_idx in range(self.n_headings):
            for speed_idx in range(self.n_speeds):
                speed = (
                    (np.exp((speed_idx + 1) / self.n_speeds) - 1)
                    / (np.e - 1)
                    * self.v_max
                )
                angle = 2 * np.pi * heading_idx / self.n_headings
                vx = float(speed * np.cos(angle))
                vy = float(speed * np.sin(angle))
                actions.append((vx, vy))
        if len(actions) != self.n_actions:
            raise RegistryError(
                f"internal error: built {len(actions)} actions, expected {self.n_actions}"
            )
        return actions

    def table_hash(self) -> str:
        table = self.build_action_table()
        return _sha256_of_obj([[round(vx, 12), round(vy, 12)] for vx, vy in table])


@dataclass(frozen=True)
class SeedRoles:
    world_train_suite_seeds: Tuple[int, ...]
    world_validation_suite_seeds: Tuple[int, ...]
    il_collection_seed_base: Tuple[int, ...]
    rl_training_seeds: Tuple[int, ...]
    checkpoint_validation_seeds: Tuple[int, ...]
    formal_test_suite_seeds: Tuple[int, ...]

    def all_seeds(self) -> List[int]:
        seeds: List[int] = []
        for field_name in (
            "world_train_suite_seeds",
            "world_validation_suite_seeds",
            "il_collection_seed_base",
            "rl_training_seeds",
            "checkpoint_validation_seeds",
            "formal_test_suite_seeds",
        ):
            seeds.extend(getattr(self, field_name))
        return seeds

    def assert_disjoint(self) -> None:
        seen: Dict[int, str] = {}
        for role_name in (
            "world_train_suite_seeds",
            "world_validation_suite_seeds",
            "il_collection_seed_base",
            "rl_training_seeds",
            "checkpoint_validation_seeds",
            "formal_test_suite_seeds",
        ):
            for seed in getattr(self, role_name):
                if seed in seen:
                    raise RegistryError(
                        f"seed {seed} used in both {seen[seed]!r} and {role_name!r}"
                    )
                seen[seed] = role_name

    @classmethod
    def frozen(cls) -> "SeedRoles":
        roles = cls(
            world_train_suite_seeds=tuple(SEED_ROLES["world_train_suite_seeds"]),
            world_validation_suite_seeds=tuple(SEED_ROLES["world_validation_suite_seeds"]),
            il_collection_seed_base=tuple(SEED_ROLES["il_collection_seed_base"]),
            rl_training_seeds=tuple(SEED_ROLES["rl_training_seeds"]),
            checkpoint_validation_seeds=tuple(SEED_ROLES["checkpoint_validation_seeds"]),
            formal_test_suite_seeds=tuple(SEED_ROLES["formal_test_suite_seeds"]),
        )
        roles.assert_disjoint()
        return roles


@dataclass(frozen=True)
class BDVLRegistry:
    """The single frozen registry object. Construct via ``build_frozen_registry``."""

    protocol_name: str
    frozen_values: Dict[str, object]
    seed_roles: SeedRoles
    action_grid_hash: str
    action_table: Tuple[Tuple[float, float], ...]
    formal_scenarios: Dict[str, Dict[str, object]]
    train_scenario: str
    semantic_modes: Tuple[str, ...]
    nonstationary_profiles: Dict[str, Dict[str, object]]
    nonstationary_protocol_source_sha256: str
    # R3R-0 fix (2026-08-07): normalization.py's scale constants determine
    # every feature the network ever sees and were previously absent from
    # the registry entirely -- a resumed run had no way to detect that
    # NORMALIZATION_CONSTANTS had drifted since the checkpoint was written.
    normalization_constants: Dict[str, float]
    # R3R-0 fix (2026-08-07, point 2): feature/reward schema and a hash of
    # normalization.py's actual code (not just its derived constant
    # values) are now part of the frozen registry itself, not only a
    # per-checkpoint field -- guide.md 2987 "BDVLRegistry加入：feature
    # schema、reward schema...normalizer内容hash".
    feature_schema: str
    reward_schema: str
    normalizer_source_sha256: str
    gradient_audit: Dict[str, object]
    training_contract_schema: str

    def content_hash(self) -> str:
        payload = {
            "protocol_name": self.protocol_name,
            "frozen_values": self.frozen_values,
            "seed_roles": asdict(self.seed_roles),
            "action_grid_hash": self.action_grid_hash,
            "action_table": [list(a) for a in self.action_table],
            "formal_scenarios": self.formal_scenarios,
            "train_scenario": self.train_scenario,
            "semantic_modes": list(self.semantic_modes),
            "nonstationary_profiles": self.nonstationary_profiles,
            "nonstationary_protocol_source_sha256": self.nonstationary_protocol_source_sha256,
            "normalization_constants": self.normalization_constants,
            "feature_schema": self.feature_schema,
            "reward_schema": self.reward_schema,
            "normalizer_source_sha256": self.normalizer_source_sha256,
            "gradient_audit": self.gradient_audit,
            "training_contract_schema": self.training_contract_schema,
        }
        return _sha256_of_obj(payload)

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "protocol_name": self.protocol_name,
            "frozen_values": self.frozen_values,
            "seed_roles": asdict(self.seed_roles),
            "action_grid_hash": self.action_grid_hash,
            "action_table": [list(a) for a in self.action_table],
            "formal_scenarios": self.formal_scenarios,
            "train_scenario": self.train_scenario,
            "semantic_modes": list(self.semantic_modes),
            "nonstationary_profiles": self.nonstationary_profiles,
            "nonstationary_protocol_source_sha256": self.nonstationary_protocol_source_sha256,
            "normalization_constants": self.normalization_constants,
            "feature_schema": self.feature_schema,
            "reward_schema": self.reward_schema,
            "normalizer_source_sha256": self.normalizer_source_sha256,
            "gradient_audit": self.gradient_audit,
            "training_contract_schema": self.training_contract_schema,
            "content_sha256": self.content_hash(),
        }


def build_frozen_registry(env_config_path: str) -> BDVLRegistry:
    grid = ActionGridSpec.from_env_config(env_config_path)
    seed_roles = SeedRoles.frozen()
    return BDVLRegistry(
        protocol_name="Bayesian Distributional Value Lookahead (BDVL)",
        frozen_values=dict(FROZEN_VALUES),
        seed_roles=seed_roles,
        action_grid_hash=grid.table_hash(),
        action_table=tuple(grid.build_action_table()),
        formal_scenarios=dict(FORMAL_SCENARIOS),
        train_scenario=TRAIN_SCENARIO,
        semantic_modes=SEMANTIC_MODES,
        nonstationary_profiles=dict(NONSTATIONARY_PROFILES),
        nonstationary_protocol_source_sha256=nonstationary_protocol_source_sha256(),
        normalization_constants=dict(NORMALIZATION_CONSTANTS),
        feature_schema=FEATURE_SCHEMA_V4,
        reward_schema=REWARD_SCHEMA_V2,
        normalizer_source_sha256=normalizer_source_sha256(),
        gradient_audit=gradient_audit_spec(),
        training_contract_schema=TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC,
    )


def write_registry(registry: BDVLRegistry, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.to_json_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _sha256_of_file(path)


def load_and_validate_registry(registry_path: str, env_config_path: Optional[str] = None) -> Dict[object, object]:
    """Load a written registry and verify its stored content hash matches
    a fresh recomputation from its own fields (fail closed on drift).

    R4-1R-2 fix (2026-08-10, guide.md "R4-1R -- 冻结 V4 registry"): a
    registry stamped with a retired ``feature_schema``/``reward_schema``
    used to load without complaint here -- the only place that actually
    rejected a schema mismatch was ``load_composed_checkpoint``, checking
    the CHECKPOINT's own recorded schema, never the registry file itself.
    That let a stale V3 registry.json keep being the default
    ``--registry`` argument everywhere while a freshly-trained V4
    checkpoint's ``registry_content_sha256`` (computed from a
    freshly-built, correctly V4-stamped in-memory registry) silently
    could never match it -- the full train -> select -> evaluate chain
    was broken with no clear error pointing at the real cause. Every
    active caller must now be on the current schema; a script that
    genuinely needs to reproduce an old (e.g. R3) run must pin a
    checked-out historical commit/worktree, not load an old-schema
    registry through current code (guide.md's "R4-0 历史复现边界").

    ``env_config_path``, when given, additionally recomputes the action
    grid from that config and fails closed if it does not match the
    registry's own ``action_grid_hash`` -- optional (default None, no
    check) because not every caller has a natural env-config path handy
    at this call site; every real CLI entry point that already owns one
    should pass it.
    """
    path = Path(registry_path)
    if not path.is_file():
        raise RegistryError(f"registry not found: {path}")
    data = json.loads(path.read_text())
    stored_hash = data.pop("content_sha256", None)
    if stored_hash is None:
        raise RegistryError(f"registry {path} missing content_sha256 field")
    recomputed = _sha256_of_obj(data)
    if recomputed != stored_hash:
        raise RegistryError(
            f"registry {path} content hash mismatch: stored={stored_hash} recomputed={recomputed}"
        )
    data["content_sha256"] = stored_hash
    if data.get("feature_schema") != FEATURE_SCHEMA_V4:
        raise RegistryError(
            f"registry {path} feature_schema is {data.get('feature_schema')!r}, "
            f"current code requires {FEATURE_SCHEMA_V4!r} -- this is a retired-schema registry, "
            "not a config typo; do not edit the field to silence this"
        )
    if data.get("reward_schema") != REWARD_SCHEMA_V2:
        raise RegistryError(
            f"registry {path} reward_schema is {data.get('reward_schema')!r}, current code requires {REWARD_SCHEMA_V2!r}"
        )
    if data.get("training_contract_schema") != TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC:
        raise RegistryError(
            f"registry {path} training_contract_schema is {data.get('training_contract_schema')!r}, "
            f"current code requires {TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC!r} (guide.md R4-2R-1) -- "
            "this registry predates the R4-2 executed-action MC-loss rewrite"
        )
    if env_config_path is not None:
        fresh_hash = ActionGridSpec.from_env_config(env_config_path).table_hash()
        if fresh_hash != data.get("action_grid_hash"):
            raise RegistryError(
                f"registry {path} action_grid_hash {data.get('action_grid_hash')!r} does not match "
                f"a fresh recomputation from {env_config_path} ({fresh_hash!r})"
            )
    return data
