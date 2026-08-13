"""Runtime configuration for the goal-intent (V6) main chain ONLY.

Order B4: the V6 chain used to import ``config.py``, which also carries the
retired SBK-HMM/R4 registry machinery -- ``BDVL_PRODUCTION_SOURCES`` (a
41-entry list of legacy sources that ``build_frozen_registry`` hashes) and
``nonstationary_protocol_source_sha256()``, which OPENS
``bayesian_pilot/protocol.py`` by path. That made a lean V6 deployment
impossible: dropping any legacy file broke registry construction.

This module carries exactly what the V6 decision path needs and nothing
else. It imports no legacy module and hashes no legacy source, so the V6
chain can ship without `bayesian_pilot`, `world_model.py`, `belief.py` and
the rest.

Values are copied VERBATIM from the frozen config.py definitions -- a test
(test_b4_runtime_config_matches_the_frozen_values) asserts they stay
identical to the originals for as long as config.py exists.
"""

from __future__ import annotations

import configparser
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# The retired SBK-HMM's five kinematic modes. Kept ONLY because
# NORMALIZATION_CONSTANTS carries n_semantic_modes, which
# normalization.normalize_entropy() reads. The V6 chain never calls that
# normalizer -- it computes entropy over its OWN goal candidates, where a
# fixed 5-mode maximum would simply be the wrong denominator.
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


FORMAL_SCENARIOS: Dict[str, Dict[str, object]] = {
    "baseline_circle": {"layout": "circle", "radius": 4.0, "humans": 5},
    "baseline_square": {"layout": "square", "width": 10.0, "humans": 10},
    "dense_circle": {"layout": "circle", "radius": 4.0, "humans": 10},
    "dense_square": {"layout": "square", "width": 10.0, "humans": 20},
    "large_circle": {"layout": "circle", "radius": 6.0, "humans": 12},
    "large_square": {"layout": "square", "width": 14.0, "humans": 20},
}


SEMANTIC_MODES: Tuple[str, ...] = ("CV", "ACC", "DECEL", "TURN_L", "TURN_R")


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

# Order 4: V3 is the SAME loss decomposition (demo -> L_MC + lambda*L_rank,
# online -> L_MC only), but lambda is no longer a frozen constant: it is set
# by AdaptiveRankBalancer from an EMA of past per-term gradient norms.
#
# This is an OBJECTIVE change, so it gets its own contract even though the
# network shape is unchanged (feature schema V5 and checkpoint schema V6
# both still hold -- they describe tensor shapes, not what was optimised).
# A V2 checkpoint must be refused for resume: its optimizer state, EMA and
# replay were all produced under lambda=380, which measurably drove the
# weighted ranking gradient to 80-350x the MC gradient. Continuing from it
# under V3 would produce a run that is neither, and could not be described
# in a paper.
TRAINING_CONTRACT_V3_ADAPTIVE_GRADIENT_BALANCE = (
    "bdvl_intent_training_contract_adaptive_gradient_balance_v3")


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


