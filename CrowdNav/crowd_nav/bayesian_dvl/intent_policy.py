"""Goal-intent main chain: FEATURE_SCHEMA_V6 feature builder, candidate
scoring, and checkpoint contract (consolidation plan Order 4 item 3/5).

Import graph is deliberately self-contained: config, normalization,
contracts, geometry_features, intent_tracker, scene_candidates, set_encoder,
iqn, model -- NONE of belief.py / rollout.py / world_model.py / counterfactual.py
/ data_coverage.py / oracle_regret.py (the old SBK-HMM + R4 chain). This is
what "final train/inference entry must not import the old branches" means
at the actual Python import-graph level, not just "doesn't call it".

Human feature layout (FEATURE_SCHEMA_V6, HUMAN_FEATURE_DIM_V6 = 62 dims).
V5 handed the network a bare p0..p7 vector whose slots were POSITIONAL;
measured on the held-out junction crowd, p0 meant "left exit" for 412
humans and "right exit" for 68 others, and no coordinates were supplied to
tell them apart. V6 pairs each probability with its own geometry and pools
the set, so candidate ORDER cannot affect the output.

  [0:7]              relative dx,dy, rel vx,vy, radius, speed, ttc
  [7]                normalized track_age
  [8]                normalized posterior entropy (entropy / log(n_valid))
  [9]                top1-minus-top2 probability margin
  [10:12]            posterior-future mean position delta vs CV, normalized
  [12]               posterior-future position spread, normalized
  [13]               candidate count / MAX_CANDIDATE_GOALS (public cardinality;
                     identical across arms, and NOT recoverable from the
                     pooled candidate embedding when the block is zeroed)
  [14 : 14+G*F]      G candidates x F features, row-major:
                       probability, endpoint relative to the human (dx,dy),
                       next waypoint relative to the human (dx,dy)
  [14+G*F : 14+G*F+G]  per-candidate validity mask
where G = MAX_CANDIDATE_GOALS = 8 and F = CANDIDATE_FEATURE_DIM = 5.

set_encoder.SetEncoder is the ONLY place that unpacks this; it is one
tensor rather than three so replay storage, batching and every call site
keep the [B, N, D] shapes they already had.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from crowd_nav.bayesian_dvl import normalization as norm
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    CANDIDATE_FEATURE_DIM, FEATURE_SCHEMA_V6, HUMAN_FEATURE_DIM_V6,
    HUMAN_SCALAR_DIM_V6, MAX_CANDIDATE_GOALS, NORMALIZATION_CONSTANTS, FROZEN_VALUES,
    TRAINING_CONTRACT_V5_CAPPED_RANK_AUDIT_ONLY,
)
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.geometry_features import _robot_feature_vector, compute_action_features_array
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank, IntentTrackerError
from crowd_nav.bayesian_dvl.model import DistributionalValueModel

# MAX_CANDIDATE_GOALS / CANDIDATE_FEATURE_DIM / HUMAN_SCALAR_DIM_V6 /
# HUMAN_FEATURE_DIM_V6 live in intent_runtime_config so the packed layout has
# ONE definition shared by the feature builder and the encoder that unpacks it.
MAX_HUMANS = 20


class IntentPolicyError(ValueError):
    pass


def remaining_time_fraction(global_time: float, time_limit: float) -> float:
    if time_limit <= 0:
        raise IntentPolicyError(f"time_limit must be positive, got {time_limit}")
    return float(np.clip((time_limit - global_time) / time_limit, 0.0, 1.0))


def _candidate_block(
    human: HumanObservation,
    goal_belief: np.ndarray,
    candidate_routes: Sequence[np.ndarray],
    waypoint_index: Sequence[int],
    show_candidates: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """[G, CANDIDATE_FEATURE_DIM] candidate features + [G] validity mask.

    Every candidate carries its OWN geometry, so the network identifies a
    candidate by where it goes rather than by which slot it landed in. See
    FEATURE_SCHEMA_V6 for why the slot index was not identifying.

    ``show_candidates`` is False for the mean/cv arms, whose frozen
    definition is that they receive NO per-goal information. Their block is
    all zeros but their VALIDITY MASK is still set, exactly as in v5: the
    number of public destinations is public scene geometry, not posterior
    information, and every arm has always been allowed to know it. What they
    must not see is which candidate is which or how likely it is.

    The COUNT itself is carried by a scalar in the row, not by this mask.
    Pooling identical all-zero candidate rows gives the same vector for two
    valid candidates as for four, so an all-zero block plus a mask conveys
    nothing about cardinality -- the mask only stops padding rows from
    entering the pool.

    Giving these arms the coordinates while zeroing only the probabilities
    would quietly widen what they can see and stop the ablation measuring
    what it claims to.
    """
    feats = np.zeros((MAX_CANDIDATE_GOALS, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_CANDIDATE_GOALS, dtype=np.float32)
    pos = np.array([human.px, human.py], dtype=np.float64)
    for ci in range(len(goal_belief)):
        mask[ci] = 1.0
        if not show_candidates:
            continue
        route = np.asarray(candidate_routes[ci], dtype=np.float64)
        end_dx, end_dy = norm.normalize_position(*(route[-1] - pos))
        wp = route[min(int(waypoint_index[ci]), len(route) - 1)]
        wp_dx, wp_dy = norm.normalize_position(*(wp - pos))
        feats[ci] = (float(goal_belief[ci]), end_dx, end_dy, wp_dx, wp_dy)
    return feats, mask


def _intent_human_feature_vector(
    robot: RobotObservation,
    human: HumanObservation,
    goal_belief: np.ndarray,
    track_age: int,
    future_mean_delta: np.ndarray,   # [2] posterior-future mean position delta vs CV, meters
    future_spread: float,             # trace of the posterior-future sample covariance, m^2
    candidate_routes: Sequence[np.ndarray],
    waypoint_index: Sequence[int],
    show_candidates: bool,
) -> np.ndarray:
    raw_dx, raw_dy = human.px - robot.px, human.py - robot.py
    raw_speed = float(np.hypot(human.vx, human.vy))
    raw_rel_vx, raw_rel_vy = human.vx - robot.vx, human.vy - robot.vy
    radius_sum = human.radius + robot.radius
    time_limit = float(FROZEN_VALUES["time_limit"])
    raw_ttc = min(_ttc(raw_dx, raw_dy, raw_rel_vx, raw_rel_vy, radius_sum), time_limit)

    dx, dy = norm.normalize_position(raw_dx, raw_dy)
    rel_vx, rel_vy = norm.normalize_relative_velocity(raw_rel_vx, raw_rel_vy)
    radius = norm.normalize_radius(human.radius)
    speed = norm.normalize_speed(raw_speed, NORMALIZATION_CONSTANTS["max_human_speed"])
    ttc = norm.normalize_ttc(raw_ttc, time_limit)
    norm_age = norm.normalize_track_age(float(track_age))

    n = len(goal_belief)
    if n == 0 or n > MAX_CANDIDATE_GOALS:
        raise IntentPolicyError(f"goal_belief cardinality must be in [1,{MAX_CANDIDATE_GOALS}], got {n}")
    cand_feats, cand_mask = _candidate_block(
        human, goal_belief, candidate_routes, waypoint_index, show_candidates)

    max_entropy = float(np.log(n)) if n > 1 else 1.0  # n=1 is deterministic; avoid /0, entropy is 0 anyway
    entropy = float(-np.sum(goal_belief * np.log(np.clip(goal_belief, 1e-12, 1.0))))
    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0)) if n > 1 else 0.0
    sorted_b = np.sort(goal_belief)[::-1]
    top1_margin = float(sorted_b[0] - (sorted_b[1] if n > 1 else 0.0))

    fdx, fdy = norm.normalize_position(float(future_mean_delta[0]), float(future_mean_delta[1]))
    # spread (m^2) normalized the same way covariance was in v1-v4: relative
    # to a max-human-speed-derived scale, clipped to a bounded range.
    spread_scale = max(NORMALIZATION_CONSTANTS["max_human_speed"] ** 2, 1e-6)
    norm_spread = float(np.clip(future_spread / spread_scale, 0.0, 4.0))

    # Candidate CARDINALITY as an explicit scalar. It is public scene
    # geometry (how many destinations exist), identical across all four arms,
    # and it cannot be recovered from the pooled candidate embedding when the
    # block is all zeros -- see HUMAN_SCALAR_DIM_V6.
    norm_count = float(n) / float(MAX_CANDIDATE_GOALS)

    # V6 packed row: scalars, then the candidate block, then its mask.
    return np.concatenate([
        np.array([dx, dy, rel_vx, rel_vy, radius, speed, ttc], dtype=np.float32),
        np.array([norm_age], dtype=np.float32),
        np.array([norm_entropy, top1_margin], dtype=np.float32),
        np.array([fdx, fdy, norm_spread], dtype=np.float32),
        np.array([norm_count], dtype=np.float32),
        cand_feats.reshape(-1), cand_mask,
    ])


def _ttc(dx: float, dy: float, rel_vx: float, rel_vy: float, radius_sum: float) -> float:
    # same linear (constant-relative-velocity) TTC model as the v1-v4 chain
    # (policy.py's private helper) -- a pure function of observable relative
    # geometry, no belief involved either version.
    rel_speed_sq = rel_vx ** 2 + rel_vy ** 2
    if rel_speed_sq < 1e-9:
        return float("inf")
    a = rel_speed_sq
    b = 2.0 * (dx * rel_vx + dy * rel_vy)
    c = dx ** 2 + dy ** 2 - radius_sum ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return float("inf")
    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    candidates = [t for t in (t1, t2) if t >= 0]
    return min(candidates) if candidates else float("inf")


def _future_summary(future_trajectories: Sequence[np.ndarray], cv_future: np.ndarray) -> Tuple[np.ndarray, float]:
    """From a set of predicted future position trajectories (posterior
    futures, from ``GoalIntentTracker.sample_futures``) and the CV-baseline
    future: mean END-POSITION delta vs CV (2,), and the trace of the
    end-position sample covariance across the futures (spread, scalar)."""
    endpoints = np.array([tr[-1] for tr in future_trajectories])  # [n_samples, 2]
    mean_delta = endpoints.mean(axis=0) - cv_future[-1]
    if len(endpoints) > 1:
        cov = np.cov(endpoints.T)
        spread = float(np.trace(np.atleast_2d(cov)))
    else:
        spread = 0.0
    return mean_delta, spread


def build_intent_human_feature_batch(
    bank: IntentBeliefBank,
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    mode: str,
    rng: np.random.Generator,
    horizon: int = 8,
    n_samples: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Padded [MAX_HUMANS, HUMAN_FEATURE_DIM_V6] feature array + mask for
    ONE decision. ``mode`` in {full, mean, cv, uniform} -- this is the ONE
    place the ablation enters the network features, so all four arms share
    identical env trajectories/seeds and differ ONLY in what belief
    information the network is shown.

    FROZEN ARM DEFINITIONS (plan section 3.2). Two successive real bugs
    were found here by audit, both fixed:

      (bug 1) the belief vector fed to the network was ALWAYS the real
      fitted posterior regardless of mode, so the "ablation" only varied
      which futures were sampled, not what the network could see.

      (bug 2, C0.4) ``mean`` was then given a MAP ONE-HOT belief vector
      while its futures were the posterior-WEIGHTED mean -- a "MAP + mean"
      hybrid arm that is neither of the two things it could be named
      after, and which still exposed the posterior's argmax.

    The frozen definitions are:
      full    -- the real fitted posterior vector + posterior-sampled
                 futures (multimodal) + real posterior spread.
      mean    -- NO per-goal probability vector at all (zeros, hence
                 entropy 0 and top1-margin 0); the ONLY posterior-derived
                 information is the posterior-WEIGHTED MEAN future, and
                 spread is exactly 0 (a single averaged trajectory).
      cv      -- no goal posterior of any kind; the future is plain
                 constant-velocity extrapolation, so its future-delta-vs-CV
                 is identically zero and spread is 0. What separates cv
                 from mean is therefore precisely "does the model get a
                 goal-conditioned predicted future at all".
      uniform -- a FLAT 1/n vector + uniformly-sampled futures: "the public
                 candidates exist but nothing distinguishes them". A
                 SUPPLEMENTARY sanity control only (plan section 3.2), not
                 a co-equal main arm, and explicitly NOT "no-belief".

    The per-candidate VALIDITY MASK is deliberately identical across all
    four arms: it encodes only how many PUBLIC candidate destinations the
    scene provides for this human, which is public scene geometry, not
    posterior information.
    """
    if mode not in ("full", "mean", "cv", "uniform"):
        raise IntentPolicyError(f"unknown mode {mode!r}")
    features = np.zeros((MAX_HUMANS, HUMAN_FEATURE_DIM_V6), dtype=np.float32)
    mask = np.zeros(MAX_HUMANS, dtype=bool)
    for i, human in enumerate(humans[:MAX_HUMANS]):
        try:
            tracker = bank.tracker_for(human.track_id)
        except IntentTrackerError:
            continue  # no active track yet (first observation not ingested) -- skip this human this step
        real_belief = tracker.belief()
        n = len(real_belief)
        if mode == "full":
            network_belief = real_belief
        elif mode in ("mean", "cv"):
            # C0.4: neither arm may see the per-goal posterior vector.
            # All-zero also makes normalized entropy and top1-margin
            # exactly 0, so no posterior SHAPE leaks through those two
            # scalars either.
            network_belief = np.zeros(n)
        else:  # uniform
            network_belief = np.ones(n) / n
        pos = np.array([human.px, human.py], dtype=np.float64)
        vel = np.array([human.vx, human.vy], dtype=np.float64)
        cv_future = np.array([pos + vel * tracker.dt * (t + 1) for t in range(horizon)])
        if mode == "cv":
            future_trajs = [cv_future]
        else:
            future_trajs = tracker.sample_futures(pos, vel, horizon=horizon, mode=mode, rng=rng, n_samples=n_samples)
        mean_delta, spread = _future_summary(future_trajs, cv_future)
        track_age = bank.track_age_for(human.track_id)
        features[i] = _intent_human_feature_vector(
            robot, human, network_belief, track_age=track_age, future_mean_delta=mean_delta,
            future_spread=spread, candidate_routes=tracker._routes,
            waypoint_index=tracker._wp_idx, show_candidates=(mode in ("full", "uniform")))
        mask[i] = True
    return features, mask


@dataclass(frozen=True)
class IntentCandidateResult:
    action_index: int
    q_mean: float  # mean over the sampled taus (risk-neutral expected return, consolidation plan §0)


def score_candidates_v5(
    model: DistributionalValueModel,
    robot: RobotObservation,
    human_features: np.ndarray,
    human_mask: np.ndarray,
    action_table: np.ndarray,
    remaining_fraction: float,
    n_taus: int = 32,
    device: str = "cpu",
) -> Tuple[IntentCandidateResult, ...]:
    """Score all 80 candidates with the SAME state embedding (batched over
    actions), default RISK-NEUTRAL expected return (mean over FIXED
    quantile points) -- consolidation plan: IQN kept, CVaR NOT in the main
    method.

    ``tau`` is a FIXED, deterministic grid of quantile points (evenly
    spaced midpoints of n_taus equal bins in (0,1)), NOT a random draw.
    Real bug found by measurement (external review): the previous version
    drew a fresh ``torch.rand`` each call, so scoring the IDENTICAL input
    twice gave a DIFFERENT top-1 action (confirmed: two back-to-back calls
    on the same state picked different actions) -- this silently broke
    both deployment determinism and ablation fairness (full/mean/cv/uniform
    would each get their OWN random tau draw even when everything else was
    controlled). Training (train_step) still legitimately samples tau
    randomly per step (that's how IQN learns the whole quantile function);
    only DEPLOYMENT/EVAL scoring needs to be deterministic, and now is.
    """
    model.eval()
    n_actions = len(action_table)
    robot_feat = _robot_feature_vector(robot, remaining_fraction)
    robot_batch = torch.as_tensor(np.tile(robot_feat, (n_actions, 1)), dtype=torch.float32, device=device)
    human_batch = torch.as_tensor(np.tile(human_features[None], (n_actions, 1, 1)), dtype=torch.float32, device=device)
    mask_batch = torch.as_tensor(np.tile(human_mask[None], (n_actions, 1)), dtype=torch.bool, device=device)
    action_feats = compute_action_features_array(robot, np.asarray(action_table, dtype=np.float64))
    action_batch = torch.as_tensor(action_feats, dtype=torch.float32, device=device)
    fixed_tau = (torch.arange(n_taus, dtype=torch.float32, device=device) + 0.5) / n_taus  # deterministic midpoints
    tau = fixed_tau.unsqueeze(0).expand(n_actions, n_taus)
    with torch.no_grad():
        quantiles = model(robot_batch, human_batch, mask_batch, action_batch, tau)  # [n_actions, n_taus]
        q_mean = quantiles.mean(dim=1).cpu().numpy()
    return tuple(IntentCandidateResult(action_index=i, q_mean=float(q_mean[i])) for i in range(n_actions))


# C0.5 (plan section 5): bumped V5 -> V6 because the TRAINING SEMANTICS
# changed in a way no shape check can detect (see
# the retired v2 contract). A V5 checkpoint loads
# with identical tensor shapes but was fit under the buggy
# online-samples-get-ranking objective, so it MUST fail closed here rather
# than be silently accepted.
CHECKPOINT_SCHEMA_V7 = "bdvl_intent_checkpoint_v7_candidate_set"


def save_intent_checkpoint(
    model: DistributionalValueModel, path: str, action_grid_hash: str, scene_registry_sha256: str,
    optimizer: Optional[torch.optim.Optimizer] = None, extra: Optional[dict] = None,
) -> None:
    """ONE checkpoint contract for the goal-intent chain (single schema, no
    compat branches). ``optimizer`` state is included whenever given, so a
    full-resume checkpoint and an eval-only checkpoint are the SAME schema
    (the field is just absent for eval-only saves), never a second
    incompatible format. ``training_contract_schema`` is a SEPARATE axis
    from ``feature_schema``: it pins the data/loss semantics the weights
    were fit under, which tensor shapes cannot distinguish."""
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_V7,
        "feature_schema": FEATURE_SCHEMA_V6,
        "training_contract_schema": TRAINING_CONTRACT_V5_CAPPED_RANK_AUDIT_ONLY,
        "model_state_dict": model.state_dict(),
        "action_grid_hash": action_grid_hash,
        "scene_registry_sha256": scene_registry_sha256,
        "return_bounds": [float(model.value_network.v_min), float(model.value_network.v_max)],
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_intent_checkpoint(
    path: str, model: DistributionalValueModel, *, optimizer: Optional[torch.optim.Optimizer] = None,
    expected_action_grid_hash: Optional[str] = None, expected_scene_registry_sha256: Optional[str] = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_fields = (
        "checkpoint_schema", "feature_schema", "training_contract_schema",
        "model_state_dict", "action_grid_hash", "scene_registry_sha256",
    )
    for required in required_fields:
        if required not in checkpoint:
            raise IntentPolicyError(f"intent checkpoint {path} missing required field {required!r}")
    # Retired schemas are refused by the equality check below. There is no
    # per-version branch any more: anything that is not the current schema
    # fails closed, which is what "no compat loading" has always meant.
    if checkpoint["checkpoint_schema"] != CHECKPOINT_SCHEMA_V7:
        raise IntentPolicyError(f"checkpoint schema {checkpoint['checkpoint_schema']!r} != {CHECKPOINT_SCHEMA_V7!r}, fail closed, no compat loading")
    if checkpoint["feature_schema"] != FEATURE_SCHEMA_V6:
        raise IntentPolicyError(f"feature schema {checkpoint['feature_schema']!r} != {FEATURE_SCHEMA_V6!r}, fail closed")
    if checkpoint["training_contract_schema"] != TRAINING_CONTRACT_V5_CAPPED_RANK_AUDIT_ONLY:
        raise IntentPolicyError(
            f"training contract {checkpoint['training_contract_schema']!r} != "
            f"{TRAINING_CONTRACT_V5_CAPPED_RANK_AUDIT_ONLY!r}, fail closed -- the loss semantics these "
            f"weights were fit under differ from what this code implements, and no shape check can "
            f"detect that")
    if expected_action_grid_hash is not None and checkpoint["action_grid_hash"] != expected_action_grid_hash:
        raise IntentPolicyError("checkpoint action_grid_hash does not match the loaded action table")
    if expected_scene_registry_sha256 is not None and checkpoint["scene_registry_sha256"] != expected_scene_registry_sha256:
        raise IntentPolicyError("checkpoint scene_registry_sha256 does not match the loaded scene registry")
    checkpoint_bounds = checkpoint.get("return_bounds")
    if checkpoint_bounds is not None:
        stored_min, stored_max = float(checkpoint_bounds[0]), float(checkpoint_bounds[1])
        if abs(stored_min - model.value_network.v_min) > 1e-9 or abs(stored_max - model.value_network.v_max) > 1e-9:
            raise IntentPolicyError("checkpoint return bounds do not match the loaded model's bounds")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise IntentPolicyError(f"checkpoint {path} has no optimizer_state_dict but an optimizer was given to resume into")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {"action_grid_hash": checkpoint["action_grid_hash"], "scene_registry_sha256": checkpoint["scene_registry_sha256"]}
