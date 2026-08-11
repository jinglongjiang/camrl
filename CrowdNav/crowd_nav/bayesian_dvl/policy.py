"""BDVLPolicy: ties world model + belief + rollout + set encoder + IQN
into the 8-step action-selection procedure (guide.md 3.1/A7).

Step order is fixed and tested (guide.md A7 acceptance):
    1. canonicalize the current real observation
    2. update the REAL belief tracker once
    3. draw shared posterior world samples (common random numbers
       across all 80 actions)
    4. build 80 robot successor states (one per frozen action)
    5. compute sampled reward + hypothetical belief for each
       (action, world sample) pair
    6. batch IQN forward over all (action, sample) pairs
    7. compute lower-tail CVaR per action
    8. pick exactly one action index (never a weighted blend)
    9. only continue past this point once the environment returns the
       NEXT real observation (i.e. this object is single-shot per
       decision call, not holding half-applied state)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from crowd_nav.bayesian_dvl.belief import BeliefTracker
from crowd_nav.bayesian_dvl.geometry_features import _robot_feature_vector, compute_action_features_array
from crowd_nav.bayesian_dvl.config import (
    ACTION_FEATURE_DIM, FEATURE_SCHEMA_V1, FEATURE_SCHEMA_V2, FEATURE_SCHEMA_V3, FEATURE_SCHEMA_V4,
    FROZEN_VALUES, NORMALIZATION_CONSTANTS, REWARD_SCHEMA_V1, REWARD_SCHEMA_V2,
    TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC, derive_return_bounds,
)
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork
from crowd_nav.bayesian_dvl import normalization as norm
from crowd_nav.bayesian_dvl.rollout import _stable_seed, sample_human_next_states
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, HUMAN_FEATURE_DIM, ROBOT_FEATURE_DIM, SetEncoder
from crowd_nav.bayesian_dvl.transition import RewardConfig
from crowd_nav.bayesian_dvl.statistics import CALIBRATION_TAUS
from crowd_nav.bayesian_dvl.transition import batch_step as bdvl_batch_step
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact


class PolicyError(ValueError):
    pass


def save_composed_checkpoint(
    encoder: SetEncoder, value_network: IQNValueNetwork, path: str,
    action_grid_hash: str, registry_content_sha256: str,
    action_encoder: Optional[ActionEncoder] = None,
    artifact_sha256: Optional[str] = None,
    training_config_sha256: Optional[str] = None,
    feature_schema: str = FEATURE_SCHEMA_V4,
    reward_schema: str = REWARD_SCHEMA_V2,
    training_contract_schema: str = TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC,
) -> None:
    """B6 fix (independent audit, 2026-08-06): ONE checkpoint file
    holding both networks' state plus the exact registry/action-grid
    hashes they were trained against, instead of two independently
    optional .pth files with no provenance binding at all.

    R2-2 fix (2026-08-07): also records the value network's own
    ``(v_min, v_max)`` return bounds, so a checkpoint trained against one
    derivation of the reward-implied return range can never be silently
    loaded under a different one.

    R4-1 fix (2026-08-10): ``action_encoder`` is a third module now
    required for any v4 checkpoint (its weights are what makes the
    Q(s,b,a) contract action-conditioned at all). Optional only so
    v1/v2/v3-schema selftests that exercise the OLD retired-schema
    rejection path can still build a payload without constructing one.

    R4-2R-1 fix (2026-08-10, guide.md "R4-2R-1"): ``training_contract_schema``
    is a SEPARATE axis from ``feature_schema`` -- see
    ``config.TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC``'s own comment for
    why shape compatibility alone (which ``feature_schema`` checks)
    cannot distinguish an R4-1-loss checkpoint from an R4-2-loss one.
    """
    payload = {
        "encoder_state_dict": encoder.state_dict(),
        "value_network_state_dict": value_network.state_dict(),
        "action_grid_hash": action_grid_hash,
        "registry_content_sha256": registry_content_sha256,
        "feature_schema": feature_schema,
        "reward_schema": reward_schema,
        "training_contract_schema": training_contract_schema,
        "return_bounds": [float(value_network.v_min), float(value_network.v_max)],
    }
    if action_encoder is not None:
        payload["action_encoder_state_dict"] = action_encoder.state_dict()
    if artifact_sha256 is not None:
        payload["artifact_sha256"] = artifact_sha256
    if training_config_sha256 is not None:
        payload["training_config_sha256"] = training_config_sha256
    torch.save(payload, path)


def load_composed_checkpoint(
    path: str,
    encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: Optional[ActionEncoder] = None,
    *,
    expected_artifact_sha256: Optional[str] = None,
    expected_registry_content_sha256: Optional[str] = None,
    expected_action_grid_hash: Optional[str] = None,
    require_formal_provenance: bool = False,
) -> dict:
    """Loads state into the given (already-constructed) modules IN
    PLACE and returns the checkpoint's provenance manifest so the
    caller can verify it against the registry actually in use.

    R4-1 fix (2026-08-10): ``action_encoder`` is optional ONLY so the
    v1/v2/v3 retired-schema selftests (which never built one) still
    exercise the schema-rejection path without constructing a real
    ActionEncoder; every real v4 caller must pass one, and this
    function requires ``action_encoder_state_dict`` to be present
    whenever an ``action_encoder`` module was actually given, so a v4
    checkpoint can never silently skip loading it.
    """
    # BDVL checkpoints are trusted, repository-owned artifacts.  They contain
    # the replay buffer's Transition records in full-resume checkpoints, so
    # PyTorch 2.6+'s weights-only default cannot deserialize them.  Keep the
    # trust boundary explicit instead of depending on a version-specific
    # default.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_fields = ("encoder_state_dict", "value_network_state_dict", "action_grid_hash", "registry_content_sha256")
    if require_formal_provenance:
        required_fields += ("artifact_sha256", "training_config_sha256", "feature_schema", "reward_schema")
    for required in required_fields:
        if required not in checkpoint:
            raise PolicyError(f"composed checkpoint {path} missing required field {required!r}")
    if expected_artifact_sha256 is not None and checkpoint.get("artifact_sha256") != expected_artifact_sha256:
        raise PolicyError("checkpoint artifact hash does not match the production artifact")
    if expected_registry_content_sha256 is not None and checkpoint.get("registry_content_sha256") != expected_registry_content_sha256:
        raise PolicyError("checkpoint registry hash does not match the loaded registry")
    if expected_action_grid_hash is not None and checkpoint.get("action_grid_hash") != expected_action_grid_hash:
        raise PolicyError("checkpoint action grid hash does not match the loaded action table")
    # R2-1/R3-2/R4-1 fix: fail closed on ANY retired schema, not just under
    # require_formal_provenance -- a v1 (6-dim, no time), v2 (7-dim,
    # unnormalized) or v3 (state-only V(s,b), no action conditioning)
    # checkpoint loaded into v4 (Q(s,b,a)) modules must never be silently
    # accepted, in smoke mode or otherwise. v3's IQNValueNetwork has a
    # DIFFERENT constructor signature (no action_embedding_dim) and a
    # DIFFERENT fusion-layer shape, so a naive load_state_dict would
    # already raise -- but the explicit schema string check gives a
    # clear diagnosis instead of a bare shape-mismatch traceback, and
    # is the only thing that would catch a same-shape future schema
    # drift. schema_present is None only for checkpoints that never
    # recorded it at all (pre-B6 legacy, already excluded by the
    # required_fields check above whenever require_formal_provenance is
    # set).
    schema_present = checkpoint.get("feature_schema")
    retired_schemas = {
        FEATURE_SCHEMA_V1: "6-dim robot features, no remaining-time signal",
        FEATURE_SCHEMA_V2: "7-dim robot features, unnormalized raw scale",
        FEATURE_SCHEMA_V3: "state-only V(s,b) value head, no explicit action conditioning (guide.md R4-1)",
    }
    if schema_present in retired_schemas:
        raise PolicyError(
            f"checkpoint {path} uses the retired {schema_present!r} feature schema "
            f"({retired_schemas[schema_present]}); current code requires "
            f"{FEATURE_SCHEMA_V4!r} and cannot warm-start from it"
        )
    if require_formal_provenance and schema_present != FEATURE_SCHEMA_V4:
        raise PolicyError(f"checkpoint feature schema is not {FEATURE_SCHEMA_V4!r}")
    if action_encoder is not None and "action_encoder_state_dict" not in checkpoint:
        raise PolicyError(
            f"checkpoint {path} has no 'action_encoder_state_dict' but an ActionEncoder module was "
            "given to load into -- this is not a valid v4 checkpoint"
        )
    # R3R-4 fix: progress_reward changed 0.01 -> 0.05 (R3-5) without a
    # schema bump at the time -- fail closed on the retired v1 reward
    # schema exactly like the retired feature schemas above, in every
    # mode, not just require_formal_provenance.
    reward_schema_present = checkpoint.get("reward_schema")
    if reward_schema_present == REWARD_SCHEMA_V1:
        raise PolicyError(
            f"checkpoint {path} uses the retired {REWARD_SCHEMA_V1!r} reward schema "
            f"(progress_reward=0.01); current code requires {REWARD_SCHEMA_V2!r} "
            f"(progress_reward=0.05) and cannot warm-start from it"
        )
    if require_formal_provenance and reward_schema_present != REWARD_SCHEMA_V2:
        raise PolicyError(f"checkpoint reward schema is not {REWARD_SCHEMA_V2!r}")
    # R4-2R-1 fix (2026-08-10, guide.md "R4-2R-1 -- 训练契约与checkpoint必须
    # 升版"): checked UNCONDITIONALLY (not gated behind
    # require_formal_provenance), exactly like the retired feature-schema
    # check above -- a checkpoint trained under R4-1's placeholder MC-loss
    # has the IDENTICAL tensor shapes as an R4-2 one (feature_schema alone
    # cannot tell them apart), so this is the only thing that can catch
    # "R4-1-loss weights silently entering the R4-2 selector/evaluator".
    # Missing entirely (None) is treated the same as any other mismatch --
    # every checkpoint saved before this field existed predates R4-2R-1
    # and must be rejected, not grandfathered in.
    training_contract_present = checkpoint.get("training_contract_schema")
    if training_contract_present != TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC:
        raise PolicyError(
            f"checkpoint {path} training_contract_schema is {training_contract_present!r}, "
            f"current code requires {TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC!r} -- this checkpoint "
            "predates the R4-2 executed-action MC-loss rewrite (or predates this field entirely) "
            "and must not be treated as an R4-2 weight"
        )
    checkpoint_bounds = checkpoint.get("return_bounds")
    if checkpoint_bounds is not None:
        stored_min, stored_max = float(checkpoint_bounds[0]), float(checkpoint_bounds[1])
        if abs(stored_min - value_network.v_min) > 1e-9 or abs(stored_max - value_network.v_max) > 1e-9:
            raise PolicyError(
                f"checkpoint return bounds {(stored_min, stored_max)} do not match "
                f"the loaded value network's bounds {(value_network.v_min, value_network.v_max)}"
            )
    elif require_formal_provenance:
        raise PolicyError("composed checkpoint missing required field 'return_bounds'")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    value_network.load_state_dict(checkpoint["value_network_state_dict"])
    if action_encoder is not None:
        action_encoder.load_state_dict(checkpoint["action_encoder_state_dict"])
    return {
        "action_grid_hash": checkpoint["action_grid_hash"],
        "registry_content_sha256": checkpoint["registry_content_sha256"],
        "artifact_sha256": checkpoint.get("artifact_sha256"),
        "training_config_sha256": checkpoint.get("training_config_sha256"),
        "feature_schema": checkpoint.get("feature_schema"),
        "reward_schema": checkpoint.get("reward_schema"),
        "training_contract_schema": checkpoint.get("training_contract_schema"),
        "return_bounds": checkpoint_bounds,
    }


@dataclass(frozen=True)
class ActionScore:
    action_index: int
    action_vx: float
    action_vy: float
    cvar: float  # mean of returns whose bootstrap term used tau ~ Uniform(0, cvar_alpha)


@dataclass(frozen=True)
class DecisionResult:
    chosen_action_index: int
    chosen_action: Tuple[float, float]
    all_scores: Tuple[ActionScore, ...]


def remaining_time_fraction(global_time: float, time_limit: float) -> float:
    """Normalized time-to-go, in [0, 1] (R2-1 fix, 2026-08-07). The ONE
    named constructor for this feature -- every caller (real state at
    time t, all 80 hypothetical successor states at t+dt, training
    collection, smoke/latency fixtures) must go through this function so
    the same clock semantics are never hand-derived twice and drift."""
    if time_limit <= 0:
        raise PolicyError(f"time_limit must be positive, got {time_limit}")
    return float(np.clip((time_limit - global_time) / time_limit, 0.0, 1.0))


def _human_feature_vector(
    robot: RobotObservation,
    human: HumanObservation,
    belief: np.ndarray,
    entropy: float,
    track_age: int,
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
) -> np.ndarray:
    # guide.md 5.1: "relative position/velocity, radius, speed, TTC,
    # belief[5], entropy, track_age, one-step predictive mean[2],
    # covariance[3]" -- this IS the Z(s,b) contract (R2 fix, independent
    # audit B0, 2026-08-06): belief and predictive moments are part of
    # the network's own input, not a side channel bolted on afterward.
    raw_dx, raw_dy = human.px - robot.px, human.py - robot.py
    raw_speed = float(np.hypot(human.vx, human.vy))
    raw_rel_vx, raw_rel_vy = human.vx - robot.vx, human.vy - robot.vy
    radius_sum = human.radius + robot.radius
    # ``inf`` means that no collision is predicted within the linear model.
    # Feeding an arbitrary 1e6 sentinel into a neural encoder makes the
    # feature scale depend on that sentinel rather than the task horizon.
    # The only relevant TTC beyond the frozen 35 s episode horizon is
    # "outside this episode", so use the frozen horizon as the explicit cap.
    time_limit = float(FROZEN_VALUES["time_limit"])
    raw_ttc = min(_ttc(raw_dx, raw_dy, raw_rel_vx, raw_rel_vy, radius_sum), time_limit)

    # R3-2: single frozen normalizer, applied identically everywhere.
    dx, dy = norm.normalize_position(raw_dx, raw_dy)
    rel_vx, rel_vy = norm.normalize_relative_velocity(raw_rel_vx, raw_rel_vy)
    radius = norm.normalize_radius(human.radius)
    speed = norm.normalize_speed(raw_speed, NORMALIZATION_CONSTANTS["max_human_speed"])
    ttc = norm.normalize_ttc(raw_ttc, time_limit)
    norm_entropy = norm.normalize_entropy(entropy)
    norm_age = norm.normalize_track_age(float(track_age))
    mean_a, mean_omega = norm.normalize_pred_mean(float(pred_mean[0]), float(pred_mean[1]))
    cov_a, cov_ao, cov_omega = norm.normalize_pred_cov_upper(
        float(pred_cov[0, 0]), float(pred_cov[0, 1]), float(pred_cov[1, 1]),
    )
    return np.concatenate([
        np.array([dx, dy, rel_vx, rel_vy, radius, speed, ttc], dtype=np.float32),
        belief.astype(np.float32),
        np.array([norm_entropy, norm_age], dtype=np.float32),
        np.array([mean_a, mean_omega], dtype=np.float32),
        np.array([cov_a, cov_ao, cov_omega], dtype=np.float32),
    ])


def build_human_feature_batch(
    tracker: BeliefTracker, robot: RobotObservation, humans: Sequence[HumanObservation],
) -> Tuple[np.ndarray, np.ndarray]:
    """Padded [MAX_HUMANS, HUMAN_FEATURE_DIM] feature array + mask for
    ONE "current state" (i.e. Z(s) capture, not a hypothetical
    successor) -- the single production builder for this, used by
    ``train_bdvl.py``'s demo/online collection AND R3-3's Stage 1
    ranking-IL loss so the two never hand-maintain equivalent-but-
    separate versions."""
    max_humans = 20
    human_feat = np.zeros((max_humans, HUMAN_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_humans, dtype=bool)
    for i, human in enumerate(humans):
        track_id = human.track_id
        belief = tracker.belief_for(track_id)
        entropy = tracker.entropy_for(track_id)
        age = tracker.track_age_for(track_id)
        mean, cov = tracker.predictive_moments_for(track_id)
        human_feat[i, :] = _human_feature_vector(
            robot, human, belief=belief, entropy=entropy, track_age=age,
            pred_mean=mean, pred_cov=cov,
        )
        mask[i] = True
    return human_feat, mask


def _ttc(px, py, vx, vy, radius_sum) -> float:
    vv = vx * vx + vy * vy
    if vv < 1e-8:
        return float("inf")
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - radius_sum * radius_sum
    disc = b * b - 4.0 * vv * c
    if disc <= 0.0:
        return float("inf")
    t1 = (-b - np.sqrt(disc)) / (2.0 * vv)
    return float(t1) if t1 > 1e-6 else float("inf")


class ReplayIntegrityError(PolicyError):
    pass


def validate_replay_sample_integrity(
    sample, action_table: Sequence[Tuple[float, float]], artifact_sha256: str, expected_source_role: str,
) -> None:
    """R4-2R-3 fix (2026-08-10, guide.md "R4-2R-3 -- Replay 记录必须
    fail closed 自校验"): ``RankingDemoSample``/``MCReturnSample`` store
    ``executed_action_features`` but the real training path
    (``_vectorized_candidate_batch``) always RECOMPUTES action features
    from ``sample.robot``+``action_table`` internally -- the stored field
    was write-only, never actually cross-checked against anything, so a
    corrupted/mismatched replay record (wrong action index, tampered
    feature array, wrong source_role, stale artifact, seed identity that
    doesn't match the sample it's attached to) would train silently
    instead of failing loudly. Called once per sample at the START of
    ``stage1_train_step``/``stage2_train_step``'s loop body, BEFORE the
    (expensive) candidate builder runs on it.

    Raises ``ReplayIntegrityError`` (never silently corrects) on the
    first violation found; does not attempt to be exhaustive about
    which check failed first.
    """
    n_actions = len(action_table)
    if not (0 <= sample.executed_action_index < n_actions):
        raise ReplayIntegrityError(
            f"executed_action_index {sample.executed_action_index} out of range [0, {n_actions})"
        )
    stored = np.asarray(sample.executed_action_features, dtype=np.float64)
    if stored.shape != (ACTION_FEATURE_DIM,):
        raise ReplayIntegrityError(f"executed_action_features shape {stored.shape} != ({ACTION_FEATURE_DIM},)")
    if not np.all(np.isfinite(stored)):
        raise ReplayIntegrityError(f"executed_action_features contains non-finite values: {stored}")
    recomputed = compute_action_features_array(
        sample.robot, np.asarray(action_table, dtype=np.float64),
    )[sample.executed_action_index].astype(np.float64)
    if not np.allclose(stored, recomputed, atol=1e-4):
        raise ReplayIntegrityError(
            f"executed_action_features {stored.tolist()} does not match a fresh recompute "
            f"{recomputed.tolist()} from the sample's own robot/executed_action_index -- "
            "the replay record and its stored feature have drifted apart"
        )
    if sample.source_role != expected_source_role:
        raise ReplayIntegrityError(
            f"sample.source_role={sample.source_role!r} does not match expected {expected_source_role!r}"
        )
    if sample.artifact_sha256 != artifact_sha256:
        raise ReplayIntegrityError(
            f"sample.artifact_sha256={sample.artifact_sha256!r} does not match the currently-loaded "
            f"artifact {artifact_sha256!r} -- this replay item was collected under a different world model"
        )
    seed_key = sample.posterior_seed_key
    if len(seed_key) != 3:
        raise ReplayIntegrityError(f"posterior_seed_key must be a 3-tuple, got {seed_key!r}")
    if int(seed_key[1]) != int(sample.episode_seed):
        raise ReplayIntegrityError(
            f"posterior_seed_key episode component {seed_key[1]} does not match "
            f"sample.episode_seed {sample.episode_seed} -- seed identity does not match sample identity"
        )


def _vectorized_candidate_batch(
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    action_table: Sequence[Tuple[float, float]],
    samples_by_track: Dict[int, Sequence[object]],
    hyp_beliefs_per_sample: Sequence[Dict[int, np.ndarray]],
    hyp_entropy_per_sample: Sequence[Dict[int, float]],
    hyp_age_per_sample: Sequence[Dict[int, int]],
    hyp_pred_moments_per_sample: Sequence[Dict[int, Tuple[np.ndarray, np.ndarray]]],
    dt: float,
    time_limit: float,
    global_time: float,
    reward_config: RewardConfig,
    suite_seed: int,
    episode_seed: int,
    decision_counter: int,
    max_human_speed: float,
    action_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all candidate tensors in one NumPy batch.

    ``action_indices``, when given, must have the same length as
    ``action_table`` and gives the GLOBAL (0..79) identity of each row in
    ``action_table`` for tau-seed purposes only (guide.md R4-2: MC-loss
    scores a single executed action via a length-1 ``action_table``, and
    must derive the SAME tau/world-sampling seed a full 80-action
    deployment/ranking call would have used for that same real action --
    seeding tau from the row's position in a truncated table, always 0,
    would silently diverge). Defaults to ``range(len(action_table))``,
    exactly reproducing prior behavior for every existing 80-action caller.

    This is deliberately a pure batch rewrite of the old action/sample loop;
    ``batch_step`` is separately checked against scalar ``transition.step``.
    The returned flattening order is action-major then world-major, exactly
    matching the previous loop and therefore preserving tau seeds and score
    ordering.
    """
    actions = np.asarray(action_table, dtype=np.float64)
    n_actions = actions.shape[0]
    n_samples = len(hyp_beliefs_per_sample)
    n_humans = len(humans)
    human_actions = np.zeros((n_samples, n_humans, 2), dtype=np.float64)
    for sample_idx in range(n_samples):
        for human_idx, human in enumerate(humans):
            sample = samples_by_track[human.track_id][sample_idx]
            human_actions[sample_idx, human_idx, 0] = sample.next_speed * float(np.cos(sample.next_heading))
            human_actions[sample_idx, human_idx, 1] = sample.next_speed * float(np.sin(sample.next_heading))
    batch = bdvl_batch_step(
        robot=robot, humans=humans, actions=actions, human_actions=human_actions,
        dt=dt, time_limit=time_limit, global_time=global_time, reward_config=reward_config,
    )

    # Robot features are action-dependent but world-independent. All 80
    # candidates share the SAME successor time t+dt (R2-1: the named
    # remaining_time_fraction constructor, evaluated once here rather
    # than re-derived per action, so it can never drift from the real
    # state's own t-based computation in decide()).
    candidate_remaining_fraction = remaining_time_fraction(global_time + dt, time_limit)
    # R3-2: same normalize_*_array functions the scalar builders use --
    # not a hand-vectorized "equivalent" formula (guide.md R3-2 scalar/
    # batch equivalence acceptance).
    raw_goal_rel = np.asarray([robot.gx, robot.gy], dtype=np.float64)[None, :] - batch.next_robot_positions
    norm_dx, norm_dy = norm.normalize_position_array(raw_goal_rel[:, 0], raw_goal_rel[:, 1])
    norm_action_vx, norm_action_vy = norm.normalize_robot_velocity_array(actions[:, 0], actions[:, 1])
    norm_radius = norm.normalize_radius_array(np.full(n_actions, robot.radius, dtype=np.float64))
    norm_v_pref = norm.normalize_speed_array(
        np.full(n_actions, robot.v_pref, dtype=np.float64), NORMALIZATION_CONSTANTS["robot_max_speed"],
    )
    robot_features_by_action = np.stack([
        norm_dx, norm_dy, norm_action_vx, norm_action_vy, norm_radius, norm_v_pref,
        np.full(n_actions, candidate_remaining_fraction, dtype=np.float64),
    ], axis=1).astype(np.float32)
    robot_feats = np.repeat(robot_features_by_action[:, None, :], n_samples, axis=1)

    # R4-1: action features are, like robot_feats, action-dependent but
    # world-independent -- computed from the robot's CURRENT (pre-action)
    # state, never from the simulated successor, so they are a genuinely
    # separate signal from robot_feats rather than a re-derivation of it.
    action_features_by_action = compute_action_features_array(robot, actions)
    action_feats = np.repeat(action_features_by_action[:, None, :], n_samples, axis=1)

    # Stack hypothetical tracker outputs once per world sample. The tracker
    # is still updated exactly as before; only the action broadcast changes.
    belief_dim = 5
    if n_humans:
        belief_dim = int(np.asarray(hyp_beliefs_per_sample[0][humans[0].track_id]).shape[0])
        beliefs = np.stack([
            np.stack([hyp_beliefs_per_sample[s][h.track_id] for h in humans], axis=0)
            for s in range(n_samples)
        ], axis=0).astype(np.float32)
        entropies = np.asarray([
            [hyp_entropy_per_sample[s][h.track_id] for h in humans]
            for s in range(n_samples)
        ], dtype=np.float32)
        ages = np.asarray([
            [hyp_age_per_sample[s][h.track_id] for h in humans]
            for s in range(n_samples)
        ], dtype=np.float32)
        pred_means = np.asarray([
            [hyp_pred_moments_per_sample[s][h.track_id][0] for h in humans]
            for s in range(n_samples)
        ], dtype=np.float32)
        pred_covs = np.asarray([
            [hyp_pred_moments_per_sample[s][h.track_id][1] for h in humans]
            for s in range(n_samples)
        ], dtype=np.float32)
    else:
        beliefs = np.zeros((n_samples, 0, belief_dim), dtype=np.float32)
        entropies = np.zeros((n_samples, 0), dtype=np.float32)
        ages = np.zeros((n_samples, 0), dtype=np.float32)
        pred_means = np.zeros((n_samples, 0, 2), dtype=np.float32)
        pred_covs = np.zeros((n_samples, 0, 2, 2), dtype=np.float32)

    # [A,S,H,*] broadcast: one vectorized TTC/feature construction replaces
    # 80*32*20 calls to _human_feature_vector.
    human_feat = np.zeros((n_actions, n_samples, 20, HUMAN_FEATURE_DIM), dtype=np.float32)
    if n_humans:
        next_human_positions = batch.next_human_positions[None, :, :, :]
        next_human_velocities = batch.next_human_velocities[None, :, :, :]
        robot_positions = batch.next_robot_positions[:, None, None, :]
        robot_velocities = actions[:, None, None, :]
        relative_positions = next_human_positions - robot_positions
        relative_velocities = next_human_velocities - robot_velocities
        speeds = np.hypot(next_human_velocities[..., 0], next_human_velocities[..., 1])
        radius_sum = np.asarray([h.radius + robot.radius for h in humans], dtype=np.float64)
        vv = np.sum(relative_velocities * relative_velocities, axis=-1)
        b = 2.0 * np.sum(relative_positions * relative_velocities, axis=-1)
        c = np.sum(relative_positions * relative_positions, axis=-1) - radius_sum[None, None, :] ** 2
        discriminant = b * b - 4.0 * vv * c
        sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
        t1 = np.divide(
            -b - sqrt_discriminant, 2.0 * vv,
            out=np.full_like(vv, np.inf), where=vv >= 1e-8,
        )
        valid_ttc = (vv >= 1e-8) & (discriminant > 0.0) & (t1 > 1e-6)
        ttc = np.where(valid_ttc, t1, np.inf)
        time_limit_val = float(FROZEN_VALUES["time_limit"])
        ttc = np.minimum(ttc, time_limit_val)
        cov_upper = np.stack([pred_covs[..., 0, 0], pred_covs[..., 0, 1], pred_covs[..., 1, 1]], axis=-1)
        tiled_shape = (n_actions, n_samples, n_humans)

        # R3-2: same normalize_*_array functions the scalar
        # _human_feature_vector builder uses.
        norm_rel_px, norm_rel_py = norm.normalize_position_array(relative_positions[..., 0], relative_positions[..., 1])
        norm_rel_vx, norm_rel_vy = norm.normalize_relative_velocity_array(relative_velocities[..., 0], relative_velocities[..., 1])
        human_radii = np.asarray([h.radius for h in humans], dtype=np.float64)
        norm_human_radius = norm.normalize_radius_array(human_radii)
        norm_speed = norm.normalize_speed_array(speeds[0], NORMALIZATION_CONSTANTS["max_human_speed"])
        norm_ttc = norm.normalize_ttc_array(ttc, time_limit_val)
        norm_entropy = norm.normalize_entropy_array(entropies)
        norm_age = norm.normalize_track_age_array(ages)
        norm_mean_a, norm_mean_omega = norm.normalize_pred_mean_array(pred_means[..., 0], pred_means[..., 1])
        norm_cov_a, norm_cov_ao, norm_cov_omega = norm.normalize_pred_cov_upper_array(
            cov_upper[..., 0], cov_upper[..., 1], cov_upper[..., 2],
        )
        norm_pred_means = np.stack([norm_mean_a, norm_mean_omega], axis=-1)
        norm_cov = np.stack([norm_cov_a, norm_cov_ao, norm_cov_omega], axis=-1)

        parts = [
            np.stack([norm_rel_px, norm_rel_py], axis=-1).astype(np.float32),
            np.stack([norm_rel_vx, norm_rel_vy], axis=-1).astype(np.float32),
            np.broadcast_to(norm_human_radius.astype(np.float32)[None, None, :, None], tiled_shape + (1,)),
            np.broadcast_to(norm_speed[None, :, :, None].astype(np.float32), tiled_shape + (1,)),
            norm_ttc[:, :, :, None].astype(np.float32),
            np.broadcast_to(beliefs[None, :, :, :], tiled_shape + (belief_dim,)),
            np.broadcast_to(norm_entropy[None, :, :, None], tiled_shape + (1,)),
            np.broadcast_to(norm_age[None, :, :, None], tiled_shape + (1,)),
            np.broadcast_to(norm_pred_means[None, :, :, :], tiled_shape + (2,)),
            np.broadcast_to(norm_cov[None, :, :, :], tiled_shape + (3,)),
        ]
        human_feat[:, :, :n_humans, :] = np.concatenate(parts, axis=-1)

    masks = np.zeros((n_actions, n_samples, 20), dtype=bool)
    masks[:, :, :n_humans] = True
    rewards = batch.rewards
    not_terminal = ~(batch.terminated | batch.truncated)
    global_action_indices = list(range(n_actions)) if action_indices is None else list(action_indices)
    if len(global_action_indices) != n_actions:
        raise ValueError(f"action_indices length {len(global_action_indices)} != n_actions {n_actions}")
    tau_seeds = np.empty((n_actions, n_samples), dtype=np.int64)
    for row, global_action_idx in enumerate(global_action_indices):
        for sample_idx in range(n_samples):
            tau_seeds[row, sample_idx] = _stable_seed(
                suite_seed, episode_seed, decision_counter, global_action_idx, sample_idx
            )
    return (
        robot_feats.reshape(n_actions * n_samples, ROBOT_FEATURE_DIM),
        human_feat.reshape(n_actions * n_samples, 20, HUMAN_FEATURE_DIM),
        masks.reshape(n_actions * n_samples, 20),
        rewards.reshape(n_actions * n_samples),
        not_terminal.reshape(n_actions * n_samples),
        tau_seeds.reshape(n_actions * n_samples),
        action_feats.reshape(n_actions * n_samples, ACTION_FEATURE_DIM),
    )


def _encode_score_quantiles(
    set_encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    robot_feats: torch.Tensor,
    human_feats: torch.Tensor,
    masks: torch.Tensor,
    action_feats: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """The ONE encode->fuse->IQN forward pass (guide.md R3-3/R4-1/R4-2):
    every caller that needs quantile values for a (state, action) batch
    -- decision scoring, ranking, and now the R4-2 executed-action MC
    target -- goes through exactly this, differing only in how they
    AGGREGATE the resulting ``[batch, n_quantiles]`` tensor afterward.
    Deliberately grad-transparent; callers wrap in ``torch.no_grad()``
    themselves where appropriate."""
    state_embeddings = set_encoder(robot_feats, human_feats, masks)
    action_embeddings = action_encoder(action_feats)
    return value_network(state_embeddings, action_embeddings, tau)  # [batch, n_quantiles]


def score_candidate_batch(
    set_encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    robot_feats: torch.Tensor,
    human_feats: torch.Tensor,
    masks: torch.Tensor,
    action_feats: torch.Tensor,
    rewards: torch.Tensor,
    not_terminal: torch.Tensor,
    tau: torch.Tensor,
    n_actions: int,
    n_samples: int,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """THE single production DECISION-SCORING path (guide.md R3-3/R4-1):
    encode -> IQN -> per-(action,sample) bootstrap -> per-action
    mean-over-samples-AND-quantiles scalar score, used for ranking/argmax.
    Deliberately grad-transparent -- this function never calls
    ``torch.no_grad()`` or ``.detach()`` itself, so the SAME code path
    serves ``BDVLPolicy.decide()`` (wrapped in ``torch.no_grad()`` by the
    caller), the Stage 1 expert-ranking loss (gradients flowing, so the
    encoder/IQN learn to rank expert actions above alternatives), and
    checkpoint selection/evaluation (also wrapped in ``torch.no_grad()``).
    There is no second "approximate" copy of this formula anywhere else
    in the codebase; every caller MUST route through this function.

    R4-1 fix (2026-08-10): the value network now scores Q(s,b,a), not
    V(s',b') -- ``action_feats`` is a genuinely separate input from
    ``robot_feats`` (the latter describes the CANDIDATE SUCCESSOR state,
    the former the action's own identity at the CURRENT decision point;
    see ``compute_action_features_array``), encoded by its own
    ``action_encoder`` and fused with the Set Encoder's state embedding
    inside ``value_network.forward``.

    NOT the right aggregation for training the quantile function itself
    (guide.md R4-2): ``bootstrap_values = quantiles.mean(dim=1)`` collapses
    the quantile axis into a single scalar per (action, sample) BEFORE
    combining with reward -- correct for a ranking/CVaR-style scalar
    decision rule, but it throws away exactly the per-tau structure IQN
    training needs. See ``compute_executed_action_quantile_target`` for
    the quantile-PRESERVING aggregation R4-2's MC loss uses instead.

    Returns ``(action_scores [n_actions], quantiles [n_actions*n_samples,
    n_quantiles])`` -- the same two outputs ``decide()`` has always
    needed (one for argmax/ranking, one for the calibration diagnostic).
    """
    quantiles = _encode_score_quantiles(set_encoder, value_network, action_encoder, robot_feats, human_feats, masks, action_feats, tau)
    bootstrap_values = quantiles.mean(dim=1) * not_terminal.to(quantiles.dtype)
    total_returns = rewards + gamma * bootstrap_values
    total_returns = total_returns.reshape(n_actions, n_samples)
    action_scores = total_returns.mean(dim=1)
    return action_scores, quantiles


def compute_executed_action_quantile_target(
    set_encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    robot_feats: torch.Tensor,
    human_feats: torch.Tensor,
    masks: torch.Tensor,
    action_feats: torch.Tensor,
    rewards: torch.Tensor,
    not_terminal: torch.Tensor,
    n_quantiles: int,
    tau_upper: float,
    seed_key: Tuple[int, int, int],
    executed_action_index: int,
    gamma: float,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """guide.md R4-2's frozen L_executed_MC target construction, for a
    SINGLE (already-taken) action across ``n_samples`` shared-posterior
    world draws:

        z[w,j]   = r(s_t,a_t,w) + gamma * IQN(E(s'_{t+1}(a_t,w), b'_{t+1,w}), E_a(a_t), tau_j)
        z_bar[j] = mean_w z[w,j]

    ``robot_feats``/``human_feats``/``masks``/``action_feats``/``rewards``/
    ``not_terminal`` are the length-``n_samples`` rows
    ``_score_all_candidates`` returns when called with a length-1 action
    table and ``action_indices=[executed_action_index]`` (so WORLD
    sampling is bit-identical to an 80-action deployment call for this
    same real action). Their TAU, however, is NOT reused here: guide.md's
    ``z_bar[j] = mean_w z[w,j]`` only means anything if index ``j`` names
    the SAME tau_j across every world sample w -- but
    ``_vectorized_candidate_batch``'s tau seeding varies PER (action,
    sample) pair by design (each row is an independent contribution to a
    ranking/CVaR score there, never compared index-for-index against
    another row's tau), so directly reusing ``_score_all_candidates``'s
    own ``quantiles``/``tau`` return would silently average together K
    DIFFERENT tau distributions instead of K shared quantile levels. This
    function draws its OWN single shared ``[n_quantiles]`` tau (seeded
    from ``seed_key`` + ``executed_action_index``, no sample index --
    still fully reproducible from stored fields, just a genuinely
    different seed than any per-sample decision-scoring draw) and
    broadcasts it identically to every world sample before the IQN
    forward pass, then aggregates over samples ONLY -- a second,
    necessarily separate, encode/IQN call from whatever
    ``_score_all_candidates`` might also have computed for this state.

    Deliberately does NOT collapse the quantile axis (unlike
    ``score_candidate_batch``'s decision-scoring aggregation, which means
    over quantiles per row BEFORE combining with reward -- correct for a
    scalar ranking score, wrong for training the quantile function
    itself): the resulting ``z_bar`` IS the network's predicted quantile
    function for this action's total return, meant to be compared
    against the single realized Monte Carlo return ``G_t`` via
    ``quantile_huber_loss(z_bar, tau, G_t)`` -- aggregating over world
    samples FIRST and comparing to ONE real number ONCE, never treating
    each hypothetical world sample as though it independently realized
    the one true future (guide.md R4-2: "禁止把同一个真实未来复制成每个
    hypothetical world的独立真值").

    Returns ``(z_bar [n_quantiles], tau [n_quantiles])`` -- the caller
    passes both straight into ``quantile_huber_loss``.
    """
    n_samples = robot_feats.shape[0]
    tau_seed = np.array([_stable_seed(seed_key[0], seed_key[1], seed_key[2], executed_action_index)], dtype=np.int64)
    shared_tau = _stateless_tau_cpu(tau_seed, n_quantiles, tau_upper).to(device)  # [1, K]
    tau_batch = shared_tau.expand(n_samples, n_quantiles)  # [n_samples, K], identical row-for-row
    quantiles = _encode_score_quantiles(set_encoder, value_network, action_encoder, robot_feats, human_feats, masks, action_feats, tau_batch)  # [n_samples, K]
    bootstrap = quantiles * not_terminal.to(quantiles.dtype).unsqueeze(-1)  # [n_samples, K]
    z = rewards.unsqueeze(-1) + gamma * bootstrap  # [n_samples, K]
    return z.mean(dim=0), shared_tau.squeeze(0)  # ([K], [K])


def _score_all_candidates(
    tracker: BeliefTracker,
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    global_time: float,
    artifact: SBKHMMArtifact,
    action_table: Sequence[Tuple[float, float]],
    reward_config: RewardConfig,
    dt: float,
    time_limit: float,
    max_human_speed: float,
    n_world_samples: int,
    n_iqn_quantiles: int,
    tau_upper: float,
    posterior_source: str,
    set_encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    device,
    seed_key: Tuple[int, int, int],
    gamma: float,
    action_indices: Optional[Sequence[int]] = None,
):
    """Steps 3 (shared posterior world samples) through 6 (batched score)
    of the 8-step decision procedure, given a belief tracker that is
    ALREADY current for this state -- ``decide()`` gets there by calling
    ``tracker.update()`` with a real observation; Stage 1 ranking-IL
    training gets there by restoring a tracker from a per-step snapshot
    captured during ORCA demonstration collection. Both cases end up
    with a tracker whose ``belief_for``/``entropy_for``/``track_age_for``
    are valid right now, which is all this function requires -- it is
    the ONE place ``decide()`` and Stage 1 training both call, so they
    can never silently diverge (guide.md R3-3).

    Returns (action_scores, quantiles, robot_feats, human_feats, masks,
    action_feats, rewards, not_terminal, n_actions, n_samples) -- the raw
    tensors are returned too because callers need them for the
    calibration diagnostic / for building the next MCReturnSample-
    equivalent record. NOT this function's own ``tau`` -- guide.md R4-2's
    executed-action MC-loss target needs a tau shared identically across
    every world sample row (see ``compute_executed_action_quantile_target``'s
    docstring for why this function's per-(action,sample) tau seeding is
    NOT reusable for that), so it draws its own rather than receiving one
    here.
    """
    real_beliefs = {h.track_id: tracker.belief_for(h.track_id) for h in humans}
    track_positions = {h.track_id: np.array([h.px, h.py]) for h in humans}
    track_speeds = {h.track_id: float(np.hypot(h.vx, h.vy)) for h in humans}
    track_headings = {h.track_id: float(np.arctan2(h.vy, h.vx)) for h in humans}
    samples_by_track = sample_human_next_states(
        artifact=artifact, track_beliefs=real_beliefs,
        track_positions=track_positions, track_speeds=track_speeds, track_headings=track_headings,
        n_samples=n_world_samples, dt=dt, max_human_speed=max_human_speed,
        source=posterior_source, seed=seed_key,
    )

    hyp_beliefs_per_sample: List[Dict[int, np.ndarray]] = []
    hyp_entropy_per_sample: List[Dict[int, float]] = []
    hyp_age_per_sample: List[Dict[int, int]] = []
    hyp_pred_moments_per_sample: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = []
    for sample_idx in range(n_world_samples):
        clone = tracker.clone_for_hypothetical()
        hyp_observations = {}
        for h in humans:
            sample = samples_by_track[h.track_id][sample_idx]
            hyp_observations[h.track_id] = (global_time + dt, sample.next_position)
        clone.update(hyp_observations)
        hyp_beliefs_per_sample.append({h.track_id: clone.belief_for(h.track_id) for h in humans})
        hyp_entropy_per_sample.append({h.track_id: clone.entropy_for(h.track_id) for h in humans})
        hyp_age_per_sample.append({h.track_id: clone.track_age_for(h.track_id) for h in humans})
        hyp_pred_moments_per_sample.append({h.track_id: clone.predictive_moments_for(h.track_id) for h in humans})

    n_actions = len(action_table)
    robot_feats, human_feats, masks, rewards, not_terminal, tau_seeds, action_feats = _vectorized_candidate_batch(
        robot=robot, humans=humans, action_table=action_table, samples_by_track=samples_by_track,
        hyp_beliefs_per_sample=hyp_beliefs_per_sample, hyp_entropy_per_sample=hyp_entropy_per_sample,
        hyp_age_per_sample=hyp_age_per_sample, hyp_pred_moments_per_sample=hyp_pred_moments_per_sample,
        dt=dt, time_limit=time_limit, global_time=global_time, reward_config=reward_config,
        suite_seed=seed_key[0], episode_seed=seed_key[1], decision_counter=seed_key[2],
        max_human_speed=max_human_speed, action_indices=action_indices,
    )

    tau_cpu = _stateless_tau_cpu(tau_seeds, n_iqn_quantiles, tau_upper)
    robot_feats_t = torch.tensor(robot_feats, dtype=torch.float32, device=device)
    human_feats_t = torch.tensor(human_feats, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, device=device)
    action_feats_t = torch.tensor(action_feats, dtype=torch.float32, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    not_terminal_t = torch.tensor(not_terminal, device=device)
    tau_t = tau_cpu.to(device)

    action_scores, quantiles = score_candidate_batch(
        set_encoder, value_network, action_encoder, robot_feats_t, human_feats_t, masks_t,
        action_feats_t, rewards_t, not_terminal_t, tau_t, n_actions, n_world_samples, gamma,
    )
    return (
        action_scores, quantiles, robot_feats_t, human_feats_t, masks_t, action_feats_t,
        rewards_t, not_terminal_t, n_actions, n_world_samples,
    )


class BDVLPolicy:
    def __init__(
        self,
        artifact: SBKHMMArtifact,
        set_encoder: SetEncoder,
        value_network: IQNValueNetwork,
        action_encoder: ActionEncoder,
        action_table: Sequence[Tuple[float, float]],
        reward_config: RewardConfig,
        dt: float,
        time_limit: float,
        max_human_speed: float,
        cvar_alpha: float = 0.20,
        n_world_samples: int = 8,
        n_iqn_quantiles: int = 16,
        posterior_source: str = "full",
        risk_neutral: bool = False,
        collect_calibration: bool = False,
    ):
        # guide.md 5.4: the risk-neutral ablation reuses the identical
        # network and pipeline, changing ONLY the tau sampling range
        # from Uniform(0, cvar_alpha) to Uniform(0, 1) -- never a
        # separately-defined network or a post-hoc reweighting.
        self.risk_neutral = risk_neutral
        self.collect_calibration = bool(collect_calibration)
        if len(action_table) != 80:
            raise PolicyError(f"expected 80 frozen actions, got {len(action_table)}")
        self.artifact = artifact
        self.set_encoder = set_encoder
        self.value_network = value_network
        self.action_encoder = action_encoder
        self.action_table = tuple(action_table)
        self.reward_config = reward_config
        self.dt = dt
        self.time_limit = time_limit
        self.max_human_speed = max_human_speed
        self.cvar_alpha = cvar_alpha
        self.n_world_samples = n_world_samples
        self.n_iqn_quantiles = n_iqn_quantiles
        self.posterior_source = posterior_source
        self.gamma = float(FROZEN_VALUES["gamma"])
        self.belief_tracker = BeliefTracker(artifact)
        self._decision_counter = 0
        self.last_seed_key: Optional[Tuple[int, int, int]] = None
        self.last_predicted_return_samples: Optional[np.ndarray] = None
        self.last_predicted_quantile_values: Optional[np.ndarray] = None
        # Input tensors built inside decide() must land on whatever
        # device the networks actually live on (A11: exercised GPU
        # latency benchmarking surfaced this -- CPU-only inputs against
        # a .to('cuda') network crashed with a device-mismatch error).
        self.device = next(set_encoder.parameters()).device

    def reset_episode_stats(self) -> None:
        """Required by CrowdNav's Explorer convention (guide.md A0.8:
        new policy must implement this identically named method) --
        clears ALL track state and the RNG counter used for
        common-random-number seeding, so nothing leaks across episodes."""
        self.belief_tracker.reset()
        self._decision_counter = 0
        self.last_seed_key = None
        self.last_predicted_return_samples = None
        self.last_predicted_quantile_values = None

    def next_seed_key(self, suite_seed: int, episode_seed: int) -> Tuple[int, int, int]:
        """R4-2R-2 fix (2026-08-10, guide.md "R4-2R-2 -- 保存真实posterior
        seed"): the ONE place ``_decision_counter`` advances, so every
        caller that needs a fresh, real seed_key -- a genuine live
        ``decide()`` call OR an exploratory action that skips scoring but
        must still consume exactly one counter tick -- gets a value that
        is provably what THIS policy instance would use next, never a
        value reconstructed after the fact from ``step_index`` (which
        silently drifts from the real counter once any exploration step
        is interleaved with greedy decisions, since only ``decide()``
        used to advance the counter)."""
        self._decision_counter += 1
        self.last_seed_key = (suite_seed, episode_seed, self._decision_counter)
        return self.last_seed_key

    def decide(
        self,
        robot: RobotObservation,
        humans: Sequence[HumanObservation],
        global_time: float,
        suite_seed: int,
        episode_seed: int,
    ) -> DecisionResult:
        # Step 1: canonicalize. Callers are required to already hand in
        # RobotObservation/HumanObservation (this IS the canonical
        # contract from contracts.py); no further conversion is needed
        # or performed here.
        if len(humans) > 20:
            raise PolicyError(f"{len(humans)} humans exceeds MAX_HUMANS=20")

        # Step 2: update REAL belief once.
        observations = {h.track_id: (global_time, np.array([h.px, h.py])) for h in humans}
        self.belief_tracker.update(observations)

        # Steps 3-6 (guide.md R3-3): the shared production scoring path,
        # also used by Stage 1 ranking-IL training on stored ORCA states.
        seed_key = self.next_seed_key(suite_seed, episode_seed)
        tau_upper = 1.0 if self.risk_neutral else self.cvar_alpha
        with torch.no_grad():
            (
                action_scores, quantiles, robot_feats_t, human_feats_t, masks_t, action_feats_t,
                rewards_t, not_terminal_t, n_actions, n_samples,
            ) = _score_all_candidates(
                tracker=self.belief_tracker, robot=robot, humans=humans, global_time=global_time,
                artifact=self.artifact, action_table=self.action_table, reward_config=self.reward_config,
                dt=self.dt, time_limit=self.time_limit, max_human_speed=self.max_human_speed,
                n_world_samples=self.n_world_samples, n_iqn_quantiles=self.n_iqn_quantiles,
                tau_upper=tau_upper, posterior_source=self.posterior_source,
                set_encoder=self.set_encoder, value_network=self.value_network,
                action_encoder=self.action_encoder, device=self.device,
                seed_key=seed_key, gamma=self.gamma,
            )
        cvar_per_action = action_scores.cpu().numpy()

        scores: List[ActionScore] = [
            ActionScore(action_index=i, action_vx=self.action_table[i][0], action_vy=self.action_table[i][1], cvar=float(cvar_per_action[i]))
            for i in range(n_actions)
        ]

        best = max(scores, key=lambda s: s.cvar)
        # Keep the posterior-predictive return samples for the evaluator's
        # calibration report.  This is diagnostic output only; action
        # selection still uses the exact score path above.
        quantile_samples = quantiles.detach().cpu().numpy().reshape(n_actions, n_samples, self.n_iqn_quantiles)
        rewards = rewards_t.cpu().numpy().reshape(n_actions, n_samples, 1)
        terminal_mask = not_terminal_t.cpu().numpy().reshape(n_actions, n_samples, 1)
        predicted_return_samples = rewards + self.gamma * quantile_samples * terminal_mask
        self.last_predicted_return_samples = predicted_return_samples[best.action_index].reshape(-1).astype(np.float64)
        if self.collect_calibration:
            # Calibration must inspect the network's direct output at a
            # fixed, ordered tau grid. Empirical quantiles of the sampled
            # rows cannot detect IQN crossing because sorting makes them
            # monotone by construction.
            fixed_tau = torch.tensor(
                np.broadcast_to(np.asarray(CALIBRATION_TAUS, dtype=np.float32), (n_actions * n_samples, len(CALIBRATION_TAUS))),
                dtype=torch.float32, device=self.device,
            )
            _, fixed_quantiles = score_candidate_batch(
                self.set_encoder, self.value_network, self.action_encoder, robot_feats_t, human_feats_t,
                masks_t, action_feats_t, rewards_t, not_terminal_t, fixed_tau, n_actions, n_samples, self.gamma,
            )
            self.last_predicted_quantile_values = fixed_quantiles.detach().cpu().numpy().reshape(
                n_actions, n_samples, len(CALIBRATION_TAUS)
            )[best.action_index].astype(np.float64)
        return DecisionResult(
            chosen_action_index=best.action_index,
            chosen_action=(best.action_vx, best.action_vy),
            all_scores=tuple(scores),
        )



def _stateless_tau_cpu(tau_seeds: np.ndarray, n_quantiles: int, upper: float) -> torch.Tensor:
    """Generate deterministic per-row tau without one RNG object per row.

    One formal decision previously created 2,560 ``torch.Generator``
    instances. This SplitMix-style counter transform preserves the frozen
    per-row seed contract and uniform support while making generation one
    vectorized CPU operation. The exact pseudo-random stream changes across
    this implementation, so old checkpoints are not claimed bit-compatible;
    same seed remains reproducible.
    """
    seeds = np.asarray(tau_seeds, dtype=np.uint64).reshape(-1, 1)
    indices = np.arange(n_quantiles, dtype=np.uint64).reshape(1, -1)
    z = seeds + np.uint64(0x9E3779B97F4A7C15) + indices * np.uint64(0xD1B54A32D192ED03)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    uniforms = (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    uniforms = np.minimum(np.maximum(uniforms, np.finfo(np.float64).tiny), 1.0 - np.finfo(np.float64).eps)
    return torch.from_numpy((uniforms * float(upper)).astype(np.float32))


class BayesianDVLPolicy:
    """CrowdNav-facing adapter (guide.md 8.1/A10). Implements the same
    duck-typed interface as ``crowd_sim.envs.policy.policy.Policy``
    (trainable/phase/model/device/env attributes, ``configure``/
    ``predict``) plus the explicit ``expects_joint_state=True`` contract
    that ``Robot.act()`` now checks first (see the one permitted edit
    to ``crowd_sim/envs/utils/robot.py``).

    Does NOT subclass ``crowd_sim.envs.policy.policy.Policy`` --
    guide.md 8.2 forbids modifying files outside the two explicitly
    listed exceptions, and duck-typing avoids importing/depending on
    that module changing this policy's own behavior. ``Robot.act()``
    only requires ``.predict()`` and (for the JointState branch)
    nothing else.
    """

    expects_joint_state = True
    kinematics = "holonomic"
    multiagent_training = True

    def __init__(self, bdvl_policy: Optional[BDVLPolicy] = None, suite_seed: int = 0):
        # Zero-arg construction (matching policy_factory's `policy_class()`
        # pattern, guide.md A10) is supported: the real BDVLPolicy is
        # built lazily from `configure(config)`'s `[bayesian_dvl]`
        # section rather than required as a constructor argument.
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        self.env = None
        self.bdvl_policy = bdvl_policy
        self.suite_seed = int(suite_seed)
        self._episode_index = 0
        self._episode_seed: Optional[int] = None
        self._global_time = 0.0
        self.last_action_index = None
        self.last_decision = None

    def configure(self, config) -> None:
        """Reads `[bayesian_dvl]` from the passed env/policy config:
            registry_path   = path to a frozen bayesian_dvl_registry.json
            artifact_path   = path to a frozen SBK-HMM artifact json
            checkpoint_path = path to a COMPOSED checkpoint (see
                              save_composed_checkpoint/load_composed_checkpoint):
                              one file holding encoder+value state dicts
                              AND the registry/artifact/action-table/
                              reward hashes it was trained against.
            smoke_mode      = bool, default false. Required to be
                              explicitly true to run with a freshly-
                              initialized (UNTRAINED) network when
                              checkpoint_path is absent -- B6 fix
                              (independent audit, 2026-08-06): formal/
                              evaluation configuration must not silently
                              fall back to random weights.
            eval_phase      = one of "train"/"validation"/"formal",
                              selects the frozen world-sample/quantile
                              budget for that phase (guide.md 5.4's
                              table) -- default "train".
        """
        if self.bdvl_policy is not None:
            return  # already constructed (e.g. directly by a test), don't rebuild
        if not config.has_section("bayesian_dvl"):
            raise PolicyError("env/policy config has no [bayesian_dvl] section")
        section = config["bayesian_dvl"]

        from crowd_nav.bayesian_dvl.config import load_and_validate_registry
        from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact
        from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder
        from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork
        from crowd_nav.bayesian_dvl.transition import RewardConfig

        registry = load_and_validate_registry(section.get("registry_path"))
        artifact = SBKHMMArtifact.load(section.get("artifact_path"), expect_tier=None if section.getboolean("smoke_mode", fallback=False) else "production")
        action_table = [tuple(a) for a in registry["action_table"]]
        frozen = registry["frozen_values"]
        if artifact.dt != frozen["dt"]:
            raise PolicyError(f"artifact.dt={artifact.dt} != registry frozen dt={frozen['dt']}")
        reward_config = RewardConfig(
            success_reward=frozen["success_reward"], collision_penalty=frozen["collision_penalty"],
            timeout_penalty=frozen["timeout_penalty"], progress_reward=frozen["progress_reward"],
            time_penalty=frozen["time_penalty"], stand_penalty=frozen["stand_penalty"],
            stand_speed_threshold=frozen["stand_speed_threshold"],
            discomfort_distance=frozen["discomfort_distance"],
            discomfort_penalty_factor=frozen["discomfort_penalty_factor"],
        )

        v_min, v_max = derive_return_bounds(frozen)
        encoder = SetEncoder()
        action_encoder = ActionEncoder()
        value_network = IQNValueNetwork(
            state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim,
            v_min=v_min, v_max=v_max,
        )
        checkpoint_path = section.get("checkpoint_path", fallback=None)
        smoke_mode = section.getboolean("smoke_mode", fallback=False)
        if checkpoint_path:
            manifest = load_composed_checkpoint(
                checkpoint_path, encoder, value_network, action_encoder,
                expected_artifact_sha256=artifact.content_sha256(),
                expected_registry_content_sha256=registry["content_sha256"],
                expected_action_grid_hash=registry["action_grid_hash"],
                require_formal_provenance=not smoke_mode,
            )
            if manifest["action_grid_hash"] != registry["action_grid_hash"]:
                raise PolicyError(
                    f"checkpoint was trained against a different action grid "
                    f"({manifest['action_grid_hash']}) than this registry ({registry['action_grid_hash']})"
                )
            if manifest["registry_content_sha256"] != registry["content_sha256"]:
                raise PolicyError("checkpoint's recorded registry hash does not match the loaded registry")
        elif not smoke_mode:
            raise PolicyError(
                "no checkpoint_path given and smoke_mode is not explicitly true -- "
                "refusing to silently run an untrained network (B6 fix)"
            )
        encoder.eval()
        value_network.eval()
        action_encoder.eval()

        eval_phase = section.get("eval_phase", fallback="train")
        phase_budgets = {
            "train": (frozen["world_samples_train"], frozen["iqn_train_quantiles"]),
            "validation": (frozen["world_samples_validation"], frozen["iqn_quantiles_validation"]),
            "formal": (frozen["world_samples_formal"], frozen["iqn_quantiles_formal"]),
        }
        if eval_phase not in phase_budgets:
            raise PolicyError(f"eval_phase must be one of {sorted(phase_budgets)}, got {eval_phase!r}")
        n_world_samples, n_iqn_quantiles = phase_budgets[eval_phase]

        self.bdvl_policy = BDVLPolicy(
            artifact=artifact, set_encoder=encoder, value_network=value_network, action_encoder=action_encoder,
            action_table=action_table, reward_config=reward_config,
            dt=frozen["dt"], time_limit=frozen["time_limit"], max_human_speed=frozen["max_human_speed"],
            cvar_alpha=frozen["cvar_alpha"],
            n_world_samples=n_world_samples, n_iqn_quantiles=n_iqn_quantiles,
        )
        self.suite_seed = int(section.get("suite_seed", fallback=0))

    def set_phase(self, phase) -> None:
        self.phase = phase

    def set_device(self, device) -> None:
        # B6 fix (independent audit, 2026-08-06): previously this only
        # recorded `device` on self without moving the composed
        # encoder/value network there, so a caller doing set_device('cuda')
        # would silently keep computing on CPU.
        self.device = device
        if self.bdvl_policy is not None:
            self.bdvl_policy.set_encoder.to(device)
            self.bdvl_policy.value_network.to(device)
            self.bdvl_policy.action_encoder.to(device)
            self.bdvl_policy.device = next(self.bdvl_policy.set_encoder.parameters()).device

    def set_env(self, env) -> None:
        self.env = env

    def reset_episode_stats(
        self,
        *,
        suite_seed: Optional[int] = None,
        episode_seed: Optional[int] = None,
    ) -> None:
        """Reset episode state and bind the deterministic sampling identity.

        Formal training/evaluation callers must pass both seeds explicitly.
        The no-argument form remains available for CrowdNav's legacy Explorer
        hook and allocates a distinct local episode identity.
        """
        if (suite_seed is None) != (episode_seed is None):
            raise PolicyError("suite_seed and episode_seed must be provided together")
        self.bdvl_policy.reset_episode_stats()
        self._episode_index += 1
        if suite_seed is None:
            self._episode_seed = self.suite_seed * 100000 + self._episode_index
        else:
            self.suite_seed = int(suite_seed)
            self._episode_seed = int(episode_seed)
        self._global_time = 0.0
        self.last_action_index = None
        self.last_decision = None

    def predict(self, state):
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation

        if self._episode_seed is None:
            raise PolicyError(
                "episode identity is unset; call reset_episode_stats() before predict()"
            )

        self_state = state.self_state
        robot = RobotObservation.from_full_state(self_state)
        humans = [
            HumanObservation(
                track_id=i, px=float(h.px), py=float(h.py),
                vx=float(h.vx), vy=float(h.vy), radius=float(h.radius),
            )
            for i, h in enumerate(state.human_states)
        ]

        result = self.bdvl_policy.decide(
            robot, humans, global_time=self._global_time,
            suite_seed=self.suite_seed, episode_seed=self._episode_seed,
        )
        if self.time_step is not None:
            self._global_time += self.time_step
        self.last_action_index = int(result.chosen_action_index)
        self.last_decision = result
        vx, vy = result.chosen_action
        return ActionXY(vx, vy)
