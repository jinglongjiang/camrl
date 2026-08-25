"""Registered BDVL self-tests (guide.md section 9, per-stage acceptance).

Run: python3 -m crowd_nav.bayesian_dvl.selftest
Each test function name starts with ``test_`` and is auto-discovered.
Failures raise AssertionError; the runner prints a pass/fail count and
exits non-zero on any failure, matching the project's existing selftest
convention (see crowd_nav/bayesian_brne/selftest.py).
"""
from __future__ import annotations
import copy
import inspect
import json
import random
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch.nn as nn
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    ACTION_FEATURE_DIM, ActionGridSpec, FEATURE_SCHEMA_V7, FROZEN_VALUES,
    NORMALIZATION_CONSTANTS, PROGRESS_REWARD_NORMALIZED_K,
    TRAINING_CONTRACT_V9_FAILED_DEMOS_MC_ONLY,
    derive_return_bounds,
)
from crowd_nav.bayesian_dvl.contracts import (
    MAX_HUMANS,
    CanonicalObservation,
    HumanObservation,
    RobotObservation,
    canonical_observation_is_permutation_invariant_content,
    canonicalize,
)
from crowd_nav.bayesian_dvl.intent_tracker import (
    CandidateGoal, GoalIntentTracker, IntentBeliefBank, IntentTrackerError, _unit as _intent_unit,
)
from crowd_nav.bayesian_dvl.scene_candidates import (
    PublicDestination, PublicScene, SceneCandidatesError, circle_scene, junction_scene, make_candidate_fn,
    square_scene,
)
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    CANDIDATE_FEATURE_DIM, FEATURE_SCHEMA_V7, HUMAN_FEATURE_DIM_V7, HUMAN_SCALAR_DIM_V7,
    MAX_CANDIDATE_GOALS,
)
from crowd_nav.bayesian_dvl.intent_policy import (
    CHECKPOINT_SCHEMA_V8, HUMAN_FEATURE_DIM_V7, IntentPolicyError, build_intent_human_feature_batch,
    load_intent_checkpoint, remaining_time_fraction as intent_remaining_time_fraction,
    save_intent_checkpoint, score_candidates_v5,
)
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
from crowd_nav.bayesian_dvl.junction_scenario import (
    junction_crowd_role_of_seed,
    AMBIGUOUS_TRACK_INDEX, CROWD_HELDOUT_BACKGROUND_SPEED_RANGE, CROWD_HELDOUT_EXIT_LEFT, CROWD_HELDOUT_EXIT_RIGHT,
    CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE, CROWD_TRAIN_BACKGROUND_SPEED_RANGE,
    EXIT_LEFT, EXIT_RIGHT, JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_HUMAN_NUM, JUNCTION_CROWD_TRAIN_SEEDS,
    JUNCTION_HELDOUT_SEEDS, JUNCTION_TRAIN_SEEDS, JUNCTION_WAYPOINT, JunctionCrowdEpisodeConfig,
    JunctionEpisodeConfig, JunctionScenarioError, PEDESTRIAN_SPEED_RANGE, WAYPOINT_RADIUS,
    build_junction_crowd_episode, build_junction_episode, crowd_exit_position, exit_position,
    maybe_reveal_crowd_exit, maybe_reveal_exit, public_junction_crowd_scene, public_junction_scene,
)
from crowd_nav.bayesian_dvl.intent_train import (
    RawEpisode, RawStep,
    FORMAL_EVAL_HELDOUT_SEEDS, FORMAL_SIX_SCENARIOS, EMAModel, IntentReplay, IntentTrainError,
    IntentBatch, _make_crowd_env, batch_to_tensors, build_formal_scenario_env, collect_online_episode,
    run_il_update,
    collect_orca_episode, collect_raw_orca_episode, compute_mc_returns, materialize_arm_transitions,
    run_formal_scenario_episode, run_formal_six_scenario_evaluation,
    run_online_training_step, summarize_scenario_results,
    train_step as intent_train_step,
)
import torch
from crowd_nav.bayesian_dvl.set_encoder import ACTION_FEATURE_DIM, ActionEncoder, HUMAN_FEATURE_DIM, ROBOT_FEATURE_DIM, SetEncoder
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork, quantile_huber_loss, quantile_huber_loss_hand_check
from crowd_nav.bayesian_dvl import normalization as norm
from crowd_nav.bayesian_dvl.iqn import expert_ranking_loss
from crowd_nav.bayesian_dvl.ranking import (
    RankingDemoSample, RankingError, build_action_equivalence_class, derive_action_equivalence_tolerance,
    nearest_action_index,
)
from crowd_nav.bayesian_dvl.statistics import (
    CALIBRATION_TAUS, EpisodeRecord, StatisticsError, check_cvar_promotion_gate, check_r2_gate,
    join_paired_episodes, paired_difference_bootstrap, quantile_calibration_metrics,
    stratified_calibration_metrics, CalibrationAccumulator,
    suite_seed_block_bootstrap,
)
from crowd_nav.bayesian_dvl.evaluate import (
    EvaluatorError, assert_no_duplicate_or_missing, assert_role_allowed_for_checkpoint_selection,
    deterministic_records_sha256, read_episode_records_csv, run_paired_evaluation, write_episode_records_csv,
)
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_CONFIG_PATH = REPO_ROOT / "crowd_nav" / "configs" / "env.config"
def _make_full_state(rng, **overrides):
    from crowd_sim.envs.utils.state import FullState

    vals = dict(
        px=rng.uniform(-5, 5), py=rng.uniform(-5, 5),
        vx=rng.uniform(-1, 1), vy=rng.uniform(-1, 1),
        radius=0.3, gx=rng.uniform(-5, 5), gy=rng.uniform(-5, 5),
        v_pref=1.0, theta=rng.uniform(-np.pi, np.pi),
    )
    vals.update(overrides)
    return FullState(**vals)
def _make_observable_state(rng, **overrides):
    from crowd_sim.envs.utils.state import ObservableState

    vals = dict(
        px=rng.uniform(-5, 5), py=rng.uniform(-5, 5),
        vx=rng.uniform(-1, 1), vy=rng.uniform(-1, 1), radius=0.3,
    )
    vals.update(overrides)
    return ObservableState(**vals)
def _track_from_kinematics(dt, n_steps, speed0, accel, omega, heading0=0.0):
    positions = [np.array([0.0, 0.0])]
    speed, heading = speed0, heading0
    for _ in range(n_steps):
        speed = max(speed + accel * dt, 0.0)
        heading = heading + omega * dt
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        positions.append(positions[-1] + vel * dt)
    return Track(positions=np.array(positions), dt=dt)
MODE_NAMES_LOOKUP = ("CV", "ACC", "DECEL", "TURN_L", "TURN_R")
def _fixture_artifact(dt=0.25) -> SBKHMMArtifact:
    rng = np.random.default_rng(7)
    trans_counts = np.abs(rng.normal(size=(N_MODES, N_MODES))) + 1.0
    np.fill_diagonal(trans_counts, trans_counts.diagonal() + 8.0)  # sticky
    mu = np.array([
        [0.0, 0.0], [0.6, 0.0], [-0.6, 0.0], [0.0, 0.8], [0.0, -0.8],
    ])
    kappa = np.full(N_MODES, 5.0)
    nu = np.full(N_MODES, 6.0)  # > d-1=1
    psi = np.tile(np.eye(2) * 1.0, (N_MODES, 1, 1))
    return SBKHMMArtifact(
        dt=dt, transition_counts=trans_counts,
        niw_mu=mu, niw_kappa=kappa, niw_nu=nu, niw_psi=psi,
        initial_counts=np.array([6.0, 1.0, 1.0, 1.0, 1.0]), n_iterations=1, converged=True,
        log_likelihood_history=(0.0,), train_data_sha256="fixture", tier="production",
    )
def _positions_from_kinematics(dt, n_steps, speed0, accel, omega, heading0=0.0):
    positions = [np.array([0.0, 0.0])]
    speed, heading = speed0, heading0
    for _ in range(n_steps):
        speed = max(speed + accel * dt, 0.0)
        heading = heading + omega * dt
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        positions.append(positions[-1] + vel * dt)
    return positions
def _sample_common_args(artifact, belief, n_samples=200, source="full", seed=(1, 2, 3)):
    return dict(
        artifact=artifact,
        track_beliefs={0: belief},
        track_positions={0: np.array([0.0, 0.0])},
        track_speeds={0: 1.0},
        track_headings={0: 0.0},
        n_samples=n_samples,
        dt=0.25,
        max_human_speed=2.0,
        source=source,
        seed=seed,
    )
def _random_crowd(rng_seed, n_humans, max_humans=20, batch=2):
    torch.manual_seed(rng_seed)
    robot_features = torch.randn(batch, ROBOT_FEATURE_DIM)
    human_features = torch.zeros(batch, max_humans, HUMAN_FEATURE_DIM)
    human_features[:, :n_humans, :] = torch.randn(batch, n_humans, HUMAN_FEATURE_DIM)
    mask = torch.zeros(batch, max_humans, dtype=torch.bool)
    mask[:, :n_humans] = True
    return robot_features, human_features, mask
def _tiny_policy(posterior_source="full", risk_neutral=False):
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    reward_cfg = _REWARD_CFG
    return BDVLPolicy(
        artifact=artifact, set_encoder=encoder, value_network=net, action_encoder=action_encoder, action_table=action_table,
        reward_config=reward_cfg, dt=0.25, time_limit=35.0, max_human_speed=2.0,
        n_world_samples=2, n_iqn_quantiles=4, posterior_source=posterior_source, risk_neutral=risk_neutral,
    )
def _fixture_robot_and_humans():
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [
        HumanObservation(track_id=0, px=2.0, py=0.1, vx=-0.5, vy=0.0, radius=0.3),
        HumanObservation(track_id=1, px=1.0, py=-1.0, vx=0.0, vy=0.3, radius=0.3),
    ]
    return robot, humans
def _dummy_transition(tag: float) -> Transition:
    return Transition(
        robot_features=np.full(4, tag, dtype=np.float32),
        human_features=np.zeros((20, 4), dtype=np.float32),
        human_mask=np.zeros(20, dtype=bool),
        belief=np.zeros(5, dtype=np.float32),
        action_index=0, reward=tag, done=False,
        next_robot_features=np.full(4, tag, dtype=np.float32),
        next_human_features=np.zeros((20, 4), dtype=np.float32),
        next_human_mask=np.zeros(20, dtype=bool),
        next_belief=np.zeros(5, dtype=np.float32),
        artifact_sha256="fixture",
    )
def _write_registry_and_artifact(tmp_dir):
    registry = build_frozen_registry(str(ENV_CONFIG_PATH))
    registry_path = Path(tmp_dir) / "registry.json"
    write_registry(registry, str(registry_path))

    artifact = _fixture_artifact()
    artifact_path = Path(tmp_dir) / "artifact.json"
    artifact.save(str(artifact_path))
    return registry, registry_path, artifact_path
class _FakeConfigSection(dict):
    def get(self, key, fallback=None):
        return super().get(key, fallback)

    def getboolean(self, key, fallback=False):
        value = super().get(key, fallback)
        return bool(value) if not isinstance(value, str) else value.lower() == "true"
class _FakeConfig(dict):
    def has_section(self, name):
        return name in self

    def __getitem__(self, key):
        return _FakeConfigSection(super().__getitem__(key))
def _episode(method, scenario, profile, suite_seed, episode_seed, outcome="success", steps=10):
    return EpisodeRecord(method=method, scenario=scenario, profile=profile, suite_seed=suite_seed, episode_seed=episode_seed, outcome=outcome, steps=steps)
def _stress_policy():
    return _tiny_policy()  # small dims: fast enough to exercise many crowd sizes in CI
def _fixture_mc_capture_state(action_table, suite_seed=7, episode_seed=1, step_index=0):
    # R4-2/R4-2R-2: build_mc_return_samples now expects the raw dict
    # train_bdvl._capture_state produces (robot/humans/belief snapshot/
    # global_time/executed_action_features/posterior_seed_key), not
    # pre-flattened arrays, and copies posterior_seed_key VERBATIM
    # (real captured value, not reconstructed from step_index).
    robot = RobotObservation(px=0.0, py=0.0, vx=0.3, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = (HumanObservation(track_id=0, px=2.0, py=0.2, vx=-0.4, vy=0.1, radius=0.3),)
    return {
        "robot": robot, "humans": humans, "global_time": 0.0, "belief_tracker_snapshot": {},
        "executed_action_features": compute_action_features_array(robot, np.asarray(action_table, dtype=np.float64))[0],
        "posterior_seed_key": (suite_seed, episode_seed, step_index + 1),
    }
def _fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=0, target_return=0.5) -> RankingDemoSample:
    robot = RobotObservation(px=0.0, py=0.0, vx=0.3, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = (HumanObservation(track_id=0, px=2.0, py=0.2, vx=-0.4, vy=0.1, radius=0.3),)
    tracker = BeliefTracker(artifact)
    tracker.update({0: (1.0, np.array([humans[0].px, humans[0].py]))})
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    tolerance = derive_action_equivalence_tolerance(action_table)
    expert_indices = build_action_equivalence_class(0.9, 0.05, action_table, tolerance)
    executed_index = nearest_action_index(0.9, 0.05, action_table)
    executed_feats = compute_action_features_array(robot, np.asarray(action_table, dtype=np.float64))[executed_index]
    return RankingDemoSample(
        robot=robot, humans=humans, global_time=1.0,
        belief_tracker_snapshot=copy.deepcopy(tracker._tracks),
        expert_action_indices=expert_indices,
        executed_action_index=executed_index, executed_action_features=executed_feats,
        posterior_seed_key=(0, episode_seed, step_index + 1),
        target_return=target_return,
        artifact_sha256=artifact.content_sha256(), episode_seed=episode_seed, step_index=step_index,
        outcome="success", source_role="demo",
    )
def _fixture_mc_return_sample(artifact, action_table, episode_seed=2, step_index=0, target_return=0.2, outcome="success", executed_action_index=3) -> MCReturnSample:
    robot = RobotObservation(px=0.0, py=0.0, vx=0.3, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = (HumanObservation(track_id=0, px=2.0, py=0.2, vx=-0.4, vy=0.1, radius=0.3),)
    tracker = BeliefTracker(artifact)
    tracker.update({0: (1.0, np.array([humans[0].px, humans[0].py]))})
    executed_feats = compute_action_features_array(robot, np.asarray(action_table, dtype=np.float64))[executed_action_index]
    return MCReturnSample(
        robot=robot, humans=humans, global_time=1.0,
        belief_tracker_snapshot=copy.deepcopy(tracker._tracks),
        executed_action_index=executed_action_index, executed_action_features=executed_feats,
        posterior_seed_key=(7, episode_seed, step_index + 1),
        target_return=target_return, artifact_sha256=artifact.content_sha256(),
        episode_seed=episode_seed, step_index=step_index, outcome=outcome, source_role="online",
    )
def _fixture_selection_result(success_rate, timeout_rate=0.0, collision_rate=0.0, mean_negative_alignment_fraction=0.0, mean_path_ratio=1.0) -> dict:
    return {
        "success_rate": success_rate, "timeout_rate": timeout_rate, "collision_rate": collision_rate,
        "mean_negative_alignment_fraction": mean_negative_alignment_fraction, "mean_path_ratio": mean_path_ratio,
    }
def _fixture_gate_result(checkpoint: str, passed: bool, **score) -> dict:
    score.setdefault("success_rate", 0.90 if passed else 0.50)
    score.setdefault("collision_rate", 0.0)
    score.setdefault("timeout_rate", 0.10 if passed else 0.50)
    return {"checkpoint": checkpoint, "r2_gate": {"passed": passed, "reasons": [] if passed else ["fixture fail"]}, **score}
def _fixture_bimodal_trap_artifact() -> SBKHMMArtifact:
    # Near-deterministic (huge kappa/nu, tiny psi), near-immovable modes
    # (huge diagonal transition stickiness) ACC/DECEL pair -- deliberately
    # engineered so the ACC and DECEL branches are each individually
    # well-clear of a collision (they cross the danger point at different
    # times than the robot occupies it), but their midpoint (what
    # posterior_mean substitutes for actually sampling a mode -- constant
    # velocity, since 0.5*(+2.0)+0.5*(-2.0)=0) arrives exactly on time
    # for a collision.
    rng = np.random.default_rng(7)
    trans_counts = np.abs(rng.normal(size=(N_MODES, N_MODES))) + 0.5
    np.fill_diagonal(trans_counts, trans_counts.diagonal() + 1e7)
    mu = np.array([[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 0.8], [0.0, -0.8]])
    kappa = np.full(N_MODES, 1e6)
    nu = np.full(N_MODES, 1e6)
    psi = np.tile(np.eye(2) * 1e-7, (N_MODES, 1, 1))
    return SBKHMMArtifact(
        dt=0.25, transition_counts=trans_counts, niw_mu=mu, niw_kappa=kappa, niw_nu=nu, niw_psi=psi,
        initial_counts=np.array([6.0, 1.0, 1.0, 1.0, 1.0]), n_iterations=1, converged=True,
        log_likelihood_history=(0.0,), train_data_sha256="fixture", tier="production",
    )
def _fixture_candidate(action_index, collision_prob, lower_tail_clearance, expected_progress, control_cost=0.0) -> CounterfactualCandidateResult:
    return CounterfactualCandidateResult(
        action_index=action_index, collision_prob=collision_prob, lower_tail_clearance=lower_tail_clearance,
        expected_progress=expected_progress, control_cost=control_cost,
    )
def _fixture_risk_records_sufficient() -> list:
    # guide.md R4-4R gate thresholds: >=100 states from >=30 episodes,
    # each profile >=5, >=5% disagreement vs mean/cv, turn+stop_or_slow
    # each from >=1 episode.
    profiles = ["nominal", "train_nonstationary", "train_non_reciprocal"]
    n_episodes = 36
    n_states_per_episode = 4
    idx = 0
    records: List[RiskOpportunityRecord] = []
    for ep in range(n_episodes):
        profile = profiles[ep % 3]
        for s in range(n_states_per_episode):
            disagree_mean = idx % 10 == 0
            disagree_cv = idx % 9 == 0
            if ep == 0 and s == 0:
                action_type = "turn"
            elif ep == 1 and s == 0:
                action_type = "stop_or_slow"
            else:
                action_type = "straight"
            records.append(RiskOpportunityRecord(
                episode_seed=700000 + ep, decision_seed=s + 1, profile=profile,
                belief_entropy_bin="medium", action_type=action_type,
                full_top_action=0, mean_top_action=(1 if disagree_mean else 0), cv_top_action=(2 if disagree_cv else 0),
                agree_with_mean=not disagree_mean, agree_with_cv=not disagree_cv,
                internal_regret_vs_mean=(0.1 if disagree_mean else 0.0), internal_regret_vs_cv=(0.1 if disagree_cv else 0.0),
            ))
            idx += 1
    return records
def _fixture_audit_records(full_rank, mean_rank, cv_rank, n_episodes=15, n_per_episode=3, seed_base=900000) -> List[AuditRecord]:
    records: List[AuditRecord] = []
    rng = np.random.default_rng(5)
    for ep in range(n_episodes):
        for s in range(n_per_episode):
            full = MethodRegret(0.0, 0.0, 0.0, float(np.clip(full_rank + rng.normal(0, 0.01), 0.0, 1.0)), True)
            mean = MethodRegret(0.0, 0.0, 0.0, float(np.clip(mean_rank + rng.normal(0, 0.01), 0.0, 1.0)), True)
            cv = MethodRegret(0.0, 0.0, 0.0, float(np.clip(cv_rank + rng.normal(0, 0.01), 0.0, 1.0)), True)
            records.append(AuditRecord(
                episode_seed=seed_base + ep, decision_seed=s + 1, profile="train_non_reciprocal",
                full=full, mean=mean, cv=cv,
            ))
    return records
_INTENT_CANDS = [
    CandidateGoal("left", ((0.0, 4.0), (-2.5, 6.0))),
    CandidateGoal("straight", ((0.0, 4.0), (0.0, 6.5))),
    CandidateGoal("right", ((0.0, 4.0), (2.5, 6.0))),
]
def _run_intent_track(true_route, n=24, speed=1.0, dt=0.25):
    tracker = GoalIntentTracker(_INTENT_CANDS, dt=dt, speed=speed)
    pos = np.array([0.0, 0.0])
    idx = 0
    beliefs = []
    for _ in range(n):
        tracker.update(pos)
        beliefs.append(tracker.belief())
        while idx < len(true_route) and np.linalg.norm(np.array(true_route[idx]) - pos) <= 0.35:
            idx += 1
        v = np.zeros(2) if idx >= len(true_route) else speed * _intent_unit(np.array(true_route[idx]) - pos)
        pos = pos + v * dt
    return np.array(beliefs)
def _bank_candidate_fn(track_id, first_position):
    # PUBLIC inputs only (a stable id + an observed entry position); no Human,
    # no hidden gx/gy. Returns the same public junction candidates for any track.
    assert np.asarray(first_position).shape == (2,)
    return _INTENT_CANDS
def _feed_bank_track(dt=0.25):
    # a pedestrian approaching then turning left, as a list of observed positions
    return [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (-0.6, 4.5), (-1.2, 5.0), (-1.8, 5.5)]
def _junction_bank_2exit():
    js = junction_scene((0.0, 4.0), [("left", (-2.5, 6.0)), ("right", (2.5, 6.0))])
    return IntentBeliefBank(make_candidate_fn(js), dt=0.25, speed=1.0)
def _env_config_path() -> Path:
    return REPO_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"


__all__ = [
    'ACTION_FEATURE_DIM',
    'AMBIGUOUS_TRACK_INDEX',
    'ActionEncoder',
    'ActionGridSpec',
    'CALIBRATION_TAUS',

    'CHECKPOINT_SCHEMA_V8',
    'CROWD_HELDOUT_BACKGROUND_SPEED_RANGE',
    'CROWD_HELDOUT_EXIT_LEFT',
    'CROWD_HELDOUT_EXIT_RIGHT',
    'CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE',
    'CROWD_TRAIN_BACKGROUND_SPEED_RANGE',
    'CalibrationAccumulator',
    'CandidateGoal',
    'CanonicalObservation',
    'DistributionalValueModel',
    'EMAModel',
    'ENV_CONFIG_PATH',
    'EXIT_LEFT',
    'EXIT_RIGHT',
    'EpisodeRecord',
    'EvaluatorError',
    'FEATURE_SCHEMA_V7',
    'FORMAL_EVAL_HELDOUT_SEEDS',
    'FORMAL_SIX_SCENARIOS',
    'FROZEN_VALUES',
    'GoalIntentTracker',
    'HUMAN_FEATURE_DIM',
    'HUMAN_FEATURE_DIM_V7', 'junction_crowd_role_of_seed', 'HUMAN_FEATURE_DIM_V7', 'HUMAN_SCALAR_DIM_V7',
    'CANDIDATE_FEATURE_DIM', 'MAX_CANDIDATE_GOALS', 'FEATURE_SCHEMA_V7',
    'HumanObservation',
    'IQNValueNetwork',
    'IntentBatch',
    'IntentBeliefBank',
    'IntentPolicyError',
    'IntentReplay',
    'IntentTrackerError',
    'IntentTrainError',
    'JUNCTION_CROWD_HELDOUT_SEEDS',
    'JUNCTION_CROWD_HUMAN_NUM',
    'JUNCTION_CROWD_TRAIN_SEEDS',
    'JUNCTION_HELDOUT_SEEDS',
    'JUNCTION_TRAIN_SEEDS',
    'JUNCTION_WAYPOINT',
    'JunctionCrowdEpisodeConfig',
    'JunctionEpisodeConfig',
    'JunctionScenarioError',
    'MAX_HUMANS',
    'MODE_NAMES_LOOKUP',
    'NORMALIZATION_CONSTANTS',
    'PEDESTRIAN_SPEED_RANGE',
    'PROGRESS_REWARD_NORMALIZED_K',
    'Path',
    'PublicDestination',
    'PublicScene',
    'REPO_ROOT',
    'ROBOT_FEATURE_DIM',
    'RankingDemoSample',
    'RankingError',
    'RawEpisode',
    'RawStep',
    'RobotObservation',
    'SceneCandidatesError',
    'SetEncoder',
    'StatisticsError',
    'TRAINING_CONTRACT_V9_FAILED_DEMOS_MC_ONLY',
    'WAYPOINT_RADIUS',
    '_FakeConfig',
    '_FakeConfigSection',
    '_INTENT_CANDS',
    '_bank_candidate_fn',
    '_dummy_transition',
    '_env_config_path',
    '_episode',
    '_feed_bank_track',
    '_fixture_artifact',
    '_fixture_audit_records',
    '_fixture_bimodal_trap_artifact',
    '_fixture_candidate',
    '_fixture_gate_result',
    '_fixture_mc_capture_state',
    '_fixture_mc_return_sample',
    '_fixture_ranking_demo_sample',
    '_fixture_risk_records_sufficient',
    '_fixture_robot_and_humans',
    '_fixture_selection_result',
    '_intent_unit',
    '_junction_bank_2exit',
    '_make_full_state',
    '_make_observable_state',
    '_make_crowd_env',
    '_positions_from_kinematics',
    '_random_crowd',
    '_run_intent_track',
    '_sample_common_args',
    '_stress_policy',
    '_tiny_policy',
    '_track_from_kinematics',
    '_write_registry_and_artifact',
    'annotations',
    'assert_no_duplicate_or_missing',
    'assert_role_allowed_for_checkpoint_selection',
    'batch_to_tensors',
    'build_action_equivalence_class',
    'build_formal_scenario_env',
    'build_intent_human_feature_batch',
    'build_junction_crowd_episode',
    'build_junction_episode',
    'canonical_observation_is_permutation_invariant_content',
    'canonicalize',
    'check_cvar_promotion_gate',
    'check_r2_gate',
    'circle_scene',
    'collect_online_episode',
    'collect_orca_episode',
    'collect_raw_orca_episode',
    'compute_mc_returns',
    'copy',
    'crowd_exit_position',
    'derive_action_equivalence_tolerance',
    'derive_return_bounds',
    'deterministic_records_sha256',
    'exit_position',
    'expert_ranking_loss',
    'inspect',
    'intent_remaining_time_fraction',
    'intent_train_step',
    'join_paired_episodes',
    'json',
    'junction_scene',
    'load_intent_checkpoint',
    'make_candidate_fn',
    'materialize_arm_transitions',
    'maybe_reveal_crowd_exit',
    'maybe_reveal_exit',
    'nearest_action_index',
    'nn',
    'norm',
    'np',
    'paired_difference_bootstrap',
    'public_junction_crowd_scene',
    'public_junction_scene',
    'quantile_calibration_metrics',
    'quantile_huber_loss',
    'quantile_huber_loss_hand_check',
    'random',
    'read_episode_records_csv',
    'run_formal_scenario_episode',
    'run_formal_six_scenario_evaluation',
    'run_il_update',
    'run_online_training_step',
    'run_paired_evaluation',
    'save_intent_checkpoint',
    'score_candidates_v5',
    'square_scene',
    'stratified_calibration_metrics',
    'suite_seed_block_bootstrap',
    'summarize_scenario_results',
    'sys',
    'tempfile',
    'torch',
    'write_episode_records_csv',
]
