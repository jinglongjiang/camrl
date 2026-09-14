"""Environment and feature runtime for the full-crowd belief-MDP.

Mamba keeps its original frozen five-nearest-pedestrian, 34-D view (matching
its pretrained checkpoint); it is used only as a history encoder that
refines the belief representation (see ``model.BeliefMDPQNetwork``). The
GDBN belief branch separately observes every pedestrian in the scenario
(``belief_num_humans``, up to 20) through the same action-conditioned
rollout already verified in the full-crowd calibration paper.

Six belief conditions share this runtime so that the network architecture
never changes across the ablation battery -- only the information reaching
it does. Every mode produces a belief vector of the SAME fixed dimension
(``2*MAX_K+5``) and the SAME candidate feature layout (6 Bayesian risk
features + 6 real action-kinematic features), regardless of the underlying
risk model's own mode count -- this was not true in the first version
(``cv``'s K=1 model produced a 7-d belief vs GDBN's 11-d, i.e. a different
network, not an ablation of one network) and is fixed here by padding.

- ``action_conditioned``: the paper's fitted K=3 GDBN, action-conditioned.
- ``cv``: ``ConstantVelocityRiskModel`` drop-in (same interface, K=1
  internally, padded to the same belief/candidate dimension).
- ``k1_belief``: the real GDBN action-conditioned pipeline (same code path
  as ``action_conditioned``, unlike ``cv``'s different model class) but
  fitted with K=1 mode instead of K=3 -- isolates whether the multi-modal
  belief itself (not just "having a GDBN at all") is what matters.
- ``state_only``: the real GDBN belief, but the *risk* portion of every
  candidate action is scored with the same neutral rollout -- action
  identity is still visible through the (unaffected) kinematic features, so
  this ablates action-conditioning of risk specifically, not all action
  information (the first version conflated the two and made all 80 Q-values
  identical).
- ``corrupted``: the real GDBN belief with pedestrian identities shuffled by
  a permutation drawn once per episode (not per step, which would average
  out to near-independent noise instead of a stable wrong mapping).
- ``no_belief``: belief vector and the six risk-derived candidate features
  are zeroed; only kinematic features and Mamba context remain. Distinct
  from ``beta=0`` on ``action_conditioned`` (which still lets the learned
  Q_R/Q_C see real Bayesian features through the encoders even though the
  explicit combination weight is zero) -- this is the only condition that
  actually removes Bayesian information from the network's inputs.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from crowd_nav.bayesian_pilot.protocol import (
    BehaviorScheduler,
    InterventionORCA,
    PROFILES,
)
from crowd_nav.belief_mdp.model import DecisionFeatures
from crowd_nav.belief_space_rl.runtime import (
    build_frozen_mamba,
    merged_policy_config,
)
from crowd_nav.contracts import (
    GRID,
    _batch_joint34_to_tokens_vectorized,
    discrete_index_to_action,
)
from crowd_nav.gdbn import GDBNIntegration
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.risk_models import ConstantVelocityRiskModel
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


BELIEF_MODES = (
    "action_conditioned",
    "cv",
    "state_only",
    "corrupted",
    "no_belief",
    "k1_belief",
)

# Canonical mode count every belief vector is padded to, regardless of the
# underlying risk model's own K (GDBN K=3, ConstantVelocityRiskModel K=1).
MAX_K = 3

RISK_FEATURE_NAMES = (
    "expected_risk",
    "tail_risk",
    "tail_excess",
    "posterior_entropy",
    "action_klda",
    "epistemic_value",
)
KINEMATIC_FEATURE_NAMES = (
    "vx",
    "vy",
    "speed",
    "goal_alignment",
    "progress",
    "turn_cost",
)
ACTION_FEATURE_NAMES = RISK_FEATURE_NAMES + KINEMATIC_FEATURE_NAMES

# Training uses PROFILES["nominal"] and PROFILES["train_nonstationary"]
# directly; formal evaluation uses PROFILES["heldout_nonstationary"]. An
# earlier version of this file defined its own "train_risk"/"decision_stress"
# profiles that copied heldout_nonstationary's duration/turn/slow ranges
# verbatim and only changed the event rate -- so "decision_stress" was a
# higher-intensity replay of the same distribution training already saw, not
# a genuinely unseen one, despite being described as a held-out stress test.
# train_nonstationary and heldout_nonstationary (defined in
# bayesian_pilot/protocol.py) differ in duration_steps, turn_degrees, and
# slow_scale, not just event rate -- that is what makes heldout_nonstationary
# an actual generalization test.

# Matches configs/env_belief_mdp.config-adjacent [eval_envs] convention from
# configs/policy_bayesian_fullcrowd_tail.config: this is the project's own
# formal six-scenario protocol, "shape:size:human_count".
SIX_SCENARIOS = {
    "baseline_circle": ("circle", 4.0, 5),
    "baseline_square": ("square", 10.0, 10),
    "dense_circle": ("circle", 4.0, 10),
    "dense_square": ("square", 10.0, 20),
    "large_circle": ("circle", 6.0, 12),
    "large_square": ("square", 14.0, 20),
}

# The single 5-person scenario used for all training data collection AND
# every in-training/post-hoc validation check (train.py's Stage 1 gate and
# DAgger convergence stopping, select_checkpoint.py's checkpoint selection).
# The other five SIX_SCENARIOS entries (10-20 people) exist only in
# evaluate.py's one-time formal test of the frozen model -- using them any
# earlier, even just to decide which checkpoint to keep, means checkpoint
# selection has already "seen" performance on the exact distribution later
# reported as zero-shot generalization.
VALIDATION_SCENARIOS = ("baseline_circle",)


def parse_scenario(name: str):
    if name in SIX_SCENARIOS:
        return SIX_SCENARIOS[name]
    shape, size, human_num = name.split(":")
    return shape, float(size), int(human_num)


def build_full_crowd_environment(
    env_config_path: str,
    scenario: str,
    robot_visible: bool = False,
):
    """Build a CrowdSim environment sized for the requested full-crowd scenario."""
    shape, size, human_num = parse_scenario(scenario)
    config = configparser.RawConfigParser()
    with open(env_config_path, "r", encoding="utf-8") as handle:
        config.read_file(handle)
    config.set("sim", "test_sim", f"{shape}_crossing")
    config.set("sim", "human_num", str(int(human_num)))
    if shape == "square":
        config.set("sim", "square_width", str(float(size)))
    else:
        config.set("sim", "circle_radius", str(float(size)))
    config.set("robot", "visible", "true" if robot_visible else "false")
    config.set("humans", "policy", "orca")

    env = CrowdSim()
    env.configure(config)
    env.phase = "test"
    robot = Robot(config, "robot")
    teacher = ORCA()
    teacher.configure(config)
    teacher.multiagent_training = True
    teacher.set_phase("test")
    robot.set_policy(teacher)
    robot.env = env
    env.set_robot(robot)
    return env, robot, teacher, config


def sort_humans_by_ttc(robot, humans):
    """Nearest-collision-time ordering, matching the production test.py path."""
    ttc_list = []
    for human in humans:
        rel_x = human.px - robot.px
        rel_y = human.py - robot.py
        rel_vx = human.vx - robot.vx
        rel_vy = human.vy - robot.vy
        dist = float(np.sqrt(rel_x ** 2 + rel_y ** 2 + 1e-6))
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
        ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10.0
        ttc_list.append((ttc, human))
    return [human for _, human in sorted(ttc_list, key=lambda item: item[0])]


def mamba_token_observation(robot, sorted_humans_top5) -> np.ndarray:
    """Robot(9) + five nearest pedestrians(5 each) = 34-D, zero-padded."""
    robot_state = robot.get_full_state().to_array()
    human_states = [
        human.get_observable_state().to_array() for human in sorted_humans_top5[:5]
    ]
    while len(human_states) < 5:
        human_states.append(np.zeros(5, dtype=np.float32))
    return np.concatenate([robot_state, *human_states]).astype(np.float32)


def belief_state_observation(robot, all_humans, num_humans: int) -> np.ndarray:
    """Robot(9) + up to ``num_humans`` pedestrians(5 each), identity-stable."""
    robot_state = robot.get_full_state().to_array()
    humans = []
    for human in all_humans[:num_humans]:
        state = human.get_observable_state()
        humans.extend([state.px, state.py, state.vx, state.vy, state.radius])
    expected = 5 * num_humans
    humans.extend([0.0] * max(0, expected - len(humans)))
    return np.concatenate(
        (robot_state, np.asarray(humans[:expected], dtype=np.float32))
    ).astype(np.float32)


# GDBNIntegration.load() sets self.K from the saved params file itself, so
# k1_belief needs its own K=1-fitted directory (produced by
# tools/fit_k1_gdbn.py from the SAME orca_demos_seq.npz as the main K=3 fit,
# per its diagnostics.json) -- passing K=1 against the K=3 params would be
# silently overridden back to K=3 by load().
DEFAULT_K1_GDBN_PARAMS = "runs/mamba_vl/gdbn_params_k1"


def build_risk_filter(
    belief_mode: str,
    K: int,
    n_particles: int,
    gdbn_params: str,
    num_humans: int,
    seed: int,
    k1_gdbn_params: str = DEFAULT_K1_GDBN_PARAMS,
):
    if belief_mode == "cv":
        return ConstantVelocityRiskModel(
            max_peds=num_humans,
            modeled_peds=num_humans,
            belief_dim=K + 2,
        )
    # action_conditioned, state_only, corrupted, no_belief all use the real
    # fitted K=3 GDBN -- only what happens to its output differs (see
    # encode()). k1_belief uses the same code path with a K=1-fitted params
    # directory instead, to isolate the effect of multi-modality itself.
    params_dir = k1_gdbn_params if belief_mode == "k1_belief" else gdbn_params
    fit_k = 1 if belief_mode == "k1_belief" else K
    return GDBNIntegration(
        K=fit_k,
        n_particles=n_particles,
        params_dir=params_dir,
        max_peds=num_humans,
        random_seed=seed,
    )


class BeliefMDPFeatureEngine:
    """Belief-conditioned decision features: GDBN belief + per-action features.

    ``context`` (the frozen Mamba history summary) feeds only the network's
    TaskEncoder (-> Q_R); ``belief``/candidate risk features feed only its
    RiskEncoder (-> Q_C) -- see ``model.BeliefMDPQNetwork``. There is no
    shared layer or joint tensor between the two paths.
    """

    def __init__(
        self,
        mamba: MambaRLPolicy,
        gdbn_params: str,
        device: torch.device,
        belief_mode: str = "action_conditioned",
        K: int = 3,
        n_particles: int = 50,
        num_humans: int = 20,
        seq_len: int = 24,
        risk_horizon: int = 5,
        safe_distance: float = 0.20,
        cvar_alpha: float = 0.80,
        pedestrian_aggregation: str = "max",
        dt: float = 0.25,
        seed: int = 2407,
        k1_gdbn_params: str = DEFAULT_K1_GDBN_PARAMS,
    ):
        if belief_mode not in BELIEF_MODES:
            raise ValueError(f"Unknown belief_mode: {belief_mode}")
        self.mamba = mamba
        self.device = device
        self.belief_mode = belief_mode
        self.num_humans = int(num_humans)
        self.seq_len = int(seq_len)
        self.risk_horizon = int(risk_horizon)
        self.safe_distance = float(safe_distance)
        self.cvar_alpha = float(cvar_alpha)
        self.pedestrian_aggregation = pedestrian_aggregation
        self.dt = float(dt)
        self._rng = np.random.default_rng(seed + 3701)
        self._corruption_permutation: Optional[np.ndarray] = None

        # `no_belief` still runs the real filter (so belief_dim bookkeeping
        # stays identical) but zeroes its contribution before it reaches the
        # network -- see encode(). `build_risk_filter` internally routes
        # `k1_belief` to a K=1-fitted params directory instead of the main
        # K=3 one, and only `cv` uses a different model class.
        self.filter = build_risk_filter(
            belief_mode, K, n_particles, gdbn_params, self.num_humans, seed,
            k1_gdbn_params=k1_gdbn_params,
        )
        self.filter_K = int(self.filter.K)
        n_actions = int(GRID["n_speeds"]) * int(GRID["n_headings"]) + int(
            bool(GRID.get("include_stop", False))
        )
        self.actions = np.asarray(
            [discrete_index_to_action(i) for i in range(n_actions)],
            dtype=np.float32,
        )
        self.history: List[np.ndarray] = []

    @property
    def belief_dim(self) -> int:
        # per-ped mean+max mode probs (padded to MAX_K), entropy mean/max,
        # klda mean/max, valid fraction.
        return 2 * MAX_K + 5

    @property
    def candidate_dim(self) -> int:
        return len(ACTION_FEATURE_NAMES)

    def reset(self):
        self.history = []
        self.filter.reset(n_peds=self.num_humans)
        if self.belief_mode == "corrupted":
            self._corruption_permutation = self._rng.permutation(self.num_humans)
        else:
            self._corruption_permutation = None

    def _mamba_context(self, token_34: np.ndarray) -> np.ndarray:
        self.history.append(token_34)
        tokens = _batch_joint34_to_tokens_vectorized(
            np.stack(self.history[-self.seq_len:])
        )
        sequence = list(tokens)
        if len(sequence) < self.seq_len:
            sequence = [sequence[0]] * (self.seq_len - len(sequence)) + sequence
        tensor = torch.as_tensor(
            np.asarray(sequence[-self.seq_len:], dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            spatial = self.mamba.spatial_encoder(tensor)
            temporal = self.mamba.temporal_encoder(spatial)
        return temporal[0, -1].float().cpu().numpy()

    def _pad_mode_probs(self, mode_probs: np.ndarray) -> np.ndarray:
        """Pad/truncate the last axis from filter_K to the canonical MAX_K."""
        if self.filter_K == MAX_K:
            return mode_probs
        if self.filter_K > MAX_K:
            return mode_probs[..., :MAX_K]
        pad_width = [(0, 0)] * (mode_probs.ndim - 1) + [(0, MAX_K - self.filter_K)]
        return np.pad(mode_probs, pad_width, mode="constant")

    def _global_belief(self, belief: np.ndarray) -> np.ndarray:
        valid = np.any(belief != 0.0, axis=-1)
        active = belief[valid] if valid.any() else belief[:1]
        mode = self._pad_mode_probs(active[:, : self.filter_K])
        entropy = active[:, self.filter_K]
        klda = active[:, self.filter_K + 1]
        return np.concatenate(
            (
                mode.mean(axis=0),
                mode.max(axis=0),
                np.asarray(
                    [
                        entropy.mean(),
                        entropy.max(),
                        klda.mean(),
                        klda.max(),
                        float(valid.mean()),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)

    def _risk_features(
        self,
        belief_state: np.ndarray,
        actions: np.ndarray,
        belief_vecs: Optional[np.ndarray],
    ) -> np.ndarray:
        candidate_count = int(len(actions))
        states = np.repeat(belief_state.reshape(1, -1), candidate_count, axis=0)
        beliefs = None
        if belief_vecs is not None:
            beliefs = np.repeat(
                belief_vecs.reshape(1, *belief_vecs.shape), candidate_count, axis=0
            )
        rollout = self.filter.predict_action_rollout_batch(
            states,
            actions,
            belief_vecs=beliefs,
            horizon=self.risk_horizon,
            dt=self.dt,
            safe_distance=self.safe_distance,
            cvar_alpha=self.cvar_alpha,
            pedestrian_aggregation=self.pedestrian_aggregation,
        )
        expected_risk = np.clip(np.asarray(rollout["risk"], dtype=np.float32), 0.0, 1.0)
        tail_risk = np.clip(
            np.asarray(rollout.get("tail_risk", rollout["risk"]), dtype=np.float32),
            0.0,
            1.0,
        )
        entropy = np.clip(np.asarray(rollout["entropy"], dtype=np.float32), 0.0, 1.0)
        klda_scale = max(float(getattr(self.filter, "klda_norm_clip", 1.0)), 1e-6)
        klda = np.clip(np.asarray(rollout["klda"], dtype=np.float32) / klda_scale, 0.0, 1.0)
        epistemic = np.clip(
            np.asarray(rollout.get("epistemic_value", entropy), dtype=np.float32), 0.0, 1.0
        )
        return np.stack(
            (
                expected_risk,
                tail_risk,
                np.maximum(tail_risk - expected_risk, 0.0),
                entropy,
                klda,
                epistemic,
            ),
            axis=-1,
        ).astype(np.float32)

    def _kinematic_features(self, robot, actions: np.ndarray) -> np.ndarray:
        full_state = robot.get_full_state()
        position = np.asarray([full_state.px, full_state.py], dtype=np.float64)
        goal = np.asarray([full_state.gx, full_state.gy], dtype=np.float64) - position
        goal_norm = max(float(np.linalg.norm(goal)), 1e-6)
        goal_unit = goal / goal_norm
        current_velocity = np.asarray([full_state.vx, full_state.vy], dtype=np.float64)
        current_norm = max(float(np.linalg.norm(current_velocity)), 1e-6)

        actions64 = actions.astype(np.float64)
        speed = np.linalg.norm(actions64, axis=-1)
        alignment = (actions64 @ goal_unit) / np.maximum(speed, 1e-6)
        progress = (actions64 @ goal_unit) * self.dt
        turn_cost = 1.0 - (actions64 @ (current_velocity / current_norm)) / np.maximum(
            speed, 1e-6
        )
        turn_cost = np.where(speed > 1e-6, turn_cost, 1.0)

        return np.stack(
            (
                actions64[:, 0],
                actions64[:, 1],
                np.clip(speed, 0.0, 1.0),
                np.clip(alignment, -1.0, 1.0),
                np.clip(progress, -0.25, 0.25) * 4.0,
                np.clip(turn_cost, 0.0, 2.0) * 0.5,
            ),
            axis=-1,
        ).astype(np.float32)

    def teacher_scores(self, robot, all_humans) -> np.ndarray:
        """Frozen strong-Mamba value-lookahead score for every discrete action.

        Replaces ORCA as the expert/teacher label source: this is the exact
        one-step lookahead the pretrained Mamba-VL policy itself would use to
        rank actions (mirrors ``belief_space_rl.runtime.BeliefFeatureEngine.
        teacher_scores``), so the "expert" the network imitates is the same
        strong navigation policy Q_R is ultimately supposed to approximate,
        not a classical planner with a different risk model (ORCA) that the
        belief branch has no reason to agree with. ``encode`` must have been
        called first this step so ``self.history`` already ends with the
        current raw 34-D token.
        """
        if not self.history:
            raise RuntimeError("encode must be called before teacher_scores")
        top5 = sort_humans_by_ttc(robot, all_humans)
        current_robot = robot.get_full_state()
        current_humans = [human.get_observable_state() for human in top5[:5]]

        # Round 12 fix: ``encode`` already appended the *current* raw token
        # to ``self.history`` before calling this. An earlier version of
        # this method excluded it (reasoning that each candidate action's
        # own next_token made it redundant), but next_token represents the
        # state *one step ahead* (t+1), not a replacement for the current
        # state (t) -- dropping the current frame fed Mamba "...older
        # history, then jump straight to t+1" with t itself missing
        # entirely. Verified on 60 real states: this changed the teacher's
        # top-1 action on 36/60 of them (60%) versus keeping t in the
        # sequence, so this was not a cosmetic difference. Include the
        # current frame; ``sequence = past_tokens + [next_token]`` below
        # already truncates to the most recent ``seq_len`` afterward.
        recent_raw = self.history[-self.seq_len:]
        past_tokens = _batch_joint34_to_tokens_vectorized(np.stack(recent_raw))

        action_sequences = []
        rewards = []
        minimum_clearances = []
        for velocity in self.actions:
            action = ActionXY(float(velocity[0]), float(velocity[1]))
            next_robot = self.mamba.propagate(current_robot, action)
            next_humans = [
                self.mamba.propagate(human, ActionXY(float(human.vx), float(human.vy)))
                for human in current_humans
            ]
            rewards.append(
                self.mamba.compute_reward(
                    next_robot, next_humans, prev_nav=current_robot, action=action
                )
            )
            minimum_clearances.append(
                min(
                    (
                        np.hypot(next_robot.px - human.px, next_robot.py - human.py)
                        - next_robot.radius
                        - human.radius
                    )
                    for human in next_humans
                )
                if next_humans
                else float("inf")
            )
            next_state = self.mamba._build_joint_state_34(next_robot, next_humans)
            next_token = _batch_joint34_to_tokens_vectorized(next_state.reshape(1, -1))[0]
            sequence = list(past_tokens) + [next_token]
            sequence = sequence[-self.seq_len:]
            if len(sequence) < self.seq_len:
                sequence = [sequence[0]] * (self.seq_len - len(sequence)) + sequence
            action_sequences.append(np.asarray(sequence, dtype=np.float32))

        tensor = torch.as_tensor(
            np.asarray(action_sequences), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            next_values = self.mamba.forward_value(tensor)
        scores = (
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            + float(self.mamba.gamma) * next_values
        )
        clearances = torch.as_tensor(minimum_clearances, dtype=torch.float32, device=self.device)
        minimum = float(self.mamba.test_min_clearance)
        risk_lambda = float(self.mamba.test_risk_lambda)
        if minimum > 0.0:
            safe = clearances >= minimum
            if safe.any():
                scores = scores.masked_fill(~safe, -1e4)
        if risk_lambda > 0.0:
            margin = minimum if minimum > 0.0 else float(self.mamba.discomfort_dist)
            scores = scores - risk_lambda * torch.clamp(margin - clearances, min=0.0)
        return scores.detach().cpu().numpy().astype(np.float32)

    def encode(self, robot, all_humans) -> DecisionFeatures:
        top5 = sort_humans_by_ttc(robot, all_humans)
        token = mamba_token_observation(robot, top5)
        belief_state = belief_state_observation(robot, all_humans, self.num_humans)

        self.filter.update(belief_state)
        if isinstance(self.filter, GDBNIntegration):
            belief_vecs = np.asarray(self.filter.get_belief_snapshot().features, dtype=np.float32)
        else:
            belief_vecs = np.asarray(self.filter.get_per_ped_belief_vec(), dtype=np.float32)

        if self.belief_mode == "corrupted" and self._corruption_permutation is not None:
            belief_vecs = belief_vecs[self._corruption_permutation]

        if self.belief_mode == "state_only":
            neutral_action = np.zeros((1, 2), dtype=np.float32)
            neutral_risk = self._risk_features(belief_state, neutral_action, belief_vecs)
            risk = np.repeat(neutral_risk, len(self.actions), axis=0)
        else:
            risk = self._risk_features(belief_state, self.actions, belief_vecs)

        kinematics = self._kinematic_features(robot, self.actions)

        if self.belief_mode == "no_belief":
            belief_vecs = np.zeros_like(belief_vecs)
            risk = np.zeros_like(risk)

        belief_global = self._global_belief(belief_vecs)
        candidates = np.concatenate((risk, kinematics), axis=-1)

        return DecisionFeatures(
            context=self._mamba_context(token),
            belief=belief_global,
            candidates=candidates,
        )


@dataclass
class StepResult:
    outcome: str
    reward: float
    done: bool
    dmin: float


class FullCrowdNavigationEnvironment:
    """Nominal/nonstationary full-crowd (up to 20-pedestrian) environment."""

    def __init__(self, env_config: str, scenario: str, robot_visible: bool = False):
        self.env, self.robot, self.teacher, self.config = build_full_crowd_environment(
            env_config, scenario, robot_visible=robot_visible
        )
        self.scheduler: Optional[BehaviorScheduler] = None
        self.policies: List[InterventionORCA] = []

    def reset(self, seed: int, profile: str, test_case: int):
        self.env.reset(seed=int(seed), options={"test_case": int(test_case)})
        self.teacher.sim = None
        self.teacher._last_pref_vel = None
        self.policies = []
        for human in self.env.humans:
            policy = InterventionORCA(self.config)
            policy.time_step = self.env.time_step
            policy.reset()
            human.set_policy(policy)
            self.policies.append(policy)
        self.scheduler = BehaviorScheduler(PROFILES[profile], seed=int(seed) + 7919)
        self.scheduler.reset(len(self.policies))
        return self.robot, list(self.env.humans)

    def joint_state(self) -> JointState:
        return JointState(
            self.robot.get_full_state(),
            [human.get_observable_state() for human in self.env.humans],
        )

    def expert_action(self) -> ActionXY:
        return self.teacher.predict(self.joint_state())

    def step(self, action: ActionXY) -> StepResult:
        assert self.scheduler is not None
        self.scheduler.advance(self.policies)
        _, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        outcome = "running"
        if done:
            event = str(info.get("event", ""))
            if "reach" in event.lower() or "success" in event.lower():
                outcome = "success"
            elif "collision" in event.lower():
                outcome = "collision"
            else:
                outcome = "timeout"
        dmin = float(info.get("dmin", float("inf")))
        return StepResult(outcome=outcome, reward=float(reward), done=done, dmin=dmin)


def nearest_action_index(action: ActionXY, actions: np.ndarray) -> int:
    delta = actions - np.asarray([action.vx, action.vy], dtype=np.float32)
    return int(np.argmin(np.sum(delta * delta, axis=-1)))


def nearest_action_indices(action: ActionXY, actions: np.ndarray, k: int = 3) -> np.ndarray:
    """Top-k discrete actions nearest a continuous velocity, by L2 distance.

    ORCA's continuous velocity rarely lands exactly on one of the 80
    discrete grid actions -- several neighboring grid points are often
    nearly equivalent in velocity space. Forcing a margin against a single
    nearest action treats those near-ties as violations; using the k
    nearest as an "expert set" (see train.expert_margin_loss) does not.
    """
    delta = actions - np.asarray([action.vx, action.vy], dtype=np.float32)
    distances = np.sum(delta * delta, axis=-1)
    return np.argsort(distances)[:k].astype(np.int64)
