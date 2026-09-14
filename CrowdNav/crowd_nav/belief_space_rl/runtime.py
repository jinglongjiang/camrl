"""Environment and feature runtime for the belief-space RL pilot."""

from __future__ import annotations

import configparser
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from crowd_nav.bayesian_pilot.collect_nonstationary import (
    build_environment,
    load_config,
    observation_34,
)
from crowd_nav.bayesian_pilot.protocol import (
    BehaviorProfile,
    BehaviorScheduler,
    InterventionORCA,
    PROFILES,
)
from crowd_nav.belief_space_rl.model import DecisionFeatures
from crowd_nav.contracts import (
    GRID,
    _batch_joint34_to_tokens_vectorized,
    discrete_index_to_action,
    init_grid_from_cfg,
)
from crowd_nav.gdbn import GDBNIntegration
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


RL_PROFILES = dict(PROFILES)
RL_PROFILES["train_risk"] = BehaviorProfile(
    name="train_risk",
    event_rate=0.06,
    duration_steps=(4, 9),
    turn_degrees=(55.0, 85.0),
    slow_scale=(0.15, 0.45),
    event_weights=(0.35, 0.20, 0.225, 0.225),
)
RL_PROFILES["decision_stress"] = BehaviorProfile(
    name="decision_stress",
    event_rate=0.08,
    duration_steps=(4, 9),
    turn_degrees=(55.0, 85.0),
    slow_scale=(0.15, 0.45),
    event_weights=(0.35, 0.20, 0.225, 0.225),
)


def merged_policy_config(
    policy_path: str,
    env_path: str,
) -> configparser.RawConfigParser:
    policy = configparser.RawConfigParser()
    environment = configparser.RawConfigParser()
    policy.read(policy_path)
    environment.read(env_path)
    for section in environment.sections():
        if not policy.has_section(section):
            policy.add_section(section)
        for key, value in environment.items(section):
            if not policy.has_option(section, key):
                policy.set(section, key, value)
    if not policy.has_section("buffer"):
        policy.add_section("buffer")
    policy.set("buffer", "seq_len", policy.get("temporal", "T"))
    if not policy.has_section("belief"):
        policy.add_section("belief")
    for key, value in {
        "enable": "false",
        "base_token_dim": "13",
        "belief_dim": "0",
        "token_dim": "13",
        "num_entities": "8",
        "human_start": "3",
        "num_humans": "5",
    }.items():
        policy.set("belief", key, value)
    if not policy.has_section("train"):
        policy.add_section("train")
    policy.set("train", "gamma", "0.99")
    policy.set("train", "epsilon_start", "0.0")
    if not policy.has_section("sarl"):
        policy.add_section("sarl")
    policy.set("sarl", "epsilon_start", "0.0")
    init_grid_from_cfg(policy)
    return policy


def checkpoint_state(path: str):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get(
        "policy_state",
        checkpoint.get("model_state_dict", checkpoint.get("value", checkpoint)),
    )
    return {
        (
            key.replace("_orig_mod.", "", 1)
            if key.startswith("_orig_mod.")
            else key
        ): value
        for key, value in state.items()
    }


def build_frozen_mamba(
    config,
    checkpoint_path: str,
    device: torch.device,
) -> MambaRLPolicy:
    # A frozen historical teacher must retain the coordinates it was trained on.
    import copy
    teacher_config = copy.deepcopy(config)
    teacher_config.set('mamba', 'coordinate_contract', 'legacy_v1')
    policy = MambaRLPolicy(config=teacher_config, device=device)
    state = checkpoint_state(checkpoint_path)
    incompatible = policy.load_state_dict(state, strict=False)
    missing_encoder = [
        key for key in incompatible.missing_keys
        if key.startswith(("spatial_encoder.", "temporal_encoder."))
    ]
    if missing_encoder:
        raise RuntimeError(
            f"Checkpoint is missing encoder parameters: {missing_encoder[:5]}"
        )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return policy


def upper_cvar(
    values: np.ndarray,
    probabilities: np.ndarray,
    tail_mass: float,
) -> np.ndarray:
    """Probability-weighted upper-tail average along the last axis."""
    order = np.argsort(values, axis=-1)[..., ::-1]
    ordered_v = np.take_along_axis(values, order, axis=-1)
    ordered_p = np.take_along_axis(probabilities, order, axis=-1)
    remaining = np.full(values.shape[:-1], float(tail_mass))
    numerator = np.zeros(values.shape[:-1], dtype=np.float64)
    for index in range(values.shape[-1]):
        take = np.minimum(ordered_p[..., index], remaining)
        numerator += take * ordered_v[..., index]
        remaining -= take
    return numerator / max(float(tail_mass), 1e-8)


class BeliefFeatureEngine:
    """Stateful Bayesian and frozen-Mamba feature extractor."""

    def __init__(
        self,
        mamba: MambaRLPolicy,
        gdbn_params: str,
        device: torch.device,
        seq_len: int = 24,
        particles: int = 50,
        risk_samples: int = 24,
        risk_horizon: int = 3,
        caution_clearance: float = 0.30,
        seed: int = 2407,
    ):
        self.mamba = mamba
        self.device = device
        self.seq_len = int(seq_len)
        self.risk_horizon = int(risk_horizon)
        self.caution_clearance = float(caution_clearance)
        self.history = deque(maxlen=self.seq_len)
        self.filter = GDBNIntegration(
            K=4,
            n_particles=particles,
            params_dir=gdbn_params,
            max_peds=5,
            random_seed=seed,
        )
        if not self.filter._fitted or not self.filter._action_fitted:
            raise RuntimeError("The selected GDBN and action model are required")
        self.K = int(self.filter.K)
        self.actions = np.asarray(
            [
                discrete_index_to_action(index)
                for index in range(
                    int(GRID["n_speeds"]) * int(GRID["n_headings"])
                    + int(bool(GRID.get("include_stop", False)))
                )
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed + 1709)
        half = max(2, int(risk_samples) // 2)
        standard = rng.normal(size=(half, 2))
        self.standard_samples = np.concatenate((standard, -standard), axis=0)
        self.noise = np.stack(
            [
                self.standard_samples
                @ np.linalg.cholesky(
                    np.asarray(cov[:2, :2], dtype=np.float64)
                    + 1e-8 * np.eye(2)
                ).T
                for cov in self.filter.gdbn.Q
            ],
            axis=0,
        )

    @property
    def belief_dim(self) -> int:
        return 2 * self.K + 5

    @property
    def candidate_dim(self) -> int:
        return 9

    def reset(self):
        self.history.clear()
        self.filter.reset(n_peds=5)

    def _mamba_context(self, state_34: np.ndarray) -> np.ndarray:
        tokens = _batch_joint34_to_tokens_vectorized(
            state_34.reshape(1, -1)
        )[0]
        self.history.append(tokens)
        sequence = list(self.history)
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

    def _global_belief(self, belief: np.ndarray, state_34: np.ndarray):
        humans = state_34[9:34].reshape(5, 5)
        valid = np.any(humans != 0.0, axis=-1)
        active = belief[valid]
        if len(active) == 0:
            active = belief[:1]
        mode = active[:, :self.K]
        entropy = active[:, self.K]
        klda = active[:, self.K + 1]
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
                        valid.mean(),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)

    def _candidate_features(
        self,
        state: np.ndarray,
        belief: np.ndarray,
    ) -> np.ndarray:
        action_count = len(self.actions)
        expected_max = np.zeros(action_count, dtype=np.float64)
        tail_max = np.zeros(action_count, dtype=np.float64)
        risk_std_max = np.zeros(action_count, dtype=np.float64)
        proximity_max = np.zeros(action_count, dtype=np.float64)
        humans = state[9:34].reshape(5, 5).astype(np.float64)
        robot_position = state[:2].astype(np.float64)
        robot_radius = float(state[4])
        A = np.stack(self.filter.gdbn.A, axis=0)
        B = np.stack(self.filter.B_action, axis=0)
        Pi = np.asarray(self.filter.gdbn.Pi, dtype=np.float64)
        actions = self.actions.astype(np.float64)
        robot_future = robot_position[None, :].repeat(action_count, axis=0)

        for ped_index, human in enumerate(humans):
            if not np.any(human):
                continue
            probabilities = np.maximum(
                belief[ped_index, :self.K].astype(np.float64),
                1e-10,
            )
            probabilities /= probabilities.sum()
            x_by_action = np.repeat(
                human[:4][None, :],
                action_count,
                axis=0,
            )
            ped_expected = np.zeros(action_count, dtype=np.float64)
            ped_tail = np.zeros(action_count, dtype=np.float64)
            ped_std = np.zeros(action_count, dtype=np.float64)
            ped_proximity = np.zeros(action_count, dtype=np.float64)
            local_robot = robot_future.copy()

            for _ in range(self.risk_horizon):
                probabilities = np.maximum(probabilities @ Pi, 1e-10)
                probabilities /= probabilities.sum()
                means = np.einsum("kij,aj->aki", A, x_by_action)
                means += np.einsum("kij,aj->aki", B, actions)
                local_robot += actions * 0.25
                samples = (
                    means[:, :, None, :2]
                    + self.noise[None, :, :, :]
                )
                distance = np.linalg.norm(
                    samples - local_robot[:, None, None, :],
                    axis=-1,
                )
                threshold = (
                    robot_radius + float(human[4])
                    + self.caution_clearance
                )
                mode_risk = (distance <= threshold).mean(axis=-1)
                expected = (mode_risk * probabilities[None, :]).sum(axis=-1)
                tail = upper_cvar(
                    mode_risk,
                    np.broadcast_to(probabilities, mode_risk.shape),
                    tail_mass=0.25,
                )
                variance = (
                    probabilities[None, :]
                    * (mode_risk - expected[:, None]) ** 2
                ).sum(axis=-1)
                mean_distance = np.linalg.norm(
                    means[:, :, :2] - local_robot[:, None, :],
                    axis=-1,
                )
                mean_clearance = (
                    mean_distance - robot_radius - float(human[4])
                )
                proximity = np.exp(
                    -np.maximum(mean_clearance.min(axis=-1), 0.0)
                    / max(self.caution_clearance, 1e-6)
                )
                ped_expected = np.maximum(ped_expected, expected)
                ped_tail = np.maximum(ped_tail, tail)
                ped_std = np.maximum(ped_std, np.sqrt(variance))
                ped_proximity = np.maximum(ped_proximity, proximity)
                x_by_action = (
                    means * probabilities[None, :, None]
                ).sum(axis=1)

            expected_max = np.maximum(expected_max, ped_expected)
            tail_max = np.maximum(tail_max, ped_tail)
            risk_std_max = np.maximum(risk_std_max, ped_std)
            proximity_max = np.maximum(proximity_max, ped_proximity)

        goal = state[5:7].astype(np.float64) - state[:2].astype(np.float64)
        goal_norm = max(float(np.linalg.norm(goal)), 1e-6)
        goal_unit = goal / goal_norm
        speed = np.linalg.norm(actions, axis=-1)
        alignment = (actions @ goal_unit) / np.maximum(speed, 1e-6)
        progress = actions @ goal_unit * 0.25
        current_velocity = state[2:4].astype(np.float64)
        current_norm = max(float(np.linalg.norm(current_velocity)), 1e-6)
        turn_cost = 1.0 - (
            actions @ (current_velocity / current_norm)
        ) / np.maximum(speed, 1e-6)
        turn_cost = np.where(speed > 1e-6, turn_cost, 1.0)
        features = np.stack(
            (
                expected_max,
                tail_max,
                np.maximum(tail_max - expected_max, 0.0),
                risk_std_max,
                proximity_max,
                np.clip(alignment, -1.0, 1.0),
                np.clip(progress, -0.25, 0.25) * 4.0,
                np.clip(speed, 0.0, 1.0),
                np.clip(turn_cost, 0.0, 2.0) * 0.5,
            ),
            axis=-1,
        )
        return features.astype(np.float32)

    def teacher_scores(self, state_34: np.ndarray) -> np.ndarray:
        """Return the frozen Mamba value-lookahead score for every action.

        This mirrors ``MambaRLPolicy.predict_sarl_style`` without mutating the
        teacher policy. The current observation has already been appended to
        ``self.history`` by ``encode``.
        """
        state = np.asarray(state_34, dtype=np.float32)
        robot = FullState(*state[:9].tolist())
        humans = []
        for pedestrian_index in range(5):
            start = 9 + 5 * pedestrian_index
            pedestrian = state[start:start + 5]
            if pedestrian.size == 5 and not np.allclose(pedestrian, 0.0):
                humans.append(ObservableState(*pedestrian.tolist()))

        history = list(self.history)
        if not history:
            raise RuntimeError("encode must be called before teacher_scores")
        action_sequences = []
        rewards = []
        minimum_clearances = []
        for velocity in self.actions:
            action = ActionXY(float(velocity[0]), float(velocity[1]))
            next_robot = self.mamba.propagate(robot, action)
            next_humans = [
                self.mamba.propagate(
                    human,
                    ActionXY(float(human.vx), float(human.vy)),
                )
                for human in humans
            ]
            rewards.append(
                self.mamba.compute_reward(
                    next_robot,
                    next_humans,
                    prev_nav=robot,
                    action=action,
                )
            )
            minimum_clearances.append(
                min(
                    (
                        np.hypot(
                            next_robot.px - human.px,
                            next_robot.py - human.py,
                        )
                        - next_robot.radius
                        - human.radius
                    )
                    for human in next_humans
                )
                if next_humans
                else float("inf")
            )
            next_state = self.mamba._build_joint_state_34(
                next_robot,
                next_humans,
            )
            next_token = _batch_joint34_to_tokens_vectorized(
                next_state.reshape(1, -1)
            )[0]
            sequence = history + [next_token]
            sequence = sequence[-self.seq_len:]
            if len(sequence) < self.seq_len:
                sequence = [sequence[0]] * (
                    self.seq_len - len(sequence)
                ) + sequence
            action_sequences.append(np.asarray(sequence, dtype=np.float32))

        tensor = torch.as_tensor(
            np.asarray(action_sequences),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            next_values = self.mamba.forward_value(tensor)
        scores = (
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            + float(self.mamba.gamma) * next_values
        )
        clearances = torch.as_tensor(
            minimum_clearances,
            dtype=torch.float32,
            device=self.device,
        )
        minimum = float(self.mamba.test_min_clearance)
        risk_lambda = float(self.mamba.test_risk_lambda)
        if minimum > 0.0:
            safe = clearances >= minimum
            if safe.any():
                scores = scores.masked_fill(~safe, -1e4)
        if risk_lambda > 0.0:
            margin = (
                minimum
                if minimum > 0.0
                else float(self.mamba.discomfort_dist)
            )
            scores = scores - risk_lambda * torch.clamp(
                margin - clearances,
                min=0.0,
            )
        if scores.numel() > 0:
            scores[0] -= 1e-3
        return scores.detach().cpu().numpy().astype(np.float32)

    def encode(self, state_34: np.ndarray) -> DecisionFeatures:
        state_34 = np.asarray(state_34, dtype=np.float32)
        self.filter.update(state_34)
        belief = np.asarray(
            self.filter.get_per_ped_belief_vec(),
            dtype=np.float32,
        )
        return DecisionFeatures(
            context=self._mamba_context(state_34),
            belief=self._global_belief(belief, state_34),
            candidates=self._candidate_features(state_34, belief),
        )


@dataclass
class StepResult:
    state_34: np.ndarray
    reward: float
    done: bool
    outcome: str


class PilotNavigationEnvironment:
    """Five-pedestrian nominal/nonstationary pilot environment."""

    def __init__(self, env_config: str, scenario: str):
        config = load_config(Path(env_config))
        self.env, self.robot, _ = build_environment(config, scenario)
        self.config = config
        self.teacher = ORCA()
        self.teacher.configure(config)
        self.teacher.multiagent_training = True
        self.teacher.set_phase("test")
        self.scheduler: Optional[BehaviorScheduler] = None
        self.policies = []

    def reset(
        self,
        seed: int,
        profile: str,
        test_case: int,
    ) -> np.ndarray:
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
        self.scheduler = BehaviorScheduler(
            RL_PROFILES[profile],
            seed=int(seed) + 7919,
        )
        self.scheduler.reset(len(self.policies))
        return observation_34(self.robot, self.env.humans)

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
        result = self.env.step(action)
        _, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return StepResult(
            state_34=observation_34(self.robot, self.env.humans),
            reward=float(reward),
            done=done,
            outcome=str(info.get("event", "running")) if done else "running",
        )


def nearest_action_index(action: ActionXY, actions: np.ndarray) -> int:
    delta = actions - np.asarray([action.vx, action.vy], dtype=np.float32)
    return int(np.argmin(np.sum(delta * delta, axis=-1)))
