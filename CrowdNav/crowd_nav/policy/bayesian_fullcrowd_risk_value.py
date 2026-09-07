"""Full-crowd Bayesian risk calibration for Mamba value lookahead."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from crowd_nav.gdbn import GDBNIntegration
from crowd_nav.policy.mamba_rl import MambaRLPolicy


class BayesianFullCrowdRiskValuePolicy(MambaRLPolicy):
    """Correct nominal candidate values with online full-crowd action risk.

    The Mamba branch retains its original five-pedestrian observation and
    constant-velocity successor model. A separate Bayesian branch observes up
    to 20 pedestrians and estimates collision risk for every candidate action.
    The risk is applied as a bounded continuous correction, not as a hard veto.
    """

    ACTION_FEATURE_NAMES = (
        "expected_risk",
        "tail_risk",
        "tail_excess",
        "posterior_entropy",
        "action_klda",
        "epistemic_value",
    )

    def __init__(self, config=None, device="cpu"):
        super().__init__(config=config, device=device)
        belief_section = "bayesian_distributional"
        risk_section = "bayesian_fullcrowd_risk"
        if not config.has_section(belief_section):
            raise ValueError(f"Missing [{belief_section}] section")
        if not config.has_section(risk_section):
            raise ValueError(f"Missing [{risk_section}] section")

        self.belief_k = config.getint(belief_section, "K")
        self.belief_num_humans = config.getint(belief_section, "num_humans")
        self.belief_particles = config.getint(belief_section, "n_particles")
        self.belief_seed = config.getint(belief_section, "belief_seed")
        self.belief_params_dir = config.get(belief_section, "params_dir")
        self.action_risk_horizon = config.getint(
            belief_section,
            "action_risk_horizon",
        )
        self.action_safe_distance = config.getfloat(
            belief_section,
            "action_safe_distance",
        )
        self.action_risk_cvar_alpha = config.getfloat(
            belief_section,
            "action_risk_cvar_alpha",
        )
        self.action_pedestrian_aggregation = config.get(
            belief_section,
            "action_pedestrian_aggregation",
            fallback="max",
        ).strip().lower()
        self.risk_statistic = config.get(
            risk_section,
            "risk_statistic",
            fallback="expected",
        ).strip().lower()
        if self.risk_statistic not in {"expected", "tail"}:
            raise ValueError(
                "risk_statistic must be 'expected' or 'tail'"
            )

        self.register_buffer(
            "collision_penalty_scale",
            torch.tensor(
                config.getfloat(risk_section, "collision_penalty_scale"),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "penalty_cap",
            torch.tensor(
                config.getfloat(risk_section, "penalty_cap"),
                dtype=torch.float32,
            ),
        )

        self.belief_filter = GDBNIntegration(
            K=self.belief_k,
            n_particles=self.belief_particles,
            params_dir=self.belief_params_dir,
            max_peds=self.belief_num_humans,
            random_seed=self.belief_seed,
        )
        if not getattr(self.belief_filter, "_fitted", False):
            raise RuntimeError(
                "Bayesian belief parameters were not loaded: "
                f"{os.path.abspath(self.belief_params_dir)}"
            )

        self._active_belief = None
        self._active_action_features = None
        self._fullcrowd_human_states = None
        self.to(self.device)
        logging.info(
            "[BAYES-FULLCROWD] K=%d pedestrians=%d aggregation=%s "
            "statistic=%s collision_scale=%.3f cap=%.3f",
            self.belief_k,
            self.belief_num_humans,
            self.action_pedestrian_aggregation,
            self.risk_statistic,
            float(self.collision_penalty_scale),
            float(self.penalty_cap),
        )

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self._active_belief = None
        self._active_action_features = None
        self._fullcrowd_human_states = None
        self.belief_filter.reset(n_peds=self.belief_num_humans)

    def set_fullcrowd_human_states(self, human_states):
        """Provide identity-stable observations for the full-crowd branch."""
        self._fullcrowd_human_states = list(human_states)

    def build_belief_state(self, robot_state, human_states) -> np.ndarray:
        """Build the full-crowd state used only by the Bayesian branch."""
        fullcrowd_humans = self._fullcrowd_human_states
        if fullcrowd_humans is not None:
            human_states = fullcrowd_humans
        robot = np.asarray(
            [
                robot_state.px,
                robot_state.py,
                robot_state.vx,
                robot_state.vy,
                robot_state.radius,
                robot_state.gx,
                robot_state.gy,
                robot_state.v_pref,
                robot_state.theta,
            ],
            dtype=np.float32,
        )
        humans = []
        for human in human_states[:self.belief_num_humans]:
            humans.extend(
                [
                    human.px,
                    human.py,
                    human.vx,
                    human.vy,
                    human.radius,
                ]
            )
        expected = 5 * self.belief_num_humans
        humans.extend([0.0] * max(0, expected - len(humans)))
        return np.concatenate(
            (robot, np.asarray(humans[:expected], dtype=np.float32))
        )

    @staticmethod
    def _rollout_array(
        rollout: dict,
        key: str,
        fallback: Optional[str] = None,
    ) -> np.ndarray:
        value = rollout.get(key)
        if value is None and fallback is not None:
            value = rollout[fallback]
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def action_features_from_rollout(self, rollout: dict) -> np.ndarray:
        expected_risk = np.clip(
            self._rollout_array(rollout, "risk"),
            0.0,
            1.0,
        )
        tail_risk = np.clip(
            self._rollout_array(rollout, "tail_risk", "risk"),
            0.0,
            1.0,
        )
        entropy = np.clip(
            self._rollout_array(rollout, "entropy"),
            0.0,
            1.0,
        )
        klda_scale = max(
            float(getattr(self.belief_filter, "klda_norm_clip", 1.0)),
            1e-6,
        )
        klda = np.clip(
            self._rollout_array(rollout, "klda") / klda_scale,
            0.0,
            1.0,
        )
        epistemic = np.clip(
            self._rollout_array(rollout, "epistemic_value"),
            0.0,
            1.0,
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
        ).astype(np.float32, copy=False)

    def compute_action_features_batch(
        self,
        states,
        actions,
        belief_vecs=None,
    ) -> np.ndarray:
        rollout = self.belief_filter.predict_action_rollout_batch(
            np.asarray(states, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            belief_vecs=(
                None
                if belief_vecs is None
                else np.asarray(belief_vecs, dtype=np.float64)
            ),
            horizon=self.action_risk_horizon,
            dt=self.time_step,
            safe_distance=self.action_safe_distance,
            cvar_alpha=self.action_risk_cvar_alpha,
            pedestrian_aggregation=self.action_pedestrian_aggregation,
        )
        return self.action_features_from_rollout(rollout)

    def compute_candidate_action_features(
        self,
        state,
        actions,
    ) -> np.ndarray:
        candidate_count = int(len(actions))
        states = np.repeat(
            np.asarray(state, dtype=np.float32).reshape(1, -1),
            candidate_count,
            axis=0,
        )
        beliefs = None
        if self._active_belief is not None:
            beliefs = np.repeat(
                self._active_belief.reshape(
                    1,
                    *self._active_belief.shape,
                ),
                candidate_count,
                axis=0,
            )
        return self.compute_action_features_batch(states, actions, beliefs)

    def corrected_values(
        self,
        base_values: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        risk_index = 1 if self.risk_statistic == "tail" else 0
        collision_risk = action_features[:, risk_index].clamp(0.0, 1.0)
        penalty = torch.minimum(
            self.collision_penalty_scale * collision_risk,
            self.penalty_cap,
        )
        value_scale = base_values.std(unbiased=False).clamp_min(1e-4)
        return base_values - value_scale * penalty

    def forward_value(self, joint_tokens, mask=None):
        base_values = MambaRLPolicy.forward_value(
            self,
            joint_tokens,
            mask=mask,
        )
        if self._active_action_features is None:
            return base_values
        action_features = torch.as_tensor(
            self._active_action_features,
            dtype=joint_tokens.dtype,
            device=joint_tokens.device,
        )
        if len(action_features) != len(base_values):
            return base_values
        return self.corrected_values(base_values, action_features)

    def predict_sarl_style(self, state):
        belief_state = self.build_belief_state(
            state.self_state,
            state.human_states,
        )
        self.belief_filter.update(belief_state)
        self._active_belief = np.asarray(
            self.belief_filter.get_per_ped_belief_vec(),
            dtype=np.float32,
        )
        if self.reach_destination(state):
            return super().predict_sarl_style(state)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        actions = np.asarray(
            [
                (float(action.vx), float(action.vy))
                for action in self.action_space
            ],
            dtype=np.float32,
        )
        self._active_action_features = self.compute_candidate_action_features(
            belief_state,
            actions,
        )
        try:
            return super().predict_sarl_style(state)
        finally:
            self._active_action_features = None


BayesianFullCrowdRiskValue = BayesianFullCrowdRiskValuePolicy
