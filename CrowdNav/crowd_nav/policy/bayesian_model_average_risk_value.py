"""Online Bayesian model averaging for full-crowd risk correction."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from crowd_nav.policy.bayesian_fullcrowd_risk_value import (
    BayesianFullCrowdRiskValuePolicy,
)
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.risk_models import ConstantVelocityRiskModel


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _gaussian_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = covariance + 1e-6 * np.eye(covariance.shape[0])
    difference = np.asarray(value, dtype=np.float64) - np.asarray(
        mean,
        dtype=np.float64,
    )
    try:
        chol = np.linalg.cholesky(covariance)
        solved = np.linalg.solve(chol, difference)
        log_det = 2.0 * np.log(np.diag(chol)).sum()
        quadratic = float(np.dot(solved, solved))
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
        _, log_det = np.linalg.slogdet(covariance)
        quadratic = float(difference @ inverse @ difference)
    return float(
        -0.5
        * (
            quadratic
            + log_det
            + len(difference) * np.log(2.0 * np.pi)
        )
    )


class BayesianModelAverageFullCrowdRiskValuePolicy(
    BayesianFullCrowdRiskValuePolicy
):
    """Blend CV and GDBN risks using online one-step model evidence."""

    def __init__(self, config=None, device="cpu"):
        super().__init__(config=config, device=device)
        section = "bayesian_model_average"
        if not config.has_section(section):
            raise ValueError(f"Missing [{section}] section")

        calibration_path = config.get(section, "cv_calibration")
        if not os.path.isfile(calibration_path):
            raise FileNotFoundError(calibration_path)
        calibration = np.load(calibration_path, allow_pickle=False)
        self.cv_covariance = np.asarray(
            calibration["cv_covariance"],
            dtype=np.float64,
        )
        if self.cv_covariance.shape != (4, 4):
            raise ValueError("cv_covariance must have shape (4, 4)")
        self.log_bayes_factor_center = float(
            calibration["log_bayes_factor_center"]
        )
        self.log_bayes_factor_scale = max(
            float(calibration["log_bayes_factor_scale"]),
            1e-6,
        )

        self.model_prior_gdbn = config.getfloat(
            section,
            "prior_gdbn",
            fallback=0.5,
        )
        if not 0.0 < self.model_prior_gdbn < 1.0:
            raise ValueError("prior_gdbn must be in (0, 1)")
        self.model_evidence_forgetting = config.getfloat(
            section,
            "evidence_forgetting",
            fallback=0.95,
        )
        self.model_evidence_temperature = config.getfloat(
            section,
            "evidence_temperature",
            fallback=1.0,
        )
        self.model_evidence_clip = config.getfloat(
            section,
            "evidence_clip",
            fallback=6.0,
        )
        self.model_probability_floor = config.getfloat(
            section,
            "probability_floor",
            fallback=0.02,
        )
        self.fixed_gdbn_weight = config.getfloat(
            section,
            "fixed_gdbn_weight",
            fallback=-1.0,
        )

        self.cv_filter = ConstantVelocityRiskModel(
            max_peds=self.belief_num_humans,
            modeled_peds=self.belief_num_humans,
            belief_dim=self.belief_k + 2,
        )
        self._prior_log_odds = float(
            np.log(self.model_prior_gdbn / (1.0 - self.model_prior_gdbn))
        )
        self._model_log_odds = np.full(
            self.belief_num_humans,
            self._prior_log_odds,
            dtype=np.float64,
        )
        self._previous_belief_state = None
        self._previous_mode_probabilities = None
        self._previous_action = None
        self._bma_weight_sum = 0.0
        self._bma_max_weight_sum = 0.0
        self._bma_weight_count = 0
        self._bma_high_weight_count = 0
        self._bma_any_high_weight_count = 0
        logging.info(
            "[BMA-FULLCROWD] prior=%.3f forgetting=%.3f temperature=%.3f "
            "fixed_weight=%.3f evidence_center=%.3f evidence_scale=%.3f "
            "calibration=%s",
            self.model_prior_gdbn,
            self.model_evidence_forgetting,
            self.model_evidence_temperature,
            self.fixed_gdbn_weight,
            self.log_bayes_factor_center,
            self.log_bayes_factor_scale,
            os.path.abspath(calibration_path),
        )

    def reset_episode_stats(self):
        super().reset_episode_stats()
        if hasattr(self, "cv_filter"):
            self.cv_filter.reset(n_peds=self.belief_num_humans)
        if hasattr(self, "_prior_log_odds"):
            self._model_log_odds.fill(self._prior_log_odds)
        self._previous_belief_state = None
        self._previous_mode_probabilities = None
        self._previous_action = None

    @staticmethod
    def _constant_velocity_matrix(dt: float) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 2] = float(dt)
        matrix[1, 3] = float(dt)
        return matrix

    def _model_probabilities(self) -> np.ndarray:
        if 0.0 <= self.fixed_gdbn_weight <= 1.0:
            return np.full(
                self.belief_num_humans,
                self.fixed_gdbn_weight,
                dtype=np.float64,
            )
        clipped = np.clip(self._model_log_odds, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped))
        floor = float(np.clip(self.model_probability_floor, 0.0, 0.49))
        return np.clip(probabilities, floor, 1.0 - floor)

    def _update_model_evidence(self, current_state: np.ndarray):
        if (
            self.fixed_gdbn_weight >= 0.0
            or self._previous_belief_state is None
            or self._previous_mode_probabilities is None
        ):
            return

        previous = self._previous_belief_state
        action = self._previous_action
        if action is None:
            action = previous[2:4]
        action = np.asarray(action, dtype=np.float64)
        cv_matrix = self._constant_velocity_matrix(self.time_step)
        transition = np.asarray(self.belief_filter.gdbn.Pi, dtype=np.float64)

        for ped_index in range(self.belief_num_humans):
            start = 9 + 5 * ped_index
            previous_ped = previous[start:start + 5]
            current_ped = current_state[start:start + 5]
            if np.all(previous_ped == 0.0) or np.all(current_ped == 0.0):
                self._model_log_odds[ped_index] = self._prior_log_odds
                continue

            previous_modes = np.maximum(
                self._previous_mode_probabilities[ped_index],
                1e-12,
            )
            previous_modes /= previous_modes.sum()
            mode_prior = np.maximum(previous_modes @ transition, 1e-12)
            mode_prior /= mode_prior.sum()
            x_previous = previous_ped[:4].astype(np.float64)
            x_current = current_ped[:4].astype(np.float64)

            gdbn_terms = []
            for mode_index in range(self.belief_k):
                mean = (
                    self.belief_filter.gdbn.A[mode_index] @ x_previous
                    + self.belief_filter.B_action[mode_index] @ action
                )
                gdbn_terms.append(
                    np.log(mode_prior[mode_index])
                    + _gaussian_logpdf(
                        x_current,
                        mean,
                        self.belief_filter.gdbn.Q[mode_index],
                    )
                )
            gdbn_log_likelihood = _logsumexp(
                np.asarray(gdbn_terms, dtype=np.float64)
            )
            cv_log_likelihood = _gaussian_logpdf(
                x_current,
                cv_matrix @ x_previous,
                self.cv_covariance,
            )
            evidence = (
                (
                    gdbn_log_likelihood
                    - cv_log_likelihood
                    - self.log_bayes_factor_center
                )
                / self.log_bayes_factor_scale
                / max(float(self.model_evidence_temperature), 1e-6)
            )
            evidence = float(
                np.clip(
                    evidence,
                    -self.model_evidence_clip,
                    self.model_evidence_clip,
                )
            )
            self._model_log_odds[ped_index] = (
                self._prior_log_odds
                + self.model_evidence_forgetting
                * (
                    self._model_log_odds[ped_index]
                    - self._prior_log_odds
                )
                + evidence
            )

    @staticmethod
    def _aggregate_per_pedestrian(
        values: np.ndarray,
        valid_mask: np.ndarray,
        aggregation: str,
    ) -> np.ndarray:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        values = np.asarray(values, dtype=np.float64)
        if aggregation == "max":
            result = np.where(valid_mask, values, -np.inf).max(axis=1)
            return np.where(valid_mask.any(axis=1), result, 0.0)
        counts = np.maximum(valid_mask.sum(axis=1), 1)
        return (values * valid_mask).sum(axis=1) / counts

    def _blended_action_features(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        belief_vecs: np.ndarray,
    ) -> np.ndarray:
        rollout_kwargs = dict(
            horizon=self.action_risk_horizon,
            dt=self.time_step,
            safe_distance=self.action_safe_distance,
            cvar_alpha=self.action_risk_cvar_alpha,
            pedestrian_aggregation=self.action_pedestrian_aggregation,
        )
        gdbn_rollout = self.belief_filter.predict_action_rollout_batch(
            states,
            actions,
            belief_vecs=belief_vecs,
            **rollout_kwargs,
        )
        cv_rollout = self.cv_filter.predict_action_rollout_batch(
            states,
            actions,
            belief_vecs=None,
            **rollout_kwargs,
        )
        model_weights = self._model_probabilities()[None, :]
        valid_mask = np.asarray(gdbn_rollout["valid_mask"], dtype=bool)
        expected_by_ped = (
            model_weights * np.asarray(gdbn_rollout["risk_by_ped"])
            + (1.0 - model_weights)
            * np.asarray(cv_rollout["risk_by_ped"])
        )
        tail_by_ped = (
            model_weights * np.asarray(gdbn_rollout["tail_risk_by_ped"])
            + (1.0 - model_weights)
            * np.asarray(cv_rollout["tail_risk_by_ped"])
        )
        expected = self._aggregate_per_pedestrian(
            expected_by_ped,
            valid_mask,
            self.action_pedestrian_aggregation,
        )
        tail = self._aggregate_per_pedestrian(
            tail_by_ped,
            valid_mask,
            self.action_pedestrian_aggregation,
        )
        average_weight = float(np.mean(self._model_probabilities()))
        rollout = {
            "risk": expected,
            "tail_risk": tail,
            "entropy": average_weight
            * np.asarray(gdbn_rollout["entropy"]),
            "klda": average_weight * np.asarray(gdbn_rollout["klda"]),
            "epistemic_value": average_weight
            * np.asarray(gdbn_rollout["epistemic_value"]),
        }
        return self.action_features_from_rollout(rollout)

    def predict_sarl_style(self, state):
        belief_state = self.build_belief_state(
            state.self_state,
            state.human_states,
        )
        self._update_model_evidence(belief_state)
        self.belief_filter.update(belief_state)
        self.cv_filter.update(belief_state)
        self._active_belief = np.asarray(
            self.belief_filter.get_per_ped_belief_vec(),
            dtype=np.float32,
        )
        current_mode_probabilities = self._active_belief[:, :self.belief_k]

        valid = np.asarray(
            [
                not np.all(
                    belief_state[9 + 5 * index:14 + 5 * index] == 0.0
                )
                for index in range(self.belief_num_humans)
            ],
            dtype=bool,
        )
        weights = self._model_probabilities()
        if np.any(valid):
            mean_weight = float(np.mean(weights[valid]))
            max_weight = float(np.max(weights[valid]))
            self._bma_weight_sum += mean_weight
            self._bma_max_weight_sum += max_weight
            self._bma_weight_count += 1
            self._bma_high_weight_count += int(mean_weight >= 0.75)
            self._bma_any_high_weight_count += int(max_weight >= 0.75)

        self._previous_belief_state = belief_state.copy()
        self._previous_mode_probabilities = current_mode_probabilities.copy()
        if self.reach_destination(state):
            return MambaRLPolicy.predict_sarl_style(self, state)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        actions = np.asarray(
            [
                (float(action.vx), float(action.vy))
                for action in self.action_space
            ],
            dtype=np.float32,
        )
        states = np.repeat(
            belief_state.reshape(1, -1),
            len(actions),
            axis=0,
        )
        beliefs = np.repeat(
            self._active_belief.reshape(
                1,
                *self._active_belief.shape,
            ),
            len(actions),
            axis=0,
        )
        self._active_action_features = self._blended_action_features(
            states,
            actions,
            beliefs,
        )
        try:
            selected = MambaRLPolicy.predict_sarl_style(self, state)
            if hasattr(selected, "vx"):
                self._previous_action = np.asarray(
                    [float(selected.vx), float(selected.vy)],
                    dtype=np.float64,
                )
            return selected
        finally:
            self._active_action_features = None


class FixedModelAverageFullCrowdRiskValuePolicy(
    BayesianModelAverageFullCrowdRiskValuePolicy
):
    """Control condition with a state-independent GDBN/CV mixture."""

    def __init__(self, config=None, device="cpu"):
        super().__init__(config=config, device=device)
        if not 0.0 <= self.fixed_gdbn_weight <= 1.0:
            raise ValueError(
                "Fixed model averaging requires fixed_gdbn_weight in [0, 1]"
            )


BayesianModelAverageFullCrowdRiskValue = (
    BayesianModelAverageFullCrowdRiskValuePolicy
)
FixedModelAverageFullCrowdRiskValue = (
    FixedModelAverageFullCrowdRiskValuePolicy
)
