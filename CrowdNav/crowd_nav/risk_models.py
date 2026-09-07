"""Evaluation-only risk-model baselines for the GDBN governance interface."""

from __future__ import annotations

import numpy as np


class ConstantVelocityRiskModel:
    """Deterministic constant-velocity pedestrian predictor.

    The class implements the rollout API consumed by ``BayesianCoreScorer`` so
    that only the risk predictor changes while vetoes, penalties, action space,
    checkpoint, and evaluation protocol remain fixed.
    """

    def __init__(self, max_peds=8, belief_dim=10, modeled_peds=5):
        self.K = 1
        self.n_particles = 0
        self.max_peds = int(max_peds)
        self.modeled_peds = min(int(modeled_peds), self.max_peds)
        self.belief_dim = int(belief_dim)
        self.klda_norm_clip = 1.0
        self._state = None

    @property
    def is_fitted(self):
        return True

    @property
    def action_fitted(self):
        return True

    def reset(self, n_peds=None):
        self._state = None

    def update(self, state_34d):
        self._state = self._normalize_states(state_34d)[0]
        return [0.0] * self.max_peds

    def get_per_ped_belief_vec(self):
        return np.zeros((self.max_peds, self.belief_dim), dtype=np.float32)

    @staticmethod
    def _clearance_risk(clearances, safe_distance):
        margin = max(float(safe_distance), 1e-6)
        clearances = np.asarray(clearances, dtype=np.float64)
        positive = np.exp(-np.maximum(clearances, 0.0) / margin)
        return np.clip(np.where(clearances <= 0.0, 1.0, positive), 0.0, 1.0)

    def _normalize_states(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        expected = 9 + 5 * self.max_peds
        if states.shape[1] < expected:
            states = np.pad(
                states,
                ((0, 0), (0, expected - states.shape[1])),
            )
        return states[:, :expected]

    def predict_action_rollout_batch(
        self,
        states_34d,
        actions_xy,
        belief_vecs=None,
        horizon=1,
        dt=0.25,
        safe_distance=0.2,
        human_radius_default=0.3,
        cvar_alpha=0.80,
        pedestrian_aggregation="mean",
    ):
        states = self._normalize_states(states_34d)
        actions = np.asarray(actions_xy, dtype=np.float64).reshape(len(states), 2)
        batch_size = len(states)
        ped_count = self.modeled_peds

        peds = np.zeros((batch_size, ped_count, 5), dtype=np.float64)
        for ped_idx in range(ped_count):
            start = 9 + 5 * ped_idx
            peds[:, ped_idx] = states[:, start:start + 5]

        valid = ~np.all(peds == 0.0, axis=-1)
        valid_count = np.maximum(valid.sum(axis=1), 1)
        robot_radius = np.where(states[:, 4] > 0.0, states[:, 4], 0.3)
        human_radius = np.where(
            peds[:, :, 4] > 0.0,
            peds[:, :, 4],
            float(human_radius_default),
        )
        combined_radius = robot_radius[:, None] + human_radius

        risk_max = np.zeros((batch_size, ped_count), dtype=np.float64)
        min_clearance = np.full((batch_size, ped_count), np.inf, dtype=np.float64)
        robot_start = states[:, :2]

        for step in range(1, max(1, int(horizon)) + 1):
            elapsed = float(step) * float(dt)
            robot_future = robot_start + actions * elapsed
            human_future = peds[:, :, :2] + peds[:, :, 2:4] * elapsed
            clearance = (
                np.linalg.norm(human_future - robot_future[:, None, :], axis=-1)
                - combined_radius
            )
            risk_max = np.maximum(
                risk_max,
                self._clearance_risk(clearance, safe_distance),
            )
            min_clearance = np.minimum(min_clearance, clearance)

        valid_float = valid.astype(np.float64)
        zeros = np.zeros(batch_size, dtype=np.float64)
        if pedestrian_aggregation not in {"mean", "max"}:
            raise ValueError(
                "pedestrian_aggregation must be 'mean' or 'max'"
            )
        if pedestrian_aggregation == "max":
            invalid_fill = np.full_like(risk_max, -np.inf)
            aggregate_risk = np.where(
                valid,
                risk_max,
                invalid_fill,
            ).max(axis=1)
            aggregate_risk = np.where(
                valid.any(axis=1),
                aggregate_risk,
                0.0,
            )
        else:
            aggregate_risk = (
                (risk_max * valid_float).sum(axis=1) / valid_count
            )

        return {
            'risk': aggregate_risk,
            'tail_risk': aggregate_risk.copy(),
            'entropy': zeros.copy(),
            'klda': zeros.copy(),
            'epistemic_value': zeros.copy(),
            'risk_by_ped': risk_max,
            'tail_risk_by_ped': risk_max.copy(),
            'valid_mask': valid,
            'min_clearance': np.min(
                np.where(valid, min_clearance, np.inf),
                axis=1,
            ),
        }

    def predict_action_rollout(
        self,
        state_34d,
        action_xy,
        horizon=1,
        dt=0.25,
        robot_radius=0.3,
        human_radius_default=0.3,
        safe_distance=0.2,
    ):
        result = self.predict_action_rollout_batch(
            np.asarray(state_34d, dtype=np.float64).reshape(1, -1),
            np.asarray(action_xy, dtype=np.float64).reshape(1, 2),
            horizon=horizon,
            dt=dt,
            safe_distance=safe_distance,
            human_radius_default=human_radius_default,
        )
        return {
            'risk': float(result['risk'][0]),
            'tail_risk': float(result['tail_risk'][0]),
            'entropy': 0.0,
            'klda': 0.0,
            'epistemic_value': 0.0,
            'min_clearance': float(result['min_clearance'][0]),
            'belief_vec': self.get_per_ped_belief_vec(),
            'action_fitted': True,
        }
