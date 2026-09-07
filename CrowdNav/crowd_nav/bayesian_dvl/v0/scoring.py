"""V0 action selection: one-step successor-state lookahead.

    a* = argmax_a [ r(s,a) + gamma * V(s'_a, b) ]

Cost note. The humans' predicted next state is independent of the robot's
action -- the robot is invisible, so pedestrians do not react to it -- and the
belief is likewise unchanged by the action. So the expensive part (posterior
future sampling, 60 samples per human) runs ONCE per decision, and each of the
80 candidates only needs the robot-relative columns rewritten. Measured on the
`full` arm: 6.37 s/step rebuilding every row, ~0.08 s/step this way, identical
numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.geometry_features import _robot_feature_vector
from crowd_nav.bayesian_dvl.intent_runtime_config import FROZEN_VALUES
from crowd_nav.bayesian_dvl.intent_policy import build_intent_human_feature_batch
from crowd_nav.bayesian_dvl.v0 import lookahead as LA


@dataclass(frozen=True)
class CandidateInputs:
    """Everything about one decision that does NOT depend on the network.

    Split out so training and deployment cannot drift apart: the ranking loss
    needs gradients through V and deployment does not, but both must score the
    same tensors built the same way. Anything that is a function of the world
    lives here; the only thing left downstream is one forward pass.
    """
    rows: np.ndarray          # [n_actions, MAX_HUMANS, HUMAN_FEATURE_DIM]
    mask: np.ndarray          # [MAX_HUMANS]
    robot_feats: np.ndarray   # [n_actions, ROBOT_FEATURE_DIM]
    rewards: np.ndarray       # [n_actions]  exact, matches crowd_sim.step()
    terminal: np.ndarray      # [n_actions]  bool


def build_candidate_inputs(bank, robot: RobotObservation, humans: Sequence[HumanObservation],
                           action_table: np.ndarray, remaining_fraction: float, *,
                           mode: str, rng: np.random.Generator, n_samples: int,
                           feature_horizon: int) -> CandidateInputs:
    """The world half of a decision: predicted crowd, 80 robot successors,
    exact immediate rewards.

    ``feature_horizon`` MUST equal the horizon the training rows were built
    with -- it sets how far ahead the posterior future is summarised into the
    fdx/fdy/spread columns, which is where the belief's prediction lives.
    Getting it wrong gives the same three columns a 1-step meaning in one place
    and an 8-step meaning in the other, the identical failure shape as the
    16-vs-32 quantile split.
    """
    n_a = len(action_table)
    # once per decision: the humans' predicted next state does not depend on the
    # robot's action (the robot is invisible), and neither does the belief
    pred_humans = LA.predict_humans(bank, humans, mode, rng, n_samples)
    base_rows, base_mask = build_intent_human_feature_batch(
        bank, robot, pred_humans, mode=mode, rng=rng,
        horizon=feature_horizon, n_samples=n_samples)

    # the successor is one interval ahead, so its time feature must be too
    rem_next = max(0.0, remaining_fraction - LA.DT / float(FROZEN_VALUES["time_limit"]))
    rows = np.empty((n_a,) + base_rows.shape, dtype=np.float32)
    robot_feats = np.empty((n_a, 7), dtype=np.float32)
    rewards = np.empty(n_a, dtype=np.float64)
    terminal = np.zeros(n_a, dtype=bool)
    for i, (vx, vy) in enumerate(action_table):
        r, term, _ = LA.immediate_reward(robot, humans, float(vx), float(vy))
        rewards[i], terminal[i] = r, term
        rp = LA.robot_successor(robot, float(vx), float(vy))
        rows[i] = LA.patch_robot_columns(base_rows, base_mask, rp, pred_humans)
        robot_feats[i] = _robot_feature_vector(rp, rem_next)
    return CandidateInputs(rows, base_mask, robot_feats, rewards, terminal)


def score_from_inputs(model, ci: CandidateInputs, gamma: float, device: str):
    """The network half. Differentiable -- the ranking loss needs gradients
    through V, and deployment simply calls it inside no_grad.

    A terminal successor has no future: bootstrapping past the end of an
    episode would credit a collision with whatever the value head predicts for
    a state the robot never reaches.
    """
    n_a = ci.rows.shape[0]
    v = model(
        torch.as_tensor(ci.robot_feats, device=device),
        torch.as_tensor(ci.rows, device=device),
        torch.as_tensor(np.tile(ci.mask[None], (n_a, 1)), device=device),
    )
    v = torch.where(torch.as_tensor(ci.terminal, device=device), torch.zeros_like(v), v)
    r = torch.as_tensor(ci.rewards, dtype=v.dtype, device=device)
    return r + gamma * v, r, v


@torch.no_grad()
def score_actions(model, bank, robot: RobotObservation, humans: Sequence[HumanObservation],
                  action_table: np.ndarray, remaining_fraction: float, *,
                  mode: str, rng: np.random.Generator, n_samples: int,
                  gamma: float, device: str,
                  feature_horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deployment scoring: the two halves above, nothing else."""
    ci = build_candidate_inputs(bank, robot, humans, action_table, remaining_fraction,
                                mode=mode, rng=rng, n_samples=n_samples,
                                feature_horizon=feature_horizon)
    model.eval()
    s, r, v = score_from_inputs(model, ci, gamma, device)
    return s.cpu().numpy().astype(np.float64), ci.rewards, v.cpu().numpy().astype(np.float64)
