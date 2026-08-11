"""Implicit Quantile Network value head + quantile Huber loss (guide.md 5.3, A6).

Outputs a full return DISTRIBUTION (one network, one objective) -- no
80-dim direct-Q head, no task/risk dual head (guide.md 3.3 explicitly
forbids both).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowd_nav.bayesian_dvl.intent_runtime_config import FROZEN_VALUES, derive_return_bounds


class CosineTauEmbedding(nn.Module):
    """tau_embedding = ReLU(Linear(cos(pi * i * tau))), i=1..n_cosines."""

    def __init__(self, embedding_dim: int, n_cosines: int = 64):
        super().__init__()
        self.n_cosines = n_cosines
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(n_cosines, embedding_dim)
        self.register_buffer(
            "_i_pi", (torch.arange(1, n_cosines + 1, dtype=torch.float32) * math.pi), persistent=False
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: [B, K] in (0,1) -> [B, K, embedding_dim]
        angles = tau.unsqueeze(-1) * self._i_pi  # [B, K, n_cosines]
        cos_features = torch.cos(angles)
        return F.relu(self.linear(cos_features))


class IQNValueNetwork(nn.Module):
    """Z_psi(state_embedding, action_embedding; tau) -> quantile value
    estimates -- Q(s,b,a;tau), not V(s',b') (guide.md R4-1).

    Not ``80`` action-indexed: this network scores ONE (state, belief,
    action) embedding triple at a time. The 80-action lookahead in
    policy.py calls it once per candidate action, batched.

    R2-2 fix (2026-08-07, independent diagnosis of BDVL's 66/200
    common-timeout episodes): the raw linear value head was unbounded,
    and a single-episode trace found a chosen action's score of 1.276
    when the reward definition makes no episode return exceed ~1.08 --
    an impossible value, i.e. genuine miscalibration/OOD extrapolation
    in the low-tau region, not "correctly conservative but too
    cautious". ``v_min``/``v_max`` (from
    ``config.derive_return_bounds``) are baked into the network via a
    monotonic sigmoid squash so NO raw output, at any tau, at any
    state, can ever leave the analytically-derived achievable range --
    this is an architectural guarantee, not a post-hoc ``clip()`` that
    would just hide a badly-trained network's symptoms at inference
    time while leaving training targets free to blow up.

    R4-1 fix (2026-08-10, guide.md "R4 -- Belief-Bypass Remediation
    Plan"): the R4-0 audit confirmed that scoring only the
    forward-simulated successor state's embedding lets the network
    infer "what happens next" without ever being forced to represent
    the acting agent's OWN choice as a first-class input, which is part
    of why continuation-value ended up bypassing belief while still
    reacting to raw human kinematics. ``action_embedding`` is now a
    second, explicit input, fused with ``state_embedding`` by a small
    linear layer BEFORE the tau-conditioned value head -- the fusion
    output has the same dimension as before so the value head itself is
    unchanged; only the input contract gained a mandatory second tensor.
    """

    def __init__(
        self, state_embedding_dim: int, action_embedding_dim: int, n_cosines: int = 64, hidden_dim: int = 128,
        v_min: Optional[float] = None, v_max: Optional[float] = None,
    ):
        super().__init__()
        derived_min, derived_max = derive_return_bounds(FROZEN_VALUES)
        v_min = derived_min if v_min is None else float(v_min)
        v_max = derived_max if v_max is None else float(v_max)
        if not (v_max > v_min):
            raise ValueError(f"v_max must exceed v_min, got v_min={v_min}, v_max={v_max}")
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.state_embedding_dim = state_embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(state_embedding_dim + action_embedding_dim, state_embedding_dim), nn.ReLU(),
        )
        self.tau_embedding = CosineTauEmbedding(state_embedding_dim, n_cosines)
        self.value_head = nn.Sequential(
            nn.Linear(state_embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_embedding: torch.Tensor, action_embedding: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # state_embedding: [B, Ds], action_embedding: [B, Da], tau: [B, K] -> quantile_values: [B, K]
        if action_embedding.shape[0] != state_embedding.shape[0]:
            raise ValueError(
                f"state_embedding batch {state_embedding.shape[0]} != action_embedding batch {action_embedding.shape[0]}"
            )
        fused = self.fusion(torch.cat([state_embedding, action_embedding], dim=-1))  # [B, Ds]
        tau_embed = self.tau_embedding(tau)  # [B, K, Ds]
        merged = fused.unsqueeze(1) * tau_embed  # [B, K, Ds]
        raw = self.value_head(merged).squeeze(-1)  # [B, K], unbounded
        quantiles = self.v_min + (self.v_max - self.v_min) * torch.sigmoid(raw)
        return quantiles


def quantile_huber_loss(
    predicted_quantiles: torch.Tensor,  # [B, K1] estimated at tau
    tau: torch.Tensor,  # [B, K1] the fractions predicted_quantiles were estimated at
    target_quantile_samples: torch.Tensor,  # [B, K2] target return samples (not necessarily at specific taus)
    kappa: float = 1.0,
) -> torch.Tensor:
    """Standard IQN pairwise quantile Huber loss (Dabney et al. 2018).
    Every predicted quantile is compared against every target sample;
    the asymmetric tau-weighting is what makes low tau underestimate
    and high tau overestimate in expectation, recovering the quantile
    function."""
    B, K1 = predicted_quantiles.shape
    K2 = target_quantile_samples.shape[1]

    pred = predicted_quantiles.unsqueeze(2)  # [B, K1, 1]
    target = target_quantile_samples.unsqueeze(1)  # [B, 1, K2]
    td_error = target - pred  # [B, K1, K2]

    huber = F.huber_loss(pred.expand(-1, -1, K2), target.expand(-1, K1, -1), delta=kappa, reduction="none")
    tau_expanded = tau.unsqueeze(2)  # [B, K1, 1]
    weight = torch.abs(tau_expanded - (td_error.detach() < 0).float())
    loss = weight * huber / kappa
    return loss.sum(dim=1).mean(dim=1).mean()


def expert_ranking_loss(action_scores: torch.Tensor, expert_indices: Sequence[int], margin: float) -> torch.Tensor:
    """guide.md R3R-1's Stage 1 ranking supervision -- a DQfD-style
    hardest-negative margin loss:

        L = relu(margin + max(score[non-expert]) - max(score[expert]))

    R3R-1 fix (2026-08-07, independent audit of the R3 implementation):
    the previous version used logsumexp over ALL non-expert actions as
    a supposed hard-negative proxy. That is WRONG, not just
    approximate: logsumexp(x_1..x_n) grows with n even when every x_i
    is identical (logsumexp of n equal values v is v + log(n)), so with
    79 non-expert actions the non-expert side carries a systematic
    +log(79)=+4.37 advantage having nothing to do with any action's
    actual score. Verified by direct computation: at equal scores the
    old loss was 4.4808 (not ~0), and even with the expert action at
    the network's theoretical MAXIMUM return and all 79 others at the
    theoretical MINIMUM, the old loss floored at 0.8479 -- it could
    architecturally never reach zero, so gradient direction was
    correct but magnitude was permanently distorted by the 80-vs-1
    set-size asymmetry, not by how well-separated the scores actually
    were.

    The max/max formulation above has none of that: it compares
    exactly two numbers (best expert, hardest actual negative) and is
    identical regardless of how many candidates were in either set --
    verified by a selftest that 1/2/4/8-sized expert sets with the
    same best-expert/best-negative gap produce IDENTICAL loss. relu
    (not softplus) is used deliberately so the loss AND its gradient
    are both exactly zero once the margin is satisfied, matching
    guide.md R3R-1's explicit acceptance criterion -- softplus's
    gradient never fully vanishes, which is a much smaller version of
    the same "the loss can't represent 'done'" problem this fix exists
    to remove.
    """
    n = action_scores.shape[0]
    if not expert_indices:
        raise ValueError("expert_indices must be non-empty")
    expert_mask = torch.zeros(n, dtype=torch.bool, device=action_scores.device)
    expert_mask[list(expert_indices)] = True
    if bool(expert_mask.all()):
        raise ValueError("expert_indices cannot cover every candidate action (no negatives to rank against)")
    expert_best = action_scores[expert_mask].max()
    hard_negative = action_scores[~expert_mask].max()
    return F.relu(margin + hard_negative - expert_best)


def quantile_huber_loss_hand_check(pred_scalar: float, tau_scalar: float, target_scalar: float, kappa: float = 1.0) -> float:
    """Pure-python reference for a single (prediction, tau, target)
    triple, used only by the selftest to hand-verify the batched/
    vectorized loss above is not silently wrong."""
    diff = target_scalar - pred_scalar
    if abs(diff) <= kappa:
        huber = 0.5 * diff * diff
    else:
        huber = kappa * (abs(diff) - 0.5 * kappa)
    indicator = 1.0 if diff < 0 else 0.0
    weight = abs(tau_scalar - indicator)
    return weight * huber / kappa
