"""Permutation-invariant, mask-aware crowd encoder (guide.md 5.2, A6).

No Mamba/GRU/LSTM/history tokens: all temporal state lives ONLY in the
Bayesian filter (belief.py); this module is a pure per-timestep set
function. Trained on 5 humans, must generalize unmodified to 10/12/20
(guide.md formal scenarios) because pooling ops (mean/max/attention)
have no fixed input arity.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# guide.md 5.1 canonical layouts (R2 fix, independent audit B0,
# 2026-08-06: the previous 4-dim human feature carried no belief/
# entropy/track_age/predictive-moment information at all, making the
# implemented network Z(s;tau) rather than the specified Z(s,b;tau)).
#
# robot_features: goal-relative dx, dy; current vx, vy; radius; v_pref;
# remaining_fraction (R2 fix, 2026-08-07: v1 had no way to distinguish
# t=5s from t=34s at the same geometry, so the value function could not
# represent "continuing to circle now costs you the timeout"). Bumping
# this dimension makes v1 checkpoints structurally incompatible
# (nn.Linear input shape mismatch) -- see config.FEATURE_SCHEMA_V2.
ROBOT_FEATURE_DIM = 7
# human_features[i]: relative dx, dy; relative dvx, dvy; radius; speed;
# TTC; belief[5]; entropy; track_age; predictive mean[2]; predictive
# covariance upper-triangle[3] (var_a, cov_a_omega, var_omega)
HUMAN_FEATURE_DIM = 19

# R4-1 (guide.md "R4 -- Belief-Bypass Remediation Plan"): action_features:
# normalized vx, vy, speed, goal_alignment (cosine of action heading vs.
# CURRENT goal-relative direction), turn_cost (0=no heading change, 1=full
# reversal relative to the robot's CURRENT velocity heading). See
# policy.compute_action_features_array for the single canonical builder.
ACTION_FEATURE_DIM = 5


class HumanMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, human_features: torch.Tensor) -> torch.Tensor:
        # human_features: [..., N, in_dim] -> [..., N, out_dim]. Applying
        # the SAME shared MLP per-human (no cross-human mixing here) is
        # what makes the eventual pooled output permutation-invariant.
        return self.net(human_features)


class AttentionPool(nn.Module):
    """Single-head additive attention pooling of human embeddings,
    queried by the robot embedding. Mask-aware: masked-out humans get
    -inf logits before softmax, contributing exactly zero weight."""

    def __init__(self, human_dim: int, query_dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(human_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, human_embeddings: torch.Tensor, robot_query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # human_embeddings: [B, N, H], robot_query: [B, Q], mask: [B, N] bool
        query = self.query_proj(robot_query).unsqueeze(1)  # [B, 1, hidden]
        keys = self.key_proj(human_embeddings)  # [B, N, hidden]
        scores = self.score_proj(torch.tanh(query + keys)).squeeze(-1)  # [B, N]
        scores = scores.masked_fill(~mask, float("-inf"))
        # A fully-masked row (zero humans) would produce all -inf ->
        # softmax NaN; guard explicitly (guide.md A11: zero-human case
        # must not produce NaN).
        any_valid = mask.any(dim=1, keepdim=True)  # [B, 1]
        safe_scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=1)  # [B, N]
        weights = weights * mask.float()
        weights = torch.where(
            any_valid, weights, torch.zeros_like(weights)
        )
        pooled = torch.einsum("bn,bnh->bh", weights, human_embeddings)
        return pooled


class ActionEncoder(nn.Module):
    """e_action(a) = MLP(ActionFeature(a)). Deliberately separate from
    SetEncoder's robot_mlp: robot_mlp encodes the CANDIDATE SUCCESSOR
    state (where the robot ends up), ActionEncoder encodes the action's
    OWN identity at the current decision point (guide.md R4-1: the
    network must not be able to infer which action was taken solely from
    the successor state -- a genuine, separate action-identifying input
    is required)."""

    def __init__(self, action_feature_dim: int = ACTION_FEATURE_DIM, hidden_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(action_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.net(action_features)


class CandidateSetEncoder(nn.Module):
    """Per-human candidate goals -> ONE permutation-invariant embedding.

    Replaces feeding the network a bare ``p0..p7`` probability vector. Those
    slots were positional, and their meaning was NOT stable: candidates are
    produced by filtering public destinations against the observed entry, so
    on the held-out junction crowd ``p0`` meant "left exit" for 412 humans
    and "right exit" for 68 others (measured). The network had no way to
    tell those apart, because the slot index was the only thing identifying
    a candidate and no coordinates were supplied.

    Each candidate now carries its own geometry (probability + where it ends
    + where its next waypoint is, all relative to the human), a shared MLP
    embeds each one independently, and masked mean/max pooling collapses the
    set. Pooling has no fixed arity and no slot order, so shuffling the
    candidates cannot change the output -- asserted by
    ``test_candidate_encoding_is_permutation_invariant``.
    """

    def __init__(self, candidate_feature_dim: int, hidden_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, candidate_features: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        # candidate_features: [B, N, G, F], candidate_mask: [B, N, G] bool
        embeds = self.net(candidate_features)                     # [B, N, G, E]
        mask_f = candidate_mask.float().unsqueeze(-1)             # [B, N, G, 1]
        n_valid = mask_f.sum(dim=2).clamp(min=1.0)                # [B, N, 1]
        mean_pool = (embeds * mask_f).sum(dim=2) / n_valid        # [B, N, E]

        very_negative = torch.finfo(embeds.dtype).min
        max_pool = embeds.masked_fill(~candidate_mask.unsqueeze(-1), very_negative).max(dim=2).values
        any_valid = candidate_mask.any(dim=2, keepdim=True)       # [B, N, 1]
        # A human whose candidate block is entirely masked (the mean/cv arms,
        # which get no per-goal information at all) must contribute exactly
        # zero here, not the min-fill sentinel.
        max_pool = torch.where(any_valid, max_pool, torch.zeros_like(max_pool))
        mean_pool = torch.where(any_valid, mean_pool, torch.zeros_like(mean_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)           # [B, N, 2E]


class SetEncoder(nn.Module):
    """robot_features [B, ROBOT_FEATURE_DIM], human_features [B, N, HUMAN_FEATURE_DIM],
    human_mask [B, N] bool -> state_embedding [B, embedding_dim].

    ``human_features`` is the packed V6 row: a scalar block, then the
    per-candidate block, then the per-candidate validity mask. It is packed
    into one tensor rather than passed as three so that replay storage,
    batching, checkpointing and every call site keep the [B, N, D] shape
    they already had; this encoder is the one place that knows the layout.
    """

    def __init__(
        self,
        robot_feature_dim: int = ROBOT_FEATURE_DIM,
        human_feature_dim: int = HUMAN_FEATURE_DIM,
        human_hidden_dim: int = 64,
        human_embed_dim: int = 32,
        robot_embed_dim: int = 32,
        embedding_dim: int = 128,
        scalar_dim: int = None,
        max_candidates: int = None,
        candidate_feature_dim: int = None,
    ):
        super().__init__()
        from crowd_nav.bayesian_dvl.intent_runtime_config import (
            CANDIDATE_FEATURE_DIM, HUMAN_SCALAR_DIM_V6, MAX_CANDIDATE_GOALS,
        )
        self.scalar_dim = HUMAN_SCALAR_DIM_V6 if scalar_dim is None else int(scalar_dim)
        self.max_candidates = MAX_CANDIDATE_GOALS if max_candidates is None else int(max_candidates)
        self.candidate_feature_dim = (
            CANDIDATE_FEATURE_DIM if candidate_feature_dim is None else int(candidate_feature_dim))
        expected = (self.scalar_dim + self.max_candidates * self.candidate_feature_dim
                    + self.max_candidates)
        if human_feature_dim != expected:
            raise ValueError(
                f"human_feature_dim {human_feature_dim} does not match the packed V6 layout "
                f"(scalar {self.scalar_dim} + {self.max_candidates}x{self.candidate_feature_dim} "
                f"candidates + {self.max_candidates} mask = {expected})")
        self.candidate_encoder = CandidateSetEncoder(self.candidate_feature_dim)
        human_mlp_in = self.scalar_dim + 2 * self.candidate_encoder.embed_dim
        self.human_mlp = HumanMLP(human_mlp_in, human_hidden_dim, human_embed_dim)
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_feature_dim, robot_embed_dim), nn.ReLU(),
        )
        self.attention_pool = AttentionPool(human_embed_dim, robot_embed_dim, hidden_dim=32)
        context_dim = human_embed_dim + human_embed_dim + human_embed_dim  # mean + max + attention
        self.fusion_mlp = nn.Sequential(
            nn.Linear(robot_embed_dim + context_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
        )

    def split_packed(self, human_features: torch.Tensor):
        """[B, N, D] packed row -> (scalars, candidate features, candidate mask)."""
        g, f = self.max_candidates, self.candidate_feature_dim
        scalars = human_features[..., :self.scalar_dim]
        block = human_features[..., self.scalar_dim:self.scalar_dim + g * f]
        cand = block.reshape(*block.shape[:-1], g, f)
        cand_mask = human_features[..., self.scalar_dim + g * f:] > 0.5
        return scalars, cand, cand_mask

    def forward(self, robot_features: torch.Tensor, human_features: torch.Tensor, human_mask: torch.Tensor) -> torch.Tensor:
        robot_embed = self.robot_mlp(robot_features)  # [B, robot_embed_dim]
        scalars, cand_feats, cand_mask = self.split_packed(human_features)
        cand_embed = self.candidate_encoder(cand_feats, cand_mask)   # [B, N, 2E]
        human_embeds = self.human_mlp(torch.cat([scalars, cand_embed], dim=-1))  # [B, N, human_embed_dim]

        mask_f = human_mask.float().unsqueeze(-1)  # [B, N, 1]
        n_valid = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1], avoid div-by-zero for zero-human states
        mean_pool = (human_embeds * mask_f).sum(dim=1) / n_valid  # [B, H]

        very_negative = torch.finfo(human_embeds.dtype).min
        masked_for_max = human_embeds.masked_fill(~human_mask.unsqueeze(-1), very_negative)
        max_pool = masked_for_max.max(dim=1).values  # [B, H]
        # Zero-human rows: max over an all-"very_negative" row is still
        # finite but meaningless; zero it out explicitly.
        any_valid = human_mask.any(dim=1, keepdim=True)  # [B, 1]
        max_pool = torch.where(any_valid, max_pool, torch.zeros_like(max_pool))

        attn_pool = self.attention_pool(human_embeds, robot_embed, human_mask)  # [B, H]

        context = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        fused = torch.cat([robot_embed, context], dim=-1)
        return self.fusion_mlp(fused)
