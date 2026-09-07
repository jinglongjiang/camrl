#!/usr/bin/env python3
"""Train the belief-MDP policy: Stage 1 teacher imitation, Stage 2 Double-DQN.

Q_score(b, a) = Q_R(b, a) - beta * Q_C(b, a) from the first training step of
Stage 1 onward. Q_R and Q_C each have their own Bellman target (task reward,
risk cost) and are combined only at decision time. Mamba's own encoder is
frozen and reused for two purposes: (1) TaskEncoder context (feeds Q_R only,
see model.BeliefMDPQNetwork), and (2) the Stage 1/DAgger expert label source
via one-step value-lookahead (``runtime.BeliefMDPFeatureEngine.
teacher_scores``, see the Round 8 revision note below) -- the frozen
checkpoint's own value head is never fine-tuned or backpropagated into by
this script.

Revision note (Round 11: teacher supervises Q_R only; hard top-3 gate
replaced with paired non-inferiority vs the frozen baseline):

The Round 10 run cleared Stage 1's nav bar easily on every single check
(nominal/stress SR 92-100%, CR 0-7%) but never passed the gate, because
``top1_in_expert_set`` -- was the network's *final* decision (Q_score =
Q_R - beta*Q_C) among the teacher's top-3 actions -- stayed flat at ~64%
across Stage 1b and all 5 DAgger rounds despite near-ceiling navigation.
Two compounding problems, not one:

1. The margin loss was applied to Q_score, not Q_R. Since Q_score =
   Q_R - beta*Q_C, demanding Q_score agree with the teacher created a
   direct incentive to shrink Q_C wherever its risk signal would otherwise
   move Q_score's argmax away from the teacher's pick -- Q_C being
   penalized for doing exactly what it exists to do. Fixed: the margin
   loss (``optimize()``) now targets Q_R only; Q_C keeps learning real
   simulator collision cost only (unchanged, see td_loss_c/mc_loss_c) and
   is never supervised by the teacher at all. ``risk_cost`` is currently a
   pure terminal-collision indicator (``run_episode``: ``float(result.done
   and result.outcome == "collision")``) -- there is no near-miss/dmin
   component in it despite some older comments elsewhere in this project's
   history describing one; do not describe it as "collision/near-miss
   cost" (Round 12 correction).
2. A hard "is the chosen action literally in the teacher's top-3" test
   cannot distinguish "picked an equally-good alternative action" (should
   not count against the policy) from "picked something the teacher
   considered clearly worse" (should). Replaced with
   ``normalized_teacher_regret`` -- (teacher's best score - teacher's score
   for the chosen action) / teacher's own score range for that state,
   using the full 80-action teacher score vector now stored per transition
   (``model.ReplayBuffer.teacher_scores``) -- and made purely a reported
   diagnostic, never a gate criterion. ``run_stage1_gate_check``'s hard
   pass/fail is now a *paired* non-inferiority comparison against the
   frozen Mamba-VL baseline (``evaluate_quick_baseline``, a dedicated
   baseline-policy instance, evaluated once at start-up on the identical
   seed/scenario/profile formula the gate reuses every check) plus the
   existing numerical-stability aborts -- teacher-agreement (both Q_R-only
   and Q_score/final-decision versions) and a "risk reduction" diagnostic
   (CR with vs without Q_C in the decision, same episodes) are logged in
   full but never block Stage 2.

Revision note (Round 10: train/validate on 5 people only; six scenarios and
heldout_nonstationary are the formal test, never touched here):

Round 8/9 collected Stage 1/DAgger/Stage 2 data and decided the Stage 1 gate
and DAgger's convergence-based revert using ``TRAIN_SCENARIOS = tuple(SIX_
SCENARIOS.keys())`` (all six scenarios, 5-20 people) and the
``heldout_nonstationary`` profile -- which are exactly the scenario/profile
combinations ``evaluate.py``'s formal generalization test later reports on.
Every automated decision in this file that used those numbers (Stage 1
gate pass/fail, DAgger round's convergence stop/revert, which checkpoint to
keep) had therefore already "seen" performance on the distribution this
project's central claim is about generalizing to zero-shot -- invalidating
that claim regardless of what the frozen final model's formal-eval numbers
turned out to be. Fixed by hard-separating three scenario/profile sets:
``TRAIN_SCENARIOS``/``VALIDATION_SCENARIOS`` (both ``("baseline_circle",)``,
5 people) for everything this file does, and the six-scenario battery +
``heldout_nonstationary`` reserved exclusively for ``evaluate.py``/
``select_checkpoint.py``'s one-time formal evaluation of the frozen model.
``VALIDATION_PROFILES = ("nominal", "train_nonstationary")`` -- both
already part of the training distribution (``run_episode`` samples between
them), so validation and training share a distribution the way train/val
splits ordinarily do; only the eventual formal test is a genuinely
different one. ``dagger_validation`` was also split into
``dagger_validation_recent`` (current round only, cleared and moved into
``dagger_validation_historical`` at the start of the next round, mirroring
``recent_dagger_buffer``/``older_dagger_buffer``) since the convergence
check needs "did *this* round's updates help on states *this* round's
policy visits," not a metric diluted by earlier, weaker rounds.
``select_checkpoint.py`` received the equivalent fix.

Revision note (Round 8: ORCA teacher replaced with frozen strong Mamba):

Rounds 1-7 (below) used ORCA as the Stage 1/DAgger expert label source. ORCA
is a classical planner with its own risk model, unrelated to the frozen
Mamba-VL policy this project is actually trying to approximate and augment
with belief -- so the imitation target and the eventual RL objective could
legitimately disagree about what a good action looks like, for reasons that
have nothing to do with the belief branch. The label source is now the
frozen Mamba checkpoint's own one-step value-lookahead ranking over the 80
discrete actions (``teacher_scores``): the "expert" the network imitates in
Stage 1/DAgger is the same strong policy Q_R is meant to approximate, so
imitation and RL point the same direction from the start. ``is_expert_action``
was also fixed to record the literal ``use_expert`` draw rather than an
``action_index == expert_index`` equality check, which could be
coincidentally true under exploration and did not actually mean "the
teacher acted here."

Revision note (Round 9: 5-way stratified replay + convergence-based DAgger
stopping, same pass as Round 8):

DAgger previously wrote every round's data into the same permanent buffer
used for Stage 1a's teacher demonstrations, mixed with online RL experience
via a single fixed demo/online ratio -- old rounds, the fixed teacher
demonstrations, and rare collision trajectories all competed for the same
undifferentiated pool with no control over the mix. Replaced with five
buffers sampled at fixed ratios (``sample_stratified``, 25/35/15/15/10 for
teacher_demo/recent_dagger/older_dagger/collision/online_rl); a round's
``recent_dagger`` data moves into ``older_dagger`` once the next round
starts (``ReplayBuffer.export_all``/``import_batch``); every colliding
episode's transitions are additionally copied into ``collision_buffer``
regardless of which stratum they normally belong to. Each DAgger round's
offline updates also no longer run a fixed count: ``run_dagger_round_updates``
runs up to a budget cap, checking every ``stage1_log_every`` updates whether
a nav SR/CR composite improved, stopping after ``dagger_convergence_patience``
consecutive non-improving checks and reverting to the round's best-seen
checkpoint (RL is not monotonic -- see the SR-drop discussion below -- so a
round can easily end on a real dip relative to its own peak otherwise).
Five other diagnostics (teacher/DAgger-state top-3 hit rate, Q_C collision
AUC/Brier, Q_R scale, beta-top1-change fraction) are logged at every check
for visibility but intentionally not folded into the stop/revert decision
itself, to avoid inventing an arbitrary weighted blend across incomparable
scales. Dueling Q (separate V_R/A_R heads) remains unimplemented: the
gradient-conflict diagnostic (``compute_gradient_conflict``) has shown no
sustained negative correlation between the value and margin-loss gradients
across many measurements, so there is still no evidence for it.

Revision note (root-cause fix after the first full training run diverged):

The first full run showed Q_C learning a real, stable, non-constant signal
(q_c_action_std settled at ~0.23-0.26, beta_top1_change_rate stayed nonzero
throughout) -- proof the Bayesian branch genuinely participates in RL
decisions. But Q_R inflated from O(1) to ~20 then collapsed to ~3-4, BC
loss rose toward log(80)=4.38 (chance level), and navigation performance
peaked mid-training (episode 1900) then degraded by the end. Root cause:
``bc_loss = F.cross_entropy(q_score, expert_action)`` used the *same* Q_R
both as a Bellman-fitted value and as a classification logit -- cross-
entropy has no notion of "close enough," so it kept demanding a larger
margin between the expert action's score and every other action's score
without limit, inflating Q_R independent of whether it still meant
anything as a return estimate. Fixed by:

1. A DQfD-style *fixed-margin* expert loss (``expert_margin_loss``) that
   stops contributing once the expert action leads by ``margin=0.05`` --
   it can no longer inflate Q values without bound.
2. Monte-Carlo return targets (``mc_task_return``, ``mc_collision_return``)
   computed once per episode by a reverse pass, added as extra regression
   terms -- every state in an eventually-colliding trajectory gets
   supervision, not just the single terminal transition risk_cost=1 covers.
3. A permanent demo buffer (never evicted by online experience) mixed at a
   fixed 25/75 ratio with the circular online buffer, so ORCA demonstrations
   cannot be crowded out as training progresses.
4. Expanded diagnostics and an automatic abort if Q_R's scale or NaNs
   indicate the same divergence recurring.
5. Periodic, unconditional checkpoints instead of an in-training "best"
   pick from a noisy 30-episode quick-eval -- model selection now happens
   after training, on independent validation seeds (see select_checkpoint.py).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_mdp.evaluate import run_mamba_baseline_episode  # noqa: E402
from crowd_nav.belief_mdp.hashing import sha256_dir, sha256_file  # noqa: E402
from crowd_nav.belief_mdp.model import (  # noqa: E402
    BeliefMDPQNetwork,
    ReplayBuffer,
    sample_stratified,
)
from crowd_nav.belief_mdp.runtime import (  # noqa: E402
    BeliefMDPFeatureEngine,
    FullCrowdNavigationEnvironment,
    VALIDATION_SCENARIOS,
)
from crowd_nav.belief_space_rl.runtime import (  # noqa: E402
    build_frozen_mamba,
    merged_policy_config,
)
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402


# Training uses a seed range disjoint from evaluate.py's formal seeds so no
# training scenario instance is ever reused as a held-out test case.
TRAIN_SEED_BASE = 300_000

# Round 10: train/validate on the 5-person scenario only (runtime.
# VALIDATION_SCENARIOS); the six-scenario battery (up to 20 people) and the
# heldout_nonstationary profile are the formal generalization *test*
# (evaluate.py/select_checkpoint.py), never touched anywhere in this file.
# Using them here -- even just to decide a gate pass/fail or which
# checkpoint to keep -- would mean checkpoint selection had already "seen"
# performance on the exact distribution later reported as zero-shot
# generalization, which is not a claim this project can make once that
# happens even once. All Stage 1/DAgger/Stage 2 data collection AND every
# in-training validation check (gate, DAgger convergence, Stage 2 periodic
# eval) use TRAIN_SCENARIOS/VALIDATION_SCENARIOS and VALIDATION_PROFILES
# exclusively. TRAIN_SCENARIOS and VALIDATION_SCENARIOS are the same tuple
# by design (train/validation share a distribution the way ordinary ML
# train/val splits do); kept as two names so call sites document intent.
TRAIN_SCENARIOS = VALIDATION_SCENARIOS
# nominal and train_nonstationary are both in-training-distribution profiles
# (the policy is trained under a random mix of the two -- see run_episode);
# heldout_nonstationary is reserved entirely for the one-time formal
# evaluation and must never appear in this file's own decision logic.
VALIDATION_PROFILES = ("nominal", "train_nonstationary")

# Auto-abort thresholds (see module docstring, item 4).
Q_R_ABS_P95_LIMIT = 3.0
SR_DROP_LIMIT = 0.20


class TrainingDiverged(RuntimeError):
    pass


def compute_artifact_hashes(args) -> dict:
    """Content hashes of every artifact that determines this checkpoint's
    behaviour, so evaluate.py can detect a config/GDBN-params/checkpoint
    swap that keeps the same path string but changes the content.

    Round 12: also hashes this project's own source files (train.py,
    model.py, runtime.py, evaluate.py) -- without this, a checkpoint from
    before a source fix (e.g. the teacher_scores() frame bug) and one from
    after it are otherwise indistinguishable from their recorded args/
    config hashes alone, since neither changed. A later evaluation run
    using patched source against an old checkpoint would silently look
    like "the same version," which is exactly the kind of mismatch that
    made Round 11's imitation numbers hard to trust in isolation.
    """
    hashes = {
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "gdbn_params_sha256": sha256_dir(args.gdbn_params),
        "policy_config_sha256": sha256_file(args.policy_config),
        "base_env_config_sha256": sha256_file(args.base_env_config),
        "env_config_sha256": sha256_file(args.env_config),
        "train_py_sha256": sha256_file(str(THIS_DIR / "train.py")),
        "model_py_sha256": sha256_file(str(THIS_DIR / "model.py")),
        "runtime_py_sha256": sha256_file(str(THIS_DIR / "runtime.py")),
        "evaluate_py_sha256": sha256_file(str(THIS_DIR / "evaluate.py")),
    }
    if args.belief_mode == "k1_belief":
        hashes["k1_gdbn_params_sha256"] = sha256_dir(args.k1_gdbn_params)
    return hashes


def linear_schedule(start: float, end: float, step: int, total: int) -> float:
    fraction = min(max(step / max(total, 1), 0.0), 1.0)
    return float(start + fraction * (end - start))


def tensor_features(features, device):
    return (
        torch.as_tensor(features.context, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(features.belief, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(features.candidates, dtype=torch.float32, device=device).unsqueeze(0),
    )


def select_action(network, features, device, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, len(features.candidates)))
    with torch.no_grad():
        q_score = network(*tensor_features(features, device))
    return int(q_score.argmax(dim=-1).item())


def expert_set_margin_loss(q_score: torch.Tensor, expert_action_set: torch.Tensor, margin: float = 0.05):
    """DQfD-style fixed-margin expert loss against a *set* of near-equivalent
    expert actions, not a single one.

    The teacher (frozen Mamba value-lookahead, see ``runtime.
    BeliefMDPFeatureEngine.teacher_scores``) often assigns nearly identical
    scores to several neighboring discrete actions -- e.g. a few adjacent
    headings that all clear the nearest pedestrian equally well this step.
    Treating only its single top-scoring action as "correct" and demanding
    a hard margin over every other action -- including those other
    near-tied ones -- makes the margin nearly always violated for reasons
    that have nothing to do with the policy being wrong. ``expert_action_set``
    (B, K) holds the K best-scoring discrete actions; the loss only asks
    that the best of those leads the best non-expert-set action by
    ``margin``.
    """
    expert_q = q_score.gather(1, expert_action_set).max(dim=1).values
    competitor = q_score.clone()
    competitor.scatter_(1, expert_action_set, -torch.inf)
    best_other = competitor.max(dim=1).values
    per_sample = F.relu(best_other + margin - expert_q)
    return per_sample


TEACHER_SCORE_MASK_THRESHOLD = -1e3  # below runtime.py's -1e4 safety-mask sentinel, with margin


def normalized_teacher_regret(
    chosen_action: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask_threshold: float = TEACHER_SCORE_MASK_THRESHOLD,
) -> torch.Tensor:
    """How much value the teacher's own full 80-action ranking assigns to
    ``chosen_action`` relative to its best action, normalized by the
    teacher's own score spread for that state -- 0 means the teacher agrees
    this was its best choice, 1 means it was as bad as or worse than its
    worst *safety-valid* action.

    Round 11: replaces the hard top-K "was the chosen action literally in
    the teacher's top-3" criterion for reporting. A hard top-K test cannot
    tell "picked an almost-equally-good alternative action" (small regret)
    apart from "picked something the teacher considered clearly worse"
    (large regret) -- exactly the ambiguity that made the old top1_in_
    expert_set gate uninformative once nav performance was already near
    ceiling (Round 10 run: ~64% top-3 hit rate, ~100% SR/~0% CR).

    Round 12 fix: ``runtime.BeliefMDPFeatureEngine.teacher_scores`` fills
    actions its own safety filter flags unsafe with a ``-1e4`` sentinel.
    Including those in the min/max range let a single masked action (found
    in 6/60 sampled real states) drag ``teacher_worst`` down to ~-1e4,
    inflating the range and compressing every regret value toward zero
    regardless of how different the real, unmasked scores actually were.
    Masked actions are now excluded from the range (falling back to the
    full range only if literally every action was masked); a chosen action
    that itself was masked clamps to a regret of 1.0 rather than an
    unbounded value that would dominate any batch average. Never used for
    gradients -- diagnostic only.
    """
    valid = teacher_scores > mask_threshold
    any_valid = valid.any(dim=-1, keepdim=True)
    effective_valid = torch.where(any_valid, valid, torch.ones_like(valid))
    teacher_best = teacher_scores.masked_fill(~effective_valid, float("-inf")).max(dim=-1).values
    teacher_worst = teacher_scores.masked_fill(~effective_valid, float("inf")).min(dim=-1).values
    teacher_range = (teacher_best - teacher_worst).clamp_min(1e-6)
    chosen_score = teacher_scores.gather(1, chosen_action[:, None]).squeeze(1)
    regret = (teacher_best - chosen_score) / teacher_range
    return regret.clamp(0.0, 1.0)


def optimize(network, target, optimizer, buffers, ratios, batch_size, gamma, lambda_e, rng, device):
    batch = sample_stratified(buffers, ratios, batch_size, rng, device)
    _, components = network(
        batch["context"], batch["belief"], batch["candidates"], return_components=True
    )
    q_r, q_c = components["q_r"], components["q_c"]
    chosen_q_r = q_r.gather(1, batch["action"][:, None]).squeeze(1)
    chosen_q_c = q_c.gather(1, batch["action"][:, None]).squeeze(1)

    with torch.no_grad():
        next_score_online, next_online = network(
            batch["next_context"], batch["next_belief"], batch["next_candidates"],
            return_components=True,
        )
        next_action = next_score_online.argmax(dim=-1)
        _, next_target = target(
            batch["next_context"], batch["next_belief"], batch["next_candidates"],
            return_components=True,
        )
        next_q_r = next_target["q_r"].gather(1, next_action[:, None]).squeeze(1)
        next_q_c = next_target["q_c"].gather(1, next_action[:, None]).squeeze(1)
        td_target_r = batch["task_reward"] + float(gamma) * (1.0 - batch["done"]) * next_q_r
        td_target_c = batch["risk_cost"] + float(gamma) * (1.0 - batch["done"]) * next_q_c

    td_loss_r = F.smooth_l1_loss(chosen_q_r, td_target_r)
    td_loss_c = F.smooth_l1_loss(chosen_q_c, td_target_c)
    mc_loss_r = F.smooth_l1_loss(chosen_q_r, batch["mc_task_return"])
    mc_loss_c = F.smooth_l1_loss(chosen_q_c, batch["mc_collision_return"])

    q_score = q_r - network.beta * q_c
    # Round 11: the teacher supervises the *task* head only. Margin loss on
    # q_score (the previous design) backpropagated into q_c_head too --
    # with a *negative* sign, since q_score = q_r - beta*q_c, so demanding
    # q_score agree with the teacher created a direct incentive to shrink
    # Q_C wherever its risk signal would otherwise move q_score's argmax
    # away from the teacher's pick. That is Q_C being penalized for doing
    # its job. Supervising q_r alone removes that fight entirely: Q_R learns
    # "how to get to the goal" from the teacher, Q_C learns real collision
    # cost from the simulator (unchanged, see td_loss_c/mc_loss_c above),
    # and only their combination at decision time is allowed to disagree
    # with the teacher.
    margin_per_sample = expert_set_margin_loss(q_r, batch["expert_action_set"], margin=0.05)
    # DQfD's large-margin supervised loss is only valid supervision for
    # transitions with a real expert label to imitate (Stage 1 demo
    # collection and DAgger-round data) -- applying it to self-generated
    # Stage-2-only experience would keep that experience anchored to the
    # teacher's instantaneous suggestion for the rest of training, contradicting "the
    # learned policy drives decisions." Weight by has_expert_label so only
    # labeled transitions contribute, regardless of who actually acted in
    # them (see model.ReplayBuffer for why that's a separate flag).
    demo_weight = batch["has_expert_label"]
    margin_loss = (margin_per_sample * demo_weight).sum() / demo_weight.sum().clamp_min(1.0)

    with torch.no_grad():
        # See module docstring for what each of these watches for.
        positive_risk_fraction = float(batch["risk_cost"].mean())
        positive_mc_collision_fraction = float((batch["mc_collision_return"] > 0).float().mean())
        q_c_action_std = float(q_c.std(dim=-1).mean())
        r_only_action = q_r.argmax(dim=-1)
        score_action = q_score.argmax(dim=-1)
        # The core "is the Bayesian branch doing anything" signal: how often
        # beta*Q_C actually changes the deployed decision away from the
        # task-only choice. Unlike the old design this is no longer in
        # tension with the margin loss (which now never touches q_c_head).
        beta_top1_change_rate = float((r_only_action != score_action).float().mean())
        q_r_abs_p95 = float(chosen_q_r.detach().abs().quantile(0.95))

        labeled_mask = batch["has_expert_label"]
        unlabeled_mask = 1.0 - labeled_mask
        labeled_count = labeled_mask.sum().clamp_min(1.0)
        unlabeled_count = unlabeled_mask.sum().clamp_min(1.0)

        # Q_R-only teacher agreement: what the margin loss actually trains
        # toward, and the meaningful "did the task head keep the navigation
        # skill the teacher demonstrated" check. Reported (see Round 11:
        # neither this nor the q_score version below gates Stage 1 anymore
        # -- run_stage1_gate_check uses a paired non-inferiority nav
        # comparison against the frozen Mamba baseline instead).
        qr_margin_violation_rate = float((margin_per_sample > 0).float().mean())
        qr_in_expert_set = (r_only_action[:, None] == batch["expert_action_set"]).any(dim=1).float()
        qr_top1_in_expert_set = float(qr_in_expert_set.mean())
        qr_expert_top1_agreement = float((r_only_action == batch["expert_action"]).float().mean())
        qr_regret = normalized_teacher_regret(r_only_action, batch["teacher_scores"])
        qr_regret_mean = float((qr_regret * labeled_mask).sum() / labeled_count)

        # Q_score (the actual deployed decision) teacher agreement: report
        # only, never a training target or a gate criterion. Expected to
        # legitimately diverge from the Q_R-only numbers above whenever
        # Q_C meaningfully changes the decision -- that divergence is the
        # point of having a risk head at all, not a defect to eliminate.
        score_margin_per_sample = expert_set_margin_loss(q_score, batch["expert_action_set"], margin=0.05)
        score_margin_violation_rate = float((score_margin_per_sample > 0).float().mean())
        score_in_expert_set = (score_action[:, None] == batch["expert_action_set"]).any(dim=1).float()
        score_top1_in_expert_set = float(score_in_expert_set.mean())
        score_expert_top1_agreement = float((score_action == batch["expert_action"]).float().mean())
        score_regret = normalized_teacher_regret(score_action, batch["teacher_scores"])
        score_regret_mean = float((score_regret * labeled_mask).sum() / labeled_count)

        demo_qr_top1_in_set = float((qr_in_expert_set * labeled_mask).sum() / labeled_count)
        demo_qr_margin_violation = float(
            ((margin_per_sample > 0).float() * labeled_mask).sum() / labeled_count
        )
        online_beta_top1_change = float(
            ((r_only_action != score_action).float() * unlabeled_mask).sum() / unlabeled_count
        )
        online_qc_action_std = float(
            (q_c.std(dim=-1) * unlabeled_mask).sum() / unlabeled_count
        )

        # Velocity/direction error between the network's own final decision
        # (score_action) and the single nearest expert action, read directly
        # from the kinematic slice of the candidate features (columns 6:8
        # are vx, vy -- see runtime.KINEMATIC_FEATURE_NAMES). Using
        # batch["action"] here would be wrong during Stage 1
        # (force_expert=True means the executed action *is* the expert
        # action by construction, so the "error" would be tautologically
        # zero) and misleading during DAgger/Stage 2 (it would measure the
        # exploration/behavior policy's error, not the learned policy's).
        chosen_velocity = batch["candidates"][:, :, 6:8].gather(
            1, score_action[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        expert_velocity = batch["candidates"][:, :, 6:8].gather(
            1, batch["expert_action"][:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        velocity_error = float(torch.linalg.norm(chosen_velocity - expert_velocity, dim=-1).mean())
        cosine = F.cosine_similarity(chosen_velocity, expert_velocity, dim=-1).clamp(-1.0, 1.0)
        angle_error_deg = float(torch.rad2deg(torch.acos(cosine)).mean())

    loss = (
        td_loss_r + 0.25 * mc_loss_r
        + td_loss_c + 1.00 * mc_loss_c
        + float(lambda_e) * margin_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "td_loss_r": float(td_loss_r.detach()),
        "td_loss_c": float(td_loss_c.detach()),
        "mc_loss_r": float(mc_loss_r.detach()),
        "mc_loss_c": float(mc_loss_c.detach()),
        "margin_loss": float(margin_loss.detach()),
        "q_r_mean": float(chosen_q_r.detach().mean()),
        "q_r_std": float(chosen_q_r.detach().std()),
        "q_r_abs_p95": q_r_abs_p95,
        "q_c_mean": float(chosen_q_c.detach().mean()),
        "q_c_std": float(chosen_q_c.detach().std()),
        "q_c_p95": float(chosen_q_c.detach().quantile(0.95)),
        "td_target_r_mean": float(td_target_r.detach().mean()),
        "td_target_c_mean": float(td_target_c.detach().mean()),
        "positive_risk_fraction": positive_risk_fraction,
        "positive_mc_collision_fraction": positive_mc_collision_fraction,
        "q_c_action_std": q_c_action_std,
        "beta_top1_change_rate": beta_top1_change_rate,
        "qr_margin_violation_rate": qr_margin_violation_rate,
        "qr_top1_in_expert_set": qr_top1_in_expert_set,
        "qr_expert_top1_agreement": qr_expert_top1_agreement,
        "qr_regret_mean": qr_regret_mean,
        "score_margin_violation_rate": score_margin_violation_rate,
        "score_top1_in_expert_set": score_top1_in_expert_set,
        "score_expert_top1_agreement": score_expert_top1_agreement,
        "score_regret_mean": score_regret_mean,
        "demo_qr_top1_in_set": demo_qr_top1_in_set,
        "demo_qr_margin_violation": demo_qr_margin_violation,
        "online_beta_top1_change": online_beta_top1_change,
        "online_qc_action_std": online_qc_action_std,
        "velocity_error": velocity_error,
        "angle_error_deg": angle_error_deg,
    }


def evaluate_demo_validation(network, buffer, device, chunk_size: int = 2048):
    """Diagnostic-only forward pass over every held-out demo transition.

    Never used for gradients; checks whether Stage 1 imitation actually
    generalized to demo episodes it never trained on, rather than reading
    only the last training minibatch (which could be an easy or lucky one).
    Reports both the Q_R-only agreement (what the margin loss actually
    trains toward -- see optimize()) and the Q_score/final-decision
    agreement (informational: expected to legitimately diverge from Q_R's
    whenever Q_C changes the decision). Round 11: neither number gates
    anything by itself; run_stage1_gate_check uses a paired non-inferiority
    nav comparison against the frozen Mamba baseline instead (a hard top-3
    membership test cannot distinguish "picked an equally-good alternative
    action" from "picked something the teacher considered clearly worse" --
    see normalized_teacher_regret).

    Uses ``ReplayBuffer.deterministic_chunks`` -- a fixed, no-replacement
    pass over the whole buffer -- instead of ``.sample()`` with the shared
    training RNG. Reusing the training RNG here would both consume from its
    stream (silently changing every later training batch's sampling order)
    and could draw the same held-out transition more than once while
    skipping others, biasing the estimate. This takes no RNG at all.
    """
    if buffer.size == 0:
        return None
    network.eval()
    total = 0
    qr_top1_hits = 0.0
    qr_margin_violations = 0.0
    qr_regret_sum = 0.0
    score_top1_hits = 0.0
    score_margin_violations = 0.0
    score_regret_sum = 0.0
    with torch.no_grad():
        for batch in buffer.deterministic_chunks(chunk_size, device):
            _, components = network(
                batch["context"], batch["belief"], batch["candidates"], return_components=True
            )
            q_r, q_c = components["q_r"], components["q_c"]
            q_score = q_r - network.beta * q_c
            r_only_action = q_r.argmax(dim=-1)
            score_action = q_score.argmax(dim=-1)

            qr_margin_per_sample = expert_set_margin_loss(q_r, batch["expert_action_set"], margin=0.05)
            qr_in_set = (r_only_action[:, None] == batch["expert_action_set"]).any(dim=1)
            score_margin_per_sample = expert_set_margin_loss(q_score, batch["expert_action_set"], margin=0.05)
            score_in_set = (score_action[:, None] == batch["expert_action_set"]).any(dim=1)

            chunk_count = r_only_action.shape[0]
            total += chunk_count
            qr_top1_hits += float(qr_in_set.float().sum())
            qr_margin_violations += float((qr_margin_per_sample > 0).float().sum())
            qr_regret_sum += float(normalized_teacher_regret(r_only_action, batch["teacher_scores"]).sum())
            score_top1_hits += float(score_in_set.float().sum())
            score_margin_violations += float((score_margin_per_sample > 0).float().sum())
            score_regret_sum += float(normalized_teacher_regret(score_action, batch["teacher_scores"]).sum())
    return {
        "episodes_held_out": buffer.size,
        "sample_size": total,
        "qr_top1_in_expert_set": qr_top1_hits / max(total, 1),
        "qr_margin_violation_rate": qr_margin_violations / max(total, 1),
        "qr_regret_mean": qr_regret_sum / max(total, 1),
        "score_top1_in_expert_set": score_top1_hits / max(total, 1),
        "score_margin_violation_rate": score_margin_violations / max(total, 1),
        "score_regret_mean": score_regret_sum / max(total, 1),
    }


def evaluate_qc_calibration(network, buffer, device, chunk_size: int = 4096):
    """How well Q_C's prediction for the *executed* action matches whether
    that trajectory actually went on to collide.

    Brier score: mean squared error between the chosen action's Q_C and its
    ``mc_collision_return`` -- exactly what Q_C's own MC loss regresses
    against, so this is a held-out-batch read of the same objective, not a
    new one. AUC: rank-based (Mann-Whitney U), treating ``mc_collision_return
    > 0`` as the binary "this trajectory eventually collided" label and
    Q_C as the score -- avoids adding a sklearn dependency for one metric.
    Diagnostic only, logged at every convergence check; never used for
    gradients or the accept/reject decision itself (see run_dagger_round_updates).
    """
    if buffer.size == 0:
        return None
    network.eval()
    labels = []
    scores = []
    brier_terms = []
    with torch.no_grad():
        for batch in buffer.deterministic_chunks(chunk_size, device):
            _, components = network(
                batch["context"], batch["belief"], batch["candidates"], return_components=True
            )
            chosen_q_c = components["q_c"].gather(1, batch["action"][:, None]).squeeze(1)
            label = (batch["mc_collision_return"] > 0).float()
            labels.append(label.cpu())
            scores.append(chosen_q_c.cpu())
            brier_terms.append(((chosen_q_c - batch["mc_collision_return"]) ** 2).cpu())
    labels = torch.cat(labels)
    scores = torch.cat(scores)
    brier = float(torch.cat(brier_terms).mean())
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    auc = None
    if positive.numel() > 0 and negative.numel() > 0:
        combined = torch.cat([positive, negative])
        ranks = combined.argsort().argsort().float() + 1.0
        n_pos, n_neg = positive.numel(), negative.numel()
        rank_sum_pos = ranks[:n_pos].sum()
        auc = float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return {
        "sample_size": int(labels.numel()),
        "positive_count": int(positive.numel()),
        "positive_fraction": float(labels.mean()),
        "q_c_brier": brier,
        "q_c_collision_auc": auc,
    }


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def compute_gradient_conflict(network, target, buffers, ratios, batch_size, gamma, rng, device):
    """Cosine similarity between the value-loss gradient (TD+MC for both
    heads) and the margin-loss gradient, over every shared parameter.

    Diagnostic only: uses ``torch.autograd.grad`` (never populates
    ``.grad`` or calls ``optimizer.step()``), so it cannot affect real
    training. This exists to settle "does the margin loss fight the value
    regression on the same Q_R output" with a measurement instead of an
    assumption -- Dueling Q (separating V_R/A_R) is only justified once
    this is confirmed persistently negative across many measurements, not
    guessed from the DAgger result alone. Returns ``None`` if the sampled
    batch happens to contain no expert-labeled transitions (the margin
    gradient is undefined without at least one).
    """
    batch = sample_stratified(buffers, ratios, batch_size, rng, device)
    if float(batch["has_expert_label"].sum()) < 1.0:
        return None
    _, components = network(
        batch["context"], batch["belief"], batch["candidates"], return_components=True
    )
    q_r, q_c = components["q_r"], components["q_c"]
    chosen_q_r = q_r.gather(1, batch["action"][:, None]).squeeze(1)
    chosen_q_c = q_c.gather(1, batch["action"][:, None]).squeeze(1)

    with torch.no_grad():
        next_score_online, _ = network(
            batch["next_context"], batch["next_belief"], batch["next_candidates"],
            return_components=True,
        )
        next_action = next_score_online.argmax(dim=-1)
        _, next_target = target(
            batch["next_context"], batch["next_belief"], batch["next_candidates"],
            return_components=True,
        )
        next_q_r = next_target["q_r"].gather(1, next_action[:, None]).squeeze(1)
        next_q_c = next_target["q_c"].gather(1, next_action[:, None]).squeeze(1)
        td_target_r = batch["task_reward"] + float(gamma) * (1.0 - batch["done"]) * next_q_r
        td_target_c = batch["risk_cost"] + float(gamma) * (1.0 - batch["done"]) * next_q_c

    value_loss = (
        F.smooth_l1_loss(chosen_q_r, td_target_r) + 0.25 * F.smooth_l1_loss(chosen_q_r, batch["mc_task_return"])
        + F.smooth_l1_loss(chosen_q_c, td_target_c) + 1.00 * F.smooth_l1_loss(chosen_q_c, batch["mc_collision_return"])
    )

    # Matches optimize()'s actual training loss (Round 11: margin on Q_R,
    # not Q_score) -- this diagnostic answers "does the margin loss, as
    # actually used, fight the value regression," so it must measure the
    # same margin loss that is actually used.
    margin_per_sample = expert_set_margin_loss(q_r, batch["expert_action_set"], margin=0.05)
    demo_weight = batch["has_expert_label"]
    margin_loss = (margin_per_sample * demo_weight).sum() / demo_weight.sum().clamp_min(1.0)

    params = [p for p in network.parameters() if p.requires_grad]
    value_grads = torch.autograd.grad(value_loss, params, retain_graph=True, allow_unused=True)
    margin_grads = torch.autograd.grad(margin_loss, params, retain_graph=False, allow_unused=True)

    def flatten(grads):
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params)
        ])

    cosine = F.cosine_similarity(flatten(value_grads)[None, :], flatten(margin_grads)[None, :])
    return float(cosine.item())


def compute_mc_returns(transitions, gamma):
    """Reverse pass: every state in the episode gets a Monte-Carlo task and
    collision return, not just a one-step bootstrap target. A trajectory
    that ends in collision gives every preceding state a nonzero (geometrically
    decaying) mc_collision_return, instead of only the single terminal
    transition where risk_cost=1 -- this is what lets Q_C learn from whole
    colliding trajectories rather than one transition per collision."""
    g_r = 0.0
    g_c = 0.0
    returns = [None] * len(transitions)
    for index in range(len(transitions) - 1, -1, -1):
        reward = transitions[index][4]
        risk_cost = transitions[index][5]
        g_r = reward + gamma * g_r
        g_c = risk_cost + gamma * g_c
        returns[index] = (g_r, g_c)
    return returns


def run_episode(network, engine, environment, args, device, epsilon, expert_prob, rng, need_teacher=True):
    """Collect one episode.

    The "expert" is the frozen strong Mamba-VL policy's own one-step
    value-lookahead ranking over the 80 discrete actions
    (``engine.teacher_scores``), not ORCA -- the label the network imitates
    is the same navigation policy Q_R is ultimately supposed to approximate,
    so there is no reason for the imitation target and the RL objective to
    disagree about what a good action looks like. ``expert_prob`` is the
    probability that the teacher's own top action is actually executed at
    each step (1.0 during Stage 1 demo collection, a decaying schedule
    during DAgger rounds, 0.0 during Stage 2). The teacher is *always*
    scored regardless of who acts -- that label is what DAgger aggregates --
    only whether it is actually *executed* varies. ``is_expert_action`` is
    the literal ``use_expert`` draw, not a coincidental action-index match
    (a non-expert action can legitimately equal the teacher's top choice by
    chance, e.g. under epsilon-greedy exploration; that should not count as
    "the expert acted here").

    ``need_teacher=False`` (Stage 2 only, where ``has_expert_label`` is
    always False downstream so the label can never contribute to any loss)
    skips the real ``teacher_scores`` forward pass -- a full 80-candidate
    Mamba value-lookahead every single step -- and stores an unused
    placeholder instead. Stage 2 is the largest phase of training (2400
    episodes by default), so paying full teacher-scoring cost there for a
    label that is mathematically guaranteed zero gradient weight would be
    pure waste, not caution.
    """
    robot, humans = environment.reset(
        seed=TRAIN_SEED_BASE + rng.integers(0, 10_000_000),
        profile=args.train_profile if rng.random() < 0.5 else "nominal",
        test_case=int(rng.integers(0, 9000)),
    )
    engine.reset()
    features = engine.encode(robot, humans)
    done = False
    steps = 0
    reward_sum = 0.0
    outcome = "timeout"
    transitions = []
    network.eval()
    placeholder_expert_set = np.zeros(args.expert_set_size, dtype=np.int64)
    placeholder_teacher_score = np.zeros(len(engine.actions), dtype=np.float32)
    # A nonzero expert_prob means the teacher's action can actually be
    # executed (not just recorded), so its real index is required
    # regardless of what the caller passed for need_teacher -- only a
    # caller that both never executes the teacher (expert_prob=0.0) and
    # never trains on its label (has_expert_label=False) may skip it.
    need_teacher = need_teacher or expert_prob > 0.0
    while not done and steps < 200:
        if need_teacher:
            teacher_score = engine.teacher_scores(environment.robot, list(environment.env.humans))
            expert_index = int(np.argmax(teacher_score))
            expert_index_set = np.argsort(-teacher_score)[:args.expert_set_size].astype(np.int64)
        else:
            teacher_score = placeholder_teacher_score
            expert_index = 0
            expert_index_set = placeholder_expert_set
        use_expert = rng.random() < expert_prob
        if use_expert:
            action_index = expert_index
        elif rng.random() < epsilon:
            action_index = int(rng.integers(0, len(engine.actions)))
        else:
            action_index = select_action(network, features, device, 0.0, rng)
        is_expert_action = use_expert
        vx, vy = engine.actions[action_index]
        result = environment.step(ActionXY(float(vx), float(vy)))
        # Q_C's one-step cost target is the simulator's own terminal outcome
        # only -- never GDBN's own risk prediction (circular) and never a
        # per-step near-miss term (unbounded accumulation under gamma=0.99).
        # See compute_mc_returns for how the *episode-level* MC target fixes
        # this transition's isolation from the rest of a colliding trajectory.
        risk_cost = float(result.done and result.outcome == "collision")
        next_features = engine.encode(environment.robot, list(environment.env.humans))
        transitions.append(
            (
                features, action_index, expert_index, expert_index_set, result.reward,
                risk_cost, next_features, result.done, is_expert_action, teacher_score,
            )
        )
        features = next_features
        reward_sum += result.reward
        steps += 1
        done = result.done
        if done:
            outcome = result.outcome
    return transitions, reward_sum, steps, outcome


def evaluate_quick(network, engine_factory, environment_factory, args, device, profile, episodes, seed_offset, scenarios, head="score"):
    """In-training-only validation rollout. ``scenarios`` is always passed
    explicitly by the caller (VALIDATION_SCENARIOS for every call site in
    this file) -- never defaulted -- so it is never accidentally left
    pointing at the six-scenario formal-test battery.

    ``head="score"`` (default) is the actual deployed policy: argmax of
    Q_score = Q_R - beta*Q_C. ``head="task_only"`` ignores Q_C entirely
    (argmax of Q_R alone) -- used only as a counterfactual to measure the
    Bayesian branch's "risk reduction" (how much lower is CR with Q_C in
    the decision than without it, on the identical episodes), never as the
    policy actually selected for deployment or checkpointing.
    """
    network.eval()
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    steps_all = []
    for episode in range(int(episodes)):
        scenario = scenarios[episode % len(scenarios)]
        environment = environment_factory(scenario)
        engine = engine_factory()
        robot, humans = environment.reset(
            seed=args.seed + seed_offset + episode * 101,
            profile=profile,
            test_case=(seed_offset + episode) % 9000,
        )
        engine.reset()
        features = engine.encode(robot, humans)
        done = False
        episode_steps = 0
        outcome = "timeout"
        while not done and episode_steps < 200:
            with torch.no_grad():
                if head == "task_only":
                    _, components = network(*tensor_features(features, device), return_components=True)
                    q_score = components["q_r"]
                else:
                    q_score = network(*tensor_features(features, device))
            action_index = int(q_score.argmax(dim=-1).item())
            vx, vy = engine.actions[action_index]
            result = environment.step(ActionXY(float(vx), float(vy)))
            done = result.done
            episode_steps += 1
            if done:
                outcome = result.outcome
            else:
                features = engine.encode(environment.robot, list(environment.env.humans))
        outcomes[outcome] += 1
        steps_all.append(episode_steps)
    total = max(sum(outcomes.values()), 1)
    return {
        "profile": profile,
        "episodes": total,
        "SR": outcomes["success"] / total,
        "CR": outcomes["collision"] / total,
        "TR": outcomes["timeout"] / total,
        "mean_steps": float(np.mean(steps_all)),
    }


def evaluate_quick_baseline(baseline_mamba, environment_factory, args, profile, episodes, seed_offset, scenarios):
    """The frozen Mamba-VL baseline (no belief, no Q_C) on the identical
    episode/seed formula ``evaluate_quick`` uses, so the Stage 1 gate can
    compare candidate vs baseline *paired* -- same scenario draw, same
    profile, same seed, same test_case for every episode index. Uses a
    dedicated ``baseline_mamba`` instance (never the ``mamba`` object
    ``teacher_scores`` and the TaskEncoder context share) because
    ``run_mamba_baseline_episode`` mutates persistent policy state
    (``set_phase``, ``use_sarl_predict``) that must never leak into the
    frozen teacher used for labels elsewhere in this script.
    """
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    steps_all = []
    for episode in range(int(episodes)):
        scenario = scenarios[episode % len(scenarios)]
        environment = environment_factory(scenario)
        outcome, steps, _ = run_mamba_baseline_episode(
            baseline_mamba, environment,
            args.seed + seed_offset + episode * 101, profile,
            (seed_offset + episode) % 9000, 200,
        )
        outcomes[outcome] += 1
        steps_all.append(steps)
    total = max(sum(outcomes.values()), 1)
    return {
        "profile": profile,
        "episodes": total,
        "SR": outcomes["success"] / total,
        "CR": outcomes["collision"] / total,
        "TR": outcomes["timeout"] / total,
        "mean_steps": float(np.mean(steps_all)),
    }


def save_checkpoint(path, network, target, optimizer, args, metrics, artifact_hashes):
    torch.save(
        {
            "network": network.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "artifact_hashes": artifact_hashes,
            "architecture": {
                "decision_core": "belief-MDP twin head: Q_R(b,a) - beta*Q_C(b,a), fully separate TaskEncoder/RiskEncoder",
                "belief_mode": args.belief_mode,
                "beta": args.beta,
                "mamba_role": "frozen TaskEncoder context (feeds Q_R only) + Stage 1/DAgger teacher label source; never fine-tuned",
            },
        },
        path,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/belief_mdp_pilot")
    parser.add_argument("--policy_config", default="configs/policy_bayesian_fullcrowd_tail.config")
    parser.add_argument("--base_env_config", default="configs/env_belief_mdp.config")
    parser.add_argument("--env_config", default="configs/env_belief_mdp.config")
    parser.add_argument("--base_checkpoint", default="runs/mamba_vl/rl_model_ep10000_T24.pth")
    parser.add_argument(
        "--gdbn_params", default="runs/bayesian_distributional/gdbn_params_cv_residual_k3"
    )
    parser.add_argument(
        "--k1_gdbn_params", default="runs/mamba_vl/gdbn_params_k1",
        help="K=1-fitted GDBN params (tools/fit_k1_gdbn.py), used only by --belief_mode k1_belief.",
    )
    parser.add_argument(
        "--belief_mode",
        default="action_conditioned",
        choices=(
            "action_conditioned", "cv", "state_only", "corrupted", "no_belief", "k1_belief",
        ),
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gdbn_K", type=int, default=3)
    parser.add_argument("--num_humans", type=int, default=20)
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--risk_horizon", type=int, default=5)
    parser.add_argument("--safe_distance", type=float, default=0.20)
    parser.add_argument("--cvar_alpha", type=float, default=0.80)
    parser.add_argument("--pedestrian_aggregation", default="max")
    parser.add_argument("--train_profile", default="train_nonstationary")
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--expert_set_size", type=int, default=3)
    parser.add_argument("--demo_episodes", type=int, default=600)
    parser.add_argument("--demo_updates", type=int, default=10000)
    parser.add_argument("--rl_episodes", type=int, default=2400)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument(
        "--stage1_nav_episodes", type=int, default=120,
        help="Episodes for the Stage 1 gate's nav check, on VALIDATION_SCENARIOS "
        "only (never the six-scenario formal battery).",
    )
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--demo_buffer_capacity", type=int, default=80000)
    parser.add_argument(
        "--demo_validation_every", type=int, default=10,
        help="Every Nth demo episode is held out into a validation-only buffer, never trained on.",
    )
    parser.add_argument(
        "--stage1_log_every", type=int, default=1000,
        help="Log a stage1b_progress record to metrics.jsonl every N offline updates.",
    )
    parser.add_argument("--warmup_transitions", type=int, default=3000)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--lambda_e_demo", type=float, default=5.0)
    parser.add_argument("--lambda_e_start", type=float, default=0.20)
    parser.add_argument("--lambda_e_end", type=float, default=0.05)
    parser.add_argument("--epsilon_start", type=float, default=0.15)
    parser.add_argument("--epsilon_end", type=float, default=0.03)
    # Stage 1 hard gate (Round 11: paired non-inferiority vs the frozen
    # Mamba-VL baseline on identical validation episodes, not an absolute
    # SR/CR bar and not teacher-imitation agreement -- see the module
    # docstring's Round 11 note for why the old top1_in_expert_set/absolute-
    # SR gate was measuring the wrong thing). Fails these and the run stops
    # before Stage 2, saving stage1_failed_model.pth instead of wasting a
    # full RL budget on a policy that regressed relative to the baseline.
    parser.add_argument(
        "--gate_sr_slack", type=float, default=0.03,
        help="Gate passes if candidate SR >= baseline SR - gate_sr_slack, per profile.",
    )
    parser.add_argument(
        "--gate_cr_slack", type=float, default=0.02,
        help="Gate passes if candidate CR <= baseline CR + gate_cr_slack, per profile.",
    )
    parser.add_argument(
        "--gate_tr_slack", type=float, default=0.02,
        help="Gate passes if candidate TR <= baseline TR + gate_tr_slack, per profile.",
    )
    # DAgger (Stage 1c): only runs if the gate right after Stage 1b fails.
    # Sequential distribution shift -- training states are all teacher-visited,
    # so one wrong action at deployment reaches states the demonstrations
    # never covered -- is a distinct failure mode from "hasn't trained
    # enough," and more offline updates on the same fixed demo set cannot
    # fix it; DAgger aggregates the teacher's label on states the *policy* visits.
    parser.add_argument(
        "--dagger_teacher_prob_schedule", default="0.8,0.6,0.4,0.2,0.0",
        help="Comma-separated probability that the teacher's action is actually executed, one value per round.",
    )
    parser.add_argument("--dagger_episodes_per_round", type=int, default=150)
    parser.add_argument(
        "--dagger_updates_per_round", type=int, default=6000,
        help="Per-round offline-update BUDGET (a cap, not a fixed count): "
        "convergence-based early stopping (see --dagger_convergence_patience) "
        "usually stops well before this many updates run.",
    )
    parser.add_argument(
        "--dagger_convergence_patience", type=int, default=2,
        help="Stop a DAgger round's offline updates after this many consecutive "
        "non-improving checks (every --stage1_log_every updates) on the nav "
        "SR/CR composite, reverting to the round's best-seen checkpoint.",
    )
    parser.add_argument(
        "--convergence_nav_episodes", type=int, default=30,
        help="Episodes (VALIDATION_SCENARIOS only, never the six-scenario formal "
        "battery) for the periodic in-round nav check that drives convergence "
        "stopping -- cheaper than the full --stage1_nav_episodes end-of-round gate check.",
    )
    parser.add_argument("--dagger_buffer_capacity", type=int, default=40000)
    parser.add_argument("--collision_buffer_capacity", type=int, default=20000)
    parser.add_argument(
        "--dagger_validation_every", type=int, default=10,
        help="Every Nth DAgger-round episode (across all rounds) is held out into "
        "dagger_validation_recent instead of recent_dagger_buffer.",
    )
    parser.add_argument(
        "--log_grad_conflict", action="store_true", default=True,
        help="Log cos(grad(value_loss), grad(margin_loss)) at every stage1_log_every checkpoint.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.belief_mode == "no_belief" and args.beta != 0.0:
        print(f"[BELIEF-MDP] no_belief forces beta=0 (was {args.beta})", flush=True)
        args.beta = 0.0
    if args.smoke:
        args.demo_episodes = 3
        args.demo_updates = 20
        args.rl_episodes = 3
        args.eval_episodes = 2
        args.stage1_nav_episodes = 2
        args.eval_interval = 3
        args.batch_size = 8
        args.replay_capacity = 400
        args.demo_buffer_capacity = 200
        args.demo_validation_every = 2
        args.stage1_log_every = 5
        args.warmup_transitions = 16
        args.particles = 12
        # Smoke is correctness-only (a handful of episodes/updates); the
        # Stage 1 hard gate is a real training-quality bar that a few dozen
        # updates cannot meaningfully pass or fail. Disable it here so smoke
        # always reaches Stage 2 and exercises the whole code path -- the
        # real, non-smoke run enforces it strictly.
        args.gate_sr_slack = 1.0
        args.gate_cr_slack = 1.0
        args.gate_tr_slack = 1.0
        # Unlike the thresholds above, the DAgger schedule itself stays
        # real (just tiny) during smoke, so that code path is exercised
        # rather than skipped -- only the final pass/fail decision is
        # forced through if DAgger still hasn't cleared the (already
        # trivial) bar by the time its tiny rounds are done.
        args.dagger_teacher_prob_schedule = "0.8,0.0"
        args.dagger_episodes_per_round = 2
        args.dagger_updates_per_round = 5
        args.dagger_convergence_patience = 1
        args.convergence_nav_episodes = 2
        args.dagger_buffer_capacity = 200
        args.collision_buffer_capacity = 200
        args.dagger_validation_every = 2

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    config = merged_policy_config(args.policy_config, args.base_env_config)
    mamba = build_frozen_mamba(config, args.base_checkpoint, device)

    def make_engine():
        return BeliefMDPFeatureEngine(
            mamba,
            args.gdbn_params,
            device,
            belief_mode=args.belief_mode,
            K=args.gdbn_K,
            n_particles=args.particles,
            num_humans=args.num_humans,
            risk_horizon=args.risk_horizon,
            safe_distance=args.safe_distance,
            cvar_alpha=args.cvar_alpha,
            pedestrian_aggregation=args.pedestrian_aggregation,
            seed=args.seed,
            k1_gdbn_params=args.k1_gdbn_params,
        )

    def make_environment(scenario):
        return FullCrowdNavigationEnvironment(args.env_config, scenario, robot_visible=False)

    # Round 11: Stage 1's hard gate is now a paired non-inferiority
    # comparison against this frozen baseline, computed once (a separate
    # instance from `mamba` above -- run_mamba_baseline_episode mutates
    # persistent policy state that must never leak into the teacher-scoring
    # `mamba` object) on the exact seed/scenario/profile formula
    # run_stage1_gate_check reuses every time, so "paired" is literal: same
    # episode index -> same scenario, seed, and test_case for baseline and
    # candidate alike.
    print("[BELIEF-MDP] evaluating frozen Mamba-VL baseline for the Stage 1 gate...", flush=True)
    baseline_mamba = build_frozen_mamba(config, args.base_checkpoint, device)
    gate_baseline_nominal = evaluate_quick_baseline(
        baseline_mamba, make_environment, args, "nominal", args.stage1_nav_episodes, 810000, VALIDATION_SCENARIOS,
    )
    gate_baseline_stress = evaluate_quick_baseline(
        baseline_mamba, make_environment, args, "train_nonstationary", args.stage1_nav_episodes, 910000, VALIDATION_SCENARIOS,
    )
    print(
        f"[BELIEF-MDP] gate baseline: nominal={gate_baseline_nominal['SR']:.1%}/{gate_baseline_nominal['CR']:.1%} "
        f"stress={gate_baseline_stress['SR']:.1%}/{gate_baseline_stress['CR']:.1%}",
        flush=True,
    )

    probe = make_engine()
    network = BeliefMDPQNetwork(
        context_dim=256,
        belief_dim=probe.belief_dim,
        candidate_dim=probe.candidate_dim,
        beta=args.beta,
    ).to(device)
    target = deepcopy(network).to(device)
    target.eval()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-5)

    buffer_kwargs = dict(
        context_dim=256,
        belief_dim=probe.belief_dim,
        action_count=len(probe.actions),
        candidate_dim=probe.candidate_dim,
        expert_set_size=args.expert_set_size,
    )
    dagger_schedule = [
        float(x) for x in args.dagger_teacher_prob_schedule.split(",") if x.strip() != ""
    ]
    # Five-way stratified replay (see module docstring, Round 8): each
    # stratum is sampled at a fixed ratio (sample_stratified redistributes
    # an empty stratum's share to the others) instead of one "demo" bucket
    # that mixes Stage 1a's fixed teacher demonstrations together with every
    # later DAgger round's data, which let old rounds get diluted/crowded
    # out by new ones with no control over the mix.
    teacher_demo_buffer = ReplayBuffer(args.demo_buffer_capacity, permanent=True, **buffer_kwargs)
    recent_dagger_buffer = ReplayBuffer(args.dagger_buffer_capacity, permanent=True, **buffer_kwargs)
    older_dagger_buffer = ReplayBuffer(
        args.dagger_buffer_capacity * max(1, len(dagger_schedule)), permanent=True, **buffer_kwargs
    )
    # Not permanent: collision trajectories from the policy's *current*
    # failure modes are more useful than a fixed early sample of them, so
    # this is a rolling window, not a permanent set.
    collision_buffer = ReplayBuffer(args.collision_buffer_capacity, permanent=False, **buffer_kwargs)
    online_rl_buffer = ReplayBuffer(args.replay_capacity, permanent=False, **buffer_kwargs)
    stratified_buffers = {
        "teacher_demo": teacher_demo_buffer,
        "recent_dagger": recent_dagger_buffer,
        "older_dagger": older_dagger_buffer,
        "collision": collision_buffer,
        "online_rl": online_rl_buffer,
    }
    STRATIFIED_BUFFER_RATIOS = {
        "teacher_demo": 0.25,
        "recent_dagger": 0.35,
        "older_dagger": 0.15,
        "collision": 0.15,
        "online_rl": 0.10,
    }

    def total_buffer_size() -> int:
        return sum(buf.size for buf in stratified_buffers.values())

    # Every demo_validation_every-th Stage 1a episode is routed here instead
    # of teacher_demo_buffer -- never trained on, used only to check Stage 1
    # actually converged on data it did not see, not on the last training
    # minibatch. dagger_validation_recent is the same idea for DAgger-round
    # data specifically: teacher_demo_buffer's held-out set only covers
    # states the *teacher's own demonstrations* visited, which says nothing
    # about how well the policy imitates the teacher on states the *policy
    # itself* (DAgger's on-policy collection) visits -- a distinct question
    # DAgger exists to answer. Split into recent (current round only,
    # cleared and moved into historical at the start of each new round --
    # mirrors recent_dagger_buffer/older_dagger_buffer) and historical
    # (accumulates every round, logged for reporting only): the convergence
    # check's stopping decision must read only the *current* round's states,
    # or it is diluted by early, weaker-policy states from earlier rounds
    # and stops meaning "did this round's updates help."
    demo_validation_buffer = ReplayBuffer(args.demo_buffer_capacity, permanent=True, **buffer_kwargs)
    dagger_validation_recent = ReplayBuffer(args.dagger_buffer_capacity, permanent=True, **buffer_kwargs)
    dagger_validation_historical = ReplayBuffer(
        args.dagger_buffer_capacity * max(1, len(dagger_schedule)), permanent=True, **buffer_kwargs
    )

    print("[BELIEF-MDP] hashing training artifacts (checkpoint, GDBN params, configs)...", flush=True)
    artifact_hashes = compute_artifact_hashes(args)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "device": str(device),
        "args": vars(args),
        "artifact_hashes": artifact_hashes,
        "architecture": (
            "Belief-MDP twin head: Q_score(b,a) = Q_R(b,a) - beta*Q_C(b,a); "
            "Q_R/Q_C each have their own Bellman + MC-return target and fully "
            "separate encoders (TaskEncoder: Mamba context + kinematics -> "
            "Q_R only; RiskEncoder: GDBN belief + risk features -> Q_C only; "
            "no shared layer or joint tensor); belief is part of the "
            "decision state from Stage 1 step 1; the frozen Mamba-VL "
            "checkpoint's own value-lookahead is the Stage 1/DAgger expert "
            "label source (teacher_scores), never fine-tuned; expert loss is "
            "a fixed-margin (DQfD-style) loss against a top-K expert set, "
            "not cross-entropy, so it cannot inflate Q without bound; replay "
            "is a 5-way stratified buffer (teacher_demo/recent_dagger/"
            "older_dagger/collision/online_rl) sampled at fixed ratios"
        ),
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.jsonl"
    updates = 0
    latest_losses = {}
    start_time = time.time()
    aborted_reason = None
    last_eval_sr = None
    # EMA (not a single update) of q_r_abs_p95: RL is not monotonic, and a
    # single noisy spike should not stop a run that would otherwise recover
    # -- only *sustained* divergence (roughly the last ~50 updates) should.
    q_r_p95_ema = None
    Q_R_EMA_DECAY = 0.98

    print(
        f"[BELIEF-MDP] mode={args.belief_mode} beta={args.beta} device={device} "
        f"num_humans={args.num_humans} belief_dim={probe.belief_dim} "
        f"candidate_dim={probe.candidate_dim}",
        flush=True,
    )

    total_stage1_episode_counter = 0  # counts Stage 1a + every DAgger round's episodes, for logging only

    def collect_and_store(expert_prob, target_buffer, has_expert_label, scenario, epsilon=0.0):
        """Collect one episode into ``target_buffer``. Every transition of an
        episode that ends in collision is *also* copied into
        ``collision_buffer`` (prioritized replay for the policy's current
        failure modes), regardless of which stratum ``target_buffer`` is --
        this is intentional double-accounting, not a bug: the same
        transitions are ordinarily sampled at their stratum's normal rate
        and additionally over-represented via the collision stratum.
        """
        nonlocal total_stage1_episode_counter
        environment = make_environment(scenario)
        engine = make_engine()
        # Skip the real teacher forward pass only when this transition will
        # never execute the teacher's action (expert_prob=0.0, guarded again
        # inside run_episode) *and* will never train on its label -- true
        # only for pure Stage 2 online RL calls (has_expert_label=False).
        transitions, reward_sum, steps, outcome = run_episode(
            network, engine, environment, args, device, epsilon, expert_prob, rng,
            need_teacher=has_expert_label,
        )
        mc_returns = compute_mc_returns(transitions, args.gamma)
        expert_action_fraction = float(np.mean([t[8] for t in transitions])) if transitions else 0.0
        is_collision_episode = outcome == "collision"
        for (
            features, action_index, expert_index, expert_index_set, reward, risk_cost,
            next_features, done, is_expert_action, teacher_score,
        ), (mc_r, mc_c) in zip(transitions, mc_returns):
            target_buffer.add(
                features, action_index, expert_index, expert_index_set, reward, risk_cost,
                mc_r, mc_c, next_features, done,
                has_expert_label=has_expert_label, is_expert_action=is_expert_action,
                teacher_scores=teacher_score,
            )
            if is_collision_episode and target_buffer is not collision_buffer:
                collision_buffer.add(
                    features, action_index, expert_index, expert_index_set, reward, risk_cost,
                    mc_r, mc_c, next_features, done,
                    has_expert_label=has_expert_label, is_expert_action=is_expert_action,
                    teacher_scores=teacher_score,
                )
        total_stage1_episode_counter += 1
        return reward_sum, steps, outcome, expert_action_fraction

    def run_offline_updates(num_updates, lambda_e_value, phase_label):
        nonlocal updates, aborted_reason, latest_losses
        ema = None
        for update_index in range(num_updates):
            latest_losses = optimize(
                network, target, optimizer, stratified_buffers, STRATIFIED_BUFFER_RATIOS,
                min(args.batch_size, total_buffer_size()), args.gamma, lambda_e_value, rng, device,
            )
            soft_update(target, network, args.tau)
            updates += 1
            if not math_finite(latest_losses.get("loss")):
                aborted_reason = f"non-finite loss during {phase_label}, update {update_index + 1}"
                return
            current_p95 = latest_losses.get("q_r_abs_p95", 0.0)
            ema = current_p95 if ema is None else Q_R_EMA_DECAY * ema + (1.0 - Q_R_EMA_DECAY) * current_p95
            if ema > Q_R_ABS_P95_LIMIT:
                aborted_reason = (
                    f"q_r_abs_p95 EMA={ema:.3f} exceeded {Q_R_ABS_P95_LIMIT} during {phase_label}, "
                    f"update {update_index + 1} -- sustained, not a single spike"
                )
                return
            if (update_index + 1) % args.stage1_log_every == 0:
                validation = evaluate_demo_validation(network, demo_validation_buffer, device)
                grad_conflict = (
                    compute_gradient_conflict(
                        network, target, stratified_buffers, STRATIFIED_BUFFER_RATIOS,
                        args.batch_size, args.gamma, rng, device,
                    )
                    if args.log_grad_conflict else None
                )
                progress = {
                    "type": "stage1b_progress",
                    "phase": phase_label,
                    "update": update_index + 1,
                    "q_r_abs_p95": current_p95,
                    "q_r_abs_p95_ema": ema,
                    "td_loss_r": latest_losses["td_loss_r"],
                    "td_loss_c": latest_losses["td_loss_c"],
                    "mc_loss_r": latest_losses["mc_loss_r"],
                    "mc_loss_c": latest_losses["mc_loss_c"],
                    "margin_loss": latest_losses["margin_loss"],
                    "velocity_error": latest_losses["velocity_error"],
                    "angle_error_deg": latest_losses["angle_error_deg"],
                    "grad_cosine_value_vs_margin": grad_conflict,
                    "held_out": validation,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress) + "\n")
                held_out_str = (
                    f"qr_top1_in_set={validation['qr_top1_in_expert_set']:.1%} "
                    f"qr_regret={validation['qr_regret_mean']:.3f}"
                    if validation is not None else "no held-out data"
                )
                grad_str = f"{grad_conflict:+.3f}" if grad_conflict is not None else "n/a"
                print(
                    f"[{phase_label.upper()}-PROGRESS] update={update_index + 1}/{num_updates} "
                    f"q_r_p95_ema={ema:.3f} vel_err={latest_losses['velocity_error']:.3f} "
                    f"angle_err={latest_losses['angle_error_deg']:.1f}deg "
                    f"grad_cos(value,margin)={grad_str} held_out=({held_out_str})",
                    flush=True,
                )

    def run_stage1_gate_check(nav_episodes, label):
        """Round 11: the hard pass/fail gate is a paired non-inferiority nav
        comparison against the frozen Mamba-VL baseline (gate_baseline_*,
        computed once at start-up on identical seeds/scenarios/profiles) --
        not teacher-imitation agreement. Q_R/Q_score teacher agreement and
        regret are still computed and logged in full below, purely for
        reporting/paper analysis; they no longer appear in ``gate_passed``.
        Numerical-stability aborts (non-finite loss, Q_R divergence) are a
        separate mechanism (see run_offline_updates/run_dagger_round_updates)
        that stops the whole run regardless of this gate.
        """
        stage1_validation = evaluate_demo_validation(network, demo_validation_buffer, device)
        if stage1_validation is None:
            print(
                f"[{label}] no held-out demo episodes available "
                "(demo_validation_every too large for demo_episodes) -- skipped",
                flush=True,
            )
        else:
            print(
                f"[{label}] teacher-agreement report (held out {stage1_validation['episodes_held_out']} "
                f"transitions, sampled {stage1_validation['sample_size']}, not gating): "
                f"Q_R top1_in_set={stage1_validation['qr_top1_in_expert_set']:.1%} "
                f"regret={stage1_validation['qr_regret_mean']:.3f} | "
                f"Q_score top1_in_set={stage1_validation['score_top1_in_expert_set']:.1%} "
                f"regret={stage1_validation['score_regret_mean']:.3f}",
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"type": "stage1_validation", "label": label, **stage1_validation}) + "\n"
                )
        print(
            f"[BELIEF-MDP] running Stage 1 navigation check ({nav_episodes} episodes, "
            f"{VALIDATION_SCENARIOS} only -- never the six-scenario formal battery)...",
            flush=True,
        )
        nominal = evaluate_quick(
            network, make_engine, make_environment, args, device,
            "nominal", nav_episodes, 810000, VALIDATION_SCENARIOS,
        )
        # "stress" here means the *training-time* nonstationary profile
        # (train_nonstationary), a validation proxy for a harder regime --
        # never heldout_nonstationary, which is reserved for the frozen
        # model's one-time formal evaluation (see the VALIDATION_PROFILES
        # comment above).
        stress = evaluate_quick(
            network, make_engine, make_environment, args, device,
            "train_nonstationary", nav_episodes, 910000, VALIDATION_SCENARIOS,
        )
        # Risk-reduction diagnostic: the same episodes, but the counterfactual
        # policy that ignores Q_C entirely (argmax of Q_R alone). A positive
        # CR delta (task_only CR minus actual CR) is direct evidence the
        # Bayesian branch is preventing collisions the task head alone would
        # not have -- never used to select or gate anything, purely to
        # answer "is Q_C doing something" with a measurement.
        nominal_task_only = evaluate_quick(
            network, make_engine, make_environment, args, device,
            "nominal", nav_episodes, 810000, VALIDATION_SCENARIOS, head="task_only",
        )
        stress_task_only = evaluate_quick(
            network, make_engine, make_environment, args, device,
            "train_nonstationary", nav_episodes, 910000, VALIDATION_SCENARIOS, head="task_only",
        )
        risk_reduction = {
            "nominal_cr_delta": nominal_task_only["CR"] - nominal["CR"],
            "stress_cr_delta": stress_task_only["CR"] - stress["CR"],
            "nominal_task_only": nominal_task_only,
            "stress_task_only": stress_task_only,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "type": "evaluation", "label": label, "nominal": nominal, "stress": stress,
                "risk_reduction": risk_reduction,
            }) + "\n")

        def non_inferior(candidate, baseline):
            return (
                candidate["SR"] >= baseline["SR"] - args.gate_sr_slack
                and candidate["CR"] <= baseline["CR"] + args.gate_cr_slack
                and candidate["TR"] <= baseline["TR"] + args.gate_tr_slack
            )

        nav_passed = (
            non_inferior(nominal, gate_baseline_nominal)
            and non_inferior(stress, gate_baseline_stress)
        )
        gate_passed = nav_passed
        print(
            f"[STAGE1-GATE:{label}] {'PASS' if gate_passed else 'FAIL'} (paired vs frozen baseline, "
            f"slack SR-{args.gate_sr_slack:.0%}/CR+{args.gate_cr_slack:.0%}/TR+{args.gate_tr_slack:.0%}) "
            f"nominal={nominal['SR']:.1%}/{nominal['CR']:.1%} (baseline {gate_baseline_nominal['SR']:.1%}/"
            f"{gate_baseline_nominal['CR']:.1%}) stress={stress['SR']:.1%}/{stress['CR']:.1%} "
            f"(baseline {gate_baseline_stress['SR']:.1%}/{gate_baseline_stress['CR']:.1%}) "
            f"risk_reduction(task_only_CR-actual_CR)={risk_reduction['nominal_cr_delta']:+.1%}/"
            f"{risk_reduction['stress_cr_delta']:+.1%}",
            flush=True,
        )
        return gate_passed, nominal, stress, stage1_validation

    def run_dagger_round_updates(max_updates, lambda_e_value, phase_label, patience):
        """DAgger-round offline updates with convergence-based early stopping.

        Runs up to ``max_updates`` (a budget cap, not a fixed count),
        checking every ``args.stage1_log_every`` updates whether the policy
        actually improved. Six diagnostics are logged at each check (teacher-
        state and DAgger-state top-3 hit rate, Q_C collision AUC/Brier, the
        six-scenario nav SR/CR/TR, Q_R scale, and beta-top1-change fraction)
        -- but rather than an arbitrary weighted blend across their very
        different scales, the actual improve/no-improve decision uses one
        composite already central to this project's own Stage 1 hard gate:
        mean SR minus mean CR across nominal+stress. The other five are
        logged for visibility/paper reporting, not folded into the trigger.
        Stops after ``patience`` consecutive non-improving checks (or once
        ``max_updates`` is exhausted) and reverts network/target/optimizer
        to the best-scoring snapshot seen this round -- a round can end with
        a real regression relative to its own best point otherwise, since
        RL is not monotonic (see the module docstring's SR-drop discussion).
        """
        nonlocal updates, aborted_reason, latest_losses
        ema = None
        best_composite = None
        best_state = None
        non_improve_streak = 0
        for update_index in range(max_updates):
            latest_losses = optimize(
                network, target, optimizer, stratified_buffers, STRATIFIED_BUFFER_RATIOS,
                min(args.batch_size, total_buffer_size()), args.gamma, lambda_e_value, rng, device,
            )
            soft_update(target, network, args.tau)
            updates += 1
            if not math_finite(latest_losses.get("loss")):
                aborted_reason = f"non-finite loss during {phase_label}, update {update_index + 1}"
                return
            current_p95 = latest_losses.get("q_r_abs_p95", 0.0)
            ema = current_p95 if ema is None else Q_R_EMA_DECAY * ema + (1.0 - Q_R_EMA_DECAY) * current_p95
            if ema > Q_R_ABS_P95_LIMIT:
                aborted_reason = (
                    f"q_r_abs_p95 EMA={ema:.3f} exceeded {Q_R_ABS_P95_LIMIT} during {phase_label}, "
                    f"update {update_index + 1} -- sustained, not a single spike"
                )
                return
            if (update_index + 1) % args.stage1_log_every == 0:
                teacher_validation = evaluate_demo_validation(network, demo_validation_buffer, device)
                # Current-round-only: this is what the improve/no-improve
                # decision below would use if it read DAgger-state imitation
                # at all (it currently drives on the nav composite instead,
                # see the docstring) -- historical is logged for reporting,
                # never for any decision, since it mixes in earlier, weaker
                # rounds' states.
                dagger_validation_current = evaluate_demo_validation(network, dagger_validation_recent, device)
                dagger_validation_historical_result = evaluate_demo_validation(
                    network, dagger_validation_historical, device
                )
                qc_calibration = evaluate_qc_calibration(network, recent_dagger_buffer, device)
                grad_conflict = (
                    compute_gradient_conflict(
                        network, target, stratified_buffers, STRATIFIED_BUFFER_RATIOS,
                        args.batch_size, args.gamma, rng, device,
                    )
                    if args.log_grad_conflict else None
                )
                nominal = evaluate_quick(
                    network, make_engine, make_environment, args, device,
                    "nominal", args.convergence_nav_episodes, 810000, VALIDATION_SCENARIOS,
                )
                stress = evaluate_quick(
                    network, make_engine, make_environment, args, device,
                    "train_nonstationary", args.convergence_nav_episodes, 910000, VALIDATION_SCENARIOS,
                )
                composite = 0.5 * (nominal["SR"] + stress["SR"]) - 0.5 * (nominal["CR"] + stress["CR"])
                improved = best_composite is None or composite > best_composite + 1e-6
                if improved:
                    best_composite = composite
                    best_state = {
                        "network": deepcopy(network.state_dict()),
                        "target": deepcopy(target.state_dict()),
                        "optimizer": deepcopy(optimizer.state_dict()),
                    }
                    non_improve_streak = 0
                else:
                    non_improve_streak += 1
                progress = {
                    "type": "dagger_convergence_check",
                    "phase": phase_label,
                    "update": update_index + 1,
                    "q_r_abs_p95": current_p95,
                    "q_r_abs_p95_ema": ema,
                    "beta_top1_change_rate": latest_losses["beta_top1_change_rate"],
                    "grad_cosine_value_vs_margin": grad_conflict,
                    "teacher_state_validation": teacher_validation,
                    "dagger_state_validation_current_round": dagger_validation_current,
                    "dagger_state_validation_historical": dagger_validation_historical_result,
                    "qc_calibration": qc_calibration,
                    "nominal": nominal,
                    "stress": stress,
                    "composite": composite,
                    "best_composite": best_composite,
                    "improved": improved,
                    "non_improve_streak": non_improve_streak,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress) + "\n")
                print(
                    f"[{phase_label.upper()}-CONVERGENCE] update={update_index + 1}/{max_updates} "
                    f"composite={composite:+.3f} best={best_composite:+.3f} "
                    f"{'IMPROVED' if improved else f'no-improve({non_improve_streak}/{patience})'} "
                    f"teacher_qr_top3={(teacher_validation['qr_top1_in_expert_set'] if teacher_validation else float('nan')):.1%} "
                    f"dagger_qr_top3={(dagger_validation_current['qr_top1_in_expert_set'] if dagger_validation_current else float('nan')):.1%} "
                    f"qc_auc={(qc_calibration['q_c_collision_auc'] if qc_calibration and qc_calibration['q_c_collision_auc'] is not None else float('nan')):.3f}",
                    flush=True,
                )
                if non_improve_streak >= patience:
                    print(
                        f"[{phase_label.upper()}-CONVERGENCE] stopping early after {patience} "
                        f"consecutive non-improving checks, reverting to best (composite={best_composite:+.3f})",
                        flush=True,
                    )
                    break
        if best_state is not None:
            network.load_state_dict(best_state["network"])
            target.load_state_dict(best_state["target"])
            optimizer.load_state_dict(best_state["optimizer"])

    # ---- Stage 1a: Mamba-teacher imitation demo collection ----
    print(f"[BELIEF-MDP] Stage 1a: {args.demo_episodes} Mamba-teacher demo episodes", flush=True)
    for episode in range(args.demo_episodes):
        scenario = TRAIN_SCENARIOS[episode % len(TRAIN_SCENARIOS)]
        is_validation_episode = (episode + 1) % args.demo_validation_every == 0
        target_buffer = demo_validation_buffer if is_validation_episode else teacher_demo_buffer
        reward_sum, steps, outcome, _ = collect_and_store(
            expert_prob=1.0, target_buffer=target_buffer, has_expert_label=True, scenario=scenario,
        )
        record = {
            "type": "train_episode", "episode": episode + 1, "stage": "demo1a",
            "scenario": scenario, "outcome": outcome, "reward": reward_sum, "steps": steps,
            "teacher_demo_buffer_size": teacher_demo_buffer.size, "updates": updates,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if (episode + 1) % 100 == 0 or episode == 0:
            print(f"[TRAIN] demo1a {episode + 1}/{args.demo_episodes} outcome={outcome} reward={reward_sum:.3f}", flush=True)

    # ---- Stage 1b: offline margin+TD+MC updates on Stage 1a data ----
    print(f"[BELIEF-MDP] Stage 1b: {args.demo_updates} offline margin+TD+MC updates", flush=True)
    network.train()
    run_offline_updates(args.demo_updates, args.lambda_e_demo, "stage1b")

    gate_passed = False
    nominal = stress = stage1_validation = None
    if not aborted_reason:
        gate_passed, nominal, stress, stage1_validation = run_stage1_gate_check(
            args.stage1_nav_episodes, "STAGE1B-VALIDATION"
        )
        save_checkpoint(
            output / f"checkpoint_ep{args.demo_episodes}.pth", network, target, optimizer, args,
            {"episode": args.demo_episodes, "nominal": nominal, "stress": stress}, artifact_hashes,
        )

    # ---- Stage 1c: DAgger rounds (only if the gate above did not pass) ----
    dagger_episode_counter = 0  # across all rounds, for dagger_validation_every
    if not aborted_reason and not gate_passed:
        print(
            f"[BELIEF-MDP] Stage 1b gate failed -- entering DAgger (Stage 1c), "
            f"{len(dagger_schedule)} rounds, teacher_prob schedule={dagger_schedule}",
            flush=True,
        )
        for round_index, teacher_prob in enumerate(dagger_schedule):
            round_label = f"dagger_round{round_index + 1}"
            if round_index > 0:
                # This round's data starts a fresh recent_dagger_buffer; the
                # previous round's data moves into older_dagger_buffer so it
                # is still sampled (at a lower fixed ratio) rather than
                # simply discarded. Same pattern for the held-out DAgger
                # validation set, so the next round's convergence checks
                # read only *this* round's states, not a mix diluted by
                # earlier, weaker-policy rounds.
                moved = recent_dagger_buffer.size
                older_dagger_buffer.import_batch(recent_dagger_buffer.export_all())
                recent_dagger_buffer.clear()
                moved_validation = dagger_validation_recent.size
                dagger_validation_historical.import_batch(dagger_validation_recent.export_all())
                dagger_validation_recent.clear()
                print(
                    f"[BELIEF-MDP] {round_label}: moved previous round's {moved} recent_dagger "
                    f"transitions into older_dagger (now {older_dagger_buffer.size} total), "
                    f"{moved_validation} dagger_validation transitions into historical "
                    f"(now {dagger_validation_historical.size} total)",
                    flush=True,
                )
            print(
                f"[BELIEF-MDP] {round_label}: collecting {args.dagger_episodes_per_round} episodes "
                f"at teacher_prob={teacher_prob:.2f}",
                flush=True,
            )
            expert_fractions = []
            for round_episode in range(args.dagger_episodes_per_round):
                scenario = TRAIN_SCENARIOS[round_episode % len(TRAIN_SCENARIOS)]
                dagger_episode_counter += 1
                is_validation_episode = dagger_episode_counter % args.dagger_validation_every == 0
                target_buffer = dagger_validation_recent if is_validation_episode else recent_dagger_buffer
                reward_sum, steps, outcome, expert_fraction = collect_and_store(
                    expert_prob=teacher_prob, target_buffer=target_buffer, has_expert_label=True, scenario=scenario,
                )
                expert_fractions.append(expert_fraction)
                record = {
                    "type": "train_episode", "episode": total_stage1_episode_counter, "stage": round_label,
                    "scenario": scenario, "outcome": outcome, "reward": reward_sum, "steps": steps,
                    "teacher_prob": teacher_prob, "recent_dagger_buffer_size": recent_dagger_buffer.size,
                    "older_dagger_buffer_size": older_dagger_buffer.size,
                    "collision_buffer_size": collision_buffer.size, "updates": updates,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            print(
                f"[BELIEF-MDP] {round_label}: collected, mean realized expert-action fraction="
                f"{float(np.mean(expert_fractions)):.2f} (target teacher_prob={teacher_prob:.2f}); "
                f"running up to {args.dagger_updates_per_round} offline updates "
                f"(convergence patience={args.dagger_convergence_patience})",
                flush=True,
            )
            run_dagger_round_updates(
                args.dagger_updates_per_round, args.lambda_e_demo, round_label,
                args.dagger_convergence_patience,
            )
            if aborted_reason:
                break
            gate_passed, nominal, stress, stage1_validation = run_stage1_gate_check(
                args.stage1_nav_episodes, round_label.upper()
            )
            save_checkpoint(
                output / f"checkpoint_{round_label}.pth", network, target, optimizer, args,
                {"episode": total_stage1_episode_counter, "nominal": nominal, "stress": stress}, artifact_hashes,
            )
            if gate_passed:
                print(f"[BELIEF-MDP] {round_label}: gate PASSED, stopping DAgger early", flush=True)
                break

    # Smoke is correctness-only: real training-quality thresholds cannot be
    # meaningfully cleared by a handful of updates. Force through to Stage 2
    # here (after DAgger's code path has actually run at least once above)
    # rather than skipping DAgger entirely.
    if args.smoke:
        gate_passed = True

    if not aborted_reason and not gate_passed:
        aborted_reason = "stage1_failed: imitation and/or navigation gate not met after Stage 1b and DAgger"
        save_checkpoint(
            output / "stage1_failed_model.pth", network, target, optimizer, args,
            {
                "episode": total_stage1_episode_counter, "nominal": nominal, "stress": stress,
                "stage1_validation": stage1_validation,
            },
            artifact_hashes,
        )
        print(
            "[BELIEF-MDP] STAGE1 FAILED: saved stage1_failed_model.pth, not entering Stage 2. "
            "See STAGE1-GATE lines above.",
            flush=True,
        )

    # ---- Stage 2: online Double-DQN fine-tuning ----
    rl_episodes_completed = 0
    if not aborted_reason and gate_passed:
        print("[BELIEF-MDP] Stage 2: online Double-DQN fine-tuning", flush=True)
        for episode in range(args.rl_episodes):
            rl_episodes_completed = episode + 1
            rl_index = episode
            scenario = TRAIN_SCENARIOS[episode % len(TRAIN_SCENARIOS)]
            epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, rl_index, args.rl_episodes)
            reward_sum, steps, outcome, _ = collect_and_store(
                expert_prob=0.0, target_buffer=online_rl_buffer, has_expert_label=False,
                scenario=scenario, epsilon=epsilon,
            )
            lambda_e = linear_schedule(args.lambda_e_start, args.lambda_e_end, rl_index, args.rl_episodes)
            if online_rl_buffer.size >= args.warmup_transitions:
                for _ in range(max(1, args.gradient_steps)):
                    latest_losses = optimize(
                        network, target, optimizer, stratified_buffers, STRATIFIED_BUFFER_RATIOS,
                        args.batch_size, args.gamma, lambda_e, rng, device,
                    )
                    soft_update(target, network, args.tau)
                    updates += 1
                    if not math_finite(latest_losses.get("loss")):
                        aborted_reason = f"non-finite loss at rl episode {episode + 1}, update {updates}"
                    else:
                        current_p95 = latest_losses.get("q_r_abs_p95", 0.0)
                        q_r_p95_ema = (
                            current_p95 if q_r_p95_ema is None
                            else Q_R_EMA_DECAY * q_r_p95_ema + (1.0 - Q_R_EMA_DECAY) * current_p95
                        )
                        if q_r_p95_ema > Q_R_ABS_P95_LIMIT:
                            aborted_reason = (
                                f"q_r_abs_p95 EMA={q_r_p95_ema:.3f} exceeded {Q_R_ABS_P95_LIMIT} "
                                f"at rl episode {episode + 1}, update {updates} -- sustained, not a single spike"
                            )
                    if aborted_reason:
                        break
            if aborted_reason:
                print(f"[BELIEF-MDP] ABORT: {aborted_reason}", flush=True)
                break

            record = {
                "type": "train_episode", "episode": episode + 1, "stage": "rl", "scenario": scenario,
                "outcome": outcome, "reward": reward_sum, "steps": steps,
                "online_rl_buffer_size": online_rl_buffer.size, "updates": updates, "lambda_e": lambda_e,
                **latest_losses,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if (episode + 1) % 10 == 0 or episode == 0:
                print(
                    f"[TRAIN] rl {episode + 1}/{args.rl_episodes} outcome={outcome} reward={reward_sum:.3f} "
                    f"online={online_rl_buffer.size} loss={latest_losses.get('loss', float('nan')):.4f} "
                    f"q_r_p95={latest_losses.get('q_r_abs_p95', float('nan')):.3f}",
                    flush=True,
                )

            if (rl_index + 1) % args.eval_interval == 0 or episode + 1 == args.rl_episodes:
                nominal = evaluate_quick(
                    network, make_engine, make_environment, args, device,
                    "nominal", args.eval_episodes, 810000, VALIDATION_SCENARIOS,
                )
                stress = evaluate_quick(
                    network, make_engine, make_environment, args, device,
                    "train_nonstationary", args.eval_episodes, 910000, VALIDATION_SCENARIOS,
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"type": "evaluation", "episode": episode + 1, "nominal": nominal, "stress": stress}) + "\n")
                mean_sr = 0.5 * (nominal["SR"] + stress["SR"])
                print(
                    f"[EVAL] episode={episode + 1} nominal={nominal['SR']:.1%}/{nominal['CR']:.1%} "
                    f"stress={stress['SR']:.1%}/{stress['CR']:.1%}",
                    flush=True,
                )
                # SR dropping between two evals is *not* an abort condition:
                # online RL is not monotonic (a prior run recovered from 33%
                # to 3% back up to 53%), and stopping on a single dip would
                # kill runs that go on to recover. Warn only; the periodic
                # checkpoint below is saved regardless so a later, better
                # episode is never lost to an earlier dip.
                if (
                    not args.smoke and last_eval_sr is not None
                    and (last_eval_sr - mean_sr) > SR_DROP_LIMIT
                ):
                    print(
                        f"[BELIEF-MDP] WARNING: mean SR dropped {last_eval_sr:.1%} -> {mean_sr:.1%} "
                        f"(> {SR_DROP_LIMIT:.0%}) at episode {episode + 1} "
                        "-- not aborting, RL is not monotonic; checkpoint still saved",
                        flush=True,
                    )
                last_eval_sr = mean_sr
                # Periodic, unconditional checkpoint -- model selection
                # happens after training via select_checkpoint.py on
                # independent validation seeds, not from this quick number.
                save_checkpoint(
                    output / f"checkpoint_rl_ep{episode + 1}.pth", network, target, optimizer, args,
                    {"episode": episode + 1, "nominal": nominal, "stress": stress}, artifact_hashes,
                )

    final_metrics = {
        "elapsed_seconds": time.time() - start_time,
        "stage1_episodes": total_stage1_episode_counter,
        "rl_episodes_completed": rl_episodes_completed,
        "rl_episodes_planned": args.rl_episodes,
        "updates": updates,
        "buffer_sizes": {name: buf.size for name, buf in stratified_buffers.items()},
        "aborted_reason": aborted_reason,
    }
    save_checkpoint(output / "final_model.pth", network, target, optimizer, args, final_metrics, artifact_hashes)
    if aborted_reason and aborted_reason.startswith("stage1_failed"):
        status = "stage1_failed"
    elif aborted_reason:
        status = "aborted"
    else:
        status = "completed"
    manifest.update(
        {
            "status": status,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **final_metrics,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[BELIEF-MDP] {status}: {output}", flush=True)


def math_finite(value) -> bool:
    if value is None:
        return True
    return value == value and abs(value) != float("inf")


if __name__ == "__main__":
    main()
