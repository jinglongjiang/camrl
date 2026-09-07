#!/usr/bin/env python3
"""Lightweight, GPU/CrowdSim-free correctness checks for belief_mdp.

Covers the five specific regressions this project has already hit once:
each test exists because a prior version of this code silently did the
wrong thing and only a real GPU run (expensive) exposed it. Run with:

    python3 -u belief_mdp/selftest.py

Exits nonzero on any failure. This is not a substitute for the GPU smoke
test (--smoke) -- it cannot touch CUDA, mamba_ssm, or CrowdSim -- it only
checks the pure-Python/PyTorch logic that a GPU run would otherwise have
to fail expensively to reveal.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, THIS_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


failures = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def test_margin_zero_when_expert_set_satisfied(train_mod):
    q_score = torch.tensor([[1.0, 0.99, 0.3, 0.1]])
    expert_set = torch.tensor([[0, 1, 2]])
    per_sample = train_mod.expert_set_margin_loss(q_score, expert_set, margin=0.05)
    check(
        "Top-3 expert-set margin loss is 0 when the arg max is any member of the set",
        per_sample.item() == 0.0,
        f"got {per_sample.item()}",
    )

    q_score2 = torch.tensor([[0.1, 0.2, 0.3, 0.99]])
    per_sample2 = train_mod.expert_set_margin_loss(q_score2, expert_set, margin=0.05)
    check(
        "Top-3 expert-set margin loss is positive when the arg max is outside the set",
        per_sample2.item() > 0.0,
        f"got {per_sample2.item()}",
    )


def test_normalized_teacher_regret(train_mod):
    # Teacher scores span [0, 10] for this state; choosing its best action
    # (index 2, score 10) must give exactly zero regret, choosing its worst
    # (index 0, score 0) must give exactly one, and a middling choice must
    # land strictly in between.
    teacher_scores = torch.tensor([[0.0, 5.0, 10.0, 7.0]])
    best_action = torch.tensor([2])
    worst_action = torch.tensor([0])
    middle_action = torch.tensor([1])
    regret_best = train_mod.normalized_teacher_regret(best_action, teacher_scores)
    regret_worst = train_mod.normalized_teacher_regret(worst_action, teacher_scores)
    regret_middle = train_mod.normalized_teacher_regret(middle_action, teacher_scores)
    check(
        "Choosing the teacher's own best action has exactly zero normalized regret",
        abs(regret_best.item() - 0.0) < 1e-6,
        f"got {regret_best.item()}",
    )
    check(
        "Choosing the teacher's worst action has exactly one normalized regret",
        abs(regret_worst.item() - 1.0) < 1e-6,
        f"got {regret_worst.item()}",
    )
    check(
        "A middling choice has regret strictly between the best and worst cases",
        0.0 < regret_middle.item() < 1.0,
        f"got {regret_middle.item()}",
    )
    # Degenerate case: every action has the identical teacher score (e.g. a
    # placeholder all-zero vector for non-expert-labeled rows) -- must not
    # divide by zero.
    flat_scores = torch.zeros((1, 4))
    regret_flat = train_mod.normalized_teacher_regret(torch.tensor([0]), flat_scores)
    check(
        "A flat (zero-range) teacher score vector gives a finite regret, not NaN/inf",
        torch.isfinite(regret_flat).all().item(),
        f"got {regret_flat.item()}",
    )

    # Round 12: a masked (-1e4 safety-sentinel) action must not be allowed
    # to drag teacher_worst down and compress every other action's regret
    # toward zero. Real scores span [5, 6] here; without excluding the
    # sentinel, regret for the middling choice (5.5) would be
    # (6 - 5.5) / (6 - (-1e4)) ~= 0.00005 instead of the correct 0.5.
    masked_scores = torch.tensor([[5.0, 5.5, 6.0, -1e4]])
    regret_middle_masked = train_mod.normalized_teacher_regret(torch.tensor([1]), masked_scores)
    check(
        "A masked (-1e4 sentinel) action is excluded from the regret normalization range",
        abs(regret_middle_masked.item() - 0.5) < 1e-4,
        f"got {regret_middle_masked.item()} (expected ~0.5, not ~0.00005)",
    )
    regret_chose_masked = train_mod.normalized_teacher_regret(torch.tensor([3]), masked_scores)
    check(
        "Choosing the masked action itself clamps to a regret of exactly 1.0, not an unbounded value",
        abs(regret_chose_masked.item() - 1.0) < 1e-6,
        f"got {regret_chose_masked.item()}",
    )


def test_margin_loss_on_qr_never_reaches_qc_head(train_mod, model_mod):
    """Round 12: fixes the specific bug this whole redesign started from --
    margin loss applied to Q_score (not Q_R) gave Q_C a real, negative-sign
    gradient path (shrink Q_C to make Q_score agree with the teacher). This
    had previously only been checked by hand (per the Round 11 report);
    fixing it in code without a permanent regression test would let it
    silently come back. Q_C's head is intentionally zero-initialized (see
    model.py), which would make every gradient trivially zero regardless of
    architecture -- so this test first perturbs it to a random nonzero
    weight, exactly like the earlier isolation test for the same reason.
    """
    network = model_mod.BeliefMDPQNetwork(context_dim=8, belief_dim=5, candidate_dim=12, beta=0.5)
    with torch.no_grad():
        network.q_c_head[2].weight.normal_(0, 0.1)
        network.q_c_head[2].bias.zero_()

    batch = 4
    context = torch.randn(batch, 8)
    belief = torch.randn(batch, 5)
    candidates = torch.randn(batch, 80, 12)
    expert_action_set = torch.randint(0, 80, (batch, 3))

    _, components = network(context, belief, candidates, return_components=True)
    margin_loss = train_mod.expert_set_margin_loss(components["q_r"], expert_action_set).mean()
    network.zero_grad(set_to_none=True)
    margin_loss.backward()

    qc_param_names = ("risk_belief_encoder", "risk_action_encoder", "q_c_head")
    qr_param_names = ("task_context_encoder", "task_kinematic_encoder", "q_r_head")

    def grad_is_exactly_zero(name):
        module = getattr(network, name)
        return all(
            p.grad is None or torch.count_nonzero(p.grad).item() == 0
            for p in module.parameters()
        )

    def grad_has_nonzero(name):
        module = getattr(network, name)
        return any(
            p.grad is not None and torch.count_nonzero(p.grad).item() > 0
            for p in module.parameters()
        )

    check(
        "Margin loss on Q_R produces exactly zero gradient in every RiskEncoder/Q_C submodule",
        all(grad_is_exactly_zero(name) for name in qc_param_names),
        f"nonzero found in: {[n for n in qc_param_names if not grad_is_exactly_zero(n)]}",
    )
    check(
        "Margin loss on Q_R produces a real (nonzero) gradient in every TaskEncoder/Q_R submodule",
        all(grad_has_nonzero(name) for name in qr_param_names),
        f"all-zero found in: {[n for n in qr_param_names if not grad_has_nonzero(n)]}",
    )


def test_margin_loss_only_from_demo_samples(train_mod):
    # Two identical-loss rows, one flagged is_demo=1, the other is_demo=0.
    # The is_demo-weighted mean must equal exactly the demo row's own loss,
    # not an average that lets the online row leak in.
    q_score = torch.tensor([[0.1, 0.2, 0.3, 0.99], [0.1, 0.2, 0.3, 0.99]])
    expert_set = torch.tensor([[0, 1, 2], [0, 1, 2]])
    per_sample = train_mod.expert_set_margin_loss(q_score, expert_set, margin=0.05)
    is_demo = torch.tensor([1.0, 0.0])
    weighted = (per_sample * is_demo).sum() / is_demo.sum().clamp_min(1.0)
    check(
        "Demo-weighted margin loss ignores online (is_demo=0) rows entirely",
        abs(weighted.item() - per_sample[0].item()) < 1e-6,
        f"weighted={weighted.item()} demo_row_loss={per_sample[0].item()}",
    )
    # And the all-online case must not divide by zero / blow up.
    is_demo_none = torch.tensor([0.0, 0.0])
    weighted_none = (per_sample * is_demo_none).sum() / is_demo_none.sum().clamp_min(1.0)
    check(
        "Margin loss is a finite 0 when a batch happens to contain zero demo rows",
        weighted_none.item() == 0.0,
    )


def test_mc_returns_cover_whole_colliding_trajectory(train_mod):
    gamma = 0.99
    transitions = [
        (None, 0, 0, None, 0.01, 0.0, None, False),
        (None, 0, 0, None, 0.01, 0.0, None, False),
        (None, 0, 0, None, 0.01, 0.0, None, False),
        (None, 0, 0, None, -0.5, 1.0, None, True),
    ]
    returns = train_mod.compute_mc_returns(transitions, gamma)
    check(
        "Every state in a colliding trajectory gets a nonzero mc_collision_return",
        all(r[1] > 0 for r in returns),
        f"got {[r[1] for r in returns]}",
    )
    non_colliding = [
        (None, 0, 0, None, 0.01, 0.0, None, False),
        (None, 0, 0, None, 1.0, 0.0, None, True),
    ]
    returns2 = train_mod.compute_mc_returns(non_colliding, gamma)
    check(
        "A non-colliding trajectory has all-zero mc_collision_return (no false positives)",
        all(r[1] == 0.0 for r in returns2),
    )


def test_q_r_ema_ignores_single_spike(train_mod):
    decay = 0.98
    limit = train_mod.Q_R_ABS_P95_LIMIT
    ema = None
    aborted = False
    # 200 updates at a safe value, one single spike far above the limit,
    # then back to safe -- the run must NOT abort on that one spike.
    values = [0.5] * 100 + [50.0] + [0.5] * 100
    for value in values:
        ema = value if ema is None else decay * ema + (1.0 - decay) * value
        if ema > limit:
            aborted = True
    check(
        "A single spike does not trip the Q_R EMA abort threshold",
        not aborted,
        f"final ema={ema}",
    )
    # But truly sustained divergence (every update at 10x the limit) must
    # eventually trip it.
    ema2 = None
    aborted2 = False
    for _ in range(300):
        value = limit * 10
        ema2 = value if ema2 is None else decay * ema2 + (1.0 - decay) * value
        if ema2 > limit:
            aborted2 = True
            break
    check(
        "Sustained divergence (every update over the limit) does trip the EMA abort",
        aborted2,
    )


def test_deterministic_chunks_full_coverage(model_mod):
    def fake_features():
        return model_mod.DecisionFeatures(
            context=np.zeros(4, dtype=np.float32),
            belief=np.zeros(3, dtype=np.float32),
            candidates=np.zeros((5, 2), dtype=np.float32),
        )

    kwargs = dict(context_dim=4, belief_dim=3, action_count=5, candidate_dim=2)
    buf = model_mod.ReplayBuffer(capacity=10, permanent=True, **kwargs)
    for i in range(7):
        buf.add(
            fake_features(), i, i, np.array([i, i, i]), 1.0, 0.0, 0.5, 0.0,
            fake_features(), False, has_expert_label=True, is_expert_action=(i % 2 == 0),
            teacher_scores=np.zeros(5, dtype=np.float32),
        )
    chunks = list(buf.deterministic_chunks(3, "cpu"))
    actions = sorted(sum([c["action"].tolist() for c in chunks], []))
    check(
        "deterministic_chunks covers every stored transition exactly once, no duplicates/gaps",
        actions == list(range(7)),
        f"got {actions}",
    )


def test_sample_stratified_ratios_and_empty_redistribution(model_mod):
    def fake_features():
        return model_mod.DecisionFeatures(
            context=np.zeros(4, dtype=np.float32),
            belief=np.zeros(3, dtype=np.float32),
            candidates=np.zeros((5, 2), dtype=np.float32),
        )

    kwargs = dict(context_dim=4, belief_dim=3, action_count=5, candidate_dim=2)

    def filled(capacity, permanent, marker):
        buf = model_mod.ReplayBuffer(capacity=capacity, permanent=permanent, **kwargs)
        for _ in range(capacity):
            buf.add(
                fake_features(), marker, marker, np.array([marker, marker, marker]),
                1.0, 0.0, 0.5, 0.0, fake_features(), False,
                has_expert_label=True, is_expert_action=False,
                teacher_scores=np.zeros(5, dtype=np.float32),
            )
        return buf

    a = filled(200, True, 0)
    b = filled(200, True, 1)
    c = model_mod.ReplayBuffer(capacity=200, permanent=False, **kwargs)  # empty
    ratios = {"a": 0.5, "b": 0.3, "c": 0.2}
    rng = np.random.default_rng(0)
    batch = model_mod.sample_stratified({"a": a, "b": b, "c": c}, ratios, 1000, rng, "cpu")
    fraction_a = float((batch["action"] == 0).float().mean())
    fraction_b = float((batch["action"] == 1).float().mean())
    check(
        "sample_stratified redistributes an empty buffer's ratio share to the non-empty ones "
        "(0.5/(0.5+0.3)=62.5% from a, 0.3/(0.5+0.3)=37.5% from b, 0% from the empty buffer)",
        abs(fraction_a - 0.625) < 0.05 and abs(fraction_b - 0.375) < 0.05,
        f"fraction_a={fraction_a}, fraction_b={fraction_b}",
    )


def test_export_import_batch_transfers_transitions_exactly(model_mod):
    def fake_features():
        return model_mod.DecisionFeatures(
            context=np.zeros(4, dtype=np.float32),
            belief=np.zeros(3, dtype=np.float32),
            candidates=np.zeros((5, 2), dtype=np.float32),
        )

    kwargs = dict(context_dim=4, belief_dim=3, action_count=5, candidate_dim=2)
    source = model_mod.ReplayBuffer(capacity=10, permanent=True, **kwargs)
    for i in range(6):
        source.add(
            fake_features(), i, i, np.array([i, i, i]), 1.0, 0.0, 0.5, 0.0,
            fake_features(), False, has_expert_label=True, is_expert_action=(i % 2 == 0),
            teacher_scores=np.zeros(5, dtype=np.float32),
        )
    dest = model_mod.ReplayBuffer(capacity=20, permanent=True, **kwargs)
    dest.import_batch(source.export_all())
    check(
        "import_batch(export_all()) transfers every transition's action field exactly, in order",
        dest.size == 6 and list(dest.action[:6]) == list(range(6)),
        f"dest.size={dest.size}, actions={list(dest.action[:6])}",
    )
    source.clear()
    check(
        "clear() resets size/position to empty without touching the destination's already-imported copy",
        source.size == 0 and dest.size == 6,
    )


def test_gradient_conflict_none_without_labels(train_mod, model_mod):
    def fake_features():
        return model_mod.DecisionFeatures(
            context=np.random.randn(256).astype(np.float32),
            belief=np.random.randn(11).astype(np.float32),
            candidates=np.random.rand(80, 12).astype(np.float32),
        )

    from copy import deepcopy
    network = train_mod.BeliefMDPQNetwork(context_dim=256, belief_dim=11, candidate_dim=12, beta=0.5)
    target = deepcopy(network)
    kwargs = dict(context_dim=256, belief_dim=11, action_count=80, candidate_dim=12)
    demo_empty = model_mod.ReplayBuffer(capacity=50, permanent=True, **kwargs)
    online = model_mod.ReplayBuffer(capacity=50, permanent=False, **kwargs)
    rng = np.random.default_rng(0)
    for i in range(50):
        expert_set = np.array([i % 80, (i + 1) % 80, (i + 2) % 80])
        online.add(
            fake_features(), i % 80, (i + 1) % 80, expert_set, 0.01, 0.0, 0.5, 0.1,
            fake_features(), False, has_expert_label=False, is_expert_action=False,
            teacher_scores=np.zeros(80, dtype=np.float32),
        )
    buffers = {"demo_empty": demo_empty, "online": online}
    ratios = {"demo_empty": 0.25, "online": 0.75}
    result = train_mod.compute_gradient_conflict(
        network, target, buffers, ratios, batch_size=16,
        gamma=0.99, rng=rng, device="cpu",
    )
    check(
        "compute_gradient_conflict returns None (not a crash) when a batch has no expert-labeled rows",
        result is None,
    )


def test_find_checkpoints_only_matches_stage2_rl_checkpoints(select_mod):
    """Round 13: a real Round-12-style run directory has checkpoint_ep600.pth
    (Stage 1b gate), checkpoint_dagger_round1..5.pth, stage1_failed_model.pth
    (only present on failure), final_model.pth, and 12 checkpoint_rl_ep*.pth
    files (the actual Stage 2 candidates select_checkpoint.py is meant to
    choose among). The previous pattern only ever matched checkpoint_ep600.pth
    -- confirmed on a real run to silently select the Stage 1b checkpoint
    (which hadn't even passed its own gate) as "best," never touching any of
    the 12 real candidates.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        non_candidates = [
            "checkpoint_ep600.pth",
            "checkpoint_dagger_round1.pth",
            "checkpoint_dagger_round5.pth",
            "stage1_failed_model.pth",
            "final_model.pth",
        ]
        rl_candidates = [f"checkpoint_rl_ep{ep}.pth" for ep in range(200, 2401, 200)]
        for name in non_candidates + rl_candidates:
            (run_dir / name).write_bytes(b"not a real checkpoint, just a name for this test")

        found = select_mod.find_checkpoints(run_dir)
        found_episodes = sorted(episode for episode, _ in found)
        expected_episodes = sorted(range(200, 2401, 200))
        check(
            "find_checkpoints returns exactly the 12 checkpoint_rl_ep*.pth files, "
            "none of checkpoint_ep600.pth/dagger/stage1_failed/final_model",
            found_episodes == expected_episodes and len(found) == len(rl_candidates),
            f"got episodes {found_episodes}, expected {expected_episodes}",
        )


def test_no_selected_model_without_eligible_checkpoint(select_mod):
    baseline = {
        "nominal": {"SR": 0.90, "CR": 0.03, "TR": 0.07, "episodes": 100},
        "train_nonstationary": {"SR": 0.85, "CR": 0.05, "TR": 0.10, "episodes": 100},
    }
    # A candidate far worse than baseline on every axis.
    bad_candidate = {
        "nominal": {"SR": 0.10, "CR": 0.40, "TR": 0.50, "episodes": 100},
        "train_nonstationary": {"SR": 0.10, "CR": 0.50, "TR": 0.40, "episodes": 100},
    }
    eligible = select_mod.is_eligible(bad_candidate, baseline, sr_slack=0.02, cr_slack=0.01, tr_slack=0.02)
    check(
        "A candidate far worse than baseline on SR/CR/TR is never eligible",
        eligible is False,
    )
    # A candidate at/above baseline on every axis within slack.
    good_candidate = {
        "nominal": {"SR": 0.89, "CR": 0.03, "TR": 0.08, "episodes": 100},
        "train_nonstationary": {"SR": 0.84, "CR": 0.05, "TR": 0.11, "episodes": 100},
    }
    eligible2 = select_mod.is_eligible(good_candidate, baseline, sr_slack=0.02, cr_slack=0.01, tr_slack=0.02)
    check(
        "A candidate within slack of baseline on every axis is eligible",
        eligible2 is True,
    )


def test_audit_non_inferior_matches_gate_semantics(audit_mod):
    """stage1_gate_audit.py's non_inferior() must implement exactly the same
    rule as train.py's Stage 1 gate (run_stage1_gate_check's inline
    non_inferior closure) -- the audit exists to re-check that decision at
    higher statistical power, not to apply a different or looser one."""
    # Seed 3407's actual Round 12 DAgger-round-5 stress-profile numbers:
    # baseline SR/CR/TR=100.0/0.0/0.0, candidate SR/CR/TR=96.67/0.0/3.33 --
    # fails on both SR (96.67 < 100-3=97.0) and TR (3.33 > 0+2=2.0).
    baseline = {"SR": 1.0, "CR": 0.0, "TR": 0.0}
    near_miss_candidate = {"SR": 0.9667, "CR": 0.0, "TR": 0.0333}
    passed = audit_mod.non_inferior(near_miss_candidate, baseline, sr_slack=0.03, cr_slack=0.02, tr_slack=0.02)
    check(
        "non_inferior reproduces seed 3407's original FAIL on its exact reported numbers",
        passed is False,
        f"got {passed}",
    )
    clearly_fine_candidate = {"SR": 0.99, "CR": 0.0, "TR": 0.01}
    passed2 = audit_mod.non_inferior(clearly_fine_candidate, baseline, sr_slack=0.03, cr_slack=0.02, tr_slack=0.02)
    check(
        "non_inferior passes a candidate clearly within slack on every axis",
        passed2 is True,
        f"got {passed2}",
    )


def main():
    train_mod = _load("belief_mdp_train_selftest", "train.py")
    select_mod = _load("belief_mdp_select_selftest", "select_checkpoint.py")
    model_mod = _load("belief_mdp_model_selftest", "model.py")
    audit_mod = _load("belief_mdp_audit_selftest", "stage1_gate_audit.py")

    test_margin_zero_when_expert_set_satisfied(train_mod)
    test_normalized_teacher_regret(train_mod)
    test_margin_loss_on_qr_never_reaches_qc_head(train_mod, model_mod)
    test_margin_loss_only_from_demo_samples(train_mod)
    test_mc_returns_cover_whole_colliding_trajectory(train_mod)
    test_q_r_ema_ignores_single_spike(train_mod)
    test_deterministic_chunks_full_coverage(model_mod)
    test_sample_stratified_ratios_and_empty_redistribution(model_mod)
    test_export_import_batch_transfers_transitions_exactly(model_mod)
    test_gradient_conflict_none_without_labels(train_mod, model_mod)
    test_find_checkpoints_only_matches_stage2_rl_checkpoints(select_mod)
    test_no_selected_model_without_eligible_checkpoint(select_mod)
    test_audit_non_inferior_matches_gate_semantics(audit_mod)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        raise SystemExit(1)
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
