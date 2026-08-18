"""Projection safety is a statement about the RAW gradient, not the step.

``combine_gradients`` removes any component of the ranking gradient that
opposes the MC gradient, so the combined RAW gradient always satisfies
g . g_MC >= |g_MC|^2 > 0. It is tempting -- and the code used to say so in
so many words -- to conclude that ranking "can never undo value regression".
That conclusion does not survive contact with Adam, which rescales each
coordinate by its own accumulated second moment: the realised parameter
change is a different direction, and its alignment with MC descent can be
driven to nearly zero without the dot product ever changing sign.

These tests pin that distinction in two places: a 2-D case where it can be
read off by hand, and the real model.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from crowd_nav.bayesian_dvl.intent_train import combine_gradients


def _cos(a, b):
    na = sum(float((x ** 2).sum()) for x in a) ** 0.5
    nb = sum(float((x ** 2).sum()) for x in b) ** 0.5
    return sum(float((x * y).sum()) for x, y in zip(a, b)) / max(na * nb, 1e-12)


def test_projection_makes_the_raw_combined_gradient_safe():
    """The guarantee that DOES hold: after projection the combined gradient
    still has a non-negative MC component."""
    g_mc = [torch.tensor([1.0, 0.0])]
    for rank in ([torch.tensor([-3.0, 1.0])], [torch.tensor([0.0, 2.0])],
                 [torch.tensor([2.0, 2.0])]):
        combined, info = combine_gradients(g_mc, rank, rho=2.0)
        dot = float((combined[0] * g_mc[0]).sum())
        assert dot > 0.0, f"projected update must keep descending MC, got {dot}"


def test_adam_can_destroy_the_alignment_the_projection_guarantees():
    """A 2-D case, worked by hand.

    Parameter 0 has seen large gradients, parameter 1 small ones. Adam
    divides by sqrt(v), so the coordinate MC cares about is damped and the
    coordinate it does not care about is amplified. The RAW combined
    gradient is 45 degrees from g_MC; the step Adam actually takes is nearly
    perpendicular to it -- while dot(g_MC, delta) stays negative throughout,
    so a sign test reports nothing wrong.
    """
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p], lr=1e-2, betas=(0.0, 0.999), eps=1e-12)

    # build asymmetric second moments: coordinate 0 large, coordinate 1 tiny
    for _ in range(200):
        opt.zero_grad()
        p.grad = torch.tensor([1.0, 1e-3])
        opt.step()

    g_mc = [torch.tensor([1.0, 0.0])]
    g_rank = [torch.tensor([0.0, 1.0])]           # orthogonal: projection removes nothing
    combined, info = combine_gradients(g_mc, g_rank, rho=2.0)
    assert info["conflict_removed"] is False
    raw_cos = _cos(combined, g_mc)
    assert raw_cos > 0.4, f"raw combined gradient is well aligned with MC ({raw_cos:.3f})"

    before = p.detach().clone()
    opt.zero_grad()
    p.grad = combined[0].clone()
    opt.step()
    delta = [p.detach() - before]

    dot = float((g_mc[0] * delta[0]).sum())
    post_cos = -dot / max(
        float((delta[0] ** 2).sum()) ** 0.5 * float((g_mc[0] ** 2).sum()) ** 0.5, 1e-12)
    assert dot < 0.0, "the step still descends MC to first order -- the sign test passes"
    assert post_cos < 0.2 * raw_cos, (
        f"but its ALIGNMENT collapsed: raw {raw_cos:.3f} -> post-Adam {post_cos:.3f}. "
        "This is why the sign of the dot product cannot be the health metric.")


def test_the_docstring_no_longer_claims_ranking_can_never_undo_mc():
    """The absolute claim was wrong once Adam is in the loop; it must not
    survive in the source as a licence to skip the measurement."""
    import inspect
    from crowd_nav.bayesian_dvl import intent_train
    src = inspect.getsource(intent_train.combine_gradients)
    lowered = src.lower()
    for banned in ("can never actively undo", "never undo value regression"):
        assert banned not in lowered, (
            f"combine_gradients still claims {banned!r}; the guarantee is about the RAW gradient only")
    assert "adam" in lowered, "the docstring must state that the guarantee is pre-optimizer"


def test_train_step_reports_both_metrics_and_only_when_asked():
    from crowd_nav.bayesian_dvl.tests._common import (
        DistributionalValueModel, HUMAN_FEATURE_DIM_V5, batch_to_tensors,
        collect_orca_episode, _env_config_path)
    import crowd_nav.bayesian_dvl.intent_train as T

    tr = []
    for sc, sd in (("standard", 2_600_000), ("junction", 96001)):
        tr += collect_orca_episode(_env_config_path(), sc, sd).transitions
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = batch_to_tensors(tr)
    gen = torch.Generator().manual_seed(0)

    off = T.train_step(model, opt, batch, gen, rho=2.0, diagnose=False)
    assert off.post_adam_cos_mc == 0.0 and off.module_diagnostics == {}, (
        "diagnostics cost two full parameter copies; they must be opt-in")

    on = T.train_step(model, opt, batch, gen, rho=2.0, diagnose=True)
    assert on.adam_step > 0
    assert 0.0 <= on.active_hinge_fraction <= 1.0
    assert set(on.module_diagnostics) == {"encoder", "action_encoder", "value_network"}
    for m in on.module_diagnostics.values():
        assert {"mc_grad_norm", "rank_grad_norm", "delta_norm", "dot_mc", "cos_mc"} <= set(m)


# ------------------------------------------------- capped ranking budget
# The old rule normalised the projected ranking gradient to a FIXED multiple
# of |g_MC|, which is a floor as well as a ceiling: a nearly-satisfied hinge
# was scaled back UP, so ranking never yielded as it converged. Measured over
# 3 diagnostic seeds x IL 2000, that degraded held-out MC on 3/3 (final/best
# 1.500 / 1.285 / 1.162) and the fixed TRAINING set with it. The budget is
# now an upper bound only.

def _norm(vs):
    return sum(float((v ** 2).sum()) for v in vs) ** 0.5


def test_a_small_ranking_gradient_is_never_amplified():
    from crowd_nav.bayesian_dvl.intent_train import combine_gradients
    g_mc = [torch.tensor([3.0, 4.0])]                       # |g_MC| = 5
    tiny = [torch.tensor([0.0, 1.0])]                       # inside a 0.5 budget of 2.5
    combined, info = combine_gradients(g_mc, tiny, rho=0.5)
    assert info["rank_scale"] == pytest.approx(1.0), "already inside budget -> pass through untouched"
    assert info["rank_scale"] * info["projected_rank_norm"] == pytest.approx(1.0)
    assert torch.allclose(combined[0], g_mc[0] + tiny[0])


@pytest.mark.parametrize("rho,expected", [(0.5, 2.5), (1.0, 5.0)])
def test_a_large_ranking_gradient_is_capped_exactly_at_rho_times_mc(rho, expected):
    from crowd_nav.bayesian_dvl.intent_train import combine_gradients
    g_mc = [torch.tensor([3.0, 4.0])]                       # |g_MC| = 5
    big = [torch.tensor([0.0, 10.0])]
    _combined, info = combine_gradients(g_mc, big, rho=rho)
    assert info["rank_scale"] * info["projected_rank_norm"] == pytest.approx(expected)
    assert info["rank_scale"] == pytest.approx(expected / 10.0)


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("scale_of_rank", [1e-6, 0.1, 1.0, 10.0, 1e4])
def test_rank_scale_is_never_greater_than_one(rho, scale_of_rank):
    from crowd_nav.bayesian_dvl.intent_train import combine_gradients
    g_mc = [torch.tensor([1.0, -2.0, 0.5])]
    rank = [torch.tensor([0.3, 0.4, -0.2]) * scale_of_rank]
    _c, info = combine_gradients(g_mc, rank, rho=rho)
    assert info["rank_scale"] <= 1.0 + 1e-12, info


def test_zero_ranking_gradient_is_bitwise_the_pure_mc_update():
    from crowd_nav.bayesian_dvl.intent_train import combine_gradients
    g_mc = [torch.tensor([3.0, 4.0]), torch.tensor([[1.0, -1.0]])]
    zero = [torch.zeros(2), torch.zeros(1, 2)]
    combined, info = combine_gradients(g_mc, zero, rho=0.5)
    assert info["rank_scale"] == 0.0
    for c, m in zip(combined, g_mc):
        assert torch.equal(c, m), "a satisfied hinge must leave the MC update untouched, bit for bit"


def test_the_conflicting_component_is_still_projected_out():
    from crowd_nav.bayesian_dvl.intent_train import combine_gradients
    g_mc = [torch.tensor([1.0, 0.0])]
    for rank in ([torch.tensor([-3.0, 1.0])], [torch.tensor([-10.0, 0.1])]):
        combined, info = combine_gradients(g_mc, rank, rho=1.0)
        assert info["conflict_removed"] is True
        assert float((combined[0] * g_mc[0]).sum()) > 0.0, "the update must still descend MC"


def test_the_retired_fixed_share_semantics_cannot_come_back():
    """rank_share normalised UP as well as down. A config still declaring it
    is describing a procedure the code no longer runs."""
    import configparser
    from crowd_nav.bayesian_dvl.intent_config import (
        DEFAULT_TRAINING_CONFIG, IntentConfigError, load_intent_training_config)

    cfg = load_intent_training_config()
    assert not hasattr(cfg, "rank_share")
    assert 0.0 <= cfg.rank_cap_rho <= 10.0

    parser = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    parser.read(str(DEFAULT_TRAINING_CONFIG))
    assert not parser.has_option("gradient_balance", "rank_share")

    with tempfile.TemporaryDirectory() as d:
        parser.set("gradient_balance", "rank_share", "2.0")
        path = Path(d) / "cfg.config"
        with open(path, "w") as fh:
            parser.write(fh)
        with pytest.raises(IntentConfigError, match="rank_share"):
            load_intent_training_config(path)
