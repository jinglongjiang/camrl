"""Aborts carry a TYPE, and the diagnostic ranking exemption is narrow.

Three different failures used to arrive as three English sentences behind one
shared 'ABORT (audit gate):' prefix. The 2x2 runner therefore had to match
prose to tell 'the experiment answered' (MC regression) from 'the run broke'
(non-finite, clipping, ranking) -- and a reworded message would have silently
broken that. These tests pin the typed contract and forbid the string match
from coming back.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from crowd_nav.bayesian_dvl.intent_train import (
    ABORT_CLIPPING, ABORT_MC_REGRESSION, ABORT_NON_FINITE, ABORT_RANKING_QUALITY,
    ABORT_TYPES, ABORT_WARMUP_FAILURE, TrainingHealthMonitor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
QUOTES = (chr(34), chr(39))
TRIPLES = (chr(34) * 3, chr(39) * 3)


def _monitor(**kw):
    kw.setdefault("mc_regression_factor", 1.5)
    kw.setdefault("rank_max", 0.075)
    kw.setdefault("consecutive_bad", 2)
    kw.setdefault("ranking_grace_il_passes", 0)
    return TrainingHealthMonitor(**kw)


def test_every_abort_type_is_declared():
    assert set(ABORT_TYPES) == {"mc_regression", "ranking_quality", "non_finite",
                                "clipping", "warmup_failure"}


def test_healthy_audit_returns_none():
    assert _monitor().observe_audit(0.10, 0.02, 0.9, il_pass=1000) is None


def test_mc_regression_is_typed():
    m = _monitor()
    assert m.observe_audit(0.10, 0.02, 0.9, il_pass=1000) is None
    verdict = m.observe_audit(0.20, 0.02, 0.9, il_pass=1000)
    assert verdict is not None
    kind, message = verdict
    assert kind == ABORT_MC_REGRESSION
    assert "1.5" in message or "2.00x" in message


def test_non_finite_is_typed_and_beats_everything():
    kind, message = _monitor().observe_audit(float("nan"), 0.02, 0.9, il_pass=1000)
    assert kind == ABORT_NON_FINITE and "non-finite" in message


def test_ranking_quality_is_typed():
    m = _monitor()
    assert m.observe_audit(0.10, 0.20, 0.9, il_pass=1000) is None      # one bad audit
    kind, _msg = m.observe_audit(0.10, 0.20, 0.9, il_pass=1000)        # two in a row
    assert kind == ABORT_RANKING_QUALITY


def test_the_diagnostic_exemption_suspends_ONLY_the_ranking_verdict():
    m = _monitor()
    m.observe_audit(0.10, 0.20, 0.9, il_pass=1000, disable_ranking_gate=True)
    assert m.observe_audit(0.10, 0.20, 0.9, il_pass=1000, disable_ranking_gate=True) is None, (
        "the ranking ABORT must be suspended for a diagnostic branch")
    assert m.consecutive_rank_bad >= 2, "the ranking metric must still be tracked, not ignored"
    # everything else still fires
    assert m.observe_audit(float("inf"), 0.02, 0.9, il_pass=1000,
                           disable_ranking_gate=True)[0] == ABORT_NON_FINITE
    m2 = _monitor()
    m2.observe_audit(0.10, 0.02, 0.9, il_pass=1000, disable_ranking_gate=True)
    assert m2.observe_audit(0.30, 0.02, 0.9, il_pass=1000,
                            disable_ranking_gate=True)[0] == ABORT_MC_REGRESSION, (
        "MC regression is the experiment's dependent variable and must never be suspended")


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "crowd_nav.bayesian_dvl.intent_train_cli", *argv],
                          cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("extra,needle", [
    (["--seed", "98201", "--fork-from", "/tmp/nope.pth"], "not a diagnostic optimizer seed"),
    (["--seed", "98211"], "--fork-from is required"),
    (["--seed", "98211", "--fork-from", "/tmp/nope.pth", "--formal-plan", "/tmp/nope.json"],
     "--formal-plan present"),
])
def test_the_exemption_is_refused_outside_a_diagnostic_branch(extra, needle):
    with tempfile.TemporaryDirectory() as d:
        r = _cli("train", "--run-dir", str(Path(d) / "r"), "--diagnostic-disable-ranking-gate",
                 "--il-passes", "1", "--target-online-episodes", "0", *extra)
        assert r.returncode != 0
        assert "--diagnostic-disable-ranking-gate refused" in r.stderr and needle in r.stderr, r.stderr


def test_the_exemption_is_refused_on_resume():
    with tempfile.TemporaryDirectory() as d:
        r = _cli("resume", "--run-dir", str(Path(d) / "r"), "--diagnostic-disable-ranking-gate",
                 "--seed", "98211", "--fork-from", "/tmp/nope.pth",
                 "--il-passes", "1", "--target-online-episodes", "0")
        assert r.returncode != 0 and "resume is not a diagnostic branch start" in r.stderr, r.stderr


def test_the_runner_branches_on_the_type_not_on_the_message():
    """The 2x2 runner must read ABORT_TYPE=. If it goes back to matching a
    sentence, a reworded message silently turns a broken run into a result."""
    runner = REPO_ROOT / "crowd_nav" / "bayesian_dvl" / "tools" / "run_2x2.sh"
    if not runner.exists():
        pytest.skip("runner script is deployed separately")
    text = runner.read_text()
    assert "ABORT_TYPE=" in text
    assert "value regression on the FIXED audit set" not in text, (
        "the runner is matching the message text again")


# ---------------------------------------------------- the fork/branch path
# This path had NO test, and a leftover `args.rank_share` in one log line
# inside it survived a full 259-test suite and then crashed all six
# calibration branches ten seconds after launch. A branch launch is a
# production entry point and gets exercised like one.

def _branch_argv(run_dir, fork, extra=()):
    return ["train", "--run-dir", str(run_dir), "--seed", "98211",
            "--fork-from", str(fork), "--rank-cap-rho", "0.5",
            "--diagnostic-disable-ranking-gate", "--il-passes", "1",
            "--target-online-episodes", "0", *extra]


def test_a_branch_launch_parses_and_passes_its_guards():
    """In-process, so it costs nothing: parse a real branch command line and
    run the diagnostic guards over it. The crash this guards against was an
    ``args.rank_share`` that no longer existed, reached only on this path --
    a subprocess test would need a fresh candidate audit and four minutes."""
    from crowd_nav.bayesian_dvl.intent_train_cli import build_parser
    args = build_parser().parse_args(_branch_argv("/tmp/x", "/tmp/f.pth"))
    assert args.rank_cap_rho == 0.5 and args.fork_from is not None
    assert args.diagnostic_disable_ranking_gate is True
    assert not hasattr(args, "rank_share"), "the retired flag must be gone from the parser"


def test_the_fork_loader_checks_identity_before_touching_the_model():
    """Exercised in process against the real loader, including the log line
    that crashed all six calibration branches ten seconds after launch."""
    import torch
    from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
    from crowd_nav.bayesian_dvl.intent_train_cli import (
        FORK_SCHEMA, IntentCLIError, load_warmup_fork, scene_registry_sha256, code_sha256)

    cfg = load_intent_training_config()

    class _Art:
        model = optimizer = ema = None
        tau_generator = sample_rng = None
        state = None

    with tempfile.TemporaryDirectory() as d:
        fork = Path(d) / "fork.pth"
        torch.save({"fork_schema": FORK_SCHEMA, "code_sha256": "deadbeef",
                    "config_content_sha256": cfg.content_hash(),
                    "scene_registry_sha256": scene_registry_sha256(cfg), "arm": "full"}, str(fork))
        with pytest.raises(IntentCLIError, match="different code_sha256"):
            load_warmup_fork(fork, _Art(), cfg, "full", reset_optimizer=False)

        torch.save({"fork_schema": "not_a_fork"}, str(fork))
        with pytest.raises(IntentCLIError, match="not a warm-up fork"):
            load_warmup_fork(fork, _Art(), cfg, "full", reset_optimizer=False)

        torch.save({"fork_schema": FORK_SCHEMA, "code_sha256": code_sha256(),
                    "config_content_sha256": cfg.content_hash(),
                    "scene_registry_sha256": scene_registry_sha256(cfg), "arm": "mean"}, str(fork))
        with pytest.raises(IntentCLIError, match="different arm"):
            load_warmup_fork(fork, _Art(), cfg, "full", reset_optimizer=False)


def test_no_production_module_still_uses_the_retired_rank_share():
    """The rename that broke the calibration launch: an ``args.rank_share``
    in one log line on the fork path survived a 259-test suite because no
    test walked that path.

    This looks for USES -- an attribute access or a keyword argument -- not
    for the word. The refusal rule that names the retired key, and the
    docstrings explaining why it was retired, must keep saying it.
    """
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"(\.rank_share\b|\brank_share\s*=(?!=))")
    offenders = []
    for f in sorted(root.glob("*.py")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or not pattern.search(stripped):
                continue
            if '"rank_share"' in stripped or "'rank_share'" in stripped:
                continue        # naming the retired key in the refusal table
            offenders.append(f"{f.name}:{n}: {stripped[:90]}")
    assert not offenders, "live rank_share uses remain:\n" + "\n".join(offenders)
