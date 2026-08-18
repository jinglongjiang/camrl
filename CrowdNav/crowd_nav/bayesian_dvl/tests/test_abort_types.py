"""Aborts carry a TYPE, and only arithmetic failures abort at all.

Three different failures used to arrive as three English sentences behind
one shared prefix, so the only way to tell them apart was to match prose.
The type survives; the QUALITY verdicts do not. MC regression, ranking
quality and the clipping-fraction window used to stop training mid-run --
they now belong to the selector, which sees the whole curve rather than the
last two samples.
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
    ABORT_NON_FINITE, ABORT_TYPES, ABORT_WARMUP_FAILURE, AuditRecorder,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
QUOTES = (chr(34), chr(39))
TRIPLES = (chr(34) * 3, chr(39) * 3)


def test_only_arithmetic_and_io_failures_can_abort():
    assert set(ABORT_TYPES) == {"non_finite", "warmup_failure"}, (
        "quality verdicts -- mc_regression, ranking_quality, clipping -- belong to the "
        "selector, not to the training loop")


def test_the_recorder_tracks_the_best_and_never_returns_a_verdict():
    r = AuditRecorder()
    assert r.observe_audit(0.20, 0.02, 0.9) is None
    assert r.best_mc == pytest.approx(0.20)
    assert r.observe_audit(0.10, 0.02, 0.9) is None
    assert r.best_mc == pytest.approx(0.10)
    # a large regression is RECORDED, not judged
    assert r.observe_audit(0.90, 0.90, 0.1) is None
    assert r.best_mc == pytest.approx(0.10) and r.n_checks == 3


def test_the_recorder_survives_a_resume_round_trip():
    r = AuditRecorder()
    r.observe_audit(0.11, 0.02, 0.9)
    r2 = AuditRecorder()
    r2.load_state_dict(r.state_dict())
    assert r2.best_mc == r.best_mc and r2.n_checks == r.n_checks


def test_a_non_finite_audit_does_not_poison_the_best():
    r = AuditRecorder()
    r.observe_audit(0.10, 0.02, 0.9)
    r.observe_audit(float("nan"), 0.02, 0.9)
    assert r.best_mc == pytest.approx(0.10)


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "crowd_nav.bayesian_dvl.intent_train_cli", *argv],
                          cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)


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
