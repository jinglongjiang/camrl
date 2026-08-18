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
