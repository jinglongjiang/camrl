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


def test_order10_no_il_checkpoint_selection_machinery_exists():
    """Order 10 forbids picking IL weights on audit metrics.

    IL runs a FIXED budget and the weights standing at the last pass go
    straight to online RL. Any of these names reappearing means a selector
    was reintroduced: a best-weight file, a feasibility filter, or a phase
    flag gating the handover.
    """
    root = Path(__file__).resolve().parents[1]
    banned = ("il_selected", "best_feasible", "il_selection_complete")
    offenders = []
    for f in sorted(root.glob("*.py")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            for name in banned:
                if name in line:
                    offenders.append(f"{f.name}:{n}: {line.strip()[:90]}")
    assert not offenders, (
        "Order 10 selection machinery present:\n" + "\n".join(offenders))


def test_order10_audit_never_drives_control_flow_in_the_il_loop():
    """Audit is telemetry. ``_audit_record`` may write logs and state, but the
    IL loop must not branch, break or return on an audit metric -- that is
    exactly the dynamic early-stopping Order 10 bans.
    """
    src = (Path(__file__).resolve().parents[1] / "intent_train_cli.py").read_text().splitlines()
    start = next(i for i, l in enumerate(src) if "ranking warm-up" in l)
    end = next(i for i, l in enumerate(src) if "online RL phase" in l)
    metrics = re.compile(r"\b(il_audit_mc_loss|il_audit_rank_loss|best_mc|audit_mc_loss)\b")
    control = re.compile(r"^\s*(if|elif|while|assert)\b|\b(break|continue|return)\b")
    offenders = [f"{n}: {src[n - 1].strip()[:90]}"
                 for n in range(start + 1, end + 1)
                 if metrics.search(src[n - 1]) and control.search(src[n - 1])]
    assert not offenders, (
        "audit metric drives control flow inside the IL phase:\n" + "\n".join(offenders))


def test_order10_il_hands_its_final_weights_straight_to_online():
    """Between the last IL pass and the first online episode nothing may
    reload weights. The pass-2400 model/optimizer/EMA continue as they are;
    a ``load_state_dict`` or checkpoint read in that window would mean an
    earlier IL checkpoint is being restored to start online from.
    """
    src = (Path(__file__).resolve().parents[1] / "intent_train_cli.py").read_text().splitlines()
    end_il = next(i for i, l in enumerate(src) if "FINAL" in l and "_audit_record" in l)
    start_online = next(i for i, l in enumerate(src) if "online RL phase" in l)
    reload_ = re.compile(r"load_state_dict|torch\.load|load_checkpoint|restore")
    offenders = [f"{n}: {src[n - 1].strip()[:90]}"
                 for n in range(end_il + 1, start_online + 1)
                 if reload_.search(src[n - 1]) and not src[n - 1].strip().startswith("#")]
    assert not offenders, (
        "weights are reloaded between IL and online:\n" + "\n".join(offenders))


def test_order10_resume_restores_the_cursor_not_a_best_checkpoint():
    """``resume`` continues the run where it stopped. The training cursor is
    restored; the audit's best-MC figure is telemetry that rides along in
    run_state and must never be used to pick which weights come back.

    The bit-identity of resume vs. a continuous run is proved separately in
    test_intent_cli; this pins the structural rule that no best-weight file
    is written or read at all.
    """
    root = Path(__file__).resolve().parents[1]
    best_file = re.compile(r"""(best|selected)[_a-z]*\.pth|["'](best|selected)[_a-z]*["']\s*\.pth""")
    offenders = []
    for f in sorted(root.glob("*.py")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if best_file.search(line):
                offenders.append(f"{f.name}:{n}: {line.strip()[:90]}")
    assert not offenders, (
        "a best/selected weight file is referenced:\n" + "\n".join(offenders))
