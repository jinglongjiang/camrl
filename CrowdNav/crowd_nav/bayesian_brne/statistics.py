"""Paired statistics, suite-seed-block bootstrap, and table output.

Single responsibility (guide.md 9.5): inner-join episode records across
methods by their unique key, error out on missing episodes (never pad/align
by row number), and compute paired-difference bootstrap CIs with SEED (not
episode) as the top-level resampling block -- matching
bayesian_decision_gate/bootstrap.py's convention, reused/ported here rather
than re-derived. Never re-runs environments to fill in gaps.

STATUS: placeholder, implemented in Step 9 once Steps 1-8 pass.
"""

from __future__ import annotations

STEP = 9
IMPLEMENTED = False


def _not_ready(*_args, **_kwargs):
    raise NotImplementedError("statistics.py is a Step 9 deliverable, not yet implemented.")
