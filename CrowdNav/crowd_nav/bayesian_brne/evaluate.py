"""Closed-loop evaluation of SM-BRNE and its ablations (guide.md 9.6).

Single responsibility: run real closed-loop episodes (robot action changes
each step -> all interactive pedestrians re-respond next step) for one
method at a time, writing a uniform episode-record schema keyed by
``(profile, scenario, suite_seed, episode_index, initial_state_hash)`` for
later pairing in statistics.py / compare_bayesian_brne.py. Does not compute
paired statistics itself, and does not re-run environments to fix missing
rows (statistics.py is explicitly barred from doing that).

STATUS: placeholder, implemented in Step 9 once Steps 1-8 pass.
"""

from __future__ import annotations

STEP = 9
IMPLEMENTED = False


def _not_ready(*_args, **_kwargs):
    raise NotImplementedError("evaluate.py is a Step 9 deliverable, not yet implemented.")
