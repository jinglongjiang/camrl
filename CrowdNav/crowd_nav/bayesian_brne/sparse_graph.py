"""Reachable-tube pruning for the 20-person sparse interaction graph.

Single responsibility (guide.md 2.6): given all observed agents' trajectory
samples, decide which pairwise edges can be dropped from BRNE's cost
computation because their reachable tubes provably cannot interact within
the planning horizon -- and ONLY those. All agents remain nodes; "nearest-K
silent truncation" is banned (guide.md's explicit prohibition). Does not
choose actions -- that is policy.py's job.

STATUS: placeholder, implemented in Step 6/8 once dense solving is validated
and benchmarked (sparse is only enabled if the dense solver misses its
latency budget at 20 agents, per guide.md 2.6/8.5).
"""

from __future__ import annotations

STEP = 6
IMPLEMENTED = False


def _not_ready(*_args, **_kwargs):
    raise NotImplementedError("sparse_graph.py is a Step 6/8 deliverable, not yet implemented.")
