#!/usr/bin/env python3
"""Real CrowdSim episode smoke test (guide.md 8.4) -- engineering
correctness only, never a statistical gate.

Checks: all six scenarios run, human counts are exactly 5/10/10/20/12/20,
actions are finite and in-bounds, belief resets every episode, 20-person
scenarios show `observed=20, planned=20` (no silent truncation), no Mamba
checkpoint is imported, and manifest/episode-CSV/latency-JSON are written.

STATUS: placeholder, implemented in Step 8 once policy.py (Step 6) exists.
"""

from __future__ import annotations

STEP = 8
IMPLEMENTED = False


def _not_ready(*_args, **_kwargs):
    raise NotImplementedError("test_integration.py is a Step 8 deliverable, not yet implemented.")


def main():
    _not_ready()


if __name__ == "__main__":
    main()
