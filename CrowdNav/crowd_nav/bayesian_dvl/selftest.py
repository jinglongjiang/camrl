"""Registered BDVL self-tests (guide.md section 9, per-stage acceptance).

Run: python3 -m crowd_nav.bayesian_dvl.selftest
Each test function name starts with ``test_`` and is auto-discovered.
Failures raise AssertionError; the runner prints a pass/fail count and
exits non-zero on any failure, matching the project's existing selftest
convention (see crowd_nav/bayesian_brne/selftest.py).

Thin entry point: the V6 test functions live in
``crowd_nav/bayesian_dvl/tests/`` (``test_intent_tracker.py``,
``test_scene_candidates.py``, ``test_intent_policy.py``,
``test_intent_training.py``), covering the goal-intent V6 chain only --
the retired SBK-HMM/R4 chain and its 220 tests were removed from this
branch (Order B). This module only
discovers and runs them -- it defines no tests of its own.
"""

from __future__ import annotations

import inspect

from crowd_nav.bayesian_dvl.tests import (
    test_intent_cli,
    test_intent_integration,
    test_intent_monitor,
    test_intent_policy,
    test_intent_tracker,
    test_intent_training,
    test_scene_candidates,
)

_TEST_MODULES = (
    test_intent_cli,
    test_intent_integration,
    test_intent_monitor,
    test_intent_tracker,
    test_scene_candidates,
    test_intent_policy,
    test_intent_training,
)


def _discover_tests():
    seen = {}
    for module in _TEST_MODULES:
        for name, obj in inspect.getmembers(module):
            if name.startswith("test_") and inspect.isfunction(obj):
                if name in seen and seen[name] is not obj:
                    raise RuntimeError(f"duplicate test name {name!r} defined in multiple test modules")
                seen[name] = obj
    for name, obj in sorted(seen.items()):
        yield name, obj


def main() -> int:
    passed = 0
    failed = 0
    for name, fn in _discover_tests():
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - selftest must report, not crash silently
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
