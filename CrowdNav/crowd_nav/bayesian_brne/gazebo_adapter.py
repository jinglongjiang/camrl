"""ROS/Gazebo message <-> PolicyObservation/action format conversion only.

Single responsibility (guide.md 7.6): tracked-humans-in -> PolicyObservation
-> BayesianBRNEPolicy.predict() -> ActionRot(v, omega) -> /cmd_vel out. No
planning/solving logic lives here -- this file only converts formats and
enforces coordinate-frame/timestamp/unit assertions.

STATUS: placeholder, deferred until Steps 1-8 (simulation-only) are
validated -- guide.md's own execution order does not start Gazebo work
before the simulated framework passes its acceptance gates.
"""

from __future__ import annotations

STEP = 8
IMPLEMENTED = False


def _not_ready(*_args, **_kwargs):
    raise NotImplementedError("gazebo_adapter.py is deferred until the simulated framework is validated.")
