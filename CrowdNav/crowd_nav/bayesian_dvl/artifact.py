"""Versioned SBK-HMM artifact + hash checks (guide.md 8.1's ``artifact.py``).

This is a thin re-export, not a second implementation: the actual
``SBKHMMArtifact``/``NIWParams``/``promote_to_production``/hash-checked
save-load logic lives in ``world_model.py`` (co-located with the EM
fitter that produces artifacts, since the two are tightly coupled and
keeping them in one file avoids a circular-import risk). This module
exists so callers who only care about the FROZEN ARTIFACT contract
(not the fitting procedure) have a name that matches guide.md's file
layout, without guide.md's fitter/artifact split forcing an actual code
duplication.
"""

from __future__ import annotations

from crowd_nav.bayesian_dvl.world_model import (  # noqa: F401
    CV, ACC, DECEL, TURN_L, TURN_R, N_MODES, MODE_NAMES,
    NIWParams, SBKHMMArtifact, Track, WorldModelError,
    promote_to_production, sample_sign_truncated_predictive,
)
