"""Auditable JSONL diagnostics for the SM-BRNE runtime (Order R4).

The writer stores one compact episode record per line.  It deliberately does
not store raw candidate tensors or model parameters: those belong in the
artifact/config manifests.  Every line carries the hashes needed to join a
runtime observation back to those immutable inputs.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


class DiagnosticSchemaError(ValueError):
    """Raised when an episode diagnostic omits an audit-critical field."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonlDiagnosticWriter:
    """Append-only, schema-checked episode diagnostics."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Path,
        *,
        resolved_config: Any,
        artifact_sha256: str,
        suite_seed: int,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"diagnostic file already exists: {self.path}")
        if not artifact_sha256:
            raise DiagnosticSchemaError("artifact_sha256 must be non-empty")
        self.config_sha256 = canonical_sha256(resolved_config)
        self.artifact_sha256 = str(artifact_sha256)
        self.suite_seed = int(suite_seed)
        if overwrite:
            self.path.write_text("", encoding="utf-8")

    def write_episode(
        self,
        *,
        episode_seed: int,
        scenario: str,
        profile: str,
        step_records: list[dict],
        termination_event: str,
        elapsed_ms: float,
    ) -> dict:
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "episode",
            "suite_seed": self.suite_seed,
            "episode_seed": int(episode_seed),
            "scenario": str(scenario),
            "profile": str(profile),
            "config_sha256": self.config_sha256,
            "artifact_sha256": self.artifact_sha256,
            "steps": int(len(step_records)),
            "termination_event": str(termination_event),
            "elapsed_ms": float(elapsed_ms),
            "step_records": step_records,
        }
        required = (
            "schema_version", "suite_seed", "episode_seed", "scenario", "profile",
            "config_sha256", "artifact_sha256", "steps", "termination_event",
        )
        if any(record[key] in (None, "") for key in required):
            raise DiagnosticSchemaError(f"missing required episode field in {record}")
        payload = json.dumps(_jsonable(record), sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
        return record


def summarize_policy_diagnostics(diagnostics: Optional[dict], step: int) -> dict:
    """Keep only scalar audit fields from one policy step."""
    diagnostics = diagnostics or {}
    iteration_records = diagnostics.get("iteration_diagnostics", [])
    fallback_count = sum(
        bool(item.get("inner_numeric_fallback_used", False))
        for item in iteration_records
        if isinstance(item, Mapping)
    )
    timing = {
        "sampling_ms": sum(float(item.get("sampling_ms", 0.0)) for item in iteration_records if isinstance(item, Mapping)),
        "solver_wall_ms": sum(float(item.get("solver_wall_ms", 0.0)) for item in iteration_records if isinstance(item, Mapping)),
        "outer_iteration_ms": sum(float(item.get("outer_iteration_ms", 0.0)) for item in iteration_records if isinstance(item, Mapping)),
    }
    return {
        "step": int(step),
        "status": str(diagnostics.get("status", "unknown")),
        "converged": bool(diagnostics.get("converged", False)),
        "outer_iterations": int(diagnostics.get("outer_iterations", 0)),
        "max_weight_residual": float(diagnostics.get("max_weight_residual", 0.0)),
        "final_consistency_residual": float(diagnostics.get("final_consistency_residual", 0.0)),
        "numeric_fallback_count": int(fallback_count),
        "timing_ms": timing,
    }
