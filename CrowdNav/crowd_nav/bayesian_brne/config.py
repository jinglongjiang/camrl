"""Frozen, validated configuration dataclasses for SM-BRNE.

Every tunable number in this package must come from one of these two
dataclasses -- no magic numbers scattered in algorithm files (guide.md 4.3).
Both are frozen (immutable after construction) so a config object can be
hashed into an artifact's provenance manifest without risk of later mutation
invalidating that hash.

``M=64, H=12`` (the defaults below) are the first formal candidates, not
fixed truths -- they may be tuned exactly once against the validation split
and the latency benchmark (Step 8), then frozen for the rest of the project
(guide.md 4.3).
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple


class ConfigError(ValueError):
    """Raised when a config value is out of its valid range."""


@dataclass(frozen=True)
class BayesianModelConfig:
    """Controls mode_model.py's fitting (K selection, particle count) and
    belief_tracker.py's numerical floors."""

    k_candidates: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    n_particles: int = 128  # reserved/legacy: the exact discrete belief filter does not use particles (see belief_tracker.py); kept for config/manifest compatibility with any future occlusion-aware particle extension
    min_mode_fraction: float = 0.03
    max_mode_similarity: float = 0.98
    covariance_floor: float = 1e-4
    dt: float = 0.25
    # Physical bounds enforced during trajectory_sampler.py rollouts (fixes
    # B3 from the 2026-08-03 independent audit: an earlier version let
    # per-step delta_v accumulate unbounded across a multi-step rollout,
    # producing e.g. a 7.55m lateral spread in 2.5s -- far beyond any real
    # pedestrian's kinematics). Defaults are generous placeholders; guide.md
    # 6.3/B3 requires these be re-derived from train-split speed/acceleration
    # quantiles and frozen before validation once real data exists (Step 5).
    max_human_speed: float = 2.0
    max_human_acceleration: float = 2.0

    def __post_init__(self) -> None:
        if not self.k_candidates or any(int(k) < 1 for k in self.k_candidates):
            raise ConfigError(f"k_candidates must be positive integers, got {self.k_candidates}")
        if len(set(self.k_candidates)) != len(self.k_candidates):
            raise ConfigError(f"k_candidates must not contain duplicates, got {self.k_candidates}")
        if self.n_particles < 1:
            raise ConfigError(f"n_particles must be >= 1, got {self.n_particles}")
        if not (0.0 < self.min_mode_fraction < 1.0):
            raise ConfigError(f"min_mode_fraction must be in (0, 1), got {self.min_mode_fraction}")
        if not (0.0 < self.max_mode_similarity <= 1.0):
            raise ConfigError(f"max_mode_similarity must be in (0, 1], got {self.max_mode_similarity}")
        if self.covariance_floor <= 0.0:
            raise ConfigError(f"covariance_floor must be > 0, got {self.covariance_floor}")
        if self.dt <= 0.0:
            raise ConfigError(f"dt must be > 0, got {self.dt}")
        if self.max_human_speed <= 0.0:
            raise ConfigError(f"max_human_speed must be > 0, got {self.max_human_speed}")
        if self.max_human_acceleration <= 0.0:
            raise ConfigError(f"max_human_acceleration must be > 0, got {self.max_human_acceleration}")


@dataclass(frozen=True)
class PlannerConfig:
    """Controls trajectory_sampler.py, robot_sampler.py, brne_adapter.py,
    and sparse_graph.py at plan/predict time."""

    horizon_steps: int = 12
    num_samples: int = 64
    max_speed: float = 2.0
    max_acceleration: float = 2.0
    max_outer_iterations: int = 10
    outer_tolerance: float = 1e-3
    outer_damping: float = 1.0
    oscillation_tolerance: float = 1e-6
    equilibrium_iterations: int = 10
    solver_mode: str = "stable_clearance"
    safe_distance: float = 0.20
    cost_sigma: float = 0.10
    cost_scale: float = 100.0
    sparse_enabled: bool = False  # R6 graph pruning is not implemented yet.
    pruning_margin: float = 0.0  # Non-zero pruning is fail-closed until R6.
    brne_root: Optional[str] = None
    max_missed_steps: int = 5
    belief_seed: int = 2407
    sampling_mode: str = "full_posterior"
    seed: int = 2407

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ConfigError(f"horizon_steps must be >= 1, got {self.horizon_steps}")
        if self.num_samples < 1:
            raise ConfigError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.max_speed <= 0.0:
            raise ConfigError(f"max_speed must be > 0, got {self.max_speed}")
        if self.max_acceleration <= 0.0:
            raise ConfigError(f"max_acceleration must be > 0, got {self.max_acceleration}")
        if self.max_outer_iterations < 1:
            raise ConfigError(f"max_outer_iterations must be >= 1, got {self.max_outer_iterations}")
        if self.outer_tolerance <= 0.0:
            raise ConfigError(f"outer_tolerance must be > 0, got {self.outer_tolerance}")
        if not (0.0 < self.outer_damping <= 1.0):
            raise ConfigError(f"outer_damping must be in (0, 1], got {self.outer_damping}")
        if self.oscillation_tolerance <= 0.0:
            raise ConfigError(
                f"oscillation_tolerance must be > 0, got {self.oscillation_tolerance}"
            )
        if self.equilibrium_iterations < 1:
            raise ConfigError(f"equilibrium_iterations must be >= 1, got {self.equilibrium_iterations}")
        if self.solver_mode not in ("official_exact", "stable", "stable_clearance"):
            raise ConfigError(f"unknown solver_mode={self.solver_mode!r}")
        if self.safe_distance < 0.0:
            raise ConfigError(f"safe_distance must be >= 0, got {self.safe_distance}")
        if self.cost_sigma <= 0.0:
            raise ConfigError(f"cost_sigma must be > 0, got {self.cost_sigma}")
        if self.cost_scale <= 0.0:
            raise ConfigError(f"cost_scale must be > 0, got {self.cost_scale}")
        if not self.sparse_enabled and self.pruning_margin != 0.0:
            raise ConfigError("pruning_margin must be 0 when sparse_enabled=false; pruning is not implemented yet")
        if self.sparse_enabled:
            raise ConfigError("sparse_enabled=true is not supported before R6; refusing a silent no-op")
        if self.pruning_margin < 0.0:
            raise ConfigError(f"pruning_margin must be >= 0, got {self.pruning_margin}")
        if self.max_missed_steps < 0:
            raise ConfigError(f"max_missed_steps must be >= 0, got {self.max_missed_steps}")


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(v.strip()) for v in raw.split(",") if v.strip())


def load_model_config(config: configparser.RawConfigParser, section: str = "bayesian_model") -> BayesianModelConfig:
    if not config.has_section(section):
        return BayesianModelConfig()
    defaults = BayesianModelConfig()
    return BayesianModelConfig(
        k_candidates=_parse_int_tuple(config.get(section, "k_candidates", fallback=",".join(str(k) for k in defaults.k_candidates))),
        n_particles=config.getint(section, "n_particles", fallback=defaults.n_particles),
        min_mode_fraction=config.getfloat(section, "min_mode_fraction", fallback=defaults.min_mode_fraction),
        max_mode_similarity=config.getfloat(section, "max_mode_similarity", fallback=defaults.max_mode_similarity),
        covariance_floor=config.getfloat(section, "covariance_floor", fallback=defaults.covariance_floor),
        dt=config.getfloat(section, "dt", fallback=defaults.dt),
        max_human_speed=config.getfloat(section, "max_human_speed", fallback=defaults.max_human_speed),
        max_human_acceleration=config.getfloat(section, "max_human_acceleration", fallback=defaults.max_human_acceleration),
    )


def load_planner_config(config: configparser.RawConfigParser, section: str = "planner") -> PlannerConfig:
    if not config.has_section(section):
        return PlannerConfig()
    defaults = PlannerConfig()
    return PlannerConfig(
        horizon_steps=config.getint(section, "horizon_steps", fallback=defaults.horizon_steps),
        num_samples=config.getint(section, "num_samples", fallback=defaults.num_samples),
        max_speed=config.getfloat(section, "max_speed", fallback=defaults.max_speed),
        max_acceleration=config.getfloat(section, "max_acceleration", fallback=defaults.max_acceleration),
        max_outer_iterations=config.getint(section, "max_outer_iterations", fallback=defaults.max_outer_iterations),
        outer_tolerance=config.getfloat(section, "outer_tolerance", fallback=defaults.outer_tolerance),
        outer_damping=config.getfloat(section, "outer_damping", fallback=defaults.outer_damping),
        oscillation_tolerance=config.getfloat(
            section, "oscillation_tolerance", fallback=defaults.oscillation_tolerance
        ),
        equilibrium_iterations=config.getint(section, "equilibrium_iterations", fallback=defaults.equilibrium_iterations),
        solver_mode=config.get(section, "solver_mode", fallback=defaults.solver_mode),
        safe_distance=config.getfloat(section, "safe_distance", fallback=defaults.safe_distance),
        cost_sigma=config.getfloat(section, "cost_sigma", fallback=defaults.cost_sigma),
        cost_scale=config.getfloat(section, "cost_scale", fallback=defaults.cost_scale),
        sparse_enabled=config.getboolean(section, "sparse_enabled", fallback=defaults.sparse_enabled),
        pruning_margin=config.getfloat(section, "pruning_margin", fallback=defaults.pruning_margin),
        max_missed_steps=config.getint(section, "max_missed_steps", fallback=defaults.max_missed_steps),
        belief_seed=config.getint(section, "belief_seed", fallback=defaults.belief_seed),
        sampling_mode=config.get(section, "sampling_mode", fallback=defaults.sampling_mode),
        seed=config.getint(section, "seed", fallback=defaults.seed),
    )


@dataclass(frozen=True)
class RuntimeConfig:
    """Non-algorithm paths and provenance expectations for one policy file."""

    artifact_path: str
    artifact_tier: str
    brne_root: str
    brne_commit: str

    def __post_init__(self) -> None:
        if self.artifact_tier not in ("engineering_only", "production"):
            raise ConfigError(f"unknown artifact_tier={self.artifact_tier!r}")
        if not self.artifact_path:
            raise ConfigError("artifact_path must be non-empty")
        if not self.brne_root:
            raise ConfigError("brne_root must be non-empty")
        if not self.brne_commit:
            raise ConfigError("brne_commit must be non-empty")


def load_runtime_config(config: configparser.RawConfigParser) -> RuntimeConfig:
    if not config.has_section("artifact"):
        raise ConfigError("policy config must define [artifact]")
    if not config.has_section("brne_upstream"):
        raise ConfigError("policy config must define [brne_upstream]")
    return RuntimeConfig(
        artifact_path=config.get("artifact", "path"),
        artifact_tier=config.get("artifact", "tier"),
        brne_root=config.get("brne_upstream", "root"),
        brne_commit=config.get("brne_upstream", "commit"),
    )


def load_policy_config_file(path: str):
    """Load the complete policy file from one source of truth.

    Returns ``(BayesianModelConfig, PlannerConfig, RuntimeConfig)``. The
    planner receives the BRNE root from the runtime section; callers do not
    get to provide a second, conflicting path.
    """
    parser = configparser.RawConfigParser()
    if not parser.read(str(Path(path).expanduser())):
        raise ConfigError(f"policy config not found: {path}")
    model = load_model_config(parser)
    runtime = load_runtime_config(parser)
    planner = load_planner_config(parser)
    planner = replace(planner, brne_root=runtime.brne_root)
    return model, planner, runtime
