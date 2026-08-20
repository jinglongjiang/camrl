"""Loader for the ONE formal training config (Order C2.1).

``train_intent_bdvl.config`` is the single source of truth for a formal
V6 run. Real problem this fixes (plan 2.2 point 3): the CLI carried its
OWN hyperparameter defaults (online_iters=0, batch 32, lr 1e-3, gamma
0.95, fixed epsilon 0.2, buffer 20000) which were smoke-test values
silently diverging from the project's actual training configuration
(5000 IL / 10000 RL / batch 256 / lr 1e-4 / gamma 0.99 / epsilon
0.30->0.05 / buffer 200000). With two places able to define a formal
run's hyperparameters, "what did we actually train with" was unanswerable.

This module is belief-free and imports no scenario/env machinery, so it
can be loaded by a preflight check without building CrowdSim.
"""

from __future__ import annotations

import configparser
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


class IntentConfigError(ValueError):
    pass


DEFAULT_TRAINING_CONFIG = Path("crowd_nav/configs/train_intent_bdvl.config")


@dataclass(frozen=True)
class IntentTrainingConfig:
    # [schema]
    feature_schema: str
    training_contract_schema: str
    checkpoint_schema: str
    scenario_registry_id: str
    # [il]
    il_episodes_total: int
    il_episodes_standard: int
    il_episodes_junction_crowd: int
    il_passes: int
    # [online]
    online_episodes_total: int
    mix_standard: float
    mix_junction_crowd: float
    updates_per_episode: int
    # [optim]
    batch_size: int
    demo_sample_ratio: float
    demo_capacity: int
    replay_capacity: int
    learning_rate: float
    gamma: float
    grad_clip_norm: float
    # [exploration]
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: int
    # [ranking]
    ranking_margin: float
    ranking_batch_size: int
    gradient_diagnostic_interval: int
    # [iqn]
    iqn_train_quantiles: int
    iqn_eval_quantiles: int
    # [belief]
    future_horizon: int
    future_n_samples: int
    tracker_sigma: float
    tracker_persistence: float
    tracker_waypoint_radius: float
    missing_timeout_steps: int
    tracker_speed_prior: float
    tracker_estimate_speed: bool
    tracker_speed_ema_alpha: float
    tracker_speed_min: float
    tracker_speed_max: float
    max_candidate_goals: int
    # [ema]
    ema_decay: float
    # [checkpoint]
    checkpoint_interval_episodes: int
    # [monitoring]
    monitor_rolling_windows: Tuple[int, ...]
    monitor_plot_interval_episodes: int
    development_eval_interval_episodes: int
    tensorboard_enabled: bool
    # [seeds]
    training_seeds: Tuple[int, ...]
    validation_seeds: Tuple[int, ...]
    # Order 2R/3R: projected-gradient balance, warm-up and effect-based health
    rank_cap_rho: float = 0.5
    audit_episodes_per_scenario: int = 50
    rank_zero_tolerance: float = 1e-8
    warmup_max_steps: int = 750
    warmup_check_interval: int = 50
    warmup_rank_loss_max: float = 0.05
    audit_interval: int = 100
    # provenance
    source_path: str = ""
    source_sha256: str = ""

    def epsilon_at(self, online_episode_index: int) -> float:
        """Linear decay across ``epsilon_decay_episodes``, then flat at
        ``epsilon_end``. ``online_episode_index`` is the GLOBAL count of
        online episodes already completed (so resume continues the
        schedule rather than restarting it -- plan 2.2 point 4)."""
        if online_episode_index < 0:
            raise IntentConfigError(f"online_episode_index must be >= 0, got {online_episode_index}")
        if self.epsilon_decay_episodes <= 0:
            return self.epsilon_end
        frac = min(1.0, float(online_episode_index) / float(self.epsilon_decay_episodes))
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac)

    #: Fields that actually determine the RAW ORCA corpus. Everything else
    #: -- optimizer settings, rank_share, gate thresholds, EMA, epsilon --
    #: provably cannot change a single recorded transition, so it must not
    #: be able to invalidate 5000 episodes of collection. Measured cost of
    #: getting this wrong: a config edit renamed the corpus file AND the
    #: in-file guard rejected the old one, making reuse impossible.
    CORPUS_IDENTITY_FIELDS = (
        "il_episodes_total", "il_episodes_standard", "il_episodes_junction_crowd",
        "gamma", "future_horizon", "future_n_samples",
        "feature_schema", "checkpoint_schema", "max_candidate_goals",
    )

    def corpus_identity_hash(self) -> str:
        """Identity of the RAW ORCA corpus: environment, teacher, reward,
        gamma, action table, scenes and seeds -- not training knobs."""
        payload = {k: getattr(self, k) for k in self.CORPUS_IDENTITY_FIELDS}
        blob = repr(sorted(payload.items())).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def content_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k not in ("source_path", "source_sha256")}
        blob = repr(sorted(payload.items())).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _seed_tuple(raw: str, field_name: str) -> Tuple[int, ...]:
    try:
        seeds = tuple(int(tok.strip()) for tok in raw.split(",") if tok.strip())
    except ValueError as exc:
        raise IntentConfigError(f"{field_name} must be a comma-separated integer list, got {raw!r}") from exc
    if not seeds:
        raise IntentConfigError(f"{field_name} must not be empty")
    if len(set(seeds)) != len(seeds):
        raise IntentConfigError(f"{field_name} contains duplicates: {seeds}")
    return seeds


def _positive_int_tuple(raw: str, field_name: str) -> Tuple[int, ...]:
    values = _seed_tuple(raw, field_name)
    if any(v <= 0 for v in values):
        raise IntentConfigError(f"{field_name} must contain only positive integers, got {values}")
    return values


def load_intent_training_config(path: Path = DEFAULT_TRAINING_CONFIG) -> IntentTrainingConfig:
    """Read + HARD-VALIDATE the formal config. Fails closed on anything
    inconsistent rather than silently training under a broken budget."""
    path = Path(path)
    parser = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not parser.read(str(path)):
        raise IntentConfigError(f"training config not found: {path}")
    # RETIRED keys: present in the file means the config is describing a
    # training procedure the code no longer runs. lambda_rank was a scalar
    # weight on the ranking term; gradients are now combined by projection
    # with rank_share, so a value here would be hashed into provenance and
    # take no part in training -- the same provenance lie as the tracker
    # knobs that were declared and never passed.
    for section, key, why in (
        ("ranking", "lambda_rank",
         "retired: gradients are combined by projection with a capped ranking budget, "
         "not by a scalar weight"),
        ("gradient_balance", "rank_share",
         "retired: it normalised the ranking gradient to a FIXED multiple of |g_MC|, which is a "
         "floor as well as a ceiling and degraded held-out MC on 3/3 diagnostic seeds. Use "
         "rank_cap_rho, which is an upper bound only"),
    ):
        if parser.has_option(section, key):
            raise IntentConfigError(f"config declares [{section}] {key}, which is {why}; remove it")

    for section in ("schema", "il", "online", "optim", "exploration", "ranking", "iqn", "belief", "ema",
                    "checkpoint", "monitoring", "seeds"):
        if not parser.has_section(section):
            raise IntentConfigError(f"training config {path} is missing required section [{section}]")

    g, gi, gf = parser.get, parser.getint, parser.getfloat
    cfg = IntentTrainingConfig(
        feature_schema=g("schema", "feature_schema").strip(),
        training_contract_schema=g("schema", "training_contract_schema").strip(),
        checkpoint_schema=g("schema", "checkpoint_schema").strip(),
        scenario_registry_id=g("schema", "scenario_registry_id").strip(),
        il_episodes_total=gi("il", "il_episodes_total"),
        il_episodes_standard=gi("il", "il_episodes_standard"),
        il_episodes_junction_crowd=gi("il", "il_episodes_junction_crowd"),
        il_passes=gi("il", "il_passes"),
        online_episodes_total=gi("online", "online_episodes_total"),
        mix_standard=gf("online", "mix_standard"),
        mix_junction_crowd=gf("online", "mix_junction_crowd"),
        updates_per_episode=gi("online", "updates_per_episode"),
        batch_size=gi("optim", "batch_size"),
        demo_sample_ratio=gf("optim", "demo_sample_ratio"),
        demo_capacity=gi("optim", "demo_capacity"),
        replay_capacity=gi("optim", "replay_capacity"),
        learning_rate=gf("optim", "learning_rate"),
        gamma=gf("optim", "gamma"),
        grad_clip_norm=gf("optim", "grad_clip_norm"),
        epsilon_start=gf("exploration", "epsilon_start"),
        epsilon_end=gf("exploration", "epsilon_end"),
        epsilon_decay_episodes=gi("exploration", "epsilon_decay_episodes"),
        ranking_margin=gf("ranking", "ranking_margin"),
        ranking_batch_size=gi("ranking", "ranking_batch_size"),
        gradient_diagnostic_interval=gi("ranking", "gradient_diagnostic_interval"),
        iqn_train_quantiles=gi("iqn", "iqn_train_quantiles"),
        iqn_eval_quantiles=gi("iqn", "iqn_eval_quantiles"),
        future_horizon=gi("belief", "future_horizon"),
        future_n_samples=gi("belief", "future_n_samples"),
        tracker_sigma=gf("belief", "tracker_sigma"),
        tracker_persistence=gf("belief", "tracker_persistence"),
        tracker_waypoint_radius=gf("belief", "tracker_waypoint_radius"),
        missing_timeout_steps=gi("belief", "missing_timeout_steps"),
        tracker_speed_prior=gf("belief", "tracker_speed_prior"),
        tracker_estimate_speed=g("belief", "tracker_estimate_speed").strip().lower() in ("1", "true", "yes"),
        tracker_speed_ema_alpha=gf("belief", "tracker_speed_ema_alpha"),
        tracker_speed_min=gf("belief", "tracker_speed_min"),
        tracker_speed_max=gf("belief", "tracker_speed_max"),
        max_candidate_goals=gi("belief", "max_candidate_goals"),
        ema_decay=gf("ema", "ema_decay"),
        checkpoint_interval_episodes=gi("checkpoint", "checkpoint_interval_episodes"),
        monitor_rolling_windows=_positive_int_tuple(
            g("monitoring", "rolling_windows"), "rolling_windows"),
        monitor_plot_interval_episodes=gi("monitoring", "plot_interval_episodes"),
        development_eval_interval_episodes=gi("monitoring", "development_eval_interval_episodes"),
        tensorboard_enabled=parser.getboolean("monitoring", "tensorboard_enabled"),
        training_seeds=_seed_tuple(g("seeds", "training_seeds"), "training_seeds"),
        validation_seeds=_seed_tuple(g("seeds", "validation_seeds"), "validation_seeds"),
        rank_cap_rho=gf("gradient_balance", "rank_cap_rho"),
        audit_episodes_per_scenario=gi("audit_split", "audit_episodes_per_scenario"),
        rank_zero_tolerance=gf("gradient_balance", "rank_zero_tolerance"),
        warmup_max_steps=gi("ranking_warmup", "warmup_max_steps"),
        warmup_check_interval=gi("ranking_warmup", "warmup_check_interval"),
        warmup_rank_loss_max=gf("ranking_warmup", "warmup_rank_loss_max"),
        audit_interval=gi("audit", "audit_interval"),
        source_path=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: IntentTrainingConfig) -> None:
    # schema must match the CODE, not just be internally consistent
    from crowd_nav.bayesian_dvl.intent_runtime_config import (
        FEATURE_SCHEMA_V6, TRAINING_CONTRACT_V6_EPISODE_BALANCED_REPLAY,
    )
    from crowd_nav.bayesian_dvl.intent_policy import CHECKPOINT_SCHEMA_V7
    if cfg.feature_schema != FEATURE_SCHEMA_V6:
        raise IntentConfigError(f"config feature_schema {cfg.feature_schema!r} != code's {FEATURE_SCHEMA_V6!r}")
    if cfg.training_contract_schema != TRAINING_CONTRACT_V6_EPISODE_BALANCED_REPLAY:
        raise IntentConfigError(
            f"config training_contract_schema {cfg.training_contract_schema!r} != "
            f"code's {TRAINING_CONTRACT_V6_EPISODE_BALANCED_REPLAY!r}")
    if cfg.checkpoint_schema != CHECKPOINT_SCHEMA_V7:
        raise IntentConfigError(f"config checkpoint_schema {cfg.checkpoint_schema!r} != code's {CHECKPOINT_SCHEMA_V7!r}")

    # Every tracker knob the config declares must be one the code actually
    # uses. These were previously hashed into provenance and then ignored,
    # because the bank was constructed without passing them; they matched the
    # defaults, so the gap was invisible. Fail closed instead.
    from crowd_nav.bayesian_dvl.junction_scenario import SCENARIO_REGISTRY_ID
    if cfg.scenario_registry_id != SCENARIO_REGISTRY_ID:
        raise IntentConfigError(
            f"config scenario_registry_id {cfg.scenario_registry_id!r} != code's {SCENARIO_REGISTRY_ID!r}. "
            "Nothing compared these before, so a V1 config passed preflight against V2 code -- the exact "
            "way two different environments get recorded as one.")
    from crowd_nav.bayesian_dvl.intent_runtime_config import TRACKER_DEFAULTS
    for cfg_name, code_name in (
        ("tracker_sigma", "sigma"), ("tracker_persistence", "persistence"),
        ("tracker_waypoint_radius", "waypoint_radius"),
        ("missing_timeout_steps", "missing_timeout_steps"),
        ("tracker_speed_prior", "speed_prior"), ("tracker_estimate_speed", "estimate_speed"),
        ("tracker_speed_ema_alpha", "speed_ema_alpha"),
        ("tracker_speed_min", "speed_min"), ("tracker_speed_max", "speed_max"),
    ):
        declared, used = getattr(cfg, cfg_name), TRACKER_DEFAULTS[code_name]
        if type(declared) is bool or type(used) is bool:
            same = bool(declared) == bool(used)
        else:
            same = abs(float(declared) - float(used)) < 1e-12
        if not same:
            raise IntentConfigError(
                f"config {cfg_name}={declared!r} but the code uses TRACKER_DEFAULTS[{code_name!r}]={used!r}; "
                "a declared value the code does not use is a provenance lie")

    if cfg.il_episodes_standard + cfg.il_episodes_junction_crowd != cfg.il_episodes_total:
        raise IntentConfigError(
            f"IL split {cfg.il_episodes_standard}+{cfg.il_episodes_junction_crowd} != total {cfg.il_episodes_total}")
    if abs(cfg.mix_standard + cfg.mix_junction_crowd - 1.0) > 1e-9:
        raise IntentConfigError(f"online mix must sum to 1.0, got {cfg.mix_standard}+{cfg.mix_junction_crowd}")
    for name, v in (
        ("il_passes", cfg.il_passes), ("online_episodes_total", cfg.online_episodes_total),
        ("batch_size", cfg.batch_size), ("replay_capacity", cfg.replay_capacity),
        ("updates_per_episode", cfg.updates_per_episode),
        ("iqn_train_quantiles", cfg.iqn_train_quantiles), ("iqn_eval_quantiles", cfg.iqn_eval_quantiles),
        ("future_horizon", cfg.future_horizon), ("future_n_samples", cfg.future_n_samples),
        ("missing_timeout_steps", cfg.missing_timeout_steps), ("max_candidate_goals", cfg.max_candidate_goals),
        ("checkpoint_interval_episodes", cfg.checkpoint_interval_episodes),
        ("monitor_plot_interval_episodes", cfg.monitor_plot_interval_episodes),
        ("development_eval_interval_episodes", cfg.development_eval_interval_episodes),
        ("demo_capacity", cfg.demo_capacity),
        ("ranking_batch_size", cfg.ranking_batch_size),
        ("gradient_diagnostic_interval", cfg.gradient_diagnostic_interval),
    ):
        if v <= 0:
            raise IntentConfigError(f"{name} must be positive, got {v}")
    for name, v in (("learning_rate", cfg.learning_rate), ("grad_clip_norm", cfg.grad_clip_norm),
                    ("ranking_margin", cfg.ranking_margin),
                    ("tracker_sigma", cfg.tracker_sigma), ("tracker_waypoint_radius", cfg.tracker_waypoint_radius)):
        if not v > 0:
            raise IntentConfigError(f"{name} must be positive, got {v}")
    if not 0.0 <= cfg.demo_sample_ratio <= 1.0:
        raise IntentConfigError(f"demo_sample_ratio must be in [0,1], got {cfg.demo_sample_ratio}")
    if cfg.ranking_batch_size > cfg.batch_size:
        raise IntentConfigError(
            f"ranking_batch_size {cfg.ranking_batch_size} cannot exceed batch_size {cfg.batch_size}")
    # Order 2R/3R validation -- fail closed on every gate parameter.
    if not (0.0 <= cfg.rank_cap_rho <= 10.0):
        raise IntentConfigError(f"rank_cap_rho must be in [0,10], got {cfg.rank_cap_rho}")
    if cfg.audit_episodes_per_scenario <= 0:
        raise IntentConfigError(
            f"audit_episodes_per_scenario must be positive, got {cfg.audit_episodes_per_scenario}")
    if not (0.0 < cfg.rank_zero_tolerance < 1.0):
        raise IntentConfigError(f"rank_zero_tolerance must be in (0,1), got {cfg.rank_zero_tolerance}")
    if cfg.warmup_max_steps <= 0 or cfg.warmup_check_interval <= 0:
        raise IntentConfigError("warm-up steps and check interval must be positive")
    if cfg.warmup_rank_loss_max <= 0:
        raise IntentConfigError(f"warmup_rank_loss_max must be positive, got {cfg.warmup_rank_loss_max}")
    if cfg.audit_interval <= 0:
        raise IntentConfigError(f"audit_interval must be positive, got {cfg.audit_interval}")
    if not 0.0 < cfg.gamma <= 1.0:
        raise IntentConfigError(f"gamma must be in (0,1], got {cfg.gamma}")
    if not 0.0 < cfg.ema_decay < 1.0:
        raise IntentConfigError(f"ema_decay must be in (0,1), got {cfg.ema_decay}")
    if not 0.0 < cfg.tracker_persistence < 1.0:
        raise IntentConfigError(f"tracker_persistence must be in (0,1), got {cfg.tracker_persistence}")
    if not 0.0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1.0:
        raise IntentConfigError(
            f"require 0 <= epsilon_end <= epsilon_start <= 1, got {cfg.epsilon_end}/{cfg.epsilon_start}")
    if cfg.epsilon_decay_episodes <= 0:
        raise IntentConfigError(f"epsilon_decay_episodes must be positive, got {cfg.epsilon_decay_episodes}")

    # the MAX_CANDIDATE_GOALS the network was built for must match
    from crowd_nav.bayesian_dvl.intent_policy import MAX_CANDIDATE_GOALS
    if cfg.max_candidate_goals != MAX_CANDIDATE_GOALS:
        raise IntentConfigError(
            f"config max_candidate_goals {cfg.max_candidate_goals} != code's {MAX_CANDIDATE_GOALS}")

    # seed roles must be mutually exclusive AND disjoint from every frozen
    # scenario/eval block (plan section 4.2: no reusing an eval seed for training)
    from crowd_nav.bayesian_dvl.junction_scenario import (
        JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_TRAIN_SEEDS, JUNCTION_CROWD_VALIDATION_SEEDS,
        JUNCTION_HELDOUT_SEEDS, JUNCTION_TRAIN_SEEDS,
    )
    from crowd_nav.bayesian_dvl.intent_train import (
        FORMAL_EVAL_HELDOUT_SEEDS, STANDARD_DEV_DIAGNOSTIC_SEEDS,
        STANDARD_SELECTION_DEV_SEEDS, JUNCTION_SELECTION_DEV_SEEDS,
    )
    blocks = {
        "training_seeds": set(cfg.training_seeds),
        "validation_seeds": set(cfg.validation_seeds),
        "junction_train": set(JUNCTION_TRAIN_SEEDS),
        "junction_heldout": set(JUNCTION_HELDOUT_SEEDS),
        "crowd_train": set(JUNCTION_CROWD_TRAIN_SEEDS),
        "crowd_heldout": set(JUNCTION_CROWD_HELDOUT_SEEDS),
        "crowd_validation": set(JUNCTION_CROWD_VALIDATION_SEEDS),
        "formal_eval": set(FORMAL_EVAL_HELDOUT_SEEDS),
        "standard_dev_diagnostic": set(STANDARD_DEV_DIAGNOSTIC_SEEDS),
        "standard_selection_dev": set(STANDARD_SELECTION_DEV_SEEDS),
        "junction_selection_dev": set(JUNCTION_SELECTION_DEV_SEEDS),
    }
    names = sorted(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = blocks[a] & blocks[b]
            if overlap:
                raise IntentConfigError(f"seed roles {a} and {b} must be disjoint, overlap={sorted(overlap)[:5]}")
