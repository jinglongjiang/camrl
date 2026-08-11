"""Formal V6 training/eval CLI for the goal-intent BDVL chain (Order C2).

Subcommands (plan section 6):

    preflight                 -- verify config/code/CUDA/action grid/scene
    train    --run-dir D --target-online-episodes N [--device cuda]
    resume   --run-dir D --target-online-episodes N [--device cuda]
    validate    --checkpoint final_ema.pth
    eval-paper  --checkpoint final_ema.pth
    eval-stress --checkpoint final_ema.pth
    ablate      --arm full|mean|cv|uniform

Hard rules enforced here:
  * EVERY formal hyperparameter comes from ``train_intent_bdvl.config``
    (intent_config.py). This module defines NO hyperparameter defaults of
    its own -- plan 2.2 point 3.
  * ``--target-online-episodes N`` is a TOTAL TARGET, not "run N more":
    resuming a run that already did 6 of 10 episodes runs 4, and resuming
    a finished run runs 0 -- plan 2.2 point 4 / section 6.
  * ``train``/``resume`` never touch formal/paper seeds -- section 6.
  * ``--device cuda`` FAILS CLOSED when CUDA is unavailable, rather than
    silently training on CPU -- plan 2.2 point 2.
  * Periodic ATOMIC checkpoints; a crash loses at most one interval --
    plan 2.2 point 15.
  * The deployment/paper artifact is ``final_ema.pth``, whose
    ``model_state_dict`` IS the EMA (not the raw weights buried in
    ``extra``) -- plan 2.2 point 14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from crowd_nav.bayesian_dvl.config import ActionGridSpec
from crowd_nav.bayesian_dvl.intent_config import (
    DEFAULT_TRAINING_CONFIG, IntentConfigError, IntentTrainingConfig, load_intent_training_config,
)
from crowd_nav.bayesian_dvl.intent_policy import (
    CHECKPOINT_SCHEMA_V6, HUMAN_FEATURE_DIM_V5, load_intent_checkpoint, save_intent_checkpoint,
)
from crowd_nav.bayesian_dvl.intent_train import (
    FORMAL_EVAL_HELDOUT_SEEDS, FORMAL_SIX_SCENARIOS, PAPER_MAIN_BASE_SEED,
    PAPER_MAIN_EPISODES_PER_SCENARIO, EMAModel, GradientRatioMonitor, IntentReplay, paper_main_jobs,
    batch_to_tensors, collect_orca_episode, run_ablation_suite, run_formal_six_scenario_evaluation,
    run_il_update, run_online_training_step, summarize_scenario_results, train_step,
)
from crowd_nav.bayesian_dvl.junction_scenario import (
    JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_IL_SEEDS, JUNCTION_CROWD_ONLINE_SEEDS,
    JUNCTION_CROWD_TRAIN_SEEDS, JUNCTION_HELDOUT_SEEDS, JUNCTION_TRAIN_SEEDS, SCENARIO_REGISTRY_ID,
    public_junction_crowd_scene,
)
from crowd_nav.bayesian_dvl.intent_evaluate import run_persistent_evaluation, summarize_csv
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
from crowd_nav.bayesian_dvl.scene_candidates import circle_scene, square_scene


class IntentCLIError(RuntimeError):
    pass


DEFAULT_ENV_CONFIG = Path("crowd_nav/configs/env_bayesian_dvl.config")
IL_CORPUS_SCHEMA = "bdvl_intent_il_corpus_v1"
STANDARD_IL_SEED_BASE = 700_001      # standard-scenario IL seeds
STANDARD_ONLINE_SEED_BASE = 800_001  # standard-scenario online seeds
FINAL_EMA_NAME = "final_ema.pth"
RUN_STATE_NAME = "run_state.json"
# C4RF.3 follow-up: TWO-SLOT rolling resume. `resume_latest.pth` used to be
# a THIRD full copy alongside slots A and B -- at the frozen
# replay_capacity=200000 that is 3 x ~1.4 GB = 4.3 GB per run (65 GB across
# 15 runs), on a host with 2.5 GB free. It is now a tiny POINTER naming the
# slot that was last written completely, so a run costs two slots, not three.
RESUME_POINTER_NAME = "resume_latest.json"
SLOT_NAMES = ("resume_slot_A.pth", "resume_slot_B.pth")


def _read_resume_pointer(run_dir: Path) -> Optional[Path]:
    ptr = Path(run_dir) / RESUME_POINTER_NAME
    if not ptr.exists():
        legacy = Path(run_dir) / "resume_latest.pth"   # pre-pointer layout
        return legacy if legacy.exists() else None
    slot = json.loads(ptr.read_text()).get("slot")
    if not slot:
        return None
    p = Path(run_dir) / slot
    return p if p.exists() else None


def _write_resume_pointer(run_dir: Path, slot_name: str, state: "RunState") -> None:
    """Written ONLY after the slot file is fully on disk, so a crash mid-save
    leaves the pointer aimed at the previous, still-valid slot."""
    ptr = Path(run_dir) / RESUME_POINTER_NAME
    tmp = ptr.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "slot": slot_name,
        "online_episodes_done": state.online_episodes_done,
        "il_passes_done": state.il_passes_done,
        "global_updates": state.global_updates,
    }, indent=2))
    os.replace(str(tmp), str(ptr))


# --------------------------------------------------------------------- #
# Scene registry hash: plan 2.2 point 13 -- must hash the REAL geometry,
# not a description string.
# --------------------------------------------------------------------- #

def scene_registry_sha256(cfg: IntentTrainingConfig) -> str:
    """Hash the ACTUAL public geometry every scenario's candidate provider
    will produce, plus the tracker parameters that turn it into a
    posterior. Real bug this fixes (plan 2.2 point 13): the previous
    version hashed a hand-written label like
    ``"junction_scene:waypoint+2_exits"``, which stays byte-identical if
    the waypoint, the exit coordinates, or the tracker's sigma change --
    exactly the drift a provenance hash exists to catch."""
    payload = {
        "scenario_registry_id": SCENARIO_REGISTRY_ID,
        "standard_circle": [list(d.position) for d in circle_scene(radius=4.0, n_sectors=8).destinations],
        "junction_crowd_train": [
            [d.name, list(d.position)] for d in public_junction_crowd_scene(is_heldout=False).destinations],
        "junction_crowd_train_junction": list(public_junction_crowd_scene(is_heldout=False).junction),
        "junction_crowd_heldout": [
            [d.name, list(d.position)] for d in public_junction_crowd_scene(is_heldout=True).destinations],
        "junction_crowd_heldout_junction": list(public_junction_crowd_scene(is_heldout=True).junction),
        "formal_square_10": [list(d.position) for d in square_scene(width=10.0, n_rows=4).destinations],
        "formal_square_14": [list(d.position) for d in square_scene(width=14.0, n_rows=4).destinations],
        "tracker": {
            "sigma": cfg.tracker_sigma, "persistence": cfg.tracker_persistence,
            "waypoint_radius": cfg.tracker_waypoint_radius, "missing_timeout_steps": cfg.missing_timeout_steps,
            "max_candidate_goals": cfg.max_candidate_goals,
            "future_horizon": cfg.future_horizon, "future_n_samples": cfg.future_n_samples,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def code_sha256() -> str:
    """Hash of the goal-intent main chain's source files, so a checkpoint
    records WHICH code produced it."""
    root = Path(__file__).resolve().parent
    names = [
        "intent_tracker.py", "scene_candidates.py", "intent_policy.py", "junction_scenario.py",
        "intent_train.py", "intent_config.py", "intent_train_cli.py", "geometry_features.py",
        "set_encoder.py", "iqn.py", "model.py", "ranking.py", "normalization.py",
    ]
    digest = hashlib.sha256()
    for name in sorted(names):
        p = root / name
        if not p.exists():
            raise IntentCLIError(f"main-chain source missing: {p}")
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    """``--device cuda`` must FAIL CLOSED when CUDA is unavailable (plan
    2.2 point 2): silently falling back to CPU is how a "4090 run" turns
    out to have been a CPU run."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise IntentCLIError(
                f"--device {requested} requested but torch.cuda.is_available() is False. "
                f"Refusing to silently fall back to CPU.")
        return torch.device(requested)
    raise IntentCLIError(f"unsupported --device {requested!r}, expected 'cpu' or 'cuda[:N]'")


# --------------------------------------------------------------------- #
# Run state: the CURSOR that makes resume a total-target operation.
# --------------------------------------------------------------------- #

@dataclass
class RunState:
    """Everything needed to answer "where is this run?" -- plan 2.2 point
    4. Without this, a resumed run re-ran IL and restarted the online loop
    (and its epsilon schedule) from index 0."""
    il_passes_done: int = 0
    online_episodes_done: int = 0
    global_updates: int = 0
    config_hash: str = ""
    code_hash: str = ""
    action_grid_hash: str = ""
    scene_registry_hash: str = ""
    seed: int = 0
    # C2.1/C5.3: a PILOT run subsamples the frozen budget for engineering
    # checks (crash/speed/VRAM/resume). Recording it means a pilot
    # checkpoint can never be silently reported as a formal result.
    is_pilot: bool = False
    pilot_overrides: str = ""
    # C4R.6: which ablation arm this run trains. Part of the run identity
    # so a mean-arm checkpoint can never be mistaken for a full-arm one.
    training_arm: str = "full"
    il_episodes_collected: int = 0
    ratio_monitor: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "RunState":
        return cls(**json.loads(blob))


def _atomic_write_bytes(path: Path, write_fn) -> None:
    """Write via a temp file + os.replace so a crash mid-write can never
    leave a truncated checkpoint where a valid one used to be."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(str(tmp), str(path))


# --------------------------------------------------------------------- #
# C4RF.2: the IL corpus is IMMUTABLE and ARM-SPECIFIC, not seed-specific.
# The 5 formal optimizer seeds for one arm differ only in initialization
# and training RNG -- they imitate the SAME demonstrations. Collecting it
# five times would cost ~4x the wall clock for byte-identical data, and
# (worse) would make "same budget, same data" an assumption rather than a
# fact. Collect once, hash it, and have every seed load the same file.
# --------------------------------------------------------------------- #

def il_corpus_path(corpus_dir: Path, arm: str, cfg: IntentTrainingConfig) -> Path:
    return Path(corpus_dir) / f"il_corpus_{arm}_{cfg.content_hash()[:12]}.pth"


def build_il_corpus(env_config_path: Path, cfg: IntentTrainingConfig, arm: str, out_path: Path,
                    n_episodes: Optional[int] = None) -> dict:
    """Collect the arm's IL corpus ONCE and write it with full identity."""
    plan = il_episode_plan(cfg)
    if n_episodes is not None:
        per = max(1, n_episodes // 2)
        plan = ([p for p in plan if p[0] == "standard"][:per]
                + [p for p in plan if p[0] == "junction_crowd"][:per])
    transitions, identities = [], []
    for scenario, ep_seed in plan:
        _assert_not_formal_seed(ep_seed)
        ep = collect_orca_episode(env_config_path, scenario, ep_seed, gamma=cfg.gamma,
                                   belief_mode=arm, n_samples=cfg.future_n_samples,
                                   horizon=cfg.future_horizon)
        transitions.extend(ep.transitions)
        identities.append([scenario, int(ep_seed), len(ep.transitions), ep.outcome])
    payload = {
        "corpus_schema": IL_CORPUS_SCHEMA,
        "training_arm": arm,
        "feature_schema": cfg.feature_schema,
        "training_contract_schema": cfg.training_contract_schema,
        "config_content_hash": cfg.content_hash(),
        "code_hash": code_sha256(),
        "gamma": cfg.gamma,
        "future_horizon": cfg.future_horizon,
        "future_n_samples": cfg.future_n_samples,
        "n_episodes": len(plan),
        "n_transitions": len(transitions),
        "episode_identities": identities,
        "transitions": transitions,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(out_path, lambda tmp: torch.save(payload, str(tmp)))
    meta = {k: v for k, v in payload.items() if k not in ("transitions", "episode_identities")}
    meta["corpus_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    meta["path"] = str(out_path)
    out_path.with_suffix(".manifest.json").write_text(json.dumps(
        {**meta, "episode_identities": identities}, indent=2, sort_keys=True))
    return meta


def load_il_corpus(path: Path, cfg: IntentTrainingConfig, arm: str) -> tuple:
    """Load + HARD-VALIDATE an IL corpus. Fails closed on any identity
    mismatch rather than training an arm on another arm's demonstrations."""
    path = Path(path)
    if not path.exists():
        raise IntentCLIError(f"IL corpus not found: {path}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("corpus_schema") != IL_CORPUS_SCHEMA:
        raise IntentCLIError(f"corpus schema {payload.get('corpus_schema')!r} != {IL_CORPUS_SCHEMA!r}")
    if payload["training_arm"] != arm:
        raise IntentCLIError(
            f"corpus was collected for arm {payload['training_arm']!r}, not {arm!r}; "
            f"an arm must never be trained on another arm's belief features")
    if payload["config_content_hash"] != cfg.content_hash():
        raise IntentCLIError("corpus was collected under a different training config")
    if payload["code_hash"] != code_sha256():
        raise IntentCLIError("corpus was collected by different main-chain code")
    meta = {k: v for k, v in payload.items() if k not in ("transitions", "episode_identities")}
    meta["corpus_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["path"] = str(path)
    return payload["transitions"], meta


@dataclass
class TrainingArtifacts:
    model: DistributionalValueModel
    optimizer: torch.optim.Optimizer
    ema: EMAModel
    buffer: IntentReplay
    tau_generator: torch.Generator
    explore_rng: np.random.Generator
    sample_rng: np.random.Generator
    state: RunState
    monitor: Optional[GradientRatioMonitor] = None
    reservoir_rng: Optional[np.random.Generator] = None
    il_corpus_ref: Optional[dict] = None


def _save_checkpoint(path: Path, art: TrainingArtifacts, cfg: IntentTrainingConfig,
                     action_grid_hash: str, scene_hash: str) -> None:
    def _write(tmp: Path):
        save_intent_checkpoint(
            art.model, str(tmp), action_grid_hash=action_grid_hash, scene_registry_sha256=scene_hash,
            optimizer=art.optimizer,
            extra={
                "ema_state_dict": art.ema.state_dict(),
                "tau_generator_state": art.tau_generator.get_state(),
                "explore_rng_state": art.explore_rng.bit_generator.state,
                "sample_rng_state": art.sample_rng.bit_generator.state,
                # C4RF.3: the immutable demo corpus is NOT embedded; only
                # its path+hash. A full checkpoint therefore holds the
                # online ring only (~2.4 GB -> a few hundred MB).
                "replay_buffer_state": art.buffer.state_dict(include_demo=False),
                "il_corpus_ref": art.il_corpus_ref,
                "reservoir_rng_state": art.reservoir_rng.bit_generator.state,
                "run_state": asdict(art.state),
                "config_hash": cfg.content_hash(),
                "config_source_sha256": cfg.source_sha256,
                "code_hash": art.state.code_hash,
            },
        )
    _atomic_write_bytes(path, _write)


def _save_milestone(path: Path, art: TrainingArtifacts, cfg: IntentTrainingConfig,
                    action_grid_hash: str, scene_hash: str) -> None:
    """C4RF.3: a periodic SMALL artifact -- model + EMA + run identity, no
    replay. Enough to evaluate or restart-from-weights at that point in
    training; a full bit-exact resume uses the rolling A/B slots."""
    ema_model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    art.ema.copy_to(ema_model)

    def _write(tmp: Path):
        save_intent_checkpoint(
            ema_model, str(tmp), action_grid_hash=action_grid_hash, scene_registry_sha256=scene_hash,
            extra={
                "artifact_role": "milestone_ema",
                "training_arm": art.state.training_arm,
                "run_state": asdict(art.state),
                "raw_model_state_dict": art.model.state_dict(),
                "config_hash": cfg.content_hash(),
                "code_hash": art.state.code_hash,
                "il_corpus_ref": art.il_corpus_ref,
            },
        )
    _atomic_write_bytes(path, _write)


def _save_final_ema(path: Path, art: TrainingArtifacts, cfg: IntentTrainingConfig,
                    action_grid_hash: str, scene_hash: str) -> None:
    """Plan 2.2 point 14: the deployment/paper artifact must have the EMA
    as its ``model_state_dict``, so an ordinary loader gets the EMA
    weights -- not the raw weights with the EMA hidden in ``extra``."""
    ema_model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    art.ema.copy_to(ema_model)

    def _write(tmp: Path):
        save_intent_checkpoint(
            ema_model, str(tmp), action_grid_hash=action_grid_hash, scene_registry_sha256=scene_hash,
            extra={
                "artifact_role": "final_ema",
                "training_arm": art.state.training_arm,
                "ema_decay": cfg.ema_decay,
                "run_state": asdict(art.state),
                "config_hash": cfg.content_hash(),
                "code_hash": art.state.code_hash,
            },
        )
    _atomic_write_bytes(path, _write)


def _build_artifacts(cfg: IntentTrainingConfig, seed: int, device: torch.device,
                     action_grid_hash: str, scene_hash: str) -> TrainingArtifacts:
    torch.manual_seed(seed)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    return TrainingArtifacts(
        model=model, optimizer=optimizer, ema=EMAModel(model, decay=cfg.ema_decay),
        buffer=IntentReplay(demo_capacity=cfg.demo_capacity, online_capacity=cfg.replay_capacity),
        tau_generator=torch.Generator().manual_seed(seed + 1),
        explore_rng=np.random.default_rng(seed + 2),
        sample_rng=np.random.default_rng(seed + 3),
        reservoir_rng=np.random.default_rng(seed + 4),
        monitor=GradientRatioMonitor(cfg.rank_gradient_ratio_min, cfg.rank_gradient_ratio_max,
                                      cfg.gradient_ratio_sustained_updates),
        state=RunState(config_hash=cfg.content_hash(), code_hash=code_sha256(),
                        action_grid_hash=action_grid_hash, scene_registry_hash=scene_hash, seed=seed),
    )


def _load_into(art: TrainingArtifacts, path: Path, cfg: IntentTrainingConfig,
               action_grid_hash: str, scene_hash: str) -> None:
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    load_intent_checkpoint(str(path), art.model, optimizer=art.optimizer,
                            expected_action_grid_hash=action_grid_hash,
                            expected_scene_registry_sha256=scene_hash)
    extra = raw.get("extra") or {}
    if extra.get("config_hash") not in (None, cfg.content_hash()):
        raise IntentCLIError(
            f"checkpoint was produced under a DIFFERENT training config "
            f"(hash {extra.get('config_hash')} != {cfg.content_hash()}); refusing to resume")
    if extra.get("code_hash") not in (None, code_sha256()):
        raise IntentCLIError(
            f"checkpoint was produced by DIFFERENT main-chain code "
            f"(hash {extra.get('code_hash')} != {code_sha256()}); refusing to resume")
    art.ema.load_state_dict(extra["ema_state_dict"])
    art.tau_generator.set_state(extra["tau_generator_state"])
    art.explore_rng.bit_generator.state = extra["explore_rng_state"]
    art.sample_rng.bit_generator.state = extra["sample_rng_state"]
    ref = extra.get("il_corpus_ref")
    if ref and not art.buffer.n_demo:
        raise IntentCLIError(
            f"checkpoint references IL corpus {ref.get('path')} (sha256 {str(ref.get('corpus_sha256'))[:12]}) "
            f"but the replay's demo side is empty -- load the corpus before resuming")
    if ref and art.il_corpus_ref and ref.get("corpus_sha256") != art.il_corpus_ref.get("corpus_sha256"):
        raise IntentCLIError(
            f"IL corpus hash mismatch: checkpoint {str(ref.get('corpus_sha256'))[:12]} != "
            f"loaded {str(art.il_corpus_ref.get('corpus_sha256'))[:12]}")
    art.buffer.load_state_dict(extra["replay_buffer_state"])
    if "reservoir_rng_state" in extra:
        art.reservoir_rng.bit_generator.state = extra["reservoir_rng_state"]
    art.state = RunState(**extra["run_state"])
    if art.state.ratio_monitor and art.monitor is not None:
        art.monitor.load_state_dict(json.loads(art.state.ratio_monitor))


# --------------------------------------------------------------------- #
# Seed schedules: deterministic functions of the GLOBAL cursor, so resume
# continues the schedule rather than restarting it.
# --------------------------------------------------------------------- #

def il_episode_plan(cfg: IntentTrainingConfig) -> List[tuple]:
    """The FROZEN IL episode identity table. C4RF.1: one DISTINCT seed per
    episode -- no cycling. Fails closed if the budget exceeds the frozen
    block rather than silently wrapping around and re-imitating the same
    initial layouts."""
    plan = [("standard", STANDARD_IL_SEED_BASE + i) for i in range(cfg.il_episodes_standard)]
    crowd = list(JUNCTION_CROWD_IL_SEEDS)
    if cfg.il_episodes_junction_crowd > len(crowd):
        raise IntentCLIError(
            f"IL budget wants {cfg.il_episodes_junction_crowd} junction_crowd episodes but the frozen "
            f"JUNCTION_CROWD_IL_SEEDS block only has {len(crowd)}; refusing to reuse layouts")
    plan += [("junction_crowd", crowd[i]) for i in range(cfg.il_episodes_junction_crowd)]
    return plan


def online_episode_at(cfg: IntentTrainingConfig, index: int) -> tuple:
    """Scenario + seed for GLOBAL online episode ``index``. Strict 1:1
    alternation implements the frozen 50/50 mix (plan section 4.1).

    C4RF.1: junction seeds come from a block DISJOINT from the IL block --
    previously online reused the exact same 200 seeds IL had already
    imitated, so the RL phase explored no new initial layouts at all."""
    if index % 2 == 0:
        return "standard", STANDARD_ONLINE_SEED_BASE + index // 2
    crowd = list(JUNCTION_CROWD_ONLINE_SEEDS)
    j = index // 2
    if j >= len(crowd):
        raise IntentCLIError(
            f"online episode {index} needs junction seed #{j} but the frozen "
            f"JUNCTION_CROWD_ONLINE_SEEDS block only has {len(crowd)}; refusing to reuse layouts")
    return "junction_crowd", crowd[j]


def _assert_not_formal_seed(seed: int) -> None:
    """Section 6: train/resume must never touch formal/paper seeds."""
    if seed in set(FORMAL_EVAL_HELDOUT_SEEDS) or seed in set(JUNCTION_CROWD_HELDOUT_SEEDS):
        raise IntentCLIError(f"training tried to use seed {seed}, which is a FORMAL/HELD-OUT evaluation seed")


# --------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------- #

def seed_inventory(cfg: IntentTrainingConfig) -> Dict[str, object]:
    """C4R.7: an explicit inventory of EVERY seed block this project's
    goal-intent chain claims, with a hard mutual-exclusion check.

    Plan 2.2 point 6: previously only the DECLARED ranges were checked
    against each other; there was no single artifact saying "these are all
    the seeds, and here is proof no role overlaps another"."""
    blocks = {
        "junction_unit_train": JUNCTION_TRAIN_SEEDS,
        "junction_unit_heldout": JUNCTION_HELDOUT_SEEDS,
        "junction_crowd_train": JUNCTION_CROWD_TRAIN_SEEDS,
        "junction_crowd_heldout": JUNCTION_CROWD_HELDOUT_SEEDS,
        "formal_eval_heldout": FORMAL_EVAL_HELDOUT_SEEDS,
        "junction_crowd_il": JUNCTION_CROWD_IL_SEEDS,
        "junction_crowd_online": JUNCTION_CROWD_ONLINE_SEEDS,
        "training_seeds": cfg.training_seeds,
        "validation_seeds": cfg.validation_seeds,
    }
    # scenarios whose seeds are DERIVED rather than enumerated
    # C4RF.1 / audit point 4: the online_standard upper bound was reported
    # one HIGHER than the schedule ever produces (the last standard episode
    # is index online_total-2, i.e. offset online_total//2 - 1).
    derived = {
        "il_standard": [STANDARD_IL_SEED_BASE, STANDARD_IL_SEED_BASE + cfg.il_episodes_standard - 1],
        "online_standard": [STANDARD_ONLINE_SEED_BASE,
                             STANDARD_ONLINE_SEED_BASE + (cfg.online_episodes_total + 1) // 2 - 1],
    }
    inventory = {
        "blocks": {k: {"n": len(v), "min": min(v), "max": max(v)} for k, v in blocks.items()},
        "derived_ranges": derived,
    }
    overlaps = []
    names = sorted(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = set(blocks[a]) & set(blocks[b])
            if common:
                overlaps.append({"a": a, "b": b, "n": len(common), "example": sorted(common)[:5]})
    # derived ranges must also not collide with any enumerated block
    for dname, (lo, hi) in derived.items():
        drange = set(range(lo, hi + 1))
        for bname, seeds in blocks.items():
            common = drange & set(seeds)
            if common:
                overlaps.append({"a": dname, "b": bname, "n": len(common), "example": sorted(common)[:5]})
    inventory["overlaps"] = overlaps
    inventory["ok"] = not overlaps
    return inventory


def cmd_preflight(args) -> int:
    cfg = load_intent_training_config(args.config)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    table = grid.build_action_table()
    scene_hash = scene_registry_sha256(cfg)
    print("=== BDVL goal-intent preflight ===")
    # Import-provenance guard. This machine has a SECOND CrowdNav checkout
    # pip-installed (soc-nav-training/CrowdNav); `import crowd_nav` resolves
    # there unless cwd is the intended repo. Launching a formal run from the
    # wrong directory would silently train a DIFFERENT source tree, and the
    # only symptom would be results that don't match the manifest.
    import crowd_nav as _cn
    resolved = Path(_cn.__file__).resolve().parent
    expected = Path(__file__).resolve().parents[1]
    print(f"crowd_nav package : {resolved}")
    if resolved != expected:
        raise IntentCLIError(
            f"import provenance mismatch: `import crowd_nav` resolves to {resolved} but this CLI lives in "
            f"{expected}. Run from the intended repo root, or fix the installed copy shadowing it.")
    print(f"python            : {platform.python_version()}   torch {torch.__version__}")
    print(f"config            : {cfg.source_path}")
    print(f"  sha256          : {cfg.source_sha256}")
    print(f"  content hash    : {cfg.content_hash()}")
    print(f"code hash         : {code_sha256()}")
    print(f"action grid       : {len(table)} actions, hash {grid.table_hash()}")
    print(f"scene registry    : {scene_hash}")
    print(f"feature schema    : {cfg.feature_schema}")
    print(f"training contract : {cfg.training_contract_schema}")
    print(f"checkpoint schema : {cfg.checkpoint_schema}")
    print(f"budget            : IL {cfg.il_episodes_total} ({cfg.il_passes} passes), online {cfg.online_episodes_total}")
    print(f"optim             : batch {cfg.batch_size}, lr {cfg.learning_rate}, gamma {cfg.gamma}, buffer {cfg.replay_capacity}")
    print(f"epsilon           : {cfg.epsilon_start} -> {cfg.epsilon_end} over {cfg.epsilon_decay_episodes} episodes")
    print(f"lambda_rank       : {cfg.lambda_rank} (margin {cfg.ranking_margin})")
    print(f"ema decay         : {cfg.ema_decay}, checkpoint every {cfg.checkpoint_interval_episodes} episodes")
    print(f"training seeds    : {list(cfg.training_seeds)}")
    cuda = torch.cuda.is_available()
    print(f"cuda available    : {cuda}")
    if cuda:
        print(f"  device 0        : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  memory          : {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    if args.device != "cpu":
        dev = resolve_device(args.device)
        probe = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5).to(dev)
        p = next(probe.parameters())
        print(f"device probe      : parameters really on {p.device}")
        if p.device.type != dev.type:
            raise IntentCLIError(f"model parameters landed on {p.device}, expected {dev}")
    inv = seed_inventory(cfg)
    print("seed inventory    :")
    for name, meta in sorted(inv["blocks"].items()):
        print(f"  {name:<24} n={meta['n']:<6d} [{meta['min']}, {meta['max']}]")
    for name, (lo, hi) in sorted(inv["derived_ranges"].items()):
        print(f"  {name:<24} derived  [{lo}, {hi}]")
    if not inv["ok"]:
        raise IntentCLIError(f"seed roles OVERLAP -- fail closed: {inv['overlaps']}")
    print("  -> all seed roles mutually exclusive")

    try:
        import shutil
        usage = shutil.disk_usage(str(Path(args.run_dir).parent if args.run_dir else "."))
        print(f"disk free         : {usage.free / 1e9:.1f} GB")
        if usage.free < 5e9:
            print(f"  WARNING: under 5 GB free; a long multi-seed run writes many checkpoints")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"disk free         : unavailable ({exc})")

    if args.write_inventory:
        Path(args.write_inventory).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_inventory).write_text(json.dumps({
            "seed_inventory": inv,
            "config_content_hash": cfg.content_hash(),
            "config_sha256": cfg.source_sha256,
            "code_hash": code_sha256(),
            "action_grid_hash": grid.table_hash(),
            "scene_registry_hash": scene_hash,
        }, indent=2, sort_keys=True))
        print(f"wrote inventory   : {args.write_inventory}")
    print("preflight OK")
    return 0


def cmd_train(args, resume: bool = False) -> int:
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    action_grid_hash = grid.table_hash()
    scene_hash = scene_registry_sha256(cfg)

    seed = args.seed if args.seed is not None else cfg.training_seeds[0]
    if seed not in cfg.training_seeds:
        raise IntentCLIError(f"--seed {seed} is not one of the frozen training seeds {list(cfg.training_seeds)}")

    overrides = {}
    if args.il_episodes is not None:
        overrides["il_episodes"] = args.il_episodes
    if args.il_passes is not None:
        overrides["il_passes"] = args.il_passes
    if args.target_online_episodes is not None and args.target_online_episodes != cfg.online_episodes_total:
        overrides["target_online_episodes"] = args.target_online_episodes

    art = _build_artifacts(cfg, seed, device, action_grid_hash, scene_hash)
    art.state.training_arm = args.training_arm
    if overrides:
        art.state.is_pilot = True
        art.state.pilot_overrides = json.dumps(overrides, sort_keys=True)
        print(f"PILOT RUN -- frozen budget overridden: {art.state.pilot_overrides}. "
              f"Results are engineering checks only, never formal results.")
    resume_path = _read_resume_pointer(run_dir)
    if resume and resume_path is None:
        raise IntentCLIError(f"--resume requested but no valid resume slot exists under {run_dir}")
    slot_index = {"n": 0}

    def _save_rolling() -> None:
        """Alternate A/B, then move the pointer. Two files total."""
        name = SLOT_NAMES[slot_index["n"] % 2]
        slot_index["n"] += 1
        _save_checkpoint(run_dir / name, art, cfg, action_grid_hash, scene_hash)
        _write_resume_pointer(run_dir, name, art.state)
    # NOTE: the actual _load_into happens AFTER the IL corpus is loaded --
    # C4RF.3 checkpoints omit the demo corpus, so the replay's demo side
    # must be populated from the immutable artifact first.

    target_online = cfg.online_episodes_total if args.target_online_episodes is None else args.target_online_episodes
    if target_online > cfg.online_episodes_total:
        raise IntentCLIError(
            f"--target-online-episodes {target_online} exceeds the frozen budget {cfg.online_episodes_total}")

    arm = art.state.training_arm
    diag_every = max(1, cfg.gradient_diagnostic_interval)

    def _sync_monitor() -> None:
        if art.monitor is not None:
            art.state.ratio_monitor = json.dumps(art.monitor.state_dict())

    def _observe(result) -> None:
        """C4R.3: feed measured ratios to the sustained-window gate.

        C4RF.4 fix (real bug found by audit): the abort path used to call
        ``_save_checkpoint`` BEFORE refreshing ``RunState.ratio_monitor``
        from the live monitor, so the checkpoint written at abort time
        carried a STALE consecutive-out-of-range count -- exactly the
        number the gate exists to preserve. Resuming from it would restart
        the streak from an older value and could sail past the window.
        The monitor state is now fixed into RunState FIRST, then saved."""
        if result is not None and result.ratio_measured and art.monitor is not None:
            reason = art.monitor.observe(result.gradient_ratio)
            _sync_monitor()
            if reason:
                _save_rolling()
                raise IntentCLIError(f"ABORT (gradient ratio gate): {reason}")

    # ---------------- IL phase ----------------
    # C4RF.2: the arm's IL corpus is collected ONCE into an immutable,
    # hashed artifact and SHARED by all 5 optimizer seeds of that arm.
    # C4R.1: updates are batch_size mini-batches drawn from the CPU-side
    # reservoir, never one device-resident full-corpus batch.
    total_il_passes = cfg.il_passes if args.il_passes is None else args.il_passes
    corpus_dir = Path(args.il_corpus_dir) if args.il_corpus_dir else run_dir.parent / "il_corpus"
    corpus_file = Path(args.il_corpus) if args.il_corpus else il_corpus_path(corpus_dir, arm, cfg)
    if args.il_episodes is not None:
        corpus_file = corpus_file.with_name(corpus_file.stem + f"_pilot{args.il_episodes}.pth")

    if not corpus_file.exists():
        print(f"collecting IL corpus for arm={arm} -> {corpus_file}")
        meta = build_il_corpus(args.env_config, cfg, arm, corpus_file, n_episodes=args.il_episodes)
        print(f"  {meta['n_episodes']} episodes, {meta['n_transitions']} transitions, "
              f"sha256 {meta['corpus_sha256'][:12]}")
    demo_transitions, art.il_corpus_ref = load_il_corpus(corpus_file, cfg, arm)
    art.buffer.add_demo(demo_transitions, art.reservoir_rng)
    print(f"IL corpus: {art.il_corpus_ref['n_episodes']} episodes -> {art.buffer.n_demo} demo "
          f"transitions in reservoir (capacity {cfg.demo_capacity}, "
          f"sha256 {art.il_corpus_ref['corpus_sha256'][:12]})")
    art.state.il_episodes_collected = int(art.il_corpus_ref["n_episodes"])

    if resume:
        # the demo side is now populated, so the checkpoint (which omits
        # the corpus) can be applied
        _load_into(art, resume_path, cfg, action_grid_hash, scene_hash)
        art.model.to(device)
        art.ema.to(device)   # keep the EMA shadow co-located with the model
        if art.state.training_arm != args.training_arm:
            raise IntentCLIError(
                f"this run was trained as arm {art.state.training_arm!r} but --training-arm "
                f"{args.training_arm!r} was requested; refusing to mix arms in one run")
        print(f"resumed: IL passes {art.state.il_passes_done}/{total_il_passes}, online "
              f"{art.state.online_episodes_done}/{cfg.online_episodes_total}, "
              f"updates {art.state.global_updates}")

    if art.state.il_passes_done < total_il_passes:
        result = None
        while art.state.il_passes_done < total_il_passes:
            measure = (art.state.global_updates % diag_every == 0)
            result = run_il_update(
                art.model, art.optimizer, art.buffer, cfg.batch_size, art.sample_rng, art.tau_generator,
                n_taus=cfg.iqn_train_quantiles, ranking_margin=cfg.ranking_margin,
                lambda_rank=cfg.lambda_rank, ranking_batch_size=cfg.ranking_batch_size,
                device=str(device), grad_clip_norm=cfg.grad_clip_norm, measure_gradient_ratio=measure)
            art.ema.update(art.model)
            art.state.il_passes_done += 1
            art.state.global_updates += 1
            _observe(result)
            if art.state.il_passes_done % max(1, total_il_passes // 10) == 0:
                print(f"IL[{art.state.il_passes_done}/{total_il_passes}] loss={result.loss:.4f} "
                      f"(mc={result.mc_loss:.4f} rank={result.rank_loss:.4f}) "
                      f"|g|={result.grad_norm_preclip:.2f}{' CLIPPED' if result.clipped else ''}"
                      + (f" ratio={result.gradient_ratio:.3f}" if result.ratio_measured else ""))
        _sync_monitor()
        _save_rolling()

    # ---------------- online RL phase ----------------
    while art.state.online_episodes_done < target_online:
        i = art.state.online_episodes_done
        scenario, ep_seed = online_episode_at(cfg, i)
        _assert_not_formal_seed(ep_seed)
        epsilon = cfg.epsilon_at(i)
        measure = (art.state.global_updates % diag_every == 0)
        result = run_online_training_step(
            args.env_config, art.model, art.optimizer, action_table, scenario, ep_seed, epsilon,
            art.buffer, cfg.batch_size, art.explore_rng, art.sample_rng, art.tau_generator,
            gamma=cfg.gamma, n_taus=cfg.iqn_train_quantiles, ranking_margin=cfg.ranking_margin,
            lambda_rank=cfg.lambda_rank, ranking_batch_size=cfg.ranking_batch_size,
            n_samples=cfg.future_n_samples, horizon=cfg.future_horizon, device=str(device),
            demo_ratio=cfg.demo_sample_ratio, grad_clip_norm=cfg.grad_clip_norm,
            measure_gradient_ratio=measure, updates=cfg.updates_per_episode, belief_mode=arm)
        art.ema.update(art.model)
        art.state.global_updates += cfg.updates_per_episode
        art.state.online_episodes_done += 1
        _observe(result)
        done = art.state.online_episodes_done
        if done % max(1, min(cfg.checkpoint_interval_episodes, max(1, target_online // 10))) == 0:
            print(f"online[{done}/{target_online}] {scenario} eps={epsilon:.3f} loss={result.loss:.4f} "
                  f"(mc={result.mc_loss:.4f} rank={result.rank_loss:.4f}) "
                  f"demo/online={result.n_demo}/{result.n_online} "
                  f"|g|={result.grad_norm_preclip:.2f}{' CLIPPED' if result.clipped else ''}"
                  + (f" ratio={result.gradient_ratio:.3f}" if result.ratio_measured else ""))
        if done % cfg.checkpoint_interval_episodes == 0:
            _sync_monitor()
            _save_rolling()
            # milestones keep only the SMALL artifacts (model + EMA +
            # manifest), never another copy of the replay.
            _save_milestone(run_dir / f"milestone_ep{done:06d}.pth", art, cfg,
                             action_grid_hash, scene_hash)

    _sync_monitor()
    _save_rolling()
    (run_dir / RUN_STATE_NAME).write_text(art.state.to_json())
    _save_final_ema(run_dir / FINAL_EMA_NAME, art, cfg, action_grid_hash, scene_hash)
    m = art.monitor
    print(f"done[arm={arm}]: IL {art.state.il_passes_done}, online "
          f"{art.state.online_episodes_done}/{cfg.online_episodes_total}, updates {art.state.global_updates}")
    if m is not None and m.n_measured:
        print(f"gradient ratio: measured {m.n_measured}x, out-of-range {m.n_out_of_range}, "
              f"current streak {m.consecutive_out_of_range}/{m.sustained_updates}")
    print(f"wrote {_read_resume_pointer(run_dir)} (2-slot rolling) and {run_dir / FINAL_EMA_NAME}")
    return 0


def _load_eval_model(checkpoint: Path, cfg: IntentTrainingConfig, device: torch.device,
                     action_grid_hash: str, scene_hash: str) -> DistributionalValueModel:
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    load_intent_checkpoint(str(checkpoint), model, expected_action_grid_hash=action_grid_hash,
                            expected_scene_registry_sha256=scene_hash)
    return model.to(device)


def _print_table(title: str, results: Dict[str, List]) -> None:
    print(f"--- {title} ---")
    width = max(len(k) for k in results)
    for name, eps in results.items():
        s = summarize_scenario_results(eps)
        print(f"  {name:<{width}}  n={int(s['n']):<4d} success={s['success_rate']:.3f} "
              f"collision={s['collision_rate']:.3f} timeout={s['timeout_rate']:.3f} "
              f"mean_steps={s['mean_steps']:.1f}")


def _provenance(cfg, args, scene_hash: str, action_grid_hash: str) -> dict:
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "config_path": cfg.source_path,
        "config_sha256": cfg.source_sha256,
        "config_content_hash": cfg.content_hash(),
        "code_hash": code_sha256(),
        "action_grid_hash": action_grid_hash,
        "scene_registry_hash": scene_hash,
        "feature_schema": cfg.feature_schema,
        "training_contract_schema": cfg.training_contract_schema,
    }


def _print_summary(title: str, csv_path: Path) -> None:
    print(f"--- {title} ---")
    summary = summarize_csv(csv_path)
    width = max((len(k) for k in summary), default=8)
    for scenario, s in sorted(summary.items()):
        print(f"  {scenario:<{width}}  n={int(s['n']):<4d} SR={s['success_rate']:.3f} CR={s['collision_rate']:.3f} "
              f"TR={s['timeout_rate']:.3f} time={s['mean_nav_time']:.1f}s path_ratio={s['mean_path_ratio']:.2f} "
              f"clr={s['mean_min_clearance']:.3f} discomfort={s['mean_discomfort_frequency']:.3f}")
    print(f"  rows: {csv_path}")


def cmd_eval(args, which: str) -> int:
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    scene_hash = scene_registry_sha256(cfg)
    model = _load_eval_model(Path(args.checkpoint), cfg, device, grid.table_hash(), scene_hash)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "results"
    prov = _provenance(cfg, args, scene_hash, grid.table_hash())
    common = dict(n_samples=cfg.future_n_samples, horizon=cfg.future_horizon, device=str(device),
                  provenance=prov, resume=not args.no_resume)

    if which == "validate":
        seeds = list(cfg.validation_seeds)[: args.episodes] if args.episodes else list(cfg.validation_seeds)
        jobs = [("standard", s, False) for s in seeds]
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "paper_main",
                                              jobs, method="intent_bdvl_validation", **common)
        _print_summary("validation (training health only -- never checkpoint selection)", csv_path)
        return 0

    if which == "eval-paper":
        # C4RF.5: the paper-main table must be episode-for-episode pairable
        # with the Mamba-VL / SARL / LSTM numbers, which come from test8.py.
        # That means test8's OWN identities -- base seed 42, 500 episodes
        # per scenario, seed = (42 + case_id*1_000_003 + ep) % (2**31-1) --
        # not a private held-out block of ours.
        n_ep = args.episodes or PAPER_MAIN_EPISODES_PER_SCENARIO
        jobs = paper_main_jobs(episodes_per_scenario=n_ep, base_seed=args.base_seed)
        prov.update({"protocol": "test8_paper_main", "base_seed": args.base_seed,
                     "episodes_per_scenario": n_ep,
                     "episode_seed_formula": "(base_seed + case_id*1_000_003 + ep) % (2**31-1)"})
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "paper_main",
                                              jobs, **common)
        _print_summary(f"paper-main: test8 protocol (base_seed={args.base_seed}, {n_ep} eps/scenario)", csv_path)
        return 0

    if which == "eval-stress":
        seeds = (list(JUNCTION_CROWD_HELDOUT_SEEDS)[: args.episodes] if args.episodes
                 else list(JUNCTION_CROWD_HELDOUT_SEEDS))
        jobs = [("junction_crowd", s, True) for s in seeds]
        csv_path = run_persistent_evaluation(args.env_config, model, action_table, out_dir, "heldout_junction",
                                              jobs, **common)
        _print_summary("held-out junction-crowd stress (shifted speeds + wider fork)", csv_path)
        return 0

    raise IntentCLIError(f"unknown eval kind {which!r}")


def cmd_ablate(args) -> int:
    cfg = load_intent_training_config(args.config)
    device = resolve_device(args.device)
    grid = ActionGridSpec.from_env_config(str(args.env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)
    scene_hash = scene_registry_sha256(cfg)
    model = _load_eval_model(Path(args.checkpoint), cfg, device, grid.table_hash(), scene_hash)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "results"
    seeds = (list(JUNCTION_CROWD_HELDOUT_SEEDS)[: args.episodes] if args.episodes
             else list(JUNCTION_CROWD_HELDOUT_SEEDS))
    jobs = [("junction_crowd", s, True) for s in seeds]
    arms = (args.arm,) if args.arm else ("full", "mean", "cv", "uniform")
    print("NOTE: this is a FEATURE INTERVENTION on ONE checkpoint -- a mechanism diagnosis.")
    print("      The paper's main ablation requires SEPARATELY TRAINING full/mean/cv under an")
    print("      identical budget (plan section 3.2); 'uniform' is a supplementary control only.")
    for arm in arms:
        csv_path = run_persistent_evaluation(
            args.env_config, model, action_table, out_dir, "ablation", jobs, belief_mode=arm,
            n_samples=cfg.future_n_samples, horizon=cfg.future_horizon, device=str(device),
            provenance=_provenance(cfg, args, scene_hash, grid.table_hash()), resume=not args.no_resume)
        _print_summary(f"ablation arm: {arm}", csv_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BDVL goal-intent (V6) formal training and evaluation.")
    p.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    p.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda[:N]'; cuda fails closed if unavailable")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("preflight")
    pf.add_argument("--run-dir", type=Path, default=None)
    pf.add_argument("--write-inventory", type=Path, default=None,
                    help="write the seed inventory + all provenance hashes to this JSON path")

    for name in ("train", "resume"):
        t = sub.add_parser(name)
        t.add_argument("--run-dir", type=Path, required=True)
        t.add_argument("--target-online-episodes", type=int, default=None,
                       help="TOTAL target, not 'run N more'")
        t.add_argument("--seed", type=int, default=None, help="must be one of the frozen training seeds")
        t.add_argument("--il-episodes", type=int, default=None, help="PILOT ONLY: subsample the frozen IL plan")
        t.add_argument("--il-passes", type=int, default=None, help="PILOT ONLY: override the frozen IL pass count")
        t.add_argument("--il-corpus", type=Path, default=None, help="explicit IL corpus artifact path")
        t.add_argument("--il-corpus-dir", type=Path, default=None,
                       help="directory holding per-arm IL corpora shared across optimizer seeds")
        t.add_argument("--training-arm", type=str, default="full", choices=["full", "mean", "cv"],
                       help="C4R.6: train an INDEPENDENT ablation arm from scratch. "
                            "'uniform' is a supplementary inference-time control only and is not trained.")

    for name in ("validate", "eval-paper", "eval-stress"):
        e = sub.add_parser(name)
        e.add_argument("--checkpoint", type=Path, required=True)
        e.add_argument("--episodes", type=int, default=None)
        e.add_argument("--out-dir", type=Path, default=None)
        e.add_argument("--no-resume", action="store_true")
        e.add_argument("--base-seed", type=int, default=PAPER_MAIN_BASE_SEED,
                       help="eval-paper only: test8.py's base seed (default 42)")

    a = sub.add_parser("ablate")
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--arm", type=str, default=None, choices=["full", "mean", "cv", "uniform"])
    a.add_argument("--episodes", type=int, default=None)
    a.add_argument("--out-dir", type=Path, default=None)
    a.add_argument("--no-resume", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "preflight":
            return cmd_preflight(args)
        if args.cmd == "train":
            return cmd_train(args, resume=False)
        if args.cmd == "resume":
            return cmd_train(args, resume=True)
        if args.cmd in ("validate", "eval-paper", "eval-stress"):
            return cmd_eval(args, args.cmd)
        if args.cmd == "ablate":
            return cmd_ablate(args)
    except (IntentCLIError, IntentConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise IntentCLIError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
