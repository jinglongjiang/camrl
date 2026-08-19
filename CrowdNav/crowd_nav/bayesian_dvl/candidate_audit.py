"""Pre-training candidate audit: refuse to train on a broken candidate model.

The junction-crowd defect cost a full 10,500-episode paper test. Nothing in
the pipeline was watching the one thing that was wrong -- whether the public
candidate goals actually described the pedestrians they were attached to.
Every check below was violated by the shipped V1 model and would have
stopped it before a single gradient step:

  coverage        candidate endpoints sat 1.90 m (max 4.01 m) from where the
                  background humans actually went
  speed residual  the preferred-velocity model was off by 1.03 m/s, roughly
                  the pedestrians' own speed
  single candidate 27% of background observations collapsed to one candidate,
                  i.e. the network was told "this person is certainly going
                  there" about someone whose destination was unknown
  flat posterior  82.6% of the remaining background steps sat at exactly
                  uniform for the whole episode -- candidates that no
                  observation could ever discriminate
  permutation     p0 meant "left exit" for 412 humans and "right exit" for
                  68 others, and nothing in the input distinguished them

This module reads hidden goals. That is legitimate and confined here: it is
an offline audit that SCORES the public model, exactly like a test does. It
is never imported by the belief chain -- ``test_main_chain_never_reads_
hidden_goal`` covers the modules that must stay clean, and this one is not
among them.
"""

from __future__ import annotations

import argparse
import sys

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from crowd_nav.bayesian_dvl.intent_runtime_config import (
    CANDIDATE_FEATURE_DIM, FEATURE_SCHEMA_V6, FROZEN_VALUES, HUMAN_SCALAR_DIM_V6,
    MAX_CANDIDATE_GOALS, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn, scene_registry_sha256


DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_intent_bdvl.config"
DEFAULT_ENV_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "env_bayesian_dvl.config"


class IntentCLIError(ValueError):
    """Audit-side CLI error. Deliberately NOT imported from the training CLI:
    the audit is an offline preflight and must not depend on training."""


class CandidateAuditError(ValueError):
    """Raised when the candidate model is not fit to train on."""


# FROZEN budgets. Set from the measured before/after gap so they sit between
# the broken model and the fixed one: V1 measured 1.90 m mean coverage error,
# 1.03 m/s speed residual, 27% single-candidate and 82.6% flat; V2 measures
# 0.26 m, and 0% single-candidate. Not to be relaxed after seeing a result.
MAX_MEAN_COVERAGE_ERROR_M = 0.60
MAX_WORST_COVERAGE_ERROR_M = 1.20

# --- representability gate (continuous-goal scenes) -------------------
# The absolute metre budgets above were measured on the junction's crossing
# band, where the public destinations are DISCRETE. Test8's circle and square
# goals are CONTINUOUS, so the achievable error is set by how finely 8
# candidates can tile the goal region -- 0.79 m for an 8-sector circle of
# radius 4 -- and no correct candidate model can beat that. Applying the
# junction number there measured the discretization, not the model.
#
# What the check always meant is: does the PUBLIC FILTER delete the candidate
# that best describes this person? That is answered without any tunable
# number at all:
#
#   oracle_error      truth -> nearest candidate in the FULL public dictionary
#   assigned_error    truth -> nearest candidate actually GIVEN to this person
#   assignment_regret assigned_error - oracle_error
#
# Regret must be zero. A non-zero regret means the filter threw away the best
# available description of that pedestrian -- exactly the junction defect,
# stated in a form that does not depend on scene scale.
MAX_ASSIGNMENT_REGRET_M = 1e-6
MAX_MEAN_SPEED_RESIDUAL = 0.45
MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE = 0.02
MAX_PERSISTENTLY_FLAT_RATE = 0.25
PERMUTATION_TOLERANCE = 1e-6


@dataclass
class AuditResult:
    scenario: str
    n_episodes: int
    n_observations: int
    mean_coverage_error_m: float
    worst_coverage_error_m: float
    mean_speed_residual: float
    single_candidate_rate: float
    persistently_flat_rate: float
    # continuous-goal scenes: representability rather than an absolute budget
    mean_oracle_error_m: float = 0.0
    worst_oracle_error_m: float = 0.0
    worst_assignment_regret_m: float = 0.0
    n_regret_offenders: int = 0
    oracle_bound_m: Optional[float] = None
    permutation_max_delta: Optional[float] = None
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["passed"] = self.passed
        return d


def _flat(belief: np.ndarray) -> bool:
    n = len(belief)
    return n > 1 and bool(np.max(np.abs(belief - 1.0 / n)) < 1e-6)


def audit_scenario(
    episodes,
    scene,
    *,
    scenario: str,
    ambiguous_index: Optional[int] = None,
    dt: Optional[float] = None,
    max_steps: int = 60,
    coverage_mode: str = "absolute",
    oracle_bound_m: Optional[float] = None,
) -> AuditResult:
    """``coverage_mode``:

      "absolute"        junction: candidates are discrete public destinations,
                        so an absolute metre budget is meaningful.
      "representability" circle/square: goals are continuous, so the budget is
                        set by the discretization. Gate on assignment regret
                        (the filter must not drop the dictionary's best
                        candidate) plus an ANALYTIC bound on the oracle error
                        derived from the 8-candidate geometry -- never from a
                        measured result.
    """
    """``episodes`` yields ``(env, advance_hidden_state)``, where the second
    element is the per-step hook that reveals the scenario's hidden state (or
    None). Pedestrians are advanced with their own policies only -- no robot,
    because the audit asks whether the candidate model describes pedestrian
    motion, and that must not depend on which robot policy is being trained.

    Coverage is scored against each human's hidden goal; the ambiguous
    pedestrian is skipped because its goal is legitimately the shared
    waypoint until it reaches the fork, so scoring it against a final exit
    would measure the scenario's design rather than the candidate model.
    """
    if coverage_mode not in ("absolute", "representability"):
        raise CandidateAuditError(f"unknown coverage_mode {coverage_mode!r}")
    if coverage_mode == "representability" and oracle_bound_m is None:
        raise CandidateAuditError(
            "representability mode needs an ANALYTIC oracle bound; leaving it unset would turn the "
            "check into whatever the measurement happened to be")
    dt = float(FROZEN_VALUES["dt"]) if dt is None else float(dt)
    cov, resid, n_obs, n_single, n_flat_tracks, n_tracks, n_eps = [], [], 0, 0, 0, 0, 0
    oracle, regret, regret_offenders = [], [], []
    for env, advance in episodes:
        n_eps += 1
        bank = IntentBeliefBank(make_candidate_fn(scene), dt=dt,
                                speed=TRACKER_DEFAULTS["speed_prior"])
        per_track_flat: Dict[int, List[bool]] = {}
        for step in range(max_steps):
            obs = {i: (float(h.px), float(h.py)) for i, h in enumerate(env.humans)}
            bank.update(obs)
            for i, h in enumerate(env.humans):
                tracker = bank.tracker_for(i)
                belief = tracker.belief()
                n_obs += 1
                if len(tracker.candidates) == 1:
                    n_single += 1
                per_track_flat.setdefault(i, []).append(_flat(belief))
                if step > 0:
                    pos = np.array([h.px, h.py])
                    v_hat = np.zeros(2)
                    for ci in range(len(belief)):
                        nxt = np.asarray(tracker.roll_candidate_future(ci, pos, 1)[0])
                        v_hat += belief[ci] * (nxt - pos) / dt
                    resid.append(float(np.linalg.norm(v_hat - np.array([h.vx, h.vy]))))
            if step == 0:
                for i, h in enumerate(env.humans):
                    if ambiguous_index is not None and i == ambiguous_index:
                        continue
                    ends = [np.asarray(c.waypoints[-1]) for c in bank.tracker_for(i).candidates]
                    assigned = min(float(np.hypot(e[0] - h.gx, e[1] - h.gy)) for e in ends)
                    cov.append(assigned)
                    # Compare against the complete public dictionary for
                    # this observable entry, not merely the scene's base
                    # exits. Junction crossers have a public crossing
                    # band whose candidates are derived from the entry;
                    # omitting that band makes a correct assignment look
                    # artificially better than the oracle.
                    dictionary = scene.public_dictionary_for(np.array([h.px, h.py]))
                    best = min(
                        float(np.hypot(c.waypoints[-1][0] - h.gx,
                                       c.waypoints[-1][1] - h.gy))
                        for c in dictionary
                    )
                    oracle.append(best)
                    r_ = assigned - best
                    regret.append(r_)
                    if r_ > MAX_ASSIGNMENT_REGRET_M:
                        regret_offenders.append(
                            {"human": i, "assigned_error_m": round(assigned, 4),
                             "oracle_error_m": round(best, 4), "regret_m": round(r_, 4),
                             "entry": [round(float(h.px), 3), round(float(h.py), 3)]})
            if advance is not None:
                advance()
            actions = [h.act([o.get_observable_state() for o in env.humans if o is not h])
                       for h in env.humans]
            for h, a in zip(env.humans, actions):
                h.step(a)
        for i, flags in per_track_flat.items():
            n_tracks += 1
            if all(flags):
                n_flat_tracks += 1

    res = AuditResult(
        scenario=scenario, n_episodes=n_eps, n_observations=n_obs,
        mean_coverage_error_m=float(np.mean(cov)) if cov else 0.0,
        worst_coverage_error_m=float(np.max(cov)) if cov else 0.0,
        mean_speed_residual=float(np.mean(resid)) if resid else 0.0,
        single_candidate_rate=n_single / max(n_obs, 1),
        persistently_flat_rate=n_flat_tracks / max(n_tracks, 1),
        mean_oracle_error_m=float(np.mean(oracle)) if oracle else 0.0,
        worst_oracle_error_m=float(np.max(oracle)) if oracle else 0.0,
        worst_assignment_regret_m=float(np.max(regret)) if regret else 0.0,
        n_regret_offenders=len(regret_offenders),
        oracle_bound_m=oracle_bound_m,
    )
    if coverage_mode == "absolute":
        if res.mean_coverage_error_m > MAX_MEAN_COVERAGE_ERROR_M:
            res.failures.append(
                f"mean candidate-coverage error {res.mean_coverage_error_m:.2f} m > "
                f"{MAX_MEAN_COVERAGE_ERROR_M} m -- the candidates do not describe where these humans go")
        if res.worst_coverage_error_m > MAX_WORST_COVERAGE_ERROR_M:
            res.failures.append(
                f"worst candidate-coverage error {res.worst_coverage_error_m:.2f} m > {MAX_WORST_COVERAGE_ERROR_M} m")
    else:
        if res.n_regret_offenders:
            ex = regret_offenders[:3]
            res.failures.append(
                f"{res.n_regret_offenders} pedestrian(s) were denied the dictionary's best candidate "
                f"(worst regret {res.worst_assignment_regret_m:.4f} m > {MAX_ASSIGNMENT_REGRET_M:g} m); "
                f"the public filter is deleting the candidate that best describes them: {ex}")
        if res.worst_oracle_error_m > oracle_bound_m:
            res.failures.append(
                f"worst oracle error {res.worst_oracle_error_m:.3f} m exceeds the ANALYTIC "
                f"{oracle_bound_m:.3f} m bound for this scene's 8-candidate geometry -- the public "
                "dictionary itself cannot represent where these pedestrians go")
    if res.mean_speed_residual > MAX_MEAN_SPEED_RESIDUAL:
        res.failures.append(
            f"mean candidate-velocity residual {res.mean_speed_residual:.2f} m/s > {MAX_MEAN_SPEED_RESIDUAL} "
            "-- the likelihood is comparing observed motion against the wrong speed")
    if res.single_candidate_rate > MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE:
        res.failures.append(
            f"{res.single_candidate_rate:.1%} of observations collapsed to ONE candidate > "
            f"{MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE:.1%} -- certainty the scene does not have")
    if res.persistently_flat_rate > MAX_PERSISTENTLY_FLAT_RATE:
        res.failures.append(
            f"{res.persistently_flat_rate:.1%} of tracks stayed EXACTLY uniform for their whole episode > "
            f"{MAX_PERSISTENTLY_FLAT_RATE:.1%} -- candidates no observation can discriminate")
    return res


def audit_permutation_invariance(model, n_candidates: int = 5, n_perms: int = 8,
                                 seed: int = 0) -> float:
    """Max output change over candidate permutations. Zero by construction if
    the encoder pools, non-zero the moment anything reads a slot index."""
    import torch
    rng = np.random.default_rng(seed)
    S, G, F = HUMAN_SCALAR_DIM_V6, MAX_CANDIDATE_GOALS, CANDIDATE_FEATURE_DIM
    row = np.zeros(S + G * F + G, dtype=np.float32)
    row[:S] = rng.normal(scale=0.3, size=S)
    cands = rng.normal(size=(n_candidates, F))
    row[S:S + n_candidates * F] = cands.reshape(-1)
    row[S + G * F:S + G * F + n_candidates] = 1.0

    robot, action = torch.zeros(1, 7), torch.zeros(1, 5)
    tau = torch.linspace(0.05, 0.95, 8)[None]
    mask = torch.ones(1, 1, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        ref = model(robot, torch.as_tensor(row[None, None, :]), mask, action, tau)
        worst = 0.0
        for p in range(n_perms):
            order = np.random.default_rng(1000 + p).permutation(n_candidates)
            shuffled = row.copy()
            shuffled[S:S + n_candidates * F] = cands[order].reshape(-1)
            got = model(robot, torch.as_tensor(shuffled[None, None, :]), mask, action, tau)
            worst = max(worst, float((ref - got).abs().max()))
    return worst


def analytic_oracle_bound(shape: str, size: float, v_pref: float = 1.0) -> float:
    """The largest oracle error the 8-candidate dictionary can possibly leave,
    derived from GEOMETRY -- never from a measured run.

    circle: goals are the antipode of a noisy entry, so they sit within
      |noise| of the radius-``size`` circle that the 8 sector points tile.
      Worst in-plane miss is the half-chord between adjacent sectors,
      2R*sin(pi/16); CrowdSim adds up to v_pref/2 of noise per axis, i.e.
      v_pref/sqrt(2) in norm (generate_circle_crossing_human).

    square: candidates sit at x = -+w/4 and at 4 band centres in y, while the
      goal spans |gx| <= w/2 and |gy| <= w/2 in the opposite half-plane. Worst
      miss is (w/4) in x and (w/16) in y, i.e. sqrt(5)/8 * w.
    """
    if shape == "circle":
        return 2.0 * size * float(np.sin(np.pi / 16)) + v_pref / float(np.sqrt(2.0)) + 1e-6
    if shape == "square":
        return float(np.sqrt(5.0)) / 8.0 * size + 1e-6
    raise CandidateAuditError(f"no analytic oracle bound for shape {shape!r}")


def run_pretraining_audit(results: Sequence[AuditResult], permutation_delta: float,
                          out_path: Optional[Path] = None) -> dict:
    """Collect every check and REFUSE to continue if any failed. There is no
    warn-and-proceed path: the whole point is that the last run proceeded."""
    payload = {
        "checks": [r.as_dict() for r in results],
        "permutation_max_delta": permutation_delta,
        "permutation_tolerance": PERMUTATION_TOLERANCE,
        "budgets": {
            "max_assignment_regret_m": MAX_ASSIGNMENT_REGRET_M,
            "max_mean_coverage_error_m": MAX_MEAN_COVERAGE_ERROR_M,
            "max_worst_coverage_error_m": MAX_WORST_COVERAGE_ERROR_M,
            "max_mean_speed_residual": MAX_MEAN_SPEED_RESIDUAL,
            "max_unexpected_single_candidate_rate": MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE,
            "max_persistently_flat_rate": MAX_PERSISTENTLY_FLAT_RATE,
        },
    }
    failures = [f"{r.scenario}: {msg}" for r in results for msg in r.failures]
    if permutation_delta > PERMUTATION_TOLERANCE:
        failures.append(
            f"candidate permutation changed the value by {permutation_delta:.2e} > {PERMUTATION_TOLERANCE:.0e} "
            "-- the encoding still depends on candidate ORDER")
    payload["passed"] = not failures
    payload["failures"] = failures
    if out_path is not None:
        Path(out_path).write_text(json.dumps(payload, indent=2))
    if failures:
        raise CandidateAuditError(
            "pre-training candidate audit FAILED; training refused:\n  - " + "\n  - ".join(failures))
    return payload


# --------------------------------------------------------------------- #
# CLI. Order 5: the audit is a ONE-TIME offline preflight over a corpus, not
# a training subcommand -- it asks whether the public candidate goals
# describe the pedestrians they are attached to, which is decided by the
# scene rules and the feature schema before any gradient exists.
# --------------------------------------------------------------------- #

def run_candidate_audit(cfg, env_config: Path, out_path: Path, n_episodes: int = 40,
                        max_steps: int = 40) -> dict:
    from crowd_nav.bayesian_dvl.candidate_audit import (
        audit_permutation_invariance, audit_scenario, run_pretraining_audit,
    )
    from crowd_nav.bayesian_dvl.junction_scenario import (
        AMBIGUOUS_TRACK_INDEX, JunctionCrowdEpisodeConfig, build_junction_crowd_episode,
        junction_crowd_role_of_seed, maybe_reveal_crowd_exit, public_junction_crowd_scene,
    )
    from crowd_nav.bayesian_dvl.model import DistributionalValueModel

    def episodes(seeds, is_heldout):
        for seed in seeds:
            env, _robot, true_exit = build_junction_crowd_episode(
                env_config, JunctionCrowdEpisodeConfig(
                    episode_seed=seed, role=junction_crowd_role_of_seed(seed)))
            state = {"wp": False}

            def advance(env=env, true_exit=true_exit, state=state, hd=is_heldout):
                state["wp"] = maybe_reveal_crowd_exit(
                    env.humans[AMBIGUOUS_TRACK_INDEX], true_exit, state["wp"], is_heldout=hd)
            yield env, advance

    results = []
    for is_heldout, block, name in (
        (False, JUNCTION_CROWD_TRAIN_SEEDS, "junction_crowd_train"),
        (True, JUNCTION_CROWD_HELDOUT_SEEDS, "junction_crowd_heldout"),
    ):
        seeds = list(block)[:n_episodes]
        r = audit_scenario(episodes(seeds, is_heldout), public_junction_crowd_scene(is_heldout=is_heldout),
                           scenario=name, ambiguous_index=AMBIGUOUS_TRACK_INDEX, max_steps=max_steps)
        results.append(r)
        print(f"  {name:24} coverage {r.mean_coverage_error_m:.3f}/{r.worst_coverage_error_m:.3f} m | "
              f"speed residual {r.mean_speed_residual:.3f} | single {r.single_candidate_rate:.2%} | "
              f"flat {r.persistently_flat_rate:.1%} | {'PASS' if r.passed else 'FAIL'}", flush=True)

    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    delta = audit_permutation_invariance(model)
    print(f"  permutation invariance   max delta {delta:.2e}", flush=True)

    payload = run_pretraining_audit(results, delta)     # raises on any failure
    payload["identity"] = audit_identity(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload



def cmd_audit_candidates(args) -> int:
    cfg = load_intent_training_config(args.config)
    out = Path(args.out)
    print("=== pre-training candidate audit ===", flush=True)
    try:
        run_candidate_audit(cfg, args.env_config, out, n_episodes=args.episodes)
    except Exception as exc:      # CandidateAuditError, and anything the scene raises
        print(f"AUDIT FAILED -- training must not start:\n{exc}", flush=True)
        return 2
    print(f"audit PASSED -> {out}", flush=True)
    return 0


def _human_v_pref(env_config: Path) -> float:
    """The pedestrian speed the circle noise bound is derived from -- read
    from the env config, not assumed."""
    import configparser
    c = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not c.read(str(env_config)):
        raise IntentCLIError(f"env config not found: {env_config}")
    return float(c.get("humans", "v_pref"))


def run_test8_candidate_audit(cfg, env_config: Path, out_path: Path, base_seed: int,
                              episodes: int, max_steps: int = 40) -> dict:
    """Model-INDEPENDENT candidate audit over the six Test8 scenarios.

    Uses the SAME scenario builder and the SAME candidate provider the formal
    evaluator uses -- build_formal_scenario_env plus circle_scene/square_scene
    -- rather than a second copy of the geometry. A private copy is how the
    formal evaluator once ran circle_scene() for square scenarios.
    """
    from crowd_nav.bayesian_dvl.candidate_audit import audit_scenario, run_pretraining_audit
    from crowd_nav.bayesian_dvl.intent_train import (
        FORMAL_SIX_SCENARIOS, build_formal_scenario_env, paper_main_episode_seed,
    )
    from crowd_nav.bayesian_dvl.intent_evaluate import initial_state_hash
    from crowd_nav.bayesian_dvl.scene_candidates import circle_scene, square_scene

    from crowd_nav.bayesian_dvl.candidate_audit import analytic_oracle_bound

    results, seed_table = [], []
    for scenario in FORMAL_SIX_SCENARIOS:
        shape, size, _humans = FORMAL_SIX_SCENARIOS[scenario]
        scene = (circle_scene(radius=size, n_sectors=8) if shape == "circle"
                 else square_scene(width=size, n_rows=4))
        bound = analytic_oracle_bound(shape, size, v_pref=_human_v_pref(env_config))
        seeds = [paper_main_episode_seed(scenario, i, base_seed) for i in range(episodes)]

        def episodes_iter(scenario=scenario, seeds=seeds):
            for sd in seeds:
                env, robot, _shape, _size = build_formal_scenario_env(env_config, scenario)
                env.case_counter["test"] = sd % (2 ** 32 - 1)
                env.reset()
                seed_table.append({"scenario": scenario, "episode_seed": int(sd),
                                   "initial_state_hash": initial_state_hash(robot, env.humans)})
                yield env, None          # no hidden state to reveal in circle/square

        r = audit_scenario(episodes_iter(), scene, scenario=scenario, ambiguous_index=None,
                           max_steps=max_steps, coverage_mode="representability",
                           oracle_bound_m=bound)
        results.append(r)
        print(f"  {scenario:16} regret max {r.worst_assignment_regret_m:.2e} m ({r.n_regret_offenders} "
              f"offenders) | oracle {r.mean_oracle_error_m:.3f}/{r.worst_oracle_error_m:.3f} m "
              f"(bound {bound:.3f}) | assigned {r.mean_coverage_error_m:.3f} m | "
              f"speed residual {r.mean_speed_residual:.3f} | single {r.single_candidate_rate:.2%} | "
              f"flat {r.persistently_flat_rate:.1%} | {'PASS' if r.passed else 'FAIL'}", flush=True)
        if not r.passed:
            raise IntentCLIError(
                f"Test8 candidate audit FAILED on {scenario}: " + "; ".join(r.failures))

    from crowd_nav.bayesian_dvl.candidate_audit import audit_permutation_invariance
    from crowd_nav.bayesian_dvl.model import DistributionalValueModel
    torch.manual_seed(0)
    delta = audit_permutation_invariance(DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6))
    print(f"  permutation invariance   max delta {delta:.2e}", flush=True)

    payload = run_pretraining_audit(results, delta)      # raises on any failure
    payload["suite"] = "test8"
    payload["base_seed"] = int(base_seed)
    payload["episodes_per_scenario"] = int(episodes)
    payload["identity"] = audit_identity(cfg)
    payload["episodes"] = seed_table
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def cmd_audit_test8_candidates(args) -> int:
    cfg = load_intent_training_config(args.config)
    if args.base_seed == PAPER_MAIN_BASE_SEED:
        raise IntentCLIError(
            f"--base-seed {args.base_seed} is the FORMAL Test8 base; the candidate audit must not touch "
            f"formal episode identities. Use {TEST8_AUDIT_BASE_SEED}.")
    if args.base_seed == TEST8_AUDIT_BASE_SEED_RETIRED:
        raise IntentCLIError(
            f"--base-seed {args.base_seed} is RETIRED. Its run is kept as INVALID_GATE_DIAGNOSTIC -- it was "
            f"gated with the junction's absolute metre budget on a continuous goal space. Use "
            f"{TEST8_AUDIT_BASE_SEED}.")
    print(f"=== Test8 candidate audit (base seed {args.base_seed}, "
          f"{args.episodes} episodes x 6 scenarios) ===", flush=True)
    try:
        run_test8_candidate_audit(cfg, args.env_config, Path(args.out), args.base_seed, args.episodes)
    except Exception as exc:
        print(f"AUDIT FAILED -- no IL collection, no training:\n{exc}", flush=True)
        return 2
    print(f"audit PASSED -> {args.out}", flush=True)
    return 0
from crowd_nav.bayesian_dvl.junction_scenario import (  # noqa: E402
    JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_TRAIN_SEEDS,
)
from crowd_nav.bayesian_dvl.model import DistributionalValueModel  # noqa: E402
from crowd_nav.bayesian_dvl.intent_policy import HUMAN_FEATURE_DIM_V6  # noqa: E402
from crowd_nav.bayesian_dvl.intent_config import (  # noqa: E402
    IntentConfigError, load_intent_training_config,
)
from crowd_nav.bayesian_dvl.evaluation_protocol import (  # noqa: E402
    PAPER_MAIN_BASE_SEED, TEST8_AUDIT_BASE_SEED, TEST8_AUDIT_BASE_SEED_RETIRED,
)


def audit_identity(cfg) -> dict:
    """What a candidate audit is valid FOR.

    Computed HERE, from the scene rules, the scenario version, the feature
    schema and the code that produces a materialized row. Deliberately not
    the main-chain code hash: the audit asks whether the public candidate
    goals describe the pedestrians they are attached to, which nothing in
    the trainer decides. Keying it on the trainer forced a four-minute
    re-measurement of an unchanged property after every unrelated edit.
    """
    import hashlib
    import inspect
    from crowd_nav.bayesian_dvl.junction_scenario import SCENARIO_REGISTRY_ID
    from crowd_nav.bayesian_dvl.intent_train import materialize_arm_transitions

    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ("intent_tracker.py", "scene_candidates.py", "intent_policy.py",
                 "junction_scenario.py", "geometry_features.py", "normalization.py"):
        h.update((here / name).read_bytes())
    h.update(inspect.getsource(materialize_arm_transitions).encode())
    return {
        "scene_registry_sha256": scene_registry_sha256(cfg),
        "scenario_registry_id": SCENARIO_REGISTRY_ID,
        "feature_schema": FEATURE_SCHEMA_V6,
        "materialization_code_sha256": h.hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    p.add_argument("--device", type=str, default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)
    ac = sub.add_parser("audit-candidates")
    ac.add_argument("--out", type=Path, default=Path("runs/candidate_audit.json"))
    ac.add_argument("--episodes", type=int, default=40)
    a8 = sub.add_parser("audit-test8-candidates")
    a8.add_argument("--base-seed", type=int, default=TEST8_AUDIT_BASE_SEED)
    a8.add_argument("--episodes", type=int, default=100)
    a8.add_argument("--out", type=Path, default=Path("runs/v2/test8_candidate_audit.json"))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "audit-test8-candidates":
            return cmd_audit_test8_candidates(args)
        return cmd_audit_candidates(args)
    except Exception as exc:
        if type(exc).__name__ not in ("IntentCLIError", "IntentConfigError", "CandidateAuditError"):
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
