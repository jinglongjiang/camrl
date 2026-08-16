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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from crowd_nav.bayesian_dvl.intent_runtime_config import (
    CANDIDATE_FEATURE_DIM, FROZEN_VALUES, HUMAN_SCALAR_DIM_V6, MAX_CANDIDATE_GOALS, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn


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
    # the FULL public dictionary, before any per-entry filtering
    dictionary = [np.asarray(d.position, dtype=np.float64) for d in scene.destinations]

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
                    best = min(float(np.hypot(d[0] - h.gx, d[1] - h.gy)) for d in dictionary)
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
