"""Suite-seed block bootstrap and paired statistics (guide.md 7.4, A9).

The unit of resampling is the SUITE SEED, not the episode: guide.md 7.4
"最高层以suite seed block bootstrap，整块带上该seed的全部场景" -- every
episode generated under one suite seed is resampled together as one
block, preserving whatever correlation exists across episodes sharing
initial conditions from that seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


class StatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float


# R3-6 fix (2026-08-07): widened from (0.2, 0.5, 0.8) to the full
# guide.md R3-6 grid so a validation run's calibration report covers
# the whole distribution, not just three points.
CALIBRATION_TAUS: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def quantile_calibration_metrics(
    predicted_samples: Sequence[Sequence[float]],
    realized_returns: Sequence[float],
    taus: Sequence[float] = CALIBRATION_TAUS,
    predicted_quantiles: Sequence[Sequence[float]] | None = None,
) -> Dict[str, object]:
    """Compute calibration diagnostics from real rollout predictions.

    ``predicted_samples[i]`` is the posterior-predictive return sample set
    emitted at decision ``i`` and ``realized_returns[i]`` is the discounted
    return actually observed from that same decision to episode termination.
    The function intentionally uses empirical quantiles of the complete
    sample set, so it cannot pretend that a lower-tail CVaR sample is a
    calibrated 0.5/0.8 quantile.  Empty, malformed, non-finite, or
    non-monotone inputs fail closed.
    """
    tau_tuple = tuple(float(tau) for tau in taus)
    if tau_tuple != CALIBRATION_TAUS:
        raise StatisticsError(f"calibration taus must be exactly {CALIBRATION_TAUS}, got {tau_tuple}")
    if len(predicted_samples) == 0 or len(predicted_samples) != len(realized_returns):
        raise StatisticsError("predicted sample rows and realized returns must be non-empty and equal length")

    if predicted_quantiles is not None and len(predicted_quantiles) != len(realized_returns):
        raise StatisticsError("predicted quantile rows and realized returns must have equal length")
    crossing_count = 0
    crossing_available = predicted_quantiles is not None
    coverage_hits = {tau: 0 for tau in tau_tuple}
    # R3R-5 fix (2026-08-07): guide.md's R3-6 calibration order also
    # requires empirical CRPS and per-tau pinball/quantile loss, not just
    # coverage -- coverage alone can look fine while the predictive
    # distribution is systematically too wide or too narrow.
    crps_values: List[float] = []
    pinball_losses: Dict[float, List[float]] = {tau: [] for tau in tau_tuple}
    for row_index, (row, realized) in enumerate(zip(predicted_samples, realized_returns)):
        samples = np.asarray(row, dtype=np.float64).reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all() or not np.isfinite(realized):
            raise StatisticsError("calibration inputs must be finite and every sample row must be non-empty")
        quantile_array = None
        if predicted_quantiles is None:
            quantiles = np.quantile(samples, tau_tuple)
        else:
            quantile_array = np.asarray(predicted_quantiles[row_index], dtype=np.float64)
            if quantile_array.ndim == 2:
                if quantile_array.shape[1] != len(tau_tuple) or not np.isfinite(quantile_array).all():
                    raise StatisticsError("fixed IQN quantile rows must be finite and match the tau grid")
                crossing_count += float(np.mean(np.any(np.diff(quantile_array, axis=1) < 0.0, axis=1)))
                quantiles = np.mean(quantile_array, axis=0)
            else:
                quantiles = quantile_array.reshape(-1)
            if quantiles.size != len(tau_tuple) or not np.isfinite(quantiles).all():
                raise StatisticsError("fixed IQN quantile rows must be finite and match the tau grid")
        if predicted_quantiles is not None and quantile_array.ndim != 2 and np.any(np.diff(quantiles) < 0.0):
            crossing_count += 1
        for tau, quantile in zip(tau_tuple, quantiles):
            coverage_hits[tau] += int(float(realized) <= float(quantile))
            diff = float(realized) - float(quantile)
            pinball_losses[tau].append(max(tau * diff, (tau - 1.0) * diff))
        # Empirical CRPS (Gneiting & Raftery form): E|X-y| - 0.5*E|X-X'|,
        # estimated from the finite sample set itself.
        mean_abs_error = float(np.mean(np.abs(samples - float(realized))))
        ordered = np.sort(samples)
        n_samples = ordered.size
        # O(n log n) CRPS: the first term is computed against the realized
        # value; the pairwise term has a closed form for sorted samples.
        indices = np.arange(1, n_samples + 1, dtype=np.float64)
        pairwise_mean_abs = float(2.0 * np.sum((2.0 * indices - n_samples - 1.0) * ordered) / (n_samples ** 2))
        crps_values.append(mean_abs_error - 0.5 * pairwise_mean_abs)

    n = len(realized_returns)
    coverage = {str(tau): coverage_hits[tau] / n for tau in tau_tuple}
    coverage_errors = {str(tau): abs(coverage[str(tau)] - tau) for tau in tau_tuple}
    mean_pinball_loss = {str(tau): float(np.mean(pinball_losses[tau])) for tau in tau_tuple}
    return {
        "n_decisions": n,
        # Keep the numeric field backward-compatible for historical reports;
        # the status explicitly prevents this placeholder zero from being
        # used as a promotion-gate measurement.
        "quantile_crossing_rate": crossing_count / n if crossing_available else 0.0,
        "quantile_crossing_status": "OK" if crossing_available else "UNAVAILABLE_FIXED_IQN_OUTPUTS_REQUIRED",
        "coverage": coverage,
        "coverage_errors": coverage_errors,
        "taus": list(tau_tuple),
        "mean_crps": float(np.mean(crps_values)),
        "mean_pinball_loss": mean_pinball_loss,
    }


# R3R-5 fix (2026-08-07): "profile x outcome" stratified calibration --
# guide.md 3052 "按profile x outcome分层；样本不足时明确INSUFFICIENT，
# 不得省略或合并." A group folded silently into the aggregate would hide
# exactly the failure mode R2's independent audit found (timeout episodes
# calibrating very differently from success episodes).
MIN_STRATUM_SAMPLES = 20


class CalibrationAccumulator:
    """Streaming calibration reducer used by formal evaluation.

    It keeps counters and a small deterministic audit sample, never the
    full 2048-value predictive row for every decision. This makes formal
    evaluation memory-bounded while preserving the same metrics as the
    batch helper above.
    """

    def __init__(self, taus: Sequence[float] = CALIBRATION_TAUS, audit_limit: int = 128):
        self.taus = tuple(float(t) for t in taus)
        if self.taus != CALIBRATION_TAUS:
            raise StatisticsError(f"calibration taus must be exactly {CALIBRATION_TAUS}")
        self.audit_limit = int(audit_limit)
        self.n = 0
        self.coverage_hits = {t: 0 for t in self.taus}
        self.pinball_sums = {t: 0.0 for t in self.taus}
        self.crps_sum = 0.0
        self.crossing_count = 0
        self.groups: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.audit_rows: List[Dict[str, object]] = []

    @staticmethod
    def _crps(samples: np.ndarray, realized: float) -> float:
        ordered = np.sort(samples)
        n = ordered.size
        indices = np.arange(1, n + 1, dtype=np.float64)
        pairwise_mean_abs = 2.0 * np.sum((2.0 * indices - n - 1.0) * ordered) / (n ** 2)
        return float(np.mean(np.abs(samples - realized)) - 0.5 * pairwise_mean_abs)

    def add(self, row: Dict[str, object]) -> None:
        samples = np.asarray(row["predicted_samples"], dtype=np.float64).reshape(-1)
        quantile_array = np.asarray(row["predicted_quantiles"], dtype=np.float64)
        realized = float(row["realized_return"])
        if quantile_array.ndim == 2:
            quantiles = np.mean(quantile_array, axis=0)
            crossing_fraction = float(np.mean(np.any(np.diff(quantile_array, axis=1) < 0.0, axis=1)))
        else:
            quantiles = quantile_array.reshape(-1)
            crossing_fraction = float(np.any(np.diff(quantiles) < 0.0))
        if samples.size == 0 or quantiles.size != len(self.taus) or not np.isfinite(samples).all() or not np.isfinite(quantile_array).all() or not np.isfinite(realized):
            raise StatisticsError("streaming calibration row is malformed or non-finite")
        self.n += 1
        self.crossing_count += crossing_fraction
        for tau, quantile in zip(self.taus, quantiles):
            self.coverage_hits[tau] += int(realized <= quantile)
            diff = realized - float(quantile)
            self.pinball_sums[tau] += max(tau * diff, (tau - 1.0) * diff)
        self.crps_sum += self._crps(samples, realized)
        key = (str(row["profile"]), str(row["outcome"]))
        group = self.groups.setdefault(key, {
            "n": 0, "coverage_hits": {t: 0 for t in self.taus},
            "pinball_sums": {t: 0.0 for t in self.taus}, "crps_sum": 0.0, "crossing_count": 0,
        })
        group["n"] += 1
        group["crossing_count"] += crossing_fraction
        group["crps_sum"] += self._crps(samples, realized)
        for tau, quantile in zip(self.taus, quantiles):
            group["coverage_hits"][tau] += int(realized <= quantile)
            diff = realized - float(quantile)
            group["pinball_sums"][tau] += max(tau * diff, (tau - 1.0) * diff)
        if len(self.audit_rows) < self.audit_limit:
            self.audit_rows.append(dict(row))

    @staticmethod
    def _metrics(n, coverage_hits, pinball_sums, crps_sum, crossing_count, taus):
        coverage = {str(t): coverage_hits[t] / n for t in taus}
        return {
            "status": "OK", "n_decisions": n, "quantile_crossing_rate": crossing_count / n,
            "quantile_crossing_status": "OK", "coverage": coverage,
            "coverage_errors": {str(t): abs(coverage[str(t)] - t) for t in taus},
            "taus": list(taus), "mean_crps": crps_sum / n,
            "mean_pinball_loss": {str(t): pinball_sums[t] / n for t in taus},
        }

    def summary(self) -> Dict[str, object]:
        if self.n == 0:
            return {"status": "NOT_AVAILABLE", "n_decisions": 0}
        return self._metrics(self.n, self.coverage_hits, self.pinball_sums, self.crps_sum, self.crossing_count, self.taus)

    def stratified_summary(self, expected_profiles: Sequence[str], min_samples: int = MIN_STRATUM_SAMPLES) -> Dict[str, object]:
        report = {}
        for profile in expected_profiles:
            for outcome in ("success", "collision", "timeout"):
                group = self.groups.get((str(profile), outcome))
                n = int(group["n"]) if group else 0
                key = f"{profile}/{outcome}"
                if not group or n < min_samples:
                    report[key] = {"status": "INSUFFICIENT", "n_decisions": n, "min_required": min_samples}
                else:
                    report[key] = self._metrics(n, group["coverage_hits"], group["pinball_sums"], group["crps_sum"], group["crossing_count"], self.taus)
        return report


def stratified_calibration_metrics(
    rows: Sequence[Dict[str, object]],
    taus: Sequence[float] = CALIBRATION_TAUS,
    min_samples: int = MIN_STRATUM_SAMPLES,
    expected_profiles: Sequence[str] | None = None,
) -> Dict[str, object]:
    """``rows`` are per-decision calibration records, each carrying at
    least ``profile``, ``outcome``, ``predicted_samples``,
    ``realized_return``. Returns one calibration report per (profile,
    outcome) stratum; a stratum with fewer than ``min_samples`` decisions
    is reported as ``{"status": "INSUFFICIENT", "n_decisions": <n>}``
    rather than silently merged into a larger group or dropped.
    """
    if len(rows) == 0:
        raise StatisticsError("stratified calibration requires at least one row")
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["profile"]), str(row["outcome"]))
        groups.setdefault(key, []).append(row)
    profiles = tuple(expected_profiles) if expected_profiles is not None else tuple(sorted({key[0] for key in groups}))
    outcomes = ("success", "collision", "timeout")
    report: Dict[str, object] = {}
    for profile in profiles:
        for outcome in outcomes:
            group_rows = groups.get((str(profile), outcome), [])
            stratum_key = f"{profile}/{outcome}"
            if len(group_rows) < min_samples:
                report[stratum_key] = {"status": "INSUFFICIENT", "n_decisions": len(group_rows), "min_required": min_samples}
                continue
            metrics = quantile_calibration_metrics(
                [row["predicted_samples"] for row in group_rows],
                [row["realized_return"] for row in group_rows],
                taus,
                predicted_quantiles=[row["predicted_quantiles"] for row in group_rows] if "predicted_quantiles" in group_rows[0] else None,
            )
            report[stratum_key] = {"status": "OK", **metrics}
    return report


def suite_seed_block_bootstrap(
    values_by_seed: Dict[int, Sequence[float]],
    statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Resample whole suite-seed blocks with replacement, recompute
    ``statistic_fn`` over the concatenated resampled values each time,
    and take the percentile CI. ``values_by_seed`` must be non-empty."""
    if not values_by_seed:
        raise StatisticsError("values_by_seed must not be empty")
    seeds = sorted(values_by_seed.keys())
    n_seeds = len(seeds)
    rng = np.random.default_rng(seed)

    all_values = np.concatenate([np.asarray(values_by_seed[s], dtype=np.float64) for s in seeds])
    point = float(statistic_fn(all_values))

    bootstrap_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen_seeds = rng.choice(seeds, size=n_seeds, replace=True)
        resampled = np.concatenate([np.asarray(values_by_seed[s], dtype=np.float64) for s in chosen_seeds])
        bootstrap_stats[i] = statistic_fn(resampled)

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return BootstrapResult(point=point, ci_low=ci_low, ci_high=ci_high, n_resamples=n_resamples, confidence_level=confidence_level)


def paired_difference_bootstrap(
    method_a_by_seed: Dict[int, Sequence[float]],
    method_b_by_seed: Dict[int, Sequence[float]],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """CI for statistic(A) - statistic(B), resampling the SAME suite
    seeds jointly for both methods each bootstrap iteration (required
    for a valid paired comparison -- guide.md 7.4 "paired episode
    identity必须一致"). Both dicts must share the identical seed set."""
    seeds_a, seeds_b = set(method_a_by_seed.keys()), set(method_b_by_seed.keys())
    if seeds_a != seeds_b:
        raise StatisticsError(
            f"method_a and method_b must share identical suite seeds; "
            f"only-in-a={seeds_a - seeds_b} only-in-b={seeds_b - seeds_a}"
        )
    seeds = sorted(seeds_a)
    n_seeds = len(seeds)
    rng = np.random.default_rng(seed)

    all_a = np.concatenate([np.asarray(method_a_by_seed[s], dtype=np.float64) for s in seeds])
    all_b = np.concatenate([np.asarray(method_b_by_seed[s], dtype=np.float64) for s in seeds])
    point = float(statistic_fn(all_a) - statistic_fn(all_b))

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen_seeds = rng.choice(seeds, size=n_seeds, replace=True)
        resampled_a = np.concatenate([np.asarray(method_a_by_seed[s], dtype=np.float64) for s in chosen_seeds])
        resampled_b = np.concatenate([np.asarray(method_b_by_seed[s], dtype=np.float64) for s in chosen_seeds])
        diffs[i] = statistic_fn(resampled_a) - statistic_fn(resampled_b)

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return BootstrapResult(point=point, ci_low=ci_low, ci_high=ci_high, n_resamples=n_resamples, confidence_level=confidence_level)


@dataclass(frozen=True)
class EpisodeRecord:
    """guide.md 7.4: "保存逐episode：method/scenario/profile/suite_seed/
    episode_seed/outcome/steps/time/dmin/path_length/decision_latency".
    R1 fix (independent audit B5, 2026-08-06): the previous version was
    missing elapsed_time/min_clearance/path_length/decision_latency
    entirely."""

    method: str
    scenario: str
    profile: str
    suite_seed: int
    episode_seed: int
    outcome: str  # "success" | "collision" | "timeout"
    steps: int
    elapsed_time: float = 0.0  # steps * dt, seconds
    min_clearance: float = 0.0  # minimum robot-human clearance over the episode (can be negative on collision)
    path_length: float = 0.0  # total robot path length traveled
    mean_decision_latency_ms: float = 0.0  # mean per-step decision wall-clock latency for this episode
    # R2-5 fix (2026-08-07): anti-circling diagnostics. The independent
    # diagnosis of BDVL's 66/200 common-timeout episodes found the robot
    # actively walking 3-4x the straight-line distance while drifting
    # AWAY from the goal, not standing still -- SR/CR/TR alone cannot
    # distinguish that failure mode from a slow-but-converging one, so
    # every validation run must log it going forward.
    initial_goal_distance: float = 0.0  # straight-line distance to goal at episode start
    final_goal_distance: float = 0.0  # straight-line distance to goal at episode end
    max_goal_distance: float = 0.0  # largest distance-to-goal reached at any point (>initial_goal_distance means net regression)
    mean_action_alignment: float = 0.0  # mean cos(angle) between chosen action and the goal-relative direction, in [-1, 1]
    negative_alignment_fraction: float = 0.0  # fraction of steps where the chosen action pointed away from the goal
    max_action_score: float = 0.0  # largest per-decision candidate score observed over the episode
    min_action_score: float = 0.0  # smallest per-decision candidate score observed over the episode
    score_bound_violations: int = 0  # count of decisions whose candidate score exceeded either derived bound


def check_r2_gate(
    success_rate: float, collision_rate: float, timeout_rate: float,
    score_bound_violations: int, n_episodes: int,
) -> Dict[str, object]:
    """guide.md R2-5's minimum engineering gate for entering 3-seed
    formal training (NOT a paper-quality bar -- guide.md is explicit:
    "这些是进入正式实验的最低工程门槛，不是论文胜出标准"). Pure/testable:
    takes already-computed rates, makes no I/O, no scenario knowledge."""
    reasons = []
    if n_episodes <= 0:
        raise StatisticsError("n_episodes must be positive")
    if success_rate < 0.80:
        reasons.append(f"success_rate={success_rate:.3f} < 0.80")
    if collision_rate > 0.05:
        reasons.append(f"collision_rate={collision_rate:.3f} > 0.05")
    if timeout_rate > 0.15:
        reasons.append(f"timeout_rate={timeout_rate:.3f} > 0.15")
    if score_bound_violations != 0:
        reasons.append(f"score_bound_violations={score_bound_violations} != 0")
    return {"passed": not reasons, "reasons": reasons}


def check_cvar_promotion_gate(
    score_bound_violations: int,
    quantile_crossing_rate: float,
    coverage_errors: Dict[float, float],
    cvar_timeout_rate: float,
    risk_neutral_timeout_rate: float,
    cvar_collision_rate: float,
    risk_neutral_collision_rate: float,
    coverage_error_threshold: float = 0.05,
    quantile_crossing_threshold: float = 0.01,
) -> Dict[str, object]:
    """guide.md R2-4/R3-6's CVaR-promotion gate: CVaR may only replace
    risk-neutral as the default decision rule once the value network's
    OWN quantile function is well-calibrated, never merely because it
    happened to score higher SR on one run. ``coverage_errors`` maps
    every tau in ``CALIBRATION_TAUS`` (R3-6 widened this to 0.1-0.9,
    matching guide.md's requirement that validation report the WHOLE
    grid, not just 0.2/0.5/0.8) to the absolute difference between that
    tau and the empirical fraction of realized returns falling below
    the network's predicted tau-quantile. Computing
    quantile_crossing_rate/coverage_errors from real rollouts is a
    separate, not-yet-wired analysis step (guide.md tracks this as an
    open item) -- this function only encodes the PASS/FAIL logic given
    those numbers, and is unit-tested against hand-picked values.
    ``required_taus`` is derived from ``CALIBRATION_TAUS`` (not a
    second hardcoded literal) so widening the grid can never leave this
    gate silently checking a stale subset."""
    reasons = []
    required_taus = {float(tau) for tau in CALIBRATION_TAUS}
    supplied_taus = {float(tau) for tau in coverage_errors}
    if supplied_taus != required_taus:
        reasons.append(f"coverage taus must be exactly {sorted(required_taus)}, got {sorted(supplied_taus)}")
    if not np.isfinite(float(quantile_crossing_rate)):
        reasons.append("quantile_crossing_rate is non-finite")
    elif quantile_crossing_rate < 0.0 or quantile_crossing_rate > 1.0:
        reasons.append(f"quantile_crossing_rate={quantile_crossing_rate:.4f} outside [0,1]")
    if score_bound_violations != 0:
        reasons.append(f"score_bound_violations={score_bound_violations} != 0")
    if quantile_crossing_rate > quantile_crossing_threshold:
        reasons.append(f"quantile_crossing_rate={quantile_crossing_rate:.4f} > {quantile_crossing_threshold}")
    for tau in sorted(coverage_errors):
        err = float(coverage_errors[tau])
        if not np.isfinite(err) or err < 0.0 or err > 1.0:
            reasons.append(f"coverage error at tau={tau} is invalid: {err!r}")
        elif err > coverage_error_threshold:
            reasons.append(f"coverage error at tau={tau} is {err:.4f} > {coverage_error_threshold}")
    if cvar_timeout_rate > risk_neutral_timeout_rate + 1e-9:
        reasons.append(f"cvar timeout_rate={cvar_timeout_rate:.3f} > risk-neutral {risk_neutral_timeout_rate:.3f}")
    if cvar_collision_rate > risk_neutral_collision_rate + 1e-9:
        reasons.append(f"cvar collision_rate={cvar_collision_rate:.3f} > risk-neutral {risk_neutral_collision_rate:.3f}")
    return {"passed": not reasons, "reasons": reasons}


CVaR_PROMOTION_SCHEMA = "bdvl_cvar_promotion_gate_v1"


def validate_cvar_promotion_report(
    report: Dict[str, object], *, registry_sha256: str, artifact_sha256: str,
    checkpoint_sha256: str, validation_result_sha256: str,
) -> None:
    """Validate a hash-bound promotion report before formal CVaR use."""
    required = {
        "schema": CVaR_PROMOTION_SCHEMA, "passed": True,
        "registry_sha256": registry_sha256, "artifact_sha256": artifact_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_result_sha256": validation_result_sha256,
        "quantile_crossing_source": "fixed_iqn_tau_outputs",
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise StatisticsError(f"CVaR promotion report field {key!r} is not bound to this run: expected {expected!r}, got {report.get(key)!r}")
    metrics = report.get("gate_metrics")
    if not isinstance(metrics, dict) or metrics.get("quantile_crossing_status") != "OK":
        raise StatisticsError("CVaR promotion report lacks fixed-IQN quantile calibration metrics")


def join_paired_episodes(
    records_a: Sequence[EpisodeRecord], records_b: Sequence[EpisodeRecord],
) -> List[Tuple[EpisodeRecord, EpisodeRecord]]:
    """Direct join on (scenario, profile, suite_seed, episode_seed).
    Fails closed on any missing or duplicate key (guide.md 7.4/A9:
    "episode缺失、重复、seed/profile/scenario不匹配直接失败")."""
    def _key(r: EpisodeRecord):
        return (r.scenario, r.profile, r.suite_seed, r.episode_seed)

    index_a: Dict[tuple, EpisodeRecord] = {}
    for r in records_a:
        k = _key(r)
        if k in index_a:
            raise StatisticsError(f"duplicate episode identity in records_a: {k}")
        index_a[k] = r

    index_b: Dict[tuple, EpisodeRecord] = {}
    for r in records_b:
        k = _key(r)
        if k in index_b:
            raise StatisticsError(f"duplicate episode identity in records_b: {k}")
        index_b[k] = r

    if set(index_a.keys()) != set(index_b.keys()):
        missing_in_b = set(index_a) - set(index_b)
        missing_in_a = set(index_b) - set(index_a)
        raise StatisticsError(
            f"episode identity mismatch: missing_in_b={missing_in_b} missing_in_a={missing_in_a}"
        )

    return [(index_a[k], index_b[k]) for k in sorted(index_a.keys())]
