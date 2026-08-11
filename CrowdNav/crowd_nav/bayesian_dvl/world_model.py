"""Semantic Bayesian Kinematic HMM (SBK-HMM), guide.md section 4 / A3.

Five FIXED semantic modes -- never searched, never grown by NLL:

    0 CV      near-zero longitudinal accel and angular rate
    1 ACC     positive longitudinal accel
    2 DECEL   negative longitudinal accel
    3 TURN_L  positive angular rate
    4 TURN_R  negative angular rate

Only observable motion quantities are used (speed, longitudinal accel,
angular rate) -- no true goal, label, or future information (guide.md
4.2). Mode transition is a sticky categorical matrix with a Dirichlet
posterior; mode emission is a sign-constrained multivariate Student-t
over ``[a_parallel, omega]``, the posterior predictive of a per-mode
Normal-Inverse-Wishart (NIW) posterior (R1 fix, independent audit B1,
2026-08-06: the previous version stored only a Gaussian point mean/
covariance with a fixed shrinkage floor -- not the NIW/Student-t family
guide.md 4.3 actually specifies, and it could sample an
acceleration/turn sign inconsistent with the selected mode).

This is approximate Bayesian parameter learning via EM on responsibility-
weighted sufficient statistics (guide.md 4.3): the artifact is a fitted
point in the posterior-predictive family, not claimed as an exact
posterior. Exact Bayes recursion is only the online per-track filter in
belief.py, applied AFTER this artifact is frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

N_MODES = 5
MODE_NAMES: Tuple[str, ...] = ("CV", "ACC", "DECEL", "TURN_L", "TURN_R")
CV, ACC, DECEL, TURN_L, TURN_R = range(N_MODES)

# Sign constraints on the emission mean for [a_parallel, omega], guide.md 4.1/4.3.
# None means "no constraint on that dimension for this mode".
_MODE_SIGN_CONSTRAINTS: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    CV: (0, 0),        # near-zero both (soft: enforced via prior, not hard-clip)
    ACC: (+1, None),
    DECEL: (-1, None),
    TURN_L: (None, +1),
    TURN_R: (None, -1),
}

MIN_SPEED_FOR_HEADING = 0.05  # below this, heading is unstable -> feature marked missing
MAX_SIGN_REJECTION_SAMPLES = 200  # bounded rejection sampling budget (guide.md B1 repair item 3)


class WorldModelError(ValueError):
    pass


@dataclass
class Track:
    """One human's raw kinematic history for fitting: positions only,
    features are derived (guide.md 4.2), never read from a label."""

    positions: np.ndarray  # [T, 2]
    dt: float

    def feature_runs(self) -> List[np.ndarray]:
        """Return a list of CONTIGUOUS valid-feature runs, each [n_i, 2]
        of [a_parallel, omega]. R1 fix (independent audit B1 repair item
        5, 2026-08-06): the previous ``features()`` returned a single
        flat (features, valid_mask) pair; fitting code then did
        ``features[valid]``, which silently COMPRESSES away invalid
        frames -- if frames 5 and 8 are the only valid ones in a track,
        that compression places them adjacent in the fitted sequence,
        so the HMM would treat a 3-step temporal gap as one Markov
        transition. Splitting into contiguous runs and fitting each run
        as its own independent sequence (guide.md's forward-backward
        already operates per-sequence) avoids this entirely."""
        pos = self.positions
        T = pos.shape[0]
        if T < 3:
            return []
        vel = (pos[1:] - pos[:-1]) / self.dt  # [T-1, 2]
        speed = np.linalg.norm(vel, axis=1)  # [T-1]
        heading = np.arctan2(vel[:, 1], vel[:, 0])  # [T-1]

        a_parallel = (speed[1:] - speed[:-1]) / self.dt  # [T-2]
        dheading = heading[1:] - heading[:-1]
        dheading = np.mod(dheading + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]
        omega = dheading / self.dt  # [T-2]

        valid = (speed[:-1] > MIN_SPEED_FOR_HEADING) & (speed[1:] > MIN_SPEED_FOR_HEADING)
        features = np.stack([a_parallel, omega], axis=1)

        runs: List[np.ndarray] = []
        start = None
        for i in range(len(valid)):
            if valid[i] and start is None:
                start = i
            elif not valid[i] and start is not None:
                runs.append(features[start:i])
                start = None
        if start is not None:
            runs.append(features[start:])
        return [r for r in runs if r.shape[0] > 0]


# --------------------------------------------------------------------- #
# Normal-Inverse-Wishart posterior + multivariate Student-t predictive
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class NIWParams:
    """One mode's NIW posterior over a 2D Gaussian's (mean, covariance).
    guide.md 4.3's ``(mu, kappa, nu, Psi)``."""

    mu: np.ndarray  # [2]
    kappa: float
    nu: float
    psi: np.ndarray  # [2,2]

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise WorldModelError(f"NIW kappa must be positive, got {self.kappa}")
        d = 2
        if self.nu <= d - 1:
            raise WorldModelError(f"NIW nu must be > d-1={d-1}, got {self.nu}")
        try:
            np.linalg.cholesky(self.psi)
        except np.linalg.LinAlgError as exc:
            raise WorldModelError(f"NIW psi is not positive-definite: {exc}") from exc

    def predictive_dof_location_scale(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """The posterior predictive of an NIW is a multivariate
        Student-t with dof = nu-d+1, location = mu,
        scale = Psi*(kappa+1)/(kappa*(nu-d+1))."""
        d = 2
        dof = self.nu - d + 1
        scale = self.psi * (self.kappa + 1) / (self.kappa * dof)
        return dof, self.mu, scale

    def predictive_covariance(self) -> np.ndarray:
        """Actual covariance of the Student-t predictive (dof/(dof-2) *
        scale, defined only for dof>2); falls back to the scale matrix
        itself for dof in (0,2] (heavy-tailed regime where the
        covariance is formally infinite/undefined) -- used only as a
        network FEATURE (guide.md 5.1's "covariance[3]"), not as a
        probability itself, so a finite proxy is acceptable there and
        is clearly documented rather than silently wrong."""
        dof, _, scale = self.predictive_dof_location_scale()
        if dof > 2:
            return scale * dof / (dof - 2)
        return scale

    def log_predictive_prob(self, x: np.ndarray) -> float:
        dof, mu, scale = self.predictive_dof_location_scale()
        return _multivariate_t_log_pdf(x[None, :], dof, mu, scale)[0]

    def sample_predictive(self, rng: np.random.Generator) -> np.ndarray:
        dof, mu, scale = self.predictive_dof_location_scale()
        return _sample_multivariate_t(rng, dof, mu, scale, n=1)[0]


def _multivariate_t_log_pdf(x: np.ndarray, dof: float, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """log p(x) for x: [T,2] under a multivariate Student-t(dof, mu, scale)."""
    d = 2
    diff = x - mu
    inv_scale = np.linalg.inv(scale)
    det = np.linalg.det(scale)
    quad = np.einsum("ti,ij,tj->t", diff, inv_scale, diff)
    log_norm = (
        gammaln((dof + d) / 2.0) - gammaln(dof / 2.0)
        - 0.5 * d * np.log(dof * np.pi) - 0.5 * np.log(det)
    )
    return log_norm - 0.5 * (dof + d) * np.log1p(quad / dof)


def _sample_multivariate_t(rng: np.random.Generator, dof: float, mu: np.ndarray, scale: np.ndarray, n: int) -> np.ndarray:
    """Standard construction: x = mu + z / sqrt(w/dof), z~N(0,scale), w~chi2(dof)."""
    z = rng.multivariate_normal(np.zeros(2), scale, size=n)
    w = rng.chisquare(dof, size=n)
    return mu[None, :] + z / np.sqrt(w / dof)[:, None]


def sample_sign_truncated_predictive(rng: np.random.Generator, niw: NIWParams, mode: int) -> np.ndarray:
    """Bounded rejection sampling (guide.md B1 repair item 3): draw from
    the Student-t predictive, discard samples that violate the mode's
    frozen sign constraint, hard-fail (not silently fall back) if the
    budget is exhausted -- this should be rare in practice since the
    posterior mean is itself sign-projected during fitting, but a wide
    posterior can still put non-trivial mass on the wrong side."""
    a_sign, w_sign = _MODE_SIGN_CONSTRAINTS[mode]
    for _ in range(MAX_SIGN_REJECTION_SAMPLES):
        sample = niw.sample_predictive(rng)
        a, w = sample
        if a_sign == 1 and a <= 0:
            continue
        if a_sign == -1 and a >= 0:
            continue
        if a_sign == 0 and abs(a) > 3.0:  # near-zero region: reject only extreme outliers
            continue
        if w_sign == 1 and w <= 0:
            continue
        if w_sign == -1 and w >= 0:
            continue
        if w_sign == 0 and abs(w) > 3.0:
            continue
        return sample
    raise WorldModelError(
        f"sign-truncated rejection sampling exhausted {MAX_SIGN_REJECTION_SAMPLES} attempts "
        f"for mode {MODE_NAMES[mode]} -- posterior is inconsistent with its own sign constraint"
    )


@dataclass
class SBKHMMArtifact:
    """Frozen fitted parameters. Immutable after construction; use
    ``fit_sbk_hmm`` to produce one, never hand-edit fields post-fit."""

    dt: float
    transition_counts: np.ndarray  # [5,5] Dirichlet posterior pseudo-counts (prior + data)
    niw_mu: np.ndarray  # [5,2]
    niw_kappa: np.ndarray  # [5]
    niw_nu: np.ndarray  # [5]
    niw_psi: np.ndarray  # [5,2,2]
    initial_counts: np.ndarray  # [5] Dirichlet posterior pseudo-counts for the initial distribution
    n_iterations: int
    converged: bool
    log_likelihood_history: Tuple[float, ...]
    train_data_sha256: str
    feature_layout: Tuple[str, ...] = ("a_parallel", "omega")
    schema_version: int = 2
    tier: str = "engineering_only"

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise WorldModelError(f"unsupported SBKHMM artifact schema_version={self.schema_version}; expected 2")
        if self.tier not in {"engineering_only", "production"}:
            raise WorldModelError(f"invalid SBKHMM artifact tier={self.tier!r}")
        if self.transition_counts.shape != (N_MODES, N_MODES):
            raise WorldModelError("transition_counts must be [5,5]")
        if self.niw_mu.shape != (N_MODES, 2):
            raise WorldModelError("niw_mu must be [5,2]")
        if self.niw_kappa.shape != (N_MODES,):
            raise WorldModelError("niw_kappa must be [5]")
        if self.niw_nu.shape != (N_MODES,):
            raise WorldModelError("niw_nu must be [5]")
        if self.niw_psi.shape != (N_MODES, 2, 2):
            raise WorldModelError("niw_psi must be [5,2,2]")
        if self.initial_counts.shape != (N_MODES,):
            raise WorldModelError("initial_counts must be [5]")
        # Constructing each mode's NIWParams runs the PD/kappa/nu checks
        # (guide.md B1 repair item 4): reject bad artifacts at
        # construction time, not silently later at first use.
        self._niw_params = tuple(
            NIWParams(mu=self.niw_mu[k], kappa=float(self.niw_kappa[k]), nu=float(self.niw_nu[k]), psi=self.niw_psi[k])
            for k in range(N_MODES)
        )

        # These arrays depend only on the frozen artifact.  The policy asks
        # for them for every human and world sample, so recomputing them in
        # properties would put repeated NIW algebra on the decision path.
        # Keep the caches read-only to preserve the artifact's immutable-by-
        # contract behavior.
        self._transition_matrix_cache = np.asarray(
            self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        )
        self._transition_matrix_cache.setflags(write=False)
        self._initial_distribution_cache = np.asarray(
            self.initial_counts / self.initial_counts.sum()
        )
        self._initial_distribution_cache.setflags(write=False)
        self._emission_cov_cache = np.stack(
            [params.predictive_covariance() for params in self._niw_params], axis=0
        )
        self._emission_cov_cache.setflags(write=False)

    def niw(self, mode: int) -> NIWParams:
        return self._niw_params[mode]

    @property
    def transition_matrix(self) -> np.ndarray:
        """Posterior-mean transition matrix (Dirichlet mean = counts / row sum)."""
        return self._transition_matrix_cache

    @property
    def initial_distribution(self) -> np.ndarray:
        return self._initial_distribution_cache

    @property
    def emission_mean(self) -> np.ndarray:
        """[5,2] posterior predictive location per mode (guide.md's
        'predictive mean' feature; equals the NIW mu)."""
        return self.niw_mu

    @property
    def emission_cov(self) -> np.ndarray:
        """[5,2,2] posterior predictive COVARIANCE per mode (see
        NIWParams.predictive_covariance's dof<=2 caveat). Cached at
        construction because the artifact is immutable by contract."""
        return self._emission_cov_cache

    def emission_log_prob(self, feature: np.ndarray) -> np.ndarray:
        """Log Student-t(feature; NIW predictive params) for each of
        the 5 modes -- R1 fix: previously a plain Gaussian log-pdf
        against a point mean/covariance, not the NIW/Student-t family
        guide.md 4.3 specifies."""
        return np.array([self._niw_params[k].log_predictive_prob(feature) for k in range(N_MODES)])

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dt": self.dt,
            "mode_names": list(MODE_NAMES),
            "feature_layout": list(self.feature_layout),
            "transition_counts": self.transition_counts.tolist(),
            "niw_mu": self.niw_mu.tolist(),
            "niw_kappa": self.niw_kappa.tolist(),
            "niw_nu": self.niw_nu.tolist(),
            "niw_psi": self.niw_psi.tolist(),
            "initial_counts": self.initial_counts.tolist(),
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "log_likelihood_history": list(self.log_likelihood_history),
            "train_data_sha256": self.train_data_sha256,
            "tier": self.tier,
        }

    def content_sha256(self) -> str:
        payload = json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str) -> str:
        data = self.to_json_dict()
        data["content_sha256"] = self.content_sha256()
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return data["content_sha256"]

    @classmethod
    def load(cls, path: str, expect_tier: Optional[str] = None) -> "SBKHMMArtifact":
        data = json.loads(Path(path).read_text())
        stored_hash = data.pop("content_sha256", None)
        if not stored_hash:
            raise WorldModelError(f"{path} is missing required content_sha256")
        schema_version = data.pop("schema_version")
        if schema_version != 2:
            raise WorldModelError(
                f"{path} is schema_version={schema_version} (pre-NIW Gaussian-point-estimate format); "
                "refit with the current fit_sbk_hmm to produce a schema_version=2 NIW artifact"
            )
        mode_names = tuple(data.pop("mode_names"))
        if mode_names != MODE_NAMES:
            raise WorldModelError(f"mode_names mismatch: {mode_names} != {MODE_NAMES}")
        artifact = cls(
            dt=data["dt"],
            transition_counts=np.array(data["transition_counts"]),
            niw_mu=np.array(data["niw_mu"]),
            niw_kappa=np.array(data["niw_kappa"]),
            niw_nu=np.array(data["niw_nu"]),
            niw_psi=np.array(data["niw_psi"]),
            initial_counts=np.array(data["initial_counts"]),
            n_iterations=data["n_iterations"],
            converged=data["converged"],
            log_likelihood_history=tuple(data["log_likelihood_history"]),
            train_data_sha256=data["train_data_sha256"],
            feature_layout=tuple(data["feature_layout"]),
            schema_version=schema_version,
            tier=data.get("tier", "engineering_only"),
        )
        if expect_tier is not None and artifact.tier != expect_tier:
            raise WorldModelError(f"{path} tier={artifact.tier!r} does not satisfy expect_tier={expect_tier!r}")
        if artifact.content_sha256() != stored_hash:
            raise WorldModelError(
                f"{path} content hash mismatch: stored={stored_hash} recomputed={artifact.content_sha256()}"
            )
        return artifact


def promote_to_production(artifact: SBKHMMArtifact) -> SBKHMMArtifact:
    """Explicit promotion gate (guide.md B1 repair item 4 / A3
    acceptance): unconverged fits must never be saved as a production
    candidate. ``fit_sbk_hmm`` itself still returns best-effort fits
    unconditionally so callers can inspect diagnostics on a failed fit;
    only THIS function is the hard gate before a fit is treated as
    usable."""
    if not artifact.converged:
        raise WorldModelError(
            f"refusing to promote an unconverged fit (ran {artifact.n_iterations} iterations) to production"
        )
    artifact.tier = "production"
    return artifact


def _initial_niw_prior() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prior NIW hyperparameters placed inside each mode's sign-
    constrained half-plane -- NOT data-dependent, so EM must actually
    move the posterior via responsibility-weighted statistics."""
    mu0 = np.array([
        [0.0, 0.0],     # CV
        [0.6, 0.0],     # ACC
        [-0.6, 0.0],    # DECEL
        [0.0, 0.8],     # TURN_L
        [0.0, -0.8],    # TURN_R
    ])
    kappa0 = np.full(N_MODES, 1.0)
    nu0 = np.full(N_MODES, 4.0)  # > d-1=1, gives a proper (finite-variance) prior predictive
    psi0 = np.tile(np.eye(2) * 0.3, (N_MODES, 1, 1))
    return mu0, kappa0, nu0, psi0


def _project_mean_to_constraint(mean_k: np.ndarray, mode: int, min_magnitude: float) -> np.ndarray:
    """Enforce guide.md 4.3's sign constraints on the fitted mean: project
    onto the mode's half-plane rather than silently letting EM drift a
    mode's mean across the CV boundary (which would let mode identity
    swap under permutation -- explicitly forbidden by A3 acceptance)."""
    a_sign, w_sign = _MODE_SIGN_CONSTRAINTS[mode]
    a, w = mean_k
    if a_sign == 1:
        a = max(a, min_magnitude)
    elif a_sign == -1:
        a = min(a, -min_magnitude)
    elif a_sign == 0:
        a = np.clip(a, -min_magnitude, min_magnitude)
    if w_sign == 1:
        w = max(w, min_magnitude)
    elif w_sign == -1:
        w = min(w, -min_magnitude)
    elif w_sign == 0:
        w = np.clip(w, -min_magnitude, min_magnitude)
    return np.array([a, w])


def _forward_backward(log_emission: np.ndarray, log_pi: np.ndarray, log_trans: np.ndarray):
    """Standard scaled forward-backward on a single sequence.
    log_emission: [T, K]. Returns (gamma [T,K], xi_sum [K,K], loglik)."""
    T, K = log_emission.shape
    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, k]) + log_emission[t, k]

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for k in range(K):
            log_beta[t, k] = np.logaddexp.reduce(log_trans[k, :] + log_emission[t + 1] + log_beta[t + 1])

    loglik = np.logaddexp.reduce(log_alpha[-1])
    log_gamma = log_alpha + log_beta - loglik
    gamma = np.exp(log_gamma)

    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None] + log_trans + log_emission[t + 1][None, :] + log_beta[t + 1][None, :] - loglik
        )
        xi_sum += np.exp(log_xi_t)

    return gamma, xi_sum, loglik


def fit_sbk_hmm(
    tracks: Sequence[Track],
    train_data_sha256: str,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    dirichlet_alpha: float = 1.0,
    sign_projection_min_magnitude: float = 0.05,
) -> SBKHMMArtifact:
    """EM fit with semantic sign constraints and an NIW/Student-t
    emission family (guide.md 4.3). Raises WorldModelError if there is
    no usable data, if tracks have inconsistent dt (guide.md B1 repair
    item 4: "reject ... mixed dt tracks"), or if any produced NIW
    posterior is degenerate. Returns a best-effort fit regardless of
    convergence so callers can inspect diagnostics; use
    ``promote_to_production`` as the hard convergence gate before
    treating a fit as usable."""
    if not tracks:
        raise WorldModelError("no tracks provided")
    dt = tracks[0].dt
    for track in tracks:
        if track.dt != dt:
            raise WorldModelError(f"mixed dt across tracks: {track.dt} != {dt}")

    sequences: List[np.ndarray] = []
    for track in tracks:
        sequences.extend(track.feature_runs())
    if not sequences:
        raise WorldModelError("no usable (valid-heading) feature frames in any track")

    mu, kappa, nu, psi = _initial_niw_prior()
    mu0, kappa0, nu0, psi0 = _initial_niw_prior()

    trans_counts = np.full((N_MODES, N_MODES), dirichlet_alpha)
    np.fill_diagonal(trans_counts, dirichlet_alpha + 5.0)  # sticky prior
    initial_counts = np.full(N_MODES, dirichlet_alpha)

    prev_loglik = -np.inf
    loglik_history: List[float] = []
    converged = False
    n_iterations = 0

    for iteration in range(max_iterations):
        n_iterations = iteration + 1
        log_trans = np.log(trans_counts / trans_counts.sum(axis=1, keepdims=True))
        log_pi = np.log(initial_counts / initial_counts.sum())

        niw_params = [NIWParams(mu=mu[k], kappa=float(kappa[k]), nu=float(nu[k]), psi=psi[k]) for k in range(N_MODES)]

        total_loglik = 0.0
        weighted_mean_num = np.zeros((N_MODES, 2))
        weighted_n = np.zeros(N_MODES)
        xi_accum = np.zeros((N_MODES, N_MODES))
        initial_accum = np.zeros(N_MODES)
        all_gammas = []
        all_feature_seqs = []

        for seq in sequences:
            mode_log_probs = []
            for k in range(N_MODES):
                dof_k, mu_k, scale_k = niw_params[k].predictive_dof_location_scale()
                mode_log_probs.append(_multivariate_t_log_pdf(seq, dof_k, mu_k, scale_k))
            log_emission = np.stack(mode_log_probs, axis=1)
            gamma, xi_sum, loglik = _forward_backward(log_emission, log_pi, log_trans)
            total_loglik += loglik
            initial_accum += gamma[0]
            xi_accum += xi_sum
            weighted_mean_num += gamma.T @ seq
            weighted_n += gamma.sum(axis=0)
            all_gammas.append(gamma)
            all_feature_seqs.append(seq)

        loglik_history.append(float(total_loglik))

        # M-step: Dirichlet posterior updates (pseudo-counts, guide.md 4.3).
        trans_counts = np.full((N_MODES, N_MODES), dirichlet_alpha) + xi_accum
        initial_counts = np.full(N_MODES, dirichlet_alpha) + initial_accum

        # M-step: NIW posterior updates from responsibility-weighted
        # sufficient statistics (standard conjugate NIW update).
        new_mu = mu.copy()
        new_kappa = kappa.copy()
        new_nu = nu.copy()
        new_psi = psi.copy()
        for k in range(N_MODES):
            n_k = weighted_n[k]
            if n_k > 1e-8:
                xbar_k = weighted_mean_num[k] / n_k
            else:
                xbar_k = mu0[k]

            scatter_k = np.zeros((2, 2))
            for gamma, seq in zip(all_gammas, all_feature_seqs):
                diff = seq - xbar_k
                scatter_k += (gamma[:, k, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)

            kappa_k = kappa0[k] + n_k
            mean_diff = xbar_k - mu0[k]
            raw_mu_k = (kappa0[k] * mu0[k] + n_k * xbar_k) / kappa_k
            new_mu[k] = _project_mean_to_constraint(raw_mu_k, k, sign_projection_min_magnitude)
            new_kappa[k] = kappa_k
            new_nu[k] = nu0[k] + n_k
            new_psi[k] = (
                psi0[k] + scatter_k
                + (kappa0[k] * n_k / kappa_k) * np.outer(mean_diff, mean_diff)
            )

        mu, kappa, nu, psi = new_mu, new_kappa, new_nu, new_psi

        if abs(total_loglik - prev_loglik) < tolerance * max(1.0, abs(prev_loglik)):
            converged = True
            break
        prev_loglik = total_loglik

    return SBKHMMArtifact(
        dt=dt,
        transition_counts=trans_counts,
        niw_mu=mu, niw_kappa=kappa, niw_nu=nu, niw_psi=psi,
        initial_counts=initial_counts,
        n_iterations=n_iterations,
        converged=converged,
        log_likelihood_history=tuple(loglik_history),
        train_data_sha256=train_data_sha256,
    )
