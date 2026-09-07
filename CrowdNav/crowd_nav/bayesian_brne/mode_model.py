"""Variable-N mode learning: feature extraction, K selection, per-mode
velocity-residual dynamics, transition matrix, artifact save/load.

Does NOT modify or import ``gdbn.py`` (guide.md 6.1): its fixed 34-D/
5-human format, slot-indexed tracker, and mean-collapsing rollout are all
unsuitable for this project's variable-N, sample-preserving design. It DOES
reuse the same proven math style already validated in
``tools/fit_behavior_gdbn_modes.py`` and ``bayesian_decision_gate/
fit_models.py`` (k-means++ clustering + ridge-regression dynamics),
generalized to arbitrary N and to VELOCITY-RESIDUAL dynamics (guide.md 2.1):

    delta_v_{t+1} = F_k @ phi_t + eps_k,   eps_k ~ N(0, Q_k)
    v_{t+1}       = v_t + delta_v_{t+1}
    p_{t+1}       = p_t + dt * v_{t+1}

instead of the old absolute-state ``A_k @ state`` form -- this keeps the
learned dynamics translation-invariant (a mode's meaning cannot become "walk
toward this specific scene coordinate").

``phi_t`` (guide.md 6.2) is built ONLY from deployment-observable
quantities: speed, delta_speed, heading_change, lateral_acceleration,
robot-relative position/velocity, clipped TTC, predicted passing side.
Pedestrian ground-truth goals and any test script's latent behavior label
are NEVER read here (guide.md 5.2/5.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from crowd_nav.bayesian_brne.config import BayesianModelConfig
from crowd_nav.bayesian_brne.schemas import robot_position, robot_velocity

FEATURE_NAMES = (
    "speed",
    "delta_speed",
    "heading_change",
    "lateral_acceleration",
    "relative_px",
    "relative_py",
    "relative_vx",
    "relative_vy",
    "ttc_clipped",
    "passing_side",
)
TTC_CLIP = 10.0


@dataclass
class TransitionRow:
    track_key: Tuple[int, int]  # (episode_index, track_id)
    t: int
    v: np.ndarray  # [2] velocity at t
    delta_v_next: np.ndarray  # [2] v_{t+1} - v_t (fit target)
    phi: np.ndarray  # [len(FEATURE_NAMES)] feature vector at t
    step_dt: float


@dataclass
class ModeModelArtifact:
    K: int
    F: List[np.ndarray]  # each [2, len(FEATURE_NAMES)]
    Q: List[np.ndarray]  # each [2, 2], positive definite
    Pi: np.ndarray  # [K, K] row-stochastic transition matrix
    feature_mean: np.ndarray
    feature_std: np.ndarray
    cluster_centers: np.ndarray  # [K, len(FEATURE_NAMES)] standardized-feature centroids
    dt: float
    model_card: dict = field(default_factory=dict)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "K": np.asarray(self.K),
            "Pi": self.Pi,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "cluster_centers": self.cluster_centers,
            "dt": np.asarray(self.dt),
        }
        for k in range(self.K):
            payload[f"F_{k}"] = self.F[k]
            payload[f"Q_{k}"] = self.Q[k]
        np.savez(output_dir / "mode_model.npz", **payload)
        (output_dir / "model_card.json").write_text(json.dumps(self.model_card, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: Path) -> "ModeModelArtifact":
        output_dir = Path(output_dir)
        data = np.load(output_dir / "mode_model.npz")
        K = int(data["K"])
        model_card = {}
        card_path = output_dir / "model_card.json"
        if card_path.exists():
            model_card = json.loads(card_path.read_text(encoding="utf-8"))
        return cls(
            K=K,
            F=[data[f"F_{k}"] for k in range(K)],
            Q=[data[f"Q_{k}"] for k in range(K)],
            Pi=data["Pi"],
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            cluster_centers=data["cluster_centers"],
            dt=float(data["dt"]),
            model_card=model_card,
        )


def _angle_wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _safe_heading(v: np.ndarray) -> float:
    speed = float(np.hypot(v[0], v[1]))
    return float(np.arctan2(v[1], v[0])) if speed > 1e-4 else 0.0


def extract_transitions(
    episodes: List[dict],
    dt: float,
) -> List[TransitionRow]:
    """``episodes`` is a list of dicts with keys ``humans`` ``[T, N, 5]``
    (px, py, vx, vy, radius), ``human_track_ids`` ``[T, N]``, ``robot``
    ``[T, 9]`` (px,py,vx,vy,radius,gx,gy,v_pref,theta -- schemas.ROBOT_STATE_FIELDS), ``valid_mask``
    ``[T, N]`` bool. Generalizes to any N.

    Only STRICTLY CONSECUTIVE raw steps (``t1 - t == 1``) are turned into a
    fit transition (fixes D3 from the 2026-08-03 audit: an earlier version
    accepted gaps of up to 2 raw steps but still fed ``delta_v_next = v_t1 -
    v_t`` straight into ``_fit_dynamics`` as if it were a single-``dt``
    target, while the rollout in trajectory_sampler.py always advances by
    exactly one fixed ``dt`` -- a 2-step gap's velocity change is roughly
    double a real 1-step delta for the same ``phi``, biasing ``F_k`` upward.
    guide.md's own recommendation is to accept gaps only if they are
    correctly re-derived against their true ``step_dt``, or else restrict to
    consecutive transitions; this takes the simpler, safer option). Any gap
    (``t1 - t != 1``) breaks the ``prev_v``/``prev_phi`` chain the same way a
    missing-track frame does in belief_tracker.py, so ``delta_speed``/
    ``heading_change``/``lateral_acc`` on the next accepted row are not
    silently computed across a skipped frame either.
    """
    rows: List[TransitionRow] = []
    for episode_index, episode in enumerate(episodes):
        humans = np.asarray(episode["humans"], dtype=np.float64)
        track_ids = np.asarray(episode["human_track_ids"])
        robot = np.asarray(episode["robot"], dtype=np.float64)
        valid_mask = np.asarray(episode["valid_mask"], dtype=bool)
        T, N = track_ids.shape[0], track_ids.shape[1]

        by_track: Dict[int, List[int]] = {}
        for t in range(T):
            for n in range(N):
                if not valid_mask[t, n]:
                    continue
                by_track.setdefault(int(track_ids[t, n]), []).append(t)

        for track_id, times in by_track.items():
            slot_at_t = {t: int(np.where(track_ids[t] == track_id)[0][0]) for t in times}
            prev_v: Optional[np.ndarray] = None
            for ii in range(len(times) - 1):
                t, t1 = times[ii], times[ii + 1]
                if t1 - t != 1:
                    prev_v = None
                    continue
                slot_t, slot_t1 = slot_at_t[t], slot_at_t[t1]
                x = humans[t, slot_t, :4]
                y = humans[t1, slot_t1, :4]
                step_dt = dt
                v_t = x[2:4]
                v_t1 = y[2:4]

                speed = float(np.hypot(v_t[0], v_t[1]))
                delta_speed = 0.0
                heading_change = 0.0
                lateral_acc = 0.0
                if prev_v is not None:
                    prev_speed = float(np.hypot(prev_v[0], prev_v[1]))
                    delta_speed = (speed - prev_speed) / step_dt
                    heading_change = float(_angle_wrap(_safe_heading(v_t) - _safe_heading(prev_v))) / step_dt
                    accel = (v_t - prev_v) / step_dt
                    unit = v_t / max(speed, 1e-4)
                    normal = np.array([-unit[1], unit[0]])
                    lateral_acc = float(np.dot(accel, normal))

                r = robot[t]
                rel_pos = x[:2] - robot_position(r)
                rel_vel = v_t - robot_velocity(r)
                dist = float(np.hypot(rel_pos[0], rel_pos[1]) + 1e-6)
                closing = float(-np.dot(rel_pos, rel_vel) / dist)
                ttc = dist / closing if closing > 0.1 else TTC_CLIP
                ttc_clipped = float(np.clip(ttc, 0.0, TTC_CLIP))
                passing_side = float((rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]) / dist)

                phi = np.array(
                    [
                        speed, delta_speed, heading_change, lateral_acc,
                        rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1],
                        ttc_clipped, passing_side,
                    ],
                    dtype=np.float64,
                )
                rows.append(
                    TransitionRow(
                        track_key=(episode_index, track_id), t=t,
                        v=v_t.copy(), delta_v_next=(v_t1 - v_t).copy(),
                        phi=phi, step_dt=step_dt,
                    )
                )
                prev_v = v_t
    if not rows:
        raise ValueError("extract_transitions: no valid transitions found in given episodes")
    return rows


def _standardize(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    return (features - mean) / std, mean, std


def _kmeans_pp(features: np.ndarray, K: int, seed: int, restarts: int = 8, max_iter: int = 100):
    rng = np.random.default_rng(seed)
    n = len(features)
    best_labels, best_centers, best_score = None, None, float("inf")
    for _ in range(max(1, restarts)):
        centers = np.empty((K, features.shape[1]))
        centers[0] = features[rng.integers(n)]
        dist2 = np.sum((features - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            probs = dist2 / max(float(dist2.sum()), 1e-12)
            centers[k] = features[rng.choice(n, p=probs)]
            dist2 = np.minimum(dist2, np.sum((features - centers[k]) ** 2, axis=1))
        labels = np.zeros(n, dtype=np.int64)
        for _it in range(max_iter):
            d = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d, axis=1)
            new_centers = centers.copy()
            for k in range(K):
                idx = new_labels == k
                if idx.any():
                    new_centers[k] = features[idx].mean(axis=0)
                else:
                    farthest = int(np.argmax(np.min(d, axis=1)))
                    new_centers[k] = features[farthest]
                    new_labels[farthest] = k
            shift = float(np.linalg.norm(new_centers - centers))
            labels, centers = new_labels, new_centers
            if shift < 1e-5:
                break
        d = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(d, axis=1)))
        counts = np.bincount(labels, minlength=K)
        balance_penalty = float(np.sum((counts < max(20, n // (K * 40))) * n))
        score = inertia + balance_penalty
        if score < best_score:
            best_score, best_labels, best_centers = score, labels.copy(), centers.copy()
    return best_labels, best_centers


def _fit_dynamics(rows: List[TransitionRow], labels: np.ndarray, K: int, ridge: float, covariance_floor: float):
    F_list, Q_list = [], []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 10:
            F_list.append(np.zeros((2, len(FEATURE_NAMES))))
            Q_list.append(covariance_floor * np.eye(2))
            continue
        Phi = np.asarray([rows[i].phi for i in idx])
        Y = np.asarray([rows[i].delta_v_next for i in idx])
        reg = ridge * np.eye(Phi.shape[1])
        F_k = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y).T  # [2, num_features]
        residual = Y - (F_k @ Phi.T).T
        cov = (residual.T @ residual) / max(1, len(residual)) + covariance_floor * np.eye(2)
        F_list.append(F_k)
        Q_list.append(cov)
    return F_list, Q_list


def _fit_transition_matrix(rows: List[TransitionRow], labels: np.ndarray, K: int) -> np.ndarray:
    counts = np.zeros((K, K))
    by_track: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i, row in enumerate(rows):
        by_track.setdefault(row.track_key, []).append((row.t, int(labels[i])))
    for items in by_track.values():
        items = sorted(items)
        for (_, a), (_, b) in zip(items[:-1], items[1:]):
            counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1, 1.0, row_sums)
    return (counts + 0.1) / (row_sums + 0.1 * K)


def _logpdf_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    diff = x - mean
    inv = np.linalg.inv(cov)
    return float(-0.5 * (diff @ inv @ diff + logdet + len(x) * np.log(2.0 * np.pi)))


def _held_out_nll(rows: List[TransitionRow], F_list, Q_list, Pi: np.ndarray) -> float:
    K = len(F_list)
    by_track: Dict[Tuple[int, int], List[TransitionRow]] = {}
    for row in rows:
        by_track.setdefault(row.track_key, []).append(row)
    nlls = []
    for items in by_track.values():
        items = sorted(items, key=lambda r: r.t)
        posterior = np.ones(K) / K
        for row in items:
            prior = np.maximum(posterior @ Pi, 1e-12)
            prior /= prior.sum()
            means = [row.v + F_list[k] @ row.phi for k in range(K)]
            log_terms = np.array([np.log(prior[k]) + _logpdf_gaussian(row.delta_v_next + row.v, means[k], Q_list[k]) for k in range(K)])
            m = log_terms.max()
            nlls.append(-(m + np.log(np.exp(log_terms - m).sum())))
            likelihood = np.exp(log_terms - log_terms.max())
            posterior = likelihood / max(float(likelihood.sum()), 1e-12)
    return float(np.mean(nlls)) if nlls else float("nan")


def _mode_similarity(F_a: np.ndarray, F_b: np.ndarray) -> float:
    """Cosine similarity between two modes' flattened dynamics. DIAGNOSTIC
    ONLY (reported in the model card) -- NOT the mode-collapse gate. Two
    modes can have very different F but nearly-identical predicted
    distributions on the actual validation feature distribution (or
    similar F but very different Q, i.e. genuinely different uncertainty
    regimes wrongly flagged as duplicates by F alone) -- see
    ``_predictive_distribution_similarity`` for the real gate (fixes B6
    from the 2026-08-03 independent audit)."""
    a, b = F_a.ravel(), F_b.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 1.0
    return float(np.dot(a, b) / denom)


def _bhattacharyya_coefficient(mu_a: np.ndarray, cov_a: np.ndarray, mu_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Bhattacharyya coefficient between two Gaussians -- 1.0 means
    identical distributions, near 0 means well-separated. Used (not cosine
    similarity of raw parameters) as the actual mode-collapse gate, because
    it directly answers "do these two modes make distinguishable
    predictions", the thing that actually matters."""
    cov = 0.5 * (cov_a + cov_b)
    diff = mu_a - mu_b
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    mahalanobis_term = 0.125 * float(diff @ inv_cov @ diff)
    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    sign, logdet = np.linalg.slogdet(cov)
    log_det_term = 0.5 * (logdet - 0.5 * logdet_a - 0.5 * logdet_b)
    bhattacharyya_distance = mahalanobis_term + log_det_term
    return float(np.exp(-max(bhattacharyya_distance, 0.0)))


def _predictive_distribution_similarity(
    F_a: np.ndarray, Q_a: np.ndarray, F_b: np.ndarray, Q_b: np.ndarray, validation_phis: np.ndarray,
) -> float:
    """Mean Bhattacharyya coefficient between mode a's and mode b's
    predicted next-velocity distribution, averaged over the ACTUAL
    validation feature distribution (not synthetic/uniform phi) -- this is
    what "these two modes are indistinguishable" should mean: not that
    their parameter vectors happen to point the same direction, but that
    they make statistically inseparable predictions on real data."""
    if len(validation_phis) == 0:
        return _bhattacharyya_coefficient(np.zeros(2), Q_a, np.zeros(2), Q_b)
    sample = validation_phis if len(validation_phis) <= 500 else validation_phis[:: max(1, len(validation_phis) // 500)]
    coefficients = [
        _bhattacharyya_coefficient(F_a @ phi, Q_a, F_b @ phi, Q_b)
        for phi in sample
    ]
    return float(np.mean(coefficients))


@dataclass
class KCandidateReport:
    K: int
    eligible: bool
    rejection_reasons: List[str]
    held_out_nll: float
    mode_counts: List[int]
    max_pairwise_similarity: float  # diagnostic only (F cosine)
    max_predictive_similarity: float  # the actual gate (Bhattacharyya coefficient)
    pi_row_entropy: float


def fit_and_select(
    train_rows: List[TransitionRow],
    validation_rows: List[TransitionRow],
    config: BayesianModelConfig,
    ridge: float = 1e-3,
    seed: int = 2407,
    require_multimodal: bool = True,
    min_nll_improvement: float = 0.01,
    return_all_fits: bool = False,
):
    """Fits K in ``config.k_candidates`` on ``train_rows``, scores each on
    ``validation_rows``, applies guide.md 6.3's hard rejection rules
    (mode-collapse gate now uses the modes' PREDICTIVE DISTRIBUTIONS on
    validation features, not raw F-parameter cosine similarity -- fixes B6),
    and selects the eligible K with the best held-out NLL.

    ``require_multimodal`` (fixes B5): when True (the default, and the ONLY
    mode the real SM-BRNE artifact may be fit with), K=1 is NEVER an
    acceptable selection -- if no K>1 candidate is eligible, this raises
    loudly rather than silently returning K=1 and letting the caller believe
    a "switching-mode Bayesian" artifact was produced when in fact no
    multimodality survived validation. Pass ``require_multimodal=False``
    only when deliberately fitting a K=1 baseline artifact for comparison
    (e.g. a "legacy_k3"-style ablation), never for the main method.

    ``min_nll_improvement`` (also part of B5): a K>1 candidate must beat
    K=1's held-out NLL by at least this many nats, not by an arbitrary
    floating-point epsilon -- guards against a spurious, statistically
    meaningless improvement being accepted as evidence of real
    multimodality.

    ``return_all_fits`` (Order 9): when True, additionally returns a third
    element -- ``{K: (F_list, Q_list, Pi, cluster_centers)}`` for EVERY K in
    ``config.k_candidates``, not just the selected winner. fit_bayesian_brne.py
    needs every candidate's own fitted parameters to report per-K Brier/
    ECE/bootstrap CI, not only the winning K's.
    """
    train_features = np.asarray([r.phi for r in train_rows])
    validation_features = np.asarray([r.phi for r in validation_rows])
    _, feature_mean, feature_std = _standardize(train_features)
    reports: List[KCandidateReport] = []
    fitted: Dict[int, Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]] = {}

    standardized_train = (train_features - feature_mean) / feature_std

    for K in config.k_candidates:
        if K == 1:
            labels = np.zeros(len(train_rows), dtype=np.int64)
            centers = standardized_train.mean(axis=0, keepdims=True)
        else:
            labels, centers = _kmeans_pp(standardized_train, K, seed=seed)
        F_list, Q_list = _fit_dynamics(train_rows, labels, K, ridge, config.covariance_floor)
        Pi = _fit_transition_matrix(train_rows, labels, K)
        held_out_nll = _held_out_nll(validation_rows, F_list, Q_list, Pi)

        counts = np.bincount(labels, minlength=K).tolist()
        cosine_pairwise = [
            _mode_similarity(F_list[i], F_list[j])
            for i in range(K) for j in range(i + 1, K)
        ]
        predictive_pairwise = [
            _predictive_distribution_similarity(F_list[i], Q_list[i], F_list[j], Q_list[j], validation_features)
            for i in range(K) for j in range(i + 1, K)
        ]
        max_similarity = max(cosine_pairwise) if cosine_pairwise else 0.0
        max_predictive_similarity = max(predictive_pairwise) if predictive_pairwise else 0.0
        pi_entropy = float(-np.mean(np.sum(Pi * np.log(np.maximum(Pi, 1e-10)), axis=1)))

        reasons = []
        min_fraction = min(counts) / max(1, sum(counts))
        if K > 1 and min_fraction < config.min_mode_fraction:
            reasons.append(f"mode_fraction {min_fraction:.4f} < min_mode_fraction {config.min_mode_fraction}")
        if K > 1 and max_predictive_similarity > config.max_mode_similarity:
            reasons.append(
                f"max_predictive_similarity {max_predictive_similarity:.4f} > threshold "
                f"{config.max_mode_similarity} (modes make statistically indistinguishable predictions)"
            )
        if not np.all(np.isfinite(Pi)) or not np.allclose(Pi.sum(axis=1), 1.0, atol=1e-6):
            reasons.append("Pi is not finite/row-normalized")
        for k in range(K):
            eigvals = np.linalg.eigvalsh(Q_list[k])
            if np.any(eigvals <= 0.0):
                reasons.append(f"Q_{k} is not positive definite")

        reports.append(
            KCandidateReport(
                K=K, eligible=(len(reasons) == 0), rejection_reasons=reasons,
                held_out_nll=held_out_nll, mode_counts=counts,
                max_pairwise_similarity=max_similarity,
                max_predictive_similarity=max_predictive_similarity,
                pi_row_entropy=pi_entropy,
            )
        )
        fitted[K] = (F_list, Q_list, Pi, centers)

    k1_nll = next(r.held_out_nll for r in reports if r.K == 1)
    for r in reports:
        if r.K > 1 and (k1_nll - r.held_out_nll) < min_nll_improvement:
            r.eligible = False
            r.rejection_reasons.append(
                f"K={r.K} held_out_nll improvement over K=1 ({k1_nll - r.held_out_nll:.4f} nats) "
                f"< min_nll_improvement ({min_nll_improvement} nats)"
            )

    multimodal_eligible = [r for r in reports if r.eligible and r.K > 1]
    if require_multimodal:
        if not multimodal_eligible:
            raise RuntimeError(
                "fit_and_select: require_multimodal=True but every K>1 candidate was rejected "
                "-- per guide.md 6.3 this must fail loudly, not silently fall back to K=1 and "
                "call it a passing SM-BRNE (switching-mode) artifact. See KCandidateReport.rejection_reasons "
                "for why each K>1 candidate failed. Pass require_multimodal=False only for an "
                "explicit K=1/legacy baseline fit, never for the main method."
            )
        best = min(multimodal_eligible, key=lambda r: r.held_out_nll)
    else:
        eligible = [r for r in reports if r.eligible]
        if not eligible:
            raise RuntimeError("fit_and_select: every K candidate was rejected, including K=1.")
        best = min(eligible, key=lambda r: r.held_out_nll)
    F_list, Q_list, Pi, centers = fitted[best.K]
    artifact = ModeModelArtifact(
        K=best.K, F=F_list, Q=Q_list, Pi=Pi,
        feature_mean=feature_mean, feature_std=feature_std, cluster_centers=centers,
        dt=config.dt,
        model_card={
            "selected_K": best.K,
            "candidates": [
                {
                    "K": r.K, "eligible": r.eligible, "rejection_reasons": r.rejection_reasons,
                    "held_out_nll": r.held_out_nll, "mode_counts": r.mode_counts,
                    "max_pairwise_similarity": r.max_pairwise_similarity,
                    "max_predictive_similarity": r.max_predictive_similarity,
                    "pi_row_entropy": r.pi_row_entropy,
                }
                for r in reports
            ],
            "feature_names": list(FEATURE_NAMES),
        },
    )
    if return_all_fits:
        return artifact, reports, fitted
    return artifact, reports


def assign_mode(artifact: ModeModelArtifact, phi: np.ndarray) -> int:
    standardized = (phi - artifact.feature_mean) / artifact.feature_std
    dists = np.linalg.norm(artifact.cluster_centers - standardized, axis=1)
    return int(np.argmin(dists))
