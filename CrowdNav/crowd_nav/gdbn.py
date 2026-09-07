# -*- coding: utf-8 -*-
"""
GDBN: Generalized Dynamic Bayesian Network for Pedestrian Intent Tracking

RegazzoniGDBN/MJPFcrowd navigation

Pipeline:
 GNG ORCAK
 GDBN Π + A_k, Q_k
 MJPF
 KLDA KL(λ(S_t) || α(S_t))/
 GDBNIntegration Mamba-VLAPImamba_rl.py / explorer.py

  Fritzke 1995. A Growing Neural Gas Network Learns Topologies.
  Regazzoni et al. Multi-modal Generative Models for GDBN.
 GDBN
 KLDAMamba-VLlookahead
"""

import os
import numpy as np
from typing import List, Optional, Tuple

# ============================================================================ #
#  1. GNG — Growing Neural Gas
# ============================================================================ #

class GNG:
    """
 Growing Neural Gas

 (5D): [rel_x, rel_y, rel_dist, rel_vx, rel_vy]
 : ORCA demo (T, 34)
 : K = K
    """

    def __init__(
        self,
        k_target: int = 8,
        max_age: int = 50,
        lambda_insert: int = 100,
        eps_b: float = 0.05,
        eps_n: float = 0.006,
        alpha: float = 0.5,
        beta_decay: float = 0.995,
    ):
        self.k_target = k_target
        self.max_age = max_age
        self.lambda_insert = lambda_insert
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.alpha = alpha
        self.beta_decay = beta_decay

        self.nodes: Optional[np.ndarray] = None
        self.errors: Optional[np.ndarray] = None
        self.edges: dict = {}
        self._step: int = 0

        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #

    def fit(self, npz_path: str, n_epochs: int = 3, random_seed: int = 42):
        """ORCA demo npzGNG"""
        data = np.load(npz_path, allow_pickle=True)
        obs_list = data['obs']

        all_feats: List[np.ndarray] = []
        for obs_seq in obs_list:
            if not (isinstance(obs_seq, np.ndarray) and obs_seq.ndim == 2
                    and obs_seq.shape[1] == 34):
                continue
            f = self._extract_features(obs_seq)
            if len(f) > 0:
                all_feats.append(f)

        if not all_feats:
            raise ValueError("[GNG] demo")

        features = np.concatenate(all_feats, axis=0).astype(np.float32)
        self._feat_mean = features.mean(axis=0)
        self._feat_std  = features.std(axis=0) + 1e-6
        features = (features - self._feat_mean) / self._feat_std

        if self.k_target == 1:
            self.nodes = np.zeros((1, features.shape[1]), dtype=np.float32)
            self.errors = np.zeros(1, dtype=np.float32)
            self.edges = {}
            self._step = 0
            print(f"[GNG] single-mode fit: 1 mode, {len(features)} samples")
            return

        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(features), 2, replace=False)
        self.nodes = features[idx].copy()
        self.errors = np.zeros(2, dtype=np.float32)
        self.edges = {}
        self._step = 0

        for _ in range(n_epochs):
            perm = rng.permutation(len(features))
            for i in perm:
                self._gng_step(features[i])

        print(f"[GNG] : {self.n_modes} {len(features)} ")

    def _extract_features(self, obs_seq: np.ndarray) -> np.ndarray:
        """ (T, 34)  (T*K, 5)"""
        robot = obs_seq[:, :9]
        feats = []
        for p in range(5):
            s = 9 + p * 5
            ped = obs_seq[:, s:s + 5]
            valid = ~np.all(ped == 0, axis=1)
            if valid.sum() == 0:
                continue
            r = robot[valid]
            pd = ped[valid]
            rel_x  = pd[:, 0] - r[:, 0]
            rel_y  = pd[:, 1] - r[:, 1]
            dist   = np.hypot(rel_x, rel_y) + 1e-6
            rel_vx = pd[:, 2] - r[:, 2]
            rel_vy = pd[:, 3] - r[:, 3]
            feats.append(np.stack([rel_x, rel_y, dist, rel_vx, rel_vy], axis=1))
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 5), dtype=np.float32)

    def _gng_step(self, x: np.ndarray):
        dists = np.linalg.norm(self.nodes - x, axis=1)
        order = np.argsort(dists)
        s1, s2 = order[0], order[1]

        self.errors[s1] += float(dists[s1] ** 2)
        self.nodes[s1] += self.eps_b * (x - self.nodes[s1])

        for (i, j) in list(self.edges.keys()):
            n = j if i == s1 else (i if j == s1 else -1)
            if n >= 0:
                self.nodes[n] += self.eps_n * (x - self.nodes[n])

        key = (min(s1, s2), max(s1, s2))
        self.edges[key] = 0

        for k in list(self.edges.keys()):
            if s1 in k:
                self.edges[k] += 1
        for k in [k for k, a in self.edges.items() if a > self.max_age]:
            del self.edges[k]

        connected = set()
        for (i, j) in self.edges:
            connected.add(i); connected.add(j)
        isolated = [i for i in range(len(self.nodes)) if i not in connected]
        if isolated and len(self.nodes) > 2:
            keep = [i for i in range(len(self.nodes)) if i not in isolated]
            old2new = {o: n for n, o in enumerate(keep)}
            self.nodes  = self.nodes[keep]
            self.errors = self.errors[keep]
            self.edges  = {
                (old2new[i], old2new[j]): a
                for (i, j), a in self.edges.items()
                if i in old2new and j in old2new
            }

        self._step += 1
        if self._step % self.lambda_insert == 0 and self.n_modes < self.k_target:
            q = int(np.argmax(self.errors))
            nbrs = ([j for (i, j) in self.edges if i == q]
                    + [i for (i, j) in self.edges if j == q])
            if nbrs:
                f = nbrs[int(np.argmax(self.errors[nbrs]))]
                r_idx = self.n_modes
                new_node = 0.5 * (self.nodes[q] + self.nodes[f])
                self.nodes  = np.vstack([self.nodes,  new_node])
                self.errors = np.append(self.errors, 0.0)
                qf_key = (min(q, f), max(q, f))
                if qf_key in self.edges:
                    del self.edges[qf_key]
                self.edges[(min(q, r_idx), max(q, r_idx))] = 0
                self.edges[(min(f, r_idx), max(f, r_idx))] = 0
                self.errors[q] *= self.alpha
                self.errors[f] *= self.alpha

        self.errors *= self.beta_decay

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #

    def assign_mode(self, feat: np.ndarray) -> int:
        """5DGNG"""
        feat_n = (np.asarray(feat, dtype=np.float32) - self._feat_mean) / self._feat_std
        return int(np.argmin(np.linalg.norm(self.nodes - feat_n, axis=1)))

    @property
    def n_modes(self) -> int:
        return len(self.nodes) if self.nodes is not None else 0

    def save(self, path: str):
        np.savez(path, nodes=self.nodes, errors=self.errors,
                 feat_mean=self._feat_mean, feat_std=self._feat_std)

    def load(self, path: str):
        d = np.load(path)
        self.nodes = d['nodes']
        self.errors = d['errors']
        self._feat_mean = d['feat_mean']
        self._feat_std  = d['feat_std']
        self.edges = {}

# ============================================================================ #
#  2. GDBN — Generalized Dynamic Bayesian Network
# ============================================================================ #

class GDBN:
    """

 S_t ∈ {0,...,K-1} GNG
 X_t ∈ R^4 [px, py, vx, vy]
 Z_t = X_t + v RVO2

 Pi : (K,K) P(S_t | S_{t-1})
 A[k]: (4,4) k
 Q[k]: (4,4) k
 R : (4,4)
    """

    def __init__(self, K: int = 8):
        self.K = K
        self.Pi = np.full((K, K), 1.0 / K)
        self.A  = [np.eye(4) for _ in range(K)]
        self.Q  = [0.1 * np.eye(4) for _ in range(K)]
        self.R  = 0.01 * np.eye(4)

    def fit(self, gng: GNG, obs_list: List[np.ndarray]):
        """ORCA demoGDBNEM-like offline fitting"""
        K = self.K
        transition_counts = np.zeros((K, K))
        data_per_mode: List[List[Tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(K)]

        for obs_seq in obs_list:
            if not (isinstance(obs_seq, np.ndarray) and obs_seq.ndim == 2
                    and obs_seq.shape[1] == 34):
                continue
            robot = obs_seq[:, :9]

            for p in range(5):
                s = 9 + p * 5
                ped = obs_seq[:, s:s + 5]
                valid_idx = np.where(~np.all(ped == 0, axis=1))[0]
                if len(valid_idx) < 2:
                    continue

                for ii in range(len(valid_idx) - 1):
                    t, t1 = valid_idx[ii], valid_idx[ii + 1]
                    if t1 - t > 2:
                        continue

                    def _feat(tidx):
                        rx, ry   = robot[tidx, 0], robot[tidx, 1]
                        rvx, rvy = robot[tidx, 2], robot[tidx, 3]
                        px_, py_ = ped[tidx, 0], ped[tidx, 1]
                        pvx, pvy = ped[tidx, 2], ped[tidx, 3]
                        drx, dry = px_ - rx, py_ - ry
                        return np.array([drx, dry, np.hypot(drx, dry) + 1e-6,
                                         pvx - rvx, pvy - rvy], dtype=np.float32)

                    k_t  = gng.assign_mode(_feat(t))
                    k_t1 = gng.assign_mode(_feat(t1))
                    transition_counts[k_t, k_t1] += 1
                    data_per_mode[k_t].append(
                        (ped[t, :4].astype(np.float64),
                         ped[t1, :4].astype(np.float64))
                    )

        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1, 1.0, row_sums)
        self.Pi = (transition_counts + 0.1) / (row_sums + K * 0.1)

        for k in range(K):
            pairs = data_per_mode[k]
            if len(pairs) < 5:
                A = np.eye(4)
                A[0, 2] = 0.25  # px += vx * dt (dt≈0.25s)
                A[1, 3] = 0.25  # py += vy * dt
                self.A[k] = A
                self.Q[k] = 0.1 * np.eye(4)
                continue

            X_prev = np.array([p[0] for p in pairs])  # (N, 4)
            X_curr = np.array([p[1] for p in pairs])  # (N, 4)

            XXT = X_prev.T @ X_prev + 1e-3 * np.eye(4)
            A_k = (X_curr.T @ X_prev) @ np.linalg.inv(XXT)

            u, sv, vt = np.linalg.svd(A_k)
            sv = np.clip(sv, 0.0, 1.5)
            self.A[k] = u @ np.diag(sv) @ vt

            residuals = X_curr - (self.A[k] @ X_prev.T).T
            self.Q[k] = (residuals.T @ residuals) / len(pairs) + 1e-4 * np.eye(4)

        h = -np.sum(self.Pi * np.log(self.Pi + 1e-10), axis=1).mean()
        print(f"[GDBN] : K={K}, ={h:.3f}")

    def save(self, path: str):
        d: dict = {'K': np.array(self.K), 'Pi': self.Pi, 'R': self.R}
        for k in range(self.K):
            d[f'A_{k}'] = self.A[k]
            d[f'Q_{k}'] = self.Q[k]
        np.savez(path, **d)

    def load(self, path: str):
        d = np.load(path)
        self.K  = int(d['K'])
        self.Pi = d['Pi']
        self.R  = d['R']
        self.A  = [d[f'A_{k}'] for k in range(self.K)]
        self.Q  = [d[f'Q_{k}'] for k in range(self.K)]

# ============================================================================ #
# ============================================================================ #

class PedestrianBeliefTracker:
    """
 Modified Markov Jump Particle Filter

 {(X_t^i, S_t^i, w_i)}

 KLDA = KL(λ(S_t) || α(S_t))
 α(S_t): = π_{t-1} @ Π
 λ(S_t): ≈ P(Z_t | S_t)
 KLDA
    """

    def __init__(
        self,
        gdbn: GDBN,
        n_particles: int = 50,
        rng: Optional[np.random.Generator] = None,
    ):
        self.gdbn = gdbn
        self.K = gdbn.K
        self.N = n_particles
        self.rng = rng or np.random.default_rng()

        self.particles_x: Optional[np.ndarray] = None  # (N, 4)
        self.particles_s: Optional[np.ndarray] = None  # (N,) int
        self.weights:     Optional[np.ndarray] = None

        self._R_inv = np.linalg.inv(gdbn.R)

    def reset(self, x0: Optional[np.ndarray] = None):
        N, K = self.N, self.K
        self.particles_s = self.rng.integers(0, K, size=N)
        if x0 is not None:
            noise = self.rng.normal(size=(N, 4)) * 0.05
            self.particles_x = np.tile(x0.astype(np.float64), (N, 1)) + noise
        else:
            self.particles_x = np.zeros((N, 4), dtype=np.float64)
        self.weights = np.full(N, 1.0 / N)

    def _marginal_s(self) -> np.ndarray:
        """ π(S_t)"""
        pi = np.zeros(self.K)
        np.add.at(pi, self.particles_s, self.weights)
        return pi / (pi.sum() + 1e-10)

    def step(self, z: np.ndarray) -> float:
        """
 -KLDA

 z: [px, py, vx, vy] ()
        """
        if self.particles_x is None:
            self.reset(x0=z)
            return 0.0

        N, K = self.N, self.K
        z = np.asarray(z, dtype=np.float64)

        pi_prev = self._marginal_s()
        alpha = pi_prev @ self.gdbn.Pi  # (K,) @ (K,K) -> (K,)
        alpha = np.maximum(alpha, 1e-10)
        alpha /= alpha.sum()

        cumpi = np.cumsum(self.gdbn.Pi[self.particles_s], axis=1)  # (N, K)
        u = self.rng.uniform(size=(N, 1))
        new_s = np.clip((u > cumpi).sum(axis=1), 0, K - 1).astype(int)

        new_x = np.empty((N, 4), dtype=np.float64)
        for k in range(K):
            idx = np.where(new_s == k)[0]
            if len(idx) == 0:
                continue
            mean_x = (self.gdbn.A[k] @ self.particles_x[idx].T).T
            noise = self.rng.multivariate_normal(
                np.zeros(4),
                self.gdbn.Q[k],
                size=len(idx),
            )
            new_x[idx] = mean_x + noise

        self.particles_s = new_s
        self.particles_x = new_x

        lambda_diag = np.zeros(K)
        x_global_mean = (self.particles_x * self.weights[:, None]).sum(axis=0)
        for k in range(K):
            idx = np.where(new_s == k)[0]
            if len(idx) > 0:
                w_k = self.weights[idx]
                w_k = w_k / (w_k.sum() + 1e-10)
                x_k = (self.particles_x[idx] * w_k[:, None]).sum(axis=0)
            else:
                x_k = self.gdbn.A[k] @ x_global_mean
            diff = z - x_k
            lambda_diag[k] = np.exp(
                -0.5 * float(diff @ self._R_inv @ diff)
            )
        lambda_diag = np.maximum(lambda_diag, 1e-10)
        lambda_diag /= lambda_diag.sum()

        # ---- Step 4: KLDA = KL(λ || α) ----
        klda = float(np.clip(
            np.sum(lambda_diag * np.log(lambda_diag / alpha)),
            0.0, 10.0
        ))

        diff_all = z[np.newaxis, :] - self.particles_x  # (N, 4)
        log_w = -0.5 * np.einsum('ni,ij,nj->n', diff_all, self._R_inv, diff_all)
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        w_sum = self.weights.sum()
        if w_sum < 1e-10:
            self.weights = np.full(N, 1.0 / N)
        else:
            self.weights /= w_sum

        n_eff = 1.0 / ((self.weights ** 2).sum())
        if n_eff < N / 2:
            cumsum = np.cumsum(self.weights)
            u_rs = (np.arange(N) + self.rng.uniform()) / N
            idx_rs = np.clip(np.searchsorted(cumsum, u_rs), 0, N - 1)
            self.particles_x = self.particles_x[idx_rs].copy()
            self.particles_s = self.particles_s[idx_rs].copy()
            self.weights = np.full(N, 1.0 / N)

        return klda

    def get_mode_distribution(self) -> np.ndarray:
        return self._marginal_s()

# ============================================================================ #
# ============================================================================ #

class GDBNIntegration:
    """
 Mamba-VLGDBNAPI

        gdbn = GDBNIntegration(K=8)
        gdbn.train_offline('orca_demos_seq.npz')
        gdbn.load('gdbn_params/')

        gdbn.reset()

        gdbn.update(state_34d)
        klda = gdbn.get_klda_risk()
        belief = gdbn.get_belief_features()
    """

    DEFAULT_PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'gdbn_params')

    def __init__(
        self,
        K: int = 8,
        n_particles: int = 50,
        params_dir: Optional[str] = None,
        max_peds: int = 5,
        klda_norm_clip: float = 5.0,
        random_seed: int = 42,
    ):
        self.K = K
        self.n_particles = n_particles
        self.max_peds = int(max_peds)
        self.klda_norm_clip = float(klda_norm_clip)
        self._rng = np.random.default_rng(int(random_seed))
        self.gng  = GNG(k_target=K)
        self.gdbn = GDBN(K=K)
        self._trackers: List[PedestrianBeliefTracker] = []
        self._n_peds = self.max_peds
        self._fitted = False
        self._last_klda: List[float] = []
        self._action_fitted = False
        self.B_action = [np.zeros((4, 2), dtype=np.float64) for _ in range(K)]
        self.action_residual_cov = [0.1 * np.eye(4) for _ in range(K)]

        load_dir = params_dir or self.DEFAULT_PARAMS_DIR
        self.params_dir = os.path.abspath(load_dir)
        if os.path.isdir(load_dir):
            try:
                self.load(load_dir)
            except Exception as e:
                print(f"[GDBN]  ({e}) train_offline() ")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #

    def train_offline(
        self,
        npz_path: str,
        save_dir: Optional[str] = None,
        gng_epochs: int = 3,
    ):
        """ORCA demoGNG + GDBN"""
        print(f"[GDBN] : {npz_path}")
        self.gng.fit(npz_path, n_epochs=gng_epochs)

        K_actual = self.gng.n_modes
        if K_actual != self.K:
            print(f"[GDBN] GNG K={K_actual}={self.K}")
            self.K = K_actual
            self.gdbn = GDBN(K=K_actual)

        data = np.load(npz_path, allow_pickle=True)
        obs_list = list(data['obs'])
        self.gdbn.fit(self.gng, obs_list)
        if 'act' in data.files:
            self._fit_action_model(obs_list, list(data['act']))
        self._fitted = True

        out_dir = save_dir or self.DEFAULT_PARAMS_DIR
        self.save(out_dir)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.params_dir = os.path.abspath(save_dir)
        self.gng.save(os.path.join(save_dir, 'gng.npz'))
        self.gdbn.save(os.path.join(save_dir, 'gdbn.npz'))
        if self._action_fitted:
            d = {'K': np.array(self.K)}
            for k in range(self.K):
                d[f'B_{k}'] = self.B_action[k]
                d[f'C_{k}'] = self.action_residual_cov[k]
            np.savez(os.path.join(save_dir, 'action_model.npz'), **d)
        print(f"[GDBN] : {save_dir}")

    def load(self, params_dir: str):
        self.params_dir = os.path.abspath(params_dir)
        self.gng.load(os.path.join(params_dir, 'gng.npz'))
        self.gdbn.load(os.path.join(params_dir, 'gdbn.npz'))
        self.K = self.gdbn.K
        self.B_action = [np.zeros((4, 2), dtype=np.float64) for _ in range(self.K)]
        self.action_residual_cov = [0.1 * np.eye(4) for _ in range(self.K)]
        action_path = os.path.join(params_dir, 'action_model.npz')
        if os.path.exists(action_path):
            d = np.load(action_path)
            for k in range(self.K):
                self.B_action[k] = d[f'B_{k}']
                self.action_residual_cov[k] = d[f'C_{k}']
            self._action_fitted = True
        self._fitted = True
        print(f"[GDBN] : K={self.K} {params_dir}")

    def train_action_model(self, npz_path: str, save_dir: Optional[str] = None):
        """Fit action-conditioned residual dynamics X_next = A_k X + B_k a + eps."""
        if not self._fitted:
            raise RuntimeError("[GDBN] train_action_model requires fitted GNG/GDBN first")
        data = np.load(npz_path, allow_pickle=True)
        if 'obs' not in data.files or 'act' not in data.files:
            raise ValueError(f"[GDBN] action model requires obs and act arrays: {npz_path}")
        self._fit_action_model(list(data['obs']), list(data['act']))
        self.save(save_dir or self.DEFAULT_PARAMS_DIR)

    def _fit_action_model(self, obs_list: List[np.ndarray], act_list: List[np.ndarray]):
        """Least-squares action coupling per mode."""
        pairs_x = [[] for _ in range(self.K)]
        pairs_y = [[] for _ in range(self.K)]

        for obs_seq, act_seq in zip(obs_list, act_list):
            if not (isinstance(obs_seq, np.ndarray) and obs_seq.ndim == 2 and obs_seq.shape[1] == 34):
                continue
            if not (isinstance(act_seq, np.ndarray) and act_seq.ndim == 2 and act_seq.shape[1] >= 2):
                continue
            T = min(len(obs_seq), len(act_seq))
            if T < 2:
                continue
            robot = obs_seq[:, :9]
            for p in range(self.max_peds):
                s = 9 + p * 5
                ped = obs_seq[:, s:s + 5]
                valid_idx = np.where(~np.all(ped[:T] == 0, axis=1))[0]
                if len(valid_idx) < 2:
                    continue
                for ii in range(len(valid_idx) - 1):
                    t, t1 = valid_idx[ii], valid_idx[ii + 1]
                    if t >= T - 1 or t1 >= T or t1 - t > 2:
                        continue
                    rx, ry = robot[t, 0], robot[t, 1]
                    rvx, rvy = robot[t, 2], robot[t, 3]
                    px, py, pvx, pvy = ped[t, :4]
                    feat = np.array([px - rx, py - ry, np.hypot(px - rx, py - ry) + 1e-6,
                                     pvx - rvx, pvy - rvy], dtype=np.float32)
                    k = self.gng.assign_mode(feat)
                    x = ped[t, :4].astype(np.float64)
                    y = ped[t1, :4].astype(np.float64) - (self.gdbn.A[k] @ x)
                    a = act_seq[t, :2].astype(np.float64)
                    pairs_x[k].append(a)
                    pairs_y[k].append(y)

        self.B_action = [np.zeros((4, 2), dtype=np.float64) for _ in range(self.K)]
        self.action_residual_cov = [0.1 * np.eye(4) for _ in range(self.K)]
        for k in range(self.K):
            if len(pairs_x[k]) < 8:
                continue
            A_mat = np.asarray(pairs_x[k], dtype=np.float64)  # [N, 2]
            Y_mat = np.asarray(pairs_y[k], dtype=np.float64)  # [N, 4]
            reg = 1e-3 * np.eye(2)
            B_t = np.linalg.solve(A_mat.T @ A_mat + reg, A_mat.T @ Y_mat)  # [2, 4]
            self.B_action[k] = B_t.T
            residual = Y_mat - A_mat @ B_t
            cov = (residual.T @ residual) / max(1, len(residual)) + 1e-4 * np.eye(4)
            self.action_residual_cov[k] = cov
        self._action_fitted = True
        print(f"[GDBN-ACTION] fitted action-conditioned residual model")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #

    def reset(self, n_peds: Optional[int] = None):
        """episodetracker"""
        self._n_peds = int(n_peds or self.max_peds)
        self._trackers = [
            PedestrianBeliefTracker(
                self.gdbn,
                n_particles=self.n_particles,
                rng=np.random.default_rng(
                    int(self._rng.integers(0, np.iinfo(np.int32).max))
                ),
            )
            for _ in range(self._n_peds)
        ]
        self._last_klda = [0.0] * self._n_peds

    def update(self, state_34d: np.ndarray) -> List[float]:
        """
 beliefKLDA

 state_34d: (34,) = robot(9) + 5×ped(5)
 : KLDA _last_klda
        """
        if not self._fitted:
            return [0.0] * self._n_peds
        if not self._trackers:
            self.reset()

        klda_list = []
        for p in range(min(self._n_peds, self.max_peds)):
            s = 9 + p * 5
            ped = state_34d[s:s + 5]
            if p >= len(self._trackers) or np.all(ped == 0):
                klda_list.append(0.0)
                continue
            obs_x = ped[:4].astype(np.float64)
            klda = self._trackers[p].step(obs_x)
            klda_list.append(klda)

        self._last_klda = klda_list
        return klda_list

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #

    def get_klda_risk(self) -> float:
        """
 KLDA

 predict_sarl_style
        """
        if not self._last_klda:
            return 0.0
        return float(max(self._last_klda))

    def get_belief_features(self) -> np.ndarray:
        """
 K

 : (5, K) —
 EnhancedSpatialEncoder token
        """
        features = np.zeros((self.max_peds, self.K), dtype=np.float32)
        if not self._fitted:
            return features
        for p, tracker in enumerate(self._trackers[:self.max_peds]):
            if tracker.particles_x is not None:
                features[p] = tracker.get_mode_distribution().astype(np.float32)
        return features

    def get_all_klda(self) -> List[float]:
        """KLDA/"""
        return list(self._last_klda)

    def get_per_ped_belief_vec(self) -> np.ndarray:
        """Per-pedestrian belief vector for token augmentation.

        Returns: (max_peds, K+2) float32
            columns [:K]  — normalized mode probabilities (sums to 1 per ped)
            column  [K]   — mode entropy normalized to [0, 1]
            column  [K+1] — KLDA normalized to [0, 1]
        """
        out_dim = self.K + 2
        features = np.zeros((self.max_peds, out_dim), dtype=np.float32)
        if not self._fitted:
            features[:, :self.K] = 1.0 / self.K  # uniform prior
            return features
        klda_clip = max(float(self.klda_norm_clip), 1e-6)
        log_K = float(np.log(max(self.K, 2)))
        for p, tracker in enumerate(self._trackers[:self.max_peds]):
            klda = float(self._last_klda[p]) if p < len(self._last_klda) else 0.0
            if tracker.particles_x is not None:
                mode_probs = tracker.get_mode_distribution().astype(np.float32)
            else:
                mode_probs = np.ones(self.K, dtype=np.float32) / self.K
            mode_probs = np.maximum(mode_probs, 1e-10)
            mode_probs /= mode_probs.sum()
            entropy = float(-np.sum(mode_probs * np.log(mode_probs)) / log_K)
            features[p, :self.K] = mode_probs
            features[p, self.K] = float(np.clip(entropy, 0.0, 1.0))
            features[p, self.K + 1] = float(np.clip(klda, 0.0, klda_clip) / klda_clip)
        return features

    @staticmethod
    def _clearance_risk(clearances: np.ndarray, safe_distance: float) -> np.ndarray:
        """Continuous collision-proximity risk for Bayesian action scoring."""
        margin = max(float(safe_distance), 1e-6)
        clearances = np.asarray(clearances, dtype=np.float64)
        positive_risk = np.exp(-np.maximum(clearances, 0.0) / margin)
        return np.clip(np.where(clearances <= 0.0, 1.0, positive_risk), 0.0, 1.0)

    @staticmethod
    def _weighted_upper_cvar(
        values: np.ndarray,
        probabilities: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Return the probability-weighted mean over the upper risk tail."""
        values = np.asarray(values, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= np.maximum(
            probabilities.sum(axis=-1, keepdims=True), 1e-12
        )

        tail_mass = max(1.0 - float(np.clip(alpha, 0.0, 0.999999)), 1e-6)
        order = np.argsort(values, axis=-1)[..., ::-1]
        ordered_values = np.take_along_axis(values, order, axis=-1)
        ordered_probabilities = np.take_along_axis(
            probabilities, order, axis=-1
        )
        cumulative_before = (
            np.cumsum(ordered_probabilities, axis=-1)
            - ordered_probabilities
        )
        included_mass = np.clip(
            tail_mass - cumulative_before,
            0.0,
            ordered_probabilities,
        )
        return (
            (ordered_values * included_mass).sum(axis=-1) / tail_mass
        )

    def predict_action_rollout(
        self,
        state_34d: np.ndarray,
        action_xy,
        horizon: int = 1,
        dt: float = 0.25,
        robot_radius: float = 0.3,
        human_radius_default: float = 0.3,
        safe_distance: float = 0.2,
    ) -> dict:
        """Action-conditioned Bayesian rollout without mutating online trackers.

        Returns risk/entropy/free-energy ingredients and predicted belief vectors
        for one candidate robot action.
        """
        out_dim = self.K + 2
        belief_vec = np.zeros((self.max_peds, out_dim), dtype=np.float32)
        if not self._fitted:
            belief_vec[:, :self.K] = 1.0 / self.K
            return {
                'risk': 0.0, 'entropy': 0.0, 'klda': 0.0,
                'min_clearance': float('inf'), 'belief_vec': belief_vec,
                'action_fitted': False,
            }

        state_34d = np.asarray(state_34d, dtype=np.float64).reshape(-1)
        if state_34d.shape[0] < 34:
            state_34d = np.pad(state_34d, (0, 34 - state_34d.shape[0]))
        else:
            state_34d = state_34d[:34]
        action_xy = np.asarray(action_xy, dtype=np.float64).reshape(-1)[:2]

        robot_px, robot_py = float(state_34d[0]), float(state_34d[1])
        robot_radius = float(state_34d[4]) if state_34d[4] > 0 else float(robot_radius)

        total_risk = 0.0
        total_entropy = 0.0
        total_klda = 0.0
        total_epistemic = 0.0
        min_clearance = float('inf')
        valid_peds = 0
        log_K = float(np.log(max(self.K, 2)))
        klda_clip = max(float(self.klda_norm_clip), 1e-6)

        for p in range(self.max_peds):
            s = 9 + p * 5
            ped = state_34d[s:s + 5]
            if ped.shape[0] < 5 or np.all(ped == 0):
                continue
            valid_peds += 1
            x = ped[:4].astype(np.float64)
            human_radius = float(ped[4]) if ped[4] > 0 else human_radius_default

            if p < len(self._trackers) and self._trackers[p].particles_x is not None:
                pi = self._trackers[p].get_mode_distribution().astype(np.float64)
            else:
                feat = self._ped_feature(state_34d, p)
                pi = np.zeros(self.K, dtype=np.float64)
                pi[self.gng.assign_mode(feat)] = 1.0

            p_safe_0 = np.maximum(pi, 1e-8)
            entropy_before = float(-np.sum(p_safe_0 * np.log(p_safe_0)) / log_K)

            robot_future = np.array([robot_px, robot_py], dtype=np.float64)
            mode_dist = pi.copy()
            x_mean = x.copy()
            ped_risk = 0.0
            ped_min_clearance = float('inf')
            klda_acc = 0.0

            for _ in range(max(1, int(horizon))):
                base_alpha = mode_dist @ self.gdbn.Pi
                base_alpha = np.maximum(base_alpha, 1e-10)
                base_alpha /= base_alpha.sum()

                pred_by_mode = np.zeros((self.K, 4), dtype=np.float64)
                for k in range(self.K):
                    pred_by_mode[k] = self.gdbn.A[k] @ x_mean + self.B_action[k] @ action_xy

                x_mean = (base_alpha[:, None] * pred_by_mode).sum(axis=0)
                robot_future = robot_future + action_xy * float(dt)

                clearances = (
                    np.linalg.norm(pred_by_mode[:, :2] - robot_future[None, :], axis=1)
                    - robot_radius - human_radius
                )
                risk_by_mode = self._clearance_risk(clearances, safe_distance)
                ped_risk = max(ped_risk, float(np.sum(base_alpha * risk_by_mode)))
                ped_min_clearance = min(ped_min_clearance, float(np.min(clearances)))

                if self._action_fitted:
                    action_mag = float(np.linalg.norm(action_xy))
                    action_shift = np.array([
                        np.linalg.norm(self.B_action[k] @ action_xy) for k in range(self.K)
                    ], dtype=np.float64)
                    lam = np.maximum(action_shift + 1e-6, 1e-10)
                    lam = lam / lam.sum()
                    klda_acc += float(np.sum(lam * np.log(lam / base_alpha))) * min(1.0, action_mag)

                mode_dist = base_alpha

            mode_dist = np.maximum(mode_dist, 1e-10)
            mode_dist /= mode_dist.sum()
            entropy = float(-np.sum(mode_dist * np.log(mode_dist)) / log_K)
            epistemic = max(0.0, entropy_before - entropy)

            belief_vec[p, :self.K] = mode_dist.astype(np.float32)
            belief_vec[p, self.K] = float(np.clip(entropy, 0.0, 1.0))
            belief_vec[p, self.K + 1] = float(np.clip(klda_acc, 0.0, klda_clip) / klda_clip)

            total_risk += ped_risk
            total_entropy += entropy
            total_klda += klda_acc
            total_epistemic += epistemic
            min_clearance = min(min_clearance, ped_min_clearance)

        denom = max(1, valid_peds)
        return {
            'risk': float(total_risk / denom),
            'entropy': float(total_entropy / denom),
            'klda': float(total_klda / denom),
            'epistemic_value': float(total_epistemic / denom),
            'min_clearance': float(min_clearance),
            'belief_vec': belief_vec,
            'action_fitted': bool(self._action_fitted),
        }

    def predict_action_rollout_from_belief(
        self,
        state_34d: np.ndarray,
        action_xy,
        initial_belief_vecs: np.ndarray,
        horizon: int = 1,
        dt: float = 0.25,
        robot_radius: float = 0.3,
        human_radius_default: float = 0.3,
        safe_distance: float = 0.2,
    ) -> dict:
        """Rollout using stored belief vectors instead of live trackers.

        Bypasses self._trackers to avoid pedestrian ordering mismatches during
        training (token slots are TTC-sorted; tracker slots are identity-sorted).
        initial_belief_vecs[:, :K] are mode probs; the rest are ignored here.
        """
        out_dim = self.K + 2
        belief_vec = np.zeros((self.max_peds, out_dim), dtype=np.float32)
        if not self._fitted:
            belief_vec[:, :self.K] = 1.0 / self.K
            return {
                'risk': 0.0, 'entropy': 0.0, 'klda': 0.0,
                'epistemic_value': 0.0,
                'min_clearance': float('inf'), 'belief_vec': belief_vec,
                'action_fitted': False,
            }

        state_34d = np.asarray(state_34d, dtype=np.float64).reshape(-1)
        if state_34d.shape[0] < 34:
            state_34d = np.pad(state_34d, (0, 34 - state_34d.shape[0]))
        else:
            state_34d = state_34d[:34]
        action_xy = np.asarray(action_xy, dtype=np.float64).reshape(-1)[:2]

        robot_px, robot_py = float(state_34d[0]), float(state_34d[1])
        robot_radius = float(state_34d[4]) if state_34d[4] > 0 else float(robot_radius)

        total_risk = 0.0
        total_entropy = 0.0
        total_klda = 0.0
        total_epistemic = 0.0
        min_clearance = float('inf')
        valid_peds = 0
        log_K = float(np.log(max(self.K, 2)))
        klda_clip = max(float(self.klda_norm_clip), 1e-6)

        for p in range(self.max_peds):
            s = 9 + p * 5
            ped = state_34d[s:s + 5]
            if ped.shape[0] < 5 or np.all(ped == 0):
                continue
            valid_peds += 1
            x = ped[:4].astype(np.float64)
            human_radius = float(ped[4]) if ped[4] > 0 else human_radius_default

            # Use stored mode probs from token buffer (columns [:K])
            if (initial_belief_vecs is not None
                    and p < initial_belief_vecs.shape[0]
                    and initial_belief_vecs.shape[1] >= self.K):
                pi = initial_belief_vecs[p, :self.K].astype(np.float64)
                pi = np.maximum(pi, 1e-10)
                pi /= pi.sum()
            else:
                pi = np.ones(self.K, dtype=np.float64) / self.K

            p_safe_0 = np.maximum(pi, 1e-8)
            entropy_before = float(-np.sum(p_safe_0 * np.log(p_safe_0)) / log_K)

            robot_future = np.array([robot_px, robot_py], dtype=np.float64)
            mode_dist = pi.copy()
            ped_risk = 0.0
            ped_min_clearance = float('inf')
            klda_acc = 0.0

            for _ in range(max(1, int(horizon))):
                base_alpha = mode_dist @ self.gdbn.Pi
                base_alpha = np.maximum(base_alpha, 1e-10)
                base_alpha /= base_alpha.sum()

                pred_by_mode = np.zeros((self.K, 4), dtype=np.float64)
                for k in range(self.K):
                    pred_by_mode[k] = self.gdbn.A[k] @ x + self.B_action[k] @ action_xy

                x = (base_alpha[:, None] * pred_by_mode).sum(axis=0)
                robot_future = robot_future + action_xy * float(dt)

                clearances = (
                    np.linalg.norm(pred_by_mode[:, :2] - robot_future[None, :], axis=1)
                    - robot_radius - human_radius
                )
                risk_by_mode = self._clearance_risk(clearances, safe_distance)
                ped_risk = max(ped_risk, float(np.sum(base_alpha * risk_by_mode)))
                ped_min_clearance = min(ped_min_clearance, float(np.min(clearances)))

                if self._action_fitted:
                    action_mag = float(np.linalg.norm(action_xy))
                    action_shift = np.array([
                        np.linalg.norm(self.B_action[k] @ action_xy) for k in range(self.K)
                    ], dtype=np.float64)
                    lam = np.maximum(action_shift + 1e-6, 1e-10)
                    lam = lam / lam.sum()
                    klda_acc += float(np.sum(lam * np.log(lam / base_alpha))) * min(1.0, action_mag)

                mode_dist = base_alpha

            mode_dist = np.maximum(mode_dist, 1e-10)
            mode_dist /= mode_dist.sum()
            entropy = float(-np.sum(mode_dist * np.log(mode_dist)) / log_K)
            epistemic = max(0.0, entropy_before - entropy)

            belief_vec[p, :self.K] = mode_dist.astype(np.float32)
            belief_vec[p, self.K] = float(np.clip(entropy, 0.0, 1.0))
            belief_vec[p, self.K + 1] = float(np.clip(klda_acc, 0.0, klda_clip) / klda_clip)

            total_risk += ped_risk
            total_entropy += entropy
            total_klda += klda_acc
            total_epistemic += epistemic
            min_clearance = min(min_clearance, ped_min_clearance)

        denom = max(1, valid_peds)
        return {
            'risk': float(total_risk / denom),
            'entropy': float(total_entropy / denom),
            'klda': float(total_klda / denom),
            'epistemic_value': float(total_epistemic / denom),
            'min_clearance': float(min_clearance),
            'belief_vec': belief_vec,
            'action_fitted': bool(self._action_fitted),
        }

    def predict_action_rollout_batch(
        self,
        states_34d: np.ndarray,
        actions_xy: np.ndarray,
        belief_vecs: Optional[np.ndarray] = None,
        horizon: int = 1,
        dt: float = 0.25,
        safe_distance: float = 0.2,
        human_radius_default: float = 0.3,
        cvar_alpha: float = 0.80,
        pedestrian_aggregation: str = "mean",
    ) -> dict:
        """Vectorized batch rollout for sarl_style_update (eliminates serial loop).

        Args:
            states_34d: (B, 34) states reconstructed from last token frame.
            actions_xy: (B, 2) continuous robot velocity actions.
            belief_vecs: (B, max_peds, K+2) from token buffer dims [belief_start:].
                         Columns [:K] are mode probs. If None → uniform init.
        Returns:
            dict with 'risk', 'entropy', 'klda', 'epistemic_value': (B,) float64 arrays.
        """
        if not self._fitted:
            B = max(1, len(states_34d))
            zero = np.zeros(B, dtype=np.float64)
            return {
                'risk': zero,
                'tail_risk': zero,
                'entropy': zero,
                'klda': zero,
                'epistemic_value': zero,
            }

        states_34d = np.asarray(states_34d, dtype=np.float64)
        actions_xy = np.asarray(actions_xy, dtype=np.float64)
        B = states_34d.shape[0]
        P = self.max_peds
        K = self.K

        # Pedestrian states: (B, P, 5)
        ped_states = np.zeros((B, P, 5), dtype=np.float64)
        for p in range(P):
            s = 9 + p * 5
            if s + 5 <= states_34d.shape[1]:
                ped_states[:, p, :] = states_34d[:, s:s + 5]

        valid_mask = ~np.all(ped_states == 0, axis=-1)  # (B, P)
        valid_count = np.maximum(valid_mask.sum(axis=1).astype(np.float64), 1.0)

        robot_r = np.where(states_34d[:, 4] > 0, states_34d[:, 4], 0.3)
        human_r = np.where(ped_states[:, :, 4] > 0, ped_states[:, :, 4], human_radius_default)
        combined_r = robot_r[:, None] + human_r  # (B, P)

        # Initialize mode distributions from stored belief tokens (first K columns = mode probs)
        if (belief_vecs is not None
                and belief_vecs.ndim == 3
                and belief_vecs.shape[1] == P
                and belief_vecs.shape[2] >= K):
            mode_dists = np.asarray(belief_vecs[:, :, :K], dtype=np.float64)
            mode_dists = np.maximum(mode_dists, 1e-10)
            mode_dists /= mode_dists.sum(axis=-1, keepdims=True)
        else:
            mode_dists = np.ones((B, P, K), dtype=np.float64) / K

        log_K = float(np.log(max(K, 2)))
        md0 = np.maximum(mode_dists, 1e-10)
        entropy_before = -(md0 * np.log(md0)).sum(axis=-1) / log_K  # (B, P)

        A_stack = np.stack(self.gdbn.A, axis=0)   # (K, 4, 4)
        B_stack = np.stack(self.B_action, axis=0)  # (K, 4, 2)
        Pi = self.gdbn.Pi

        x_batch = ped_states[:, :, :4].copy()
        robot_future = states_34d[:, :2].copy()

        # B[k] @ a[b] for all k, b: (B, K, 4)
        B_a = np.einsum('kij,bj->bki', B_stack, actions_xy)

        risk_max_bp = np.zeros((B, P), dtype=np.float64)
        tail_risk_max_bp = np.zeros((B, P), dtype=np.float64)
        klda_sum_bp = np.zeros((B, P), dtype=np.float64)
        min_clear_bp = np.full((B, P), np.inf, dtype=np.float64)

        if self._action_fitted:
            action_shift = np.linalg.norm(B_a, axis=-1)  # (B, K)
            lam_base = np.maximum(action_shift + 1e-6, 1e-10)
            lam_base /= lam_base.sum(axis=-1, keepdims=True)
            action_mag = np.linalg.norm(actions_xy, axis=-1)  # (B,)

        for _ in range(max(1, int(horizon))):
            # Transition: mode_dists(B,P,K) @ Pi(K,K) → (B,P,K)
            base_alpha = mode_dists @ Pi
            base_alpha = np.maximum(base_alpha, 1e-10)
            base_alpha /= base_alpha.sum(axis=-1, keepdims=True)

            # A[k]@x[b,p] for all k,b,p: (B,P,K,4)
            A_x = np.einsum('kij,bpj->bpki', A_stack, x_batch)
            pred_by_mode = A_x + B_a[:, None, :, :]  # (B,P,K,4)

            x_batch = (base_alpha[:, :, :, None] * pred_by_mode).sum(axis=2)
            robot_future = robot_future + actions_xy * dt

            pred_pos = pred_by_mode[:, :, :, :2]
            dist = np.linalg.norm(pred_pos - robot_future[:, None, None, :], axis=-1)
            clearances = dist - combined_r[:, :, None]

            risk_by_mode = self._clearance_risk(clearances, safe_distance)
            ped_risk = (base_alpha * risk_by_mode).sum(axis=-1)  # (B, P)
            ped_tail_risk = self._weighted_upper_cvar(
                risk_by_mode,
                base_alpha,
                cvar_alpha,
            )
            risk_max_bp = np.maximum(risk_max_bp, ped_risk)
            tail_risk_max_bp = np.maximum(
                tail_risk_max_bp, ped_tail_risk
            )
            min_clear_bp = np.minimum(min_clear_bp, clearances.min(axis=-1))

            if self._action_fitted:
                lam_bp = lam_base[:, None, :]  # (B, 1, K)
                klda_step = (lam_bp * np.log(lam_bp / np.maximum(base_alpha, 1e-10))).sum(axis=-1)
                klda_sum_bp += klda_step * np.minimum(action_mag[:, None], 1.0)

            mode_dists = base_alpha

        mode_dists = np.maximum(mode_dists, 1e-10)
        mode_dists /= mode_dists.sum(axis=-1, keepdims=True)
        entropy_after = -(mode_dists * np.log(mode_dists)).sum(axis=-1) / log_K  # (B, P)

        epistemic_bp = np.maximum(entropy_before - entropy_after, 0.0)
        klda_clip = max(float(self.klda_norm_clip), 1e-6)
        klda_clipped = np.clip(klda_sum_bp, 0.0, klda_clip)

        vf = valid_mask.astype(np.float64)
        if pedestrian_aggregation not in {"mean", "max"}:
            raise ValueError(
                "pedestrian_aggregation must be 'mean' or 'max'"
            )
        if pedestrian_aggregation == "max":
            invalid_fill = np.full_like(risk_max_bp, -np.inf)
            aggregate_risk = np.where(
                valid_mask,
                risk_max_bp,
                invalid_fill,
            ).max(axis=1)
            aggregate_tail_risk = np.where(
                valid_mask,
                tail_risk_max_bp,
                invalid_fill,
            ).max(axis=1)
            aggregate_risk = np.where(
                valid_mask.any(axis=1),
                aggregate_risk,
                0.0,
            )
            aggregate_tail_risk = np.where(
                valid_mask.any(axis=1),
                aggregate_tail_risk,
                0.0,
            )
        else:
            aggregate_risk = (
                (risk_max_bp * vf).sum(axis=1) / valid_count
            )
            aggregate_tail_risk = (
                (tail_risk_max_bp * vf).sum(axis=1) / valid_count
            )
        return {
            'risk':            aggregate_risk,
            'tail_risk':       aggregate_tail_risk,
            'entropy':         (entropy_after * vf).sum(axis=1) / valid_count,
            'klda':            (klda_clipped * vf).sum(axis=1) / valid_count,
            'epistemic_value': (epistemic_bp  * vf).sum(axis=1) / valid_count,
            'risk_by_ped':     risk_max_bp,
            'tail_risk_by_ped': tail_risk_max_bp,
            'valid_mask':      valid_mask,
        }

    def train_incremental(
        self,
        obs_seqs: List[np.ndarray],
        act_seqs: List[np.ndarray],
        alpha: float = 0.05,
        min_count: int = 5,
    ) -> bool:
        """EMA update of B_action[k] from new RL experience.

        Keeps the action-coupling model current as the robot explores new
        movement patterns. A_k (base pedestrian dynamics) is held fixed
        because pedestrian locomotion is independent of robot policy.

        Args:
            obs_seqs: list of (T, 34) state sequences from RL episodes.
            act_seqs: list of (T, 2+) continuous action sequences.
            alpha: EMA blend coefficient (fraction of new estimate to accept).
            min_count: minimum transitions per mode required for an update.
        Returns:
            True if at least one mode matrix was updated.
        """
        if not self._fitted or not self._action_fitted:
            return False

        pairs_x = [[] for _ in range(self.K)]
        pairs_y = [[] for _ in range(self.K)]

        for obs_seq, act_seq in zip(obs_seqs, act_seqs):
            if not (isinstance(obs_seq, np.ndarray) and obs_seq.ndim == 2
                    and obs_seq.shape[1] == 34):
                continue
            act_arr = np.asarray(act_seq, dtype=np.float64)
            if act_arr.ndim != 2 or act_arr.shape[1] < 2:
                continue
            T = min(len(obs_seq), len(act_arr))
            if T < 2:
                continue
            robot = obs_seq[:, :9]
            for p in range(self.max_peds):
                s = 9 + p * 5
                ped = obs_seq[:, s:s + 5]
                valid_idx = np.where(~np.all(ped[:T] == 0, axis=1))[0]
                if len(valid_idx) < 2:
                    continue
                for ii in range(len(valid_idx) - 1):
                    t, t1 = valid_idx[ii], valid_idx[ii + 1]
                    if t >= T - 1 or t1 >= T or t1 - t > 2:
                        continue
                    rx, ry = robot[t, 0], robot[t, 1]
                    rvx, rvy = robot[t, 2], robot[t, 3]
                    px_t, py_t, pvx_t, pvy_t = ped[t, :4]
                    feat = np.array([
                        px_t - rx, py_t - ry,
                        np.hypot(px_t - rx, py_t - ry) + 1e-6,
                        pvx_t - rvx, pvy_t - rvy,
                    ], dtype=np.float32)
                    k = self.gng.assign_mode(feat)
                    x = ped[t, :4].astype(np.float64)
                    y = ped[t1, :4].astype(np.float64) - (self.gdbn.A[k] @ x)
                    pairs_x[k].append(act_arr[t, :2])
                    pairs_y[k].append(y)

        any_updated = False
        for k in range(self.K):
            if len(pairs_x[k]) < min_count:
                continue
            A_mat = np.asarray(pairs_x[k], dtype=np.float64)  # (N, 2)
            Y_mat = np.asarray(pairs_y[k], dtype=np.float64)  # (N, 4)
            reg = 1e-3 * np.eye(2)
            try:
                B_t_new = np.linalg.solve(A_mat.T @ A_mat + reg, A_mat.T @ Y_mat)
                self.B_action[k] = (1.0 - alpha) * self.B_action[k] + alpha * B_t_new.T
                any_updated = True
            except np.linalg.LinAlgError:
                continue

        return any_updated

    def _ped_feature(self, state_34d: np.ndarray, p: int) -> np.ndarray:
        s = 9 + p * 5
        ped = state_34d[s:s + 5]
        robot = state_34d[:9]
        rel_x = ped[0] - robot[0]
        rel_y = ped[1] - robot[1]
        return np.array([
            rel_x, rel_y, np.hypot(rel_x, rel_y) + 1e-6,
            ped[2] - robot[2], ped[3] - robot[3]
        ], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def action_fitted(self) -> bool:
        return self._action_fitted
