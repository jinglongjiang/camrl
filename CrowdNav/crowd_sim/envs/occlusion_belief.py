"""Geometric occlusion and a dynamic Bayesian occupancy belief.

Order 17. Five modes share one implementation so that the four experimental
arms differ ONLY in which observation the policy is handed:

    off             the module is inert; the legacy fully-observable path runs
                    untouched and must stay bit-identical
    gt              ground-truth pedestrians (the oracle arm's upper bound)
    sensor          only pedestrians the robot can actually see
    deterministic   sensor plus a decaying memory of where a pedestrian was
                    last seen, propagated by constant velocity -- no belief,
                    no uncertainty, the ablation that isolates "is a posterior
                    doing anything a simple memory could not"
    bayes           sensor plus a recursive Bernoulli occupancy posterior over
                    the occluded region

The geometry is ported from PaS_CrowdNav (MIT), files generateLabelGrid.py and
generateSensorGrid.py: shadow polygons cast by the tangent lines from the robot
to each pedestrian circle, nearest first, plus a field-of-view disc. The numba
decorators, the VAE, the PPO stack and grid_utils are deliberately NOT copied;
the three geometric primitives are reimplemented here in plain numpy so the
module has no new dependency.

Leakage contract, enforced by construction rather than by convention:
`update()` splits the world once into visible and occluded, and the belief
update is fed ONLY the sensor grid and its own previous prediction. A hidden
pedestrian's position, velocity and id are never read on the bayes or sensor
path. Ground truth is reachable exclusively through `oracle_entities()`, which
the student policy must never call.
"""
import numpy as np
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class BeliefEntity:
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    p_exist: float
    uncertainty: float
    visible: float
    hidden: float
    id: int

    def __post_init__(self):
        values = (self.px, self.py, self.vx, self.vy, self.radius,
                  self.p_exist, self.uncertainty, self.visible, self.hidden)
        if not np.isfinite(values).all() or self.radius < 0:
            raise ValueError('Invalid belief entity')
        if not (0 <= self.p_exist <= 1 and 0 <= self.uncertainty <= 1):
            raise ValueError('Invalid belief confidence')


@dataclass(frozen=True)
class BeliefSnapshot:
    frame_index: int
    time_step: float
    entities: Tuple[BeliefEntity, ...]
    coordinate_frame: str = 'world_xy_velocity'
    probability_semantics: str = 'occupancy_peak_per_entity_not_normalized_across_entities'
    uncertainty_semantics: str = 'normalized_spatial_spread_not_covariance'

    def entity_dicts(self):
        return [asdict(entity) for entity in self.entities]

MODES = ("off", "sensor", "deterministic", "bayes", "gt", "oracle_belief")
SCHEMA_VERSION = 1


# ----------------------------------------------------------- primitives -----
def _point_in_circle(mesh_x, mesh_y, center, radius):
    return (mesh_x - center[0]) ** 2 + (mesh_y - center[1]) ** 2 <= radius ** 2


def _line_extend(x0, y0, x1, y1, x_target):
    """y of the ray through (x0,y0)->(x1,y1) at x_target; PaS's linefunction."""
    if abs(x1 - x0) < 1e-12:
        return y1 + (1e6 if y1 >= y0 else -1e6)
    return y0 + (y1 - y0) / (x1 - x0) * (x_target - x0)


def _points_in_polygon(px, py, poly):
    """Vectorised even-odd ray casting. Replaces PaS's numba routine; same
    result, no numba dependency."""
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (py - yi) / np.where(np.abs(yj - yi) < 1e-12,
                                                    1e-12, (yj - yi)) + xi
        inside ^= cond & (px < xint)
        j = i
    return inside


class OcclusionBelief:
    """Per-episode occlusion geometry and occupancy belief.

    Grids are robot-centred and rebuilt each step, so nothing about the map is
    remembered; only the belief over occupancy persists across steps, which is
    the whole point of the module.
    """

    def __init__(self, mode="off", grid_resolution=0.25, grid_extent=5.0,
                 fov_radius=5.0, max_entities=5, p_prior=0.05, p_hit=0.85,
                 p_miss=0.01, decay=0.90, diffuse=0.35, p_report=0.30,
                 vmax=1.0, time_step=0.25):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.res = float(grid_resolution)
        self.extent = float(grid_extent)
        self.fov_radius = float(fov_radius)
        self.max_entities = int(max_entities)
        # Bernoulli filter parameters, all in probability space; converted to
        # log-odds internally so the update is an addition
        self.p_prior = float(p_prior)
        # p_miss is P(occupied | observed free) and p_hit is P(occupied |
        # observed occupied). Evidence only points the right way when
        # p_miss < p_prior < p_hit; with p_miss above the prior, seeing a cell
        # empty would RAISE its occupancy probability. Caught by the unit test,
        # so it fails loudly here instead of quietly inverting the filter.
        if not (p_miss < p_prior < p_hit):
            raise ValueError(
                f"require p_miss < p_prior < p_hit, got {p_miss} / {p_prior} / {p_hit}")
        self.l_hit = np.log(p_hit / (1.0 - p_hit))
        self.l_miss = np.log(p_miss / (1.0 - p_miss))
        self.l_prior = np.log(p_prior / (1.0 - p_prior))
        self.decay = float(decay)
        self.diffuse = float(diffuse)
        self.p_report = float(p_report)
        self.vmax = float(vmax)
        self.dt = float(time_step)
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        self._frame_index = -1
        self._snapshot = BeliefSnapshot(-1, self.dt, ())
        self.logodds = None          # belief over the local grid, log-odds
        self._grid_shape = None
        self._prev_modes = []        # for velocity estimation of belief modes
        self.visible_ids = []
        self.occluded_ids = []
        self.sensor_grid = None
        self.label_grid = None
        self._mesh = None
        self._last_seen = {}         # id -> (px, py, vx, vy, radius, age)
        self._ever_seen = set()
        self._robot_xy = np.zeros(2)
        self._robot_radius = 0.0

    @property
    def enabled(self):
        return self.mode != "off"

    # ---------------------------------------------------------- geometry
    def _build_mesh(self, robot_xy):
        n = int(round(2 * self.extent / self.res))
        xs = robot_xy[0] - self.extent + self.res * (np.arange(n) + 0.5)
        ys = robot_xy[1] - self.extent + self.res * (np.arange(n) + 0.5)
        mx, my = np.meshgrid(xs, ys)
        return mx, my

    def _reproject_belief(self, old_mesh):
        """Keep the posterior fixed in world coordinates as the robot moves."""
        if self.logodds is None or old_mesh is None or self._mesh is None:
            return

        old_mx, old_my = old_mesh
        new_mx, new_my = self._mesh
        old_p = 1.0 / (1.0 + np.exp(-self.logodds))
        rows, cols = old_p.shape
        fx = (new_mx - old_mx[0, 0]) / self.res
        fy = (new_my - old_my[0, 0]) / self.res
        c0 = np.floor(fx).astype(np.int64)
        r0 = np.floor(fy).astype(np.int64)
        wx = fx - c0
        wy = fy - r0

        result = np.full(new_mx.shape, self.p_prior, dtype=np.float64)
        valid = (r0 >= 0) & (r0 < rows - 1) & (c0 >= 0) & (c0 < cols - 1)
        if np.any(valid):
            rv, cv = r0[valid], c0[valid]
            ax, ay = wx[valid], wy[valid]
            result[valid] = (
                (1.0 - ax) * (1.0 - ay) * old_p[rv, cv]
                + ax * (1.0 - ay) * old_p[rv, cv + 1]
                + (1.0 - ax) * ay * old_p[rv + 1, cv]
                + ax * ay * old_p[rv + 1, cv + 1]
            )
        result = np.clip(result, 1e-4, 1.0 - 1e-4)
        self.logodds = np.log(result / (1.0 - result))

    def _label_and_sensor(self, robot_xy, humans):
        """Ground-truth occupancy, then what the robot can actually see.

        Ported from PaS generateLabelGrid / generateSensorGrid: shadow polygons
        from nearest pedestrian outward, then the field-of-view disc, then a
        pedestrian whose cells are entirely shadowed counts as occluded.
        """
        mx, my = self._build_mesh(robot_xy)
        occ = np.zeros(mx.shape, dtype=np.float32)
        ids = np.full(mx.shape, -1, dtype=np.int32)
        for h in humans:
            m = _point_in_circle(mx, my, (h["px"], h["py"]), h["radius"])
            occ[m] = 1.0
            ids[m] = h["id"]

        sensor = np.zeros(mx.shape, dtype=np.float32)   # 0 free, 1 occ, .5 unknown
        order = np.argsort([np.hypot(h["px"] - robot_xy[0], h["py"] - robot_xy[1])
                            for h in humans]) if humans else []
        shadowed = np.zeros(mx.shape, dtype=bool)
        for k in order:
            h = humans[k]
            dx, dy = h["px"] - robot_xy[0], h["py"] - robot_xy[1]
            d = float(np.hypot(dx, dy))
            if d <= h["radius"] or d < 1e-6:
                continue
            alpha = np.arctan2(dy, dx)
            theta = np.arcsin(np.clip(h["radius"] / d, -1.0, 1.0))
            t = h["radius"] / max(np.tan(theta), 1e-6)
            x1 = robot_xy[0] + t * np.cos(alpha - theta)
            y1 = robot_xy[1] + t * np.sin(alpha - theta)
            x2 = robot_xy[0] + t * np.cos(alpha + theta)
            y2 = robot_xy[1] + t * np.sin(alpha + theta)
            far = 2.5 * self.extent
            x3 = robot_xy[0] - far if x1 <= robot_xy[0] else robot_xy[0] + far
            x4 = robot_xy[0] - far if x2 <= robot_xy[0] else robot_xy[0] + far
            y3 = _line_extend(robot_xy[0], robot_xy[1], x1, y1, x3)
            y4 = _line_extend(robot_xy[0], robot_xy[1], x2, y2, x4)
            poly = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])
            shadowed |= _points_in_polygon(mx, my, poly)

        fov = _point_in_circle(mx, my, robot_xy, self.fov_radius)
        sensor[shadowed] = 0.5
        sensor[~fov] = 0.5
        # a pedestrian's own cells are visible unless entirely shadowed
        visible, occluded = [], []
        for h in humans:
            m = ids == h["id"]
            if not m.any():
                occluded.append(h["id"])
                continue
            if np.all(sensor[m] == 0.5):
                occluded.append(h["id"])
            else:
                sensor[m] = 1.0
                visible.append(h["id"])
        self._mesh = (mx, my)
        return occ, ids, sensor, visible, occluded

    # ------------------------------------------------------------- belief
    def _predict(self):
        """Motion model: a pedestrian believed to be somewhere may have moved.

        Probability mass is spread by one grid neighbourhood per step (a
        pedestrian covers at most vmax*dt) and pulled back toward the prior, so
        an unconfirmed belief decays instead of persisting forever.
        """
        if self.logodds is None:
            return
        p = 1.0 / (1.0 + np.exp(-self.logodds))
        k = max(1, int(round(self.vmax * self.dt / self.res)))
        sp = p.copy()
        for _ in range(k):
            padded = np.pad(sp, 1, mode="constant", constant_values=self.p_prior)
            sp = np.maximum.reduce([
                padded[1:-1, 1:-1],
                padded[:-2, 1:-1], padded[2:, 1:-1],
                padded[1:-1, :-2], padded[1:-1, 2:]])
        p = (1.0 - self.diffuse) * p + self.diffuse * sp
        p = self.p_prior + self.decay * (p - self.p_prior)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        self.logodds = np.log(p / (1.0 - p))

    def _correct(self, sensor):
        """Bernoulli update from the SENSOR grid only.

        Free cells are strong negative evidence, occupied cells positive, and
        cells marked unknown are left alone -- not seeing something is not
        evidence of absence when the view is blocked. This is the only place
        the belief is allowed to learn anything, and it never touches ground
        truth.
        """
        if self.logodds is None:
            self.logodds = np.full(sensor.shape, self.l_prior, dtype=np.float64)
        free = sensor == 0.0
        occ = sensor == 1.0
        self.logodds[free] += self.l_miss - self.l_prior
        self.logodds[occ] += self.l_hit - self.l_prior
        np.clip(self.logodds, -8.0, 8.0, out=self.logodds)

    def _extract_modes(self, sensor):
        """Turn the posterior over the OCCLUDED region into at most
        max_entities pseudo-pedestrians, so the belief can be consumed by the
        same token layout as a real pedestrian."""
        if self.logodds is None:
            return []
        p = 1.0 / (1.0 + np.exp(-self.logodds))
        mx, my = self._mesh
        cand = (p >= self.p_report) & (sensor == 0.5)
        if not cand.any():
            self._prev_modes = []
            return []
        idx = np.argwhere(cand)
        w = p[cand]
        order = np.argsort(-w)
        modes, used = [], np.zeros(len(order), dtype=bool)
        min_sep = max(2.0 * self.res, 0.6)
        for a, oi in enumerate(order):
            if used[a]:
                continue
            r, c = idx[oi]
            cx, cy = float(mx[r, c]), float(my[r, c])
            sel = [a]
            for b in range(a + 1, len(order)):
                if used[b]:
                    continue
                rb, cb = idx[order[b]]
                if np.hypot(mx[rb, cb] - cx, my[rb, cb] - cy) <= min_sep:
                    used[b] = True
                    sel.append(b)
            used[a] = True
            rs = idx[order[sel]]
            ws = w[order[sel]]
            cx = float((mx[rs[:, 0], rs[:, 1]] * ws).sum() / ws.sum())
            cy = float((my[rs[:, 0], rs[:, 1]] * ws).sum() / ws.sum())
            pk = float(ws.max())
            spread = float(np.sqrt(((mx[rs[:, 0], rs[:, 1]] - cx) ** 2 +
                                    (my[rs[:, 0], rs[:, 1]] - cy) ** 2)
                                   .dot(ws) / ws.sum())) if len(sel) > 1 else self.res
            modes.append([cx, cy, pk, spread])
            if len(modes) >= self.max_entities:
                break
        # velocity by nearest-neighbour association with the previous step's
        # modes; a newly appeared mode is reported as stationary rather than
        # given an invented velocity
        out = []
        for cx, cy, pk, spread in modes:
            vx = vy = 0.0
            if self._prev_modes:
                d = [np.hypot(cx - m[0], cy - m[1]) for m in self._prev_modes]
                j = int(np.argmin(d))
                if d[j] <= self.vmax * self.dt * 2.0:
                    vx = (cx - self._prev_modes[j][0]) / self.dt
                    vy = (cy - self._prev_modes[j][1]) / self.dt
            out.append({"px": cx, "py": cy, "vx": vx, "vy": vy,
                        "radius": 0.3, "p_exist": pk,
                        "uncertainty": min(1.0, spread / max(self.extent, 1e-6)),
                        "visible": 0.0, "hidden": 1.0, "id": -1})
        self._prev_modes = modes
        return out

    # -------------------------------------------------------------- update
    def update(self, robot_state, humans):
        """One step. `humans` is ground truth; the split happens here, once,
        and everything downstream of the split is derived from the sensor grid
        alone."""
        if not self.enabled:
            return
        self._robot_xy = np.array([robot_state.px, robot_state.py], dtype=np.float64)
        self._robot_radius = float(robot_state.radius)
        hs = [{"id": i, "px": h.px, "py": h.py, "vx": h.vx, "vy": h.vy,
               "radius": h.radius} for i, h in enumerate(humans)]
        old_mesh = self._mesh
        occ, ids, sensor, vis, occl = self._label_and_sensor(self._robot_xy, hs)
        if self.mode == "bayes":
            self._reproject_belief(old_mesh)
        self.label_grid = np.stack([occ, ids.astype(np.float32)])
        self.sensor_grid = sensor
        self.visible_ids, self.occluded_ids = vis, occl
        self._visible = [hs[i] for i in vis]
        self._all = hs
        # last-seen memory drives the deterministic arm; it is written ONLY
        # from what was visible this step
        for h in self._visible:
            self._ever_seen.add(h["id"])
            self._last_seen[h["id"]] = [h["px"], h["py"], h["vx"], h["vy"],
                                        h["radius"], 0]
        for k in list(self._last_seen):
            if k not in vis:
                self._last_seen[k][5] += 1
        if self.mode == "bayes":
            self._predict()
            self._correct(sensor)
            # Association advances exactly once per sensor update, never on read.
            entities = self._visible_entities() + self._extract_modes(sensor)
            self._frame_index += 1
            self._snapshot = BeliefSnapshot(
                self._frame_index, self.dt,
                tuple(BeliefEntity(**entity) for entity in entities))

    def get_belief_snapshot(self):
        if self.mode != 'bayes':
            raise RuntimeError('Occupancy snapshot requires bayes mode')
        return self._snapshot

    # ------------------------------------------------------------ entities
    def _visible_entities(self):
        return [{"px": h["px"], "py": h["py"], "vx": h["vx"], "vy": h["vy"],
                 "radius": h["radius"], "p_exist": 1.0, "uncertainty": 0.0,
                 "visible": 1.0, "hidden": 0.0, "id": h["id"]}
                for h in self._visible]

    def policy_entities(self, max_age=20, include_hidden_belief=True):
        """What the STUDENT policy is allowed to see, per arm.

        ``include_hidden_belief=False`` is an evaluation-only ablation for a
        trained Bayes policy. It keeps the Bayes tracker and occlusion geometry
        active, but returns exactly the visible entities that the sensor arm
        would expose. Dropping only token columns 9--12 would be invalid: the
        inferred entity position and velocity also occupy columns 0--8.
        """
        if not self.enabled:
            return None
        if not include_hidden_belief:
            if self.mode != "bayes":
                raise ValueError(
                    "hidden-belief input ablation is defined only for bayes mode")
            return self._visible_entities()
        if self.mode == "gt":
            return [{"px": h["px"], "py": h["py"], "vx": h["vx"], "vy": h["vy"],
                     "radius": h["radius"], "p_exist": 1.0, "uncertainty": 0.0,
                     "visible": 1.0, "hidden": 0.0, "id": h["id"]}
                    for h in self._all]
        if self.mode == "oracle_belief":
            visible = set(self.visible_ids)
            return [{"px": h["px"], "py": h["py"], "vx": h["vx"], "vy": h["vy"],
                     "radius": h["radius"], "p_exist": 1.0, "uncertainty": 0.0,
                     "visible": float(h["id"] in visible),
                     "hidden": float(h["id"] not in visible), "id": h["id"]}
                    for h in self._all]
        ents = self._visible_entities()
        if self.mode == "sensor":
            return ents
        if self.mode == "deterministic":
            # constant-velocity dead reckoning of pedestrians last seen, with a
            # confidence that decays with age. No probabilistic content: this
            # is the control that isolates what the posterior adds.
            for k, v in self._last_seen.items():
                if k in self.visible_ids or v[5] == 0 or v[5] > max_age:
                    continue
                t = v[5] * self.dt
                ents.append({"px": v[0] + v[2] * t, "py": v[1] + v[3] * t,
                             "vx": v[2], "vy": v[3], "radius": v[4],
                             "p_exist": float(max(0.0, 1.0 - v[5] / max_age)),
                             "uncertainty": float(min(1.0, v[5] / max_age)),
                             "visible": 0.0, "hidden": 1.0, "id": -1})
            return ents
        return self.get_belief_snapshot().entity_dicts()

    def oracle_entities(self):
        """Ground truth. For the oracle arm and for evaluation only; the
        student path must never call this."""
        return [{"px": h["px"], "py": h["py"], "vx": h["vx"], "vy": h["vy"],
                 "radius": h["radius"], "p_exist": 1.0, "uncertainty": 0.0,
                 "visible": 1.0, "hidden": 0.0, "id": h["id"]}
                for h in self._all]

    # ------------------------------------------------------------- metrics
    def stats(self, near_radius=2.0):
        if not self.enabled or self.sensor_grid is None:
            return {}
        n = self.sensor_grid.size
        occluded = set(self.occluded_ids)
        hidden = [h for h in self._all if h["id"] in occluded]
        unseen_hidden = [h for h in hidden if h["id"] not in self._ever_seen]

        def clearances(items):
            return [
                float(np.hypot(h["px"] - self._robot_xy[0],
                               h["py"] - self._robot_xy[1])
                      - self._robot_radius - h["radius"])
                for h in items
            ]

        hidden_clearances = clearances(hidden)
        unseen_clearances = clearances(unseen_hidden)
        return {"occluded_frac": float((self.sensor_grid == 0.5).sum()) / n,
                "n_visible": len(self.visible_ids),
                "n_occluded": len(self.occluded_ids),
                "n_near_occluded": int(sum(
                    d <= float(near_radius) for d in hidden_clearances)),
                "n_unseen_occluded": len(unseen_hidden),
                "n_near_unseen_occluded": int(sum(
                    d <= float(near_radius) for d in unseen_clearances)),
                "min_occluded_clearance_m": (
                    min(hidden_clearances) if hidden_clearances else np.nan),
                "min_unseen_occluded_clearance_m": (
                    min(unseen_clearances) if unseen_clearances else np.nan)}

    def belief_scores(self):
        """Brier and NLL of the posterior against ground-truth occupancy on the
        occluded cells only -- the region the belief is actually responsible
        for. Evaluation only."""
        if self.mode != "bayes" or self.logodds is None or self.label_grid is None:
            return {}
        m = self.sensor_grid == 0.5
        if not m.any():
            return {}
        p = np.clip(1.0 / (1.0 + np.exp(-self.logodds[m])), 1e-6, 1 - 1e-6)
        y = (self.label_grid[0][m] > 0.5).astype(np.float64)
        result = {"belief_brier": float(((p - y) ** 2).mean()),
                  "belief_nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())}

        full_p = 1.0 / (1.0 + np.exp(-self.logodds))
        mx, my = self._mesh
        hidden = [h for h in self._all if h["id"] in set(self.occluded_ids)]
        hits, seen_hits, unseen_hits = 0, 0, 0
        seen_total, unseen_total = 0, 0
        true_prob = []
        for h in hidden:
            footprint = _point_in_circle(mx, my, (h["px"], h["py"]),
                                         max(h["radius"], self.res))
            footprint &= self.sensor_grid == 0.5
            peak = float(full_p[footprint].max()) if footprint.any() else 0.0
            hit = peak >= self.p_report
            was_seen = h["id"] in self._ever_seen
            hits += int(hit)
            seen_total += int(was_seen)
            unseen_total += int(not was_seen)
            seen_hits += int(hit and was_seen)
            unseen_hits += int(hit and not was_seen)
            true_prob.append(peak)
        result.update({
            "belief_hidden_count": float(len(hidden)),
            "belief_hidden_recall": float(hits / len(hidden)) if hidden else np.nan,
            "belief_seen_hidden_recall": (float(seen_hits / seen_total)
                                           if seen_total else np.nan),
            "belief_unseen_hidden_recall": (float(unseen_hits / unseen_total)
                                             if unseen_total else np.nan),
            "belief_hidden_true_prob": float(np.mean(true_prob)) if true_prob else np.nan,
        })
        return result
