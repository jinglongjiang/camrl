"""Public-geometry candidate-destination provider (BDVL final main chain,
Order 4 item 1 / item 4).

Produces the ``candidate_fn(track_id, first_observed_position)`` that
``IntentBeliefBank`` needs, deriving candidate destinations ONLY from a
PUBLIC scene description + an observable entry position. It NEVER reads a
Human object or a hidden per-human goal -- that is the label-leakage hard
contract (consolidation plan Decision 2). Enforced two ways:

  * the callable's signature is ``(track_id: int, first_position) -> ...``
    -- there is no parameter through which a Human/hidden goal could enter;
  * a static test (``test_main_chain_never_reads_hidden_goal``) greps the
    main-chain belief modules and asserts they never access a human's hidden
    goal attributes (the dotted g-x / g-y coordinates).

A PublicScene is a fixed, public map object:
  * ``junction`` scenes expose discrete exits reached via a shared junction
    waypoint -- the multimodal stress structure (collision risk depends on
    the human's unknown exit choice).
  * ``circle`` scenes (the standard CrowdNav negative control) expose a set
    of public boundary destinations; an open crossing disambiguates quickly,
    so the belief collapses fast -> full ~= mean (expected negative control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.intent_tracker import CandidateGoal


class SceneCandidatesError(ValueError):
    pass


@dataclass(frozen=True)
class PublicDestination:
    """A PUBLIC map destination (never a hidden per-human goal)."""

    name: str
    position: Tuple[float, float]

    def __post_init__(self) -> None:
        arr = np.asarray(self.position, dtype=np.float64)
        if arr.shape != (2,) or not np.all(np.isfinite(arr)):
            raise SceneCandidatesError(f"destination {self.name!r} position must be a finite 2D point, got {self.position!r}")


@dataclass(frozen=True)
class PublicScene:
    """Public map: destinations + an optional shared via-waypoint (the
    junction). ``forward_only`` keeps only destinations roughly ahead of the
    observed entry (a human does not turn around), so an entry position alone
    -- no hidden goal -- selects a plausible candidate set."""

    destinations: Tuple[PublicDestination, ...]
    junction: Optional[Tuple[float, float]] = None
    forward_only: bool = True
    # ---- junction-crowd fix (paper-test post-mortem) -------------------
    # The junction-crowd scene contains TWO populations with different
    # public motion rules, and the old code applied the junction rule to
    # both. Measured consequence on the 500-episode paper test: the four
    # lateral crossers were each handed "will exit left or right at the
    # junction" candidates whose endpoints sat 1.90 m (max 4.01 m) from
    # where they actually went, with a 1.03 m/s velocity residual. A
    # counterfactual that replaced only those four humans' belief features
    # cut navigation time 37%, path length 19% and heading jitter 99% at no
    # safety cost -- i.e. the wrong candidate model, not the Bayesian
    # method, produced the result.
    #
    # The fix keeps the derivation PUBLIC: which rule applies is decided by
    # the observable entry position against a declared corridor, never by
    # who the pedestrian is. An entry inside the approach corridor is on
    # the junction approach and gets the exit candidates; an entry outside
    # it is a lateral crosser and gets crossing candidates.
    #
    # ``approach_corridor`` = (half_width, y_min): |x| <= half_width and
    # y >= y_min. ``crossing_band`` = (y_min, y_max, n_bands): crossers
    # traverse to the mirrored-x side, ending somewhere in this y band,
    # which is discretized into ``n_bands`` candidate endpoints.
    approach_corridor: Optional[Tuple[float, float]] = None
    crossing_band: Optional[Tuple[float, float, int]] = None
    # C1.1: some scene TYPES carry a stronger public structural rule than a
    # forward cone. In CrowdSim's ``square_crossing``, a pedestrian's goal
    # is ALWAYS in the opposite half-plane from its entry
    # (``generate_square_crossing_human``: px uses ``sign``, gx uses
    # ``-sign``). That is public knowledge about the scenario type -- the
    # same kind of public fact as "this junction has two exits" -- and is
    # NOT per-pedestrian label leakage, since it does not depend on any
    # individual's hidden goal. A forward cone cannot express it: from the
    # far edge, same-side destinations are still within any reasonable
    # cone angle. ``"x"``/``"y"`` keep only destinations strictly across
    # that axis from the entry; ``None`` (default) preserves the original
    # cone-only behaviour for circle/junction scenes.
    opposite_half_plane_axis: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.destinations) == 0:
            raise SceneCandidatesError("PublicScene requires at least one destination")
        names = [d.name for d in self.destinations]
        if len(set(names)) != len(names):
            raise SceneCandidatesError(f"destination names must be unique, got {names}")
        if self.junction is not None:
            j = np.asarray(self.junction, dtype=np.float64)
            if j.shape != (2,) or not np.all(np.isfinite(j)):
                raise SceneCandidatesError(f"junction must be a finite 2D point, got {self.junction!r}")
        if self.opposite_half_plane_axis not in (None, "x", "y"):
            raise SceneCandidatesError(
                f"opposite_half_plane_axis must be None, 'x' or 'y', got {self.opposite_half_plane_axis!r}")
        if self.approach_corridor is not None:
            hw, y_min = self.approach_corridor
            if not (np.isfinite(hw) and hw > 0 and np.isfinite(y_min)):
                raise SceneCandidatesError(f"approach_corridor must be (half_width>0, y_min), got {self.approach_corridor!r}")
            if self.junction is None:
                raise SceneCandidatesError("approach_corridor requires a junction -- a corridor approaches something")
        if self.crossing_band is not None:
            y0, y1, n = self.crossing_band
            if not (np.isfinite(y0) and np.isfinite(y1) and y1 > y0):
                raise SceneCandidatesError(f"crossing_band needs y_max > y_min, got {self.crossing_band!r}")
            if int(n) < 2:
                raise SceneCandidatesError(
                    f"crossing_band n_bands must be >= 2, got {n}: a crosser's endpoint along the band is "
                    "genuinely unknown, and collapsing it to one candidate would fake certainty")
            if int(n) > 8:
                raise SceneCandidatesError(f"crossing_band n_bands must be <= MAX_CANDIDATE_GOALS (8), got {n}")
            if self.approach_corridor is None:
                raise SceneCandidatesError(
                    "crossing_band requires an approach_corridor: without one there is no public rule "
                    "separating approachers from crossers, which is the bug this pair exists to fix")

    def in_approach_corridor(self, first_position: np.ndarray) -> bool:
        """PUBLIC test: is this observable entry on the junction approach?"""
        if self.approach_corridor is None:
            return False
        entry = np.asarray(first_position, dtype=np.float64)
        half_width, y_min = self.approach_corridor
        return bool(abs(entry[0]) <= half_width and entry[1] >= y_min)

    def _crossing_candidates(self, entry: np.ndarray) -> List[CandidateGoal]:
        """Lateral crossers traverse to the mirrored-x side of the scene and
        stop somewhere in the public crossing band. Public rule + observable
        entry only -- the band is discretized, never read off a hidden goal,
        so the endpoint along it stays genuinely uncertain."""
        y0, y1, n_bands = self.crossing_band
        n_bands = int(n_bands)
        target_x = -float(entry[0])
        edges = np.linspace(float(y0), float(y1), n_bands + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return [CandidateGoal(f"cross{i}", ((target_x, float(y)),)) for i, y in enumerate(centres)]

    def candidates_for(self, first_position: np.ndarray) -> List[CandidateGoal]:
        """PUBLIC-only derivation: from an entry position (observable) +
        this public map, build one CandidateGoal route per plausible
        destination. Route = [junction, destination] when a junction is
        defined and still ahead, else [destination]."""
        entry = np.asarray(first_position, dtype=np.float64)
        if entry.shape != (2,) or not np.all(np.isfinite(entry)):
            raise SceneCandidatesError(f"first_position must be a finite 2D point, got {first_position!r}")

        # Two populations, two public rules. Decided by the observable entry
        # against the declared corridor -- never by pedestrian identity.
        if self.crossing_band is not None and not self.in_approach_corridor(entry):
            return self._crossing_candidates(entry)

        # forward direction: toward the junction if present, else scene centroid
        if self.junction is not None:
            fwd = np.asarray(self.junction, dtype=np.float64) - entry
        else:
            centroid = np.mean([d.position for d in self.destinations], axis=0)
            fwd = np.asarray(centroid, dtype=np.float64) - entry
        fwd_n = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-9 else np.zeros(2)

        axis_i = {"x": 0, "y": 1}.get(self.opposite_half_plane_axis) if self.opposite_half_plane_axis else None

        cands: List[CandidateGoal] = []
        for d in self.destinations:
            dest = np.asarray(d.position, dtype=np.float64)
            if axis_i is not None and entry[axis_i] * dest[axis_i] > 0:
                continue  # same half-plane as the entry -> not a plausible crossing goal
            if self.forward_only and fwd_n.any():
                to_dest = dest - entry
                if np.linalg.norm(to_dest) > 1e-9 and float(np.dot(to_dest / np.linalg.norm(to_dest), fwd_n)) < -0.3:
                    continue  # destination is behind the entry -> implausible
            if self.junction is not None:
                jd = np.asarray(self.junction, dtype=np.float64)
                # include the junction waypoint only while the entry is still before it
                if float(np.linalg.norm(jd - entry)) > 0.35:
                    cands.append(CandidateGoal(d.name, (tuple(jd), tuple(dest))))
                    continue
            cands.append(CandidateGoal(d.name, (tuple(dest),)))
        if not cands:  # fallback: everything filtered -> keep all (never empty)
            cands = [
                CandidateGoal(d.name, ((tuple(np.asarray(self.junction, dtype=np.float64)), d.position)
                                       if self.junction is not None else (d.position,)))
                for d in self.destinations
            ]
        return cands


def make_candidate_fn(scene: PublicScene) -> Callable[[int, np.ndarray], List[CandidateGoal]]:
    """Adapter for ``IntentBeliefBank(candidate_fn=...)``. Receives ONLY a
    stable track_id + observable entry position -- never a Human/hidden goal."""
    def candidate_fn(track_id: int, first_position) -> List[CandidateGoal]:
        return scene.candidates_for(np.asarray(first_position, dtype=np.float64))
    return candidate_fn


def circle_scene(radius: float, n_sectors: int = 8) -> PublicScene:
    """Standard CrowdNav negative control: public boundary destinations
    evenly around the circle (no junction -> open crossing)."""
    if radius <= 0 or n_sectors < 2:
        raise SceneCandidatesError(f"radius>0 and n_sectors>=2 required, got {radius}/{n_sectors}")
    dests = tuple(
        PublicDestination(
            f"b{i}", (float(radius * np.cos(2 * np.pi * i / n_sectors)), float(radius * np.sin(2 * np.pi * i / n_sectors)))
        )
        for i in range(n_sectors)
    )
    return PublicScene(destinations=dests, junction=None, forward_only=True)


def square_scene(width: float, n_rows: int = 4) -> PublicScene:
    """Public destinations matching CrowdSim's ``square_crossing`` geometry.

    Real bug found by audit (plan 2.2 point 12 / Order C1.1): the formal
    six-scenario evaluator called ``circle_scene()`` for BOTH circle and
    square scenarios, so in every square scenario the "public candidate
    destinations" the tracker reasoned over sat on a circle that has
    nothing to do with where square-crossing pedestrians actually go.

    ``CrowdSim.generate_square_crossing_human`` starts a pedestrian at
    ``px in [0, +-w/2], py in [-w/2, +w/2]`` and gives it a goal in the
    OPPOSITE half-plane: ``gx in [0, -+w/2], gy in [-w/2, +w/2]``. So the
    real goal region is the two half-planes, not a ring. This discretizes
    that region into the centroids of a 2 x ``n_rows`` partition: two
    columns (left/right half-plane centroids at ``x = -+w/4``) times
    ``n_rows`` bands evenly covering ``y in [-w/2, +w/2]``.

    Public geometry only -- derived from the scene's configured width, never
    from any pedestrian's hidden gx/gy.
    """
    if width <= 0 or n_rows < 1:
        raise SceneCandidatesError(f"width>0 and n_rows>=1 required, got {width}/{n_rows}")
    half = float(width) * 0.5
    col_x = (-half * 0.5, half * 0.5)  # centroid of each half-plane
    dests = []
    for ci, x in enumerate(col_x):
        for r in range(n_rows):
            # band centroid: evenly spaced midpoints across [-half, +half]
            y = -half + (2.0 * half) * (r + 0.5) / n_rows
            dests.append(PublicDestination(f"{'l' if ci == 0 else 'r'}{r}", (float(x), float(y))))
    return PublicScene(
        destinations=tuple(dests), junction=None, forward_only=True,
        # CrowdSim's square_crossing always sends a pedestrian to the
        # OPPOSITE half-plane; see PublicScene.opposite_half_plane_axis.
        opposite_half_plane_axis="x",
    )


def junction_scene(junction: Tuple[float, float], exits: Sequence[Tuple[str, Tuple[float, float]]],
                   approach_corridor: Optional[Tuple[float, float]] = None,
                   crossing_band: Optional[Tuple[float, float, int]] = None) -> PublicScene:
    """Multimodal stress structure: discrete public exits reached via a
    shared junction waypoint (collision risk depends on the unknown exit).

    Pass ``approach_corridor``/``crossing_band`` for a scene that also
    contains lateral crossers, so the two populations get the two different
    public rules instead of the junction rule being applied to everyone."""
    dests = tuple(PublicDestination(name, pos) for name, pos in exits)
    return PublicScene(destinations=dests, junction=junction, forward_only=True,
                       approach_corridor=approach_corridor, crossing_band=crossing_band)
