"""Closed-loop, robot-action-dependent pedestrian behavior protocol
(guide.md 5.2) plus a standalone synthetic multi-agent loop used for the
Step 4 smoke-scale data collector and the real six-scenario train/
validation/heldout_interactive/nominal environment construction (guide.md
5.3).

Single responsibility (guide.md section 3): defines the behavior FSM
(continue / yield / assertive / stop_go / turn / goal_switch), the scenario
table (circle/square geometry x baseline/dense/large size, matching
configs/policy_bayesian_brne.config's ``[eval_envs]``), and the
train/validation vs heldout_interactive non-overlapping parameter ranges.
Must NOT contain mode fitting, belief tracking, or BRNE solving -- those
live in mode_model.py, belief_tracker.py, and brne_adapter.py respectively.

Scope note: ``run_episode``/``SyntheticEnv`` below is a standalone,
self-contained synthetic loop -- it does NOT (yet) wire into the real
``crowd_sim.envs.CrowdSim`` Gym environment. That integration is Step 8's
explicit deliverable ("CrowdSim integration smoke", guide.md 10.7/8.4).
Conflating "the protocol produces valid smoke data" with "the protocol is
wired into the real simulator" would be exactly the kind of overstated
claim this project has repeatedly had to walk back; this docstring exists
so that distinction is never silently lost.

Human-human collision avoidance here is a lightweight reciprocal repulsion
term, NOT the real RVO2 ORCA algorithm from ``crowd_sim/envs/policy/
orca*.py``. guide.md 5.2 allows "ORCA可以作为每种类型底层的人-人碰撞约束"
as an option, not a requirement; wiring the real ORCA implementation in is
deferred to Step 8's CrowdSim integration, where the real environment (and
its already-tested ORCA policy) exists natively -- reimplementing/importing
it here would either duplicate untested glue code or touch files outside
the two guide.md 0.2 permits (policy_factory.py, robot.py).

Only the ``goal_directed`` robot controller is implemented (Step 4 smoke
scope); the full goal_directed/ORCA/original_BRNE/scripted_probe mixture is
a Step 5 formal-collection requirement (guide.md 5.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import os

import numpy as np

from crowd_nav.bayesian_brne.schemas import build_robot_state_row

STEP = 4
IMPLEMENTED = True


class BehaviorType(str, Enum):
    CONTINUE = "continue"
    YIELD = "yield"
    ASSERTIVE = "assertive"
    STOP_GO = "stop_go"
    TURN = "turn"
    GOAL_SWITCH = "goal_switch"


DEFAULT_BEHAVIOR_MIX: Dict[BehaviorType, float] = {
    BehaviorType.CONTINUE: 1.0,
    BehaviorType.YIELD: 1.0,
    BehaviorType.ASSERTIVE: 1.0,
    BehaviorType.STOP_GO: 1.0,
    BehaviorType.TURN: 1.0,
    BehaviorType.GOAL_SWITCH: 1.0,
}

# A profile whose behavior mix places ALL mass on CONTINUE -- guide.md 5.1's
# "test nominal:六场景,原始nominal人群" (the original, non-interactive
# crowd), used only as the label for what data this produces; it is not
# claimed to be numerically identical to the project's existing ORCA
# baseline crowd.
NOMINAL_BEHAVIOR_MIX: Dict[BehaviorType, float] = {bt: (1.0 if bt == BehaviorType.CONTINUE else 0.0) for bt in BehaviorType}


class ProtocolConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProtocolConfig:
    """Every tunable number the FSM uses -- no magic numbers in the branch
    logic below (same discipline as config.py's BayesianModelConfig)."""

    ttc_threshold: float = 3.0
    clearance_threshold: float = 1.0
    conflict_horizon_steps: int = 12
    yield_gain: float = 0.6
    turn_gain: float = 0.7
    stop_release_steps: int = 8
    goal_switch_lateral: float = 1.5
    assertive_repulsion_scale: float = 0.3
    max_human_speed: float = 2.0
    max_human_acceleration: float = 2.0
    # C5 fix (2026-08-03 third Step 4 audit): guide.md 5.3 names "yield
    # threshold" and "goal-switch location/time" as two of the six
    # dimensions that MUST differ, non-overlapping, between train/
    # validation and heldout_interactive. An earlier version only varied
    # ``yield_gain`` (a response MAGNITUDE) and ``goal_switch_lateral`` (an
    # offset magnitude) -- neither is a trigger threshold/timing, so the
    # audit correctly found the "six dimensions" claim unmet. These two new
    # fields are the actual TRIGGER conditions (a TTC threshold each,
    # standing in for "location/time" per the guide's own suggested
    # ``goal_switch_trigger_ttc`` alternative), independent of the generic
    # ``ttc_threshold`` every other behavior type still shares.
    yield_ttc_threshold: float = 3.0
    goal_switch_ttc_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.ttc_threshold <= 0.0:
            raise ProtocolConfigError(f"ttc_threshold must be > 0, got {self.ttc_threshold}")
        if self.clearance_threshold <= 0.0:
            raise ProtocolConfigError(f"clearance_threshold must be > 0, got {self.clearance_threshold}")
        if self.conflict_horizon_steps < 1:
            raise ProtocolConfigError(f"conflict_horizon_steps must be >= 1, got {self.conflict_horizon_steps}")
        if not (0.0 < self.yield_gain <= 1.0):
            raise ProtocolConfigError(f"yield_gain must be in (0, 1], got {self.yield_gain}")
        if self.turn_gain <= 0.0:
            raise ProtocolConfigError(f"turn_gain must be > 0, got {self.turn_gain}")
        if self.stop_release_steps < 1:
            raise ProtocolConfigError(f"stop_release_steps must be >= 1, got {self.stop_release_steps}")
        if not (0.0 < self.assertive_repulsion_scale <= 1.0):
            raise ProtocolConfigError(f"assertive_repulsion_scale must be in (0, 1], got {self.assertive_repulsion_scale}")
        if self.max_human_speed <= 0.0:
            raise ProtocolConfigError(f"max_human_speed must be > 0, got {self.max_human_speed}")
        if self.max_human_acceleration <= 0.0:
            raise ProtocolConfigError(f"max_human_acceleration must be > 0, got {self.max_human_acceleration}")
        if self.yield_ttc_threshold <= 0.0:
            raise ProtocolConfigError(f"yield_ttc_threshold must be > 0, got {self.yield_ttc_threshold}")
        if self.goal_switch_ttc_threshold <= 0.0:
            raise ProtocolConfigError(f"goal_switch_ttc_threshold must be > 0, got {self.goal_switch_ttc_threshold}")


@dataclass
class HumanBehaviorState:
    """Mutable per-human, per-episode FSM state. ``behavior_type`` is
    sampled once at episode start (a fixed latent type, guide.md 5.2) and
    held fixed for the whole episode; only the FSM sub-state (stop timer,
    goal-switch flag/local goal) evolves as the episode proceeds."""

    behavior_type: BehaviorType
    stop_timer: int = 0
    goal_switched: bool = False
    local_goal: Optional[np.ndarray] = None

    def copy(self) -> "HumanBehaviorState":
        return HumanBehaviorState(
            behavior_type=self.behavior_type,
            stop_timer=self.stop_timer,
            goal_switched=self.goal_switched,
            local_goal=None if self.local_goal is None else self.local_goal.copy(),
        )


def sample_behavior_type(rng: np.random.Generator, mix: Optional[Dict[BehaviorType, float]] = None) -> BehaviorType:
    mix = mix or DEFAULT_BEHAVIOR_MIX
    types = list(mix.keys())
    weights = np.array([mix[t] for t in types], dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(types), p=weights))
    return types[idx]


def compute_conflict(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    human_pos: np.ndarray,
    human_vel: np.ndarray,
    combined_radius: float,
    config: ProtocolConfig,
    dt: float,
    ttc_threshold: Optional[float] = None,
    clearance_threshold: Optional[float] = None,
) -> Tuple[bool, float, float]:
    """``conflict = TTC(robot, human) < ttc_threshold and
    predicted_min_clearance < clearance_threshold`` (guide.md 5.2), using
    constant-velocity linear extrapolation of both agents over
    ``conflict_horizon_steps``. This is deliberately the simplest possible
    predictor -- it is NOT the Bayesian belief tracker's job (that module
    handles the actual multi-modal prediction the robot reasons about); this
    FSM only needs a cheap, transparent trigger for the pedestrian's OWN
    immediate reaction. Returns ``(in_conflict, ttc, min_clearance)``.

    ``ttc_threshold``/``clearance_threshold`` default to ``config``'s
    generic values but can be overridden per call -- this is how YIELD and
    GOAL_SWITCH get their OWN trigger thresholds (``config.
    yield_ttc_threshold``/``config.goal_switch_ttc_threshold``) distinct
    from the shared threshold CONTINUE/ASSERTIVE/STOP_GO/TURN report
    against (C5 fix, guide.md 5.3)."""
    ttc_threshold = config.ttc_threshold if ttc_threshold is None else ttc_threshold
    clearance_threshold = config.clearance_threshold if clearance_threshold is None else clearance_threshold

    rel_pos = human_pos - robot_pos
    rel_vel = human_vel - robot_vel
    dist0 = float(np.hypot(rel_pos[0], rel_pos[1]))

    min_clearance = dist0
    for step in range(1, config.conflict_horizon_steps + 1):
        p = rel_pos + rel_vel * dt * step
        min_clearance = min(min_clearance, float(np.hypot(p[0], p[1])))

    dist0_safe = max(dist0, 1e-6)
    closing_speed = float(-np.dot(rel_pos, rel_vel) / dist0_safe)
    ttc = dist0_safe / closing_speed if closing_speed > 1e-3 else float("inf")

    in_conflict = (ttc < ttc_threshold) and (min_clearance - combined_radius < clearance_threshold)
    return in_conflict, ttc, min_clearance


def predicted_passing_side(rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
    """Signed cross product ``rel_pos x rel_vel``: sign indicates which side
    the robot is predicted to pass the human on. Same convention as
    ``mode_model.py``'s ``passing_side`` feature."""
    return float(rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0])


def compute_human_response(
    human_pos: np.ndarray,
    human_vel: np.ndarray,
    human_pref_speed: float,
    goal: np.ndarray,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    combined_radius: float,
    state: HumanBehaviorState,
    config: ProtocolConfig,
    dt: float,
) -> Tuple[np.ndarray, HumanBehaviorState, bool, float]:
    """Core FSM step (guide.md 5.2). Returns
    ``(preferred_velocity[2], new_state, in_conflict, human_human_repulsion_scale)``.
    ``preferred_velocity`` is this human's robot-facing intent; a
    human-human collision-avoidance layer runs on top of it, scaled by
    ``human_human_repulsion_scale`` (see ``_apply_human_human_avoidance``
    below). Deliberately a pure function of its arguments (plus the
    immutable ``behavior_type`` inside ``state``) -- this is what makes the
    branching test possible: same snapshot + same robot action must give
    bit-identical output, different robot action must give a different
    output for reactive types.

    CONTINUE and ASSERTIVE are both conflict-blind in ``preferred_velocity``
    (guide.md 5.2: assertive gives "低让行概率,继续前进" toward the ROBOT),
    but the 2026-08-03 audit correctly flagged that leaving them otherwise
    IDENTICAL makes them unobservable as distinct latent modes. ASSERTIVE
    therefore reduces its human-human repulsion specifically while in
    robot-conflict (``assertive_repulsion_scale < 1``) -- it does not yield
    to the ROBOT, but it also does not make extra room for other HUMANS
    while asserting through a robot-conflict, which CONTINUE (always
    ``repulsion_scale == 1``, conflict-independent) does not do."""
    to_goal = goal - human_pos
    dist_to_goal = float(np.hypot(to_goal[0], to_goal[1]))
    goal_dir = to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.zeros(2)
    nominal_pref = goal_dir * human_pref_speed

    # Generic conflict check (shared by CONTINUE/ASSERTIVE/STOP_GO/TURN).
    # YIELD and GOAL_SWITCH use their OWN trigger thresholds below (C5 fix,
    # guide.md 5.3) -- each behavior type's RETURNED ``in_conflict`` is the
    # one it actually used to decide, not always this generic one.
    in_conflict, _, _ = compute_conflict(robot_pos, robot_vel, human_pos, human_vel, combined_radius, config, dt)
    rel_pos = human_pos - robot_pos
    rel_vel = human_vel - robot_vel
    side = predicted_passing_side(rel_pos, rel_vel)
    lateral = np.array([-goal_dir[1], goal_dir[0]])

    bt = state.behavior_type
    new_state = state.copy()
    repulsion_scale = 1.0

    if bt == BehaviorType.CONTINUE:
        pref = nominal_pref

    elif bt == BehaviorType.ASSERTIVE:
        # Conflict-blind toward the ROBOT by design (guide.md 5.2: "对机器
        # 人低让行概率，继续前进"); differs from CONTINUE only in the
        # human-human channel (see docstring above).
        pref = nominal_pref
        repulsion_scale = config.assertive_repulsion_scale if in_conflict else 1.0

    elif bt == BehaviorType.YIELD:
        in_conflict, _, _ = compute_conflict(
            robot_pos, robot_vel, human_pos, human_vel, combined_radius, config, dt,
            ttc_threshold=config.yield_ttc_threshold,
        )
        if in_conflict:
            sign = -1.0 if side >= 0 else 1.0
            pref = nominal_pref * (1.0 - config.yield_gain) + lateral * sign * config.yield_gain * human_pref_speed
        else:
            pref = nominal_pref

    elif bt == BehaviorType.STOP_GO:
        if in_conflict and new_state.stop_timer == 0:
            new_state.stop_timer = config.stop_release_steps
        if new_state.stop_timer > 0:
            new_state.stop_timer -= 1
            pref = np.zeros(2)
        else:
            pref = nominal_pref

    elif bt == BehaviorType.TURN:
        if in_conflict:
            sign = 1.0 if side >= 0 else -1.0
            pref = nominal_pref + lateral * sign * config.turn_gain * human_pref_speed
            norm = float(np.hypot(pref[0], pref[1]))
            if norm > human_pref_speed:
                pref = pref / norm * human_pref_speed
        else:
            pref = nominal_pref

    elif bt == BehaviorType.GOAL_SWITCH:
        in_conflict, _, _ = compute_conflict(
            robot_pos, robot_vel, human_pos, human_vel, combined_radius, config, dt,
            ttc_threshold=config.goal_switch_ttc_threshold,
        )
        if in_conflict and not new_state.goal_switched:
            sign = -1.0 if side >= 0 else 1.0
            new_state.local_goal = goal + lateral * sign * config.goal_switch_lateral
            new_state.goal_switched = True
        if new_state.goal_switched and new_state.local_goal is not None:
            to_local = new_state.local_goal - human_pos
            d = float(np.hypot(to_local[0], to_local[1]))
            direction = to_local / d if d > 1e-6 else np.zeros(2)
            pref = direction * human_pref_speed
        else:
            pref = nominal_pref

    else:
        raise ValueError(f"unknown behavior type {bt!r}")

    return pref, new_state, in_conflict, repulsion_scale


def _apply_human_human_avoidance(
    idx: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    preferred: np.ndarray,
    repulsion_scale: float = 1.0,
    repulsion_gain: float = 1.5,
    repulsion_range: float = 1.0,
) -> np.ndarray:
    """Lightweight reciprocal repulsion (NOT the real RVO2 ORCA algorithm --
    see module docstring). Adds a velocity term pushing agent ``idx`` away
    from any other agent whose gap is inside ``repulsion_range``, scaled by
    how deep the encroachment is and by ``repulsion_scale`` (ASSERTIVE
    reduces this while in robot-conflict; see compute_human_response).
    Purely a Step-4-scope placeholder so the synthetic multi-human loop does
    not produce agents walking through each other; real ORCA fidelity is
    Step 8's job."""
    correction = np.zeros(2)
    for j in range(positions.shape[0]):
        if j == idx:
            continue
        rel = positions[idx] - positions[j]
        dist = float(np.hypot(rel[0], rel[1]))
        gap = dist - (radii[idx] + radii[j])
        if gap < repulsion_range and dist > 1e-6:
            correction += (rel / dist) * repulsion_gain * repulsion_scale * max(repulsion_range - gap, 0.0)
    return preferred + correction


def _clip_speed_and_accel(v_prev: np.ndarray, v_target: np.ndarray, dt: float, config: ProtocolConfig) -> np.ndarray:
    accel = (v_target - v_prev) / dt
    accel_norm = float(np.hypot(accel[0], accel[1]))
    if accel_norm > config.max_human_acceleration:
        accel = accel / accel_norm * config.max_human_acceleration
    v_new = v_prev + accel * dt
    speed = float(np.hypot(v_new[0], v_new[1]))
    if speed > config.max_human_speed:
        v_new = v_new / speed * config.max_human_speed
    return v_new


@dataclass
class SyntheticHuman:
    track_id: int
    pos: np.ndarray
    vel: np.ndarray
    goal: np.ndarray
    radius: float
    pref_speed: float
    behavior_state: HumanBehaviorState


@dataclass
class SyntheticEnv:
    """Standalone multi-agent loop for Step 4 protocol tests and the smoke
    data collector. See module docstring for the CrowdSim-integration scope
    boundary (Step 8, not here)."""

    humans: List[SyntheticHuman]
    robot_pos: np.ndarray
    robot_vel: np.ndarray
    robot_goal: np.ndarray
    robot_radius: float
    robot_pref_speed: float
    dt: float
    config: ProtocolConfig = field(default_factory=ProtocolConfig)
    # Order 5d: the actually-sampled per-episode profile parameters, kept on
    # the env so run_episode can record them into the saved data's
    # ``profile_params`` field (traceability -- "谁生成、用哪组行为参数").
    sampled_params: Optional["SampledEpisodeParams"] = None

    def robot_state_row(self) -> np.ndarray:
        """Canonical FullState.to_array() order (schemas.ROBOT_STATE_FIELDS),
        built ONLY via schemas.build_robot_state_row (guide.md B8 fix) --
        never a positional literal that could silently drift out of sync
        with the canonical field order."""
        theta = float(np.arctan2(self.robot_vel[1], self.robot_vel[0]))
        return build_robot_state_row(
            px=self.robot_pos[0], py=self.robot_pos[1], vx=self.robot_vel[0], vy=self.robot_vel[1],
            radius=self.robot_radius, gx=self.robot_goal[0], gy=self.robot_goal[1],
            v_pref=self.robot_pref_speed, theta=theta,
        )

    def step_robot_goal_directed(self) -> np.ndarray:
        """Deterministic ``goal_directed`` robot controller (guide.md 5.4):
        no obstacle avoidance, straight toward the goal at pref speed,
        respecting the same accel/speed caps as humans."""
        to_goal = self.robot_goal - self.robot_pos
        dist = float(np.hypot(to_goal[0], to_goal[1]))
        direction = to_goal / dist if dist > 1e-6 else np.zeros(2)
        target = direction * self.robot_pref_speed
        return _clip_speed_and_accel(self.robot_vel, target, self.dt, self.config)

    def step(self, robot_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance one dt given an explicit robot velocity ACTION (guide.md
        5.2's branching test calls the lower-level ``compute_human_response``
        directly with two different actions from the same snapshot;
        ``run_episode`` below instead calls one of the ``step_robot_*``
        controllers each step for full rollouts)."""
        n = len(self.humans)
        positions = np.stack([h.pos for h in self.humans]) if n else np.zeros((0, 2))
        velocities = np.stack([h.vel for h in self.humans]) if n else np.zeros((0, 2))
        radii = np.array([h.radius for h in self.humans]) if n else np.zeros((0,))

        new_vels = np.zeros_like(velocities)
        conflicts = np.zeros(n, dtype=bool)
        new_states: List[HumanBehaviorState] = []
        for i, human in enumerate(self.humans):
            combined_radius = human.radius + self.robot_radius
            pref, new_state, in_conflict, repulsion_scale = compute_human_response(
                human.pos, human.vel, human.pref_speed, human.goal,
                self.robot_pos, robot_action, combined_radius,
                human.behavior_state, self.config, self.dt,
            )
            pref = _apply_human_human_avoidance(i, positions, velocities, radii, pref, repulsion_scale=repulsion_scale)
            new_vels[i] = _clip_speed_and_accel(human.vel, pref, self.dt, self.config)
            conflicts[i] = in_conflict
            new_states.append(new_state)

        for i, human in enumerate(self.humans):
            human.vel = new_vels[i]
            human.pos = human.pos + human.vel * self.dt
            human.behavior_state = new_states[i]

        self.robot_vel = robot_action
        self.robot_pos = self.robot_pos + self.robot_vel * self.dt

        return {"conflicts": conflicts}


# --------------------------------------------------------------------- #
# Order 6: four ROBOT controllers (guide.md 5.4). These are data-collection
# diversity controllers, not the paper's evaluated policy -- their job is
# only to keep collected pedestrian-response data from being narrowed to
# whatever ONE robot policy induces (guide.md 5.4's own justification).
# ``goal_directed`` lives on SyntheticEnv already; the other three are
# implemented below as free functions dispatched by ``compute_robot_action``.
# --------------------------------------------------------------------- #

GENERATOR_VERSION = "sm_brne_order6.1"
CONTROLLER_TYPES = ("goal_directed", "orca", "original_brne", "scripted_probe")
_BRNE_DEFAULT_ROOT = os.environ.get("SM_BRNE_UPSTREAM_ROOT", "/home/abc/temp/brne")


def allocate_controller_type(episode_index: int) -> str:
    """Deterministic EQUAL allocation by episode index (guide.md Order
    6.1: "按 episode index 和 suite seed 确定性等量分配，不能看结果后改比
    例"). A fixed round-robin over CONTROLLER_TYPES, decided purely by
    ``episode_index`` -- never by any post-hoc measurement of which
    controller "worked better"."""
    return CONTROLLER_TYPES[episode_index % len(CONTROLLER_TYPES)]


class ControllerConfigError(ValueError):
    pass


@dataclass
class ScriptedProbeState:
    """FSM sub-state for the ``scripted_probe`` controller (guide.md 5.4.4):
    a reproducible, bounded left/right lateral offset + brief stop,
    triggered by a conflict window against the nearest human, cycling
    through the three probe types in a fixed rotation so the same
    conflict-triggering episode always produces the same probe sequence."""

    probe_phase: Optional[str] = None
    phase_timer: int = 0
    trigger_count: int = 0


PROBE_CYCLE = ("left", "right", "stop")
PROBE_DURATION_STEPS = 4


@dataclass
class RobotControllerState:
    """Everything a controller needs to persist ACROSS steps within one
    episode (an RVO2 sim handle for ``orca``, an RNG stream for
    ``original_brne``'s CV sampling, FSM sub-state for ``scripted_probe``).
    One instance per episode, created fresh at episode start -- never
    shared across episodes (that would silently correlate episodes that
    are supposed to be independent)."""

    controller_type: str
    brne_root: str = _BRNE_DEFAULT_ROOT
    orca_policy: Optional[object] = None
    brne_rng: Optional[np.random.Generator] = None
    scripted_probe_state: ScriptedProbeState = field(default_factory=ScriptedProbeState)
    fallback_count: int = 0

    def __post_init__(self) -> None:
        if self.controller_type not in CONTROLLER_TYPES:
            raise ControllerConfigError(f"unknown controller_type {self.controller_type!r}, expected one of {CONTROLLER_TYPES}")
        if self.brne_rng is None:
            self.brne_rng = np.random.default_rng(0)


def _step_robot_orca(env: SyntheticEnv, ctrl_state: RobotControllerState) -> np.ndarray:
    """Real RVO2 ORCA (guide.md Order 6.2: "orca 必须调用项目现有 RVO2/ORCA
    ...不能用轻量 repulsion 冒充"). Reuses ``crowd_sim.envs.policy.orca.ORCA``
    UNMODIFIED (read-only import -- guide.md 0.2 only permits editing
    policy_factory.py/robot.py) via the project's own ``FullState``/
    ``ObservableState``/``JointState`` wrappers, so this is byte-for-byte
    the same RVO2 call path the rest of the codebase already uses and has
    tested, not a reimplementation."""
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState

    if ctrl_state.orca_policy is None:
        policy = ORCA()
        policy.time_step = env.dt
        policy.max_speed = env.robot_pref_speed
        ctrl_state.orca_policy = policy

    theta = float(np.arctan2(env.robot_vel[1], env.robot_vel[0]))
    self_state = FullState(
        px=float(env.robot_pos[0]), py=float(env.robot_pos[1]),
        vx=float(env.robot_vel[0]), vy=float(env.robot_vel[1]),
        radius=env.robot_radius, gx=float(env.robot_goal[0]), gy=float(env.robot_goal[1]),
        v_pref=env.robot_pref_speed, theta=theta,
    )
    human_states = [
        ObservableState(px=float(h.pos[0]), py=float(h.pos[1]), vx=float(h.vel[0]), vy=float(h.vel[1]), radius=h.radius)
        for h in env.humans
    ]
    action = ctrl_state.orca_policy.predict(JointState(self_state, human_states))
    raw = np.array([action.vx, action.vy])
    return _clip_speed_and_accel(env.robot_vel, raw, env.dt, env.config)


def _sample_robot_candidates_for_original_brne(env: SyntheticEnv, num_samples: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Holonomic velocity-perturbation robot candidates around the
    goal-directed nominal command -- scoped ONLY to the ``original_brne``
    data-collection controller. This is deliberately NOT ``robot_sampler.py``
    (Step 6): that module must be reused IDENTICALLY across every
    BRNE-family evaluation ablation so a difference in robot sampling can
    never be mistaken for a Bayesian effect (guide.md 2.5); this helper's
    only job is to give the original-BRNE data-collection controller
    something to equilibrate over, at data-collection (not evaluation)
    fidelity. Returns ``(robot_traj[M,H,2], robot_controls[M,H,2])``."""
    to_goal = env.robot_goal - env.robot_pos
    dist = float(np.hypot(to_goal[0], to_goal[1]))
    direction = to_goal / dist if dist > 1e-6 else np.zeros(2)
    nominal = direction * env.robot_pref_speed
    lateral = np.array([-direction[1], direction[0]])
    offsets = np.linspace(-0.5, 0.5, num_samples)
    robot_controls = np.zeros((num_samples, horizon, 2))
    for m in range(num_samples):
        robot_controls[m, :, :] = nominal + lateral * offsets[m]
    robot_traj = np.cumsum(robot_controls * env.dt, axis=1) + env.robot_pos[None, None, :]
    return robot_traj, robot_controls


def _step_robot_original_brne(env: SyntheticEnv, ctrl_state: RobotControllerState) -> np.ndarray:
    """The existing Bayesian-game baseline, reused as a data-collection
    controller (guide.md Order 6.3): fixed CV/GP prior for pedestrians
    (``trajectory_sampler.sample_cv`` -- no learned/Bayesian posterior),
    the locked upstream BRNE solver (``brne_adapter.BRNESolver``,
    ``solver_mode='official_exact'``), and ``weighted_first_control`` for
    the executed action -- never SM-BRNE's new posterior machinery."""
    from crowd_nav.bayesian_brne.brne_adapter import BRNESolver, weighted_first_control
    from crowd_nav.bayesian_brne.trajectory_sampler import sample_cv

    horizon, num_samples = 8, 16
    robot_traj, robot_controls = _sample_robot_candidates_for_original_brne(env, num_samples, horizon)

    trajectories = [robot_traj]
    radii = [env.robot_radius]
    for human in env.humans:
        state0 = np.array([human.pos[0], human.pos[1], human.vel[0], human.vel[1]])
        traj = sample_cv(state0, horizon, num_samples, env.dt, ctrl_state.brne_root, ctrl_state.brne_rng)
        trajectories.append(traj)
        radii.append(human.radius)

    solver = BRNESolver(solver_mode="official_exact", brne_root=ctrl_state.brne_root)
    result = solver.solve(np.stack(trajectories, axis=0), np.array(radii), edge_mask=None, equilibrium_iterations=10)
    action = weighted_first_control(result.weights[0], robot_controls)
    return _clip_speed_and_accel(env.robot_vel, action, env.dt, env.config)


def _step_robot_scripted_probe(env: SyntheticEnv, ctrl_state: RobotControllerState) -> np.ndarray:
    """Deterministic, bounded, conflict-triggered probe (guide.md Order
    6.4): when the nearest human enters a conflict window, execute one of
    a fixed rotation of {left offset, right offset, brief stop} for
    ``PROBE_DURATION_STEPS`` steps, then resume goal-directed nominal.
    Reproducible: the probe SEQUENCE is entirely determined by how many
    times a conflict has been triggered so far, not by any RNG draw."""
    state = ctrl_state.scripted_probe_state
    to_goal = env.robot_goal - env.robot_pos
    dist = float(np.hypot(to_goal[0], to_goal[1]))
    direction = to_goal / dist if dist > 1e-6 else np.zeros(2)
    nominal = direction * env.robot_pref_speed
    lateral = np.array([-direction[1], direction[0]])

    nearest_conflict = False
    if env.humans:
        dists = [float(np.hypot(h.pos[0] - env.robot_pos[0], h.pos[1] - env.robot_pos[1])) for h in env.humans]
        nearest = env.humans[int(np.argmin(dists))]
        combined_radius = env.robot_radius + nearest.radius
        nearest_conflict, _, _ = compute_conflict(env.robot_pos, env.robot_vel, nearest.pos, nearest.vel, combined_radius, env.config, env.dt)

    if state.probe_phase is None and nearest_conflict:
        state.probe_phase = PROBE_CYCLE[state.trigger_count % len(PROBE_CYCLE)]
        state.trigger_count += 1
        state.phase_timer = PROBE_DURATION_STEPS

    if state.probe_phase is not None:
        if state.probe_phase == "left":
            target = nominal * 0.5 + lateral * env.robot_pref_speed * 0.6
        elif state.probe_phase == "right":
            target = nominal * 0.5 - lateral * env.robot_pref_speed * 0.6
        else:
            target = np.zeros(2)
        state.phase_timer -= 1
        if state.phase_timer <= 0:
            state.probe_phase = None
    else:
        target = nominal

    return _clip_speed_and_accel(env.robot_vel, target, env.dt, env.config)


def compute_robot_action(env: SyntheticEnv, ctrl_state: RobotControllerState) -> Tuple[np.ndarray, Optional[dict]]:
    """Dispatch to the controller named in ``ctrl_state.controller_type``.
    Any exception (e.g. a degenerate BRNE solve, an RVO2 failure) is caught,
    counted in ``ctrl_state.fallback_count``, and falls back to
    ``goal_directed`` -- guide.md Order 6.6: "controller 失败必须计数并写
    event，不能静默换 controller 后保留原标签". Returns
    ``(action, fallback_event_or_None)``; the caller is responsible for
    appending the event to the episode's ``events`` list (so it is saved,
    not just counted in memory)."""
    try:
        if ctrl_state.controller_type == "goal_directed":
            return env.step_robot_goal_directed(), None
        elif ctrl_state.controller_type == "orca":
            return _step_robot_orca(env, ctrl_state), None
        elif ctrl_state.controller_type == "original_brne":
            return _step_robot_original_brne(env, ctrl_state), None
        elif ctrl_state.controller_type == "scripted_probe":
            return _step_robot_scripted_probe(env, ctrl_state), None
        raise ControllerConfigError(f"unknown controller_type {ctrl_state.controller_type!r}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: ANY controller failure must fall back+count, not crash collection
        ctrl_state.fallback_count += 1
        fallback_event = {"type": "controller_fallback", "controller": ctrl_state.controller_type, "error": str(exc)[:200]}
        return env.step_robot_goal_directed(), fallback_event


# --------------------------------------------------------------------- #
# Profile ranges (guide.md 5.3): train/validation and test_heldout_
# interactive MUST have non-overlapping ranges across at least stop
# duration, turn angle/rate, yield threshold, assertiveness probability,
# speed range, and goal-switch location/time. Asserted disjoint at import
# time below -- this is a hard invariant, not just a selftest check.
# --------------------------------------------------------------------- #


class ProfileConfigError(ValueError):
    pass


# C5 fix (2026-08-03 third Step 4 audit): the six dimensions here are
# exactly guide.md 5.3's named list -- stop duration, turn angle/rate,
# YIELD THRESHOLD (a trigger condition, not the response magnitude
# ``yield_gain``), assertiveness probability, speed range, and GOAL-SWITCH
# LOCATION/TIME (a trigger condition, not the offset magnitude
# ``goal_switch_lateral``). An earlier version varied the two magnitude
# fields instead of the two trigger fields, which the audit correctly
# rejected as not meeting the six-dimension requirement.
_PROFILE_FIELDS = ("stop_duration_steps", "turn_gain", "yield_ttc_threshold", "assertive_probability", "speed", "goal_switch_ttc_threshold")


@dataclass(frozen=True)
class ProfileRanges:
    stop_duration_steps: Tuple[int, int]
    turn_gain: Tuple[float, float]
    yield_ttc_threshold: Tuple[float, float]
    assertive_probability: Tuple[float, float]
    speed: Tuple[float, float]
    goal_switch_ttc_threshold: Tuple[float, float]

    def __post_init__(self) -> None:
        for name in _PROFILE_FIELDS:
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ProfileConfigError(f"{name} range must have lo <= hi, got ({lo}, {hi})")


TRAIN_VALIDATION_PROFILE = ProfileRanges(
    stop_duration_steps=(4, 8),
    turn_gain=(0.4, 0.7),
    yield_ttc_threshold=(2.0, 3.0),
    assertive_probability=(0.1, 0.4),
    speed=(0.8, 1.2),
    goal_switch_ttc_threshold=(2.0, 3.0),
)

HELDOUT_INTERACTIVE_PROFILE = ProfileRanges(
    stop_duration_steps=(10, 16),
    turn_gain=(0.9, 1.3),
    yield_ttc_threshold=(4.0, 6.0),
    assertive_probability=(0.5, 0.8),
    speed=(1.4, 1.8),
    goal_switch_ttc_threshold=(4.0, 6.0),
)

# test_nominal reuses train/validation's kinematic ranges -- irrelevant in
# practice since NOMINAL_BEHAVIOR_MIX puts 100% mass on CONTINUE, which
# ignores every one of these fields.
NOMINAL_PROFILE = TRAIN_VALIDATION_PROFILE


def assert_profiles_disjoint(a: ProfileRanges, b: ProfileRanges) -> None:
    for name in _PROFILE_FIELDS:
        a_lo, a_hi = getattr(a, name)
        b_lo, b_hi = getattr(b, name)
        if a_hi >= b_lo and b_hi >= a_lo:
            raise ProfileConfigError(f"{name} ranges overlap: train/validation={a_lo, a_hi} vs heldout_interactive={b_lo, b_hi}")


assert_profiles_disjoint(TRAIN_VALIDATION_PROFILE, HELDOUT_INTERACTIVE_PROFILE)


@dataclass(frozen=True)
class SampledEpisodeParams:
    stop_release_steps: int
    turn_gain: float
    yield_ttc_threshold: float
    assertive_probability: float
    speed_lo: float
    speed_hi: float
    goal_switch_ttc_threshold: float


def sample_profile_params(rng: np.random.Generator, profile: ProfileRanges) -> SampledEpisodeParams:
    return SampledEpisodeParams(
        stop_release_steps=int(rng.integers(profile.stop_duration_steps[0], profile.stop_duration_steps[1] + 1)),
        turn_gain=float(rng.uniform(*profile.turn_gain)),
        yield_ttc_threshold=float(rng.uniform(*profile.yield_ttc_threshold)),
        assertive_probability=float(rng.uniform(*profile.assertive_probability)),
        speed_lo=profile.speed[0],
        speed_hi=profile.speed[1],
        goal_switch_ttc_threshold=float(rng.uniform(*profile.goal_switch_ttc_threshold)),
    )


def _behavior_mix_from_assertive_probability(p_assertive: float) -> Dict[BehaviorType, float]:
    others = [bt for bt in BehaviorType if bt != BehaviorType.ASSERTIVE]
    remaining = max(1.0 - p_assertive, 1e-6)
    mix = {bt: remaining / len(others) for bt in others}
    mix[BehaviorType.ASSERTIVE] = p_assertive
    return mix


def _config_from_params(params: SampledEpisodeParams) -> ProtocolConfig:
    return ProtocolConfig(
        turn_gain=params.turn_gain, stop_release_steps=params.stop_release_steps,
        yield_ttc_threshold=params.yield_ttc_threshold, goal_switch_ttc_threshold=params.goal_switch_ttc_threshold,
    )


VALID_SPLITS = ("train", "validation", "test_nominal", "test_heldout_interactive")


class ScenarioConfigError(ValueError):
    pass


def _profile_and_mix_for_split(split: str, rng: np.random.Generator) -> Tuple[SampledEpisodeParams, Dict[BehaviorType, float]]:
    if split in ("train", "validation"):
        params = sample_profile_params(rng, TRAIN_VALIDATION_PROFILE)
        mix = _behavior_mix_from_assertive_probability(params.assertive_probability)
    elif split == "test_heldout_interactive":
        params = sample_profile_params(rng, HELDOUT_INTERACTIVE_PROFILE)
        mix = _behavior_mix_from_assertive_probability(params.assertive_probability)
    elif split == "test_nominal":
        params = sample_profile_params(rng, NOMINAL_PROFILE)
        mix = NOMINAL_BEHAVIOR_MIX
    else:
        raise ScenarioConfigError(f"unknown split {split!r}, expected one of {VALID_SPLITS}")
    return params, mix


# --------------------------------------------------------------------- #
# Scenario table (guide.md 5.3/10.4): matches
# configs/policy_bayesian_brne.config's [eval_envs] geometry:size:n_humans
# exactly. Fail-closed dispatch (make_scenario) is the C2/C3 fix: an
# earlier version accepted all six scenario NAMES but silently routed every
# one of them through the circle generator, so e.g. "baseline_square"
# produced circle-shaped data still labeled "baseline_square". Now "circle"
# and "square" are genuinely different geometries, and any name/split not
# in the tables below raises immediately.
# --------------------------------------------------------------------- #

SCENARIO_TABLE: Dict[str, Tuple[str, float, int]] = {
    "baseline_circle": ("circle", 4.0, 5),
    "baseline_square": ("square", 10.0, 10),
    "dense_circle": ("circle", 4.0, 10),
    "dense_square": ("square", 10.0, 20),
    "large_circle": ("circle", 6.0, 12),
    "large_square": ("square", 14.0, 20),
}

# C3 fix (2026-08-03 third Step 4 audit): an earlier version sampled each
# agent's position independently, with no minimum-clearance check against
# already-placed agents. Independent audit found 25.6%-55.8% of initial
# robot-human/human-human pairs across the six scenarios started in
# physical overlap (center distance < combined radius) -- exactly the kind
# of contaminated initial state that would corrupt mode fitting, NLL, and
# every downstream metric if fed into formal collection. Every agent below
# is now placed via bounded rejection sampling against ALL previously
# placed agents (including the robot), requiring a full
# ``radius_i + radius_j + INITIAL_CLEARANCE_MARGIN`` gap; running out of
# attempts is a hard, explicit ScenarioConfigError, never a silent
# best-effort placement.
INITIAL_CLEARANCE_MARGIN = 0.2  # matches configs/policy_bayesian_brne.config's discomfort_threshold
MAX_PLACEMENT_ATTEMPTS = 500


def _min_required_distance(radius_a: float, radius_b: float) -> float:
    return radius_a + radius_b + INITIAL_CLEARANCE_MARGIN


def _is_clear(candidate_pos: np.ndarray, candidate_radius: float,
              placed_positions: List[np.ndarray], placed_radii: List[float]) -> bool:
    for other_pos, other_radius in zip(placed_positions, placed_radii):
        dist = float(np.hypot(candidate_pos[0] - other_pos[0], candidate_pos[1] - other_pos[1]))
        if dist < _min_required_distance(candidate_radius, other_radius):
            return False
    return True


def _place_circle_point(rng: np.random.Generator, base_angle: float, jitter_range: float, radius: float,
                         own_radius: float, placed_positions: List[np.ndarray], placed_radii: List[float]) -> np.ndarray:
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        angle = base_angle + rng.uniform(-jitter_range, jitter_range)
        pos = radius * np.array([np.cos(angle), np.sin(angle)])
        if _is_clear(pos, own_radius, placed_positions, placed_radii):
            return pos
    raise ScenarioConfigError(
        f"failed to place an agent on the circle (radius={radius}) without overlap after "
        f"{MAX_PLACEMENT_ATTEMPTS} attempts -- {len(placed_positions)} agents already placed, "
        f"required clearance={own_radius}+other+{INITIAL_CLEARANCE_MARGIN}. Scenario may be too "
        "dense for this radius/margin."
    )


def _make_circle_scenario(rng: np.random.Generator, n_humans: int, radius: float, dt: float,
                           params: SampledEpisodeParams, mix: Dict[BehaviorType, float]) -> SyntheticEnv:
    """N humans placed on a circle, goals on the opposite side; robot on
    the circle too. All placements are collision-free at t=0 (see
    ``_place_circle_point``)."""
    config = _config_from_params(params)
    humans: List[SyntheticHuman] = []
    placed_positions: List[np.ndarray] = []
    placed_radii: List[float] = []
    for i in range(n_humans):
        base_angle = 2.0 * np.pi * i / n_humans
        pos = _place_circle_point(rng, base_angle, 0.1, radius, 0.3, placed_positions, placed_radii)
        placed_positions.append(pos)
        placed_radii.append(0.3)
        goal = -pos
        pref_speed = float(rng.uniform(params.speed_lo, params.speed_hi))
        behavior_type = sample_behavior_type(rng, mix)
        humans.append(
            SyntheticHuman(
                track_id=i, pos=pos, vel=np.zeros(2), goal=goal, radius=0.3,
                pref_speed=pref_speed, behavior_state=HumanBehaviorState(behavior_type=behavior_type),
            )
        )
    robot_pos = _place_circle_point(rng, 0.0, np.pi, radius, 0.3, placed_positions, placed_radii)
    robot_goal = -robot_pos
    return SyntheticEnv(
        humans=humans, robot_pos=robot_pos, robot_vel=np.zeros(2), robot_goal=robot_goal,
        robot_radius=0.3, robot_pref_speed=1.0, dt=dt, config=config, sampled_params=params,
    )


def _square_perimeter_point(t: float, half: float) -> np.ndarray:
    """``t`` in ``[0, 1)`` parameterizes a ``2*half``-side square's
    perimeter, starting at the right edge's bottom corner and going
    counterclockwise. A genuinely different spatial layout from the circle
    generator: every point here has ``max(|x|, |y|) == half`` (square
    perimeter), never ``hypot(x, y) == radius`` (circle) -- the two
    geometries are distinguishable from the generated positions alone."""
    t = t % 1.0
    side = min(int(t * 4), 3)
    frac = t * 4 - side
    if side == 0:
        return np.array([half, -half + 2 * half * frac])
    elif side == 1:
        return np.array([half - 2 * half * frac, half])
    elif side == 2:
        return np.array([-half, half - 2 * half * frac])
    else:
        return np.array([-half + 2 * half * frac, -half])


def _place_square_point(rng: np.random.Generator, base_t: float, jitter_range: float, half: float,
                         own_radius: float, placed_positions: List[np.ndarray], placed_radii: List[float]) -> np.ndarray:
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t = base_t + rng.uniform(-jitter_range, jitter_range)
        pos = _square_perimeter_point(t, half)
        if _is_clear(pos, own_radius, placed_positions, placed_radii):
            return pos
    raise ScenarioConfigError(
        f"failed to place an agent on the square perimeter (half={half}) without overlap after "
        f"{MAX_PLACEMENT_ATTEMPTS} attempts -- {len(placed_positions)} agents already placed, "
        f"required clearance={own_radius}+other+{INITIAL_CLEARANCE_MARGIN}. Scenario may be too "
        "dense for this side length/margin."
    )


def _make_square_scenario(rng: np.random.Generator, n_humans: int, side: float, dt: float,
                           params: SampledEpisodeParams, mix: Dict[BehaviorType, float]) -> SyntheticEnv:
    """N humans placed on the perimeter of a ``side x side`` square
    (centered at the origin), each walking to the point reflected through
    the center -- same crossing-paths interaction intent as the circle
    scenario, but a genuinely different geometry (see
    ``_square_perimeter_point``). All placements are collision-free at t=0
    (see ``_place_square_point``)."""
    half = side / 2.0
    config = _config_from_params(params)
    humans: List[SyntheticHuman] = []
    placed_positions: List[np.ndarray] = []
    placed_radii: List[float] = []
    for i in range(n_humans):
        base_t = i / n_humans
        pos = _place_square_point(rng, base_t, 0.02, half, 0.3, placed_positions, placed_radii)
        placed_positions.append(pos)
        placed_radii.append(0.3)
        goal = -pos
        pref_speed = float(rng.uniform(params.speed_lo, params.speed_hi))
        behavior_type = sample_behavior_type(rng, mix)
        humans.append(
            SyntheticHuman(
                track_id=i, pos=pos, vel=np.zeros(2), goal=goal, radius=0.3,
                pref_speed=pref_speed, behavior_state=HumanBehaviorState(behavior_type=behavior_type),
            )
        )
    robot_pos = _place_square_point(rng, 0.5, 0.5, half, 0.3, placed_positions, placed_radii)
    robot_goal = -robot_pos
    return SyntheticEnv(
        humans=humans, robot_pos=robot_pos, robot_vel=np.zeros(2), robot_goal=robot_goal,
        robot_radius=0.3, robot_pref_speed=1.0, dt=dt, config=config, sampled_params=params,
    )


# C4 fix (2026-08-03 third Step 4 audit): guide.md 5.3 is explicit that
# Train/Validation use ONLY ``baseline_circle`` (5-person) for fitting; all
# six scenarios are for TEST (nominal and heldout_interactive) only. An
# earlier version let ``make_scenario`` build any of the six scenarios under
# any split -- e.g. ``make_scenario('dense_square', 'train', ...)``
# succeeded and returned a 20-person environment, which the audit correctly
# flagged as contradicting the 5-person-fitting constraint. The valid
# (scenario, split) combinations are now exactly 2 (baseline_circle x
# {train, validation}) + 12 (all six scenarios x the two test splits) = 14,
# not the full 6x4=24 cartesian product.
TRAIN_VALIDATION_SCENARIO = "baseline_circle"


def make_scenario(name: str, split: str, rng: np.random.Generator, dt: float) -> SyntheticEnv:
    """Fail-closed dispatcher (C2/C3/C4 fix): unknown scenario names or
    splits raise ``ScenarioConfigError`` immediately rather than silently
    falling back to some default geometry and mislabeling the resulting
    data; so does any (scenario, split) combination outside the 14 valid
    ones (see ``TRAIN_VALIDATION_SCENARIO`` above)."""
    if name not in SCENARIO_TABLE:
        raise ScenarioConfigError(f"unknown scenario {name!r}, expected one of {sorted(SCENARIO_TABLE)}")
    if split not in VALID_SPLITS:
        raise ScenarioConfigError(f"unknown split {split!r}, expected one of {VALID_SPLITS}")
    if split in ("train", "validation") and name != TRAIN_VALIDATION_SCENARIO:
        raise ScenarioConfigError(
            f"split {split!r} only permits scenario {TRAIN_VALIDATION_SCENARIO!r} (guide.md 5.3: "
            f"Train/Validation 只用 5 人 baseline_circle 拟合), got {name!r}. All six scenarios are "
            "for test_nominal/test_heldout_interactive only."
        )
    geometry, size, n_humans = SCENARIO_TABLE[name]
    params, mix = _profile_and_mix_for_split(split, rng)
    if geometry == "circle":
        return _make_circle_scenario(rng, n_humans, size, dt, params, mix)
    elif geometry == "square":
        return _make_square_scenario(rng, n_humans, size, dt, params, mix)
    raise ScenarioConfigError(f"scenario {name!r} has unknown geometry {geometry!r}")


def _profile_params_dict(params: Optional["SampledEpisodeParams"], config: ProtocolConfig) -> Dict[str, object]:
    """Order 5d.2: canonical-serializable dict of every profile parameter
    guide.md names (stop duration, turn gain, yield gain, yield trigger
    threshold, assertive probability, speed range, goal-switch lateral,
    goal-switch trigger window/threshold)."""
    if params is None:
        return {
            "stop_release_steps": int(config.stop_release_steps), "turn_gain": float(config.turn_gain),
            "yield_gain": float(config.yield_gain), "yield_ttc_threshold": float(config.yield_ttc_threshold),
            "assertive_probability": None, "speed_lo": None, "speed_hi": None,
            "goal_switch_lateral": float(config.goal_switch_lateral),
            "goal_switch_ttc_threshold": float(config.goal_switch_ttc_threshold),
        }
    return {
        "stop_release_steps": int(params.stop_release_steps), "turn_gain": float(params.turn_gain),
        "yield_gain": float(config.yield_gain), "yield_ttc_threshold": float(params.yield_ttc_threshold),
        "assertive_probability": float(params.assertive_probability),
        "speed_lo": float(params.speed_lo), "speed_hi": float(params.speed_hi),
        "goal_switch_lateral": float(config.goal_switch_lateral),
        "goal_switch_ttc_threshold": float(params.goal_switch_ttc_threshold),
    }


def run_episode(
    env: SyntheticEnv,
    horizon_steps: int,
    controller: str = "goal_directed",
    scenario: str = "unknown_scenario",
    profile: str = "unset_profile",
    split: str = "unset_split",
    profile_name: str = "unset_profile_name",
    brne_root: str = _BRNE_DEFAULT_ROOT,
) -> Dict[str, object]:
    """Roll ``env`` forward ``horizon_steps`` dt's, recording exactly the
    guide.md 5.5 schema fields (minus schema_version/robot_state_layout,
    which data_io.save_episode stamps) PLUS Order 5d's traceability fields
    (split/controller_type/profile_name/profile_params/initial_state_hash/
    generator_version/behavior_type_map). ``controller`` selects the
    robot's own policy each step (guide.md 5.4, Order 6): one of
    CONTROLLER_TYPES. ``scenario``/``profile`` are recorded as-is into the
    returned dict (callers should pass the SAME name used to build ``env``
    via ``make_scenario``, so the saved label always matches the actual
    generated geometry -- the exact mislabeling bug C2 flagged)."""
    from crowd_nav.bayesian_brne.data_io import compute_initial_state_hash

    n = len(env.humans)
    T = horizon_steps
    robot_arr = np.zeros((T, 9))
    humans_arr = np.zeros((T, n, 5))
    track_ids_arr = np.zeros((T, n), dtype=np.int64)
    robot_actions = np.zeros((T, 2))
    human_actions = np.zeros((T, n, 2))
    latent_labels = np.zeros((T, n), dtype=np.int64)
    valid_mask = np.ones((T, n), dtype=bool)
    events: List[dict] = []

    behavior_to_int = {bt: i for i, bt in enumerate(BehaviorType)}
    behavior_type_map = {str(human.track_id): human.behavior_state.behavior_type.value for human in env.humans}
    profile_params = _profile_params_dict(env.sampled_params, env.config)

    ctrl_state = RobotControllerState(controller_type=controller, brne_root=brne_root, brne_rng=np.random.default_rng(0))

    initial_state_hash = compute_initial_state_hash(
        env.robot_state_row(),
        np.array([[h.pos[0], h.pos[1], h.vel[0], h.vel[1], h.radius] for h in env.humans]) if n else np.zeros((0, 5)),
        behavior_type_map, controller, profile_name, profile_params,
    )

    for t in range(T):
        robot_arr[t] = env.robot_state_row()
        for i, human in enumerate(env.humans):
            humans_arr[t, i] = [human.pos[0], human.pos[1], human.vel[0], human.vel[1], human.radius]
            track_ids_arr[t, i] = human.track_id
            latent_labels[t, i] = behavior_to_int[human.behavior_state.behavior_type]

        robot_action, fallback_event = compute_robot_action(env, ctrl_state)
        if fallback_event is not None:
            events.append({"t": t, **fallback_event})
        robot_actions[t] = robot_action
        result = env.step(robot_action)
        for i, human in enumerate(env.humans):
            human_actions[t, i] = human.vel
            if result["conflicts"][i]:
                events.append({"t": t, "track_id": human.track_id, "type": "conflict"})

    return {
        "suite_seed": 0,
        "episode_seed": 0,
        "scenario": scenario,
        "profile": profile,
        "dt": env.dt,
        "split": split,
        "controller_type": controller,
        "profile_name": profile_name,
        "generator_version": GENERATOR_VERSION,
        "initial_state_hash": initial_state_hash,
        "profile_params": profile_params,
        "behavior_type_map": behavior_type_map,
        "robot": robot_arr,
        "humans": humans_arr,
        "human_track_ids": track_ids_arr,
        "robot_actions": robot_actions,
        "human_actions": human_actions,
        "events": events,
        "latent_behavior_labels": latent_labels,
        "valid_mask": valid_mask,
    }
