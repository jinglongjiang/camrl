"""FROZEN multimodal junction scenario (consolidation plan Order 5).

Per the reset investigation's hard-won principle (guide.md / bdvl_
consolidation_plan.md §4b, found via reset_exp/step4-step4b): belief only
has decision value when the COLLISION RISK ITSELF depends on the human's
unknown choice -- temporal overlap between "ambiguous" and "in conflict"
is NOT sufficient (step4b showed 120/120 identical actions despite real
temporal overlap, because the robot only met the pedestrian during its
UNAMBIGUOUS pre-fork descent). This scenario is built so the robot is
FORCED to commit to a side of the junction WHILE the pedestrian's exit is
still genuinely unknown, and a wrong guess collides.

Non-invasive by construction (same pattern as R4-4's InterventionORCA):
built entirely from existing CrowdSim/Robot/Human/ORCA machinery, calling
``env.reset()`` normally then REPLACING the human list with junction-
scenario humans -- crowd_sim.py itself is never touched.

FROZEN geometry + seed ranges. Per consolidation plan hard requirement 1:
frozen BEFORE any RL collection, train/heldout seeds are DISJOINT ranges
(same convention as config.SEED_ROLES's 9xxxx blocks), and are NOT to be
adjusted after seeing results.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.intent_runtime_config import FROZEN_VALUES, robot_visible_from
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.scene_candidates import PublicScene, junction_scene
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.robot import Robot


class JunctionScenarioError(ValueError):
    pass


# --- FROZEN geometry (do not tune post-hoc; a change here is a new,
# separately-versioned scenario, not an edit to this one). ---
# guide/consolidation plan §4b hard principle: collision risk must depend
# on the pedestrian's UNKNOWN exit choice. A first attempt at this geometry
# (junction 4m from robot, exits 2m past the junction) was measured and
# found NOT to force any conflict at all -- even a fully naive, goal-blind
# robot (straight line to its own goal, ignoring the human) never got
# closer than 0.86m across 6 real seeds. Root cause: pedestrian and robot
# occupied disjoint y-bands for almost the whole episode (pedestrian
# finished its short walk long before the robot's slower crossing arrived).
# This COMPACT geometry (mirrors the validated bimodal_ceiling.py scale --
# short pedestrian route, robot crossing the SAME region at a matched time)
# was verified by measurement (not assumed) to force a real conflict: see
# reset_exp-style timing check before this was frozen.
JUNCTION_WAYPOINT: Tuple[float, float] = (0.0, 3.0)
EXIT_LEFT: Tuple[float, float] = (-1.2, 1.5)
EXIT_RIGHT: Tuple[float, float] = (1.2, 1.5)
PEDESTRIAN_START_Y = 4.0
PEDESTRIAN_SPEED_RANGE: Tuple[float, float] = (0.6, 1.1)
ROBOT_GOAL: Tuple[float, float] = (0.0, 5.0)
# robot starts below the whole junction/exit region and crosses straight
# through it -- timed (by construction: comparable speeds, comparable
# distances) to be IN the fork region while the pedestrian's exit is still
# being resolved, not before or after it.
ROBOT_START_Y_RANGE: Tuple[float, float] = (0.3, 0.7)
# V2: the junction-crowd BIRTH DISTRIBUTION and the CANDIDATE SEMANTICS both
# changed (backgrounds are rejection-sampled out of the approach corridor, and
# crossers get crossing candidates instead of junction exits). Episodes from
# v1 and v2 are different environments and must never be pooled or compared;
# v1 corpora, weights and test seeds are all retired.
SCENARIO_REGISTRY_ID = "bdvl_junction_scenario_v2_corridor_split"

# --- FROZEN, DISJOINT seed ranges (consolidation plan hard requirement 1) ---
JUNCTION_TRAIN_SEEDS: Tuple[int, ...] = tuple(range(96001, 96201))     # 200
JUNCTION_HELDOUT_SEEDS: Tuple[int, ...] = tuple(range(96501, 96601))   # 100, disjoint block


def _assert_seed_ranges_disjoint() -> None:
    train, heldout = set(JUNCTION_TRAIN_SEEDS), set(JUNCTION_HELDOUT_SEEDS)
    if train & heldout:
        raise JunctionScenarioError(f"train/heldout seed ranges must be disjoint, overlap={train & heldout}")


_assert_seed_ranges_disjoint()


def public_junction_scene() -> PublicScene:
    """The PUBLIC scene description the candidate provider (and the label-
    leakage tests) may use -- exits only, no hidden per-episode goal."""
    return junction_scene(JUNCTION_WAYPOINT, [("left", EXIT_LEFT), ("right", EXIT_RIGHT)])


@dataclass(frozen=True)
class JunctionEpisodeConfig:
    episode_seed: int
    is_heldout: bool

    def __post_init__(self) -> None:
        role_set = JUNCTION_HELDOUT_SEEDS if self.is_heldout else JUNCTION_TRAIN_SEEDS
        if self.episode_seed not in role_set:
            raise JunctionScenarioError(
                f"episode_seed {self.episode_seed} is not in the frozen "
                f"{'heldout' if self.is_heldout else 'train'} seed range"
            )


def make_env(env_config_path: Path, human_num: int = 1):
    if env_config_path.name != "env_bayesian_dvl.config":
        raise JunctionScenarioError("junction scenario must use env_bayesian_dvl.config")
    if human_num < 1:
        raise JunctionScenarioError(f"human_num must be >= 1, got {human_num}")
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not env_config.read(str(env_config_path)):
        raise JunctionScenarioError(f"env config not found: {env_config_path}")
    env_config.set("sim", "human_num", str(int(human_num)))
    env_config.set("robot", "policy", "orca")
    env_config.set("sim", "train_val_sim", "circle_crossing")  # placeholder; humans replaced below
    env_config.set("sim", "test_sim", "circle_crossing")
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "train"
    return env, env_config


WAYPOINT_RADIUS = 0.35


def exit_position(true_exit: str) -> Tuple[float, float]:
    if true_exit not in ("left", "right"):
        raise JunctionScenarioError(f"true_exit must be 'left' or 'right', got {true_exit!r}")
    return EXIT_LEFT if true_exit == "left" else EXIT_RIGHT


def maybe_reveal_exit(pedestrian, true_exit: str, waypoint_reached: bool) -> bool:
    """Call ONCE PER STEP, before ``pedestrian.act()``. Real bug found by
    external review: the original version set the pedestrian's goal
    DIRECTLY to its final exit at episode start, so ORCA's preferred
    velocity diverged left/right from the very first simulated step --
    measured: shared direction [0,-1] vs left [-0.433,-0.902] vs right
    [0.433,-0.902], i.e. the "ambiguity" the tracker measured was the
    TRACKER's own model being wrong about the real (already-resolved)
    trajectory, not genuine environmental unidentifiability.

    Fix: the pedestrian's goal starts at the SHARED junction waypoint
    (identical regardless of true_exit) and is switched to the true exit
    ONLY once the pedestrian is physically within WAYPOINT_RADIUS of the
    junction -- so two pedestrians with the same start position/speed and
    DIFFERENT true_exit produce BIT-IDENTICAL position sequences until the
    switch point (see test_junction_pedestrian_left_right_identical_
    before_fork), which is what "ambiguous before the fork" is actually
    supposed to mean.
    """
    if waypoint_reached:
        return True
    d = float(np.hypot(pedestrian.px - JUNCTION_WAYPOINT[0], pedestrian.py - JUNCTION_WAYPOINT[1]))
    if d <= WAYPOINT_RADIUS:
        ex, ey = exit_position(true_exit)
        pedestrian.gx, pedestrian.gy = ex, ey
        return True
    return False


def build_junction_episode(env_config_path: Path, cfg: JunctionEpisodeConfig):
    """Builds ONE junction episode: standard CrowdSim/Robot/ORCA machinery,
    with the human list REPLACED (never crowd_sim.py edited) by a single
    pedestrian whose hidden true exit is drawn from the frozen seed. The
    pedestrian's goal starts at the SHARED junction waypoint (not the
    exit) -- callers MUST invoke ``maybe_reveal_exit`` once per step (see
    its docstring) to switch the goal once the junction is physically
    reached. Returns (env, robot, pedestrian_true_exit_name) -- the true
    exit name is for driving the sim + post-hoc scoring ONLY, never for
    the tracker/provider.
    """
    env, env_config = make_env(env_config_path)
    robot = Robot(env_config, "robot")
    robot_orca = ORCA(); robot_orca.configure(env_config)
    # see intent_train.py's _make_standard_env comment: CrowdSim.reset()
    # forces human_num=1 unless this is set, regardless of config. This
    # scenario wants exactly 1 pedestrian anyway, but set it explicitly
    # rather than relying on that as an accident.
    robot_orca.multiagent_training = True
    robot.set_policy(robot_orca)
    # Order 12: invisible robot; the config is the single definition.
    robot.visible = robot_visible_from(env_config)
    robot.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    env.case_counter["train"] = cfg.episode_seed % (2**32 - 1)
    env.reset()

    rng = np.random.default_rng(cfg.episode_seed)
    true_exit = "left" if rng.random() < 0.5 else "right"
    ped_speed = float(rng.uniform(*PEDESTRIAN_SPEED_RANGE))
    ped_x = float(rng.uniform(-0.3, 0.3))

    pedestrian = env.humans[0]
    ped_orca = ORCA(); ped_orca.configure(env_config)
    pedestrian.set_policy(ped_orca)
    # goal starts at the SHARED junction waypoint, NOT the exit -- see
    # maybe_reveal_exit's docstring. Callers must call maybe_reveal_exit
    # every step to switch the goal once the junction is physically reached.
    pedestrian.set(ped_x, PEDESTRIAN_START_Y, JUNCTION_WAYPOINT[0], JUNCTION_WAYPOINT[1], 0.0, 0.0, 0.0)
    pedestrian.v_pref = ped_speed

    robot_y = float(rng.uniform(*ROBOT_START_Y_RANGE))
    robot_x = float(rng.uniform(-0.3, 0.3))
    robot.set(robot_x, robot_y, ROBOT_GOAL[0], ROBOT_GOAL[1], 0.0, 0.0, np.pi / 2)

    return env, robot, true_exit


# --------------------------------------------------------------------- #
# Order C1.2/C1.3: the 5-person `junction_crowd` scenario.
#
# Plan 2.2 point 11: the 1-person junction above is a good MECHANISM unit
# test (it isolates the hidden L/R choice with nothing else going on) but
# cannot by itself support a crowd-navigation stress claim. The training
# scenario is "1 ambiguous pedestrian + N background ORCA pedestrians":
# the multimodal conflict must still come from the hidden L/R choice, while
# the robot simultaneously has to handle ordinary crowd traffic.
#
# Background pedestrians are ordinary CrowdSim agents with their OWN goals.
# Those goals are hidden state for the tracker exactly like the ambiguous
# pedestrian's exit is -- the tracker only ever sees observable positions
# (enforced by test_main_chain_never_reads_hidden_goal + the
# candidate_fn(track_id, position) signature).
# --------------------------------------------------------------------- #

JUNCTION_CROWD_HUMAN_NUM = 5           # 1 ambiguous + 4 background
AMBIGUOUS_TRACK_INDEX = 0              # env.humans[0] is always the ambiguous one

# PUBLIC approach corridor: the strip that leads into the junction. An entry
# inside it is on the approach and gets the left/right exit candidates; an
# entry outside it is a lateral crosser and gets crossing candidates.
#
# This is the geometric fact the scene previously ASSERTED but did not
# ENFORCE. The old background sampler could place a crosser at, say,
# (0.1, 4.0) -- indistinguishable from the ambiguous pedestrian's start --
# so no observable rule could have separated the two populations, and the
# candidate function applied the junction rule to all five. Backgrounds are
# now rejection-sampled OUT of the corridor, which makes corridor membership
# a sound public discriminator instead of a claim.
JUNCTION_APPROACH_HALF_WIDTH = 0.6     # ped_x is drawn from [-0.3, 0.3], so the ambiguous entry is always inside
JUNCTION_APPROACH_Y_MIN = JUNCTION_WAYPOINT[1]   # the corridor is the stretch BEFORE the junction
CROSSING_BAND_N = 4                    # crossers' endpoint along the band stays genuinely uncertain

# (The first-frame speed prior lives in intent_runtime_config.TRACKER_DEFAULTS
# so it is one global, hashed value rather than a per-scene constant that
# nothing reads -- an earlier draft of this fix defined it here and never
# wired it into the call chain.)

# TRAIN parameter ranges (frozen before any collection).
CROWD_TRAIN_BACKGROUND_X_RANGE: Tuple[float, float] = (-2.2, 2.2)
CROWD_TRAIN_BACKGROUND_Y_RANGE: Tuple[float, float] = (1.0, 4.5)
CROWD_TRAIN_BACKGROUND_SPEED_RANGE: Tuple[float, float] = (0.5, 1.0)

# HELD-OUT parameter ranges: plan 2.2 point 10 / section 4.2 -- a genuine
# distribution SHIFT, not merely unseen seeds. Every range below is
# deliberately DISJOINT from, or strictly wider than, its TRAIN
# counterpart, and the exit geometry itself is different (wider fork,
# faster pedestrians, denser background band). Frozen before training;
# never to be retuned after seeing results.
CROWD_HELDOUT_EXIT_LEFT: Tuple[float, float] = (-1.9, 1.35)      # wider fork than EXIT_LEFT (-1.2, 1.5)
CROWD_HELDOUT_EXIT_RIGHT: Tuple[float, float] = (1.9, 1.35)
CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE: Tuple[float, float] = (1.15, 1.45)  # DISJOINT from train's (0.6, 1.1)
CROWD_HELDOUT_BACKGROUND_X_RANGE: Tuple[float, float] = (-2.8, 2.8)
CROWD_HELDOUT_BACKGROUND_Y_RANGE: Tuple[float, float] = (0.8, 5.0)
CROWD_HELDOUT_BACKGROUND_SPEED_RANGE: Tuple[float, float] = (1.05, 1.40)  # DISJOINT from train's (0.5, 1.0)

# FROZEN, disjoint seed blocks for the crowd variant (the 1-person unit
# scenario keeps 96001-96200 / 96501-96600; these are separate blocks so a
# crowd episode can never be confused with a unit episode).
# ---------------------------------------------------------------------
# V2 SEED BLOCKS. Every V1 junction block is RETIRED: those episodes were
# generated by a birth distribution that could place background pedestrians
# inside the approach corridor, so their raw ORCA trajectories cannot be
# re-materialised under the V2 candidate rules -- the crossers in them are
# still indistinguishable from the ambiguous approacher. Nothing from
# 96001-97700 or 1_100_000-1_400_499 may be reused, including the paper-test
# block, which has been seen.
# ---------------------------------------------------------------------
JUNCTION_CROWD_TRAIN_SEEDS: Tuple[int, ...] = tuple(range(2_000_000, 2_000_200))   # 200, mechanism audit
JUNCTION_CROWD_HELDOUT_SEEDS: Tuple[int, ...] = tuple(range(2_010_000, 2_010_100))  # 100, mechanism audit

# C4RF.1: the 200-seed block above is the MECHANISM/unit block (used by the
# tests). Formal training needs one distinct initial layout PER EPISODE.
# Real problem found by audit: the frozen budget says 2500 IL + 5000 online
# junction episodes, but both cycled the SAME 200 seeds -- 12.5x and 25x
# reuse -- and worse, the IL and online sets were IDENTICAL, so the online
# phase explored exactly the layouts it had already imitated. These two
# blocks are large enough for the full budget with NO reuse and NO overlap.
JUNCTION_CROWD_IL_SEEDS: Tuple[int, ...] = tuple(range(2_100_000, 2_102_500))       # 2500
JUNCTION_CROWD_ONLINE_SEEDS: Tuple[int, ...] = tuple(range(2_200_000, 2_205_000))   # 5000
# Development-only navigation-health checks during training. These are
# neither optimizer seeds nor paper/formal held-out identities.
JUNCTION_CROWD_VALIDATION_SEEDS: Tuple[int, ...] = tuple(range(2_300_000, 2_300_010))    # 10
JUNCTION_CROWD_SELECTION_DEV_SEEDS: Tuple[int, ...] = tuple(range(2_400_000, 2_400_100))  # 100
# Never touched by training or selection. Training and selection code must
# REFUSE these; the V1 paper-test block was spent the moment it was read.
JUNCTION_CROWD_PAPER_TEST_SEEDS: Tuple[int, ...] = tuple(range(2_500_000, 2_500_500))     # 500
# Order 14 section 7: STAGE ACCEPTANCE. Closed-loop layouts that decide whether
# the NEXT training stage may start. NOMINAL geometry -- the same junction the
# training distribution uses -- because a stage gate must measure whether the
# policy can finish an episode, not whether it generalises to a shifted scene;
# mixing those two questions makes a failed gate uninterpretable.
#
# Defined HERE and nowhere else. evaluation_protocol imports this constant
# rather than restating the range: the first version of this block was written
# as a literal range in evaluation_protocol only, so the role resolver never
# learned about it and the 90-episode gate crashed on its 61st layout with
# 'seed 3020000 is not in any frozen junction_crowd block'.
JUNCTION_CROWD_STAGE_ACCEPT_SEEDS: Tuple[int, ...] = tuple(range(3_020_000, 3_020_030))   # 30

# Seed -> ROLE, and ROLE -> geometry. One seed belongs to exactly one role.
JUNCTION_CROWD_SEED_ROLES: Dict[str, Tuple[int, ...]] = {
    "mechanism_train": JUNCTION_CROWD_TRAIN_SEEDS,
    "mechanism_heldout": JUNCTION_CROWD_HELDOUT_SEEDS,
    "il": JUNCTION_CROWD_IL_SEEDS,
    "online": JUNCTION_CROWD_ONLINE_SEEDS,
    "validation": JUNCTION_CROWD_VALIDATION_SEEDS,
    "selection_dev": JUNCTION_CROWD_SELECTION_DEV_SEEDS,
    "paper_test": JUNCTION_CROWD_PAPER_TEST_SEEDS,
    "stage_accept": JUNCTION_CROWD_STAGE_ACCEPT_SEEDS,
}
TRAIN_GEOMETRY_ROLES = frozenset({"mechanism_train", "il", "online", "validation", "stage_accept"})
HELDOUT_GEOMETRY_ROLES = frozenset({"mechanism_heldout", "selection_dev", "paper_test"})
# Roles training must never touch, whatever geometry they use. stage_accept is
# the first member that uses TRAIN geometry: "never trained on" and "shifted
# scene" are independent properties, and a stage gate needs the first without
# the second.
FORMAL_ONLY_ROLES = frozenset({"mechanism_heldout", "selection_dev", "paper_test", "stage_accept"})


def _assert_roles_partition() -> None:
    if set(JUNCTION_CROWD_SEED_ROLES) != TRAIN_GEOMETRY_ROLES | HELDOUT_GEOMETRY_ROLES:
        raise JunctionScenarioError("every junction_crowd role must map to exactly one geometry")
    seen: Dict[int, str] = {}
    for role, seeds in JUNCTION_CROWD_SEED_ROLES.items():
        for s_ in seeds:
            if s_ in seen:
                raise JunctionScenarioError(
                    f"seed {s_} is claimed by both {seen[s_]!r} and {role!r}; roles must be a partition")
            seen[s_] = role


_assert_roles_partition()


def _role_of_seed(seed: int):
    for role, seeds in JUNCTION_CROWD_SEED_ROLES.items():
        if seed in seeds:
            return role
    return None


def junction_crowd_role_of_seed(seed: int) -> str:
    role = _role_of_seed(seed)
    if role is None:
        raise JunctionScenarioError(f"seed {seed} is not in any frozen junction_crowd block")
    return role


def _assert_crowd_seed_ranges_disjoint() -> None:
    blocks = {
        "junction_train": set(JUNCTION_TRAIN_SEEDS),
        "junction_heldout": set(JUNCTION_HELDOUT_SEEDS),
        "crowd_train": set(JUNCTION_CROWD_TRAIN_SEEDS),
        "crowd_heldout": set(JUNCTION_CROWD_HELDOUT_SEEDS),
        "crowd_il": set(JUNCTION_CROWD_IL_SEEDS),
        "crowd_online": set(JUNCTION_CROWD_ONLINE_SEEDS),
        "crowd_validation": set(JUNCTION_CROWD_VALIDATION_SEEDS),
    }
    names = sorted(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = blocks[a] & blocks[b]
            if overlap:
                raise JunctionScenarioError(f"seed blocks {a} and {b} overlap: {sorted(overlap)[:5]}")


_assert_crowd_seed_ranges_disjoint()


def _assert_heldout_ranges_are_shifted() -> None:
    """Plan 2.2 point 10: held-out must be a real distribution shift, not
    just new seeds. Enforce that mechanically so it cannot silently rot."""
    for name, train, heldout in (
        ("pedestrian_speed", PEDESTRIAN_SPEED_RANGE, CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE),
        ("background_speed", CROWD_TRAIN_BACKGROUND_SPEED_RANGE, CROWD_HELDOUT_BACKGROUND_SPEED_RANGE),
    ):
        # disjoint intervals: heldout_low must exceed train_high
        if not heldout[0] > train[1]:
            raise JunctionScenarioError(
                f"{name} held-out range {heldout} must be DISJOINT from (above) the train range {train}, "
                f"otherwise 'held-out' is only new seeds, not a distribution shift")
    if abs(CROWD_HELDOUT_EXIT_LEFT[0]) <= abs(EXIT_LEFT[0]):
        raise JunctionScenarioError("held-out fork must be geometrically wider than the train fork")


_assert_heldout_ranges_are_shifted()


@dataclass(frozen=True)
class JunctionCrowdEpisodeConfig:
    """ONE junction-crowd episode, identified by seed AND the ROLE it plays.

    The old interface took ``is_heldout`` alone, so any caller holding the
    heldout flag could run any heldout-block seed -- including the paper-test
    block. Roles make the mapping explicit and one-way: a seed belongs to
    exactly one role, the role determines the geometry, and a caller that
    names a role the seed does not belong to is an error rather than a
    silent reinterpretation.
    """

    episode_seed: int
    role: str

    def __post_init__(self) -> None:
        if self.role not in JUNCTION_CROWD_SEED_ROLES:
            raise JunctionScenarioError(
                f"unknown junction_crowd seed role {self.role!r}; expected one of "
                f"{sorted(JUNCTION_CROWD_SEED_ROLES)}")
        if self.episode_seed not in JUNCTION_CROWD_SEED_ROLES[self.role]:
            actual = _role_of_seed(self.episode_seed)
            raise JunctionScenarioError(
                f"episode_seed {self.episode_seed} is not in the frozen {self.role!r} block"
                + (f" -- it belongs to {actual!r}" if actual else " -- it belongs to no frozen block"))

    @property
    def is_heldout(self) -> bool:
        """Geometry follows from the ROLE, never from a caller's flag."""
        return self.role in HELDOUT_GEOMETRY_ROLES


def public_junction_crowd_scene(is_heldout: bool = False) -> PublicScene:
    """PUBLIC scene for the crowd variant. The held-out variant has a
    genuinely different (wider) fork, so its public geometry differs too --
    which is exactly what makes it a distribution shift rather than a
    reseed. Still public-only: exits and the shared waypoint, never a
    per-pedestrian goal."""
    left = CROWD_HELDOUT_EXIT_LEFT if is_heldout else EXIT_LEFT
    right = CROWD_HELDOUT_EXIT_RIGHT if is_heldout else EXIT_RIGHT
    y0, y1 = CROWD_HELDOUT_BACKGROUND_Y_RANGE if is_heldout else CROWD_TRAIN_BACKGROUND_Y_RANGE
    return junction_scene(
        JUNCTION_WAYPOINT, [("left", left), ("right", right)],
        approach_corridor=(JUNCTION_APPROACH_HALF_WIDTH, JUNCTION_APPROACH_Y_MIN),
        crossing_band=(y0, y1, CROSSING_BAND_N))


def crowd_exit_position(true_exit: str, is_heldout: bool = False) -> Tuple[float, float]:
    if true_exit not in ("left", "right"):
        raise JunctionScenarioError(f"true_exit must be 'left' or 'right', got {true_exit!r}")
    if is_heldout:
        return CROWD_HELDOUT_EXIT_LEFT if true_exit == "left" else CROWD_HELDOUT_EXIT_RIGHT
    return EXIT_LEFT if true_exit == "left" else EXIT_RIGHT


def maybe_reveal_crowd_exit(pedestrian, true_exit: str, waypoint_reached: bool, is_heldout: bool = False) -> bool:
    """``maybe_reveal_exit`` for the crowd variant (held-out uses the wider
    fork). Call ONCE PER STEP on the AMBIGUOUS pedestrian only, before the
    environment steps. Background pedestrians are never routed through it."""
    if waypoint_reached:
        return True
    d = float(np.hypot(pedestrian.px - JUNCTION_WAYPOINT[0], pedestrian.py - JUNCTION_WAYPOINT[1]))
    if d <= WAYPOINT_RADIUS:
        ex, ey = crowd_exit_position(true_exit, is_heldout=is_heldout)
        pedestrian.gx, pedestrian.gy = ex, ey
        return True
    return False


def build_junction_crowd_episode(env_config_path: Path, cfg: JunctionCrowdEpisodeConfig):
    """ONE 5-person junction episode: ``env.humans[0]`` is the AMBIGUOUS
    pedestrian (shared waypoint, hidden L/R exit revealed only on arrival);
    ``env.humans[1:]`` are ordinary ORCA background pedestrians crossing
    the same region with their own fixed goals.

    Returns ``(env, robot, true_exit)``. ``true_exit`` drives the simulator
    and post-hoc scoring ONLY -- never the tracker or candidate provider.
    """
    env, env_config = make_env(env_config_path, human_num=JUNCTION_CROWD_HUMAN_NUM)
    robot = Robot(env_config, "robot")
    robot_orca = ORCA(); robot_orca.configure(env_config)
    robot_orca.multiagent_training = True  # else CrowdSim.reset() silently forces human_num=1
    robot.set_policy(robot_orca)
    # Order 12: invisible robot; the config is the single definition.
    robot.visible = robot_visible_from(env_config)
    robot.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    env.case_counter["train"] = cfg.episode_seed % (2**32 - 1)
    env.reset()
    if len(env.humans) != JUNCTION_CROWD_HUMAN_NUM:
        raise JunctionScenarioError(
            f"expected {JUNCTION_CROWD_HUMAN_NUM} humans after reset, got {len(env.humans)} "
            f"(multiagent_training gate?)")

    rng = np.random.default_rng(cfg.episode_seed)
    hd = cfg.is_heldout
    speed_range = CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE if hd else PEDESTRIAN_SPEED_RANGE
    bg_x_range = CROWD_HELDOUT_BACKGROUND_X_RANGE if hd else CROWD_TRAIN_BACKGROUND_X_RANGE
    bg_y_range = CROWD_HELDOUT_BACKGROUND_Y_RANGE if hd else CROWD_TRAIN_BACKGROUND_Y_RANGE
    bg_speed_range = CROWD_HELDOUT_BACKGROUND_SPEED_RANGE if hd else CROWD_TRAIN_BACKGROUND_SPEED_RANGE

    true_exit = "left" if rng.random() < 0.5 else "right"
    ped_speed = float(rng.uniform(*speed_range))
    ped_x = float(rng.uniform(-0.3, 0.3))

    ambiguous = env.humans[AMBIGUOUS_TRACK_INDEX]
    ped_orca = ORCA(); ped_orca.configure(env_config)
    ambiguous.set_policy(ped_orca)
    # goal starts at the SHARED waypoint; the hidden exit is revealed only
    # by maybe_reveal_crowd_exit once the junction is physically reached.
    ambiguous.set(ped_x, PEDESTRIAN_START_Y, JUNCTION_WAYPOINT[0], JUNCTION_WAYPOINT[1], 0.0, 0.0, 0.0)
    ambiguous.v_pref = ped_speed

    placed = [(ped_x, PEDESTRIAN_START_Y, ambiguous.radius)]
    robot_y = float(rng.uniform(*ROBOT_START_Y_RANGE))
    robot_x = float(rng.uniform(-0.3, 0.3))
    robot.set(robot_x, robot_y, ROBOT_GOAL[0], ROBOT_GOAL[1], 0.0, 0.0, np.pi / 2)
    placed.append((robot_x, robot_y, robot.radius))

    for i in range(1, JUNCTION_CROWD_HUMAN_NUM):
        bg = env.humans[i]
        bg_orca = ORCA(); bg_orca.configure(env_config)
        bg.set_policy(bg_orca)
        # rejection-sample a non-overlapping start (C1.5: no initial overlap)
        for _attempt in range(200):
            bx = float(rng.uniform(*bg_x_range))
            by = float(rng.uniform(*bg_y_range))
            if abs(bx) <= JUNCTION_APPROACH_HALF_WIDTH and by >= JUNCTION_APPROACH_Y_MIN:
                continue  # inside the approach corridor -> would be indistinguishable from the ambiguous entry
            if all(np.hypot(bx - px, by - py) >= bg.radius + r + 0.15 for px, py, r in placed):
                break
        else:
            raise JunctionScenarioError(f"could not place background pedestrian {i} without overlap")
        # background pedestrians cross the junction region laterally, so
        # they interact with the robot without themselves being ambiguous.
        gx = -bx
        gy = float(rng.uniform(*bg_y_range))
        bg.set(bx, by, gx, gy, 0.0, 0.0, 0.0)
        bg.v_pref = float(rng.uniform(*bg_speed_range))
        placed.append((bx, by, bg.radius))

    return env, robot, true_exit
