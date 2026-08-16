"""Regression tests for the junction-crowd V2 candidate model.

The paper-test post-mortem found that four of five pedestrians in the core
ablation scenario were handed the wrong candidate goals. The suite was green
throughout, because it checked that posteriors normalise -- not that the
candidates described the humans they were attached to. These tests pin the
properties that were actually violated, so the same class of defect cannot
pass again.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from crowd_nav.bayesian_dvl.intent_policy import (
    HUMAN_FEATURE_DIM_V5, HumanObservation, RobotObservation, build_intent_human_feature_batch,
)
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    CANDIDATE_FEATURE_DIM, FROZEN_VALUES, HUMAN_SCALAR_DIM_V6, MAX_CANDIDATE_GOALS, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_tracker import CandidateGoal, GoalIntentTracker, IntentBeliefBank
from crowd_nav.bayesian_dvl.junction_scenario import (
    junction_crowd_role_of_seed,
    AMBIGUOUS_TRACK_INDEX, CROSSING_BAND_N, JUNCTION_APPROACH_HALF_WIDTH, JUNCTION_APPROACH_Y_MIN,
    JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_HUMAN_NUM, JUNCTION_CROWD_TRAIN_SEEDS,
    JunctionCrowdEpisodeConfig, SCENARIO_REGISTRY_ID, build_junction_crowd_episode,
    public_junction_crowd_scene,
)
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn

ENV = Path(__file__).resolve().parents[2] / "configs" / "env_bayesian_dvl.config"

# Measured budgets. Before the fix the background humans' nearest candidate
# endpoint sat 1.90 m from where they actually went (max 4.01 m); after it,
# 0.26 m mean / 0.52 m max over 60 held-out episodes. These bounds sit
# between the two, so a regression toward the old behaviour fails here.
MAX_MEAN_CANDIDATE_ERROR_M = 0.60
MAX_WORST_CANDIDATE_ERROR_M = 1.20


def _episodes(seeds, is_heldout):
    for seed in seeds:
        env, robot, true_exit = build_junction_crowd_episode(
            ENV, JunctionCrowdEpisodeConfig(episode_seed=seed, role=junction_crowd_role_of_seed(seed)))
        yield env, robot, true_exit


@pytest.mark.parametrize("is_heldout", [False, True])
def test_corridor_entry_gets_the_two_junction_exits(is_heldout):
    """A pedestrian entering the approach corridor is on the junction
    approach, so its candidates are the two exits and nothing else."""
    scene = public_junction_crowd_scene(is_heldout=is_heldout)
    fn = make_candidate_fn(scene)
    for x in (0.0, 0.3, -0.3, JUNCTION_APPROACH_HALF_WIDTH):
        cands = fn(0, np.array([x, 4.0]))
        assert len(cands) == 2, f"entry ({x}, 4.0) is in the corridor, expected 2 exits, got {len(cands)}"
        assert {c.name for c in cands} == {"left", "right"}


@pytest.mark.parametrize("is_heldout", [False, True])
def test_outside_the_corridor_gets_crossing_candidates_not_junction_exits(is_heldout):
    scene = public_junction_crowd_scene(is_heldout=is_heldout)
    fn = make_candidate_fn(scene)
    for entry in ((2.0, 4.0), (-2.0, 3.5), (1.0, 4.9), (-1.5, 1.0), (0.0, 1.0)):
        cands = fn(0, np.array(entry))
        assert len(cands) == CROSSING_BAND_N, (
            f"entry {entry} is outside the corridor, expected {CROSSING_BAND_N} crossing candidates")
        assert not ({c.name for c in cands} & {"left", "right"}), (
            f"entry {entry} must not be given junction exit candidates: {[c.name for c in cands]}")
        # a crosser traverses to the mirrored-x side
        for c in cands:
            assert c.waypoints[-1][0] == pytest.approx(-entry[0]), (
                f"crossing candidate should end on the mirrored side of {entry}, got {c.waypoints[-1]}")


def test_ambiguous_pedestrian_always_starts_inside_the_corridor():
    scene = public_junction_crowd_scene(is_heldout=True)
    for env, _robot, _exit in _episodes(list(JUNCTION_CROWD_HELDOUT_SEEDS)[:40], True):
        h = env.humans[AMBIGUOUS_TRACK_INDEX]
        assert scene.in_approach_corridor((h.px, h.py)), (
            f"the ambiguous pedestrian started at ({h.px:.3f}, {h.py:.3f}), outside the corridor")


@pytest.mark.parametrize("is_heldout,seeds", [
    (False, list(JUNCTION_CROWD_TRAIN_SEEDS)[:60]),
    (True, list(JUNCTION_CROWD_HELDOUT_SEEDS)[:60]),
])
def test_no_background_pedestrian_is_ever_born_in_the_corridor(is_heldout, seeds):
    """This is what makes corridor membership a sound public discriminator
    rather than an assertion. The old sampler could place a crosser at
    (0.1, 4.0), indistinguishable from the ambiguous start."""
    scene = public_junction_crowd_scene(is_heldout=is_heldout)
    offenders = []
    for env, _robot, _exit in _episodes(seeds, is_heldout):
        for i, h in enumerate(env.humans):
            if i != AMBIGUOUS_TRACK_INDEX and scene.in_approach_corridor((h.px, h.py)):
                offenders.append((i, round(float(h.px), 3), round(float(h.py), 3)))
    assert not offenders, f"{len(offenders)} background pedestrians born inside the corridor: {offenders[:5]}"


@pytest.mark.parametrize("is_heldout", [False, True])
def test_candidate_endpoints_actually_cover_where_pedestrians_go(is_heldout):
    """Coverage, measured against the hidden goals. The hidden goals are read
    ONLY here, in a test, to score the public model -- never by the tracker."""
    seeds = list(JUNCTION_CROWD_HELDOUT_SEEDS if is_heldout else JUNCTION_CROWD_TRAIN_SEEDS)[:40]
    fn = make_candidate_fn(public_junction_crowd_scene(is_heldout=is_heldout))
    errors = []
    for env, _robot, _exit in _episodes(seeds, is_heldout):
        for i, h in enumerate(env.humans):
            if i == AMBIGUOUS_TRACK_INDEX:
                continue    # its goal is the shared waypoint until the fork is reached
            cands = fn(i, np.array([h.px, h.py]))
            errors.append(min(float(np.hypot(c.waypoints[-1][0] - h.gx, c.waypoints[-1][1] - h.gy))
                              for c in cands))
    assert np.mean(errors) <= MAX_MEAN_CANDIDATE_ERROR_M, (
        f"mean candidate-coverage error {np.mean(errors):.2f} m exceeds {MAX_MEAN_CANDIDATE_ERROR_M} m "
        "-- the candidate model no longer describes where these pedestrians go")
    assert np.max(errors) <= MAX_WORST_CANDIDATE_ERROR_M, (
        f"worst candidate-coverage error {np.max(errors):.2f} m exceeds {MAX_WORST_CANDIDATE_ERROR_M} m")


def test_every_junction_crowd_human_gets_at_least_two_candidates():
    """Not a global ban on single candidates -- a genuinely unique public
    destination may legitimately give [1.0]. But in THIS scene every
    pedestrian's destination is unresolved at entry, and a single candidate
    would report certainty the scene does not have."""
    fn = make_candidate_fn(public_junction_crowd_scene(is_heldout=True))
    for env, _robot, _exit in _episodes(list(JUNCTION_CROWD_HELDOUT_SEEDS)[:40], True):
        assert len(env.humans) == JUNCTION_CROWD_HUMAN_NUM
        for i, h in enumerate(env.humans):
            n = len(fn(i, np.array([h.px, h.py])))
            assert n >= 2, f"human {i} at ({h.px:.2f}, {h.py:.2f}) collapsed to {n} candidate(s)"


def test_scenario_registry_id_is_v2():
    """v1 and v2 differ in birth distribution AND candidate semantics. If the
    id did not move, two different environments would be recorded as one."""
    assert SCENARIO_REGISTRY_ID == "bdvl_junction_scenario_v2_corridor_split"


# ---------------------------------------------------------------- speed EMA

def _straight_line_tracker(speed, dt=0.25, **kw):
    cands = [CandidateGoal("a", ((0.0, 100.0),)), CandidateGoal("b", ((100.0, 0.0),))]
    t = GoalIntentTracker(cands, dt=dt, speed=1.0, **kw)
    for k in range(12):
        t.update((0.0, speed * dt * k))
    return t


def test_speed_estimate_tracks_observed_motion():
    for true_speed in (0.5, 1.3, 2.0):
        t = _straight_line_tracker(true_speed)
        assert abs(t._speed_est - true_speed) < 0.1, (
            f"observed {true_speed} m/s, tracker estimated {t._speed_est:.3f}")


def test_speed_estimate_is_clipped_to_the_frozen_range():
    assert _straight_line_tracker(20.0)._speed_est == pytest.approx(TRACKER_DEFAULTS["speed_max"])
    assert _straight_line_tracker(0.001)._speed_est == pytest.approx(TRACKER_DEFAULTS["speed_min"])


def test_speed_estimation_can_be_switched_off_and_then_never_moves():
    t = _straight_line_tracker(2.0, estimate_speed=False)
    assert t._speed_est == pytest.approx(1.0), "with estimation off the prior must be used verbatim"


def test_speed_ema_updates_after_the_likelihood_not_before():
    """Order matters: if the estimate absorbed this frame's displacement
    BEFORE the likelihood, every candidate would be scored against a model
    already fitted to that frame and the posterior would stop discriminating.
    One update at a speed far from the prior must leave the estimate at the
    PRIOR while that frame is being scored."""
    cands = [CandidateGoal("a", ((0.0, 100.0),)), CandidateGoal("b", ((100.0, 0.0),))]
    t = GoalIntentTracker(cands, dt=0.25, speed=1.0)
    t.update((0.0, 0.0))
    assert t._speed_est == pytest.approx(1.0), "no displacement seen yet -> still the prior"
    t.update((0.0, 0.5))                                    # 2.0 m/s
    alpha = TRACKER_DEFAULTS["speed_ema_alpha"]
    assert t._speed_est == pytest.approx((1 - alpha) * 1.0 + alpha * 2.0), (
        "the first real displacement must move the estimate by exactly one EMA step, no more")


def test_speed_model_is_global_not_junction_only():
    """A deliberate, declared decision: a tracker that assumes a fixed speed
    is wrong everywhere, and scoping the fix to one scenario would put two
    tracker behaviours inside one experiment."""
    assert TRACKER_DEFAULTS["estimate_speed"] is True
    bank = IntentBeliefBank(lambda tid, pos: [CandidateGoal("a", ((0.0, 9.0),)),
                                              CandidateGoal("b", ((9.0, 0.0),))],
                            dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
    for k in range(10):
        bank.update({0: (0.0, 0.4 * k)})
    assert bank.speed_estimate_for(0) != pytest.approx(TRACKER_DEFAULTS["speed_prior"]), (
        "the estimator must be active for any scene, not just junction_crowd")


# ------------------------------------------------- V6 candidate-set encoding

def _packed(row):
    S, G, F = HUMAN_SCALAR_DIM_V6, MAX_CANDIDATE_GOALS, CANDIDATE_FEATURE_DIM
    return row[S:S + G * F].reshape(G, F), row[S + G * F:]


def test_candidate_encoding_is_permutation_invariant():
    """The defect V6 exists to remove: p0 meant 'left exit' for 412 humans
    and 'right exit' for 68 others, and slot index was the only thing
    identifying a candidate. Shuffling the candidate order must now be
    invisible to the network."""
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    model.eval()
    n_c = 5
    row = np.zeros(HUMAN_FEATURE_DIM_V5, dtype=np.float32)
    row[:HUMAN_SCALAR_DIM_V6] = np.linspace(-0.5, 0.5, HUMAN_SCALAR_DIM_V6)
    cands, mask = _packed(row)
    rng = np.random.default_rng(7)
    cands[:n_c] = rng.normal(size=(n_c, CANDIDATE_FEATURE_DIM))
    mask[:n_c] = 1.0

    feats = torch.as_tensor(row[None, None, :], dtype=torch.float32)
    human_mask = torch.ones(1, 1, dtype=torch.bool)
    robot = torch.zeros(1, 7)
    action = torch.zeros(1, 5)
    tau = torch.linspace(0.05, 0.95, 8)[None]
    with torch.no_grad():
        ref = model(robot, feats, human_mask, action, tau)

    for perm_seed in range(5):
        order = np.random.default_rng(perm_seed).permutation(n_c)
        shuffled = row.copy()
        s_cands, s_mask = _packed(shuffled)
        s_cands[:n_c] = cands[:n_c][order]
        s_mask[:n_c] = 1.0
        with torch.no_grad():
            got = model(robot, torch.as_tensor(shuffled[None, None, :], dtype=torch.float32),
                        human_mask, action, tau)
        assert torch.allclose(ref, got, atol=1e-6), (
            f"permutation {order.tolist()} changed the output by "
            f"{(ref - got).abs().max().item():.2e}")


def test_candidate_geometry_actually_reaches_the_network():
    """Permutation invariance would also be satisfied by ignoring the
    candidates entirely, so check the opposite: moving a candidate's endpoint
    must change the output."""
    torch.manual_seed(1)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    model.eval()
    row = np.zeros(HUMAN_FEATURE_DIM_V5, dtype=np.float32)
    cands, mask = _packed(row)
    cands[:2] = [(0.5, 0.1, 0.2, 0.1, 0.2), (0.5, -0.1, 0.3, -0.1, 0.3)]
    mask[:2] = 1.0
    moved = row.copy()
    m_cands, _ = _packed(moved)
    m_cands[1] = (0.5, 0.9, -0.4, 0.9, -0.4)

    args = (torch.zeros(1, 7), None, torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 5), torch.linspace(0.05, 0.95, 8)[None])
    with torch.no_grad():
        a = model(args[0], torch.as_tensor(row[None, None, :]), args[2], args[3], args[4])
        b = model(args[0], torch.as_tensor(moved[None, None, :]), args[2], args[3], args[4])
    assert not torch.allclose(a, b, atol=1e-6), "candidate endpoints must influence the value"


def test_mean_and_cv_get_a_valid_mask_but_an_empty_candidate_block():
    """The frozen arm definitions: candidate CARDINALITY is public geometry
    every arm may know; which candidate is which, and how likely, is not."""
    scene = public_junction_crowd_scene(is_heldout=True)
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"],
                            speed=TRACKER_DEFAULTS["speed_prior"])
    for k in range(6):
        bank.update({0: (0.0, 4.0 - 0.2 * k)})
    robot = RobotObservation(px=0.0, py=0.5, gx=0.0, gy=5.0, vx=0.0, vy=0.5,
                             radius=0.3, v_pref=1.0, theta=np.pi / 2)
    humans = [HumanObservation(0, 0.0, 2.9, 0.0, -0.8, 0.3)]
    out = {}
    for mode in ("full", "mean", "cv"):
        f, _m = build_intent_human_feature_batch(
            bank, robot, humans, mode=mode, rng=np.random.default_rng(0), n_samples=64)
        out[mode] = _packed(f[0])
    assert np.any(out["full"][0] != 0.0), "full must carry real candidate features"
    for mode in ("mean", "cv"):
        assert np.all(out[mode][0] == 0.0), f"{mode} must not see any candidate probability or geometry"
        assert np.array_equal(out[mode][1], out["full"][1]), (
            f"{mode}'s validity mask must match full's -- cardinality is public")


def test_candidate_count_reaches_the_network():
    """The gap this test exists for: an earlier version claimed the validity
    mask carried candidate cardinality to the mean/cv arms. It does not.
    Masked mean/max pooling over IDENTICAL all-zero candidate rows returns
    the same vector whether two or four are valid, so without an explicit
    count scalar those arms could not see how many public destinations
    existed -- a public fact they had in v5."""
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    model.eval()
    robot, action = torch.zeros(1, 7), torch.zeros(1, 5)
    tau = torch.linspace(0.05, 0.95, 8)[None]
    hmask = torch.ones(1, 1, dtype=torch.bool)

    def row_for(n_cand):
        # exactly the mean/cv arm's shape: zeroed candidate block, valid mask
        row = np.zeros(HUMAN_FEATURE_DIM_V5, dtype=np.float32)
        row[HUMAN_SCALAR_DIM_V6 - 1] = n_cand / MAX_CANDIDATE_GOALS
        _c, mask = _packed(row)
        mask[:n_cand] = 1.0
        return torch.as_tensor(row[None, None, :])

    with torch.no_grad():
        two, four = model(robot, row_for(2), hmask, action, tau), model(robot, row_for(4), hmask, action, tau)
    assert not torch.allclose(two, four, atol=1e-6), (
        "2 and 4 public destinations must be distinguishable even when the candidate block is zeroed")

    # and the count must be the ONLY thing carrying it: blank the scalar and
    # the two become identical again, which is the bug this guards.
    def row_without_count(n_cand):
        row = np.zeros(HUMAN_FEATURE_DIM_V5, dtype=np.float32)
        _c, mask = _packed(row)
        mask[:n_cand] = 1.0
        return torch.as_tensor(row[None, None, :])

    with torch.no_grad():
        a = model(robot, row_without_count(2), hmask, action, tau)
        b = model(robot, row_without_count(4), hmask, action, tau)
    assert torch.allclose(a, b, atol=1e-7), (
        "precondition: with the count scalar blanked, the mask alone conveys nothing about cardinality")


def test_all_arms_report_the_same_candidate_count():
    """Cardinality is public geometry, so it must be identical across arms --
    it is not part of what the ablation withholds."""
    scene = public_junction_crowd_scene(is_heldout=True)
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"],
                            speed=TRACKER_DEFAULTS["speed_prior"])
    for k in range(6):
        bank.update({0: (2.0, 4.0 - 0.2 * k)})       # outside the corridor -> 4 crossing candidates
    robot = RobotObservation(px=0.0, py=0.5, gx=0.0, gy=5.0, vx=0.0, vy=0.5,
                             radius=0.3, v_pref=1.0, theta=np.pi / 2)
    humans = [HumanObservation(0, 2.0, 2.9, 0.0, -0.8, 0.3)]
    counts = {}
    for mode in ("full", "mean", "cv", "uniform"):
        f, _m = build_intent_human_feature_batch(
            bank, robot, humans, mode=mode, rng=np.random.default_rng(0), n_samples=32)
        counts[mode] = float(f[0][HUMAN_SCALAR_DIM_V6 - 1])
    assert len(set(counts.values())) == 1, f"arms disagree on public candidate count: {counts}"
    assert counts["full"] == pytest.approx(CROSSING_BAND_N / MAX_CANDIDATE_GOALS)
