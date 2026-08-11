"""Split from selftest.py (guide/review point 1): PublicScene / candidate-provider label-leakage tests. Shares the
FULL original test namespace via ``from ..tests._common import *`` so
every test body is copied VERBATIM (byte-identical) from the original
monolithic file -- zero risk of a name/import mismatch during the split.
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403


def test_scene_candidates_junction_and_circle() -> None:
    js = junction_scene((0.0, 4.0), [("left", (-2.5, 6.0)), ("straight", (0.0, 6.5)), ("right", (2.5, 6.0))])
    cands = js.candidates_for(np.array([0.0, 0.0]))
    assert {c.name for c in cands} == {"left", "straight", "right"}
    # routes go via the junction waypoint while the entry is still before it
    assert all(tuple(c.waypoints[0]) == (0.0, 4.0) for c in cands)
    # end-to-end with the bank: approach + left turn collapses belief to left
    bank = IntentBeliefBank(make_candidate_fn(js), dt=0.25, speed=1.0)
    for p in [(0.0, 1.0), (0.0, 2.5), (0.0, 4.0), (-0.6, 4.5), (-1.2, 5.0)]:
        bank.update({1: p})
    assert bank.belief_for(1)[0] > 0.8
    # circle negative control: multiple public boundary candidates exist
    cs = circle_scene(4.0, 8)
    assert len(cs.candidates_for(np.array([4.0, 0.0]))) >= 2

def test_scene_candidates_no_label_leakage() -> None:
    # HARD CONTRACT (review item 4): candidates derive from PUBLIC scene +
    # observable entry ONLY. Same public scene + same entry must yield the
    # identical candidate set no matter what a human's hidden goal is (the
    # provider never sees it). And the callable signature exposes no channel
    # for a Human/hidden goal.
    js = junction_scene((0.0, 4.0), [("left", (-2.5, 6.0)), ("right", (2.5, 6.0))])
    cf = make_candidate_fn(js)
    import inspect
    assert list(inspect.signature(cf).parameters) == ["track_id", "first_position"], "provider must take only (track_id, position)"
    c1 = cf(1, np.array([0.0, 0.0]))
    c2 = cf(1, np.array([0.0, 0.0]))  # a "different hidden goal" cannot change anything -- it is never an input
    assert [(c.name, c.waypoints) for c in c1] == [(c.name, c.waypoints) for c in c2]

def test_main_chain_never_reads_hidden_goal() -> None:
    # THE label-leakage hard contract. The precise rule is: no module on the
    # BELIEF/DECISION path may read a HUMAN's goal. Two things are NOT
    # violations and must not be confused with one:
    #   * reading the ROBOT's OWN goal (robot.gx/gy) -- the robot obviously
    #     knows where it is going; that is what navigation means.
    #   * WRITING a pedestrian's goal to drive the simulator
    #     (junction_scenario's maybe_reveal_exit), or reading one purely
    #     post-hoc for an initial-state identity hash that never reaches the
    #     policy (intent_evaluate.initial_state_hash).
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    # every module that participates in producing the belief or the action
    decision_path = (
        "intent_tracker.py", "scene_candidates.py", "intent_policy.py",
        "intent_crowdnav_policy.py", "intent_train.py",
    )
    human_goal = re.compile(r"\b(?:human|humans\[[^\]]*\]|h|ped|pedestrian|bg|amb|ambiguous)\s*\.\s*g[xy]\b")
    for fname in decision_path:
        src = (root / fname).read_text()
        hits = human_goal.findall(src)
        assert not hits, f"{fname} reads a hidden HUMAN goal: {hits}"
    # the two belief modules are held to the STRICTEST form: no .gx/.gy at
    # all, not even the robot's own, since they must be pure functions of
    # observable positions plus public scene geometry.
    for fname in ("intent_tracker.py", "scene_candidates.py"):
        src = (root / fname).read_text()
        assert not re.search(r"\.gx\b", src), f"{fname} must not read .gx"
        assert not re.search(r"\.gy\b", src), f"{fname} must not read .gy"
    # and the candidate provider's signature must admit no Human channel
    import inspect
    from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn
    fn = make_candidate_fn(circle_scene(radius=4.0, n_sectors=8))
    params = list(inspect.signature(fn).parameters)
    assert params == ["track_id", "first_position"], (
        f"candidate_fn must take only (track_id, first_position), got {params}")


def test_scene_candidates_fail_closed() -> None:
    for bad in [(0.0, 1.0, 2.0), (float("nan"), 0.0)]:
        try:
            PublicDestination("d", bad)
            assert False, f"expected SceneCandidatesError on {bad!r}"
        except SceneCandidatesError:
            pass
    try:
        PublicScene(destinations=())
        assert False, "expected SceneCandidatesError on empty destinations"
    except SceneCandidatesError:
        pass
    try:
        PublicScene(destinations=(PublicDestination("a", (0.0, 1.0)), PublicDestination("a", (0.0, 2.0))))
        assert False, "expected SceneCandidatesError on duplicate destination names"
    except SceneCandidatesError:
        pass
    try:
        circle_scene(-1.0)
        assert False, "expected SceneCandidatesError on non-positive radius"
    except SceneCandidatesError:
        pass


def test_c1_square_scene_matches_crowdsim_square_crossing_goal_region() -> None:
    # plan 2.2 point 12 / Order C1.1: a real bug -- the formal six-scenario
    # evaluator used circle_scene() for SQUARE scenarios too, so the public
    # candidate destinations sat on a ring unrelated to where square-
    # crossing pedestrians actually go. Verify square_scene covers
    # CrowdSim.generate_square_crossing_human's real goal region:
    #   gx in [0, +-w/2], gy in [-w/2, +w/2], opposite half-plane from entry.
    width = 10.0
    scene = square_scene(width=width, n_rows=4)
    assert len(scene.destinations) == 8, "2 half-planes x 4 bands"
    half = width * 0.5
    xs = sorted({d.position[0] for d in scene.destinations})
    ys = sorted({d.position[1] for d in scene.destinations})
    assert len(xs) == 2 and xs[0] < 0 < xs[1], f"must span BOTH half-planes, got x={xs}"
    assert all(abs(x) <= half for x in xs), "destinations must stay inside the square"
    assert len(ys) == 4
    assert all(-half <= y <= half for y in ys), f"y must stay inside [-w/2, w/2], got {ys}"
    # bands must actually cover the range, not clump in the middle
    assert ys[0] < -half * 0.4 and ys[-1] > half * 0.4, f"y bands must span the width, got {ys}"

    # a pedestrian entering on the LEFT must get forward (right-hand)
    # candidates, never be told its plausible goals are behind it
    cands = scene.candidates_for(np.array([-half, 0.0]))
    assert len(cands) >= 1
    assert all(c.waypoints[-1][0] > 0 for c in cands), (
        f"entering from the left, every plausible destination must be to the right, got {[c.waypoints[-1] for c in cands]}")
    cands_r = scene.candidates_for(np.array([half, 0.0]))
    assert all(c.waypoints[-1][0] < 0 for c in cands_r)

    # fail closed
    for bad_w, bad_rows in ((0.0, 4), (-1.0, 4), (10.0, 0)):
        try:
            square_scene(width=bad_w, n_rows=bad_rows)
            assert False, f"expected SceneCandidatesError on width={bad_w} n_rows={bad_rows}"
        except SceneCandidatesError:
            pass


def test_c1_square_scene_no_label_leakage() -> None:
    # same hard contract as every other provider: public geometry only.
    import inspect as _inspect
    src = _inspect.getsource(square_scene)
    assert ".gx" not in src and ".gy" not in src, "square_scene must never read a hidden human goal"
    # it must be a pure function of the configured width -- same width in,
    # bit-identical destinations out, regardless of anything else.
    a = square_scene(width=14.0, n_rows=4)
    b = square_scene(width=14.0, n_rows=4)
    assert [d.position for d in a.destinations] == [d.position for d in b.destinations]
