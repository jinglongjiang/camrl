"""Split from selftest.py (guide/review point 1): GoalIntentTracker + IntentBeliefBank tests. Shares the
FULL original test namespace via ``from ..tests._common import *`` so
every test body is copied VERBATIM (byte-identical) from the original
monolithic file -- zero risk of a name/import mismatch during the split.
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403


def test_intent_tracker_prefork_multimodal_then_collapses() -> None:
    # guide/consolidation plan: the predictive intent belief must stay
    # genuinely multimodal BEFORE the fork (both left and right alive) and
    # collapse to the realized exit after -- the exact property the reactive
    # kinematic SBK-HMM tracker could not produce (its pre-fork L/R mass was
    # ~0.001; this must be substantial).
    beliefs = _run_intent_track([(0.0, 4.0), (-2.5, 6.0)])  # true goal = left
    pre = beliefs[[3, 5, 7]].mean(axis=0)
    assert min(pre[0], pre[2]) > 0.25, f"pre-fork L/R mass too small: {pre}"
    assert beliefs[-1][0] > 0.8, f"did not collapse to the true (left) exit: {beliefs[-1]}"

def test_intent_tracker_no_label_leakage() -> None:
    # HARD CONTRACT (consolidation plan Decision 2): the tracker consumes
    # ONLY observable motion + public candidates. Two pedestrians with
    # IDENTICAL observed motion but DIFFERENT hidden goals must produce the
    # IDENTICAL belief -- proving the hidden goal never enters. (Here the
    # "hidden goal" differs only as a label; the fed positions are the same.)
    b_left_label = _run_intent_track([(0.0, 4.0), (-2.5, 6.0)])
    b_same_motion = _run_intent_track([(0.0, 4.0), (-2.5, 6.0)])
    assert np.allclose(b_left_label, b_same_motion)
    # and the tracker's only mutating entry point takes a position, nothing else
    import inspect
    params = list(inspect.signature(GoalIntentTracker.update).parameters)
    assert params == ["self", "observed_position"], f"update must take only an observable position, got {params}"

def test_intent_tracker_sample_futures_modes() -> None:
    tracker = GoalIntentTracker(_INTENT_CANDS, dt=0.25, speed=1.0)
    for p in [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]:  # pre-fork approach -> belief stays ~uniform
        tracker.update(p)
    rng = np.random.default_rng(0)
    full = tracker.sample_futures((0.0, 3.0), (0.0, 1.0), horizon=8, mode="full", rng=rng, n_samples=60)
    mean = tracker.sample_futures((0.0, 3.0), (0.0, 1.0), horizon=8, mode="mean", rng=rng, n_samples=60)
    cv = tracker.sample_futures((0.0, 3.0), (0.0, 1.0), horizon=8, mode="cv", rng=rng, n_samples=60)
    assert len(full) == 60 and len(mean) == 1 and len(cv) == 1
    # full is genuinely multimodal (samples end at different exits); mean is one averaged path
    endpoints = np.array([tr[-1] for tr in full])
    assert endpoints[:, 0].std() > 0.3, "full futures should spread across exits (multimodal)"
    assert all(t.shape == (8, 2) for t in full)

def test_intent_tracker_full_vs_uniform_sampling_distribution() -> None:
    # guide/review: when the posterior is NON-uniform, `full` must sample by
    # the posterior while `uniform` keeps sampling uniformly -- and the two
    # resulting future distributions must differ. (The earlier test only
    # exercised full/mean/cv; uniform was never really checked.)
    tracker = GoalIntentTracker(_INTENT_CANDS, dt=0.25, speed=1.0)
    # drive the pedestrian PAST the junction and toward the LEFT exit so the
    # posterior becomes strongly non-uniform (mass on 'left').
    for p in [(0.0, 2.0), (0.0, 3.5), (0.0, 4.0), (-0.6, 4.5), (-1.2, 5.0)]:
        tracker.update(p)
    belief = tracker.belief()
    assert belief[0] > 0.6, f"posterior should be non-uniform (left-dominant), got {belief}"

    rng = np.random.default_rng(1)
    full = tracker.sample_futures((-1.2, 5.0), (-1.0, 1.0), horizon=8, mode="full", rng=rng, n_samples=300)
    unif = tracker.sample_futures((-1.2, 5.0), (-1.0, 1.0), horizon=8, mode="uniform", rng=rng, n_samples=300)
    full_end_x = np.array([tr[-1][0] for tr in full])
    unif_end_x = np.array([tr[-1][0] for tr in unif])
    # full follows the posterior -> endpoints concentrate at the left exit;
    # uniform spreads across all exits -> a distinctly different (less left,
    # higher-variance) endpoint distribution.
    assert full_end_x.mean() < unif_end_x.mean() - 0.3, (
        f"full should be more left-concentrated than uniform: full_mean={full_end_x.mean():.3f} unif_mean={unif_end_x.mean():.3f}")
    assert unif_end_x.std() > full_end_x.std(), "uniform endpoints should be more spread out than full's"

def test_intent_tracker_fail_closed() -> None:
    # empty candidate set
    try:
        GoalIntentTracker([], dt=0.25, speed=1.0)
        assert False, "expected IntentTrackerError on empty candidates"
    except IntentTrackerError:
        pass
    # duplicate candidate names
    try:
        GoalIntentTracker([CandidateGoal("a", ((0.0, 1.0),)), CandidateGoal("a", ((0.0, 2.0),))], dt=0.25, speed=1.0)
        assert False, "expected IntentTrackerError on duplicate names"
    except IntentTrackerError:
        pass
    # CandidateGoal: empty route / non-2D waypoint / non-finite waypoint
    for bad in [(), ((0.0, 1.0, 2.0),), ((float("nan"), 1.0),)]:
        try:
            CandidateGoal("bad", bad)
            assert False, f"expected IntentTrackerError on bad waypoints {bad!r}"
        except IntentTrackerError:
            pass
    # constructor: non-finite / non-positive dt/speed/sigma/wp_radius
    for bad_kwargs in [dict(dt=float("inf")), dict(speed=float("nan")), dict(sigma=0.0), dict(wp_radius=0.0), dict(wp_radius=-0.1)]:
        kw = dict(dt=0.25, speed=1.0)
        kw.update(bad_kwargs)
        try:
            GoalIntentTracker(_INTENT_CANDS, **kw)
            assert False, f"expected IntentTrackerError on {bad_kwargs!r}"
        except IntentTrackerError:
            pass
    tracker = GoalIntentTracker(_INTENT_CANDS, dt=0.25, speed=1.0)
    # update: bad position shape
    try:
        tracker.update((0.0, 0.0, 0.0))
        assert False, "expected IntentTrackerError on bad position shape"
    except IntentTrackerError:
        pass
    # update: non-finite position
    try:
        tracker.update((float("nan"), 0.0))
        assert False, "expected IntentTrackerError on non-finite position"
    except IntentTrackerError:
        pass
    # sample_futures: invalid mode / n_samples / shape / non-finite
    rng = np.random.default_rng(0)
    for kwargs in [
        dict(mode="bogus"),
        dict(mode="full", n_samples=0),
        dict(position=(0.0, 0.0, 0.0)),
        dict(velocity=(float("inf"), 0.0)),
        dict(horizon=0),
    ]:
        call = dict(position=(0.0, 3.0), velocity=(0.0, 1.0), horizon=8, mode="full", rng=rng, n_samples=10)
        call.update(kwargs)
        try:
            tracker.sample_futures(**call)
            assert False, f"expected IntentTrackerError on {kwargs!r}"
        except IntentTrackerError:
            pass

def test_intent_belief_bank_identity_by_track_id_not_list_position() -> None:
    # guide/review Order-4 item 5: identity is bound to STABLE track_id, never
    # to dict/list ordering. Feeding the SAME per-track observations in a
    # different insertion order must yield the SAME per-track belief.
    left = _feed_bank_track()
    right = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (0.6, 4.5), (1.2, 5.0), (1.8, 5.5)]

    bank_a = IntentBeliefBank(_bank_candidate_fn, dt=0.25, speed=1.0)
    bank_b = IntentBeliefBank(_bank_candidate_fn, dt=0.25, speed=1.0)
    for k in range(len(left)):
        bank_a.update({7: left[k], 3: right[k]})          # order {7,3}
        bank_b.update({3: right[k], 7: left[k]})          # order {3,7}
    assert np.allclose(bank_a.belief_for(7), bank_b.belief_for(7))
    assert np.allclose(bank_a.belief_for(3), bank_b.belief_for(3))
    # and identity really tracks the id: track 7 (left) leans left, track 3 (right) leans right
    assert bank_a.belief_for(7)[0] > 0.6 and bank_a.belief_for(3)[2] > 0.6

def test_intent_belief_bank_reset_and_reappearance_timeout() -> None:
    bank = IntentBeliefBank(_bank_candidate_fn, dt=0.25, speed=1.0, missing_timeout_steps=3)
    # drive belief NON-uniform (past junction toward left) so "preserved" is
    # distinguishable from "reset to fresh uniform".
    for p in [(0.0, 3.0), (0.0, 4.0), (-0.6, 4.5), (-1.2, 5.0)]:
        bank.update({1: p})
    b_before = bank.belief_for(1).copy()
    assert b_before[0] > 0.6, f"precondition: non-uniform (left) belief, got {b_before}"

    # short gap (<= timeout): track survives, non-uniform belief PRESERVED
    # (re-baselined, not reset to uniform, not blown up by a fake velocity)
    for _ in range(3):
        bank.update({})
    assert 1 in bank.active_tracks(), "track should survive a gap within the timeout"
    bank.update({1: (-1.8, 5.5)})
    assert np.allclose(bank.belief_for(1), b_before), (
        f"short-gap reappearance must preserve the pre-gap belief, got {bank.belief_for(1)} vs {b_before}")

    # long gap (> timeout): track expires, reappearance is a FRESH uniform tracker
    for _ in range(4):
        bank.update({})
    assert 1 not in bank.active_tracks(), "track should expire after the timeout"
    bank.update({1: (0.0, 0.0)})
    assert np.allclose(bank.belief_for(1), np.ones(3) / 3), "reappearance after expiry must start from a fresh uniform prior"

    bank.reset()
    assert bank.active_tracks() == set()

def test_intent_belief_bank_short_gap_rebaselines_no_fake_velocity() -> None:
    # guide/review Order-4 item 5 stale-gap fix: when a track is missing for a
    # few frames (within the timeout) and reappears at a jumped position,
    # update() must NOT diff the jumped position against the pre-gap position
    # over a single dt (that fabricates a huge velocity and wrongly compresses
    # the posterior). Required contract:
    #   - reappearance FIRST frame belief == pre-gap belief (re-baseline);
    #   - the NEXT frame updates on the real single-step velocity;
    #   - a long gap still expires to a fresh uniform prior.
    bank = IntentBeliefBank(_bank_candidate_fn, dt=0.25, speed=1.0, missing_timeout_steps=3)
    # drive PAST the junction toward left so the posterior is strongly
    # non-uniform (this is where the fake velocity actually distorts).
    for p in [(0.0, 3.0), (0.0, 4.0), (-0.6, 4.5), (-1.2, 5.0)]:
        bank.update({1: p})
    b_before = bank.belief_for(1).copy()
    assert b_before[0] > 0.6, f"precondition: non-uniform (left) belief, got {b_before}"

    bank.update({}); bank.update({})               # miss 2 frames (within timeout)
    # reappear jumped far: a single-dt diff vs pre-gap (-1.2,5.0) would be
    # speed ~8 (real single-step is ~1). The fix must re-baseline instead.
    bank.update({1: (-2.7, 6.4)})
    assert np.allclose(bank.belief_for(1), b_before), (
        f"reappearance frame must re-baseline (belief unchanged), got {bank.belief_for(1)} vs {b_before}")
    # next real single-step keeps evolving sanely (stays left-dominant, not blown up)
    bank.update({1: (-3.3, 6.9)})
    nb = bank.belief_for(1)
    assert nb[0] > 0.6 and np.all(np.isfinite(nb)), f"post-rebaseline update should be sane, got {nb}"

def test_intent_belief_bank_fail_closed() -> None:
    try:
        IntentBeliefBank(candidate_fn=None, dt=0.25, speed=1.0)  # type: ignore[arg-type]
        assert False, "expected IntentTrackerError on non-callable candidate_fn"
    except IntentTrackerError:
        pass
    bank = IntentBeliefBank(_bank_candidate_fn, dt=0.25, speed=1.0)
    try:
        bank.belief_for(999)
        assert False, "expected IntentTrackerError for an inactive track"
    except IntentTrackerError:
        pass
