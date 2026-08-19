"""Split from selftest.py (guide/review point 1): FEATURE_SCHEMA_V6 feature-batch, checkpoint, and score_candidates_v5 tests. Shares the
FULL original test namespace via ``from ..tests._common import *`` so
every test body is copied VERBATIM (byte-identical) from the original
monolithic file -- zero risk of a name/import mismatch during the split.
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403


def test_intent_policy_no_sbk_hmm_import() -> None:
    # HARD CONTRACT (consolidation plan item 3): the new main chain must not
    # transitively import the old SBK-HMM/R4 branches. Must run in a FRESH
    # subprocess (same pattern as test_bdvl_package_does_not_import_mamba_or_
    # legacy_bayesian_packages) -- selftest.py's OWN top-level imports (of the
    # old chain's own tests) already pollute sys.modules in-process, which
    # would make an in-process check pass/fail for the wrong reason.
    import subprocess
    import sys as _sys
    code = (
        "import sys\n"
        "import crowd_nav.bayesian_dvl.intent_policy\n"
        "forbidden = ('bayesian_dvl.belief', 'bayesian_dvl.rollout', 'bayesian_dvl.world_model', "
        "'bayesian_dvl.counterfactual', 'bayesian_dvl.data_coverage', 'bayesian_dvl.oracle_regret', "
        "'bayesian_dvl.replay', 'bayesian_dvl.trainer')\n"
        "bad = [m for m in sys.modules if any(f in m for f in forbidden)]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout

def test_intent_policy_feature_batch_shape_and_finite() -> None:
    bank = _junction_bank_2exit()
    for p in [(0.0, 1.0), (0.0, 2.5), (0.0, 4.0), (-0.6, 4.5)]:
        bank.update({0: p})
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=1.0, radius=0.3, gx=0.0, gy=8.0, v_pref=1.0, theta=1.5708)
    humans = [HumanObservation(0, -0.6, 4.5, -1.0, 1.0, 0.3)]
    rng = np.random.default_rng(0)
    feat, mask = build_intent_human_feature_batch(bank, robot, humans, mode="full", rng=rng)
    assert feat.shape == (20, HUMAN_FEATURE_DIM_V6)
    assert mask.shape == (20,) and mask.dtype == bool
    assert np.all(np.isfinite(feat))
    assert mask[0] and not mask[1:].any()

def test_intent_policy_full_vs_mean_differ_in_multimodal_state() -> None:
    # guide/review's local-e2e hard requirement 5 item 3: full and mean must
    # produce DIFFERENT network inputs AND different action scores in a
    # genuinely multimodal (pre-fork) risk state.
    bank = _junction_bank_2exit()
    for p in [(0.0, 1.0), (0.0, 2.5), (0.0, 3.9)]:
        bank.update({0: p})
    belief = bank.belief_for(0)
    assert abs(belief[0] - belief[1]) < 0.05, f"precondition: near-uniform bimodal belief, got {belief}"

    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=1.0, radius=0.3, gx=0.0, gy=8.0, v_pref=1.0, theta=1.5708)
    humans = [HumanObservation(0, 0.0, 3.9, 0.0, 1.0, 0.3)]
    action_table = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH)).build_action_table()
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    rf = intent_remaining_time_fraction(3.0, 35.0)

    feat_full, mask_full = build_intent_human_feature_batch(bank, robot, humans, mode="full", rng=np.random.default_rng(42), n_samples=200)
    feat_mean, mask_mean = build_intent_human_feature_batch(bank, robot, humans, mode="mean", rng=np.random.default_rng(42), n_samples=200)
    assert not np.allclose(feat_full, feat_mean), "full and mean must produce different network inputs in a multimodal state"
    # the spread feature is the clearest signature: full (multiple sampled
    # futures) has nonzero endpoint spread, mean (one averaged path) has zero.
    # Indexed by NAME, not as [-1]: in V5 the spread happened to be the last
    # dim, in V6 the last dim is the candidate validity mask, and a negative
    # index silently followed the layout instead of failing.
    SPREAD = 12
    assert feat_full[0, SPREAD] > 0.05 and feat_mean[0, SPREAD] == 0.0

    res_full = score_candidates_v5(model, robot, feat_full, mask_full, action_table, rf)
    res_mean = score_candidates_v5(model, robot, feat_mean, mask_mean, action_table, rf)
    q_full = np.array([r.q_mean for r in res_full])
    q_mean_scores = np.array([r.q_mean for r in res_mean])
    assert not np.allclose(q_full, q_mean_scores), "full and mean must produce different action scores in a multimodal state"

def test_intent_policy_checkpoint_roundtrip_and_fail_closed() -> None:
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ckpt.pth")
        save_intent_checkpoint(model, path, action_grid_hash="hash_a", scene_registry_sha256="scene_a")
        model2 = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
        manifest = load_intent_checkpoint(path, model2, expected_action_grid_hash="hash_a", expected_scene_registry_sha256="scene_a")
        assert manifest["action_grid_hash"] == "hash_a"
        for p1, p2 in zip(model.state_dict().values(), model2.state_dict().values()):
            assert torch.equal(p1, p2)
        # fail-closed: wrong action grid hash / wrong scene registry hash / tampered schema
        try:
            load_intent_checkpoint(path, model2, expected_action_grid_hash="wrong_hash")
            assert False, "expected IntentPolicyError on action_grid_hash mismatch"
        except IntentPolicyError:
            pass
        try:
            load_intent_checkpoint(path, model2, expected_scene_registry_sha256="wrong_scene")
            assert False, "expected IntentPolicyError on scene_registry_sha256 mismatch"
        except IntentPolicyError:
            pass
        tampered = torch.load(path, weights_only=False)
        tampered["checkpoint_schema"] = "bogus_schema"
        torch.save(tampered, path + ".tampered")
        try:
            load_intent_checkpoint(path + ".tampered", model2)
            assert False, "expected IntentPolicyError on tampered checkpoint_schema"
        except IntentPolicyError:
            pass

def test_intent_policy_remaining_time_fraction_and_bad_mode() -> None:
    assert intent_remaining_time_fraction(0.0, 35.0) == 1.0
    assert intent_remaining_time_fraction(35.0, 35.0) == 0.0
    try:
        intent_remaining_time_fraction(1.0, 0.0)
        assert False, "expected IntentPolicyError on non-positive time_limit"
    except IntentPolicyError:
        pass
    bank = _junction_bank_2exit()
    bank.update({0: (0.0, 1.0)})
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=1.0, radius=0.3, gx=0.0, gy=8.0, v_pref=1.0, theta=1.5708)
    humans = [HumanObservation(0, 0.0, 1.0, 0.0, 1.0, 0.3)]
    try:
        build_intent_human_feature_batch(bank, robot, humans, mode="bogus", rng=np.random.default_rng(0))
        assert False, "expected IntentPolicyError on unknown mode"
    except IntentPolicyError:
        pass

def test_score_candidates_v5_repeated_calls_are_bit_identical() -> None:
    # permanent regression test for the nondeterminism bug found by review
    # (top-1 flipped between identical calls because tau was drawn from
    # torch.rand instead of a fixed grid) -- promoted from an ad-hoc manual
    # check into a real, permanent test.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    scene = public_junction_scene()
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
    bank.update({0: (0.1, 3.9)})
    bank.update({0: (0.05, 3.6)})
    robot_obs = RobotObservation(px=0.0, py=0.5, gx=0.0, gy=5.0, vx=0.0, vy=0.5, radius=0.3, v_pref=1.0, theta=np.pi / 2)
    humans = [HumanObservation(0, 0.05, 3.6, 0.0, -0.8, 0.3)]
    human_feats, human_mask = build_intent_human_feature_batch(bank, robot_obs, humans, mode="full", rng=np.random.default_rng(0))
    results_1 = score_candidates_v5(model, robot_obs, human_feats, human_mask, action_table, remaining_fraction=0.5)
    results_2 = score_candidates_v5(model, robot_obs, human_feats, human_mask, action_table, remaining_fraction=0.5)
    q1 = [r.q_mean for r in results_1]
    q2 = [r.q_mean for r in results_2]
    assert q1 == q2, "identical input must score bit-identically across repeated calls"
    top1_a = max(results_1, key=lambda r: r.q_mean).action_index
    top1_b = max(results_2, key=lambda r: r.q_mean).action_index
    assert top1_a == top1_b


def test_c0_frozen_four_arm_definitions() -> None:
    # plan section 3.2 / C0.4. Two successive real bugs were found here:
    # (1) every arm secretly saw the real posterior; (2) `mean` was then
    # given a MAP ONE-HOT vector while its futures were the posterior-
    # WEIGHTED mean -- a hybrid arm that is neither pure MAP nor pure
    # posterior-mean, and which still leaked the posterior's argmax.
    # This test pins the FROZEN definitions so neither can regress.
    scene = public_junction_scene()
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
    # drive the tracker to a genuinely NON-uniform, non-degenerate posterior
    for py in (4.0, 3.7, 3.4, 3.1, 2.9):
        bank.update({0: (0.0, py)})
    bank.update({0: (-0.25, 2.6)})  # veer left -> posterior must become lopsided
    real_belief = bank.belief_for(0)
    assert len(real_belief) == 2
    assert abs(real_belief[0] - real_belief[1]) > 0.05, (
        f"test precondition: the real posterior must be clearly non-uniform, got {real_belief}")

    robot = RobotObservation(px=0.0, py=0.5, gx=0.0, gy=5.0, vx=0.0, vy=0.5, radius=0.3, v_pref=1.0, theta=np.pi / 2)
    humans = [HumanObservation(0, -0.25, 2.6, -0.5, -0.6, 0.3)]

    # V6 layout: 13 scalars, then G x F candidate features, then the G-wide
    # validity mask. The per-goal probability is column 0 of each candidate
    # row rather than a standalone p0..p7 block.
    n_cand = len(real_belief)
    G, F, S = MAX_CANDIDATE_GOALS, CANDIDATE_FEATURE_DIM, HUMAN_SCALAR_DIM_V6
    ENTROPY, MARGIN = 8, 9
    FUT_DX, FUT_DY, SPREAD = 10, 11, 12
    MASK = slice(S + G * F, S + G * F + G)

    def probs(row):
        return row[S:S + G * F].reshape(G, F)[:, 0]

    def cand_rows(row):
        return row[S:S + G * F].reshape(G, F)

    feats = {}
    for mode in ("full", "mean", "cv", "uniform"):
        f, m = build_intent_human_feature_batch(
            bank, robot, humans, mode=mode, rng=np.random.default_rng(0), n_samples=200)
        assert bool(m[0])
        feats[mode] = f[0]

    # --- full: the real posterior, verbatim ---
    assert np.allclose(probs(feats["full"])[:n_cand], real_belief, atol=1e-6)
    assert feats["full"][ENTROPY] > 0.0

    # --- mean: NO per-goal vector at all (hence entropy 0 and margin 0),
    #     spread exactly 0, but a real posterior-weighted mean future ---
    assert np.all(cand_rows(feats["mean"]) == 0.0), (
        "mean must not expose any per-goal probability OR candidate geometry, got "
        f"{cand_rows(feats['mean'])[:n_cand]}")
    assert feats["mean"][ENTROPY] == 0.0 and feats["mean"][MARGIN] == 0.0, (
        "mean must not leak posterior shape through the entropy/margin scalars either")
    assert feats["mean"][SPREAD] == 0.0, "mean is a single averaged trajectory -> spread must be exactly 0"
    assert not (feats["mean"][FUT_DX] == 0.0 and feats["mean"][FUT_DY] == 0.0), (
        "mean's whole content is the posterior-weighted mean future: its delta vs CV must be non-zero")

    # --- cv: no goal information whatsoever; future IS the CV future ---
    assert np.all(cand_rows(feats["cv"]) == 0.0)
    assert feats["cv"][ENTROPY] == 0.0 and feats["cv"][MARGIN] == 0.0
    assert feats["cv"][SPREAD] == 0.0
    assert abs(feats["cv"][FUT_DX]) < 1e-6 and abs(feats["cv"][FUT_DY]) < 1e-6, (
        "cv's predicted future is the CV baseline itself, so its delta vs CV must be ~0")

    # --- mean vs cv must be genuinely distinguishable, and the ONLY thing
    #     separating them is the goal-conditioned predicted future ---
    assert not np.allclose(feats["mean"], feats["cv"]), "mean and cv must not collapse to the same feature vector"
    differing = np.where(~np.isclose(feats["mean"], feats["cv"], atol=1e-6))[0]
    assert set(differing.tolist()) <= {FUT_DX, FUT_DY}, (
        f"mean and cv may differ ONLY in the future-mean-delta dims, but differ at {differing.tolist()}")

    # --- uniform: flat 1/n, a SUPPLEMENTARY control, never the posterior ---
    assert np.allclose(probs(feats["uniform"])[:n_cand], np.ones(n_cand) / n_cand, atol=1e-6)
    assert not np.allclose(probs(feats["uniform"])[:n_cand], real_belief, atol=1e-3), (
        "uniform must never be the fitted posterior")

    # --- the public validity mask is identical across all four arms
    #     (it is public scene geometry, not posterior information) ---
    for mode in ("mean", "cv", "uniform"):
        assert np.array_equal(feats[mode][MASK], feats["full"][MASK]), (
            f"{mode}'s candidate validity mask must match full's -- it encodes only how many PUBLIC "
            f"destinations exist, which is not posterior information")
