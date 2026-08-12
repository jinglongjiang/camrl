"""Split from selftest.py (guide/review point 1): junction_scenario + intent_train (IL/online-RL/ablation) tests. Shares the
FULL original test namespace via ``from ..tests._common import *`` so
every test body is copied VERBATIM (byte-identical) from the original
monolithic file -- zero risk of a name/import mismatch during the split.
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403


def test_junction_scenario_seed_ranges_frozen_and_disjoint() -> None:
    assert set(JUNCTION_TRAIN_SEEDS) & set(JUNCTION_HELDOUT_SEEDS) == set()
    assert len(JUNCTION_TRAIN_SEEDS) == 200 and len(JUNCTION_HELDOUT_SEEDS) == 100
    try:
        JunctionEpisodeConfig(episode_seed=JUNCTION_HELDOUT_SEEDS[0], is_heldout=False)
        assert False, "expected JunctionScenarioError: heldout seed used as train"
    except JunctionScenarioError:
        pass
    try:
        JunctionEpisodeConfig(episode_seed=JUNCTION_TRAIN_SEEDS[0], is_heldout=True)
        assert False, "expected JunctionScenarioError: train seed used as heldout"
    except JunctionScenarioError:
        pass

def test_junction_scenario_forces_genuine_conflict_for_naive_robot() -> None:
    # guide/consolidation plan §4b hard principle, verified by measurement:
    # collision risk must depend on the pedestrian's unknown exit -- a
    # goal-blind (ignores the human entirely) robot must come genuinely
    # close to at least SOME fraction of episodes, proving the geometry
    # itself (not any clever policy) creates the conflict.
    env_config_path = REPO_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"
    clearances = []
    for seed in JUNCTION_TRAIN_SEEDS[:8]:
        cfg = JunctionEpisodeConfig(episode_seed=seed, is_heldout=False)
        env, robot, true_exit = build_junction_episode(env_config_path, cfg)
        assert true_exit in ("left", "right")
        ped = env.humans[0]
        min_dist = float("inf")
        waypoint_reached = False
        for _ in range(60):
            waypoint_reached = maybe_reveal_exit(ped, true_exit, waypoint_reached)
            d = float(np.hypot(robot.px - ped.px, robot.py - ped.py)) - robot.radius - ped.radius
            min_dist = min(min_dist, d)
            goal_vec = np.array([robot.gx - robot.px, robot.gy - robot.py])
            n = float(np.linalg.norm(goal_vec))
            if n < robot.radius:
                break
            vel = robot.v_pref * goal_vec / n
            robot.px += float(vel[0]) * FROZEN_VALUES["dt"]
            robot.py += float(vel[1]) * FROZEN_VALUES["dt"]
            ped.step(ped.act([robot.get_observable_state()]))
        clearances.append(min_dist)
    n_at_risk = sum(1 for c in clearances if c < 0.3)
    assert n_at_risk >= len(clearances) // 2, (
        f"geometry must force genuine conflict for a naive robot in at least half the episodes, got {n_at_risk}/{len(clearances)} (clearances={clearances})")

def test_junction_scenario_ambiguity_overlaps_conflict() -> None:
    # the OTHER half of §4b's principle: the conflict must occur WHILE the
    # real tracker's belief is still ambiguous, not only after it resolves.
    env_config_path = REPO_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"
    scene = public_junction_scene()
    n_overlap = 0
    seeds = JUNCTION_TRAIN_SEEDS[:6]
    for seed in seeds:
        cfg = JunctionEpisodeConfig(episode_seed=seed, is_heldout=False)
        env, robot, true_exit = build_junction_episode(env_config_path, cfg)
        ped = env.humans[0]
        bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
        conflict_step, collapse_step = None, None
        waypoint_reached = False
        for step in range(60):
            waypoint_reached = maybe_reveal_exit(ped, true_exit, waypoint_reached)
            bank.update({0: (ped.px, ped.py)})
            b = bank.belief_for(0)
            if collapse_step is None and min(b) < 0.15:
                collapse_step = step
            d = float(np.hypot(robot.px - ped.px, robot.py - ped.py)) - robot.radius - ped.radius
            if conflict_step is None and d < 0.5:
                conflict_step = step
            goal_vec = np.array([robot.gx - robot.px, robot.gy - robot.py])
            n = float(np.linalg.norm(goal_vec))
            if n < robot.radius:
                break
            vel = robot.v_pref * goal_vec / n
            robot.px += float(vel[0]) * FROZEN_VALUES["dt"]
            robot.py += float(vel[1]) * FROZEN_VALUES["dt"]
            ped.step(ped.act([robot.get_observable_state()]))
        if conflict_step is not None and (collapse_step is None or conflict_step <= collapse_step):
            n_overlap += 1
    assert n_overlap == len(seeds), f"ambiguity must overlap the conflict window in every episode, got {n_overlap}/{len(seeds)}"

def test_junction_pedestrian_left_right_identical_before_fork() -> None:
    # directly enforces the fix for the real bug found by external review:
    # two pedestrians with the SAME start position/speed but DIFFERENT
    # true_exit must produce BIT-IDENTICAL position sequences up until the
    # waypoint-switch step, then diverge after -- proving the "ambiguity"
    # measured elsewhere is genuine (the trajectory itself carries no
    # information about the hidden exit before the switch), not an
    # artifact of the tracker's own wrong model of an already-resolved path.
    import copy
    env_config_path = REPO_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"
    cfg = JunctionEpisodeConfig(episode_seed=JUNCTION_TRAIN_SEEDS[0], is_heldout=False)
    env, robot, _true_exit = build_junction_episode(env_config_path, cfg)
    ped_template = env.humans[0]

    ped_left = copy.deepcopy(ped_template)
    ped_right = copy.deepcopy(ped_template)
    wp_left, wp_right = False, False
    left_positions, right_positions = [], []
    fork_step = None
    for step in range(80):
        wp_left = maybe_reveal_exit(ped_left, "left", wp_left)
        wp_right = maybe_reveal_exit(ped_right, "right", wp_right)
        left_positions.append((ped_left.px, ped_left.py))
        right_positions.append((ped_right.px, ped_right.py))
        if fork_step is None and wp_left:
            fork_step = step
        ped_left.step(ped_left.act([]))
        ped_right.step(ped_right.act([]))

    assert fork_step is not None, "pedestrian never reached the shared junction waypoint within 80 steps"
    for i in range(fork_step + 1):
        assert left_positions[i] == right_positions[i], (
            f"left/right trajectories must be bit-identical before the fork, diverged at step {i}: "
            f"{left_positions[i]} != {right_positions[i]}")
    diverged = any(
        left_positions[i] != right_positions[i] for i in range(fork_step + 1, len(left_positions))
    )
    assert diverged, "left/right trajectories must diverge after the hidden exit is revealed"

def test_junction_scenario_no_sbk_hmm_import() -> None:
    import subprocess
    import sys as _sys
    code = (
        "import sys\n"
        "import crowd_nav.bayesian_dvl.junction_scenario\n"
        "forbidden = ('bayesian_dvl.belief', 'bayesian_dvl.rollout', 'bayesian_dvl.world_model')\n"
        "bad = [m for m in sys.modules if any(f in m for f in forbidden)]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    result = subprocess.run([_sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout

def test_intent_train_standard_scenario_has_multiple_humans() -> None:
    # regression test for a real bug (found by measurement): CrowdSim.reset()
    # gates human_num behind robot.policy.multiagent_training (defaults to
    # None on a bare ORCA()) -- without setting it explicitly, reset()
    # silently forces human_num=1 regardless of config, which meant
    # AttentionPool never received a gradient (only one human is ever a
    # trivial softmax). Verify the standard scenario really has >1 human.
    res = collect_orca_episode(_env_config_path(), "standard", 700001)
    assert res.transitions[0].human_mask.sum() > 1, "standard scenario must have multiple humans, not 1"

def test_intent_train_collect_both_scenarios_valid_termination() -> None:
    # guide/review local-e2e hard requirement 5 item 5: standard (unimodal)
    # and multimodal (junction) scenarios must both legally terminate.
    for scenario, seed in [("standard", 700001), ("junction", 96001)]:
        res = collect_orca_episode(_env_config_path(), scenario, seed)
        assert res.outcome in ("success", "collision", "timeout")
        assert len(res.transitions) > 0
        assert all(np.isfinite(t.mc_return) for t in res.transitions)

def test_intent_train_executed_action_matches_80action_label_exactly() -> None:
    # real bug found by review: an earlier version stepped CrowdSim with the
    # RAW continuous ORCA velocity while labeling the transition with the
    # nearest grid action index -- so the recorded reward/return did not
    # strictly correspond to the labeled action. Fixed by stepping with
    # action_table[executed_idx] directly (holonomic kinematics -> robot.vx/
    # vy become EXACTLY that grid velocity). Verify directly, not just via
    # collect_orca_episode's black-box output.
    from crowd_sim.envs.utils.action import ActionXY
    env_config_path = _env_config_path()
    env, robot = _make_standard_env(env_config_path)
    env.case_counter["train"] = 700001 % (2**32 - 1)
    env.reset()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)
    for _ in range(5):
        orca_action = env.robot.act([h.get_observable_state() for h in env.humans])
        dists = np.hypot(action_table[:, 0] - orca_action.vx, action_table[:, 1] - orca_action.vy)
        executed_idx = int(np.argmin(dists))
        gvx, gvy = action_table[executed_idx]
        env.step(ActionXY(float(gvx), float(gvy)))
        assert abs(env.robot.vx - gvx) < 1e-9 and abs(env.robot.vy - gvy) < 1e-9, (
            f"robot velocity after step ({env.robot.vx}, {env.robot.vy}) must exactly match the "
            f"80-action label velocity ({gvx}, {gvy})")

def test_intent_train_compute_mc_returns_hand_computed() -> None:
    returns = compute_mc_returns([1.0, 1.0, 1.0], gamma=0.5)
    assert abs(returns[-1] - 1.0) < 1e-9
    assert abs(returns[-2] - 1.5) < 1e-9
    assert abs(returns[0] - 1.75) < 1e-9
    try:
        compute_mc_returns([], gamma=0.9)
        assert False, "expected IntentTrainError on empty rewards"
    except IntentTrainError:
        pass
    try:
        compute_mc_returns([1.0], gamma=0.0)
        assert False, "expected IntentTrainError on gamma<=0"
    except IntentTrainError:
        pass

def test_intent_train_real_gradient_flow_through_every_parameter() -> None:
    # guide/review local-e2e hard requirement 5 items 3-4: IL and RL must
    # produce REAL gradients. Collect real ORCA episodes from BOTH scenarios,
    # take real gradient steps, and verify EVERY parameter tensor moved and
    # the loss meaningfully decreased -- not just "the code ran".
    env_config_path = _env_config_path()
    transitions = []
    for scenario, seed in [("standard", 700001), ("standard", 700002), ("junction", 96001), ("junction", 96002), ("junction", 96003)]:
        transitions.extend(collect_orca_episode(env_config_path, scenario, seed).transitions)
    assert len(transitions) > 20

    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    il_batch = batch_to_tensors(transitions)
    gen = torch.Generator().manual_seed(0)
    results = [intent_train_step(model, opt, il_batch, gen) for _ in range(30)]
    losses = [r.loss for r in results]
    after = model.state_dict()
    unchanged = [k for k in before if torch.equal(before[k], after[k])]
    assert not unchanged, f"every parameter must receive real gradient, unchanged: {unchanged}"
    assert losses[-1] < losses[0] * 0.5, f"loss should meaningfully decrease with real learning, {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(np.isfinite(l) for l in losses)
    assert all(np.isfinite(r.mc_loss) and np.isfinite(r.rank_loss) for r in results)

def test_intent_train_rank_loss_teaches_executed_action_to_outrank_others() -> None:
    # real bug found by review: an earlier version had NO action-
    # classification/ranking supervision at all -- only MC-return
    # regression on the executed action, so nothing ever told the network
    # the executed action was chosen FOR A REASON over the other 79 grid
    # actions. Directly verify the fix teaches this: rank_loss must trend
    # down with training (the margin loss is exactly zero once the
    # executed action is well-separated from the hardest negative, so a
    # falling mean is real evidence of learned separation, not noise).
    env_config_path = _env_config_path()
    transitions = []
    for scenario, seed in [("standard", 700001), ("junction", 96001), ("junction", 96002)]:
        transitions.extend(collect_orca_episode(env_config_path, scenario, seed).transitions)
    assert len(transitions) > 10

    torch.manual_seed(1)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    il_batch = batch_to_tensors(transitions)
    gen = torch.Generator().manual_seed(1)
    results = [intent_train_step(model, opt, il_batch, gen, lambda_rank=1.0) for _ in range(40)]
    rank_losses = [r.rank_loss for r in results]
    early_mean = float(np.mean(rank_losses[:5]))
    late_mean = float(np.mean(rank_losses[-5:]))
    assert late_mean < early_mean, f"rank_loss should trend down with training, {early_mean:.4f} -> {late_mean:.4f}"

    # a zero ranking_batch_size must fail loudly, not silently skip the term
    try:
        intent_train_step(model, opt, il_batch, gen, ranking_batch_size=0)
        assert False, "expected IntentTrainError on ranking_batch_size=0"
    except IntentTrainError:
        pass

def test_intent_train_resume_is_bit_identical_to_continuous() -> None:
    # guide/review local-e2e hard requirement 5 item 4: resume before/after
    # must give consistent results. Regression test for a real bug found by
    # measurement: an earlier train_step drew tau from the UNSEEDED global
    # torch RNG, so reconstructing a fresh model mid-script (even though its
    # weights are immediately overwritten by load) shifted every subsequent
    # tau draw and silently made "resumed" training diverge from what
    # "continuous" training would have done. Fixed with an explicit,
    # checkpointable torch.Generator; this test proves resumed == continuous
    # bit-for-bit, not just "close".
    env_config_path = _env_config_path()
    transitions = []
    for scenario, seed in [("standard", 700001), ("junction", 96001), ("junction", 96002)]:
        transitions.extend(collect_orca_episode(env_config_path, scenario, seed).transitions)
    il_batch = batch_to_tensors(transitions)

    torch.manual_seed(7)
    model_a = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    gen_a = torch.Generator().manual_seed(123)
    losses_a = [intent_train_step(model_a, opt_a, il_batch, gen_a).loss for _ in range(20)]

    torch.manual_seed(7)
    model_b = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    gen_b = torch.Generator().manual_seed(123)
    for _ in range(10):
        intent_train_step(model_b, opt_b, il_batch, gen_b)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ckpt.pth")
        save_intent_checkpoint(model_b, path, action_grid_hash="h", scene_registry_sha256="s",
                                optimizer=opt_b, extra={"generator_state": gen_b.get_state()})
        model_c = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)  # extra random draws here, on purpose
        opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)
        raw = torch.load(path, weights_only=False)
        load_intent_checkpoint(path, model_c, optimizer=opt_c, expected_action_grid_hash="h", expected_scene_registry_sha256="s")
        gen_c = torch.Generator(); gen_c.set_state(raw["extra"]["generator_state"])
        losses_resumed = [intent_train_step(model_c, opt_c, il_batch, gen_c).loss for _ in range(10)]

    assert all(a == b for a, b in zip(losses_a[10:], losses_resumed)), (losses_a[10:], losses_resumed)
    assert all(torch.equal(pa, pc) for pa, pc in zip(model_a.parameters(), model_c.parameters()))

def test_online_replay_buffer_fail_closed() -> None:
    for dc, oc in ((0, 4), (4, 0), (-1, 4)):
        try:
            IntentReplay(demo_capacity=dc, online_capacity=oc)
            assert False, f"expected IntentTrainError on capacities {dc}/{oc}"
        except IntentTrainError:
            pass
    buf = IntentReplay(demo_capacity=4, online_capacity=4)
    try:
        buf.sample(1, np.random.default_rng(0), demo_ratio=0.2)
        assert False, "expected IntentTrainError sampling an empty replay"
    except IntentTrainError:
        pass
    env_config_path = _env_config_path()
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions[:2]
    buf.add_demo(demo, np.random.default_rng(0))
    for bad in (-0.1, 1.1):
        try:
            buf.sample(2, np.random.default_rng(0), demo_ratio=bad)
            assert False, f"expected IntentTrainError on demo_ratio {bad}"
        except IntentTrainError:
            pass
    try:
        buf.sample(0, np.random.default_rng(0), demo_ratio=1.0)
        assert False, "expected IntentTrainError on batch_size 0"
    except IntentTrainError:
        pass
    # role enforcement: a demo transition may not enter the online ring
    try:
        buf.add_online(demo)
        assert False, "expected IntentTrainError putting a demo sample in the online ring"
    except IntentTrainError:
        pass


def test_online_replay_buffer_ring_overflow_and_sampling() -> None:
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ep1 = collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=1.0,
                                  explore_rng=np.random.default_rng(1))
    ep2 = collect_online_episode(env_config_path, model, action_table, "standard", 700002, epsilon=1.0,
                                  explore_rng=np.random.default_rng(2))
    assert len(ep1.transitions) > 0 and ep1.outcome in ("success", "collision", "timeout")

    capacity = 5
    buf = IntentReplay(demo_capacity=100, online_capacity=capacity)
    buf.add_online(ep1.transitions)
    buf.add_online(ep2.transitions)
    assert buf.n_online == min(capacity, len(ep1.transitions) + len(ep2.transitions))
    assert any(t is ep2.transitions[-1] for t in buf._online), "ring must retain the most recent adds"
    batch = buf.sample(3, np.random.default_rng(0), demo_ratio=0.0)
    assert len(batch) == 3 and all(t.source_role == "online" for t in batch)


def test_online_replay_buffer_state_roundtrip() -> None:
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ep = collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=1.0,
                                 explore_rng=np.random.default_rng(1))
    demo = collect_orca_episode(env_config_path, "standard", 700002).transitions
    a = IntentReplay(demo_capacity=100, online_capacity=100)
    a.add_online(ep.transitions)
    a.add_demo(demo, np.random.default_rng(0))
    state = a.state_dict()

    b = IntentReplay(demo_capacity=100, online_capacity=100)
    b.load_state_dict(state)
    assert (b.n_demo, b.n_online) == (a.n_demo, a.n_online)
    assert b._demo_seen == a._demo_seen, "the reservoir counter must survive resume"
    assert all(x.action_index == y.action_index for x, y in zip(a._online, b._online))

    c = IntentReplay(demo_capacity=50, online_capacity=100)
    try:
        c.load_state_dict(state)
        assert False, "expected IntentTrainError on capacity mismatch"
    except IntentTrainError:
        pass


def test_c4r_mixed_replay_keeps_demo_supervision_alive_during_online() -> None:
    # Order C4R.2 / audit 2.2 point 2: the previous online-only buffer made
    # rank_loss permanently ZERO once IL ended -- mathematically no longer
    # wrong, but it discards the expert supervision entirely and invites
    # forgetting. Verify a mixed batch really contains demo rows and really
    # produces a non-zero ranking loss.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions
    online = collect_online_episode(env_config_path, model, action_table, "standard", 700002, epsilon=1.0,
                                     explore_rng=np.random.default_rng(3)).transitions
    buf = IntentReplay(demo_capacity=1000, online_capacity=1000)
    buf.add_demo(demo, np.random.default_rng(0))
    buf.add_online(online)

    rng = np.random.default_rng(7)
    batch = buf.sample(100, rng, demo_ratio=0.20)
    n_demo = sum(1 for t in batch if t.source_role == "demo")
    assert n_demo == 20, f"a 0.20 demo ratio over 100 must give exactly 20 demo rows, got {n_demo}"
    tb = batch_to_tensors(batch)
    assert int(tb.demo_mask.sum()) == 20
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    r = intent_train_step(model, opt, tb, torch.Generator().manual_seed(0), lambda_rank=1.0)
    assert r.rank_loss > 0.0, "a mixed batch must still produce a real ranking loss"
    assert r.n_demo == 20 and r.n_online == 80

    # a pure-online batch is still exactly zero (the C0 contract holds)
    ob = batch_to_tensors(buf.sample(50, rng, demo_ratio=0.0))
    r0 = intent_train_step(model, opt, ob, torch.Generator().manual_seed(0), lambda_rank=1000.0)
    assert r0.rank_loss == 0.0 and abs(r0.loss - r0.mc_loss) < 1e-12


def test_c4r_demo_reservoir_is_bounded_and_uniform() -> None:
    # the demo side must stay within a fixed memory budget no matter how
    # much IL data is offered (the formal budget is ~195k transitions).
    env_config_path = _env_config_path()
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions
    cap = 10
    buf = IntentReplay(demo_capacity=cap, online_capacity=10)
    buf.add_demo(demo, np.random.default_rng(0))
    assert buf.n_demo == cap, f"reservoir must cap at {cap}, got {buf.n_demo}"
    assert buf._demo_seen == len(demo), "the reservoir must count every item ever offered"
    # every retained item must be a real demo sample with an expert set
    for t in buf._demo:
        assert t.source_role == "demo" and t.expert_action_indices


def test_c4r_grad_clipping_and_gradient_ratio_are_really_applied() -> None:
    # Order C4R.3 / audit 2.2 point 3: grad_clip_norm and the ratio gate
    # were declared in the frozen config, validated on load, and then
    # NEVER used by train_step -- contract drift. Verify both are real.
    env_config_path = _env_config_path()
    trans = collect_orca_episode(env_config_path, "standard", 700001).transitions[:16]
    batch = batch_to_tensors(trans)
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt = torch.optim.Adam(model.parameters(), lr=0.0)

    # (a) with a tiny clip the post-clip gradient norm must equal the clip
    r = intent_train_step(model, opt, batch, torch.Generator().manual_seed(0),
                          lambda_rank=380.0, grad_clip_norm=0.01)
    post = float(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5)
    assert r.clipped is True and r.grad_norm_preclip > 0.01
    assert abs(post - 0.01) < 1e-4, f"gradients must actually be clipped to 0.01, post-clip norm {post}"

    # (b) with a huge clip nothing is clipped and the norm is untouched
    r2 = intent_train_step(model, opt, batch, torch.Generator().manual_seed(0),
                           lambda_rank=380.0, grad_clip_norm=1e9)
    post2 = float(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5)
    assert r2.clipped is False and abs(post2 - r2.grad_norm_preclip) < 1e-3

    # (c) an invalid clip fails loudly
    try:
        intent_train_step(model, opt, batch, torch.Generator().manual_seed(0), grad_clip_norm=0.0)
        assert False, "expected IntentTrainError on a non-positive grad_clip_norm"
    except IntentTrainError:
        pass

    # (d) the ratio is measured PER LOSS TERM, not faked from the total
    r3 = intent_train_step(model, opt, batch, torch.Generator().manual_seed(0),
                           lambda_rank=380.0, measure_gradient_ratio=True)
    assert r3.ratio_measured and r3.mc_grad_norm > 0 and r3.rank_grad_norm > 0
    assert abs(r3.weighted_rank_grad_norm - 380.0 * r3.rank_grad_norm) < 1e-6
    assert abs(r3.gradient_ratio - r3.weighted_rank_grad_norm / r3.mc_grad_norm) < 1e-6
    assert r3.mc_grad_norm != r3.grad_norm_preclip, (
        "the per-term MC gradient must not be the TOTAL gradient norm in disguise")
    # at the frozen lambda_rank the two terms should be roughly balanced
    assert 0.05 <= r3.gradient_ratio <= 50.0, (
        f"at the audited lambda_rank the ratio should sit inside the frozen gate, got {r3.gradient_ratio}")
    # and when not asked for, it is not computed (it costs 2 extra passes)
    r4 = intent_train_step(model, opt, batch, torch.Generator().manual_seed(0), measure_gradient_ratio=False)
    assert not r4.ratio_measured and r4.mc_grad_norm == 0.0


def test_c4r_gradient_ratio_monitor_sustained_window() -> None:
    m = GradientRatioMonitor(ratio_min=0.05, ratio_max=50.0, sustained_updates=3)
    assert m.observe(1.0) is None
    # isolated excursions must NOT abort
    assert m.observe(1e6) is None
    assert m.observe(1.0) is None and m.consecutive_out_of_range == 0
    # a sustained run must
    assert m.observe(1e6) is None
    assert m.observe(1e6) is None
    reason = m.observe(1e6)
    assert reason is not None and "consecutive" in reason
    assert m.n_measured == 6 and m.n_out_of_range == 4
    state = m.state_dict()
    m2 = GradientRatioMonitor(0.05, 50.0, 3)
    m2.load_state_dict(state)
    assert m2.consecutive_out_of_range == m.consecutive_out_of_range
    for bad in ((0.0, 1.0, 3), (1.0, 0.5, 3), (0.05, 50.0, 0)):
        try:
            GradientRatioMonitor(*bad)
            assert False, f"expected IntentTrainError on {bad}"
        except IntentTrainError:
            pass


def test_c4r_il_update_uses_minibatches_not_the_whole_corpus() -> None:
    # Order C4R.1 / audit 2.2 point 1: IL previously moved EVERY demo
    # transition to the device as one batch and ran full-batch updates.
    # An update must now cost exactly batch_size rows regardless of how
    # much demo data exists.
    env_config_path = _env_config_path()
    demo = []
    for s in (700001, 700002, 700003):
        demo.extend(collect_orca_episode(env_config_path, "standard", s).transitions)
    assert len(demo) > 64
    buf = IntentReplay(demo_capacity=100000, online_capacity=100)
    buf.add_demo(demo, np.random.default_rng(0))
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for bs in (8, 32):
        r = run_il_update(model, opt, buf, bs, np.random.default_rng(1), torch.Generator().manual_seed(0),
                          ranking_batch_size=4, grad_clip_norm=10.0)
        assert r.n_demo == bs, f"an IL update must consume exactly batch_size={bs} rows, got {r.n_demo}"
        assert r.n_online == 0, "IL updates draw from the demo side only"
        assert r.rank_loss > 0.0


def test_collect_online_episode_epsilon_zero_is_deterministic_given_model() -> None:
    # epsilon=0 -> purely greedy over the model's own scoring -- no
    # exploration randomness should affect the trajectory at all (only the
    # per-step belief-sampling rng, which is seeded from episode_seed, same
    # as collect_orca_episode/run_ablation_episode).
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ep_a = collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=0.0,
                                   explore_rng=np.random.default_rng(111))
    ep_b = collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=0.0,
                                   explore_rng=np.random.default_rng(999))
    assert [t.action_index for t in ep_a.transitions] == [t.action_index for t in ep_b.transitions]
    assert ep_a.outcome == ep_b.outcome

    try:
        collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=1.5,
                                explore_rng=np.random.default_rng(0))
        assert False, "expected IntentTrainError on epsilon out of [0,1]"
    except IntentTrainError:
        pass

def test_online_training_step_real_gradient_and_resume_bit_identical() -> None:
    # guide/review point 5: real online RL requires policy-driven
    # exploration, a replay buffer, and MC updates from online experience,
    # with EXACT resume of that loop's state -- not just the model. Verify
    # both: (a) online training moves every parameter with a real gradient,
    # (b) reconstructing buffer + explore_rng + sample_rng + tau_generator
    # from persisted state reproduces the continuous run bit-for-bit.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    seeds = [700001, 700002, 700003, 700004, 700005, 700006]

    def make_state(seed):
        torch.manual_seed(seed)
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        buf = IntentReplay(demo_capacity=200, online_capacity=200)
        explore_rng = np.random.default_rng(1000 + seed)
        sample_rng = np.random.default_rng(2000 + seed)
        tau_gen = torch.Generator().manual_seed(3000 + seed)
        return model, opt, buf, explore_rng, sample_rng, tau_gen

    model_a, opt_a, buf_a, exr_a, smr_a, tg_a = make_state(7)
    before = {k: v.clone() for k, v in model_a.state_dict().items()}
    results_a = [
        run_online_training_step(env_config_path, model_a, opt_a, action_table, "standard", seeds[i % len(seeds)],
                                  epsilon=0.3, buffer=buf_a, batch_size=8, explore_rng=exr_a, sample_rng=smr_a,
                                  tau_generator=tg_a)
        for i in range(12)
    ]
    after = model_a.state_dict()
    unchanged = [k for k in before if torch.equal(before[k], after[k])]
    assert not unchanged, f"every parameter must receive real gradient from online training, unchanged: {unchanged}"
    assert all(np.isfinite(r.loss) for r in results_a)

    model_b, opt_b, buf_b, exr_b, smr_b, tg_b = make_state(7)
    for i in range(6):
        run_online_training_step(env_config_path, model_b, opt_b, action_table, "standard", seeds[i % len(seeds)],
                                  epsilon=0.3, buffer=buf_b, batch_size=8, explore_rng=exr_b, sample_rng=smr_b,
                                  tau_generator=tg_b)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "online_ckpt.pth")
        save_intent_checkpoint(
            model_b, path, action_grid_hash="h", scene_registry_sha256="s", optimizer=opt_b,
            extra={
                "tau_generator_state": tg_b.get_state(),
                "explore_rng_state": exr_b.bit_generator.state,
                "sample_rng_state": smr_b.bit_generator.state,
                "replay_buffer_state": buf_b.state_dict(),
            },
        )
        model_c = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)  # extra random draws, on purpose
        opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)
        raw = torch.load(path, weights_only=False)
        load_intent_checkpoint(path, model_c, optimizer=opt_c, expected_action_grid_hash="h", expected_scene_registry_sha256="s")
        tg_c = torch.Generator(); tg_c.set_state(raw["extra"]["tau_generator_state"])
        exr_c = np.random.default_rng(0); exr_c.bit_generator.state = raw["extra"]["explore_rng_state"]
        smr_c = np.random.default_rng(0); smr_c.bit_generator.state = raw["extra"]["sample_rng_state"]
        buf_c = IntentReplay(demo_capacity=200, online_capacity=200)
        buf_c.load_state_dict(raw["extra"]["replay_buffer_state"])

        results_resumed = [
            run_online_training_step(env_config_path, model_c, opt_c, action_table, "standard", seeds[i % len(seeds)],
                                      epsilon=0.3, buffer=buf_c, batch_size=8, explore_rng=exr_c, sample_rng=smr_c,
                                      tau_generator=tg_c)
            for i in range(6, 12)
        ]

    assert all(a.loss == b.loss for a, b in zip(results_a[6:], results_resumed)), (
        [r.loss for r in results_a[6:]], [r.loss for r in results_resumed])
    assert all(torch.equal(pa, pc) for pa, pc in zip(model_a.parameters(), model_c.parameters()))

def test_intent_ablation_suite_shares_identical_seeds_across_modes() -> None:
    # guide/review consolidation plan hard requirement 4: full/mean/cv/
    # no-belief(uniform) must share the identical env trajectory, episode
    # seed, and checkpoint -- only the belief representation differs.
    from crowd_nav.bayesian_dvl.intent_train import run_ablation_suite
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    seeds = list(JUNCTION_TRAIN_SEEDS[:3])
    results = run_ablation_suite(env_config_path, model, action_table, "junction", seeds, n_samples=20)
    assert set(results.keys()) == {"full", "mean", "cv", "uniform"}
    seed_sequences = {mode: tuple(e.episode_seed for e in eps) for mode, eps in results.items()}
    assert len(set(seed_sequences.values())) == 1, f"all modes must share identical episode seeds: {seed_sequences}"
    for mode, eps in results.items():
        assert all(e.outcome in ("success", "collision", "timeout") for e in eps)
        assert all(e.steps > 0 for e in eps)

def test_build_junction_episode_initial_state_is_deterministic_from_seed() -> None:
    # review point 7: the ablation suite's "all modes share identical
    # seeds" claim is only meaningful if the same seed really does produce
    # a bit-identical SETUP (initial state), not merely the same integer in
    # a list. Directly verify that property at its source: two independent
    # constructions from the same episode_seed must be bit-identical.
    env_config_path = _env_config_path()
    cfg = JunctionEpisodeConfig(episode_seed=JUNCTION_TRAIN_SEEDS[5], is_heldout=False)
    env1, robot1, exit1 = build_junction_episode(env_config_path, cfg)
    env2, robot2, exit2 = build_junction_episode(env_config_path, cfg)
    ped1, ped2 = env1.humans[0], env2.humans[0]
    assert exit1 == exit2
    assert (robot1.px, robot1.py, robot1.gx, robot1.gy, robot1.v_pref, robot1.theta) == (
        robot2.px, robot2.py, robot2.gx, robot2.gy, robot2.v_pref, robot2.theta)
    assert (ped1.px, ped1.py, ped1.gx, ped1.gy, ped1.v_pref) == (ped2.px, ped2.py, ped2.gx, ped2.gy, ped2.v_pref)

def test_ablation_cv_mode_outcome_is_independent_of_planner_seed() -> None:
    # review point 7: "exogenous random-stream independence" -- cv mode
    # never samples from planner_rng (it's a deterministic constant-
    # velocity extrapolation, see build_intent_human_feature_batch), so its
    # trajectory must be IDENTICAL regardless of which planner_seed is
    # used. This directly proves the shared-planner_seed convention in
    # run_ablation_suite isn't silently leaking mode-specific randomness
    # into what's supposed to be a pure belief-representation ablation.
    from crowd_nav.bayesian_dvl.intent_train import run_ablation_episode
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    episode_seed = JUNCTION_TRAIN_SEEDS[7]
    result_a = run_ablation_episode(env_config_path, model, action_table, "junction", episode_seed, "cv",
                                     planner_seed=5_000_000 + episode_seed, n_samples=20)
    result_b = run_ablation_episode(env_config_path, model, action_table, "junction", episode_seed, "cv",
                                     planner_seed=999_999_999, n_samples=20)
    assert result_a.outcome == result_b.outcome and result_a.steps == result_b.steps, (
        f"cv mode must be independent of planner_seed: {result_a} vs {result_b}")


def test_ema_model_fail_closed() -> None:
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    try:
        EMAModel(model, decay=1.5)
        assert False, "expected IntentTrainError on decay outside (0,1)"
    except IntentTrainError:
        pass
    try:
        EMAModel(model, decay=0.0)
        assert False, "expected IntentTrainError on decay=0.0"
    except IntentTrainError:
        pass


def test_ema_model_tracks_a_smoothed_average_not_the_raw_weights() -> None:
    # real bug found by review: point 8 explicitly requires EMA -- without
    # it, deployment/eval reads the raw, noisy training weights of
    # whatever step happened to be last. Directly verify the EMA state
    # differs from (and is a real weighted average trailing) the raw
    # model's weights once the raw model has moved.
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ema = EMAModel(model, decay=0.9)
    initial_param = next(iter(model.state_dict().values())).clone()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    env_config_path = _env_config_path()
    transitions = collect_orca_episode(env_config_path, "standard", 700001).transitions
    il_batch = batch_to_tensors(transitions)
    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        intent_train_step(model, opt, il_batch, gen)
        ema.update(model)

    raw_param = next(iter(model.state_dict().values()))
    ema_param = next(iter(ema.state_dict().values()))
    assert not torch.equal(raw_param, initial_param), "sanity: raw model must have actually moved"
    assert not torch.equal(ema_param, raw_param), "EMA shadow must differ from the raw (noisier) weights"
    assert not torch.equal(ema_param, initial_param), "EMA shadow must have moved from its init too"

    # copy_to loads the smoothed weights into a fresh model exactly
    eval_model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ema.copy_to(eval_model)
    for k, v in eval_model.state_dict().items():
        assert torch.equal(v, ema.shadow[k])


def test_ema_model_state_roundtrip_and_key_mismatch_fail_closed() -> None:
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ema_a = EMAModel(model, decay=0.9)
    with torch.no_grad():
        for v in model.parameters():
            v.add_(1.0)
    ema_a.update(model)
    state = ema_a.state_dict()

    ema_b = EMAModel(model, decay=0.9)
    ema_b.load_state_dict(state)
    assert all(torch.equal(ema_a.shadow[k], ema_b.shadow[k]) for k in ema_a.shadow)

    bad_state = dict(state)
    bad_state.pop(next(iter(bad_state)))
    try:
        ema_b.load_state_dict(bad_state)
        assert False, "expected IntentTrainError on key mismatch"
    except IntentTrainError:
        pass


def test_formal_six_scenarios_cover_baseline_dense_large_circle_square() -> None:
    assert set(FORMAL_SIX_SCENARIOS.keys()) == {
        "baseline_circle", "baseline_square", "dense_circle", "dense_square", "large_circle", "large_square",
    }
    for name, (shape, size, human_num) in FORMAL_SIX_SCENARIOS.items():
        assert shape in ("circle", "square")
        assert size > 0
        assert human_num > 0
    assert len(FORMAL_EVAL_HELDOUT_SEEDS) == 100
    # disjoint from the junction train/heldout seed blocks (same "9Xxxx
    # block" convention as junction_scenario.py's frozen seed ranges).
    assert not (set(FORMAL_EVAL_HELDOUT_SEEDS) & set(JUNCTION_TRAIN_SEEDS))
    assert not (set(FORMAL_EVAL_HELDOUT_SEEDS) & set(JUNCTION_HELDOUT_SEEDS))


def test_build_formal_scenario_env_rejects_unknown_scenario() -> None:
    env_config_path = _env_config_path()
    try:
        build_formal_scenario_env(env_config_path, "not_a_real_scenario")
        assert False, "expected IntentTrainError on unknown scenario name"
    except IntentTrainError:
        pass


def test_run_formal_scenario_episode_baseline_circle_matches_human_count() -> None:
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    env, robot, shape, size = build_formal_scenario_env(env_config_path, "baseline_circle")
    assert shape == "circle"
    assert size == FORMAL_SIX_SCENARIOS["baseline_circle"][1]
    result = run_formal_scenario_episode(env_config_path, model, action_table, "baseline_circle",
                                          FORMAL_EVAL_HELDOUT_SEEDS[0], n_samples=15)
    assert result.outcome in ("success", "collision", "timeout")
    assert result.steps > 0


def test_run_formal_six_scenario_evaluation_covers_every_scenario_and_summarizes() -> None:
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    seeds = FORMAL_EVAL_HELDOUT_SEEDS[:2]
    results = run_formal_six_scenario_evaluation(env_config_path, model, action_table, episode_seeds=seeds, n_samples=15)
    assert set(results.keys()) == set(FORMAL_SIX_SCENARIOS.keys())
    for name, eps in results.items():
        assert len(eps) == len(seeds)
        summary = summarize_scenario_results(eps)
        assert abs(summary["success_rate"] + summary["collision_rate"] + summary["timeout_rate"] - 1.0) < 1e-9
        assert summary["n"] == len(seeds)

    try:
        summarize_scenario_results([])
        assert False, "expected IntentTrainError on empty results"
    except IntentTrainError:
        pass


def test_intent_train_cli_end_to_end_collect_il_rl_resume_checkpoint_ablation() -> None:
    # the FORMAL end-to-end smoke test: collect -> IL -> RL -> checkpoint
    # -> resume -> RL continues -> four-arm ablation -> six-scenario eval,
    # through the real CLI. Rewritten for the Order C2 SUBCOMMAND
    # interface (preflight/train/resume/eval-paper/ablate); the previous
    # flat-flag form no longer exists.
    import subprocess
    import sys as _sys
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        results = Path(d) / "results"
        base = [_sys.executable, "-m", "crowd_nav.bayesian_dvl.intent_train_cli"]
        pilot = ["--il-episodes", "2", "--il-passes", "3", "--seed", "97201"]

        r = subprocess.run(base + ["preflight"], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "preflight OK" in r.stdout, r.stderr

        r = subprocess.run(base + ["train", "--run-dir", str(run),
                                    "--target-online-episodes", "2", "--keep-resume"] + pilot,
                            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert "IL corpus:" in r.stdout and "IL[" in r.stdout and "RL[" in r.stdout
        # C4R.2/C4R.3 must be visible in a real run: online batches carry
        # demo rows (so ranking supervision survives IL), and the gradient
        # norm is reported per update.
        assert "demo/online=" in r.stdout, "online updates must report the demo/online mix"
        assert "|g|=" in r.stdout, "each update must report its pre-clip gradient norm"
        assert "outcome=" in r.stdout and "ROLL@" in r.stdout
        assert (run / "train.log").exists() and (run / "metrics.jsonl").exists()
        assert (run / "curves.png").exists()
        from crowd_nav.bayesian_dvl.intent_train_cli import RESUME_NAME
        assert (run / RESUME_NAME).exists() and (run / "final_ema.pth").exists()

        r = subprocess.run(base + ["resume", "--run-dir", str(run),
                                    "--target-online-episodes", "4"] + pilot,
                            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert "resumed:" in r.stdout and "online 2/" in r.stdout

        ck = str(run / "final_ema.pth")
        r = subprocess.run(base + ["ablate", "--checkpoint", ck, "--episodes", "1",
                                    "--out-dir", str(results)],
                            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        for arm in ("full", "mean", "cv", "uniform"):
            assert f"ablation arm: {arm}" in r.stdout
            assert (results / "ablation" / f"arm_{arm}" / "episodes.csv").exists()

        r = subprocess.run(base + ["eval-paper", "--checkpoint", ck, "--episodes", "1",
                                    "--out-dir", str(results)],
                            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert (results / "paper_main" / "episodes.csv").exists()
        assert (results / "paper_main" / "manifest.json").exists()


def test_intent_train_cli_no_sbk_hmm_import() -> None:
    import subprocess
    import sys as _sys
    code = (
        "import sys\n"
        "import crowd_nav.bayesian_dvl.intent_train_cli\n"
        "forbidden = ('bayesian_dvl.belief', 'bayesian_dvl.rollout', 'bayesian_dvl.world_model', "
        "'bayesian_dvl.trainer', 'bayesian_dvl.replay')\n"
        "bad = [m for m in sys.modules if any(f in m for f in forbidden)]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    result = subprocess.run([_sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


# --------------------------------------------------------------------- #
# Order C0.3: demo/online role separation in the training objective.
# --------------------------------------------------------------------- #

def test_c0_sample_roles_fail_closed() -> None:
    # plan section 3.1: online samples must NEVER carry an expert set, and
    # demo samples must always carry one. Enforced at construction so a
    # mislabeled sample can never reach the trainer.
    env_config_path = _env_config_path()
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions[0]
    assert demo.source_role == "demo"
    assert len(demo.expert_action_indices) >= 1
    assert demo.action_index in demo.expert_action_indices, (
        "the executed (single nearest) action must itself be inside the tolerance-widened expert set")

    import dataclasses
    try:
        dataclasses.replace(demo, source_role="online")  # keeps the non-empty expert set
        assert False, "expected IntentTrainError: online sample with a non-empty expert set"
    except IntentTrainError:
        pass
    try:
        dataclasses.replace(demo, expert_action_indices=())  # demo with an empty set
        assert False, "expected IntentTrainError: demo sample with an empty expert set"
    except IntentTrainError:
        pass
    try:
        dataclasses.replace(demo, source_role="teacher")
        assert False, "expected IntentTrainError on an unknown source_role"
    except IntentTrainError:
        pass


def test_c0_online_collection_never_labels_its_own_action_as_expert() -> None:
    # the BLOCKING objective bug this order exists to fix: with epsilon=1.0
    # every executed action is uniformly RANDOM. None of them may be
    # recorded as an expert demonstration.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ep = collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=1.0,
                                 explore_rng=np.random.default_rng(1))
    assert len(ep.transitions) > 0
    assert all(t.source_role == "online" for t in ep.transitions)
    assert all(t.expert_action_indices == () for t in ep.transitions)
    batch = batch_to_tensors(ep.transitions)
    assert not bool(batch.demo_mask.any()), "an epsilon=1 online episode must contain zero demo samples"


def test_c0_online_only_batch_has_exactly_zero_rank_loss() -> None:
    # plan section 3.1: "mixed batch: L_rank 只在 demo mask 上求均值;没有
    # demo 时严格为 0". Verify the online-only case is EXACTLY 0.0 (not a
    # small number), and that the total loss equals the MC loss alone, so
    # lambda_rank provably cannot influence an online-only update.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ep = collect_online_episode(env_config_path, model, action_table, "standard", 700001, epsilon=0.5,
                                 explore_rng=np.random.default_rng(3))
    batch = batch_to_tensors(ep.transitions)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    result = intent_train_step(model, opt, batch, gen, lambda_rank=1000.0)
    assert result.rank_loss == 0.0, f"online-only batch must have rank_loss exactly 0.0, got {result.rank_loss}"
    assert abs(result.loss - result.mc_loss) < 1e-12, (
        f"with no demo samples the total loss must equal the MC loss even at lambda_rank=1000, "
        f"got loss={result.loss} mc={result.mc_loss}")


def test_c0_mixed_batch_rank_loss_averages_over_demo_mask_only() -> None:
    # the core C0 semantics: in a mixed demo+online batch, L_rank must be
    # the mean over the DEMO samples only. Verify against a hand-computed
    # reference: the same demo samples alone must give the identical
    # rank_loss as the mixed batch that also contains online samples.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    demo_ts = collect_orca_episode(env_config_path, "standard", 700001).transitions[:4]
    online_ts = collect_online_episode(env_config_path, model, action_table, "standard", 700002, epsilon=1.0,
                                        explore_rng=np.random.default_rng(5)).transitions[:6]
    assert len(demo_ts) == 4 and len(online_ts) == 6

    opt = torch.optim.Adam(model.parameters(), lr=0.0)  # lr=0 so repeated calls see identical weights

    mixed = batch_to_tensors(list(demo_ts) + list(online_ts))
    assert int(mixed.demo_mask.sum()) == 4 and len(mixed.demo_mask) == 10
    r_mixed = intent_train_step(model, opt, mixed, torch.Generator().manual_seed(11), lambda_rank=1.0)

    demo_only = batch_to_tensors(list(demo_ts))
    r_demo = intent_train_step(model, opt, demo_only, torch.Generator().manual_seed(11), lambda_rank=1.0)

    assert abs(r_mixed.rank_loss - r_demo.rank_loss) < 1e-9, (
        f"mixed-batch rank_loss must equal the demo-only rank_loss (mean over the demo mask), "
        f"got mixed={r_mixed.rank_loss} demo_only={r_demo.rank_loss}")
    # and the MC losses must genuinely differ, proving the online samples
    # DID contribute to L_MC (they are not simply being dropped entirely).
    assert abs(r_mixed.mc_loss - r_demo.mc_loss) > 1e-9, (
        "online samples must still contribute to L_MC; only the ranking term excludes them")


def test_c0_rank_loss_uses_equivalence_set_not_single_nearest_action() -> None:
    # plan C0.2 / audit point 6: adjacent grid actions can be genuinely
    # equivalent to ORCA's continuous velocity; a margin loss against the
    # SINGLE nearest action wrongly penalizes them. Verify the recorded
    # expert set is the tolerance-widened class and that at least some
    # decisions really do have more than one equivalent expert action
    # (otherwise this fix would be vacuous).
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)
    tol = derive_action_equivalence_tolerance(action_table)
    assert tol > 0

    transitions = []
    for seed in (700001, 700002, 700003):
        transitions.extend(collect_orca_episode(env_config_path, "standard", seed).transitions)
    sizes = [len(t.expert_action_indices) for t in transitions]
    assert all(s >= 1 for s in sizes)
    assert max(sizes) > 1, (
        f"the tolerance-widened expert set must be strictly larger than the single nearest action for at "
        f"least some real ORCA decisions, else C0.2 changes nothing (sizes seen: {sorted(set(sizes))})")
    # every recorded expert set must be reproducible from the public
    # geometry alone (no hidden state): rebuilding it must be a subset of
    # the grid and must contain the executed action.
    for t in transitions[:20]:
        assert t.action_index in t.expert_action_indices
        assert all(0 <= i < len(action_table) for i in t.expert_action_indices)


def test_c0_checkpoint_schema_v6_rejects_retired_v5_and_wrong_training_contract() -> None:
    # plan C0.5: V5 weights were fit under the buggy objective; loading
    # them must FAIL CLOSED, since no shape check can tell them apart.
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ckpt.pth")
        save_intent_checkpoint(model, path, action_grid_hash="h", scene_registry_sha256="s")
        raw = torch.load(path, weights_only=False)
        assert raw["checkpoint_schema"] == CHECKPOINT_SCHEMA_V6
        assert raw["training_contract_schema"] == TRAINING_CONTRACT_V2_DEMO_RANK_ONLINE_MC

        model2 = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
        load_intent_checkpoint(path, model2, expected_action_grid_hash="h", expected_scene_registry_sha256="s")

        # a genuine retired-V5 payload must be rejected by name
        v5 = dict(raw)
        v5["checkpoint_schema"] = CHECKPOINT_SCHEMA_V5_RETIRED
        v5_path = str(Path(d) / "v5.pth")
        torch.save(v5, v5_path)
        try:
            load_intent_checkpoint(v5_path, model2)
            assert False, "expected IntentPolicyError: retired V5 checkpoint schema"
        except IntentPolicyError as exc:
            assert "RETIRED" in str(exc)

        # right schema, wrong training contract -> still rejected
        bad = dict(raw)
        bad["training_contract_schema"] = "bdvl_intent_training_contract_something_else"
        bad_path = str(Path(d) / "bad_contract.pth")
        torch.save(bad, bad_path)
        try:
            load_intent_checkpoint(bad_path, model2)
            assert False, "expected IntentPolicyError on training_contract_schema mismatch"
        except IntentPolicyError as exc:
            assert "training contract" in str(exc)

        # missing the field entirely -> rejected
        missing = {k: v for k, v in raw.items() if k != "training_contract_schema"}
        missing_path = str(Path(d) / "missing.pth")
        torch.save(missing, missing_path)
        try:
            load_intent_checkpoint(missing_path, model2)
            assert False, "expected IntentPolicyError on a missing training_contract_schema"
        except IntentPolicyError:
            pass


def test_c0_mixed_loss_matches_hand_computation() -> None:
    # Order C0 completion criterion: "手算 mixed loss 与生产函数一致".
    # Recompute L = L_MC + lambda*mean_over_demo(L_rank) from first
    # principles -- independently of train_step's internals -- and require
    # bit-level agreement with the production function's reported values.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)

    demo_ts = collect_orca_episode(env_config_path, "standard", 700001).transitions[:3]
    online_ts = collect_online_episode(env_config_path, model, action_table, "standard", 700002, epsilon=1.0,
                                        explore_rng=np.random.default_rng(7)).transitions[:5]
    batch = batch_to_tensors(list(demo_ts) + list(online_ts))
    B, n_taus, lam, margin = len(batch.demo_mask), 16, 0.37, 0.1

    # lr=0 so the production step cannot move the weights before we
    # recompute against them.
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    produced = intent_train_step(model, opt, batch, torch.Generator().manual_seed(99),
                                  n_taus=n_taus, ranking_margin=margin, lambda_rank=lam)

    # --- independent recomputation ---
    model.train()
    tau = torch.rand(B, n_taus, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        predicted = model(batch.robot_feats, batch.human_feats, batch.human_mask, batch.action_feats, tau)
        expected_mc = quantile_huber_loss(predicted, tau, batch.mc_returns.expand(B, 1)).mean()

        n_actions = batch.all_action_feats.shape[1]
        fixed_tau = (torch.arange(n_taus, dtype=torch.float32) + 0.5) / n_taus
        per_demo = []
        for i in range(B):
            if not bool(batch.demo_mask[i]):
                continue
            q = model(
                batch.robot_feats[i:i + 1].expand(n_actions, -1),
                batch.human_feats[i:i + 1].expand(n_actions, -1, -1),
                batch.human_mask[i:i + 1].expand(n_actions, -1),
                batch.all_action_feats[i],
                fixed_tau.unsqueeze(0).expand(n_actions, n_taus),
            ).mean(dim=1)
            experts = batch.expert_indices[i]
            expert_mask = torch.zeros(n_actions, dtype=torch.bool)
            expert_mask[list(experts)] = True
            # expert_ranking_loss's definition, written out longhand here
            # rather than called, so this really is an independent check:
            #   relu(margin + max(non-expert) - max(expert))
            hand = torch.relu(margin + q[~expert_mask].max() - q[expert_mask].max())
            per_demo.append(hand)
        assert len(per_demo) == 3, f"expected exactly the 3 demo samples, got {len(per_demo)}"
        expected_rank = torch.stack(per_demo).mean()
        expected_total = expected_mc + lam * expected_rank

    assert abs(produced.mc_loss - float(expected_mc)) < 1e-6, (produced.mc_loss, float(expected_mc))
    assert abs(produced.rank_loss - float(expected_rank)) < 1e-6, (produced.rank_loss, float(expected_rank))
    assert abs(produced.loss - float(expected_total)) < 1e-6, (produced.loss, float(expected_total))


# --------------------------------------------------------------------- #
# Order C1.2-C1.5: the 5-person junction_crowd scenario.
# --------------------------------------------------------------------- #

def test_c1_crowd_seed_blocks_frozen_and_mutually_disjoint() -> None:
    assert len(JUNCTION_CROWD_TRAIN_SEEDS) == 200
    assert len(JUNCTION_CROWD_HELDOUT_SEEDS) == 100
    blocks = {
        "junction_train": set(JUNCTION_TRAIN_SEEDS),
        "junction_heldout": set(JUNCTION_HELDOUT_SEEDS),
        "crowd_train": set(JUNCTION_CROWD_TRAIN_SEEDS),
        "crowd_heldout": set(JUNCTION_CROWD_HELDOUT_SEEDS),
        "formal_eval": set(FORMAL_EVAL_HELDOUT_SEEDS),
    }
    names = sorted(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (blocks[a] & blocks[b]), f"{a} and {b} overlap: {sorted(blocks[a] & blocks[b])[:5]}"
    for wrong in (JUNCTION_CROWD_HELDOUT_SEEDS[0], JUNCTION_TRAIN_SEEDS[0]):
        try:
            JunctionCrowdEpisodeConfig(episode_seed=wrong, is_heldout=False)
            assert False, f"expected JunctionScenarioError for seed {wrong} as crowd-train"
        except JunctionScenarioError:
            pass


def test_c1_crowd_heldout_is_a_real_distribution_shift_not_just_new_seeds() -> None:
    # plan 2.2 point 10: "所谓 held-out stress 目前只是新 seed". The crowd
    # held-out set must differ in PARAMETERS, not merely in seed integers.
    assert CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE[0] > PEDESTRIAN_SPEED_RANGE[1], (
        f"held-out pedestrian speed {CROWD_HELDOUT_PEDESTRIAN_SPEED_RANGE} must be disjoint from (above) "
        f"train {PEDESTRIAN_SPEED_RANGE}")
    assert CROWD_HELDOUT_BACKGROUND_SPEED_RANGE[0] > CROWD_TRAIN_BACKGROUND_SPEED_RANGE[1]
    assert abs(CROWD_HELDOUT_EXIT_LEFT[0]) > abs(EXIT_LEFT[0]), "held-out fork must be geometrically wider"
    assert abs(CROWD_HELDOUT_EXIT_RIGHT[0]) > abs(EXIT_RIGHT[0])
    # and the PUBLIC scene must reflect that shift (a shifted world the
    # provider still describes with the train geometry would be a bug)
    train_scene = public_junction_crowd_scene(is_heldout=False)
    heldout_scene = public_junction_crowd_scene(is_heldout=True)
    train_pos = sorted(d.position for d in train_scene.destinations)
    heldout_pos = sorted(d.position for d in heldout_scene.destinations)
    assert train_pos != heldout_pos, "held-out public geometry must differ from train's"


def test_c1_crowd_episode_has_five_humans_and_no_initial_overlap() -> None:
    # C1.5: actual human count, and no two agents (robot included) start
    # overlapping. Real regression risk: CrowdSim.reset()'s
    # multiagent_training gate silently forces human_num=1.
    env_config_path = _env_config_path()
    for heldout, seeds in ((False, JUNCTION_CROWD_TRAIN_SEEDS[:4]), (True, JUNCTION_CROWD_HELDOUT_SEEDS[:3])):
        for seed in seeds:
            cfg = JunctionCrowdEpisodeConfig(episode_seed=seed, is_heldout=heldout)
            env, robot, true_exit = build_junction_crowd_episode(env_config_path, cfg)
            assert true_exit in ("left", "right")
            assert len(env.humans) == JUNCTION_CROWD_HUMAN_NUM == 5, (
                f"expected 5 humans, got {len(env.humans)}")
            agents = [(robot.px, robot.py, robot.radius)] + [(h.px, h.py, h.radius) for h in env.humans]
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    (ax, ay, ar), (bx, by, br) = agents[i], agents[j]
                    assert np.hypot(ax - bx, ay - by) >= ar + br, (
                        f"seed {seed}: agents {i} and {j} start overlapping")
            # the ambiguous pedestrian's goal must START at the shared
            # waypoint, never already at an exit
            amb = env.humans[AMBIGUOUS_TRACK_INDEX]
            assert (amb.gx, amb.gy) == JUNCTION_WAYPOINT, (
                f"ambiguous pedestrian must start heading for the shared waypoint, got ({amb.gx}, {amb.gy})")


def test_c1_crowd_hidden_exit_identical_before_reveal_and_diverges_after() -> None:
    # C1.4, the crowd analogue of the 1-person fork-identity test: with the
    # WHOLE crowd stepping, the ambiguous pedestrian's trajectory must still
    # be bit-identical under true_exit="left" vs "right" until the reveal.
    import copy
    env_config_path = _env_config_path()
    cfg = JunctionCrowdEpisodeConfig(episode_seed=JUNCTION_CROWD_TRAIN_SEEDS[0], is_heldout=False)
    env_l, robot_l, _ = build_junction_crowd_episode(env_config_path, cfg)
    env_r, robot_r, _ = build_junction_crowd_episode(env_config_path, cfg)

    def roll(env, robot, true_exit):
        amb = env.humans[AMBIGUOUS_TRACK_INDEX]
        wp, positions, fork = False, [], None
        for step in range(70):
            wp_new = maybe_reveal_crowd_exit(amb, true_exit, wp)
            if fork is None and wp_new and not wp:
                fork = step
            wp = wp_new
            positions.append((amb.px, amb.py))
            gv = np.array([robot.gx - robot.px, robot.gy - robot.py])
            n = float(np.linalg.norm(gv))
            if n >= robot.radius:
                v = robot.v_pref * gv / n
                robot.px += float(v[0]) * FROZEN_VALUES["dt"]
                robot.py += float(v[1]) * FROZEN_VALUES["dt"]
            for h in env.humans:
                ob = [robot.get_observable_state()] + [o.get_observable_state() for o in env.humans if o is not h]
                h.step(h.act(ob))
        return positions, fork

    pos_l, fork_l = roll(env_l, robot_l, "left")
    pos_r, fork_r = roll(env_r, robot_r, "right")
    assert fork_l is not None and fork_l == fork_r, f"reveal step must be exit-independent, {fork_l} vs {fork_r}"
    for i in range(fork_l + 1):
        assert pos_l[i] == pos_r[i], (
            f"crowd ambiguous pedestrian must be bit-identical before the reveal; diverged at step {i}: "
            f"{pos_l[i]} != {pos_r[i]}")
    assert any(pos_l[i] != pos_r[i] for i in range(fork_l + 1, len(pos_l))), (
        "trajectories must diverge once the hidden exit is revealed")


def test_c1_crowd_background_pedestrians_are_not_ambiguous_and_leak_nothing() -> None:
    # C1.4 second half: background pedestrians are ordinary ORCA agents
    # with their own goals; nothing in the belief main chain may read them.
    # The tracker sees positions only -- verified structurally (the
    # candidate_fn signature admits no Human) and statically (no .gx/.gy in
    # the belief modules, covered by test_main_chain_never_reads_hidden_goal).
    env_config_path = _env_config_path()
    cfg = JunctionCrowdEpisodeConfig(episode_seed=JUNCTION_CROWD_TRAIN_SEEDS[1], is_heldout=False)
    env, robot, _true_exit = build_junction_crowd_episode(env_config_path, cfg)
    amb = env.humans[AMBIGUOUS_TRACK_INDEX]
    backgrounds = env.humans[AMBIGUOUS_TRACK_INDEX + 1:]
    assert len(backgrounds) == 4
    for bg in backgrounds:
        assert (bg.gx, bg.gy) != JUNCTION_WAYPOINT, "background pedestrians are not routed via the fork"
        assert np.isfinite(bg.gx) and np.isfinite(bg.gy)
    # the tracker is fed observable positions only, and still produces a
    # valid posterior for EVERY track including the background ones
    scene = public_junction_crowd_scene(is_heldout=False)
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
    for _ in range(3):
        bank.update({i: (h.px, h.py) for i, h in enumerate(env.humans)})
        for h in env.humans:
            h.step(h.act([robot.get_observable_state()] + [o.get_observable_state() for o in env.humans if o is not h]))
    for i in range(len(env.humans)):
        b = bank.belief_for(i)
        assert abs(float(np.sum(b)) - 1.0) < 1e-9 and np.all(b >= 0)


def test_c1_crowd_geometry_forces_conflict_while_belief_is_ambiguous() -> None:
    # C1.5 final clause, and the whole reason this scenario exists (plan
    # section 1 / 4b hard principle): a goal-blind robot must be driven into
    # genuine conflict, and that conflict must overlap the window where the
    # exit is still unresolved. Measured, not assumed.
    env_config_path = _env_config_path()
    scene = public_junction_crowd_scene(is_heldout=False)
    clearances, n_overlap = [], 0
    seeds = JUNCTION_CROWD_TRAIN_SEEDS[:8]
    for seed in seeds:
        cfg = JunctionCrowdEpisodeConfig(episode_seed=seed, is_heldout=False)
        env, robot, true_exit = build_junction_crowd_episode(env_config_path, cfg)
        amb = env.humans[AMBIGUOUS_TRACK_INDEX]
        bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
        wp, min_d, conflict, collapse = False, float("inf"), None, None
        for step in range(70):
            wp = maybe_reveal_crowd_exit(amb, true_exit, wp)
            bank.update({0: (amb.px, amb.py)})
            b = bank.belief_for(0)
            if collapse is None and min(b) < 0.15:
                collapse = step
            d = float(np.hypot(robot.px - amb.px, robot.py - amb.py)) - robot.radius - amb.radius
            min_d = min(min_d, d)
            if conflict is None and d < 0.5:
                conflict = step
            gv = np.array([robot.gx - robot.px, robot.gy - robot.py])
            n = float(np.linalg.norm(gv))
            if n < robot.radius:
                break
            v = robot.v_pref * gv / n
            robot.px += float(v[0]) * FROZEN_VALUES["dt"]
            robot.py += float(v[1]) * FROZEN_VALUES["dt"]
            for h in env.humans:
                ob = [robot.get_observable_state()] + [o.get_observable_state() for o in env.humans if o is not h]
                h.step(h.act(ob))
        clearances.append(min_d)
        if conflict is not None and (collapse is None or conflict <= collapse):
            n_overlap += 1
    n_at_risk = sum(1 for c in clearances if c < 0.3)
    assert n_at_risk >= len(seeds) * 3 // 4, (
        f"crowd geometry must force genuine conflict for a naive robot in most episodes, "
        f"got {n_at_risk}/{len(seeds)} (clearances={[round(c, 3) for c in clearances]})")
    assert n_overlap >= len(seeds) * 3 // 4, (
        f"conflict must overlap the still-ambiguous window in most episodes, got {n_overlap}/{len(seeds)}")
