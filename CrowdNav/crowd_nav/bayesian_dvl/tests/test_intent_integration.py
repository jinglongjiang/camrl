"""Order C3 tests: CrowdNav policy registration + Test5/Gazebo-style
loading, and the persistent evaluator (per-episode rows, manifest,
resume by identity, separated result kinds).
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403

import configparser
import csv
import subprocess
import sys as _sys

from crowd_nav.bayesian_dvl.intent_crowdnav_policy import IntentBDVLPolicy, IntentPolicyAdapterError
from crowd_nav.bayesian_dvl.intent_evaluate import (
    IntentEvaluateError, RESULT_KINDS, completed_identities, initial_state_hash,
    run_persistent_evaluation, summarize_csv,
)
from crowd_nav.bayesian_dvl.intent_train import FORMAL_SIX_SCENARIOS, _make_standard_env
from crowd_nav.bayesian_dvl.junction_scenario import JUNCTION_CROWD_HELDOUT_SEEDS


def _tiny_checkpoint(d: Path) -> Path:
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    p = d / "final_ema.pth"
    save_intent_checkpoint(model, str(p), action_grid_hash="h", scene_registry_sha256="s",
                            extra={"artifact_role": "final_ema"})
    return p


def _policy_config(checkpoint: Path, **over) -> configparser.RawConfigParser:
    cfg = configparser.RawConfigParser()
    cfg.add_section("intent_bdvl")
    cfg.set("intent_bdvl", "checkpoint_path", str(checkpoint))
    cfg.set("intent_bdvl", "env_config_path", str(_env_config_path()))
    cfg.set("intent_bdvl", "scene", "circle:4.0")
    for k, v in over.items():
        cfg.set("intent_bdvl", k, str(v))
    return cfg


def test_c3_intent_policy_is_registered_in_policy_factory() -> None:
    # plan 2.2 point 17: Test5 / Gazebo could not load this policy at all
    # because it was never registered.
    from crowd_nav.policy.policy_factory import policy_factory
    assert "intent_bdvl" in policy_factory
    for alias in ("intent_bdvl", "bdvl_intent", "goal_intent_bdvl"):
        assert policy_factory[alias] is IntentBDVLPolicy
    # zero-arg constructible, like every other registered policy
    assert isinstance(policy_factory["intent_bdvl"](), IntentBDVLPolicy)


def test_c3_intent_policy_declares_explicit_capability_not_class_name_guessing() -> None:
    # plan section 6: Robot.act() must route by an explicit capability
    # flag, never by whether the class name happens to contain
    # 'mamba'/'orca'/'sarl'. IntentBDVLPolicy's name contains none of them,
    # so without the flag it would silently get the WRONG input format.
    pol = IntentBDVLPolicy()
    assert pol.expects_joint_state is True
    name = IntentBDVLPolicy.__name__.lower()
    assert not any(k in name for k in ("orca", "cadrl", "sarl", "multi_human_rl", "mamba")), (
        "this policy must not rely on the legacy class-name heuristic")
    import inspect
    from crowd_sim.envs.utils import robot as robot_mod
    src = inspect.getsource(robot_mod.Robot.act)
    assert "expects_joint_state" in src, "Robot.act must consult the explicit capability flag"
    assert src.index("expects_joint_state") < src.index("policy_name"), (
        "the explicit flag must be checked BEFORE the class-name fallback")


def test_c3_intent_policy_configure_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = _tiny_checkpoint(d)
        # missing section
        empty = configparser.RawConfigParser()
        try:
            IntentBDVLPolicy().configure(empty)
            assert False, "expected IntentPolicyAdapterError on a missing [intent_bdvl] section"
        except IntentPolicyAdapterError:
            pass
        # no checkpoint -> must refuse to deploy random weights
        cfg = _policy_config(ck)
        cfg.set("intent_bdvl", "checkpoint_path", "")
        try:
            IntentBDVLPolicy().configure(cfg)
            assert False, "expected IntentPolicyAdapterError when checkpoint_path is empty"
        except IntentPolicyAdapterError:
            pass
        # unknown belief mode / scene
        try:
            IntentBDVLPolicy().configure(_policy_config(ck, belief_mode="bogus"))
            assert False, "expected IntentPolicyAdapterError on an unknown belief_mode"
        except IntentPolicyAdapterError:
            pass
        try:
            IntentBDVLPolicy().configure(_policy_config(ck, scene="hexagon:3"))
            assert False, "expected IntentPolicyAdapterError on an unknown scene kind"
        except IntentPolicyAdapterError:
            pass
        # predict before configure
        try:
            IntentBDVLPolicy().predict(None)
            assert False, "expected IntentPolicyAdapterError predicting before configure"
        except IntentPolicyAdapterError:
            pass


def test_c3_intent_policy_rejects_retired_v5_checkpoint() -> None:
    # a V5 checkpoint must not be deployable: those weights were fit under
    # the buggy online-ranking objective (C0.5).
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = _tiny_checkpoint(d)
        raw = torch.load(str(ck), map_location="cpu", weights_only=False)
        raw["checkpoint_schema"] = 'bdvl_intent_checkpoint_v5'
        v5 = d / "v5.pth"
        torch.save(raw, str(v5))
        try:
            IntentBDVLPolicy().configure(_policy_config(v5))
            assert False, "expected the retired V5 checkpoint to fail closed at deployment"
        except IntentPolicyError:
            pass


def test_c3_intent_policy_runs_a_real_episode_through_robot_act() -> None:
    # the Test5/Gazebo path: Robot.act() -> JointState -> predict() -> one
    # of the frozen 80 actions, in a real CrowdSim episode.
    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        pol = IntentBDVLPolicy()
        pol.configure(_policy_config(ck))
        env, robot = _make_standard_env(_env_config_path())
        env.case_counter["train"] = 700001
        env.reset()
        robot.set_policy(pol)
        pol.set_env(env)
        pol.set_time_step(FROZEN_VALUES["dt"])
        pol.reset_episode_stats(suite_seed=0, episode_seed=700001)
        assert len(env.humans) == 5

        indices = []
        for _ in range(12):
            action = robot.act([h.get_observable_state() for h in env.humans])
            assert hasattr(action, "vx") and hasattr(action, "vy")
            idx = pol.last_action_index
            assert 0 <= idx < len(pol.action_table)
            # the returned action must BE the table entry, not a blend
            assert abs(action.vx - pol.action_table[idx][0]) < 1e-12
            assert abs(action.vy - pol.action_table[idx][1]) < 1e-12
            indices.append(idx)
            _, _r, term, trunc, _info = env.step(action)
            if term or trunc:
                break
        assert indices


def test_c3_intent_policy_belief_mode_selects_the_arm_at_deployment() -> None:
    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        for arm in ("full", "mean", "cv", "uniform"):
            pol = IntentBDVLPolicy()
            pol.configure(_policy_config(ck, belief_mode=arm))
            assert pol.belief_mode == arm


def test_c3_intent_policy_reset_clears_belief_between_episodes() -> None:
    # a stale bank would carry one episode's posterior into the next --
    # track ids restart at 0 every episode.
    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        pol = IntentBDVLPolicy()
        pol.configure(_policy_config(ck))
        env, robot = _make_standard_env(_env_config_path())
        env.case_counter["train"] = 700001
        env.reset()
        robot.set_policy(pol); pol.set_env(env)
        pol.reset_episode_stats(suite_seed=0, episode_seed=700001)
        for _ in range(4):
            env.step(robot.act([h.get_observable_state() for h in env.humans]))
        assert len(pol.bank.active_tracks()) > 0
        pol.reset_episode_stats(suite_seed=0, episode_seed=700002)
        assert len(pol.bank.active_tracks()) == 0, "reset must clear all per-episode belief state"


def test_c3_intent_policy_no_legacy_chain_import() -> None:
    code = (
        "import sys\n"
        "import crowd_nav.bayesian_dvl.intent_crowdnav_policy\n"
        "forbidden = ('bayesian_dvl.belief', 'bayesian_dvl.rollout', 'bayesian_dvl.world_model',\n"
        "             'bayesian_dvl.trainer', 'bayesian_dvl.replay', 'bayesian_dvl.counterfactual')\n"
        "bad = [m for m in sys.modules if any(f in m for f in forbidden)]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    r = subprocess.run([_sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"


# ---------------- evaluator ----------------

def test_c3_evaluator_persists_every_episode_with_the_full_metric_set() -> None:
    # plan section 7: one row per episode, written IMMEDIATELY, with the
    # required metrics -- not aggregates printed to stdout at the end.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = _tiny_checkpoint(d)
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
        load_intent_checkpoint(str(ck), model)
        action_table = np.asarray(
            ActionGridSpec.from_env_config(str(_env_config_path())).build_action_table(), dtype=np.float64)
        jobs = [("baseline_circle", FORMAL_EVAL_HELDOUT_SEEDS[i], False) for i in range(2)]
        csv_path = run_persistent_evaluation(
            _env_config_path(), model, action_table, d / "out", "paper_main", jobs,
            n_samples=10, provenance={"checkpoint": str(ck)})
        assert csv_path.exists()
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        for r in rows:
            for field in ("outcome", "steps", "elapsed_time", "path_length", "path_ratio",
                          "min_clearance", "discomfort_frequency", "mean_speed", "smoothness",
                          "initial_state_hash", "scenario", "episode_seed"):
                assert field in r and r[field] != "", f"missing metric {field}"
            assert r["outcome"] in ("success", "collision", "timeout")
            assert len(r["initial_state_hash"]) == 64
        manifest = json.loads((csv_path.parent / "manifest.json").read_text())
        for key in ("kind", "belief_mode", "n_total_rows", "episodes_csv_sha256", "provenance",
                    "python", "torch", "argv", "cuda_available"):
            assert key in manifest, f"manifest missing {key}"
        summary = summarize_csv(csv_path)
        assert "baseline_circle" in summary and summary["baseline_circle"]["n"] == 2.0


def test_c3_evaluator_resumes_by_identity_without_duplicating_or_skipping() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = _tiny_checkpoint(d)
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
        load_intent_checkpoint(str(ck), model)
        action_table = np.asarray(
            ActionGridSpec.from_env_config(str(_env_config_path())).build_action_table(), dtype=np.float64)
        first = [("baseline_circle", FORMAL_EVAL_HELDOUT_SEEDS[i], False) for i in range(2)]
        run_persistent_evaluation(_env_config_path(), model, action_table, d / "out", "paper_main",
                                   first, n_samples=10)
        extended = first + [("baseline_circle", FORMAL_EVAL_HELDOUT_SEEDS[i], False) for i in (2, 3)]
        csv_path = run_persistent_evaluation(_env_config_path(), model, action_table, d / "out", "paper_main",
                                              extended, n_samples=10)
        manifest = json.loads((csv_path.parent / "manifest.json").read_text())
        assert manifest["n_new_episodes"] == 2, "resume must run ONLY the new episodes"
        ids = completed_identities(csv_path)
        assert len(ids) == 4, f"expected 4 distinct episodes, got {len(ids)}"
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4, "resume must not duplicate rows"


def test_c3_evaluator_keeps_result_kinds_and_ablation_arms_separate() -> None:
    # plan C3.3: paper-main / held-out stress / ablation must never share
    # a directory, and each ablation arm gets its own.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = _tiny_checkpoint(d)
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
        load_intent_checkpoint(str(ck), model)
        action_table = np.asarray(
            ActionGridSpec.from_env_config(str(_env_config_path())).build_action_table(), dtype=np.float64)
        out = d / "out"
        paper = [("baseline_circle", FORMAL_EVAL_HELDOUT_SEEDS[0], False)]
        stress = [("junction_crowd", JUNCTION_CROWD_HELDOUT_SEEDS[0], True)]
        p1 = run_persistent_evaluation(_env_config_path(), model, action_table, out, "paper_main", paper, n_samples=10)
        p2 = run_persistent_evaluation(_env_config_path(), model, action_table, out, "heldout_junction", stress, n_samples=10)
        arms = {}
        for arm in ("full", "mean", "cv", "uniform"):
            arms[arm] = run_persistent_evaluation(_env_config_path(), model, action_table, out, "ablation",
                                                   stress, belief_mode=arm, n_samples=10)
        paths = {p1, p2, *arms.values()}
        assert len(paths) == 6, f"every kind/arm needs its own file, got {sorted(str(p) for p in paths)}"
        for arm, p in arms.items():
            assert p.parent.name == f"arm_{arm}"
        assert p1.parent.name == "paper_main" and p2.parent.name == "heldout_junction"
        try:
            run_persistent_evaluation(_env_config_path(), model, action_table, out, "not_a_kind", paper)
            assert False, "expected IntentEvaluateError on an unknown result kind"
        except IntentEvaluateError:
            pass


def test_c3_ablation_arms_share_the_initial_state_but_may_diverge_after() -> None:
    # plan section 7: fairness is "same initial conditions and exogenous
    # randomness", NOT "same trajectories" -- closed-loop actions differ
    # between arms, so trajectories legitimately diverge after step 0.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = _tiny_checkpoint(d)
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
        load_intent_checkpoint(str(ck), model)
        action_table = np.asarray(
            ActionGridSpec.from_env_config(str(_env_config_path())).build_action_table(), dtype=np.float64)
        stress = [("junction_crowd", JUNCTION_CROWD_HELDOUT_SEEDS[i], True) for i in range(2)]
        hashes = {}
        for arm in ("full", "mean", "cv", "uniform"):
            p = run_persistent_evaluation(_env_config_path(), model, action_table, d / "out", "ablation",
                                           stress, belief_mode=arm, n_samples=10)
            with p.open(newline="") as fh:
                hashes[arm] = {int(r["episode_seed"]): r["initial_state_hash"] for r in csv.DictReader(fh)}
        base = hashes["full"]
        for arm in ("mean", "cv", "uniform"):
            assert hashes[arm] == base, (
                f"arm {arm} must start from the IDENTICAL initial state as full: {hashes[arm]} != {base}")


def test_c3_initial_state_hash_detects_a_changed_start() -> None:
    env, robot = _make_standard_env(_env_config_path())
    env.case_counter["train"] = 700001
    env.reset()
    h1 = initial_state_hash(robot, env.humans)
    assert len(h1) == 64
    assert initial_state_hash(robot, env.humans) == h1
    robot.px += 0.01
    assert initial_state_hash(robot, env.humans) != h1


# --------------------------------------------------------------------- #
# Order C4R.5: evaluation seam -- Test5 protocol equivalence + ROS/Gazebo
# adapter smoke.
# --------------------------------------------------------------------- #

def test_c4r_six_scenarios_match_test5_eval_envs_config_exactly() -> None:
    # plan C4R.5: the paper-main table must be comparable with Mamba-VL /
    # SARL / LSTM, which are evaluated under the project's own Test5
    # [eval_envs] protocol. Our FORMAL_SIX_SCENARIOS was written by hand;
    # this test makes the equivalence AUTOMATIC so the two cannot drift.
    import configparser
    ref_path = REPO_ROOT / "crowd_nav" / "configs" / "policy_bayesian_fullcrowd_tail.config"
    assert ref_path.exists(), f"Test5 protocol config not found: {ref_path}"
    parser = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    parser.read(str(ref_path))
    assert parser.has_section("eval_envs"), "Test5 config must define [eval_envs]"
    reference = {}
    for name, spec in parser.items("eval_envs"):
        shape, size, human_num = spec.split(":")
        reference[name] = (shape.strip(), float(size), int(human_num))
    assert reference == FORMAL_SIX_SCENARIOS, (
        f"the goal-intent evaluator's six scenarios must match Test5's [eval_envs] EXACTLY.\n"
        f"  Test5 : {reference}\n  ours  : {FORMAL_SIX_SCENARIOS}")

    # and the env we actually BUILD must realize that spec (human count,
    # shape and size), not merely name it
    for name, (shape, size, human_num) in sorted(FORMAL_SIX_SCENARIOS.items()):
        env, robot, built_shape, built_size = build_formal_scenario_env(_env_config_path(), name)
        assert built_shape == shape and built_size == size, (name, built_shape, built_size)
        env.case_counter["test"] = FORMAL_EVAL_HELDOUT_SEEDS[0]
        env.reset()
        assert len(env.humans) == human_num, (
            f"{name}: Test5 specifies {human_num} humans, env built {len(env.humans)}")


def test_c4r_ros_gazebo_adapter_smoke_without_a_crowdsim_env() -> None:
    # plan C4R.5: Gazebo adapter single-episode smoke. rclpy is not
    # installed in this environment, so this exercises the exact POLICY
    # CONTRACT the ROS nodes use (crowd_nav/crowdnav_ros.py lines ~86-217):
    #   policy_factory[name]() -> configure(cfg) -> set_device -> set_phase
    #   -> predict(JointState(FullState, [ObservableState, ...]))
    # with NO CrowdSim env attached -- a real robot has no env.global_time,
    # and that is precisely the wiring that silently breaks.
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
    from crowd_nav.policy.policy_factory import policy_factory

    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        policy = policy_factory["intent_bdvl"]()
        policy.configure(_policy_config(ck))
        policy.set_device(torch.device("cpu"))
        policy.set_phase("test")
        assert policy.env is None, "the ROS path must work with no env attached"
        policy.reset_episode_stats(suite_seed=0, episode_seed=0)

        # a moving pedestrian observed over several ROS frames
        for step in range(6):
            robot_state = FullState(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 4.0, 1.0, np.pi / 2)
            humans = [
                ObservableState(px=0.4, py=3.5 - 0.3 * step, vx=0.0, vy=-0.6, radius=0.3),
                ObservableState(px=-0.8, py=2.0 + 0.2 * step, vx=0.1, vy=0.4, radius=0.3),
            ]
            action = policy.predict(JointState(robot_state, humans))
            assert np.isfinite(action.vx) and np.isfinite(action.vy)
            speed = float(np.hypot(action.vx, action.vy))
            assert speed <= 1.0 + 1e-6, f"action must respect the 1.0 m/s grid bound, got {speed}"
            idx = policy.last_action_index
            assert 0 <= idx < len(policy.action_table)
            assert abs(action.vx - policy.action_table[idx][0]) < 1e-12
            assert abs(action.vy - policy.action_table[idx][1]) < 1e-12

        # a variable number of tracked people (detector dropouts) must not crash
        for n in (0, 1, 5, 12):
            robot_state = FullState(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 4.0, 1.0, np.pi / 2)
            humans = [ObservableState(px=0.5 * i, py=2.0, vx=0.0, vy=-0.5, radius=0.3) for i in range(n)]
            action = policy.predict(JointState(robot_state, humans))
            assert np.isfinite(action.vx) and np.isfinite(action.vy)


def test_c4r_final_ema_records_its_training_arm() -> None:
    # C4R.6: three independently trained arms each produce a final_ema.pth;
    # the artifact must say WHICH arm it is or they are indistinguishable.
    import inspect
    from crowd_nav.bayesian_dvl import intent_train_cli
    src = inspect.getsource(intent_train_cli._save_final_ema)
    assert '"training_arm"' in src, "final_ema must record its training arm"


def test_order1r_clearance_uses_the_simulator_swept_dmin() -> None:
    """Order 1R: the reported clearance must be CrowdSim's swept ``dmin``
    for the interval the action just covered.

    The bug this pins: clearance used to be a snapshot taken BEFORE the
    action (robot vs pre-step human positions), and the loop ``break``ed on
    termination, so the interval in which the collision actually happened
    was never measured. CrowdSim declares a collision exactly when its
    swept dmin goes negative, so the old metric could report "no negative
    clearance" for an episode that ended in a collision -- which is what
    made every earlier "zero negative clearance" reading meaningless.

    Constructed so the two disagree in the direction that matters: the
    humans are placed FAR away (pre-action snapshot would be large and
    positive) while the simulator reports a negative swept dmin.
    """
    from crowd_nav.bayesian_dvl.intent_evaluate import _run_one_episode, IntentEvaluateError
    from crowd_nav.bayesian_dvl.intent_train import _ScenarioEpisode
    from crowd_nav.bayesian_dvl.intent_policy import HUMAN_FEATURE_DIM_V6

    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    model.eval()
    action_table = np.asarray(
        ActionGridSpec.from_env_config(str(_env_config_path())).build_action_table(), dtype=np.float64)

    def run(patch_step):
        ep = _ScenarioEpisode(_env_config_path(), "standard", 2_600_000)
        env = ep.env
        # push every human far away so a PRE-ACTION snapshot is large
        for h in env.humans:
            h.px, h.py = 50.0, 50.0
        real_step = env.step

        def stepped(action):
            obs, reward, term, trunc, info = real_step(action)
            return patch_step(obs, reward, term, trunc, dict(info))

        env.step = stepped
        return _run_one_episode(env, ep.robot, ep.scene, model, action_table, "full", 0, n_samples=6)

    # a negative swept dmin must surface as a negative min_clearance, even
    # though the pre-action snapshot was ~70 m
    m = run(lambda o, r, te, tr, info: (o, r, True, tr, {**info, "event": "collision", "dmin": -0.037}))
    assert m.outcome == "collision"
    assert m.min_clearance < 0, f"swept penetration must be reported, got {m.min_clearance}"
    assert abs(m.min_clearance - (-0.037)) < 1e-9, m.min_clearance
    # and it must count as a discomfort step
    assert m.discomfort_frequency > 0

    # a positive swept dmin is reported as-is, NOT as the far-away snapshot
    m2 = run(lambda o, r, te, tr, info: (o, r, True, tr, {**info, "event": "reach_goal", "dmin": 0.11}))
    assert m2.outcome == "success"
    assert abs(m2.min_clearance - 0.11) < 1e-9, m2.min_clearance

    # missing dmin must FAIL CLOSED -- a silent fallback would reinstate the bug
    try:
        run(lambda o, r, te, tr, info: (o, r, True, tr, {"event": "reach_goal"}))
        assert False, "expected IntentEvaluateError when env.step() returns no dmin"
    except IntentEvaluateError:
        pass
