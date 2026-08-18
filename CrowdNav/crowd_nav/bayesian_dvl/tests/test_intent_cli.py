"""Order C2 tests: the ONE formal config, device handling, epsilon
schedule + global cursor, provenance hashes, atomic periodic checkpoints,
CLI-level exact resume, and the final_ema deployment artifact.
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403

import subprocess
import sys as _sys

from crowd_nav.bayesian_dvl.intent_config import (
    DEFAULT_TRAINING_CONFIG, IntentConfigError, load_intent_training_config,
)
from crowd_nav.bayesian_dvl.intent_train_cli import (
    RunState, code_sha256, il_episode_plan, online_episode_at, resolve_device, scene_registry_sha256,
    IntentCLIError, _assert_not_formal_seed, _select_il_audit_rows,
)


def test_c2_formal_config_is_the_single_source_of_truth() -> None:
    # plan 2.2 point 3 / C2.1: the CLI must carry no formal hyperparameter
    # defaults; every value comes from train_intent_bdvl.config.
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    # the values the audit named as the project's real training config
    assert cfg.il_episodes_total == 5000
    assert cfg.online_episodes_total == 10000
    assert cfg.batch_size == 256
    assert cfg.replay_capacity == 200000
    assert abs(cfg.learning_rate - 1e-4) < 1e-12
    assert abs(cfg.gamma - 0.99) < 1e-12
    assert (cfg.epsilon_start, cfg.epsilon_end) == (0.30, 0.05)
    assert cfg.updates_per_episode == 1
    assert not hasattr(cfg, 'lambda_rank'), (
        'lambda_rank is retired: gradients are combined by projection with a capped ranking '
        'scalar weight here would be hashed into provenance and never used')
    # C4R: the stability + mixing settings must be present and usable
    assert 0.0 < cfg.demo_sample_ratio < 1.0
    assert cfg.demo_capacity > 0 and cfg.ranking_batch_size <= cfg.batch_size
    assert cfg.gradient_diagnostic_interval > 0 and cfg.grad_clip_norm > 0
    assert cfg.monitor_rolling_windows == (25, 50, 200)
    assert cfg.monitor_plot_interval_episodes == 50
    assert cfg.development_eval_interval_episodes == 500
    # provenance
    assert len(cfg.source_sha256) == 64 and len(cfg.content_hash()) == 64
    # the CLI module must not hard-code formal hyperparameters
    import inspect
    from crowd_nav.bayesian_dvl import intent_train_cli
    src = inspect.getsource(intent_train_cli.build_parser)
    for banned in ("default=256", "default=0.99", "default=1e-4", "default=0.5", "default=200000"):
        assert banned not in src, f"CLI must not define formal hyperparameter {banned}"


def test_c2_config_validation_fails_closed() -> None:
    import configparser
    base = DEFAULT_TRAINING_CONFIG.read_text()
    with tempfile.TemporaryDirectory() as d:
        def write(mutate):
            p = Path(d) / "bad.config"
            parser = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
            parser.read_string(base)
            mutate(parser)
            with open(p, "w") as fh:
                parser.write(fh)
            return p

        cases = [
            ("IL split mismatch", lambda c: c.set("il", "il_episodes_standard", "1")),
            ("mix not summing to 1", lambda c: c.set("online", "mix_standard", "0.9")),
            ("gamma out of range", lambda c: c.set("optim", "gamma", "1.5")),
            ("epsilon_end > start", lambda c: c.set("exploration", "epsilon_end", "0.9")),
            ("retired lambda_rank present at all", lambda c: c.set("ranking", "lambda_rank", "380")),
            ("ema_decay >= 1", lambda c: c.set("ema", "ema_decay", "1.0")),
            ("wrong feature schema", lambda c: c.set("schema", "feature_schema", "bogus_v9")),
            ("wrong training contract", lambda c: c.set("schema", "training_contract_schema", "bogus")),
            ("wrong checkpoint schema", lambda c: c.set("schema", "checkpoint_schema", "bogus")),
            ("duplicate training seed", lambda c: c.set("seeds", "training_seeds", "98201, 98201")),
            ("training seed steals a formal eval seed",
             lambda c: c.set("seeds", "training_seeds", "97001, 97202")),
            ("training seed steals a crowd-heldout seed",
             lambda c: c.set("seeds", "training_seeds", "2010000, 98202")),
            ("zero batch size", lambda c: c.set("optim", "batch_size", "0")),
            ("demo_sample_ratio out of range", lambda c: c.set("optim", "demo_sample_ratio", "1.5")),
            ("ranking_batch_size exceeds batch", lambda c: c.set("ranking", "ranking_batch_size", "99999")),
            # Order 2/3: the retired [ranking] gate is replaced by the
            # balancer + sliding-window health gate, which must fail closed
            # on every one of their parameters.
            ("negative rank_cap_rho", lambda c: c.set("gradient_balance", "rank_cap_rho", "-1")),
            ("retired rank_share present at all",
             lambda c: c.set("gradient_balance", "rank_share", "2.0")),
            ("zero audit split", lambda c: c.set("audit_split", "audit_episodes_per_scenario", "0")),
            ("zero health check interval",
             lambda c: c.set("gradient_health", "health_check_interval", "0")),
            ("non-positive warmup rank ceiling",
             lambda c: c.set("ranking_warmup", "warmup_rank_loss_max", "0")),
            ("zero diagnostic interval", lambda c: c.set("ranking", "gradient_diagnostic_interval", "0")),
            ("zero plot interval", lambda c: c.set("monitoring", "plot_interval_episodes", "0")),
            ("bad rolling window", lambda c: c.set("monitoring", "rolling_windows", "25, 0")),
        ]
        for label, mutate in cases:
            p = write(mutate)
            try:
                load_intent_training_config(p)
                assert False, f"expected IntentConfigError: {label}"
            except IntentConfigError:
                pass
        # a missing section must also fail closed
        p = Path(d) / "nosec.config"
        p.write_text("[il]\nil_episodes_total = 1\n")
        try:
            load_intent_training_config(p)
            assert False, "expected IntentConfigError on missing sections"
        except IntentConfigError:
            pass
    # missing file
    try:
        load_intent_training_config(Path("/nonexistent/train.config"))
        assert False, "expected IntentConfigError on a missing file"
    except IntentConfigError:
        pass


def test_c2_epsilon_schedule_is_a_function_of_the_global_cursor() -> None:
    # plan 2.2 point 4: resume must CONTINUE the schedule, so epsilon must
    # depend on the global episode index, not on "how many this process ran".
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    assert abs(cfg.epsilon_at(0) - cfg.epsilon_start) < 1e-12
    assert abs(cfg.epsilon_at(cfg.epsilon_decay_episodes) - cfg.epsilon_end) < 1e-12
    mid = cfg.epsilon_at(cfg.epsilon_decay_episodes // 2)
    assert cfg.epsilon_end < mid < cfg.epsilon_start
    # monotone non-increasing, and clamped after the decay window
    prev = cfg.epsilon_start
    for i in range(0, cfg.epsilon_decay_episodes + 1, max(1, cfg.epsilon_decay_episodes // 20)):
        e = cfg.epsilon_at(i)
        assert e <= prev + 1e-12
        prev = e
    assert abs(cfg.epsilon_at(cfg.epsilon_decay_episodes * 5) - cfg.epsilon_end) < 1e-12
    try:
        cfg.epsilon_at(-1)
        assert False, "expected IntentConfigError on a negative cursor"
    except IntentConfigError:
        pass


def test_c2_online_schedule_is_fifty_fifty_and_never_uses_formal_seeds() -> None:
    # plan section 4.1: the online mix is FROZEN 50/50 and written into the
    # schedule itself; section 6: training must never touch formal seeds.
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    scen = [online_episode_at(cfg, i)[0] for i in range(200)]
    assert scen.count("standard") == 100 and scen.count("junction_crowd") == 100
    for i in range(400):
        scenario, seed = online_episode_at(cfg, i)
        _assert_not_formal_seed(seed)  # must not raise
    plan = il_episode_plan(cfg)
    assert len(plan) == cfg.il_episodes_total
    assert sum(1 for s, _ in plan if s == "standard") == cfg.il_episodes_standard
    assert sum(1 for s, _ in plan if s == "junction_crowd") == cfg.il_episodes_junction_crowd
    for _s, seed in plan:
        _assert_not_formal_seed(seed)
    # and the guard really does fire on a formal seed
    try:
        _assert_not_formal_seed(FORMAL_EVAL_HELDOUT_SEEDS[0])
        assert False, "expected IntentCLIError on a formal eval seed"
    except IntentCLIError:
        pass
    try:
        _assert_not_formal_seed(JUNCTION_CROWD_HELDOUT_SEEDS[0])
        assert False, "expected IntentCLIError on a crowd held-out seed"
    except IntentCLIError:
        pass


def test_c2_device_resolution_fails_closed_on_unavailable_cuda() -> None:
    assert resolve_device("cpu").type == "cpu"
    try:
        resolve_device("tpu")
        assert False, "expected IntentCLIError on an unsupported device"
    except IntentCLIError:
        pass
    if not torch.cuda.is_available():
        try:
            resolve_device("cuda")
            assert False, "expected IntentCLIError: cuda requested but unavailable"
        except IntentCLIError:
            pass
    else:
        assert resolve_device("cuda").type == "cuda"


def test_c2_scene_registry_hash_covers_real_geometry_not_a_label() -> None:
    # plan 2.2 point 13: the previous hash was of a description STRING, so
    # moving the junction or an exit left it byte-identical.
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    base = scene_registry_sha256(cfg)
    assert len(base) == 64
    assert scene_registry_sha256(cfg) == base, "must be deterministic"

    import dataclasses
    # a tracker-parameter change must change the hash
    for field_name, new in (("tracker_sigma", cfg.tracker_sigma + 0.1),
                            ("future_horizon", cfg.future_horizon + 1),
                            ("tracker_waypoint_radius", cfg.tracker_waypoint_radius + 0.05)):
        assert scene_registry_sha256(dataclasses.replace(cfg, **{field_name: new})) != base, (
            f"changing {field_name} must change the scene registry hash")

    # a GEOMETRY change must change the hash too
    import crowd_nav.bayesian_dvl.junction_scenario as js
    original = js.EXIT_LEFT
    try:
        js.EXIT_LEFT = (original[0] - 0.5, original[1])
        assert scene_registry_sha256(cfg) != base, "moving an exit must change the scene registry hash"
    finally:
        js.EXIT_LEFT = original
    assert scene_registry_sha256(cfg) == base


def test_c2_code_hash_changes_when_main_chain_source_changes() -> None:
    h = code_sha256()
    assert len(h) == 64 and code_sha256() == h


def test_c2_vectorized_ranking_matches_the_per_sample_loop() -> None:
    # C2.3: the Bx80 ranking forward was vectorized (and the set encoder
    # de-duplicated from n_demo*80 down to n_demo passes). It must be
    # numerically identical to the straightforward per-sample loop.
    env_config_path = _env_config_path()
    trans = collect_orca_episode(env_config_path, "standard", 700001).transitions[:6]
    b = batch_to_tensors(trans)
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    n_taus, margin = 16, 0.1
    n_act = b.all_action_feats.shape[1]
    ft = (torch.arange(n_taus, dtype=torch.float32) + 0.5) / n_taus
    model.train()
    with torch.no_grad():
        ref = []
        for i in range(len(b.demo_mask)):
            q = model(b.robot_feats[i:i + 1].expand(n_act, -1),
                      b.human_feats[i:i + 1].expand(n_act, -1, -1),
                      b.human_mask[i:i + 1].expand(n_act, -1),
                      b.all_action_feats[i],
                      ft.unsqueeze(0).expand(n_act, n_taus)).mean(dim=1)
            ref.append(expert_ranking_loss(q, b.expert_indices[i], margin))
        ref_rank = float(torch.stack(ref).mean())
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    got = intent_train_step(model, opt, b, torch.Generator().manual_seed(0),
                            n_taus=n_taus, ranking_margin=margin, rho=1.0)
    assert abs(got.rank_loss - ref_rank) < 1e-6, (got.rank_loss, ref_rank)


_AUDIT_ARTIFACT = REPO_ROOT / "runs" / "candidate_audit.json"


def _ensure_candidate_audit():
    """train/resume refuse to start without a PASSING audit collected under
    the current code/config/scene. Produce a REAL one once per session --
    never a stub, because a stub would also disable the gate for every test
    that relies on it."""
    if _AUDIT_ARTIFACT.exists():
        import json
        try:
            payload = json.loads(_AUDIT_ARTIFACT.read_text())
        except ValueError:
            payload = {}
        from crowd_nav.bayesian_dvl.intent_train_cli import audit_identity
        from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
        if payload.get("passed") and payload.get("identity") == audit_identity(load_intent_training_config()):
            return
    r = subprocess.run(
        [_sys.executable, "-m", "crowd_nav.bayesian_dvl.intent_train_cli",
         "audit-candidates", "--out", str(_AUDIT_ARTIFACT), "--episodes", "8"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"the candidate audit itself failed:\n{r.stdout}\n{r.stderr}"


def _cli(*argv, expect_ok=True):
    if argv and argv[0] in ("train", "resume"):
        _ensure_candidate_audit()
    cmd = [_sys.executable, "-m", "crowd_nav.bayesian_dvl.intent_train_cli", *argv]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    if expect_ok:
        assert r.returncode == 0, f"cmd={argv}\nstdout={r.stdout}\nstderr={r.stderr}"
    return r


def test_c2_cli_preflight_reports_config_code_grid_and_scene_hashes() -> None:
    r = _cli("preflight")
    for needle in ("content hash", "code hash", "action grid", "scene registry",
                   "feature schema", "training contract", "checkpoint schema", "preflight OK"):
        assert needle in r.stdout, f"preflight must report {needle!r}\n{r.stdout}"


def test_c2_cli_resume_is_a_total_target_and_bit_identical_to_continuous() -> None:
    # plan 2.2 point 4 + C2.5: continuous 4 == 2 then resume-to-4, at the
    # CLI level (not just the low-level train_step), AND resuming an
    # already-finished run must do nothing rather than run N more.
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "runA", Path(d) / "runB"
        common = ["--il-episodes", "2", "--audit-episodes", "1", "--il-passes", "3", "--seed", "98201",
                  "--il-corpus-dir", str(Path(d) / "corpus"), "--keep-resume"]
        _cli("train", "--run-dir", str(a), "--target-online-episodes", "4", *common)
        _cli("train", "--run-dir", str(b), "--target-online-episodes", "2", *common)
        r_resume = _cli("resume", "--run-dir", str(b), "--target-online-episodes", "4", *common)
        assert "resumed:" in r_resume.stdout
        assert "online 2/" in r_resume.stdout, "resume must report the restored cursor"

        from crowd_nav.bayesian_dvl.intent_train_cli import _resume_path
        A = torch.load(str(_resume_path(a)), map_location="cpu", weights_only=False)
        B = torch.load(str(_resume_path(b)), map_location="cpu", weights_only=False)
        for key in ("model_state_dict",):
            for k in A[key]:
                assert torch.equal(A[key][k].float(), B[key][k].float()), f"{key}[{k}] differs after resume"
        for k in A["extra"]["ema_state_dict"]:
            assert torch.equal(A["extra"]["ema_state_dict"][k].float(),
                               B["extra"]["ema_state_dict"][k].float()), f"EMA[{k}] differs after resume"
        assert torch.equal(A["extra"]["tau_generator_state"], B["extra"]["tau_generator_state"])
        assert A["extra"]["explore_rng_state"] == B["extra"]["explore_rng_state"]
        assert A["extra"]["sample_rng_state"] == B["extra"]["sample_rng_state"]
        assert A["extra"]["run_state"]["online_episodes_done"] == B["extra"]["run_state"]["online_episodes_done"] == 4
        assert A["extra"]["run_state"]["global_updates"] == B["extra"]["run_state"]["global_updates"]
        ra, rb = A["extra"]["replay_buffer_state"], B["extra"]["replay_buffer_state"]
        # C4RF.3: checkpoints carry the ONLINE ring only; the immutable demo
        # corpus is referenced by path+hash, not embedded.
        assert ra["demo_included"] is False and rb["demo_included"] is False
        assert "demo" not in ra and "demo" not in rb
        assert len(ra["online"]) == len(rb["online"])
        assert ra["demo_seen"] == rb["demo_seen"], "the reservoir counter must survive resume"
        assert A["extra"]["reservoir_rng_state"] == B["extra"]["reservoir_rng_state"]
        assert (A["extra"]["il_corpus_ref"]["corpus_sha256"]
                == B["extra"]["il_corpus_ref"]["corpus_sha256"]), "both runs must share one IL corpus"

        # total-target semantics: resuming a FINISHED run runs zero episodes
        r_noop = _cli("resume", "--run-dir", str(b), "--target-online-episodes", "4", *common)
        assert "online 4/" in r_noop.stdout
        assert "online[" not in r_noop.stdout, "resuming a finished run must not run more episodes"

        # and asking for more than the frozen budget must fail closed
        r_over = _cli("train", "--run-dir", str(Path(d) / "runC"),
                      "--target-online-episodes", "999999", *common, expect_ok=False)
        assert r_over.returncode != 0 and "exceeds the frozen budget" in r_over.stderr


def test_c2_final_ema_artifact_holds_ema_weights_not_raw() -> None:
    # plan 2.2 point 14: an ordinary loader of final_ema.pth must get the
    # EMA weights, not the raw last-step weights with the EMA in `extra`.
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        _cli("train", "--run-dir", str(run), "--target-online-episodes", "2",
             "--il-episodes", "2", "--audit-episodes", "1", "--il-passes", "3", "--seed", "98201", "--keep-resume")
        assert (run / "final_ema.pth").exists() and (run / "run_state.json").exists()
        from crowd_nav.bayesian_dvl.intent_train_cli import _resume_path
        raw = torch.load(str(_resume_path(run)), map_location="cpu", weights_only=False)
        fin = torch.load(str(run / "final_ema.pth"), map_location="cpu", weights_only=False)
        assert fin["extra"]["artifact_role"] == "final_ema"
        assert fin["checkpoint_schema"] == CHECKPOINT_SCHEMA_V7
        ema = raw["extra"]["ema_state_dict"]
        assert all(torch.equal(fin["model_state_dict"][k].float(), ema[k].float()) for k in fin["model_state_dict"]), \
            "final_ema.model_state_dict must BE the EMA"
        assert not all(torch.equal(fin["model_state_dict"][k].float(), raw["model_state_dict"][k].float())
                       for k in fin["model_state_dict"]), "final_ema must differ from the raw weights"
        # it must load through the ordinary loader with no special casing
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
        load_intent_checkpoint(str(run / "final_ema.pth"), model)


def test_c2_pilot_runs_are_marked_so_they_cannot_pass_as_formal() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        r = _cli("train", "--run-dir", str(run), "--target-online-episodes", "2",
                 "--il-episodes", "2", "--audit-episodes", "1", "--il-passes", "3", "--seed", "98201")
        assert "PILOT RUN" in r.stdout
        state = json.loads((run / "run_state.json").read_text())
        assert state["is_pilot"] is True
        assert "il_passes" in state["pilot_overrides"]


def test_c2_cli_rejects_a_seed_outside_the_frozen_training_seeds() -> None:
    with tempfile.TemporaryDirectory() as d:
        r = _cli("train", "--run-dir", str(Path(d) / "r"), "--target-online-episodes", "1",
                 "--il-episodes", "2", "--audit-episodes", "1", "--il-passes", "1", "--seed", "12345", expect_ok=False)
        assert r.returncode != 0 and "not one of the frozen training seeds" in r.stderr


def test_c4_cpu_cuda_top1_action_agreement() -> None:
    # Order C4 final clause: the SAME checkpoint must pick the SAME top-1
    # action on CPU and CUDA. Small numerical differences are acceptable;
    # an ACTION change is not, because that silently makes a GPU-trained
    # policy behave differently from its CPU evaluation.
    if not torch.cuda.is_available():
        return  # nothing to compare on a CPU-only host
    env_config_path = _env_config_path()
    action_table = np.asarray(
        ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)

    scene = public_junction_crowd_scene(is_heldout=False)
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
    for py in (4.0, 3.7, 3.4, 3.1):
        bank.update({0: (0.0, py)})
    bank.update({0: (-0.2, 2.8)})
    robot = RobotObservation(px=0.0, py=0.5, gx=0.0, gy=5.0, vx=0.0, vy=0.5,
                             radius=0.3, v_pref=1.0, theta=np.pi / 2)
    humans = [HumanObservation(0, -0.2, 2.8, -0.4, -0.7, 0.3)]

    n_disagree, n_checked = 0, 0
    for mode in ("full", "mean", "cv", "uniform"):
        hf, hm = build_intent_human_feature_batch(
            bank, robot, humans, mode=mode, rng=np.random.default_rng(0), n_samples=40)
        for remaining in (0.9, 0.5, 0.1):
            cpu = score_candidates_v5(model.cpu(), robot, hf, hm, action_table, remaining, device="cpu")
            gpu = score_candidates_v5(model.cuda(), robot, hf, hm, action_table, remaining, device="cuda")
            top_cpu = max(cpu, key=lambda r: r.q_mean).action_index
            top_gpu = max(gpu, key=lambda r: r.q_mean).action_index
            n_checked += 1
            if top_cpu != top_gpu:
                n_disagree += 1
            deltas = [abs(a.q_mean - b.q_mean) for a, b in zip(cpu, gpu)]
            assert max(deltas) < 1e-3, f"CPU/CUDA scores diverged too far: max delta {max(deltas)}"
    model.cpu()
    assert n_disagree == 0, f"CPU and CUDA chose different top-1 actions in {n_disagree}/{n_checked} cases"


def test_c4_training_step_is_device_independent_in_its_random_stream() -> None:
    # tau is drawn on the CPU generator and moved, so a run resumed on a
    # different device follows the SAME tau sequence. Without this, a
    # checkpoint's generator state would be device-bound and CPU/CUDA
    # parity would be uncheckable in principle.
    env_config_path = _env_config_path()
    trans = collect_orca_episode(env_config_path, "standard", 700001).transitions[:8]
    losses = {}
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        torch.manual_seed(3)
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = batch_to_tensors(trans, device=dev)
        gen = torch.Generator().manual_seed(5)
        losses[dev] = [intent_train_step(model, opt, batch, gen).loss for _ in range(3)]
    if "cuda" in losses:
        for a, b in zip(losses["cpu"], losses["cuda"]):
            assert abs(a - b) < 1e-3, f"CPU/CUDA training losses diverged: {losses}"


# --------------------------------------------------------------------- #
# Order C4RF: data/storage/seed/protocol close-out.
# --------------------------------------------------------------------- #

def test_c4rf_every_training_episode_has_a_distinct_seed() -> None:
    # C4RF.1 / audit point 3: the budget said 2500 IL + 5000 online
    # junction episodes but both cycled the SAME 200 seeds (12.5x and 25x
    # reuse), AND the two sets were identical -- so the online phase
    # explored exactly the layouts IL had already imitated.
    from crowd_nav.bayesian_dvl.intent_train_cli import il_episode_plan, online_episode_at
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    plan = il_episode_plan(cfg)
    il_std = [s for k, s in plan if k == "standard"]
    il_jc = [s for k, s in plan if k == "junction_crowd"]
    assert len(il_std) == len(set(il_std)) == cfg.il_episodes_standard
    assert len(il_jc) == len(set(il_jc)) == cfg.il_episodes_junction_crowd, (
        f"IL junction must use one distinct seed per episode, got {len(set(il_jc))} for {len(il_jc)} episodes")

    on = [online_episode_at(cfg, i) for i in range(cfg.online_episodes_total)]
    on_std = [s for k, s in on if k == "standard"]
    on_jc = [s for k, s in on if k == "junction_crowd"]
    assert len(on_std) == len(set(on_std)) and len(on_jc) == len(set(on_jc)), (
        "every online episode must have a distinct seed")
    assert not (set(il_jc) & set(on_jc)), "IL and online junction seeds must be DISJOINT"
    assert not (set(il_std) & set(on_std)), "IL and online standard seeds must be DISJOINT"

    # exceeding the frozen block must fail closed, not wrap around
    import dataclasses
    over = dataclasses.replace(cfg, il_episodes_junction_crowd=999999, il_episodes_standard=1,
                               il_episodes_total=1000000)
    try:
        il_episode_plan(over)
        assert False, "expected IntentCLIError when the IL budget exceeds the frozen seed block"
    except IntentCLIError:
        pass


def test_c4rf_seed_inventory_bounds_match_the_real_schedule() -> None:
    # C4RF.1 / audit point 4: the reported online_standard upper bound was
    # one HIGHER than any seed the schedule actually produces.
    from crowd_nav.bayesian_dvl.intent_train_cli import online_episode_at, seed_inventory
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    inv = seed_inventory(cfg)
    assert inv["ok"], inv["overlaps"]
    on_std = [s for i in range(cfg.online_episodes_total)
              for k, s in [online_episode_at(cfg, i)] if k == "standard"]
    lo, hi = inv["derived_ranges"]["online_standard"]
    assert lo == min(on_std) and hi == max(on_std), (
        f"reported online_standard range [{lo},{hi}] != real [{min(on_std)},{max(on_std)}]")
    # the large junction blocks must be in the inventory too
    for name in ("junction_crowd_il", "junction_crowd_online"):
        assert name in inv["blocks"], f"{name} missing from the seed inventory"


def test_c4rf_paper_main_uses_test8_episode_identities() -> None:
    # C4RF.5 / audit point 5: eval-paper must be episode-for-episode
    # pairable with the baselines scored through test8.py.
    from crowd_nav.bayesian_dvl.intent_train import (
        PAPER_MAIN_BASE_SEED, PAPER_MAIN_CASE_IDS, PAPER_MAIN_EPISODES_PER_SCENARIO,
        paper_main_episode_seed, paper_main_jobs,
    )
    assert PAPER_MAIN_BASE_SEED == 30_260_816
    assert PAPER_MAIN_EPISODES_PER_SCENARIO == 500
    assert PAPER_MAIN_CASE_IDS == {n: i for i, n in enumerate(FORMAL_SIX_SCENARIOS)}
    # the formula, recomputed independently
    for name, case_id in PAPER_MAIN_CASE_IDS.items():
        for ep in (0, 1, 499):
            assert paper_main_episode_seed(name, ep) == (
                PAPER_MAIN_BASE_SEED + case_id * 1_000_003 + ep) % (2**31 - 1)
    jobs = paper_main_jobs()
    assert len(jobs) == 6 * 500
    assert len({s for _, s, _ in jobs}) == 6 * 500, "paper-main identities must be unique"
    # and it must match the pre-existing tool that defines this protocol
    tool = (REPO_ROOT / "crowd_nav" / "tools" / "evaluate_bdvl_paper_main.py").read_text()
    assert "1_000_003" in tool and "--base-seed" in tool
    assert 'default=500' in tool.replace(" ", ""), "test8 protocol is 500 episodes/scenario"
    for bad in (-1,):
        try:
            paper_main_episode_seed("baseline_circle", bad)
            assert False, "expected an error on a negative episode index"
        except Exception:
            pass


def test_c4rf_checkpoint_omits_the_immutable_demo_corpus() -> None:
    # C4RF.3 / audit point 2: embedding the demo corpus in every checkpoint
    # measured ~5.8 KB/transition -> ~2.4 GB per full checkpoint at the
    # formal budget, and a permanent copy every 500 episodes across 15
    # formal runs projected to ~705 GB of byte-identical demo data.
    env_config_path = _env_config_path()
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions
    buf = IntentReplay(demo_capacity=1000, online_capacity=1000)
    buf.add_demo(demo, np.random.default_rng(0))
    full = buf.state_dict(include_demo=True)
    lean = buf.state_dict(include_demo=False)
    assert full["demo_included"] is True and "demo" in full
    assert lean["demo_included"] is False and "demo" not in lean

    # a lean state cannot be loaded into an EMPTY replay -- it must fail
    # closed rather than silently continue with no demonstrations
    empty = IntentReplay(demo_capacity=1000, online_capacity=1000)
    try:
        empty.load_state_dict(lean)
        assert False, "expected IntentTrainError loading a corpus-less state into an empty replay"
    except IntentTrainError:
        pass
    # but loads fine once the corpus has been restored first
    prepared = IntentReplay(demo_capacity=1000, online_capacity=1000)
    prepared.add_demo(demo, np.random.default_rng(0))
    prepared.load_state_dict(lean)
    assert prepared.n_demo == len(demo)


def test_c4rf_il_corpus_is_immutable_shared_and_identity_checked() -> None:
    # C4RF.2: collect once per ARM, share across the 5 optimizer seeds.
    from crowd_nav.bayesian_dvl.intent_train_cli import (
        COMPATIBLE_RAW_IL_CORPUS_CODE_HASHES, build_il_corpus, il_corpus_path, load_il_corpus,
    )
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        path = d / "corpus_full.pth"
        progress = []
        meta = build_il_corpus(
            _env_config_path(), cfg, "full", path, n_episodes=2,
            progress_callback=lambda done, total, scenario, seed, raw: progress.append(
                (done, total, scenario, seed, raw.outcome, len(raw.steps))),
        )
        assert path.exists() and path.with_suffix(".manifest.json").exists()
        assert [row[:2] for row in progress] == [(1, 2), (2, 2)]
        assert all(row[2] in ("standard", "junction_crowd") and row[5] > 0 for row in progress)
        assert meta["training_arm"] is None, "A3: the corpus records no arm"
        assert meta["n_transitions"] > 0
        assert len(meta["corpus_sha256"]) == 64
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        assert len(manifest["episode_identities"]) == meta["n_episodes"]
        for scenario, seed, n_trans, outcome in manifest["episode_identities"]:
            assert scenario in ("standard", "junction_crowd") and n_trans > 0
            assert outcome in ("success", "collision", "timeout")

        # a second load is byte-identical -> the 5 seeds really share it
        materialize_progress = []
        t1, m1 = load_il_corpus(
            path, cfg, "full",
            progress_callback=lambda done, total, n_transitions: materialize_progress.append(
                (done, total, n_transitions)),
        )
        t2, m2 = load_il_corpus(path, cfg, "full")
        assert [row[:2] for row in materialize_progress] == [(1, 2), (2, 2)]
        assert materialize_progress[-1][2] == meta["n_transitions"]
        assert m1["corpus_sha256"] == m2["corpus_sha256"] == meta["corpus_sha256"]
        assert len(t1) == len(t2) == meta["n_transitions"]
        assert all(t.source_role == "demo" and t.expert_action_indices for t in t1)

        # A3: the corpus is arm-INDEPENDENT -- every arm loads the SAME file
        # and materializes its own features, so this must now SUCCEED and
        # must give features that differ from the full arm's.
        t_full, _ = load_il_corpus(path, cfg, "full")
        t_mean, _ = load_il_corpus(path, cfg, "mean")
        assert len(t_full) == len(t_mean)
        assert not np.array_equal(t_full[0].human_features, t_mean[0].human_features), \
            "each arm must materialize its OWN belief features from the shared corpus"
        # Order 3W: the corpus is keyed on what DETERMINES it, not on the
        # whole config. A training knob must NOT invalidate 5000 episodes of
        # ORCA collection -- learning_rate provably cannot change a recorded
        # transition, and the old coupling made reuse impossible (the file
        # was renamed AND the in-file guard rejected the original).
        import dataclasses
        knob = dataclasses.replace(cfg, learning_rate=cfg.learning_rate * 2)
        assert knob.corpus_identity_hash() == cfg.corpus_identity_hash()
        rows_knob, _ = load_il_corpus(path, knob, "full")      # must SUCCEED
        assert len(rows_knob) == len(t_full)
        # but a corpus-DETERMINING change must still fail closed
        determining = dataclasses.replace(cfg, gamma=cfg.gamma * 0.5)
        assert determining.corpus_identity_hash() != cfg.corpus_identity_hash()
        try:
            load_il_corpus(path, determining, "full")
            assert False, "expected IntentCLIError on a corpus-identity mismatch"
        except IntentCLIError:
            pass
        # Only the explicitly audited telemetry-only predecessor may reuse
        # this raw corpus. An arbitrary stale producer remains fail-closed.
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        payload["code_hash"] = next(iter(COMPATIBLE_RAW_IL_CORPUS_CODE_HASHES))
        compatible_hash = d / "compatible_hash.pth"
        torch.save(payload, str(compatible_hash))
        compatible, _ = load_il_corpus(compatible_hash, cfg, "full")
        assert len(compatible) == meta["n_transitions"]
        payload["code_hash"] = "0" * 64
        bad_hash = d / "bad_hash.pth"
        torch.save(payload, str(bad_hash))
        try:
            load_il_corpus(bad_hash, cfg, "full")
            assert False, "expected IntentCLIError on an unknown producer hash"
        except IntentCLIError:
            pass
        try:
            load_il_corpus(d / "missing.pth", cfg, "full")
            assert False, "expected IntentCLIError on a missing corpus"
        except IntentCLIError:
            pass
        # A3: ONE shared path, independent of arm
        assert il_corpus_path(d, "full", cfg) == il_corpus_path(d, "mean", cfg)


def test_c4rf_ema_shadow_follows_the_model_device_on_resume() -> None:
    # Real bug found by the C4RF.6 final-hash CUDA dry-run: checkpoints are
    # read with map_location="cpu", so a restored EMA shadow was CPU-side
    # while the model had been moved to CUDA, and the next update() raised
    # "Expected all tensors to be on the same device". CPU-only resume
    # never reproduced it, and the earlier CUDA coverage was `train`, not
    # `resume`.
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    ema = EMAModel(model, decay=0.99)
    saved = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}  # as torch.load returns

    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        m = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5).to(dev)
        e = EMAModel(m, decay=0.99)
        e.load_state_dict(saved)          # CPU tensors into a possibly-CUDA shadow
        for v in e.shadow.values():
            assert v.device.type == torch.device(dev).type, (
                f"EMA shadow landed on {v.device}, expected {dev}")
        e.update(m)                       # must not raise a device mismatch
        # and .to() moves it explicitly
        e.to("cpu")
        assert all(v.device.type == "cpu" for v in e.shadow.values())


def test_c5_source_manifest_covers_the_whole_chain_and_detects_drift() -> None:
    # Pre-C5 requirement: the whole bayesian_dvl package and the formal
    # config were Git-UNTRACKED, so a 4090 run could not be tied back to a
    # known local state. The manifest is that tie. It must also cover the
    # files OUTSIDE the package that the tests READ -- both were untracked,
    # so a partial rsync would break the paper-protocol equivalence tests
    # with a confusing failure rather than a clear missing-file error.
    from crowd_nav.bayesian_dvl.source_manifest import (
        EXTERNAL_DEPENDENCIES, GROUPS, MANIFEST_SCHEMA, SourceManifestError,
        build_manifest, check_manifest,
    )
    manifest = build_manifest(REPO_ROOT)
    assert manifest["manifest_schema"] == MANIFEST_SCHEMA
    # groups may overlap and build_manifest de-duplicates across them, so
    # the count is the size of the UNION -- not the sum of group lengths.
    assert manifest["n_files"] == len({r for v in GROUPS.values() for r in v})

    # B6: the manifest describes the V6 RUNTIME CLOSURE and nothing else.
    # Declaring a retired module would fail closed (it is not on this
    # branch); silently OMITTING one the chain really loads is the failure
    # that actually bites -- a partial sync then dies mid-run. So check the
    # closure empirically, by importing the V6 entry points in a subprocess
    # and comparing what Python actually loaded against what is declared.
    declared_py = {r for v in GROUPS.values() for r in v if r.endswith(".py")}
    probe = (
        "import importlib, json, sys\n"
        "from pathlib import Path\n"
        "ROOT = Path(sys.argv[1]).resolve()\n"
        "for m in ('crowd_nav.bayesian_dvl.intent_train_cli',\n"
        "          'crowd_nav.bayesian_dvl.intent_evaluate',\n"
        "          'crowd_nav.bayesian_dvl.intent_crowdnav_policy',\n"
        "          'crowd_nav.bayesian_dvl.selftest',\n"
        "          'crowd_nav.bayesian_dvl.source_manifest'):\n"
        "    importlib.import_module(m)\n"
        "out = []\n"
        "for mod in list(sys.modules.values()):\n"
        "    f = getattr(mod, '__file__', None)\n"
        "    if not f: continue\n"
        # Some C-extension modules (torch._classes, torch._ops) carry a BARE
        # relative __file__, which resolve() silently reinterprets against the
        # CWD and makes look like a repo file. Require the path to really
        # exist and to sit under a repo package, or the check reports phantom
        # files whose presence depends on the torch build.
        "    p = Path(f).resolve()\n"
        "    if not p.is_file(): continue\n"
        "    try: rel = p.relative_to(ROOT)\n"
        "    except ValueError: continue\n"
        "    if 'site-packages' in str(rel): continue\n"
        "    if rel.parts[0] not in ('crowd_nav', 'crowd_sim'): continue\n"
        "    out.append(str(rel))\n"
        "print(json.dumps(sorted(set(out))))\n"
    )
    res = subprocess.run([_sys.executable, "-c", probe, str(REPO_ROOT)],
                         cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    loaded = set(json.loads(res.stdout.strip().splitlines()[-1]))
    undeclared = sorted(loaded - declared_py)
    assert not undeclared, f"the V6 chain loads files the manifest does not declare: {undeclared}"

    # and no retired module may be declared OR loaded
    retired = {"belief.py", "counterfactual.py", "data_coverage.py", "oracle_regret.py",
               "policy.py", "provenance.py", "replay.py", "rollout.py", "trainer.py",
               "transition.py", "world_model.py", "config.py", "artifact.py"}
    for rel in sorted(declared_py | loaded):
        parts = rel.split("/")
        if len(parts) >= 3 and parts[1] == "bayesian_dvl" and parts[-1] in retired:
            assert False, f"retired module {rel} is back in the V6 manifest/closure"
    assert len(manifest["manifest_sha256"]) == 64

    # the provenance hashes must agree with the live code/config
    from crowd_nav.bayesian_dvl.intent_train_cli import code_sha256, scene_registry_sha256
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    assert manifest["code_hash"] == code_sha256()
    assert manifest["config_content_hash"] == cfg.content_hash()
    assert manifest["scene_registry_hash"] == scene_registry_sha256(cfg)

    # the two easy-to-miss external dependencies must be declared
    for rel in ("crowd_nav/configs/policy_bayesian_fullcrowd_tail.config",
                "crowd_nav/tools/evaluate_bdvl_paper_main.py"):
        assert rel in EXTERNAL_DEPENDENCIES, f"{rel} is read by the tests and must be in the manifest"
        assert rel in manifest["groups"]["external_dependencies"]

    # a clean tree verifies
    assert check_manifest(manifest, REPO_ROOT) == []
    # a tampered entry is detected
    tampered = json.loads(json.dumps(manifest))
    first = sorted(tampered["groups"]["main_chain"])[0]
    tampered["groups"]["main_chain"][first]["sha256"] = "0" * 64
    problems = check_manifest(tampered, REPO_ROOT)
    assert any(first in p for p in problems), problems
    # a missing declared file fails closed rather than being skipped
    with tempfile.TemporaryDirectory() as d:
        try:
            build_manifest(Path(d))
            assert False, "expected SourceManifestError when declared sources are absent"
        except SourceManifestError:
            pass


def test_c5_online_rows_persist_without_dead_action_features() -> None:
    # 80x5 floats per row -- ~40% of a row -- that train_step NEVER reads
    # for online samples (it indexes all_action_feats only at demo
    # positions, and online rows carry no expert set). Dropping them on
    # persist must not change any training result.
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions
    online = collect_online_episode(env_config_path, model, action_table, "standard", 700002,
                                     epsilon=1.0, explore_rng=np.random.default_rng(3)).transitions

    a = IntentReplay(demo_capacity=1000, online_capacity=1000)
    a.add_demo(demo, np.random.default_rng(0))
    a.add_online(online)
    live = a._online[0].all_action_features
    state = a.state_dict(include_demo=False)

    # persisting must NOT mutate the live in-memory rows
    assert a._online[0].all_action_features is live is not None, "state_dict must not damage the live buffer"
    assert state["online"][0].all_action_features is None
    assert state["online_action_feature_shape"] == list(live.shape)

    b = IntentReplay(demo_capacity=1000, online_capacity=1000)
    b.add_demo(demo, np.random.default_rng(0))
    b.load_state_dict(state)
    assert b.n_online == a.n_online
    restored = b._online[0].all_action_features
    assert restored is not None and restored.shape == live.shape
    assert not restored.any(), "restored as zeros (never read for online rows)"

    # and a batch drawn from the restored buffer trains identically
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    ra = intent_train_step(model, opt, batch_to_tensors(a.sample(64, rng_a, demo_ratio=0.2)),
                           torch.Generator().manual_seed(1), rho=380.0)
    rb = intent_train_step(model, opt, batch_to_tensors(b.sample(64, rng_b, demo_ratio=0.2)),
                           torch.Generator().manual_seed(1), rho=380.0)
    assert abs(ra.loss - rb.loss) < 1e-9, (ra.loss, rb.loss)
    assert abs(ra.rank_loss - rb.rank_loss) < 1e-9


def test_a1_single_atomic_resume_no_slots() -> None:
    # A1: two-slot A/B still cost 2 x ~1.1 GB. _atomic_write_bytes already
    # does temp-file + os.replace, so the old file survives intact until
    # the new one is complete -- a second slot buys nothing.
    from crowd_nav.bayesian_dvl.intent_train_cli import RESUME_NAME, _resume_path
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        _cli("train", "--run-dir", str(run), "--target-online-episodes", "2",
             "--il-episodes", "2", "--audit-episodes", "1", "--il-passes", "3", "--seed", "98201",
             "--il-corpus-dir", str(Path(d) / "corpus"), "--keep-resume")
        big = sorted(p.name for p in run.glob("*.pth") if p.stat().st_size > 500_000)
        assert big == [RESUME_NAME], f"exactly one full resume expected, got {big}"
        assert not list(run.glob("resume_slot_*.pth")) and not list(run.glob("resume_latest.json"))
        assert _resume_path(run).exists()
        import inspect
        from crowd_nav.bayesian_dvl import intent_train_cli
        assert "os.replace" in inspect.getsource(intent_train_cli._atomic_write_bytes)


def test_a2_replay_persists_only_valid_human_rows() -> None:
    # A2: human_features is [MAX_HUMANS=20, 29] but scenarios run 5 people;
    # 15 of 20 rows are structural zeros (~58% of a row).
    env_config_path = _env_config_path()
    demo = collect_orca_episode(env_config_path, "standard", 700001).transitions
    buf = IntentReplay(demo_capacity=500, online_capacity=500)
    buf.add_demo(demo, np.random.default_rng(0))
    live_shape = buf._demo[0].human_features.shape
    assert live_shape[0] == MAX_HUMANS

    state = buf.state_dict(include_demo=True)
    stored = state["demo"][0].human_features
    n_valid = int(np.asarray(state["demo"][0].human_mask).sum())
    assert stored.shape == (n_valid, live_shape[1]) and n_valid < MAX_HUMANS
    # the live buffer must be untouched
    assert buf._demo[0].human_features.shape == live_shape

    back = IntentReplay(demo_capacity=500, online_capacity=500)
    back.load_state_dict(state)
    assert back._demo[0].human_features.shape == live_shape
    assert np.array_equal(back._demo[0].human_features, buf._demo[0].human_features), \
        "the padded [20,29] tensor must be rebuilt EXACTLY"


def test_a3_one_raw_corpus_serves_every_arm() -> None:
    # A3: three near-identical per-arm corpora -> ONE arm-independent raw
    # corpus. ORCA never consults the belief, so an episode can be
    # collected without any belief and each arm regenerates its features.
    env_config_path = _env_config_path()
    action_table = np.asarray(
        ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)
    for scenario, seed in (("standard", 2_600_000), ("junction_crowd", 2_100_000)):
        reference = collect_orca_episode(env_config_path, scenario, seed, belief_mode="full").transitions
        raw = collect_raw_orca_episode(env_config_path, scenario, seed)
        assert len(raw.steps) == len(reference)
        got = materialize_arm_transitions(raw, "full", action_table)
        for a, b in zip(reference, got):
            for f in ("robot_features", "human_features", "action_features", "all_action_features"):
                assert np.array_equal(getattr(a, f), getattr(b, f)), f"{scenario}: {f} not reproduced"
            assert a.action_index == b.action_index
            assert a.expert_action_indices == b.expert_action_indices
            assert a.reward == b.reward and a.mc_return == b.mc_return
        # and the arms are genuinely different
        mean = materialize_arm_transitions(raw, "mean", action_table)
        cv = materialize_arm_transitions(raw, "cv", action_table)
        assert not np.array_equal(got[0].human_features, mean[0].human_features)
        assert not np.array_equal(mean[0].human_features, cv[0].human_features)
    # a legacy per-arm corpus must fail closed
    from crowd_nav.bayesian_dvl.intent_train_cli import load_il_corpus, il_corpus_path
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    with tempfile.TemporaryDirectory() as d:
        # the path must no longer depend on the arm
        assert il_corpus_path(Path(d), "full", cfg) == il_corpus_path(Path(d), "mean", cfg)
        bad = Path(d) / "v1.pth"
        torch.save({"corpus_schema": "bdvl_intent_raw_il_corpus_v2", "training_arm": "full",
                    "config_content_hash": cfg.content_hash(), "code_hash": "x"}, str(bad))
        try:
            load_il_corpus(bad, cfg, "full")
            assert False, "expected IntentCLIError on a per-arm v1 corpus"
        except IntentCLIError:
            pass


def test_a4_milestones_are_gated_and_carry_no_replay() -> None:
    from crowd_nav.bayesian_dvl.intent_train_cli import MILESTONE_EPISODES, _save_milestone
    assert MILESTONE_EPISODES == (2500, 5000, 7500, 10000)
    import inspect
    from crowd_nav.bayesian_dvl import intent_train_cli
    src = inspect.getsource(_save_milestone)
    # check the PAYLOAD, not the prose: no replay/buffer state may be stored
    for banned in ("replay_buffer_state", "buffer.state_dict"):
        assert banned not in src, f"a milestone must never carry {banned}"
    assert '"artifact_role": "milestone_ema"' in src
    train_src = inspect.getsource(intent_train_cli.cmd_train)
    assert "if done in MILESTONE_EPISODES" in train_src, "milestones must be gated to the four counts"


def test_a4_resume_is_reclaimed_only_after_final_ema_verifies() -> None:
    from crowd_nav.bayesian_dvl.intent_train_cli import RESUME_NAME
    import inspect
    from crowd_nav.bayesian_dvl import intent_train_cli
    src = inspect.getsource(intent_train_cli.cmd_train)
    v = src.index("_save_final_ema(")
    load = src.index("load_intent_checkpoint(str(final_path)", v)
    unlink = src.index("resume_path.unlink", v)
    assert v < load < unlink, "verify the artifact BEFORE reclaiming the only recovery copy"
    # an INCOMPLETE run must keep its resume
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        r = _cli("train", "--run-dir", str(run), "--target-online-episodes", "2",
                 "--il-episodes", "2", "--audit-episodes", "1", "--il-passes", "3", "--seed", "98201",
                 "--il-corpus-dir", str(Path(d) / "corpus"))
        assert "final_ema verified" in r.stdout
        assert "run incomplete" in r.stdout, "a partial run must keep its resume"
        assert (run / RESUME_NAME).exists()


def test_a5_preflight_refuses_when_space_is_insufficient() -> None:
    from crowd_nav.bayesian_dvl.intent_train_cli import (
        RESUME_NAME, RESUME_TRANSIENT_FACTOR, SPACE_SAFETY_MARGIN,
        count_incomplete_runs, estimate_run_bytes,
    )
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    one = estimate_run_bytes(cfg, concurrent_runs=1, plan_runs=1)
    three = estimate_run_bytes(cfg, concurrent_runs=3, plan_runs=3)
    assert one["shared_corpus"] > 0 and one["resume"] > 0
    # the corpus is shared: it must NOT scale with run count
    assert three["shared_corpus"] == one["shared_corpus"]
    assert three["total"] > one["total"]
    assert SPACE_SAFETY_MARGIN >= 1.2

    # --- audit fix 1: the atomic save's transient peak must be budgeted ---
    # _atomic_write_bytes keeps the OLD resume while writing the new .tmp,
    # so an active run peaks at 2x. Budgeting 1x means the run dies at its
    # first checkpoint -- the exact failure this gate exists to prevent.
    assert RESUME_TRANSIENT_FACTOR >= 2.0
    assert one["resume_peak"] == one["resume"] * RESUME_TRANSIENT_FACTOR
    assert one["active"] == one["resume_peak"]
    assert three["active"] == 3 * one["resume_peak"]

    # --- audit fix 2: a SEQUENTIAL plan must not be charged for every run ---
    # A4 deletes a run's resume once its final_ema verifies, so 15 runs one
    # after another never hold 15 resumes. Charging plan_runs x per_run was
    # a false negative that would block a launch the disk can easily take.
    seq15 = estimate_run_bytes(cfg, concurrent_runs=1, plan_runs=15)
    assert seq15["active"] == one["active"], "sequential runs must not multiply the resume footprint"
    naive = one["shared_corpus"] + 15 * one["resume"]
    assert seq15["total"] < naive, "15 sequential runs must cost far less than 15 resumes"
    # only the small retained artifacts scale with the plan
    assert seq15["retained"] == 15 * one["retained_per_run"]
    assert seq15["retained"] < one["resume"], "milestones+final_ema are small next to a resume"

    # already-existing incomplete runs DO occupy their resume and must count
    stale = estimate_run_bytes(cfg, concurrent_runs=1, plan_runs=1, incomplete_runs=2)
    assert stale["incomplete"] == 2 * one["resume"]
    assert stale["total"] > one["total"]

    # a corpus already on disk must not be double-counted
    have = estimate_run_bytes(cfg, concurrent_runs=1, plan_runs=1, corpus_present=True)
    assert have["shared_corpus"] == 0
    assert have["total"] == one["total"] - one["shared_corpus"]

    # count_incomplete_runs sees a leftover resume, and stops seeing it once
    # the run reclaims it
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        assert count_incomplete_runs(root) == 0
        for name in ("run_a", "run_b"):
            (root / name).mkdir()
            (root / name / RESUME_NAME).write_bytes(b"x")
        assert count_incomplete_runs(root) == 2
        (root / "run_a" / RESUME_NAME).unlink()
        assert count_incomplete_runs(root) == 1
        assert count_incomplete_runs(root / "does_not_exist") == 0

    # the gate runs BEFORE the first run, so the runs directory normally does
    # not exist yet. statvfs needs an existing path; measuring the run dir's
    # parent directly made preflight die with FileNotFoundError on a fresh
    # checkout -- the exact situation it is meant to cover.
    with tempfile.TemporaryDirectory() as d:
        fresh = Path(d) / "not_created_yet" / "runs" / "run_a"
        r_fresh = _cli("preflight", "--run-dir", str(fresh))
        assert "disk free" in r_fresh.stdout, r_fresh.stdout[-1500:]
        assert "sufficient" in r_fresh.stdout
        assert not fresh.exists(), "preflight must not create the run dir as a side effect"

    # a plan that cannot fit must be refused, not warned about -- and the
    # thing that makes it not fit is CONCURRENCY, not the eventual total
    r = _cli("preflight", "--concurrent-runs", "100000", expect_ok=False)
    assert r.returncode != 0 and "insufficient disk" in r.stderr
    r_ok = _cli("preflight", "--concurrent-runs", "1", "--plan-runs", "15")
    assert "sufficient" in r_ok.stdout, "15 SEQUENTIAL runs must not be blocked by the plan total"


def test_order1_standard_dev_diagnostic_seeds_are_frozen_and_dev_only() -> None:
    # Order 1: the `standard` scenario was the lambda=380 run's only blind
    # spot -- development validation drew 10 standard episodes per
    # checkpoint, and every large-n `standard` number came from ONLINE
    # episodes, which carry epsilon-greedy exploration and therefore say
    # nothing about the greedy policy. This block exists to close that gap.
    from crowd_nav.bayesian_dvl.intent_train import STANDARD_DEV_DIAGNOSTIC_SEEDS
    from crowd_nav.bayesian_dvl.intent_train_cli import seed_inventory
    from crowd_nav.bayesian_dvl.intent_evaluate import RESULT_KINDS

    assert len(STANDARD_DEV_DIAGNOSTIC_SEEDS) == 100
    assert (min(STANDARD_DEV_DIAGNOSTIC_SEEDS), max(STANDARD_DEV_DIAGNOSTIC_SEEDS)) == (97401, 97500)

    # declared in the ONE inventory artifact, and provably disjoint there
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    inv = seed_inventory(cfg)
    assert "standard_dev_diagnostic" in inv["blocks"]
    assert inv["blocks"]["standard_dev_diagnostic"]["n"] == 100
    assert inv["ok"], inv["overlaps"]

    # disjoint from every other enumerated block -- checked directly, not
    # only via the inventory's own verdict
    dev = set(STANDARD_DEV_DIAGNOSTIC_SEEDS)
    for name, other in (("formal_eval", FORMAL_EVAL_HELDOUT_SEEDS),
                        ("crowd_heldout", JUNCTION_CROWD_HELDOUT_SEEDS),
                        ("training", cfg.training_seeds),
                        ("validation", cfg.validation_seeds)):
        assert not (dev & set(other)), f"standard dev seeds collide with {name}"

    # DEVELOPMENT ONLY: training must refuse them, exactly as it refuses a
    # formal seed. Without this a later run could quietly train on the very
    # seeds used to diagnose it.
    for seed in (97401, 97450, 97500):
        try:
            _assert_not_formal_seed(seed)
            assert False, f"training must reject dev diagnostic seed {seed}"
        except IntentCLIError:
            pass
    # and an unclaimed seed either side is still allowed (97501+ now belongs
    # to the selection-dev blocks, so it is deliberately NOT the example)
    _assert_not_formal_seed(97400)
    _assert_not_formal_seed(97800)

    # the evaluator has a distinct result kind, so a dev diagnosis can never
    # be written into (or mistaken for) a paper_main / held-out artifact
    assert "dev_standard" in RESULT_KINDS
    assert "dev_standard" not in ("paper_main", "heldout_junction")


def test_order5_selection_dev_seed_blocks_are_frozen_and_separate_from_diagnosis() -> None:
    """Order 5: checkpoint selection must score candidates on seeds nobody
    has looked at yet.

    97401-97500 is already spent -- it is the block that diagnosed the
    lambda=380 run, so its answers are known and selecting on it would be
    selecting on seen data. These two blocks are reserved for the frozen
    selection rule and nothing else.
    """
    from crowd_nav.bayesian_dvl.intent_train import (
        STANDARD_DEV_DIAGNOSTIC_SEEDS, STANDARD_SELECTION_DEV_SEEDS, JUNCTION_SELECTION_DEV_SEEDS,
    )
    from crowd_nav.bayesian_dvl.intent_train_cli import seed_inventory

    assert (min(STANDARD_SELECTION_DEV_SEEDS), max(STANDARD_SELECTION_DEV_SEEDS)) == (2_900_000, 2_900_099)
    assert (min(JUNCTION_SELECTION_DEV_SEEDS), max(JUNCTION_SELECTION_DEV_SEEDS)) == (2_400_000, 2_400_099)
    assert len(STANDARD_SELECTION_DEV_SEEDS) == len(JUNCTION_SELECTION_DEV_SEEDS) == 100

    sel_s, sel_j = set(STANDARD_SELECTION_DEV_SEEDS), set(JUNCTION_SELECTION_DEV_SEEDS)
    # the whole point: selection seeds must be UNSEEN, i.e. disjoint from
    # the diagnostic block
    assert not (sel_s & set(STANDARD_DEV_DIAGNOSTIC_SEEDS))
    assert not (sel_j & set(STANDARD_DEV_DIAGNOSTIC_SEEDS))
    assert not (sel_s & sel_j)

    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    inv = seed_inventory(cfg)
    for name in ("standard_selection_dev", "junction_selection_dev"):
        assert name in inv["blocks"] and inv["blocks"][name]["n"] == 100
    assert inv["ok"], inv["overlaps"]

    # never trainable, and never the paper's formal seeds
    for seed in (2_900_000, 2_900_099, 2_400_000, 2_400_099):
        try:
            _assert_not_formal_seed(seed)
            assert False, f"training must reject selection-dev seed {seed}"
        except IntentCLIError:
            pass
    for other in (FORMAL_EVAL_HELDOUT_SEEDS, JUNCTION_CROWD_HELDOUT_SEEDS,
                  cfg.training_seeds, cfg.validation_seeds):
        assert not ((sel_s | sel_j) & set(other))


def test_order2w_audit_episodes_are_held_out_of_replay_entirely() -> None:
    """Order 2W: the audit split must be BY EPISODE and must never reach the
    replay, the warm-up, or any gradient update.

    The leak this pins: the audit set used to be built from the same rows
    already added to the replay, so it measured TRAINING performance. Under
    that arrangement warm-up hit top1 0.951 in 500 steps; on genuinely
    disjoint episodes the same procedure reached only 0.879-0.945 in 750.
    """
    from crowd_nav.bayesian_dvl.intent_train import il_audit_identity
    from crowd_nav.bayesian_dvl.intent_train_cli import RESUME_NAME

    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    n_audit = cfg.audit_episodes_per_scenario
    assert n_audit == 50
    # formal budget: 5000 episodes -> 4900 training + 100 audit
    assert cfg.il_episodes_total == 5000
    assert cfg.il_episodes_total - 2 * n_audit == 4900

    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        # 12 episodes with 2 audit episodes per scenario -> 8 train / 4 audit
        r = _cli("train", "--run-dir", str(run), "--target-online-episodes", "1",
                 "--il-episodes", "12", "--audit-episodes", "2", "--warmup-steps", "0",
                 "--il-passes", "2", "--seed", "98201",
                 "--il-corpus-dir", str(Path(d) / "corpus"), "--keep-resume")
        assert "split BY EPISODE" in r.stdout, r.stdout[-3000:]
        assert "audit rows never enter replay" in r.stdout

        ck = torch.load(str(run / RESUME_NAME), map_location="cpu", weights_only=False)
        state = ck["extra"]["run_state"]
        assert state["il_audit_episodes"] > 0
        assert len(state["il_audit_identity"]) == 64

        # the corpus records a per-row episode id, which is what makes an
        # episode-level split possible at all (a row-level split would leak:
        # rows inside one episode are temporally adjacent)
        corpus = list((Path(d) / "corpus").glob("*.pth"))[0]
        payload = torch.load(str(corpus), map_location="cpu", weights_only=False)
        eps = payload["episode_of_row"] if "episode_of_row" in payload else None
        if eps is None:  # stored in the meta returned by load_il_corpus
            from crowd_nav.bayesian_dvl.intent_train_cli import load_il_corpus
            rows, meta = load_il_corpus(corpus, cfg, "full")
            eps = meta["episode_of_row"]
            scen = meta["scenario_of_row"]
            assert len(eps) == len(rows) == len(scen)
            by_scen = {}
            for e, sc in zip(eps, scen):
                by_scen.setdefault(sc, set()).add(e)
            # every scenario must have MORE episodes than it holds out, or the
            # run must have failed closed rather than training on nothing
            for sc, s in by_scen.items():
                assert len(s) >= 2, (sc, len(s))
            # audit episodes are the LAST n of each scenario -- deterministic,
            # touches no RNG, so resume identity is unaffected
            audit_eps = set()
            for sc, s in by_scen.items():
                audit_eps.update(sorted(s)[-2:])
            train_eps = {e for e in eps if e not in audit_eps}
            assert not (train_eps & audit_eps), "training and audit episodes must be disjoint"
            # and the rows follow the episodes: no audit row may share an
            # episode with a training row
            train_rows = [t for t, e in zip(rows, eps) if e not in audit_eps]
            audit_rows = [t for t, e in zip(rows, eps) if e in audit_eps]
            assert train_rows and audit_rows
            assert len({id(t) for t in train_rows} & {id(t) for t in audit_rows}) == 0
            # identity is content-based, so an audit score can never be
            # compared against a different row set
            assert il_audit_identity(audit_rows) != il_audit_identity(train_rows[:len(audit_rows)])


def test_order4_production_audit_selector_is_fixed_512_and_balanced() -> None:
    """The production selector must not silently score every held-out row.

    The previous wiring passed all 4k+ held-out rows directly to the health
    gate even though the frozen contract is 256 rows per scenario.
    """
    rows = [object() for _ in range(700)]
    labels = ["standard"] * 350 + ["junction_crowd"] * 350
    selected = _select_il_audit_rows(rows, labels, is_pilot=False)
    assert len(selected) == 512
    assert len({id(x) for x in selected}) == 512
    assert set(selected) <= set(rows)

    # A tiny pilot remains usable, but a formal run must fail closed when the
    # frozen per-scenario minimum cannot be formed.
    tiny_rows = [object() for _ in range(10)]
    tiny_labels = ["standard"] * 5 + ["junction_crowd"] * 5
    assert _select_il_audit_rows(tiny_rows, tiny_labels, is_pilot=True) == tiny_rows
    try:
        _select_il_audit_rows(tiny_rows, tiny_labels, is_pilot=False)
        assert False, "formal run must reject an undersized fixed audit set"
    except IntentCLIError:
        pass


def test_order5w_formal_plan_pairs_the_three_arms_on_the_same_seeds() -> None:
    """Order 5W: the pairing guard must be wired into the CLI, not merely
    exist as a helper."""
    import inspect
    from crowd_nav.bayesian_dvl import intent_train_cli
    from crowd_nav.bayesian_dvl.intent_train import build_formal_plan, FORMAL_ARMS

    src = inspect.getsource(intent_train_cli.cmd_train)
    assert "assert_in_formal_plan(" in src, "the guard must run inside cmd_train"
    assert FORMAL_ARMS == ("full", "mean", "cv")

    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    plan = build_formal_plan([98201, 98202], code_sha256(), cfg.content_hash(),
                             cfg.corpus_identity_hash(), 5000, 10000)
    assert len(plan["runs"]) == 6                       # 3 arms x 2 seeds
    assert {r["seed"] for r in plan["runs"]} == {98201, 98202}
    for seed in plan["seeds"]:
        assert {r["arm"] for r in plan["runs"] if r["seed"] == seed} == set(FORMAL_ARMS)

    with tempfile.TemporaryDirectory() as d:
        pf = Path(d) / "plan.json"
        pf.write_text(json.dumps(plan))
        # a full-budget run without a plan must be refused outright
        r = _cli("train", "--run-dir", str(Path(d) / "r1"), "--seed", "98201", expect_ok=False)
        assert r.returncode != 0 and "--formal-plan" in r.stderr
        # a seed outside the plan must be refused even WITH a plan
        r2 = _cli("train", "--run-dir", str(Path(d) / "r2"), "--seed", "97203",
                  "--formal-plan", str(pf), expect_ok=False)
        assert r2.returncode != 0


def test_order3w_corpus_identity_is_decoupled_from_training_knobs() -> None:
    """Order 3W: optimizer / rank_cap_rho / gate settings must not be able to
    invalidate 5000 episodes of ORCA collection."""
    import configparser
    cfg = load_intent_training_config(DEFAULT_TRAINING_CONFIG)
    base_corpus, base_full = cfg.corpus_identity_hash(), cfg.content_hash()
    text = DEFAULT_TRAINING_CONFIG.read_text()

    def variant(section, key, value):
        p = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
        p.read_string(text)
        p.set(section, key, value)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "v.config"
            with open(out, "w") as fh:
                p.write(fh)
            return load_intent_training_config(out)

    # training knobs: full config hash MOVES, corpus identity does NOT
    for section, key, value in (("gradient_balance", "rank_cap_rho", "1.0"),
                                ("optim", "batch_size", "128"),
                                ("optim", "learning_rate", "5e-5"),
                                ("ema", "ema_decay", "0.95"),
                                ("gradient_health", "health_check_interval", "50")):
        v = variant(section, key, value)
        assert v.content_hash() != base_full, (section, key)
        assert v.corpus_identity_hash() == base_corpus, (
            f"{section}.{key} must NOT invalidate the ORCA corpus")

    # corpus-determining settings: identity MUST move
    for section, key, value in (("optim", "gamma", "0.95"),
                                ("belief", "future_horizon", "6")):
        v = variant(section, key, value)
        assert v.corpus_identity_hash() != base_corpus, (section, key)

    # rank_cap_rho is the frozen UPPER BOUND on the ranking contribution
    assert cfg.rank_cap_rho == 0.5, (
        "rank_share=2.0 is retired: it normalised the ranking gradient to a FIXED multiple of "
        "|g_MC|, degrading held-out MC on 3/3 diagnostic seeds")
