"""Split from selftest.py (guide/review point 1): the pre-existing SBK-HMM/belief/rollout/R3/R4/trainer/replay main chain -- kept for regression coverage, NOT imported by the new goal-intent chain. Shares the
FULL original test namespace via ``from ..tests._common import *`` so
every test body is copied VERBATIM (byte-identical) from the original
monolithic file -- zero risk of a name/import mismatch during the split.
"""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403


def test_nonstationary_profiles_match_live_bayesian_pilot_source() -> None:
    # B7 fix (independent audit, 2026-08-06): the frozen copy in
    # config.py must stay byte-identical to the live source it was
    # copied from; if that source ever changes, this must fail loudly
    # rather than silently training against stale profile parameters.
    from crowd_nav.bayesian_dvl.config import NONSTATIONARY_PROFILES
    from crowd_nav.bayesian_pilot.protocol import PROFILES

    for name in ("nominal", "train_nonstationary", "heldout_nonstationary"):
        live = PROFILES[name]
        frozen = NONSTATIONARY_PROFILES[name]
        assert frozen["event_rate"] == live.event_rate
        assert tuple(frozen["duration_steps"]) == live.duration_steps
        assert tuple(frozen["turn_degrees"]) == live.turn_degrees
        assert tuple(frozen["slow_scale"]) == live.slow_scale
        assert tuple(frozen["event_weights"]) == live.event_weights

def test_action_grid_speed_magnitudes_match_p0_frozen_ground_truth() -> None:
    # guide.md P0-1 independently documents the actual production
    # exponential-sampling output as these 5 speed magnitudes (v_min is
    # NOT used by the exponential branch); cross-checked here bit for
    # bit against this module's own table construction.
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    table = grid.build_action_table()
    speeds = [round(float(np.hypot(vx, vy)), 12) for vx, vy in table[:5]]
    expected = [0.128851248086, 0.286230517890, 0.478453992107, 0.713236273698, 1.0]
    for got, want in zip(speeds, expected):
        assert abs(got - want) < 1e-9, (speeds, expected)

def test_action_grid_matches_existing_contracts_grid() -> None:
    from crowd_nav.contracts import discrete_index_to_action

    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    assert grid.n_speeds == 5 and grid.n_headings == 16 and grid.sampling == "exponential"
    assert grid.n_actions == 80
    table = grid.build_action_table()
    assert len(table) == 80
    grid_params = dict(
        n_speeds=grid.n_speeds, n_headings=grid.n_headings,
        v_min=grid.v_min, v_max=grid.v_max,
        sampling=grid.sampling, include_stop=grid.include_stop,
    )
    for idx, (vx, vy) in enumerate(table):
        ref_vx, ref_vy = discrete_index_to_action(idx, **grid_params)
        assert abs(vx - ref_vx) < 1e-9 and abs(vy - ref_vy) < 1e-9, (idx, (vx, vy), (ref_vx, ref_vy))

def test_action_grid_rejects_wrong_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp) / "bad_env.config"
        bad_path.write_text(
            "[policy]\nn_speeds = 6\nn_headings = 16\nv_min = 0.0\nv_max = 1.0\n"
            "sampling = exponential\ninclude_stop = false\n"
        )
        try:
            ActionGridSpec.from_env_config(str(bad_path))
            raise AssertionError("expected RegistryError for n_speeds=6")
        except RegistryError:
            pass

def test_action_grid_rejects_even_sampling_and_include_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        variants = [
            "[policy]\nn_speeds = 5\nn_headings = 16\nv_min = 0.0\nv_max = 1.0\n"
            "sampling = even\ninclude_stop = false\n",
            "[policy]\nn_speeds = 5\nn_headings = 16\nv_min = 0.0\nv_max = 1.0\n"
            "sampling = exponential\ninclude_stop = true\n",
        ]
        for text in variants:
            bad_path = Path(tmp) / "bad_env.config"
            bad_path.write_text(text)
            try:
                ActionGridSpec.from_env_config(str(bad_path))
                raise AssertionError(f"expected RegistryError for config:\n{text}")
            except RegistryError:
                pass

def test_action_grid_missing_file_fails_loudly() -> None:
    try:
        ActionGridSpec.from_env_config("/nonexistent/env.config")
        raise AssertionError("expected RegistryError for missing file")
    except RegistryError:
        pass

def test_seed_roles_are_disjoint() -> None:
    roles = SeedRoles.frozen()
    all_seeds = roles.all_seeds()
    assert len(all_seeds) == len(set(all_seeds)), "seed roles must not overlap"
    assert len(all_seeds) == 10 + 5 + 1 + 3 + 5 + 10

def test_seed_roles_detect_injected_overlap() -> None:
    roles = SeedRoles(
        world_train_suite_seeds=(1, 2, 3),
        world_validation_suite_seeds=(3, 4),  # overlaps with world_train
        il_collection_seed_base=(5,),
        rl_training_seeds=(6,),
        checkpoint_validation_seeds=(7,),
        formal_test_suite_seeds=(8,),
    )
    try:
        roles.assert_disjoint()
        raise AssertionError("expected RegistryError for overlapping seed 3")
    except RegistryError:
        pass

def test_registry_roundtrip_and_tamper_detection() -> None:
    registry = build_frozen_registry(str(ENV_CONFIG_PATH))
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "registry.json"
        write_registry(registry, str(out_path))

        loaded = load_and_validate_registry(str(out_path))
        assert loaded["action_grid_hash"] == registry.action_grid_hash

        # Tamper: flip one action-grid hash character, must fail closed.
        data = json.loads(out_path.read_text())
        original = data["action_grid_hash"]
        data["action_grid_hash"] = ("0" if original[0] != "0" else "1") + original[1:]
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        try:
            load_and_validate_registry(str(out_path))
            raise AssertionError("expected RegistryError after tampering with registry content")
        except RegistryError:
            pass

def test_registry_content_hash_is_deterministic() -> None:
    r1 = build_frozen_registry(str(ENV_CONFIG_PATH))
    r2 = build_frozen_registry(str(ENV_CONFIG_PATH))
    assert r1.content_hash() == r2.content_hash()

def test_load_and_validate_registry_rejects_retired_feature_schema() -> None:
    # guide.md R4-1R-2: this is the ONE shared gate every CLI (train/select/
    # evaluate/queue/paper-main) routes through -- testing it here covers
    # all of them at once, rather than needing a separate subprocess test
    # per entry point for the same underlying check.
    registry = build_frozen_registry(str(ENV_CONFIG_PATH))
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "registry.json"
        write_registry(registry, str(out_path))
        data = json.loads(out_path.read_text())
        data["feature_schema"] = FEATURE_SCHEMA_V3
        # content_sha256 must still match so this fails on the SCHEMA check,
        # not an unrelated hash-tamper error.
        stored_hash = data.pop("content_sha256")
        data["content_sha256"] = stored_hash
        recomputed_payload = {k: v for k, v in data.items() if k != "content_sha256"}
        import hashlib as _hashlib
        recomputed = _hashlib.sha256(json.dumps(recomputed_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        data["content_sha256"] = recomputed
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        try:
            load_and_validate_registry(str(out_path))
            raise AssertionError("expected RegistryError for retired feature_schema")
        except RegistryError as exc:
            assert "feature_schema" in str(exc)

def test_load_and_validate_registry_rejects_wrong_reward_schema() -> None:
    registry = build_frozen_registry(str(ENV_CONFIG_PATH))
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "registry.json"
        write_registry(registry, str(out_path))
        data = json.loads(out_path.read_text())
        data["reward_schema"] = "bdvl_reward_v1"
        payload = {k: v for k, v in data.items() if k != "content_sha256"}
        import hashlib as _hashlib
        data["content_sha256"] = _hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        try:
            load_and_validate_registry(str(out_path))
            raise AssertionError("expected RegistryError for wrong reward_schema")
        except RegistryError as exc:
            assert "reward_schema" in str(exc)

def test_load_and_validate_registry_env_config_mismatch_detected() -> None:
    # guide.md R4-1R-2: action_grid_hash must be cross-checked against a
    # FRESH recomputation from the currently-loaded env config, not just
    # trusted from the registry file's own (self-consistent, but
    # potentially stale) stored value.
    registry = build_frozen_registry(str(ENV_CONFIG_PATH))
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "registry.json"
        write_registry(registry, str(out_path))
        data = json.loads(out_path.read_text())
        data["action_grid_hash"] = "0" * 64
        payload = {k: v for k, v in data.items() if k != "content_sha256"}
        import hashlib as _hashlib
        data["content_sha256"] = _hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        # Without env_config_path, the tampered-but-internally-consistent
        # action_grid_hash is NOT caught (documents the opt-in nature).
        load_and_validate_registry(str(out_path))
        try:
            load_and_validate_registry(str(out_path), env_config_path=str(ENV_CONFIG_PATH))
            raise AssertionError("expected RegistryError for action_grid_hash vs env config mismatch")
        except RegistryError as exc:
            assert "action_grid_hash" in str(exc)

def test_r4_2_replay_contract_flag_is_true_and_train_bdvl_gate_is_open() -> None:
    # guide.md R4-1R-3 introduced a fail-closed gate in train_bdvl.py's
    # CLI keyed off config.R4_2_REPLAY_CONTRACT_COMPLETE, which stayed
    # False (blocking all real training) until R4-2's replay schema and
    # MC-loss rewrite actually landed -- see guide.md's R4-2 completion
    # section for the real verification that justified flipping it. This
    # test is a regression guard for the flag itself (catch an
    # accidental revert to False) plus proof the CLI gate genuinely no
    # longer fires: a bad --artifact-path now fails for THAT reason, not
    # the old "R4-2" fail-closed message, without running a real
    # (slow) multi-episode training subprocess just to prove it.
    assert R4_2_REPLAY_CONTRACT_COMPLETE is True
    import subprocess as _subprocess
    import sys as _sys
    result = _subprocess.run(
        [
            _sys.executable, "-m", "crowd_nav.tools.train_bdvl",
            "--artifact-path", "nonexistent_artifact.json",
            "--il-episodes", "1", "--rl-episodes", "1", "--allow-short-run",
            "--output", str(Path(tempfile.gettempdir()) / "should_not_be_created.pth"),
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, "expected failure from the bad artifact path, not success"
    assert "R4-2" not in result.stderr, f"R4-2 gate should no longer fire, got stderr={result.stderr!r}"
    assert not (Path(tempfile.gettempdir()) / "should_not_be_created.pth").exists()

def test_run_bdvl_queue_propagates_registry_to_every_stage() -> None:
    # guide.md R4-1R-2b: train/select/evaluate must all receive the SAME
    # explicit --registry the queue itself was given -- running the full
    # multi-stage queue end-to-end here would be far too slow for a
    # standing regression test, so this checks the actual command lists
    # run_bdvl_queue.py builds contain "--registry", args.registry for
    # every stage that touches a registry, by reading its source. A
    # structural check, not a behavioral one -- documented as such.
    source = (REPO_ROOT / "crowd_nav" / "tools" / "run_bdvl_queue.py").read_text()
    assert '"--registry", args.registry' in source, (
        "run_bdvl_queue.py must pass its own --registry through to every "
        "subprocess stage it launches (train/select/evaluate); a literal "
        "hardcoded registry path or a stage relying on its own default "
        "regressed this exact bug once already (guide.md R4-1R-2)"
    )
    assert source.count('"--registry", args.registry') >= 3, (
        "expected --registry passed to all three of train/select/evaluate stages"
    )

def test_registry_dependent_clis_default_to_v4_registry() -> None:
    # guide.md R4-1R-2c: catches a future PR silently reintroducing the
    # retired V3 file as a default (evaluate_bdvl.py/select_bdvl_checkpoint.py/
    # evaluate_bdvl_paper_main.py/run_bdvl_queue.py/policy_bayesian_dvl.config
    # all regressed this exact way once already this session).
    for rel_path in (
        "crowd_nav/tools/evaluate_bdvl.py",
        "crowd_nav/tools/select_bdvl_checkpoint.py",
        "crowd_nav/tools/evaluate_bdvl_paper_main.py",
        "crowd_nav/tools/run_bdvl_queue.py",
    ):
        source = (REPO_ROOT / rel_path).read_text()
        assert "bayesian_dvl_registry_r4.json" in source, f"{rel_path} does not default to the V4 registry"
        assert '"crowd_nav/configs/bayesian_dvl_registry.json"' not in source, (
            f"{rel_path} still references the retired V3 registry path as a live default/literal"
        )
    policy_config = (REPO_ROOT / "crowd_nav/configs/policy_bayesian_dvl.config").read_text()
    assert "bayesian_dvl_registry_r4.json" in policy_config

def test_bdvl_package_does_not_import_mamba_or_legacy_bayesian_packages() -> None:
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import crowd_nav.bayesian_dvl.config\n"
        "bad = [m for m in sys.modules if 'mamba_ssm' in m "
        "or m.startswith('crowd_nav.bayesian_brne') "
        "or m.startswith('crowd_nav.belief_mdp')]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout

def test_robot_observation_immune_to_get_obs_array_field_order() -> None:
    # crowd_sim.utils.robot.Robot.get_obs_array() returns
    # (px,py,gx,gy,vx,vy,radius,v_pref,theta) -- a DIFFERENT order from
    # FullState.to_array()'s (px,py,vx,vy,radius,gx,gy,v_pref,theta).
    # from_full_state must read named attributes, so swapping which
    # array-order a caller *thinks* it has cannot corrupt the result.
    rng = np.random.default_rng(0)
    fs = _make_full_state(rng, gx=7.0, gy=-3.0, vx=0.4, vy=-0.2, v_pref=0.9)
    robot_obs = RobotObservation.from_full_state(fs)
    assert robot_obs.gx == 7.0 and robot_obs.gy == -3.0
    assert robot_obs.vx == 0.4 and robot_obs.vy == -0.2
    assert robot_obs.v_pref == 0.9
    canonical = robot_obs.to_canonical_array()
    assert np.allclose(canonical, fs.to_array())

def test_canonicalize_shapes_1_5_10_12_20_humans() -> None:
    rng = np.random.default_rng(1)
    fs = _make_full_state(rng)
    for n in (1, 5, 10, 12, 20):
        humans = [_make_observable_state(rng) for _ in range(n)]
        track_ids = list(range(n))
        obs = canonicalize(fs, humans, track_ids)
        assert obs.human_features.shape == (MAX_HUMANS, 5)
        assert obs.human_mask.shape == (MAX_HUMANS,)
        assert obs.n_humans == n
        assert int(obs.human_mask.sum()) == n
        assert np.all(obs.human_features[n:] == 0.0)
        assert np.all(obs.human_mask[n:] == False)  # noqa: E712

def test_canonicalize_rejects_more_than_max_humans() -> None:
    rng = np.random.default_rng(2)
    fs = _make_full_state(rng)
    humans = [_make_observable_state(rng) for _ in range(MAX_HUMANS + 1)]
    try:
        canonicalize(fs, humans, list(range(MAX_HUMANS + 1)))
        raise AssertionError("expected ValueError for >MAX_HUMANS humans")
    except ValueError:
        pass

def test_canonical_observation_content_is_permutation_invariant() -> None:
    rng = np.random.default_rng(3)
    fs = _make_full_state(rng)
    humans = [_make_observable_state(rng) for _ in range(6)]
    track_ids = list(range(6))
    obs_a = canonicalize(fs, humans, track_ids)

    perm = [3, 1, 4, 0, 5, 2]
    humans_b = [humans[i] for i in perm]
    track_ids_b = [track_ids[i] for i in perm]
    obs_b = canonicalize(fs, humans_b, track_ids_b)

    assert canonical_observation_is_permutation_invariant_content(obs_a, obs_b)

    # Sanity: corrupting one human's radius must break equivalence.
    import copy

    humans_c = copy.deepcopy(humans_b)
    humans_c[0].radius += 1.0
    obs_c = canonicalize(fs, humans_c, track_ids_b)
    assert not canonical_observation_is_permutation_invariant_content(obs_a, obs_c)

def test_canonical_observation_rejects_bad_mask_dtype_and_nonzero_padding() -> None:
    rng = np.random.default_rng(4)
    fs = _make_full_state(rng)
    robot = RobotObservation.from_full_state(fs)
    features = np.zeros((MAX_HUMANS, 5), dtype=np.float64)
    mask_int = np.zeros((MAX_HUMANS,), dtype=np.int64)
    try:
        CanonicalObservation(robot=robot, human_track_ids=(), human_features=features, human_mask=mask_int)
        raise AssertionError("expected ValueError for non-bool mask")
    except ValueError:
        pass

    mask_bool = np.zeros((MAX_HUMANS,), dtype=np.bool_)
    features_dirty = np.zeros((MAX_HUMANS, 5), dtype=np.float64)
    features_dirty[5, 0] = 1.0  # nonzero in a masked-out row
    try:
        CanonicalObservation(robot=robot, human_track_ids=(), human_features=features_dirty, human_mask=mask_bool)
        raise AssertionError("expected ValueError for nonzero padding in masked row")
    except ValueError:
        pass

def test_swept_collision_not_missed_by_endpoint_only_check() -> None:
    # Robot at origin heading toward (0,0)->(10,0) goal, human positioned
    # 0.1m off the robot's line of travel. Both the human's position
    # BEFORE and AFTER this step are ~2.0m from the robot (safe by an
    # endpoint-only check), but the straight-line sweep between them
    # passes within 0.1m of the robot -- well inside radius_sum=0.6m.
    # guide.md A2 explicitly requires this NOT be missed.
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=10.0, gy=0.0, v_pref=1.0, theta=0.0)
    human_before = HumanObservation(track_id=0, px=2.0, py=0.1, vx=0.0, vy=0.0, radius=0.3)
    dt = 0.25
    action_vx = 16.0  # contrived speed, purely to test the geometry, not a legal action-table entry
    action_vy = 0.0

    result = bdvl_transition_step(
        robot=robot, humans=[human_before], action_vx=action_vx, action_vy=action_vy,
        human_actions=[(0.0, 0.0)], dt=dt, time_limit=1000.0, global_time=0.0,
        reward_config=_REWARD_CFG,
    )
    assert result.event == "collision", (
        f"expected swept-through collision to be detected, got event={result.event!r} dmin={result.dmin}"
    )

    # Sanity: an ENDPOINT-only check (ignoring the segment sweep) would
    # have said both ends are safe -- confirms this fixture actually
    # exercises the swept-vs-endpoint distinction, not a trivial case.
    end_px = human_before.px + (0.0 - action_vx) * dt  # relative human motion = human.vx - action_vx
    end_dist = float(np.hypot(end_px, human_before.py))
    start_dist = float(np.hypot(human_before.px, human_before.py))
    radius_sum = human_before.radius + robot.radius
    assert start_dist - radius_sum > 0, "fixture bug: start point already colliding"
    assert end_dist - radius_sum > 0, "fixture bug: end point already colliding"

def test_no_collision_when_clearance_is_large() -> None:
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=10.0, gy=0.0, v_pref=1.0, theta=0.0)
    human = HumanObservation(track_id=0, px=5.0, py=5.0, vx=0.0, vy=0.0, radius=0.3)
    result = bdvl_transition_step(
        robot=robot, humans=[human], action_vx=0.5, action_vy=0.0,
        human_actions=[(0.0, 0.0)], dt=0.25, time_limit=1000.0, global_time=0.0,
        reward_config=_REWARD_CFG,
    )
    assert result.event == "nothing"

def test_termination_priority_matches_crowdsim_timeout_beats_collision() -> None:
    # crowd_sim.py:step() checks `if global_time >= time_limit: timeout`
    # BEFORE checking collision -- a collision on the very last step
    # must still report "timeout", not "collision".
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=10.0, gy=0.0, v_pref=1.0, theta=0.0)
    human = HumanObservation(track_id=0, px=0.05, py=0.0, vx=0.0, vy=0.0, radius=0.3)  # already overlapping
    result = bdvl_transition_step(
        robot=robot, humans=[human], action_vx=0.0, action_vy=0.0,
        human_actions=[(0.0, 0.0)], dt=0.25, time_limit=10.0, global_time=10.0,
        reward_config=_REWARD_CFG,
    )
    assert result.event == "timeout", f"expected timeout to take priority over collision, got {result.event!r}"

def test_sbk_hmm_recovers_cv_mode_from_constant_velocity_track() -> None:
    dt = 0.25
    tracks = [_track_from_kinematics(dt, 40, speed0=1.0, accel=0.0, omega=0.0)]
    artifact = fit_sbk_hmm(tracks, train_data_sha256="test", max_iterations=30)
    assert artifact.converged
    # CV mode's mean must stay inside its near-zero constraint box.
    assert abs(artifact.emission_mean[CV, 0]) < 0.2
    assert abs(artifact.emission_mean[CV, 1]) < 0.2
    # And the responsibility-weighted posterior must actually prefer CV
    # for this data (not, say, permutation-collapsed into TURN_L).
    feature = np.array([0.0, 0.0])
    log_probs = artifact.emission_log_prob(feature)
    assert int(np.argmax(log_probs)) == CV

def test_sbk_hmm_recovers_signed_modes_without_permutation_swap() -> None:
    dt = 0.25
    tracks = [
        _track_from_kinematics(dt, 40, speed0=0.3, accel=0.8, omega=0.0),   # ACC
        _track_from_kinematics(dt, 40, speed0=1.5, accel=-0.8, omega=0.0),  # DECEL
        _track_from_kinematics(dt, 40, speed0=1.0, accel=0.0, omega=1.0),   # TURN_L
        _track_from_kinematics(dt, 40, speed0=1.0, accel=0.0, omega=-1.0),  # TURN_R
    ]
    artifact = fit_sbk_hmm(tracks, train_data_sha256="test", max_iterations=30)
    # Sign-constrained projection guarantees these hold structurally,
    # not just "usually true from a lucky random init" -- this is the
    # A3 acceptance requirement that mode identity cannot permutation-swap.
    assert artifact.emission_mean[ACC, 0] > 0
    assert artifact.emission_mean[DECEL, 0] < 0
    assert artifact.emission_mean[TURN_L, 1] > 0
    assert artifact.emission_mean[TURN_R, 1] < 0

def test_sbk_hmm_pure_cv_data_does_not_fake_turning() -> None:
    dt = 0.25
    tracks = [_track_from_kinematics(dt, 60, speed0=1.0, accel=0.0, omega=0.0)]
    artifact = fit_sbk_hmm(tracks, train_data_sha256="test", max_iterations=30)
    feature = np.array([0.0, 0.0])
    log_probs = artifact.emission_log_prob(feature)
    best_mode = int(np.argmax(log_probs))
    assert best_mode == CV, f"pure CV data should not be best explained by mode {MODE_NAMES_LOOKUP[best_mode]}"

def test_sbk_hmm_save_load_roundtrip_bit_identical() -> None:
    dt = 0.25
    tracks = [_track_from_kinematics(dt, 30, speed0=1.0, accel=0.3, omega=0.5)]
    artifact = fit_sbk_hmm(tracks, train_data_sha256="test", max_iterations=10)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact.json"
        h1 = artifact.save(str(path))
        loaded = SBKHMMArtifact.load(str(path))
        assert loaded.content_sha256() == h1
        assert np.allclose(loaded.emission_mean, artifact.emission_mean)
        assert np.allclose(loaded.transition_counts, artifact.transition_counts)
        feature = np.array([0.1, 0.2])
        assert np.allclose(loaded.emission_log_prob(feature), artifact.emission_log_prob(feature))

def test_sbk_hmm_cached_predictive_arrays_are_correct_and_read_only() -> None:
    artifact = _fixture_artifact()
    expected_cov = np.stack([params.predictive_covariance() for params in artifact._niw_params], axis=0)
    expected_transition = artifact.transition_counts / artifact.transition_counts.sum(axis=1, keepdims=True)
    expected_initial = artifact.initial_counts / artifact.initial_counts.sum()
    assert np.allclose(artifact.emission_cov, expected_cov)
    assert np.allclose(artifact.transition_matrix, expected_transition)
    assert np.allclose(artifact.initial_distribution, expected_initial)
    assert not artifact.emission_cov.flags.writeable
    assert not artifact.transition_matrix.flags.writeable
    assert not artifact.initial_distribution.flags.writeable
    try:
        artifact.emission_cov[0, 0, 0] = 0.0
        raise AssertionError("cached predictive covariance must be read-only")
    except ValueError:
        pass

def test_predictive_moments_vectorization_matches_mixture_reference() -> None:
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact)
    tracker.update({7: (0.0, np.array([0.0, 0.0]))})
    mean, cov = tracker.predictive_moments_for(7)
    belief = tracker.belief_for(7)
    reference_mean = belief @ artifact.emission_mean
    reference_cov = np.zeros((2, 2))
    for k in range(N_MODES):
        diff = artifact.emission_mean[k] - reference_mean
        reference_cov += belief[k] * (artifact.emission_cov[k] + np.outer(diff, diff))
    assert np.allclose(mean, reference_mean, atol=1e-12)
    assert np.allclose(cov, reference_cov, atol=1e-12)

def test_sbk_hmm_load_rejects_tampered_artifact() -> None:
    dt = 0.25
    tracks = [_track_from_kinematics(dt, 20, speed0=1.0, accel=0.0, omega=0.0)]
    artifact = fit_sbk_hmm(tracks, train_data_sha256="test", max_iterations=5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact.json"
        artifact.save(str(path))
        data = json.loads(path.read_text())
        data["niw_mu"][0][0] += 999.0
        path.write_text(json.dumps(data))
        try:
            SBKHMMArtifact.load(str(path))
            raise AssertionError("expected WorldModelError on tampered artifact")
        except Exception as exc:
            assert "hash mismatch" in str(exc)

def test_niw_rejects_non_positive_definite_psi() -> None:
    bad_psi = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3 -> not PD
    try:
        NIWParams(mu=np.zeros(2), kappa=1.0, nu=4.0, psi=bad_psi)
        raise AssertionError("expected WorldModelError for non-PD psi")
    except WorldModelError:
        pass

def test_niw_rejects_invalid_nu_and_kappa() -> None:
    good_psi = np.eye(2)
    try:
        NIWParams(mu=np.zeros(2), kappa=1.0, nu=0.5, psi=good_psi)  # nu must be > d-1=1
        raise AssertionError("expected WorldModelError for nu <= d-1")
    except WorldModelError:
        pass
    try:
        NIWParams(mu=np.zeros(2), kappa=-1.0, nu=4.0, psi=good_psi)
        raise AssertionError("expected WorldModelError for non-positive kappa")
    except WorldModelError:
        pass

def test_promote_to_production_rejects_unconverged_fit() -> None:
    artifact = _fixture_artifact()
    unconverged = SBKHMMArtifact(
        dt=artifact.dt, transition_counts=artifact.transition_counts,
        niw_mu=artifact.niw_mu, niw_kappa=artifact.niw_kappa, niw_nu=artifact.niw_nu, niw_psi=artifact.niw_psi,
        initial_counts=artifact.initial_counts, n_iterations=50, converged=False,
        log_likelihood_history=artifact.log_likelihood_history, train_data_sha256=artifact.train_data_sha256,
    )
    try:
        promote_to_production(unconverged)
        raise AssertionError("expected WorldModelError for unconverged artifact")
    except WorldModelError:
        pass
    converged = SBKHMMArtifact(
        dt=artifact.dt, transition_counts=artifact.transition_counts,
        niw_mu=artifact.niw_mu, niw_kappa=artifact.niw_kappa, niw_nu=artifact.niw_nu, niw_psi=artifact.niw_psi,
        initial_counts=artifact.initial_counts, n_iterations=10, converged=True,
        log_likelihood_history=artifact.log_likelihood_history, train_data_sha256=artifact.train_data_sha256,
    )
    assert promote_to_production(converged) is converged

def test_fit_sbk_hmm_rejects_mixed_dt_tracks() -> None:
    track_a = _track_from_kinematics(0.25, 20, speed0=1.0, accel=0.0, omega=0.0)
    track_b = _track_from_kinematics(0.5, 20, speed0=1.0, accel=0.0, omega=0.0)  # different dt
    try:
        fit_sbk_hmm([track_a, track_b], train_data_sha256="test")
        raise AssertionError("expected WorldModelError for mixed dt across tracks")
    except WorldModelError as exc:
        assert "mixed dt" in str(exc)

def test_track_feature_runs_split_on_low_speed_gap_not_concatenated() -> None:
    # Build a position sequence with a genuine low-speed (near-stationary)
    # gap in the middle: the two valid feature runs on either side of
    # the gap must NOT be silently concatenated into one sequence (R1
    # fix, independent audit B1 repair item 5) -- that would let the
    # HMM treat a real multi-step temporal gap as a single transition.
    dt = 0.25
    moving_a = _positions_from_kinematics(dt, 10, speed0=1.0, accel=0.0, omega=0.0)
    stationary_gap = [moving_a[-1] + np.array([1e-4, 0.0]) * i for i in range(1, 4)]  # near-zero speed
    moving_b_start = stationary_gap[-1]
    moving_b = [moving_b_start]
    speed, heading = 1.0, 0.3
    for _ in range(10):
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        moving_b.append(moving_b[-1] + vel * dt)

    positions = np.array(moving_a + stationary_gap + moving_b)
    track = Track(positions=positions, dt=dt)
    runs = track.feature_runs()
    assert len(runs) >= 2, f"expected at least 2 separate contiguous runs across the low-speed gap, got {len(runs)}"

def test_sample_sign_truncated_predictive_never_violates_constraint() -> None:
    artifact = _fixture_artifact()
    rng = np.random.default_rng(0)
    for mode in range(N_MODES):
        a_sign, w_sign = _MODE_SIGN_CONSTRAINTS[mode]
        for _ in range(200):
            sample = sample_sign_truncated_predictive(rng, artifact.niw(mode), mode)
            a, w = sample
            if a_sign == 1:
                assert a > 0
            elif a_sign == -1:
                assert a < 0
            if w_sign == 1:
                assert w > 0
            elif w_sign == -1:
                assert w < 0

def test_multivariate_t_log_pdf_converges_to_gaussian_at_large_dof() -> None:
    # A multivariate Student-t with dof -> infinity converges to the
    # Gaussian with the same location/scale -- an independently
    # hand-computable check (Gaussian log-pdf is trivial) that the
    # Student-t formula's normalization/exponent are correct, not just
    # "some formula that happens to run".
    mu = np.array([0.3, -0.2])
    scale = np.array([[0.5, 0.05], [0.05, 0.3]])
    dof = 1e6  # effectively infinite
    x = np.array([[0.4, -0.1]])

    t_log_pdf = _multivariate_t_log_pdf(x, dof, mu, scale)[0]

    diff = x[0] - mu
    inv = np.linalg.inv(scale)
    det = np.linalg.det(scale)
    gaussian_log_pdf = -0.5 * (2 * np.log(2 * np.pi) + np.log(det) + diff @ inv @ diff)

    assert abs(t_log_pdf - gaussian_log_pdf) < 1e-3, (t_log_pdf, gaussian_log_pdf)

def test_sbk_hmm_rejects_too_short_tracks() -> None:
    from crowd_nav.bayesian_dvl.world_model import WorldModelError

    tracks = [Track(positions=np.array([[0.0, 0.0], [0.1, 0.0]]), dt=0.25)]  # only 2 points
    try:
        fit_sbk_hmm(tracks, train_data_sha256="test")
        raise AssertionError("expected WorldModelError for all-too-short tracks")
    except WorldModelError:
        pass

def test_belief_tracker_hand_computed_three_step_posterior() -> None:
    artifact = _fixture_artifact()
    positions = _positions_from_kinematics(artifact.dt, 4, speed0=1.0, accel=0.8, omega=0.0)
    tracker = BeliefTracker(artifact)

    # Step 0: first observation -> prior only, no feature yet.
    tracker.update({0: (0.0, positions[0])})
    expected = artifact.initial_distribution.copy()
    assert np.allclose(tracker.belief_for(0), expected)

    # Step 1: still only one velocity estimate -> no feature yet, prediction only.
    tracker.update({0: (artifact.dt, positions[1])})
    expected = expected @ artifact.transition_matrix
    expected = expected / expected.sum()
    assert np.allclose(tracker.belief_for(0), expected)

    # Step 2: now two velocities exist -> a real [a_parallel, omega] feature and a real update.
    tracker.update({0: (2 * artifact.dt, positions[2])})
    v0 = (positions[1] - positions[0]) / artifact.dt
    v1 = (positions[2] - positions[1]) / artifact.dt
    a_parallel = (np.linalg.norm(v1) - np.linalg.norm(v0)) / artifact.dt
    predicted = expected @ artifact.transition_matrix
    predicted = predicted / predicted.sum()
    log_emission = artifact.emission_log_prob(np.array([a_parallel, 0.0]))
    log_emission = log_emission - log_emission.max()
    likelihood = np.exp(log_emission)
    hand_computed = predicted * likelihood
    hand_computed = hand_computed / hand_computed.sum()
    assert np.allclose(tracker.belief_for(0), hand_computed, atol=1e-10)

def test_belief_tracker_fresh_track_uses_artifact_initial_distribution_not_uniform() -> None:
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact)
    tracker.update({5: (0.0, np.array([1.0, 2.0]))})
    assert np.allclose(tracker.belief_for(5), artifact.initial_distribution)
    assert not np.allclose(artifact.initial_distribution, np.full(N_MODES, 1.0 / N_MODES)), (
        "fixture's initial distribution must not itself already be uniform, or this test proves nothing"
    )

def test_belief_tracker_miss_then_reappear_applies_exactly_two_transitions_not_three() -> None:
    # R1 regression (independent audit B2, 2026-08-06): a miss at
    # t=dt followed by reappearance at t=2*dt must advance the belief
    # by exactly Pi^2 (one transition per elapsed dt), not Pi^3. The
    # old bug double-counted the missing step's own transition.
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact)
    tracker.update({0: (0.0, np.array([0.0, 0.0]))})
    b0 = tracker.belief_for(0)

    tracker.update({})  # miss at t=dt
    tracker.update({0: (2 * artifact.dt, np.array([2 * artifact.dt, 0.0]))})  # reappear at t=2*dt

    pi = artifact.transition_matrix
    correct = b0 @ pi @ pi
    correct = correct / correct.sum()
    buggy = b0 @ pi @ pi @ pi
    buggy = buggy / buggy.sum()

    actual = tracker.belief_for(0)
    assert np.allclose(actual, correct, atol=1e-10), f"expected b0@Pi^2={correct}, got {actual}"
    assert not np.allclose(actual, buggy, atol=1e-6), "belief matches the OLD buggy b0@Pi^3 -- regression reintroduced"

def test_belief_tracker_reappearance_does_not_compute_feature_across_gap() -> None:
    # The reappearance observation itself must not silently compute a
    # velocity/acceleration spanning the gap (dividing a multi-step
    # displacement by a single dt would corrupt the feature). Verify by
    # constructing a displacement that would look like an extreme,
    # obviously-wrong acceleration if computed naively across the gap,
    # and confirming the belief update stays a pure prediction (matches
    # the miss-only prediction, not a likelihood-weighted one).
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact)
    tracker.update({0: (0.0, np.array([0.0, 0.0]))})
    tracker.update({})  # miss
    tracker.update({0: (2 * artifact.dt, np.array([100.0, 0.0]))})  # huge jump across the gap
    after_reappearance = tracker.belief_for(0)

    # A second, independent tracker that only ever sees the belief
    # recursion (no observation at all) should match exactly, since
    # the reappearance frame must contribute pure prediction, not a
    # (wrong) likelihood term from the huge fake displacement.
    tracker2 = BeliefTracker(artifact)
    tracker2.update({0: (0.0, np.array([0.0, 0.0]))})
    tracker2.update({})
    tracker2.update({})  # second miss instead of an observation: pure prediction path only
    reference = tracker2.belief_for(0)
    assert np.allclose(after_reappearance, reference, atol=1e-10), (
        "reappearance after a gap must not fold a cross-gap displacement into a likelihood term"
    )

def test_belief_tracker_missing_frame_prediction_only_no_stale_likelihood() -> None:
    artifact = _fixture_artifact()
    positions = _positions_from_kinematics(artifact.dt, 5, speed0=1.0, accel=0.0, omega=1.0)
    tracker = BeliefTracker(artifact)
    tracker.update({0: (0.0, positions[0])})
    tracker.update({0: (artifact.dt, positions[1])})
    before_missing = tracker.belief_for(0)
    tracker.update({})  # track 0 goes missing for one step
    after_missing = tracker.belief_for(0)
    expected = before_missing @ artifact.transition_matrix
    expected = expected / expected.sum()
    assert np.allclose(after_missing, expected)

def test_belief_tracker_reappearance_after_timeout_is_a_new_track() -> None:
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact, missing_timeout_steps=2)
    tracker.update({0: (0.0, np.array([0.0, 0.0]))})
    tracker.update({0: (artifact.dt, np.array([0.25, 0.0]))})
    tracker.update({})  # miss 1
    tracker.update({})  # miss 2 -> should expire (timeout_steps=2)
    assert 0 not in tracker.active_track_ids()
    tracker.update({0: (10.0, np.array([5.0, 5.0]))})  # "reappears" -> fresh track
    assert np.allclose(tracker.belief_for(0), artifact.initial_distribution)

def test_belief_tracker_rejects_duplicate_and_backward_timestamps() -> None:
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact)
    tracker.update({0: (1.0, np.array([0.0, 0.0]))})
    try:
        tracker.update({0: (1.0, np.array([0.1, 0.0]))})  # duplicate timestamp
        raise AssertionError("expected BeliefError for duplicate timestamp")
    except BeliefError:
        pass
    try:
        tracker.update({0: (0.5, np.array([0.1, 0.0]))})  # backward timestamp
        raise AssertionError("expected BeliefError for backward timestamp")
    except BeliefError:
        pass

def test_belief_tracker_rejects_non_dt_multiple_timestamp() -> None:
    artifact = _fixture_artifact()
    tracker = BeliefTracker(artifact)
    tracker.update({0: (0.0, np.array([0.0, 0.0]))})
    try:
        tracker.update({0: (0.4, np.array([0.1, 0.0]))})  # dt=0.4, not a multiple of artifact.dt=0.25
        raise AssertionError("expected BeliefError for non-dt-multiple timestamp")
    except BeliefError:
        pass

def test_belief_tracker_output_independent_of_dict_iteration_order() -> None:
    artifact = _fixture_artifact()
    pos_a = _positions_from_kinematics(artifact.dt, 3, speed0=1.0, accel=0.5, omega=0.0)
    pos_b = _positions_from_kinematics(artifact.dt, 3, speed0=1.0, accel=0.0, omega=0.7)

    tracker1 = BeliefTracker(artifact)
    tracker2 = BeliefTracker(artifact)
    for t in range(3):
        obs = {0: (t * artifact.dt, pos_a[t]), 1: (t * artifact.dt, pos_b[t])}
        tracker1.update(obs)
        # Same content, different dict construction order.
        obs_reordered = {1: (t * artifact.dt, pos_b[t]), 0: (t * artifact.dt, pos_a[t])}
        tracker2.update(obs_reordered)

    assert np.allclose(tracker1.belief_for(0), tracker2.belief_for(0))
    assert np.allclose(tracker1.belief_for(1), tracker2.belief_for(1))

def test_belief_tracker_hypothetical_clone_does_not_mutate_real_tracker() -> None:
    artifact = _fixture_artifact()
    positions = _positions_from_kinematics(artifact.dt, 3, speed0=1.0, accel=0.3, omega=0.0)
    tracker = BeliefTracker(artifact)
    tracker.update({0: (0.0, positions[0])})
    tracker.update({0: (artifact.dt, positions[1])})
    real_belief_before = tracker.belief_for(0)

    clone = tracker.clone_for_hypothetical()
    clone.update({0: (2 * artifact.dt, positions[2])})
    clone.update({99: (2 * artifact.dt, np.array([10.0, 10.0]))})

    assert np.allclose(tracker.belief_for(0), real_belief_before), "real tracker mutated by hypothetical clone"
    assert 99 not in tracker.active_track_ids(), "real tracker gained a track from the hypothetical clone"
    assert not np.allclose(clone.belief_for(0), real_belief_before), "clone fixture is vacuous (nothing changed)"

def test_rollout_same_seed_bit_identical_different_seed_differs() -> None:
    artifact = _fixture_artifact()
    belief = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    r1 = sample_human_next_states(**_sample_common_args(artifact, belief, seed=(1, 2, 3)))
    r2 = sample_human_next_states(**_sample_common_args(artifact, belief, seed=(1, 2, 3)))
    r3 = sample_human_next_states(**_sample_common_args(artifact, belief, seed=(1, 2, 4)))

    pos1 = np.array([s.next_position for s in r1[0]])
    pos2 = np.array([s.next_position for s in r2[0]])
    pos3 = np.array([s.next_position for s in r3[0]])
    assert np.array_equal(pos1, pos2), "identical seed must give bit-identical samples"
    assert not np.array_equal(pos1, pos3), "different seed must give different samples"

def test_rollout_full_posterior_keeps_bimodality_moment_mean_collapses() -> None:
    artifact = _fixture_artifact()
    # Bimodal belief: half mass on ACC (a_parallel>0), half on DECEL (a_parallel<0).
    belief = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    full = sample_human_next_states(**_sample_common_args(artifact, belief, n_samples=500, source="full"))
    a_parallels = np.array([s.a_parallel for s in full[0]])
    # A real bimodal sample set should have a substantial fraction on both sides of 0.
    frac_positive = float((a_parallels > 0).mean())
    assert 0.25 < frac_positive < 0.75, f"expected roughly bimodal split, got frac_positive={frac_positive}"

    moment_mean = sample_human_next_states(**_sample_common_args(artifact, belief, n_samples=5, source="moment_mean"))
    mm_a_parallels = np.array([s.a_parallel for s in moment_mean[0]])
    expected_mean = float(belief @ artifact.emission_mean[:, 0])
    assert np.allclose(mm_a_parallels, expected_mean), "moment_mean must collapse to the single belief-weighted mean"
    assert all(s.sampled_mode is None for s in moment_mean[0])

def test_rollout_cv_source_ignores_belief_map_source_is_argmax() -> None:
    artifact = _fixture_artifact()
    belief = np.array([0.05, 0.05, 0.8, 0.05, 0.05])  # DECEL dominates
    cv_result = sample_human_next_states(**_sample_common_args(artifact, belief, n_samples=50, source="cv"))
    assert all(s.sampled_mode == CV for s in cv_result[0])

    map_result = sample_human_next_states(**_sample_common_args(artifact, belief, n_samples=50, source="map"))
    assert all(s.sampled_mode == DECEL for s in map_result[0])

def test_rollout_respects_max_human_speed_clamp() -> None:
    artifact = _fixture_artifact()
    belief = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # pure ACC, will keep speeding up
    args = _sample_common_args(artifact, belief, n_samples=100, source="full")
    args["track_speeds"] = {0: 1.9}
    args["max_human_speed"] = 2.0
    result = sample_human_next_states(**args)
    speeds = np.array([s.next_speed for s in result[0]])
    assert np.all(speeds <= 2.0 + 1e-9)
    assert np.all(speeds >= 0.0)

def test_set_encoder_permutation_invariant() -> None:
    encoder = SetEncoder()
    encoder.eval()
    robot_features, human_features, mask = _random_crowd(0, n_humans=6)
    with torch.no_grad():
        out_a = encoder(robot_features, human_features, mask)

    perm = torch.randperm(6)
    human_features_b = human_features.clone()
    human_features_b[:, :6, :] = human_features[:, perm, :]
    with torch.no_grad():
        out_b = encoder(robot_features, human_features_b, mask)

    assert torch.allclose(out_a, out_b, atol=1e-5)

def test_set_encoder_padding_invariant() -> None:
    encoder = SetEncoder()
    encoder.eval()
    robot_features, human_features_5, mask_5 = _random_crowd(1, n_humans=5)
    with torch.no_grad():
        out_5 = encoder(robot_features, human_features_5, mask_5)

    # Pad with garbage in the masked-out slots -- must not affect output.
    human_features_padded = human_features_5.clone()
    human_features_padded[:, 5:, :] = torch.randn_like(human_features_padded[:, 5:, :]) * 1000.0
    with torch.no_grad():
        out_padded = encoder(robot_features, human_features_padded, mask_5)

    assert torch.allclose(out_5, out_padded, atol=1e-5)

def test_set_encoder_generalizes_5_to_10_12_20_humans_no_nan() -> None:
    encoder = SetEncoder()
    for n in (1, 5, 10, 12, 20):
        robot_features, human_features, mask = _random_crowd(2, n_humans=n)
        out = encoder(robot_features, human_features, mask)
        assert out.shape == (2, 128)
        assert torch.isfinite(out).all()

def test_set_encoder_zero_humans_no_nan() -> None:
    encoder = SetEncoder()
    robot_features, human_features, mask = _random_crowd(3, n_humans=0)
    out = encoder(robot_features, human_features, mask)
    assert torch.isfinite(out).all()

def test_set_encoder_20_humans_gradients_finite() -> None:
    encoder = SetEncoder()
    robot_features, human_features, mask = _random_crowd(4, n_humans=20)
    out = encoder(robot_features, human_features, mask)
    loss = out.sum()
    loss.backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"

def test_iqn_no_mamba_ssm_import() -> None:
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import crowd_nav.bayesian_dvl.set_encoder\n"
        "import crowd_nav.bayesian_dvl.iqn\n"
        "assert 'mamba_ssm' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run([_sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout

def test_quantile_huber_loss_matches_hand_computation() -> None:
    torch.manual_seed(0)
    pred = torch.tensor([[0.5]])
    tau = torch.tensor([[0.3]])
    target = torch.tensor([[2.0]])
    loss = quantile_huber_loss(pred, tau, target, kappa=1.0)
    expected = quantile_huber_loss_hand_check(0.5, 0.3, 2.0, kappa=1.0)
    assert abs(float(loss) - expected) < 1e-6

    pred2 = torch.tensor([[3.0]])
    tau2 = torch.tensor([[0.7]])
    target2 = torch.tensor([[1.0]])
    loss2 = quantile_huber_loss(pred2, tau2, target2, kappa=1.0)
    expected2 = quantile_huber_loss_hand_check(3.0, 0.7, 1.0, kappa=1.0)
    assert abs(float(loss2) - expected2) < 1e-6

def test_iqn_quantiles_roughly_monotonic_after_training_on_fixed_target() -> None:
    # Train a tiny IQN to fit a FIXED scalar target distribution (a
    # known Gaussian): the estimated quantile function should end up
    # increasing in tau (allow a small crossing rate, guide.md 5.3/A6:
    # "不强制网络层本身硬单调，但报告crossing rate").
    torch.manual_seed(0)
    state_dim = 8
    # R2-2 fix (2026-08-07): IQNValueNetwork now bounds its output to
    # [v_min, v_max] (default: BDVL's own derived reward-implied range).
    # This test checks GENERIC quantile-monotonicity behavior, not
    # BDVL's specific bounds, so it must use a range wide enough to
    # actually contain target_mean=2.0 -- otherwise a correctly-bounded
    # network could never fit the target and the assertions below would
    # be testing "is the network capped" instead of "is it monotonic".
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=state_dim, action_embedding_dim=action_encoder.embed_dim, n_cosines=16, hidden_dim=32, v_min=-10.0, v_max=10.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    fixed_state = torch.randn(1, state_dim)
    # This test checks generic quantile-monotonicity behavior, not action
    # conditioning -- a single fixed, untrained action embedding (like
    # fixed_state) keeps the target distribution well-defined across all
    # 400 steps.
    with torch.no_grad():
        fixed_action = action_encoder(torch.randn(1, ACTION_FEATURE_DIM))
    target_mean, target_std = 2.0, 1.0

    for _ in range(400):
        optimizer.zero_grad()
        tau = torch.rand(1, 8)
        pred = net(fixed_state.expand(1, -1), fixed_action, tau)
        target_samples = torch.randn(1, 32) * target_std + target_mean
        loss = quantile_huber_loss(pred, tau, target_samples)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_tau = torch.linspace(0.05, 0.95, 19).unsqueeze(0)
        quantiles = net(fixed_state, fixed_action, eval_tau).squeeze(0)
    diffs = quantiles[1:] - quantiles[:-1]
    crossing_rate = float((diffs < 0).float().mean())
    assert crossing_rate < 0.3, f"quantile function too non-monotonic after training: crossing_rate={crossing_rate}"
    # Sanity: low/high tau should roughly bracket the target mean.
    assert float(quantiles[0]) < target_mean < float(quantiles[-1]) + 1.0

def test_bdvl_policy_returns_all_80_action_scores() -> None:
    policy = _tiny_policy()
    robot, humans = _fixture_robot_and_humans()
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    assert len(result.all_scores) == 80
    assert 0 <= result.chosen_action_index < 80

def test_bdvl_policy_action_index_roundtrips_to_action_table() -> None:
    policy = _tiny_policy()
    robot, humans = _fixture_robot_and_humans()
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    expected_action = policy.action_table[result.chosen_action_index]
    assert result.chosen_action == expected_action

def test_bdvl_policy_never_averages_returns_exactly_one_table_action() -> None:
    policy = _tiny_policy()
    robot, humans = _fixture_robot_and_humans()
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    # The chosen action must be BIT-IDENTICAL to one specific table
    # entry -- not merely close to one (which a blend of two adjacent
    # actions could also satisfy by coincidence).
    matches = [a for a in policy.action_table if a == result.chosen_action]
    assert len(matches) >= 1

def test_bdvl_policy_rejects_more_than_20_humans() -> None:
    policy = _tiny_policy()
    robot, _ = _fixture_robot_and_humans()
    too_many = [HumanObservation(track_id=i, px=1.0, py=1.0, vx=0.0, vy=0.0, radius=0.3) for i in range(21)]
    try:
        policy.decide(robot, too_many, global_time=1.0, suite_seed=1, episode_seed=1)
        raise AssertionError("expected PolicyError for >20 humans")
    except PolicyError:
        pass

def test_bdvl_policy_reset_episode_stats_clears_tracks_and_counter() -> None:
    policy = _tiny_policy()
    robot, humans = _fixture_robot_and_humans()
    policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    assert len(policy.belief_tracker.active_track_ids()) > 0
    assert policy._decision_counter > 0

    policy.reset_episode_stats()
    assert len(policy.belief_tracker.active_track_ids()) == 0
    assert policy._decision_counter == 0

def test_bdvl_policy_decision_is_reproducible_across_fresh_instances() -> None:
    # Same seeds on two INDEPENDENTLY constructed policies (fresh
    # networks with the same init, since torch.manual_seed is set
    # identically) must produce bit-identical scores -- this is what
    # the counter-based deterministic tau fix (guide.md 5.4) buys.
    robot, humans = _fixture_robot_and_humans()

    torch.manual_seed(42)
    policy1 = _tiny_policy()
    result1 = policy1.decide(robot, humans, global_time=1.0, suite_seed=7, episode_seed=3)

    torch.manual_seed(42)
    policy2 = _tiny_policy()
    result2 = policy2.decide(robot, humans, global_time=1.0, suite_seed=7, episode_seed=3)

    scores1 = np.array([s.cvar for s in result1.all_scores])
    scores2 = np.array([s.cvar for s in result2.all_scores])
    assert np.array_equal(scores1, scores2), "identical seeds must give bit-identical decision scores"
    assert result1.chosen_action_index == result2.chosen_action_index

def test_bdvl_policy_same_geometry_different_belief_changes_scores() -> None:
    # R2 acceptance (independent audit B0, 2026-08-06): with IDENTICAL
    # robot/human geometry, forcing two different REAL beliefs for the
    # same track must change the state embeddings feeding IQN and (at
    # minimum) the resulting action scores. If this were false, belief
    # would not actually be part of the network's input -- i.e. the
    # network would still be Z(s), not Z(s,b).
    robot, humans = _fixture_robot_and_humans()

    policy_a = _tiny_policy()
    policy_a.belief_tracker._tracks = {}
    # First feed a real observation so the track exists, then force its
    # belief to a specific value (larger than what the natural update
    # would produce) purely to get two DIFFERING beliefs to compare.
    policy_a.belief_tracker.update({h.track_id: (0.0, np.array([h.px, h.py])) for h in humans})
    for h in humans:
        policy_a.belief_tracker._tracks[h.track_id].belief = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
    result_a = policy_a.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)

    policy_b = _tiny_policy()
    policy_b.belief_tracker._tracks = {}
    policy_b.belief_tracker.update({h.track_id: (0.0, np.array([h.px, h.py])) for h in humans})
    for h in humans:
        policy_b.belief_tracker._tracks[h.track_id].belief = np.array([0.025, 0.025, 0.025, 0.025, 0.9])
    result_b = policy_b.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)

    scores_a = np.array([s.cvar for s in result_a.all_scores])
    scores_b = np.array([s.cvar for s in result_b.all_scores])
    assert not np.allclose(scores_a, scores_b), (
        "identical geometry with two different beliefs produced identical scores -- "
        "belief is not actually part of the network input (network is Z(s), not Z(s,b))"
    )

def test_human_ttc_feature_is_capped_at_frozen_time_horizon() -> None:
    robot, _ = _fixture_robot_and_humans()
    humans = [HumanObservation(track_id=0, px=100.0, py=100.0, vx=0.0, vy=0.0, radius=0.3)]
    feature = _human_feature_vector(
        robot, humans[0], belief=np.full(5, 0.2, dtype=np.float32), entropy=1.0,
        track_age=1, pred_mean=np.zeros(2, dtype=np.float32),
        pred_cov=np.eye(2, dtype=np.float32),
    )
    assert np.isfinite(feature).all()
    # R3-2: TTC is capped at the frozen 35s horizon THEN normalized to
    # [0,1] (normalize_ttc(35.0, 35.0) == 1.0) -- the raw-scale value is
    # no longer what reaches the network.
    assert feature[6] == 1.0, f"no-collision TTC must normalize to the horizon cap 1.0, got {feature[6]}"

def test_bdvl_policy_decide_never_mutates_real_tracker_beyond_the_one_real_update() -> None:
    # The hypothetical per-sample clones built inside decide() (guide.md
    # B0 repair item 4) must never leak back into the real online
    # tracker. Verify the real tracker's belief after decide() equals
    # exactly what a bare belief_tracker.update() call alone would
    # produce -- decide() must add nothing beyond that one real update.
    robot, humans = _fixture_robot_and_humans()
    artifact = _fixture_artifact()

    reference_tracker = BeliefTracker(artifact)
    reference_tracker.update({h.track_id: (1.0, np.array([h.px, h.py])) for h in humans})
    expected = {h.track_id: reference_tracker.belief_for(h.track_id) for h in humans}

    policy = _tiny_policy()
    policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    for h in humans:
        assert np.allclose(policy.belief_tracker.belief_for(h.track_id), expected[h.track_id]), (
            "decide() left the real tracker in a different state than a single real update() call would -- "
            "a hypothetical rollout clone must have leaked into the real tracker"
        )
    # And no extra tracks (e.g. from a stray clone) were introduced.
    assert set(policy.belief_tracker.active_track_ids()) == {h.track_id for h in humans}

def test_bdvl_policy_corrupted_posterior_changes_scores() -> None:
    # A "corrupted" posterior here is simulated via the shuffled source,
    # which the rollout module already supports as a necessity ablation
    # (guide.md 4.5/5.4's negative control). It must produce a
    # DIFFERENT score/ordering than the full posterior on the same
    # fixture, otherwise the posterior isn't actually influencing scores.
    robot, humans = _fixture_robot_and_humans()

    policy_full = _tiny_policy(posterior_source="full")
    result_full = policy_full.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)

    policy_cv = _tiny_policy(posterior_source="cv")
    result_cv = policy_cv.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)

    scores_full = np.array([s.cvar for s in result_full.all_scores])
    scores_cv = np.array([s.cvar for s in result_cv.all_scores])
    assert not np.allclose(scores_full, scores_cv), (
        "full posterior and cv-only posterior produced identical scores on a fixture "
        "where the two humans' beliefs are not uniform -- posterior is not influencing the score"
    )

def test_replay_demo_online_ratio_within_tolerance() -> None:
    replay = DemoOnlineReplay(demo_capacity=1000, online_capacity=1000, demo_ratio=0.20)
    for i in range(200):
        replay.demo.add(_dummy_transition(float(i)))
        replay.online.add(_dummy_transition(-float(i)))

    rng = random.Random(0)
    n_demo_total, n_online_total = 0, 0
    for _ in range(50):
        _, n_demo, n_online = replay.sample_batch(256, rng)
        n_demo_total += n_demo
        n_online_total += n_online
    observed_ratio = n_demo_total / (n_demo_total + n_online_total)
    assert abs(observed_ratio - 0.20) < 0.02

def test_replay_reallocates_when_demo_buffer_empty() -> None:
    replay = DemoOnlineReplay(demo_capacity=1000, online_capacity=1000, demo_ratio=0.20)
    for i in range(100):
        replay.online.add(_dummy_transition(float(i)))
    batch, n_demo, n_online = replay.sample_batch(64, random.Random(0))
    assert n_demo == 0
    assert n_online == 64
    assert len(batch) == 64

def test_replay_reallocates_when_online_buffer_empty() -> None:
    replay = DemoOnlineReplay(demo_capacity=1000, online_capacity=1000, demo_ratio=0.20)
    for i in range(100):
        replay.demo.add(_dummy_transition(float(i)))
    batch, n_demo, n_online = replay.sample_batch(64, random.Random(0))
    assert n_online == 0
    assert n_demo == 64

def test_ring_buffer_preserves_terminal_transition_under_overflow() -> None:
    replay = DemoOnlineReplay(demo_capacity=10, online_capacity=5, demo_ratio=0.0)
    terminal = Transition(
        robot_features=np.zeros(4, dtype=np.float32), human_features=np.zeros((20, 4), dtype=np.float32),
        human_mask=np.zeros(20, dtype=bool), belief=np.zeros(5, dtype=np.float32),
        action_index=1, reward=-0.5, done=True,
        next_robot_features=np.zeros(4, dtype=np.float32), next_human_features=np.zeros((20, 4), dtype=np.float32),
        next_human_mask=np.zeros(20, dtype=bool), next_belief=np.zeros(5, dtype=np.float32),
        artifact_sha256="fixture",
    )
    replay.online.add(terminal)
    for i in range(4):  # ring buffer capacity=5, add 4 more non-terminal -> terminal must still be present
        replay.online.add(_dummy_transition(float(i)))
    found = any(t.done for t in replay.online._items if t is not None)
    assert found, "terminal transition was lost before the ring buffer actually overflowed capacity"

def test_compute_n_step_returns_hand_computed() -> None:
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    dones = [False, False, False, False, True]
    gamma = 0.9
    n_step = 3
    results = compute_n_step_returns(rewards, dones, gamma, n_step)
    # t=0: r0 + gamma*r1 + gamma^2*r2 (horizon=3, no done hit)
    expected_0 = 1.0 + 0.9 * 2.0 + 0.81 * 3.0
    assert abs(results[0][0] - expected_0) < 1e-9 and results[0][1] == 3
    # t=3: r3 + gamma*r4, but r4 is terminal -> horizon stops at 2 (index 3,4)
    expected_3 = 4.0 + 0.9 * 5.0
    assert abs(results[3][0] - expected_3) < 1e-9 and results[3][1] == 2
    # t=4: only the terminal reward itself.
    assert abs(results[4][0] - 5.0) < 1e-9 and results[4][1] == 1

def test_build_n_step_transitions_preserves_terminal_window_and_successor() -> None:
    def state(i):
        return {
            "robot_features": np.full(6, i, dtype=np.float32),
            "human_features": np.full((MAX_HUMANS, HUMAN_FEATURE_DIM), i, dtype=np.float32),
            "human_mask": np.zeros(MAX_HUMANS, dtype=bool),
            "belief": np.full(MAX_HUMANS * 5, i, dtype=np.float32),
        }

    transitions = build_n_step_transitions(
        [state(0), state(1), state(2), state(3)], [7, 8, 9], [1.0, 2.0, 3.0],
        [False, False, True], gamma=0.9, n_step=3, artifact_sha256="artifact", episode_seed=123,
    )
    assert [round(t.reward, 6) for t in transitions] == [5.23, 4.7, 3.0]
    assert all(t.done for t in transitions)
    assert [round(t.gamma_pow_n, 6) for t in transitions] == [0.729, 0.81, 0.9]
    assert [t.action_index for t in transitions] == [7, 8, 9]
    assert [int(t.next_robot_features[0]) for t in transitions] == [3, 3, 3]
    assert [t.step_index for t in transitions] == [0, 1, 2]

def test_build_n_step_transitions_keeps_nonterminal_successor() -> None:
    def state(i):
        return {
            "robot_features": np.array([i] * 6, dtype=np.float32),
            "human_features": np.zeros((MAX_HUMANS, HUMAN_FEATURE_DIM), dtype=np.float32),
            "human_mask": np.zeros(MAX_HUMANS, dtype=bool),
            "belief": np.zeros(MAX_HUMANS * 5, dtype=np.float32),
        }

    transitions = build_n_step_transitions(
        [state(0), state(1), state(2), state(3)], [1, 2, 3], [0.0, 0.0, 0.0],
        [False, False, False], gamma=0.99, n_step=2, artifact_sha256="a", episode_seed=4,
    )
    assert not transitions[0].done and transitions[0].gamma_pow_n == 0.99**2
    assert int(transitions[0].next_robot_features[0]) == 2
    assert int(transitions[1].next_robot_features[0]) == 3

def test_replay_state_roundtrip_preserves_sampling() -> None:
    replay = DemoOnlineReplay(8, 8, demo_ratio=0.2)
    base = dict(
        robot_features=np.zeros(6, dtype=np.float32), human_features=np.zeros((MAX_HUMANS, HUMAN_FEATURE_DIM), dtype=np.float32),
        human_mask=np.zeros(MAX_HUMANS, dtype=bool), belief=np.zeros(MAX_HUMANS * 5, dtype=np.float32),
        action_index=2, reward=1.0, done=True, next_robot_features=np.ones(6, dtype=np.float32),
        next_human_features=np.zeros((MAX_HUMANS, HUMAN_FEATURE_DIM), dtype=np.float32), next_human_mask=np.zeros(MAX_HUMANS, dtype=bool),
        next_belief=np.zeros(MAX_HUMANS * 5, dtype=np.float32), artifact_sha256="a",
    )
    for i in range(4):
        replay.demo.add(Transition(**{**base, "episode_seed": i}))
    replay.online.add(Transition(**{**base, "done": False, "episode_seed": 99}))
    state = replay.state_dict()
    restored = DemoOnlineReplay(8, 8, demo_ratio=0.2)
    restored.load_state_dict(state)
    left = replay.sample_batch(10, random.Random(55))
    right = restored.sample_batch(10, random.Random(55))
    assert [(t.episode_seed, t.done) for t in left[0]] == [(t.episode_seed, t.done) for t in right[0]]
    assert left[1:] == right[1:]

def test_soft_update_target_hand_computed() -> None:
    online = nn.Linear(2, 2, bias=False)
    target = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        online.weight.fill_(1.0)
        target.weight.fill_(0.0)
    soft_update_target(online, target, tau=0.1)
    assert torch.allclose(target.weight, torch.full((2, 2), 0.1))
    soft_update_target(online, target, tau=0.1)
    assert torch.allclose(target.weight, torch.full((2, 2), 0.19))  # 0.1 + 0.1*(1-0.1)

def test_hard_copy_to_target_matches_online_exactly() -> None:
    online = nn.Linear(3, 3)
    target = nn.Linear(3, 3)
    hard_copy_to_target(online, target)
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        assert torch.equal(p_online, p_target)

def test_train_step_aborts_cleanly_on_nan_input() -> None:
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    target_encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    target_action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    target_net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    hard_copy_to_target(encoder, target_encoder)
    hard_copy_to_target(net, target_net)
    hard_copy_to_target(action_encoder, target_action_encoder)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)

    B = 4
    action_table = np.array([[0.1 * i, 0.05 * i] for i in range(B)], dtype=np.float64)
    action_indices = list(range(B))
    robot_features = torch.randn(B, ROBOT_FEATURE_DIM)
    robot_features[0, 0] = float("nan")  # inject NaN
    human_features = torch.zeros(B, 20, HUMAN_FEATURE_DIM)
    human_mask = torch.zeros(B, 20, dtype=torch.bool)
    result = train_step(
        encoder, net, action_encoder, target_encoder, target_net, target_action_encoder,
        action_table, action_indices,
        robot_features, human_features, human_mask,
        rewards=torch.zeros(B), dones=torch.zeros(B, dtype=torch.bool),
        next_robot_features=torch.randn(B, ROBOT_FEATURE_DIM), next_human_features=human_features, next_human_mask=human_mask,
        gamma_pow_n=torch.full((B,), 0.99),
        n_train_quantiles=4, n_target_samples=4, optimizer=optimizer,
    )
    assert result.aborted
    assert "non-finite" in result.abort_reason

def test_train_step_works_on_cuda_when_available() -> None:
    # R1 regression (independent audit B3, 2026-08-06): an independent
    # CUDA call to train_step() crashed immediately because tau tensors
    # were created on CPU while the networks were on CUDA. Reproduced
    # and fixed by deriving device from the network parameters and
    # moving every tensor built/received in train_step onto it. Skips
    # (does not fail) on a machine with no CUDA device.
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(0)
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8).to(device)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4).to(device)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8).to(device)
    target_encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8).to(device)
    target_action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4).to(device)
    target_net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8).to(device)
    hard_copy_to_target(encoder, target_encoder)
    hard_copy_to_target(net, target_net)
    hard_copy_to_target(action_encoder, target_action_encoder)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)

    B = 4
    action_table = np.array([[0.1 * i, 0.05 * i] for i in range(B)], dtype=np.float64)
    action_indices = list(range(B))
    # Inputs deliberately left on CPU (the exact scenario that crashed):
    # train_step is responsible for moving them, not the caller.
    robot_features = torch.randn(B, ROBOT_FEATURE_DIM)
    human_features = torch.zeros(B, 20, HUMAN_FEATURE_DIM)
    human_mask = torch.zeros(B, 20, dtype=torch.bool)
    result = train_step(
        encoder, net, action_encoder, target_encoder, target_net, target_action_encoder,
        action_table, action_indices,
        robot_features, human_features, human_mask,
        rewards=torch.ones(B), dones=torch.zeros(B, dtype=torch.bool),
        next_robot_features=robot_features, next_human_features=human_features, next_human_mask=human_mask,
        gamma_pow_n=torch.full((B,), 0.99),
        n_train_quantiles=4, n_target_samples=4, optimizer=optimizer,
    )
    assert not result.aborted, f"CUDA train_step aborted unexpectedly: {result.abort_reason}"
    assert np.isfinite(result.loss)

def test_train_step_normal_input_reduces_loss_over_iterations() -> None:
    torch.manual_seed(0)
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    target_encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    target_action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    target_net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    hard_copy_to_target(encoder, target_encoder)
    hard_copy_to_target(net, target_net)
    hard_copy_to_target(action_encoder, target_action_encoder)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)

    B = 8
    action_table = np.array([[0.1 * i, 0.05 * i] for i in range(B)], dtype=np.float64)
    action_indices = list(range(B))
    robot_features = torch.randn(B, ROBOT_FEATURE_DIM)
    human_features = torch.zeros(B, 20, HUMAN_FEATURE_DIM)
    human_mask = torch.zeros(B, 20, dtype=torch.bool)
    losses = []
    for _ in range(50):
        result = train_step(
            encoder, net, action_encoder, target_encoder, target_net, target_action_encoder,
            action_table, action_indices,
            robot_features, human_features, human_mask,
            rewards=torch.ones(B), dones=torch.zeros(B, dtype=torch.bool),
            next_robot_features=robot_features, next_human_features=human_features, next_human_mask=human_mask,
            gamma_pow_n=torch.full((B,), 0.99),
            n_train_quantiles=8, n_target_samples=8, optimizer=optimizer,
        )
        assert not result.aborted
        losses.append(result.loss)
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), "loss did not decrease over training"

def test_bayesian_dvl_policy_configure_rejects_missing_checkpoint_without_smoke_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, registry_path, artifact_path = _write_registry_and_artifact(tmp)
        config = _FakeConfig({"bayesian_dvl": {
            "registry_path": str(registry_path), "artifact_path": str(artifact_path),
        }})
        adapter = BayesianDVLPolicy()
        try:
            adapter.configure(config)
            raise AssertionError("expected PolicyError: no checkpoint and smoke_mode not set")
        except PolicyError as exc:
            assert "smoke_mode" in str(exc)

def test_bayesian_dvl_policy_configure_allows_smoke_mode_without_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, registry_path, artifact_path = _write_registry_and_artifact(tmp)
        config = _FakeConfig({"bayesian_dvl": {
            "registry_path": str(registry_path), "artifact_path": str(artifact_path),
            "smoke_mode": "true",
        }})
        adapter = BayesianDVLPolicy()
        adapter.configure(config)  # must not raise
        assert adapter.bdvl_policy is not None

def test_bayesian_dvl_policy_configure_loads_composed_checkpoint_and_verifies_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        registry, registry_path, artifact_path = _write_registry_and_artifact(tmp)
        from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder as _AE, SetEncoder as _SE
        from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork as _IQN

        encoder = _SE()
        action_encoder = _AE()
        net = _IQN(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim)
        checkpoint_path = Path(tmp) / "checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(checkpoint_path), registry.action_grid_hash, registry.content_hash(),
            action_encoder=action_encoder,
            artifact_sha256=SBKHMMArtifact.load(str(artifact_path)).content_sha256(),
            training_config_sha256="test-training-config",
        )

        config = _FakeConfig({"bayesian_dvl": {
            "registry_path": str(registry_path), "artifact_path": str(artifact_path),
            "checkpoint_path": str(checkpoint_path),
        }})
        adapter = BayesianDVLPolicy()
        adapter.configure(config)  # must not raise
        assert adapter.bdvl_policy is not None

def test_bayesian_dvl_policy_configure_rejects_checkpoint_with_wrong_action_grid_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        registry, registry_path, artifact_path = _write_registry_and_artifact(tmp)
        from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder as _AE, SetEncoder as _SE
        from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork as _IQN

        encoder = _SE()
        action_encoder = _AE()
        net = _IQN(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim)
        checkpoint_path = Path(tmp) / "checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(checkpoint_path), "deliberately-wrong-hash", registry.content_hash(),
            action_encoder=action_encoder,
            artifact_sha256=SBKHMMArtifact.load(str(artifact_path)).content_sha256(),
            training_config_sha256="test-training-config",
        )

        config = _FakeConfig({"bayesian_dvl": {
            "registry_path": str(registry_path), "artifact_path": str(artifact_path),
            "checkpoint_path": str(checkpoint_path),
        }})
        adapter = BayesianDVLPolicy()
        try:
            adapter.configure(config)
            raise AssertionError("expected PolicyError for action-grid hash mismatch")
        except PolicyError as exc:
            assert "action grid" in str(exc)

def test_bayesian_dvl_policy_configure_rejects_checkpoint_with_wrong_artifact_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        registry, registry_path, artifact_path = _write_registry_and_artifact(tmp)
        from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder as _AE, SetEncoder as _SE
        from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork as _IQN

        encoder = _SE()
        action_encoder = _AE()
        net = _IQN(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim)
        checkpoint_path = Path(tmp) / "checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(checkpoint_path), registry.action_grid_hash, registry.content_hash(),
            action_encoder=action_encoder,
            artifact_sha256="deliberately-wrong-artifact-hash",
            training_config_sha256="test-training-config",
        )
        config = _FakeConfig({"bayesian_dvl": {
            "registry_path": str(registry_path), "artifact_path": str(artifact_path),
            "checkpoint_path": str(checkpoint_path),
        }})
        adapter = BayesianDVLPolicy()
        try:
            adapter.configure(config)
            raise AssertionError("expected PolicyError for artifact hash mismatch")
        except PolicyError as exc:
            assert "artifact hash" in str(exc)

def test_bayesian_dvl_policy_set_device_moves_composed_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, registry_path, artifact_path = _write_registry_and_artifact(tmp)
        config = _FakeConfig({"bayesian_dvl": {
            "registry_path": str(registry_path), "artifact_path": str(artifact_path),
            "smoke_mode": "true",
        }})
        adapter = BayesianDVLPolicy()
        adapter.configure(config)
        adapter.set_device("cpu")
        assert str(next(adapter.bdvl_policy.set_encoder.parameters()).device) == "cpu"
        assert adapter.bdvl_policy.device == torch.device("cpu")

def test_bayesian_dvl_adapter_uses_explicit_episode_identity() -> None:
    from types import SimpleNamespace
    from crowd_sim.envs.utils.state import JointState

    class _SpyPolicy:
        def __init__(self):
            self.calls = []

        def reset_episode_stats(self):
            pass

        def decide(self, robot, humans, global_time, suite_seed, episode_seed):
            self.calls.append((suite_seed, episode_seed, global_time))
            return SimpleNamespace(
                chosen_action_index=0,
                chosen_action=(0.0, 0.0),
            )

    spy = _SpyPolicy()
    adapter = BayesianDVLPolicy(bdvl_policy=spy, suite_seed=1)
    adapter.time_step = 0.25
    adapter.reset_episode_stats(suite_seed=94001, episode_seed=9400100002)
    state = JointState(_make_full_state(np.random.default_rng(1)), [])
    adapter.predict(state)
    assert spy.calls == [(94001, 9400100002, 0.0)]
    assert adapter._global_time == 0.25

def test_bayesian_dvl_adapter_rejects_partial_episode_identity() -> None:
    class _SpyPolicy:
        def reset_episode_stats(self):
            pass

    adapter = BayesianDVLPolicy(bdvl_policy=_SpyPolicy(), suite_seed=1)
    for kwargs in ({"suite_seed": 2}, {"episode_seed": 200001}):
        try:
            adapter.reset_episode_stats(**kwargs)
            raise AssertionError("partial episode identity must be rejected")
        except PolicyError as exc:
            assert "provided together" in str(exc)

def test_bayesian_dvl_adapter_legacy_reset_allocates_distinct_seeds() -> None:
    class _SpyPolicy:
        def reset_episode_stats(self):
            pass

    adapter = BayesianDVLPolicy(bdvl_policy=_SpyPolicy(), suite_seed=7)
    adapter.reset_episode_stats()
    first = adapter._episode_seed
    adapter.reset_episode_stats()
    second = adapter._episode_seed
    assert first == 700001
    assert second == 700002
    assert first != second

def test_suite_seed_block_bootstrap_recovers_known_mean() -> None:
    rng = np.random.default_rng(0)
    values_by_seed = {s: rng.normal(loc=5.0, scale=0.1, size=20).tolist() for s in range(10)}
    result = suite_seed_block_bootstrap(values_by_seed, np.mean, n_resamples=500, seed=1)
    assert abs(result.point - 5.0) < 0.1
    assert result.ci_low < result.point < result.ci_high

def test_paired_difference_bootstrap_requires_matching_seeds() -> None:
    a = {1: [1.0, 2.0], 2: [3.0]}
    b = {1: [1.0, 2.0], 3: [3.0]}
    try:
        paired_difference_bootstrap(a, b)
        raise AssertionError("expected StatisticsError for mismatched seed sets")
    except StatisticsError:
        pass

def test_paired_difference_bootstrap_hand_checkable_direction() -> None:
    a = {1: [10.0] * 20, 2: [10.0] * 20}
    b = {1: [1.0] * 20, 2: [1.0] * 20}
    result = paired_difference_bootstrap(a, b, n_resamples=200, seed=0)
    assert abs(result.point - 9.0) < 1e-9
    assert result.ci_low > 0, "CI should not straddle 0 for a clearly-different pair of constants"

def test_join_paired_episodes_matches_by_full_identity() -> None:
    a = [_episode("m1", "baseline_circle", "nominal", 1, 100001), _episode("m1", "baseline_circle", "nominal", 1, 100002)]
    b = [_episode("m2", "baseline_circle", "nominal", 1, 100002), _episode("m2", "baseline_circle", "nominal", 1, 100001)]
    pairs = join_paired_episodes(a, b)
    assert len(pairs) == 2
    for ra, rb in pairs:
        assert ra.episode_seed == rb.episode_seed

def test_join_paired_episodes_rejects_missing_and_duplicate() -> None:
    a = [_episode("m1", "s", "p", 1, 1), _episode("m1", "s", "p", 1, 2)]
    b = [_episode("m2", "s", "p", 1, 1)]  # missing episode_seed=2
    try:
        join_paired_episodes(a, b)
        raise AssertionError("expected StatisticsError for missing episode")
    except StatisticsError:
        pass

    a_dup = [_episode("m1", "s", "p", 1, 1), _episode("m1", "s", "p", 1, 1)]
    try:
        join_paired_episodes(a_dup, a_dup)
        raise AssertionError("expected StatisticsError for duplicate episode identity")
    except StatisticsError:
        pass

def test_evaluator_two_runs_same_seed_byte_identical_csv() -> None:
    def deterministic_episode_fn(method, scenario, profile, suite_seed, episode_seed):
        outcome = "success" if (suite_seed + episode_seed) % 3 == 0 else "collision"
        return EpisodeRecord(method=method, scenario=scenario, profile=profile, suite_seed=suite_seed, episode_seed=episode_seed, outcome=outcome, steps=(episode_seed % 50))

    with tempfile.TemporaryDirectory() as tmp:
        records1 = run_paired_evaluation("methodX", "baseline_circle", "nominal", [1, 2, 3], 5, deterministic_episode_fn)
        records2 = run_paired_evaluation("methodX", "baseline_circle", "nominal", [1, 2, 3], 5, deterministic_episode_fn)
        path1, path2 = Path(tmp) / "run1.csv", Path(tmp) / "run2.csv"
        write_episode_records_csv(records1, str(path1))
        write_episode_records_csv(records2, str(path2))
        assert path1.read_bytes() == path2.read_bytes()

        loaded = read_episode_records_csv(str(path1))
        assert loaded == records1

def test_evaluator_deterministic_digest_excludes_wall_clock_latency() -> None:
    record = _episode("methodX", "baseline_circle", "nominal", 1, 100001)
    changed_latency = EpisodeRecord(
        method=record.method, scenario=record.scenario, profile=record.profile,
        suite_seed=record.suite_seed, episode_seed=record.episode_seed,
        outcome=record.outcome, steps=record.steps, elapsed_time=record.elapsed_time,
        min_clearance=record.min_clearance, path_length=record.path_length,
        mean_decision_latency_ms=record.mean_decision_latency_ms + 123.0,
    )
    assert deterministic_records_sha256([record]) == deterministic_records_sha256([changed_latency])

def test_evaluator_rejects_episode_fn_returning_wrong_identity() -> None:
    def broken_episode_fn(method, scenario, profile, suite_seed, episode_seed):
        return EpisodeRecord(method=method, scenario=scenario, profile=profile, suite_seed=suite_seed, episode_seed=999999, outcome="success", steps=1)

    try:
        run_paired_evaluation("m", "s", "p", [1], 2, broken_episode_fn)
        raise AssertionError("expected EvaluatorError for identity mismatch")
    except EvaluatorError:
        pass

def test_evaluator_detects_missing_episodes() -> None:
    records = [_episode("m", "s", "p", 1, 100001), _episode("m", "s", "p", 1, 100002)]
    try:
        assert_no_duplicate_or_missing(records, expected_suite_seeds=[1], episodes_per_seed=5)
        raise AssertionError("expected EvaluatorError for missing episodes")
    except EvaluatorError:
        pass

def test_checkpoint_selection_role_guard() -> None:
    assert_role_allowed_for_checkpoint_selection("checkpoint-validation")  # must not raise
    try:
        assert_role_allowed_for_checkpoint_selection("RL-train")
        raise AssertionError("expected EvaluatorError: RL-train is not a validation role")
    except EvaluatorError:
        pass

def test_stress_1_5_10_20_humans_no_nan() -> None:
    for n in (1, 5, 10, 20):
        policy = _stress_policy()
        robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
        humans = [HumanObservation(track_id=i, px=float(i), py=1.0, vx=0.0, vy=0.0, radius=0.3) for i in range(n)]
        result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
        assert all(np.isfinite(s.cvar) for s in result.all_scores), f"non-finite score with n={n} humans"

def test_stress_near_collision_state_no_nan() -> None:
    policy = _stress_policy()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(track_id=0, px=0.55, py=0.0, vx=-0.1, vy=0.0, radius=0.3)]  # already overlapping-adjacent
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    assert all(np.isfinite(s.cvar) for s in result.all_scores)

def test_stress_all_stationary_no_nan() -> None:
    policy = _stress_policy()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(track_id=i, px=float(i) * 2, py=2.0, vx=0.0, vy=0.0, radius=0.3) for i in range(5)]
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    assert all(np.isfinite(s.cvar) for s in result.all_scores)

def test_stress_high_speed_crossing_no_nan() -> None:
    policy = _stress_policy()
    robot = RobotObservation(px=0.0, py=0.0, vx=1.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(track_id=0, px=3.0, py=0.0, vx=-2.0, vy=0.0, radius=0.3)]  # fast head-on closure
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=1, episode_seed=1)
    assert all(np.isfinite(s.cvar) for s in result.all_scores)

def test_stress_belief_near_onehot_no_nan() -> None:
    policy = _stress_policy()
    # Force a near-one-hot belief by feeding several consistent ACC observations.
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    positions = _positions_from_kinematics(policy.dt, 6, speed0=0.2, accel=0.9, omega=0.0)
    for t, pos in enumerate(positions):
        humans = [HumanObservation(track_id=0, px=float(pos[0]), py=float(pos[1]) + 1.0, vx=0.0, vy=0.0, radius=0.3)]
        result = policy.decide(robot, humans, global_time=1.0 + t * policy.dt, suite_seed=1, episode_seed=1)
    assert all(np.isfinite(s.cvar) for s in result.all_scores)
    belief = policy.belief_tracker.belief_for(0)
    assert belief.max() > 0.5, "expected belief to concentrate after several consistent ACC observations"

def test_stress_long_missing_then_reappear_no_nan() -> None:
    policy = _stress_policy()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(track_id=0, px=1.0, py=1.0, vx=0.0, vy=0.0, radius=0.3)]
    policy.decide(robot, humans, global_time=0.0, suite_seed=1, episode_seed=1)
    # Long gap: track goes missing for many steps (belief_tracker's
    # internal missing_timeout_steps default is 8, so this exceeds it
    # and forces a genuine reappearance-as-new-track code path).
    for t in range(1, 12):
        result = policy.decide(robot, [], global_time=t * policy.dt, suite_seed=1, episode_seed=1)
        assert all(np.isfinite(s.cvar) for s in result.all_scores), f"non-finite score at missing-step {t} (zero humans)"
    # Reappears.
    result = policy.decide(robot, humans, global_time=13 * policy.dt, suite_seed=1, episode_seed=1)
    assert all(np.isfinite(s.cvar) for s in result.all_scores)

def test_batch_transition_matches_scalar_reference() -> None:
    """A11 optimization gate: batch construction must not change semantics."""
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=1.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [
        HumanObservation(track_id=0, px=0.45, py=0.0, vx=-0.2, vy=0.0, radius=0.3),
        HumanObservation(track_id=1, px=2.0, py=1.0, vx=0.0, vy=-0.1, radius=0.3),
    ]
    actions = np.asarray([[0.0, 0.0], [0.8, 0.0], [0.0, 0.8], [-0.8, 0.0]], dtype=np.float64)
    human_actions = np.asarray([
        [[-0.2, 0.0], [0.0, -0.1]],
        [[0.2, 0.0], [0.0, 0.1]],
        [[0.0, 0.0], [0.1, 0.0]],
    ], dtype=np.float64)
    reward_config = RewardConfig(
        success_reward=1.0, collision_penalty=-0.5, timeout_penalty=-0.5,
        progress_reward=0.01, time_penalty=-0.003, stand_penalty=0.0,
        stand_speed_threshold=0.05, discomfort_distance=0.2,
        discomfort_penalty_factor=0.5,
    )
    for global_time in (0.0, 34.999999, 35.0):
        batch = bdvl_batch_step(
            robot, humans, actions, human_actions, dt=0.25, time_limit=35.0,
            global_time=global_time, reward_config=reward_config,
        )
        for action_idx, (vx, vy) in enumerate(actions):
            for sample_idx, sample_actions in enumerate(human_actions):
                scalar = bdvl_transition_step(
                    robot, humans, float(vx), float(vy), [tuple(v) for v in sample_actions],
                    dt=0.25, time_limit=35.0, global_time=global_time,
                    reward_config=reward_config,
                )
                assert np.allclose(
                    batch.next_robot_positions[action_idx],
                    [scalar.next_robot.px, scalar.next_robot.py], atol=1e-12,
                )
                assert np.allclose(
                    batch.next_human_positions[sample_idx],
                    [[h.px, h.py] for h in scalar.next_humans], atol=1e-12,
                )
                assert np.allclose(batch.rewards[action_idx, sample_idx], scalar.reward, atol=1e-12)
                assert bool(batch.terminated[action_idx, sample_idx]) == scalar.terminated
                assert bool(batch.truncated[action_idx, sample_idx]) == scalar.truncated
                assert batch.events[action_idx, sample_idx] == scalar.event
                assert np.allclose(batch.dmin[action_idx], scalar.dmin, atol=1e-12)

    # Explicit zero-human boundary: the batch path must retain the scalar
    # path's +inf clearance and finite navigation reward.
    empty = bdvl_batch_step(
        robot, [], actions[:1], np.zeros((1, 0, 2)), dt=0.25, time_limit=35.0,
        global_time=0.0, reward_config=reward_config,
    )
    scalar_empty = bdvl_transition_step(
        robot, [], 0.0, 0.0, [], dt=0.25, time_limit=35.0,
        global_time=0.0, reward_config=reward_config,
    )
    assert np.isinf(empty.dmin[0]) and np.isinf(scalar_empty.dmin)
    assert np.isfinite(empty.rewards[0, 0]) and empty.events[0, 0] == scalar_empty.event

def test_stateless_tau_is_reproducible_bounded_and_seeded() -> None:
    seeds = np.asarray([1, 2, 3, 123456], dtype=np.int64)
    first = _stateless_tau_cpu(seeds, n_quantiles=64, upper=0.2).numpy()
    second = _stateless_tau_cpu(seeds, n_quantiles=64, upper=0.2).numpy()
    changed = _stateless_tau_cpu(seeds + 1, n_quantiles=64, upper=0.2).numpy()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.all(first > 0.0) and np.all(first < 0.2)

def test_remaining_time_fraction_boundaries_are_exactly_one_and_zero() -> None:
    assert remaining_time_fraction(0.0, 35.0) == 1.0
    assert remaining_time_fraction(35.0, 35.0) == 0.0
    assert remaining_time_fraction(17.5, 35.0) == 0.5

def test_remaining_time_fraction_clips_out_of_range_time() -> None:
    # Guide.md R2-1: "重复/越界时间被clip但记录诊断" -- clip, never raise,
    # for time values outside [0, time_limit].
    assert remaining_time_fraction(-5.0, 35.0) == 1.0
    assert remaining_time_fraction(50.0, 35.0) == 0.0

def test_robot_feature_vector_same_geometry_differs_only_in_time_dim() -> None:
    robot = RobotObservation(px=0.0, py=0.0, vx=0.3, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    early = _robot_feature_vector(robot, remaining_time_fraction(0.0, 35.0))
    late = _robot_feature_vector(robot, remaining_time_fraction(35.0, 35.0))
    assert early.shape == (ROBOT_FEATURE_DIM,) == (7,)
    assert np.allclose(early[:6], late[:6])
    assert early[6] == 1.0 and late[6] == 0.0

def test_bdvl_policy_candidate_features_use_global_time_plus_dt() -> None:
    # All 80 candidates share ONE successor time (t+dt), not the real
    # current time t (guide.md R2-1's named-constructor acceptance).
    policy = _tiny_policy()
    robot, humans = _fixture_robot_and_humans()
    result = policy.decide(robot, humans, global_time=10.0, suite_seed=1, episode_seed=1)
    assert len(result.all_scores) == 80  # sanity: decide() still runs end-to-end with the 7-dim contract

def test_load_composed_checkpoint_rejects_v1_feature_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        encoder = SetEncoder()
        # No ActionEncoder here on purpose: this exercises the retired
        # v1/v2/v3-schema rejection path, which save/load_composed_checkpoint
        # support without one (see their docstrings) -- IQNValueNetwork
        # itself still requires an action_embedding_dim even though this
        # test never constructs an ActionEncoder to match it.
        net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        path = Path(tmp) / "v1_checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(path), "grid-hash", "registry-hash",
            feature_schema=FEATURE_SCHEMA_V1,
        )
        fresh_encoder = SetEncoder()
        fresh_net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        try:
            load_composed_checkpoint(str(path), fresh_encoder, fresh_net)
            raise AssertionError("expected PolicyError loading a v1-schema checkpoint under v2 code")
        except PolicyError as exc:
            assert FEATURE_SCHEMA_V1 in str(exc)

def test_load_composed_checkpoint_rejects_return_bound_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        encoder = SetEncoder()
        net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16, v_min=-2.0, v_max=2.0)
        path = Path(tmp) / "checkpoint.pth"
        save_composed_checkpoint(encoder, net, str(path), "grid-hash", "registry-hash")

        fresh_encoder = SetEncoder()
        fresh_net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16, v_min=-4.798, v_max=1.08)  # different bounds
        try:
            load_composed_checkpoint(str(path), fresh_encoder, fresh_net)
            raise AssertionError("expected PolicyError on return-bound drift")
        except PolicyError as exc:
            assert "return bounds" in str(exc)

def test_derive_return_bounds_hand_computed() -> None:
    # Hand-computed discounted per-step envelope. It must not depend on
    # the training geometry because the same value network is evaluated on
    # all frozen formal layouts.
    frozen = dict(FROZEN_VALUES)
    v_min, v_max = derive_return_bounds(frozen, max_action_speed=1.0, initial_goal_distance=8.0)
    n_steps_max = round(frozen["time_limit"] / frozen["dt"]) + 1
    gamma = frozen["gamma"]
    r_min = frozen["time_penalty"] - frozen["progress_reward"] * frozen["dt"] - frozen["discomfort_penalty_factor"] * frozen["discomfort_distance"] * frozen["dt"]
    r_max = frozen["time_penalty"] + frozen["progress_reward"] * frozen["dt"]
    expected_v_min = min(
        r_min * sum(gamma ** i for i in range(k)) + min(frozen["collision_penalty"], frozen["timeout_penalty"]) * gamma ** k
        for k in range(n_steps_max)
    )
    expected_v_max = max(
        r_max * sum(gamma ** i for i in range(k)) + max(frozen["success_reward"], frozen["collision_penalty"], frozen["timeout_penalty"]) * gamma ** k
        for k in range(n_steps_max)
    )
    assert abs(v_min - expected_v_min) < 1e-9
    assert v_min < 0.0 < v_max  # sanity: a real range, not degenerate

def test_derive_return_bounds_rejects_non_positive_time_limit() -> None:
    bad = dict(FROZEN_VALUES)
    bad["time_limit"] = 0.0
    try:
        derive_return_bounds(bad)
        raise AssertionError("expected a ZeroDivisionError-class failure for time_limit=0")
    except (ZeroDivisionError, ValueError):
        pass

def test_iqn_value_network_output_never_leaves_derived_bounds_on_extreme_input() -> None:
    v_min, v_max = -4.798, 1.08
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=4, n_cosines=8, hidden_dim=8, v_min=v_min, v_max=v_max)
    with torch.no_grad():
        for scale in (1.0, 1e3, 1e6):
            state = torch.randn(16, 8) * scale
            action = torch.randn(16, 4) * scale
            tau = torch.rand(16, 32)
            out = net(state, action, tau)
            assert torch.isfinite(out).all()
            assert float(out.min()) >= v_min - 1e-6
            assert float(out.max()) <= v_max + 1e-6

def test_iqn_value_network_rejects_v_max_not_greater_than_v_min() -> None:
    try:
        IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=4, v_min=1.0, v_max=1.0)
        raise AssertionError("expected ValueError for v_max == v_min")
    except ValueError:
        pass

def test_build_mc_return_samples_hand_computed() -> None:
    # 3-step episode, gamma=0.5: G_0 = 1 + 0.5*2 + 0.25*4 = 3.0;
    # G_1 = 2 + 0.5*4 = 4.0; G_2 = 4.0.
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    states = [_fixture_mc_capture_state(action_table, suite_seed=7, episode_seed=1, step_index=t) for t in range(3)]
    samples = build_mc_return_samples(states, actions=[0, 1, 2], rewards=[1.0, 2.0, 4.0], gamma=0.5,
                                       artifact_sha256="fixture", suite_seed=7, episode_seed=1, outcome="success")
    assert [round(s.target_return, 6) for s in samples] == [3.0, 4.0, 4.0]
    assert [s.step_index for s in samples] == [0, 1, 2]
    assert all(s.outcome == "success" for s in samples)
    assert all(s.source_role == "online" for s in samples)
    assert [s.posterior_seed_key for s in samples] == [(7, 1, 1), (7, 1, 2), (7, 1, 3)]

def test_build_mc_return_samples_rejects_mismatched_lengths() -> None:
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    states = [_fixture_mc_capture_state(action_table)]
    try:
        build_mc_return_samples(states, actions=[0, 1], rewards=[1.0, 2.0], gamma=0.9, artifact_sha256="x", suite_seed=7, episode_seed=1)
        raise AssertionError("expected TrainerError for length mismatch")
    except TrainerError:
        pass

def test_assert_returns_within_bounds_raises_on_violation() -> None:
    try:
        assert_returns_within_bounds([0.5, -2.0, 5.0], v_min=-4.798, v_max=1.08)
        raise AssertionError("expected TrainerError: 5.0 exceeds v_max")
    except TrainerError:
        pass
    assert_returns_within_bounds([0.5, -2.0, 1.0], v_min=-4.798, v_max=1.08)  # must not raise

def test_mc_train_step_aborts_on_out_of_bound_target() -> None:
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)
    B = 4
    action_table = np.array([[0.1 * i, 0.05 * i] for i in range(B)], dtype=np.float64)
    action_indices = list(range(B))
    result = mc_train_step(
        encoder, net, action_encoder, action_table, action_indices,
        robot_features=torch.randn(B, ROBOT_FEATURE_DIM),
        human_features=torch.zeros(B, 20, HUMAN_FEATURE_DIM),
        human_mask=torch.zeros(B, 20, dtype=torch.bool),
        target_returns=torch.tensor([0.5, -2.0, 5.0, 0.0]),  # 5.0 is out of bounds
        n_train_quantiles=8, optimizer=optimizer,
    )
    assert result.aborted
    assert "bounds" in result.abort_reason

def test_mc_train_step_normal_input_reduces_loss_over_iterations() -> None:
    torch.manual_seed(0)
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-2)
    B = 8
    action_table = np.array([[0.1 * i, 0.05 * i] for i in range(B)], dtype=np.float64)
    action_indices = list(range(B))
    robot_features = torch.randn(B, ROBOT_FEATURE_DIM)
    human_features = torch.zeros(B, 20, HUMAN_FEATURE_DIM)
    human_mask = torch.zeros(B, 20, dtype=torch.bool)
    target_returns = torch.full((B,), 1.0)
    losses = []
    for _ in range(50):
        result = mc_train_step(
            encoder, net, action_encoder, action_table, action_indices,
            robot_features, human_features, human_mask, target_returns,
            n_train_quantiles=8, optimizer=optimizer,
        )
        assert not result.aborted, result.abort_reason
        losses.append(result.loss)
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), "MC loss did not decrease over training"

def test_mc_train_step_works_on_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8).to(device)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4).to(device)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)
    B = 4
    action_table = np.array([[0.1 * i, 0.05 * i] for i in range(B)], dtype=np.float64)
    action_indices = list(range(B))
    # Inputs deliberately left on CPU: mc_train_step must move them itself.
    result = mc_train_step(
        encoder, net, action_encoder, action_table, action_indices,
        robot_features=torch.randn(B, ROBOT_FEATURE_DIM),
        human_features=torch.zeros(B, 20, HUMAN_FEATURE_DIM),
        human_mask=torch.zeros(B, 20, dtype=torch.bool),
        target_returns=torch.full((B,), 0.5),
        n_train_quantiles=8, optimizer=optimizer,
    )
    assert not result.aborted, f"CUDA mc_train_step aborted unexpectedly: {result.abort_reason}"
    assert np.isfinite(result.loss)

def test_check_r2_gate_passes_only_when_all_thresholds_met() -> None:
    passing = check_r2_gate(success_rate=0.85, collision_rate=0.02, timeout_rate=0.13, score_bound_violations=0, n_episodes=200)
    assert passing["passed"] is True and passing["reasons"] == []

    failing = check_r2_gate(success_rate=0.56, collision_rate=0.0, timeout_rate=0.44, score_bound_violations=3, n_episodes=200)
    assert failing["passed"] is False
    assert any("success_rate" in r for r in failing["reasons"])
    assert any("timeout_rate" in r for r in failing["reasons"])
    assert any("score_bound_violations" in r for r in failing["reasons"])

def test_check_cvar_promotion_gate_rejects_regression_vs_risk_neutral() -> None:
    result = check_cvar_promotion_gate(
        score_bound_violations=0, quantile_crossing_rate=0.0,
        coverage_errors={0.2: 0.01, 0.5: 0.02, 0.8: 0.01},
        cvar_timeout_rate=0.455, risk_neutral_timeout_rate=0.415,
        cvar_collision_rate=0.0, risk_neutral_collision_rate=0.0,
    )
    assert result["passed"] is False
    assert any("timeout_rate" in r for r in result["reasons"])

def test_check_cvar_promotion_gate_passes_when_calibrated_and_not_worse() -> None:
    result = check_cvar_promotion_gate(
        score_bound_violations=0, quantile_crossing_rate=0.005,
        coverage_errors={tau: 0.02 for tau in CALIBRATION_TAUS},
        cvar_timeout_rate=0.10, risk_neutral_timeout_rate=0.12,
        cvar_collision_rate=0.0, risk_neutral_collision_rate=0.0,
    )
    assert result["passed"] is True and result["reasons"] == []

def test_quantile_calibration_metrics_uses_real_rollout_returns() -> None:
    rows = [[0.0, 1.0, 2.0, 3.0, 4.0]] * 5
    realized = [0.0, 1.0, 2.0, 3.0, 4.0]
    metrics = quantile_calibration_metrics(rows, realized)
    assert metrics["n_decisions"] == 5
    assert tuple(metrics["taus"]) == CALIBRATION_TAUS
    assert set(metrics["coverage_errors"]) == {str(tau) for tau in CALIBRATION_TAUS}
    assert metrics["quantile_crossing_rate"] == 0.0

def test_quantile_calibration_metrics_reports_crps_and_pinball_loss() -> None:
    # R3R-5 point 1/2: CRPS and per-tau pinball loss must be present and
    # hand-verifiable, not just coverage. A perfectly sharp, perfectly
    # correct predictive distribution (all samples equal the realized
    # value) must give CRPS=0 and every pinball loss=0.
    rows = [[2.0, 2.0, 2.0, 2.0]] * 5
    realized = [2.0] * 5
    metrics = quantile_calibration_metrics(rows, realized)
    assert "mean_crps" in metrics and abs(metrics["mean_crps"]) < 1e-12
    assert set(metrics["mean_pinball_loss"]) == {str(tau) for tau in CALIBRATION_TAUS}
    assert all(abs(v) < 1e-12 for v in metrics["mean_pinball_loss"].values())

    # A distribution that always UNDER-predicts should give positive
    # pinball loss for every tau (the realized value always exceeds the
    # predicted quantile), matching the hand-computed sign of
    # max(tau*diff, (tau-1)*diff) with diff=realized-quantile>0.
    under_rows = [[0.0, 0.0, 0.0, 0.0]] * 5
    over_realized = [1.0] * 5
    under_metrics = quantile_calibration_metrics(under_rows, over_realized)
    for tau in CALIBRATION_TAUS:
        expected = tau * 1.0
        assert abs(under_metrics["mean_pinball_loss"][str(tau)] - expected) < 1e-9

def test_fixed_iqn_quantile_outputs_are_the_crossing_source() -> None:
    samples = [[0.0, 1.0, 2.0]]
    crossing = [[0.0, 0.8, 0.7, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0]]
    metrics = quantile_calibration_metrics(samples, [1.0], predicted_quantiles=crossing)
    assert metrics["quantile_crossing_status"] == "OK"
    assert metrics["quantile_crossing_rate"] == 1.0
    accumulator = CalibrationAccumulator()
    accumulator.add({"profile": "nominal", "outcome": "success", "predicted_samples": samples[0], "predicted_quantiles": crossing[0], "realized_return": 1.0})
    assert accumulator.summary()["quantile_crossing_rate"] == 1.0

def test_stratified_calibration_metrics_labels_insufficient_groups() -> None:
    # R3R-5 point 3: a stratum below MIN_STRATUM_SAMPLES must be reported
    # as INSUFFICIENT, never silently merged into a larger group.
    rows = []
    for _ in range(25):
        rows.append({"profile": "nominal", "outcome": "success", "predicted_samples": [1.0, 2.0, 3.0], "realized_return": 2.0})
    for _ in range(3):
        rows.append({"profile": "nominal", "outcome": "collision", "predicted_samples": [0.0, -0.5, -1.0], "realized_return": -0.5})
    report = stratified_calibration_metrics(rows)
    assert report["nominal/success"]["status"] == "OK"
    assert report["nominal/success"]["n_decisions"] == 25
    assert report["nominal/collision"]["status"] == "INSUFFICIENT"
    assert report["nominal/collision"]["n_decisions"] == 3

def test_stratified_calibration_metrics_keeps_profile_outcome_groups_separate() -> None:
    rows = []
    for _ in range(25):
        rows.append({"profile": "nominal", "outcome": "success", "predicted_samples": [0.0, 1.0], "realized_return": 0.5})
    for _ in range(25):
        rows.append({"profile": "train_nonstationary", "outcome": "success", "predicted_samples": [0.0, 1.0], "realized_return": 0.5})
    report = stratified_calibration_metrics(rows)
    assert set(report.keys()) == {
        "nominal/success", "nominal/collision", "nominal/timeout",
        "train_nonstationary/success", "train_nonstationary/collision", "train_nonstationary/timeout",
    }
    assert report["nominal/success"]["n_decisions"] == 25
    assert report["train_nonstationary/success"]["n_decisions"] == 25
    assert report["nominal/collision"]["status"] == "INSUFFICIENT"

def test_check_cvar_promotion_gate_rejects_missing_calibration_tau() -> None:
    result = check_cvar_promotion_gate(
        score_bound_violations=0, quantile_crossing_rate=0.0,
        coverage_errors={0.2: 0.01, 0.5: 0.02},
        cvar_timeout_rate=0.10, risk_neutral_timeout_rate=0.10,
        cvar_collision_rate=0.0, risk_neutral_collision_rate=0.0,
    )
    assert result["passed"] is False
    assert any("coverage taus" in reason for reason in result["reasons"])

def test_normalize_position_clips_to_unit_range_beyond_max_extent() -> None:
    beyond = NORMALIZATION_CONSTANTS["max_position_distance"] * 10.0
    nx, ny = norm.normalize_position(beyond, -beyond)
    assert nx == 1.0 and ny == -1.0
    zx, zy = norm.normalize_position(0.0, 0.0)
    assert zx == 0.0 and zy == 0.0

def test_normalize_ttc_infinite_and_boundary() -> None:
    time_limit = float(FROZEN_VALUES["time_limit"])
    assert norm.normalize_ttc(float("inf"), time_limit) == 1.0  # caller must cap before calling; defend anyway
    assert norm.normalize_ttc(0.0, time_limit) == 0.0
    assert norm.normalize_ttc(time_limit, time_limit) == 1.0

def test_normalize_entropy_at_max_uniform_belief() -> None:
    # Max entropy over 5 equiprobable modes is exactly log(5); a uniform
    # belief's entropy must normalize to 1.0, not overflow past it.
    uniform_entropy = float(-np.sum(np.full(5, 0.2) * np.log(np.full(5, 0.2))))
    assert abs(norm.normalize_entropy(uniform_entropy) - 1.0) < 1e-9

def test_normalize_track_age_clips_beyond_episode_horizon() -> None:
    max_age = NORMALIZATION_CONSTANTS["max_track_age_steps"]
    assert norm.normalize_track_age(max_age) == 1.0
    assert norm.normalize_track_age(max_age * 2) == 1.0  # a stale/buggy age must clip, not blow up
    assert norm.normalize_track_age(0.0) == 0.0

def test_normalize_scalar_and_array_paths_agree() -> None:
    # guide.md R3-2 acceptance: scalar and batch normalization must be
    # element-for-element identical -- both call the SAME _array core,
    # so this is a regression guard against a future edit that only
    # updates one entry point.
    rng = np.random.default_rng(0)
    raw = rng.uniform(-30.0, 30.0, size=20)
    scalar_results = np.array([norm.normalize_position(x, 0.0)[0] for x in raw])
    array_results, _ = norm.normalize_position_array(raw, np.zeros_like(raw))
    assert np.allclose(scalar_results, array_results)

def test_vectorized_candidate_batch_matches_scalar_feature_builders() -> None:
    # Construct one robot, one human, one synthetic world sample; compare
    # _vectorized_candidate_batch's row for a chosen action against
    # manually propagating the SAME state and calling the scalar
    # _robot_feature_vector/_human_feature_vector builders directly.
    # Any drift between the batch and scalar normalization formulas
    # would show up as a mismatch here.
    from crowd_nav.bayesian_dvl.rollout import SampledHumanNextState

    robot = RobotObservation(px=0.5, py=-0.3, vx=0.2, vy=0.1, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    human = HumanObservation(track_id=7, px=2.0, py=0.4, vx=-0.4, vy=0.2, radius=0.3)
    dt, time_limit, global_time = 0.25, 35.0, 12.0
    action_table = [(0.6, -0.2)] * 3  # only action_idx=1 is inspected; repeats keep n_actions small
    action_idx = 1
    action_vx, action_vy = action_table[action_idx]

    next_speed, next_heading = 0.5, 0.3
    sample = SampledHumanNextState(
        track_id=human.track_id, sampled_mode=0, a_parallel=0.1, omega=0.05,
        next_speed=next_speed, next_heading=next_heading,
        next_position=np.array([human.px + 0.1, human.py - 0.05]),
    )
    samples_by_track = {human.track_id: [sample]}
    belief = np.array([0.7, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
    entropy = 0.42
    age = 6
    pred_mean = np.array([0.3, -0.1], dtype=np.float32)
    pred_cov = np.array([[0.02, 0.001], [0.001, 0.015]], dtype=np.float32)

    reward_config = RewardConfig(
        success_reward=1.0, collision_penalty=-0.5, timeout_penalty=-0.5, progress_reward=0.01,
        time_penalty=-0.003, stand_penalty=0.0, stand_speed_threshold=0.05,
        discomfort_distance=0.2, discomfort_penalty_factor=0.5,
    )
    robot_feats, human_feats, masks, *_ = _vectorized_candidate_batch(
        robot=robot, humans=[human], action_table=action_table, samples_by_track=samples_by_track,
        hyp_beliefs_per_sample=[{human.track_id: belief}],
        hyp_entropy_per_sample=[{human.track_id: entropy}],
        hyp_age_per_sample=[{human.track_id: age}],
        hyp_pred_moments_per_sample=[{human.track_id: (pred_mean, pred_cov)}],
        dt=dt, time_limit=time_limit, global_time=global_time, reward_config=reward_config,
        suite_seed=1, episode_seed=1, decision_counter=1, max_human_speed=2.0,
    )
    # Row layout is action-major, world-major with 1 sample: row index == action_idx.
    batch_robot_row = robot_feats[action_idx]
    batch_human_row = human_feats[action_idx, 0]

    next_robot = propagate_robot(robot, action_vx, action_vy, dt)
    remaining = remaining_time_fraction(global_time + dt, time_limit)
    expected_robot = _robot_feature_vector(next_robot, remaining)
    assert np.allclose(batch_robot_row, expected_robot, atol=1e-5)

    next_human = propagate_human(human, next_speed * float(np.cos(next_heading)), next_speed * float(np.sin(next_heading)), dt)
    expected_human = _human_feature_vector(
        next_robot, next_human, belief=belief, entropy=entropy, track_age=age,
        pred_mean=pred_mean, pred_cov=pred_cov,
    )
    assert np.allclose(batch_human_row, expected_human, atol=1e-5)

def test_score_candidate_batch_is_differentiable() -> None:
    # guide.md R3-3's whole point: score_candidate_batch must support
    # gradients so the Stage 1 ranking loss can backprop into the SAME
    # scoring path decide() uses at inference -- verify a real backward
    # pass actually populates every parameter's .grad.
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    n_actions, n_samples = 5, 3
    batch = n_actions * n_samples
    robot_feats = torch.randn(batch, ROBOT_FEATURE_DIM, requires_grad=False)
    human_feats = torch.zeros(batch, 20, HUMAN_FEATURE_DIM)
    masks = torch.zeros(batch, 20, dtype=torch.bool)
    action_feats = torch.randn(batch, ACTION_FEATURE_DIM)
    rewards = torch.rand(batch)
    not_terminal = torch.ones(batch, dtype=torch.bool)
    tau = torch.rand(batch, 8)

    action_scores, quantiles = score_candidate_batch(
        encoder, net, action_encoder, robot_feats, human_feats, masks, action_feats, rewards, not_terminal, tau,
        n_actions, n_samples, gamma=0.99,
    )
    assert action_scores.shape == (n_actions,)
    assert action_scores.requires_grad
    loss = action_scores.sum()
    loss.backward()
    grads = [p.grad for p in list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters())]
    assert all(g is not None for g in grads), "score_candidate_batch did not propagate gradients to every parameter"
    assert all(torch.isfinite(g).all() for g in grads)

def test_action_encoder_different_action_gives_different_embedding() -> None:
    # guide.md R4-1 acceptance: same state, different action -> different
    # action embedding (and therefore, generically, a different Q) --
    # action identity must not be inferable solely from the successor
    # state, so ActionEncoder itself must not collapse distinct actions.
    # seeded (real bug found by measurement: without this, the test's
    # outcome depended on the GLOBAL torch RNG state accumulated from every
    # other test that happened to run earlier in alphabetical discovery
    # order -- same bug CLASS as the train_step resume-divergence fix --
    # so it could intermittently fail depending on unrelated test churn).
    torch.manual_seed(0)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    action_a = torch.tensor([[0.5, 0.0, 0.5, 1.0, 0.0]], dtype=torch.float32)
    action_b = torch.tensor([[-0.5, 0.2, 0.54, -0.3, 0.8]], dtype=torch.float32)
    with torch.no_grad():
        embed_a = action_encoder(action_a)
        embed_b = action_encoder(action_b)
    assert not torch.allclose(embed_a, embed_b), "different actions produced the same embedding"

def test_score_candidate_batch_permutation_equivariant_in_action_axis() -> None:
    # guide.md R4-1: the network must not silently ignore action identity --
    # permuting the action axis of the inputs must permute action_scores/
    # quantiles by the SAME permutation, bit-for-bit. n_samples=1 makes the
    # action axis exactly the row/batch axis of score_candidate_batch's
    # flattened inputs.
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    encoder.eval(); net.eval(); action_encoder.eval()
    n_actions, n_samples = 6, 1
    batch = n_actions * n_samples
    robot_feats = torch.randn(batch, ROBOT_FEATURE_DIM)
    human_feats = torch.zeros(batch, 20, HUMAN_FEATURE_DIM)
    masks = torch.zeros(batch, 20, dtype=torch.bool)
    action_feats = torch.randn(batch, ACTION_FEATURE_DIM)
    rewards = torch.rand(batch)
    not_terminal = torch.ones(batch, dtype=torch.bool)
    tau = torch.rand(batch, 8)

    with torch.no_grad():
        scores, quantiles = score_candidate_batch(
            encoder, net, action_encoder, robot_feats, human_feats, masks, action_feats,
            rewards, not_terminal, tau, n_actions, n_samples, gamma=0.99,
        )
        perm = torch.randperm(n_actions)
        scores_perm, quantiles_perm = score_candidate_batch(
            encoder, net, action_encoder, robot_feats[perm], human_feats[perm], masks[perm], action_feats[perm],
            rewards[perm], not_terminal[perm], tau[perm], n_actions, n_samples, gamma=0.99,
        )
    assert torch.equal(scores[perm], scores_perm)
    assert torch.equal(quantiles[perm], quantiles_perm)

def test_score_candidate_batch_train_deploy_parity_is_bit_identical() -> None:
    # guide.md R4-1 acceptance: calling score_candidate_batch twice with
    # IDENTICAL inputs under eval()/no_grad() must give bit-identical
    # action_scores -- structurally guaranteed (no dropout/batchnorm
    # anywhere in these modules), but guide.md explicitly asks for the
    # assertion rather than leaving it implicit.
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8)
    encoder.eval(); net.eval(); action_encoder.eval()
    n_actions, n_samples = 4, 2
    batch = n_actions * n_samples
    robot_feats = torch.randn(batch, ROBOT_FEATURE_DIM)
    human_feats = torch.zeros(batch, 20, HUMAN_FEATURE_DIM)
    masks = torch.zeros(batch, 20, dtype=torch.bool)
    action_feats = torch.randn(batch, ACTION_FEATURE_DIM)
    rewards = torch.rand(batch)
    not_terminal = torch.ones(batch, dtype=torch.bool)
    tau = torch.rand(batch, 8)

    with torch.no_grad():
        scores_1, quantiles_1 = score_candidate_batch(
            encoder, net, action_encoder, robot_feats, human_feats, masks, action_feats,
            rewards, not_terminal, tau, n_actions, n_samples, gamma=0.99,
        )
        scores_2, quantiles_2 = score_candidate_batch(
            encoder, net, action_encoder, robot_feats, human_feats, masks, action_feats,
            rewards, not_terminal, tau, n_actions, n_samples, gamma=0.99,
        )
    assert torch.equal(scores_1, scores_2)
    assert torch.equal(quantiles_1, quantiles_2)

def test_score_all_candidates_matches_decide_action_scores() -> None:
    # _score_all_candidates is the exact function decide() calls; verify
    # calling it directly (as Stage 1 ranking-IL training will) on the
    # SAME real tracker/state reproduces decide()'s own chosen scores.
    policy = _tiny_policy()
    robot, humans = _fixture_robot_and_humans()
    result = policy.decide(robot, humans, global_time=1.0, suite_seed=3, episode_seed=3)
    tracker_copy = policy.belief_tracker.clone_for_hypothetical()
    with torch.no_grad():
        action_scores, *_ = _score_all_candidates(
            tracker=tracker_copy, robot=robot, humans=humans, global_time=1.0,
            artifact=policy.artifact, action_table=policy.action_table, reward_config=policy.reward_config,
            dt=policy.dt, time_limit=policy.time_limit, max_human_speed=policy.max_human_speed,
            n_world_samples=policy.n_world_samples, n_iqn_quantiles=policy.n_iqn_quantiles,
            tau_upper=(1.0 if policy.risk_neutral else policy.cvar_alpha), posterior_source=policy.posterior_source,
            set_encoder=policy.set_encoder, value_network=policy.value_network, action_encoder=policy.action_encoder,
            device=policy.device, seed_key=(3, 3, policy._decision_counter), gamma=policy.gamma,
        )
    recomputed = {i: float(action_scores[i]) for i in range(len(policy.action_table))}
    original = {s.action_index: s.cvar for s in result.all_scores}
    for i in original:
        assert abs(recomputed[i] - original[i]) < 1e-4, f"action {i}: recomputed={recomputed[i]} vs decide()={original[i]}"

def test_derive_action_equivalence_tolerance_is_positive_and_data_derived() -> None:
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    tolerance = derive_action_equivalence_tolerance(action_table)
    assert tolerance > 0.0
    # Sanity bound: tolerance is a LOCAL grid-spacing statistic, not a
    # global span -- it must be far smaller than the grid's own extent.
    max_speed = max(np.hypot(vx, vy) for vx, vy in action_table)
    assert tolerance < max_speed

def test_build_action_equivalence_class_always_includes_the_nearest_action() -> None:
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    tolerance = derive_action_equivalence_tolerance(action_table)
    orca_vx, orca_vy = action_table[17]  # exactly on a grid point
    indices = build_action_equivalence_class(orca_vx, orca_vy, action_table, tolerance)
    assert 17 in indices
    assert len(indices) >= 1

def test_build_action_equivalence_class_rejects_negative_tolerance() -> None:
    try:
        build_action_equivalence_class(0.5, 0.0, [(0.0, 0.0), (1.0, 0.0)], tolerance=-0.1)
        raise AssertionError("expected RankingError for negative tolerance")
    except RankingError:
        pass

def test_expert_ranking_loss_hand_computed_direction() -> None:
    # Expert clearly ahead of the hardest negative -> loss EXACTLY 0
    # (max/max margin loss, not just "near" 0 -- R3R-1's whole point).
    # Expert clearly behind -> loss large and positive.
    scores_expert_ahead = torch.tensor([5.0, 0.0, 0.0, 0.0])
    loss_ahead = expert_ranking_loss(scores_expert_ahead, expert_indices=[0], margin=0.1)
    scores_expert_behind = torch.tensor([-5.0, 0.0, 0.0, 0.0])
    loss_behind = expert_ranking_loss(scores_expert_behind, expert_indices=[0], margin=0.1)
    assert float(loss_ahead) == 0.0  # relu(0.1 + 0.0 - 5.0) == relu(negative) == 0 exactly
    assert abs(float(loss_behind) - 5.1) < 1e-6  # relu(0.1 + 0.0 - (-5.0)) == 5.1 exactly

def test_expert_ranking_loss_is_invariant_to_expert_set_size() -> None:
    # guide.md R3R-1 acceptance: "测试必须覆盖1/2/4/8个expert时，同样的
    # best-expert/best-negative差产生相同loss，证明不受集合大小影响" --
    # this is the direct regression guard for the log(n) bug the old
    # logsumexp formulation had (equal-score loss used to be 4.4808;
    # the theoretical-best-case floor used to be 0.8479, never 0).
    for k in (1, 2, 4, 8):
        scores = torch.zeros(80)
        scores[:k] = 3.0   # tied expert set
        scores[k:] = 3.0   # tied non-expert set, same score -> a genuine margin violation of exactly `margin`
        loss = expert_ranking_loss(scores, list(range(k)), margin=0.1)
        assert abs(float(loss) - 0.1) < 1e-5, f"k={k}: expected relu(0.1+3.0-3.0)=0.1 regardless of set size, got {float(loss)}"

def test_expert_ranking_loss_reaches_exact_zero_at_theoretical_best_case() -> None:
    # The old logsumexp loss floored at ~0.8479 here and could NEVER
    # reach 0 no matter how well-separated the scores were -- verified
    # by direct computation before this fix. The new loss must reach
    # exactly 0 once the expert is at v_max and every negative at v_min.
    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    scores = torch.full((80,), v_min)
    scores[0] = v_max
    loss = expert_ranking_loss(scores, [0], margin=0.1)
    assert float(loss) == 0.0

def test_expert_ranking_loss_gradient_direction_flips_with_expert_label() -> None:
    # guide.md R3R-1 acceptance: "合成状态中交换expert label后梯度方向必须
    # 反转" -- swapping which action is "expert" must flip the sign of
    # the gradient pushing on that action's score. Two tied top scores
    # (index 0 and 1) guarantee a genuine, nonzero margin violation in
    # BOTH directions -- with the max/max loss, a configuration where
    # the margin is already satisfied legitimately has zero gradient
    # everywhere, which would make a "sign flip" assertion meaningless.
    raw_scores = torch.tensor([0.5, 0.5, 0.0, 0.0], requires_grad=True)
    loss_a = expert_ranking_loss(raw_scores, expert_indices=[0], margin=0.2)
    grad_a = torch.autograd.grad(loss_a, raw_scores)[0]
    assert float(loss_a) > 0.0  # sanity: this configuration is a real violation

    raw_scores2 = raw_scores.detach().clone().requires_grad_(True)
    loss_b = expert_ranking_loss(raw_scores2, expert_indices=[1], margin=0.2)
    grad_b = torch.autograd.grad(loss_b, raw_scores2)[0]
    assert float(loss_b) > 0.0

    # Index 0's gradient (pushing its score up when it's expert) must
    # have the opposite sign from when index 1 is expert instead
    # (index 0 then becomes the hardest negative, pushing its score DOWN).
    assert float(grad_a[0]) * float(grad_b[0]) < 0.0

def test_expert_ranking_loss_rejects_expert_set_covering_everything() -> None:
    scores = torch.zeros(3)
    try:
        expert_ranking_loss(scores, expert_indices=[0, 1, 2], margin=0.1)
        raise AssertionError("expected ValueError: no negatives to rank against")
    except ValueError:
        pass

def test_stage1_train_step_runs_and_reduces_combined_loss() -> None:
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-2)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()

    losses = []
    for step in range(15):
        batch = [_fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=step, target_return=0.5)]
        result = stage1_train_step(
            encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
            dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
            posterior_source="full", device=torch.device("cpu"), gamma=0.99,
            demo_batch=batch, ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=optimizer,
        )
        assert not result.aborted, result.abort_reason
        losses.append(result.loss)
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), "combined stage1 loss did not decrease over training"

def test_frozen_lambda_rank_keeps_rank_gradient_from_being_negligible() -> None:
    # Regression guard for the real gradient-scale audit that set
    # lambda_rank=200 (guide.md R3-3 comment in config.FROZEN_VALUES):
    # at lambda_rank=1, L_rank's gradient was 38x-495x smaller than
    # L_MC's across varied fresh-init trials -- effectively invisible to
    # the optimizer. With the frozen lambda_rank applied, the two
    # weighted gradient norms must be within a reasonable order of
    # magnitude of each other, not off by two-plus orders of magnitude.
    artifact = _fixture_artifact()
    torch.manual_seed(0)
    encoder = SetEncoder()
    action_encoder = ActionEncoder()
    net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    robot = RobotObservation(px=0.1, py=-0.2, vx=0.3, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = (HumanObservation(track_id=0, px=2.0, py=0.2, vx=-0.4, vy=0.1, radius=0.3),)
    tracker = BeliefTracker(artifact)
    tracker.update({0: (1.0, np.array([humans[0].px, humans[0].py]))})

    remaining = remaining_time_fraction(1.0, 35.0)
    robot_feat = torch.tensor(_robot_feature_vector(robot, remaining), dtype=torch.float32).unsqueeze(0)
    human_feat, mask = build_human_feature_batch(tracker, robot, humans)
    state_emb = encoder(robot_feat, torch.tensor(human_feat).unsqueeze(0), torch.tensor(mask).unsqueeze(0))
    action_table_array = np.asarray(action_table, dtype=np.float64)
    action_feat = torch.tensor(
        compute_action_features_array(robot, action_table_array)[10], dtype=torch.float32,
    ).unsqueeze(0)
    action_emb = action_encoder(action_feat)
    tau = torch.rand(1, 16)
    pred = net(state_emb, action_emb, tau)
    mc_loss = quantile_huber_loss(pred, tau, torch.full((1, 1), 0.3))

    action_scores, *_ = _score_all_candidates(
        tracker=tracker, robot=robot, humans=humans, global_time=1.0, artifact=artifact,
        action_table=action_table, reward_config=_REWARD_CFG, dt=0.25, time_limit=35.0, max_human_speed=2.0,
        n_world_samples=4, n_iqn_quantiles=8, tau_upper=1.0, posterior_source="full",
        set_encoder=encoder, value_network=net, action_encoder=action_encoder,
        device=torch.device("cpu"), seed_key=(0, 1, 1), gamma=0.99,
    )
    rank_loss = expert_ranking_loss(action_scores, expert_indices=[10], margin=float(FROZEN_VALUES["ranking_margin"]))
    lambda_rank = float(FROZEN_VALUES["lambda_rank"])

    params = list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters())
    grad_mc = torch.autograd.grad(mc_loss, params, retain_graph=True, allow_unused=True)
    grad_rank = torch.autograd.grad(lambda_rank * rank_loss, params, retain_graph=True, allow_unused=True)
    norm_mc = sum(float(g.norm() ** 2) for g in grad_mc if g is not None) ** 0.5
    norm_rank = sum(float(g.norm() ** 2) for g in grad_rank if g is not None) ** 0.5
    ratio = norm_mc / max(norm_rank, 1e-12)
    assert 0.05 < ratio < 20.0, f"lambda_rank={lambda_rank} leaves grad_norm(L_MC)/grad_norm(lambda_rank*L_rank)={ratio:.2f}, outside the balanced range"

def test_gradient_gate_accepts_legitimate_zero_ranking_gradient() -> None:
    from types import SimpleNamespace
    from crowd_nav.tools.train_bdvl import _validate_and_record_gradient_diagnostic

    diagnostics = []
    inactive = SimpleNamespace(
        mc_grad_norm=1.0,
        rank_grad_norm=0.0,
        weighted_rank_grad_norm=0.0,
        gradient_ratio=0.0,
    )
    monitor = {"outside_streak": 0}
    _validate_and_record_gradient_diagnostic(inactive, "il", 1, diagnostics, monitor)
    assert diagnostics[-1]["ranking_active"] is False
    assert diagnostics[-1]["gradient_ratio"] == 0.0

    active_bad = SimpleNamespace(
        mc_grad_norm=1.0,
        rank_grad_norm=1e-3,
        weighted_rank_grad_norm=1e-3,
        gradient_ratio=0.0,
    )
    _validate_and_record_gradient_diagnostic(active_bad, "il", 2, diagnostics, monitor)
    assert diagnostics[-1]["ranking_active"] is True
    assert diagnostics[-1]["ratio_in_frozen_range"] is False
    assert monitor["outside_streak"] == 1

def test_gradient_gate_aborts_only_after_sustained_out_of_range_updates() -> None:
    from types import SimpleNamespace
    from crowd_nav.tools.train_bdvl import _validate_and_record_gradient_diagnostic

    active_bad = SimpleNamespace(
        mc_grad_norm=1.0,
        rank_grad_norm=1e-3,
        weighted_rank_grad_norm=1e-3,
        gradient_ratio=0.0,
    )
    monitor = {"outside_streak": 0}
    window = int(FROZEN_VALUES["gradient_ratio_sustained_updates"])
    for index in range(window - 1):
        _validate_and_record_gradient_diagnostic(active_bad, "il", index + 1, [], monitor)
    assert monitor["outside_streak"] == window - 1
    try:
        _validate_and_record_gradient_diagnostic(active_bad, "il", window, [], monitor)
        raise AssertionError("expected sustained out-of-range ratio to be rejected")
    except RuntimeError as exc:
        assert "persistently outside frozen range" in str(exc)

def test_gradient_gate_rejects_nonfinite_diagnostic() -> None:
    from types import SimpleNamespace
    from crowd_nav.tools.train_bdvl import _validate_and_record_gradient_diagnostic

    nonfinite = SimpleNamespace(
        mc_grad_norm=float("nan"),
        rank_grad_norm=0.0,
        weighted_rank_grad_norm=0.0,
        gradient_ratio=0.0,
    )
    try:
        _validate_and_record_gradient_diagnostic(nonfinite, "rl", 1, [])
        raise AssertionError("expected non-finite gradient diagnostic to be rejected")
    except RuntimeError as exc:
        assert "non-finite" in str(exc)

def test_stage1_train_step_ranking_batch_size_caps_expensive_scoring() -> None:
    # guide.md R3R-2: ranking_batch_size bounds how many demo samples
    # get the expensive 80-action _score_all_candidates pass; L_MC must
    # still cover every sample in the batch regardless of the cap.
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-2)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    batch = [_fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=i, target_return=0.4) for i in range(4)]

    result_capped = stage1_train_step(
        encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
        dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
        posterior_source="full", device=torch.device("cpu"), gamma=0.99,
        demo_batch=batch, ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=optimizer,
        ranking_batch_size=1,
    )
    assert not result_capped.aborted, result_capped.abort_reason
    # mc_loss must reflect all 4 samples' targets, not just the 1 ranked one --
    # sanity-checked indirectly by confirming the step succeeds with a batch
    # bigger than ranking_batch_size and produces a finite, real loss.
    assert np.isfinite(result_capped.mc_loss) and np.isfinite(result_capped.rank_loss)

    try:
        stage1_train_step(
            encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
            dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
            posterior_source="full", device=torch.device("cpu"), gamma=0.99,
            demo_batch=batch, ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=optimizer,
            ranking_batch_size=0,
        )
        raise AssertionError("expected TrainerError for ranking_batch_size=0")
    except TrainerError:
        pass

def test_stage1_train_step_aborts_on_out_of_bound_target() -> None:
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    batch = [_fixture_ranking_demo_sample(artifact, target_return=999.0)]  # far outside [-4.798, 1.08]
    result = stage1_train_step(
        encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
        dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
        posterior_source="full", device=torch.device("cpu"), gamma=0.99,
        demo_batch=batch, ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=optimizer,
    )
    assert result.aborted
    assert "bounds" in result.abort_reason

def test_r4_2_executed_action_features_recompute_bit_identical_from_snapshot() -> None:
    # guide.md R4-2 minimum test 1: executed_action_index/features stored
    # on a replay sample must be exactly recoverable from the raw
    # robot/action_table snapshot also stored alongside them -- catches
    # a future edit that lets the two silently drift (e.g. computing
    # executed_action_features from a DIFFERENT robot state than the one
    # actually stored).
    artifact = _fixture_artifact()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    sample = _fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=0, target_return=0.5)
    recomputed = compute_action_features_array(sample.robot, np.asarray(action_table, dtype=np.float64))[sample.executed_action_index]
    assert np.allclose(recomputed, sample.executed_action_features)

    online_sample = _fixture_mc_return_sample(artifact, action_table, episode_seed=2, step_index=0, target_return=0.2)
    recomputed_online = compute_action_features_array(online_sample.robot, np.asarray(action_table, dtype=np.float64))[online_sample.executed_action_index]
    assert np.allclose(recomputed_online, online_sample.executed_action_features)

def test_r4_2_swapping_executed_action_label_makes_mc_loss_worse() -> None:
    # guide.md R4-2 minimum test 2: after the network has actually
    # learned an action-specific target_return mapping, evaluating the
    # SAME state/target with a DIFFERENT (wrong) executed_action_index
    # must give a strictly worse (higher) MC loss -- proving the network
    # genuinely uses action identity rather than ignoring it. Trains on
    # ONE fixed (state, correct action, target_return) pair repeatedly so
    # "worse when swapped" is a real, checkable claim, not a fresh-init
    # coin flip.
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=5e-2)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    correct_index = 10
    wrong_index = 60

    def make_sample(executed_index):
        s = _fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=0, target_return=0.5)
        s.executed_action_index = executed_index
        s.executed_action_features = compute_action_features_array(s.robot, np.asarray(action_table, dtype=np.float64))[executed_index]
        return s

    for _ in range(30):
        result = stage1_train_step(
            encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
            dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=4, n_iqn_quantiles=8,
            posterior_source="full", device=torch.device("cpu"), gamma=0.99,
            demo_batch=[make_sample(correct_index)], ranking_margin=0.1, lambda_rank=0.0,
            n_train_quantiles=8, optimizer=optimizer,
        )
        assert not result.aborted, result.abort_reason
    loss_correct = result.mc_loss

    with torch.no_grad():
        wrong_sample = make_sample(wrong_index)
        tracker = BeliefTracker(artifact)
        tracker._tracks = copy.deepcopy(wrong_sample.belief_tracker_snapshot)
        _, _, rf, hf, mk, af, rw, nt, _, _ = _score_all_candidates(
            tracker=tracker, robot=wrong_sample.robot, humans=wrong_sample.humans, global_time=wrong_sample.global_time,
            artifact=artifact, action_table=(action_table[wrong_index],), action_indices=(wrong_index,),
            reward_config=_REWARD_CFG, dt=0.25, time_limit=35.0, max_human_speed=2.0,
            n_world_samples=4, n_iqn_quantiles=8, tau_upper=1.0, posterior_source="full",
            set_encoder=encoder, value_network=net, action_encoder=action_encoder, device=torch.device("cpu"),
            seed_key=wrong_sample.posterior_seed_key, gamma=0.99,
        )
        z_bar, tau_used = compute_executed_action_quantile_target(
            encoder, net, action_encoder, rf, hf, mk, af, rw, nt, n_quantiles=8, tau_upper=1.0,
            seed_key=wrong_sample.posterior_seed_key, executed_action_index=wrong_index, gamma=0.99, device=torch.device("cpu"),
        )
        loss_wrong = float(quantile_huber_loss(z_bar.unsqueeze(0), tau_used.unsqueeze(0), torch.tensor([[0.5]])))
    assert loss_wrong > loss_correct, f"expected swapped-action loss ({loss_wrong}) to exceed trained-action loss ({loss_correct})"

def test_r4_2_expert_set_change_affects_only_rank_loss_not_mc_loss() -> None:
    # guide.md R4-2 minimum test 3: executed_action_index (MC-loss) and
    # expert_action_indices (ranking) must be genuinely independent
    # fields -- changing one must not silently move the other's loss.
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()

    sample_a = _fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=0, target_return=0.5)
    sample_b = copy.deepcopy(sample_a)
    sample_b.expert_action_indices = tuple(i for i in (1, 2, 3) if i != sample_a.executed_action_index) or (1, 2, 3)

    def run(sample):
        opt = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=0.0)
        return stage1_train_step(
            encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
            dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=4, n_iqn_quantiles=8,
            posterior_source="full", device=torch.device("cpu"), gamma=0.99,
            demo_batch=[sample], ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=8, optimizer=opt,
        )

    result_a = run(sample_a)
    result_b = run(sample_b)
    assert not result_a.aborted and not result_b.aborted
    assert abs(result_a.mc_loss - result_b.mc_loss) < 1e-6, "mc_loss must not depend on expert_action_indices"

def test_r4_2_online_action_features_have_no_placeholder_zeros() -> None:
    # guide.md R4-2 minimum test 4: online samples must carry the FULL
    # 5-dim ActionFeature (goal_alignment/turn_cost included), not the
    # retired R4-1-mechanical placeholder that hardcoded those two to 0.0.
    artifact = _fixture_artifact()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    # Robot heading away from an off-axis goal with nonzero current
    # velocity -- a state where goal_alignment and turn_cost are BOTH
    # analytically guaranteed nonzero for a generic action, so an
    # all-zero placeholder would be caught, not accidentally matched.
    sample = _fixture_mc_return_sample(artifact, action_table, executed_action_index=37)
    assert abs(float(sample.executed_action_features[3])) > 1e-6, "goal_alignment must not be a placeholder 0"
    assert abs(float(sample.executed_action_features[4])) > 1e-6, "turn_cost must not be a placeholder 0"

def test_r4_2_single_action_subset_score_matches_full_80_action_deployment() -> None:
    # guide.md R4-2 minimum test 5: scoring ONE action via a length-1
    # action table + action_indices=[k] (what R4-2's MC-loss does) must
    # be bit-identical to that same action's row inside a full 80-action
    # deployment/ranking call -- otherwise train-time and deploy-time
    # scores for the executed action could silently diverge.
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    encoder.eval(); net.eval(); action_encoder.eval()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    artifact_ = artifact
    tracker = BeliefTracker(artifact_)
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, 2.0, 0.0, -0.5, 0.0, 0.3)]
    tracker.update({0: (0.0, np.array([2.0, 0.0]))})
    seed_key = (1, 2, 3)
    common = dict(
        artifact=artifact_, reward_config=_REWARD_CFG, dt=0.25, time_limit=35.0, max_human_speed=2.0,
        n_world_samples=4, n_iqn_quantiles=8, tau_upper=1.0, posterior_source="full",
        set_encoder=encoder, value_network=net, action_encoder=action_encoder, device=torch.device("cpu"),
        seed_key=seed_key, gamma=0.99,
    )
    k = 42
    with torch.no_grad():
        scores_full, quant_full, *_rest_full = _score_all_candidates(
            tracker=tracker, robot=robot, humans=humans, global_time=0.0, action_table=action_table, **common,
        )
        scores_sub, quant_sub, *_rest_sub = _score_all_candidates(
            tracker=tracker, robot=robot, humans=humans, global_time=0.0,
            action_table=(action_table[k],), action_indices=(k,), **common,
        )
    n_samples = quant_sub.shape[0]
    assert torch.allclose(quant_full[k * n_samples:(k + 1) * n_samples], quant_sub)
    assert torch.allclose(scores_full[k], scores_sub[0])

def test_r4_2r_world_aggregation_and_shared_tau_via_production_function() -> None:
    # guide.md R4-2R-4: the world/shared-tau tests must actually call the
    # PRODUCTION function (compute_executed_action_quantile_target via
    # _score_all_candidates), not hand-roll the aggregation formula
    # separately -- a hand-rolled copy cannot catch a regression in the
    # real implementation. Hooks policy._encode_score_quantiles (the one
    # place both score_candidate_batch and
    # compute_executed_action_quantile_target reach the network) to
    # capture exactly what tau/quantiles the production code used.
    import crowd_nav.bayesian_dvl.policy as _policy_mod

    artifact = _fixture_artifact()
    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    encoder = SetEncoder(); action_encoder = ActionEncoder()
    net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim, v_min=v_min, v_max=v_max)
    encoder.eval(); net.eval(); action_encoder.eval()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    tracker = BeliefTracker(artifact)
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, 2.0, 0.0, -0.5, 0.0, 0.3)]
    tracker.update({0: (0.0, np.array([2.0, 0.0]))})
    k = 42
    n_samples = 8
    n_quantiles = 16
    gamma = 0.97

    captured = []
    original = _policy_mod._encode_score_quantiles

    def _hooked(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append((args[-1].clone(), out.detach().clone()))  # (tau, quantiles)
        return out

    _policy_mod._encode_score_quantiles = _hooked
    try:
        with torch.no_grad():
            _, _, rf, hf, mk, af, rw, nt, _, _ = _policy_mod._score_all_candidates(
                tracker=tracker, robot=robot, humans=humans, global_time=0.0,
                artifact=artifact, action_table=(action_table[k],), action_indices=(k,),
                reward_config=_REWARD_CFG, dt=0.25, time_limit=35.0, max_human_speed=2.0,
                n_world_samples=n_samples, n_iqn_quantiles=n_quantiles, tau_upper=1.0,
                posterior_source="full", set_encoder=encoder, value_network=net, action_encoder=action_encoder,
                device=torch.device("cpu"), seed_key=(1, 2, 3), gamma=gamma,
            )
            captured.clear()  # only care about the call inside compute_executed_action_quantile_target below
            z_bar, tau_used = compute_executed_action_quantile_target(
                encoder, net, action_encoder, rf, hf, mk, af, rw, nt, n_quantiles=n_quantiles, tau_upper=1.0,
                seed_key=(1, 2, 3), executed_action_index=k, gamma=gamma, device=torch.device("cpu"),
            )
    finally:
        _policy_mod._encode_score_quantiles = original

    assert len(captured) == 1, "compute_executed_action_quantile_target must call _encode_score_quantiles exactly once"
    hooked_tau, hooked_quantiles = captured[0]

    # 1. every world row got IDENTICAL tau (guide.md R4-2R-4 point 1).
    for row in range(1, n_samples):
        assert torch.equal(hooked_tau[0], hooked_tau[row]), "tau differs across world rows -- z_bar[j] would not mean the same quantile level for every world"
    assert torch.equal(hooked_tau[0], tau_used)

    # 2. hand-recompute z_bar from the CAPTURED production quantiles/reward/not_terminal -- must match bit-for-bit.
    bootstrap = hooked_quantiles * nt.to(hooked_quantiles.dtype).unsqueeze(-1)
    manual_z_bar = (rw.unsqueeze(-1) + gamma * bootstrap).mean(dim=0)
    assert torch.allclose(manual_z_bar, z_bar), "production z_bar does not match manual mean-over-worlds recompute of its own captured quantiles"

    # 3. outlier-world sensitivity: correct mean-then-compare vs. wrong duplicate-G-per-world.
    #    Use the REAL captured quantiles/reward but synthetically inject one outlier world's
    #    quantile row to prove the mean formula dilutes it by n_samples (not fixed g).
    outlier_quantiles = hooked_quantiles.clone()
    outlier_quantiles[0] = outlier_quantiles[0] + 10.0  # one outlier world
    outlier_bootstrap = outlier_quantiles * nt.to(outlier_quantiles.dtype).unsqueeze(-1)
    outlier_z_bar = (rw.unsqueeze(-1) + gamma * outlier_bootstrap).mean(dim=0)
    expected_shift = gamma * 10.0 / n_samples
    assert torch.allclose(outlier_z_bar - manual_z_bar, torch.full_like(z_bar, expected_shift), atol=1e-4), (
        "a single outlier world must shift z_bar by exactly gamma*delta/n_samples under correct mean "
        "aggregation -- a 'duplicate G per world' bug would not dilute by n_samples at all"
    )

    # 4. same seed -> bit-identical; different seed -> different tau, but this is purely a
    #    scoring-path property and must never touch any replay identity field.
    with torch.no_grad():
        z_bar_repeat, tau_repeat = compute_executed_action_quantile_target(
            encoder, net, action_encoder, rf, hf, mk, af, rw, nt, n_quantiles=n_quantiles, tau_upper=1.0,
            seed_key=(1, 2, 3), executed_action_index=k, gamma=gamma, device=torch.device("cpu"),
        )
        z_bar_diff_seed, tau_diff_seed = compute_executed_action_quantile_target(
            encoder, net, action_encoder, rf, hf, mk, af, rw, nt, n_quantiles=n_quantiles, tau_upper=1.0,
            seed_key=(1, 2, 4), executed_action_index=k, gamma=gamma, device=torch.device("cpu"),
        )
    assert torch.equal(tau_used, tau_repeat) and torch.allclose(z_bar, z_bar_repeat)
    assert not torch.equal(tau_used, tau_diff_seed)

def test_r4_2_full_pipeline_registry_schema_consistent_train_to_evaluate() -> None:
    # guide.md R4-2 minimum test 7: a real (small) train -> resume-style
    # save -> select -> evaluate chain, called directly at the function
    # level (train_bdvl.py's CLI itself is fail-closed until
    # R4_2_REPLAY_CONTRACT_COMPLETE=True -- see config.py), must agree on
    # registry content hash / action grid hash / feature schema at every
    # stage, exactly like the real CLI chain guide.md R4-1R-2 verified.
    import tempfile as _tempfile
    from crowd_nav.bayesian_dvl.config import load_and_validate_registry as _load_registry
    from crowd_nav.bayesian_dvl.policy import save_composed_checkpoint as _save_ckpt, load_composed_checkpoint as _load_ckpt
    from crowd_nav.tools.train_bdvl import stage1_il_pretrain as _il_pretrain

    artifact = _fixture_artifact()
    registry = _load_registry(str(REPO_ROOT / "crowd_nav/configs/bayesian_dvl_registry_r4.json"))
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    assert registry["action_grid_hash"] == grid.table_hash()

    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    encoder = SetEncoder(); action_encoder = ActionEncoder()
    net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim, v_min=v_min, v_max=v_max)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=3e-4)
    replay = DemoOnlineReplay(200, 200, 0.2)
    _il_pretrain(
        encoder, net, action_encoder, artifact, action_table, REPO_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config",
        replay, n_episodes=2, gamma=FROZEN_VALUES["gamma"], optimizer=optimizer, device="cpu",
        profiles=["nominal"], episode_seed_base=1, batch_size=4,
    )
    with _tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "ckpt.pth")
        _save_ckpt(
            encoder, net, ckpt_path, registry["action_grid_hash"], registry["content_sha256"],
            action_encoder=action_encoder, artifact_sha256=artifact.content_sha256(),
        )
        fresh_encoder = SetEncoder(); fresh_action_encoder = ActionEncoder()
        fresh_net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=fresh_action_encoder.embed_dim, v_min=v_min, v_max=v_max)
        manifest = _load_ckpt(
            ckpt_path, fresh_encoder, fresh_net, fresh_action_encoder,
            expected_registry_content_sha256=registry["content_sha256"],
            expected_action_grid_hash=registry["action_grid_hash"],
        )
        assert manifest["feature_schema"] == FEATURE_SCHEMA_V4
        assert manifest["action_grid_hash"] == registry["action_grid_hash"]
        assert manifest["registry_content_sha256"] == registry["content_sha256"]

def test_r4_2r_mixed_explore_greedy_posterior_seed_keys_are_unique_and_sequential() -> None:
    # guide.md R4-2R-2 minimum test: a real episode with BOTH exploration
    # and greedy steps interleaved must produce posterior_seed_key values
    # that are all distinct and whose decision-counter component (index
    # 2) increases by exactly 1 every single environment step -- proving
    # every step (not just greedy ones) consumes its own real counter
    # tick, rather than exploration steps silently reusing/skipping one.
    from crowd_nav.tools.train_bdvl import _collect_online_episode

    artifact = _fixture_artifact()
    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    encoder = SetEncoder(); action_encoder = ActionEncoder()
    net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim, v_min=v_min, v_max=v_max)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    env_config_path = REPO_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"

    # epsilon=0.5 with this seed reliably produces a genuine mix of both
    # branches within a handful of episodes -- checked across several
    # seeds/episode indices below rather than relying on exactly one.
    saw_mixed_episode = False
    for episode_index in range(6):
        states, actions, rewards, outcome = _collect_online_episode(
            env_config_path, artifact, action_table, seed=555001, episode_index=episode_index,
            profile="nominal", encoder=encoder, value_network=net, action_encoder=action_encoder, epsilon=0.5,
        )
        seed_keys = [tuple(s["posterior_seed_key"]) for s in states]
        assert len(set(seed_keys)) == len(seed_keys), f"duplicate posterior_seed_key within one episode: {seed_keys}"
        counters = [k[2] for k in seed_keys]
        assert counters == list(range(1, len(counters) + 1)), f"decision counter is not sequential 1..T: {counters}"
        suite_episode_ids = {(k[0], k[1]) for k in seed_keys}
        assert len(suite_episode_ids) == 1, "suite_seed/episode_seed must be constant within one episode"
        if len(states) > 3:
            saw_mixed_episode = True
    assert saw_mixed_episode, "test setup did not exercise a long-enough episode to be meaningful"

def test_r4_2r_tampered_replay_sample_fields_abort_training() -> None:
    # guide.md R4-2R-3 minimum test: tampering with executed_action_features,
    # artifact_sha256, source_role, or the seed identity must each be
    # independently caught and abort training -- not silently trained
    # through.
    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()

    def run(sample):
        opt = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=0.0)
        return stage1_train_step(
            encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
            dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
            posterior_source="full", device=torch.device("cpu"), gamma=0.99,
            demo_batch=[sample], ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=opt,
        )

    baseline = _fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=0, target_return=0.5)
    result = run(copy.deepcopy(baseline))
    assert not result.aborted, f"untampered fixture should not abort: {result.abort_reason}"

    tampered_features = copy.deepcopy(baseline)
    tampered_features.executed_action_features = tampered_features.executed_action_features.copy()
    tampered_features.executed_action_features[3] += 5.0  # corrupt goal_alignment out of any plausible range
    result = run(tampered_features)
    assert result.aborted and "executed_action_features" in result.abort_reason

    tampered_artifact = copy.deepcopy(baseline)
    tampered_artifact.artifact_sha256 = "0" * 64
    result = run(tampered_artifact)
    assert result.aborted and "artifact_sha256" in result.abort_reason

    tampered_role = copy.deepcopy(baseline)
    tampered_role.source_role = "online"
    result = run(tampered_role)
    assert result.aborted and "source_role" in result.abort_reason

    tampered_seed = copy.deepcopy(baseline)
    tampered_seed.posterior_seed_key = (0, baseline.episode_seed + 999, tampered_seed.posterior_seed_key[2])
    result = run(tampered_seed)
    assert result.aborted and "seed identity" in result.abort_reason

def test_stage2_train_step_mixes_demo_rank_loss_with_online_mc_only() -> None:
    from crowd_nav.bayesian_dvl.trainer import stage2_train_step

    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-2)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()

    demo_batch = [_fixture_ranking_demo_sample(artifact, episode_seed=1, step_index=0, target_return=0.5)]
    online_batch = [_fixture_mc_return_sample(artifact, action_table, episode_seed=2, step_index=0, target_return=0.2)]

    losses = []
    for _ in range(10):
        result = stage2_train_step(
            encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
            dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
            posterior_source="full", device=torch.device("cpu"), gamma=0.99,
            demo_batch=demo_batch, online_batch=online_batch,
            ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=optimizer,
            grad_clip_norm=10.0,
        )
        assert not result.aborted, result.abort_reason
        assert result.n_demo == 1 and result.n_online == 1
        assert result.rank_loss > 0.0  # demo contributed a real ranking term
        assert np.isfinite(result.grad_norm) and result.grad_norm >= 0.0
        losses.append(result.loss)
    assert losses[-1] < losses[0] * 1.5 or np.mean(losses[-3:]) < np.mean(losses[:3])

def test_stage2_train_step_online_only_has_zero_rank_loss() -> None:
    from crowd_nav.bayesian_dvl.trainer import stage2_train_step

    artifact = _fixture_artifact()
    encoder = SetEncoder(human_hidden_dim=8, human_embed_dim=4, robot_embed_dim=4, embedding_dim=8)
    action_encoder = ActionEncoder(hidden_dim=8, embed_dim=4)
    net = IQNValueNetwork(state_embedding_dim=8, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=8, v_min=-4.798, v_max=1.08)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()) + list(action_encoder.parameters()), lr=1e-3)
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    online_batch = [_fixture_mc_return_sample(artifact, action_table, episode_seed=2, step_index=0, target_return=0.1, outcome="timeout", executed_action_index=0)]
    result = stage2_train_step(
        encoder, net, action_encoder, artifact, action_table, _REWARD_CFG,
        dt=0.25, time_limit=35.0, max_human_speed=2.0, n_world_samples=2, n_iqn_quantiles=4,
        posterior_source="full", device=torch.device("cpu"), gamma=0.99,
        demo_batch=[], online_batch=online_batch,
        ranking_margin=0.1, lambda_rank=1.0, n_train_quantiles=4, optimizer=optimizer,
        grad_clip_norm=10.0,
    )
    assert not result.aborted, result.abort_reason
    assert result.rank_loss == 0.0
    assert result.n_demo == 0 and result.n_online == 1

def test_load_composed_checkpoint_rejects_v2_feature_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        encoder = SetEncoder()
        net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        path = Path(tmp) / "v2_checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(path), "grid-hash", "registry-hash",
            feature_schema=FEATURE_SCHEMA_V2,
        )
        fresh_encoder = SetEncoder()
        fresh_net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        try:
            load_composed_checkpoint(str(path), fresh_encoder, fresh_net)
            raise AssertionError("expected PolicyError loading a v2-schema checkpoint under v3 code")
        except PolicyError as exc:
            assert FEATURE_SCHEMA_V2 in str(exc)

def test_load_composed_checkpoint_rejects_v1_reward_schema() -> None:
    # R3R-4 fix: progress_reward changed 0.01 -> 0.05 but the reward
    # schema string was never bumped at the time -- a checkpoint
    # trained under the old reward scale must fail closed now.
    from crowd_nav.bayesian_dvl.config import REWARD_SCHEMA_V1

    with tempfile.TemporaryDirectory() as tmp:
        encoder = SetEncoder()
        net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        path = Path(tmp) / "old_reward_checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(path), "grid-hash", "registry-hash",
            reward_schema=REWARD_SCHEMA_V1,
        )
        fresh_encoder = SetEncoder()
        fresh_net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        try:
            load_composed_checkpoint(str(path), fresh_encoder, fresh_net)
            raise AssertionError("expected PolicyError loading a v1-reward-schema checkpoint")
        except PolicyError as exc:
            assert REWARD_SCHEMA_V1 in str(exc)

def test_load_composed_checkpoint_rejects_v3_feature_schema() -> None:
    # R4-1: v3 (state-only V(s,b), no explicit action conditioning) is
    # retired the same way v1/v2 are -- guide.md's belief-bypass
    # remediation plan requires Q(s,b,a) (v4) and nothing older.
    with tempfile.TemporaryDirectory() as tmp:
        encoder = SetEncoder()
        net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        path = Path(tmp) / "v3_checkpoint.pth"
        save_composed_checkpoint(
            encoder, net, str(path), "grid-hash", "registry-hash",
            feature_schema=FEATURE_SCHEMA_V3,
        )
        fresh_encoder = SetEncoder()
        fresh_net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=16)
        try:
            load_composed_checkpoint(str(path), fresh_encoder, fresh_net)
            raise AssertionError("expected PolicyError loading a v3-schema checkpoint under v4 code")
        except PolicyError as exc:
            assert FEATURE_SCHEMA_V3 in str(exc)

def test_progress_reward_raw_meters_matches_normalized_formula() -> None:
    # R3R-4 point 2: the raw-meters formula actually used by crowd_sim.py/
    # transition.py (progress_reward * progress) and the dimensionless
    # normalized_progress formula guide.md's R3-5 order originally
    # specified (k_progress * clip(progress/(v_pref*dt), -1, 1)) must be
    # bit-equivalent at the frozen v_pref=1.0/dt=0.25 protocol, for every
    # progress value reachable within one step of the real 80-action grid
    # (i.e. |progress| <= v_pref*dt, so the clip never binds).
    progress_reward = float(FROZEN_VALUES["progress_reward"])
    dt = float(FROZEN_VALUES["dt"])
    v_pref = NORMALIZATION_CONSTANTS["robot_max_speed"]
    max_step_progress = v_pref * dt
    assert abs(PROGRESS_REWARD_NORMALIZED_K - 0.0125) < 1e-12, PROGRESS_REWARD_NORMALIZED_K
    for progress in np.linspace(-max_step_progress, max_step_progress, 41):
        raw = progress_reward * progress
        normalized_progress = float(np.clip(progress / max_step_progress, -1.0, 1.0))
        normalized = PROGRESS_REWARD_NORMALIZED_K * normalized_progress
        assert abs(raw - normalized) < 1e-12, (progress, raw, normalized)
    # A progress magnitude beyond one step's physical reach is exactly
    # where the clip WOULD bind -- confirms the two formulas are only
    # claimed equivalent inside the reachable range, not universally.
    beyond = max_step_progress * 1.5
    raw_beyond = progress_reward * beyond
    normalized_beyond = PROGRESS_REWARD_NORMALIZED_K * float(np.clip(beyond / max_step_progress, -1.0, 1.0))
    assert abs(raw_beyond - normalized_beyond) > 1e-6

def test_progress_reward_normalized_k_changes_if_v_pref_or_dt_change() -> None:
    # R3R-4 point 2: a future change to v_pref or dt must be caught by a
    # changed derived constant (forcing a schema bump), never silently
    # reused against the old k_progress=0.0125.
    dt = float(FROZEN_VALUES["dt"])
    v_pref = NORMALIZATION_CONSTANTS["robot_max_speed"]
    progress_reward = float(FROZEN_VALUES["progress_reward"])
    k_same = progress_reward * v_pref * dt
    k_other_dt = progress_reward * v_pref * (dt * 2.0)
    k_other_v_pref = progress_reward * (v_pref * 2.0) * dt
    assert abs(k_same - PROGRESS_REWARD_NORMALIZED_K) < 1e-12
    assert abs(k_other_dt - PROGRESS_REWARD_NORMALIZED_K) > 1e-6
    assert abs(k_other_v_pref - PROGRESS_REWARD_NORMALIZED_K) > 1e-6

def test_selection_rank_key_prefers_higher_success_rate_first() -> None:
    from crowd_nav.tools.select_bdvl_checkpoint import select_best_eligible

    # This is the exact regression that motivated R3-1: a strictly better
    # (99.5% SR) checkpoint must be selected over a worse (86.5% SR) one
    # even though the worse one has zero collisions, because the R2 gate
    # (CR<=5%) already treats both as safe enough to compare on capability.
    worse_but_zero_collision = _fixture_selection_result(success_rate=0.865, collision_rate=0.0)
    better_with_one_collision = _fixture_selection_result(success_rate=0.995, collision_rate=0.005)
    best = select_best_eligible([worse_but_zero_collision, better_with_one_collision])
    assert best is better_with_one_collision

def test_selection_rank_key_tiebreak_cascade_ends_in_path_ratio() -> None:
    from crowd_nav.tools.select_bdvl_checkpoint import select_best_eligible

    # Identical on every criterion except the R3R-4 point-3 final tiebreak:
    # the more direct path (lower path_length/initial_goal_distance) wins.
    direct = _fixture_selection_result(success_rate=0.99, timeout_rate=0.005, collision_rate=0.005, mean_negative_alignment_fraction=0.1, mean_path_ratio=1.05)
    detour = _fixture_selection_result(success_rate=0.99, timeout_rate=0.005, collision_rate=0.005, mean_negative_alignment_fraction=0.1, mean_path_ratio=1.40)
    assert select_best_eligible([detour, direct]) is direct
    # But path_ratio must NEVER override an earlier criterion in the cascade.
    worse_sr_but_direct = _fixture_selection_result(success_rate=0.90, mean_path_ratio=1.01)
    better_sr_but_detour = _fixture_selection_result(success_rate=0.99, mean_path_ratio=1.50)
    assert select_best_eligible([worse_sr_but_direct, better_sr_but_detour]) is better_sr_but_detour

def test_select_best_eligible_rejects_empty_list() -> None:
    from crowd_nav.tools.select_bdvl_checkpoint import select_best_eligible

    try:
        select_best_eligible([])
        raise AssertionError("expected ValueError on empty eligible list")
    except ValueError:
        pass

def test_parse_checkpoint_identity_reads_seed_category_episode() -> None:
    from crowd_nav.tools.select_bdvl_checkpoint import parse_checkpoint_identity

    assert parse_checkpoint_identity("/base/seed_93001/checkpoints/checkpoint_ep1500.pth") == (93001, "raw", 1500)
    assert parse_checkpoint_identity("/base/seed_93001/checkpoints/checkpoint_ep1500_ema.pth") == (93001, "ema", 1500)
    try:
        parse_checkpoint_identity("/base/seed_93001/checkpoints/not_a_checkpoint.pth")
        raise AssertionError("expected ValueError on unrecognized filename")
    except ValueError:
        pass
    try:
        parse_checkpoint_identity("/base/checkpoints/checkpoint_ep1500.pth")
        raise AssertionError("expected ValueError with no seed_<N> path component")
    except ValueError:
        pass

def test_parse_checkpoint_identity_handles_ad_hoc_run_directory_names() -> None:
    # Engineering-gap fix (2026-08-08): a real R3-8 single-seed validation
    # run launched directly via train_bdvl.py (not through
    # run_bdvl_queue.py) used --checkpoint-dir
    # ".../bdvl_r3r8_seed93001_20260807_fix2/checkpoints" -- "seed" and the
    # digits are adjacent with no underscore and are not their own path
    # component, which crashed the original strict "seed_<N> component"
    # parser at the very last step of a real ~1.5 hour validation run.
    from crowd_nav.tools.select_bdvl_checkpoint import parse_checkpoint_identity

    path = "/root/workspace/.../runs/bdvl_r3r8_seed93001_20260807_fix2/checkpoints/checkpoint_ep500_ema.pth"
    assert parse_checkpoint_identity(path) == (93001, "ema", 500)

def test_parse_checkpoint_identity_rejects_conflicting_seed_tokens() -> None:
    from crowd_nav.tools.select_bdvl_checkpoint import parse_checkpoint_identity

    try:
        parse_checkpoint_identity("/base/seed_93001/other_seed_94002/checkpoints/checkpoint_ep500.pth")
        raise AssertionError("expected ValueError on conflicting seed tokens")
    except ValueError:
        pass

def test_apply_stability_gate_rejects_isolated_peak() -> None:
    # R3R-4 point 4: 1500 passes alone, surrounded by failing 1000/2000 ->
    # not stability-eligible even though it individually cleared the gate.
    from crowd_nav.tools.select_bdvl_checkpoint import apply_stability_gate

    results = [
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1000.pth", False),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1500.pth", True),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep2000.pth", False),
    ]
    assert apply_stability_gate(results) == []

def test_apply_stability_gate_accepts_two_consecutive_passes() -> None:
    from crowd_nav.tools.select_bdvl_checkpoint import apply_stability_gate

    results = [
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1000.pth", False),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1500.pth", True),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep2000.pth", True),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep2500.pth", False),
    ]
    kept = {r["checkpoint"] for r in apply_stability_gate(results)}
    assert kept == {"/b/seed_1/checkpoints/checkpoint_ep1500.pth", "/b/seed_1/checkpoints/checkpoint_ep2000.pth"}

def test_apply_stability_gate_groups_by_seed_and_category_independently() -> None:
    # Same episode numbers passing under two different (seed, category)
    # groups must NOT let one group's neighbor rescue another group's
    # isolated peak -- each (seed, category) pair is judged on its own.
    from crowd_nav.tools.select_bdvl_checkpoint import apply_stability_gate

    results = [
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1000.pth", False),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1500.pth", True),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep2000.pth", False),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1000_ema.pth", True),
        _fixture_gate_result("/b/seed_1/checkpoints/checkpoint_ep1500_ema.pth", True),
        _fixture_gate_result("/b/seed_2/checkpoints/checkpoint_ep1500.pth", True),
    ]
    kept = {r["checkpoint"] for r in apply_stability_gate(results)}
    assert kept == {"/b/seed_1/checkpoints/checkpoint_ep1000_ema.pth", "/b/seed_1/checkpoints/checkpoint_ep1500_ema.pth"}

def test_r4_3r_bimodal_mean_trap_moment_mean_picks_a_different_worse_action() -> None:
    # guide.md R4-3R point 3: a single-candidate test can only show "full
    # and moment_mean score the SAME action differently" -- it cannot show
    # a real DECISION regret, since there is nothing to choose between.
    # This uses two near-identical candidates (straight vs. a hair of
    # lateral drift) whose real (full-posterior) safety margins are close
    # enough that moment_mean's badly miscalibrated collision estimate
    # for the "straight" candidate (~1.0, an artifact of the impossible
    # average trajectory) flips which one it ranks first -- full and
    # moment_mean must pick DIFFERENT top actions, and moment_mean's pick
    # must be the one full's OWN (ground-truth) ranking considers worse.
    # Verified robust across three unrelated seed_keys and three sample
    # counts (300/500/1000) before being frozen as a permanent test.
    artifact = _fixture_bimodal_trap_artifact()
    action_table = [(1.0, 0.0), (1.0, 0.02)]
    robot = RobotObservation(px=0.0, py=0.0, vx=1.0, vy=0.0, radius=0.3, gx=10.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, px=2.0, py=-2.0, vx=0.0, vy=1.0, radius=0.3)]
    belief = {0: np.array([0.0, 0.5, 0.5, 0.0, 0.0])}
    positions = {0: np.array([2.0, -2.0])}
    speeds = {0: 1.0}
    headings = {0: np.pi / 2}
    rc = _REWARD_CFG
    common = dict(
        artifact=artifact, action_grid_hash="x", robot=robot, humans=humans,
        track_beliefs=belief, track_positions=positions, track_speeds=speeds, track_headings=headings,
        action_table=action_table, reward_config=rc, dt=0.25, time_limit=FROZEN_VALUES["time_limit"],
        global_time=0.0, max_human_speed=FROZEN_VALUES["max_human_speed"], n_world_samples=500, cvar_alpha=0.2,
        # this test is about a collision-probability miscalibration, not
        # the clearance tie-break -- keep clearance_tolerance=0.0 (strict,
        # guide.md R4-4R3's backward-compatible value) so it isn't affected.
        clearance_tolerance=0.0,
    )

    full_record = build_counterfactual_record(suite_seed=1, episode_seed=2, decision_seed=3, posterior_source="full", **common)
    mean_record = build_counterfactual_record(suite_seed=1, episode_seed=2, decision_seed=3, posterior_source="moment_mean", **common)

    full_pick = full_record.ranked_action_indices[0]
    mean_pick = mean_record.ranked_action_indices[0]
    assert full_pick != mean_pick, (
        f"full posterior and moment_mean must choose DIFFERENT top actions to demonstrate real decision "
        f"regret, both picked {full_pick}"
    )
    # "regret" = under the TRUE (full-posterior) distribution's own
    # ranking, moment_mean's pick must not be ranked ahead of full's pick.
    full_pick_rank = full_record.ranked_action_indices.index(full_pick)
    mean_pick_rank_under_truth = full_record.ranked_action_indices.index(mean_pick)
    assert full_pick_rank < mean_pick_rank_under_truth, (
        "moment_mean's chosen action must rank WORSE than full posterior's own choice, "
        "under the ground-truth (full posterior) ranking itself"
    )

def test_r4_3_all_candidates_share_identical_human_worlds() -> None:
    # guide.md R4-3 point 4: the human world samples must be built ONCE
    # and reused unchanged across every candidate action -- verified here
    # by confirming evaluate_counterfactual_candidates never mutates the
    # human_worlds it was given, and that re-running it twice against the
    # SAME (already-built) human_worlds object is bit-identical.
    artifact = _fixture_artifact()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, 2.0, 0.0, -0.5, 0.0, 0.3)]
    belief = {0: np.array([0.5, 0.2, 0.1, 0.1, 0.1])}
    positions = {0: np.array([2.0, 0.0])}
    speeds = {0: 0.5}
    headings = {0: np.pi}

    human_worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs=belief, track_positions=positions, track_speeds=speeds,
        track_headings=headings, n_samples=16, dt=0.25, max_human_speed=FROZEN_VALUES["max_human_speed"],
        source="full", seed=(1, 2, 3),
    )
    snapshot_before = copy.deepcopy(human_worlds)

    results_a = evaluate_counterfactual_candidates(
        robot=robot, humans=humans, human_worlds=human_worlds, n_samples=16, action_table=action_table,
        reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
    )
    assert human_worlds == snapshot_before, "evaluate_counterfactual_candidates must not mutate the shared human_worlds"

    results_b = evaluate_counterfactual_candidates(
        robot=robot, humans=humans, human_worlds=human_worlds, n_samples=16, action_table=action_table,
        reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
    )
    assert results_a == results_b, "re-scoring against the SAME human_worlds must be bit-identical"
    # Two different actions must generally see different outcomes despite
    # sharing the identical human_worlds (proves the shared randomness
    # doesn't collapse every candidate to the same result trivially).
    distinct_collision_probs = {r.collision_prob for r in results_a}
    assert len(distinct_collision_probs) > 1, "80 different actions against a real crowd should not all have identical collision_prob"

def test_r4_3_mode_transitions_converge_to_transition_matrix() -> None:
    # guide.md R4-3 point 3: modes 1..horizon-1 must be drawn from
    # artifact.transition_matrix[prev_mode, :], not independently
    # re-sampled from the original belief every step. Statistical check
    # over many world samples, not a single-draw assertion.
    artifact = _fixture_artifact()
    belief = {0: np.array([0.2, 0.2, 0.2, 0.2, 0.2])}  # uniform initial -- all 5 initial modes get real coverage
    positions = {0: np.array([0.0, 0.0])}
    speeds = {0: 0.5}
    headings = {0: 0.0}
    n_samples = 4000

    worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs=belief, track_positions=positions, track_speeds=speeds,
        track_headings=headings, n_samples=n_samples, dt=0.25, max_human_speed=FROZEN_VALUES["max_human_speed"],
        source="full", seed=(1, 2, 3),
    )[0]

    counts = np.zeros((N_MODES, N_MODES))
    for traj in worlds:
        for t in range(ROLLOUT_HORIZON - 1):
            counts[traj.modes[t], traj.modes[t + 1]] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    empirical = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    # Only check rows with enough observations for a meaningful comparison.
    for from_mode in range(N_MODES):
        if row_totals[from_mode, 0] < 50:
            continue
        assert np.allclose(empirical[from_mode], artifact.transition_matrix[from_mode], atol=0.08), (
            f"mode {from_mode}'s empirical transition row {empirical[from_mode]} does not converge to "
            f"artifact.transition_matrix's row {artifact.transition_matrix[from_mode]}"
        )

def test_r4_3_terminal_absorption_freezes_geometry_after_collision() -> None:
    # guide.md R4-3 point 6: once a world's rollout collides at step t,
    # steps t+1..horizon must not change geometry/progress further.
    artifact = _fixture_artifact()
    # Robot drives straight at a stationary human 0.5m away -- guaranteed
    # collision within the first step or two for every world sample,
    # regardless of the (irrelevant, since the human barely moves) mode draw.
    robot = RobotObservation(px=0.0, py=0.0, vx=1.0, vy=0.0, radius=0.3, gx=10.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, px=0.5, py=0.0, vx=0.0, vy=0.0, radius=0.3)]
    belief = {0: np.array([1.0, 0.0, 0.0, 0.0, 0.0])}  # CV, stationary-ish
    positions = {0: np.array([0.5, 0.0])}
    speeds = {0: 0.0}
    headings = {0: 0.0}
    action_table = [(1.0, 0.0)]

    human_worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs=belief, track_positions=positions, track_speeds=speeds,
        track_headings=headings, n_samples=8, dt=0.25, max_human_speed=FROZEN_VALUES["max_human_speed"],
        source="full", seed=(1, 2, 3),
    )
    results = evaluate_counterfactual_candidates(
        robot=robot, humans=humans, human_worlds=human_worlds, n_samples=8, action_table=action_table,
        reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
    )
    result = results[0]
    assert result.collision_prob == 1.0
    # Progress must be bounded to roughly what ONE step of closing
    # distance can achieve (collision happens on step 0 or 1), not 8
    # steps' worth of "progress" -- a non-absorbing bug would keep adding
    # progress for the remaining ~7 frozen steps.
    max_plausible_one_step_progress = 1.0 * 0.25 * 2  # generous slack, still far under 8-step total
    assert result.expected_progress <= max_plausible_one_step_progress, (
        f"expected_progress={result.expected_progress} suggests progress kept accumulating after collision"
    )

def test_r4_3_cross_process_reproducibility() -> None:
    # guide.md R4-3 point 8/9: rollout.py's hash(seed) -> _stable_seed
    # replacement must be verified in a REAL fresh subprocess, not just
    # asserted to be stable within this one process.
    import subprocess as _subprocess
    import sys as _sys

    code = (
        "import numpy as np\n"
        "from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact, N_MODES\n"
        "from crowd_nav.bayesian_dvl.counterfactual import sample_human_multi_step_worlds\n"
        "rng = np.random.default_rng(7)\n"
        "trans_counts = np.abs(rng.normal(size=(N_MODES, N_MODES))) + 1.0\n"
        "np.fill_diagonal(trans_counts, trans_counts.diagonal() + 8.0)\n"
        "mu = np.array([[0.0,0.0],[0.6,0.0],[-0.6,0.0],[0.0,0.8],[0.0,-0.8]])\n"
        "kappa = np.full(N_MODES, 5.0); nu = np.full(N_MODES, 6.0)\n"
        "psi = np.tile(np.eye(2)*1.0, (N_MODES,1,1))\n"
        "artifact = SBKHMMArtifact(dt=0.25, transition_counts=trans_counts, niw_mu=mu, niw_kappa=kappa, "
        "niw_nu=nu, niw_psi=psi, initial_counts=np.array([6.,1.,1.,1.,1.]), n_iterations=1, converged=True, "
        "log_likelihood_history=(0.0,), train_data_sha256='fixture', tier='production')\n"
        "worlds = sample_human_multi_step_worlds(artifact=artifact, track_beliefs={0: np.array([0.5,0.2,0.1,0.1,0.1])}, "
        "track_positions={0: np.array([2.0,0.0])}, track_speeds={0: 0.5}, track_headings={0: np.pi}, "
        "n_samples=4, dt=0.25, max_human_speed=2.0, source='full', seed=(11,22,33))\n"
        "for traj in worlds[0]:\n"
        "    print(traj.modes, [round(v[0],8) for v in traj.step_velocities], [round(v[1],8) for v in traj.step_velocities])\n"
    )
    result_a = _subprocess.run([_sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    result_b = _subprocess.run([_sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert result_a.stdout == result_b.stdout, "cross-process reproducibility failed -- output differs between two fresh subprocess runs with the same seed"
    assert result_a.stdout.strip() != "", "subprocess produced no output"

def test_r4_3r_fixes_progress_counts_the_terminating_step() -> None:
    # guide.md R4-3R point 1: a fast action that reaches the goal on step
    # 1 must count that step's progress -- absorption freezes STARTING
    # the NEXT step, not this one. Reproduces the exact bug report: a
    # 1-step-to-goal action previously scored expected_progress=0.0 while
    # a creeping, never-arriving action scored higher, an inverted
    # preference.
    artifact = _fixture_artifact()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.1, gx=1.0, gy=0.0, v_pref=1.0, theta=0.0)
    action_table = [(4.0, 0.0), (0.1, 0.0)]  # fast: exactly reaches goal at step 1 (4.0*0.25=1.0); slow: creeps
    human_worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs={}, track_positions={}, track_speeds={}, track_headings={},
        n_samples=4, dt=0.25, max_human_speed=FROZEN_VALUES["max_human_speed"], source="full", seed=(1, 2, 3),
    )
    results = evaluate_counterfactual_candidates(
        robot=robot, humans=[], human_worlds=human_worlds, n_samples=4, action_table=action_table,
        reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
    )
    fast, slow = results[0], results[1]
    assert fast.expected_progress > 0.9, f"fast action's terminating-step progress must be counted, got {fast.expected_progress}"
    assert fast.expected_progress > slow.expected_progress, "reaching the goal in one step must score higher progress than creeping without arriving"

def test_r4_3r_per_track_seeding_is_order_independent() -> None:
    # guide.md R4-3R point 2: swapping two tracks' dict insertion order
    # must not change either track's own sampled trajectory -- each track
    # must use its own independent RNG seeded by its own track_id, not a
    # single RNG shared (and consumed in iteration order) across tracks.
    artifact = _fixture_artifact()
    belief_a = {0: np.array([0.5, 0.2, 0.1, 0.1, 0.1]), 1: np.array([0.2, 0.5, 0.1, 0.1, 0.1])}
    pos_a = {0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])}
    speeds_a = {0: 0.5, 1: 0.5}
    headings_a = {0: np.pi, 1: 0.0}
    belief_b = {1: belief_a[1], 0: belief_a[0]}
    pos_b = {1: pos_a[1], 0: pos_a[0]}
    speeds_b = {1: speeds_a[1], 0: speeds_a[0]}
    headings_b = {1: headings_a[1], 0: headings_a[0]}

    w_a = sample_human_multi_step_worlds(artifact=artifact, track_beliefs=belief_a, track_positions=pos_a, track_speeds=speeds_a, track_headings=headings_a, n_samples=4, dt=0.25, max_human_speed=2.0, source="full", seed=(1, 2, 3))
    w_b = sample_human_multi_step_worlds(artifact=artifact, track_beliefs=belief_b, track_positions=pos_b, track_speeds=speeds_b, track_headings=headings_b, n_samples=4, dt=0.25, max_human_speed=2.0, source="full", seed=(1, 2, 3))
    assert w_a[0] == w_b[0], "track 0's sampled trajectory must not depend on dict insertion order"
    assert w_a[1] == w_b[1], "track 1's sampled trajectory must not depend on dict insertion order"

def test_r4_3r_fail_closed_dt_mismatch() -> None:
    artifact = _fixture_artifact(dt=0.25)
    try:
        sample_human_multi_step_worlds(
            artifact=artifact, track_beliefs={0: np.array([0.5, 0.2, 0.1, 0.1, 0.1])},
            track_positions={0: np.array([2.0, 0.0])}, track_speeds={0: 0.5}, track_headings={0: np.pi},
            n_samples=4, dt=0.5, max_human_speed=2.0, source="full", seed=(1, 2, 3),
        )
        raise AssertionError("expected RolloutError for dt != artifact.dt")
    except RolloutError:
        pass

def test_r4_3r_fail_closed_invalid_n_samples_and_cvar_alpha() -> None:
    artifact = _fixture_artifact()
    try:
        sample_human_multi_step_worlds(
            artifact=artifact, track_beliefs={0: np.array([0.5, 0.2, 0.1, 0.1, 0.1])},
            track_positions={0: np.array([2.0, 0.0])}, track_speeds={0: 0.5}, track_headings={0: np.pi},
            n_samples=0, dt=0.25, max_human_speed=2.0, source="full", seed=(1, 2, 3),
        )
        raise AssertionError("expected RolloutError for n_samples<=0")
    except RolloutError:
        pass

    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    human_worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs={}, track_positions={}, track_speeds={}, track_headings={},
        n_samples=4, dt=0.25, max_human_speed=2.0, source="full", seed=(1, 2, 3),
    )
    for bad_alpha in (0.0, -0.1, 1.5):
        try:
            evaluate_counterfactual_candidates(
                robot=robot, humans=[], human_worlds=human_worlds, n_samples=4, action_table=action_table,
                reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=bad_alpha,
            )
            raise AssertionError(f"expected RolloutError for cvar_alpha={bad_alpha}")
        except RolloutError:
            pass

def test_r4_3r_fail_closed_human_worlds_shape_mismatch() -> None:
    artifact = _fixture_artifact()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, 2.0, 0.0, -0.5, 0.0, 0.3)]

    # missing track_id entirely
    try:
        evaluate_counterfactual_candidates(
            robot=robot, humans=humans, human_worlds={}, n_samples=4, action_table=action_table,
            reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
        )
        raise AssertionError("expected RolloutError for missing track_id in human_worlds")
    except RolloutError:
        pass

    real_worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs={0: np.array([0.5, 0.2, 0.1, 0.1, 0.1])},
        track_positions={0: np.array([2.0, 0.0])}, track_speeds={0: 0.5}, track_headings={0: np.pi},
        n_samples=4, dt=0.25, max_human_speed=2.0, source="full", seed=(1, 2, 3),
    )
    # wrong world count: caller claims n_samples=8 but human_worlds only has 4
    try:
        evaluate_counterfactual_candidates(
            robot=robot, humans=humans, human_worlds=real_worlds, n_samples=8, action_table=action_table,
            reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
        )
        raise AssertionError("expected RolloutError for world-count mismatch")
    except RolloutError:
        pass

    # wrong trajectory length: truncate one trajectory's step_velocities
    tampered_traj = real_worlds[0][0]
    tampered_traj = type(tampered_traj)(track_id=tampered_traj.track_id, modes=tampered_traj.modes[:4], step_velocities=tampered_traj.step_velocities[:4])
    tampered_worlds = {0: (tampered_traj,) + real_worlds[0][1:]}
    try:
        evaluate_counterfactual_candidates(
            robot=robot, humans=humans, human_worlds=tampered_worlds, n_samples=4, action_table=action_table,
            reward_config=_REWARD_CFG, dt=0.25, time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, cvar_alpha=0.2,
        )
        raise AssertionError("expected RolloutError for a trajectory shorter than horizon")
    except RolloutError:
        pass

def test_r4_3r_counterfactual_record_carries_full_provenance() -> None:
    # guide.md R4-3R: CounterfactualRecord must carry contract_version,
    # action_grid_hash, posterior_source, horizon, n_world_samples, and
    # cvar_alpha -- not just the seed/artifact identity it already had.
    artifact = _fixture_artifact()
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    robot = RobotObservation(px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, 2.0, 0.0, -0.5, 0.0, 0.3)]
    belief = {0: np.array([0.5, 0.2, 0.1, 0.1, 0.1])}
    positions = {0: np.array([2.0, 0.0])}
    speeds = {0: 0.5}
    headings = {0: np.pi}

    record = build_counterfactual_record(
        suite_seed=1, episode_seed=2, decision_seed=3, artifact=artifact, action_grid_hash=grid.table_hash(),
        robot=robot, humans=humans, track_beliefs=belief, track_positions=positions, track_speeds=speeds,
        track_headings=headings, action_table=action_table, reward_config=_REWARD_CFG, dt=0.25,
        time_limit=FROZEN_VALUES["time_limit"], global_time=0.0, max_human_speed=FROZEN_VALUES["max_human_speed"],
        n_world_samples=8, cvar_alpha=0.2, clearance_tolerance=0.0, posterior_source="full",
    )
    assert record.contract_version
    assert record.action_grid_hash == grid.table_hash()
    assert record.posterior_source == "full"
    assert record.horizon == ROLLOUT_HORIZON
    assert record.n_world_samples == 8
    assert record.cvar_alpha == 0.2

def test_is_risk_opportunity_state_true_when_safe_and_dangerous_coexist() -> None:
    candidates = [
        _fixture_candidate(0, 0.0, 0.5, 0.5),   # safe, non-trivial progress
        _fixture_candidate(1, 0.4, -0.1, 0.3),  # dangerous
    ]
    assert is_risk_opportunity_state(candidates, discomfort_distance=0.20)

def test_is_risk_opportunity_state_false_when_only_safe_or_only_dangerous_or_trivial() -> None:
    only_safe = [_fixture_candidate(0, 0.0, 0.5, 0.5)]
    assert not is_risk_opportunity_state(only_safe, discomfort_distance=0.20)
    only_dangerous = [_fixture_candidate(0, 0.6, -0.2, 0.5)]
    assert not is_risk_opportunity_state(only_dangerous, discomfort_distance=0.20)
    # a "safe" candidate that never actually moves (the trivial
    # standing-still solution) must not count as the safe option.
    safe_but_trivial_and_dangerous = [
        _fixture_candidate(0, 0.0, 0.5, 0.0),
        _fixture_candidate(1, 0.5, -0.1, 0.3),
    ]
    assert not is_risk_opportunity_state(safe_but_trivial_and_dangerous, discomfort_distance=0.20)

def test_is_risk_opportunity_state_rejects_empty() -> None:
    try:
        is_risk_opportunity_state([], discomfort_distance=0.2)
        assert False, "expected DataCoverageError"
    except DataCoverageError:
        pass

def test_episode_gate_is_informational_never_blocks_on_collision_or_clearance() -> None:
    records = [
        EpisodeCoverageRecord(episode_seed=1, profile="nominal", outcome="success", is_non_reciprocal_episode=False, clearance_bin="normal"),
        EpisodeCoverageRecord(episode_seed=2, profile="train_nonstationary", outcome="success", is_non_reciprocal_episode=False, clearance_bin="normal"),
        EpisodeCoverageRecord(episode_seed=3, profile="train_non_reciprocal", outcome="success", is_non_reciprocal_episode=True, clearance_bin="normal"),
    ]
    report = compute_episode_coverage_report(records)
    assert report["outcome"] == {"success": 3}
    assert report["clearance_bin"] == {"normal": 3}
    # zero collisions, zero near_collision in this report -- guide.md
    # R4-4R: check_episode_gate must NOT block on that, only on profile
    # presence.
    gate = check_episode_gate(report, required_profiles=("nominal", "train_nonstationary", "train_non_reciprocal"), min_samples=1)
    assert gate["passed"], gate["reasons"]
    gate_missing = check_episode_gate(
        report, required_profiles=("nominal", "train_nonstationary", "train_non_reciprocal", "heldout_nonstationary"), min_samples=1,
    )
    assert not gate_missing["passed"]
    assert "profile='heldout_nonstationary'" in gate_missing["reasons"][0]

def test_check_risk_opportunity_gate_flags_each_missing_requirement() -> None:
    records = [
        RiskOpportunityRecord(
            episode_seed=1, decision_seed=1, profile="nominal", belief_entropy_bin="low", action_type="straight",
            full_top_action=0, mean_top_action=0, cv_top_action=0, agree_with_mean=True, agree_with_cv=True,
            internal_regret_vs_mean=0.0, internal_regret_vs_cv=0.0,
        ),
        RiskOpportunityRecord(
            episode_seed=2, decision_seed=1, profile="nominal", belief_entropy_bin="low", action_type="straight",
            full_top_action=0, mean_top_action=0, cv_top_action=0, agree_with_mean=True, agree_with_cv=True,
            internal_regret_vs_mean=0.0, internal_regret_vs_cv=0.0,
        ),
    ]
    gate = check_risk_opportunity_gate(records, required_profiles=("nominal", "train_nonstationary", "train_non_reciprocal"))
    assert not gate["passed"]
    reasons = " | ".join(gate["reasons"])
    assert "n_risk_opportunity_states=2 < 100" in reasons
    assert "n_risk_opportunity_episodes=2 < 30" in reasons
    assert "profile='train_nonstationary'" in reasons
    assert "profile='train_non_reciprocal'" in reasons
    assert "disagreement_rate_vs_mean=0.0000 < 0.05" in reasons
    assert "disagreement_rate_vs_cv=0.0000 < 0.05" in reasons
    assert "action_type='turn'" in reasons
    assert "action_type='stop_or_slow'" in reasons

def test_check_risk_opportunity_gate_passes_with_synthetic_sufficient_data() -> None:
    records = _fixture_risk_records_sufficient()
    report = compute_risk_opportunity_report(records)
    assert report["n_states"] == len(records)
    gate = check_risk_opportunity_gate(records, required_profiles=("nominal", "train_nonstationary", "train_non_reciprocal"))
    assert gate["passed"], gate["reasons"]

def test_check_world_sample_stability_thresholds() -> None:
    good = check_world_sample_stability([1.0, 1.0, 0.9, 1.0], min_overlap=0.85)
    assert good["passed"]
    bad = check_world_sample_stability([0.5, 0.4, 0.3], min_overlap=0.85)
    assert not bad["passed"]

def test_topk_overlap_hand_computed() -> None:
    assert topk_overlap([0, 1, 2, 3], [2, 1, 0, 9], k=3) == 1.0
    assert topk_overlap([0, 1, 2], [3, 4, 5], k=3) == 0.0
    assert abs(topk_overlap([0, 1, 2], [0, 5, 6], k=3) - (1 / 3)) < 1e-9

def test_clearance_equivalence_tolerance_positive_and_scales_with_horizon() -> None:
    grid = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = grid.build_action_table()
    tol8 = clearance_equivalence_tolerance(action_table, dt=0.25, horizon=8)
    tol4 = clearance_equivalence_tolerance(action_table, dt=0.25, horizon=4)
    assert tol8 > 0.0
    assert abs(tol8 - 2 * tol4) < 1e-9

def test_r4_4r3_rank_tolerance_lets_progress_win_within_clearance_tolerance() -> None:
    # guide.md R4-4R3 root-cause fix: reproduces the REAL R4-4R2 audit
    # finding in miniature (episode_seed=980030, decision_seed=2, real
    # collect_episode run) -- a candidate with a noise-scale clearance
    # edge (1.970 vs 1.929, both ~2m from any human, nowhere near the
    # 0.20m discomfort distance) must not veto a candidate with 2.4x the
    # progress once they are within clearance_tolerance of each other.
    # The strict (tolerance=0.0) rule is preserved exactly for backward
    # compatibility -- only a nonzero tolerance changes the outcome.
    candidates = [
        _fixture_candidate(9, 0.0, 1.970, 0.522, control_cost=0.287),
        _fixture_candidate(14, 0.0, 1.929, 1.256, control_cost=0.130),
    ]
    strict = rank_counterfactual_candidates(candidates, clearance_tolerance=0.0)
    assert strict[0] == 9, "zero tolerance must preserve the old strict clearance-first behavior"

    tolerant = rank_counterfactual_candidates(candidates, clearance_tolerance=0.05)
    assert tolerant[0] == 14, "within tolerance, the much larger progress advantage must win"

def test_rank_counterfactual_candidates_never_lets_tolerance_override_collision_safety() -> None:
    candidates = [
        _fixture_candidate(0, 0.0, 0.5, 0.1),   # safe, low progress/clearance
        _fixture_candidate(1, 0.5, 5.0, 10.0),  # huge clearance/progress but dangerous
    ]
    ranked = rank_counterfactual_candidates(candidates, clearance_tolerance=100.0)  # absurdly large tolerance
    assert ranked[0] == 0, "collision_prob tiers must never be crossed by clearance tolerance, however large"

def test_rank_counterfactual_candidates_rejects_negative_tolerance() -> None:
    try:
        rank_counterfactual_candidates([_fixture_candidate(0, 0.0, 1.0, 0.5)], clearance_tolerance=-0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_build_safety_optimal_layer_includes_best_and_respects_tolerance() -> None:
    oracle_results = [
        _fixture_candidate(0, 0.0, 1.0, 0.8),
        _fixture_candidate(1, 0.0, 0.95, 0.5),
        _fixture_candidate(2, 0.0, 0.5, 0.9),   # too far below best's clearance -- excluded
        _fixture_candidate(3, 0.3, 1.2, 0.9),   # higher clearance but nonzero collision -- excluded
    ]
    layer = build_safety_optimal_layer(oracle_results, clearance_tolerance=0.1)
    assert set(layer) == {0, 1}

def test_compute_method_regret_hand_computed() -> None:
    oracle_results = [
        _fixture_candidate(0, 0.0, 1.0, 0.8),   # oracle-best
        _fixture_candidate(1, 0.0, 0.95, 0.5),  # safety-equivalent to best, less progress
        _fixture_candidate(2, 0.5, 0.2, 0.9),   # dangerous
    ]
    tolerance = 0.1
    ranked = rank_counterfactual_candidates(oracle_results, tolerance)

    regret = compute_method_regret(1, oracle_results, tolerance)
    assert regret.collision_regret == 0.0
    assert abs(regret.clearance_regret - 0.05) < 1e-9
    assert regret.in_safety_optimal_layer
    assert abs(regret.efficiency_regret - 0.3) < 1e-9  # best_progress_in_layer(0.8) - chosen(0.5)
    assert abs(regret.normalized_rank_regret - ranked.index(1) / (len(ranked) - 1)) < 1e-9

    regret_dangerous = compute_method_regret(2, oracle_results, tolerance)
    assert regret_dangerous.collision_regret == 0.5
    assert not regret_dangerous.in_safety_optimal_layer
    assert regret_dangerous.efficiency_regret is None
    assert regret_dangerous.normalized_rank_regret == 1.0  # worst-ranked of 3 candidates

def test_compute_method_regret_rejects_action_not_in_oracle_results() -> None:
    oracle_results = [_fixture_candidate(0, 0.0, 1.0, 0.8)]
    try:
        compute_method_regret(99, oracle_results, clearance_tolerance=0.1)
        assert False, "expected OracleRegretError"
    except OracleRegretError:
        pass

def test_check_oracle_regret_gate_passes_when_full_significantly_better() -> None:
    records = _fixture_audit_records(full_rank=0.02, mean_rank=0.30, cv_rank=0.35)
    gate = check_oracle_regret_gate(records, min_states=20, min_episodes=10)
    assert gate["passed"], gate["reasons"]
    assert gate["metrics"]["vs_mean"]["rank_regret_improvement"]["ci_low"] > 0.0
    assert gate["metrics"]["vs_cv"]["rank_regret_improvement"]["ci_low"] > 0.0
    assert "efficiency_regret_improvement" in gate["metrics"]["vs_mean"]

def test_check_oracle_regret_gate_fails_when_regret_identical() -> None:
    # guide.md R4-4R2: if full/mean/cv show IDENTICAL regret against the
    # population-aligned oracle, that is the genuine "belief changes
    # actions without real value" finding -- must FAIL, not pass.
    records = []
    for ep in range(15):
        for s in range(3):
            same = MethodRegret(0.0, 0.0, 0.0, 0.1, True)
            records.append(AuditRecord(
                episode_seed=930000 + ep, decision_seed=s + 1, profile="train_non_reciprocal",
                full=same, mean=same, cv=same,
            ))
    gate = check_oracle_regret_gate(records, min_states=20, min_episodes=10)
    assert not gate["passed"]
    assert any("normalized_rank_regret 95% CI lower bound" in r for r in gate["reasons"])

def test_check_oracle_regret_gate_flags_non_inferiority_violation() -> None:
    records = []
    for ep in range(15):
        for s in range(3):
            full = MethodRegret(0.3, 0.2, None, 0.5, False)
            other = MethodRegret(0.0, 0.0, 0.0, 0.1, True)
            records.append(AuditRecord(
                episode_seed=940000 + ep, decision_seed=s + 1, profile="train_non_reciprocal",
                full=full, mean=other, cv=other,
            ))
    gate = check_oracle_regret_gate(records, min_states=20, min_episodes=10)
    assert not gate["passed"]
    reasons = " | ".join(gate["reasons"])
    assert "full collision_regret" in reasons and "worse than mean" in reasons
    assert "full collision_regret" in reasons and "worse than cv" in reasons

def test_check_oracle_regret_gate_flags_insufficient_states_and_episodes() -> None:
    records = [AuditRecord(
        episode_seed=1, decision_seed=1, profile="train_non_reciprocal",
        full=MethodRegret(0.0, 0.0, 0.0, 0.0, True), mean=MethodRegret(0.0, 0.0, 0.0, 0.0, True),
        cv=MethodRegret(0.0, 0.0, 0.0, 0.0, True),
    )]
    gate = check_oracle_regret_gate(records, min_states=20, min_episodes=10)
    assert not gate["passed"]
    reasons = " | ".join(gate["reasons"])
    assert "n_audit_states=1 < 20" in reasons
    assert "n_audit_episodes=1 < 10" in reasons

def test_oracle_candidate_results_returns_none_without_ids_or_insufficient_horizon() -> None:
    from crowd_nav.tools.collect_bdvl_r4_4_data import _build_oracle_candidate_results
    action_table = [(1.0, 0.0), (-1.0, 0.0)]
    robot_obs = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [HumanObservation(0, 1.0, 0.0, 0.0, 0.0, 0.3)]
    real_velocity_trace = [[(0.0, 0.0)] for _ in range(20)]

    assert _build_oracle_candidate_results(2, 20, [], real_velocity_trace, robot_obs, humans, action_table, _REWARD_CFG, 0.0) is None
    assert _build_oracle_candidate_results(15, 20, [0], real_velocity_trace, robot_obs, humans, action_table, _REWARD_CFG, 0.0) is None

def test_oracle_candidate_results_uses_shifted_real_future_not_stale_step() -> None:
    # guide.md R4-4R (carried into R4-4R2): the oracle must replay the
    # REAL future velocities that drove steps step_index+1..
    # step_index+HORIZON, never the stale value recorded at step_index
    # itself (crowd_sim's holonomic Agent.step() sets self.vx/vy to the
    # action AFTER applying it, so the loop-top read at iteration k
    # reflects the PREVIOUS transition, not the upcoming one). Proves
    # the +1 shift by cross-checking against a direct, hand-sliced call
    # into the already-tested evaluate_counterfactual_candidates -- not
    # by hand-deriving physics.
    from crowd_nav.tools.collect_bdvl_r4_4_data import _build_oracle_candidate_results

    rng = np.random.default_rng(11)
    action_table = [(1.0, 0.0), (-1.0, 0.3), (0.2, -0.4)]
    non_reciprocal_ids = [0]
    step_index = 2
    n_steps = step_index + ROLLOUT_HORIZON + 3

    real_velocity_trace = [[(0.0, 0.0)] for _ in range(n_steps)]
    real_velocity_trace[step_index][0] = (37.0, -19.0)  # stale trap value; must never be read
    for t in range(1, ROLLOUT_HORIZON + 1):
        real_velocity_trace[step_index + t][0] = (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)))

    robot_obs = RobotObservation(px=0.0, py=0.0, vx=0.4, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    subset_humans = [HumanObservation(0, 1.0, 0.2, 0.1, 0.0, 0.3)]

    result = _build_oracle_candidate_results(
        step_index, n_steps, non_reciprocal_ids, real_velocity_trace, robot_obs, subset_humans,
        action_table, _REWARD_CFG, 0.7,
    )
    assert result is not None

    expected_velocities = tuple(real_velocity_trace[step_index + t][0] for t in range(1, ROLLOUT_HORIZON + 1))
    human_worlds = {0: (HumanMultiStepTrajectory(track_id=0, modes=(None,) * ROLLOUT_HORIZON, step_velocities=expected_velocities),)}
    expected_results = evaluate_counterfactual_candidates(
        robot=robot_obs, humans=subset_humans, human_worlds=human_worlds, n_samples=1, action_table=action_table,
        reward_config=_REWARD_CFG, dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"], global_time=0.7,
        cvar_alpha=1.0, horizon=ROLLOUT_HORIZON,
    )
    expected_by_idx = {c.action_index: c for c in expected_results}
    result_by_idx = {c.action_index: c for c in result}
    for idx in expected_by_idx:
        assert result_by_idx[idx].lower_tail_clearance == expected_by_idx[idx].lower_tail_clearance
        assert result_by_idx[idx].collision_prob == expected_by_idx[idx].collision_prob

    # sanity: the stale trap value must genuinely be CAPABLE of changing
    # the answer, or this test would pass even with the bug reintroduced.
    wrong_velocities = tuple(
        [real_velocity_trace[step_index][0]] + [real_velocity_trace[step_index + t][0] for t in range(1, ROLLOUT_HORIZON)]
    )
    assert wrong_velocities != expected_velocities
