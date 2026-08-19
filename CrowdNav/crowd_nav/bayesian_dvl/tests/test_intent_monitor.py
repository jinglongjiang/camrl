"""Training-observability tests for the V6 main chain."""

from crowd_nav.bayesian_dvl.tests._common import *  # noqa: F401,F403

from types import SimpleNamespace

from crowd_nav.bayesian_dvl.intent_monitor import (
    DevelopmentSummary, IntentMonitorError, TrainingMonitor, run_development_validation,
)


def _result(outcome: str, episode_seed: int = 1):
    return SimpleNamespace(
        outcome=outcome,
        episode_return={"success": 1.0, "collision": -1.0, "timeout": -0.5}[outcome],
        episode_steps=12,
        navigation_time=3.0,
        path_length=2.9,
        path_ratio=0.97,
        min_clearance=0.12,
        discomfort_frequency=0.25,
        scenario="standard",
        episode_seed=episode_seed,
        epsilon=0.2,
        loss=0.4,
        mc_loss=0.3,
        rank_loss=0.1,
        grad_norm_preclip=2.0,
        clipped=False,
        n_demo=2,
        n_online=6,
    )


def test_monitor_writes_episode_metrics_log_curves_and_rolling_rates() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        monitor = TrainingMonitor(run, 0, 0, False, (2, 3), tensorboard_enabled=False)
        monitor.record_online(1, 3, _result("success", 1))
        monitor.record_online(2, 3, _result("collision", 2))
        rolling = monitor.record_online(3, 3, _result("timeout", 3))
        assert rolling[3]["success_rate"] == 1 / 3
        assert rolling[3]["collision_rate"] == 1 / 3
        assert rolling[3]["timeout_rate"] == 1 / 3
        monitor.plot()
        monitor.close()
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [r["online_episode"] for r in rows] == [1, 2, 3]
        assert "MONITOR START" in (run / "train.log").read_text()
        png = (run / "curves.png").read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 10_000


def test_monitor_resume_truncates_rows_ahead_of_checkpoint_without_duplicates() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        first = TrainingMonitor(run, 0, 0, False, (2,), tensorboard_enabled=False)
        first.record_il(1, 2, _result("success"))
        first.record_il(2, 2, _result("success"))
        first.record_online(1, 2, _result("success", 1))
        first.record_online(2, 2, _result("collision", 2))
        first.record_validation(DevelopmentSummary(
            online_episode=2, n=2, success_rate=0.5, collision_rate=0.5, timeout_rate=0.0,
            mean_return=0.0, mean_navigation_time=3.0, mean_path_ratio=1.0,
            mean_min_clearance=0.1, mean_discomfort_frequency=0.25, by_scenario={},
        ))
        first.close()

        resumed = TrainingMonitor(run, 1, 1, True, (2,), tensorboard_enabled=False)
        assert [r.get("online_episode") for r in resumed.records if r["record_type"] == "online"] == [1]
        assert [r.get("il_pass") for r in resumed.records if r["record_type"] == "il"] == [1]
        assert not resumed.has_validation(2)
        resumed.record_online(2, 2, _result("timeout", 2))
        resumed.close()
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [r["online_episode"] for r in rows if r["record_type"] == "online"] == [1, 2]


def test_monitor_fails_when_checkpoint_is_ahead_of_durable_metrics() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        first = TrainingMonitor(run, 0, 0, False, (2,), tensorboard_enabled=False)
        first.record_online(1, 1, _result("success"))
        first.close()
        try:
            TrainingMonitor(run, 2, 0, True, (2,), tensorboard_enabled=False)
            assert False, "expected fail-closed cursor mismatch"
        except IntentMonitorError:
            pass


def test_monitor_recovers_only_a_truncated_final_jsonl_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        first = TrainingMonitor(run, 0, 0, False, (2,), tensorboard_enabled=False)
        first.record_online(1, 1, _result("success"))
        first.close()
        with (run / "metrics.jsonl").open("a") as fh:
            fh.write('{"record_type":"online","online_episode":2')
        recovered = TrainingMonitor(run, 1, 0, True, (2,), tensorboard_enabled=False)
        assert [r["online_episode"] for r in recovered.records if r["record_type"] == "online"] == [1]
        recovered.close()
        assert len((run / "metrics.jsonl").read_text().splitlines()) == 1


def test_cli_checkpoints_before_development_and_resume_fills_a_missing_record() -> None:
    import inspect
    from crowd_nav.bayesian_dvl import intent_train_cli

    source = inspect.getsource(intent_train_cli.cmd_train)
    marker = source.index("# Save BEFORE the read-only validation")
    checkpoint = source.index("_save_rolling()", marker)
    validation = source.index("_run_development_if_due(done)", checkpoint)
    assert checkpoint < validation
    assert "if resume:\n        _run_development_if_due(art.state.online_episodes_done)" in source


def test_online_training_result_keeps_navigation_outcome_and_episode_metrics() -> None:
    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(19)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = IntentReplay(demo_capacity=20, online_capacity=200)
    result = run_online_training_step(
        env_config_path, model, optimizer, action_table, "standard", 700001, 0.2,
        buffer, 8, np.random.default_rng(1), np.random.default_rng(2),
        torch.Generator().manual_seed(3),
    )
    assert result.outcome in ("success", "collision", "timeout")
    assert result.episode_steps > 0 and np.isfinite(result.episode_return)
    assert result.navigation_time > 0 and result.path_length >= 0
    assert np.isfinite(result.path_ratio) and np.isfinite(result.min_clearance)
    assert result.scenario == "standard" and result.episode_seed == 700001


def test_development_validation_uses_legal_disjoint_seeds_and_does_not_mutate_model() -> None:
    from crowd_nav.bayesian_dvl.junction_scenario import JUNCTION_CROWD_VALIDATION_SEEDS

    env_config_path = _env_config_path()
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table())
    torch.manual_seed(23)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows = run_development_validation(
        env_config_path, model, action_table, [97301], [JUNCTION_CROWD_VALIDATION_SEEDS[0]],
        belief_mode="full", n_samples=4, horizon=3, device="cpu",
    )
    assert [(r["scenario"], r["episode_seed"]) for r in rows] == [
        ("standard", 97301), ("junction_crowd", JUNCTION_CROWD_VALIDATION_SEEDS[0])]
    assert all(r["outcome"] in ("success", "collision", "timeout") for r in rows)
    assert model.training, "validation must restore the caller's train/eval mode"
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
