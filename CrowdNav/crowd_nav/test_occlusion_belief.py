"""Order 17 item 16: the acceptance suite for the occlusion work.

Five modes, one of which must change nothing at all. The tests are written so
that the failure a reviewer would most fear -- ground truth leaking into the
student's observation -- is checked by construction rather than by reading the
code, and so that the untouched path is proven bit-identical against a trace
captured before any edit.

    --unit           geometry, probability ranges, filter behaviour, token layout
    --leakage        moving a fully occluded pedestrian must not move a token
    --legacy-parity  off mode reproduces the pre-edit SHA256 exactly
    --smoke          every arm x backbone runs and back-propagates
"""
import argparse
import hashlib
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from crowd_sim.envs.occlusion_belief import OcclusionBelief, MODES
from crowd_nav.contracts import (
    entities_to_tokens, select_belief_entities_for_tokens,
    token_shape_for_contract,
)


class _S:
    """Minimal stand-in for FullState / ObservableState."""
    def __init__(self, px, py, vx=0.0, vy=0.0, radius=0.3,
                 gx=0.0, gy=4.0, v_pref=1.0, theta=0.0):
        self.px, self.py, self.vx, self.vy = px, py, vx, vy
        self.radius, self.gx, self.gy = radius, gx, gy
        self.v_pref, self.theta = v_pref, theta


def _mk(mode, **kw):
    return OcclusionBelief(mode=mode, grid_resolution=0.25, grid_extent=5.0,
                           fov_radius=5.0, max_entities=5, **kw)


# ------------------------------------------------------------------ unit ---
def test_geometry_occludes():
    """A pedestrian directly behind another, along the same ray, must be
    reported as occluded; one out to the side must not be."""
    ob = _mk("sensor")
    robot = _S(0.0, 0.0)
    behind = [_S(0.0, 1.0), _S(0.0, 2.2)]
    ob.update(robot, behind)
    assert 1 in ob.occluded_ids, f"blocker did not occlude: {ob.occluded_ids}"
    assert 0 in ob.visible_ids
    ob2 = _mk("sensor")
    ob2.update(robot, [_S(0.0, 1.0), _S(2.5, 1.0)])
    assert ob2.occluded_ids == [], f"unexpected occlusion: {ob2.occluded_ids}"
    print("  geometry: blocker occludes collinear pedestrian, spares lateral one  OK")


def test_fov_limit():
    ob = _mk("sensor", )
    ob.fov_radius = 2.0
    ob.update(_S(0.0, 0.0), [_S(0.0, 4.0)])
    assert 0 in ob.occluded_ids, "pedestrian beyond the FOV radius stayed visible"
    print("  geometry: beyond field of view counts as occluded  OK")


def test_probability_range_and_growth():
    """The posterior must stay a probability, and an unconfirmed belief must
    decay rather than persist forever."""
    ob = _mk("bayes")
    robot = _S(0.0, 0.0)
    hs = [_S(0.0, 1.0), _S(0.0, 2.2)]
    for _ in range(6):
        ob.update(robot, hs)
    p = 1.0 / (1.0 + np.exp(-ob.logodds))
    assert p.min() >= 0.0 and p.max() <= 1.0, "posterior left [0,1]"
    hi = float(p.max())
    for _ in range(40):
        ob.update(robot, [_S(4.9, 4.9)])
    p2 = 1.0 / (1.0 + np.exp(-ob.logodds))
    assert float(p2.max()) <= hi + 1e-9, "belief grew without evidence"
    print(f"  filter: posterior in [0,1]; peak {hi:.3f} -> {float(p2.max()):.3f} "
          f"without confirmation  OK")


def test_free_space_update():
    """A cell the robot can see to be empty must lose probability mass."""
    ob = _mk("bayes")
    ob.update(_S(0.0, 0.0), [])
    p = 1.0 / (1.0 + np.exp(-ob.logodds))
    seen = ob.sensor_grid == 0.0
    assert seen.any(), "nothing was observed free"
    assert p[seen].max() < ob.p_prior + 1e-6, "observed free space kept its prior"
    print(f"  filter: observed free cells fall below the prior "
          f"({p[seen].max():.4f} < {ob.p_prior})  OK")


def test_world_frame_reprojection_and_boundaries():
    """Robot motion must not drag belief mass, and diffusion must not wrap."""
    ob = _mk("bayes", decay=1.0, diffuse=1.0)
    ob._mesh = ob._build_mesh(np.array([0.0, 0.0]))
    p = np.full(ob._mesh[0].shape, ob.p_prior, dtype=np.float64)
    target = np.unravel_index(
        np.argmin((ob._mesh[0] - 1.0) ** 2 + ob._mesh[1] ** 2), p.shape)
    p[target] = 0.9
    ob.logodds = np.log(p / (1.0 - p))
    old_mesh = ob._mesh
    ob._mesh = ob._build_mesh(np.array([0.25, 0.0]))
    ob._reproject_belief(old_mesh)
    moved_p = 1.0 / (1.0 + np.exp(-ob.logodds))
    same_world = np.unravel_index(
        np.argmin((ob._mesh[0] - 1.0) ** 2 + ob._mesh[1] ** 2), moved_p.shape)
    assert moved_p[same_world] > 0.85, "robot motion dragged the world-fixed belief"

    edge = np.full_like(moved_p, ob.p_prior)
    edge[len(edge) // 2, 0] = 0.9
    ob.logodds = np.log(edge / (1.0 - edge))
    ob._predict()
    predicted = 1.0 / (1.0 + np.exp(-ob.logodds))
    assert predicted[:, -1].max() < 0.1, "diffusion wrapped across the grid boundary"
    print("  filter: world-frame reprojection and non-wrapping diffusion  OK")


def test_fair_oracle_marks_occluded_entities_hidden():
    ob = _mk("oracle_belief")
    ob.update(_S(0.0, 0.0), [_S(0.0, 1.0), _S(0.0, 2.2)])
    entities = ob.policy_entities()
    assert len(entities) == 2
    assert sum(e["visible"] > 0.5 for e in entities) == 1
    assert sum(e["hidden"] > 0.5 for e in entities) == 1
    print("  oracle: exact hidden state uses the same hidden-only consumer path  OK")


def test_never_seen_episode_metrics():
    """Never-seen means hidden now and never visible earlier this episode."""
    robot = _S(0.0, 0.0, radius=0.3)
    ob = _mk("sensor")
    ob.update(robot, [_S(0.0, 1.0), _S(0.0, 2.2)])
    first = ob.stats(near_radius=2.0)
    assert first["n_unseen_occluded"] == 1, first
    assert first["n_near_unseen_occluded"] == 1, first
    assert np.isfinite(first["min_unseen_occluded_clearance_m"]), first

    # Make pedestrian 1 visible once, then hide it behind pedestrian 0 again.
    ob.update(robot, [_S(0.0, 1.0), _S(2.0, 1.0)])
    ob.update(robot, [_S(0.0, 1.0), _S(0.0, 2.2)])
    later = ob.stats(near_radius=2.0)
    assert later["n_occluded"] == 1, later
    assert later["n_unseen_occluded"] == 0, later
    assert np.isnan(later["min_unseen_occluded_clearance_m"]), later
    print("  metrics: never-seen is episode-causal and distance-stratified  OK")


def test_token_layout():
    """Columns 0-8 keep the paper-1 semantics; 9-12 carry the belief."""
    r = _S(0.0, 0.0, 0.5, 0.0)
    ents = [{"px": 1.0, "py": 0.0, "vx": -0.5, "vy": 0.0, "radius": 0.3,
             "p_exist": 1.0, "uncertainty": 0.0, "visible": 1.0, "hidden": 0.0},
            {"px": 3.0, "py": 1.0, "vx": 0.0, "vy": 0.0, "radius": 0.3,
             "p_exist": 0.4, "uncertainty": 0.7, "visible": 0.0, "hidden": 1.0}]
    t = entities_to_tokens(r, ents)
    assert t.shape == (8, 13), t.shape
    assert t[0, 12] == 1.0, "robot flag missing"
    assert abs(t[3, 0] - 1.0) < 1e-6 and abs(t[3, 2] - 1.0) < 1e-3, "relative geometry wrong"
    assert t[3, 11] == 1.0 and t[3, 12] == 0.0, "visible entity mislabelled"
    assert abs(t[4, 9] - 0.4) < 1e-6 and abs(t[4, 10] - 0.7) < 1e-6, "belief columns lost"
    assert t[4, 11] == 0.0 and t[4, 12] == 1.0, "hidden entity mislabelled"
    print("  tokens: [8,13], columns 0-8 legacy, 9-12 carry existence/uncertainty/flags  OK")


def test_belief_v3_wide_contract():
    """The holdout-approved contract must survive selection, tokenisation,
    replay storage and a differentiable set encoding without another top-5
    truncation."""
    import torch
    from crowd_nav.policy.mamba_rl import EnhancedSpatialEncoder
    from crowd_nav.utils.ppo_buffer import ReplayBufferIQL

    robot = _S(0.0, 0.0, 0.2, 0.0)
    visible = [
        {"px": 0.6 + i * 0.2, "py": (-1) ** i * 0.2,
         "vx": -0.1, "vy": 0.0, "radius": 0.3,
         "p_exist": 1.0, "uncertainty": 0.0,
         "visible": 1.0, "hidden": 0.0, "id": i}
        for i in range(8)
    ]
    hidden = [
        {"px": 1.0 + i * 0.1, "py": 1.0, "vx": 0.0, "vy": 0.0,
         "radius": 0.3, "p_exist": 0.20 + i * 0.05,
         "uncertainty": 0.1, "visible": 0.0, "hidden": 1.0,
         "id": 100 + i}
        for i in range(12)
    ]
    selected = select_belief_entities_for_tokens(
        robot, visible + hidden, visible_slots=5, hidden_slots=10)
    assert len(selected) == 15
    assert sum(e["visible"] > 0.5 for e in selected) == 5
    assert sum(e["hidden"] > 0.5 for e in selected) == 10
    selected_hidden_p = [e["p_exist"] for e in selected if e["hidden"] > 0.5]
    assert selected_hidden_p == sorted(selected_hidden_p, reverse=True)

    full = entities_to_tokens(
        robot, visible + hidden, token_contract="belief_v3",
        visible_slots=5, hidden_slots=10, belief_features="full")
    fixed = entities_to_tokens(
        robot, visible + hidden, token_contract="belief_v3",
        visible_slots=5, hidden_slots=10,
        belief_features="fixed_confidence")
    assert full.shape == fixed.shape == token_shape_for_contract("belief_v3", 5, 10)
    assert np.array_equal(full[:, :9], fixed[:, :9])
    assert np.array_equal(full[:, 11:13], fixed[:, 11:13])
    assert int((full[3:, 11] > 0.5).sum()) == 5
    assert int((full[3:, 12] > 0.5).sum()) == 10

    replay = ReplayBufferIQL(
        capacity=8, seq_len=2, obs_shape=(18, 13),
        occlusion_mode="bayes", belief_features="full")
    payload = dict(
        states=np.stack([full, full]),
        actions=np.zeros((2, 2), dtype=np.float32),
        action_indices=np.zeros(2, dtype=np.int64),
        rewards=np.zeros(2, dtype=np.float32),
        dones=np.array([False, True]),
    )
    replay.store_episode(payload)
    assert replay.states.shape[2:] == (18, 13)
    assert np.array_equal(replay.states[0, -1], full)

    # Exercise the production Explorer -> replay handoff as well. This catches
    # any future conversion back to the lossy 34-D representation.
    from crowd_nav.utils.explorer import Explorer
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
    robot_state = FullState(
        0.0, 0.0, 0.2, 0.0, 0.3, 0.0, 4.0, 1.0, 0.0)
    js = JointState(robot_state, [
        ObservableState(e["px"], e["py"], e["vx"], e["vy"], e["radius"])
        for e in visible + hidden])
    js.policy_entities = visible + hidden
    js.provenance = "bayes"
    js.belief_features = "full"
    js.token_contract = "belief_v3"
    js.visible_slots = 5
    js.hidden_slots = 10
    explorer_replay = ReplayBufferIQL(
        capacity=8, seq_len=2, obs_shape=(18, 13),
        occlusion_mode="bayes", belief_features="full")
    explorer = Explorer(
        env=None, robot=None, device="cpu", memory=explorer_replay)
    explorer.update_memory(
        states=[js, js], actions=[ActionXY(0.1, 0.0), ActionXY(0.1, 0.0)],
        rewards=[0.0, 0.0], action_indices=[0, 0])
    assert explorer_replay.size == 2
    assert np.array_equal(explorer_replay.states[0, -1], full)

    encoder = EnhancedSpatialEncoder(d_model=64)
    x = torch.from_numpy(np.stack([full, fixed])).float().unsqueeze(1)
    y = encoder(x)
    assert y.shape == (2, 1, 64) and torch.isfinite(y).all()
    y.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in encoder.parameters())
    print("  belief-v3: 15 entities survive selection, replay and backward  OK")


def test_stateless_mlp_and_teacher_observability():
    import configparser
    import torch
    from crowd_nav.policy.mamba_rl import MLPTemporalEncoder
    from crowd_nav.utils.explorer import Explorer

    encoder = MLPTemporalEncoder(d_model=16, dropout=0.0).eval()
    x1 = torch.randn(2, 4, 16)
    x2 = x1.clone()
    x2[:, :-1] = torch.randn_like(x2[:, :-1]) * 100.0
    with torch.no_grad():
        y1, y2 = encoder(x1), encoder(x2)
    assert torch.equal(y1[:, -1], y2[:, -1]), (
        "stateless MLP leaked earlier frames into the current decision")

    class _Env:
        def __init__(self):
            self.config = configparser.RawConfigParser()
            self.config.add_section('imitation_learning')
            self.oracle = object()

        def get_oracle_state(self):
            return self.oracle

    env = _Env()
    policy_state = object()
    explorer = Explorer(env=env, robot=None, device='cpu')
    env.config.set('imitation_learning', 'teacher_observation', 'policy')
    assert explorer._teacher_state_for_il(policy_state) is policy_state
    env.config.set('imitation_learning', 'teacher_observation', 'oracle')
    assert explorer._teacher_state_for_il(policy_state) is env.oracle
    env.config.set('imitation_learning', 'teacher_observation', 'invalid')
    try:
        explorer._teacher_state_for_il(policy_state)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid teacher observability did not fail closed")
    print("  paper2: MLP is stateless and IL teacher observability is explicit  OK")


def test_paired_belief_contract_reaches_replay():
    """The paired arms share geometry and status flags. Only posterior
    existence/uncertainty differ, and those values must survive Explorer and
    ReplayBuffer without a 34-D round trip."""
    from crowd_nav.utils.explorer import Explorer
    from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState

    robot = FullState(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 4.0, 1.0, 0.0)
    entities = [
        {"px": 1.0, "py": 0.0, "vx": 0.0, "vy": 0.0, "radius": 0.3,
         "p_exist": 1.0, "uncertainty": 0.0, "visible": 1.0, "hidden": 0.0},
        {"px": 2.0, "py": 0.0, "vx": -0.2, "vy": 0.0, "radius": 0.3,
         "p_exist": 0.4, "uncertainty": 0.7, "visible": 0.0, "hidden": 1.0},
    ]

    def make_state(feature_mode):
        js = JointState(robot, [
            ObservableState(e["px"], e["py"], e["vx"], e["vy"], e["radius"])
            for e in entities
        ])
        js.policy_entities = entities
        js.provenance = "bayes"
        js.belief_features = feature_mode
        return js

    full_state = make_state("full")
    fixed_state = make_state("fixed_confidence")
    explorer = Explorer(env=None, robot=None, device="cpu")
    full = explorer._state_to_array(full_state)
    fixed = explorer._state_to_array(fixed_state)
    assert full.shape == fixed.shape == (8, 13)
    assert np.array_equal(full[:, :9], fixed[:, :9]), "paired geometry drifted"
    assert np.array_equal(full[:, 11:13], fixed[:, 11:13]), "status flags drifted"
    hidden_row = 3 + int(np.flatnonzero(full[3:, 12] > 0.5)[0])
    assert np.allclose(full[hidden_row, 9:11], [0.4, 0.7])
    assert np.allclose(fixed[hidden_row, 9:11], [1.0, 0.0])

    payload = dict(
        actions=np.zeros((2, 2), dtype=np.float32),
        action_indices=np.zeros(2, dtype=np.int64),
        rewards=np.zeros(2, dtype=np.float32),
        dones=np.array([False, True]),
    )
    full_replay = ReplayBufferIQL(
        capacity=8, seq_len=2, occlusion_mode="bayes", belief_features="full")
    full_replay.store_episode(dict(payload, states=np.stack([full, full])))
    fixed_replay = ReplayBufferIQL(
        capacity=8, seq_len=2, occlusion_mode="bayes",
        belief_features="fixed_confidence")
    fixed_replay.store_episode(dict(payload, states=np.stack([fixed, fixed])))
    assert full_replay.belief_contract["nontrivial_confidence_rows"] > 0
    assert np.array_equal(full_replay.states[0, -1], full)
    assert np.array_equal(fixed_replay.states[0, -1], fixed)
    print("  replay-contract: paired columns survive Explorer and replay exactly  OK")


def test_arms_differ():
    """The four arms must produce genuinely different observations, otherwise
    the comparison is vacuous."""
    robot = _S(0.0, 0.0)
    hs = [_S(0.0, 1.0), _S(0.0, 2.2)]
    seen = {}
    for m in ("gt", "sensor", "deterministic", "bayes"):
        ob = _mk(m)
        for _ in range(5):
            ob.update(robot, hs)
        seen[m] = ob.policy_entities()
    assert len(seen["gt"]) == 2, "oracle lost a pedestrian"
    assert len(seen["sensor"]) == 1, f"sensor arm saw {len(seen['sensor'])}, expected 1"
    assert len(seen["deterministic"]) >= 1
    print(f"  arms: gt={len(seen['gt'])} sensor={len(seen['sensor'])} "
          f"deterministic={len(seen['deterministic'])} bayes={len(seen['bayes'])}  OK")


def test_bayes_input_ablation_matches_sensor():
    """The evaluation ablation must remove the complete inferred entity, not
    merely its four belief columns, and must then equal the sensor arm exactly."""
    robot = _S(0.0, 0.0)
    visible_layout = [_S(1.0, 1.0), _S(0.0, 2.2)]
    hidden_layout = [_S(0.0, 1.0), _S(0.0, 2.2)]
    bayes, sensor = _mk("bayes"), _mk("sensor")
    for tracker in (bayes, sensor):
        tracker.update(robot, visible_layout)
        tracker.update(robot, hidden_layout)

    full = bayes.policy_entities()
    ablated = bayes.policy_entities(include_hidden_belief=False)
    sensed = sensor.policy_entities()
    assert any(e["hidden"] > 0.5 for e in full), \
        "test setup produced no hidden Bayes entity"
    assert all(e["visible"] > 0.5 and e["hidden"] < 0.5 for e in ablated)
    assert ablated == sensed, "Bayes input ablation differs from sensor entities"
    assert np.array_equal(entities_to_tokens(robot, ablated),
                          entities_to_tokens(robot, sensed)), \
        "Bayes input ablation differs from sensor tokens"
    print("  ablation: Bayes visible-only entities and tokens equal sensor exactly  OK")


def test_off_is_inert():
    ob = _mk("off")
    assert not ob.enabled
    ob.update(_S(0.0, 0.0), [_S(0.0, 1.0)])
    assert ob.sensor_grid is None and ob.policy_entities() is None
    print("  off: module is inert and returns no payload  OK")


def test_metrics_and_checkpoint_contract():
    import configparser
    import torch
    from crowd_nav.policy.mamba_rl import (
        assert_checkpoint_compatible,
        occlusion_checkpoint_meta,
        warm_start_from_legacy,
    )

    ob = _mk("sensor")
    ob.update(_S(0.0, 0.0), [_S(0.0, 1.0), _S(0.0, 2.2)])
    stats = ob.stats(near_radius=2.0)
    assert stats['n_occluded'] == 1 and stats['n_near_occluded'] == 1, stats

    cfg = configparser.RawConfigParser()
    cfg.add_section('occlusion')
    cfg.set('occlusion', 'grid_resolution', '0.25')
    cfg.set('occlusion', 'grid_extent', '5.0')
    cfg.set('occlusion', 'fov_radius', '5.0')
    meta = occlusion_checkpoint_meta(cfg, 'bayes', 'gru')
    assert_checkpoint_compatible(meta, 'bayes', 'gru', expected_meta=meta)
    bad = dict(meta, grid_extent=6.0)
    try:
        assert_checkpoint_compatible(bad, 'bayes', 'gru', expected_meta=meta)
    except ValueError:
        pass
    else:
        raise AssertionError("grid mismatch was accepted")
    other_arm = dict(meta, belief_features='fixed_confidence')
    try:
        assert_checkpoint_compatible(
            other_arm, 'bayes', 'gru', expected_meta=meta)
    except ValueError:
        pass
    else:
        raise AssertionError("belief-feature arm mismatch was accepted")

    class _Legacy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.human_encoder = torch.nn.Sequential(torch.nn.Linear(21, 4))

    src, dst = _Legacy(), _Legacy()
    with torch.no_grad():
        src.human_encoder[0].weight.fill_(1.0)
    warm_start_from_legacy(dst, src.state_dict(), verbose=False)
    weight = dst.human_encoder[0].weight.detach()
    assert torch.count_nonzero(weight[:, [9, 10, 11, 12]]) == 0
    assert torch.all(weight[:, 8] == 1.0) and torch.all(weight[:, 13] == 1.0)
    print("  contracts: near-hidden metrics, strict metadata, legacy zero-init  OK")


def test_gt_matches_legacy_tokens():
    """GT may add values only in the four reserved human columns. The robot
    rows and human columns 0-8 must exactly match the pretrained 34-D path,
    including with 20 humans and for a propagated successor state."""
    from crowd_nav.contracts import (
        _batch_joint34_to_tokens_vectorized,
        select_entities_for_tokens,
    )

    robot = _S(0.0, -4.0, 0.0, 0.0, 0.3, 0.0, 4.0, 1.0, np.pi / 2)
    entities = []
    for idx in range(20):
        angle = 2.0 * np.pi * idx / 20.0
        distance = 1.0 + 0.15 * idx
        entities.append({
            "id": idx,
            "px": distance * np.cos(angle),
            "py": distance * np.sin(angle),
            "vx": -0.2 * np.cos(angle),
            "vy": -0.2 * np.sin(angle),
            "radius": 0.3,
            "p_exist": 1.0,
            "uncertainty": 0.0,
            "visible": 1.0,
            "hidden": 0.0,
        })

    def legacy_tokens(robot_state, selected):
        robot_row = np.array([
            robot_state.px, robot_state.py, robot_state.vx, robot_state.vy,
            robot_state.radius, robot_state.gx, robot_state.gy,
            robot_state.v_pref, robot_state.theta,
        ], dtype=np.float32)
        humans = [value for entity in selected for value in (
            entity['px'], entity['py'], entity['vx'], entity['vy'], entity['radius'])]
        humans.extend([0.0] * (25 - len(humans)))
        state_34 = np.concatenate([robot_row, humans[:25]])
        return _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]

    selected = select_entities_for_tokens(robot, entities)
    old = legacy_tokens(robot, selected)
    new = entities_to_tokens(robot, entities)
    assert np.array_equal(old[:3], new[:3]), "GT changed legacy robot tokens"
    assert np.array_equal(old[3:, :9], new[3:, :9]), "GT changed legacy human tokens"

    dt = 0.25
    next_robot = _S(
        robot.px + 0.4 * dt, robot.py + 0.2 * dt, 0.4, 0.2,
        robot.radius, robot.gx, robot.gy, robot.v_pref, robot.theta)
    next_entities = [dict(
        entity,
        px=entity['px'] + entity['vx'] * dt,
        py=entity['py'] + entity['vy'] * dt,
    ) for entity in selected]
    old_next = legacy_tokens(next_robot, next_entities)
    new_next = entities_to_tokens(next_robot, next_entities, preselected=True)
    assert np.array_equal(old_next[:3], new_next[:3]), "GT successor changed robot tokens"
    assert np.array_equal(old_next[3:, :9], new_next[3:, :9]), \
        "GT successor changed human tokens"
    print("  gt-parity: 20-human current and successor legacy columns exact  OK")


def test_legacy_warm_start_is_inert():
    """Legacy weights must ignore the four newly activated token columns."""
    import torch
    from crowd_nav.policy.mamba_rl import warm_start_from_legacy

    class _Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.human_encoder = torch.nn.Sequential(torch.nn.Linear(21, 8))

    src, dst = _Policy(), _Policy()
    warm_start_from_legacy(dst, src.state_dict(), verbose=False)
    legacy = torch.randn(32, 21)
    active = legacy.clone()
    active[:, 9:13] = torch.randn(32, 4)
    legacy[:, 9:13] = 0.0
    with torch.no_grad():
        baseline = dst.human_encoder(legacy)
        with_occlusion = dst.human_encoder(active)
    assert torch.equal(baseline, with_occlusion), \
        "legacy warm-start still consumes occlusion columns"
    print("  warm-start: activated columns are exactly inert for legacy weights  OK")


# --------------------------------------------------------------- leakage ---
def test_no_truth_leak():
    """The decisive one. Move a FULLY OCCLUDED pedestrian while leaving
    everything the robot can see untouched: sensor, deterministic and bayes
    must produce identical tokens, and only the oracle may change.

    If this fails, every downstream number is meaningless, because the student
    would be reading ground truth through a side channel.
    """
    robot = _S(0.0, 0.0)
    base = [_S(0.0, 1.0), _S(0.0, 2.2)]
    moved = [_S(0.0, 1.0), _S(0.0, 2.9)]      # only the hidden one moves
    ok = True
    for m in ("sensor", "deterministic", "bayes"):
        a, b = _mk(m), _mk(m)
        for _ in range(4):
            a.update(robot, base)
            b.update(robot, base)
        a.update(robot, base)
        b.update(robot, moved)
        assert 1 in b.occluded_ids, "the moved pedestrian was not actually occluded"
        ta = entities_to_tokens(robot, a.policy_entities())
        tb = entities_to_tokens(robot, b.policy_entities())
        same = np.array_equal(ta, tb)
        print(f"  leakage[{m}]: tokens identical = {same}")
        ok &= same
    g1, g2 = _mk("gt"), _mk("gt")
    g1.update(robot, base)
    g2.update(robot, moved)
    changed = not np.array_equal(entities_to_tokens(robot, g1.policy_entities()),
                                 entities_to_tokens(robot, g2.policy_entities()))
    print(f"  leakage[gt]: tokens changed = {changed} (must be True)")
    assert ok, "TRUTH LEAK: a hidden pedestrian's motion reached the student"
    assert changed, "oracle arm failed to react to ground truth"
    print("  leakage: hidden motion invisible to student arms, visible to oracle  OK")


# ---------------------------------------------------------------- parity ---
def test_legacy_parity(trace_path, checkpoint):
    """off mode must reproduce the pre-edit trace bit for bit."""
    import runpy
    ref = np.load(trace_path)
    ref_sha = str(ref["sha256"][0])
    rec = {"states": [], "actions": []}
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    orig = MambaRLPolicy.predict_sarl_style

    def hook(self, state):
        s34 = self._build_joint_state_34(state.self_state, state.human_states)
        a = orig(self, state)
        rec["states"].append(np.asarray(s34, dtype=np.float64).copy())
        rec["actions"].append(np.array([a.vx, a.vy], dtype=np.float64))
        return a

    MambaRLPolicy.predict_sarl_style = hook
    # The baseline was captured from inside crowd_nav/ with a model_dir
    # relative to it. Bit-exact parity requires reproducing that invocation
    # exactly, working directory included: test.py resolves its config and
    # checkpoint paths relative to cwd.
    argv, cwd = sys.argv, os.getcwd()
    ckpt = os.path.normpath(checkpoint)
    if ckpt.startswith("crowd_nav" + os.sep):
        ckpt = ckpt[len("crowd_nav" + os.sep):]
    sys.argv = ["test.py", "--policy", "mamba_rl",
                "--model_dir", os.path.dirname(ckpt),
                "--weights", os.path.basename(ckpt), "--gpu",
                "--episodes", "5", "--test_case", "0", "--seed", "42",
                "--no_progress"]
    try:
        os.chdir(os.path.join(_ROOT, "crowd_nav"))
        runpy.run_path("test.py", run_name="__main__")
    finally:
        os.chdir(cwd)
        sys.argv = argv
        MambaRLPolicy.predict_sarl_style = orig
    S = np.array(rec["states"]); A = np.array(rec["actions"])
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(S).tobytes())
    h.update(np.ascontiguousarray(A).tobytes())
    sha = h.hexdigest()
    print(f"  parity: {len(rec['states'])} decisions")
    print(f"  parity: reference {ref_sha}")
    print(f"  parity: current   {sha}")
    assert S.shape == ref["states"].shape, f"shape drift {S.shape} vs {ref['states'].shape}"
    assert sha == ref_sha, "LEGACY PARITY BROKEN: off mode changed behaviour"
    print("  parity: off mode is bit-identical to the pre-edit baseline  OK")


def test_gt_off_action_parity(checkpoint):
    """Exercise the production evaluator and require GT to reproduce off.

    A legacy checkpoint is deliberately used for both arms. GT exposes the
    same physical humans as off, while the diagnostic warm-start makes the
    four new human columns inert. Any action difference therefore identifies
    observation/reward/lookahead drift rather than an occlusion effect.
    """
    import contextlib
    import runpy

    from crowd_nav.policy.mamba_rl import MambaRLPolicy

    captured = {"off": [], "gt": []}
    current_mode = [None]
    original = MambaRLPolicy.predict_sarl_style

    def hook(self, state):
        action = original(self, state)
        captured[current_mode[0]].append((float(action.vx), float(action.vy)))
        return action

    ckpt = os.path.normpath(checkpoint)
    if ckpt.startswith("crowd_nav" + os.sep):
        ckpt = ckpt[len("crowd_nav" + os.sep):]
    argv, cwd = sys.argv, os.getcwd()
    MambaRLPolicy.predict_sarl_style = hook
    try:
        os.chdir(os.path.join(_ROOT, "crowd_nav"))
        for mode in ("off", "gt"):
            current_mode[0] = mode
            sys.argv = [
                "test.py", "--policy", "mamba_rl",
                "--model_dir", os.path.dirname(ckpt),
                "--weights", os.path.basename(ckpt), "--gpu",
                "--episodes", "1", "--test-cases", "0,1,2,3,4,5",
                "--seed", "42", "--no_progress", "--occlusion-mode", mode,
            ]
            if mode == "gt":
                sys.argv.append("--legacy-diagnostic")
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                runpy.run_path("test.py", run_name="__main__")
    finally:
        os.chdir(cwd)
        sys.argv = argv
        MambaRLPolicy.predict_sarl_style = original

    off = np.asarray(captured["off"], dtype=np.float64)
    gt = np.asarray(captured["gt"], dtype=np.float64)
    assert off.shape == gt.shape, f"GT/off decision-count drift: {gt.shape} vs {off.shape}"
    if not np.array_equal(off, gt):
        differing = np.flatnonzero(np.any(off != gt, axis=1))
        first = int(differing[0])
        raise AssertionError(
            f"GT/off action drift at decision {first}: off={off[first]} gt={gt[first]} "
            f"({len(differing)}/{len(off)} decisions differ)"
        )
    print(f"  gt-off-parity: {len(off)} production decisions are bit-identical  OK")


# ----------------------------------------------------------------- smoke ---
def test_smoke(modes, backbones, episodes, use_gpu, belief_features="full"):
    """Each arm x backbone must run the environment and take one optimiser step
    without NaN, without a silent shape fallback, and without touching truth."""
    import configparser
    import torch
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_nav.policy.policy_factory import policy_factory

    cfgdir = os.path.join(_ROOT, "crowd_nav", "configs")
    results = []
    for mode in modes:
        for backbone in backbones:
            cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'),
                                               strict=False)
            for f in ("env.config", "policy.config", "train.config"):
                fp = os.path.join(cfgdir, f)
                if os.path.exists(fp):
                    cfg.read(fp, encoding="utf-8")
            if not cfg.has_section('occlusion'):
                cfg.add_section('occlusion')
            cfg.set('occlusion', 'mode', mode)
            cfg.set('occlusion', 'enabled', 'false' if mode == 'off' else 'true')
            cfg.set('occlusion', 'belief_features', belief_features)
            if cfg.has_section('temporal'):
                cfg.set('temporal', 'temporal_backbone', backbone)
            else:
                cfg.add_section('temporal')
                cfg.set('temporal', 'temporal_backbone', backbone)
            # Production MambaRLPolicy reads the controlled backbone from the
            # [mamba] section; setting only [temporal] silently tested Mamba.
            if not cfg.has_section('mamba'):
                cfg.add_section('mamba')
            cfg.set('mamba', 'temporal_backbone', backbone)

            env = CrowdSim()
            env.configure(cfg)
            env.phase = 'test'
            robot = Robot(cfg, 'robot')
            policy = policy_factory['mamba_rl'](cfg)
            dev = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
            # MambaRLPolicy has no set_device; test.py moves it with .to() and
            # sets .device separately (test.py:1183). A hasattr('set_device')
            # check silently leaves the model on the CPU, and mamba_ssm then
            # fails deep inside the selective scan rather than at the call site.
            if hasattr(policy, 'to'):
                policy.to(dev)
            policy.device = torch.device(dev)
            if hasattr(policy, 'set_phase'):
                policy.set_phase('test')
            # occlusion arms are value-lookahead only; see the guard in predict()
            policy.use_sarl_predict = True
            robot.set_policy(policy)
            robot.env = env
            env.set_robot(robot)

            policy.set_phase('train')
            policy.set_epsilon(1.0)
            env.reset()
            random_action, random_idx = policy.act(env.get_policy_state())
            assert isinstance(random_idx, int)
            assert policy.action_space[random_idx] == random_action
            policy.set_epsilon(0.0)
            policy.set_phase('test')

            nan = False
            steps = 0
            replay_tokens = []
            replay_actions = []
            replay_indices = []
            for _ in range(episodes):
                env.reset()
                for _ in range(30):
                    st = env.get_policy_state()
                    tok = policy._state_to_policy_tokens(st)
                    expected_shape = ((8, 13) if mode == 'off' else
                                      token_shape_for_contract(
                                          cfg.get('occlusion', 'token_contract',
                                                  fallback='legacy_top5'),
                                          cfg.getint('occlusion', 'visible_slots', fallback=5),
                                          cfg.getint('occlusion', 'hidden_slots', fallback=10)))
                    assert tok.shape == expected_shape, f"shape fallback: {tok.shape}"
                    if not np.all(np.isfinite(tok)):
                        nan = True
                    a, action_idx = policy.act(st)
                    assert isinstance(action_idx, int)
                    assert 0 <= action_idx < len(policy.action_space)
                    replay_tokens.append(tok)
                    replay_actions.append([a.vx, a.vy])
                    replay_indices.append(action_idx)
                    out = env.step(a)
                    steps += 1
                    if out[2] or out[3]:
                        break
            # one backward pass through the value network
            x = torch.from_numpy(
                np.zeros((1, policy.seq_len, *expected_shape), np.float32)).to(dev)
            v = policy.forward_value(x)
            loss = v.sum()
            loss.backward()
            gnan = any((p.grad is not None and not torch.all(torch.isfinite(p.grad)))
                       for p in policy.parameters())
            occ = env.occlusion.stats() if env.occlusion_enabled() else {}
            from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
            replay = ReplayBufferIQL(capacity=1000, seq_len=policy.seq_len,
                                     obs_shape=expected_shape, occlusion_mode=mode,
                                     belief_features=belief_features)
            replay.push_episode({
                'states': np.asarray(replay_tokens, dtype=np.float32),
                'actions': np.asarray(replay_actions, dtype=np.float32),
                'action_indices': np.asarray(replay_indices, dtype=np.int64),
                'rewards': np.zeros(len(replay_tokens), dtype=np.float32),
                'dones': np.zeros(len(replay_tokens), dtype=bool),
            })
            assert replay.size == len(replay_tokens), (
                f"online replay lost transitions: {replay.size}/{len(replay_tokens)}")
            results.append((mode, backbone, steps, nan or gnan, occ))
            print(f"  smoke[{mode}/{backbone}/{belief_features}]: {steps} steps, "
                  f"nan={nan or gnan}, belief={replay.belief_contract}, {occ}")
    bad = [r for r in results if r[3]]
    assert not bad, f"NaN produced by {[(r[0], r[1]) for r in bad]}"
    print("  smoke: every arm x backbone ran and back-propagated cleanly  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', action='store_true')
    ap.add_argument('--leakage', action='store_true')
    ap.add_argument('--legacy-parity', action='store_true')
    ap.add_argument('--gt-off-parity', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--trace', default='/root/mamba_legacy_trace.npz')
    ap.add_argument('--checkpoint', default='crowd_nav/runs/mamba_t24/rl_model_ep10000.pth')
    ap.add_argument('--modes', default='sensor,deterministic,bayes,gt,oracle_belief')
    ap.add_argument('--backbones', default='mamba,gru')
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--belief-features', choices=['full', 'fixed_confidence'],
                    default='full')
    a = ap.parse_args()
    if a.unit:
        print("[UNIT]")
        test_off_is_inert(); test_geometry_occludes(); test_fov_limit()
        test_probability_range_and_growth(); test_free_space_update()
        test_world_frame_reprojection_and_boundaries()
        test_fair_oracle_marks_occluded_entities_hidden()
        test_never_seen_episode_metrics()
        test_token_layout(); test_belief_v3_wide_contract()
        test_stateless_mlp_and_teacher_observability()
        test_paired_belief_contract_reaches_replay()
        test_arms_differ(); test_bayes_input_ablation_matches_sensor()
        test_metrics_and_checkpoint_contract()
        test_gt_matches_legacy_tokens(); test_legacy_warm_start_is_inert()
        print("[UNIT] all passed")
    if a.leakage:
        print("[LEAKAGE]")
        test_no_truth_leak()
        print("[LEAKAGE] passed")
    if a.legacy_parity:
        print("[LEGACY-PARITY]")
        test_legacy_parity(a.trace, a.checkpoint)
        print("[LEGACY-PARITY] passed")
    if a.gt_off_parity:
        print("[GT-OFF-PARITY]")
        test_gt_off_action_parity(a.checkpoint)
        print("[GT-OFF-PARITY] passed")
    if a.smoke:
        print("[SMOKE]")
        test_smoke(a.modes.split(','), a.backbones.split(','), a.episodes,
                   a.gpu, a.belief_features)
        print("[SMOKE] passed")


if __name__ == '__main__':
    main()
