#!/usr/bin/env python3
"""BDVL A12 acceptance: exact-resume equivalence (guide.md R3R-3 point 5).

Runs a tiny real online-RL segment two ways with the SAME seed:

    straight:  stage2_online_rl(start=0, n=N,   total=N)
    resumed:   stage2_online_rl(start=0, n=N/2, total=N) -> save -> load
               into FRESH modules -> stage2_online_rl(start=N/2, n=N/2, total=N)

and asserts the two final states are bit-identical: online/EMA network
weights, optimizer state, replay buffer contents, and RNG state. This is
the exact test guide.md flagged as missing -- unit tests on the pieces
(EMA hard-copy, epsilon schedule, checkpoint format v4) do not catch a
resume that silently diverges from an uninterrupted run.
"""

from __future__ import annotations

import sys
import tempfile
import hashlib
import pickle
import random
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import ActionGridSpec, FROZEN_VALUES, build_frozen_registry, derive_return_bounds  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact, Track, fit_sbk_hmm  # noqa: E402
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder  # noqa: E402
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork  # noqa: E402
from crowd_nav.bayesian_dvl.replay import DemoOnlineReplay  # noqa: E402
from crowd_nav.tools.train_bdvl import (  # noqa: E402
    _load_training_checkpoint, _save_training_checkpoint, _seed_everything, stage2_online_rl,
)

ENV_CONFIG_PATH = PACKAGE_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"
SEED = 77001
N_TOTAL = 4
N_HALF = 2
BATCH_SIZE = 2
UPDATES_PER_EPISODE = 1


def _synthetic_artifact() -> SBKHMMArtifact:
    dt = FROZEN_VALUES["dt"]

    def track(speed0, accel, omega):
        pos = [np.array([0.0, 0.0])]
        speed, heading = speed0, 0.0
        for _ in range(30):
            speed = max(speed + accel * dt, 0.0)
            heading += omega * dt
            vel = speed * np.array([np.cos(heading), np.sin(heading)])
            pos.append(pos[-1] + vel * dt)
        return Track(positions=np.array(pos), dt=dt)

    tracks = [track(1.0, 0.0, 0.0), track(0.3, 0.8, 0.0), track(1.5, -0.8, 0.0), track(1.0, 0.0, 1.0), track(1.0, 0.0, -1.0)]
    return fit_sbk_hmm(tracks, train_data_sha256="a12_resume_equivalence_fixture", max_iterations=15)


def _fresh_modules(v_min: float, v_max: float, device: str):
    encoder = SetEncoder(human_hidden_dim=16, human_embed_dim=8, robot_embed_dim=8, embedding_dim=16).to(device)
    action_encoder = ActionEncoder(hidden_dim=16, embed_dim=8).to(device)
    value_network = IQNValueNetwork(state_embedding_dim=16, action_embedding_dim=action_encoder.embed_dim, n_cosines=8, hidden_dim=16, v_min=v_min, v_max=v_max).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(value_network.parameters()) + list(action_encoder.parameters()), lr=1e-3)
    ema_encoder = SetEncoder(human_hidden_dim=16, human_embed_dim=8, robot_embed_dim=8, embedding_dim=16).to(device)
    ema_action_encoder = ActionEncoder(hidden_dim=16, embed_dim=8).to(device)
    ema_value_network = IQNValueNetwork(state_embedding_dim=16, action_embedding_dim=ema_action_encoder.embed_dim, n_cosines=8, hidden_dim=16, v_min=v_min, v_max=v_max).to(device)
    for p in list(ema_encoder.parameters()) + list(ema_value_network.parameters()) + list(ema_action_encoder.parameters()):
        p.requires_grad_(False)
    return encoder, value_network, action_encoder, optimizer, ema_encoder, ema_value_network, ema_action_encoder


def _state_dict_signature(module: torch.nn.Module) -> str:
    parts = [t.detach().cpu().numpy().tobytes() for t in module.state_dict().values()]
    import hashlib
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _canonical_bytes(value) -> bytes:
    """Stable content encoding; pickle memo ordering is not a state hash."""
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"B1" if value else b"B0"
    if isinstance(value, (int, float, str, bytes)):
        return (type(value).__name__ + ":" + repr(value)).encode("utf-8")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return b"T" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode() + tensor.numpy().tobytes()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return b"A" + str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
    if is_dataclass(value):
        return b"D" + type(value).__name__.encode() + b"".join(
            _canonical_bytes(getattr(value, field.name)) for field in fields(value)
        )
    if isinstance(value, dict):
        return b"M" + b"".join(
            _canonical_bytes(key) + _canonical_bytes(value[key])
            for key in sorted(value, key=lambda item: repr(item))
        )
    if isinstance(value, (list, tuple)):
        return (b"L" if isinstance(value, list) else b"Q") + b"".join(_canonical_bytes(item) for item in value)
    return (type(value).__name__ + ":" + repr(value)).encode("utf-8")


def _object_signature(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _runtime_signature(optimizer, replay):
    return {
        "optimizer": _object_signature(optimizer.state_dict()),
        "replay": _object_signature(replay.state_dict()),
        "python_rng": _object_signature(random.getstate()),
        "numpy_rng": _object_signature(np.random.get_state()),
        "torch_rng": _object_signature(torch.get_rng_state()),
        "cuda_rng": _object_signature(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def _run_straight(device: str, registry: dict, training_config_sha256: str):
    _seed_everything(SEED)
    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    artifact = _synthetic_artifact()
    action_spec = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = action_spec.build_action_table()
    encoder, value_network, action_encoder, optimizer, ema_encoder, ema_value_network, ema_action_encoder = _fresh_modules(v_min, v_max, device)
    ema_encoder.load_state_dict(encoder.state_dict())
    ema_value_network.load_state_dict(value_network.state_dict())
    ema_action_encoder.load_state_dict(action_encoder.state_dict())
    replay = DemoOnlineReplay(1000, 1000, 0.2)
    losses = stage2_online_rl(
        encoder, value_network, action_encoder, artifact, action_table, ENV_CONFIG_PATH, replay, optimizer, device, SEED,
        n_episodes=N_TOTAL, start_episode=0, total_rl_episodes=N_TOTAL, gamma=0.99,
        epsilon_start=0.9, epsilon_end=0.1, batch_size=BATCH_SIZE, updates_per_episode=UPDATES_PER_EPISODE,
        profiles=["nominal"], checkpoint_dir=None, registry=registry, training_config_sha256=training_config_sha256,
        ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder, ema_initialized=True,
    )
    return encoder, value_network, action_encoder, optimizer, ema_encoder, ema_value_network, ema_action_encoder, replay, losses, _runtime_signature(optimizer, replay)


def _run_resumed(device: str, registry: dict, training_config_sha256: str, tmp_dir: Path):
    _seed_everything(SEED)
    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    artifact = _synthetic_artifact()
    action_spec = ActionGridSpec.from_env_config(str(ENV_CONFIG_PATH))
    action_table = action_spec.build_action_table()
    encoder, value_network, action_encoder, optimizer, ema_encoder, ema_value_network, ema_action_encoder = _fresh_modules(v_min, v_max, device)
    ema_encoder.load_state_dict(encoder.state_dict())
    ema_value_network.load_state_dict(value_network.state_dict())
    ema_action_encoder.load_state_dict(action_encoder.state_dict())
    replay = DemoOnlineReplay(1000, 1000, 0.2)
    losses_a = stage2_online_rl(
        encoder, value_network, action_encoder, artifact, action_table, ENV_CONFIG_PATH, replay, optimizer, device, SEED,
        n_episodes=N_HALF, start_episode=0, total_rl_episodes=N_TOTAL, gamma=0.99,
        epsilon_start=0.9, epsilon_end=0.1, batch_size=BATCH_SIZE, updates_per_episode=UPDATES_PER_EPISODE,
        profiles=["nominal"], checkpoint_dir=None, registry=registry, training_config_sha256=training_config_sha256,
        ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder, ema_initialized=True,
    )
    ckpt_path = tmp_dir / "resume_test.pth"
    _save_training_checkpoint(
        ckpt_path, encoder, value_network, action_encoder, optimizer, replay,
        {
            "seed": SEED, "episode": N_HALF, "action_grid_hash": registry["action_grid_hash"],
            "registry_content_sha256": registry["content_sha256"], "artifact_sha256": artifact.content_sha256(),
            "world_train_data_sha256": artifact.train_data_sha256, "training_config_sha256": training_config_sha256,
            "ema_initialized": True,
        },
        ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder,
    )
    # Fresh modules to prove the checkpoint -- not process memory -- carries the state.
    encoder2, value_network2, action_encoder2, optimizer2, ema_encoder2, ema_value_network2, ema_action_encoder2 = _fresh_modules(v_min, v_max, device)
    replay2 = DemoOnlineReplay(1000, 1000, 0.2)
    resume_state = _load_training_checkpoint(
        ckpt_path, encoder2, value_network2, action_encoder2, optimizer2, replay2, device,
        ema_encoder=ema_encoder2, ema_value_network=ema_value_network2, ema_action_encoder=ema_action_encoder2,
    )
    assert resume_state["episode"] == N_HALF, f"expected resume episode={N_HALF}, got {resume_state['episode']}"
    losses_b = stage2_online_rl(
        encoder2, value_network2, action_encoder2, artifact, action_table, ENV_CONFIG_PATH, replay2, optimizer2, device, SEED,
        n_episodes=N_TOTAL - N_HALF, start_episode=N_HALF, total_rl_episodes=N_TOTAL, gamma=0.99,
        epsilon_start=0.9, epsilon_end=0.1, batch_size=BATCH_SIZE, updates_per_episode=UPDATES_PER_EPISODE,
        profiles=["nominal"], checkpoint_dir=None, registry=registry, training_config_sha256=training_config_sha256,
        ema_encoder=ema_encoder2, ema_value_network=ema_value_network2, ema_action_encoder=ema_action_encoder2, ema_initialized=True,
    )
    return encoder2, value_network2, action_encoder2, optimizer2, ema_encoder2, ema_value_network2, ema_action_encoder2, replay2, losses_a + losses_b, _runtime_signature(optimizer2, replay2)


if __name__ == "__main__":
    device = "cpu"
    registry = build_frozen_registry(str(ENV_CONFIG_PATH)).to_json_dict()
    training_config_sha256 = "a12_resume_equivalence_fixture_config_hash"

    enc_a, val_a, act_a, opt_a, ema_enc_a, ema_val_a, ema_act_a, replay_a, losses_a, runtime_a = _run_straight(device, registry, training_config_sha256)
    with tempfile.TemporaryDirectory() as tmp:
        enc_b, val_b, act_b, opt_b, ema_enc_b, ema_val_b, ema_act_b, replay_b, losses_b, runtime_b = _run_resumed(device, registry, training_config_sha256, Path(tmp))

    sig_enc_a, sig_enc_b = _state_dict_signature(enc_a), _state_dict_signature(enc_b)
    sig_val_a, sig_val_b = _state_dict_signature(val_a), _state_dict_signature(val_b)
    sig_act_a, sig_act_b = _state_dict_signature(act_a), _state_dict_signature(act_b)
    sig_ema_enc_a, sig_ema_enc_b = _state_dict_signature(ema_enc_a), _state_dict_signature(ema_enc_b)
    sig_ema_val_a, sig_ema_val_b = _state_dict_signature(ema_val_a), _state_dict_signature(ema_val_b)
    sig_ema_act_a, sig_ema_act_b = _state_dict_signature(ema_act_a), _state_dict_signature(ema_act_b)

    assert sig_enc_a == sig_enc_b, f"encoder weights diverged after resume: {sig_enc_a} != {sig_enc_b}"
    assert sig_val_a == sig_val_b, f"value_network weights diverged after resume: {sig_val_a} != {sig_val_b}"
    assert sig_act_a == sig_act_b, f"action_encoder weights diverged after resume: {sig_act_a} != {sig_act_b}"
    assert sig_ema_enc_a == sig_ema_enc_b, f"EMA encoder weights diverged after resume: {sig_ema_enc_a} != {sig_ema_enc_b}"
    assert sig_ema_val_a == sig_ema_val_b, f"EMA value_network weights diverged after resume: {sig_ema_val_a} != {sig_ema_val_b}"
    assert sig_ema_act_a == sig_ema_act_b, f"EMA action_encoder weights diverged after resume: {sig_ema_act_a} != {sig_ema_act_b}"
    assert len(replay_a.online) == len(replay_b.online), f"online replay size diverged: {len(replay_a.online)} != {len(replay_b.online)}"
    assert losses_a == losses_b, f"loss trajectories diverged after resume:\nstraight={losses_a}\nresumed ={losses_b}"
    assert runtime_a == runtime_b, f"optimizer/replay/RNG state diverged after resume:\nstraight={runtime_a}\nresumed={runtime_b}"

    print(f"A12_RESUME_EQUIVALENCE_PASS n_total={N_TOTAL} n_half={N_HALF} "
          f"online_size={len(replay_a.online)} losses_match=True weights_match=True "
          f"ema_match=True optimizer_replay_rng_match=True")
