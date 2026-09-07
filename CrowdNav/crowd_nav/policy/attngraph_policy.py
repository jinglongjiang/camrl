"""
AttnGraph policy adapter for the local CrowdNav evaluation harness.

This wrapper loads the official ICRA 2023 AttnGraph policy checkpoint and
exposes the same predict(state) interface used by the local test scripts.
The default path uses the official GST predictor to fill the future part of
AttnGraph's spatial_edges input.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import pickle
from collections import deque
from typing import Optional

import numpy as np
import torch
from gym.spaces import Box, Dict

from crowd_sim.envs.utils.action import ActionXY


ATTNGRAPH_ROOT = "/home/abc/temp/CrowdNav_Prediction_AttnGraph"
DEFAULT_MODEL_DIR = os.path.join(ATTNGRAPH_ROOT, "trained_models", "GST_predictor_rand")
DEFAULT_GST_MODEL_DIR = os.path.join(
    ATTNGRAPH_ROOT,
    "gst_updated",
    "results",
    "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand",
    "sj",
)


def _prepend_attngraph_root(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_saved_args(model_dir: str):
    args_path = os.path.join(model_dir, "arguments.py")
    spec = importlib.util.spec_from_file_location("attngraph_saved_arguments", args_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AttnGraph arguments from {args_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        return module.get_args()
    finally:
        sys.argv = old_argv


def _load_gst_args(gst_model_dir: str):
    args_path = os.path.join(gst_model_dir, "checkpoint", "args.pickle")
    with open(args_path, "rb") as f:
        return pickle.load(f)


class AttnGraphPolicy:
    """Adapter around the official AttnGraph PPO policy."""

    multiagent_training = False
    kinematics = "holonomic"
    trainable = False
    phase = None
    time_step = 0.25

    def __init__(
        self,
        config=None,
        attngraph_root: str = ATTNGRAPH_ROOT,
        model_dir: str = DEFAULT_MODEL_DIR,
        gst_model_dir: str = DEFAULT_GST_MODEL_DIR,
        prediction: str = "gst",
        device: Optional[torch.device] = None,
    ):
        self.attngraph_root = attngraph_root
        self.model_dir = model_dir
        self.gst_model_dir = gst_model_dir
        self.prediction = prediction
        self.device = device or torch.device("cpu")
        self.max_humans = 20
        self.predict_steps = 5
        self.obs_seq_len = 5
        self.pred_interval = 1
        self.sensor_range = 5.0

        _prepend_attngraph_root(self.attngraph_root)
        from rl.networks.model import Policy  # imported after sys.path setup

        self.args = _load_saved_args(self.model_dir)
        self.args.no_cuda = self.device.type != "cuda"
        self.args.cuda = self.device.type == "cuda"
        self.args.num_processes = 1
        self.args.num_mini_batch = 1
        self.args.env_name = "CrowdSimPredRealGST-v0"
        self.args.sort_humans = True

        obs_spaces = Dict({
            "robot_node": Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
            "temporal_edges": Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            "spatial_edges": Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_humans, 2 * (self.predict_steps + 1)),
                dtype=np.float32,
            ),
            "visible_masks": Box(low=0, high=1, shape=(self.max_humans,), dtype=np.bool_),
            "detected_human_num": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
        action_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.model = Policy(
            obs_spaces,
            action_space,
            base_kwargs=self.args,
            base="selfAttn_merge_srnn",
        )
        self.model.base.nenv = 1

        self.predictor = None
        if self.prediction == "gst":
            from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import (
                CrowdNavPredInterfaceMultiEnv,
            )

            self.gst_args = _load_gst_args(self.gst_model_dir)
            self.obs_seq_len = int(self.gst_args.obs_seq_len)
            self.predict_steps = int(self.gst_args.pred_seq_len)
            self.predictor = CrowdNavPredInterfaceMultiEnv(
                load_path=self.gst_model_dir,
                device=self.device,
                config=self.gst_args,
                num_env=1,
            )

        self.reset_episode_stats()
        self.model.to(self.device)
        self.model.eval()

    def load_state_dict(self, state_dict):
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.base.nenv = 1
        self.model.to(self.device)
        self.model.eval()

    def to(self, device):
        self.set_device(device)
        return self

    def eval(self):
        self.model.eval()

    def set_device(self, device):
        self.device = torch.device(device)
        self.args.no_cuda = self.device.type != "cuda"
        self.args.cuda = self.device.type == "cuda"
        self.model.to(self.device)
        self.reset_episode_stats()

    def set_phase(self, phase):
        self.phase = phase

    def configure(self, config):
        if config is not None and config.has_option("env", "time_step"):
            self.time_step = config.getfloat("env", "time_step")
        # Match AttnGraph's native visibility model by default. The local
        # benchmark config may not define robot.sensor_range, so keep the
        # official AttnGraph value (5m) unless explicitly provided.
        if config is not None and config.has_option("robot", "sensor_range"):
            self.sensor_range = config.getfloat("robot", "sensor_range")

    def reset_episode_stats(self):
        num_edges = self.max_humans + 1
        self.rnn_hxs = {
            "human_node_rnn": torch.zeros(
                1, 1, self.args.human_node_rnn_size, device=self.device
            ),
            "human_human_edge_rnn": torch.zeros(
                1, num_edges, self.args.human_human_edge_rnn_size, device=self.device
            ),
        }
        self.masks = torch.zeros(1, 1, device=self.device)
        invalid_pos = -torch.ones(
            (1, self.max_humans, 2),
            dtype=torch.float32,
            device=self.device,
        ) * 999.0
        invalid_mask = torch.zeros(
            (1, self.max_humans, 1),
            dtype=torch.bool,
            device=self.device,
        )
        self.traj_buffer = deque(
            [invalid_pos.clone() for _ in range(self.obs_seq_len)],
            maxlen=self.obs_seq_len,
        )
        self.mask_buffer = deque(
            [invalid_mask.clone() for _ in range(self.obs_seq_len)],
            maxlen=self.obs_seq_len,
        )
        self.last_human_states = np.zeros((self.max_humans, 5), dtype=np.float32)
        self.last_human_states[:, 0] = 15.0
        self.last_human_states[:, 1] = 15.0
        self.last_human_states[:, 4] = 0.3

    def predict(self, state):
        robot = state.self_state
        humans = state.human_states
        obs = self._convert_to_attngraph_obs(robot, humans)

        with torch.no_grad():
            _, action, _, self.rnn_hxs = self.model.act(
                obs,
                self.rnn_hxs,
                self.masks,
                deterministic=True,
            )
        self.masks = torch.ones(1, 1, device=self.device)

        action = action.detach().cpu().numpy().reshape(-1)
        vx, vy = float(action[0]), float(action[1])
        v_pref = float(getattr(robot, "v_pref", 1.0))
        speed = float(np.linalg.norm([vx, vy]))
        if speed > v_pref and speed > 1e-8:
            vx = vx / speed * v_pref
            vy = vy / speed * v_pref
        return ActionXY(vx, vy)

    def _convert_to_attngraph_obs(self, robot, humans):
        robot_node = np.array([[
            robot.px,
            robot.py,
            robot.radius,
            robot.gx,
            robot.gy,
            robot.v_pref,
            getattr(robot, "theta", 0.0),
        ]], dtype=np.float32)

        temporal_edges = np.array([[robot.vx, robot.vy]], dtype=np.float32)
        current_abs = np.zeros((self.max_humans, 2), dtype=np.float32)
        spatial_edges = np.ones((self.max_humans, 12), dtype=np.float32) * 15.0
        visible_masks = np.zeros((self.max_humans,), dtype=np.bool_)

        ordered_humans = []

        for raw_i, human in enumerate(list(humans)[: self.max_humans]):
            i = int(getattr(human, "id", raw_i))
            if i < 0 or i >= self.max_humans:
                continue
            ordered_humans.append((i, human))
            center_dist = float(np.linalg.norm([human.px - robot.px, human.py - robot.py]))
            clearance = center_dist - float(getattr(robot, "radius", 0.3)) - float(getattr(human, "radius", 0.3))
            visible_masks[i] = clearance <= self.sensor_range
            if visible_masks[i]:
                self.last_human_states[i] = [
                    human.px,
                    human.py,
                    getattr(human, "vx", 0.0),
                    getattr(human, "vy", 0.0),
                    getattr(human, "radius", 0.3),
                ]
            else:
                self.last_human_states[i, 0] += self.last_human_states[i, 2] * self.time_step
                self.last_human_states[i, 1] += self.last_human_states[i, 3] * self.time_step
            current_abs[i] = self.last_human_states[i, :2]

        if self.prediction == "gst" and self.predictor is not None:
            future_rel, pred_mask = self._future_edges_gst(robot, current_abs, visible_masks)
        else:
            future_rel, pred_mask = self._future_edges_cv(robot, ordered_humans, visible_masks)

        for i, human in ordered_humans:
            if not visible_masks[i]:
                continue
            rel = [human.px - robot.px, human.py - robot.py]
            rel.extend(future_rel[i].reshape(-1).tolist())
            spatial_edges[i] = np.asarray(rel, dtype=np.float32)
            if not pred_mask[i]:
                # Keep current observation but avoid trusting unavailable futures.
                for k in range(1, self.predict_steps + 1):
                    spatial_edges[i, 2 * k: 2 * k + 2] = spatial_edges[i, :2]

        sort_idx = np.argsort(np.linalg.norm(spatial_edges[:, :2], axis=1))
        spatial_edges = spatial_edges[sort_idx].astype(np.float32)
        detected_human_num = np.array([max(1, int(visible_masks.sum()))], dtype=np.float32)

        return {
            "robot_node": torch.from_numpy(robot_node).unsqueeze(0).to(self.device),
            "temporal_edges": torch.from_numpy(temporal_edges).unsqueeze(0).to(self.device),
            "spatial_edges": torch.from_numpy(spatial_edges).unsqueeze(0).to(self.device),
            "visible_masks": torch.from_numpy(visible_masks).unsqueeze(0).to(self.device),
            "detected_human_num": torch.from_numpy(detected_human_num).unsqueeze(0).to(self.device),
        }

    def _future_edges_cv(self, robot, humans, visible_masks):
        future = np.zeros((self.max_humans, self.predict_steps, 2), dtype=np.float32)
        pred_mask = visible_masks.astype(bool).copy()
        for i, human in humans:
            if i >= self.max_humans:
                continue
            for k in range(1, self.predict_steps + 1):
                px = human.px + human.vx * self.time_step * k
                py = human.py + human.vy * self.time_step * k
                future[i, k - 1] = [px - robot.px, py - robot.py]
        return future, pred_mask

    def _future_edges_gst(self, robot, current_abs, visible_masks):
        current_tensor = torch.as_tensor(
            current_abs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            visible_masks,
            dtype=torch.bool,
            device=self.device,
        ).reshape(1, self.max_humans, 1)

        self.traj_buffer.append(current_tensor)
        self.mask_buffer.append(mask_tensor)
        in_traj = torch.stack(list(self.traj_buffer)).permute(1, 2, 0, 3)
        in_mask = torch.stack(list(self.mask_buffer)).permute(1, 2, 0, 3).float()

        out_traj, out_mask = self.predictor.forward(
            input_traj=in_traj,
            input_binary_mask=in_mask,
            sampling=False,
        )
        robot_pos = torch.tensor([robot.px, robot.py], dtype=torch.float32, device=self.device)
        future_rel = out_traj[0, :, :, :2] - robot_pos.reshape(1, 1, 2)
        pred_mask = out_mask[0, :, 0].bool()
        future_rel = future_rel.detach().cpu().numpy().astype(np.float32)
        pred_mask = pred_mask.detach().cpu().numpy().astype(bool)
        future_rel[~pred_mask] = 0.0
        return future_rel, pred_mask
