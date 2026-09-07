"""
DSRNN Policy Wrapper - 适配到CrowdNav测试框架
使用本地复制的pytorchBaselines模块
"""
import torch
import numpy as np
import sys
import os

# 使用本地复制的模块
_policy_dir = os.path.dirname(os.path.abspath(__file__))
if _policy_dir not in sys.path:
    sys.path.insert(0, _policy_dir)

from .pytorchBaselines.a2c_ppo_acktr.model import Policy
from .dsrnn_configs.config import Config as DSRNNConfig
from gym.spaces import Box, Dict
from crowd_sim.envs.utils.action import ActionXY


class DSRNNPolicy:
    """DSRNN策略，兼容CrowdNav测试框架"""

    multiagent_training = False  # 兼容CrowdSim环境
    kinematics = 'holonomic'  # 运动学模型
    trainable = True  # 可训练
    phase = None  # 训练/测试阶段
    time_step = None  # 时间步长

    def __init__(self, config=None):
        self.device = torch.device('cpu')
        self.dsrnn_config = DSRNNConfig()
        self.human_num = 5

        # 构建observation space
        obs_spaces = Dict({
            'robot_node': Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
            'temporal_edges': Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            'spatial_edges': Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype=np.float32),
        })

        # 构建action space
        action_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # 创建模型
        self.model = Policy(
            obs_spaces,
            action_space,
            base_kwargs=self.dsrnn_config,
            base=self.dsrnn_config.robot.policy
        )

        # 初始化RNN hidden states
        num_edges = self.human_num + 1
        self.rnn_hxs = {
            'human_node_rnn': torch.zeros(1, 1, self.dsrnn_config.SRNN.human_node_rnn_size),
            'human_human_edge_rnn': torch.zeros(1, num_edges, self.dsrnn_config.SRNN.human_human_edge_rnn_size),
        }
        self.masks = torch.ones(1, 1)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.base.nenv = 1
        self.dsrnn_config.training.cuda = False
        self.model.to(self.device)
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()

    def reset_episode_stats(self):
        num_edges = self.human_num + 1
        self.rnn_hxs = {
            'human_node_rnn': torch.zeros(1, 1, self.dsrnn_config.SRNN.human_node_rnn_size).to(self.device),
            'human_human_edge_rnn': torch.zeros(1, num_edges, self.dsrnn_config.SRNN.human_human_edge_rnn_size).to(self.device),
        }
        self.masks = torch.ones(1, 1).to(self.device)

    def predict(self, state):
        robot = state.self_state
        humans = state.human_states
        obs = self._convert_to_dsrnn_obs(robot, humans)

        with torch.no_grad():
            _, action, _, self.rnn_hxs = self.model.act(
                obs, self.rnn_hxs, self.masks, deterministic=True
            )

        action = action.cpu().numpy().flatten()
        vx, vy = float(action[0]), float(action[1])

        v_pref = robot.v_pref if hasattr(robot, 'v_pref') else 1.0
        speed = np.sqrt(vx**2 + vy**2)
        if speed > v_pref:
            vx = vx / speed * v_pref
            vy = vy / speed * v_pref

        return ActionXY(vx, vy)

    def configure(self, config):
        """配置策略参数"""
        if config.has_option('env', 'time_step'):
            self.time_step = config.getfloat('env', 'time_step')

    def set_phase(self, phase):
        """设置训练/测试阶段"""
        self.phase = phase

    def set_device(self, device):
        """设置设备"""
        self.device = device
        self.model.to(device)
        # Reset RNN states on new device
        self.reset_episode_stats()

    def _convert_to_dsrnn_obs(self, robot, humans):
        robot_node = np.array([[
            robot.px, robot.py, robot.radius,
            robot.gx, robot.gy, robot.v_pref, robot.theta
        ]], dtype=np.float32)

        temporal_edges = np.array([[robot.vx, robot.vy]], dtype=np.float32)

        spatial_edges = np.zeros((self.human_num, 2), dtype=np.float32)
        for i, human in enumerate(humans[:self.human_num]):
            spatial_edges[i] = [human.px - robot.px, human.py - robot.py]

        obs = {
            'robot_node': torch.from_numpy(robot_node).unsqueeze(0).to(self.device),
            'temporal_edges': torch.from_numpy(temporal_edges).unsqueeze(0).to(self.device),
            'spatial_edges': torch.from_numpy(spatial_edges).unsqueeze(0).to(self.device),
        }
        return obs
