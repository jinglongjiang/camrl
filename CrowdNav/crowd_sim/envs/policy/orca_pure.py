# -*- coding: utf-8 -*-
"""
纯ORCA实现 - 回归理论本质 (方案A + 方案C优化)

核心改动：
- 移除候选评分机制（破坏ORCA安全保证的根源）
- 直接使用RVO2库计算结果，信任ORCA理论
- 优化参数配置，提升拥挤环境适应性
- 代码从400行精简到150行

预期效果：
- 成功率：33% → 60-70%
- 碰撞率：23.8% → <5%
- 超时率：42.8% → 25-35%
"""

import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'

        # ORCA参数（优化后的默认值 - 方案C）
        self.safety_space = 0.10        # 降低（原0.20 → 0.10）
        self.neighbor_dist = 4.5        # 扩大（原3.0 → 4.5）
        self.max_neighbors = 12         # 增加（原10 → 12）
        self.time_horizon = 5.0         # 延长（原3.0 → 5.0）
        self.time_horizon_obst = 3.0
        self.radius = 0.3
        self.max_speed = 1.0

        self.sim = None
        self._sim_timestep = None

    def configure(self, cfg):
        """从配置文件读取ORCA参数（优化后）"""
        import logging

        if hasattr(cfg, 'getfloat'):
            # 从[orca]段读取参数（使用优化后的fallback值）
            try:
                self.neighbor_dist = cfg.getfloat('orca', 'neighbor_dist', fallback=4.5)
                self.max_neighbors = cfg.getint('orca', 'max_neighbors', fallback=12)
                self.time_horizon = cfg.getfloat('orca', 'time_horizon', fallback=5.0)
                self.time_horizon_obst = cfg.getfloat('orca', 'time_horizon_obst', fallback=3.0)
                self.safety_space = cfg.getfloat('orca', 'safety_space', fallback=0.10)

                # 速度限制
                robot_v_pref = cfg.getfloat('robot', 'v_pref', fallback=1.0)
                grid_v_max = cfg.getfloat('policy', 'v_max', fallback=1.0)
                self.max_speed = min(self.max_speed, robot_v_pref, grid_v_max)

                # 时间步
                self.time_step = cfg.getfloat('env', 'time_step', fallback=0.25)

                # Use print for immediate output
                print(f"[ORCA-PURE] neighbor_dist={self.neighbor_dist:.2f}, "
                      f"time_horizon={self.time_horizon:.2f}, "
                      f"safety_space={self.safety_space:.2f}, "
                      f"max_speed={self.max_speed:.2f}, "
                      f"max_neighbors={self.max_neighbors}", flush=True)

            except Exception as e:
                print(f"[ORCA-PURE] Failed to read config: {e}", flush=True)
                logging.warning(f"[ORCA-PURE] Failed to read config: {e}")

        elif isinstance(cfg, dict):
            for k in ("neighbor_dist", "max_neighbors", "time_horizon",
                     "time_horizon_obst", "safety_space", "max_speed"):
                if k in cfg:
                    setattr(self, k, type(getattr(self, k))(cfg[k]))

    def predict(self, state):
        """预测动作"""
        if isinstance(state, list):
            if len(state) == 1:
                return [self._predict_single(state[0])]
            else:
                return self._predict_multi(state)
        else:
            return self._predict_single(state)

    def _predict_single(self, state):
        """单agent预测 - 纯ORCA实现（方案A）"""
        agent_state = state.self_state

        # 计算朝向目标的期望速度
        base_velocity = self._toward_goal(agent_state)

        # 如果没有其他agent，直接返回期望速度
        if not hasattr(state, 'human_states') or not state.human_states:
            return ActionXY(base_velocity[0], base_velocity[1])

        # 创建RVO2仿真器
        self._setup_sim()

        # 添加机器人
        self.sim.addAgent(
            (agent_state.px, agent_state.py),
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            agent_state.radius + self.safety_space,
            self.max_speed,
            (0.0, 0.0)
        )

        # 添加人类agents
        for human_state in state.human_states:
            self.sim.addAgent(
                (human_state.px, human_state.py),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                human_state.radius + self.safety_space,
                getattr(human_state, 'v_pref', 1.0),
                (0.0, 0.0)
            )

        # 设置首选速度
        self.sim.setAgentPrefVelocity(0, tuple(base_velocity))
        for i, human_state in enumerate(state.human_states):
            human_velocity = self._human_pref_velocity(human_state)
            self.sim.setAgentPrefVelocity(i + 1, tuple(human_velocity))

        # 执行ORCA计算
        self.sim.doStep()

        # ⭐ 核心改动：直接使用ORCA结果，不做任何"优化"
        velocity = np.array(self.sim.getAgentVelocity(0), dtype=np.float32)

        # 仅限制最大速度（不改变方向）
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed > 1e-6:
            velocity = velocity * (self.max_speed / speed)

        return ActionXY(float(velocity[0]), float(velocity[1]))

    def _predict_multi(self, states):
        """多agent预测"""
        if not states:
            return []

        self._setup_sim()

        # 添加所有agents
        for state in states:
            agent_state = state.self_state
            self.sim.addAgent(
                (agent_state.px, agent_state.py),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                agent_state.radius + self.safety_space,
                agent_state.v_pref,
                (0.0, 0.0)
            )

        # 设置首选速度
        for i, state in enumerate(states):
            base_velocity = self._toward_goal(state.self_state)
            self.sim.setAgentPrefVelocity(i, tuple(base_velocity))

        # 执行ORCA计算
        self.sim.doStep()

        # 收集速度
        actions = []
        for i in range(len(states)):
            velocity = self.sim.getAgentVelocity(i)
            actions.append(ActionXY(velocity[0], velocity[1]))

        return actions

    def _toward_goal(self, agent_state):
        """计算朝向目标的期望速度"""
        if hasattr(agent_state, 'gx') and hasattr(agent_state, 'gy'):
            dx = agent_state.gx - agent_state.px
            dy = agent_state.gy - agent_state.py
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 1e-6:
                return np.array([0.0, 0.0], dtype=np.float32)

            direction = np.array([dx / distance, dy / distance], dtype=np.float32)
            v_pref = getattr(agent_state, 'v_pref', 1.0)
            speed = min(v_pref, distance / getattr(self, 'time_step', 0.25))

            return direction * speed
        else:
            # 人类状态
            vx = getattr(agent_state, 'vx', 0.0)
            vy = getattr(agent_state, 'vy', 0.0)
            return np.array([vx, vy], dtype=np.float32)

    def _human_pref_velocity(self, human_state):
        """计算人类的期望速度"""
        dx = getattr(human_state, 'gx', human_state.px) - human_state.px
        dy = getattr(human_state, 'gy', human_state.py) - human_state.py
        vec = np.array([dx, dy], dtype=np.float32)
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            vx = getattr(human_state, 'vx', 0.0)
            vy = getattr(human_state, 'vy', 0.0)
            return np.array([vx, vy], dtype=np.float32)

        v_pref = getattr(human_state, 'v_pref', 1.0)
        return vec / dist * v_pref

    def _setup_sim(self):
        """设置RVO2仿真器"""
        time_step = getattr(self, 'time_step', 0.25)
        self.sim = rvo2.PyRVOSimulator(
            time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed
        )
        self._sim_timestep = time_step

    # 兼容接口
    def get_model(self):
        return None

    def get_state_dict(self):
        return {}

    def get_normalized_gamma(self):
        return None

    def load_model(self, model_path):
        pass

    def save_model(self, model_path):
        pass
