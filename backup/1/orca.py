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

        # defaults，可被 configure() 覆盖
        self.safety_space = 0.05
        self.neighbor_dist = 2.0
        self.max_neighbors = 5
        self.time_horizon = 3.0
        self.time_horizon_obst = 2.0
        self.radius = 0.3
        self.max_speed = 1.0

        # 训练用保守标签的额外膨胀（仅作用于机器人半径）
        self.label_inflate = 0.0
        # 接近终点时线性降速阈值系数：dist < slow_k * radius 开始减速
        self.slow_k = 1.5
        # 轻微破对称（避免停滞），0 关闭
        self.eps_noise = 0.0

        self.sim = None
        self._sim_timestep = None

    # 允许从配置覆盖参数
    def configure(self, config):
        if hasattr(config, 'getfloat'):
            g = config.getfloat
            gi = config.getint
            sec = 'orca'
            self.safety_space     = g(sec, 'safety_space',     fallback=self.safety_space)
            self.neighbor_dist    = g(sec, 'neighbor_dist',    fallback=self.neighbor_dist)
            self.max_neighbors    = gi(sec, 'max_neighbors',   fallback=self.max_neighbors)
            self.time_horizon     = g(sec, 'time_horizon',     fallback=self.time_horizon)
            self.time_horizon_obst= g(sec, 'time_horizon_obst',fallback=self.time_horizon_obst)
            self.radius           = g(sec, 'radius',           fallback=self.radius)
            self.max_speed        = g(sec, 'max_speed',        fallback=self.max_speed)
            self.label_inflate    = g(sec, 'label_inflate',    fallback=self.label_inflate)
            self.slow_k           = g(sec, 'slow_k',           fallback=self.slow_k)
            self.eps_noise        = g(sec, 'eps_noise',        fallback=self.eps_noise)

    def set_phase(self, phase):
        return

    def _need_rebuild(self, n_agents):
        return (self.sim is None
                or self._sim_timestep is None
                or abs(self._sim_timestep - float(self.time_step)) > 1e-9
                or self.sim.getNumAgents() != n_agents)

    def predict(self, state):
        """
        用 ORCA 给机器人算一帧速度标签。只输出机器人动作，不改环境人群。
        """
        self_state = state.self_state
        humans = state.human_states
        n_agents = 1 + len(humans)

        params = (self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst)

        # —— 构建/复用仿真器
        if self._need_rebuild(n_agents):
            if self.sim is not None:
                del self.sim
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self._sim_timestep = float(self.time_step)

            # 机器人 agent（半径叠加 safety + label_inflate）
            self.sim.addAgent(self_state.position, *params,
                              self_state.radius + 0.01 + self.safety_space + self.label_inflate,
                              max(self_state.v_pref, 1e-6), self_state.velocity)
            # 人类 agents（用各自 v_pref 作为 maxSpeed，更真实）
            for h in humans:
                h_max = getattr(h, 'v_pref', self.max_speed)
                self.sim.addAgent(h.position, *params, h.radius + 0.01 + self.safety_space,
                                  max(h_max, 1e-6), h.velocity)
        else:
            # 同步位置与速度
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, h in enumerate(humans):
                self.sim.setAgentPosition(i + 1, h.position)
                self.sim.setAgentVelocity(i + 1, h.velocity)

        # —— 设定首选速度（关键修复）
        # 方向
        to_goal = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py), dtype=float)
        dist = np.linalg.norm(to_goal)
        if dist < 1e-8:
            pref_vel = (0.0, 0.0)
        else:
            dir_vec = to_goal / dist
            # 目标速度：不超 v_pref，不超 max_speed，不超一帧可达距离/time_step
            v_lim = min(self_state.v_pref, self.max_speed, dist / max(self.time_step, 1e-6))
            # 近终点线性降速，避免在终点附近来回抖动
            if dist < self.slow_k * self_state.radius:
                v_lim *= dist / max(self.slow_k * self_state.radius, 1e-6)
            pref_vel = tuple(dir_vec * v_lim)

        # 可选：轻微破对称
        if self.eps_noise > 0.0 and dist > 1e-6:
            ang = np.random.uniform(0, 2 * np.pi)
            jitter = self.eps_noise * np.array([np.cos(ang), np.sin(ang)], dtype=float)
            pref_vel = tuple(np.asarray(pref_vel) + jitter)

        self.sim.setAgentPrefVelocity(0, pref_vel)

        # 其他人类未知目标，设 0（ORCA 只用他们的当前位置与速度约束，不用首选速度）
        for i in range(len(humans)):
            self.sim.setAgentPrefVelocity(i + 1, (0.0, 0.0))

        # —— 单步求解并取机器人速度
        self.sim.doStep()
        vx, vy = self.sim.getAgentVelocity(0)
        return ActionXY(float(vx), float(vy))

