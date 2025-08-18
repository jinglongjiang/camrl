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

        # —— ORCA 基本参数（可被 configure 覆盖）
        self.safety_space = 0.05
        self.neighbor_dist = 2.5          # ↑ 适度放大，配合不可见机器人更稳
        self.max_neighbors = 8            # ↑ 稍多邻居可减少漏检
        self.time_horizon = 4.0           # ↑ 略长的预见时间更保守
        self.time_horizon_obst = 3.0
        self.radius = 0.3
        self.max_speed = 1.2              # 不会超过各自 v_pref，这里给上限稍宽

        # —— 标注期的额外安全（机器人侧）
        self.label_inflate = 0.10         # ↑ 默认更保守
        self.slow_k = 2.0                 # ↑ 临近终点提前减速
        self.eps_noise = 0.03             # ↑ 轻微破对称，避免对向僵持

        # 行人首选速度模式
        self.human_pref_mode = 'goal'     # 'goal' | 'current' | 'zero'

        # —— 新增：TTC 前馈减速参数（不重建 ORCA，仅缩放首选速度）
        self.ttc_brake = 2.0              # < 2s 进入制动区
        self.brake_min_ratio = 0.3        # 最低保留 30% 速度
        self.ttc_eps = 1e-6

        self.sim = None
        self._sim_timestep = None

    def configure(self, config):
        if hasattr(config, 'getfloat'):
            g = config.getfloat; gi = config.getint; gs = config.get
            sec = 'orca'
            self.safety_space      = g(sec, 'safety_space',      fallback=self.safety_space)
            self.neighbor_dist     = g(sec, 'neighbor_dist',     fallback=self.neighbor_dist)
            self.max_neighbors     = gi(sec, 'max_neighbors',    fallback=self.max_neighbors)
            self.time_horizon      = g(sec, 'time_horizon',      fallback=self.time_horizon)
            self.time_horizon_obst = g(sec, 'time_horizon_obst', fallback=self.time_horizon_obst)
            self.radius            = g(sec, 'radius',            fallback=self.radius)
            self.max_speed         = g(sec, 'max_speed',         fallback=self.max_speed)
            self.label_inflate     = g(sec, 'label_inflate',     fallback=self.label_inflate)
            self.slow_k            = g(sec, 'slow_k',            fallback=self.slow_k)
            self.eps_noise         = g(sec, 'eps_noise',         fallback=self.eps_noise)
            self.human_pref_mode   = gs(sec, 'human_pref_mode',  fallback=self.human_pref_mode).strip().lower()
            # 新增项支持从 config 读
            self.ttc_brake         = g(sec, 'ttc_brake',         fallback=self.ttc_brake)
            self.brake_min_ratio   = g(sec, 'brake_min_ratio',   fallback=self.brake_min_ratio)

    def set_phase(self, phase):
        return

    def _need_rebuild(self, n_agents):
        return (self.sim is None
                or self._sim_timestep is None
                or abs(self._sim_timestep - float(self.time_step)) > 1e-9
                or self.sim.getNumAgents() != n_agents)

    def _goal_pref_vel(self, px, py, gx, gy, v_pref, radius):
        """朝目标方向的首选速度（带临近目标的线性降速 + 单步距离限幅）"""
        to_goal = np.array((gx - px, gy - py), dtype=float)
        dist = np.linalg.norm(to_goal)
        if dist < 1e-8:
            return (0.0, 0.0)
        dir_vec = to_goal / dist
        v_lim = min(max(v_pref, 1e-6), self.max_speed, dist / max(self.time_step, 1e-6))
        if dist < self.slow_k * radius:
            v_lim *= dist / max(self.slow_k * radius, 1e-6)
        return tuple(dir_vec * v_lim)

    @staticmethod
    def _ttc_lin(px, py, vx, vy, rx, ry, rad_sum):
        """
        线性 TTC 近似：相对位置 p=(px,py)，相对速度 v=(vx,vy)，接触半径 rad_sum。
        返回 ttc（<=0 或不可解则 +inf）
        """
        p = np.array([px, py], dtype=float)
        v = np.array([vx, vy], dtype=float)
        vv = float(v @ v)
        if vv < 1e-8:
            return np.inf
        R2 = rad_sum * rad_sum
        b = 2.0 * float(p @ v)
        c = float(p @ p) - R2
        disc = b * b - 4.0 * vv * c
        if disc <= 0.0:
            return np.inf
        t1 = (-b - np.sqrt(disc)) / (2.0 * vv)
        if t1 <= 1e-6:
            return np.inf
        return float(t1)

    def _ttc_min_to_humans(self, self_state, humans):
        """
        用当前速度估计与所有行人的最小 TTC（无需重建 ORCA；作为前馈制动触发）
        """
        ttc_min = np.inf
        for h in humans:
            px = h.px - self_state.px
            py = h.py - self_state.py
            vx = h.vx - self_state.vx
            vy = h.vy - self_state.vy
            rad_sum = (getattr(h, 'radius', 0.3) + getattr(self_state, 'radius', 0.3))
            ttc = self._ttc_lin(px, py, vx, vy, 0.0, 0.0, rad_sum)
            if ttc < ttc_min:
                ttc_min = ttc
        return ttc_min

    def predict(self, state):
        """
        用 ORCA 给机器人算一帧速度标签。只输出机器人动作，不改真实环境的人群。
        """
        self_state = state.self_state
        humans = state.human_states
        n_agents = 1 + len(humans)

        params = (self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst)

        # —— 构建/复用 ORCA 模拟器
        if self._need_rebuild(n_agents):
            if self.sim is not None:
                del self.sim
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self._sim_timestep = float(self.time_step)

            # 0号：机器人（半径叠加 safety + label_inflate，更保守）
            self.sim.addAgent(self_state.position, *params,
                              self_state.radius + 0.01 + self.safety_space + self.label_inflate,
                              max(self_state.v_pref, 1e-6), self_state.velocity)
            # 1..N：行人（各自 v_pref 作为 maxSpeed）
            for h in humans:
                h_max = getattr(h, 'v_pref', self.max_speed)
                self.sim.addAgent(h.position, *params,
                                  h.radius + 0.01 + self.safety_space,
                                  max(h_max, 1e-6), h.velocity)
        else:
            # 同步位置与速度
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, h in enumerate(humans):
                self.sim.setAgentPosition(i + 1, h.position)
                self.sim.setAgentVelocity(i + 1, h.velocity)

        # —— 1) 机器人目标向首选速度
        pref_vel_robot = np.asarray(self._goal_pref_vel(
            self_state.px, self_state.py, self_state.gx, self_state.gy,
            max(self_state.v_pref, 1e-6), self_state.radius
        ), dtype=float)

        # —— 2) TTC 前馈减速（不重建 ORCA，只缩放幅值）
        if len(humans) > 0 and self.ttc_brake > self.ttc_eps:
            ttc_min = self._ttc_min_to_humans(self_state, humans)
            if np.isfinite(ttc_min) and ttc_min < self.ttc_brake:
                ratio = float(np.clip(ttc_min / max(self.ttc_brake, self.ttc_eps),
                                      self.brake_min_ratio, 1.0))
                pref_vel_robot *= ratio

        # —— 3) 破对称（小抖动）
        if self.eps_noise > 0.0:
            ang = np.random.uniform(0, 2 * np.pi)
            jitter = self.eps_noise * np.array([np.cos(ang), np.sin(ang)], dtype=float)
            pref_vel_robot = pref_vel_robot + jitter

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel_robot.tolist()))

        # —— 行人首选速度
        for i, h in enumerate(humans):
            if self.human_pref_mode == 'goal' and hasattr(h, 'gx') and hasattr(h, 'gy'):
                hv = self._goal_pref_vel(h.px, h.py, h.gx, h.gy,
                                         max(getattr(h, 'v_pref', self.max_speed), 1e-6), h.radius)
            elif self.human_pref_mode == 'current':
                hv = tuple(h.velocity)
            else:
                hv = (0.0, 0.0)
            self.sim.setAgentPrefVelocity(i + 1, hv)

        # —— 单步 ORCA
        self.sim.doStep()
        vx, vy = self.sim.getAgentVelocity(0)
        return ActionXY(float(vx), float(vy))

