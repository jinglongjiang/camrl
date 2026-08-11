import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        """
        ORCA (Optimal Reciprocal Collision Avoidance)

        SARL原版实现，配置参数可通过configure()设置
        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0.0
        self.neighbor_dist = 10.0
        self.max_neighbors = 10
        self.time_horizon = 5.0
        self.time_horizon_obst = 5.0
        self.radius = 0.3
        self.max_speed = 1.0
        self.time_step = 0.25
        self.sim = None

        # Optimization #3: smoother TTC braking (enabled by default)
        self.ttc_brake = True
        self.ttc_threshold = 2.0
        self.ttc_min_scale = 0.0
        self.ttc_smoothing = 0.5
        self._last_pref_vel = None

    def configure(self, config):
        """从config读取ORCA参数（Mamba增强：支持动态配置）"""
        if not hasattr(config, 'has_option'):
            return

        # time_step可以从env或action_space读取
        if config.has_option('env', 'time_step'):
            self.time_step = config.getfloat('env', 'time_step')
        elif config.has_option('action_space', 'time_step'):
            self.time_step = config.getfloat('action_space', 'time_step')

        # ORCA参数
        if config.has_option('orca', 'neighbor_dist'):
            self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        if config.has_option('orca', 'max_neighbors'):
            self.max_neighbors = config.getint('orca', 'max_neighbors')
        if config.has_option('orca', 'time_horizon'):
            self.time_horizon = config.getfloat('orca', 'time_horizon')
        if config.has_option('orca', 'time_horizon_obst'):
            self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        if config.has_option('orca', 'radius'):
            self.radius = config.getfloat('orca', 'radius')
        if config.has_option('orca', 'max_speed'):
            self.max_speed = config.getfloat('orca', 'max_speed')
        if config.has_option('orca', 'safety_space'):
            self.safety_space = config.getfloat('orca', 'safety_space')

        # TTC braking options
        if config.has_option('orca', 'ttc_brake'):
            self.ttc_brake = config.getboolean('orca', 'ttc_brake')
        if config.has_option('orca', 'ttc_threshold'):
            self.ttc_threshold = config.getfloat('orca', 'ttc_threshold')
        if config.has_option('orca', 'ttc_min_scale'):
            self.ttc_min_scale = config.getfloat('orca', 'ttc_min_scale')
        if config.has_option('orca', 'ttc_smoothing'):
            self.ttc_smoothing = config.getfloat('orca', 'ttc_smoothing')

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        SARL原版ORCA实现

        Create a rvo2 simulation at each time step and run one step.
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx

        关键假设：
        - 机器人知道自己的目标位置
        - 人类的目标位置未知，假设preferred velocity为(0,0)
        """
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst

        # 重建simulator（如果agent数量变化）
        if self.sim is not None:
            if self.sim.getNumAgents() != len(state.human_states) + 1:
                del self.sim
                self.sim = None

        if self.sim is None:
            # 首次创建simulator
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            # addAgent(position, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity)
            self.sim.addAgent(self_state.position, *params,
                            self_state.radius + 0.01 + self.safety_space,
                            self_state.v_pref,  # maxSpeed使用机器人的v_pref
                            self_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params,
                                human_state.radius + 0.01 + self.safety_space,
                                self.max_speed,  # 人类maxSpeed使用默认值
                                human_state.velocity)
        else:
            # 更新agent位置和速度
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # 设置机器人的preferred velocity（朝向目标）
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        if self.ttc_brake and len(state.human_states) > 0:
            pref_vel = self._apply_ttc_brake(pref_vel, state)

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        # 设置人类的preferred velocity（SARL原版：假设为(0,0)，因为不知道目标）
        for i, human_state in enumerate(state.human_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        # 执行一步ORCA仿真
        self.sim.doStep()

        # 获取机器人的计算速度
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action

    def _apply_ttc_brake(self, pref_vel, state):
        min_ttc = self._compute_min_ttc(state)
        if min_ttc is None or min_ttc >= self.ttc_threshold:
            target = pref_vel
        else:
            ratio = np.clip(min_ttc / max(self.ttc_threshold, 1e-6), 0.0, 1.0)
            ratio = max(ratio, self.ttc_min_scale)
            target = pref_vel * ratio

        if self._last_pref_vel is None:
            smoothed = target
        else:
            alpha = np.clip(self.ttc_smoothing, 0.0, 1.0)
            smoothed = alpha * target + (1.0 - alpha) * self._last_pref_vel

        self._last_pref_vel = smoothed
        return smoothed

    def _compute_min_ttc(self, state):
        self_state = state.self_state
        min_ttc = None
        for human_state in state.human_states:
            rel_px = human_state.px - self_state.px
            rel_py = human_state.py - self_state.py
            rel_vx = human_state.vx - self_state.vx
            rel_vy = human_state.vy - self_state.vy

            a = rel_vx * rel_vx + rel_vy * rel_vy
            if a < 1e-8:
                continue

            combined_radius = self_state.radius + human_state.radius + self.safety_space
            b = 2.0 * (rel_px * rel_vx + rel_py * rel_vy)
            c = rel_px * rel_px + rel_py * rel_py - combined_radius * combined_radius
            disc = b * b - 4.0 * a * c
            if disc <= 0.0:
                continue

            sqrt_disc = float(np.sqrt(disc))
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            if t2 < 0.0:
                continue
            ttc = t1 if t1 > 0.0 else t2

            if min_ttc is None or ttc < min_ttc:
                min_ttc = ttc

        return min_ttc
