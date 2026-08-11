import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState

class Agent(object):
    def __init__(self, config, section):
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        # 从action_space读取kinematics，而不是从policy读取
        self.kinematics = config.get('action_space', 'kinematics', fallback='holonomic')
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None
        # 局部RNG：由env在reset()时设置，避免全局np.random污染
        self.rng = None

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        # 保持从配置文件读取的kinematics设置，不从policy覆盖

    def sample_random_attributes(self):
        """
        随机化agent属性（v_pref, radius）
        使用局部RNG确保可复现性，避免全局np.random污染
        """
        if self.rng is None:
            # Fallback：如果env未设置rng，使用全局（向后兼容）
            # 🔥 降级日志：WARNING→DEBUG（避免刷屏，每个episode都会randomize）
            logging.debug("[AGENT] rng not set, using global np.random (not reproducible)")
            self.v_pref = np.random.uniform(0.5, 1.5)
            self.radius = np.random.uniform(0.3, 0.5)
        else:
            # 正常路径：使用局部RNG
            self.v_pref = self.rng.uniform(0.5, 1.5)
            self.radius = self.rng.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

        # 强化参数验证 - 防止"喂零"问题
        if radius is not None:
            if radius > 0.05:  # 确保radius有意义
                self.radius = radius
            else:
                logging.warning(f"[AGENT-SET] 忽略无效radius={radius}，保持原值{self.radius}")

        if v_pref is not None:
            if v_pref > 0.1:  # 确保v_pref有意义
                self.v_pref = v_pref
            else:
                logging.warning(f"[AGENT-SET] 忽略无效v_pref={v_pref}，保持原值{self.v_pref}")

    # ====== Policy/Replay等用的接口 ======
    def get_observable_state(self):
        # 返回 ObservableState 对象（原始逻辑，不能改成np.array）
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        state = FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
        # 数据污染追踪：在环境层记录状态
        import logging
        if hasattr(self, '__class__') and 'Robot' in self.__class__.__name__:
            # 仅对Robot添加追踪，避免刷屏
            global _robot_state_track_count
            if '_robot_state_track_count' not in globals():
                _robot_state_track_count = 0
            _robot_state_track_count += 1
            if _robot_state_track_count <= 3:  # 只记录前3次，减少刷屏
                logging.info(f"[ENV-TRACK] Robot get_full_state #{_robot_state_track_count}: "
                           f"radius={self.radius:.3f}, v_pref={self.v_pref:.3f}, "
                           f"gx={self.gx:.3f}, gy={self.gy:.3f}")
        return state

    # ====== Gym obs拼接专用的接口 ======
    def get_obs_array(self):
        # crowdnav标准：px, py, vx, vy, radius
        return np.array([self.px, self.py, self.vx, self.vy, self.radius], dtype=np.float32)

    def get_next_obs_array(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return np.array([next_px, next_py, next_vx, next_vy, self.radius], dtype=np.float32)

    # ====== 通用 ======
    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY), f"Holonomic需要ActionXY，但得到{type(action)}"
        else:
            assert isinstance(action, ActionRot), f"Unicycle需要ActionRot，但得到{type(action)}"

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t
        return px, py

    def step(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius
