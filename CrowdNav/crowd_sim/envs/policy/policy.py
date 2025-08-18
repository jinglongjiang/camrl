import abc
import numpy as np

class Policy(object):
    def __init__(self):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        pass

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action
        """
        pass

    def transform(self, state):
        """
        通用状态转array方法
        - 如果 state 有 to_array（如 JointState），转为 array
        - 否则直接返回（通常已经是 array 或 tensor）
        """
        if hasattr(state, "to_array"):
            return state.to_array()
        return state

    @staticmethod
    def reach_destination(state):
        # 注意 self_state 坐标顺序 (py, px, gy, gx) 的兼容性
        self_state = state.self_state if hasattr(state, "self_state") else None
        if self_state is not None:
            if np.linalg.norm([self_state.px - self_state.gx, self_state.py - self_state.gy]) < self_state.radius:
                return True
        return False
