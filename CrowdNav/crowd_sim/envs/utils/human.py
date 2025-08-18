from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import numpy as np


class Human(Agent):
    _obs_shape_printed = False  # 类变量，全局只打印一次

    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def get_obs_array(self):
        arr = np.array([
            self.px, self.py, self.vx, self.vy, self.radius
        ], dtype=np.float32)
        if not Human._obs_shape_printed:
            print('[DEBUG] Human get_obs_array shape:', arr.shape)
            Human._obs_shape_printed = True
        return arr