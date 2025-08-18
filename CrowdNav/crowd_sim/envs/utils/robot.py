from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState  # 加上JointState
import numpy as np

class Robot(Agent):
    _obs_shape_printed = False  # 类变量

    def __init__(self, config, section):
        super().__init__(config, section)

    def get_obs_array(self):
        arr = np.array([
            self.px, self.py, self.gx, self.gy, self.vx, self.vy, self.radius, self.v_pref, self.theta
        ], dtype=np.float32)
        if not Robot._obs_shape_printed:
            print('[DEBUG] Robot get_obs_array shape:', arr.shape)
            Robot._obs_shape_printed = True
        return arr

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        # --- 判断policy类型 ---
        if self.policy.__class__.__name__.lower() == "orca":
            # ORCA policy 依然用 JointState
            # ob 应该是 [ObservableState,...]
            from crowd_sim.envs.utils.state import FullState
            if not isinstance(ob, (list, tuple)):
                # 一般不这样，保险处理下
                ob_list = ob.tolist() if hasattr(ob, 'tolist') else [ob]
            else:
                ob_list = ob
            state = JointState(self.get_full_state(), ob_list)
            action = self.policy.predict(state)
        else:
            # 其它 policy 用 float32 array
            if not isinstance(ob, (list, tuple)):
                obs_array = ob  # already array
            else:
                obs_array = np.concatenate(
                    [h.to_array() if hasattr(h, 'to_array') else np.array(h, dtype=np.float32) for h in ob]
                ).astype(np.float32)
            action = self.policy.predict(obs_array)
        return action

class Human(Agent):
    _obs_shape_printed = False  # 类变量

    def get_obs_array(self):
        arr = np.array([
            self.px, self.py, self.vx, self.vy, self.radius
        ], dtype=np.float32)
        if not Human._obs_shape_printed:
            print('[DEBUG] Human get_obs_array shape:', arr.shape)
            Human._obs_shape_printed = True
        return arr
