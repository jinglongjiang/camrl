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
            # Debug print disabled
            Robot._obs_shape_printed = True
        return arr

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        # --- 判断policy类型 ---
        # 优先读取policy自己声明的expects_joint_state（guide.md A0.8/A10），
        # 只有policy没有这个属性时才回退到旧的类名字符串猜测。
        explicit = getattr(self.policy, 'expects_joint_state', None)
        if explicit is not None:
            needs_joint_state = bool(explicit)
        else:
            policy_name = self.policy.__class__.__name__.lower()
            # MambaRL和一些其他RL策略需要JointState格式
            needs_joint_state = any(key in policy_name for key in ['orca', 'cadrl', 'sarl', 'multi_human_rl', 'mamba'])
        
        if needs_joint_state:
            # 这些策略需要JointState格式
            # 需要从环境获取人类的ObservableState
            if hasattr(self, 'env') and hasattr(self.env, 'humans'):
                human_states = [h.get_observable_state() for h in self.env.humans]
            else:
                # 如果无法获取环境，尝试从ob构建
                from crowd_sim.envs.utils.state import ObservableState
                if not isinstance(ob, (list, tuple)):
                    ob_list = [ob]
                else:
                    ob_list = ob
                human_states = []
                for h in ob_list:
                    if isinstance(h, ObservableState):
                        human_states.append(h)
                    elif hasattr(h, 'to_observable_state'):
                        human_states.append(h.to_observable_state())
                    else:
                        # 尝试从数组构建ObservableState
                        try:
                            h_arr = np.array(h, dtype=np.float32) if not isinstance(h, np.ndarray) else h
                            if len(h_arr) >= 5:
                                human_states.append(ObservableState(h_arr[0], h_arr[1], h_arr[2], h_arr[3], h_arr[4]))
                        except:
                            pass
            state = JointState(self.get_full_state(), human_states)
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
            # Debug print disabled
            Human._obs_shape_printed = True
        return arr
