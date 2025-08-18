import numpy as np

class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])

    def to_array(self):
        return np.array([
            self.px, self.py, self.vx, self.vy, self.radius,
            self.gx, self.gy, self.v_pref, self.theta
        ], dtype=np.float32)

class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

    def to_array(self):
        return np.array([self.px, self.py, self.vx, self.vy, self.radius], dtype=np.float32)

class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)
        self.self_state = self_state
        self.human_states = human_states

    def to_array(self):
        """
        展平为 1‑D np.ndarray[float32]：[robot_self_state | human1_state | human2_state | … ]
        """
        self_vec = self.self_state.to_array() if hasattr(self.self_state, 'to_array') else np.asarray(self.self_state, dtype=np.float32)
        human_vecs = [
            h.to_array() if hasattr(h, 'to_array')
            else np.asarray(h, dtype=np.float32)
            for h in self.human_states
        ]
        return np.concatenate([self_vec] + human_vecs).astype(np.float32)

    @classmethod
    def from_array(cls, arr):
        """
        arr: 展平后的 np.ndarray/list/torch.tensor, 形如 [robot(9), human1(5), human2(5), ...]
        """
        arr = np.array(arr, dtype=np.float32).flatten()
        robot_dim = 9
        human_dim = 5
        assert len(arr) > robot_dim and (len(arr) - robot_dim) % human_dim == 0, \
            f"obs长度异常，不能分割为JointState: {arr.shape}"
        n_human = (len(arr) - robot_dim) // human_dim

        # 解析robot
        robot_obs = arr[:robot_dim]
        robot = FullState(*robot_obs.tolist())
        # 解析human
        human_states = []
        for i in range(n_human):
            s = arr[robot_dim + i*human_dim:robot_dim + (i+1)*human_dim]
            human_states.append(ObservableState(*s.tolist()))
        return cls(robot, human_states)
