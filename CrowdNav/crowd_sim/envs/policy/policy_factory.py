# from crowd_sim.envs.policy.linear import Linear
# from crowd_sim.envs.policy.orca import ORCA
# from crowd_nav.policy.cadrl import CADRL


# def none_policy():
#     return None


# policy_factory = dict()
# policy_factory['linear'] = Linear
# policy_factory['orca'] = ORCA
# policy_factory['none'] = none_policy
# policy_factory['cadrl'] = CADRL

# Env-side policy factory: ONLY environment-local policies.
# Do NOT import anything from crowd_nav.* here, or you'll create circular imports.

from .policy import Policy
from .linear import Linear
from .orca import ORCA


class NonePolicy(Policy):
    """
    安全占位策略（用于 [robot] policy = none）。
    机器人真实 RL 策略会在外部 robot.set_policy(...) 注入。
    这里提供必要属性以避免初始化时报错。
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        # 给 Agent.__init__ 用到的字段
        self.kinematics = 'holonomic'
        self.time_step = 0.25  # 任意占位值，env 会在 reset 时覆盖到各个 agent 上
        # 兼容常见接口
        self.device = None
        self.env = None
        self.phase = 'test'

    # 兼容接口（无实际行为）
    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def set_phase(self, phase):
        self.phase = phase

    def predict(self, state):
        # 原地不动（ActionXY 是环境里的二维速度动作用）
        try:
            from crowd_sim.envs.utils.action import ActionXY
            return ActionXY(0.0, 0.0)
        except Exception:
            return None


# 仅暴露环境侧策略（ORCA/Linear/None），不要把 crowd_nav 下的 RL 策略放进来
policy_factory = {
    'linear': Linear,
    'orca': ORCA,
    'none': NonePolicy,
}
