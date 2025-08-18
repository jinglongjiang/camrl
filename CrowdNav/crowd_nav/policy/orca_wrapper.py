# crowd_nav/policy/orca_wrapper.py

from crowd_sim.envs.policy.orca import ORCA as RawORCA

class ORCA_WRAPPER(RawORCA):
    def __init__(self, config=None):
        super().__init__()  # 不用 config，兼容CrowdNav风格
        self.multiagent_training = False
        self.trainable = False   # 关键点：CrowdNav靠它区分IL用法

    def configure(self, config):  # 空实现，兼容所有policy调用
        pass

    def set_device(self, device):
        pass
