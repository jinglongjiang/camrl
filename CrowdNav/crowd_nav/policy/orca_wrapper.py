# crowd_nav/policy/orca_wrapper.py

from crowd_sim.envs.policy.orca import ORCA as RawORCA

class ORCA_WRAPPER(RawORCA):
    def __init__(self, config=None):
        super().__init__()
        self.multiagent_training = False
        self.trainable = False

    def configure(self, config):
        """Forward configuration to parent ORCA class (修复空方法bug)"""
        super().configure(config)

    def set_device(self, device):
        pass
