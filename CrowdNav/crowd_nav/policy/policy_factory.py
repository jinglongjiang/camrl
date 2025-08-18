from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.mamba_rl import MambaRL
# 新增
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.orca_wrapper import ORCA_WRAPPER


policy_factory = dict()
policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['mamba'] = MambaRL
policy_factory['orca'] = ORCA      # <--- 加这一句
policy_factory['orca'] = ORCA_WRAPPER