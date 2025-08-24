try:
    from gymnasium.envs.registration import register
except ImportError:
    from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)
