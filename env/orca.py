import numpy as np
import rvo2
import random, math
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class ORCA(Policy):
    def __init__(self,
                 time_step=0.1,
                 neighbor_dist=8.0,
                 max_neighbors=10,
                 time_horizon=5.0,
                 time_horizon_obst=5.0,
                 radius=0.3,
                 max_speed=1.5,
                 safety_space=0.0,
                 random_perturb=True,
                 perturb_strength=0.03,
                 behavior_mode="normal",  # "normal" / "aggressive" / "cautious"
                 speed_up_gain=1.0,
                 dynamic_neighbor=False,
                 all_agents_active=True):
        super().__init__()
        self.name = "ORCA"
        self.kinematics = "holonomic"
        self.trainable = False

        # 参数
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed
        self.safety_space = safety_space

        # 行为模式
        self.random_perturb = random_perturb
        self.perturb_strength = perturb_strength
        self.behavior_mode = behavior_mode
        self.speed_up_gain = speed_up_gain
        self.dynamic_neighbor = dynamic_neighbor
        self.all_agents_active = all_agents_active  # human 也走自己的目标

        self.sim = None

        # 模式参数调整
        if behavior_mode == "aggressive":
            self.neighbor_dist *= 0.7
            self.time_horizon *= 0.6
            self.speed_up_gain *= 1.2
        elif behavior_mode == "cautious":
            self.neighbor_dist *= 1.3
            self.time_horizon *= 1.4
            self.speed_up_gain *= 0.9

    def configure(self, *_): ...
    def set_phase(self, *_): ...

    def predict(self, state):
        s = state.self_state
        humans = state.human_states

        nb = self.neighbor_dist
        if self.dynamic_neighbor and humans:
            d_min = min(np.linalg.norm([h.px - s.px, h.py - s.py]) for h in humans)
            nb = max(self.radius*2, d_min * 1.5)

        params = (nb, self.max_neighbors, self.time_horizon, self.time_horizon_obst)

        if self.sim is None or self.sim.getNumAgents() != len(humans) + 1:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self._add_agents(s, humans, params)
        else:
            self._update_agents(s, humans, params)

        # Robot 设置 prefVel
        self.sim.setAgentPrefVelocity(0, tuple(self._toward_goal(s, self.speed_up_gain)))
        # Human 设置 prefVel
        for i, h in enumerate(humans):
            if self.all_agents_active:
                # human 也有自己的目标点，走目标，注意human自己的v_pref
                self.sim.setAgentPrefVelocity(i+1, tuple(self._toward_goal(h, 1.0)))
            else:
                # 如果你想让human不动，写(0,0)
                self.sim.setAgentPrefVelocity(i+1, (0, 0))

        self.sim.doStep()
        vx, vy = self.sim.getAgentVelocity(0)
        return ActionXY(vx, vy)

    def _toward_goal(self, agent_state, speed_gain=1.0):
        vec = np.array([agent_state.gx - agent_state.px, agent_state.gy - agent_state.py])
        dist = np.linalg.norm(vec)
        if dist < 1e-3:
            base = np.zeros(2)
        else:
            base = vec / dist * agent_state.v_pref * speed_gain
        # 随机扰动避免对称死锁
        if self.random_perturb and dist > 0.5:
            ang = random.random() * 2 * math.pi
            base += np.array([math.cos(ang), math.sin(ang)]) * self.perturb_strength
        # 限速
        speed = np.linalg.norm(base)
        if speed > self.max_speed:
            base = base / speed * self.max_speed
        return base

    def _add_agents(self, s, humans, params):
        self.sim.addAgent(
            s.position, *params,
            s.radius + 0.01 + self.safety_space,
            s.v_pref, s.velocity)
        for h in humans:
            self.sim.addAgent(
                h.position, *params,
                h.radius + 0.01 + self.safety_space,
                self.max_speed, h.velocity)

    def _update_agents(self, s, humans, params):
        self.sim.setAgentPosition(0, s.position)
        self.sim.setAgentVelocity(0, s.velocity)
        for i, h in enumerate(humans):
            self.sim.setAgentPosition(i+1, h.position)
            self.sim.setAgentVelocity(i+1, h.velocity)
            if self.dynamic_neighbor:
                self.sim.setAgentNeighborDist(i+1, params[0])
