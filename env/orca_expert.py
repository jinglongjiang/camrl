import numpy as np
import rvo2

class ORCAExpert:
    def __init__(
        self, 
        radius=0.3, 
        max_speed=1.0, 
        dt=0.1,
        neighbor_dist=10,
        max_neighbors=10,
        time_horizon=5,
        time_horizon_obst=5,
        ped_radius=0.3,
        ped_max_speed=0.6,
    ):
        self.radius = radius
        self.max_speed = max_speed
        self.dt = dt
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.ped_radius = ped_radius
        self.ped_max_speed = ped_max_speed

    def get_action(self, robot_pos, robot_vel, goal, human_states):
        """
        robot_pos: [2], robot_vel: [2], goal: [2]
        human_states: list of (pos[2], vel[2])
        Return: [2] action velocity
        """
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        robot_vel = np.asarray(robot_vel, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        sim = rvo2.PyRVOSimulator(
            self.dt, 
            self.neighbor_dist, 
            self.max_neighbors,
            self.time_horizon, 
            self.time_horizon_obst, 
            self.radius, 
            self.max_speed
        )
        rid = sim.addAgent(
            tuple(robot_pos), self.neighbor_dist, self.max_neighbors,
            self.time_horizon, self.time_horizon_obst,
            self.radius, self.max_speed, velocity=tuple(robot_vel)
        )
        # 行人agent
        for h_pos, h_vel in human_states:
            sim.addAgent(
                tuple(h_pos), self.neighbor_dist, self.max_neighbors,
                self.time_horizon, self.time_horizon_obst,
                self.ped_radius, self.ped_max_speed, velocity=tuple(h_vel)
            )
        # 机器人目标速度
        v_goal = goal - robot_pos
        speed = np.linalg.norm(v_goal)
        if speed > 1e-6:
            pref_vel = v_goal / speed * min(speed, self.max_speed)
        else:
            pref_vel = np.zeros(2, dtype=np.float32)
        sim.setAgentPrefVelocity(rid, tuple(pref_vel))

        # 行人pref vel设为0（可根据数据增强设为其他）
        for i in range(len(human_states)):
            sim.setAgentPrefVelocity(i + 1, (0, 0))

        sim.doStep()
        act = np.array(sim.getAgentVelocity(rid), dtype=np.float32)
        if np.any(np.isnan(act)) or np.any(np.isinf(act)):
            act = np.zeros(2, dtype=np.float32)
        return act

