import gym
from gym import spaces
import numpy as np
import rvo2
import logging
from collections import deque

logging.basicConfig(
    filename='crowd_env_debug.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


# ------------------------------------------------------------
# CrowdNavEnv  — 含 robot / humans 包装器（可读写 px,py,vx,vy,gx,gy）
# ------------------------------------------------------------
class CrowdNavEnv(gym.Env):
    # ============ 初始化 ============
    def __init__(self, config):
        super().__init__()
        self.T = config['T']
        self.n_peds = config.get('n_peds', 5)
        self.obs_dim = 6 + self.n_peds * 4
        self.action_dim = config.get('action_dim', 2)
        self.max_steps = config.get('max_steps', 500)
        self.scene_size = config.get('scene_size', 10)
        self.robot_radius = config.get('robot_radius', 0.3)
        self.ped_radius = config.get('ped_radius', 0.3)
        self.dt = config.get('dt', 0.1)
        self.robot_max_speed = config.get('robot_max_speed', 1.0)
        self.ped_max_speed = config.get('ped_max_speed', 0.6)
        self.goal_radius = config.get('goal_radius', 0.5)

        reward_cfg = config.get('reward', {})
        self.success_reward = reward_cfg.get('success_reward', 1.0)
        self.collision_penalty = reward_cfg.get('collision_penalty', -0.25)
        self.discomfort_dist = reward_cfg.get('discomfort_dist', self.robot_radius + 0.2)
        self.discomfort_penalty_factor = reward_cfg.get('discomfort_penalty_factor', 0.1)
        self.step_penalty = reward_cfg.get('step_penalty', -0.01)
        self.dist_reward_factor = reward_cfg.get('dist_reward_factor', 0.05)

        self.time_limit = self.max_steps

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.T + 1, self.obs_dim),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0] * self.action_dim, dtype=np.float32),
            high=np.array([1.0] * self.action_dim, dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0
        self.traj_buffer = deque(maxlen=self.T + 1)
        self.sim = None
        self.agent_ids = []
        self.ped_goals = None
        self.robot_goal = None
        self.prev_dist_to_goal = None
        
        # 优化：预分配内存
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_vel = np.zeros(2, dtype=np.float32)
        self.ped_positions = np.zeros((self.n_peds, 2), dtype=np.float32)
        self.ped_velocities = np.zeros((self.n_peds, 2), dtype=np.float32)
        
        # 优化：缓存观察空间
        self.obs_cache = np.zeros((self.T + 1, self.obs_dim), dtype=np.float32)
        
        # 优化：环境配置缓存
        self._setup_simulator()

    def _setup_simulator(self):
        """预设置仿真器配置"""
        self.sim_config = {
            'timeStep': self.dt,
            'neighborDist': 2.0,
            'maxNeighbors': 8,
            'timeHorizon': 1.0,
            'timeHorizonObst': 1.0,
            'radius': self.robot_radius,
            'maxSpeed': self.robot_max_speed,
        }

    # ============ reset ============
    def reset(self):
        while True:
            self.current_step = 0
            self.traj_buffer.clear()
            self.agent_ids = []

            self.sim = rvo2.PyRVOSimulator(**self.sim_config)

            # 优化：使用预分配的内存
            theta = np.random.uniform(0, 2 * np.pi)
            r = self.scene_size / 2 * 0.85
            self.robot_pos[0] = r * np.cos(theta)
            self.robot_pos[1] = r * np.sin(theta)
            self.robot_goal = -self.robot_pos.copy()
            robot_id = self.sim.addAgent(tuple(self.robot_pos))
            self.agent_ids.append(robot_id)

            self.ped_goals = []
            for i in range(self.n_peds):
                t = 2 * np.pi * i / self.n_peds
                self.ped_positions[i, 0] = r * np.cos(t)
                self.ped_positions[i, 1] = r * np.sin(t)
                ped_goal = -self.ped_positions[i].copy()
                ped_id = self.sim.addAgent(tuple(self.ped_positions[i]))
                self.agent_ids.append(ped_id)
                self.ped_goals.append(ped_goal)
            self.ped_goals = np.array(self.ped_goals)

            self.prev_dist_to_goal = np.linalg.norm(self.robot_pos - self.robot_goal)
            if np.linalg.norm(self.robot_pos - self.robot_goal) > self.goal_radius * 2:
                break

        # 优化：预填充轨迹buffer
        for _ in range(self.T + 1):
            obs = self._get_observation()
            self.traj_buffer.append(obs)
            
        return np.array(list(self.traj_buffer), dtype=np.float32)

    def _get_observation(self):
        """优化：高效的观察获取"""
        # 优化：直接使用缓存数组
        obs = self.obs_cache[0]  # 重用缓存
        
        # Robot state (px, py, vx, vy)
        obs[0:2] = self.robot_pos
        obs[2:4] = self.robot_vel
        
        # Goal state (gx, gy)
        obs[4:6] = self.robot_goal
        
        # Pedestrian states
        for i in range(self.n_peds):
            start_idx = 6 + i * 4
            obs[start_idx:start_idx+2] = self.ped_positions[i]
            obs[start_idx+2:start_idx+4] = self.ped_velocities[i]
            
        return obs.copy()

    def _compute_reward(self, action):
        """优化：更高效的奖励计算"""
        # 获取当前状态
        robot_pos = self.sim.getAgentPosition(self.agent_ids[0])
        dist_to_goal = np.linalg.norm(robot_pos - self.robot_goal)
        
        # 基础奖励：距离目标更近
        dist_reward = (self.prev_dist_to_goal - dist_to_goal) * self.dist_reward_factor
        
        # 进度奖励：基于距离目标的相对进度
        progress_reward = 0
        if hasattr(self, 'initial_dist_to_goal'):
            progress = (self.initial_dist_to_goal - dist_to_goal) / self.initial_dist_to_goal
            progress_reward = progress * self.reward.get('progress_reward_factor', 0.1)
        else:
            self.initial_dist_to_goal = dist_to_goal
        
        # 碰撞检测和安全奖励
        collision_penalty = 0
        min_dist_to_ped = float('inf')
        safety_reward = 0
        
        for i in range(1, len(self.agent_ids)):
            ped_pos = self.sim.getAgentPosition(self.agent_ids[i])
            dist_to_ped = np.linalg.norm(robot_pos - ped_pos)
            min_dist_to_ped = min(min_dist_to_ped, dist_to_ped)
            
            if dist_to_ped < self.robot_radius + self.ped_radius:
                collision_penalty = self.collision_penalty
                break
            elif dist_to_ped < self.discomfort_dist:
                # 渐进式不适惩罚
                discomfort_factor = (self.discomfort_dist - dist_to_ped) / self.discomfort_dist
                collision_penalty += self.discomfort_penalty_factor * discomfort_factor ** 2
            elif dist_to_ped < self.discomfort_dist + self.reward.get('safety_margin', 0.2):
                # 安全奖励：鼓励保持安全距离
                safety_factor = (dist_to_ped - self.discomfort_dist) / self.reward.get('safety_margin', 0.2)
                safety_reward += 0.1 * safety_factor
        
        # 成功奖励
        success_reward = 0
        if dist_to_goal < self.goal_radius:
            success_reward = self.success_reward
            
        # 动作平滑度惩罚
        action_penalty = -0.01 * np.linalg.norm(action)
        
        # 方向奖励：鼓励朝向目标
        robot_vel = np.array(self.sim.getAgentVelocity(self.agent_ids[0]))
        if np.linalg.norm(robot_vel) > 0.1:
            goal_direction = (self.robot_goal - robot_pos) / np.linalg.norm(self.robot_goal - robot_pos)
            vel_direction = robot_vel / np.linalg.norm(robot_vel)
            alignment = np.dot(goal_direction, vel_direction)
            direction_reward = 0.05 * max(0, alignment)  # 只奖励正对齐
        else:
            direction_reward = 0
        
        total_reward = (dist_reward + progress_reward + collision_penalty + 
                       success_reward + action_penalty + safety_reward + 
                       direction_reward + self.step_penalty)
        
        self.prev_dist_to_goal = dist_to_goal
        return total_reward

    def step(self, action):
        # 设置机器人目标速度
        action = np.clip(action, -1.0, 1.0)
        target_vel = action * self.robot_max_speed
        self.sim.setAgentPrefVelocity(self.agent_ids[0], tuple(target_vel))
        
        # 设置行人目标速度
        for i in range(self.n_peds):
            ped_id = self.agent_ids[i + 1]
            ped_pos = self.sim.getAgentPosition(ped_id)
            ped_goal = self.ped_goals[i]
            direction = ped_goal - ped_pos
            if np.linalg.norm(direction) > 0.1:
                direction = direction / np.linalg.norm(direction) * self.ped_max_speed
            else:
                direction = np.zeros(2)
            self.sim.setAgentPrefVelocity(ped_id, tuple(direction))
        
        # 仿真一步
        self.sim.doStep()
        
        # 更新状态
        self.robot_pos = np.array(self.sim.getAgentPosition(self.agent_ids[0]))
        self.robot_vel = np.array(self.sim.getAgentVelocity(self.agent_ids[0]))
        
        for i in range(self.n_peds):
            ped_id = self.agent_ids[i + 1]
            self.ped_positions[i] = np.array(self.sim.getAgentPosition(ped_id))
            self.ped_velocities[i] = np.array(self.sim.getAgentVelocity(ped_id))
        
        # 计算奖励
        reward = self._compute_reward(action)
        
        # 检查终止条件
        done = False
        info = {}
        
        dist_to_goal = np.linalg.norm(self.robot_pos - self.robot_goal)
        if dist_to_goal < self.goal_radius:
            done = True
            info['event'] = 'reach_goal'
        elif self.current_step >= self.max_steps:
            done = True
            info['event'] = 'timeout'
        else:
            # 检查碰撞
            for i in range(1, len(self.agent_ids)):
                ped_pos = self.sim.getAgentPosition(self.agent_ids[i])
                if np.linalg.norm(self.robot_pos - ped_pos) < self.robot_radius + self.ped_radius:
                    done = True
                    info['event'] = 'collision'
                    break
        
        # 更新轨迹buffer
        obs = self._get_observation()
        self.traj_buffer.append(obs)
        
        self.current_step += 1
        
        return np.array(list(self.traj_buffer), dtype=np.float32), reward, done, info

    # ------------------------------------------------------------------
    # ============ Agent Wrapper (支持读写 px/py/vx/vy/gx/gy) ===========
    # ------------------------------------------------------------------
    class _AgentWrapper:
        def __init__(self, env, idx, is_robot=False):
            self._env = env
            self._idx = idx           # 对应 env.agent_ids 中的索引
            self.is_robot = is_robot
            self.is_human = not is_robot

        # ---- 位置 px / py ----
        @property
        def px(self):
            return self._env.sim.getAgentPosition(self._env.agent_ids[self._idx])[0]

        @px.setter
        def px(self, v):
            self._env.sim.setAgentPosition(self._env.agent_ids[self._idx], (v, self.py))

        @property
        def py(self):
            return self._env.sim.getAgentPosition(self._env.agent_ids[self._idx])[1]

        @py.setter
        def py(self, v):
            self._env.sim.setAgentPosition(self._env.agent_ids[self._idx], (self.px, v))

        # ---- 速度 vx / vy ----
        @property
        def vx(self):
            return self._env.sim.getAgentVelocity(self._env.agent_ids[self._idx])[0]

        @vx.setter
        def vx(self, v):
            self._env.sim.setAgentVelocity(self._env.agent_ids[self._idx], (v, self.vy))

        @property
        def vy(self):
            return self._env.sim.getAgentVelocity(self._env.agent_ids[self._idx])[1]

        @vy.setter
        def vy(self, v):
            self._env.sim.setAgentVelocity(self._env.agent_ids[self._idx], (self.vx, v))

        # ---- 目标 gx / gy ----
        @property
        def gx(self):
            return self._env.robot_goal[0] if self.is_robot else self._env.ped_goals[self._idx - 1][0]

        @gx.setter
        def gx(self, v):
            if self.is_robot:
                self._env.robot_goal[0] = v
            else:
                self._env.ped_goals[self._idx - 1][0] = v

        @property
        def gy(self):
            return self._env.robot_goal[1] if self.is_robot else self._env.ped_goals[self._idx - 1][1]

        @gy.setter
        def gy(self, v):
            if self.is_robot:
                self._env.robot_goal[1] = v
            else:
                self._env.ped_goals[self._idx - 1][1] = v

    # ---- 公开属性：env.robot / env.humans ----
    @property
    def robot(self):
        return self._AgentWrapper(self, 0, is_robot=True)

    @property
    def humans(self):
        return [self._AgentWrapper(self, i + 1) for i in range(self.n_peds)]
