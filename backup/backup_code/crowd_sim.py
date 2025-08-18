import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import math
import numpy as np
from math import hypot
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.state import JointState


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        # ------------- 新增：定义动作空间和观测空间 -------------
        from gym import spaces
        import numpy as np
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        # -----------------------------------------------------


    def configure(self, config):
        self.config = config
        # —— 环境基础参数 —— 
        self.time_limit           = config.getint('env', 'time_limit')
        self.time_step            = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')

        # —— 读取原有 reward 参数 —— 
        self.success_reward     = config.getfloat('reward', 'success_reward')
        self.collision_penalty  = config.getfloat('reward', 'collision_penalty')
        self.discomfort_penalty = config.getfloat('reward', 'discomfort_penalty')
        self.step_penalty       = config.getfloat('reward', 'step_penalty')
        self.danger_dist        = config.getfloat('reward', 'danger_dist')
        # 保持兼容，discomfort_dist 等同 danger_dist
        self.discomfort_dist    = self.danger_dist
        self.goal_thresh        = config.getfloat('reward', 'goal_thresh')

        # —— 读取新增的 Mamba 奖励权重参数 —— 
        self.w_prog   = config.getfloat('reward', 'w_prog')
        self.w_goal   = config.getfloat('reward', 'w_goal')
        self.w_coll   = config.getfloat('reward', 'w_coll')
        self.w_soc    = config.getfloat('reward', 'w_soc')
        self.soc_dist = config.getfloat('reward', 'soc_dist')
        self.alpha    = config.getfloat('reward', 'alpha')
        self.v_safe   = config.getfloat('reward', 'v_safe')
        self.w_relv   = config.getfloat('reward', 'w_relv')
        self.w_time   = config.getfloat('reward', 'w_time')
        self.w_shape  = config.getfloat('reward', 'w_shape')
        self.w_align = config.getfloat('reward', 'w_align')
        self.max_dist = config.getfloat('reward', 'max_dist')


        # —— 模拟配置（humans policy） —— 
        if config.get('humans', 'policy') == 'orca':
            # 人数及场景容量设置
            self.case_capacity = {
                'train': np.iinfo(np.uint32).max - 2000,
                'val':   1000,
                'test':  1000
            }
            self.case_size = {
                'train': np.iinfo(np.uint32).max - 2000,
                'val':   config.getint('env', 'val_size'),
                'test':  config.getint('env', 'test_size')
            }
            self.train_val_sim  = config.get('sim', 'train_val_sim')
            self.test_sim       = config.get('sim', 'test_sim')
            self.square_width   = config.getfloat('sim', 'square_width')
            self.circle_radius  = config.getfloat('sim', 'circle_radius')
            self.human_num      = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError(f"Unknown human policy: {config.get('humans','policy')}")

        # —— 统计计数器初始化 —— 
        self.case_counter = {'train': 0, 'val': 0, 'test': 0}

        # —— 日志输出 —— 
        logging.info(f'human number: {self.human_num}')
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info(f'Training simulation: {self.train_val_sim}, test simulation: {self.test_sim}')
        logging.info(f'Square width: {self.square_width}, circle radius: {self.circle_radius}')



    def set_robot(self, robot):
        self.robot = robot

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        try:
            human.radius = float(self.config.get('humans', 'radius'))
        except Exception:
            human.radius = 0.3   # fallback

        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human


    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        Compatible with both IL采集和RL训练
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        # === 人类时间数组初始化 ===
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * self.human_num

        # === 控制IL采样时human数量，与RL一致 ===
        # （不管multiagent_training与否，都用human_num）
        human_num = self.human_num

        # === 随机化策略 ===
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                            'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                # 只在val/test阶段设定seed，train采样完全系统随机
                if phase in ['val', 'test']:
                    np.random.seed(counter_offset[phase] + self.case_counter[phase])
                # train采样不用seed，保证采样多样性
                sim_rule = self.train_val_sim if phase in ['train', 'val'] else self.test_sim
                self.generate_random_human_position(human_num=human_num, rule=sim_rule)
                # case_counter自动递增
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # debug手动设置
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        # === 动态参数同步 ===
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # === 返回与RL兼容的obs ===
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob, {}







    def onestep_lookahead(self, action):
        return self.step(action, update=False)

   



    def step(self, action):
        # ---------- 运动学更新前先记录 ----------
        prev_pos = np.array([self.robot.px, self.robot.py])
        goal_pos = np.array(self.robot.get_goal_position())
        dist_before = np.linalg.norm(prev_pos - goal_pos)
        
        # 记录之前的速度用于平滑性惩罚
        prev_vel = np.array([self.robot.vx, self.robot.vy])

        # ---------- 执行机器人动作 ----------
        self.robot.step(action)

        # ---------- 所有人类动作 ----------
        human_actions = [
            human.act([other.get_observable_state()
                    for other in self.humans if other != human])
            for human in self.humans
        ]
        for human, ha in zip(self.humans, human_actions):
            human.step(ha)

        # ---------- 本轮结束后的基础量 ----------
        curr_pos = np.array([self.robot.px, self.robot.py])
        curr_vel = np.array([self.robot.vx, self.robot.vy])
        dist_after = np.linalg.norm(curr_pos - goal_pos)
        
        # 计算与所有人类的距离
        human_positions = [np.array([h.px, h.py]) for h in self.humans]
        distances_to_humans = [np.linalg.norm(curr_pos - h_pos) for h_pos in human_positions]
        min_dist = min(distances_to_humans) if distances_to_humans else float('inf')
        
        reached_goal = dist_after <= self.goal_thresh
        collision = any(dist < (self.robot.radius + h.radius) for dist, h in zip(distances_to_humans, self.humans))

        # ---------- 改进的奖励计算 ----------
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # ---- 1. 终止情况 ----
        if reached_goal:
            # 成功奖励：基础奖励 + 时间奖励（越快到达奖励越高）
            time_bonus = max(0, (self.time_limit - self.global_time) / self.time_limit)
            reward = 10.0 + 5.0 * time_bonus
            terminated = True
            info['event'] = 'ReachGoal'

        elif collision:
            # 碰撞惩罚：严重惩罚
            reward = -10.0
            terminated = True
            info['event'] = 'Collision'

        elif self.global_time >= self.time_limit - 1:
            # 超时惩罚：根据剩余距离调整
            distance_penalty = dist_after / np.linalg.norm(goal_pos - np.array([0, -self.circle_radius]))
            reward = -5.0 * (1 + distance_penalty)
            truncated = True
            info['event'] = 'Timeout'

        # ---- 2. 正常步进的多重奖励 ----
        else:
            # (a) 进度奖励：显著增加权重
            progress = (dist_before - dist_after)
            reward += 2.0 * progress / self.time_step
            
            # (b) 目标导向奖励：鼓励朝目标方向移动
            to_goal = goal_pos - curr_pos
            if np.linalg.norm(to_goal) > 0:
                to_goal_unit = to_goal / np.linalg.norm(to_goal)
                vel_magnitude = np.linalg.norm(curr_vel)
                if vel_magnitude > 0:
                    vel_unit = curr_vel / vel_magnitude
                    alignment = np.dot(to_goal_unit, vel_unit)
                    reward += 0.5 * alignment * vel_magnitude
            
            # (c) 安全距离奖励/惩罚：分层处理
            safety_radius = self.robot.radius + 0.5  # 安全距离
            danger_radius = self.robot.radius + 0.2   # 危险距离
            
            if min_dist < danger_radius:
                # 危险区域：严重惩罚
                danger_penalty = (danger_radius - min_dist) / danger_radius
                reward += -5.0 * danger_penalty
            elif min_dist < safety_radius:
                # 不舒适区域：轻微惩罚
                discomfort_penalty = (safety_radius - min_dist) / (safety_radius - danger_radius)
                reward += -1.0 * discomfort_penalty
            else:
                # 安全区域：小奖励
                reward += 0.1
            
            # (d) 速度惩罚：避免过慢或过快
            speed = np.linalg.norm(curr_vel)
            optimal_speed = self.robot.v_pref
            if speed < 0.1:  # 太慢
                reward += -0.5
            elif speed > optimal_speed * 1.2:  # 太快
                reward += -0.3
            else:  # 合理速度范围
                reward += 0.1
                
            # (e) 动作平滑性：惩罚急剧变化
            if hasattr(self, 'prev_action'):
                action_diff = np.linalg.norm(np.array(action) - self.prev_action)
                if action_diff > 0.5:  # 动作变化过大
                    reward += -0.2 * action_diff
            
            # (f) 距离目标的相对奖励：越接近目标，基础奖励越高
            max_dist = np.linalg.norm(goal_pos - np.array([0, -self.circle_radius]))
            proximity_bonus = 0.5 * (1 - dist_after / max_dist)
            reward += proximity_bonus
            
            # (g) 时间惩罚：鼓励尽快完成任务
            reward += -0.01  # 小的时间步惩罚
            
            info['event'] = 'Step'
            info['min_dist'] = min_dist
            info['progress'] = progress
            info['speed'] = speed

        # 记录当前动作用于下一步的平滑性计算
        self.prev_action = np.array(action)

        # ---------- 时间推进 ----------
        self.global_time += self.time_step

        # ---------- 观测拼接 ----------
        robot_obs = self.robot.get_full_state().to_array()
        humans_obs = np.concatenate([h.get_observable_state().to_array() for h in self.humans])
        obs = np.concatenate([robot_obs, humans_obs]).astype(np.float32)

        return obs, reward, terminated, truncated, info





    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy],
                     color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')

            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError