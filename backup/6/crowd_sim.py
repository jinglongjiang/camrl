import logging
import gym
from gym import spaces
import numpy as np
from numpy.linalg import norm
import rvo2
from matplotlib import patches
import matplotlib.lines as mlines

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
# ★ NEW: 允许把 [-1,1] 数组动作转换为物理动作
from crowd_sim.envs.utils.action import ActionXY, ActionRot


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # ------------ core states ------------
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None

        # ------------ reward params (base) ------------
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None

        # ------------ reward weights (shaping) ------------
        self.w_prog = 1.0
        self.w_goal = 1.0
        self.w_coll = 1.0
        self.w_soc  = 0.0
        self.w_time = 0.0
        self.w_shape = 0.0
        self.w_align = 0.0
        self.w_relv = 0.0
        self.max_dist = 10.0
        self.alpha = 10.0
        self.v_safe = 0.5
        self.soc_dist = 0.5

        # ------------ industry/common additions ------------
        self.w_ttc = 0.0
        self.ttc_thresh = 2.0
        self.w_speed = 0.0
        self.speed_target = 0.6
        self.noprog_eps = 0.02
        self.noprog_patience = 20
        self._no_prog_steps = 0

        # ------------ sim config ------------
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

        # ------------ vis buffers ------------
        self.states = None
        self.action_values = None
        self.attention_weights = None

        # ------------ Gym spaces ------------
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        _obs_dim_placeholder = 9 + 5 * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dim_placeholder,), dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    def configure(self, config):
        self.config = config

        # env
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')

        # reward (base)
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')

        # reward (weights)
        _g = config.getfloat
        self.w_prog  = _g('reward', 'w_prog',  fallback=1.0)
        self.w_goal  = _g('reward', 'w_goal',  fallback=1.0)
        self.w_coll  = _g('reward', 'w_coll',  fallback=1.0)
        self.w_soc   = _g('reward', 'w_soc',   fallback=0.0)
        self.soc_dist= _g('reward', 'soc_dist',fallback=0.5)
        self.alpha   = _g('reward', 'alpha',   fallback=10.0)
        self.v_safe  = _g('reward', 'v_safe',  fallback=0.5)
        self.w_relv  = _g('reward', 'w_relv',  fallback=0.0)
        self.w_time  = _g('reward', 'w_time',  fallback=0.0)
        self.w_shape = _g('reward', 'w_shape', fallback=0.0)
        self.w_align = _g('reward', 'w_align', fallback=0.0)
        self.max_dist= _g('reward', 'max_dist',fallback=10.0)

        # additions
        self.w_ttc   = _g('reward', 'w_ttc',   fallback=0.0)
        self.ttc_thresh = _g('reward', 'ttc_thresh', fallback=2.0)
        self.w_speed = _g('reward', 'w_speed', fallback=0.0)
        self.speed_target = _g('reward', 'speed_target', fallback=0.6)
        self.noprog_eps = _g('reward', 'noprog_eps', fallback=0.02)
        self.noprog_patience = int(config.get('reward', 'noprog_patience', fallback='20'))

        # sim
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {
                'train': np.iinfo(np.uint32).max - 2000,
                'val': config.getint('env', 'val_size'),
                'test': config.getint('env', 'test_size')
            }
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        # obs dim = robot(9) + humans(human_num*5)
        obs_dim = 9 + 5 * self.human_num
        if self.observation_space is None or self.observation_space.shape[0] != obs_dim:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        logging.info('human number: %d', self.human_num)
        logging.info("Randomize human's radius and preferred speed" if self.randomize_attributes
                     else "Not randomize human's radius and preferred speed")
        logging.info('Training simulation: %s, test simulation: %s', self.train_val_sim, self.test_sim)
        logging.info('Square width: %.1f, circle radius: %.1f', self.square_width, self.circle_radius)

    def set_robot(self, robot):
        self.robot = robot

    # ★ NEW: 供并行环境在运行时切换策略（collect_vec 里会调用）
    def set_robot_policy(self, policy):
        try:
            if self.robot is not None:
                self.robot.set_policy(policy)
            return True
        except Exception:
            return False

    # ★ NEW: 供 AsyncVectorEnv.reset_done() 使用；没有细粒度 done 管理就退回 reset
    def reset_done(self):
        obs, _ = self.reset()
        return obs

    # -------------------------------------------------------------------------
    # Position generators
    # -------------------------------------------------------------------------
    def generate_random_human_position(self, human_num, rule):
        if rule == 'square_crossing':
            self.humans = [self.generate_square_crossing_human() for _ in range(human_num)]
        elif rule == 'circle_crossing':
            self.humans = [self.generate_circle_crossing_human() for _ in range(human_num)]
        elif rule == 'mixed':
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            target = static_human_num if static else dynamic_human_num
            for key, value in sorted(target.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                width, height = 4, 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for _ in range(human_num):
                    human = Human(self.config, 'humans')
                    sign = -1 if np.random.random() > 0.5 else 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        if all(norm((px - a.px, py - a.py)) >= human.radius + a.radius + self.discomfort_dist
                               for a in [self.robot] + (self.humans or [])):
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                for i in range(human_num):
                    h = self.generate_circle_crossing_human() if i < 2 else self.generate_square_crossing_human()
                    self.humans.append(h)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * 2 * np.pi
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            agents = [self.robot] + (self.humans or [])
            min_ok = all(
                norm((px - a.px, py - a.py)) >= human.radius + a.radius + self.discomfort_dist and
                norm((px - a.gx, py - a.gy)) >= human.radius + a.radius + self.discomfort_dist
                for a in agents
            )
            if min_ok:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        sign = -1 if np.random.random() > 0.5 else 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            if all(norm((px - a.px, py - a.py)) >= human.radius + a.radius + self.discomfort_dist
                   for a in [self.robot] + (self.humans or [])):
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            if all(norm((gx - a.gx, gy - a.gy)) >= human.radius + a.radius + self.discomfort_dist
                   for a in [self.robot] + (self.humans or [])):
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    # -------------------------------------------------------------------------
    # Reset / Step
    # -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                from gym.utils.seeding import np_random
                self.np_random, _ = np_random(seed)
            except Exception:
                np.random.seed(seed)

        phase = getattr(self, 'phase', 'test')
        test_case = None
        if options and isinstance(options, dict):
            test_case = options.get('test_case', None)

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.global_time = 0.0
        self._no_prog_steps = 0
        if phase == 'test':
            self.human_times = [0.0] * self.human_num
        else:
            self.human_times = [0.0] * (self.human_num if self.robot.policy.multiagent_training else 1)

        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        self.humans = []

        if self.config.get('humans', 'policy') != 'trajnet':
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if test_case is not None:
                self.case_counter[phase] = test_case

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = []
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = []
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = []

        if self.robot.sensor == 'coordinates':
            ob = np.concatenate([self.robot.get_obs_array()] +
                                [h.get_obs_array() for h in self.humans]).astype(np.float32)
        else:
            raise NotImplementedError

        return ob, {}

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def _rel_vel_robot(self, action):
        if self.robot.kinematics == 'holonomic':
            rvx = -action.vx
            rvy = -action.vy
        else:
            rvx = -action.v * np.cos(action.r + self.robot.theta)
            rvy = -action.v * np.sin(action.r + self.robot.theta)
        return rvx, rvy

    @staticmethod
    def _ttc_rel(px, py, vx, vy, radius_sum):
        """
        线性相对运动 TTC，返回秒；不可解或非正返回 +inf
        解 ||p + v t|| = R
        """
        p = np.array([px, py], dtype=float)
        v = np.array([vx, vy], dtype=float)
        vv = v @ v
        if vv < 1e-8:
            return np.inf
        R2 = radius_sum * radius_sum
        b = 2.0 * (p @ v)
        c = (p @ p) - R2
        disc = b * b - 4.0 * vv * c
        if disc <= 0.0:
            return np.inf
        t1 = (-b - np.sqrt(disc)) / (2.0 * vv)
        if t1 <= 1e-6:
            return np.inf
        return float(t1)

    # ★ NEW: 允许数组动作（[-1,1]）→ 物理速度动作（ActionXY）
    def _to_action(self, action):
        # 已是 Action 对象：原样返回
        if hasattr(action, 'vx') or hasattr(action, 'v'):
            return action
        # 否则视为 tanh 归一化动作
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f'Action as array must have shape (2,), got {a.shape}')
        ax, ay = float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
        v_pref = getattr(self.robot, 'v_pref', 1.0)
        scale = v_pref * getattr(getattr(self.robot, 'policy', None), 'action_scale', 1.0)
        return ActionXY(ax * scale, ay * scale)

    def step(self, action, update=True):
        # humans act
        human_actions = []
        for human in self.humans:
            ob = [other.get_observable_state() for other in self.humans if other != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # ★ NEW: 统一动作类型
        action = self._to_action(action)

        # relative geometry
        rvx, rvy = self._rel_vel_robot(action)

        dmin = float('inf')
        collision = False
        closest_id = -1
        ttc_min = np.inf

        # compute min separation & TTC wrt robot
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            vx = human.vx + rvx
            vy = human.vy + rvy
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            closest = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest < 0:
                collision = True
                closest_id = i
                dmin = closest
                ttc_min = 0.0
                break
            if closest < dmin:
                dmin = closest
                closest_id = i
            # TTC
            ttc = self._ttc_rel(px, py, vx, vy, human.radius + self.robot.radius)
            if ttc < ttc_min:
                ttc_min = ttc

        # human-human collision (log only)
        hn = len(self.humans)
        for i in range(hn):
            for j in range(i + 1, hn):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** 0.5 - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    logging.debug('Collision happens between humans in step()')

        # check reaching goal (using next robot pos)
        next_pos = self.robot.compute_position(action, self.time_step)
        reaching_goal = norm(np.array(next_pos) - np.array(self.robot.get_goal_position())) < self.robot.radius

        # distance before/after
        prev_dist = norm(np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
        curr_dist = norm(np.array(next_pos) - np.array(self.robot.get_goal_position()))
        progress = prev_dist - curr_dist

        # ===== reward shaping =====
        reward = 0.0
        terminated = False
        truncated = False

        # terminals
        if self.global_time >= self.time_limit - 1e-6:
            reward += -self.w_goal * 0.1
            truncated = True
            info = {"result": "timeout", "event": "timeout",
                    "min_dist": max(dmin, 0.0), "dmin": max(dmin, 0.0), "ttc": float(ttc_min)}
        elif collision:
            reward += self.w_coll * self.collision_penalty
            terminated = True
            info = {"result": "collision", "event": "collision",
                    "min_dist": dmin, "dmin": dmin, "hid": closest_id, "ttc": 0.0}
        elif reaching_goal:
            reward += self.w_goal * self.success_reward
            terminated = True
            info = {"result": "reach_goal", "event": "reach_goal",
                    "min_dist": max(dmin, 0.0), "dmin": max(dmin, 0.0), "ttc": float(ttc_min)}
        else:
            # 1) progress
            reward += self.w_prog * progress
            # 2) time cost per step
            reward += -self.w_time * self.time_step
            # 3) social discomfort (quadratic, smooth)
            if dmin < self.discomfort_dist:
                penalty_mag = abs(self.discomfort_penalty_factor) * (self.discomfort_dist - dmin) ** 2
                reward -= self.w_soc * penalty_mag
            # 4) TTC penalty
            if self.w_ttc > 0.0 and np.isfinite(ttc_min) and ttc_min < self.ttc_thresh:
                reward -= self.w_ttc * (self.ttc_thresh - ttc_min) / max(self.ttc_thresh, 1e-6)
            # 5) global shape potential
            if self.w_shape > 0.0:
                reward += -self.w_shape * min(curr_dist / max(self.max_dist, 1e-6), 1.0)
            # 6) align to goal direction (holonomic)
            if self.w_align > 0.0 and self.robot.kinematics == 'holonomic':
                desired = np.array(self.robot.get_goal_position()) - np.array([self.robot.px, self.robot.py])
                if norm(desired) > 1e-6:
                    desired /= norm(desired)
                    a_vec = np.array([action.vx, action.vy], dtype=float)
                    if norm(a_vec) > 1e-6:
                        a_vec /= norm(a_vec)
                        align = float(np.clip(desired @ a_vec, -1.0, 1.0))
                        reward += self.w_align * align * 0.1
            # 7) relative speed penalty（近距离高速接近）
            if self.w_relv > 0.0 and closest_id >= 0:
                h = self.humans[closest_id]
                v_rel = norm(np.array([h.vx + rvx, h.vy + rvy]))
                if dmin < self.soc_dist:
                    reward += -self.w_relv * max(0.0, v_rel - self.v_safe)
            # 8) target speed shaping —— 二次偏差惩罚
            if self.w_speed > 0.0 and self.robot.kinematics == 'holonomic':
                v_obs = norm(np.array([action.vx, action.vy], dtype=float))
                v_pref = max(getattr(self.robot, 'v_pref', 1.0), 1e-6)
                target = self.speed_target * v_pref
                if curr_dist > 1.5 * self.robot.radius:
                    dev = (v_obs - target) / v_pref
                    reward -= self.w_speed * (dev * dev)

            # no-progress early stop
            if progress < self.noprog_eps:
                self._no_prog_steps += 1
            else:
                self._no_prog_steps = 0
            if self._no_prog_steps >= self.noprog_patience:
                truncated = True
                reward += -self.w_goal * 0.1
                info = {"result": "no_progress", "event": "no_progress",
                        "min_dist": max(dmin, 0.0), "dmin": max(dmin, 0.0), "ttc": float(ttc_min)}
            else:
                in_danger = (dmin < self.discomfort_dist) or (np.isfinite(ttc_min) and ttc_min < self.ttc_thresh)
                info = {
                    "result": "danger" if in_danger else "nothing",
                    "event":  "danger" if in_danger else "nothing",
                    "min_dist": max(dmin, 0.0),
                    "dmin": max(dmin, 0.0),
                    "ttc": float(ttc_min),
                    "hid": closest_id
                }

        # ===== update world & next obs =====
        if update:
            self.states.append([self.robot.get_full_state(), [h.get_full_state() for h in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            self.robot.step(action)
            for i, ha in enumerate(human_actions):
                self.humans[i].step(ha)
            self.global_time += self.time_step

            for i, h in enumerate(self.humans):
                if self.human_times[i] == 0 and h.reached_destination():
                    self.human_times[i] = self.global_time

            if self.robot.sensor == 'coordinates':
                ob = np.concatenate([self.robot.get_obs_array()] +
                                    [h.get_obs_array() for h in self.humans]).astype(np.float32)
            else:
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = np.concatenate(
                    [self.robot.get_next_obs_array(action)] +
                    [h.get_next_obs_array(a) for h, a in zip(self.humans, human_actions)]
                ).astype(np.float32)
            else:
                raise NotImplementedError

        return ob, float(reward), bool(terminated), bool(truncated), info

    # -------------------------------------------------------------------------
    # (Optional) human-only finishing time sim — kept for compatibility
    # -------------------------------------------------------------------------
    def get_human_times(self):
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

            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    # -------------------------------------------------------------------------
    # Render (kept largely as-is)
    # -------------------------------------------------------------------------
    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16); ax.set_ylabel('y(m)', fontsize=16)

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
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16); ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color='red', marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color='yellow')
            ax.add_artist(robot); ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - 0.11, humans[i].center[1] - 0.11, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human); ax.add_artist(human_numbers[i])

            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16); ax.add_artist(time)

            def update(frame_num):
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - 0.11, human.center[1] - 0.11))
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            from matplotlib import animation
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                import matplotlib.pyplot as plt
                plt.show()
        else:
            raise NotImplementedError

