"""Current observations -> recursive belief -> continuous unicycle actions."""
import copy
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from crowd_nav.belief_mdp.runtime import FullCrowdNavigationEnvironment, belief_state_observation
from crowd_nav.gdbn import GDBNIntegration
from crowd_sim.envs.utils.action import ActionRot

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('no_belief', 'map', 'full')
MAX_HUMANS = 20


def transform_observation(observation, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    result = {k: v.copy() for k, v in observation.items()}
    if arm == 'no_belief':
        result['humans'][:, 5:] = 0
    elif arm == 'map':
        valid = result['mask'] > 0
        labels = result['humans'][valid, 5:8].argmax(1)
        result['humans'][valid, 5:8] = np.eye(3, dtype=np.float32)[labels]
        result['humans'][:, 8] = 0
    return result


def project_orca(velocity, heading, dt, vmax=1., wmax=1.2):
    velocity = np.asarray(velocity, dtype=float)
    if not np.isfinite(velocity).all():
        raise ValueError('Invalid teacher velocity')
    if np.linalg.norm(velocity) < 1e-8:
        return np.zeros(2, dtype=np.float32)
    target = np.arctan2(velocity[1], velocity[0])
    angle = (target - heading + np.pi) % (2*np.pi) - np.pi
    omega = np.clip(angle / dt, -wmax, wmax)
    next_heading = heading + omega * dt
    speed = np.clip(velocity @ np.array([np.cos(next_heading), np.sin(next_heading)]), 0., vmax)
    return np.array([speed, omega], dtype=np.float32)


class BeliefEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, params, arm='full', scenario='baseline_circle', training=True,
                 seed=2407, particles=50):
        super().__init__()
        if training and scenario != 'baseline_circle':
            raise ValueError('Training/model selection is restricted to five humans')
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.training = arm, training
        self.world = FullCrowdNavigationEnvironment(str(ROOT / 'crowd_nav/configs/env_belief_mdp.config'),
                                                    scenario, robot_visible=False)
        self.filter = GDBNIntegration(params_dir=str(params), max_peds=MAX_HUMANS,
                                      n_particles=particles, random_seed=seed)
        if self.filter.K != 3:
            raise ValueError('Frozen K=3 protocol')
        self.filter.B_action = [np.zeros_like(b) for b in self.filter.B_action]
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(np.array([0., -1.2], np.float32), np.array([1., 1.2], np.float32))
        self.observation_space = spaces.Dict({
            'robot': spaces.Box(-np.inf, np.inf, (9,), np.float32),
            'humans': spaces.Box(-np.inf, np.inf, (MAX_HUMANS, 9), np.float32),
            'mask': spaces.Box(0., 1., (MAX_HUMANS,), np.float32)})
        self.episode_records = []
        self._teacher_cache = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        layout_seed = int(options.get('layout_seed', 300000 + self.rng.integers(10_000_000)))
        profile = options.get('profile', 'nominal')
        if self.training and profile not in ('nominal', 'train_nonstationary'):
            raise ValueError('Held-out profile cannot enter training')
        case = int(options.get('test_case', self.rng.integers(9000)))
        self.world.reset(layout_seed, profile, case)
        # Human agents remain holonomic; only the robot changes executor.
        self.world.robot.kinematics = 'unicycle'
        self.filter.reset()
        self.elapsed = 0
        self.total_reward = 0.
        self.bound_violations = 0
        self.layout_seed, self.case = layout_seed, case
        self._teacher_cache = None
        return self._observation(), {'layout_seed': layout_seed, 'test_case': case}

    def _observation(self):
        robot = self.world.robot
        humans = self.world.env.humans
        if len(humans) > MAX_HUMANS:
            raise ValueError('Entity truncation is forbidden')
        mask = np.arange(MAX_HUMANS) < len(humans)
        self.filter.update(belief_state_observation(robot, humans, MAX_HUMANS), valid_mask=mask)
        belief = np.asarray(self.filter.get_belief_snapshot().features, np.float32)
        c, s = np.cos(robot.theta), np.sin(robot.theta)
        rotation = np.array([[c, s], [-s, c]])
        goal = rotation @ np.array([robot.gx-robot.px, robot.gy-robot.py])
        velocity = rotation @ np.array([robot.vx, robot.vy])
        remaining = max(0., (self.world.env.time_limit-self.world.env.global_time)/self.world.env.time_limit)
        ego = np.array([*goal, *velocity, robot.radius, robot.v_pref, s, c, remaining], np.float32)
        tokens = np.zeros((MAX_HUMANS, 9), np.float32)
        for i, h in enumerate(humans):
            relative = rotation @ np.array([h.px-robot.px, h.py-robot.py])
            rel_velocity = rotation @ np.array([h.vx-robot.vx, h.vy-robot.vy])
            tokens[i] = [*relative, *rel_velocity, h.radius, *belief[i, :4]]
        return transform_observation({'robot': ego, 'humans': tokens, 'mask': mask.astype(np.float32)}, self.arm)

    def expert_action(self):
        if self._teacher_cache is None:
            action = self.world.expert_action()
            self._teacher_cache = project_orca([action.vx, action.vy], self.world.robot.theta,
                                               self.world.env.time_step)
        return self._teacher_cache.copy()

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Action must be finite (v, omega)')
        if not self.action_space.contains(action):
            self.bound_violations += 1
            raise ValueError('Out-of-bounds continuous action')
        before_heading = float(self.world.robot.theta)
        before_position = np.array([self.world.robot.px, self.world.robot.py])
        dt = self.world.env.time_step
        result = self.world.step(ActionRot(float(action[0]), float(action[1]*dt)))
        expected_heading = (before_heading + action[1]*dt) % (2*np.pi)
        expected_position = before_position + action[0]*dt*np.array([np.cos(expected_heading), np.sin(expected_heading)])
        if not np.allclose([self.world.robot.px, self.world.robot.py], expected_position, atol=1e-6):
            raise AssertionError('Unicycle executor contract mismatch')
        self.elapsed += 1
        self.total_reward += result.reward
        self._teacher_cache = None
        obs = self._observation()
        info = {'outcome': result.outcome, 'dmin': result.dmin, 'action': action.copy()}
        if result.done:
            record = dict(layout_seed=self.layout_seed, test_case=self.case, outcome=result.outcome,
                          steps=self.elapsed, reward=self.total_reward, bound_violations=self.bound_violations)
            self.episode_records.append(record)
            info['episode_result'] = record
        # Timeout is a terminal finite-horizon task failure, not an external truncation.
        return obs, float(result.reward), bool(result.done), False, info
