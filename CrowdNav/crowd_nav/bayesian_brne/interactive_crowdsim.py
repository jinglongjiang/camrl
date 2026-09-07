"""Action-conditioned held-out protocol on the real CrowdSim backend.

This adapter preserves CrowdSim's own collision, goal, timeout, and motion
integration.  It only replaces each human's policy after reset so the human
reads the robot action staged for the current ``CrowdSim.step`` call.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.action import ActionXY

from crowd_nav.bayesian_brne.interaction_protocol import (
    HELDOUT_INTERACTIVE_PROFILE,
    HumanBehaviorState,
    ProtocolConfig,
    _apply_human_human_avoidance,
    _behavior_mix_from_assertive_probability,
    _clip_speed_and_accel,
    _config_from_params,
    compute_human_response,
    sample_behavior_type,
    sample_profile_params,
)


class _ActionConditionedHumanPolicy:
    """CrowdSim human policy for one fixed human object index."""

    def __init__(self, env: "HeldoutInteractiveCrowdSim", human_index: int):
        self.env = env
        self.human_index = int(human_index)
        self.time_step = float(env.time_step)

    def predict(self, _state):
        env = self.env
        human = env.humans[self.human_index]
        action = env._pending_robot_action
        if action is None:
            raise RuntimeError("interactive human policy used before a robot action was staged")

        human_pos = np.array([human.px, human.py], dtype=np.float64)
        human_vel = np.array([human.vx, human.vy], dtype=np.float64)
        human_goal = np.array([human.gx, human.gy], dtype=np.float64)
        robot_pos = np.array([env.robot.px, env.robot.py], dtype=np.float64)
        robot_vel = np.array([action.vx, action.vy], dtype=np.float64)
        positions = np.asarray([[h.px, h.py] for h in env.humans], dtype=np.float64)
        velocities = np.asarray([[h.vx, h.vy] for h in env.humans], dtype=np.float64)
        radii = np.asarray([h.radius for h in env.humans], dtype=np.float64)

        preferred, new_state, _in_conflict, repulsion_scale = compute_human_response(
            human_pos=human_pos,
            human_vel=human_vel,
            human_pref_speed=float(human.v_pref),
            goal=human_goal,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            combined_radius=float(human.radius + env.robot.radius),
            state=human.behavior_state,
            config=env.protocol_config,
            dt=float(env.time_step),
        )
        preferred = _apply_human_human_avoidance(
            self.human_index, positions, velocities, radii, preferred,
            repulsion_scale=float(repulsion_scale),
        )
        velocity = _clip_speed_and_accel(
            human_vel, preferred, float(env.time_step), env.protocol_config,
        )
        human.behavior_state = new_state
        return ActionXY(float(velocity[0]), float(velocity[1]))


class HeldoutInteractiveCrowdSim(CrowdSim):
    """Real CrowdSim with the held-out action-conditioned human protocol."""

    def __init__(self):
        super().__init__()
        self._pending_robot_action: Optional[ActionXY] = None
        self._interactive_rng: Optional[np.random.Generator] = None
        self.protocol_config = ProtocolConfig()
        self.profile_params = None

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._interactive_rng = np.random.default_rng(seed)
        self.profile_params = sample_profile_params(
            self._interactive_rng, HELDOUT_INTERACTIVE_PROFILE,
        )
        self.protocol_config = _config_from_params(self.profile_params)
        behavior_mix = _behavior_mix_from_assertive_probability(
            self.profile_params.assertive_probability,
        )
        for index, human in enumerate(self.humans):
            human.v_pref = float(self._interactive_rng.uniform(
                self.profile_params.speed_lo, self.profile_params.speed_hi,
            ))
            human.behavior_state = HumanBehaviorState(
                behavior_type=sample_behavior_type(self._interactive_rng, behavior_mix),
            )
            human.policy = _ActionConditionedHumanPolicy(self, index)
            human.policy.time_step = float(self.time_step)
        self._pending_robot_action = ActionXY(float(self.robot.vx), float(self.robot.vy))
        return observation, info

    def step(self, action, update=True):
        staged = self._to_action(action)
        if not isinstance(staged, ActionXY):
            raise TypeError("HeldoutInteractiveCrowdSim requires ActionXY actions")
        self._pending_robot_action = staged
        return super().step(staged, update=update)

