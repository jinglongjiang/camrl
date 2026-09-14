"""Receding-horizon unicycle rollout teacher, with swept CV obstacle checks.

This teacher is an empirical baseline, not a certified collision-free expert.
It consumes current full observations only; no human goals or future states.
"""
import numpy as np


def unicycle_teacher(world):
    robot, dt = world.robot, world.env.time_step
    speeds, turns = np.meshgrid(np.linspace(0., 1., 9), np.linspace(-1.2, 1.2, 17))
    commands = np.stack([speeds.ravel(), turns.ravel()], axis=1)
    n = len(commands)
    pos = np.tile([robot.px, robot.py], (n, 1)).astype(float)
    theta = np.full(n, robot.theta)
    goal = np.array([robot.gx, robot.gy])
    humans = world.env.humans
    hp = np.array([[h.px, h.py] for h in humans]).reshape(-1, 2)
    hv = np.array([[h.vx, h.vy] for h in humans]).reshape(-1, 2)
    radii = np.array([h.radius + robot.radius for h in humans])
    clearance = np.full(n, np.inf)
    first_clearance = clearance.copy()
    for k in range(16):
        # Hold candidate for one second, then use bounded goal steering.
        omega = commands[:, 1] if k < 4 else np.clip(
            (np.arctan2(goal[1]-pos[:, 1], goal[0]-pos[:, 0])-theta+np.pi) %
            (2*np.pi)-np.pi, -1.2, 1.2)
        theta = theta + omega*dt
        speed = commands[:, 0].copy()
        if k >= 4:
            speed = np.minimum(speed, np.linalg.norm(goal-pos, axis=1)/dt)
            heading_error = np.arctan2(goal[1]-pos[:, 1], goal[0]-pos[:, 0])-theta
            speed *= np.maximum(0., np.cos(heading_error))
        nxt = pos + speed[:, None]*dt*np.stack([np.cos(theta), np.sin(theta)], axis=1)
        if len(humans):
            start = pos[:, None] - (hp + hv*(k*dt))[None]
            end = nxt[:, None] - (hp + hv*((k+1)*dt))[None]
            delta = end-start
            fraction = np.clip(-(start*delta).sum(-1) /
                               np.maximum((delta*delta).sum(-1), 1e-12), 0., 1.)
            gap = np.linalg.norm(start+fraction[..., None]*delta, axis=-1)-radii
            clearance = np.minimum(clearance, gap.min(1))
            if k == 0:
                first_clearance = clearance.copy()
        pos = nxt
    score = -np.linalg.norm(pos-goal, axis=1) - .03*np.abs(commands[:, 1])
    score -= 2.*np.maximum(.25-clearance, 0.)
    safe = clearance >= .1
    if safe.any():
        score[~safe] = -np.inf
    else:
        # Explicit degraded choice, not an assertion that braking is safe.
        score = 100.*first_clearance + clearance + .01*score
    return commands[int(np.argmax(score))].astype(np.float32)
