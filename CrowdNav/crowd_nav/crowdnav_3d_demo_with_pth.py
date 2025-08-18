#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math
import random
import time
import threading
from rclpy.executors import MultiThreadedExecutor

# ROS2/Gazebo
from gazebo_msgs.srv import GetEntityState, SetEntityState
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

import torch
import configparser
import os

# 根据你本地结构, 引入 FullState, ObservableState, JointState
# 下面假设 crowd_sim.envs.utils.state 里有
from crowd_sim.envs.utils.state import FullState, ObservableState, JointState

###############################################################################
# 1) RandomActorController
###############################################################################
class RandomActorController(Node):
    def __init__(self, actor_names=['actor1','actor2'], rate_hz=2.0):
        super().__init__('random_actor_controller')
        self.actor_names = actor_names
        self.rate_hz = rate_hz

        self.get_state_cli = self.create_client(GetEntityState, '/get_entity_state')
        self.set_state_cli = self.create_client(SetEntityState, '/set_entity_state')

        self.get_logger().info("Waiting for /get_entity_state & /set_entity_state ...")
        self.get_state_cli.wait_for_service()
        self.set_state_cli.wait_for_service()

        self.get_logger().info("RandomActorController started.")
        self.actor_vel = {}
        for a in actor_names:
            self.actor_vel[a] = [0.0, 0.0]

        # Timer for random movement
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)

    def timer_callback(self):
        dt = 1.0 / self.rate_hz
        for actor_name in self.actor_names:
            # get_entity_state
            req = GetEntityState.Request()
            req.name = actor_name
            future = self.get_state_cli.call_async(req)
            while not future.done():
                time.sleep(0.001)
            if not future.result() or not future.result().success:
                continue
            state = future.result().state

            x = state.pose.position.x
            y = state.pose.position.y

            # randomize velocity
            if random.random() < 0.1:
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(0.0, 1.0)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)
                self.actor_vel[actor_name] = [vx, vy]

            vx, vy = self.actor_vel[actor_name]
            new_x = x + vx*dt
            new_y = y + vy*dt
            # boundary check
            if new_x > 5.0: new_x = 5.0; vx *= -1
            if new_x < -5.0: new_x = -5.0; vx *= -1
            if new_y > 5.0: new_y = 5.0; vy *= -1
            if new_y < -5.0: new_y = -5.0; vy *= -1
            self.actor_vel[actor_name] = [vx, vy]

            set_req = SetEntityState.Request()
            set_req.state.name = actor_name
            set_req.state.pose.position.x = new_x
            set_req.state.pose.position.y = new_y
            set_req.state.pose.position.z = 0.0
            set_req.state.reference_frame = 'world'

            set_fut = self.set_state_cli.call_async(set_req)
            while not set_fut.done():
                time.sleep(0.001)


###############################################################################
# 2) GazeboCrowdEnv
###############################################################################
class GazeboCrowdEnv:
    def __init__(self, node: Node, robot_name='pioneer2dx', rate_hz=5.0):
        self.node = node
        self.robot_name = robot_name
        self.rate_hz = rate_hz

        self.get_state_cli = node.create_client(GetEntityState, '/get_entity_state')
        self.pause_cli = node.create_client(Empty, '/pause_physics')
        self.unpause_cli = node.create_client(Empty, '/unpause_physics')

        self.cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)

        self.node.get_logger().info("Waiting for /get_entity_state, /pause_physics, /unpause_physics ...")
        self.get_state_cli.wait_for_service()
        self.pause_cli.wait_for_service()
        self.unpause_cli.wait_for_service()

        self.actor_names = ['actor1','actor2']
        self.robot_radius = 0.3
        self.human_radius = 0.3
        self.goal_x = 3.0
        self.goal_y = 3.0
        self.goal_threshold = 0.3

        self.node.get_logger().info("GazeboCrowdEnv init done.")

    def reset(self):
        # pause
        pause_req = Empty.Request()
        pfut = self.pause_cli.call_async(pause_req)
        while not pfut.done():
            time.sleep(0.001)

        # unpause
        unpause_req = Empty.Request()
        ufut = self.unpause_cli.call_async(unpause_req)
        while not ufut.done():
            time.sleep(0.001)

        return self._get_observation()

    def step(self, action):
        twist = Twist()
        twist.linear.x = action[0]
        twist.angular.z = action[1]
        self.cmd_pub.publish(twist)

        dt = 1.0 / self.rate_hz
        t0 = time.time()
        while time.time() - t0 < dt:
            time.sleep(0.001)

        obs = self._get_observation()
        reward, done = self._compute_reward_done(obs)
        return obs, reward, done, {}

    def _get_observation(self):
        obs = {}
        # robot
        st_robot = self._sync_get_entity_state(self.robot_name)
        if st_robot:
            rx = st_robot.pose.position.x
            ry = st_robot.pose.position.y
            rvx = st_robot.twist.linear.x
            rvy = st_robot.twist.linear.y
        else:
            rx, ry, rvx, rvy = (0,0,0,0)
        obs['robot'] = (rx, ry, rvx, rvy)

        # actors
        actor_list = []
        for a in self.actor_names:
            st_actor = self._sync_get_entity_state(a)
            if st_actor:
                ax = st_actor.pose.position.x
                ay = st_actor.pose.position.y
                avx = st_actor.twist.linear.x
                avy = st_actor.twist.linear.y
            else:
                ax, ay, avx, avy = (0,0,0,0)
            actor_list.append((ax, ay, avx, avy))

        obs['actors'] = actor_list
        return obs

    def _sync_get_entity_state(self, name):
        req = GetEntityState.Request()
        req.name = name
        fut = self.get_state_cli.call_async(req)
        while not fut.done():
            time.sleep(0.001)

        res = fut.result()
        if res and res.success:
            return res.state
        return None

    def _compute_reward_done(self, obs):
        rx, ry, rvx, rvy = obs['robot']
        done = False
        reward = -0.01

        # check collision
        for (ax, ay, avx, avy) in obs['actors']:
            dist = math.sqrt((rx-ax)**2 + (ry-ay)**2)
            if dist < (self.robot_radius + self.human_radius):
                reward -= 15.
                done = True
                return reward, done

        # check goal
        dist2goal = math.sqrt((rx - self.goal_x)**2 + (ry - self.goal_y)**2)
        if dist2goal < self.goal_threshold:
            reward += 10.
            done = True
        return reward, done


###############################################################################
# 3) 加载策略 (ini + pth)
###############################################################################
def load_crowdnav_policy(config_file, model_file, device='cpu'):
    from crowd_nav.policy.policy_factory import policy_factory

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found")

    parser = configparser.ConfigParser()
    parser.read(config_file)
    if 'rl' not in parser.sections():
        raise ValueError("No [rl] section in policy.config")
    if 'policy' not in parser['rl']:
        raise ValueError("No 'policy' key in [rl]")

    policy_name = parser['rl']['policy']
    if policy_name not in policy_factory:
        raise ValueError(f"Unknown policy {policy_name} in policy_factory")

    policy_cls = policy_factory[policy_name]
    policy = policy_cls()
    # configure
    policy.configure(parser)
    policy.set_phase('test')
    policy.set_device(device)

    ckpt = torch.load(model_file, map_location=device)
    if policy.model is None:
        raise RuntimeError("policy.model is None after configure()!")
    policy.model.load_state_dict(ckpt)
    policy.model.eval()

    print(f"[load_crowdnav_policy] Loaded policy '{policy_name}' from {model_file}")
    return policy, policy_name


###############################################################################
# 4) PolicyWrapper: obs -> JointState -> policy.predict
###############################################################################
class CrowdNavPolicyWrapper:
    def __init__(self, policy):
        self.policy = policy
        self.robot_radius = 0.3
        self.human_radius = 0.3
        self.goal_x = 3.0
        self.goal_y = 3.0

    def predict(self, obs):
        rx, ry, rvx, rvy = obs['robot']
        robot_state = FullState(
            px=rx, py=ry, vx=rvx, vy=rvy,
            radius=self.robot_radius,
            theta=0.0,
            gx=self.goal_x, gy=self.goal_y,
            v_pref=1.0
        )

        human_states = []
        for (ax, ay, avx, avy) in obs['actors']:
            h = ObservableState(px=ax, py=ay, vx=avx, vy=avy, radius=self.human_radius)
            human_states.append(h)

        joint_state = JointState(robot_state, human_states)

        action = self.policy.predict(joint_state)  # maybe returns ActionXY or (vx, vy)
        # unify
        if hasattr(action, 'vx') and hasattr(action, 'vy'):
            vx, vy = action.vx, action.vy
        else:
            vx, vy = action[0], action[1]

        lin = math.sqrt(vx*vx + vy*vy)
        ang = math.atan2(vy, vx)
        k_w = 1.0
        angular_vel = k_w * ang

        max_v = 0.5
        max_w = 1.0
        if lin > max_v:
            lin = max_v
        if abs(angular_vel) > max_w:
            angular_vel = max_w * math.copysign(1.0, angular_vel)

        return (lin, angular_vel)


###############################################################################
# 5) main: 使用多线程executor
###############################################################################
def main(args=None):
    rclpy.init(args=args)

    # 创建节点
    actor_node = RandomActorController(actor_names=['actor1','actor2'], rate_hz=2.0)
    env_node = rclpy.create_node('gazebo_rl_env_node')

    # 多线程执行器
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(actor_node)
    executor.add_node(env_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 用 env_node 创建环境
    env = GazeboCrowdEnv(node=env_node, robot_name='pioneer2dx', rate_hz=5.0)

    # 加载策略
    config_file = 'data/output/policy.config'
    model_file = 'data/output/rl_model.pth'
    policy_obj, policy_name = load_crowdnav_policy(config_file, model_file, device='cpu')
    policy = CrowdNavPolicyWrapper(policy_obj)
    print(f"Loaded policy={policy_name}, start episodes...")

    # 训练回合
    num_episodes = 3
    for ep in range(num_episodes):
        print(f"=== Episode {ep} ===")
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < 50:
            action_cmd = policy.predict(obs)  # (v, w)
            obs, reward, done, _ = env.step(action_cmd)
            ep_reward += reward
            step += 1

        print(f"Episode {ep} done, steps={step}, reward={ep_reward}")

    # 结束
    executor.shutdown()
    spin_thread.join()
    actor_node.destroy_node()
    env_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
