#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import random
import time
import threading

from gazebo_msgs.srv import GetEntityState, SetEntityState
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

############################################################
# 1) RandomActorController
############################################################
class RandomActorController(Node):
    """
    Moves actor1, actor2 around using /get_entity_state, /set_entity_state
    in a synchronous manner to ensure we get real data.
    """

    def __init__(self, actor_names=['actor1','actor2'], rate_hz=2.0):
        super().__init__('random_actor_controller')
        self.actor_names = actor_names
        self.rate_hz = rate_hz

        # Create clients to actual service names (no /gazebo prefix)
        self.get_state_cli = self.create_client(GetEntityState, '/get_entity_state')
        self.set_state_cli = self.create_client(SetEntityState, '/set_entity_state')

        self.get_logger().info("Waiting for get_entity_state & set_entity_state services...")

        # Wait for service
        for cli in [self.get_state_cli, self.set_state_cli]:
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"waiting for {cli.srv_name}...")

        self.get_logger().info("RandomActorController started.")
        self.actor_vel = {}
        for a in actor_names:
            self.actor_vel[a] = [0.0, 0.0]

        # Timer
        self.timer = self.create_timer(1.0/self.rate_hz, self.timer_callback)

    def get_state_sync(self, name):
        """
        Use spin_until_future_complete to ensure we get the result.
        """
        req = GetEntityState.Request()
        req.name = name
        future = self.get_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.done() and future.result() and future.result().success:
            return future.result().state
        else:
            return None

    def set_state_sync(self, name, x, y):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.0
        req.state.reference_frame = 'world'
        future = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.done() and future.result():
            return future.result().success
        return False

    def timer_callback(self):
        dt = 1.0/self.rate_hz
        for actor_name in self.actor_names:
            state = self.get_state_sync(actor_name)
            if not state:
                # skip if cannot get
                continue
            x = state.pose.position.x
            y = state.pose.position.y

            # random velocity sometimes
            if random.random() < 0.1:
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(0.0, 1.0)
                vx = speed*math.cos(angle)
                vy = speed*math.sin(angle)
                self.actor_vel[actor_name] = [vx, vy]

            vx, vy = self.actor_vel[actor_name]
            new_x = x + vx*dt
            new_y = y + vy*dt
            # bounding
            if new_x > 5.0: new_x=5.0; vx*=-1
            if new_x < -5.0: new_x=-5.0; vx*=-1
            if new_y > 5.0: new_y=5.0; vy*=-1
            if new_y < -5.0: new_y=-5.0; vy*=-1
            self.actor_vel[actor_name] = [vx, vy]

            self.set_state_sync(actor_name, new_x, new_y)


############################################################
# 2) GazeboCrowdEnv
############################################################
class GazeboCrowdEnv:
    """
    A Gym-like environment. We'll use synchronous get_entity_state & debug prints
    to see why we might be getting immediate collision.
    """

    def __init__(self, node: Node, robot_name='pioneer2dx', rate_hz=5.0):
        self.node = node
        self.robot_name = robot_name
        self.rate_hz = rate_hz

        self.get_state_cli = self.node.create_client(GetEntityState, '/get_entity_state')
        self.pause_cli = self.node.create_client(Empty, '/pause_physics')
        self.unpause_cli = self.node.create_client(Empty, '/unpause_physics')

        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        self.node.get_logger().info("Waiting for get_entity_state, pause_physics, unpause_physics...")
        for cli in [self.get_state_cli, self.pause_cli, self.unpause_cli]:
            while not cli.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().info(f"waiting for {cli.srv_name}...")

        # these names must match /model_states or your actor naming
        self.actor_names = ['actor1','actor2']
        self.robot_radius = 0.3
        self.human_radius = 0.3
        self.goal_x = 3.0
        self.goal_y = 3.0
        self.goal_threshold = 0.3

        self.node.get_logger().info("GazeboCrowdEnv init done.")

    def _sync_get_entity_state(self, name):
        req = GetEntityState.Request()
        req.name = name
        future = self.get_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)
        if future.done():
            result = future.result()
            if result and result.success:
                return result.state
        return None

    def reset(self):
        self._pause_sim()
        # optionally set robot to (0,0) or random, skipping for brevity
        self._unpause_sim()
        obs = self._get_observation()
        return obs

    def step(self, action):
        # publish cmd
        twist = Twist()
        twist.linear.x = action[0]
        twist.angular.z = action[1]
        self.cmd_pub.publish(twist)

        dt = 1.0/self.rate_hz
        t_start = time.time()
        while time.time()-t_start < dt:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_observation()
        reward, done = self._compute_reward_done(obs)
        return obs, reward, done, {}

    def _pause_sim(self):
        req = Empty.Request()
        self.pause_cli.call_async(req)
        rclpy.spin_once(self.node, timeout_sec=0.2)

    def _unpause_sim(self):
        req = Empty.Request()
        self.unpause_cli.call_async(req)
        rclpy.spin_once(self.node, timeout_sec=0.2)

    def _get_observation(self):
        obs = {}
        # robot
        robot_state = self._sync_get_entity_state(self.robot_name)
        if robot_state:
            rx = robot_state.pose.position.x
            ry = robot_state.pose.position.y
            rvx = robot_state.twist.linear.x
            rvy = robot_state.twist.linear.y
        else:
            rx, ry, rvx, rvy = (0,0,0,0)

        obs['robot'] = (rx, ry, rvx, rvy)

        # actors
        actor_list = []
        for a in self.actor_names:
            st = self._sync_get_entity_state(a)
            if st:
                ax = st.pose.position.x
                ay = st.pose.position.y
                avx = st.twist.linear.x
                avy = st.twist.linear.y
            else:
                ax, ay, avx, avy = (0,0,0,0)
            actor_list.append((ax, ay, avx, avy))

        obs['actors'] = actor_list
        return obs

    def _compute_reward_done(self, obs):
        """
        debug print the positions to see the actual distances
        """
        rx, ry, rvx, rvy = obs['robot']
        self.node.get_logger().info(f"robot=({rx:.2f},{ry:.2f}), vs. goal=({self.goal_x},{self.goal_y})")

        # check collision
        done = False
        reward = -0.01  # step cost
        for (ax, ay, avx, avy) in obs['actors']:
            dist = math.sqrt( (rx-ax)**2 + (ry-ay)**2 )
            self.node.get_logger().info(f"actor=({ax:.2f},{ay:.2f}), dist={dist:.2f}")
            if dist < (self.robot_radius + self.human_radius):
                reward -= 15.
                self.node.get_logger().warn(f"Collision detected with dist={dist:.3f} < 0.6!")
                done = True
                return reward, done

        # check goal
        dist2goal = math.sqrt( (rx-self.goal_x)**2 + (ry-self.goal_y)**2 )
        if dist2goal < self.goal_threshold:
            reward += 10.
            done = True
            self.node.get_logger().info(f"Reached goal, dist2goal={dist2goal:.3f}")

        return reward, done


############################################################
# 3) main
############################################################
def main(args=None):
    rclpy.init(args=args)

    # Start random actor node
    actor_node = RandomActorController(actor_names=['actor1','actor2'], rate_hz=2.0)
    actor_thread = threading.Thread(target=rclpy.spin, args=(actor_node,), daemon=True)
    actor_thread.start()

    # Env node
    env_node = rclpy.create_node('gazebo_rl_env_node')
    env = GazeboCrowdEnv(node=env_node, robot_name='pioneer2dx', rate_hz=5.0)

    # dummy policy
    class DummyPolicy:
        def predict(self, obs):
            # e.g. forward
            return (0.3, 0.0)
    policy = DummyPolicy()

    num_episodes = 3
    for ep in range(num_episodes):
        env_node.get_logger().info(f"=== Episode {ep} ===")
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done and step_count < 50:
            obs, reward, done, info = env.step( policy.predict(obs) )
            ep_reward += reward
            step_count += 1

        env_node.get_logger().info(f"Episode {ep} ends, step={step_count}, reward={ep_reward:.2f}")

    env_node.destroy_node()
    actor_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
