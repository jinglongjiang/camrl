#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# 重要：ROS2 Foxy下，pause_physics/unpause_physics/reset_world都用std_srvs/Empty
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetEntityState, SetEntityState  # 这个在Foxy有

from geometry_msgs.msg import Twist

class GazeboEnv(Node):
    def __init__(self):
        super().__init__('gazebo_env_node')

        # example: create client for pause_physics
        self.pause_cli = self.create_client(Empty, '/gazebo/pause_physics')
        self.unpause_cli = self.create_client(Empty, '/gazebo/unpause_physics')
        self.reset_world_cli = self.create_client(Empty, '/gazebo/reset_world')
        self.get_state_cli = self.create_client(GetEntityState, '/gazebo/get_entity_state')

        # wait for services
        for cli in [self.pause_cli, self.unpause_cli, self.reset_world_cli, self.get_state_cli]:
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'waiting for {cli.srv_name}...')

    def pause_sim(self):
        req = Empty.Request()
        self.pause_cli.call_async(req)

    def unpause_sim(self):
        req = Empty.Request()
        self.unpause_cli.call_async(req)

    def reset_world(self):
        req = Empty.Request()
        self.reset_world_cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboEnv()
    node.get_logger().info("Env node started. Pausing in 2s...")
    
    rclpy.spin_once(node, timeout_sec=2.0)
    node.pause_sim()
    node.get_logger().info("Paused physics.")
    
    rclpy.spin_once(node, timeout_sec=2.0)
    node.unpause_sim()
    node.get_logger().info("Unpaused physics.")

    rclpy.spin_once(node, timeout_sec=2.0)
    node.reset_world()
    node.get_logger().info("Reset world done.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
