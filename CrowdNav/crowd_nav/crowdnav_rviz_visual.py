#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

from gazebo_msgs.srv import GetEntityState
from visualization_msgs.msg import Marker, MarkerArray

class RvizVisualizerNode(Node):
    def __init__(self):
        super().__init__('rviz_visualizer_node')

        # we call get_entity_state to get robot/actors pose
        self.get_state_cli = self.create_client(GetEntityState, '/get_entity_state')
        self.robot_name = 'pioneer2dx'
        self.actor_names = ['actor1','actor2']

        self.marker_pub = self.create_publisher(MarkerArray, '/crowdnav_markers', 10)

        self.get_logger().info("Waiting for /get_entity_state service...")
        while not self.get_state_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("waiting for /get_entity_state...")

        self.timer_period = 0.2  # 5Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info("RvizVisualizerNode started.")

    def get_entity_pose(self, name):
        from gazebo_msgs.srv import GetEntityState
        req = GetEntityState.Request()
        req.name = name
        future = self.get_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        if future.done():
            res = future.result()
            if res and res.success:
                px = res.state.pose.position.x
                py = res.state.pose.position.y
                pz = res.state.pose.position.z
                return (px, py, pz)
        return (None, None, None)

    def timer_callback(self):
        # gather positions
        rx, ry, rz = self.get_entity_pose(self.robot_name)
        if rx is None:
            # not found or service fail
            return
        actor_positions = []
        for an in self.actor_names:
            ax, ay, az = self.get_entity_pose(an)
            actor_positions.append((an, ax, ay, az))

        ma = MarkerArray()

        # Robot marker
        robot_marker = Marker()
        robot_marker.header.frame_id = "world"
        robot_marker.header.stamp = self.get_clock().now().to_msg()
        robot_marker.id = 0
        robot_marker.type = Marker.SPHERE
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = rx
        robot_marker.pose.position.y = ry
        robot_marker.pose.position.z = 0.1
        robot_marker.scale.x = 0.4
        robot_marker.scale.y = 0.4
        robot_marker.scale.z = 0.4
        # color: blue
        robot_marker.color.r = 0.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 1.0
        robot_marker.color.a = 1.0
        robot_marker.lifetime.sec = 1
        ma.markers.append(robot_marker)

        # Actors marker
        mid = 1
        for (an, ax, ay, az) in actor_positions:
            actor_marker = Marker()
            actor_marker.header.frame_id = "world"
            actor_marker.header.stamp = self.get_clock().now().to_msg()
            actor_marker.id = mid
            mid+=1
            actor_marker.type = Marker.SPHERE
            actor_marker.action = Marker.ADD
            if ax is not None:
                actor_marker.pose.position.x = ax
                actor_marker.pose.position.y = ay
                actor_marker.pose.position.z = 0.1
            else:
                actor_marker.pose.position.x = 999
                actor_marker.pose.position.y = 999
                actor_marker.pose.position.z = 0
            actor_marker.scale.x = 0.4
            actor_marker.scale.y = 0.4
            actor_marker.scale.z = 0.4
            # color: red
            actor_marker.color.r = 1.0
            actor_marker.color.g = 0.0
            actor_marker.color.b = 0.0
            actor_marker.color.a = 1.0
            actor_marker.lifetime.sec=1
            ma.markers.append(actor_marker)

        self.marker_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = RvizVisualizerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()
