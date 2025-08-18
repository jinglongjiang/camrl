# sarl_real_navigation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import torch
import numpy as np
# 载入你自己的sarl网络类, utils
from sarl_network import SARLNetwork  # 只是示例导入

class SARLNode(Node):
    def __init__(self):
        super().__init__('sarl_node')
        # 加载模型
        self.model = SARLNetwork(...)
        self.model.load_state_dict(torch.load('models/sarl_model.pth'))
        self.model.eval()

        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_data = None
        self.odom_data = None

    def scan_cb(self, msg):
        self.scan_data = msg.ranges
        self.update_cmd()

    def odom_cb(self, msg):
        self.odom_data = msg.pose.pose
        self.update_cmd()

    def update_cmd(self):
        if self.scan_data is None or self.odom_data is None:
            return

        # 转成你的网络输入格式
        input_data = ...
        # 前向推理
        with torch.no_grad():
            v, w = self.model(input_data)  # 比如输出线速度v, 角速度w

        # 发布cmd_vel
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub_cmd_vel.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SARLNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
