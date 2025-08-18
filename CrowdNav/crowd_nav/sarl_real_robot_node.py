#!/usr/bin/env python3
"""
Demonstration script showing how to:
1) Use different QoS for /scan (BEST_EFFORT) and /odom (RELIABLE).
2) Fill a 150-dim state vector to match sarl.py's requirement (mlp1_dims=150).
   Here we do a naive approach:
   - first 100 dims from LaserScan
   - last 50 dims = 0
   - shape => (batch=1, humans=1, 150)
In real usage, you must replicate the exact structure from training:
some dims for self_state, laser, etc. to truly match your trained model.
"""

import os
import sys
import math
import configparser

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# QoS
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import torch
import numpy as np

from crowd_nav.policy.policy_factory import policy_factory

class SarlRealRobotNode(Node):
    def __init__(self):
        print(">>> Entered SarlRealRobotNode.__init__()")
        super().__init__('sarl_real_robot_node')

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(script_dir, 'data', 'output', 'rl_model.pth')
        self.policy_config_path = os.path.join(script_dir, 'configs', 'policy.config')

        self.get_logger().info(f"Will load RL state_dict from {self.model_path}")
        self.get_logger().info(f"Will read policy config from {self.policy_config_path}")

        # 1) 构造并配置 sarl policy
        self.policy = policy_factory['sarl']()
        if os.path.exists(self.policy_config_path):
            cf = configparser.RawConfigParser()
            cf.read(self.policy_config_path)
            self.policy.configure(cf)
            self.get_logger().info(f"Policy configured from {self.policy_config_path}")
        else:
            self.get_logger().warn("Policy config file not found, using defaults...")

        # 2) 加载模型
        self.model = self.policy.get_model()
        device = torch.device('cpu')
        try:
            checkpoint = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.get_logger().info("Policy model loaded & set to eval.")
        except Exception as e:
            self.get_logger().error(f"Failed to load state_dict: {e}")
            sys.exit(1)

        # 3) QoS
        #    /scan => BEST_EFFORT, /odom => RELIABLE
        qos_scan = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        qos_odom = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_scan)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_odom)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.last_scan = None
        self.last_odom = None

        self.timer = self.create_timer(0.5, self.main_loop)
        print(">>> Finished init setup. Ready to spin.")

    def scan_callback(self, msg):
        self.last_scan = msg

    def odom_callback(self, msg):
        self.last_odom = msg

    def main_loop(self):
        print(">>> main_loop tick")

        # 没收到scan或odom就跳过
        if self.last_scan is None or self.last_odom is None:
            print(">>> Odom or Scan is None => skip this loop.")
            return

        # -------------- STEP1: 取激光 --------------
        scan_arr = np.array(self.last_scan.ranges, dtype=np.float32)
        scan_arr[scan_arr == float('inf')] = 5.0
        # 只拿前100
        if scan_arr.size < 100:
            print(f">>> scan_arr.size={scan_arr.size}, skip loop")
            return
        scan_arr = scan_arr[:100]

        # -------------- STEP2: 拼成150维 --------------
        # sarl.py => "mlp1_dims = 150, 100" => expects input=150
        # 这里演示: 先把激光的100维放前面, 剩下50维用0填充
        state_1d = np.zeros((150,), dtype=np.float32)
        state_1d[:100] = scan_arr

        # -------------- STEP3: 构造 (batch=1, humans=1, 150) --------------
        # sarl code => state[:,0,:self.self_state_dim] => robot
        # if we had 1 human, it'd be shape [1,2,150].
        # here we do "no actual human" => #humans=1 means just robot row
        state_np = np.zeros((1, 1, 150), dtype=np.float32)
        state_np[0, 0, :] = state_1d

        print(f">>> final state shape: {state_np.shape}")  # (1,1,150)

        # -------------- STEP4: forward --------------
        state_tensor = torch.from_numpy(state_np)
        with torch.no_grad():
            value = self.model(state_tensor)  # shape probably [1,1], or [1,xxx]
        # sarl ends with mlp3_dims=150,100,100,1 => output shape [1,1]
        # Not actually v,w => It's a "value" (a single scalar) if from code
        # But let's pretend we interpret it as (v,w)...

        # For demonstration, let's pretend the net outputs 2-dim => we do:
        # Actually sarl code => mlp3_dims=150,100,100,1 => 1 dim => value[0,0]? 
        # We'll forcibly interpret the last 2 dims for "v,w" => just a hack.
        # You might actually do: v=some function(value)...

        # HACK: let's expand to 2 dims for demonstration
        if value.shape[-1] == 1:
            # artificially forging v,w = (0.1, 0)
            v, w = 0.1, 0.0
            print(">>> WARNING: sarl output is 1-dim VALUE, not velocity! We'll just do v=0.1, w=0.0 as placeholder.")
        else:
            # if it were shape [1,2], do something like:
            v = value[0,0].item()
            w = value[0,1].item()

        # -------------- STEP5: 发布cmd_vel --------------
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)
        print(f">>> published cmd_vel => v={v:.3f}, w={w:.3f}")

def main(args=None):
    print(">>> SCRIPT STARTED: sarl_real_robot_node.py")
    print(">>> Entered main()")
    rclpy.init(args=args)
    node = SarlRealRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
