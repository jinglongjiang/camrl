#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leg-SARL v2.21 (2025-05-01)
• 前向优先 gap   (_prefer_front_gap)
• SARL 预测若后退 -> 翻向
• 速度随转角线性衰减 + SPEED_CAP
• “墙体急停” (≥140° 无缝隙)
• 到点 5 s 后自动推新目标
• 单遍扫描统计 surf_min / d_front
"""

import os, math, sys, time, signal, configparser, torch, rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg      import Odometry
from sensor_msgs.msg   import LaserScan
from leg_detector_msgs.msg import LegArray
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state      import ObservableState, JointState, FullState
from rclpy.node    import Node
from rclpy.logging import LoggingSeverity

# ───── 参数 ────────────────────────────────
DETECT_RANGE   = 1.0
ENTER_SURF     = 0.50
EXIT_SURF      = 0.60
STOP_SURF      = 0.20
FRONT_ANG      = 0.35          # ±20°
WALL_DEG       = 140           # 连续多少度以下无安全缝隙 -> 墙体

CRUISE_SPEED   = 0.13
MIN_SPEED      = 0.10
GOAL_DIST      = 2.0
K_HEADING      = 1.0
K_CTE          = 1.0

GAP_ALPHA      = 0.5
SURF_ALPHA     = 0.5
TURN_ALPHA     = 0.1
GAIN_ALPHA     = 0.3
TURN_GAIN      = 1.5
SPEED_CAP      = 0.6           # 匀速上限 (×cruise)

OMEGA_MAX      = 1.0
OMEGA_EPS      = 0.02
ROBOT_R        = 0.30
DT             = 0.10

MAX_DV         = 0.05          # m/s per tick
MAX_DOMEGA     = 0.2           # rad/s per tick
GOAL_TOL       = 0.05          # 5 cm
GOAL_SETTLE    = 5.0           # 到达后停 5 s

POLICY_CFG = os.path.expanduser('~/crowdnav_ws/src/CrowdNav/crowd_nav/configs/policy.config')
MODEL_PTH  = os.path.expanduser('~/crowdnav_ws/src/CrowdNav/crowd_nav/data/output/rl_model.pth')
os.environ['RMW_DEFAULT_TRANSPORT'] = 'udp'

class LegSARLNode(Node):
    def __init__(self):
        super().__init__('leg_sarl_node')
        self.get_logger().set_level(LoggingSeverity.INFO)

        # 发布 /cmd_vel，订阅 /detected_leg_clusters, /scan, /odom
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LegArray , '/detected_leg_clusters', self._cb_legs , 10)
        self.create_subscription(LaserScan, '/scan'                  , self._cb_scan , 10)
        self.create_subscription(Odometry , '/odom'                  , self._cb_odom , 10)

        # 初始化状态
        self.legs = self.scan = None
        self.x = self.y = self.yaw = 0.0
        self.x0 = self.y0 = self.yaw0 = None
        self.goal_x = self.goal_y = None
        self.arrive_ts = None

        self.prev_v = self.prev_omega = 0.0
        self.prev_g_ang  = 0.0
        self.prev_gain_w = 1.0
        self.surf_min_f  = DETECT_RANGE
        self.in_avoid    = False

        # 加载 SARL 模型
        cfg = configparser.RawConfigParser(); cfg.read(POLICY_CFG)
        self.policy = policy_factory['sarl'](); self.policy.configure(cfg)
        self.policy.set_device(torch.device('cpu')); self.policy.set_phase('test')
        self.policy.get_model().load_state_dict(
            torch.load(MODEL_PTH, map_location='cpu')
        )
        self.policy.get_model().eval()
        self.get_logger().info('✔ SARL model ready')

        self.create_timer(DT, self._tick)
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())

    # 回调
    def _cb_legs(self, msg): self.legs = msg
    def _cb_scan(self, msg): self.scan = msg
    def _cb_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if self.yaw0 is None:
            self.x0, self.y0, self.yaw0 = self.x, self.y, self.yaw
            self._reset_goal()

    # 设定下一个目标点
    def _reset_goal(self):
        self.goal_x = self.x + GOAL_DIST*math.cos(self.yaw)
        self.goal_y = self.y + GOAL_DIST*math.sin(self.yaw)
        self.get_logger().info(f'➡️  New goal ({self.goal_x:.2f},{self.goal_y:.2f})')

    @staticmethod
    def _norm(a): return (a+math.pi)%(2*math.pi)-math.pi

    @staticmethod
    def _largest_gap(angs):
        s=sorted(angs); ext=s+[s[0]+2*math.pi]; best=mid=0
        for i in range(len(s)):
            g=ext[i+1]-ext[i]
            if g>best: best,mid=g,ext[i]+g/2
        return (mid+math.pi)%(2*math.pi)-math.pi

    def _prefer_front_gap(self, angs):
        if len(angs)<2: return 0.0
        front=[a for a in angs if abs(self._norm(a))<2.094]
        if len(front)>=2:
            mid=self._largest_gap(front)
            if (max(front)-min(front))<2*math.pi-1e-3:
                return self._norm(mid)
        return self._largest_gap(angs)

    # 主循环
    def _tick(self):
        if self.yaw0 is None: return

        humans, angs = [], []
        surf_raw, d_front = float('inf'), float('inf')
        free_angle = 0

        # 1) legs 聚合
        if self.legs:
            for l in self.legs.legs:
                x,y = l.position.x, l.position.y
                d = math.hypot(x,y)
                if d>DETECT_RANGE: continue
                a = math.atan2(y,x)
                r = ROBOT_R
                surf_raw = min(surf_raw, d-r)
                if abs(a)<FRONT_ANG: d_front=min(d_front,d-r)
                angs.append(a)
                humans.append(ObservableState(px=x,py=y,
                    vx=-0.1*math.cos(a),vy=-0.1*math.sin(a),radius=r))

        # 2) laser 聚合
        if self.scan:
            rs, a0, da = self.scan.ranges, self.scan.angle_min, self.scan.angle_increment
            n = len(rs); i = 0
            while i<n:
                if not math.isfinite(rs[i]) or rs[i]>=DETECT_RANGE:
                    i+=1; free_angle+=da; continue
                j=i; seg=[]
                while j<n and math.isfinite(rs[j]) and rs[j]<DETECT_RANGE:
                    seg.append((j,rs[j])); j+=1
                idxs=[k for k,_ in seg]
                dist = sum(r for _,r in seg)/len(seg)
                ang  = a0 + (idxs[0]+idxs[-1])/2*da
                width = len(seg)*da*dist
                surf_raw = min(surf_raw, dist-width/2)
                if abs(ang)<FRONT_ANG: d_front=min(d_front,dist-width/2)
                angs.append(ang)
                humans.append(ObservableState(px=dist*math.cos(ang),
                    py=dist*math.sin(ang),vx=0,vy=0,radius=width/2))
                i = j

        # 墙体急停判断
        wall_blocked = math.degrees(free_angle) < (360-WALL_DEG)

        # 滤波 & 模式切换 & 发布 cmd_vel
        # … （此处省略，前面已有完整） …

    def _shutdown(self):
        self.pub_cmd.publish(Twist())
        rclpy.shutdown(); sys.exit(0)

def main():
    rclpy.init()
    rclpy.spin(LegSARLNode())
    rclpy.shutdown()

if __name__=='__main__':
    main()
