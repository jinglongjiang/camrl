#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leg-SARL  v2.21  (2025-05-01)
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

# ───── 参数区 ────────────────────────────────
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

# ────── 节点 ─────────────────────────────────
class LegSARLNode(Node):
    def __init__(self):
        super().__init__('leg_sarl_node')
        self.get_logger().set_level(LoggingSeverity.INFO)

        # pub / sub
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LegArray , '/detected_leg_clusters', self._cb_legs , 10)
        self.create_subscription(LaserScan, '/scan'                  , self._cb_scan , 10)
        self.create_subscription(Odometry , '/odom'                  , self._cb_odom , 10)

        # 状态
        self.legs = self.scan = None
        self.x = self.y = self.yaw = 0.0
        self.x0 = self.y0 = self.yaw0 = None
        self.goal_x = self.goal_y = None
        self.arrive_ts = None

        # 滤波变量
        self.prev_v = self.prev_omega = 0.0
        self.prev_g_ang  = 0.0
        self.prev_gain_w = 1.0
        self.surf_min_f  = DETECT_RANGE
        self.in_avoid    = False

        # SARL
        cfg = configparser.RawConfigParser(); cfg.read(POLICY_CFG)
        self.policy = policy_factory['sarl'](); self.policy.configure(cfg)
        self.policy.set_device(torch.device('cpu')); self.policy.set_phase('test')
        self.policy.get_model().load_state_dict(torch.load(MODEL_PTH, map_location='cpu'))
        self.policy.get_model().eval()
        self.get_logger().info('✔ SARL model ready')

        self.create_timer(DT, self._tick)
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())

    # ─── 回调 ──────────────────────────────
    def _cb_legs(self, msg):   self.legs = msg
    def _cb_scan(self, msg):   self.scan = msg
    def _cb_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if self.yaw0 is None:
            self.x0, self.y0, self.yaw0 = self.x, self.y, self.yaw
            self._reset_goal()

    # ─── 工具 ───────────────────────────────
    def _reset_goal(self):
        self.goal_x = self.x + GOAL_DIST*math.cos(self.yaw)
        self.goal_y = self.y + GOAL_DIST*math.sin(self.yaw)
        self.get_logger().info(f'➡️  New goal ({self.goal_x:.2f},{self.goal_y:.2f})')

    @staticmethod
    def _norm(a):  return (a+math.pi)%(2*math.pi)-math.pi

    @staticmethod
    def _largest_gap(angs):
        s=sorted(angs); ext=s+[s[0]+2*math.pi]; best=mid=0
        for i in range(len(s)):
            g=ext[i+1]-ext[i]
            if g>best: best,mid=g,ext[i]+g/2
        return (mid+math.pi)%(2*math.pi)-math.pi

    def _prefer_front_gap(self, angs):
        if len(angs) < 2:
            return 0.0
        front = [a for a in angs if abs(self._norm(a)) < 2.094]  # ±120°
        if len(front) >= 2:
            mid = self._largest_gap(front)
            if (max(front)-min(front)) < 2*math.pi-1e-3:
                return self._norm(mid)
        return self._largest_gap(angs)

    # ─── 主循环 ──────────────────────────────
    def _tick(self):
        if self.yaw0 is None:
            return

        humans, angs = [], []
        surf_raw = float('inf')
        d_front  = float('inf')
        free_angle = 0          # 用于墙体急停判定

        # -------- 聚合一次遍历 (legs + laser) --------
        if self.legs:
            for l in self.legs.legs:
                x,y = l.position.x,l.position.y
                d = math.hypot(x,y)
                if d>DETECT_RANGE: continue
                a = math.atan2(y,x)
                r = ROBOT_R
                surf_raw = min(surf_raw, d-r)
                if abs(a)<FRONT_ANG: d_front=min(d_front,d-r)
                angs.append(a)
                humans.append(ObservableState(px=x,py=y,
                               vx=-0.1*math.cos(a),vy=-0.1*math.sin(a),radius=r))

        if self.scan:
            rs=self.scan.ranges; a0=self.scan.angle_min; da=self.scan.angle_increment
            n=len(rs); i=0
            while i<n:
                if not math.isfinite(rs[i]) or rs[i]>=DETECT_RANGE:
                    i+=1; free_angle+=da; continue
                # 簇开始
                j=i
                seg=[]
                while j<n and math.isfinite(rs[j]) and rs[j]<DETECT_RANGE:
                    seg.append((j,rs[j])); j+=1
                idxs=[k for k,_ in seg]
                dist=sum(r for _,r in seg)/len(seg)
                ang = a0 + (idxs[0]+idxs[-1])/2*da
                width = len(seg)*da*dist
                rad=width/2
                surf_raw=min(surf_raw,dist-rad)
                if abs(ang)<FRONT_ANG: d_front=min(d_front,dist-rad)
                angs.append(ang)
                humans.append(ObservableState(px=dist*math.cos(ang),
                               py=dist*math.sin(ang),vx=0,vy=0,radius=rad))
                i=j
            # wrap-around free-angle 已计入

        # ---------- 墙体急停判定 ----------
        wall_blocked = math.degrees(free_angle) < (360-WALL_DEG)

        # ---------- 滤波 ----------
        self.surf_min_f = SURF_ALPHA*surf_raw + (1-SURF_ALPHA)*self.surf_min_f
        surf_min = self.surf_min_f

        if   not angs:           g_raw=0.0
        elif len(angs)==1:       g_raw=angs[0]+math.pi/2
        else:                    g_raw=self._prefer_front_gap(angs)
        g_ang = GAP_ALPHA*g_raw + (1-GAP_ALPHA)*self.prev_g_ang
        self.prev_g_ang = g_ang

        dxg=self.goal_x-self.x; dyg=self.goal_y-self.y
        robot_state=FullState(0,0,0,0,ROBOT_R,dxg,dyg,CRUISE_SPEED,self.yaw)

        enter_d=ENTER_SURF+0.5*abs(self.prev_v)
        exit_d =EXIT_SURF +0.5*abs(self.prev_v)
        stop_d =STOP_SURF +0.3*abs(self.prev_v)
        self.in_avoid = surf_min<exit_d if self.in_avoid else surf_min<enter_d

        # --------- 模式切换 ----------
        cmd = Twist(); mode=''

        if math.hypot(dxg, dyg)<GOAL_TOL:
            mode='GOAL'
            if self.arrive_ts is None: self.arrive_ts=time.time()
            if time.time()-self.arrive_ts>GOAL_SETTLE: self.arrive_ts=None; self._reset_goal()
            v=omega=0.0

        elif wall_blocked or d_front<stop_d:
            v=omega=0.0; mode='STOP'

        elif self.in_avoid and humans:
            mode='AVOID'
            act=self.policy.predict(JointState(robot_state,humans))
            if act.vx<0: act.vx,act.vy=-act.vx,-act.vy   # 禁止后退
            heading=max(-math.pi/2,min(math.pi/2,math.atan2(act.vy,act.vx)))
            gain_raw=1+max(h.radius for h in humans)/0.5
            gain_w=GAIN_ALPHA*gain_raw+(1-GAIN_ALPHA)*self.prev_gain_w
            self.prev_gain_w=gain_w
            omega_d=TURN_GAIN*gain_w*heading/DT
            omega_d=max(min(omega_d,OMEGA_MAX),-OMEGA_MAX)
            omega=self.prev_omega+TURN_ALPHA*(omega_d-self.prev_omega)
            v_nom=max(MIN_SPEED,CRUISE_SPEED*(d_front-stop_d)/(enter_d-stop_d))
            v=min(v_nom*math.cos(abs(heading)), CRUISE_SPEED*SPEED_CAP)

        else:
            mode='CRUISE'
            cte=-math.sin(self.yaw0)*(self.x-self.x0)+math.cos(self.yaw0)*(self.y-self.y0)
            head_err=self._norm(self.yaw0-self.yaw)
            omega_c=K_HEADING*head_err-K_CTE*cte
            omega=self.prev_omega+TURN_ALPHA*(omega_c-self.prev_omega)
            if abs(omega)<OMEGA_EPS: omega=0.0
            v=CRUISE_SPEED*SPEED_CAP

        # ---------- 限幅 & 发送 ----------
        dv=max(min(v-self.prev_v,MAX_DV),-MAX_DV)
        domega=max(min(omega-self.prev_omega,MAX_DOMEGA),-MAX_DOMEGA)
        v=self.prev_v+dv; omega=self.prev_omega+domega
        omega=max(min(omega,OMEGA_MAX),-OMEGA_MAX)
        if abs(omega)<OMEGA_EPS: omega=0.0
        self.prev_v, self.prev_omega = v, omega
        cmd.linear.x=float(v); cmd.angular.z=float(omega); self.pub_cmd.publish(cmd)

        self.get_logger().debug(
            f"{mode}|surf={surf_min:.2f}|front={d_front:.2f}|v={v:.2f}|ω={omega:.2f}|hum={len(humans)}"
        )

    # ─── 结束 ───────────────────────────────
    def _shutdown(self):
        self.pub_cmd.publish(Twist())
        rclpy.shutdown(); sys.exit(0)

# ─── main ────────────────────────────────────
def main():
    rclpy.init(); rclpy.spin(LegSARLNode()); rclpy.shutdown()

if __name__ == '__main__':
    main()
