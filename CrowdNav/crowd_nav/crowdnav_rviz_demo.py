#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math
import random
import time
import os
import datetime
import copy

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import torch

# CrowdNav / CrowdSim
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot

###############################################################################
# --------- 1) 全局可调参数 -----------
###############################################################################
TIME_LIMIT    = 25.0   # 超过这个秒数 => done
SUCCESS_REWARD= 3.5    # 到达目标一次性奖励
COLLISION_PENALTY= -0.2
DISCOMFORT_DIST= 0.2
DISCOMFORT_PENALTY_FACTOR= 0.3
OUT_OF_BOUND_PENALTY= -0.15  # 超出 [-5,5] 边界
PROGRESS_GAIN= 0.35     # 距目标进展奖励
HEADING_SMOOTHNESS_WEIGHT= 0.1  # 转向惩罚

###############################################################################
# 2) DeepcopyEnv: 真环境 => step => (ob, reward, done, info)
#    + heading smoothness penalty
###############################################################################
class DeepcopyEnv:
    def __init__(self, num_humans=10, dt=0.2):
        self.num_humans= num_humans
        self.dt= dt
        self.global_time=0.0
        self.time_limit= TIME_LIMIT

        # Robot init
        rx= random.uniform(-2,2)
        ry= random.uniform(-2,2)
        self.robot_x= rx
        self.robot_y= ry
        self.robot_vx=0.0
        self.robot_vy=0.0
        self.robot_r= 0.15
        self.v_pref= 0.8
        self.robot_theta=0.0

        # random goal
        gx= random.uniform(-5,5)
        gy= random.uniform(-5,5)
        while math.hypot(gx-rx, gy-ry)<1.0:
            gx= random.uniform(-5,5)
            gy= random.uniform(-5,5)
        self.goal_x= gx
        self.goal_y= gy

        # humans
        self.humans=[]
        for i in range(num_humans):
            hx= random.uniform(-2,2)
            hy= random.uniform(-2,2)
            hvx=0.0
            hvy=0.0
            rr= 0.15
            self.humans.append([hx, hy, hvx, hvy, rr])

        # shaping dist
        self.last_dist2goal= self.compute_dist2goal()

        # 记录上一时刻(vx,vy) 以计算heading变化
        self.last_robot_vx= 0.0
        self.last_robot_vy= 0.0

    def compute_dist2goal(self):
        return math.hypot(self.robot_x - self.goal_x,
                          self.robot_y - self.goal_y)

    def make_clone(self):
        return copy.deepcopy(self)

    def restore_from(self, cloned_env):
        self.__dict__= copy.deepcopy(cloned_env.__dict__)

    def step(self, vx, vy, update=True):
        """ => (ob, reward, done, info) with shaping + collision + boundary """
        old_dist2= self.compute_dist2goal()
        if update:
            self.global_time += self.dt

        # robot
        nx= self.robot_x + vx*self.dt
        ny= self.robot_y + vy*self.dt
        if update:
            self.robot_x= nx
            self.robot_y= ny
            self.robot_vx= vx
            self.robot_vy= vy

        newhum=[]
        for i,h in enumerate(self.humans):
            hx, hy, hvx, hvy, rr= h
            if random.random()<0.1:
                angle= random.uniform(0,2*math.pi)
                spd= random.uniform(0.0,0.6)
                hvx= spd*math.cos(angle)
                hvy= spd*math.sin(angle)
            hhx= hx+ hvx*self.dt
            hhy= hy+ hvy*self.dt
            # bounce
            if hhx>5: hhx=5; hvx=-hvx
            if hhx<-5: hhx=-5; hvx=-hvx
            if hhy>5: hhy=5; hvy=-hvy
            if hhy<-5: hhy=-5; hvy=-hvy

            if update:
                self.humans[i]= [hhx,hhy,hvx,hvy, rr]
            newhum.append( [hhx,hhy,hvx,hvy, rr] )

        reward=0.0
        done=False
        info="nothing"

        # collision / discomfort
        min_dist=9999.0
        for (hhx,hhy,hvx,hvy, rr2) in newhum:
            dist= math.hypot(nx-hhx, ny-hhy)
            if dist< min_dist:
                min_dist= dist
            margin= dist- (self.robot_r+ rr2)
            if margin<0:
                reward+= COLLISION_PENALTY
                info="collision"
                done=True
                break
            elif dist< (self.robot_r+ rr2+ DISCOMFORT_DIST):
                penalty= DISCOMFORT_PENALTY_FACTOR*( self.robot_r+ rr2+ DISCOMFORT_DIST - dist )
                reward-= penalty

        # boundary check
        if not done:
            if (nx>5 or nx<-5 or ny>5 or ny<-5):
                reward+= OUT_OF_BOUND_PENALTY
                info="out_of_bounds"
                done=True

        # success => dist2goal
        dist2= math.hypot(nx- self.goal_x, ny- self.goal_y)
        if not done:
            if dist2<0.2:
                reward+= SUCCESS_REWARD
                info="success"
                done=True

        # time limit
        if not done:
            if update and self.global_time> self.time_limit:
                info="time_up"
                done=True

        # progress shaping
        if not done:
            progress= (old_dist2- dist2)* PROGRESS_GAIN
            reward+= progress

        # heading smoothness => 与上一步速度方向的夹角
        old_speed= math.hypot(self.last_robot_vx, self.last_robot_vy)
        new_speed= math.hypot(vx, vy)
        if old_speed>1e-5 and new_speed>1e-5:
            old_dir= math.atan2(self.last_robot_vy, self.last_robot_vx)
            new_dir= math.atan2(vy, vx)
            diff= abs((new_dir- old_dir + math.pi)%(2*math.pi)- math.pi)
            # 0 <= diff <= pi
            heading_penalty= diff * HEADING_SMOOTHNESS_WEIGHT
            reward-= heading_penalty
        if update:
            self.last_robot_vx= vx
            self.last_robot_vy= vy

        ob=[]
        for (hhx,hhy, hvx,hvy, rr2) in newhum:
            ob.append((hhx,hhy,hvx,hvy,rr2))

        return ob, reward, done, info

###############################################################################
# 3) ClonableEnvForCadrl => onestep_lookahead => deep copy => step
###############################################################################
class ClonableEnvForCadrl:
    def __init__(self):
        self.base_env=None
    def set_base_env(self, base_env):
        self.base_env= base_env
    def onestep_lookahead(self, action):
        return self.step(action, update=False)
    def step(self, action, update=False):
        if self.base_env is None:
            # dummy
            return ([(0,0,0,0,0)],0.0,False,{})
        env_clone= self.base_env.make_clone()
        if hasattr(action,'vx') and hasattr(action,'vy'):
            vx, vy= action.vx, action.vy
        else:
            vx, vy= action[0], action[1]
        ob, reward, done, info= env_clone.step(vx, vy, update=True)
        return ob, reward, done, info

###############################################################################
# 4) MyTrainedPolicy => 载入cadrl => set_env(ClonableEnvForCadrl)
###############################################################################
class MyTrainedPolicy:
    def __init__(self,
                 policy_config='data/output/output_cadrl/policy.config',
                 model_file='data/output/output_cadrl/rl_model.pth',
                 device='cpu'):
        import configparser
        cfg= configparser.ConfigParser()
        cfg.read(policy_config)
        if 'rl' not in cfg or 'policy' not in cfg['rl']:
            raise ValueError("No [rl]/policy in config")
        policy_name= cfg['rl']['policy']

        from crowd_nav.policy.policy_factory import policy_factory
        policy_cls= policy_factory[policy_name]
        self.policy= policy_cls()
        self.policy.configure(cfg)
        self.policy.set_phase('test')
        self.policy.set_device(device)

        ckpt= torch.load(model_file,map_location=device)
        if self.policy.model is None:
            raise RuntimeError("policy.model is None after configure!")
        self.policy.model.load_state_dict(ckpt)
        self.policy.model.eval()

        print(f"[MyTrainedPolicy] loaded {policy_name} from {model_file}")

        self.env_wrapper= ClonableEnvForCadrl()
        self.policy.set_env(self.env_wrapper)

    def set_base_env(self, base_env):
        self.env_wrapper.set_base_env(base_env)

    def predict(self, robot_state, humans):
        # => (vx,vy)
        (rx, ry, rvx, rvy, rr, gx, gy, v_pref, theta)= robot_state
        r_st= FullState(rx, ry, rvx,rvy, rr, gx, gy, v_pref, theta)
        h_sts=[]
        for (hx,hy,hvx,hvy,hr) in humans:
            h_sts.append(ObservableState(hx, hy, hvx,hvy, hr))
        j_st= JointState(r_st, h_sts)

        action= self.policy.predict(j_st)
        if hasattr(action,'vx') and hasattr(action,'vy'):
            return (action.vx, action.vy)
        else:
            return (action[0], action[1])

###############################################################################
# 5) FullSim => 管理 DeepcopyEnv + MyTrainedPolicy => step
###############################################################################
class FullSim:
    def __init__(self, num_humans=10, dt=0.2,
                 policy_config='data/output/output_cadrl/policy.config',
                 model_file='data/output/output_cadrl/rl_model.pth',
                 device='cpu'):
        self.env= DeepcopyEnv(num_humans, dt)
        self.policy= MyTrainedPolicy(policy_config, model_file, device)
        self.policy.set_base_env(self.env)

        self.episode_id=0
        self.ep_step=0

    def reset_episode(self):
        self.episode_id+=1
        self.ep_step=0
        # 只换goal => keep robot
        rx, ry= (self.env.robot_x, self.env.robot_y)
        nx= random.uniform(-5,5)
        ny= random.uniform(-5,5)
        while math.hypot(nx-rx, ny-ry)<1.0:
            nx= random.uniform(-5,5)
            ny= random.uniform(-5,5)
        self.env.goal_x= nx
        self.env.goal_y= ny
        self.env.global_time= 0.0
        # 重置 shaping
        self.env.last_dist2goal= self.env.compute_dist2goal()

    def step(self):
        self.ep_step+=1
        # gather
        rx, ry= (self.env.robot_x, self.env.robot_y)
        rr= self.env.robot_r
        rvx, rvy= (self.env.robot_vx, self.env.robot_vy)
        gx, gy= (self.env.goal_x, self.env.goal_y)
        robot_st= (rx, ry, rvx, rvy, rr, gx, gy, 0.8,0.0)

        hum_list=[]
        for (hx, hy, hvx,hvy, rr2) in self.env.humans:
            hum_list.append((hx,hy,hvx,hvy,rr2))

        vx, vy= self.policy.predict(robot_st, hum_list)
        ob, reward, done, info= self.env.step(vx, vy, update=True)
        if done:
            self.reset_episode()

    # getters => for rviz
    def get_robot(self):
        return (self.env.robot_x, self.env.robot_y, self.env.robot_r,
                self.env.robot_vx, self.env.robot_vy)
    def get_goal(self):
        return (self.env.goal_x, self.env.goal_y)
    def get_humans(self):
        arr=[]
        for (hx,hy,hvx,hvy, rr) in self.env.humans:
            arr.append((hx,hy,rr))
        return arr
    def get_distance_info(self):
        rx, ry, rr, rvx, rvy= self.get_robot()
        arr=[]
        for (hx, hy, hvx,hvy, rr2) in self.env.humans:
            dist= math.hypot(rx-hx, ry-hy)
            arr.append((hx,hy, dist))
        return arr
    def get_episode_id(self):
        return self.episode_id

###############################################################################
# 6) Node => Timer => step => markers
###############################################################################
class CrowdNavDeepcopyShapingNode(Node):
    def __init__(self):
        super().__init__('crowdnav_deepcopy_shaping_node')
        self.get_logger().info("Deepcopy-based CADRL with reward shaping, boundary penalty, heading smoothness")

        self.max_episodes=10
        self.sim= FullSim(num_humans=10, dt=0.2,
                          policy_config='data/output/output_cadrl/policy.config',
                          model_file='data/output/output_cadrl/rl_model.pth',
                          device='cpu')

        self.pub= self.create_publisher(MarkerArray,'/crowdnav_markers',10)
        self.timer= self.create_timer(0.2, self.timer_callback)

        time_str= time.strftime("%Y%m%d_%H%M%S")
        self.log_filename= f"demo_deepcopy_shaping_{time_str}.csv"
        self.log_file= open(self.log_filename,'w')
        header= "step,episode_id,time_s,rx,ry,rvx,rvy\n"
        self.log_file.write(header)

        self.step_count=0
        now= self.get_clock().now()
        self.start_time_s= now.nanoseconds*1e-9
        print(f"[Logger] => {self.log_filename}")

        self.episode_count=0

    def timer_callback(self):
        if self.episode_count>= self.max_episodes:
            self.finish_and_exit()
            return

        self.step_count+=1
        now= self.get_clock().now()
        time_s= now.nanoseconds*1e-9- self.start_time_s

        old_ep_id= self.sim.get_episode_id()
        self.sim.step()
        new_ep_id= self.sim.get_episode_id()

        if new_ep_id!=old_ep_id:
            self.episode_count+=1
            if self.episode_count>= self.max_episodes:
                self.finish_and_exit()

        arr= MarkerArray()
        stamp_msg= now.to_msg()

        # robot
        (rx, ry, rr, rvx, rvy)= self.sim.get_robot()
        mk_robot= self.make_sphere_marker("robot",0, rx, ry,0.0, rr*2,(0,1,0,1))
        mk_robot.header.stamp= stamp_msg
        arr.markers.append(mk_robot)

        # goal
        gx, gy= self.sim.get_goal()
        mk_goal= self.make_arrow_marker("goal",1,gx,gy,0.0,0.5,0.08,(1,0,0,1))
        mk_goal.header.stamp= stamp_msg
        arr.markers.append(mk_goal)

        # humans
        hums= self.sim.get_humans()
        pid=2
        for (hx,hy, rr2) in hums:
            mk= self.make_sphere_marker("humans",pid,hx,hy,0.0, rr2*2,(0,0,1,1))
            mk.header.stamp= stamp_msg
            arr.markers.append(mk)
            pid+=1

        # distance text
        dist_list= self.sim.get_distance_info()
        tid=100
        for (hx,hy, dist) in dist_list:
            txt= self.make_text_marker("dist_txt",tid,hx,hy,0.3,f"{dist:.2f}",0.15,(1,1,1,1))
            txt.header.stamp= stamp_msg
            arr.markers.append(txt)
            tid+=1

        self.pub.publish(arr)

        # log
        line= f"{self.step_count},{old_ep_id},{time_s:.3f},{rx:.3f},{ry:.3f},{rvx:.3f},{rvy:.3f}\n"
        self.log_file.write(line)

    def finish_and_exit(self):
        print(f"=== All {self.episode_count} episodes done. Exiting. ===")
        self.log_file.close()
        self.destroy_node()

    # marker helper
    def make_sphere_marker(self, ns, idx, x,y,z, scale, color):
        from visualization_msgs.msg import Marker
        mk= Marker()
        mk.header.frame_id="map"
        mk.ns= ns
        mk.id= idx
        mk.type= Marker.SPHERE
        mk.action= Marker.ADD
        mk.pose.position.x=float(x)
        mk.pose.position.y=float(y)
        mk.pose.position.z=float(z)
        mk.scale.x=scale
        mk.scale.y=scale
        mk.scale.z=scale
        mk.color.r=float(color[0])
        mk.color.g=float(color[1])
        mk.color.b=float(color[2])
        mk.color.a=float(color[3])
        return mk

    def make_arrow_marker(self, ns, idx, x,y,z, length, radius, color):
        from visualization_msgs.msg import Marker
        mk=Marker()
        mk.header.frame_id="map"
        mk.ns=ns
        mk.id=idx
        mk.type=Marker.ARROW
        mk.action=Marker.ADD
        mk.pose.position.x=float(x)
        mk.pose.position.y=float(y)
        mk.pose.position.z=float(z)
        mk.scale.x=length
        mk.scale.y=radius
        mk.scale.z=radius
        mk.color.r=float(color[0])
        mk.color.g=float(color[1])
        mk.color.b=float(color[2])
        mk.color.a=float(color[3])
        return mk

    def make_text_marker(self, ns, idx, x,y,z, text, scale_z, color):
        from visualization_msgs.msg import Marker
        mk= Marker()
        mk.header.frame_id="map"
        mk.ns= ns
        mk.id= idx
        mk.type= Marker.TEXT_VIEW_FACING
        mk.action= Marker.ADD
        mk.pose.position.x=float(x)
        mk.pose.position.y=float(y)
        mk.pose.position.z=float(z)
        mk.scale.z= scale_z
        mk.color.r=float(color[0])
        mk.color.g=float(color[1])
        mk.color.b=float(color[2])
        mk.color.a=float(color[3])
        mk.text= text
        return mk

def main(args=None):
    rclpy.init(args=args)
    node= CrowdNavDeepcopyShapingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()
