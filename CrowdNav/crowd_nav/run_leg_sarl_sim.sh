#!/usr/bin/env bash
#
# run_leg_sarl_sim.sh — 空地仿真：Gazebo + leg_detector + SARL + 圆柱行人
#
set -e

# 1) 环境
export TURTLEBOT3_MODEL=burger
source /opt/ros/foxy/setup.bash

# 2) 启动 Gazebo 空地 world + TurtleBot3
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py \
    use_sim_time:=true \
    world:="$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/worlds/empty.world" &
GZ_PID=$!
sleep 5

# 3) spawn 三个圆柱体
for i in 1 2 3; do
  ros2 run gazebo_ros spawn_entity.py \
    -entity ped_$i \
    -database cylinder \
    -x $((i-2)) -y 0 -z 0.5 &
done

# 4) 启动 leg_detector
ros2 run leg_detector leg_detector --ros-args -p use_sim_time:=true &
LEGDET_PID=$!

# 5) 启动你的 SARL 节点
python3 ~/crowdnav_ws/src/CrowdNav/crowd_nav/leg_sarl_node.py &
SARL_PID=$!

# 6) Ctrl-C 时清理
trap "kill $GZ_PID $LEGDET_PID $SARL_PID 2>/dev/null" INT TERM
wait
