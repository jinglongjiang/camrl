#!/usr/bin/env python3
import os

# 1) 假定系统里, GPU激光插件若存在, 路径一般位于 /opt/ros/foxy/lib/libgazebo_ros_gpu_laser.so
GPU_PLUGIN_PATH = "/opt/ros/foxy/lib/libgazebo_ros_gpu_laser.so"

# 2) SDF模板，留个“{}”给后续拼接Sensor配置
SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="gpu_or_cpu_world">

    <!-- 基础：光照 + 地面 -->
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- 简易机器人（仅一个link），其中插入激光sensor -->
    <model name="robot_with_laser">
      <static>false</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="chassis">
        {}
      </link>
    </model>

  </world>
</sdf>
"""

# 3) GPU 激光sensor片段
GPU_SENSOR = """
<sensor name="gpu_laser" type="gpu_ray">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <pose>0.3 0 0.3 0 0 0</pose>
  <gpu_ray>
    <scan>
      <horizontal_samples>720</horizontal_samples>
    </scan>
    <range>
      <min>0.12</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </gpu_ray>

  <!-- GPU 插件 -->
  <plugin name="gpu_laser_plugin" filename="libgazebo_ros_gpu_laser.so">
    <ros>
      <namespace>gpu_laser_ns</namespace>
      <!-- 最终雷达话题, 例如 /gpu_laser_ns/scan -->
      <argument>scan:=scan</argument>
    </ros>
  </plugin>
</sensor>
"""

# 4) CPU (Ray) 激光sensor片段
CPU_SENSOR = """
<sensor name="cpu_laser" type="ray">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <pose>0.3 0 0.3 0 0 0</pose>
  <ray>
    <scan>
      <horizontal_samples>720</horizontal_samples>
    </scan>
    <range>
      <min>0.12</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>

  <!-- CPU (ray) 插件 -->
  <plugin name="cpu_laser_plugin" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>cpu_laser_ns</namespace>
      <!-- 最终雷达话题, 例如 /cpu_laser_ns/scan -->
      <argument>scan:=scan</argument>
    </ros>
  </plugin>
</sensor>
"""

def main():
    # 判断是否存在 GPU 插件
    if os.path.exists(GPU_PLUGIN_PATH):
        print("[INFO] Detected GPU laser plugin => Using GPU laser sensor.")
        sensor_sdf = GPU_SENSOR
    else:
        print("[WARN] GPU laser plugin NOT found => Falling back to CPU laser sensor.")
        sensor_sdf = CPU_SENSOR

    # 将 sensor 片段插入模板
    final_sdf = SDF_TEMPLATE.format(sensor_sdf)

    # 写入本地 world 文件
    output_world = "gpu_or_cpu_world.world"
    with open(output_world, "w") as f:
        f.write(final_sdf)

    print(f"[INFO] Generated '{output_world}'. You can now launch it via:")
    print(f"  ros2 launch gazebo_ros gazebo.launch.py world:={os.path.join(os.getcwd(), output_world)}")

if __name__ == '__main__':
    main()

