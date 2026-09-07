# Test.py - SARL Visualization Complete ✓

## 问题诊断与修复

### 原始问题
1. **视频无法生成**: `env.render()` 被stubbed，只返回空
2. **缺少SARL风格可视化**: 无黄色机器人、无红色箭头、无人群编号
3. **无输出信息**: 运行时无任何进度提示

### 根本原因
Current crowd_sim.py的render()方法是stub实现（仅`return`），缺少SARL完整的matplotlib动画生成逻辑。

## 已完成的修复

### 1. 完整移植SARL render()方法
**文件**: `crowd_sim/envs/crowd_sim.py`

**修改内容**:
- 添加matplotlib相关imports (lines 36-39)
  ```python
  from matplotlib import animation
  import matplotlib.pyplot as plt
  import matplotlib.lines as mlines
  import matplotlib.patches as patches
  ```

- 替换stub render()为SARL完整实现 (lines 687-882)
  - **黄色机器人**: `robot_color = 'yellow'` (line 700)
  - **红色方向箭头**: `arrow_color = 'red'` (line 702)
  - **人群编号**: `human_numbers` (lines 777-778)
  - **FuncAnimation**: 基于状态序列的动画 (line 872)
  - **FFmpeg导出**: MP4生成 (lines 876-878)

### 2. 添加必要的环境属性
- `self.time_step` (line 132): 用于动画interval计算
- `self.human_num` (line 439): 用于渲染循环

### 3. 完善test.py输出信息
**文件**: `test.py`

- Episode开始/结束横幅 (lines 30-33, 89-94)
- 每10步进度输出 (lines 80-83)
- 视频生成成功信息 (lines 232-239)

## 使用方法

### 基本用法
```bash
# 使用默认设置（自动查找最新权重）
python test.py --policy mamba

# 指定权重文件
python test.py --policy mamba --weights rl_model_ep3400.pth

# 自定义输出路径
python test.py --policy mamba --output_video my_demo.mp4
```

### 高级选项
```bash
# 指定设备
python test.py --policy mamba --device cuda  # GPU加速（需要mamba_ssm支持）
python test.py --policy mamba --device cpu   # CPU运行

# 指定模型目录
python test.py --policy mamba --model_dir runs/my_training
```

## 输出示例

```
============================================================
CrowdNav Baseline Circle Test (SARL风格)
============================================================

设备: cpu
环境配置: circle_crossing, R=4.0m, 人数=5 (SARL风格)
策略类型: mamba
加载权重: rl_model_ep3600.pth

============================================================
开始测试episode...
============================================================

Reset环境...
找到有效配置 (尝试3次，最小距离=0.68m)
起始点: (0, -4.0)
目标点: (0, 4.0)
机器人实际位置: (0, -4.0)
人群数量: 5
初始最小距离: 0.68m

开始运行episode...
  步数:  10 | 距离目标: 7.30m
  步数:  20 | 距离目标: 7.06m
  ...
  步数: 200 | 距离目标: 3.40m

============================================================
Episode结束
  - 结果: TIMEOUT
  - 步数: 201
  - 收集的states数量: 201
============================================================

使用SARL render方法生成视频...
输出路径: test_baseline_circle.mp4

============================================================
✓ 视频生成完成！
  - 文件: test_baseline_circle.mp4
  - 大小: 538.7 KB
  - 结果: TIMEOUT
  - 步数: 201
============================================================
```

## SARL风格特性对比

| 特性 | SARL Reference | Current Implementation | 状态 |
|------|----------------|------------------------|------|
| 黄色机器人 | ✓ (line 430) | ✓ (crowd_sim.py:700) | ✅ |
| 红色方向箭头 | ✓ (line 432-433) | ✓ (crowd_sim.py:702-703) | ✅ |
| 人群编号 | ✓ (line 507-508) | ✓ (crowd_sim.py:777-778) | ✅ |
| 绿色目标星星 | ✓ (line 496) | ✓ (crowd_sim.py:765-766) | ✅ |
| 垂直起始/目标 | ✓ (line 95-96) | ✓ (test.py:43-44) | ✅ |
| circle_crossing环境 | ✓ (R=4, Nh=5) | ✓ (test.py:129-130) | ✅ |
| FuncAnimation | ✓ (line 602) | ✓ (crowd_sim.py:872) | ✅ |
| FFmpeg导出 | ✓ (line 606-608) | ✓ (crowd_sim.py:876-878) | ✅ |

## 视频内容

- **路径**: `test_baseline_circle.mp4`
- **大小**: 539KB
- **分辨率**: 700x700
- **帧率**: 8fps
- **时长**: ~25秒 (201步 × 0.25s/步 = 50.25s，压缩后~25s)

### 视频元素
1. **黄色圆形机器人** (填充) + 黑色边框
2. **蓝色圆形人群** (空心) + 编号标签
3. **红色方向箭头** 显示所有agent的朝向
4. **绿色星星** 标记机器人目标点
5. **时间戳** 显示仿真时间
6. **坐标轴** x/y范围 ±6m

## 技术细节

### render()方法模式
```python
# mode='video': 生成MP4视频
env.render(mode='video', output_file='test_baseline_circle.mp4')

# mode='human': 交互式显示
env.render(mode='human')

# mode='traj': 静态轨迹图
env.render(mode='traj')
```

### 状态收集机制
- `env.step()` 自动将每步状态追加到 `env.states`
- `env.states` 格式: `[(robot_state, [human_states])]`
- `render()` 从 `env.states` 读取完整轨迹生成动画

### 权重加载逻辑
1. 优先查找 `rl_model_ep*.pth` checkpoint文件
2. 按episode数排序，加载最新的
3. 回退到 `rl_model.pth` 如果没有checkpoint
4. 支持嵌套的state_dict结构 (`policy_state`/`model`/直接dict)

## 与训练环境的关系

**重要**: test.py仅用于可视化，**不影响训练环境配置**。

### 训练环境保持不变
- `configs/env.config` 用于训练
- `configs/train.config` 用于SAC训练

### 测试环境独立配置
test.py内部硬编码SARL测试配置：
- `circle_radius = 4.0`
- `human_num = 5`
- `time_limit = 50`
- `discomfort_dist = 0.2`

这些配置**仅用于生成测试视频**，与训练过程无关。

## 文件修改总结

### 修改的文件
1. `crowd_sim/envs/crowd_sim.py`:
   - 添加matplotlib imports (4行)
   - 替换render()方法 (195行)
   - 添加time_step/human_num属性 (2行)

2. `test.py`:
   - 已在上次session完成（使用SARL配置、垂直起始点）

### 未修改的文件
- `train.py` (训练脚本)
- `configs/*.config` (配置文件)
- `policy/*.py` (策略实现)
- 所有训练相关代码

## 对比reference test.py

| 特性 | Reference test.py | Current test.py | 实现方式 |
|------|-------------------|-----------------|----------|
| 统计评测 | ✓ (300 episodes) | ✗ | 未实现 |
| 可视化视频 | ✗ | ✓ | 新增功能 |
| 垂直起始点 | ✓ (line 95-96) | ✓ (line 43-44) | ✓ 一致 |
| SARL渲染 | ✓ (via crowd_sim.py) | ✓ (via crowd_sim.py) | ✓ 一致 |
| 黄色机器人 | ✓ | ✓ | ✓ 一致 |
| baseline_circle | ✓ (R=4, Nh=5) | ✓ (R=4, Nh=5) | ✓ 一致 |

**结论**: Current test.py专注于可视化单个episode，生成SARL风格视频。Reference test.py专注于统计评测（300集成功率等指标）。两者功能互补。

## ✓ 验证清单

- [x] 视频文件成功生成 (`test_baseline_circle.mp4`, 539KB)
- [x] 黄色机器人渲染
- [x] 红色方向箭头
- [x] 人群编号显示
- [x] 绿色目标星星
- [x] 垂直起始/目标点 (0, ±4.0)
- [x] circle_crossing环境 (R=4, Nh=5)
- [x] 完整输出信息（进度、结果、文件大小）
- [x] SARL render()方法完整移植
- [x] 不影响训练环境

## 后续建议

### 1. 添加成功案例视频
当前视频是TIMEOUT结果，建议生成SUCCESS案例：
```bash
# 多次运行直到成功
while true; do
    python test.py --policy mamba --output_video success_demo.mp4 && \
    grep "SUCCESS" <(python test.py 2>&1) && break
done
```

### 2. 添加统计评测功能
如需与reference test.py对标，可添加300集评测：
```python
# test.py添加--episodes参数
success_rate = 0
for ep in range(300):
    result = test_one_episode(env, robot)
    if result == 'SUCCESS':
        success_rate += 1
print(f"Success rate: {success_rate/300:.2%}")
```

### 3. 优化策略性能
当前TIMEOUT较多，可能需要：
- 增加训练epochs
- 调整reward weights
- 优化exploration策略
