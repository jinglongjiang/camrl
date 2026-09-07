# Testing Scripts Usage Guide

## Overview

两个测试脚本已完成统一改造，除策略加载逻辑外，其他所有部分（环境配置、评估循环、随机种子、日志级别等）完全一致。

---

## test.py - Mamba Policy Testing

**用途**: 专门测试 Mamba 权重

**特点**:
- 通过 `policy_factory` 动态加载 Mamba 策略
- 支持 Mamba 的所有变体（mamba, mamba_rl 等）
- 默认模型目录: `runs/mamba_vl`

**使用方法**:
```bash
# 基础测试（使用默认权重）
python test.py --policy mamba --gpu --episodes 100

# 指定权重文件
python test.py --policy mamba --weights rl_model_ep10000.pth --gpu --episodes 100

# 使用最新权重
python test.py --policy mamba --weights latest --gpu --episodes 100

# 测试单个场景
python test.py --policy mamba --test_case 0 --gpu --episodes 100
```

**参数说明**:
- `--policy`: 策略名称（默认: mamba）
- `--model_dir`: 模型目录（默认: runs/mamba_vl）
- `--weights`: 权重文件名或 "latest"（默认: rl_model_ep10000_2.pth）
- `--gpu`: 使用 GPU
- `--episodes`: 每个场景测试的 episode 数量
- `--test_case`: 测试单个场景（0-5）

---

## test2.py - CrowdNav Baseline Testing

**用途**: 专门测试 CrowdNav 原生策略（SARL/LSTM/CADRL）

**特点**:
- 通过 `policy_factory` 动态加载 CrowdNav 策略
- 支持 SARL, LSTM, CADRL
- 默认模型目录: `data/output`
- 包含完整的 SARL 配置注入（网络架构、动作空间、OM 参数）

**使用方法**:
```bash
# 测试 SARL
python test2.py --policy sarl --gpu --episodes 100

# 测试 LSTM
python test2.py --policy lstm --gpu --episodes 100

# 测试 CADRL
python test2.py --policy cadrl --gpu --episodes 100

# 指定权重文件
python test2.py --policy sarl --weights rl_model_ep5000.pth --gpu --episodes 100

# 使用 SARL Value Network 模式
python test2.py --policy sarl --sarl_value --gpu --episodes 100
```

**参数说明**:
- `--policy`: 策略名称（默认: sarl，可选: lstm, cadrl）
- `--model_dir`: 模型目录（默认: data/output）
- `--weights`: 权重文件名或 "latest"（默认: rl_model.pth）
- `--sarl_value`: 使用独立的 SARL Value Network（仅 SARL）
- `--gpu`: 使用 GPU
- `--episodes`: 每个场景测试的 episode 数量
- `--test_case`: 测试单个场景（0-5）

---

## 测试场景（已统一）

两个脚本使用相同的测试场景（与 test-ssm.py 一致）：

| ID | 场景名称 | 类型 | 人数 | 参数 |
|----|---------|------|------|------|
| 0 | baseline_circle | circle_crossing | 5 | R=4.0 |
| 1 | baseline_square | square_crossing | 10 | W=10.0 |
| 2 | dense_circle | circle_crossing | 10 | R=4.0 |
| 3 | dense_square | square_crossing | 20 | W=10.0 |
| 4 | large_circle | circle_crossing | 12 | R=6.0 |
| 5 | large_square | square_crossing | 20 | W=14.0 |

---

## 统一的特性

两个脚本现在共享以下特性：

1. **日志级别**: INFO（显示详细信息）
2. **随机种子**: 全局种子初始化（基于时间戳）
3. **设备管理**: 显式调用 `set_device()` 和 `set_phase()`
4. **SARL 模式**: 强制启用 `use_sarl_predict=True`（如果支持）
5. **错误处理**: 详细的异常信息输出

**配置注入差异**:
- **test.py**: 最小配置（仅 epsilon_start，Mamba 不需要 SARL 网络参数）
- **test2.py**: 完整 SARL 配置（mlp_dims, action_space, OM 等，CrowdNav 策略需要）

---

## 关键差异总结

| 特性 | test.py | test2.py |
|------|---------|----------|
| **目标策略** | Mamba | SARL/LSTM/CADRL |
| **策略加载** | `policy_factory[args.policy]` | `policy_factory[args.policy]` |
| **默认策略** | mamba | sarl |
| **默认目录** | runs/mamba_vl | data/output |
| **默认权重** | rl_model_ep10000_2.pth | rl_model.pth |
| **特殊模式** | 无 | `--sarl_value` 支持 |

---

## 公平比较示例

### 比较 Mamba vs SARL（相同场景）

```bash
# 测试 Mamba
cd /home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav
python test.py --policy mamba --gpu --episodes 100 > mamba_results.txt

# 测试 SARL
python test2.py --policy sarl --gpu --episodes 100 > sarl_results.txt

# 比较结果
diff mamba_results.txt sarl_results.txt
```

### 测试所有 CrowdNav 基线

```bash
# SARL
python test2.py --policy sarl --gpu --episodes 100

# LSTM
python test2.py --policy lstm --gpu --episodes 100

# CADRL
python test2.py --policy cadrl --gpu --episodes 100
```

---

## 注意事项

1. **权重路径**: 确保 `--model_dir` 指向正确的权重目录
2. **GPU 内存**: 使用 `--gpu` 时注意显存占用
3. **随机性**: 虽然设置了全局种子，但每次运行仍会有轻微差异（环境随机性）
4. **日志输出**: 两个脚本都使用 INFO 级别，输出较详细

---

## 改造完成清单

✅ 统一测试场景（dense_square=20人, large_square=W=14）
✅ 统一日志级别（INFO）
✅ 统一随机种子初始化
✅ 统一设备和阶段设置
✅ 统一错误处理
✅ 明确策略加载逻辑（test.py=Mamba, test2.py=CrowdNav）
✅ test.py: 最小配置注入（仅 epsilon_start）
✅ test2.py: 完整 SARL 配置注入（网络架构/动作空间/OM）
✅ 语法检查通过

---

## 快速参考

```bash
# Mamba ���试
python test.py --policy mamba --gpu --episodes 100

# SARL 测试
python test2.py --policy sarl --gpu --episodes 100

# LSTM 测试
python test2.py --policy lstm --gpu --episodes 100

# CADRL 测试
python test2.py --policy cadrl --gpu --episodes 100
```
