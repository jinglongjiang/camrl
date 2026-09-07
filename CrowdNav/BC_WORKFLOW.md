# BC模块化训练工作流

## 概述

将BC (Behavioral Cloning) 训练从RL训练中分离，实现：
- ✅ **模块化**: BC和RL训练独立
- ✅ **灵活性**: 可用不同数据集训练多个BC baseline
- ✅ **效率**: RL训练直接加载BC，无需重复训练
- ✅ **内存安全**: 分步骤处理大数据集，避免OOM

---

## 完整工作流

### 步骤1: 生成IL数据集（可选，如果已有数据集跳过）

```bash
# 使用现有的v4.0数据集（30,000集，304MB，已验证可用）
ls -lh data/il_dataset_diverse_v4.0.pth

# 或生成新数据集（注意：50,000集会导致OOM，建议用30,000集）
python scripts/generate_il_dataset.py \
    --env-config crowd_nav/configs/env.config \
    --output data/il_dataset_diverse_v4.0.pth \
    --gpu
```

**v4.0数据集配置：**
- 人数: 5人（固定，baseline实验要求）
- 半径: [2.5, 3.0, 4.0, 5.0, 6.0] - 密度范围 0.044-0.255 人/m²
- 速度: [0.8, 1.0, 1.2] - 慢速到快速
- Seeds: 20个
- 总集数: 30,000 episodes
- 成功率: ~91%

---

### 步骤2: 训练BC模型

使用新创建的`train_bc.py`脚本单独训练BC：

```bash
python train_bc.py \
    --policy mamba_rl \
    --dataset data/il_dataset_diverse_v4.0.pth \
    --output data/bc_mamba_v4.0.pth \
    --gpu \
    --epochs 50 \
    --batch-size 512 \
    --policy-lr 1e-4 \
    --q-lr 1e-4 \
    --seq-len 12
```

**训练参数说明：**
- `--dataset`: IL数据集路径
- `--output`: 保存BC权重的路径
- `--epochs`: BC训练轮数（默认50）
- `--batch-size`: 批次大小（默认512）
- `--policy-lr`: Policy网络学习率
- `--q-lr`: Q网络学习率
- `--seq-len`: Mamba序列长度（默认12）

**训练输出：**
```
================================================================================
BC训练完成！
================================================================================
使用方法（在RL训练中）:
  python train.py --policy mamba_rl --load-bc data/bc_mamba_v4.0.pth --skip-il --gpu
================================================================================

BC权重文件: data/bc_mamba_v4.0.pth
日志文件: data/bc_mamba_v4.0.log
```

---

### 步骤3: RL训练（加载预训练BC）

#### 选项A: 跳过IL，直接RL训练（推荐）

```bash
python train.py \
    --policy mamba_rl \
    --load-bc data/bc_mamba_v4.0.pth \
    --skip-il \
    --gpu
```

**流程：**
1. 加载BC权重
2. 评估BC模型（100 episodes）
3. 用BC policy预填充replay buffer（200 episodes）
4. 直接进入RL训练

**适用场景：** BC已经很好（>85%），不需要额外IL数据

#### 选项B: 加载BC + 执行IL phase（加载数据到buffer）

```bash
python train.py \
    --policy mamba_rl \
    --load-bc data/bc_mamba_v4.0.pth \
    --gpu
```

**流程：**
1. 加载BC权重
2. 执行IL phase（从离线数据集加载ORCA轨迹到buffer）
3. 不重新训练BC（已加载外部BC）
4. 进入RL训练

**适用场景：** 想用BC作为初始化，但希望buffer中包含ORCA专家轨迹

---

## 工作流对比

### 传统工作流（train.py一体化）
```
IL数据收集 → BC训练 → RL训练
    ↑          ↑         ↑
  15-20min   2-5min   数小时
  (每次重复) (每次重复)
```

**问题：**
- ❌ 每次RL训练都要重复IL+BC（浪费15-25分钟）
- ❌ 不同实验无法共享BC baseline
- ❌ 无法单独评估BC质量

### 模块化工作流（train_bc.py + train.py）
```
[一次性]
IL数据集生成 → BC训练 → 保存BC权重
   15-20min      2-5min    (bc_mamba_v4.0.pth)

[每次RL实验]
加载BC权重 → RL训练
   5秒         数小时
```

**优势：**
- ✅ RL训练启动快（5秒 vs 25分钟）
- ✅ 多个实验共享BC（节省资源）
- ✅ BC可独立评估和版本管理
- ✅ 灵活选择不同BC baseline对比

---

## 文件结构

```
CrowdNav/
├── scripts/
│   └── generate_il_dataset.py  # IL数据集生成（v5.0）
├── train_bc.py                  # BC训练脚本（新增）
├── train.py                     # RL训练脚本（已修改，支持--load-bc）
├── data/
│   ├── il_dataset_diverse_v4.0.pth    # IL数据集（30k集，304MB）
│   ├── bc_mamba_v4.0.pth              # BC权重
│   └── bc_mamba_v4.0.log              # BC训练日志
└── crowd_nav/runs/mamba_vl/
    ├── il_policy.pth            # RL训练中的BC checkpoint（自动生成）
    └── ckpt_il.pt               # IL完成checkpoint（自动生成）
```

---

## 常见问题

### Q1: v4.1/v5.0 数据集保存失败（0字节）怎么办？

**原因：** 50,000集数据在`torch.save()`时触发OOM killer

**解决方案：** 使用v4.0数据集（30,000集，已验证可用）

```bash
# 检查v4.0是否存在
ls -lh data/il_dataset_diverse_v4.0.pth
# 应显示: -rw-r--r-- 1 abc abc 304M Nov  2 13:34 ...

# 如果不存在，重新生成（配置降回30k集）
# 修改 scripts/generate_il_dataset.py:
# RADIUS_CONFIGS = [2.5, 3.0, 4.0, 5.0, 6.0]  # 保持5个
# HUMAN_V_PREF_CONFIGS = [0.8, 1.0, 1.2]      # 降回3个
# 总计: 20×5×3×100 = 30,000集
```

### Q2: BC训练时内存不足？

**解决方案：** 减小batch size

```bash
python train_bc.py \
    --dataset data/il_dataset_diverse_v4.0.pth \
    --output data/bc_mamba_v4.0.pth \
    --batch-size 256 \  # 从512降到256
    --gpu
```

### Q3: 如何评估BC模型质量？

**方法1：** BC训练完成时会自动显示训练损失

**方法2：** RL训练加载BC时会评估100 episodes并报告成功率

```bash
python train.py --load-bc data/bc_mamba_v4.0.pth --skip-il --gpu
# 输出: [BC-EXTERNAL] BC模型成功率: 87.0% (87/100 episodes)
```

**方法3：** 使用test.py单独测试

```bash
# 首先将BC权重转换为test.py格式
python -c "
import torch
bc = torch.load('data/bc_mamba_v4.0.pth')
torch.save(bc['policy_state_dict'], 'data/bc_test.pth')
"

# 测试
python test.py --policy mamba_rl --model_dir data --phase test --test_case 500
```

### Q4: 不同数据集训练的BC如何比较？

创建多个BC版本并对比：

```bash
# BC from v4.0 (30k, R=[2.5,3,4,5,6], V=[0.8,1.0,1.2])
python train_bc.py --dataset data/il_dataset_diverse_v4.0.pth \
    --output data/bc_mamba_v4.0.pth --gpu

# BC from v3.1 (if exists)
python train_bc.py --dataset data/il_dataset_diverse_v3.1.pth \
    --output data/bc_mamba_v3.1.pth --gpu

# RL训练时选择不同BC
python train.py --load-bc data/bc_mamba_v4.0.pth --skip-il --gpu --outdir runs/rl_bc_v4.0
python train.py --load-bc data/bc_mamba_v3.1.pth --skip-il --gpu --outdir runs/rl_bc_v3.1
```

---

## 高级用法

### 使用自定义数据集

```bash
# 1. 生成自定义数据集
python scripts/generate_il_dataset.py \
    --output data/my_custom_il.pth \
    --gpu

# 2. 训练BC
python train_bc.py \
    --dataset data/my_custom_il.pth \
    --output data/bc_custom.pth \
    --epochs 100 \  # 更多epochs
    --gpu

# 3. RL训练
python train.py --load-bc data/bc_custom.pth --skip-il --gpu
```

### 继续训练（BC微调）

如果BC质量不够，可以在RL中继续微调：

```bash
# 加载BC但不跳过IL（会执行IL phase加载ORCA数据）
python train.py --load-bc data/bc_mamba_v4.0.pth --gpu

# 这会：
# 1. 加载BC权重作为初始化
# 2. 加载IL数据集到buffer（不重新训练BC）
# 3. RL训练时可以在BC基础上继续学习
```

---

## 性能建议

### 内存优化

| 数据集规模 | 内存需求 | 状态 |
|-----------|---------|------|
| 30,000集  | ~10GB   | ✅ 可用 |
| 50,000集  | ~15GB+  | ❌ OOM |

**推荐配置：**
- 数据集: 30,000集（v4.0）
- BC batch size: 512（16GB GPU）或256（8GB GPU）
- RL batch size: 512-1024

### 训练时间估算

| 阶段 | 时间 | 备注 |
|-----|------|------|
| IL数据集生成（30k） | 15-20分钟 | 一次性 |
| BC训练（50 epochs） | 2-5分钟 | 一次性 |
| RL训练（1000 eps） | 2-4小时 | 每次实验 |

**时间节省：**
- 传统流程：20+5+120 = 145分钟/实验
- 模块化流程：0.1+120 = 120分钟/实验
- 节省：~25分钟/实验（17%）

---

## 总结

**模块化BC训练的核心价值：**

1. **一次训练，多次使用**: BC权重可在多个RL实验中复用
2. **快速迭代**: RL实验启动时间从25分钟降至5秒
3. **版本管理**: 不同数据集的BC可以独立管理和对比
4. **内存安全**: 分步处理避免大数据集OOM
5. **实验公平性**: 所有RL实验使用相同BC baseline，确保对比公平

**建议工作流：**

```bash
# [一次性准备]
# 1. 确认v4.0数据集存在
ls -lh data/il_dataset_diverse_v4.0.pth  # 应为304MB

# 2. 训练BC（只需一次）
python train_bc.py --policy mamba_rl \
    --dataset data/il_dataset_diverse_v4.0.pth \
    --output data/bc_mamba_v4.0.pth --gpu

# [每次RL实验]
# 3. 加载BC进行RL训练
python train.py --policy mamba_rl \
    --load-bc data/bc_mamba_v4.0.pth \
    --skip-il --gpu
```

完成! 🎉
