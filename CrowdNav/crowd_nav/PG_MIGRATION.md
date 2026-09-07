# Policy Gradient迁移文档

## 改动总结

### 核心改动：从AWR+BC改成Policy Gradient

#### 之前（AWR + Behavioral Cloning）
```python
# RL阶段
policy_loss = (policy(s) - action_from_buffer)²  # BC loss
loss = v_loss + policy_loss
```

**问题**：
- Policy模仿buffer中的action（包含随机噪声）
- 没有利用reward信号改进
- 性能持续下降（91% → 70% → 50%）

#### 现在（Policy Gradient）
```python
# RL阶段
policy_loss = -V(s)  # 最大化价值函数
loss = v_loss + policy_loss
```

**优势**：
- Policy学习最大化V(s)，而不是模仿action
- Critic提供改进信号（V(s)的梯度）
- 符合标准IL+RL范式

## 代码改动位置

### 1. train.py - optimize_step_sequence() (Line 523-558)

**移除**：
- AWR权重计算 (`exp(advantage/beta)`)
- BC loss (`(pred - action)²`)
- PER权重应用到policy loss

**新增**：
- Policy Gradient loss (`-V(s).mean()`)
- Q值监控日志

### 2. train.config (Line 39-58)

**修改**：
- `policy_loss_coef`: 0.1 → 1.0 （actor和critic同等重要）
- `epsilon_start`: 0.10 → 0.05 （PG本身提供改进）
- `epsilon_end`: 0.05 → 0.02
- `epsilon_decay`: 2000 → 3000

### 3. 监控日志 (Line 632-635)

**新增**：
```
[PG-MONITOR] policy_loss=X avg_Q=Y
[PG-MONITOR] Q_std=X Q_min=Y Q_max=Z
```

## 训练逻辑对比

### IL阶段（保持不变）
```
ORCA专家数据 → BC loss → Policy初始化（91%成功率）
```

### RL阶段

#### 之前（AWR+BC）
```
1. 采样：policy + 10% noise → buffer
2. Critic学习：V(s) ≈ expected return
3. Actor学习：模仿buffer中的action
结果：模仿混合分布，性能下降
```

#### 现在（Policy Gradient）
```
1. 采样：policy + 5% noise → buffer
2. Critic学习：V(s) ≈ expected return
3. Actor学习：argmax V(s)
结果：有改进方向，性能有望提升
```

## 预期效果

### 训练曲线
```
Episode 0-50:   维持91%（IL初始化）
Episode 50-200: 逐步提升92-95%（RL改进）
Episode 200+:   稳定在95%+（收敛）
```

### 监控指标
```
avg_Q: 应该逐步上升（Critic学习准确）
policy_loss: 应该逐步下降（Actor改进）
success_rate: 应该稳定提升（整体性能）
```

## 架构说明

当前使用V(s)而不是Q(s,a)：
- 标准DDPG/TD3使用Q(s,a)
- 当前架构是V(s)（类似A3C/PPO）
- 在on-policy设置下等价

**为什么可行**：
- V(s) = E[Q(s,a)] under current policy
- Policy更新方向：∇V(s) 指向更高价值的state
- 虽然不如Q(s,a)精确，但方向正确

## 验证清单

- [x] 移除AWR权重计算
- [x] 移除BC loss
- [x] 实现Policy Gradient loss
- [x] 调整超参数
- [x] 添加PG监控日志
- [ ] 启动训练验证
- [ ] 确认性能不再下降
- [ ] 确认性能有提升趋势

## 下一步

1. 启动训练测试
2. 监控前50 episodes确认稳定性
3. 监控50-200 episodes确认改进
4. 如需要，微调超参数（lr, epsilon等）
