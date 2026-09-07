# 灾难性遗忘修复总结

## 问题诊断（基于 todo.md 和 train.log）

### 核心问题
- **灾难性遗忘**: 成功率从100%快速降至42.9%（7个episodes内）
- **BC基线**: 91%成功率，但RL训练破坏了BC知识
- **根本原因**:
  1. IL兜底机制未生效（BC prefill的episodes未标记为IL）
  2. 探索率过高（35%）产生大量失败经验
  3. IL采样比例不足以保护BC知识

### todo.md的理论指导
> "使用异策略算法配合经验回放，确保混合池中同时包含IL成功数据和RL探索数据"
> "不能丢弃91%成功率的好经验"
> "防止错误累加导致的恶性循环"

---

## 已完成的修复

### 修复1: BC Prefill调用时机 ✅
**文件**: `train.py:2587-2594`

**问题**: IL完成后没有BC prefill，导致训练不一致

**修复**:
```python
# 🔥 修复：IL完成后也应该进行BC预填充（与skip_il分支保持一致）
bc_prefill_eps = int(cfg.getfloat('train', 'bc_prefill_episodes', fallback=0))
if bc_prefill_eps > 0:
    logging.info(f"[IL→RL-SWITCH] Running BC prefill before RL training ({bc_prefill_eps} episodes)...")
    _prefill_buffer_with_bc(policy, explorer, rl_buf, bc_prefill_eps)
```

**效果**: 确保无论从头训练还是resume，都会进行BC prefill

---

### 修复2: BC Prefill标记为IL ✅ **关键修复**
**文件**: `train.py:1246`

**问题**: BC prefill使用`phase='train'`，episodes被标记为`source='RL'`，IL兜底机制无法识别

**修复前**:
```python
explorer.run_k_episodes(episodes, 'train', update_memory=True, ...)
# → episodes标记为source='RL'，不被il_ratio采样
```

**修复后**:
```python
# 🔥 修复IL兜底：使用phase='il'让BC episodes被标记为IL数据
explorer.run_k_episodes(episodes, 'il', update_memory=True, ...)
# → episodes标记为source='IL'，被il_ratio优先采样
```

**效果**: 32个BC prefill episodes现在被正确识别为IL数据，il_ratio=70%时会优先采样

---

### 修复3: BC Prefill验证 ✅
**文件**: `train.py:1250-1258`

**问题**: 无法验证BC prefill是否正确标记

**修复**:
```python
# 🔥 验证BC prefill的episodes是否被正确标记为IL
if hasattr(rl_buf, 'episodes'):
    il_count_after = sum(1 for ep in rl_buf.episodes if ep.get('meta', {}).get('source') != 'RL')
    rl_count_after = len(rl_buf.episodes) - il_count_after
    logging.info(f"[BC-PREFILL-VERIFY] Buffer composition: IL={il_count_after}, RL={rl_count_after}, Total={len(rl_buf.episodes)}")
```

**效果**:
- 日志会显示: `[BC-PREFILL-VERIFY] Buffer composition: IL=3032, RL=0, Total=3032`
- 可以立即发现IL标记问题

---

### 修复4: 增加IL采样统计频率 ✅
**文件**: `train.py:1864-1871`

**问题**: 采样统计只在ep=25, 75...输出，无法及时发现问题

**修复前**:
```python
if ep % 50 == 25:  # 只在25, 75, 125...输出
```

**修复后**:
```python
# 🔥 修复：训练早期更频繁地输出IL采样统计
should_log_mix = (ep <= 100 and ep % 10 == 1) or (ep > 100 and ep % 50 == 25)
if should_log_mix:  # ep=1, 11, 21, 31, ..., 91, 125, 175...
```

**效果**:
- Episode 1, 11, 21...都会输出IL采样统计
- 可以立即验证IL兜底是否生效

---

### 修复5: Episode 1 Buffer统计 ✅
**文件**: `train.py:1466-1470`

**问题**: 不知道训练开始时buffer的组成

**修复**:
```python
# 🔥 第一个episode输出详细的buffer统计，验证IL兜底
if ep == 1 and hasattr(rl_buf, 'episodes'):
    il_count_buf = sum(1 for e in rl_buf.episodes if e.get('meta', {}).get('source') != 'RL')
    rl_count_buf = len(rl_buf.episodes) - il_count_buf
    logging.info(f"[BUFFER-STAT] ep=1 buffer: IL={il_count_buf}, RL={rl_count_buf}, Total={len(rl_buf.episodes)} | il_ratio_target={current_il_ratio:.1%}")
```

**效果**:
- 立即显示buffer初始状态
- 日志示例: `[BUFFER-STAT] ep=1 buffer: IL=3032, RL=0, Total=3032 | il_ratio_target=70.0%`

---

### 修复6: 降低探索率 ✅ **关键修复**
**文件**: `configs/train.config:54-56`

**问题**: epsilon=0.35对于91%成功率的BC模型太高，产生大量失败经验

**修复前**:
```ini
epsilon_start = 0.35  # 35%随机探索 → 太多失败
epsilon_end = 0.1
epsilon_decay_episodes = 2000
```

**修复后**:
```ini
# 🔥 修复灾难性遗忘：BC已91%成功，降低探索率保护BC知识
epsilon_start = 0.10  # 降至10%（信任BC策略，减少失败经验）
epsilon_end = 0.05    # 最终5%
epsilon_decay_episodes = 3000  # 更平滑衰减
```

**理由**:
- BC已91%成功，应该信任BC学到的策略
- 35%探索 = 每3步有1步随机 → 大量失败 → 破坏BC知识
- 10%探索足够发现改进，同时保护BC基础

**预期效果**:
- 失败经验减少70% (35%→10%)
- 成功率下降更平滑: 90%→85%→80% (而非100%→42%)

---

### 修复7: 增强IL采样保护 ✅ **关键修复**
**文件**: `configs/train.config:122-124`

**问题**: IL采样比例不足以对抗失败经验的破坏

**修复前**:
```ini
il_ratio_start = 0.60  # 60% IL
il_ratio_end = 0.30    # 30% IL
il_ratio_decay_episodes = 3000
```

**修复后**:
```ini
# 🔥 IL兜底保护：防止灾难性遗忘（基于todo.md建议）
il_ratio_start = 0.70  # 提高到70% IL（更强保护）
il_ratio_end = 0.40    # 提高到40% IL（始终保持强锚点）
il_ratio_decay_episodes = 5000  # 更缓慢衰减
```

**理由**:
- todo.md强调"不能丢弃91%成功率的好经验"
- 70% IL采样 = 每个batch中179/256个样本来自IL
- 即使后期，40% IL仍能提供稳定锚点

**预期效果**:
- 每次梯度更新，70%的样本来自成功经验
- BC知识得到持续强化，防止遗忘

---

### 修复8: 改善数据/更新比例 ✅
**文件**: `configs/train.config:137`

**问题**: 4 episodes : 80 updates = 1:20，可能过拟合失败经验

**修复前**:
```ini
num_workers = 4
episodes_per_worker = 1  # 4×1=4 episodes
# 配合updates_per_ep=80 → 1:20比例
```

**修复后**:
```ini
# 🔥 改善数据/更新比例（从1:20→1:10）
num_workers = 4
episodes_per_worker = 2  # 4×2=8 episodes
# 配合updates_per_ep=80 → 1:10比例
```

**理由**:
- 更多episodes提供更丰富的状态覆盖
- 1:10比例更健康，减少过拟合风险
- SARL是1:100（更保守），我们1:10已经比较激进

**预期效果**:
- 每批次收集8个episodes（更多样化）
- 即使有失败，也不会被反复学习太多次

---

## 修复机制说明

### IL兜底如何工作

1. **Buffer组成** (修复后):
   - 3000个IL episodes (离线数据集)
   - 32个BC prefill episodes (✅ 现在标记为IL)
   - 随着训练进行，RL episodes逐渐增加

2. **采样机制** (memory.py:400-433):
   ```python
   if il_ratio > 0.0:
       # 分离IL和RL episodes
       il_indices = [i for i in range(len(self.episodes))
                    if self.episodes[i].get('meta', {}).get('source', 'RL') != 'RL']
       rl_indices = [i for i in range(len(self.episodes))
                    if self.episodes[i].get('meta', {}).get('source', 'RL') == 'RL']

       # 计算目标采样数量
       target_il_count = int(batch_size * il_ratio)  # 256 * 0.7 = 179
       target_rl_count = batch_size - target_il_count  # 256 - 179 = 77

       # 从IL pool采样179个，从RL pool采样77个
       il_chosen = self._pick(il_indices, target_il_count)
       rl_chosen = self._pick(rl_indices, target_rl_count)
   ```

3. **梯度更新**:
   - 每个batch: 179个IL样本 (成功经验) + 77个RL样本 (探索经验)
   - 70%的梯度来自成功策略 → 保护BC知识
   - 30%的梯度来自探索 → 允许改进

### 探索率降低的影响

**修复前** (epsilon=0.35):
- 4 workers × 2 episodes = 8 episodes/batch
- 35%探索 → 约2.8个episodes是随机动作 → 可能失败
- 如果失败，80次更新会强化失败策略

**修复后** (epsilon=0.10):
- 8 episodes/batch
- 10%探索 → 约0.8个episodes有随机动作
- 大部分episodes使用BC策略 → 成功率高
- 即使有失败，IL兜底会提供179/256的成功样本对抗

---

## 验证清单

运行下次训练时，检查以下日志：

### ✅ 应该看到的日志

1. **BC Prefill验证** (IL→RL切换后):
   ```
   [BC-PREFILL] Buffer size after BC prefill: 3032
   [BC-PREFILL-VERIFY] Buffer composition: IL=3032, RL=0, Total=3032
   [BC-PREFILL-VERIFY] ✓ BC episodes correctly tagged as IL
   ```

2. **Episode 1 统计**:
   ```
   [BUFFER-STAT] ep=1 buffer: IL=3032, RL=0, Total=3032 | il_ratio_target=70.0%
   [IL-RATIO] ep=1 il_ratio=70.0% (buffer采样比例)
   [EPSILON] ep=1 eps=0.100 succ_rate=0.000
   ```

3. **Episode 1 采样** (ep=1完成后):
   ```
   [THREE-TIER-SAMPLE] ep=1 last_batch: IL=179/256 (70.0%), Success=..., Timeout=..., Collision=...
   ```
   - IL应该约等于 256 * 0.7 = 179个

4. **Episode 11, 21, 31... 持续验证**:
   ```
   [THREE-TIER-SAMPLE] ep=11 last_batch: IL=175/256 (68.4%), ...
   [THREE-TIER-SAMPLE] ep=21 last_batch: IL=178/256 (69.5%), ...
   ```

5. **成功率趋势** (应该平滑):
   ```
   ep=1: s/c/t=100.0%/0.0%/0.0%
   ep=2: s/c/t=100.0%/0.0%/0.0%
   ep=3: s/c/t=87.5%/12.5%/0.0%   ← 可能有轻微下降
   ep=10: s/c/t=85.0%/10.0%/5.0%  ← 稳定在高成功率
   ```

### ⚠️ 问题信号

1. **IL标记失败**:
   ```
   [BC-PREFILL-VERIFY] ⚠️ Unexpected IL/RL ratio, check phase setting
   ```
   → BC prefill没有被标记为IL，需要检查explorer.py

2. **IL采样比例错误**:
   ```
   [THREE-TIER-SAMPLE] ep=1 last_batch: IL=20/256 (7.8%), ...
   ```
   → 配置70%但实际只有7.8%，采样逻辑有bug

3. **成功率断崖**:
   ```
   ep=1: 100% → ep=5: 40% → ep=10: 20%
   ```
   → IL兜底完全失效，需要深入调试

---

## 预期效果

### 成功率曲线

**修复前**:
```
ep=1-3: 100% → ep=5: 60% → ep=7: 42.9% ⚠️ 断崖式下降
```

**修复后 (预期)**:
```
ep=1-10: 95-100% (BC策略主导，10%探索)
ep=11-50: 90-95% (IL兜底70%，逐渐适应RL)
ep=51-100: 85-92% (IL兜底降至60%，RL成熟)
ep=100+: 稳定在85-95%，可能超越BC的91%
```

### 训练曲线特征

1. **平滑过渡**: 不应该有断崖式下降
2. **IL锚点**: 成功率应该维持在IL数据的水平附近
3. **逐步改进**: 随着RL学习，可能在某些episode超越91%
4. **稳定性**: 方差应该小，不会大幅波动

---

## 理论支撑 (来自 todo.md)

### 问题根源
> "行为克隆的'错误累加'：小错误→陌生状态→更差决策→恶性循环"

### 解决方案
> "异策略 + 经验回放：混合IL成功数据和RL探索数据"
> "不能丢弃91%成功率的好经验"

### 关键机制
1. **经验回放缓存**: 3032个IL episodes持续保留
2. **异策略学习**: AWR可以学习不同策略的数据
3. **混合采样**: 70% IL + 30% RL，平衡学习
4. **降低探索**: 10% vs 35%，减少失败经验

---

## 总结

### 核心修复
1. ✅ **BC Prefill标记为IL** - 让IL兜底机制能识别高质量episodes
2. ✅ **降低探索率** - 从35%降至10%，减少失败经验70%
3. ✅ **提高IL采样比例** - 从60%升至70%，更强的BC保护
4. ✅ **改善数据比例** - 从1:20到1:10，减少过拟合

### 预期改善
- 成功率不会断崖式下降（100%→42%）
- 而是平滑过渡（95%→90%→85%）
- IL兜底确保始终学习成功经验
- 最终可能超越BC的91%基线

### 下一步
1. 运行训练，观察新增日志
2. 验证IL采样比例是否符合70%
3. 确认成功率曲线是否平滑
4. 如有问题，参考FIXES_ANALYSIS.md调试
