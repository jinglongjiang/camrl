# 训练问题分析与修复建议

## 基于 todo.md 和 train.log 的综合分析

### 观察到的问题

1. **成功率快速下降**
   - 前7个episodes: 100% → 100% → 100% → 75% → 60% → 50% → 42.9%
   - 符合todo.md中描述的"灾难性遗忘"问题

2. **BC模型基线很强**
   - BC checkpoint: 91% 成功率
   - BC prefill评估: 100% 成功率（16 episodes）
   - 这说明BC已经学到了很好的策略

3. **探索率过高**
   - epsilon_start = 0.35 (35%随机探索)
   - 对于已有91%成功率的BC模型，35%探索率会产生大量失败经验
   - 这些失败经验如果被过度学习，会破坏BC的知识

4. **数据/更新比例不平衡**
   - 每个episode做80次梯度更新
   - 但每次只收集4个新episodes (4 workers × 1 episode)
   - 比例：4 episodes : 80 updates = 1:20
   - 如果新episodes中有失败案例，会被反复学习80次

### 已完成的修复

✅ **修复1: 添加BC prefill到IL→RL切换** (train.py:2587-2594)
- 在IL完成后添加32个BC episodes
- 提供更多高质量初始数据

✅ **修复2: BC prefill使用phase='il'** (train.py:1246)
- 确保BC episodes被标记为IL数据
- 使IL兜底机制能够识别这些高质量episodes

✅ **修复3: 添加BC prefill验证** (train.py:1250-1258)
- 验证BC episodes是否正确标记为IL
- 输出buffer的IL/RL组成

✅ **修复4: 增加IL采样统计频率** (train.py:1864-1871)
- 前100 episodes每10个输出一次（原来是每50个）
- 可以更早发现IL兜底是否生效

✅ **修复5: 添加episode 1的buffer统计** (train.py:1466-1470)
- 立即验证buffer的初始状态
- 确认IL/RL比例是否符合预期

### 待完成的优化建议

#### 优化1: 降低初始探索率 ⭐⭐⭐ (高优先级)

**问题**: 35%探索率对于91%成功率的BC模型太高

**当前配置** (train.config):
```ini
epsilon_start = 0.35
epsilon_end = 0.1
epsilon_decay_episodes = 2000
```

**建议修改**:
```ini
epsilon_start = 0.10    # 从35%降到10%，BC已经很强了
epsilon_end = 0.05      # 最终保留5%探索
epsilon_decay_episodes = 3000  # 延长衰减周期，更平滑
```

**理由**:
- BC已91%成功，应该信任BC的策略
- 35%探索会产生大量失败，破坏BC知识
- 10%起始探索率足够发现新策略，同时保护BC基础
- SARL论文中，探索率范围是0.5→0.1，但那是从头训练；我们有BC基础，应该更保守

#### 优化2: 调整数据/更新比例 ⭐⭐ (中优先级)

**问题**: 4 episodes : 80 updates = 1:20，可能过拟合

**方案A: 增加每次收集的episodes**
```ini
[vectorize]
num_workers = 4
episodes_per_worker = 2  # 从1改为2
# 这样每次收集8个episodes，比例变成 8:80 = 1:10
```

**方案B: 减少更新次数**
```ini
updates_per_ep = 40  # 从80减半
# 比例变成 4:40 = 1:10
```

**推荐**: 方案A（增加采集）优于方案B（减少更新）
- 更多数据能提供更丰富的状态覆盖
- 保持足够的更新次数有利于充分学习

#### 优化3: 验证IL采样真实比例 ⭐⭐⭐ (高优先级)

**当前状态**:
- 配置: il_ratio_start = 0.60 (60% IL数据)
- 日志显示: [IL-RATIO] ep=1 il_ratio=60.0%
- **但实际采样比例未验证**

**需要检查**:
1. 运行训练，观察新增的日志:
   - `[BC-PREFILL-VERIFY]` - BC episodes是否标记为IL
   - `[BUFFER-STAT]` - buffer的实际IL/RL组成
   - `[THREE-TIER-SAMPLE]` - 每个batch的实际IL采样数量

2. 如果ep=1的`[THREE-TIER-SAMPLE]`显示IL占比远低于60%，说明采样逻辑有bug

#### 优化4: IL ratio衰减策略调整 ⭐ (低优先级)

**当前配置**:
```ini
il_ratio_start = 0.60   # 60% IL
il_ratio_end = 0.30     # 30% IL
il_ratio_decay_episodes = 3000
```

**考虑调整**:
```ini
il_ratio_start = 0.70   # 提高到70%，更强的IL保护
il_ratio_end = 0.40     # 提高到40%，始终保持较强IL锚点
il_ratio_decay_episodes = 5000  # 延长衰减，更平滑过渡
```

**理由**:
- todo.md强调"不能丢弃91%成功率的好经验"
- 更高的IL比例能更好地防止灾难性遗忘
- 即使在后期，40%的IL数据仍能提供稳定的锚点

### 关键验证点

运行下次训练时，重点观察以下日志：

1. **BC Prefill验证** (应在IL→RL切换后立即出现)
   ```
   [BC-PREFILL-VERIFY] Buffer composition: IL=3032, RL=0, Total=3032
   [BC-PREFILL-VERIFY] ✓ BC episodes correctly tagged as IL
   ```

2. **Episode 1 Buffer统计**
   ```
   [BUFFER-STAT] ep=1 buffer: IL=3032, RL=0, Total=3032 | il_ratio_target=60.0%
   ```

3. **Episode 1 采样统计** (应在ep=1完成后出现)
   ```
   [THREE-TIER-SAMPLE] ep=1 last_batch: IL=154/256 (60.2%), Success=..., ...
   ```
   - IL应该约等于 256 * 0.6 = 154个

4. **Episode 11 采样统计** (验证持续性)
   ```
   [THREE-TIER-SAMPLE] ep=11 last_batch: IL=~150/256 (58-62%), ...
   ```

### 成功标准

如果IL兜底正确工作，应该看到：

1. **成功率稳定**: 不应该从100%快速降到42%
   - 预期：90% → 85% → 80% → ... (缓慢下降)
   - 而非：100% → 42% (断崖式下降)

2. **IL采样比例符合配置**:
   - 配置60%，实际应在55-65%之间（有随机性）

3. **Buffer组成正确**:
   - 初始3032个episodes都应该是IL
   - 随着训练进行，RL episodes逐渐增加

### 根本原因分析 (基于todo.md理论)

todo.md指出的核心问题：
> "错误累加"（error accumulation）- BC在陌生状态下犯错，导致恶性循环

**现在的情况**:
1. ✅ 有经验回放缓存（3000 IL + 32 BC = 3032条好经验）
2. ✅ 使用异策略算法（AWR + SAC风格的value learning）
3. ✅ 配置了il_ratio=60%的混合采样
4. ❓ **待验证**: IL采样是否真的生效
5. ⚠️ **问题**: 35%探索率产生太多"坏经验"

**解决路径**:
1. 验证IL采样生效（通过新增日志）
2. 降低探索率（减少"坏经验"产生）
3. 确保IL数据始终在训练中占主导地位

### 下一步行动

1. **立即**: 运行训练，检查新增的验证日志
2. **如果IL采样生效**: 降低epsilon_start到0.10
3. **如果IL采样未生效**: 调试memory.py的sample函数
4. **观察100个episodes**: 成功率曲线应该平滑而非断崖

### 参考

- todo.md: 强调IL+RL混合采样的重要性
- train.log: 显示成功率快速下降的问题
- 《深度强化学习》: Off-policy + Experience Replay的理论基础
