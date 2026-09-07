# Round 12 修改报告（发给 Codex 审阅）

## 结论

审查指出的两个阻断性问题（教师时间序列错位、baseline未走TTC排序）均已确认为真实bug并修复；
清单中另外6项（等价性测试、梯度单测固化、regret归一化排除哨兵值、文档修正、源码哈希入manifest、
先测试后重跑）也已全部完成。双头Belief-MDP架构本身未改动。

修复后已跑通：
- `selftest.py`：23项全过（本地+远程）。
- `test_teacher_equivalence.py`（新增，需GPU）：162个真实状态，输入token最大误差9.5e-7
  （浮点噪声级别），top-1动作0处不一致，全部通过。
- 完整 `--smoke`：全流程无报错，`status=completed`。

Round11（跑到第156集、尚未开始梯度更新即被人工停止）的目录和日志原样保留，未删除。
已在新目录 `runs/belief_mdp_action_conditioned_beta0.5_round12_seed2407` 从零启动正式训练
（`action_conditioned`, `beta=0.5`, `seed=2407`），不复用Round11任何数据。

---

## 逐项修复

### 1. `teacher_scores()` 教师时间序列错位一帧（阻断性，已确认，已修复）

**位置**：`belief_mdp/runtime.py`，`BeliefMDPFeatureEngine.teacher_scores()`。

**问题**：修复前代码为：
```python
recent_raw = self.history[:-1][-(self.seq_len - 1):]
```
即排除了`encode()`刚追加的当前帧，理由是"每个候选动作自己提供next_token"——这个理由是错的：
`next_token`代表**下一步**（t+1）的状态，不是当前帧（t）的替代品。结果是喂给Mamba的序列变成
`[...更旧的历史] + [候选动作的t+1状态]`，当前帧t整体缺失。

**验证**：用Mamba自己的生产级lookahead实现`MambaRLPolicy.predict_sarl_style()`（`policy/mamba_rl.py:746`，
`test.py`同款路径）做对照，该函数自己的代码注释就写着：
```python
# 🔥 修复：先把当前state token加入历史（Mamba需要当前帧）
base_hist = list(self._history) + [current_token]
```
这本身就是原始项目里已经踩过并修复过的同一个坑，我在移植到`belief_mdp/runtime.py`时重新引入了它。
实测60个真实状态，修复前后教师Top-1动作36/60（60%）不同。

**修复**：
```python
recent_raw = self.history[-self.seq_len:]
past_tokens = _batch_joint34_to_tokens_vectorized(np.stack(recent_raw))
```
（后续`sequence = past_tokens + [next_token]`再截断到`seq_len`的逻辑不变，本身是对的。）

**影响范围**：Round 8引入Mamba教师之后的所有轮次（8、9、10，以及被停止的Round11开头156集）都
用的是错位的教师标签。Round 1-7用ORCA教师，不受影响。好消息是：即便教师标签错位，Stage2的
Q_R/Q_C自身TD+MC损失完全不依赖`teacher_scores()`，所以Round10导航表现依然接近满分——但Stage1/
DAgger阶段的模仿监督信号确实是错的，这也是Round10"导航很好但模仿一致率卡在64%"现象里，除了
Round11已修的"教师监督接错头"之外的另一个真实的混淆因素。

**新增回归测试**：`test_teacher_equivalence.py`（见第3项）。

---

### 2. 冻结Mamba baseline未走TTC排序（已确认，已修复）

**位置**：`belief_mdp/evaluate.py`，`run_mamba_baseline_episode()`。

**问题**：
```python
action = mamba.predict(environment.joint_state())
```
`environment.joint_state()`（`runtime.py:618`）按仿真器原始顺序传递行人，未按TTC排序。而候选
策略（`BeliefMDPFeatureEngine`）构造Mamba输入时始终先调用`sort_humans_by_ttc`
（`runtime.py:183`，与`test.py:547`的生产路径逐行一致）。这意味着baseline和候选策略走的输入
构造代码路径不等价。

**实测**：30个初始状态的抽查未观察到最终动作差异，但代码路径不统一本身就是问题——无法保证在
所有场景下都不产生差异，且不严谨的地方会被审稿人质疑。

**修复**：`run_mamba_baseline_episode()`改为自己构造TTC排序后的`JointState`：
```python
sorted_humans = sort_humans_by_ttc(environment.robot, list(environment.env.humans))
state = JointState(
    environment.robot.get_full_state(),
    [human.get_observable_state() for human in sorted_humans],
)
action = mamba.predict(state)
```
`FullCrowdNavigationEnvironment.joint_state()`本身未改动（它还被已经废弃不用的ORCA
`expert_action()`路径引用，ORCA是顺序无关的planner，不受影响，故未改动该共享方法本身，只改了
真正需要TTC排序的调用点）。

---

### 3. 新增真正的教师等价性测试（新文件 `belief_mdp/test_teacher_equivalence.py`）

不是smoke test，是专门针对第1项bug类型设计的回归测试，需要GPU（无法在3060开发机跑）：

- 用真实CrowdSim环境跑3个场景、每场景最多60步、共162个真实状态（覆盖历史从空到填满
  `seq_len`的全过程，不只测第一帧）。
- 通过临时monkeypatch `mamba.forward_value`（不修改被测函数本身任何代码）分别捕获
  `teacher_scores()`和`predict_sarl_style()`真正喂给Mamba的输入张量。
- 断言：≥100个真实状态、输入张量误差<1e-4、Top-1动作一致率100%。
- 实测结果：162个状态，最大误差9.5e-7，0处不一致。

---

### 4. 新增"教师margin不进入Q_C"的梯度单测（固化进 `selftest.py`）

Round11报告里"Q_R梯度非零、Q_C梯度严格为零"是人工验证的，现在固化为
`test_margin_loss_on_qr_never_reaches_qc_head`：构造网络，把`q_c_head`最后一层从零初始化
扰动为随机非零权重（否则零初始化会让所有梯度都恰好是零，掩盖真实的架构隔离情况），对`Q_R`
算margin loss并反向传播，断言`risk_belief_encoder`/`risk_action_encoder`/`q_c_head`三个子
模块梯度严格为零，`task_context_encoder`/`task_kinematic_encoder`/`q_r_head`三个子模块梯度
非零。已通过。

---

### 5. `normalized_teacher_regret` 未排除安全过滤的`-1e4`哨兵值（已确认，已修复）

**问题**：`runtime.py`的`teacher_scores()`会把被安全过滤判定为不安全的动作分数设为`-1e4`
哨兵值。旧版`normalized_teacher_regret`直接对全部80个分数取min/max做归一化，只要有一个动作
被标记为不安全，`teacher_worst`就会被拖到约`-1e4`，导致归一化区间被撑到~1e4量级，把所有其他
动作的regret值压缩到接近0——不管它们真实分数差异有多大。实测60个状态中6个含哨兵值。

**修复**：min/max计算时排除低于`-1e3`阈值（哨兵值`-1e4`之上留有余量）的动作；若整行80个动作
全部被标记不安全（极端退化情形）才回退到用全部分数计算范围。若网络自己选择的动作恰好就是被
标记为不安全的那个，regret直接钳制为1.0，而不是产生一个无界的大数字污染批量平均值。

**新增测试**：`selftest.py`新增2项，验证哨兵值被正确排除、选中哨兵动作本身钳制为1.0。

---

### 6. 文档修正：Q_C标签目前只有终止碰撞，没有近碰成本

`run_episode()`里`risk_cost = float(result.done and result.outcome == "collision")`——纯
终止碰撞二值指示，不含任何`dmin`近碰分量。之前Round11引入的一处注释错误地写成"collision/
near-miss cost"，已改正为准确描述，并明确指出"Fixed in round 2"那段历史记录（曾经加过近碰
项）已不再反映当前代码，是历史记录不是当前行为描述。

---

### 7. 源码哈希写入manifest

`compute_artifact_hashes()`新增`train_py_sha256`/`model_py_sha256`/`runtime_py_sha256`/
`evaluate_py_sha256`四项，写入每个checkpoint的`artifact_hashes`和`run_manifest.json`。
`evaluate.py`加载checkpoint后会对比这四个哈希，若当前源码与训练时不同，打印WARNING（不阻断——
旧checkpoint仍需能被新代码评测）。

---

### 8. 修复后先测试、再从新目录重跑（已按顺序执行）

1. `selftest.py`（本地+远程）：23项全过。
2. `test_teacher_equivalence.py`（远程GPU）：162个状态全过。
3. 完整`--smoke`（远程GPU）：`status=completed`，无报错。
4. 以上全部通过后，才在新目录`runs/belief_mdp_action_conditioned_beta0.5_round12_seed2407`
   从零启动正式训练——不复用Round11的任何数据或checkpoint。

---

## 已独立复核、确认仍然正确的部分（不需要再动）

- 教师margin loss只作用于`Q_R`，`Q_C`梯度严格为零（第4项新测试验证）。
- `Q_C`仍然只用仿真器真实碰撞结果训练，从未被教师监督。
- 最终决策规则仍是`Q_score = Q_R - beta*Q_C`。
- 教师一致率（Q_R版和Q_score版）已经完全退出Stage1硬门槛，只报告不拦截。
- Stage1门槛确实使用相同的seed/test_case/profile做baseline配对比较。
- 本地与4090五个核心文件（runtime.py/train.py/model.py/evaluate.py/selftest.py）SHA256
  在复核时完全一致。

**结论：双头Belief-MDP架构本身不需要推倒重来。这轮问题是"教师复刻代码与Mamba原生实现不等价"
和"候选/baseline输入协议不统一"，不是决策架构的问题。**

## 当前状态

Round12正式训练已启动（`action_conditioned, beta=0.5, seed=2407`），Stage1a教师数据收集阶段。
后续会持续监控并汇报。
