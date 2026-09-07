## 13. S1严格科学必要性门禁：唯一执行规范（CC从这里开始）

本节冻结于2026-08-04，优先级高于旧`u3_necessity_gate.py`和所有历史Order。CC只能按`S1-0 -> S1-7`串行执行；一次只做一个Order，验收通过后再进入下一个。不得在看到necessity/audit结果后改seed、K范围、先验、EM阈值、bootstrap方法或PASS条件。

### 13.1 S1要回答什么

S1只回答三个科学问题，不跑导航SR/CR：

1. 多模态切换是否必要：action-conditioned `K*>1`是否稳定优于同结构`K=1`。
2. 机器人动作条件是否必要：action-conditioned是否稳定优于`self_only`。
3. 优势是否来自正确动作-结果关系：action-conditioned是否稳定优于`constrained_action_shuffle`，并且在没有机器人因果响应的nominal负对照中不应继续表现出显著优势。

S1 GO只允许声称“在预注册的闭环交互仿真协议内，完整切换后验和正确机器人动作条件具有独立的预测必要性”。它不等于导航更安全，导航层必要性必须由S2证明；也不等于真实人类行为已经验证，因为当前数据仍由项目的交互FSM生成。论文必须如实限定这一外部有效性边界，除非以后另加真实交互数据。

### 13.2 预注册常量（禁止结果后修改）

实现时新增`crowd_nav/configs/s1_strict_registry.json`，内容必须逐项包含并校验以下值：

```text
schema_version: 1
experiment_id: s1_strict_20260804
scenario: baseline_circle
n_humans: 5
dt: 0.25
horizon_steps: 40
controllers: [goal_directed, orca, original_brne, scripted_probe]

train:
  path: runs/bayesian_brne/data_formal/train/baseline_circle
  suite_seeds: [11,12,13,14,15]
  episodes_per_seed: 400

selection:
  path: runs/bayesian_brne/data_formal/validation/baseline_circle
  suite_seeds: [21,22,23,24,25]
  episodes_per_seed: 100

necessity_id:
  environment_split: validation
  output_root: runs/bayesian_brne/s1_strict_20260804/data/necessity_id
  suite_seeds: [31,32,33,34,35,36,37,38,39,40]
  episodes_per_seed: 100
  profile_name: s1_strict_necessity

audit_interactive:
  environment_split: test_heldout_interactive
  output_root: runs/bayesian_brne/s1_strict_20260804/data/audit_interactive
  suite_seeds: [41,42,43,44,45,46,47,48,49,50]
  episodes_per_seed: 100
  profile_name: s1_strict_audit_interactive

audit_nominal_negative_control:
  environment_split: test_nominal
  output_root: runs/bayesian_brne/s1_strict_20260804/data/audit_nominal
  suite_seeds: [51,52,53,54,55,56,57,58,59,60]
  episodes_per_seed: 100
  profile_name: s1_strict_audit_nominal

primary_k_candidates: [1,2,3,4]
boundary_extension_k_candidates: [5,6]
restart_seeds: [2407,3407,4407]
sticky_kappa: 10.0
dirichlet_alpha: 2.0
shrinkage_scale: 1.0
inverse_wishart_dof: 6.0
inverse_wishart_scale: 0.01
em_max_iters: 200
em_tol: 0.0001
bootstrap_resamples: 10000
bootstrap_seed: 72407
confidence_level: 0.95
nominal_equivalence_margin_nats_per_row: 0.01
min_mode_fraction: 0.03
max_predictive_similarity: 0.98
```

seed `31–60`已经过当前workspace扫描，未出现在现有3960个SM-BRNE episode中。实现仍必须在S1-0重新扫描；若发现冲突，不得自行换seed，必须停止汇报。

### Order S1-0：冻结registry、代码身份和回滚点

目标：任何数据生成或拟合发生前，先冻结方法和判据。

必须实现：

- 创建上述registry；JSON使用排序key和稳定序列化。
- 创建`crowd_nav/bayesian_brne/s1_protocol.py`，集中实现registry解析、数据身份检查、restart选择、K选择、受约束shuffle、bootstrap和三态判定；统计逻辑不得散落在CLI脚本里。
- 创建`crowd_nav/tools/s1_strict_gate.py`，CLI固定为：

```bash
python3 -m crowd_nav.tools.s1_strict_gate \
  --registry crowd_nav/configs/s1_strict_registry.json \
  --stage preflight|collect_necessity|fit_ac|select_k|fit_controls|necessity|collect_audit|audit|promote
```

- 创建`crowd_nav/tools/run_s1_queue.py`作为唯一自动队列入口；它只调用上面的stage，不复制统计实现。
- 输出根固定为`runs/bayesian_brne/s1_strict_20260804/`，至少包含：

```text
frozen_registry.json
preflight_manifest.json
source_manifest.json
status.json
controller.log
data_manifests/
fits/action_conditioned/K{K}/restart_{seed}/
fits/self_only/K{K}/restart_{seed}/
fits/constrained_action_shuffle/K{K}/restart_{seed}/
selection_report.json
necessity_report.json
audit_unlock.json
audit_report.json
production_artifact/
```

- `source_manifest.json`逐文件记录以下源码/config SHA256：AR-HMM、data_io、interaction_protocol、s1_protocol、s1 CLI、collector、registry。记录Python/NumPy/SciPy/numba版本、git HEAD和相关文件的`git status --short`；不得因全仓其他脏文件而失败，但相关文件在冻结后变化必须失败。
- 创建只包含SM-BRNE相关源码、registry和manifest的tar备份及SHA256，不夹带旧大模型和无关工作区文件。
- 扫描所有现有SM-BRNE episode的`suite_seed/episode_seed/initial_state_hash`；train、selection和预注册新seed不得重叠。
- 旧`u3_necessity_gate.py`及其结果只读冻结，S1不得import、覆盖或把其GO当先验。

验收与停止条件：

- 新增S1测试组并证明registry字段缺失、seed重叠、源码hash变化、输出目录身份不一致时全部fail closed。
- `preflight_manifest.json`、tar和hash齐全才PASS。
- 任一身份冲突为`PRECHECK_FAIL`，停止，不采集数据。

### Order S1-1：实现严格数据角色与受约束shuffle

目标：selection、necessity、audit永不混用；shuffle只破坏动作因果关系，不破坏controller和场景边际。

数据角色规则：

- train只拟合参数。
- selection只选择K，绝不计算最终necessity结论。
- necessity_id只做开发阶段必要性门禁，不参与K选择、restart选择或拟合。
- audit_interactive和audit_nominal在necessity GO前禁止加载；queue代码的audit loader只能位于GO分支后。
- `latent_behavior_labels`永远不得进入特征、拟合、K选择或PASS条件；最终GO后可以单独用于解释性可视化，但必须标记post-hoc。

受约束shuffle必须满足：

1. 只在相同`controller_type`内部配对。
2. donor episode必须来自不同`suite_seed`且不能是自身。
3. action序列shape和有效长度必须一致；有效长度按5-step bin匹配。
4. 在同controller/长度块内，根据不含未来结果和latent标签的初始可观测context摘要做确定性最近邻异seed配对。摘要至少包含初始robot-human最小距离、初始平均相对速度、profile speed区间、yield/goal-switch TTC阈值和assertive probability；各维用train统计标准化。
5. donor分配必须是derangement；每个donor恰好使用一次，保证动作序列多重集合hash在shuffle前后相同。
6. 配对由registry中的固定seed一次性生成并保存为`shuffle_map.json`；三个restart共用同一映射。

必须测试：

- 无self-pair、无跨controller、无同suite-seed、无shape/长度不匹配。
- shuffle前后动作序列多重集合SHA256一致，至少95%的episode动作内容实际改变。
- 同registry重复生成的`shuffle_map.json`逐字节一致。
- 故意构造无法异seed配对的block时明确失败，不回退到全局乱洗。
- self_only训练输入中`u_robot`逐位为0；评分时使用真实观测序列，且artifact的`B_k`数值应为0到浮点容差。

S1-1只在本地运行单测，不采正式数据。全部新增测试、原96项selftest和15项上游等价测试通过后才进入S1-2。

### Order S1-2：只采necessity数据，audit保持未生成

heavy任务运行在4090；本地只同步代码、检查manifest和读取结果。CC不得在自己的隔离sandbox后台跑正式采集。

queue按registry为seed `31–40`逐个运行等价命令：

```bash
python3 -m crowd_nav.bayesian_brne.collect_dataset \
  --split validation --scenario baseline_circle --episodes 100 \
  --seed <31..40> --profile-name s1_strict_necessity \
  --output-dir runs/bayesian_brne/s1_strict_20260804/data/necessity_id \
  --horizon-steps 40 --dt 0.25
```

S1 wrapper必须提供可恢复语义：目标文件不存在才生成；已存在且SHA/metadata完全一致则跳过；存在但不一致则fail closed，禁止覆盖。每个seed完成后立即写子manifest，全部完成后写aggregate manifest。

necessity数据门禁：

- 精确1000集，10个suite seed各100集；每seed四控制器各25集，总计每控制器250集。
- 全部`schema_version=2`、5人、baseline_circle、split=validation、profile_name正确、dt=0.25。
- fallback event为0，初始实体overlap为0，NaN/Inf为0。
- 与train/selection的suite_seed、episode_seed和initial_state_hash交集均为空。
- generator/source/config hash与frozen registry一致。
- audit两个目录此时必须不存在；若已经存在，状态记为`AUDIT_CONTAMINATED`并停止。

### Order S1-3：action-conditioned多restart拟合与K选择

拟合只读train；selection只用于K选择。每个fit结束立即原子保存artifact、model card、完整objective history、convergence、耗时、峰值内存和日志。未完成fit不得留下可被resume误认的完成标志。

第一阶段固定运行12个fit：

```text
K = 1,2,3,4
restart seed = 2407,3407,4407
variant = action_conditioned
```

每个K内的restart选择规则：只在`converged=true`的restart中，选择train penalized objective最高者。不得用selection/necessity/audit选择restart。若某K三个restart都未收敛，该K无资格参加选择。

K资格检查：

- artifact数值finite，所有Q正定，Pi每行和initial distribution归一。
- 每个mode的train occupancy `>=0.03`。
- 最大predictive similarity `<=0.98`。
- 对应restart明确`em_tol_met`；`em_max_iters_exhausted`不得保存为production或进入选择。

K选择规则：

1. 用每个K选中的restart在selection seeds `21–25`计算逐suite-seed sequential NLL/row。
2. 在合格K中找selection平均NLL最低的`K_best`。
3. 计算`K_best`五个suite-seed NLL的标准误`SE_best = sample_std/sqrt(5)`。
4. 选择满足`mean_NLL(K) <= mean_NLL(K_best) + SE_best`的最小K，记为`K*`。
5. `K*`必须大于1，并且`K*`相对K=1的suite-seed block bootstrap NLL improvement 95% CI下界必须`>0`；否则S1直接`NO_GO_MULTIMODALITY`，不拟合controls、不打开audit。

边界规则：如果K=4是`K_best`，且K4相对K3的paired bootstrap improvement CI下界`>0`，标记`BOUNDARY_HIT`。只允许一次预注册扩展到K=5,6，同样各3个restart、同样`em_max_iters=200`；禁止扩到K>6。若K5/6均不能收敛或K6仍显著优于K5，最终为`INCONCLUSIVE_BOUNDARY`并停止，不得挑K4凑GO。

输出`selection_report.json`必须包含所有K/restart，而不是只保留胜者；报告未收敛原因、mode occupancy、predictive similarity、Q特征值、Pi对角线、NLL/seed、one-SE计算和边界判定。

### Order S1-4：拟合两个对照并运行necessity门禁

只有S1-3得到已解析的`K*>1`才执行。

在同一train、同一K*、同一三个restart seed和同一先验下分别拟合：

- `self_only`：训练动作逐位清零。
- `constrained_action_shuffle`：使用S1-1冻结的`shuffle_map.json`。

每个variant仍只按train penalized objective选择最佳收敛restart。任何variant三个restart全部未收敛为`INCONCLUSIVE_OPTIMIZATION`，不是PASS，也不得打开audit。

必要性评分只使用necessity_id，主比较为：

```text
AC(K*) - self_only(K*)
AC(K*) - constrained_action_shuffle(K*)
AC(K*) - AC(K=1)
```

这里“AC - baseline improvement”定义为`NLL_baseline - NLL_AC`，正数表示action-conditioned更好。CI使用suite-seed block bootstrap，只重采样10个suite seed，10000次；episode和transition不能当独立样本伪增样本量。

necessity GO必须同时满足：

1. 三个主比较的95% CI下界全部`>0`。
2. 对AC-vs-self和AC-vs-shuffle，四个controller中至少3个controller点估计`>0`。
3. 不允许任何controller的95% CI上界`<0`，防止总体收益由牺牲某一controller换来。
4. 三个restart全部单独报告；至少2/3 restart的AC-vs-self和AC-vs-shuffle点估计方向为正。主统计仍使用预先按train objective选中的artifact，不按necessity结果换restart。
5. 数据身份、源码hash、registry hash和shuffle hash均与S1-0一致。

NLL之外同时报告Brier、coverage/ECE、mode occupancy和action-identifiability condition number，但这些是secondary diagnostics，不得用来挽救NLL门禁失败。

若上述任一科学比较失败，判定`NO_GO`并停止；不得生成audit数据。若只是数据/优化/身份不足，判定`INCONCLUSIVE`并停止，不能写成科学FAIL或PASS。

### Order S1-5：necessity GO后自动生成并一次性打开audit

只有`necessity_report.json`为GO且所有hash复核通过，queue才原子写入`audit_unlock.json`。该文件必须包含necessity报告SHA256、registry SHA256、源码aggregate SHA256和时间；无unlock时audit stage必须拒绝运行。

随后按预注册seed自动采集，期间不允许人工修改代码：

```bash
# heldout interactive正对照，seed 41..50，每seed 100集
python3 -m crowd_nav.bayesian_brne.collect_dataset \
  --split test_heldout_interactive --scenario baseline_circle --episodes 100 \
  --seed <41..50> --profile-name s1_strict_audit_interactive \
  --output-dir runs/bayesian_brne/s1_strict_20260804/data/audit_interactive \
  --horizon-steps 40 --dt 0.25

# nominal因果负对照，seed 51..60，每seed 100集
python3 -m crowd_nav.bayesian_brne.collect_dataset \
  --split test_nominal --scenario baseline_circle --episodes 100 \
  --seed <51..60> --profile-name s1_strict_audit_nominal \
  --output-dir runs/bayesian_brne/s1_strict_20260804/data/audit_nominal \
  --horizon-steps 40 --dt 0.25
```

两组各1000集、每controller 250集、0 fallback、0 overlap、身份不重叠；使用与S1-2相同的原子保存和manifest规则。

audit_interactive使用冻结artifact重复S1-4三个主比较，PASS条件与necessity完全相同。audit_nominal只做因果负对照：AC-vs-self improvement的95% CI必须完整落在`[-0.01,+0.01] nats/row`等价区间内；否则说明action项可能在利用代理相关性，而不是只捕获真实机器人响应，最终判定`NO_GO_CONFOUNDING`。

audit数据只允许评估一次。任何源码、registry、artifact或necessity报告hash变化都判定`AUDIT_INVALIDATED`；不得在同一audit seed上修代码后重跑。若是纯I/O崩溃且尚未产出任何统计，可按相同hash恢复；一旦`audit_report.json`写出，就永久封存。

### Order S1-6：最终三态判定与production artifact

最终状态只能是：

- `GO`：selection、necessity、interactive audit和nominal negative control全部通过。
- `NO_GO`：数据机会充分、执行正确，但任一预注册科学比较失败；停止SM-BRNE路线，不做F7加速和S2。
- `INCONCLUSIVE`：边界未解析、模型无法收敛、数据/身份/代码完整性失败；不得包装成正结果，必须由用户决定是否投入一次明确修复。

只有GO时，才把“训练train-only、K*、按train objective选中的action-conditioned restart”原样晋升为`production_artifact`；不允许在GO后合并selection/necessity/audit重新拟合。production model card必须包含：

- 完整AR-HMM参数和`converged=true`。
- train aggregate hash、源码aggregate hash、registry hash。
- K/restart选择规则及selection/necessity/audit报告hash。
- feature schema、时间语义、物理边界、完整training config。
- `tier=production`和内容hash。

晋升后重新load并逐位比较数组；任一差异或hash不匹配失败。config中的artifact path暂不自动切换，由下一Order F7-remediation开始前单独提交，避免S1工具悄悄改变部署配置。

### Order S1-7：运行位置、完整命令和CC汇报格式

本地阶段：实现代码、selftest、py_compile、manifest静态检查。heavy阶段：数据采集、全部EM fit和bootstrap统一同步到4090后运行。禁止CC sandbox后台正式跑。

同步前在本地运行：

```bash
cd /home/abc/workspace/nav_data/mamba/camrl/CrowdNav
python3 -m py_compile crowd_nav/bayesian_brne/*.py crowd_nav/tools/s1_strict_gate.py crowd_nav/tools/run_s1_queue.py
python3 -m crowd_nav.bayesian_brne.selftest
python3 -m crowd_nav.bayesian_brne.selftest --group s1
python3 -m crowd_nav.bayesian_brne.test_upstream_equivalence
python3 -m crowd_nav.tools.s1_strict_gate --registry crowd_nav/configs/s1_strict_registry.json --stage preflight
```

4090唯一正式入口必须由宿主可见的`nohup`进程启动：

```bash
cd /root/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav
nohup /root/miniconda3/envs/mamba/bin/python3 -u -m crowd_nav.tools.run_s1_queue \
  --registry configs/s1_strict_registry.json \
  --output-dir runs/bayesian_brne/s1_strict_20260804 \
  > runs/bayesian_brne/s1_strict_20260804/controller.log 2>&1 < /dev/null &
```

queue必须把PID写入`pid`、当前stage/fit写入`status.json`，每个EM iteration至少输出K、variant、restart、iteration、objective、delta、elapsed和RSS。只允许单个fit进程串行运行，禁止并行restart导致内存/线程差异。完成一个fit后立即保存，queue重启时只跳过manifest/hash一致且`status=completed`的fit。

CC每个Order汇报必须固定包含：

```text
Order:
修改文件:
输入数据角色/seed/count/hash:
执行命令与执行机器:
新增测试及旧实现失败证据:
selftest / s1 group / upstream结果:
产物路径与SHA256:
原始指标（不只给PASS字样）:
PASS / NO_GO / INCONCLUSIVE:
未完成项:
下一Order（只写一个）:
```

### 13.3 CC现在可以做什么

CC现在只允许开始`S1-0`，完成registry、代码骨架、preflight、回滚包和测试后停下汇报。不得一边实现一边直接跑全部fit；Codex/用户验收S1-0R、S1-1和后述`S1-BUILD`后，才同步4090进入S1-2。后续S2/S3的具体规模不在本节提前冻结，因为它们只有S1 GO后才有科学意义。

### 必须先修的阻断项（严格按R0R-1到R0R-7执行）

#### R0R-1：移除本机绝对路径，统一仓库根定位

当前`s1_strict_gate.py`和`run_s1_queue.py`把`REPO_ROOT`硬编码为
`/home/abc/workspace/nav_data/mamba/camrl/CrowdNav`。4090真实路径是
`/root/workspace/nav_data/mamba/camrl/CrowdNav`，因此代码原样同步后会读写错误机器路径，这是正式运行阻断项。

修复要求：

- 唯一仓库根由源码位置推导：`Path(__file__).resolve().parents[2]`，或集中放到
  `s1_protocol.repo_root()`；CLI不得再出现`/home/abc`或`/root`字面量。
- 所有registry相对路径、输出路径和subprocess `cwd`都相对该根解析。
- `run_s1_queue --output-dir`必须解析后严格等于
  `<repo_root>/runs/bayesian_brne/<registry.experiment_id>`；不一致立即fail closed。
- 新增测试：把S1源码复制/映射到不同临时根后，根定位、输出根和registry解析仍正确；至少静态断言两个CLI中
  不含`/home/abc`、`/root/workspace`。
- 4090正式命令统一从仓库根运行：

```bash
cd /root/workspace/nav_data/mamba/camrl/CrowdNav
/root/miniconda3/envs/mamba/bin/python3 -m crowd_nav.tools.s1_strict_gate \
  --registry crowd_nav/configs/s1_strict_registry.json --stage preflight
```

#### R0R-2：统一`status.json` schema并原子写入

当前gate写`{stage,status}`，queue写`{queue_stage,queue_status}`，双方整体覆盖同一个文件。这不是小风险：
恢复时可能看不到stage结果，错误跳过或重复正式步骤。

修复要求：

- 在`s1_protocol.py`集中定义一个`STATUS_SCHEMA_VERSION`和唯一`update_status_atomic()`。
- 两个CLI都只能调用这个函数，禁止各自`write_text/json.dump`覆盖全文件。
- 固定字段至少为：`schema_version, experiment_id, registry_sha256, pid, queue_status,
  current_stage, stage_status, returncode, completed_stages, active_fit, started_at, updated_at`。
- 使用同目录临时文件+`os.replace`原子替换；更新时保留另一方已有字段。
- `--output-dir`身份、registry hash或experiment id与已有status不一致时fail closed。
- 测试覆盖：gate->queue、queue->gate两种写入顺序字段都不丢；中断后可准确恢复；损坏JSON拒绝运行。

#### R0R-3：preflight必须严格核验train/selection，而不只是检查新seed未使用

当前`check_seed_disjoint()`只验证31-60未出现在整个runs中，没有程序化证明registry指定路径内的数据
就是声明的2000/500集，也没有拒绝重复episode身份/initial-state身份。

修复要求：

- 只通过`data_io.load_episode()`读取registry中train/selection的精确目录，禁止裸`np.load`绕过schema校验。
- 对每个role验证：目录存在、文件总数、每个suite seed的精确集数、没有额外suite seed、scenario、split、
  profile=`formal`、5人、dt、horizon、controller均属于冻结registry。
- 验证每个role内`(suite_seed, episode_seed)`唯一，train与selection之间suite seed、episode seed、
  `initial_state_hash`全部不重叠。
- 全runs扫描仍保留，用于确认31-60从未使用；扫描候选episode时遇到损坏schema不得静默当作
  `skipped_non_episode`。非episode artifact可以按明确文件类型跳过，但疑似episode损坏必须PRECHECK_FAIL。
- 将逐seed count、identity hash集合摘要和每个目录的文件清单SHA256写入`preflight_manifest.json`。
- 测试必须故意制造：少1集、多1集、额外seed、重复episode id、重复initial hash、错误split/profile/scenario，
  六类都fail closed。

#### R0R-4：修正源码冻结生命周期，避免“冻结后又必须改同一文件”的矛盾

当前`source_manifest.json`冻结了`s1_protocol.py`和CLI，但S1-1到S1-6又明确要求继续修改这些文件；
现有preflight会在下一次运行时立刻因hash漂移失败。正确纪律不是放松hash，而是先把水管全部搭完，再最终封存。

修复要求：

- 现有manifest改名/标记为`source_manifest_s1_0_scaffold.json`，只表示S1-0回滚基线，不可冒充正式方法锁。
- S1-1完成后进入新增`S1-BUILD`：一次性实现9个stage、恢复逻辑和所有统计函数，只用合成fixture/已有
  train-selection数据做单测，不读取或生成necessity/audit数据。
- `S1-BUILD`全部验收后生成`method_source_manifest.json`和`method_lock.json`；从这一刻到S1-6结束，
  任一结果相关源码变化都必须使queue失败，不能自动重冻。
- 正式manifest至少覆盖：`crowd_nav/bayesian_brne/*.py`、两个S1 CLI、实际拟合/采集所调用的tools、
  registry和相关config；当前manifest漏掉`run_s1_queue.py`，必须补入。
- 将第13节冻结规范复制为输出目录内`frozen_protocol_spec.md`并记录SHA256。之后guide其他历史文字可继续更新，
  但这份冻结spec不可变化。

#### R0R-5：rollback验证必须校验tar文件自身SHA256

当前`verify_rollback_archive()`只核对解压后的逐文件hash，不核对manifest中的`archive_sha256`；现有tamper
测试改的是per-file manifest，不是tar本体。

修复要求：

- 解压前先计算tar实际SHA256并与`archive_sha256`常数时间比较；不一致直接失败。
- 捕获损坏tar的读取/解压异常并返回明确失败，不得抛出后被queue误判为未执行。
- 新增两条红案例：直接修改tar一个字节；只篡改`archive_sha256`。二者都必须失败。
- rollback manifest中的路径改为相对repo root或相对manifest位置，禁止把本机`/home/abc`写入将同步到4090的身份产物。

#### R0R-6：补齐registry精确常量与队列边界测试

- 当前registry文件内容与13.2一致，但validator大多只检查“类型/范围合法”，没有证明被改成另一组合法数字时会拒绝。
- 增加对冻结实验ID、scenario、5人、dt、horizon、五组seed/count、K集合、restart、先验、EM和bootstrap常量
  的精确回归测试；冻结文件任何合法值漂移都必须被`frozen_registry.json`或预期hash拒绝。
- `_S1_REGISTRY_PATH`不得依赖当前工作目录，测试路径也必须由repo root解析。
- 未实现stage保持返回码2；不得创建data/fits/audit_unlock等半成品。

#### R0R-7：S1-0R复验和停止条件

只在本地执行，不同步4090、不采数据：

```bash
cd /home/abc/workspace/nav_data/mamba/camrl/CrowdNav
python3 -m py_compile crowd_nav/bayesian_brne/*.py crowd_nav/tools/s1_strict_gate.py crowd_nav/tools/run_s1_queue.py
python3 -m crowd_nav.bayesian_brne.selftest --group s1
python3 -m crowd_nav.bayesian_brne.selftest
python3 -m crowd_nav.bayesian_brne.test_upstream_equivalence
python3 -m crowd_nav.tools.s1_strict_gate \
  --registry crowd_nav/configs/s1_strict_registry.json --stage preflight
```

验收必须同时满足：

1. 上述测试全绿，新增红案例真实覆盖R0R-1到R0R-6。
2. `preflight_manifest`含精确train/selection身份报告。
3. 两个CLI没有机器专属绝对路径，status统一且原子写入。
4. scaffold manifest、rollback和未来method lock三种概念不再混用。
5. `runs/bayesian_brne/s1_strict_20260804/data`与`fits`仍为空/不存在。

完成后必须停下汇报，下一Order仍然只能是`S1-1`，不得启动4090。

### 新增S1-BUILD（S1-1之后、S1-2之前）

为落实“完整框架先搭好，再通正式数据”的纪律，S1-1验收后不能立即采necessity数据。必须先完成以下建设：

1. 实现`collect_necessity/fit_ac/select_k/fit_controls/necessity/collect_audit/audit/promote`全部stage。
2. 用合成小数据验证3 restart选择、one-SE K选择、边界扩展、受约束shuffle、suite-seed bootstrap、
   nominal等价性、GO/NO_GO/INCONCLUSIVE、audit锁和production promotion。
3. 验证audit loader在`audit_unlock.json`出现前不可达；静态检查和运行时测试都要有。
4. 验证queue断点恢复：人为中止一个fit后只重跑未完成项；hash不一致的“completed”不得跳过。
5. 所有stage测试通过后才生成最终`method_source_manifest.json/method_lock.json`，并重新运行preflight。
6. Codex/用户验收method lock后才允许S1-2在4090采集seed 31-40。此后源码漂移只能作废本次实验，不能重冻续跑。

### A1：冻结协议未包含S1-0R与S1-BUILD（阻断）

当前`frozen_protocol_spec.md`只有340行；程序按“从`## 13`到下一个`##`”提取，因此在
`## Order S1-0 执行报告`处停止。实测以下三个关键字符串全部不在冻结文件中：

```text
Order S1-0R 独立验收
R0R-1
新增S1-BUILD
```

这意味着现在冻结的只是旧13.1-13.3，不是当前真正执行规范。此问题的源头包括Codex之前把补丁作为新的
二级标题追加到Section 13之后；CC实现没有验证“关键规范确实进入冻结文本”，双方都有责任。

修复要求：

1. 把所有规范性内容（原13.1-13.3、R0R-1到R0R-7、S1-BUILD）整理进连续的Section 13规范区；执行报告必须在
   明确的规范结束标记之后。
2. 使用显式标记`S1_PROTOCOL_FREEZE_START/END`提取，不再依赖“遇到下一个`##`就停止”的脆弱规则。
3. 因尚无正式S1数据，允许做一次有记录的协议冻结修正：生成`protocol_amendment.json`，记录旧hash、新hash、
   原因=`incomplete extraction boundary before formal data`、时间和registry hash；不得静默覆盖。
4. 新冻结文件必须包含上述三个关键字符串及全部预注册判据；测试必须断言关键章节存在、执行报告不存在。
5. 4090即使没有本地`guide.md`，也必须同步并校验仓库内已冻结的protocol spec/hash，不能“跳过就算通过”。

### A2：正式数据身份报告仍少三类证据（阻断）

`verify_formal_data_role()`当前返回字段只有seed count和initial-state hash集合；没有R0R-3明确要求的文件清单hash、
episode identity摘要和controller分层计数。`verify_train_selection_cross_disjoint()`也只比较initial-state hash，
没有比较原始`episode_seed`集合。

修复要求：

- 每个role新增并写入manifest：`file_list_sha256`、`episode_ids_sha256`、`episode_seeds_sha256`、
  `controller_counts`、`n_unique_episode_id`。
- train/selection跨角色同时检查suite seed、原始episode seed、`(suite_seed, episode_seed)`和initial-state hash。
- 当前正式数据必须锁定四控制器均衡：train每类500、selection每类125；不均衡fail closed。
- 增加错误scenario/profile/dt/horizon/n_humans/controller和controller缺失/失衡红案例；当前代码虽有部分检查，
  但现有测试只实际覆盖了wrong split。

### A3：全runs身份扫描允许缺失`initial_state_hash`（阻断）

独立反例：创建只含`suite_seed=31`和`episode_seed=3100000`、不含`initial_state_hash`的npz，当前扫描结果是
`matched=1, skipped=0`，没有报错。它不符合函数自己声明的三元身份契约。

修复要求：

- 任何同时带`suite_seed`/`episode_seed`的疑似episode必须同时带非空、格式合法的`initial_state_hash`；否则
  `PreflightError`。
- 对suite/episode/hash任一缺失、错误dtype、空hash和重复身份分别增加红案例。
- 明确非episode artifact仍可跳过，但只能依据一组显式允许的artifact schema/key签名，不得用“少一个字段”泛化跳过。

### A4：`status.json`仍混淆queue PID和stage PID（阻断恢复语义）

真实复验后`status.json`同时保留旧`queue_status=COMPLETED_REQUESTED_STAGES`，却把`pid`更新成刚结束的直接
preflight进程PID；`started_at`则仍是旧queue时间。三个字段描述的不是同一次执行。未来恢复器可能把一个已结束/
被复用的PID误当成controller进程。queue当前还会在每个stage重新写`started_at`。

修复要求：

- status schema升为v2，分开`queue_pid`与`stage_pid`，增加`invocation_mode=queue|direct`；移除含义模糊的单一`pid`。
- queue的`started_at`只在队列首次启动时设置一次；每阶段使用独立`stage_started_at/stage_finished_at`。
- stage退出后`stage_pid=null`；直接preflight不得继承一个旧queue的RUNNING/COMPLETED状态并伪装成同一次运行。
- 恢复时同时验证PID存活、命令行包含本experiment id、registry hash一致；仅PID数字相同不算同一任务。
- 新增真实子进程测试：queue完成后再直接preflight、直接preflight完成后再queue、伪造活PID/错误cmdline、
  stage中断四种情况，状态必须可解释且可恢复。

### A5：registry精确测试补齐声明与实际覆盖差异

现有测试声称覆盖“全部28字段”，实际expected表未覆盖五个data-role block的path/environment split/output root/
profile/generated episodes-per-seed等所有值。冻结registry hash目前能挡住真实文件漂移，但测试声明不准确。

修复要求：把完整registry作为规范对象逐层深比较，或对canonical JSON固定预期SHA；再分别做至少3个“合法类型但
错误值”的突变测试（output_root、profile_name、environment_split），证明会被冻结身份拒绝。

### A6：method-lock行为只允许在S1-BUILD完成，不得提前宣称

当前soft drift是合理的开发期过渡；但一旦`method_lock.json`出现，preflight仍会拿最终源码与旧scaffold manifest
比较，理论上会立即失败。该部分原计划就在S1-BUILD实现，所以不要求本轮提前造完整method lock；但S1-0R报告
不得把R0R-4描述成“全链条已完成”。S1-BUILD必须改为校验`method_source_manifest.json`，scaffold只作历史基线。

### S1-0R-A验收命令与停止条件

仍只在本地执行，不启动4090、不进入S1-1：

```bash
cd /home/abc/workspace/nav_data/mamba/camrl/CrowdNav
python3 -m py_compile crowd_nav/bayesian_brne/*.py crowd_nav/tools/s1_strict_gate.py crowd_nav/tools/run_s1_queue.py
python3 -m crowd_nav.bayesian_brne.selftest --group s1
python3 -m crowd_nav.bayesian_brne.selftest
python3 -m crowd_nav.bayesian_brne.test_upstream_equivalence
python3 -m crowd_nav.tools.s1_strict_gate \
  --registry crowd_nav/configs/s1_strict_registry.json \
  --stage preflight --protocol-spec-path /home/abc/temp/guide.md
```

完成后报告必须给出：新测试名称、红案例旧行为、修复后行为、新旧protocol hash/amendment、完整role身份字段、
status v2实例，以及确认`data/fits`仍不存在。完成后停下，等待Codex复验。
