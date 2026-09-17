# CC 协方差跨人数报告：原始数据复核

## 结论

CC 的位置协方差效应数字复现。简单替代同样出现跨人数效应放大，
本批数据未证明贝叶斯独立优势，也不是五人训练策略的零样本泛化实验。

## 数据与口径

直接逐行读取 `/home/abc/workspace/bayes_occ_mpc_hermite/results/occlusion_closeout/episodes.jsonl`，
筛选 `sensor=range_and_occlusion`，共 3,600 行，六方法各 600 回合。
使用 `method` 而非底层统一的 `arm=bayes` 区分实际干预。
成功采用 `success_without_overlap`，碰撞采用 `collision_union`。
以 (method, scene, case_id) 配对并检查重复、缺失。

这是独立确认批 case 3001017--3001116，不是旧脚本文档里的 5000--5099。
六场景共享 case seed，因此区间重采样整个六场景 case block，10,000 次。
以下区间是探索性逐比较 95% 区间，未作多重比较校正。

## 原始成功计数（每格 100）

| 方法 | 5圆 | 10方 | 10圆 | 20密方 | 12圆 | 20大方 |
| --- | --- | --- | --- | --- | --- | --- |
| posterior_mean | 99 | 93 | 95 | 76 | 99 | 92 |
| covariance | 100 | 96 | 100 | 94 | 100 | 99 |
| bayes | 100 | 97 | 100 | 90 | 100 | 99 |
| age_margin | 100 | 97 | 100 | 95 | 100 | 94 |
| ewma | 100 | 95 | 100 | 90 | 100 | 97 |
| conformal | 100 | 95 | 99 | 87 | 100 | 97 |

## 加入强对照后

相对 posterior_mean，单位为成功率百分点：

| 方法 | 5人 | 10人 | 12人 | 20人 | 20减5效应差及区间 |
| --- | --- | --- | --- | --- | --- |
| covariance | +1 | +4 | +1 | +12.5 | +11.5 [6,17] |
| bayes | +1 | +4.5 | +1 | +10.5 | +9.5 [4,15] |
| age_margin | +1 | +4.5 | +1 | +10.5 | +9.5 [4,15] |
| ewma | +1 | +3.5 | +1 | +9.5 | +8.5 [2.5,14] |
| conformal | +1 | +3 | +1 | +8 | +7 [1,13] |

Bayes 与 age_margin 分人数聚合增量完全相同，但不代表逐布局动作或结果相同。
Bayes-age_margin 的20减5效应差为 0，区间 [-3.5,+3.5] pp。
Bayes-EWMA 为 +1，区间 [-2,+4] pp。
covariance-age_margin 为 +2，区间 [-1,+5] pp。
因此“增量随人数增大”不是贝叶斯特有证据。

## 20人配对计数

| 对比 | 前者胜/后者胜 | 精确 McNemar p |
| --- | --- | --- |
| covariance - posterior_mean | 29/4 | 0.00001093 |
| bayes - covariance | 0/4 | 0.125 |
| bayes - age_margin | 7/7 | 1.000 |
| bayes - EWMA | 5/3 | 0.7266 |
| covariance - age_margin | 7/3 | 0.3438 |
| covariance - EWMA | 8/2 | 0.1094 |

这些 McNemar 数字复现 CC 的计算，但它将不同场景的 scene-case 对当独立样本；
跨场景重复 seed 的依赖由上面的 block bootstrap 保留。
单个人数组内显著，不是人数交互效应显著的替代证明。
存在概率只有四个不一致对，百分位区间退化且与精确检验不同，不据此宣称显著负贡献。

## 代码核验

1. `archive/legacy/evaluate_matched_safety.py:109` 的 posterior_mean 清空位置协方差和存在概率。
   `experiments/occlusion_confirmation.py:70` 将 covariance 的 existence_override 设为1。
   三个核心臂的其余 MPC 参数相同；该组件消融成立。
2. 强替代沿用各自在开发阶段冻结的工作点，不是全部相同工作点。
   protocol 的 chance_limit: posterior_mean/covariance/bayes 为0.75，age_margin 为0.25。
   它们是校准后的方法对照，不是同工作点的单变量实验。
3. 本确认批没有 static_cov。不能拿别批或全观测 static_cov 数字拼入本表。
4. `bayes_continuous/environment.py:104-120` 对所有真人读取真实位置速度，
   确认没有在该观测入口实施遮挡；token 尾部是 `belief[i,:4]`。
5. `crowd_nav/gdbn.py:799` 明确为模式概率、熵、KLDA；当前 K=3 时前四列为三模式概率和熵，
   不含位置协方差，也不是 RFS 存在概率。CC 将旧模态实验归于存在概率一侧的说法错误。

## 能写与不能写

能写：在本遮挡 MPC 协议下，引入位置不确定性较均值方案改善导航，20人场景改善更大。
时长裕量、EWMA 等替代亦出现相似的放大效应。

不能写：已经证明贝叶斯帮助五人训练到二十人泛化。本实验没有五人训练 Actor，
且人数与形状、尺度、遮挡共同变化；5到10到12到20的效应也不单调。
不能写：模态后验无效是由存在概率消融证明的。
不能写：贝叶斯与强替代等价；这里是未检测到差异，而非通过预注册等价检验。

## 可复算入口

现有 `crowd_nav/bayes_continuous/stage_audit.py::audit_occlusion_density(source,out)`。
`analysis.json` 保存配置、全部计数、区间和原始文件 SHA256。
此次只读审计，不训练、不改模型、不生成 PDF、不修改稿件结论。
