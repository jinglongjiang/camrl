# 测试脚本优化说明

## 🚀 优化内容

### 原始问题
- 必须运行300个episodes才能生成1个视频 → **太慢**
- 只能生成1个指定case的视频 → **功能单一**  
- 为了生成视频要完整评估所有case → **流程冗余**

### 优化方案
- ✅ **快速视频生成**：直接运行指定case，无需全量测试
- ✅ **多视频批量生成**：一次生成3个随机视频
- ✅ **灵活模式选择**：video_only / quick_eval / full_eval
- ✅ **智能文件命名**：包含时间戳、case号、结果状态

## 📋 使用方法

### 方式1: 使用优化脚本（推荐）

```bash
# 快速生成3个随机视频 (最推荐，速度最快)
python test_optimized.py \
    --policy mamba \
    --model_dir data/output \
    --weights rl_model_ep1000.pth \
    --device cuda:0 \
    --mode video_only \
    --num_videos 3 \
    --random_cases

# 生成指定case的视频
python test_optimized.py \
    --policy mamba \
    --model_dir data/output \
    --weights rl_model_ep1000.pth \
    --device cuda:0 \
    --mode video_only \
    --specific_cases 66 123 200

# 快速评估 + 视频生成 (推荐)
python test_optimized.py \
    --policy mamba \
    --model_dir data/output \
    --weights rl_model_ep1000.pth \
    --device cuda:0 \
    --mode quick_eval \
    --num_videos 2 \
    --quick_eval_samples 50
```

### 方式2: 运行示例脚本

```bash
# 运行所有示例
./test_examples.sh

# 或者单独运行某个示例
bash test_examples.sh
```

## 🎯 模式说明

### video_only 模式 (最快)
- **用途**: 只生成视频，不做性能评估
- **速度**: ⭐⭐⭐⭐⭐ 最快 (几秒到几分钟)
- **推荐场景**: 想快速查看训练效果

### quick_eval 模式 (推荐)
- **用途**: 运行少量episodes获取性能指标 + 生成视频
- **速度**: ⭐⭐⭐⭐ 较快 (几分钟)
- **推荐场景**: 既想了解性能又想看视频

### full_eval 模式 (最全面)
- **用途**: 完整评估 + 可选视频生成
- **速度**: ⭐⭐ 慢 (几十分钟)
- **推荐场景**: 需要准确的性能评估

## 📊 输出文件命名

生成的视频文件命名格式:
```
demo_20241221_143052_case066_reach_goal.mp4
demo_20241221_143052_case123_collision.mp4
demo_20241221_143052_case200_timeout.mp4
```

格式说明:
- `demo`: 基础名称
- `20241221_143052`: 时间戳 (YYYYMMDD_HHMMSS)
- `case066`: case编号
- `reach_goal/collision/timeout`: 结果状态

## 🔧 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `video_only` | 运行模式: video_only/quick_eval/full_eval |
| `--num_videos` | `3` | 生成视频数量 |
| `--random_cases` | `True` | 是否使用随机case |
| `--specific_cases` | `None` | 指定具体case (如: 66 123 200) |
| `--quick_eval_samples` | `50` | 快速评估的episode数量 |
| `--max_case_range` | `1000` | 随机选择的case范围 |

## ⚡ 性能对比

| 方法 | 时间消耗 | 功能 |
|------|----------|------|
| 原始方法 | 15-30分钟 | 300 episodes + 1个视频 |
| video_only | 30秒-2分钟 | 3个随机视频 |
| quick_eval | 3-5分钟 | 50 episodes评估 + 2个视频 |

**提升效果**: 速度提升 **10-20倍**，功能增强 **3倍视频**

## 🎥 训练监控建议

现在可以查看训练日志来选择合适的权重文件：

```bash
# 查看最新的训练进度
tail -20 data/output/output.log

# 查看成功率趋势 
grep "TRAIN ep=" data/output/output.log | tail -10

# 选择高成功率的checkpoint
ls data/output/rl_model_ep*.pth | sort -V | tail -5
```

基于当前训练情况 (EP660左右，成功率75-100%)，建议等训练完成后使用:
- `rl_model_ep1000.pth` 或更高episode的权重
- 先用 `quick_eval` 模式快速评估性能
- 再用 `video_only` 模式生成多个视频查看效果