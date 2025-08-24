#!/bin/bash
# 优化版测试脚本使用示例

echo "=== CrowdNav Test Optimization Examples ==="

# 基础路径配置
MODEL_DIR="data/output"
WEIGHTS="rl_model_ep1000.pth"  # 根据你的实际权重文件名调整
POLICY="mamba"
DEVICE="cuda:0"

echo "Using model: $MODEL_DIR/$WEIGHTS"
echo "Policy: $POLICY, Device: $DEVICE"
echo

# 示例1: 快速生成3个随机视频 (推荐用法)
echo "=== 示例1: 快速生成3个随机视频 (最快) ==="
echo "命令: python test_optimized.py --policy $POLICY --model_dir $MODEL_DIR --weights $WEIGHTS --device $DEVICE --mode video_only --num_videos 3 --random_cases"
echo "开始执行..."
python test_optimized.py \
    --policy $POLICY \
    --model_dir $MODEL_DIR \
    --weights $WEIGHTS \
    --device $DEVICE \
    --mode video_only \
    --num_videos 3 \
    --random_cases \
    --output_dir $MODEL_DIR

echo
echo "=== 示例2: 生成指定case的视频 ==="
echo "命令: python test_optimized.py --policy $POLICY --model_dir $MODEL_DIR --weights $WEIGHTS --device $DEVICE --mode video_only --specific_cases 66 123 200"
echo "跳过执行 (需要时取消注释)"
# python test_optimized.py \
#     --policy $POLICY \
#     --model_dir $MODEL_DIR \
#     --weights $WEIGHTS \
#     --device $DEVICE \
#     --mode video_only \
#     --specific_cases 66 123 200 \
#     --output_dir $MODEL_DIR

echo
echo "=== 示例3: 快速评估 + 生成视频 ==="
echo "python test_optimized.py --policy $POLICY --model_dir $MODEL_DIR --weights $WEIGHTS --device $DEVICE --mode quick_eval --num_videos 2 --quick_eval_samples 30"
python test_optimized.py \
    --policy $POLICY \
    --model_dir $MODEL_DIR \
    --weights $WEIGHTS \
    --device $DEVICE \
    --mode quick_eval \
    --num_videos 2 \
    --quick_eval_samples 30

echo
echo "=== 示例4: 完整评估 + 视频 (最慢，但最全面) ==="
echo "python test_optimized.py --policy $POLICY --model_dir $MODEL_DIR --weights $WEIGHTS --device $DEVICE --mode full_eval --episodes 100 --num_videos 2"
# python test_optimized.py \
#     --policy $POLICY \
#     --model_dir $MODEL_DIR \
#     --weights $WEIGHTS \
#     --device $DEVICE \
#     --mode full_eval \
#     --episodes 100 \
#     --num_videos 2

echo "示例4被注释，如需完整评估请取消注释"

echo
echo "=== 所有示例完成 ==="
echo "生成的视频文件将保存在 $MODEL_DIR 目录下"
echo "文件名格式: demo_YYYYMMDD_HHMMSS_caseXXX_result.mp4"