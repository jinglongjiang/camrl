#!/bin/bash
# 使用v3.2离线IL数据集训练Mamba策略

set -e

echo "================================================================"
echo "          CrowdNav Mamba训练 - 使用v3.2离线IL数据集"
echo "================================================================"

cd /home/abc/workspace/nav_data/mamba/camrl/CrowdNav

# 检查数据集
if [ ! -f "data/il_dataset_diverse_v3.1.pth" ]; then
    echo "✗ 错误: 未找到v3.2数据集"
    echo "  请先运行: python scripts/generate_il_dataset.py --gpu"
    exit 1
fi

echo "✓ 数据集就绪: $(ls -lh data/il_dataset_diverse_v3.1.pth | awk '{print $5}')"
echo ""
echo "训练配置:"
echo "  - IL数据: 5000条离线轨迹（4种ORCA风格）"
echo "  - BC训练: 自动执行"
echo "  - RL训练: BC完成后自动开始"
echo "  - 图表: 每25集生成"
echo ""

# 启动训练
PYTHONPATH=/home/abc/workspace/nav_data/mamba/camrl/CrowdNav:$PYTHONPATH \
PYTHONDONTWRITEBYTECODE=1 \
python crowd_nav/train.py \
  --outdir crowd_nav/runs/mamba_v3.2 \
  --config crowd_nav/configs/env.config \
  --policy mamba_rl \
  --gpu
