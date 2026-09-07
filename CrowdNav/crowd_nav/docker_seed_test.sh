#!/bin/bash
# Docker内种子测试一键脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐳 Docker 种子测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 进入正确路径
cd /workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav

# 检查文件是否存在
if [ ! -f "simple_seed_test.py" ]; then
    echo "❌ 错误: simple_seed_test.py 不存在"
    echo "当前路径: $(pwd)"
    echo "请确认路径是否正确"
    exit 1
fi

echo "✅ 找到测试脚本"
echo "当前路径: $(pwd)"
echo ""

# 运行测试
echo "开始测试 (16个种子 × 100 episodes)..."
echo ""

python simple_seed_test.py --num_seeds 16 --test_episodes 100

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 测试完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
