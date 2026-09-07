#!/bin/bash
# 训练收敛监控脚本

echo "=== CrowdNav训练收敛监控 ==="

# 1. 最新成功率
echo "📊 最新成功率趋势:"
tail -n 200 runs/mamba_vl/train.log | grep "EVAL\[.*\] ep=" | tail -5 | \
    grep -oE "succ=[0-9]+\.[0-9]+" | sed 's/succ=//'

# 2. V-loss趋势
echo "📈 V-loss最新趋势:"
tail -n 100 runs/mamba_vl/train.log | grep "v_loss=" | tail -5 | \
    grep -oE "v_loss=[0-9]+\.[0-9]+" | sed 's/v_loss=//'

# 3. Q-spread健康度
echo "🎯 Q-spread健康度:"
tail -n 100 runs/mamba_vl/train.log | grep "q_spread=" | tail -5 | \
    grep -oE "q_spread=[0-9]+\.[0-9]+" | sed 's/q_spread=//'

# 4. 梯度统计
echo "⚡ 梯度健康状态:"
tail -n 50 runs/mamba_vl/train.log | grep "GRAD-STATS" | tail -3

# 5. 当前episode和buffer状态
echo "📋 训练进度:"
tail -n 10 runs/mamba_vl/train.log | grep "A-MODE ep=" | tail -1 | \
    grep -oE "ep=[0-9]+" | sed 's/ep=//'
echo "Buffer大小:"
tail -n 10 runs/mamba_vl/train.log | grep "\[BUF\]" | tail -1 | \
    grep -oE "\[BUF\] [0-9]+" | sed 's/\[BUF\] //'

# 6. 收敛判断
echo ""
echo "🎯 收敛判断标准:"
echo "✅ 成功率 > 80% 且稳定"
echo "✅ V-loss < 0.1 且稳定"
echo "✅ Q-spread: 0.1-0.15"
echo "✅ 梯度norm < 1.0"
echo "✅ 无梯度爆炸"