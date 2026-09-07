#!/bin/bash
# ============================================================
# Learning Rate Search Script
# ============================================================
# Purpose: Find optimal learning rate for Mamba policy training
# Method: Test multiple LR values with quick 200-episode runs
# ============================================================

set -e  # Exit on error

# Configuration
BASE_DIR="/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav"
RESULTS_DIR="$BASE_DIR/lr_search_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Learning rates to test (in order of priority)
LR_VALUES=(
    "5e-5"   # 0.00005 - More conservative
    "1e-4"   # 0.0001  - Current baseline
    "2e-4"   # 0.0002  - 2x current
    "3e-4"   # 0.0003  - 3x current
    "5e-4"   # 0.0005  - 5x current
)

# Training parameters for quick validation
TRAIN_EPISODES=200
EVAL_EVERY=25
GPU_FLAG="--gpu"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo "Learning Rate Search Experiment"
echo "========================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Testing LR values: ${LR_VALUES[@]}"
echo "Episodes per LR: $TRAIN_EPISODES"
echo "Results directory: $RESULTS_DIR"
echo "========================================================================"
echo ""

# Function to run training with specific LR
run_lr_experiment() {
    local lr=$1
    local run_name="lr_${lr}_${TIMESTAMP}"
    local outdir="$RESULTS_DIR/$run_name"

    echo "=========================================="
    echo "Testing LR = $lr"
    echo "=========================================="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output directory: $outdir"
    echo ""

    # Create temporary config with modified learning rate
    local temp_config="$RESULTS_DIR/train_config_${lr}.tmp"
    cp "$BASE_DIR/configs/train.config" "$temp_config"

    # Modify learning rate in config
    sed -i "s/^learning_rate = .*/learning_rate = ${lr}/" "$temp_config"

    # Modify train_episodes
    sed -i "s/^train_episodes = .*/train_episodes = ${TRAIN_EPISODES}/" "$temp_config"

    # Modify eval_every
    sed -i "s/^eval_every = .*/eval_every = ${EVAL_EVERY}/" "$temp_config"

    # Create output directory
    mkdir -p "$outdir"

    # Run training
    cd "$BASE_DIR"
    export PYTHONPATH="/home/abc/workspace/nav_data/mamba/camrl/CrowdNav:$PYTHONPATH"
    python train.py \
        --outdir "$outdir" \
        --config "$temp_config" \
        $GPU_FLAG \
        2>&1 | tee "$outdir/console.log"

    local exit_code=$?

    # Cleanup temp config
    rm -f "$temp_config"

    echo ""
    echo "Finished LR = $lr (exit code: $exit_code)"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    return $exit_code
}

# Main experiment loop
failed_runs=()

for lr in "${LR_VALUES[@]}"; do
    if ! run_lr_experiment "$lr"; then
        echo "⚠️  WARNING: LR=$lr failed, continuing with next..."
        failed_runs+=("$lr")
    fi

    # Brief pause between runs
    sleep 5
done

# Summary
echo ""
echo "========================================================================"
echo "Learning Rate Search Complete"
echo "========================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Results saved to: $RESULTS_DIR"
echo ""

if [ ${#failed_runs[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!"
else
    echo "⚠️  Failed runs: ${failed_runs[@]}"
fi

echo ""
echo "Next steps:"
echo "1. Run analysis script: python analyze_lr_results.py"
echo "2. Check results in: $RESULTS_DIR"
echo "========================================================================"
