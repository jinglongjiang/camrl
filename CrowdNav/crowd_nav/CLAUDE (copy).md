# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is CrowdNav, a deep reinforcement learning framework for crowd-aware robot navigation. The codebase implements various navigation policies including CADRL, LSTM-RL, SARL, and Mamba-based approaches for robots operating in crowded environments with sophisticated SAC-based training.

## Key Architecture

### Core Training Framework
The training system is built around a highly optimized SAC (Soft Actor-Critic) implementation with:
- **Multi-policy support**: SARL (attention-based), CADRL, LSTM-RL, Mamba (state-space models), ORCA baseline
- **Dual-phase training**: Imitation learning (ORCA demonstrations) → Reinforcement learning (SAC)
- **Advanced features**: AMP (Automatic Mixed Precision), torch.compile optimization, parallel sampling

### Policy Architecture Patterns
All RL policies follow unified interface:
- **Encoder**: Processes robot+human observations → feature embeddings
- **Value Networks**: Twin Q-networks (Q1, Q2) for stability
- **Actor Network**: Stochastic policy with continuous action output
- **Attention Mechanism**: SARL uses multi-head attention for human-robot interaction modeling
- **Mamba Integration**: Uses mamba_ssm for sequence modeling with state-space models

### Training Components
- **Trainer** (`utils/trainer.py`): SAC implementation with BC regularization, gradient clipping, target networks
- **Explorer** (`utils/explorer.py`): Environment interaction with vectorized parallel sampling
- **Memory** (`utils/memory.py`): Replay buffer with expert demonstrations
- **ParallelSampler** (`utils/parallel_sampler.py`): Multi-worker episode collection

## Common Commands

### Installation and Setup
```bash
# Install from root CrowdNav directory
pip install -e .

# Install with test dependencies  
pip install -e .[test]

# Key dependencies:
# 1. First install Python-RVO2: https://github.com/sybrenstuvel/Python-RVO2
# 2. For mamba policies: pip install mamba-ssm (requires CUDA)
# 3. Core: torch, gym, matplotlib, numpy, scipy
# 4. Optional: gitpython (for version tracking)
```

### Training
```bash
# Standard training
python train.py --policy sarl --gpu

# Optimized training (recommended - addresses training instability)
./run_optimized_training.sh
# Note: The optimized configs referenced in the script need to be created first
# or use standard configs with modified parameters:
python train_optimized.py --env_config configs/env.config --train_config configs/train.config --output_dir data/output --gpu --seed 42

# Resume training from checkpoint
python train.py --policy sarl --model_dir data/output --resume --gpu

# Train specific policies
python train.py --policy cadrl --gpu  # Basic MLP policy
python train.py --policy lstm_rl --gpu  # LSTM-based policy
python train.py --policy mamba_rl --gpu  # Mamba SSM policy (requires CUDA)
```

### Testing and Evaluation
```bash
# Test with deterministic policy
python test.py --policy sarl --model_dir data/output --phase test

# Visualize single episode
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0

# ORCA baseline testing  
python test.py --policy orca --phase test

# Specify number of test cases
python test.py --policy sarl --model_dir data/output --phase test --test_case 500
```

### Development and Analysis
```bash
# Plot training curves from logs
python utils/plot.py data/output/output.log

# Run linting (after installing pylint)
pylint crowd_nav/
pylint crowd_nav/policy/sarl.py  # Check specific module

# Run tests (after installing pytest)
pytest
pytest crowd_nav/test.py::TestPolicy  # Run specific test class
pytest -v  # Verbose output
pytest -k "test_action"  # Run tests matching pattern

# Type checking (if mypy installed)
mypy crowd_nav/
```

## Configuration System

Three-tier configuration:
- **env.config**: Environment (crowd size, robot/human properties, reward function weights)
- **policy.config**: Neural network architecture (hidden dims, attention heads, etc.)
- **train.config**: Training hyperparameters (learning rates, batch sizes, SAC parameters)

Key config sections:
- `[reward]`: Success/collision rewards, progress shaping weights, social compliance penalties
- `[trainer]`: SAC parameters (gamma, tau, learning rates, target_entropy)
- `[vectorize]`: Parallel sampling configuration (workers, episodes_per_worker)

## ROS/Gazebo Integration

### Real Robot Deployment
```bash
# Run SARL policy on real robot
python sarl_real_robot_node.py

# Run with leg detection
python leg_sarl_node.py
./run_leg_sarl_sim.sh  # Launch simulation with leg detection

# ROS navigation node
python crowdnav_ros.py

# RViz visualization
python crowdnav_rviz_demo.py
python crowdnav_rviz_visual.py
```

### Gazebo Simulation
```bash
# Fine-tune policy in Gazebo
python gazebo_finetune.py

# 3D visualization demos
python crowdnav_3d_demo.py
python crowdnav_3d_demo_with_pth.py --model_dir data/output
```

## Training Architecture Details

### Two-Phase Training Process
1. **Imitation Learning**: Collect ORCA expert demonstrations (typically 100-300 episodes)
2. **RL Training**: SAC training with expert data in replay buffer

### SAC Implementation Features
- **Twin Q-networks**: Reduces overestimation bias
- **Automatic entropy tuning**: Adaptive target_entropy based on action dimensionality
- **BC Regularization**: Advantage-weighted behavioral cloning (AWBC) for stability
- **AMP Support**: Mixed precision training for speed
- **Gradient Clipping**: Prevents gradient explosions

### Vectorized Training
- **Parallel sampling**: Multiple workers collect episodes simultaneously
- **Device optimization**: Workers run on GPU for mamba_ssm compatibility
- **Broadcasting**: Model parameters synced across workers

## File Structure and Key Modules

### Policy Implementations
- `policy/sarl.py`: Attention-based policy with human interaction modeling
- `policy/mamba_rl.py`: Mamba state-space model integration 
- `policy/lstm_rl.py`: LSTM-based sequence modeling
- `policy/cadrl.py`: Basic multi-layer perceptron policy
- `policy/orca_wrapper.py`: Classical collision avoidance baseline

### Training Infrastructure  
- `train.py`: Main training script with full SAC implementation
- `train_optimized.py`: Optimized version addressing known stability issues
- `test.py`: Evaluation with config-matched deterministic testing
- `utils/trainer.py`: Core SAC trainer with advanced features
- `utils/parallel_sampler.py`: Multi-worker episode collection

### State and Observation Processing
Robot state: [px, py, vx, vy, radius, gx, gy, v_pref, theta] (9D)
Human states: [px, py, vx, vy, radius] per human (5D each, up to 5 humans)
Total observation space: 9 + 5*5 = 34D (padded/masked for variable human count)

### ROS Integration
The codebase includes ROS integration for real robot deployment:
- `crowdnav_ros.py`: ROS node for crowd navigation
- `sarl_real_robot_node.py`: SARL policy on real robots
- `leg_sarl_node.py`: Integration with leg detection
- `crowdnav_rviz_demo.py`: RViz visualization
- `gazebo_finetune.py`: Gazebo simulation fine-tuning

## Performance Optimization

The codebase includes extensive optimizations documented in `OPTIMIZATION_GUIDE.md`:

### Training Stability Issues Solved
- **High variance success rates**: Larger batch sizes (1024-2048), lower learning rates (3e-4)
- **Slow convergence**: Optimized reward shaping, adaptive exploration schedules
- **Training speed**: GPU acceleration, parallel sampling, AMP, torch.compile

### Key Optimization Techniques
- **TF32 acceleration**: Enabled for Ampere+ GPUs
- **Thread limiting**: Single-threaded CPU ops to avoid overhead
- **Pinned memory**: Faster CPU-GPU transfers
- **Observation normalization**: Running mean/std for stable training

## Known Issues and Solutions

### Common Training Problems
1. **Mamba SSM GPU Requirement**: The mamba_ssm library requires CUDA. Workers must run on GPU.
2. **High Memory Usage**: Use smaller batch sizes or reduce number of parallel workers
3. **Slow Convergence**: Use optimized configurations with adaptive exploration
4. **Training Instability**: Check OPTIMIZATION_GUIDE.md for detailed solutions

### Debugging Tips
- Monitor `output.log` for real-time training metrics
- Check for NaN values in gradients when training fails
- Verify environment rendering with `--visualize` flag
- Use smaller human counts (2-3) for initial debugging