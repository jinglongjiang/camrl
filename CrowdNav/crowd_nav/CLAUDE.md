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
```

### Training
```bash
# Standard training
python train.py --policy sarl --gpu

# Optimized training (recommended - addresses training instability)
./run_optimized_training.sh
# or manually:
python train_optimized.py --env_config configs/env.config --train_config configs/train.config --output_dir data/output --gpu --seed 42

# Resume training from checkpoint
python train.py --policy sarl --model_dir data/output --resume --gpu
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

# Run linting (if available)
pylint crowd_nav/

# Run tests (if available)
pytest
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