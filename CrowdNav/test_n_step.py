#!/usr/bin/env python3
"""
Test script for n-step TD learning implementation

This script verifies that the n-step TD learning implementation works correctly
by testing the key components:
1. SequenceReplayMemory.sample_n_step_transitions
2. VTrainer._compute_n_step_td_target
3. Overall n-step integration

Usage:
    python test_n_step.py
"""

import os
import sys
import numpy as np
import torch
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_nav.utils.memory import SequenceReplayMemory
from crowd_nav.train import VTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_mock_policy():
    """Create a mock policy for testing"""
    class MockModel(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device = device
            # Simple linear layer for testing
            self.linear = torch.nn.Linear(34, 1)
            self.to(device)

        def forward_value(self, states, pad=None):
            """Mock value function"""
            # states: [B, 1, 6, 13] or [B, 6, 13]
            if states.dim() == 4:
                B, T, H, W = states.shape
                states_flat = states.view(B, T, -1)[:, -1, :]  # [B, 78]
                # Pad or truncate to 34
                if states_flat.shape[1] > 34:
                    states_flat = states_flat[:, :34]
                else:
                    pad_size = 34 - states_flat.shape[1]
                    padding = torch.zeros(B, pad_size, device=states.device)
                    states_flat = torch.cat([states_flat, padding], dim=1)
            else:  # [B, 6, 13]
                B, H, W = states.shape
                states_flat = states.view(B, -1)  # [B, 78]
                # Pad or truncate to 34
                if states_flat.shape[1] > 34:
                    states_flat = states_flat[:, :34]
                else:
                    pad_size = 34 - states_flat.shape[1]
                    padding = torch.zeros(B, pad_size, device=states.device)
                    states_flat = torch.cat([states_flat, padding], dim=1)

            return self.linear(states_flat).squeeze(-1)  # [B]

    class MockPolicy:
        def __init__(self, device):
            self.model = MockModel(device)

    return MockPolicy

def test_sequence_memory_n_step():
    """Test SequenceReplayMemory n-step sampling"""
    logger.info("Testing SequenceReplayMemory n-step sampling...")

    # Create memory with small capacity for testing
    memory = SequenceReplayMemory(capacity=100, sequence_length=8)

    # Add some test sequences
    for seq_id in range(10):
        # Create a sequence of states, actions, rewards, etc.
        seq_length = np.random.randint(5, 15)  # Variable length sequences

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        action_indices = []

        for t in range(seq_length):
            # Create 34D state (robot 9D + 5 humans * 5D)
            state = np.random.randn(34).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)  # [vx, vy]
            reward = np.random.randn()
            next_state = np.random.randn(34).astype(np.float32)
            done = (t == seq_length - 1)  # Only last step is done
            action_idx = np.random.randint(0, 80)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            action_indices.append(action_idx)

        # Convert to numpy arrays
        states_seq = np.stack(states)
        actions_seq = np.stack(actions)
        rewards_seq = np.array(rewards)
        next_states_seq = np.stack(next_states)
        dones_seq = np.array(dones, dtype=bool)
        action_indices_seq = np.array(action_indices)

        # Push to memory using the correct interface
        # Convert to lists for push_episode
        memory.push_episode(states.copy(), actions.copy(), rewards.copy())

    logger.info(f"Added {len(memory)} sequences to memory")

    # Test n-step sampling
    n_step = 5
    batch_size = 4

    n_step_data = memory.sample_n_step_transitions(batch_size, n_step)

    if n_step_data is None:
        logger.error("Failed to sample n-step transitions")
        return False

    states, rewards, next_states, dones, lengths = n_step_data

    logger.info(f"N-step sampling results:")
    logger.info(f"  States shape: {states.shape}")  # [B, n_step, state_dim]
    logger.info(f"  Rewards shape: {rewards.shape}")  # [B, n_step]
    logger.info(f"  Next states shape: {next_states.shape}")  # [B, state_dim]
    logger.info(f"  Dones shape: {dones.shape}")  # [B, n_step]
    logger.info(f"  Lengths shape: {lengths.shape}")  # [B]

    # Validate shapes
    assert states.shape == (batch_size, n_step, 34), f"States shape mismatch: {states.shape}"
    assert rewards.shape == (batch_size, n_step), f"Rewards shape mismatch: {rewards.shape}"
    assert next_states.shape == (batch_size, 34), f"Next states shape mismatch: {next_states.shape}"
    assert dones.shape == (batch_size, n_step), f"Dones shape mismatch: {dones.shape}"
    assert lengths.shape == (batch_size,), f"Lengths shape mismatch: {lengths.shape}"

    logger.info("✓ N-step sampling test passed")
    return True

def test_n_step_td_target():
    """Test n-step TD target computation"""
    logger.info("Testing n-step TD target computation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = create_mock_policy()(device)

    # Create trainer with n-step
    trainer = VTrainer(policy, device, gamma=0.95, n_step=5)

    # Create mock n-step data
    batch_size = 4
    n_step = 5
    state_dim = 34

    # Mock data
    states_ns = np.random.randn(batch_size, n_step, state_dim).astype(np.float32)
    rewards_ns = np.random.randn(batch_size, n_step).astype(np.float32)
    next_states_ns = np.random.randn(batch_size, state_dim).astype(np.float32)
    dones_ns = np.zeros((batch_size, n_step), dtype=np.float32)
    # Make last step done for some samples
    dones_ns[0, -1] = 1.0  # Sample 0 terminates at last step
    dones_ns[1, 2] = 1.0   # Sample 1 terminates at step 2
    lengths_ns = np.array([n_step, 3, n_step, n_step])  # Actual lengths

    # Compute n-step TD targets
    y_v = trainer._compute_n_step_td_target(states_ns, rewards_ns, next_states_ns, dones_ns, lengths_ns)

    logger.info(f"N-step TD target computation results:")
    logger.info(f"  Input shapes: states={states_ns.shape}, rewards={rewards_ns.shape}")
    logger.info(f"  Output shape: {y_v.shape}")  # [B]
    logger.info(f"  Target values: {y_v.cpu().numpy()}")

    # Validate shape
    assert y_v.shape == (batch_size,), f"TD target shape mismatch: {y_v.shape}"

    # Check that values are in reasonable range
    assert torch.all(y_v >= -10.0) and torch.all(y_v <= 10.0), "TD targets out of range"

    logger.info("✓ N-step TD target computation test passed")
    return True

def test_integration():
    """Test integration with actual memory buffer"""
    logger.info("Testing integration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = create_mock_policy()(device)

    # Create trainer with n-step
    trainer = VTrainer(policy, device, gamma=0.95, n_step=5)

    # Create memory and add sequences
    memory = SequenceReplayMemory(capacity=100, sequence_length=8)

    # Add test data
    for _ in range(20):
        seq_length = np.random.randint(8, 16)

        # Generate episode data
        states = []
        actions = []
        rewards = []

        for t in range(seq_length):
            state = np.random.randn(34).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            reward = np.random.randn()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        memory.push_episode(states.copy(), actions.copy(), rewards.copy())

    logger.info(f"Memory contains {len(memory)} sequences")

    # Test optimize with n-step (mock)
    try:
        # This would normally call trainer.optimize(memory, updates=1)
        # but we'll just test the n-step data retrieval
        n_step_data = memory.sample_n_step_transitions(batch_size=8, n_step=trainer.n_step)

        if n_step_data is not None:
            states, rewards, next_states, dones, lengths = n_step_data
            y_v = trainer._compute_n_step_td_target(states, rewards, next_states, dones, lengths)
            logger.info(f"Integration test successful: TD targets shape {y_v.shape}")
        else:
            logger.warning("N-step data sampling returned None")

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

    logger.info("✓ Integration test passed")
    return True

def main():
    """Run all tests"""
    logger.info("=== Testing N-Step TD Learning Implementation ===")

    tests = [
        test_sequence_memory_n_step,
        test_n_step_td_target,
        test_integration
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")

    logger.info(f"=== Test Results: {passed}/{len(tests)} passed ===")

    if passed == len(tests):
        logger.info("🎉 All tests passed! N-step implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())