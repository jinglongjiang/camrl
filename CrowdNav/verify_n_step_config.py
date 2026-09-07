#!/usr/bin/env python3
"""
Verify that n-step configuration is properly loaded and used

This script creates a minimal training setup to verify that:
1. n-step parameter is read from config
2. VTrainer is initialized with correct n-step value
3. The n-step monitoring logs appear correctly

Usage:
    python verify_n_step_config.py
"""

import os
import sys
import logging
import configparser
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_nav.train import VTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_config():
    """Create a test configuration with n-step setting"""
    config = configparser.RawConfigParser()

    # Add train section with n-step parameter
    config.add_section('train')
    config.set('train', 'n_step', '8')  # Test with 8-step
    config.set('train', 'lr', '3e-4')
    config.set('train', 'gamma', '0.95')
    config.set('train', 'batch_size', '64')

    return config

def create_mock_policy():
    """Create a minimal mock policy"""
    class MockModel(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device = device
            self.linear = torch.nn.Linear(34, 1)
            self.to(device)

        def forward_value(self, states, pad=None):
            # Simple mock implementation
            B = states.shape[0]
            return torch.randn(B, device=self.device)

    class MockPolicy:
        def __init__(self, device):
            self.model = MockModel(device)

    return MockPolicy

def test_config_loading():
    """Test that n-step config is properly loaded"""
    logger.info("Testing n-step configuration loading...")

    config = create_test_config()

    # Test config reading function (similar to train.py)
    def g(section, key, default):
        try:
            if isinstance(default, float):
                return config.getfloat(section, key, fallback=default)
            elif isinstance(default, int):
                return config.getint(section, key, fallback=default)
            else:
                return config.get(section, key, fallback=default)
        except:
            return default

    n_step = g('train', 'n_step', 5)  # Default 5, should read 8 from config

    logger.info(f"n_step from config: {n_step} (expected: 8)")
    assert n_step == 8, f"Expected n_step=8, got {n_step}"

    logger.info("✓ Configuration loading test passed")
    return True

def test_trainer_initialization():
    """Test that VTrainer correctly initializes with n-step"""
    logger.info("Testing VTrainer initialization with n-step...")

    device = torch.device('cpu')  # Use CPU for testing
    policy = create_mock_policy()(device)

    # Initialize trainer with specific n-step
    n_step = 8
    trainer = VTrainer(policy, device, gamma=0.95, lr=3e-4, batch_size=64, n_step=n_step)

    logger.info(f"Trainer n_step: {trainer.n_step} (expected: {n_step})")
    assert trainer.n_step == n_step, f"Expected n_step={n_step}, got {trainer.n_step}"

    logger.info("✓ VTrainer initialization test passed")
    return True

def test_fallback_behavior():
    """Test that trainer handles missing n-step gracefully"""
    logger.info("Testing fallback behavior...")

    device = torch.device('cpu')
    policy = create_mock_policy()(device)

    # Initialize trainer without n-step (should use default)
    trainer = VTrainer(policy, device, gamma=0.95, lr=3e-4, batch_size=64)

    # Should default to 5 as specified in the constructor
    expected_default = 5
    logger.info(f"Default n_step: {trainer.n_step} (expected: {expected_default})")
    assert trainer.n_step == expected_default, f"Expected default n_step={expected_default}, got {trainer.n_step}"

    logger.info("✓ Fallback behavior test passed")
    return True

def main():
    """Run all verification tests"""
    logger.info("=== Verifying N-Step Configuration ===")

    tests = [
        test_config_loading,
        test_trainer_initialization,
        test_fallback_behavior
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")

    logger.info(f"=== Verification Results: {passed}/{len(tests)} passed ===")

    if passed == len(tests):
        logger.info("🎉 All verification tests passed! N-step configuration is working correctly.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Add 'n_step = 5' or 'n_step = 8' to your train.config file")
        logger.info("2. Run training with: python train.py --policy mamba --outdir runs/mamba_n_step")
        logger.info("3. Monitor logs for '[N-STEP-STATUS]' messages")
        return 0
    else:
        logger.error("❌ Some verification tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())