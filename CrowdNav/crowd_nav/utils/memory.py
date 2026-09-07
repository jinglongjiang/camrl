# -*- coding: utf-8 -*-
"""
Memory module shim to redirect legacy imports to ppo_buffer.
"""
from .ppo_buffer import ReplayBufferIQL, ReplayBufferSAC

# Aliases for backward compatibility
ReplayMemory = ReplayBufferSAC
ExpertReplayMemory = ReplayBufferSAC
SequenceReplayMemory = ReplayBufferSAC
