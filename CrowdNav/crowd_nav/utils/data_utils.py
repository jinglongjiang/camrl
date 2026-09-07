# -*- coding: utf-8 -*-
"""
P2/P3 DataConverter - 

：
- ""
-  contracts ：ensure_tensor/DEFAULT_DTYPE/validate_state_shape
-  to_tensor_unified()/to_numpy_unified() （ActionXY → [vx,vy]）
-  contracts 
"""

import numpy as np
import torch
from typing import Union, List, Tuple, Optional

from crowd_nav.contracts import (
    ensure_tensor, DEFAULT_DTYPE, validate_state_shape,
    to_btnd, tokens_to_34d, joint34_to_tokens
)


class DataConverter:

    @staticmethod
    def to_tensor_unified(data, dtype=None, device=None):
        """
        

        Args:
            data: （numpy, list, tensor）
            dtype: 
            device: 

        Returns:
            torch.Tensor: 
        """
        if dtype is None:
            dtype = DEFAULT_DTYPE

        return ensure_tensor(data, dtype=dtype, device=device)

    @staticmethod
    def to_numpy_unified(data):
        """
        numpy

        Args:
            data: （tensor, list）

        Returns:
            np.ndarray: numpy
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data, dtype=np.float32)

    @staticmethod
    def normalize_action(action) -> Tuple[float, float]:
        """
        ：ActionXY → [vx, vy]

        Args:
            action: ActionXY(vx, vy)

        Returns:
            Tuple[float, float]: (vx, vy)
        """
        if hasattr(action, 'vx') and hasattr(action, 'vy'):
            return float(action.vx), float(action.vy)
        elif isinstance(action, (tuple, list)) and len(action) == 2:
            return float(action[0]), float(action[1])
        elif isinstance(action, (torch.Tensor, np.ndarray)) and len(action) == 2:
            return float(action[0]), float(action[1])
        else:
            raise ValueError(f"Unsupported action format: {type(action)}")

    @staticmethod
    def validate_joint_state(state, expected_shape=(34,)):
        """
         - contracts

        Args:
            state: 
            expected_shape: 

        Returns:
            bool: 
        """
        try:
            validate_state_shape(state, expected_shape)
            return True
        except (ValueError, AssertionError):
            return False

    @staticmethod
    def joint_state_to_tokens(state_34d):
        """
        34D → 6×13 tokens
        contracts.joint34_to_tokens
        """
        return joint34_to_tokens(state_34d)

    @staticmethod
    def tokens_to_joint_state(tokens):
        """
        6×13 tokens → 34D
        contracts.tokens_to_34d
        """
        return tokens_to_34d(tokens)

    @staticmethod
    def prepare_batch_tokens(states_list, sequence_length=8):
        """
        tokens

        Args:
            states_list:  [state1, state2, ...]
            sequence_length: 

        Returns:
            torch.Tensor: [B, T, 6, 13] tokens
        """
        if not states_list:
            raise ValueError("Empty states list")

        states_34d = []
        for state in states_list:
            if hasattr(state, 'to_array'):
                state_34d = state.to_array()
            else:
                state_34d = np.array(state, dtype=np.float32)

            if len(state_34d) != 34:
                if len(state_34d) < 34:
                    state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant')
                else:
                    state_34d = state_34d[:34]

            states_34d.append(state_34d)

        tokens_list = []
        for state_34d in states_34d:
            tokens = DataConverter.joint_state_to_tokens(state_34d)
            tokens_list.append(tokens)

        while len(tokens_list) < sequence_length:
            tokens_list.append(tokens_list[-1])

        if len(tokens_list) > sequence_length:
            tokens_list = tokens_list[-sequence_length:]

        batch_tokens = np.stack(tokens_list, axis=0)  # [T, 6, 13]
        batch_tokens = np.expand_dims(batch_tokens, axis=0)  # [1, T, 6, 13]

        return DataConverter.to_tensor_unified(batch_tokens)

    @staticmethod
    def batch_normalize_actions(actions):
        """
        

        Args:
            actions: 

        Returns:
            List[Tuple[float, float]]: 
        """
        normalized_actions = []
        for action in actions:
            vx, vy = DataConverter.normalize_action(action)
            normalized_actions.append((vx, vy))
        return normalized_actions

    @staticmethod
    def prepare_training_batch(states, actions, rewards, next_states, dones):
        """
        

        Args:
            states: 
            actions: 
            rewards: 
            next_states: 
            dones: 

        Returns:
            dict: 
        """
        batch_size = len(states)
        assert len(actions) == batch_size
        assert len(rewards) == batch_size
        assert len(next_states) == batch_size
        assert len(dones) == batch_size

        states_tensor = DataConverter.to_tensor_unified(states)
        next_states_tensor = DataConverter.to_tensor_unified(next_states)

        normalized_actions = DataConverter.batch_normalize_actions(actions)
        actions_tensor = DataConverter.to_tensor_unified(normalized_actions)

        rewards_tensor = DataConverter.to_tensor_unified(rewards)
        dones_tensor = DataConverter.to_tensor_unified(dones, dtype=torch.bool)

        return {
            'states': states_tensor,
            'actions': actions_tensor,
            'rewards': rewards_tensor,
            'next_states': next_states_tensor,
            'dones': dones_tensor
        }


def to_tensor(data, dtype=None, device=None):
    return DataConverter.to_tensor_unified(data, dtype, device)


def to_numpy(data):
    return DataConverter.to_numpy_unified(data)


def normalize_action(action):
    return DataConverter.normalize_action(action)


def validate_state(state, expected_shape=(34,)):
    return DataConverter.validate_joint_state(state, expected_shape)


__all__ = [
    'DataConverter',
    'to_tensor', 'to_numpy', 'normalize_action', 'validate_state'
]