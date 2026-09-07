# -*- coding: utf-8 -*-
"""
Explorer — compatible with simplified contracts.py (no ensure_tensor/DEFAULT_DTYPE)
"""
from __future__ import annotations

import logging
import copy
import random
from typing import Optional, Tuple, List, Any, Dict
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from crowd_nav.contracts import (
    to_btnd, tokens_to_34d, _batch_joint34_to_tokens_vectorized
)
from crowd_sim.envs.utils.action import ActionXY

log = logging.getLogger(__name__)

def _vec_to_speed_heading(vec: np.ndarray) -> Tuple[float, float]:
    """(vx,vy) → (speed, heading[rad])"""
    vx = float(vec[0])
    vy = float(vec[1]) if len(vec) > 1 else 0.0
    spd = float(np.hypot(vx, vy))
    ang = float(np.arctan2(vy, vx))
    return spd, ang


def _grid_spec(n_headings: int, n_speeds: int, v_min: float, v_max: float) -> Dict[str, float]:
    return dict(n_headings=int(n_headings), n_speeds=int(n_speeds), v_min=float(v_min), v_max=float(v_max))


def _as_np32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _extract_event_token(info: Any) -> str:
    """鲁棒提取 episode 结束事件标记，兼容 dict/对象/枚举/字符串。
    返回值统一为小写、下划线连接的 token（如 reach_goal/collision/timeout）。
    """
    try:
        if info is None:
            return ""
        # dict 风格
        if isinstance(info, dict):
            val = info.get("event") or info.get("Event") or info.get("status") or info.get("done_event")
            if val is not None:
                return str(val).replace(" ", "_").lower().strip()
        # 对象/枚举风格
        for attr in ("event", "name", "value"):
            if hasattr(info, attr):
                try:
                    return str(getattr(info, attr)).replace(" ", "_").lower().strip()
                except Exception:
                    pass
        # 兜底：类名或 str()
        s = str(info)
        # 处理类似 Enum 形式 'Event.ReachGoal'
        if "." in s:
            s = s.split(".")[-1]
        return s.replace(" ", "_").lower().strip()
    except Exception:
        return ""


# ------------------------------------------------------------------
# Explorer
# ------------------------------------------------------------------
class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma: Optional[float] = None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self._last_safe_frac = 1.0
        self.target_model = None
        self._last_trajectories: Optional[List[Tuple[Tuple[list, list, list], dict]]] = None
        self._obs_mode_logged = {}
        # 轨迹记录器（只记录RL阶段）
        self.trajectory_logger = None
        self.enable_trajectory_logging = False
        self._logged_state_dims = set()

    @property
    def last_trajectories(self):
        return self._last_trajectories

    def _teacher_state_for_il(self, policy_state):
        """Select the teacher's observability contract explicitly.

        The paper-2 protocol uses ``policy`` so ORCA plans on the same belief
        means available to the student. ``oracle`` remains available only for
        legacy experiments whose config explicitly requests it.
        """
        config = getattr(self.env, 'config', None)
        teacher_observation = 'oracle'
        if config is not None and config.has_section('imitation_learning'):
            teacher_observation = config.get(
                'imitation_learning', 'teacher_observation',
                fallback='oracle').strip().lower()
        if teacher_observation == 'policy':
            return policy_state
        if teacher_observation == 'oracle':
            return (self.env.get_oracle_state()
                    if hasattr(self.env, 'get_oracle_state') else policy_state)
        raise ValueError(
            "[imitation_learning] teacher_observation must be policy or oracle, "
            f"got {teacher_observation!r}")

    # PPO不需要update_target_model：删除target network更新（SAC/DQN专用）

    # --------------------------------------------------------------
    # 形状/格式转换（确保 34D）
    # --------------------------------------------------------------
    def _state_to_array(self, state) -> np.ndarray:
        # Occlusion states must stay as [rows,13] tokens. Converting them through
        # JointState.to_array() irreversibly drops belief columns 9-12.
        if hasattr(state, 'to_policy_tokens'):
            tokens = state.to_policy_tokens()
            if tokens is not None:
                tokens = _as_np32(tokens)
                if tokens.ndim != 2 or tokens.shape[1] != 13 or tokens.shape[0] < 4:
                    raise RuntimeError(
                        f"invalid policy token contract {tokens.shape}; expected [rows>=4,13]")
                return tokens
        arr = np.asarray(state) if isinstance(state, np.ndarray) else None
        if (arr is not None and arr.ndim == 2 and arr.shape[1] == 13 and
                arr.shape[0] >= 4):
            return _as_np32(arr)
        if hasattr(state, 'to_array'):
            arr = state.to_array()
            allowed_dims = {14, 19, 24, 29, 34}
            if arr.shape[0] in allowed_dims:
                if arr.shape[0] < 34:
                    pad = np.zeros(34 - arr.shape[0], dtype=np.float32)
                    arr_padded = np.concatenate([_as_np32(arr), pad], axis=0)
                else:
                    arr_padded = _as_np32(arr)
                if hasattr(self, 'env') and hasattr(self.env, 'phase') and self.env.phase == 'train':
                    noise = np.random.normal(0, 0.02, size=arr_padded.shape).astype(np.float32)
                    arr_padded = arr_padded + noise
                return arr_padded
            else:
                if arr.shape[0] not in self._logged_state_dims:
                    logging.warning(f"[EXPLORER] Unexpected state dimension: {arr.shape[0]}, expected one of {sorted(allowed_dims)}")
                    self._logged_state_dims.add(arr.shape[0])
                if arr.shape[0] < 34:
                    pad = np.zeros(34 - arr.shape[0], dtype=np.float32)
                    arr_padded = np.concatenate([_as_np32(arr), pad], axis=0)
                else:
                    arr_padded = _as_np32(arr[:34])
                if hasattr(self, 'env') and hasattr(self.env, 'phase') and self.env.phase == 'train':
                    noise = np.random.normal(0, 0.02, size=arr_padded.shape).astype(np.float32)
                    arr_padded = arr_padded + noise
                return arr_padded
        try:
            t = to_btnd(torch.as_tensor(state)) 
            if t.dim() == 4:
                t = t[:, -1] 
            if t.dim() == 3:
                t = t[0]
            arr34 = tokens_to_34d(t.unsqueeze(0)).squeeze(0).cpu().numpy()
            return _as_np32(arr34)
        except Exception:
            return _as_np32(state)

    def _action_to_array(self, action) -> np.ndarray:
        if hasattr(action, 'vx') and hasattr(action, 'vy'):
            return np.array([float(action.vx), float(action.vy)], dtype=np.float32)
        return _as_np32(action)

    # --------------------------------------------------------------
    # PPO不需要collect_with_labels：删除IL标签收集（已由generate_il_dataset.py处理）

    # --------------------------------------------------------------
    # 主采样：与训练/评估循环对接
    # --------------------------------------------------------------
    def run_k_episodes(
        self, 
        k: int,
        phase: str,
        update_memory: bool = False,
        imitation_learning: bool = False,
        episode: Optional[int] = None,
        print_failure: bool = False,
        force_joint_state_policy: bool = False,
        return_stats: bool = False,
        store_on=("success", "collision", "timeout"),
        show_tqdm: Optional[bool] = None,
        il_mix_ratio: float = 0.0,  # 兼容老签名，但逻辑不使用
        # 【精简】已删除 expert_policy 和 student_prob 参数
    ):
        from crowd_sim.envs.utils.state import JointState

        if show_tqdm is None:
            show_tqdm = bool(imitation_learning or update_memory)

        if hasattr(self.robot.policy, 'set_phase'):
            self.robot.policy.set_phase(phase)

        success_times: List[float] = []; collision_times: List[float] = []; timeout_times: List[float] = []
        success = collision = timeout = 0
        rewards_hist: List[float] = []; cumulative_rewards: List[float] = []

        # 论文对标指标：收集新增3项
        time_taken_list: List[float] = []
        discomfort_freq_list: List[float] = []
        discomfort_dist_list: List[float] = []

        # todo.md C1: Extended metrics（路径长度、平滑度、能量效率）
        path_length_list: List[float] = []
        smoothness_list: List[float] = []  # 转向频率（angular jerk）
        speed_list: List[float] = []  # 平均速度

        _SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
        _COLLISION_TOKENS = {"collision"}
        _TIMEOUT_TOKENS = {"timeout", "no_progress"}  # no_progress也算timeout
        store_on = tuple(s.lower() for s in store_on)

        # 直接开始IL训练，无需重复的探针测试
        if phase == 'il':
            # CrowdSim.reset only supports train/val/test
            phase = 'train'
            if not hasattr(self, '_il_phase_warned'):
                logging.info("[IL] phase='il' remapped to 'train' for env.reset compatibility")
                self._il_phase_warned = True

        self._last_trajectories = []

        iterator = range(k)
        if show_tqdm:
            import sys
            tag = "IL" if imitation_learning else "RUN"
            iterator = tqdm(iterator, desc=f"{phase.upper()} {tag}", dynamic_ncols=True, leave=False,
                            position=0, file=sys.stdout, miniters=max(1, k//20), maxinterval=1.0,
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}', ncols=80)

        for i in iterator:
            self.env.phase = phase
            # 设置policy的phase，用于predict()中区分训练/评估阶段
            if hasattr(self.robot.policy, 'phase'):
                self.robot.policy.phase = phase
            if hasattr(self.robot.policy, 'reset_episode_stats'):
                self.robot.policy.reset_episode_stats()

            # 环境reset：phase已通过self.env.phase设置，reset()不接受phase参数
            obs = self.env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs = obs[0]

            terminated = False; truncated = False; info: Dict[str, Any] = {}
            states: List[Any] = []; actions: List[np.ndarray] = []; rewards: List[float] = []
            action_indices: List[int] = []  # ✅ 新增：存储离散动作索引
            log_probs: List[float] = []; values: List[float] = []  # PPO需要
            on_policy_mask: List[float] = []  # Gated Demonstrator: 1.0=学生决策, 0.0=老师决策

            # todo.md C1: Extended metrics tracking（每个episode）
            episode_path_length = 0.0
            episode_angular_changes = []
            prev_action = None
            initial_distance = np.hypot(self.robot.gx - self.robot.px, self.robot.gy - self.robot.py)

            # 轨迹记录：记录episode起点（总是记录，用于trajectory分析）
            trajectory_positions = []
            start_pos = (float(self.robot.px), float(self.robot.py))
            goal_pos = (float(self.robot.gx), float(self.robot.gy))
            trajectory_positions.append(start_pos)  # 记录起点

            while not (terminated or truncated):
                # 选择状态类型：JointState 或 观测数组
                policy_name = self.robot.policy.__class__.__name__.lower()
                use_joint = force_joint_state_policy or any(k in policy_name for k in ['orca','cadrl','sarl','multi_human_rl','mamba'])

                if use_joint:
                    # Order 17 item 10: the STUDENT may only ever see what the
                    # active arm allows. With occlusion off get_policy_state()
                    # returns the same ground-truth JointState as before, so
                    # the legacy path is unchanged.
                    state = (self.env.get_policy_state()
                             if hasattr(self.env, 'get_policy_state')
                             else JointState(self.robot.get_full_state(),
                                             [h.get_observable_state() for h in self.env.humans]))
                    # debug.md P0: IL阶段ORCA直接接管，不经过量化/掩码
                    if imitation_learning:
                        # debug.md P0: ORCA真接管 - 直接获取连续动作，不量化到80动作格
                        if not hasattr(self, '_orca_policy'):
                            try:
                                from crowd_nav.policy.policy_factory import policy_factory
                                self._orca_policy = policy_factory['orca']()
                                if hasattr(self._orca_policy, 'set_env'):
                                    self._orca_policy.set_env(self.env)
                                if hasattr(self._orca_policy, 'time_step'):
                                    self._orca_policy.time_step = getattr(self.env, 'time_step', 0.25)

                                # debug.md P0.2修复：配置ORCA标准参数
                                if hasattr(self._orca_policy, 'configure') and hasattr(self.env, 'config'):
                                    self._orca_policy.configure(self.env.config)
                                    logging.info("[DEBUG.MD-P0.2] ORCA policy configured with standard parameters")
                                    # 🔥 调试：输出ORCA实际参数
                                    logging.info(f"[ORCA-DEBUG] safety_space={self._orca_policy.safety_space}, "
                                               f"neighbor_dist={self._orca_policy.neighbor_dist}, "
                                               f"time_horizon={self._orca_policy.time_horizon}")
                                else:
                                    logging.warning(f"[ORCA-DEBUG] configure() NOT called! "
                                                  f"has_configure={hasattr(self._orca_policy, 'configure')}, "
                                                  f"has_config={hasattr(self.env, 'config')}")

                                # debug.md P0.2修复：IL阶段固定种子提升成功率稳定性
                                if hasattr(self._orca_policy, 'seed'):
                                    self._orca_policy.seed(42)  # 固定种子确保IL成功率稳定
                                    logging.info("[DEBUG.MD-P0.2] ORCA policy seeded for stable IL success rate")

                                # debug.md P1: ORCA参数回归CrowdNav口径（移除强化参数）
                                # 删除强化参数覆盖，让ORCA使用标准配置
                                logging.info("[DEBUG.MD-P0] Created ORCA policy with standard parameters for IL stage")
                            except Exception as e:
                                logging.warning(f"[DEBUG.MD-P0] Failed to create ORCA policy: {e}, using current policy")
                                self._orca_policy = self.robot.policy
                        # Dual-Head修复：IL阶段直接使用ORCA连续动作（不量化）
                        # 这样BC可以学习原始的精细控制，不受离散化精度损失
                        _teacher_state = self._teacher_state_for_il(state)
                        orca_action = self._orca_policy.predict(_teacher_state)  # 连续动作 (vx, vy)

                        # IL阶段：使用连续ORCA动作（SARL原版口径）
                        from crowd_sim.envs.utils.action import ActionXY
                        if hasattr(orca_action, 'vx'):
                            vx = float(orca_action.vx)
                            vy = float(orca_action.vy)
                        else:
                            vx = float(orca_action[0])
                            vy = float(orca_action[1] if len(orca_action) > 1 else 0.0)
                        action = ActionXY(vx, vy)
                        # 🔥 修复：IL阶段量化ORCA动作为离散索引（用于RL prefill）
                        if hasattr(self.robot.policy, 'continuous_to_discrete_index'):
                            try:
                                action_idx = self.robot.policy.continuous_to_discrete_index(action)
                            except Exception:
                                action_idx = None
                        else:
                            action_idx = None
                        tag = "IL-ORCA"
                        # IL阶段：ORCA不提供log_prob和value，填充dummy值
                        current_log_prob, current_value = 0.0, 0.0
                        on_policy_flag = 0.0  # IL步不进入PPO更新
                    else:
                        # 【精简】RL阶段：直接使用学生策略（已删除Gated Demonstrator）
                        policy_obj = self.robot.policy
                        policy_base = getattr(policy_obj, "_orig_mod", policy_obj)
                        if hasattr(policy_base, 'predict_with_value') or hasattr(policy_obj, 'predict_with_value'):
                            action, current_log_prob, current_value = policy_obj.predict_with_value(state)
                            action_idx = None  # PPO不返回idx
                            tag = "RL-MAMBA-PPO"
                        else:
                            # ✅ DoubleQ: 优先使用act()以返回(action, idx)
                            act_fn = getattr(policy_base, 'act', None) or getattr(policy_obj, 'act', None)
                            if act_fn is not None:
                                result = act_fn(state)
                            else:
                                pred_fn = getattr(policy_base, 'predict', None) or getattr(policy_obj, 'predict', None)
                                if pred_fn is None:
                                    raise AttributeError("Policy missing both act() and predict() in RL stage")
                                try:
                                    result = pred_fn(state, deterministic=False, return_idx=True)
                                except TypeError:
                                    result = pred_fn(state)
                            if isinstance(result, tuple) and len(result) == 2:
                                action, action_idx = result
                            else:
                                action = result
                                action_idx = None
                                # 兜底：尝试从predict(return_idx=True)获取索引
                                pred_fn = getattr(policy_base, 'predict', None) or getattr(policy_obj, 'predict', None)
                                if pred_fn is not None:
                                    try:
                                        action, action_idx = pred_fn(state, deterministic=False, return_idx=True)
                                    except TypeError:
                                        pass
                            current_log_prob, current_value = 0.0, 0.0
                            tag = "RL-MAMBA-DOUBLEQ"
                        on_policy_flag = 1.0  # RL阶段所有步骤参与PPO更新
                    # 首次记录policy类型（仅一次）
                    if len(states) == 0 and not self._obs_mode_logged.get(tag, False):
                        try:
                            log.info(f"[EXPLORER] {tag}: JointState, state_dim={state.to_array().shape}")
                        except Exception:
                            pass
                        self._obs_mode_logged[tag] = True
                else:
                    # 非JointState分支：为Mamba policy构造JointState（需要34维完整输入）
                    # Order 17 item 10: the STUDENT may only ever see what the
                    # active arm allows. With occlusion off get_policy_state()
                    # returns the same ground-truth JointState as before, so
                    # the legacy path is unchanged.
                    state = (self.env.get_policy_state()
                             if hasattr(self.env, 'get_policy_state')
                             else JointState(self.robot.get_full_state(),
                                             [h.get_observable_state() for h in self.env.humans]))

                    # debug.md P0: 非JointState情况下ORCA直接接管
                    if imitation_learning:
                        if not hasattr(self, '_orca_policy'):
                            try:
                                from crowd_nav.policy.policy_factory import policy_factory
                                self._orca_policy = policy_factory['orca']()
                                if hasattr(self._orca_policy, 'set_env'):
                                    self._orca_policy.set_env(self.env)
                                if hasattr(self._orca_policy, 'time_step'):
                                    self._orca_policy.time_step = getattr(self.env, 'time_step', 0.25)

                                # debug.md P0.2修复：配置ORCA标准参数
                                if hasattr(self._orca_policy, 'configure') and hasattr(self.env, 'config'):
                                    self._orca_policy.configure(self.env.config)
                                    logging.info("[DEBUG.MD-P0.2] ORCA policy configured with standard parameters (non-joint)")

                                # debug.md P0.2修复：IL阶段固定种子提升成功率稳定性
                                if hasattr(self._orca_policy, 'seed'):
                                    self._orca_policy.seed(42)  # 固定种子确保IL成功率稳定
                                    logging.info("[DEBUG.MD-P0.2] ORCA policy seeded for stable IL success rate (non-joint)")

                                # debug.md P1: ORCA参数回归CrowdNav口径（移除强化参数）
                                # 删除强化参数覆盖，让ORCA使用标准配置
                                logging.info("[DEBUG.MD-P0] Created ORCA policy for IL stage (non-joint, standard)")
                            except Exception as e:
                                logging.warning(f"[DEBUG.MD-P0] Failed to create ORCA policy: {e}, using current policy")
                                self._orca_policy = self.robot.policy
                        # debug.md P0: 直接使用ORCA输出（使用JointState）
                        _teacher_state = self._teacher_state_for_il(state)
                        orca_action = self._orca_policy.predict(_teacher_state)
                        from crowd_sim.envs.utils.action import ActionXY
                        if hasattr(orca_action, 'vx'):
                            vx = float(orca_action.vx)
                            vy = float(orca_action.vy)
                        else:
                            vx = float(orca_action[0])
                            vy = float(orca_action[1] if len(orca_action) > 1 else 0.0)
                        action = ActionXY(vx, vy)
                        # 🔥 修复：IL阶段量化ORCA动作为离散索引（用于RL prefill）
                        if hasattr(self.robot.policy, 'continuous_to_discrete_index'):
                            try:
                                action_idx = self.robot.policy.continuous_to_discrete_index(action)
                            except Exception:
                                action_idx = None
                        else:
                            action_idx = None
                        tag = "IL-ORCA-DISCRETE"
                        # IL阶段：ORCA不提供log_prob和value，填充dummy值
                        current_log_prob, current_value = 0.0, 0.0
                        on_policy_flag = 0.0  # IL步不进入PPO更新
                    else:
                        # 【精简】RL阶段：直接使用学生策略（已删除Gated Demonstrator）
                        policy_obj = self.robot.policy
                        policy_base = getattr(policy_obj, "_orig_mod", policy_obj)
                        if hasattr(policy_base, 'predict_with_value') or hasattr(policy_obj, 'predict_with_value'):
                            action, current_log_prob, current_value = policy_obj.predict_with_value(state)
                            action_idx = None  # PPO不返回idx
                            tag = "RL-MAMBA-PPO"
                        else:
                            # ✅ DoubleQ: 优先使用act()以返回(action, idx)
                            act_fn = getattr(policy_base, 'act', None) or getattr(policy_obj, 'act', None)
                            if act_fn is not None:
                                result = act_fn(state)
                            else:
                                pred_fn = getattr(policy_base, 'predict', None) or getattr(policy_obj, 'predict', None)
                                if pred_fn is None:
                                    raise AttributeError("Policy missing both act() and predict() in RL stage")
                                try:
                                    result = pred_fn(state, deterministic=False, return_idx=True)
                                except TypeError:
                                    result = pred_fn(state)
                            if isinstance(result, tuple) and len(result) == 2:
                                action, action_idx = result
                            else:
                                action = result
                                action_idx = None
                                # 兜底：尝试从predict(return_idx=True)获取索引
                                pred_fn = getattr(policy_base, 'predict', None) or getattr(policy_obj, 'predict', None)
                                if pred_fn is not None:
                                    try:
                                        action, action_idx = pred_fn(state, deterministic=False, return_idx=True)
                                    except TypeError:
                                        pass
                            current_log_prob, current_value = 0.0, 0.0
                            tag = "RL-MAMBA-DOUBLEQ"
                        on_policy_flag = 1.0  # RL阶段所有步骤参与PPO更新
                    # 首次记录policy类型（仅一次）
                    if len(states) == 0 and not self._obs_mode_logged.get(tag, False):
                        log.info(f"[EXPLORER] {tag}: JointState constructed, state_dim={state.to_array().shape}")
                        self._obs_mode_logged[tag] = True

                # 🔥 ORCA Safety Shield: REMOVED (RL fully responsible)
                orca_override_flag = False

                a_arr = self._action_to_array(action)
                s_arr = self._state_to_array(state)

                # 记录ORCA覆盖标志（用于后续分析）
                if orca_override_flag:
                    on_policy_flag = 0.0  # ORCA强制干预的步骤不参与RL更新

                step_out = self.env.step(action)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                elif isinstance(step_out, tuple) and len(step_out) == 4:
                    obs, reward, terminated, info = step_out
                    truncated = False
                else:
                    raise RuntimeError(f"Unexpected env.step() output length: {len(step_out)}")

                states.append(s_arr)
                actions.append(a_arr)
                # ✅ 收集action_idx（如果policy返回了）
                # 🔥 修复：直接检查action_idx变量，不用locals()
                action_indices.append(action_idx if action_idx is not None else None)
                rewards.append(float(reward))
                log_probs.append(current_log_prob)  # PPO: 保存log_prob
                values.append(current_value)  # PPO: 保存value
                on_policy_mask.append(float(on_policy_flag))  # Gated Demonstrator: 0 or 1

                # todo.md C1: Track path length and smoothness
                dt = getattr(self.env, 'time_step', 0.25)
                vx, vy = float(a_arr[0]), float(a_arr[1])
                step_distance = np.hypot(vx, vy) * dt
                episode_path_length += step_distance

                # Angular jerk (转向频率)
                if prev_action is not None:
                    prev_vx, prev_vy = float(prev_action[0]), float(prev_action[1])
                    prev_angle = np.arctan2(prev_vy, prev_vx)
                    curr_angle = np.arctan2(vy, vx)
                    angle_change = abs(curr_angle - prev_angle)
                    # Normalize to [0, pi]
                    angle_change = min(angle_change, 2*np.pi - angle_change)
                    episode_angular_changes.append(angle_change)
                prev_action = a_arr

                # 轨迹记录：每步后记录机器人位置（总是记录）
                curr_pos = (float(self.robot.px), float(self.robot.py))
                trajectory_positions.append(curr_pos)

            # ⚠️ 关键修复：优先使用环境返回的事件，步数只做兜底判断
            raw = _extract_event_token(info)
            is_time_limit = bool(truncated) or (isinstance(info, dict) and bool(info.get("TimeLimit.truncated", False)))

            episode_steps = len(rewards)
            dt = float(getattr(self.env, 'time_step', 0.25))
            time_limit = float(getattr(self.env.config, 'time_limit', 25.0))
            max_steps = int(time_limit / dt)

            # ✅ 正确优先级：1)环境事件token  2)truncated标志  3)步数兜底
            if raw in _SUCCESS_TOKENS:
                event = "success"; success += 1
            elif raw in _COLLISION_TOKENS:
                event = "collision"; collision += 1
            elif raw in _TIMEOUT_TOKENS:
                event = "timeout"; timeout += 1
            elif is_time_limit or episode_steps >= max_steps:
                event = "timeout"; timeout += 1  # 步数到上限作为兜底
            else:
                # 未识别的事件（nothing/danger等中间状态），根据步数判断为timeout
                event = "timeout"; timeout += 1
                # 降低日志级别：nothing/danger是正常的no_progress终止，不需要警告
                if raw not in ('nothing', 'danger'):
                    try:
                        log.warning('[P0-3] Unrecognized end signal: %s, steps=%d/%d. Classify as timeout.',
                                   getattr(info, 'event', str(info)), episode_steps, max_steps)
                    except Exception:
                        log.warning('[P0-3] Unrecognized end signal: <unknown>, steps=%d/%d. Classify as timeout.',
                                   episode_steps, max_steps)

            # debug.md: 事件真相源断言强化
            assert event in ("success", "collision", "timeout"), \
                f"[ASSERT] Invalid event: {event}, expected one of success/collision/timeout"

            # 统计时间/回报（已在上面定义，无需重复）
            actual_time = episode_steps * dt
            if event == 'success':
                success_times.append(actual_time)
            elif event == 'collision':
                collision_times.append(actual_time)
            elif event == 'timeout':
                timeout_times.append(actual_time)

            should_store = (event in store_on)

            if update_memory:
                if should_store:  # 修复bug：直接使用should_store，不再忽略RL阶段的store_on
                    self.update_memory(states, actions, rewards, phase, event, log_probs, values, on_policy_mask, action_indices)

            if (not imitation_learning) or should_store:
                # 添加trajectory positions到info中（用于vectorized_sampler记录）
                info_dict = dict(info) if isinstance(info, dict) else {}
                info_dict['trajectory_positions'] = trajectory_positions
                info_dict['start_pos'] = trajectory_positions[0] if trajectory_positions else (0.0, 0.0)
                info_dict['goal_pos'] = (float(self.robot.gx), float(self.robot.gy))
                info_dict['event'] = event
                # 记录离散动作索引（用于DoubleQ训练回放）
                info_dict['action_indices'] = list(action_indices)
                self._last_trajectories.append(((states, actions, rewards), info_dict))

            # 轨迹记录：写入文件（只在RL阶段且enable时）
            if self.enable_trajectory_logging and phase == 'train' and not imitation_learning and self.trajectory_logger:
                try:
                    # 写入轨迹记录：[EP=X] START=(x,y) GOAL=(x,y) RESULT=status STEPS=n
                    start_pos = trajectory_positions[0] if trajectory_positions else (0.0, 0.0)
                    goal_pos = (float(self.robot.gx), float(self.robot.gy))
                    self.trajectory_logger.write(f"[EP={i}] START={start_pos} GOAL={goal_pos} RESULT={event.upper()} STEPS={len(trajectory_positions)}\n")
                    # 写入轨迹坐标：TRAJECTORY: (x1,y1) (x2,y2) ...
                    trajectory_str = ' '.join(f"{pos}" for pos in trajectory_positions)
                    self.trajectory_logger.write(f"TRAJECTORY: {trajectory_str}\n\n")
                    self.trajectory_logger.flush()  # 立即刷新到磁盘
                except Exception as e:
                    logging.warning(f"[TRAJECTORY-LOG] Failed to write trajectory: {e}")

            # 论文对标指标：从info中提取（仅当info为dict时）
            if isinstance(info, dict):
                if "time_taken" in info:
                    time_taken_list.append(info["time_taken"])
                if "discomfort_freq" in info:
                    discomfort_freq_list.append(info["discomfort_freq"])
                if "discomfort_dist" in info:
                    discomfort_dist_list.append(info["discomfort_dist"])

            # todo.md C1: Collect extended metrics for this episode
            path_length_list.append(episode_path_length)
            if len(episode_angular_changes) > 0:
                # Smoothness: 转向频率的倒数（值越大越平滑）
                avg_angular_jerk = np.mean(episode_angular_changes)
                smoothness = 1.0 / (1.0 + avg_angular_jerk)  # 归一化到[0,1]
                smoothness_list.append(smoothness)
            else:
                smoothness_list.append(1.0)  # 单步episode视为完全平滑

            # 平均速度（m/s）
            episode_time = len(rewards) * dt
            if episode_time > 0:
                avg_speed = episode_path_length / episode_time
                speed_list.append(avg_speed)
            else:
                speed_list.append(0.0)

            if self.gamma is None:
                disc = rewards
            else:
                disc = [pow(self.gamma, t) * r for t, r in enumerate(rewards)]
            cumulative_rewards.append(sum(disc))
            rewards_hist.append(sum(rewards))

            if show_tqdm and (i % max(1, k//10) == 0 or i == k-1):
                n = i + 1
                iterator.set_postfix_str(f"S:{success/n:.2f} C:{collision/n:.2f}", refresh=False)

        k = max(1, int(k))
        # debug.md: 事件真相源断言强化 - 验证事件计数与episode数一致
        assert (success + collision + timeout) == k, \
            f"[ASSERT] Event count mismatch: succ={success} + coll={collision} + tout={timeout} = {success+collision+timeout} != episodes={k}"

        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        all_times = success_times + collision_times + timeout_times
        avg_nav_time = (sum(all_times) / len(all_times)) if len(all_times) else getattr(self.env, 'time_limit', 35.0)
        avg_reward = float(np.mean(cumulative_rewards)) if len(cumulative_rewards) else 0.0

        # 论文对标指标：计算平均值
        avg_time_taken = float(np.mean(time_taken_list)) if time_taken_list else avg_nav_time
        avg_discomfort_freq = float(np.mean(discomfort_freq_list)) if discomfort_freq_list else 0.0
        avg_discomfort_dist = float(np.mean(discomfort_dist_list)) if discomfort_dist_list else 0.0

        # todo.md C1: Compute extended metrics
        avg_path_length = float(np.mean(path_length_list)) if path_length_list else 0.0
        avg_smoothness = float(np.mean(smoothness_list)) if smoothness_list else 0.0
        avg_speed = float(np.mean(speed_list)) if speed_list else 0.0

        # debug.md step ③: 完全移除所有旁路统计，只允许train.py中的官方IL-EVAL输出
        # 删除所有"phase has success rate"类型的旁路日志，确保唯一真相源
        if phase.lower() in ['train']:
            log.debug('{:<5} summary: succ={:.2f}, coll={:.2f}, timeout={:.2f}, nav={:.2f}, reward={:.4f}'
                     .format(phase.upper(), success_rate, collision_rate, timeout_rate, avg_nav_time, avg_reward))

        if return_stats:
            return {
                "success_rate": float(success_rate),
                "collision_rate": float(collision_rate),
                "timeout_rate": float(timeout_rate),
                "nav_time": float(avg_nav_time),
                "total_reward": float(avg_reward),
                "time_taken": avg_time_taken,
                "discomfort_freq": avg_discomfort_freq,
                "discomfort_dist": avg_discomfort_dist,
                # todo.md C1: Extended metrics
                "path_length": avg_path_length,
                "smoothness": avg_smoothness,
                "avg_speed": avg_speed,
            }
        else:
            # 扩展返回格式以包含新指标：(succ, coll, timeout, ep_reward, nav_t, time_taken, discomfort_freq, discomfort_dist, path_length, smoothness, speed)
            return (float(success_rate), float(collision_rate), float(timeout_rate),
                   float(avg_reward), float(avg_nav_time),
                   avg_time_taken, avg_discomfort_freq, avg_discomfort_dist,
                   avg_path_length, avg_smoothness, avg_speed)


    # --------------------------------------------------------------
    # ReplayMemory 接口对接（V-learning 使用）
    # --------------------------------------------------------------
    def update_memory(self, states, actions, rewards, phase: str = 'rl', event: str = 'unknown',
                      log_probs: List[float] = None, values: List[float] = None, on_policy_mask: List[float] = None,
                      action_indices: List[int] = None):  # ✅ 新增：离散动作索引
        """
        Episode级别写入memory（DoubleQ扩展：添加action_indices）
        """
        # Active occlusion episodes are already [T,8,13] tokens. Preserve
        # them; the replay buffer validates the belief-column contract.
        states_payload = np.array(
            [self._state_to_array(s) for s in states], dtype=np.float32)
        rewards_arr = np.array(rewards, dtype=np.float32)

        # 构造dones和timeouts
        dones_arr = np.zeros(len(states), dtype=np.bool_)
        dones_arr[-1] = True
        timeouts_arr = np.zeros(len(states), dtype=np.bool_)
        if 'timeout' in event.lower():
            timeouts_arr[-1] = True

        # 连续控制：存储真实速度（不归一化）
        # ✅ 修复：Buffer存储真实速度[-v_max, v_max]，与环境动作一致
        # 这样BC loss和Q网络都处理真实速度，避免尺度混乱
        actions_continuous = []
        for act in actions:
            a_arr = self._action_to_array(act)
            vx, vy = float(a_arr[0]), float(a_arr[1])  # 真实速度，不除以v_max
            actions_continuous.append([vx, vy])
        actions_continuous_arr = np.array(actions_continuous, dtype=np.float32)  # [T, 2] 真实速度

        # PPO扩展：转换log_probs和values和on_policy_mask为numpy数组
        log_probs_arr = np.array(log_probs, dtype=np.float32) if log_probs else np.zeros(len(states), dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32) if values else np.zeros(len(states), dtype=np.float32)
        on_policy_arr = np.array(on_policy_mask, dtype=np.float32) if on_policy_mask else np.ones(len(states), dtype=np.float32)

        # 构造meta（todo.md方案二：添加initial_distance用于效率评分）
        initial_distance = 0.0
        if (states_payload.ndim == 3 and states_payload.shape[-1] == 13 and
                states_payload.shape[1] >= 4):
            first_state = states_payload[0]
            initial_distance = np.hypot(
                float(first_state[1, 0]), float(first_state[1, 1]))
        elif len(states_payload) > 0 and states_payload.shape[-1] >= 34:
            # 从34D状态提取robot位置和目标位置
            first_state = states_payload[0]
            px, py = float(first_state[0]), float(first_state[1])
            gx, gy = float(first_state[5]), float(first_state[6])
            initial_distance = np.hypot(gx - px, gy - py)

        event_lower = event.lower()
        is_success = 1 if 'success' in event_lower or 'reach' in event_lower else 0
        is_timeout = 1 if 'timeout' in event_lower else 0
        is_collision = 1 if 'collision' in event_lower else 0

        # 根据phase判断来源
        source_tag = 'IL' if phase.lower() == 'il' else 'RL'

        meta = {
            'source': source_tag,
            'event': event,
            'total_reward': float(rewards_arr.sum()),
            'initial_distance': float(initial_distance),  # todo.md方案二
            'flag_success': is_success,
            'flag_timeout': is_timeout,
            'flag_collision': is_collision,
        }

        # Episode级别写入
        if self.memory is not None:
            self.memory.push_episode(
                states=states_payload,
                rewards=rewards_arr,
                dones=dones_arr,
                timeouts=timeouts_arr,
                meta=meta,
                actions_continuous=actions_continuous_arr,
                log_probs=log_probs_arr,  # PPO需要
                values=values_arr,  # PPO需要
                on_policy_mask=on_policy_arr,  # Gated Demonstrator需要
                action_indices=np.array(action_indices, dtype=np.int64) if any(x is not None for x in action_indices) else None  # ✅ DoubleQ需要
            )
            # 分层探索：记录episode来源到日志（debug）
            import logging
            logging.debug(f"[STRATIFIED-PUSH] Pushed episode: source={meta['source']}, event={event}, len={len(states_payload)}")
