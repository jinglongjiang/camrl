import logging
import copy
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # 极端环境没有 tqdm 时降级
    def tqdm(x, **kwargs): return x


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # -------------------- 工具：状态/动作转数组 --------------------
    @staticmethod
    def _state_to_array(state):
        if hasattr(state, 'to_array'):
            return state.to_array().astype(np.float32)
        if isinstance(state, np.ndarray):
            return state.astype(np.float32)
        return np.asarray(state, dtype=np.float32)

    @staticmethod
    def _action_to_array(action):
        if hasattr(action, 'to_array'):
            return action.to_array().astype(np.float32)
        if hasattr(action, 'vx') and hasattr(action, 'vy'):
            return np.array([action.vx, action.vy], dtype=np.float32)
        return np.asarray(action, dtype=np.float32)

    # -------------------- DAgger：单条 episode 采样（可选步级进度条） --------------------
    def collect_with_labels(
        self,
        env,
        rl_policy,
        expert_policy,
        rl_buf,
        exp_buf,
        eps=0.0,
        max_steps=None,
        return_stats=False,
        store_on=("success", "collision", "timeout"),
        show_step_bar=False,   # <<< 新增：显示步级进度条
    ):
        """
        用 RL 动作执行环境，同时记录专家标签到 exp_buf。
        """
        from crowd_sim.envs.utils.state import JointState

        # 同步步长给策略（避免 ORCA 无 time_step）
        if hasattr(rl_policy, 'time_step'):
            rl_policy.time_step = self.env.time_step
        if hasattr(expert_policy, 'time_step'):
            expert_policy.time_step = self.env.time_step

        store_on = tuple(s.lower() for s in store_on)
        _SUCCESS_TOKENS = {"reachgoal", "reach_goal", "success"}
        _COLLISION_TOKENS = {"collision"}
        _TIMEOUT_TOKENS = {"timeout"}

        # reset 兼容 gym/gymnasium
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out

        phase = getattr(self.env, "phase", "train")
        if hasattr(rl_policy, "set_phase"): rl_policy.set_phase(phase)
        if hasattr(expert_policy, "set_phase"): expert_policy.set_phase(phase)

        done = False
        truncated = False
        steps = 0
        total_reward = 0.0
        too_close = 0
        min_dist_list = []
        success_time = None
        info = {}

        def build_joint(_):
            return JointState(self.robot.get_full_state(),
                              [h.get_observable_state() for h in self.env.humans])

        use_joint = True
        state = build_joint(obs) if use_joint else obs
        state_arr = self._state_to_array(state)

        # epsilon 探索（若策略支持）
        old_eps = getattr(rl_policy, "epsilon", 0.0)
        if eps is not None and hasattr(rl_policy, "set_epsilon"):
            rl_policy.set_epsilon(float(eps))

        step_bar = None
        if show_step_bar:
            step_bar = tqdm(total=max_steps, desc="IL steps", dynamic_ncols=True, leave=False)

        while not (done or truncated):
            # RL 行动
            try:
                a_rl = rl_policy.predict(state if use_joint else obs)
            except Exception:
                a_rl = rl_policy.predict(build_joint(obs))
            a_rl_arr = self._action_to_array(a_rl)

            # 专家标签
            try:
                a_expert = expert_policy.predict(state if use_joint else obs)
            except Exception:
                a_expert = expert_policy.predict(build_joint(obs))
            a_exp_arr = self._action_to_array(a_expert)

            # step 兼容
            step_out = env.step(a_rl)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs_next, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                obs_next, reward, done, info = step_out
                truncated = False
            else:
                raise RuntimeError(f"Unexpected env.step() output length: {len(step_out)}")

            total_reward += float(reward)
            steps += 1
            if step_bar is not None:
                step_bar.update(1)

            if str(info.get("event", "")).lower() == "danger":
                too_close += 1
                min_dist_list.append(float(info.get("min_dist", 0.0)))

            next_state = build_joint(obs_next) if use_joint else obs_next
            next_state_arr = self._state_to_array(next_state)

            # 写入回放
            rl_buf.push(state_arr, a_rl_arr.astype(np.float32), float(reward), next_state_arr, bool(done or truncated))
            exp_buf.push(state_arr, a_exp_arr.astype(np.float32))

            # 下一步
            state = next_state
            state_arr = next_state_arr
            obs = obs_next

            if max_steps is not None and steps >= int(max_steps):
                truncated = True
                info = dict(info or {})
                info["TimeLimit.truncated"] = True
                break

        if step_bar is not None:
            step_bar.close()

        if hasattr(rl_policy, "set_epsilon"):
            rl_policy.set_epsilon(old_eps)

        # 归一化结束事件
        raw = str((info or {}).get("event", "")).replace(" ", "_").lower().strip()
        is_time_limit = bool(truncated) or bool((info or {}).get("TimeLimit.truncated", False))
        if raw in _SUCCESS_TOKENS:
            event = "success"; success_time = float(self.env.global_time)
        elif raw in _COLLISION_TOKENS:
            event = "collision"
        elif raw in _TIMEOUT_TOKENS or is_time_limit:
            event = "timeout"
        else:
            event = "success"
            logging.warning('[Explorer] Unrecognized end signal in collect_with_labels: %s. Treat as success.',
                            (info or {}).get("event"))

        # 统计
        freq_danger = float(too_close) / float(max(1, steps))
        avg_min_sep = float(np.mean(min_dist_list)) if len(min_dist_list) else 0.0

        if return_stats:
            return {
                "event": event,
                "steps": int(steps),
                "total_reward": float(total_reward),
                "success_time": float(success_time) if success_time is not None else None,
                "freq_danger": float(freq_danger),
                "avg_min_sep": float(avg_min_sep),
            }

    # -------------------- 批量运行（加 episode 级进度条） --------------------
    def run_k_episodes(
        self,
        k,
        phase,
        update_memory=False,
        imitation_learning=False,
        episode=None,
        print_failure=False,
        force_joint_state_policy=False,
        return_stats=False,
        store_on=("success", "collision", "timeout"),
        show_tqdm=None,   # <<< 新增：是否显示进度条（默认：IL 或 update_memory 时显示）
    ):
        from crowd_sim.envs.utils.state import JointState

        if show_tqdm is None:
            show_tqdm = bool(imitation_learning or update_memory)

        self.robot.policy.set_phase(phase)
        success_times, collision_times, timeout_times = [], [], []
        success = collision = timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        _SUCCESS_TOKENS = {"reachgoal", "reach_goal", "success"}
        _COLLISION_TOKENS = {"collision"}
        _TIMEOUT_TOKENS = {"timeout"}

        store_on = tuple(s.lower() for s in store_on)

        rewards_hist = []

        iterator = range(k)
        if show_tqdm:
            tag = "IL" if imitation_learning else "RUN"
            iterator = tqdm(iterator, desc=f"{phase.upper()} {tag} | eps={k}",
                            dynamic_ncols=True, leave=True)

        for i in iterator:
            self.env.phase = phase
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                ob, _ = reset_out
            else:
                ob = reset_out

            terminated = False
            truncated = False
            states, actions, rewards = [], [], []
            info = {}

            while not (terminated or truncated):
                # JointState 或原始 obs
                use_joint = False
                if imitation_learning and force_joint_state_policy:
                    use_joint = True
                elif not imitation_learning:
                    policy_name = self.robot.policy.__class__.__name__.lower()
                    use_joint = any(key in policy_name for key in
                                    ['orca', 'cadrl', 'sarl', 'multi_human_rl', 'mamba'])

                if use_joint:
                    state = JointState(self.robot.get_full_state(),
                                       [h.get_observable_state() for h in self.env.humans])
                    action = self.robot.policy.predict(state)
                else:
                    state = ob
                    action = self.robot.policy.predict(ob)

                step_out = self.env.step(action)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    ob, reward, terminated, truncated, info = step_out
                elif isinstance(step_out, tuple) and len(step_out) == 4:
                    ob, reward, terminated, info = step_out
                    truncated = False
                else:
                    raise RuntimeError(f"Unexpected env.step() output length: {len(step_out)}")

                states.append(state)
                actions.append(action)
                rewards.append(float(reward))

                if str(info.get("event", "")).lower() == "danger":
                    too_close += 1
                    min_dist.append(float(info.get("min_dist", 0.0)))

            # 结束事件
            raw = str(info.get("event", "")).replace(" ", "_").lower().strip()
            is_time_limit = bool(truncated) or bool(info.get("TimeLimit.truncated", False))
            if raw in _SUCCESS_TOKENS:
                event = "success"
            elif raw in _COLLISION_TOKENS:
                event = "collision"
            elif raw in _TIMEOUT_TOKENS or is_time_limit:
                event = "timeout"
            else:
                event = "success"
                logging.warning('[Explorer] Unrecognized end signal: %s, terminated=%s, truncated=%s. Treat as success.',
                                info.get("event"), terminated, truncated)

            # 统计
            if event == "success":
                success += 1; success_times.append(self.env.global_time)
            elif event == "collision":
                collision += 1; collision_cases.append(i); collision_times.append(self.env.global_time)
            elif event == "timeout":
                timeout += 1; timeout_cases.append(i); timeout_times.append(self.env.time_limit)

            # 写经验
            if update_memory and (event in store_on):
                self.update_memory(states, actions, rewards)

            # 折扣回报
            if self.gamma is None:
                disc = rewards
            else:
                disc = [pow(self.gamma, t) * r for t, r in enumerate(rewards)]
            cumulative_rewards.append(sum(disc))
            rewards_hist.append(sum(rewards))

            if show_tqdm:
                n = i + 1
                iterator.set_postfix({
                    "r(mean50)": f"{np.mean(rewards_hist[-50:]):.2f}",
                    "succ": f"{success/max(1,n):.2f}",
                    "coll": f"{collision/max(1,n):.2f}",
                    "timeout": f"{timeout/max(1,n):.2f}",
                })

        # 指标与日志
        k = max(1, int(k))
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = max(0.0, 1.0 - success_rate - collision_rate)
        avg_nav_time = (sum(success_times) / len(success_times)) if len(success_times) else self.env.time_limit
        avg_reward = float(np.mean(cumulative_rewards)) if len(cumulative_rewards) else 0.0

        extra_info = '' if episode is None else f'in episode {episode} '
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'
                     .format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time, avg_reward))

        if phase in ['val', 'test']:
            # 近似步数统计危险频率
            step_count = 0
            robot_time_step = getattr(self.robot, "time_step", 0.0)
            if robot_time_step and robot_time_step > 0:
                all_times = success_times + collision_times + timeout_times
                step_count = int(sum(all_times) / max(1e-6, robot_time_step))
            step_count = max(1, step_count)
            freq_danger = too_close / step_count
            avg_min_sep = (sum(min_dist) / len(min_dist)) if len(min_dist) else 0.0
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         freq_danger, avg_min_sep)

        if print_failure:
            if collision_cases:
                logging.info('Collision cases: %s', ' '.join([str(x) for x in collision_cases]))
            if timeout_cases:
                logging.info('Timeout cases: %s', ' '.join([str(x) for x in timeout_cases]))

        if return_stats:
            return {
                "success_rate": float(success_rate),
                "collision_rate": float(collision_rate),
                "timeout_rate": float(timeout_rate),
                "nav_time": float(avg_nav_time),
                "total_reward": float(avg_reward),
            }

    def update_memory(self, states, actions, rewards):
        """
        存储标准五元组 (state, action, reward, next_state, done)
        """
        for i in range(len(states)):
            state_arr = self._state_to_array(states[i])
            action_arr = self._action_to_array(actions[i])
            reward = float(rewards[i])

            if i < len(states) - 1:
                next_state = states[i + 1]; done = False
            else:
                next_state = states[i]; done = True
            next_state_arr = self._state_to_array(next_state)

            self.memory.push(state_arr, action_arr, reward, next_state_arr, done)


def average(input_list):
    return sum(input_list) / len(input_list) if input_list else 0.0
