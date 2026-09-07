    def sample(self, batch_size=256, seq_len=16, device='cpu', il_ratio=0.0):
        """
        【回归简单】均匀随机采样RL数据（SARL范式）
        il_ratio已废弃，保留参数兼容性
        """
        if seq_len is None:
            seq_len = self.sequence_length
        if device is None:
            device = torch.device('cpu')

        assert len(self.episodes) > 0, "Empty replay memory."

        import logging
        b = int(batch_size)

        # 【回归简单】只采样RL数据，均匀随机（SARL范式）
        rl_episodes = [i for i, ep in enumerate(self.episodes) if ep.get('meta', {}).get('source', 'RL') == 'RL']

        if len(rl_episodes) == 0:
            # 如果没有RL数据，全部采样（包括IL）
            logging.warning("[UNIFORM-SAMPLE] No RL episodes, sampling from all episodes")
            chosen = self._pick(list(range(len(self.episodes))), b)
        else:
            # 均匀随机采样RL episodes
            chosen = self._pick(rl_episodes, b)

        logging.debug(f"[UNIFORM-SAMPLE] Sampled {len(chosen)}/{b} episodes from {len(rl_episodes)} RL episodes")

        # 确保至少有1个样本
        if len(chosen) == 0:
            raise RuntimeError(f"Cannot sample from empty buffer. Total episodes: {len(self.episodes)}")

        # 准备数据数组
        any_ep = self.episodes[chosen[0]]
        token_shape = tuple(any_ep["tokens"].shape[1:])  # (1,6,13)
        states  = np.zeros((b, seq_len) + token_shape, dtype=np.float32)
        nstates = np.zeros((b, seq_len) + token_shape, dtype=np.float32)
        rewards = np.zeros((b, seq_len), dtype=np.float32)
        dones   = np.zeros((b, seq_len), dtype=np.bool_)
        timeouts= np.zeros((b, seq_len), dtype=np.bool_)
        mask    = np.zeros((b, seq_len), dtype=np.float32)
        actions_continuous = np.zeros((b, seq_len, 2), dtype=np.float32)

        # 统计分布（简化版，只统计采样总数）
        chosen_unique = list(set(chosen))
        actual_count = len(chosen_unique)

        # 如果chosen有重复或不足batch_size，记录警告
        if len(chosen_unique) < len(chosen):
            logging.warning(f"[SAMPLE-DEDUP] chosen had {len(chosen)-len(chosen_unique)} duplicates")
        if len(chosen) < b:
            logging.warning(f"[SAMPLE-SHORTAGE] only sampled {len(chosen)}/{b}, buffer may be too small")

        # 对每个选中的episode切片
        for bi, epi in enumerate(chosen):
            ep = self.episodes[epi]
            T = ep["length"]
            terminal = bool(ep["dones"][-1])

            start = self._choose_start(T, seq_len, terminal)
            end   = min(start + seq_len, T)
            K     = end - start
            if K <= 0:
                continue

            # 使用缓存的tokens
            states[bi, :K]   = ep["tokens"][start:end]
            rewards[bi, :K]  = ep["rewards"][start:end]
            dones[bi, :K]    = ep["dones"][start:end]
            timeouts[bi, :K] = ep["timeouts"][start:end]
            mask[bi, :K]     = 1.0

            # 连续动作
            if 'actions_continuous' in ep:
                actions_continuous[bi, :K] = ep['actions_continuous'][start:end]

            # Next states
            if K >= 2:
                nstates[bi, :K-1] = ep["tokens"][start+1:end]
                nstates[bi, K-1]  = ep["tokens"][end-1] if end == T else ep["tokens"][end]
            else:
                nstates[bi, 0] = ep["tokens"][start]

        return dict(
            states      = torch.as_tensor(states).to(device, non_blocking=True),
            next_states = torch.as_tensor(nstates).to(device, non_blocking=True),
            rewards     = torch.as_tensor(rewards).to(device, non_blocking=True),
            dones       = torch.as_tensor(dones).to(device, non_blocking=True),
            timeouts    = torch.as_tensor(timeouts).to(device, non_blocking=True),
            mask        = torch.as_tensor(mask).to(device, non_blocking=True),
            actions_continuous = torch.as_tensor(actions_continuous).to(device, non_blocking=True),
            mix_info    = dict(
                actual_batch_size=len(chosen),  # 实际采样数（可能<b）
            )
        )
