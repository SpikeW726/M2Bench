"""数据缓冲区：用于存储 RL 训练数据"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from data.batch import TransitionBatch, SequenceBatch

class ReplayBuffer:
    """
    Off-policy 循环缓冲区

    用于 DQN, SAC 等 off-policy 算法。
    支持随机采样、action masking。
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        max_size: int,
        has_action_mask: bool = False,
        action_dim: Optional[int] = None,
        has_state: bool = False,
        state_dim: int = 0,
        has_active_mask: bool = False,
        has_gamma_power: bool = False,
    ):
        """
        Args:
            obs_shape: Shape of observation
            act_shape: Shape of action (for storing action values)
            max_size: Maximum buffer size
            has_action_mask: Whether to store action masks
            action_dim: Number of discrete actions (for action_mask shape).
                       Required if has_action_mask=True, since act_shape
                       is for action value storage, not number of actions.
            has_state: Whether to store global state (for CTDE: VDN/QMIX)
            state_dim: Dimension of global state. Required if has_state=True.
            has_active_mask: Whether to store active masks (1=READY, 0=ON_EDGE)
            has_gamma_power: Whether to store per-transition γ^k (sync_replay)
        """
        self.obs_shape = obs_shape
        self.act_shape = act_shape if act_shape else (1,)
        self.max_size = max_size
        self.has_action_mask = has_action_mask
        self.has_state = has_state
        self.has_active_mask = has_active_mask
        self.has_gamma_power = has_gamma_power

        # 预分配数组
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.act = np.zeros((max_size, *self.act_shape), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)

        # Action masking (for discrete action spaces)
        if self.has_action_mask:
            if action_dim is None:
                raise ValueError(
                    "action_dim must be specified when has_action_mask=True. "
                    "This is the number of discrete actions (e.g., 5 for a 5-action problem)."
                )
            self.action_dim = action_dim
            self.action_mask = np.zeros((max_size, action_dim), dtype=np.bool_)
            self.next_action_mask = np.zeros((max_size, action_dim), dtype=np.bool_)

        # Global state (for CTDE algorithms)
        if self.has_state:
            self.state = np.zeros((max_size, state_dim), dtype=np.float32)
            self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)

        # Active mask (1=READY 决策步, 0=ON_EDGE 无效步)
        if self.has_active_mask:
            self.active_mask = np.ones(max_size, dtype=np.float32)

        # γ^k per-transition bootstrap 折扣 (sync_replay 模式)
        if self.has_gamma_power:
            self.gamma_power = np.ones(max_size, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def peek_write_index(self) -> int:
        """下一次 add 将写入的槽位下标（循环数组）。"""
        return self._ptr

    def overwrite(
        self,
        idx: int,
        rew: float,
        next_obs: np.ndarray,
        gamma_power: float,
        active_mask: float,
        next_action_mask: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        done: Optional[float] = None,
    ) -> None:
        """回填 idx 槽位的 rew / next_obs / gamma_power / active_mask（shared_sync 延迟 finalize）。"""
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        if done is not None:
            self.done[idx] = float(done)
        if self.has_gamma_power:
            self.gamma_power[idx] = gamma_power
        if self.has_active_mask:
            self.active_mask[idx] = active_mask
        if self.has_action_mask:
            if next_action_mask is not None:
                self.next_action_mask[idx] = next_action_mask
            else:
                self.next_action_mask[idx] = True
        if self.has_state and next_state is not None:
            self.next_state[idx] = next_state

    def add(
        self,
        obs: np.ndarray,
        act: Union[int, float, np.ndarray],
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        active_mask: Optional[float] = None,
        gamma_power: Optional[float] = None,
    ):
        """
        添加单个 transition（循环覆盖）

        Args:
            obs: observation
            act: action (int for discrete, array for continuous)
            rew: reward
            next_obs: next observation
            done: terminal flag
            action_mask: boolean mask for valid actions at current step
            next_action_mask: boolean mask for valid actions at next step
            state: global state (for CTDE algorithms)
            next_state: next global state (for CTDE algorithms)
            active_mask: 1=READY (valid decision), 0=ON_EDGE (no-op)
            gamma_power: γ^k bootstrap 折扣 (sync_replay 模式)
        """
        self.obs[self._ptr] = obs
        self.act[self._ptr] = act
        self.rew[self._ptr] = rew
        self.next_obs[self._ptr] = next_obs
        self.done[self._ptr] = float(done)

        if self.has_action_mask:
            if action_mask is not None:
                self.action_mask[self._ptr] = action_mask
            else:
                self.action_mask[self._ptr] = True

            if next_action_mask is not None:
                self.next_action_mask[self._ptr] = next_action_mask
            else:
                self.next_action_mask[self._ptr] = True

        if self.has_state:
            if state is not None:
                self.state[self._ptr] = state
            if next_state is not None:
                self.next_state[self._ptr] = next_state

        if self.has_active_mask:
            self.active_mask[self._ptr] = active_mask if active_mask is not None else 1.0

        if self.has_gamma_power:
            self.gamma_power[self._ptr] = gamma_power if gamma_power is not None else 1.0

        self._ptr = (self._ptr + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def _build_batch(self, indices: np.ndarray) -> TransitionBatch:
        """从给定 indices 构建 TransitionBatch（内部共用逻辑）。"""
        act = self.act[indices]
        if self.act_shape == (1,):
            act = act.squeeze(-1)

        result = TransitionBatch(
            obs=self.obs[indices].copy(),
            act=act.copy(),
            rew=self.rew[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            done=self.done[indices].copy(),
        )

        if self.has_action_mask:
            result.action_mask = self.action_mask[indices].copy()
            result.next_action_mask = self.next_action_mask[indices].copy()

        if self.has_state:
            result.state = self.state[indices].copy()
            result.next_state = self.next_state[indices].copy()

        if self.has_active_mask:
            result.active_mask = self.active_mask[indices].copy()

        if self.has_gamma_power:
            result.gamma_power = self.gamma_power[indices].copy()

        return result

    def sample(self, batch_size: int) -> TransitionBatch:
        """随机采样"""
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) > buffer size ({self._size})")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return self._build_batch(indices)

    def sample_by_indices(self, indices: np.ndarray) -> TransitionBatch:
        """用外部索引采样（支持多 buffer 共享索引对齐）。"""
        return self._build_batch(indices)

    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        gamma_power: Optional[np.ndarray] = None,
    ):
        """
        批量添加 transitions（向量化写入，避免逐条 Python 循环）。

        所有数组的第 0 维为 batch_size。
        """
        batch_size = len(obs)
        if batch_size == 0:
            return

        act_arr = np.asarray(act, dtype=np.float32)
        if act_arr.ndim == 1 and self.act_shape != (1,):
            pass  # act 本身就是 1-D (e.g. discrete scalar per env)
        if act_arr.ndim == 1:
            act_arr = act_arr.reshape(-1, 1)

        end = self._ptr + batch_size
        if end <= self.max_size:
            s = slice(self._ptr, end)
            self.obs[s] = obs
            self.act[s] = act_arr
            self.rew[s] = rew
            self.next_obs[s] = next_obs
            self.done[s] = done
            if self.has_action_mask:
                if action_mask is not None:
                    self.action_mask[s] = action_mask
                else:
                    self.action_mask[s] = True
                if next_action_mask is not None:
                    self.next_action_mask[s] = next_action_mask
                else:
                    self.next_action_mask[s] = True
            if self.has_state:
                if state is not None:
                    self.state[s] = state
                if next_state is not None:
                    self.next_state[s] = next_state
            if self.has_active_mask:
                self.active_mask[s] = active_mask if active_mask is not None else 1.0
            if self.has_gamma_power:
                self.gamma_power[s] = gamma_power if gamma_power is not None else 1.0
        else:
            first = self.max_size - self._ptr
            second = batch_size - first

            s1 = slice(self._ptr, self.max_size)
            s2 = slice(0, second)

            self.obs[s1] = obs[:first]
            self.obs[s2] = obs[first:]
            self.act[s1] = act_arr[:first]
            self.act[s2] = act_arr[first:]
            self.rew[s1] = rew[:first]
            self.rew[s2] = rew[first:]
            self.next_obs[s1] = next_obs[:first]
            self.next_obs[s2] = next_obs[first:]
            self.done[s1] = done[:first]
            self.done[s2] = done[first:]

            if self.has_action_mask:
                if action_mask is not None:
                    self.action_mask[s1] = action_mask[:first]
                    self.action_mask[s2] = action_mask[first:]
                else:
                    self.action_mask[s1] = True
                    self.action_mask[s2] = True
                if next_action_mask is not None:
                    self.next_action_mask[s1] = next_action_mask[:first]
                    self.next_action_mask[s2] = next_action_mask[first:]
                else:
                    self.next_action_mask[s1] = True
                    self.next_action_mask[s2] = True
            if self.has_state:
                if state is not None:
                    self.state[s1] = state[:first]
                    self.state[s2] = state[first:]
                if next_state is not None:
                    self.next_state[s1] = next_state[:first]
                    self.next_state[s2] = next_state[first:]
            if self.has_active_mask:
                if active_mask is not None:
                    self.active_mask[s1] = active_mask[:first]
                    self.active_mask[s2] = active_mask[first:]
                else:
                    self.active_mask[s1] = 1.0
                    self.active_mask[s2] = 1.0
            if self.has_gamma_power:
                if gamma_power is not None:
                    self.gamma_power[s1] = gamma_power[:first]
                    self.gamma_power[s2] = gamma_power[first:]
                else:
                    self.gamma_power[s1] = 1.0
                    self.gamma_power[s2] = 1.0

        self._ptr = end % self.max_size
        self._size = min(self._size + batch_size, self.max_size)

    def __len__(self) -> int:
        return self._size


class SequenceReplayBuffer:
    """R2D2 风格序列 replay buffer，用于 RNN off-policy 训练。

    episode 存入时预切为固定长度 total_len = burn_in_len + seq_len 的重叠序列
    （stride = seq_len），存入扁平循环数组。采样时纯 numpy fancy indexing，零循环。

    mask 语义:
        - burn-in 区间 [:burn_in_len] 始终为 0（参与 RNN forward 但不计 loss）
        - 训练区间 [burn_in_len:] 有效步为 1，padding 为 0
    """

    def __init__(
        self,
        obs_dim: int,
        seq_len: int,
        burn_in_len: int = 0,
        max_seqs: int = 50_000,
        has_action_mask: bool = False,
        action_dim: int = 0,
        has_active_mask: bool = False,
        has_state: bool = False,
        state_dim: int = 0,
    ):
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.burn_in_len = burn_in_len
        self.total_len = burn_in_len + seq_len
        self.max_seqs = max_seqs
        self.has_action_mask = has_action_mask
        self.has_active_mask = has_active_mask
        self.has_state = has_state
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 预分配扁平循环数组
        T = self.total_len
        self.obs = np.zeros((max_seqs, T, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_seqs, T), dtype=np.float32)
        self.rew = np.zeros((max_seqs, T), dtype=np.float32)
        self.next_obs = np.zeros((max_seqs, T, obs_dim), dtype=np.float32)
        self.done = np.zeros((max_seqs, T), dtype=np.float32)
        self.mask = np.zeros((max_seqs, T), dtype=np.float32)

        if has_action_mask:
            self.action_mask_buf = np.zeros((max_seqs, T, action_dim), dtype=np.bool_)
            self.next_action_mask_buf = np.zeros((max_seqs, T, action_dim), dtype=np.bool_)

        if has_active_mask:
            self.active_mask_buf = np.zeros((max_seqs, T), dtype=np.float32)

        if has_state:
            self.state_buf = np.zeros((max_seqs, T, state_dim), dtype=np.float32)
            self.next_state_buf = np.zeros((max_seqs, T, state_dim), dtype=np.float32)

        self._ptr = 0
        self._size = 0
        self._total_steps = 0

    def add_episode(self, episode: dict):
        """将 episode 预切为重叠序列写入循环数组。

        切片方式 (R2D2 stride = seq_len):
            训练区间起点: [0, seq_len, 2*seq_len, ...]
            每条序列: episode[start - burn_in_len : start + seq_len]
            左侧不足时 zero-pad (mask=0)，右侧不足时 zero-pad (mask=0)。
        """
        ep_len = len(episode["obs"])
        if ep_len == 0:
            return

        has_am = "action_mask" in episode
        has_actm = "active_mask" in episode
        s = self.seq_len
        b = self.burn_in_len
        T = self.total_len

        num_seqs = max(1, math.ceil(ep_len / s))

        for k in range(num_seqs):
            train_start = k * s
            src_start = train_start - b  # 可能为负（burn-in 超出 episode 左边界）

            # 源数据的有效范围（裁剪到 [0, ep_len]）
            src_lo = max(0, src_start)
            src_hi = min(ep_len, train_start + s)
            src_data_len = src_hi - src_lo

            # 目标数组内的偏移
            dst_offset = src_lo - src_start  # burn-in 左侧 padding 步数

            has_st = "state" in episode and self.has_state

            # 写入一条序列到循环数组
            idx = self._ptr
            self.obs[idx] = 0
            self.act[idx] = 0
            self.rew[idx] = 0
            self.next_obs[idx] = 0
            self.done[idx] = 0
            self.mask[idx] = 0
            if has_am and self.has_action_mask:
                self.action_mask_buf[idx] = False
                self.next_action_mask_buf[idx] = False
            if has_actm and self.has_active_mask:
                self.active_mask_buf[idx] = 0
            if has_st:
                self.state_buf[idx] = 0
                self.next_state_buf[idx] = 0

            self.obs[idx, dst_offset:dst_offset + src_data_len] = episode["obs"][src_lo:src_hi]
            self.act[idx, dst_offset:dst_offset + src_data_len] = episode["act"][src_lo:src_hi]
            self.rew[idx, dst_offset:dst_offset + src_data_len] = episode["rew"][src_lo:src_hi]
            self.next_obs[idx, dst_offset:dst_offset + src_data_len] = episode["next_obs"][src_lo:src_hi]
            self.done[idx, dst_offset:dst_offset + src_data_len] = episode["done"][src_lo:src_hi]

            if has_am and self.has_action_mask:
                self.action_mask_buf[idx, dst_offset:dst_offset + src_data_len] = episode["action_mask"][src_lo:src_hi]
                self.next_action_mask_buf[idx, dst_offset:dst_offset + src_data_len] = episode["next_action_mask"][src_lo:src_hi]

            if has_actm and self.has_active_mask:
                self.active_mask_buf[idx, dst_offset:dst_offset + src_data_len] = episode["active_mask"][src_lo:src_hi]

            if has_st:
                self.state_buf[idx, dst_offset:dst_offset + src_data_len] = episode["state"][src_lo:src_hi]
                self.next_state_buf[idx, dst_offset:dst_offset + src_data_len] = episode["next_state"][src_lo:src_hi]

            # mask: burn-in 区间=0, 训练区间有效步=1, padding=0
            train_data_start = max(0, b - dst_offset)  # 训练数据在 src 中的起始偏移
            train_valid_lo = b  # 训练区间在 total_len 中的起始位置
            train_valid_hi = min(T, dst_offset + src_data_len)
            if train_valid_hi > train_valid_lo:
                self.mask[idx, train_valid_lo:train_valid_hi] = 1.0

            self._ptr = (self._ptr + 1) % self.max_seqs
            self._size = min(self._size + 1, self.max_seqs)

        self._total_steps += ep_len

    def _build_seq_batch(self, indices: np.ndarray) -> SequenceBatch:
        """按给定 indices 构建 SequenceBatch（内部共用逻辑）。"""
        result = SequenceBatch(
            obs=self.obs[indices].copy(),
            act=self.act[indices].copy(),
            rew=self.rew[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            done=self.done[indices].copy(),
            mask=self.mask[indices].copy(),
            burn_in_len=self.burn_in_len,
        )
        if self.has_action_mask:
            result.action_mask = self.action_mask_buf[indices].copy()
            result.next_action_mask = self.next_action_mask_buf[indices].copy()
        if self.has_active_mask:
            result.active_mask = self.active_mask_buf[indices].copy()
        if self.has_state:
            result.state = self.state_buf[indices].copy()
            result.next_state = self.next_state_buf[indices].copy()
        return result

    def sample(self, batch_size: int) -> SequenceBatch:
        """随机采样，零 Python 循环。"""
        if self._size == 0:
            raise ValueError("SequenceReplayBuffer is empty")
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) > buffer size ({self._size})")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return self._build_seq_batch(indices)

    def sample_by_indices(self, indices: np.ndarray) -> SequenceBatch:
        """按外部指定 indices 采样（供 shared_indices 模式使用）。"""
        return self._build_seq_batch(indices)

    def __len__(self) -> int:
        return self._size

    @property
    def total_steps(self) -> int:
        return self._total_steps


class BufferManager:
    """
    统一管理单/多智能体的buffer操作。

    支持两种模式:
    - 单智能体: buffer 是 ReplayBuffer，返回 TransitionBatch
    - 多智能体: buffer 是 Dict[str, ReplayBuffer]，返回 Dict[str, TransitionBatch]

    使用方式:
        # 单智能体
        single_buffer = ReplayBuffer(...)
        mgr = BufferManager(single_buffer)

        # 多智能体 (IQL)
        multi_buffer = {aid: ReplayBuffer(...) for aid in agent_ids}
        mgr = BufferManager(multi_buffer)
    """

    def __init__(
        self,
        buffer: Union[ReplayBuffer, Dict[str, ReplayBuffer]],
    ):
        """
        Args:
            buffer: 单个 ReplayBuffer 或 {agent_id: ReplayBuffer} 字典
        """
        self.buffer = buffer
        self.is_multi = isinstance(buffer, dict)

        if self.is_multi:
            self.agent_ids = list(buffer.keys())
        else:
            self.agent_ids = None

    def add(
        self,
        batch: Union[TransitionBatch, Dict[str, TransitionBatch]],
    ) -> None:
        """
        添加数据到 buffer。

        Args:
            batch: 单个 TransitionBatch 或 {agent_id: TransitionBatch}
        """
        if self.is_multi:
            # 多智能体: 每个agent的batch添加到对应的buffer
            if not isinstance(batch, dict):
                raise ValueError(
                    f"Expected Dict[str, TransitionBatch] for multi-agent buffer, "
                    f"got {type(batch)}"
                )
            for agent_id, agent_batch in batch.items():
                if agent_id not in self.buffer:
                    continue  # 跳过没有buffer的agent
                self._add_to_single_buffer(self.buffer[agent_id], agent_batch)
        else:
            # 单智能体: 直接添加
            if not isinstance(batch, TransitionBatch):
                raise ValueError(
                    f"Expected TransitionBatch for single-agent buffer, "
                    f"got {type(batch)}"
                )
            self._add_to_single_buffer(self.buffer, batch)

    def _add_to_single_buffer(
        self,
        buffer: ReplayBuffer,
        batch: TransitionBatch,
    ) -> None:
        """添加单个batch到单个buffer"""
        batch_size = len(batch)
        for i in range(batch_size):
            obs = batch.obs[i]
            act = batch.act[i]
            rew = batch.rew[i]
            next_obs = batch.next_obs[i]
            done = batch.done[i]

            action_mask = getattr(batch, 'action_mask', None)
            next_action_mask = getattr(batch, 'next_action_mask', None)

            if action_mask is not None:
                am = action_mask[i]
                nam = next_action_mask[i] if next_action_mask is not None else None
            else:
                am = None
                nam = None

            buffer.add(obs, act, rew, next_obs, done, am, nam)

    def sample(
        self,
        batch_size: int,
        agent_ids: Optional[list] = None,
    ) -> Union[TransitionBatch, Dict[str, TransitionBatch]]:
        """
        从buffer采样。

        Args:
            batch_size: 采样数量
            agent_ids: 指定采样的agents（多智能体模式）

        Returns:
            TransitionBatch 或 Dict[str, TransitionBatch]
        """
        if self.is_multi:
            # 多智能体: 从每个buffer采样
            target_ids = agent_ids or self.agent_ids
            return {
                aid: self.buffer[aid].sample(batch_size)
                for aid in target_ids
                if aid in self.buffer
            }
        else:
            # 单智能体
            return self.buffer.sample(batch_size)

    def __len__(self) -> int:
        """返回buffer大小（多智能体时返回最小值）"""
        if self.is_multi:
            if not self.buffer:
                return 0
            return min(len(v) for v in self.buffer.values()) if self.buffer else 0
        return len(self.buffer)

    def can_sample(self, batch_size: int, agent_ids: Optional[list] = None) -> bool:
        """检查是否可以采样指定数量的数据"""
        if self.is_multi:
            target_ids = agent_ids or self.agent_ids
            return all(
                len(self.buffer[aid]) >= batch_size
                for aid in target_ids
                if aid in self.buffer
            )
        return len(self.buffer) >= batch_size

    def get_agent_ids(self) -> Optional[list]:
        """获取agent IDs（多智能体模式）"""
        return self.agent_ids
