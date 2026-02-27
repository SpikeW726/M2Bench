"""数据缓冲区：用于存储 RL 训练数据"""

from collections import deque
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
        """
        self.obs_shape = obs_shape
        self.act_shape = act_shape if act_shape else (1,)
        self.max_size = max_size
        self.has_action_mask = has_action_mask
        self.has_state = has_state
        self.has_active_mask = has_active_mask

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

        self._ptr = 0
        self._size = 0

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

        self._ptr = end % self.max_size
        self._size = min(self._size + batch_size, self.max_size)

    def __len__(self) -> int:
        return self._size


class EpisodeReplayBuffer:
    """
    按 episode 存储的 replay buffer，用于 RNN off-policy 训练。

    存储完整 episode，采样时切 seq_len 长度的子序列并 zero-pad。
    """

    def __init__(self, max_episodes: int):
        self._episodes: deque = deque(maxlen=max_episodes)
        self._total_steps = 0

    def add_episode(self, episode: dict):
        """添加一条 episode。

        episode 字段: obs, act, rew, next_obs, done，shape=(ep_len, ...)。
        可选字段: action_mask, next_action_mask。
        """
        ep_len = len(episode["obs"])
        if ep_len == 0:
            return
        if len(self._episodes) == self._episodes.maxlen:
            self._total_steps -= len(self._episodes[0]["obs"])
        self._episodes.append(episode)
        self._total_steps += ep_len

    def sample(self, batch_size: int, seq_len: int) -> SequenceBatch:
        """随机选 episode + 随机起始位置，切 seq_len 子序列，短序列 zero-pad + mask=0。"""
        n_episodes = len(self._episodes)
        if n_episodes == 0:
            raise ValueError("EpisodeReplayBuffer is empty")

        ep_indices = np.random.randint(0, n_episodes, size=batch_size)

        # 预分配 numpy 数组
        sample_ep = self._episodes[ep_indices[0]]
        obs_shape = sample_ep["obs"].shape[1:]
        has_action_mask = "action_mask" in sample_ep

        all_obs = np.zeros((batch_size, seq_len, *obs_shape), dtype=np.float32)
        all_next_obs = np.zeros_like(all_obs)
        all_act = np.zeros((batch_size, seq_len), dtype=np.float32)
        all_rew = np.zeros((batch_size, seq_len), dtype=np.float32)
        all_done = np.zeros((batch_size, seq_len), dtype=np.float32)
        all_mask = np.zeros((batch_size, seq_len), dtype=np.float32)

        if has_action_mask:
            act_dim = sample_ep["action_mask"].shape[1:]
            all_am = np.zeros((batch_size, seq_len, *act_dim), dtype=np.bool_)
            all_nam = np.zeros_like(all_am)
        else:
            all_am = None
            all_nam = None

        for i, ep_idx in enumerate(ep_indices):
            ep = self._episodes[ep_idx]
            ep_len = len(ep["obs"])

            # 随机选起始位置
            max_start = max(0, ep_len - seq_len)
            start = np.random.randint(0, max_start + 1)
            end = min(start + seq_len, ep_len)
            actual_len = end - start

            all_obs[i, :actual_len] = ep["obs"][start:end]
            all_next_obs[i, :actual_len] = ep["next_obs"][start:end]
            all_act[i, :actual_len] = ep["act"][start:end]
            all_rew[i, :actual_len] = ep["rew"][start:end]
            all_done[i, :actual_len] = ep["done"][start:end]
            all_mask[i, :actual_len] = 1.0

            if has_action_mask and all_am is not None:
                all_am[i, :actual_len] = ep["action_mask"][start:end]
                all_nam[i, :actual_len] = ep["next_action_mask"][start:end]

        return SequenceBatch(
            obs=all_obs,
            act=all_act,
            rew=all_rew,
            next_obs=all_next_obs,
            done=all_done,
            mask=all_mask,
            action_mask=all_am,
            next_action_mask=all_nam,
        )

    def __len__(self) -> int:
        return len(self._episodes)

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
