"""Replay buffers for transition and recurrent-sequence training data."""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from data.batch import TransitionBatch, SequenceBatch

class ReplayBuffer:
    """Preallocated circular buffer for off-policy transitions.

    Optional arrays store discrete-action masks, centralized state, active-agent
    masks, and per-transition bootstrap discounts used by synchronized replay.
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
            has_gamma_power: Whether to store per-transition gamma**k (sync_replay)
        """
        self.obs_shape = obs_shape
        self.act_shape = act_shape if act_shape else (1,)
        self.max_size = max_size
        self.has_action_mask = has_action_mask
        self.has_state = has_state
        self.has_active_mask = has_active_mask
        self.has_gamma_power = has_gamma_power

        self.obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.act = np.zeros((max_size, *self.act_shape), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)

        # Action masking (for discrete action spaces).
        if self.has_action_mask:
            if action_dim is None:
                raise ValueError(
                    "action_dim must be specified when has_action_mask=True. "
                    "This is the number of discrete actions (e.g., 5 for a 5-action problem)."
                )
            self.action_dim = action_dim
            self.action_mask = np.zeros((max_size, action_dim), dtype=np.bool_)
            self.next_action_mask = np.zeros((max_size, action_dim), dtype=np.bool_)

        # Global state (for CTDE algorithms).
        if self.has_state:
            self.state = np.zeros((max_size, state_dim), dtype=np.float32)
            self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)

        if self.has_active_mask:
            self.active_mask = np.ones(max_size, dtype=np.float32)

        # gamma^k per-transition bootstrap.
        if self.has_gamma_power:
            self.gamma_power = np.ones(max_size, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def peek_write_index(self) -> int:
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
        """Finalize a previously reserved transition for shared synchronization."""

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
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) > buffer size ({self._size})")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return self._build_batch(indices)

    def sample_by_indices(self, indices: np.ndarray) -> TransitionBatch:
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
        batch_size = len(obs)
        if batch_size == 0:
            return

        act_arr = np.asarray(act, dtype=np.float32)
        if act_arr.ndim == 1 and self.act_shape != (1,):
            pass
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
    """Circular buffer of fixed-length recurrent training sequences.

    Each episode is divided into ``seq_len`` training windows preceded by up to
    ``burn_in_len`` context steps. Missing context and tail steps are zero-padded;
    only valid optimization steps are enabled in ``mask``.
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
        """Split and add one complete episode with burn-in and padding masks."""

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
            src_start = train_start - b

            src_lo = max(0, src_start)
            src_hi = min(ep_len, train_start + s)
            src_data_len = src_hi - src_lo

            dst_offset = src_lo - src_start

            has_st = "state" in episode and self.has_state

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

            train_data_start = max(0, b - dst_offset)
            train_valid_lo = b
            train_valid_hi = min(T, dst_offset + src_data_len)
            if train_valid_hi > train_valid_lo:
                self.mask[idx, train_valid_lo:train_valid_hi] = 1.0

            self._ptr = (self._ptr + 1) % self.max_seqs
            self._size = min(self._size + 1, self.max_seqs)

        self._total_steps += ep_len

    def _build_seq_batch(self, indices: np.ndarray) -> SequenceBatch:
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
        if self._size == 0:
            raise ValueError("SequenceReplayBuffer is empty")
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) > buffer size ({self._size})")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return self._build_seq_batch(indices)

    def sample_by_indices(self, indices: np.ndarray) -> SequenceBatch:
        return self._build_seq_batch(indices)

    def __len__(self) -> int:
        return self._size

    @property
    def total_steps(self) -> int:
        return self._total_steps

class BufferManager:
    """Present one replay buffer or per-agent buffers through one interface.

    Sampling from per-agent buffers is independent; the manager reports the
    smallest available size so callers know when every requested agent is ready.
    """

    def __init__(
        self,
        buffer: Union[ReplayBuffer, Dict[str, ReplayBuffer]],
    ):
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
        if self.is_multi:

            if not isinstance(batch, dict):
                raise ValueError(
                    f"Expected Dict[str, TransitionBatch] for multi-agent buffer, "
                    f"got {type(batch)}"
                )
            for agent_id, agent_batch in batch.items():
                if agent_id not in self.buffer:
                    continue
                self._add_to_single_buffer(self.buffer[agent_id], agent_batch)
        else:

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
        if self.is_multi:

            target_ids = agent_ids or self.agent_ids
            return {
                aid: self.buffer[aid].sample(batch_size)
                for aid in target_ids
                if aid in self.buffer
            }
        else:

            return self.buffer.sample(batch_size)

    def __len__(self) -> int:
        if self.is_multi:
            if not self.buffer:
                return 0
            return min(len(v) for v in self.buffer.values()) if self.buffer else 0
        return len(self.buffer)

    def can_sample(self, batch_size: int, agent_ids: Optional[list] = None) -> bool:
        if self.is_multi:
            target_ids = agent_ids or self.agent_ids
            return all(
                len(self.buffer[aid]) >= batch_size
                for aid in target_ids
                if aid in self.buffer
            )
        return len(self.buffer) >= batch_size

    def get_agent_ids(self) -> Optional[list]:
        return self.agent_ids
