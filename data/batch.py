from dataclasses import dataclass, field, fields
from typing import Dict, Iterator, List, Union
try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python 3.10 兼容
import math
import numpy as np
import torch


@dataclass
class BaseBatch:
    """Base batch with common utilities."""
    
    def to_tensor(self, device: torch.device) -> Self:
        """Convert all numpy arrays to tensors on device."""
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif isinstance(val, np.ndarray):
                # 根据 dtype 选择合适的 torch dtype
                if val.dtype == np.bool_:
                    dtype = torch.bool
                elif val.dtype in [np.float32, np.float64]:
                    dtype = torch.float32
                else:
                    dtype = torch.long
                kwargs[f.name] = torch.as_tensor(val, dtype=dtype, device=device)
            elif isinstance(val, torch.Tensor):
                kwargs[f.name] = val.to(device)
            else:
                kwargs[f.name] = val
        return self.__class__(**kwargs)
    
    def __len__(self) -> int:
        """Return batch size (first dimension)."""
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None and hasattr(val, '__len__'):
                return len(val)
        return 0
    
    def __getitem__(self, indices) -> Self:
        """Index/slice the batch. Supports int, slice, or array indices."""
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif isinstance(val, (np.ndarray, torch.Tensor)):
                kwargs[f.name] = val[indices]
            else:
                kwargs[f.name] = val
        return self.__class__(**kwargs)
    
    def split(
        self,
        size: int,
        shuffle: bool = True,
        merge_last: bool = False,
    ) -> Iterator[Self]:
        """
        Split batch into minibatches.
        
        Args:
            size: minibatch size. -1 means no split (return whole batch).
            shuffle: randomly shuffle indices before splitting.
            merge_last: merge last small batch into previous one.
        
        Yields:
            Minibatch of the same type.
        """
        length = len(self)
        if length == 0:
            return
        
        if size == -1 or size >= length:
            yield self
            return
        
        # Generate indices
        indices = np.random.permutation(length) if shuffle else np.arange(length)
        
        # Check if we need to merge last batch
        merge_last = merge_last and (length % size > 0)
        
        for start in range(0, length, size):
            end = start + size
            # Merge last small batch
            if merge_last and end + size > length:
                yield self[indices[start:]]
                break
            yield self[indices[start:end]]


@dataclass
class RolloutBatch(BaseBatch):
    """Batch for on-policy algorithms (PPO, A2C, etc.)."""
    obs: torch.Tensor | np.ndarray = None          # (batch, *obs_shape)
    act: torch.Tensor | np.ndarray = None          # (batch,) or (batch, act_dim)
    rew: torch.Tensor | np.ndarray = None          # (batch,)
    done: torch.Tensor | np.ndarray = None         # (batch,) termination | truncation
    truncated: torch.Tensor | np.ndarray = None    # (batch,) 是否为时间截断（用于正确的 value bootstrap）
    log_prob: torch.Tensor | np.ndarray = None     # (batch,) old log_prob from collection
    value: torch.Tensor | np.ndarray = None        # (batch,) value estimate
    adv: torch.Tensor | np.ndarray = None          # (batch,) advantage
    ret: torch.Tensor | np.ndarray = None          # (batch,) return (adv + value)
    # For MARL centralized critic
    global_state: torch.Tensor | np.ndarray = None # (batch, global_state_dim)
    # For action masking
    action_mask: torch.Tensor | np.ndarray = None  # (batch, num_actions)
    # For active agent masking (1=READY 需要决策, 0=ON_EDGE 跳过)
    active_mask: torch.Tensor | np.ndarray = None  # (batch,)
    # For truncation value bootstrap (List[List[ndarray or None]]，不转为 tensor)
    final_global_state: list = None
    # Actor RNN hidden state at each step (batch, recurrent_N, hidden_size)。MLP 时为 None。
    rnn_hidden: torch.Tensor | np.ndarray = None
    # Critic RNN hidden state at each step (batch, recurrent_N, hidden_size)。MLP critic 时为 None。
    # 由 prepare_batch 在处理 RNN critic 序列时填充。
    critic_rnn_hidden: torch.Tensor | np.ndarray = None

    # -----------------------------------------------------------------
    #  chunk_split: RNN chunk-based minibatch splitting
    # -----------------------------------------------------------------

    def chunk_split(
        self,
        chunk_len: int,
        T: int,
        N: int,
        num_agents: int = 1,
        minibatch_size: int = -1,
    ) -> Iterator[Self]:
        """
        RNN chunk-based splitting。

        数据布局假设: flat batch = (M*T*N, ...) 其中 M=num_agents。

        步骤:
        1. reshape → (M, T, N, ...)
        2. T 轴 pad 到 chunk_len 的整数倍并切 chunk → (M, num_chunks, L, N, ...)
        3. (M, num_chunks) 展平为 S，对 S 维 shuffle
        4. 按 minibatch_size 切分（单位=序列条数，每条包含 L*N 步）
        5. yield 的 minibatch 布局: 各字段 (L, mb_S*N, ...) — 时间维在前

        rnn_hidden 特殊处理：只取每个 chunk 的首步 hidden 作为 h0，
        yield 时形状 (recurrent_N, mb_S*N, hidden_dim)。

        padding 保护：当 T 不是 chunk_len 整数倍时，填充步的 active_mask 为 0，
        若原始 active_mask 为 None 则自动合成。
        """
        M = num_agents
        L = chunk_len
        total = M * T * N

        num_chunks = math.ceil(T / L)
        T_padded = num_chunks * L

        # 当需要 padding 且 active_mask 不存在时，合成一个全 1 mask；
        # _reshape_and_chunk 的零填充会自动使 padding 位置变为 0。
        _val_overrides: dict = {}
        if T_padded > T and self.active_mask is None:
            ref = self.obs
            _val_overrides["active_mask"] = torch.ones(
                total, dtype=torch.float32, device=ref.device,
            )

        def _reshape_and_chunk(x: torch.Tensor, is_hidden: bool = False):
            """(total, *feat) -> (M, num_chunks, L, N, *feat)，hidden 只取首步"""
            feat = x.shape[1:]
            v = x.view(M, T, N, *feat)

            if T_padded > T:
                pad_shape = (M, T_padded - T, N, *feat)
                v = torch.cat([v, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=1)

            v = v.view(M, num_chunks, L, N, *feat)

            if is_hidden:
                return v[:, :, 0, :, :]
            return v

        S = M * num_chunks
        perm = torch.randperm(S)

        if minibatch_size <= 0 or minibatch_size >= S:
            minibatch_size = S

        for mb_start in range(0, S, minibatch_size):
            mb_end = min(mb_start + minibatch_size, S)
            mb_idx = perm[mb_start:mb_end]
            mb_S = len(mb_idx)

            kwargs = {}
            for f in fields(self):
                val = _val_overrides.get(f.name, getattr(self, f.name))
                if val is None:
                    kwargs[f.name] = None
                    continue

                if f.name == "final_global_state":
                    kwargs[f.name] = None
                    continue

                if not isinstance(val, torch.Tensor):
                    kwargs[f.name] = val
                    continue

                is_hidden = f.name in ("rnn_hidden", "critic_rnn_hidden")

                if is_hidden:
                    chunked = _reshape_and_chunk(val, is_hidden=True)
                    flat_s = chunked.reshape(S, N, *chunked.shape[3:])
                    selected = flat_s[mb_idx]
                    rN, H = selected.shape[-2], selected.shape[-1]
                    kwargs[f.name] = selected.reshape(mb_S * N, rN, H).permute(1, 0, 2).contiguous()
                else:
                    chunked = _reshape_and_chunk(val)
                    feat = chunked.shape[4:]
                    flat_s = chunked.reshape(S, L, N, *feat)
                    selected = flat_s[mb_idx]
                    selected = selected.reshape(mb_S, L, N, *feat)
                    selected = selected.permute(1, 0, 2, *range(3, 3 + len(feat))).contiguous()
                    selected = selected.reshape(L, mb_S * N, *feat)
                    kwargs[f.name] = selected

            yield self.__class__(**kwargs)


@dataclass 
class TransitionBatch(BaseBatch):
    """Batch for off-policy algorithms (DQN, SAC, etc.)."""
    obs: torch.Tensor | np.ndarray = None          # (batch, *obs_shape)
    act: torch.Tensor | np.ndarray = None          # (batch,) or (batch, act_dim)
    rew: torch.Tensor | np.ndarray = None          # (batch,)
    next_obs: torch.Tensor | np.ndarray = None     # (batch, *obs_shape)
    done: torch.Tensor | np.ndarray = None         # (batch,)
    # For action masking
    action_mask: torch.Tensor | np.ndarray = None  # (batch, num_actions)
    next_action_mask: torch.Tensor | np.ndarray = None
    # For CTDE algorithms (VDN, QMIX) — global state
    state: torch.Tensor | np.ndarray = None         # (batch, state_dim)
    next_state: torch.Tensor | np.ndarray = None    # (batch, state_dim)
    # For active agent masking (1=READY 决策步, 0=ON_EDGE 无效步)
    active_mask: torch.Tensor | np.ndarray = None   # (batch,)
    # 同步 transition 的 bootstrap 折扣 γ^k（sync_replay 模式）
    gamma_power: torch.Tensor | np.ndarray = None   # (batch,)


@dataclass
class SequenceBatch(BaseBatch):
    """Off-policy RNN 训练用序列 batch，所有字段含 seq_len 维度。

    L = burn_in_len + seq_len（total_len）。
    mask 语义: burn-in 区间=0, 训练有效步=1, padding=0。
    """
    obs: torch.Tensor | np.ndarray = None               # (B, L, obs_dim)
    act: torch.Tensor | np.ndarray = None               # (B, L)
    rew: torch.Tensor | np.ndarray = None               # (B, L)
    next_obs: torch.Tensor | np.ndarray = None          # (B, L, obs_dim)
    done: torch.Tensor | np.ndarray = None              # (B, L)
    mask: torch.Tensor | np.ndarray = None              # (B, L) — 有效步=1, padding=0
    action_mask: torch.Tensor | np.ndarray = None       # (B, L, act_dim)
    next_action_mask: torch.Tensor | np.ndarray = None  # (B, L, act_dim)
    active_mask: torch.Tensor | np.ndarray = None       # (B, L) 1=READY, 0=ON_EDGE
    burn_in_len: int = 0                                # burn-in 步数（标量元数据，不转 tensor）


@dataclass
class CollectResult:
    """采集结果，包含 batch 数据和统计信息"""
    batch: Union[RolloutBatch, Dict[str, RolloutBatch]]  # 采集的数据
    n_steps: int = 0                                      # 总步数
    n_episodes: int = 0                                   # 完成的 episode 数
    episode_rewards: List[float] = field(default_factory=list)  # 每个完成 episode 的总奖励
    episode_lengths: List[int] = field(default_factory=list)    # 每个完成 episode 的长度