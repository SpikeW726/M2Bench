from dataclasses import dataclass, field, fields
from typing import Dict, Iterator, List, Union
try:
    from typing import Self  # Python 3.11+.
except ImportError:
    from typing_extensions import Self  # Python 3.10 compatibility.
import math
import numpy as np
import torch

@dataclass
class BaseBatch:
    """Base batch with common utilities."""

    def to_tensor(self, device: torch.device) -> Self:
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif isinstance(val, np.ndarray):
                # Select the matching PyTorch dtype.
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

        # Generate indices on the same device as tensor batches to avoid CPU index hops.
        index_device = None
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                index_device = val.device
                break
        if index_device is not None:
            indices = (
                torch.randperm(length, device=index_device)
                if shuffle
                else torch.arange(length, device=index_device)
            )
        else:
            indices = np.random.permutation(length) if shuffle else np.arange(length)

        # Check if we need to merge last batch.
        merge_last = merge_last and (length % size > 0)

        for start in range(0, length, size):
            end = start + size
            # Merge last small batch.
            if merge_last and end + size > length:
                yield self[indices[start:]]
                break
            yield self[indices[start:end]]

@dataclass
class RolloutBatch(BaseBatch):
    """Batch for on-policy algorithms (PPO, A2C, etc.)."""
    obs: torch.Tensor | np.ndarray = None          # (batch, *obs_shape).
    act: torch.Tensor | np.ndarray = None          # (batch,) or (batch, act_dim).
    rew: torch.Tensor | np.ndarray = None          # (batch,).
    done: torch.Tensor | np.ndarray = None         # (batch,) termination | truncation.
    truncated: torch.Tensor | np.ndarray = None
    log_prob: torch.Tensor | np.ndarray = None     # (batch,) old log_prob from collection.
    value: torch.Tensor | np.ndarray = None        # (batch,) value estimate.
    adv: torch.Tensor | np.ndarray = None          # (batch,) advantage.
    ret: torch.Tensor | np.ndarray = None          # (batch,) return (adv + value).

    global_state: torch.Tensor | np.ndarray = None # (batch, global_state_dim).

    action_mask: torch.Tensor | np.ndarray = None  # (batch, num_actions).
    # For active agent masking.
    active_mask: torch.Tensor | np.ndarray = None  # (batch,).
    # For truncation value bootstrap.
    final_global_state: list = None

    final_obs: list = None

    boundary_global_state: np.ndarray = None
    # Actor RNN hidden state at each step (batch, recurrent_N, hidden_size); None for MLP policies.
    rnn_hidden: torch.Tensor | np.ndarray = None
    # Critic RNN hidden state at each step (batch, recurrent_N, hidden_size); None for MLP critics.
    # Filled by prepare_batch when processing critic RNN sequences.
    critic_rnn_hidden: torch.Tensor | np.ndarray = None

    def __getitem__(self, indices) -> Self:
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name == "boundary_global_state":
                kwargs[f.name] = val
            elif isinstance(val, (np.ndarray, torch.Tensor)):
                kwargs[f.name] = val[indices]
            else:
                kwargs[f.name] = val
        return self.__class__(**kwargs)

    # chunk_split: RNN chunk-based minibatch splitting.

    def chunk_split(
        self,
        chunk_len: int,
        T: int,
        N: int,
        num_agents: int = 1,
        minibatch_size: int = -1,
    ) -> Iterator[Self]:
        """Split a flattened rollout into shuffled recurrent minibatches.

        Input layout is ``(M * T * N, ...)``, where ``M`` is the number of
        agents and ``N`` the number of environments. Time is padded to a multiple
        of ``chunk_len`` and split into sequences. Yielded tensors use layout
        ``(L, minibatch_sequences * N, ...)``. Recurrent states contain only each
        chunk's initial state with shape ``(recurrent_N, minibatch_sequences * N,
        hidden_dim)``. Padded steps receive a zero ``active_mask``.
        """

        M = num_agents
        L = chunk_len
        total = M * T * N

        num_chunks = math.ceil(T / L)
        T_padded = num_chunks * L

        _val_overrides: dict = {}
        if T_padded > T and self.active_mask is None:
            ref = self.obs
            _val_overrides["active_mask"] = torch.ones(
                total, dtype=torch.float32, device=ref.device,
            )

        def _reshape_and_chunk(x: torch.Tensor, is_hidden: bool = False):
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

                if f.name in ("final_global_state", "final_obs"):
                    kwargs[f.name] = None
                    continue

                # Shape: (N, state_dim).
                if f.name == "boundary_global_state":
                    kwargs[f.name] = val
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
    obs: torch.Tensor | np.ndarray = None          # (batch, *obs_shape).
    act: torch.Tensor | np.ndarray = None          # (batch,) or (batch, act_dim).
    rew: torch.Tensor | np.ndarray = None          # (batch,).
    next_obs: torch.Tensor | np.ndarray = None     # (batch, *obs_shape).
    done: torch.Tensor | np.ndarray = None         # (batch,).

    action_mask: torch.Tensor | np.ndarray = None  # (batch, num_actions).
    next_action_mask: torch.Tensor | np.ndarray = None
    # For CTDE algorithms (VDN, QMIX) - global state.
    state: torch.Tensor | np.ndarray = None         # (batch, state_dim).
    next_state: torch.Tensor | np.ndarray = None    # (batch, state_dim).
    # For active agent masking.
    active_mask: torch.Tensor | np.ndarray = None   # (batch,).

    gamma_power: torch.Tensor | np.ndarray = None   # (batch,).

@dataclass
class SequenceBatch(BaseBatch):
    """Sequence batch for recurrent off-policy training.

    The sequence length is ``burn_in_len + seq_len``. ``mask`` is zero during
    burn-in and padding and one for valid optimization steps.
    """

    obs: torch.Tensor | np.ndarray = None               # (B, L, obs_dim).
    act: torch.Tensor | np.ndarray = None               # (B, L).
    rew: torch.Tensor | np.ndarray = None               # (B, L).
    next_obs: torch.Tensor | np.ndarray = None          # (B, L, obs_dim).
    done: torch.Tensor | np.ndarray = None              # (B, L).
    mask: torch.Tensor | np.ndarray = None              # Shape: (B, L).
    action_mask: torch.Tensor | np.ndarray = None       # (B, L, act_dim).
    next_action_mask: torch.Tensor | np.ndarray = None  # (B, L, act_dim).
    active_mask: torch.Tensor | np.ndarray = None       # (B, L) 1=READY, 0=ON_EDGE.

    state: torch.Tensor | np.ndarray = None             # (B, L, state_dim).
    next_state: torch.Tensor | np.ndarray = None        # (B, L, state_dim).
    burn_in_len: int = 0

@dataclass
class CollectResult:
    batch: Union[RolloutBatch, Dict[str, RolloutBatch]]
    n_steps: int = 0
    n_episodes: int = 0
    episode_rewards: List[float] = field(default_factory=list)  # Returns for completed episodes.
    episode_lengths: List[int] = field(default_factory=list)    # Lengths of completed episodes.
