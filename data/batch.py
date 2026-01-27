from dataclasses import dataclass, field, fields
from typing import Dict, Iterator, List, Union
try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python 3.10 兼容
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


@dataclass
class CollectResult:
    """采集结果，包含 batch 数据和统计信息"""
    batch: Union[RolloutBatch, Dict[str, RolloutBatch]]  # 采集的数据
    n_steps: int = 0                                      # 总步数
    n_episodes: int = 0                                   # 完成的 episode 数
    episode_rewards: List[float] = field(default_factory=list)  # 每个完成 episode 的总奖励
    episode_lengths: List[int] = field(default_factory=list)    # 每个完成 episode 的长度