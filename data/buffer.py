"""数据缓冲区：用于存储 RL 训练数据"""

from typing import Optional, Tuple
import numpy as np

from data.batch import RolloutBatch, TransitionBatch


class RolloutBuffer:
    """
    On-policy 数据缓冲区
    
    用于存储单个采集周期的数据，支持追加和清空。
    数据按 step 顺序存储，不支持采样。
    
    Args:
        obs_shape: 观测空间形状
        act_shape: 动作空间形状（标量动作用 () 或 (1,)）
        max_size: 最大容量
        global_state_shape: 全局状态形状（可选，用于 MAPPO）
        num_actions: 动作数量（用于 action_mask，可选）
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        max_size: int,
        global_state_shape: Optional[Tuple[int, ...]] = None,
        num_actions: Optional[int] = None,
    ):
        self.obs_shape = obs_shape
        self.act_shape = act_shape if act_shape else (1,)
        self.max_size = max_size
        
        # 预分配数组
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.act = np.zeros((max_size, *self.act_shape), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.log_prob = np.zeros(max_size, dtype=np.float32)
        self.value = np.zeros(max_size, dtype=np.float32)
        self.adv = np.zeros(max_size, dtype=np.float32)
        self.ret = np.zeros(max_size, dtype=np.float32)
        
        # 可选字段
        if global_state_shape is not None:
            self.global_state = np.zeros((max_size, *global_state_shape), dtype=np.float32)
        else:
            self.global_state = None
        
        if num_actions is not None:
            self.action_mask = np.zeros((max_size, num_actions), dtype=bool)
        else:
            self.action_mask = None
        
        self._ptr = 0
        self._size = 0
    
    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        log_prob: float,
        value: float = 0.0,
        global_state: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ):
        """添加单个 transition"""
        if self._ptr >= self.max_size:
            raise RuntimeError(f"Buffer 已满 (max_size={self.max_size})")
        
        self.obs[self._ptr] = obs
        self.act[self._ptr] = act
        self.rew[self._ptr] = rew
        self.done[self._ptr] = float(done)
        self.log_prob[self._ptr] = log_prob
        self.value[self._ptr] = value
        
        if global_state is not None and self.global_state is not None:
            self.global_state[self._ptr] = global_state
        
        if action_mask is not None and self.action_mask is not None:
            self.action_mask[self._ptr] = action_mask
        
        self._ptr += 1
        self._size = max(self._size, self._ptr)
    
    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: Optional[np.ndarray] = None,
        global_state: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ):
        """批量添加 transitions"""
        batch_size = len(obs)
        if self._ptr + batch_size > self.max_size:
            raise RuntimeError(f"Buffer 空间不足")
        
        idx = slice(self._ptr, self._ptr + batch_size)
        self.obs[idx] = obs
        self.act[idx] = act.reshape(batch_size, *self.act_shape)
        self.rew[idx] = rew
        self.done[idx] = done.astype(np.float32)
        self.log_prob[idx] = log_prob
        
        if value is not None:
            self.value[idx] = value
        
        if global_state is not None and self.global_state is not None:
            self.global_state[idx] = global_state
        
        if action_mask is not None and self.action_mask is not None:
            self.action_mask[idx] = action_mask
        
        self._ptr += batch_size
        self._size = max(self._size, self._ptr)
    
    def get(self) -> RolloutBatch:
        """获取所有已存储的数据"""
        idx = slice(0, self._ptr)
        
        # 处理动作维度
        act = self.act[idx]
        if self.act_shape == (1,):
            act = act.squeeze(-1)
        
        return RolloutBatch(
            obs=self.obs[idx].copy(),
            act=act.copy(),
            rew=self.rew[idx].copy(),
            done=self.done[idx].copy(),
            log_prob=self.log_prob[idx].copy(),
            value=self.value[idx].copy(),
            adv=self.adv[idx].copy(),
            ret=self.ret[idx].copy(),
            global_state=self.global_state[idx].copy() if self.global_state is not None else None,
            action_mask=self.action_mask[idx].copy() if self.action_mask is not None else None,
        )
    
    def reset(self):
        """清空缓冲区"""
        self._ptr = 0
        self._size = 0
    
    def __len__(self) -> int:
        return self._ptr
    
    @property
    def is_full(self) -> bool:
        return self._ptr >= self.max_size


class ReplayBuffer:
    """
    Off-policy 循环缓冲区（后续实现）
    
    用于 DQN, SAC 等 off-policy 算法。
    支持随机采样。
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        max_size: int,
    ):
        self.obs_shape = obs_shape
        self.act_shape = act_shape if act_shape else (1,)
        self.max_size = max_size
        
        # 预分配数组
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.act = np.zeros((max_size, *self.act_shape), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        
        self._ptr = 0
        self._size = 0
    
    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """添加单个 transition（循环覆盖）"""
        self.obs[self._ptr] = obs
        self.act[self._ptr] = act
        self.rew[self._ptr] = rew
        self.next_obs[self._ptr] = next_obs
        self.done[self._ptr] = float(done)
        
        self._ptr = (self._ptr + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> TransitionBatch:
        """随机采样"""
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) > buffer size ({self._size})")
        
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        
        act = self.act[indices]
        if self.act_shape == (1,):
            act = act.squeeze(-1)
        
        return TransitionBatch(
            obs=self.obs[indices].copy(),
            act=act.copy(),
            rew=self.rew[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            done=self.done[indices].copy(),
        )
    
    def __len__(self) -> int:
        return self._size
