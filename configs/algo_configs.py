"""算法参数 dataclass，继承层次与算法类一一对应。"""

from dataclasses import dataclass
from typing import Optional
from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class AlgoParams(ToStringMixin):
    """算法参数基类"""
    pass


# =============================================================================
#                          PPO 系列
# =============================================================================

@dataclass(kw_only=True)
class PPOParams(AlgoParams):
    """PPO 系列共享超参"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


@dataclass(kw_only=True)
class IPPOParams(PPOParams):
    """IPPO 参数 (单优化器)"""
    lr: float = 3e-4


@dataclass(kw_only=True)
class MAPPOParams(PPOParams):
    """MAPPO 参数 (双优化器 + LR scheduler + Value Normalization)"""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_vloss: bool = True
    target_kl: Optional[float] = None

    # LR scheduler
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    critic_lr_start_factor: float = 1.0
    critic_lr_end_factor: float = 0.1
    critic_lr_decay_ratio: float = 0.8

    # Value normalization
    use_value_norm: bool = False
