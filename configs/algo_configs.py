"""算法参数 dataclass，继承层次与算法类一一对应。

AlgoParams
└── OnPolicyParams         ← Actor-Critic On-Policy 通用参数
    └── PPOParams          ← PPO 家族特有参数 (clip, KL)
        ├── IPPOParams     ← 单优化器 (PPO / IPPO 共用)
        ├── MAPPOParams    ← 双优化器 + MAPPO 特有功能
        └── VDPPOParams    ← 值分解 + PPO actor + 双优化器
"""

from dataclasses import dataclass, field
from typing import List, Optional
from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class AlgoParams(ToStringMixin):
    """算法参数基类"""
    pass


# =============================================================================
#                     On-Policy Actor-Critic 通用参数
# =============================================================================

@dataclass(kw_only=True)
class OnPolicyParams(AlgoParams):
    """Actor-Critic On-Policy 通用超参 (PPO / A2C 等共享)"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    use_active_mask: bool = True        # A+ 方案: 透明 GAE + Loss Masking
    use_value_norm: bool = False        # return 归一化 (RunningMeanStd)


# =============================================================================
#                          PPO 系列
# =============================================================================

@dataclass(kw_only=True)
class PPOParams(OnPolicyParams):
    """PPO 家族特有超参"""
    clip_range: float = 0.2
    clip_vloss: bool = True             # value loss clipping (PPO2 style)
    target_kl: Optional[float] = None   # KL early stopping (None=禁用)


@dataclass(kw_only=True)
class IPPOParams(PPOParams):
    """单优化器 PPO 参数 (PPO / IPPO 共用)"""
    lr: float = 3e-4
    # LR scheduler (单优化器)
    use_lr_scheduler: bool = False
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.1
    lr_decay_ratio: float = 0.8


@dataclass(kw_only=True)
class MAPPOParams(PPOParams):
    """MAPPO 参数 (双优化器 + LR scheduler)"""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # LR scheduler (双优化器，actor/critic 分离)
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    critic_lr_start_factor: float = 1.0
    critic_lr_end_factor: float = 0.1
    critic_lr_decay_ratio: float = 0.8


@dataclass(kw_only=True)
class VDPPOParams(PPOParams):
    """VDPPO 参数 (PPO actor + Q-decomposition + 双优化器)"""
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    target_update_freq: int = 200       # 目标网络硬更新间隔 (update 调用次数)
    mixer_embed_dim: int = 64           # QPLEXMixer lambda_net 隐藏层维度
    q_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    # LR scheduler (双优化器，actor/q 分离)
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    q_lr_start_factor: float = 1.0
    q_lr_end_factor: float = 0.1
    q_lr_decay_ratio: float = 0.8
