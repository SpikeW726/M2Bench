"""算法参数 dataclass，继承层次与算法类一一对应。

AlgoParams
└── OnPolicyParams         ← Actor-Critic On-Policy 通用参数
    └── PPOParams          ← PPO 家族特有参数 (clip, KL)
        ├── IPPOParams     ← 单优化器 (PPO / IPPO 共用)
        ├── MAPPOParams    ← 双优化器 + MAPPO 特有功能
        └── VDPPOParams    ← 值分解 + PPO actor + 双优化器
"""

from dataclasses import dataclass
from typing import Optional
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
    data_chunk_length: int = 10         # RNN chunk 长度（MLP 时忽略）


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
    # LR scheduler (双优化器，actor/q 分离)
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    q_lr_start_factor: float = 1.0
    q_lr_end_factor: float = 0.1
    q_lr_decay_ratio: float = 0.8


# =============================================================================
#                     Off-Policy 系列
# =============================================================================

@dataclass(kw_only=True)
class OffPolicyParams(AlgoParams):
    """Off-Policy 通用超参 (DQN / SAC 等共享)。

    tau < 1.0 → soft update（每次 update 后加权混合）；
    tau >= 1.0 → hard update（每 target_update_freq 次 update 后全量拷贝）。
    """
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    tau: float = 0.005                # soft update 混合系数
    target_update_freq: int = 1       # target 更新周期（update 调用次数）
    # RNN 序列训练参数（MLP 时忽略）
    seq_len: int = 20                 # RNN 训练序列长度
    burn_in_len: int = 0              # 可选 burn-in 长度（暂未实现）
    max_episodes: int = 5000          # EpisodeReplayBuffer 容量


@dataclass(kw_only=True)
class D3QNParams(OffPolicyParams):
    """D3QN (Double Dueling DQN) 特有超参"""
    lr: float = 1e-4
    # Epsilon-greedy 衰减
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.992       # 指数衰减系数（per update）
    epsilon_decay_by_step: bool = False  # True=按环境步数线性衰减, False=按 update 指数衰减
    exploration_fraction: float = 0.1    # 线性衰减模式下，衰减到 epsilon_end 的步数占比
    use_double_dqn: bool = True


@dataclass(kw_only=True)
class IQLParams(D3QNParams):
    """IQL (Independent Q-Learning) 参数，继承 D3QNParams。"""
    shared_policy: bool = False  # False=每个 agent 独立 Q-network
