"""算法参数 dataclass，继承层次与算法类一一对应。

AlgoParams
├── OnPolicyParams         ← Actor-Critic On-Policy 通用参数
│   ├── A2CParams          ← 单优化器 A2C
│   ├── MAA2CParams        ← 双优化器 MAA2C
│   └── PPOParams          ← PPO 家族特有参数 (clip, KL)
│       ├── IPPOParams     ← 单优化器 (PPO / IPPO 共用)
│       ├── MAPPOParams    ← 双优化器 + MAPPO 特有功能
│       └── VDPPOParams    ← 值分解 + PPO actor + 双优化器
├── OffPolicyParams        ← Off-Policy 通用参数
│   └── D3QNParams
│       └── IQLParams      ← 独立 Q-Learning
│           ├── VDNParams  ← VDN (SumMixer, 默认参数共享)
│           └── QMIXParams ← QMIX (继承 VDN + 超网络 Mixer)
└── QTableParams           ← Tabular Q-learning
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
    # Target 网络更新: tau < 1.0 → soft update, tau >= 1.0 → hard update
    tau: float = 0.1
    target_update_freq: int = 1         # 目标网络更新间隔 (update 调用次数)
    mixer_embed_dim: int = 64           # QPLEXMixer lambda_net 隐藏层维度
    q_clip_range: Optional[float] = None  # Q 值 clipping（None = 不 clip，与 PPO clip_range 解耦）
    # LR scheduler (双优化器，actor/q 分离)
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    q_lr_start_factor: float = 1.0
    q_lr_end_factor: float = 0.1
    q_lr_decay_ratio: float = 0.8
    # True:  环境所有 agent reward 相同，取 agent_0 的值作为 r_tot
    # False: 各 agent reward 求和作为 r_tot
    reward_global: bool = False
    # Q-network 热身期：前 N 次 update 冻结 actor，仅训练 Q-network + mixer。
    # 0 = 不冻结（默认）。适用于有预训练 actor 权重、Q-network 随机初始化的场景。
    freeze_actor_iters: int = 0


# =============================================================================
#                          A2C 系列
# =============================================================================

@dataclass(kw_only=True)
class A2CParams(OnPolicyParams):
    """单优化器 A2C 参数。"""
    lr: float = 7e-4
    use_lr_scheduler: bool = False
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.1
    lr_decay_ratio: float = 0.8


@dataclass(kw_only=True)
class MAA2CParams(OnPolicyParams):
    """MAA2C 参数（双优化器 + LR scheduler）。"""
    actor_lr: float = 7e-4
    critic_lr: float = 7e-4
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    critic_lr_start_factor: float = 1.0
    critic_lr_end_factor: float = 0.1
    critic_lr_decay_ratio: float = 0.8


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
    # 经验回放同步（仅 IQL 生效，VDN/QMIX 不支持）
    sync_replay: bool = False
    # RNN 序列训练参数（MLP 时忽略）
    seq_len: int = 20                 # RNN 训练序列长度
    burn_in_len: int = 0              # R2D2 burn-in 长度（预热 RNN hidden state）
    max_episodes: int = 5000          # SequenceReplayBuffer 最大序列条数


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


@dataclass(kw_only=True)
class VDNParams(IQLParams):
    """VDN 参数，默认共享 Q-network。"""
    shared_policy: bool = True
    # True: 环境对所有 agent 返回相同奖励（取 agent_0 的值作为全局奖励）
    # False: 各 agent 奖励不同，对全体 agent 奖励求和得到全局奖励
    reward_global: bool = False


@dataclass(kw_only=True)
class QMIXParams(VDNParams):
    """QMIX 参数：继承 VDN（含 reward_global / shared_policy），仅增加 mixer 维度。"""
    mixer_embed_dim: int = 32


# =============================================================================
#                     Tabular 系列
# =============================================================================

@dataclass(kw_only=True)
class QTableParams(AlgoParams):
    """Q-table 参数，用于 BBLA / GBLA / ExGBLA 论文复现。"""
    lr: float = 0.1               # Q-learning alpha
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995  # 指数衰减（per episode）
    sync_update: bool = False      # 同步更新: 决策→到达折叠为一次 Q-update
