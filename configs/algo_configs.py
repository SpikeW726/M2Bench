"""Algorithm parameter dataclasses aligned with the algorithm hierarchy.

Inheritance hierarchy::

    AlgoParams
    |-- OnPolicyParams
    |   |-- A2CParams
    |   |-- MAA2CParams
    |   `-- PPOParams
    |       |-- IPPOParams
    |       |-- MAPPOParams
    |       `-- VDPPOParams
    |-- OffPolicyParams
    |   `-- D3QNParams
    |       `-- IQLParams
    |           |-- VDNParams
    |           `-- QMIXParams
    |-- MAPPOMATParams
    `-- QTableParams
"""

from dataclasses import dataclass
from typing import Optional
from sensai.util.string import ToStringMixin

@dataclass(kw_only=True)
class AlgoParams(ToStringMixin):
    """Base class for algorithm parameters."""

    pass

@dataclass(kw_only=True)
class OnPolicyParams(AlgoParams):
    """Parameters shared by on-policy actor-critic algorithms."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    use_active_mask: bool = True  # Transparent GAE and loss masking.
    use_value_norm: bool = False  # Normalize returns with running statistics.
    data_chunk_length: int = 10  # RNN chunk length; ignored by MLPs.

@dataclass(kw_only=True)
class PPOParams(OnPolicyParams):
    """Parameters shared by the PPO family."""

    clip_range: float = 0.2
    clip_vloss: bool = True             # value loss clipping (PPO2 style).
    target_kl: Optional[float] = None   # KL early stopping.

@dataclass(kw_only=True)
class IPPOParams(PPOParams):
    """IPPO parameters.

    Independent actor parameters are used by default. Set ``shared_policy`` in
    YAML to opt into parameter sharing.
    """

    lr: float = 3e-4
    shared_policy: bool = False

    use_lr_scheduler: bool = False
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.1
    lr_decay_ratio: float = 0.8

@dataclass(kw_only=True)
class MAPPOParams(PPOParams):
    """MAPPO parameters with separate actor and critic optimizers."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    critic_lr_start_factor: float = 1.0
    critic_lr_end_factor: float = 0.1
    critic_lr_decay_ratio: float = 0.8

@dataclass(kw_only=True)
class VDPPOParams(PPOParams):
    """VDPPO parameters for a PPO actor and decomposed Q-functions.

    ``tau < 1`` enables soft target updates; otherwise targets are copied every
    ``target_update_freq`` updates. ``reward_global`` selects an already shared
    reward from agent 0; otherwise agent rewards are summed. ``freeze_actor_iters``
    can warm up a randomly initialized Q-network before updating a pretrained actor.
    """

    actor_lr: float = 3e-4
    q_lr: float = 3e-4

    tau: float = 0.1
    target_update_freq: int = 1
    mixer_embed_dim: int = 64
    q_clip_range: Optional[float] = None

    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    q_lr_start_factor: float = 1.0
    q_lr_end_factor: float = 0.1
    q_lr_decay_ratio: float = 0.8

    reward_global: bool = False

    freeze_actor_iters: int = 0

@dataclass(kw_only=True)
class A2CParams(OnPolicyParams):
    """Single-optimizer A2C parameters."""

    lr: float = 7e-4
    use_lr_scheduler: bool = False
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.1
    lr_decay_ratio: float = 0.8

@dataclass(kw_only=True)
class MAA2CParams(OnPolicyParams):
    """MAA2C parameters with separate actor and critic optimizers."""

    actor_lr: float = 7e-4
    critic_lr: float = 7e-4
    use_lr_scheduler: bool = False
    actor_lr_start_factor: float = 1.0
    actor_lr_end_factor: float = 0.1
    actor_lr_decay_ratio: float = 0.8
    critic_lr_start_factor: float = 1.0
    critic_lr_end_factor: float = 0.1
    critic_lr_decay_ratio: float = 0.8

@dataclass(kw_only=True)
class OffPolicyParams(AlgoParams):
    """Parameters shared by replay-based value-learning algorithms.

    ``tau < 1`` performs a soft target update after each update. Otherwise, a
    hard copy is made every ``target_update_freq`` updates. Sequence parameters
    apply only to recurrent networks.
    """

    gamma: float = 0.99
    max_grad_norm: float = 0.5
    tau: float = 0.005
    target_update_freq: int = 1

    sync_replay: bool = False

    seq_len: int = 20
    burn_in_len: int = 0
    max_episodes: int = 5000

@dataclass(kw_only=True)
class D3QNParams(OffPolicyParams):
    """Parameters for Double Dueling DQN."""

    lr: float = 1e-4

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1_000_000
    use_double_dqn: bool = True

@dataclass(kw_only=True)
class IQLParams(D3QNParams):
    """IQL parameters; agents use independent Q-networks by default."""

    shared_policy: bool = False

@dataclass(kw_only=True)
class VDNParams(IQLParams):
    """VDN parameters with a shared Q-network by default.

    ``shared_sync`` advances MLP targets to the next shared READY event without
    compacting replay data. ``peng_q_lambda`` enables Peng's Q(lambda) for
    recurrent sequence training.
    """

    shared_policy: bool = True

    reward_global: bool = False

    shared_sync: bool = False

    peng_q_lambda: bool = False
    peng_lambda: float = 0.6

@dataclass(kw_only=True)
class QMIXParams(VDNParams):
    """QMIX parameters extending VDN with a hypernetwork mixer."""

    mixer_embed_dim: int = 32

@dataclass(kw_only=True)
class MAPPOMATParams(AlgoParams):
    """PPO parameters for the Multi-Agent Transformer implementation."""

    lr: float = 5e-5
    gamma: float = 0.99
    clip_coef: float = 0.2          # PPO clip range.
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 10.0
    policy_update_steps: int = 5
    factor: float = 0.5             # ReduceLROnPlateau factor.
    patience: int = 100             # ReduceLROnPlateau patience.

    hidden_dim: int = 64
    num_heads: int = 4
    encoder_layers: int = 2

@dataclass(kw_only=True)
class QTableParams(AlgoParams):
    """Tabular Q-learning parameters used by BBLA, GBLA, and ExGBLA."""

    lr: float = 0.1               # Q-learning alpha.
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1_000_000
    sync_update: bool = False
