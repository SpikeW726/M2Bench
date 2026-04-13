"""网络架构参数 dataclass。

每种网络类型对应一个 Config 类，YAML 中通过 actor_type / critic_type / q_type
分发字段选择对应的 Config 和网络类。

继承关系:
    NetworkConfig (base, 无字段)
    ├── MLPConfig (hidden)
    │   └── QMLPConfig (hidden, dueling)
    ├── RNNConfig (rnn_type, hidden_size, num_layers)
    │   └── QRNNConfig (rnn_type, hidden_size, num_layers, dueling)
    ├── SUNConfig (num_nodes, node_feat_dim, f1_hidden, f2_hidden)
    ├── MPNNConfig (graph_path, agent_num, hidden_dim, gnn_layers, ...)
    ├── SAGEConfig  (graph_path, agent_num, hidden_dim, gnn_layers, ...)
    └── MASUPBaseConfig (MASUP 专供网络公共基础)
        ├── MASUPActorConfig        (masup_mlp actor)
        ├── MASUPActorRNNConfig     (masup_rnn actor)
        ├── MASUPCriticConfig       (masup_mlp critic)
        ├── MASUPCriticRNNConfig    (masup_rnn critic)
        ├── MASUPQConfig            (masup_q_mlp，IQL/VDN/QMIX)
        ├── MASUPQRNNConfig         (masup_q_rnn，IQL RNN)
        ├── MASUPVDPPOQConfig       (masup_vdppo_mlp，VDPPO MLP Q)
        └── MASUPVDPPOQRNNConfig    (masup_vdppo_rnn，VDPPO RNN Q)
"""

from dataclasses import dataclass, field
from typing import List
from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class NetworkConfig(ToStringMixin):
    """网络配置基类（无字段）"""
    pass


@dataclass(kw_only=True)
class MLPConfig(NetworkConfig):
    """MLP 网络参数，用于 ActorMLP / CriticMLP"""
    hidden: List[int] = field(default_factory=lambda: [256, 256])


@dataclass(kw_only=True)
class RNNConfig(NetworkConfig):
    """RNN 网络参数，用于 ActorRNN / CriticRNN

    fc_hidden: GRU/LSTM 前的编码层尺寸列表，如 [256, 256]。
              为空时退化为单层 [hidden_size] 保持向后兼容。
    """
    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)


@dataclass(kw_only=True)
class QMLPConfig(MLPConfig):
    """Q-network MLP 参数（继承 MLPConfig，增加 dueling）"""
    dueling: bool = False


@dataclass(kw_only=True)
class QRNNConfig(RNNConfig):
    """Q-network RNN 参数（继承 RNNConfig，增加 dueling）"""
    dueling: bool = False


@dataclass(kw_only=True)
class SUNConfig(NetworkConfig):
    """SUN (Spatial Utility Network) 参数，用于 SUNActor / SUNCritic"""
    num_nodes: int
    node_feat_dim: int = 2
    f1_hidden: int = 4
    f2_hidden: int = 6
    num_layers: int = 1       # k: GNN 堆叠次数，增大可扩展感知范围


@dataclass(kw_only=True)
class MPNNConfig(NetworkConfig):
    """MPNN Actor 参数，用于 MPNNActor"""
    graph_path: str
    agent_num: int
    role_imformation: str = "agent-index"
    hidden_dim: int = 64
    gnn_layers: int = 2
    actor_mlp_layers: int = 1
    gnn_mlp_layers: int = 2
    mlp_activation: str = "silu"
    node_feat_dim: int = 2
    edge_feat_dim: int = 1
    global_feat_dim: int = 2


# =============================================================================
#                    MASUP 专供网络配置（与 masup_nets.py 约定绑定）
# =============================================================================

@dataclass(kw_only=True)
class MASUPBaseConfig(NetworkConfig):
    """MASUP 专供网络公共基础配置。"""
    graph_path: str = ""
    num_agents: int = 3
    num_nodes: int = 12
    role_imformation: str = "agent-index"
    gpe_dim: int = 8
    proj_dim: int = 16
    use_log_idleness: bool = True
    T_time: float = 0.0


@dataclass(kw_only=True)
class MASUPActorConfig(MASUPBaseConfig):
    """MASUPActorMLP 参数（actor_type: masup_mlp）。"""
    hidden: List[int] = field(default_factory=lambda: [256, 256])


@dataclass(kw_only=True)
class MASUPActorRNNConfig(MASUPBaseConfig):
    """MASUPActorRNN 参数（actor_type: masup_rnn）。"""
    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)


@dataclass(kw_only=True)
class MASUPCriticConfig(MASUPBaseConfig):
    """MASUPCriticMLP 参数（critic_type: masup_mlp）。

    input_mode="state"  : MAPPO，接 state() + agent_one_hot
    input_mode="actor"  : IPPO，接 per-agent obs
    """
    hidden: List[int] = field(default_factory=lambda: [256, 256])
    input_mode: str = "state"


@dataclass(kw_only=True)
class MASUPCriticRNNConfig(MASUPBaseConfig):
    """MASUPCriticRNN 参数（critic_type: masup_rnn）。"""
    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)
    input_mode: str = "state"


@dataclass(kw_only=True)
class MASUPQConfig(MASUPBaseConfig):
    """MASUPQMLP 参数（q_type: masup_q_mlp），用于 IQL / VDN / QMIX。"""
    hidden: List[int] = field(default_factory=lambda: [256, 256])
    dueling: bool = False


@dataclass(kw_only=True)
class MASUPQRNNConfig(MASUPBaseConfig):
    """MASUPQRNN 参数（q_type: masup_q_rnn），用于 IQL RNN。"""
    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)
    dueling: bool = False


@dataclass(kw_only=True)
class MASUPVDPPOQConfig(MASUPBaseConfig):
    """MASUPVDPPOQmlp 参数（q_type: masup_vdppo_mlp），用于 VDPPO MLP Q-network。

    state_dim 由 num_agents / num_nodes 自动推算（= 3N+K+2），无需显式配置。
    """
    hidden: List[int] = field(default_factory=lambda: [64, 64])
    dueling: bool = False


@dataclass(kw_only=True)
class MASUPVDPPOQRNNConfig(MASUPBaseConfig):
    """MASUPVDPPOQrnn 参数（q_type: masup_vdppo_rnn），用于 VDPPO RNN Q-network。"""
    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)
    dueling: bool = False


@dataclass(kw_only=True)
class SAGEConfig(NetworkConfig):
    """GraphSAGE Actor 参数，用于 GraphSageActor。

    与 MPNNConfig 的区别：
    - 无 gnn_mlp_layers（每层仅单线性变换 W_k，无内部 MLP）
    - mlp_activation 默认 relu（原论文默认非线性）
    - global_feat_dim 仅用于观测解码，不进入网络

    neighbor_scoring=True 时启用论文 MAGEC 的 Neighbor Scoring 机制：
    - action_dim = max_degree + 1（最后维为专用 no-op 槽）
    - 网络结构变为 neighbor_scorer (h→1) + selector (action_dim→action_dim)
    - 不再使用 id_encoder 和 actor_head

    use_jk=True 时启用 Jumping Knowledge（各 GNN 层输出均值聚合）。
    """
    graph_path: str
    agent_num: int
    role_imformation: str = "agent-index"
    hidden_dim: int = 64
    gnn_layers: int = 2        # Algorithm 1 中的 K
    actor_mlp_layers: int = 1  # actor head / neighbor_scorer / selector MLP 深度
    mlp_activation: str = "relu"
    node_feat_dim: int = 2
    edge_feat_dim: int = 1
    global_feat_dim: int = 2   # 仅用于 obs 解码定位 identity，不送入网络
    neighbor_scoring: bool = False  # 启用 MAGEC Neighbor Scoring 分支
    use_jk: bool = False            # 启用 Jumping Knowledge（层输出均值聚合）
