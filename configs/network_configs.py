"""网络架构参数 dataclass。

每种网络类型对应一个 Config 类，YAML 中通过 actor_type / critic_type / q_type
分发字段选择对应的 Config 和网络类。

继承关系:
    NetworkConfig (base, 无字段)
    ├── MLPConfig (hidden)
    │   └── QMLPConfig (hidden, dueling)
    ├── RNNConfig (rnn_type, hidden_size, num_layers)
    │   └── QRNNConfig (rnn_type, hidden_size, num_layers, dueling)
    └── SUNConfig (num_nodes, node_feat_dim, f1_hidden, f2_hidden)
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
