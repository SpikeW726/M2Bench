"""Dataclasses for network architecture parameters.

Each network type has a corresponding configuration class. The ``actor_type``,
``critic_type``, and ``q_type`` fields in YAML select both the configuration
class and the network implementation.

Inheritance hierarchy::

    NetworkConfig
    |-- MLPConfig
    |   `-- QMLPConfig
    |-- RNNConfig
    |   `-- QRNNConfig
    |-- SUNConfig
    |-- MPNNConfig
    |-- SAGEConfig
    `-- MASUPBaseConfig
        |-- MASUPActorConfig
        |-- MASUPActorRNNConfig
        |-- MASUPCriticConfig
        |-- MASUPCriticRNNConfig
        |-- MASUPQConfig
        |-- MASUPQRNNConfig
        |-- MASUPVDPPOQConfig
        `-- MASUPVDPPOQRNNConfig
"""

from dataclasses import dataclass, field
from typing import List
from sensai.util.string import ToStringMixin

@dataclass(kw_only=True)
class NetworkConfig(ToStringMixin):
    """Base class for network configurations."""

    pass

@dataclass(kw_only=True)
class MLPConfig(NetworkConfig):
    """Configuration shared by MLP actors and critics."""

    hidden: List[int] = field(default_factory=lambda: [256, 256])

@dataclass(kw_only=True)
class RNNConfig(NetworkConfig):
    """Configuration shared by recurrent actors and critics.

    ``fc_hidden`` specifies encoder layers before the GRU or LSTM. An empty
    list falls back to a single ``hidden_size`` layer for compatibility.
    """

    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)

@dataclass(kw_only=True)
class QMLPConfig(MLPConfig):
    """MLP Q-network configuration with optional dueling heads."""

    dueling: bool = False

@dataclass(kw_only=True)
class QRNNConfig(RNNConfig):
    """Recurrent Q-network configuration with optional dueling heads."""

    dueling: bool = False

@dataclass(kw_only=True)
class SUNConfig(NetworkConfig):
    """Configuration for SUN actors and critics."""

    num_nodes: int
    node_feat_dim: int = 2
    f1_hidden: int = 4
    f2_hidden: int = 6
    num_layers: int = 1  # Number of stacked GNN layers.

@dataclass(kw_only=True)
class MPNNConfig(NetworkConfig):
    """Configuration for MPNN actors."""

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

@dataclass(kw_only=True)
class MASUPBaseConfig(NetworkConfig):
    """Parameters shared by networks specialized for MASUP observations."""

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
    """MLP actor selected by ``actor_type: masup_mlp``."""

    hidden: List[int] = field(default_factory=lambda: [256, 256])

@dataclass(kw_only=True)
class MASUPActorRNNConfig(MASUPBaseConfig):
    """Recurrent actor selected by ``actor_type: masup_rnn``."""

    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)

@dataclass(kw_only=True)
class MASUPCriticConfig(MASUPBaseConfig):
    """MLP critic selected by ``critic_type: masup_mlp``.

    ``input_mode="state"`` consumes the global state plus an agent identity
    vector for MAPPO. ``input_mode="actor"`` consumes per-agent observations
    for IPPO.
    """

    hidden: List[int] = field(default_factory=lambda: [256, 256])
    input_mode: str = "state"

@dataclass(kw_only=True)
class MASUPCriticRNNConfig(MASUPBaseConfig):
    """Recurrent critic selected by ``critic_type: masup_rnn``."""

    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)
    input_mode: str = "state"

@dataclass(kw_only=True)
class MASUPQConfig(MASUPBaseConfig):
    """MLP Q-network used by IQL, VDN, and QMIX."""

    hidden: List[int] = field(default_factory=lambda: [256, 256])
    dueling: bool = False

@dataclass(kw_only=True)
class MASUPQRNNConfig(MASUPBaseConfig):
    """Recurrent Q-network used by recurrent IQL."""

    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)
    dueling: bool = False

@dataclass(kw_only=True)
class MASUPVDPPOQConfig(MASUPBaseConfig):
    """MLP Q-network used by VDPPO.

    The state dimension is inferred as ``3 * num_nodes + num_agents + 2``.
    """

    hidden: List[int] = field(default_factory=lambda: [64, 64])
    dueling: bool = False

@dataclass(kw_only=True)
class MASUPVDPPOQRNNConfig(MASUPBaseConfig):
    """Recurrent Q-network used by VDPPO."""

    rnn_type: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    fc_hidden: List[int] = field(default_factory=list)
    dueling: bool = False

@dataclass(kw_only=True)
class SAGEConfig(NetworkConfig):
    """Configuration for GraphSAGE actors.

    Unlike :class:`MPNNConfig`, each GNN layer applies one linear transform
    instead of an internal MLP, and ``global_feat_dim`` is used only to decode
    observations. With ``neighbor_scoring`` enabled, the actor uses the MAGEC
    neighbor scorer and selector, where the final action is a dedicated no-op.
    ``use_jk`` enables mean aggregation over all GNN layer outputs.
    """

    graph_path: str
    agent_num: int
    role_imformation: str = "agent-index"
    hidden_dim: int = 64
    gnn_layers: int = 2  # K in the GraphSAGE update rule.
    actor_mlp_layers: int = 1  # actor head / neighbor_scorer / selector MLP.
    mlp_activation: str = "relu"
    node_feat_dim: int = 2
    edge_feat_dim: int = 1
    global_feat_dim: int = 2  # Used for observation decoding, not as input.
    neighbor_scoring: bool = False
    use_jk: bool = False
