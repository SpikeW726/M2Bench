"""
MPNN Actor: Message Passing Neural Network for graph-structured observations.
Extracted from gnn_MPNN_mlpcritic.py, no RLlib dependency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    def scatter_mean(src, index, dim=0, dim_size=None):
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        out = out.index_add_(0, index, src)
        count = torch.zeros((dim_size, 1), device=src.device).index_add_(
            0, index, torch.ones((src.size(0), 1), device=src.device)
        )
        return out / torch.clamp(count, min=1.0)

    def scatter_max(src, index, dim=0, dim_size=None):
        out = torch.full((dim_size, src.size(1)), -1e9, device=src.device)
        return torch.scatter_reduce(
            out,
            0,
            index.unsqueeze(1).expand(-1, src.size(1)),
            src,
            reduce="amax",
            include_self=True,
        )


def _ortho_init(layer: nn.Linear, gain: float = 1.0):
    """Orthogonal init, replaces RLlib normc_initializer."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class ResidualFC(nn.Module):
    """Residual FC block, no RLlib dependency."""

    def __init__(self, in_size: int, out_size: int, activation_fn: str = "silu"):
        super().__init__()
        act = nn.SiLU if activation_fn == "silu" else (nn.ReLU if activation_fn == "relu" else activation_fn)
        self.fc = nn.Linear(in_size, out_size)
        _ortho_init(self.fc, gain=1.0)
        self.act = act()
        self.use_residual = in_size == out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.fc(x))
        return out + x if self.use_residual else out


class MPNNLayer(nn.Module):
    """MPNN message passing layer: edge -> node -> global."""

    def __init__(self, h_dim: int, mlp_layers: int = 2, activation_fn: str = "silu"):
        super().__init__()
        self.edge_norm = nn.LayerNorm(h_dim * 3)
        self.node_norm = nn.LayerNorm(h_dim * 2)
        self.global_norm = nn.LayerNorm(h_dim * 3)
        self.edge_mlp = self._build_gnn_mlp(h_dim * 3, h_dim, mlp_layers, activation_fn)
        self.node_mlp = self._build_gnn_mlp(h_dim * 2, h_dim, mlp_layers, activation_fn)
        self.global_mlp = self._build_gnn_mlp(h_dim * 3, h_dim, mlp_layers, activation_fn)

    def _build_gnn_mlp(self, input_dim: int, hidden_dim: int, num_layers: int, activation_fn: str):
        num_layers = max(1, num_layers)
        layers = [ResidualFC(input_dim, hidden_dim, activation_fn=activation_fn)]
        for _ in range(num_layers - 1):
            layers.append(ResidualFC(hidden_dim, hidden_dim, activation_fn=activation_fn))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, g, batch_idx, edge_batch):
        row, col = edge_index
        e_in = torch.cat([edge_attr, x[row], x[col]], dim=-1)
        e_in = self.edge_norm(e_in)
        edge_attr = edge_attr + self.edge_mlp(e_in)

        e_agg = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
        x_in = torch.cat([x, e_agg], dim=-1)
        x_in = self.node_norm(x_in)
        x = x + self.node_mlp(x_in)

        x_mean = scatter_mean(x, batch_idx, dim=0, dim_size=g.size(0))
        e_mean = scatter_mean(edge_attr, edge_batch, dim=0, dim_size=g.size(0))
        g_in = torch.cat([g, x_mean, e_mean], dim=-1)
        g_in = self.global_norm(g_in)
        g = g + self.global_mlp(g_in)

        return x, edge_attr, g


class MPNNActor(nn.Module):
    """
    MPNN Actor for graph-structured obs. forward(obs) -> logits.
    identity_dim follows masup_gnn.py: agent-index -> agent_num, position -> 1, decision -> 1+agent_num.
    """

    is_recurrent = False

    def __init__(self, obs_dim: int, action_dim: int, config):
        super().__init__()
        from utils.graph_utils import Graph

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.h_dim = config.hidden_dim
        self.agent_num = config.agent_num
        self.role_ifm = config.role_imformation or "agent-index"
        # 保存 config 字段供 get_config_dict 序列化
        self._graph_path = config.graph_path
        self._gnn_layers = config.gnn_layers
        self._actor_mlp_layers = config.actor_mlp_layers
        self._gnn_mlp_layers = config.gnn_mlp_layers
        self._mlp_activation = config.mlp_activation

        graph = Graph(config.graph_path)
        self.static_node_num = len(graph.nodes)
        num_static_edges = sum(len(graph.adj_list[u]) for u in graph.nodes)
        self.max_edges = num_static_edges + (self.agent_num * 2)
        self.num_nodes = self.static_node_num + self.agent_num

        # identity_dim per masup_gnn.py
        if self.role_ifm == "agent-index":
            self.identity_dim = self.agent_num
        elif self.role_ifm == "position":
            self.identity_dim = 1
        elif self.role_ifm == "decision":
            self.identity_dim = 1 + self.agent_num
        else:
            self.identity_dim = self.agent_num

        self.node_init = nn.Linear(2, self.h_dim)
        self.edge_init = nn.Linear(1, self.h_dim)
        self.global_init = nn.Linear(2, self.h_dim)
        self.id_encoder = nn.Linear(self.identity_dim, self.h_dim)

        self.layers = nn.ModuleList([
            MPNNLayer(self.h_dim, config.gnn_mlp_layers, config.mlp_activation)
            for _ in range(config.gnn_layers)
        ])

        self.actor_head = self._build_mlp(
            self.h_dim * 3, action_dim, self.h_dim, config.actor_mlp_layers, config.mlp_activation
        )
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.constant_(self.actor_head[-1].bias, 0)

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_fn: str,
    ) -> nn.Sequential:
        num_layers = max(1, num_layers)
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(ResidualFC(prev_dim, hidden_dim, activation_fn=activation_fn))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _compute_joint_embedding(self, obs_tensor: torch.Tensor):
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        batch_size = obs_tensor.shape[0]
        device = obs_tensor.device

        curr = 0
        x_raw = obs_tensor[:, curr : curr + self.num_nodes * 2].reshape(-1, 2)
        curr += self.num_nodes * 2
        e_src = obs_tensor[:, curr : curr + self.max_edges]
        e_dst = obs_tensor[:, curr + self.max_edges : curr + self.max_edges * 2]
        curr += self.max_edges * 2
        e_attr_raw = obs_tensor[:, curr : curr + self.max_edges].reshape(-1, 1)
        curr += self.max_edges
        masks = obs_tensor[:, curr : curr + self.max_edges].reshape(-1).bool()
        curr += self.max_edges
        g_raw = obs_tensor[:, curr : curr + 2]
        curr += 2
        identity = obs_tensor[:, curr : curr + self.identity_dim]

        offsets = (torch.arange(batch_size, device=device) * self.num_nodes).unsqueeze(1)
        edge_index = torch.stack(
            [
                (e_src + offsets).reshape(-1)[masks].long(),
                (e_dst + offsets).reshape(-1)[masks].long(),
            ],
            dim=0,
        )
        edge_attr = e_attr_raw[masks]
        edge_batch = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .repeat(1, self.max_edges)
            .reshape(-1)[masks]
            .long()
        )
        batch_idx = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .repeat(1, self.num_nodes)
            .reshape(-1)
            .long()
        )

        x = F.silu(self.node_init(x_raw))
        e = F.silu(self.edge_init(edge_attr))
        g = F.silu(self.global_init(g_raw))

        for layer in self.layers:
            x, e, g = layer(x, edge_index, e, g, batch_idx, edge_batch)

        h_reshaped = x.reshape(batch_size, self.num_nodes, -1)
        if self.role_ifm == "decision":
            agent_idx = identity[:, 1].long() - 1
            agent_idx = torch.clamp(agent_idx, min=0, max=self.agent_num - 1)
        else:
            agent_idx = torch.argmax(identity, dim=1)
        self_node_idx = self.static_node_num + agent_idx
        self_embed = h_reshaped[torch.arange(batch_size, device=device), self_node_idx, :]

        id_embed = F.silu(self.id_encoder(identity))

        actor_input = torch.cat([self_embed, g, id_embed], dim=-1)
        return actor_input

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actor_input = self._compute_joint_embedding(obs)
        return self.actor_head(actor_input)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "graph_path": self._graph_path,
            "agent_num": self.agent_num,
            "role_imformation": self.role_ifm,
            "hidden_dim": self.h_dim,
            "gnn_layers": self._gnn_layers,
            "actor_mlp_layers": self._actor_mlp_layers,
            "gnn_mlp_layers": self._gnn_mlp_layers,
            "mlp_activation": self._mlp_activation,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MPNNActor":
        from configs.network_configs import MPNNConfig
        mpnn_config = MPNNConfig(
            graph_path=cfg["graph_path"],
            agent_num=cfg["agent_num"],
            role_imformation=cfg.get("role_imformation", "agent-index"),
            hidden_dim=cfg.get("hidden_dim", 64),
            gnn_layers=cfg.get("gnn_layers", 2),
            actor_mlp_layers=cfg.get("actor_mlp_layers", 1),
            gnn_mlp_layers=cfg.get("gnn_mlp_layers", 2),
            mlp_activation=cfg.get("mlp_activation", "silu"),
        )
        return cls(obs_dim=cfg["input_dim"], action_dim=cfg["output_dim"], config=mpnn_config)
