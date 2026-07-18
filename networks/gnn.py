"""Graph-based actor networks without RLlib or PyG dependencies.

``MPNNActor`` updates node, edge, and global embeddings. ``GraphSageActor``
implements edge-aware GraphSAGE with static edge features, one linear transform
per layer, and L2-normalized node embeddings. Both actors consume the same flat
observation layout::

    node features   num_nodes * node_feat_dim
    edge sources    max_edges
    edge targets    max_edges
    edge features   max_edges * edge_feat_dim
    edge mask       max_edges
    global features global_feat_dim
    identity        identity_dim

``num_nodes`` is the number of static graph nodes plus one virtual node per
agent. Static nodes are indexed first; virtual node ``static_node_num + i``
represents agent ``i``. Edge indices use this combined index space, and padded
edges are ignored by the edge mask.

Identity encoding depends on ``role_imformation``: ``agent-index`` uses an
agent one-hot vector, ``position`` uses the current node index, and ``decision``
combines that index with a decision-order one-hot vector. YAML dimensions must
match the observation constructed by the environment. GraphSAGE advances over
the global-feature segment while decoding but does not feed it to the network.
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

        self.input_dim = obs_dim
        self.output_dim = action_dim
        self.h_dim = config.hidden_dim
        self.agent_num = config.agent_num
        self.role_ifm = config.role_imformation or "agent-index"

        self._graph_path = config.graph_path
        self._gnn_layers = config.gnn_layers
        self._actor_mlp_layers = config.actor_mlp_layers
        self._gnn_mlp_layers = config.gnn_mlp_layers
        self._mlp_activation = config.mlp_activation
        self._node_feat_dim = config.node_feat_dim
        self._edge_feat_dim = config.edge_feat_dim
        self._global_feat_dim = config.global_feat_dim

        graph = Graph(config.graph_path)
        self.static_node_num = len(graph.nodes)
        num_static_edges = sum(len(graph.adj_list[u]) for u in graph.nodes)
        self.max_edges = num_static_edges + (self.agent_num * 2)
        self.num_nodes = self.static_node_num + self.agent_num

        if self.role_ifm == "agent-index":
            self.identity_dim = self.agent_num
        elif self.role_ifm == "position":
            self.identity_dim = 1
        elif self.role_ifm == "decision":
            self.identity_dim = 1 + self.agent_num
        else:
            self.identity_dim = self.agent_num

        self.node_init = nn.Linear(self._node_feat_dim, self.h_dim)
        self.edge_init = nn.Linear(self._edge_feat_dim, self.h_dim)
        self.global_init = nn.Linear(self._global_feat_dim, self.h_dim)
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

        self._mpnn_spatial_cache_key = None

    def _ensure_mpnn_spatial_cache(self, batch_size: int, device: torch.device) -> None:
        key = (batch_size, str(device))
        if self._mpnn_spatial_cache_key == key:
            return
        self._mpnn_spatial_cache_key = key
        ar = torch.arange(batch_size, device=device)
        self._mpnn_arange_b = ar
        self._mpnn_offsets = (ar * self.num_nodes).unsqueeze(1)
        self._mpnn_batch_idx = (
            ar.unsqueeze(1).repeat(1, self.num_nodes).reshape(-1).long()
        )
        self._mpnn_edge_batch_flat = (
            ar.unsqueeze(1).repeat(1, self.max_edges).reshape(-1).long()
        )

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
        self._ensure_mpnn_spatial_cache(batch_size, device)

        curr = 0
        x_raw = obs_tensor[:, curr : curr + self.num_nodes * self._node_feat_dim].reshape(-1, self._node_feat_dim)
        curr += self.num_nodes * self._node_feat_dim
        e_src = obs_tensor[:, curr : curr + self.max_edges]
        e_dst = obs_tensor[:, curr + self.max_edges : curr + self.max_edges * 2]
        curr += self.max_edges * 2
        e_attr_raw = obs_tensor[:, curr : curr + self.max_edges * self._edge_feat_dim].reshape(-1, self._edge_feat_dim)
        curr += self.max_edges * self._edge_feat_dim
        masks = obs_tensor[:, curr : curr + self.max_edges].reshape(-1).bool()
        curr += self.max_edges
        g_raw = obs_tensor[:, curr : curr + self._global_feat_dim]
        curr += self._global_feat_dim
        identity = obs_tensor[:, curr : curr + self.identity_dim]

        offsets = self._mpnn_offsets
        edge_index = torch.stack(
            [
                (e_src + offsets).reshape(-1)[masks].long(),
                (e_dst + offsets).reshape(-1)[masks].long(),
            ],
            dim=0,
        )
        edge_attr = e_attr_raw[masks]
        edge_batch = self._mpnn_edge_batch_flat[masks]
        batch_idx = self._mpnn_batch_idx

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
        self_embed = h_reshaped[self._mpnn_arange_b, self_node_idx, :]

        id_embed = F.silu(self.id_encoder(identity))

        actor_input = torch.cat([self_embed, g, id_embed], dim=-1)
        return actor_input

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actor_input = self._compute_joint_embedding(obs)
        return self.actor_head(actor_input)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "agent_num": self.agent_num,
            "role_imformation": self.role_ifm,
            "hidden_dim": self.h_dim,
            "gnn_layers": self._gnn_layers,
            "actor_mlp_layers": self._actor_mlp_layers,
            "gnn_mlp_layers": self._gnn_mlp_layers,
            "mlp_activation": self._mlp_activation,
            "node_feat_dim": self._node_feat_dim,
            "edge_feat_dim": self._edge_feat_dim,
            "global_feat_dim": self._global_feat_dim,
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
            node_feat_dim=cfg.get("node_feat_dim", 2),
            edge_feat_dim=cfg.get("edge_feat_dim", 1),
            global_feat_dim=cfg.get("global_feat_dim", 2),
        )
        return cls(obs_dim=cfg["input_dim"], action_dim=cfg["output_dim"], config=mpnn_config)

# GraphSAGE with Edge Attributes - Algorithm 1.

class GraphSageLayer(nn.Module):
    """One edge-aware GraphSAGE layer.

    Each source embedding is concatenated with its edge embedding, incoming
    messages are mean-aggregated at the destination, and the result is combined
    with the destination embedding before activation and L2 normalization.
    Edge embeddings remain unchanged across layers.
    """

    def __init__(self, h_dim: int, activation_fn: str = "relu"):
        super().__init__()
        # W_k: CONCAT(h^{k-1}_v [h], h^k_{N(v)} [2h]) -> h.
        self.W_k = nn.Linear(h_dim * 3, h_dim, bias=True)
        _ortho_init(self.W_k, gain=1.0)
        act_map = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}
        self.activation = act_map.get(activation_fn, nn.ReLU)()

    def forward(
        self,
        x: torch.Tensor,           # (batch*num_nodes, h_dim).
        edge_index: torch.Tensor,  # (2, num_valid_edges).
        edge_attr: torch.Tensor,   # (num_valid_edges, h_dim).
    ) -> torch.Tensor:
        row, col = edge_index                                          # row=src u, col=dst v.
        messages = torch.cat([x[row], edge_attr], dim=-1)
        h_neigh = scatter_mean(messages, col, dim=0, dim_size=x.size(0))
        x_new = self.activation(self.W_k(torch.cat([x, h_neigh], dim=-1)))
        return F.normalize(x_new, p=2, dim=-1)

class GraphSageActor(nn.Module):
    """Edge-aware GraphSAGE actor using the module-level observation protocol.

    The standard head combines the agent's virtual-node embedding with its
    identity embedding. In MAGEC neighbor-scoring mode, neighbor embeddings are
    scored individually and passed through a selector; the final action slot is
    reserved for no-op. Jumping Knowledge optionally averages all layer outputs.
    """

    is_recurrent = False

    def __init__(self, obs_dim: int, action_dim: int, config):
        super().__init__()
        from utils.graph_utils import Graph

        self.input_dim = obs_dim
        self.output_dim = action_dim
        self.h_dim = config.hidden_dim
        self.agent_num = config.agent_num
        self.role_ifm = config.role_imformation or "agent-index"

        self._graph_path = config.graph_path
        self._gnn_layers = config.gnn_layers
        self._actor_mlp_layers = config.actor_mlp_layers
        self._mlp_activation = config.mlp_activation
        self._node_feat_dim = config.node_feat_dim
        self._edge_feat_dim = config.edge_feat_dim
        self._global_feat_dim = config.global_feat_dim
        self._neighbor_scoring = getattr(config, "neighbor_scoring", False)
        self._use_jk = getattr(config, "use_jk", False)

        graph = Graph(config.graph_path)
        self.static_node_num = len(graph.nodes)
        num_static_edges = sum(len(graph.adj_list[u]) for u in graph.nodes)
        max_degree = max(len(graph.adj_list[u]) for u in graph.nodes)

        if self._neighbor_scoring:
            # READY agents expose their current node and at most max_degree neighbors.
            max_agent_edges = 1 + max_degree
        else:
            # The original MASUP graph observation uses at most two dynamic edges.
            max_agent_edges = 2

        self.max_edges = num_static_edges + self.agent_num * max_agent_edges
        self.num_nodes = self.static_node_num + self.agent_num

        if self.role_ifm == "agent-index":
            self.identity_dim = self.agent_num
        elif self.role_ifm == "position":
            self.identity_dim = 1
        elif self.role_ifm == "decision":
            self.identity_dim = 1 + self.agent_num
        else:
            self.identity_dim = self.agent_num

        self.node_init = nn.Linear(self._node_feat_dim, self.h_dim)

        self.edge_init = nn.Linear(self._edge_feat_dim, self.h_dim)
        _ortho_init(self.node_init, gain=1.0)
        _ortho_init(self.edge_init, gain=1.0)

        self.layers = nn.ModuleList([
            GraphSageLayer(self.h_dim, config.mlp_activation)
            for _ in range(config.gnn_layers)
        ])

        if self._neighbor_scoring:
            # MAGEC Neighbor Scoring.
            # action_dim = max_degree + 1.
            self._action_dim = action_dim
            self._max_degree = action_dim - 1

            self.neighbor_scorer = self._build_mlp(
                self.h_dim, 1, self.h_dim, config.actor_mlp_layers, config.mlp_activation
            )
            nn.init.orthogonal_(self.neighbor_scorer[-1].weight, gain=0.01)
            nn.init.constant_(self.neighbor_scorer[-1].bias, 0)

            self.selector = self._build_mlp(
                action_dim, action_dim, self.h_dim, config.actor_mlp_layers, config.mlp_activation
            )
            nn.init.orthogonal_(self.selector[-1].weight, gain=0.01)
            nn.init.constant_(self.selector[-1].bias, 0)
        else:

            self.id_encoder = nn.Linear(self.identity_dim, self.h_dim)
            _ortho_init(self.id_encoder, gain=1.0)
            self.actor_head = self._build_mlp(
                self.h_dim * 2, action_dim, self.h_dim, config.actor_mlp_layers, config.mlp_activation
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

    def _decode_obs(self, obs_tensor: torch.Tensor):
        """Decode flat observations into batched graph tensors.

        Returns flattened node features, masked and batch-offset edge tensors,
        node batch indices, identity vectors, and unmasked two-dimensional edge
        tensors required by the neighbor-scoring path.
        """

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        batch_size = obs_tensor.shape[0]
        device = obs_tensor.device

        curr = 0
        x_raw = obs_tensor[:, curr : curr + self.num_nodes * self._node_feat_dim].reshape(-1, self._node_feat_dim)
        curr += self.num_nodes * self._node_feat_dim
        e_src = obs_tensor[:, curr : curr + self.max_edges]           # (B, E).
        e_dst = obs_tensor[:, curr + self.max_edges : curr + self.max_edges * 2]  # (B, E).
        curr += self.max_edges * 2
        e_attr_raw = obs_tensor[:, curr : curr + self.max_edges * self._edge_feat_dim]  # (B, E*F).
        curr += self.max_edges * self._edge_feat_dim
        masks_flat = obs_tensor[:, curr : curr + self.max_edges].reshape(-1).bool()     # (B*E,).
        curr += self.max_edges

        curr += self._global_feat_dim
        identity = obs_tensor[:, curr : curr + self.identity_dim]

        offsets = (torch.arange(batch_size, device=device) * self.num_nodes).unsqueeze(1)
        edge_index = torch.stack(
            [
                (e_src + offsets).reshape(-1)[masks_flat].long(),
                (e_dst + offsets).reshape(-1)[masks_flat].long(),
            ],
            dim=0,
        )
        e_attr_flat = e_attr_raw.reshape(-1, self._edge_feat_dim)   # (B*E, F).
        edge_attr_masked = e_attr_flat[masks_flat]
        batch_idx = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .repeat(1, self.num_nodes)
            .reshape(-1)
            .long()
        )

        # Neighbor Scoring.
        e_src_2d  = e_src                                                    # (B, E).
        e_dst_2d  = e_dst                                                    # (B, E).
        e_attr_2d = e_attr_raw.reshape(batch_size, self.max_edges, self._edge_feat_dim)  # (B, E, F).
        e_mask_2d = masks_flat.reshape(batch_size, self.max_edges)          # (B, E).

        return (x_raw, edge_index, edge_attr_masked, batch_idx,
                identity, batch_size,
                e_src_2d, e_dst_2d, e_attr_2d, e_mask_2d)

    def _run_gnn(self, x_raw, edge_index, edge_attr_masked):
        """Encode inputs and run GraphSAGE, optionally averaging layer outputs."""

        x = F.relu(self.node_init(x_raw))
        e = F.relu(self.edge_init(edge_attr_masked))

        if self._use_jk:
            layer_outputs = []
            for layer in self.layers:
                x = layer(x, edge_index, e)
                layer_outputs.append(x)

            h = torch.stack(layer_outputs, dim=0).mean(dim=0)
        else:
            for layer in self.layers:
                x = layer(x, edge_index, e)
            h = x
        return h

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        (x_raw, edge_index, edge_attr_masked, batch_idx,
         identity, batch_size,
         e_src_2d, e_dst_2d, e_attr_2d, e_mask_2d) = self._decode_obs(obs)
        device = obs.device if obs.dim() > 1 else obs.unsqueeze(0).device

        h = self._run_gnn(x_raw, edge_index, edge_attr_masked)   # (B*N, h).

        if self._neighbor_scoring:
            return self._forward_neighbor_scoring(
                h, identity, batch_size, device,
                e_src_2d, e_dst_2d, e_attr_2d, e_mask_2d)
        else:
            return self._forward_self_embed(h, identity, batch_size, device)

    def _forward_self_embed(self, h, identity, batch_size, device):
        """Apply the standard actor head to the current agent embedding."""

        h_reshaped = h.reshape(batch_size, self.num_nodes, -1)
        if self.role_ifm == "decision":
            agent_idx = identity[:, 1].long() - 1
            agent_idx = torch.clamp(agent_idx, min=0, max=self.agent_num - 1)
        else:
            agent_idx = torch.argmax(identity, dim=1)
        self_node_idx = self.static_node_num + agent_idx
        self_embed = h_reshaped[torch.arange(batch_size, device=device), self_node_idx, :]
        id_embed = F.relu(self.id_encoder(identity))
        return self.actor_head(torch.cat([self_embed, id_embed], dim=-1))

    def _forward_neighbor_scoring(self, h, identity, batch_size, device,
                                   e_src_2d, e_dst_2d, e_attr_2d, e_mask_2d):
        """Score MAGEC neighbors and map scores to action logits.

        Valid edges leaving the agent's virtual node identify candidate
        neighbors. Their embeddings are scored, scattered by ``neighborIndex``,
        and passed through the selector. The no-op score remains zero.
        """

        h_reshaped = h.reshape(batch_size, self.num_nodes, -1)   # (B, N, h).
        agent_idx = torch.argmax(identity, dim=1)                 # (B,).
        virt_idx = (self.static_node_num + agent_idx).float().unsqueeze(1)  # (B,1).

        # neighbor_scores shape = (B, action_dim).
        neighbor_scores = torch.zeros(
            batch_size, self._action_dim, device=device, dtype=h.dtype
        )

        is_from_virt = e_src_2d == virt_idx                       # (B, E).
        nbr_idx_vals = e_attr_2d[:, :, 1]
        is_valid = is_from_virt & e_mask_2d & (nbr_idx_vals >= 0.0)
        if not is_valid.any():
            return self.selector(neighbor_scores)

        pair_idx = torch.nonzero(is_valid, as_tuple=False)        # (K, 2) [b, e].
        b_idx = pair_idx[:, 0]
        e_idx = pair_idx[:, 1]
        dst_nodes = e_dst_2d[b_idx, e_idx].long()
        nbr_indices = nbr_idx_vals[b_idx, e_idx].long().clamp(
            0, self._max_degree - 1
        )
        nbr_embeds = h_reshaped[b_idx, dst_nodes, :]              # (K, h).
        scores = self.neighbor_scorer(nbr_embeds).squeeze(-1)     # (K,).

        lin = b_idx * self._action_dim + nbr_indices
        neighbor_scores.view(-1).index_copy_(0, lin, scores)

        return self.selector(neighbor_scores)                      # (B, action_dim).

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "agent_num": self.agent_num,
            "role_imformation": self.role_ifm,
            "hidden_dim": self.h_dim,
            "gnn_layers": self._gnn_layers,
            "actor_mlp_layers": self._actor_mlp_layers,
            "mlp_activation": self._mlp_activation,
            "node_feat_dim": self._node_feat_dim,
            "edge_feat_dim": self._edge_feat_dim,
            "global_feat_dim": self._global_feat_dim,
            "neighbor_scoring": self._neighbor_scoring,
            "use_jk": self._use_jk,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "GraphSageActor":
        from configs.network_configs import SAGEConfig
        sage_config = SAGEConfig(
            graph_path=cfg["graph_path"],
            agent_num=cfg["agent_num"],
            role_imformation=cfg.get("role_imformation", "agent-index"),
            hidden_dim=cfg.get("hidden_dim", 64),
            gnn_layers=cfg.get("gnn_layers", 2),
            actor_mlp_layers=cfg.get("actor_mlp_layers", 1),
            mlp_activation=cfg.get("mlp_activation", "relu"),
            node_feat_dim=cfg.get("node_feat_dim", 2),
            edge_feat_dim=cfg.get("edge_feat_dim", 1),
            global_feat_dim=cfg.get("global_feat_dim", 2),
            neighbor_scoring=cfg.get("neighbor_scoring", False),
            use_jk=cfg.get("use_jk", False),
        )
        return cls(obs_dim=cfg["input_dim"], action_dim=cfg["output_dim"], config=sage_config)
