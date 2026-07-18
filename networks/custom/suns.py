import torch
import torch.nn as nn

class SUN_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)

class SUNBase(nn.Module):
    is_recurrent = False

    def __init__(self, obs_dim, num_nodes, node_feat_dim=2,
                 f1_hidden=4, f2_hidden=6, num_layers=1):
        super().__init__()
        self.input_dim = obs_dim
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.num_layers = num_layers
        assert self.input_dim == num_nodes * node_feat_dim + num_nodes ** 2

        self.f1 = SUN_mlp(node_feat_dim, f1_hidden)
        self.f2 = SUN_mlp(node_feat_dim + 1, f2_hidden)  # +1 for edge weight.

    def _parse_obs(self, obs):
        """Map ``(B, obs_dim)`` to node features and an ``(B, N, N)`` weight matrix."""
        N, d = self.num_nodes, self.node_feat_dim
        node_feat = obs[:, :N * d].reshape(-1, N, d)
        weight_mat = obs[:, N * d:].reshape(-1, N, N)
        return node_feat, weight_mat

    def _single_pass(self, node_feat, weight_mat):
        N = self.num_nodes

        self_util = self.f1(node_feat).squeeze(-1)             # (B, N).

        v_j = node_feat.unsqueeze(1).expand(-1, N, -1, -1)    # (B, N, N, d).
        e_ij = weight_mat.unsqueeze(-1)                        # (B, N, N, 1).
        f2_input = torch.cat([v_j, e_ij], dim=-1)             # (B, N, N, d+1).
        f2_out = self.f2(f2_input).squeeze(-1)                 # (B, N, N).

        adj = (weight_mat > 0).float()                         # (B, N, N).
        neighbor_sum = (f2_out * adj).sum(dim=-1)              # (B, N).

        return self_util + neighbor_sum                        # (B, N).

    def _compute_utilities(self, node_feat, weight_mat, apply_tanh=True):
        utilities = self._single_pass(node_feat, weight_mat)

        for _ in range(1, self.num_layers):

            node_feat = node_feat.clone()
            node_feat[:, :, 0] = utilities
            utilities = self._single_pass(node_feat, weight_mat)

        if apply_tanh:
            utilities = torch.tanh(utilities)
        return utilities

class SUNActor(SUNBase):
    def __init__(self, obs_dim, num_nodes, node_feat_dim=2,
                 f1_hidden=4, f2_hidden=6, num_layers=1):
        super().__init__(obs_dim, num_nodes, node_feat_dim,
                         f1_hidden, f2_hidden, num_layers)
        self.output_dim = num_nodes

    def forward(self, obs):
        node_feat, weight_mat = self._parse_obs(obs)
        return self._compute_utilities(node_feat, weight_mat, apply_tanh=True)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_nodes": self.num_nodes,
            "node_feat_dim": self.node_feat_dim,
            "f1_hidden": self.f1.fc1.out_features,
            "f2_hidden": self.f2.fc1.out_features,
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "SUNActor":
        return cls(
            obs_dim=cfg["input_dim"],
            num_nodes=cfg["num_nodes"],
            node_feat_dim=cfg.get("node_feat_dim", 2),
            f1_hidden=cfg.get("f1_hidden", 4),
            f2_hidden=cfg.get("f2_hidden", 6),
            num_layers=cfg.get("num_layers", 1),
        )

class SUNCritic(SUNBase):
    """SUN critic producing ``(B, 1)`` values through one-dimensional max pooling."""

    def __init__(self, obs_dim, num_nodes, node_feat_dim=2,
                 f1_hidden=4, f2_hidden=6, num_layers=1):
        super().__init__(obs_dim, num_nodes, node_feat_dim,
                         f1_hidden, f2_hidden, num_layers)
        self.output_dim = 1

    def forward(self, obs):
        node_feat, weight_mat = self._parse_obs(obs)
        utilities = self._compute_utilities(node_feat, weight_mat, apply_tanh=False)
        return utilities.max(dim=-1, keepdim=True)[0]  # (B, 1).

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_nodes": self.num_nodes,
            "node_feat_dim": self.node_feat_dim,
            "f1_hidden": self.f1.fc1.out_features,
            "f2_hidden": self.f2.fc1.out_features,
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "SUNCritic":
        return cls(
            obs_dim=cfg["input_dim"],
            num_nodes=cfg["num_nodes"],
            node_feat_dim=cfg.get("node_feat_dim", 2),
            f1_hidden=cfg.get("f1_hidden", 4),
            f2_hidden=cfg.get("f2_hidden", 6),
            num_layers=cfg.get("num_layers", 1),
        )
