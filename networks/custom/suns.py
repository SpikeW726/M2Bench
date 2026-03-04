import torch
import torch.nn as nn


class SUN_mlp(nn.Module):
    """论文 3-layer MLP: input → hidden → output, LeakyReLU(0.3) on hidden."""
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)


class SUNBase(nn.Module):
    """SUN GNN 基础模块：从 flat obs 重建图结构并计算 node utilities。

    obs 布局: [node_features_flat(2N), weight_mat_flat(N^2)]
    """
    is_recurrent = False

    def __init__(self, obs_dim, num_nodes, node_feat_dim=2,
                 f1_hidden=4, f2_hidden=6, num_layers=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.num_layers = num_layers
        assert obs_dim == num_nodes * node_feat_dim + num_nodes ** 2

        self.f1 = SUN_mlp(node_feat_dim, f1_hidden)
        self.f2 = SUN_mlp(node_feat_dim + 1, f2_hidden)  # +1 for edge weight

    def _parse_obs(self, obs):
        """(B, obs_dim) → node_features (B, N, d), weight_mat (B, N, N)"""
        N, d = self.num_nodes, self.node_feat_dim
        node_feat = obs[:, :N * d].reshape(-1, N, d)
        weight_mat = obs[:, N * d:].reshape(-1, N, N)
        return node_feat, weight_mat

    def _single_pass(self, node_feat, weight_mat):
        """单次 GNN pass: u_i = f1(v_i) + sum_{j in N_i} f2([v_j | e_ij])"""
        N = self.num_nodes

        self_util = self.f1(node_feat).squeeze(-1)             # (B, N)

        v_j = node_feat.unsqueeze(1).expand(-1, N, -1, -1)    # (B, N, N, d)
        e_ij = weight_mat.unsqueeze(-1)                        # (B, N, N, 1)
        f2_input = torch.cat([v_j, e_ij], dim=-1)             # (B, N, N, d+1)
        f2_out = self.f2(f2_input).squeeze(-1)                 # (B, N, N)

        adj = (weight_mat > 0).float()                         # (B, N, N)
        neighbor_sum = (f2_out * adj).sum(dim=-1)              # (B, N)

        return self_util + neighbor_sum                        # (B, N)

    def _compute_utilities(self, node_feat, weight_mat, apply_tanh=True):
        """k 层堆叠 GNN：每轮用上一轮 utility 替换 node_feat 的第 0 维（idleness）。

        k=1 时退化为单次 pass，与之前行为完全一致。
        """
        utilities = self._single_pass(node_feat, weight_mat)

        for _ in range(1, self.num_layers):
            # 用 utility 替换 idleness 分量，保留 distance 分量
            node_feat = node_feat.clone()
            node_feat[:, :, 0] = utilities
            utilities = self._single_pass(node_feat, weight_mat)

        if apply_tanh:
            utilities = torch.tanh(utilities)
        return utilities


class SUNActor(SUNBase):
    """SUN Actor: 输出 (B, N) logits，由 ActorPolicy 中 Categorical 消费。"""

    def __init__(self, obs_dim, num_nodes, node_feat_dim=2,
                 f1_hidden=4, f2_hidden=6, num_layers=1):
        super().__init__(obs_dim, num_nodes, node_feat_dim,
                         f1_hidden, f2_hidden, num_layers)

    def forward(self, obs):
        node_feat, weight_mat = self._parse_obs(obs)
        return self._compute_utilities(node_feat, weight_mat, apply_tanh=True)


class SUNCritic(SUNBase):
    """SUN Critic: 1D max-pool → (B, 1) value。"""

    def __init__(self, obs_dim, num_nodes, node_feat_dim=2,
                 f1_hidden=4, f2_hidden=6, num_layers=1):
        super().__init__(obs_dim, num_nodes, node_feat_dim,
                         f1_hidden, f2_hidden, num_layers)

    def forward(self, obs):
        node_feat, weight_mat = self._parse_obs(obs)
        utilities = self._compute_utilities(node_feat, weight_mat, apply_tanh=False)
        return utilities.max(dim=-1, keepdim=True)[0]  # (B, 1)
