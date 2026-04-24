"""MAT Networks — 移植自 MARL_to_solve_patrol/MARL_patrol/patrol_policy/asy_patrol/。

GATEncoder: 图注意力网络 Encoder（来自 asy_encoder.py GraphEncoder）
MATDecoder: 多智能体 Transformer 自回归 Decoder（来自 asy_decoder.py AttentionModel）

工程适配（不改动网络结构）:
- 移除全局 device / obsMap 依赖，改为参数传入
- adj 矩阵通过 __init__ 注入（nn.Buffer，跟随模型移动）
- node_coords 通过 __init__ 注入（(G,2) float32，供 Decoder action_embedding 使用）
- node_id_to_idx dict 通过 __init__ 注入，替换 obsMap.patrol_nodes.index(...)
- 实现 get_config_dict / from_config_dict（满足框架 checkpoint 协议）
- self.input_dim / self.output_dim（必须）
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#                            GATEncoder
# =============================================================================

def _transpose_qkv(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch, num_agents, graph_size, hidden) -> (batch*num_heads, num_agents, graph_size, h/nh)"""
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_heads, -1)
    X = X.permute(0, 3, 1, 2, 4)
    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])


def _transpose_output(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """逆 _transpose_qkv，还原到 (batch, num_agents, graph_size, hidden)"""
    X = X.permute(2, 0, 1, 3)
    return X.reshape(X.shape[2] // num_heads, X.shape[0], X.shape[1], num_heads * X.shape[3])


class _DotProductGraphAttention(nn.Module):
    """图结构的点积注意力（只关注邻接节点）"""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, adj: torch.Tensor) -> torch.Tensor:
        e = torch.matmul(queries, keys.transpose(2, 3)) / 8.0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, values)
        return h_prime.view(h_prime.shape[2], h_prime.shape[0], h_prime.shape[1], h_prime.shape[3])


class _MultiHeadGraphAttention(nn.Module):
    def __init__(self, adj: torch.Tensor, key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = _DotProductGraphAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        # adj 存为 buffer（不是参数，但随 .to(device) 移动）
        self.register_buffer("adj", adj)

    def forward(self, queries, keys, values) -> torch.Tensor:
        queries = _transpose_qkv(self.W_q(queries), self.num_heads)
        keys = _transpose_qkv(self.W_k(keys), self.num_heads)
        values = _transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(queries, keys, values, self.adj)
        output_concat = _transpose_output(output, self.num_heads)
        return F.relu(self.W_o(output_concat))


class _AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class _PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class _EncoderBlock(nn.Module):
    def __init__(self, adj, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout):
        super().__init__()
        self.graph_attention = _MultiHeadGraphAttention(
            adj, key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = _AddNorm(norm_shape, dropout)
        self.ffn = _PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = _AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.graph_attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))


class GATEncoder(nn.Module):
    """图注意力 Encoder。

    输入: (batch, num_agents, graph_size, input_dim=3)
    输出:
        node_emb:    (batch, num_agents, graph_size, hidden_dim)
        state_value: (batch, num_agents, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        adj: np.ndarray,
        graph_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.graph_size = graph_size

        adj_t = torch.as_tensor(adj, dtype=torch.float32)  # (G, G)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"block{i}",
                _EncoderBlock(
                    adj_t, hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                    hidden_dim, hidden_dim, hidden_dim, num_heads, dropout,
                ),
            )
        self.state_value = nn.Linear(hidden_dim * graph_size, 1)
        # adj 注册为 buffer，供 _MultiHeadGraphAttention 内部 buffer 使用
        # 这里额外存一份，供 get_config_dict 序列化 shape 信息
        self.register_buffer("_adj_buf", adj_t)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: (batch, num_agents, graph_size, input_dim)
        Returns:
            node_emb:    (batch, num_agents, graph_size, hidden_dim)
            state_value: (batch, num_agents, 1)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=self._adj_buf.device)
        if X.ndim == 3:
            X = X.unsqueeze(0)  # (1, N, G, D)
        X = X.float()
        X = self.embedding(X)
        for blk in self.blks:
            X = blk(X)
        Y = X.contiguous().reshape(X.shape[0], X.shape[1], -1)  # (B, N, G*H)
        state_value = self.state_value(Y)  # (B, N, 1)
        return X, state_value

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": "GATEncoder",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "graph_size": self.graph_size,
            "adj": self._adj_buf.cpu().numpy().tolist(),
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "GATEncoder":
        adj = np.array(cfg["adj"], dtype=np.float32)
        return cls(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            adj=adj,
            graph_size=cfg["graph_size"],
        )


# =============================================================================
#                            MATDecoder
# =============================================================================

class _DotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, adj=None):
        attention = torch.matmul(queries, keys.transpose(2, 3)) / 16.0
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, values)
        return h_prime.view(h_prime.shape[2], h_prime.shape[0], h_prime.shape[1], h_prime.shape[3])


class _Attention_(nn.Module):
    """Cross-attention between node embeddings and action embedding"""

    def __init__(self, n_embd=32, n_head=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.key_ = nn.Linear(n_embd, n_embd)
        self.query_ = nn.Linear(n_embd, n_embd)
        self.value_ = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj2 = nn.Linear(n_embd, n_embd)

    def forward(self, query, key, value):
        B, L, G, D = key.size()
        K = self.key_(key).view(B, L, self.n_head, G, D // self.n_head).transpose(1, 2).transpose(2, 3)
        V = self.value_(value).view(B, L, self.n_head, G, D // self.n_head).transpose(1, 2).transpose(2, 3)
        Q = self.query_(query).view(B, L, self.n_head, G, D // self.n_head).transpose(1, 2).transpose(2, 3)
        att = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, V)
        y = y.transpose(2, 3).transpose(1, 2).contiguous().view(B, L, G, D)
        y = F.relu(self.proj(y))
        y = self.proj2(y)
        return y


class _SelfAttention_(nn.Module):
    """Self-attention over agent dimension"""

    def __init__(self, n_embd=32, n_head=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.key_ = nn.Linear(n_embd, n_embd)
        self.query_ = nn.Linear(n_embd, n_embd)
        self.value_ = nn.Linear(n_embd, n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj2 = nn.Linear(n_embd, n_embd)

    def forward(self, query, key, value):
        B, L, D = key.size()
        K = self.ln(key + self.key_(key)).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        V = self.ln(value + self.value_(value)).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        Q = self.ln(query + self.query_(query)).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        att = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, V)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = F.relu(self.proj2(y))
        y = self.ln(self.proj(y))
        return y


class MATDecoder(nn.Module):
    """Multi-Agent Transformer 自回归 Decoder。

    严格移植自 asy_decoder.py AttentionModel，仅做工程适配：
    - 移除 obsMap / env 全局依赖
    - adj + node_coords + node_id_to_idx 通过 __init__ 注入
    - decision_flag → active_mask (0/1)
    - shift_action 使用节点坐标 (B, N, N, 2) 格式（与原版完全一致）
    """

    def __init__(
        self,
        hidden_dim: int,
        n_agents: int,
        graph_size: int,
        n_heads: int,
        adj: np.ndarray,
        node_coords: np.ndarray,
        node_id_to_idx: Dict[int, int],
        sorted_nodes: List[int],
        tanh_clipping: float = 10.0,
        select_type: str = "sampling",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.graph_size = graph_size
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.select_type = select_type
        self.temp = 1.0
        self.input_dim = hidden_dim
        self.output_dim = graph_size

        # 保存映射关系（用于动作索引查询）
        self.register_buffer("_adj_buf", torch.as_tensor(adj, dtype=torch.float32))
        self.register_buffer(
            "_node_coords_buf", torch.as_tensor(node_coords, dtype=torch.float32)
        )
        self._node_id_to_idx = node_id_to_idx   # dict，不是 tensor
        self._sorted_nodes = sorted_nodes

        # 网络组件（与原版一致）
        self.project_node_embeddings = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.project_fixed_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.project_step_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.project_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.action_embedding = nn.Linear(2, hidden_dim)   # 2D 坐标 → hidden
        self.self_attn = _SelfAttention_(hidden_dim, n_heads)
        self.attn = _Attention_(hidden_dim, n_heads)
        self.ln = nn.LayerNorm(hidden_dim)

    # -------------------------------------------------------------------------

    def forward(
        self,
        node_emb: torch.Tensor,
        node_last_idx: np.ndarray,
        active_mask: np.ndarray,
        shift_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自回归解码所有 agent 的动作。

        Args:
            node_emb:    (B, N, G, H)  GATEncoder 输出
            node_last_idx: (B, N) int — 各 agent 当前所在节点的 graph index
            active_mask: (B, N) int/bool — 1=READY 需决策，0=ON_EDGE 跳过
            shift_action: (B, N, N, 2) float — 各 agent slot 的当前目标 2D 坐标

        Returns:
            log_prob_total: (B, N, G)
            action_total:   (B, N) int — graph index
            shift_action:   (B, N, N, 2) 更新后
        """
        B = node_emb.shape[0]
        G = self.graph_size
        N = self.n_agents

        log_prob_total = torch.zeros(B, N, G, device=node_emb.device)
        action_total = torch.zeros(B, N, dtype=torch.long, device=node_emb.device)

        active_mask_t = torch.as_tensor(
            active_mask, dtype=torch.bool, device=node_emb.device
        ).reshape(B, N)

        is_train = (B > 1)

        for agent in range(N):
            # action embedding for shift_action (B, N, 2) -> (B, N, H)
            action_embed = self.action_embedding(shift_action[:, agent, :, :])
            action_embed = self.self_attn(action_embed, action_embed, action_embed)

            # precompute
            action_embed_exp = action_embed.unsqueeze(2).expand(-1, -1, G, -1)
            enc_mod = self.ln(node_emb + self.attn(node_emb, action_embed_exp, action_embed_exp))
            graph_embed = enc_mod.mean(dim=-2)   # (B, N, H)
            fixed_context = self.project_fixed_context(graph_embed)[:, :, None, :]  # (B,N,1,H)

            glimpse_key_f, glimpse_val_f, logit_key_f = \
                self.project_node_embeddings(enc_mod).chunk(3, dim=-1)
            # (B, N, G, H) → split heads → (n_heads, B, N, G, H/n_heads)
            gk = self._make_heads(glimpse_key_f)
            gv = self._make_heads(glimpse_val_f)
            lk = logit_key_f.contiguous()  # (B, N, G, H)

            # recent node embedding for this agent
            idx_agent = node_last_idx[:, agent]  # (B,) int
            recent_node = torch.stack([
                enc_mod[b, agent, idx_agent[b], :] for b in range(B)
            ]).unsqueeze(1)  # (B, 1, H)

            query = (fixed_context[:, agent, :, :] +
                     self.project_step_context(recent_node))  # (B, 1, H)

            gK = gk[:, :, agent, :, :]  # (n_heads, B, G, H/nh)
            gV = gv[:, :, agent, :, :]
            lK = lk[:, agent, :, :]     # (B, G, H)

            gQ = query.view(B, self.n_heads, 1, self.hidden_dim // self.n_heads).permute(1, 0, 2, 3)

            dim = gK.size(-1)
            weight = torch.matmul(gQ, gK.transpose(-2, -1)) / math.sqrt(dim)  # (nh,B,1,G)

            # 邻接掩码
            adj_row = []
            for b in range(B):
                adj_row.append(self._adj_buf[idx_agent[b]])  # (G,)
            adj_m = torch.stack(adj_row).unsqueeze(1)  # (B,1,G)

            zero_vec = -9e15 * torch.ones_like(weight)
            weight = torch.where(adj_m.unsqueeze(0) > 0, weight, zero_vec)

            score = torch.matmul(F.softmax(weight, dim=-1), gV)  # (nh,B,1,H/nh)
            final_Q = self.project_out(
                score.permute(1, 2, 0, 3).contiguous().view(B, 1, self.n_heads * score.shape[-1])
            )  # (B, 1, H)

            logits = torch.matmul(final_Q, lK.transpose(-2, -1)) / math.sqrt(dim)  # (B,1,G)
            logits = torch.tanh(logits) * self.tanh_clipping
            logits = torch.where(adj_m > 0, logits, -9e15 * torch.ones_like(logits))
            log_prob = torch.log_softmax(logits / self.temp, dim=-1)  # (B,1,G)

            prob = log_prob.exp().squeeze(1)  # (B, G)
            selected = self._select_node(prob)  # (B,)

            # 自回归更新：将已选动作写入 shift_action 的下一个 slot
            if agent < N - 1:
                if not is_train:
                    # 推理：决策 agent 写入坐标；非决策 agent 沿用上一个 slot
                    shift_action[:, agent + 1, :, :] = shift_action[:, agent, :, :].clone()
                    for b in range(B):
                        if active_mask_t[b, agent]:
                            shift_action[b, agent + 1, agent + 1, :] = \
                                self._node_coords_buf[selected[b].item()]
                        else:
                            # 非决策 agent：log_prob 置为 -inf，selected 沿用 shift_action
                            next_coord = shift_action[b, agent, agent + 1, :]
                            nidx = self._coord_to_idx(next_coord)
                            selected[b] = nidx
                            log_prob[b, :, :] = -1e8
                else:
                    # 训练（teacher forcing）：同原文逻辑
                    log_prob = log_prob.clone()
                    for b in range(B):
                        if not active_mask_t[b, agent]:
                            log_prob[b, :, :] = -1e8
                            next_coord = shift_action[b, agent, agent + 1, :]
                            selected[b] = self._coord_to_idx(next_coord)

            log_prob = log_prob.reshape(B, G)
            log_prob_total[:, agent, :] = log_prob
            action_total[:, agent] = selected

        return log_prob_total, action_total, shift_action

    def _make_heads(self, v: torch.Tensor) -> torch.Tensor:
        """(B, N, G, H) -> (n_heads, B, N, G, H/n_heads)"""
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )

    def _select_node(self, probs: torch.Tensor) -> torch.Tensor:
        """(B, G) -> (B,) int"""
        if self.select_type == "greedy":
            return probs.argmax(dim=-1)
        else:
            return probs.multinomial(1).squeeze(-1)

    def _coord_to_idx(self, coord: torch.Tensor) -> int:
        """将 2D 坐标近似匹配到最近的节点 index。"""
        coords_all = self._node_coords_buf  # (G, 2)
        dists = ((coords_all - coord.unsqueeze(0)) ** 2).sum(-1)
        return int(dists.argmin().item())

    def get_action_idx(self, node_id: int) -> int:
        """node_id -> graph index（供算法层 log_prob 索引）"""
        return self._node_id_to_idx.get(node_id, 0)

    def node_idx_to_coord(self, idx: int) -> torch.Tensor:
        """graph index -> 2D 坐标 (2,)"""
        return self._node_coords_buf[idx]

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": "MATDecoder",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "n_agents": self.n_agents,
            "graph_size": self.graph_size,
            "n_heads": self.n_heads,
            "tanh_clipping": self.tanh_clipping,
            "select_type": self.select_type,
            "adj": self._adj_buf.cpu().numpy().tolist(),
            "node_coords": self._node_coords_buf.cpu().numpy().tolist(),
            "sorted_nodes": self._sorted_nodes,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MATDecoder":
        adj = np.array(cfg["adj"], dtype=np.float32)
        node_coords = np.array(cfg["node_coords"], dtype=np.float32)
        sorted_nodes = list(cfg["sorted_nodes"])
        node_id_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
        return cls(
            hidden_dim=cfg["hidden_dim"],
            n_agents=cfg["n_agents"],
            graph_size=cfg["graph_size"],
            n_heads=cfg["n_heads"],
            adj=adj,
            node_coords=node_coords,
            node_id_to_idx=node_id_to_idx,
            sorted_nodes=sorted_nodes,
            tanh_clipping=cfg.get("tanh_clipping", 10.0),
            select_type=cfg.get("select_type", "sampling"),
        )
