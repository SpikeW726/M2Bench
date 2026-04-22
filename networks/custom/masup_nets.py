"""MASUP 环境专供网络（与 MASUPEnv obs 结构约定绑定）。

与通用网络（mlp.py / rnn.py）的区别：
- 原始 obs/state 向量在进入 MLP/RNN 主干前先经 MASUPPreprocessor 异构处理
- 处理策略：节点 index → GPE Embed + Linear，连续量 → RunningMeanStd，
  obs_timer → /T_time，二值/类别量 → 直通

包含网络类：
  MASUPActorMLP       actor（所有 on-policy 算法，MLP）
  MASUPActorRNN       actor（RNN）
  MASUPCriticMLP      MAPPO / IPPO critic（MLP，input_mode 区分）
  MASUPCriticRNN      MAPPO / IPPO critic（RNN）
  MASUPQMLP           IQL / VDN / QMIX Q-network（MLP）
  MASUPQRNN           IQL Q-network（RNN）
  MASUPVDPPOQmlp      VDPPO Q-network（MLP，接 state|one_hot）
  MASUPVDPPOQrnn      VDPPO Q-network（RNN，接 state|one_hot|prev_act）
"""

import json
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from networks.mlp import layer_init
from utils.train_utils import RunningMeanStd


# =============================================================================
#                          GPE 辅助计算
# =============================================================================

def _compute_gpe(graph_json_path: str, gpe_dim: int) -> Tuple[np.ndarray, list]:
    """计算 Laplacian Eigenmaps（图位置编码）。

    Returns:
        gpe_matrix: (num_nodes, gpe_dim) float32
        node_order:  图 JSON nodes 列表（与 MASUPEnv._obs_node_order 一致）
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    with open(graph_json_path) as f:
        g = json.load(f)

    node_order = list(g["nodes"])
    n = len(node_order)
    node_to_idx = {nid: i for i, nid in enumerate(node_order)}

    # 对称邻接矩阵（有向图边双向叠加）
    A = np.zeros((n, n), dtype=np.float32)
    for e in g["edges"]:
        i = node_to_idx[e["from"]]
        j = node_to_idx[e["to"]]
        A[i, j] += 1.0
        A[j, i] += 1.0

    # 归一化 Laplacian: L = I - D^{-1/2} A D^{-1/2}
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # 取 gpe_dim+1 个最小特征向量，跳过第 0 列（常数分量）
    k = min(gpe_dim + 1, n - 1)
    L_sp = sparse.csr_matrix(L_sym.astype(np.float64))
    _, eigvecs = eigsh(L_sp, k=k, which="SM")

    # 跳过第 0 个特征向量，取 gpe_dim 个
    gpe = eigvecs[:, 1 : gpe_dim + 1].astype(np.float32)

    # 若可用特征向量不足 gpe_dim，右侧补零
    if gpe.shape[1] < gpe_dim:
        pad = np.zeros((n, gpe_dim - gpe.shape[1]), dtype=np.float32)
        gpe = np.concatenate([gpe, pad], axis=1)

    return gpe, node_order


# =============================================================================
#                          核心预处理模块
# =============================================================================

class MASUPPreprocessor(nn.Module):
    """MASUP obs / state 的异构字段预处理器。

    mode="actor"  : 处理 MASUPEnv 的 per-agent obs（含 ready_flag + role_info）
    mode="state"  : 处理 state() + agent_one_hot（MAPPO Critic 输入）
    mode="vdppo_q": 处理 state() 前段（VDPPO Q-net；state_dim 后直通）

    GPE Embedding frozen，不参与梯度；node_proj 可训练。
    RMS 统计量在 self.training=True 时自动更新，eval 时冻结。
    """

    def __init__(
        self,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_matrix: np.ndarray,      # (num_nodes, gpe_dim)
        node_order: list,            # 图 JSON 中的 nodes 列表
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        mode: str,                   # "actor" | "state" | "vdppo_q"
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.role_imformation = role_imformation
        self.proj_dim = proj_dim
        self.use_log_idleness = use_log_idleness
        self.T_time = float(T_time)
        self.mode = mode

        gpe_dim = gpe_matrix.shape[1]
        self.gpe_dim = gpe_dim

        # --- GPE Embedding（冻结）+ 线性投影（可训练，无激活）---
        gpe_tensor = torch.from_numpy(gpe_matrix)
        self.node_embedding = nn.Embedding.from_pretrained(gpe_tensor, freeze=True)
        self.node_proj = nn.Linear(gpe_dim, proj_dim)

        # --- node_id → embed_idx 映射（应对非 0 起始或稀疏节点 ID）---
        max_id = max(node_order)
        id_remap = torch.zeros(max_id + 1, dtype=torch.long)
        for embed_idx, node_id in enumerate(node_order):
            id_remap[node_id] = embed_idx
        self.register_buffer("id_remap", id_remap)

        # --- 连续字段 RMS（nn.Module，buffers 随 state_dict 保存/加载）---
        self.latency_rms = RunningMeanStd(shape=(num_agents,))
        self.idle_node_rms = RunningMeanStd(shape=(num_nodes,))
        self.worst_rms = RunningMeanStd(shape=())

        # --- 预计算各字段在 obs 中的位置索引 ---
        self._build_index_maps()

    # ------------------------------------------------------------------
    # 索引映射构建
    # ------------------------------------------------------------------

    def _build_index_maps(self):
        """根据 obs 布局预计算各字段的 slice 索引，存为 buffer（整型，不做梯度）。"""
        N = self.num_agents
        K = self.num_nodes

        # last_pos 和 target 在 shared 前缀中的散布位置
        node_positions = []
        for i in range(N):
            node_positions.append(3 * i)        # last_pos_i
            node_positions.append(3 * i + 1)    # target_i

        latency_positions = [3 * i + 2 for i in range(N)]

        idle_start = 3 * N
        idle_end = 3 * N + K

        if self.mode == "actor":
            # tail: ready_flag | worst | timer | role_info
            ready_idx = 3 * N + K
            worst_idx = 3 * N + K + 1
            timer_idx = 3 * N + K + 2

            if self.role_imformation == "agent-index":
                # tail role: one-hot N 维，直通
                passthrough_positions = [ready_idx] + list(range(3 * N + K + 3, 4 * N + K + 3))
            elif self.role_imformation == "position":
                # tail role: float(ag.position) → 也是节点 index，需 embed
                node_positions.append(3 * N + K + 3)   # ag.position
                passthrough_positions = [ready_idx]
            elif self.role_imformation == "decision":
                # tail role: float(ag.position) + one-hot N
                node_positions.append(3 * N + K + 3)
                passthrough_positions = [ready_idx] + list(range(3 * N + K + 4, 4 * N + K + 4))
            else:
                passthrough_positions = [ready_idx]

        elif self.mode in ("state", "vdppo_q"):
            # state() 布局：shared prefix + worst + timer（无 ready_flag）
            worst_idx = 3 * N + K
            timer_idx = 3 * N + K + 1
            # agent_one_hot（state 模式）或 one_hot+prev_act（vdppo_q）均在 state_dim 之后
            # 这里 passthrough 仅包含 state_dim 之后的部分，由网络类在 forward 里拼接
            passthrough_positions = []

        else:
            raise ValueError(f"未知 MASUPPreprocessor mode: {self.mode}")

        # 存为 buffer（.to(device) 时自动迁移，不参与梯度）
        self.register_buffer(
            "node_idx_buf",
            torch.tensor(node_positions, dtype=torch.long),
        )
        self.register_buffer(
            "latency_buf",
            torch.tensor(latency_positions, dtype=torch.long),
        )
        self.idle_start = idle_start
        self.idle_end = idle_end
        self.worst_idx = worst_idx
        self.timer_idx = timer_idx
        self.register_buffer(
            "passthrough_buf",
            torch.tensor(passthrough_positions, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # output_dim
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """预处理后的输出维度（供下游 backbone 的 input_dim 使用）。"""
        N = self.num_agents
        K = self.num_nodes

        num_node_indices = len(self.node_idx_buf)
        node_feat_dim = num_node_indices * self.proj_dim
        passthrough_dim = len(self.passthrough_buf)

        return node_feat_dim + N + K + 1 + 1 + passthrough_dim

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, update_rms: Optional[bool] = None) -> torch.Tensor:
        """
        x: (batch, input_dim)
        update_rms: None 时跟随 self.training（训练时更新，eval 时不更新）
        Returns: (batch, output_dim)
        """
        if update_rms is None:
            update_rms = self.training

        B = x.shape[0]

        # --- 1. 节点 index → GPE Embedding → Linear 投影 ---
        node_ids = x[:, self.node_idx_buf].long()       # (B, cnt)
        node_ids = self.id_remap[node_ids]              # 映射到 0..K-1
        gpe = self.node_embedding(node_ids)             # (B, cnt, gpe_dim)
        projected = self.node_proj(gpe).reshape(B, -1)  # (B, cnt*proj_dim)

        # --- 2. latency RMS ---
        latency = x[:, self.latency_buf]                # (B, N)
        if update_rms:
            self.latency_rms.update(latency)
        lat_n = self.latency_rms.normalize(latency)

        # --- 3. weighted_idle RMS ---
        idle = x[:, self.idle_start : self.idle_end]    # (B, K)
        if update_rms:
            self.idle_node_rms.update(idle)
        idle_n = self.idle_node_rms.normalize(idle)

        # --- 4. worst_idleness（可选 log1p）+ RMS ---
        worst = x[:, self.worst_idx : self.worst_idx + 1]   # (B, 1)
        w = torch.log1p(worst) if self.use_log_idleness else worst
        if update_rms:
            self.worst_rms.update(w.squeeze(-1))
        worst_n = self.worst_rms.normalize(w)

        # --- 5. obs_timer → / T_time ---
        timer = x[:, self.timer_idx : self.timer_idx + 1]   # (B, 1)
        timer_n = timer / self.T_time if self.T_time > 0.0 else timer

        # --- 6. pass-through（ready_flag, role_info, agent_one_hot 等）---
        if len(self.passthrough_buf) > 0:
            pt = x[:, self.passthrough_buf]
            return torch.cat([projected, lat_n, idle_n, worst_n, timer_n, pt], dim=-1)

        return torch.cat([projected, lat_n, idle_n, worst_n, timer_n], dim=-1)


# =============================================================================
#                       共享工厂辅助函数
# =============================================================================

def _make_preprocessor(
    graph_path: str,
    num_agents: int,
    num_nodes: int,
    role_imformation: str,
    gpe_dim: int,
    proj_dim: int,
    use_log_idleness: bool,
    T_time: float,
    mode: str,
) -> MASUPPreprocessor:
    gpe_matrix, node_order = _compute_gpe(graph_path, gpe_dim)
    n_graph = len(node_order)
    # YAML 常省略 num_nodes 或沿用 MASUPBaseConfig 默认 12；与图不一致会导致切片错位、Embedding 越界
    if num_nodes != n_graph:
        warnings.warn(
            f"MASUP: num_nodes={num_nodes} 与 {graph_path} 节点数 {n_graph} 不一致，已按图修正。",
            UserWarning,
            stacklevel=2,
        )
        num_nodes = n_graph
    return MASUPPreprocessor(
        num_agents=num_agents,
        num_nodes=num_nodes,
        role_imformation=role_imformation,
        gpe_matrix=gpe_matrix,
        node_order=node_order,
        proj_dim=proj_dim,
        use_log_idleness=use_log_idleness,
        T_time=T_time,
        mode=mode,
    )


def _build_mlp_layers(input_dim: int, hidden: List[int], output_dim: int, output_std: float) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden:
        layers.append(layer_init(nn.Linear(prev, h), std=np.sqrt(2)))
        layers.append(nn.Tanh())
        prev = h
    layers.append(layer_init(nn.Linear(prev, output_dim), std=output_std))
    return nn.Sequential(*layers)


# =============================================================================
#                       MASUPActorMLP
# =============================================================================

class MASUPActorMLP(nn.Module):
    """MASUP Actor（MLP），适用于所有 on-policy 算法。"""
    is_recurrent = False

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden: List[int],
        output_dim: int,
        input_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._role_imformation = role_imformation
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time
        self._hidden = list(hidden)

        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, role_imformation,
            gpe_dim, proj_dim, use_log_idleness, T_time, mode="actor",
        )
        self._num_nodes = self.preprocessor.num_nodes
        self.network = _build_mlp_layers(
            self.preprocessor.output_dim, hidden, output_dim, output_std=0.01,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.preprocessor(x))

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "role_imformation": self._role_imformation,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden": self._hidden,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPActorMLP":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            role_imformation=cfg["role_imformation"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden=cfg["hidden"],
            output_dim=cfg["output_dim"],
            input_dim=cfg["input_dim"],
        )


# =============================================================================
#                       MASUPActorRNN
# =============================================================================

class MASUPActorRNN(nn.Module):
    """MASUP Actor（GRU/LSTM），适用于 on-policy 算法 RNN 配置。"""
    is_recurrent = True

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden_size: int,
        output_dim: int,
        input_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        fc_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self._fc_hidden = list(fc_hidden) if fc_hidden else []
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._role_imformation = role_imformation
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time

        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, role_imformation,
            gpe_dim, proj_dim, use_log_idleness, T_time, mode="actor",
        )
        self._num_nodes = self.preprocessor.num_nodes
        prep_out = self.preprocessor.output_dim

        # fc_in 编码层
        fc_sizes = list(fc_hidden) if fc_hidden else [hidden_size]
        fc_layers = []
        prev = prep_out
        for sz in fc_sizes:
            fc_layers.append(layer_init(nn.Linear(prev, sz)))
            fc_layers.append(nn.Tanh())
            prev = sz
        self.fc_in = nn.Sequential(*fc_layers)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(fc_sizes[-1], hidden_size, num_layers, batch_first=False)
        for name, param in self.rnn.named_parameters():
            nn.init.constant_(param, 0) if "bias" in name else nn.init.orthogonal_(param)

        self.fc_out = layer_init(nn.Linear(hidden_size, output_dim), std=0.01)

    @property
    def recurrent_N(self) -> int:
        return self.num_layers * (2 if self.rnn_type == "lstm" else 1)

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.recurrent_N, batch_size, self.hidden_size, device=device)

    def _to_rnn_state(self, h: torch.Tensor):
        if self.rnn_type == "lstm":
            a, b = h.chunk(2, dim=0)
            return a.contiguous(), b.contiguous()
        return h.contiguous()

    def _from_rnn_state(self, s) -> torch.Tensor:
        return torch.cat(s, dim=0) if self.rnn_type == "lstm" else s

    def _rnn_forward(self, x: torch.Tensor, hidden: torch.Tensor):
        if hasattr(self.rnn, "flatten_parameters"):
            p0 = next(self.rnn.parameters(), None)
            if p0 is not None and p0.is_cuda:
                self.rnn.flatten_parameters()
        out, new_h = self.rnn(x, self._to_rnn_state(hidden))
        return out, self._from_rnn_state(new_h)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor = None):
        if hidden_state is None:
            hidden_state = self.get_initial_hidden(obs.shape[0], obs.device)
        feat = self.preprocessor(obs)
        x = self.fc_in(feat).unsqueeze(0)
        rnn_out, new_h = self._rnn_forward(x, hidden_state)
        return self.fc_out(rnn_out.squeeze(0)), new_h

    def forward_sequence(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor):
        T, B, _ = obs_seq.shape
        feat = self.preprocessor(obs_seq.reshape(T * B, -1), update_rms=self.training)
        x = self.fc_in(feat).view(T, B, -1)
        rnn_out, final_h = self._rnn_forward(x, hidden_state)
        out = self.fc_out(rnn_out.reshape(T * B, -1)).view(T, B, -1)
        return out, final_h

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "role_imformation": self._role_imformation,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "fc_hidden": self._fc_hidden,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPActorRNN":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            role_imformation=cfg["role_imformation"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden_size=cfg["hidden_size"],
            output_dim=cfg["output_dim"],
            input_dim=cfg["input_dim"],
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            fc_hidden=cfg.get("fc_hidden") or None,
        )


# =============================================================================
#                       MASUPCriticMLP / MASUPCriticRNN
# =============================================================================

class MASUPCriticMLP(nn.Module):
    """MASUP Critic（MLP）。

    input_mode="state"  : MAPPO，接 state() + agent_one_hot
    input_mode="actor"  : IPPO，接 per-agent obs
    """
    is_recurrent = False

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden: List[int],
        input_dim: int,
        input_mode: str = "state",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._role_imformation = role_imformation
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time
        self._hidden = list(hidden)
        self._input_mode = input_mode

        mode = "state" if input_mode == "state" else "actor"
        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, role_imformation,
            gpe_dim, proj_dim, use_log_idleness, T_time, mode=mode,
        )
        self._num_nodes = self.preprocessor.num_nodes

        # state 模式下，state_dim 之后是 agent_one_hot（pass-through 已在 preprocessor 处理）
        # agent_one_hot 在 state 模式中由 preprocessor passthrough 处理不了（字段在 state_dim 后）
        # 因此 critic 的 forward 里手动拼接
        if input_mode == "state":
            self._state_dim = 3 * num_agents + self._num_nodes + 2
            net_input = self.preprocessor.output_dim + num_agents
        else:
            self._state_dim = None
            net_input = self.preprocessor.output_dim

        self.network = _build_mlp_layers(net_input, hidden, 1, output_std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._input_mode == "state":
            state_part = x[:, : self._state_dim]
            one_hot = x[:, self._state_dim :]
            feat = torch.cat([self.preprocessor(state_part), one_hot], dim=-1)
        else:
            feat = self.preprocessor(x)
        return self.network(feat)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "role_imformation": self._role_imformation,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden": self._hidden,
            "input_mode": self._input_mode,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPCriticMLP":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            role_imformation=cfg["role_imformation"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden=cfg["hidden"],
            input_dim=cfg["input_dim"],
            input_mode=cfg.get("input_mode", "state"),
        )


class MASUPCriticRNN(nn.Module):
    """MASUP Critic（GRU/LSTM）。input_mode 同 MASUPCriticMLP。"""
    is_recurrent = True

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden_size: int,
        input_dim: int,
        input_mode: str = "state",
        num_layers: int = 1,
        rnn_type: str = "gru",
        fc_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self._fc_hidden = list(fc_hidden) if fc_hidden else []
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._role_imformation = role_imformation
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time
        self._input_mode = input_mode

        mode = "state" if input_mode == "state" else "actor"
        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, role_imformation,
            gpe_dim, proj_dim, use_log_idleness, T_time, mode=mode,
        )
        self._num_nodes = self.preprocessor.num_nodes

        if input_mode == "state":
            self._state_dim = 3 * num_agents + self._num_nodes + 2
            rnn_input_base = self.preprocessor.output_dim + num_agents
        else:
            self._state_dim = None
            rnn_input_base = self.preprocessor.output_dim

        fc_sizes = list(fc_hidden) if fc_hidden else [hidden_size]
        fc_layers = []
        prev = rnn_input_base
        for sz in fc_sizes:
            fc_layers.append(layer_init(nn.Linear(prev, sz)))
            fc_layers.append(nn.Tanh())
            prev = sz
        self.fc_in = nn.Sequential(*fc_layers)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(fc_sizes[-1], hidden_size, num_layers, batch_first=False)
        for name, param in self.rnn.named_parameters():
            nn.init.constant_(param, 0) if "bias" in name else nn.init.orthogonal_(param)

        self.fc_out = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    @property
    def recurrent_N(self) -> int:
        return self.num_layers * (2 if self.rnn_type == "lstm" else 1)

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.recurrent_N, batch_size, self.hidden_size, device=device)

    def _to_rnn_state(self, h):
        if self.rnn_type == "lstm":
            a, b = h.chunk(2, dim=0)
            return a.contiguous(), b.contiguous()
        return h.contiguous()

    def _from_rnn_state(self, s) -> torch.Tensor:
        return torch.cat(s, dim=0) if self.rnn_type == "lstm" else s

    def _prep(self, x: torch.Tensor, update_rms: bool) -> torch.Tensor:
        if self._input_mode == "state":
            state_part = x[:, : self._state_dim]
            one_hot = x[:, self._state_dim :]
            return torch.cat([self.preprocessor(state_part, update_rms=update_rms), one_hot], dim=-1)
        return self.preprocessor(x, update_rms=update_rms)

    def _rnn_forward(self, x, hidden):
        if hasattr(self.rnn, "flatten_parameters"):
            p0 = next(self.rnn.parameters(), None)
            if p0 is not None and p0.is_cuda:
                self.rnn.flatten_parameters()
        out, new_h = self.rnn(x, self._to_rnn_state(hidden))
        return out, self._from_rnn_state(new_h)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor = None):
        if hidden_state is None:
            hidden_state = self.get_initial_hidden(obs.shape[0], obs.device)
        feat = self._prep(obs, update_rms=self.training)
        x = self.fc_in(feat).unsqueeze(0)
        rnn_out, new_h = self._rnn_forward(x, hidden_state)
        return self.fc_out(rnn_out.squeeze(0)), new_h

    def forward_sequence(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor):
        T, B, _ = obs_seq.shape
        flat = obs_seq.reshape(T * B, -1)
        feat = self._prep(flat, update_rms=self.training)
        x = self.fc_in(feat).view(T, B, -1)
        rnn_out, final_h = self._rnn_forward(x, hidden_state)
        out = self.fc_out(rnn_out.reshape(T * B, -1)).view(T, B, -1)
        return out, final_h

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "role_imformation": self._role_imformation,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "fc_hidden": self._fc_hidden,
            "input_mode": self._input_mode,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPCriticRNN":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            role_imformation=cfg["role_imformation"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden_size=cfg["hidden_size"],
            input_dim=cfg["input_dim"],
            input_mode=cfg.get("input_mode", "state"),
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            fc_hidden=cfg.get("fc_hidden") or None,
        )


# =============================================================================
#                       MASUPQMLP / MASUPQRNN  (IQL / VDN / QMIX)
# =============================================================================

class MASUPQMLP(nn.Module):
    """MASUP Q-network（MLP），用于 IQL / VDN / QMIX，接 actor obs。"""
    is_recurrent = False

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden: List[int],
        output_dim: int,
        input_dim: int,
        dueling: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dueling = dueling
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._role_imformation = role_imformation
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time
        self._hidden = list(hidden)

        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, role_imformation,
            gpe_dim, proj_dim, use_log_idleness, T_time, mode="actor",
        )
        self._num_nodes = self.preprocessor.num_nodes
        prev = self.preprocessor.output_dim
        shared_layers = []
        for h in hidden:
            shared_layers.append(layer_init(nn.Linear(prev, h), std=np.sqrt(2)))
            shared_layers.append(nn.Tanh())
            prev = h
        self.shared = nn.Sequential(*shared_layers)

        if dueling:
            self.v_stream = layer_init(nn.Linear(prev, 1), std=1.0)
            self.a_stream = layer_init(nn.Linear(prev, output_dim), std=1.0)
        else:
            self.q_head = layer_init(nn.Linear(prev, output_dim), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(self.preprocessor(x))
        if self.dueling:
            v = self.v_stream(feat)
            a = self.a_stream(feat)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.q_head(feat)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "role_imformation": self._role_imformation,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden": self._hidden,
            "dueling": self.dueling,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPQMLP":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            role_imformation=cfg["role_imformation"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden=cfg["hidden"],
            output_dim=cfg["output_dim"],
            input_dim=cfg["input_dim"],
            dueling=cfg.get("dueling", False),
        )


class MASUPQRNN(nn.Module):
    """MASUP Q-network（GRU/LSTM），用于 IQL RNN 配置，接 actor obs。"""
    is_recurrent = True

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        role_imformation: str,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden_size: int,
        output_dim: int,
        input_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dueling: bool = False,
        fc_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dueling = dueling
        self._fc_hidden = list(fc_hidden) if fc_hidden else []
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._role_imformation = role_imformation
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time

        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, role_imformation,
            gpe_dim, proj_dim, use_log_idleness, T_time, mode="actor",
        )
        self._num_nodes = self.preprocessor.num_nodes
        fc_sizes = list(fc_hidden) if fc_hidden else [hidden_size]
        fc_layers = []
        prev = self.preprocessor.output_dim
        for sz in fc_sizes:
            fc_layers.append(layer_init(nn.Linear(prev, sz)))
            fc_layers.append(nn.Tanh())
            prev = sz
        self.fc_in = nn.Sequential(*fc_layers)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(fc_sizes[-1], hidden_size, num_layers, batch_first=False)
        for name, param in self.rnn.named_parameters():
            nn.init.constant_(param, 0) if "bias" in name else nn.init.orthogonal_(param)

        if dueling:
            self.v_stream = layer_init(nn.Linear(hidden_size, 1), std=1.0)
            self.a_stream = layer_init(nn.Linear(hidden_size, output_dim), std=1.0)
        else:
            self.fc_out = layer_init(nn.Linear(hidden_size, output_dim), std=1.0)

    @property
    def recurrent_N(self) -> int:
        return self.num_layers * (2 if self.rnn_type == "lstm" else 1)

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.recurrent_N, batch_size, self.hidden_size, device=device)

    def _to_rnn_state(self, h):
        if self.rnn_type == "lstm":
            a, b = h.chunk(2, dim=0)
            return a.contiguous(), b.contiguous()
        return h.contiguous()

    def _from_rnn_state(self, s):
        return torch.cat(s, dim=0) if self.rnn_type == "lstm" else s

    def _head(self, rnn_out: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            v = self.v_stream(rnn_out)
            a = self.a_stream(rnn_out)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.fc_out(rnn_out)

    def _rnn_forward(self, x, hidden):
        if hasattr(self.rnn, "flatten_parameters"):
            p0 = next(self.rnn.parameters(), None)
            if p0 is not None and p0.is_cuda:
                self.rnn.flatten_parameters()
        out, new_h = self.rnn(x, self._to_rnn_state(hidden))
        return out, self._from_rnn_state(new_h)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor = None):
        if hidden_state is None:
            hidden_state = self.get_initial_hidden(obs.shape[0], obs.device)
        feat = self.preprocessor(obs)
        x = self.fc_in(feat).unsqueeze(0)
        rnn_out, new_h = self._rnn_forward(x, hidden_state)
        return self._head(rnn_out.squeeze(0)), new_h

    def forward_sequence(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor):
        T, B, _ = obs_seq.shape
        feat = self.preprocessor(obs_seq.reshape(T * B, -1), update_rms=self.training)
        x = self.fc_in(feat).view(T, B, -1)
        rnn_out, final_h = self._rnn_forward(x, hidden_state)
        out = self._head(rnn_out.reshape(T * B, -1)).view(T, B, -1)
        return out, final_h

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "role_imformation": self._role_imformation,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "fc_hidden": self._fc_hidden,
            "dueling": self.dueling,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPQRNN":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            role_imformation=cfg["role_imformation"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden_size=cfg["hidden_size"],
            output_dim=cfg["output_dim"],
            input_dim=cfg["input_dim"],
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            dueling=cfg.get("dueling", False),
            fc_hidden=cfg.get("fc_hidden") or None,
        )


# =============================================================================
#                       MASUPVDPPOQmlp / MASUPVDPPOQrnn  (VDPPO)
# =============================================================================

class MASUPVDPPOQmlp(nn.Module):
    """VDPPO Q-network（MLP），接 [state() | one_hot_i] 拼接输入。

    前 state_dim = 3*N+K+2 维走 state 预处理器，后续 N 维（one_hot）直通。
    """
    is_recurrent = False

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden: List[int],
        output_dim: int,
        input_dim: int,
        dueling: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dueling = dueling
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time
        self._hidden = list(hidden)

        # role_imformation 在 state 模式下无关，传 "agent-index" 占位
        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, "agent-index",
            gpe_dim, proj_dim, use_log_idleness, T_time, mode="state",
        )
        self._num_nodes = self.preprocessor.num_nodes
        self._state_dim = 3 * num_agents + self._num_nodes + 2
        net_input = self.preprocessor.output_dim + num_agents  # + one_hot passthrough

        prev = net_input
        shared_layers = []
        for h in hidden:
            shared_layers.append(layer_init(nn.Linear(prev, h), std=np.sqrt(2)))
            shared_layers.append(nn.Tanh())
            prev = h
        self.shared = nn.Sequential(*shared_layers)

        if dueling:
            self.v_stream = layer_init(nn.Linear(prev, 1), std=1.0)
            self.a_stream = layer_init(nn.Linear(prev, output_dim), std=1.0)
        else:
            self.q_head = layer_init(nn.Linear(prev, output_dim), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_part = x[:, : self._state_dim]
        passthrough = x[:, self._state_dim :]    # one_hot_i
        feat = torch.cat([self.preprocessor(state_part), passthrough], dim=-1)
        h = self.shared(feat)
        if self.dueling:
            v = self.v_stream(h)
            a = self.a_stream(h)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.q_head(h)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden": self._hidden,
            "dueling": self.dueling,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPVDPPOQmlp":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden=cfg["hidden"],
            output_dim=cfg["output_dim"],
            input_dim=cfg["input_dim"],
            dueling=cfg.get("dueling", False),
        )


class MASUPVDPPOQrnn(nn.Module):
    """VDPPO Q-network（GRU/LSTM），接 [state() | one_hot_i | prev_act_oh_i]。

    前 state_dim 维走 state 预处理器，后续 N+action_dim 维直通。
    """
    is_recurrent = True

    def __init__(
        self,
        graph_path: str,
        num_agents: int,
        num_nodes: int,
        gpe_dim: int,
        proj_dim: int,
        use_log_idleness: bool,
        T_time: float,
        hidden_size: int,
        output_dim: int,
        input_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dueling: bool = False,
        fc_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dueling = dueling
        self._fc_hidden = list(fc_hidden) if fc_hidden else []
        self._graph_path = graph_path
        self._num_agents = num_agents
        self._gpe_dim = gpe_dim
        self._proj_dim = proj_dim
        self._use_log_idleness = use_log_idleness
        self._T_time = T_time

        self.preprocessor = _make_preprocessor(
            graph_path, num_agents, num_nodes, "agent-index",
            gpe_dim, proj_dim, use_log_idleness, T_time, mode="state",
        )
        self._num_nodes = self.preprocessor.num_nodes
        self._state_dim = 3 * num_agents + self._num_nodes + 2
        passthrough_dim = input_dim - self._state_dim   # N + action_dim
        rnn_input_base = self.preprocessor.output_dim + passthrough_dim

        fc_sizes = list(fc_hidden) if fc_hidden else [hidden_size]
        fc_layers = []
        prev = rnn_input_base
        for sz in fc_sizes:
            fc_layers.append(layer_init(nn.Linear(prev, sz)))
            fc_layers.append(nn.Tanh())
            prev = sz
        self.fc_in = nn.Sequential(*fc_layers)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(fc_sizes[-1], hidden_size, num_layers, batch_first=False)
        for name, param in self.rnn.named_parameters():
            nn.init.constant_(param, 0) if "bias" in name else nn.init.orthogonal_(param)

        if dueling:
            self.v_stream = layer_init(nn.Linear(hidden_size, 1), std=1.0)
            self.a_stream = layer_init(nn.Linear(hidden_size, output_dim), std=1.0)
        else:
            self.fc_out = layer_init(nn.Linear(hidden_size, output_dim), std=1.0)

    @property
    def recurrent_N(self) -> int:
        return self.num_layers * (2 if self.rnn_type == "lstm" else 1)

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.recurrent_N, batch_size, self.hidden_size, device=device)

    def _to_rnn_state(self, h):
        if self.rnn_type == "lstm":
            a, b = h.chunk(2, dim=0)
            return a.contiguous(), b.contiguous()
        return h.contiguous()

    def _from_rnn_state(self, s):
        return torch.cat(s, dim=0) if self.rnn_type == "lstm" else s

    def _head(self, rnn_out: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            v = self.v_stream(rnn_out)
            a = self.a_stream(rnn_out)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.fc_out(rnn_out)

    def _rnn_forward(self, x, hidden):
        if hasattr(self.rnn, "flatten_parameters"):
            p0 = next(self.rnn.parameters(), None)
            if p0 is not None and p0.is_cuda:
                self.rnn.flatten_parameters()
        out, new_h = self.rnn(x, self._to_rnn_state(hidden))
        return out, self._from_rnn_state(new_h)

    def _prep(self, x: torch.Tensor, update_rms: bool) -> torch.Tensor:
        state_part = x[:, : self._state_dim]
        passthrough = x[:, self._state_dim :]
        return torch.cat([self.preprocessor(state_part, update_rms=update_rms), passthrough], dim=-1)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor = None):
        if hidden_state is None:
            hidden_state = self.get_initial_hidden(obs.shape[0], obs.device)
        feat = self._prep(obs, update_rms=self.training)
        x = self.fc_in(feat).unsqueeze(0)
        rnn_out, new_h = self._rnn_forward(x, hidden_state)
        return self._head(rnn_out.squeeze(0)), new_h

    def forward_sequence(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor):
        T, B, _ = obs_seq.shape
        feat = self._prep(obs_seq.reshape(T * B, -1), update_rms=self.training)
        x = self.fc_in(feat).view(T, B, -1)
        rnn_out, final_h = self._rnn_forward(x, hidden_state)
        out = self._head(rnn_out.reshape(T * B, -1)).view(T, B, -1)
        return out, final_h

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "graph_path": self._graph_path,
            "num_agents": self._num_agents,
            "num_nodes": self._num_nodes,
            "gpe_dim": self._gpe_dim,
            "proj_dim": self._proj_dim,
            "use_log_idleness": self._use_log_idleness,
            "T_time": self._T_time,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "fc_hidden": self._fc_hidden,
            "dueling": self.dueling,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MASUPVDPPOQrnn":
        return cls(
            graph_path=cfg["graph_path"],
            num_agents=cfg["num_agents"],
            num_nodes=cfg["num_nodes"],
            gpe_dim=cfg["gpe_dim"],
            proj_dim=cfg["proj_dim"],
            use_log_idleness=cfg["use_log_idleness"],
            T_time=cfg["T_time"],
            hidden_size=cfg["hidden_size"],
            output_dim=cfg["output_dim"],
            input_dim=cfg["input_dim"],
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            dueling=cfg.get("dueling", False),
            fc_hidden=cfg.get("fc_hidden") or None,
        )
