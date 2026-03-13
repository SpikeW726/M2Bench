import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Any, Dict, List, Sequence, Tuple, Set
from collections.abc import Mapping
from gym.spaces import Box, Dict as GymDict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override

try:
    from torch_scatter import scatter_sum, scatter_mean, scatter_max
except ImportError:
    def scatter_sum(src, index, dim=0, dim_size=None):
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        return out.index_add_(0, index, src)
    def scatter_mean(src, index, dim=0, dim_size=None):
        out = scatter_sum(src, index, dim, dim_size)
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


class ResidualFC(nn.Module):
    def __init__(self, in_size, out_size, activation_fn="silu"):
        super().__init__()
        
        # 💡 这里的处理可以兼容字符串或直接的激活函数类
        if activation_fn == "silu":
            act = nn.SiLU
        elif activation_fn == "relu":
            act = nn.ReLU
        else:
            act = activation_fn # 保持原样

        self.layer = SlimFC(
            in_size=in_size,
            out_size=out_size,
            initializer=normc_initializer(1.0),
            activation_fn=act, # 传入解析后的激活函数
        )
        self.use_residual = (in_size == out_size)

    def forward(self, x):
        out = self.layer(x)
        return out + x if self.use_residual else out


class SAGELayer(nn.Module):
    def __init__(self, h_dim, mlp_layers=2, activation_fn="silu"):
        super().__init__()
        self.node_norm = nn.LayerNorm(h_dim * 3)
        self.global_norm = nn.LayerNorm(h_dim * 3)
        self.node_mlp = self._build_gnn_mlp(h_dim * 3, h_dim, mlp_layers, activation_fn)
        self.global_mlp = self._build_gnn_mlp(h_dim * 3, h_dim, mlp_layers, activation_fn)

    # --- 修改前 (这里的循环会导致残差全部失效) ---
    # for _ in range(num_layers):
    #     layers.append(ResidualFC(in_size=prev_dim, out_size=hidden_dim, ...))

    # --- 修改后 (瓶颈式：对齐 -> 纯残差) ---
    def _build_gnn_mlp(self, input_dim, hidden_dim, num_layers, activation_fn):
        num_layers = max(1, num_layers)
        layers = []
        
        # 1. 第一层：维度转换 (从拼接后的高维降到 h_dim)
        # 这一层 in != out，不触发残差，但完成了“对齐”
        layers.append(ResidualFC(input_dim, hidden_dim, activation_fn=activation_fn))
        
        # 2. 后续层：纯残差空间
        # 因为 in_size == out_size == hidden_dim，ResidualFC 内部的残差会完美开启
        for _ in range(num_layers - 1):
            layers.append(ResidualFC(hidden_dim, hidden_dim, activation_fn=activation_fn))
            
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, g, batch_idx, edge_batch):
        row, col = edge_index
        messages = torch.cat([x[row], edge_attr], dim=-1)
        e_agg = scatter_mean(messages, col, dim=0, dim_size=x.size(0))
        x_in = torch.cat([x, e_agg], dim=-1)
        x_in = self.node_norm(x_in)
        x = x + self.node_mlp(x_in)

        x_mean = scatter_mean(x, batch_idx, dim=0, dim_size=g.size(0))
        e_mean = scatter_mean(edge_attr, edge_batch, dim=0, dim_size=g.size(0))
        g_in = torch.cat([g, x_mean, e_mean], dim=-1)
        g_in = self.global_norm(g_in)
        g = g + self.global_mlp(g_in)

        return x, edge_attr, g


class SAGEDynamicGraph(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        raw_cfg = model_config.get("custom_model_config", {})
        self.cfg = self._flatten_cfg(raw_cfg)
        self.h_dim = self.cfg.get("hidden_dim", 64)
        self.k_layers = self.cfg.get("gnn_layers", 2)
        self.actor_mlp_layers = int(self.cfg.get("actor_mlp_layers", 1))
        self.critic_mlp_layers = int(self.cfg.get("critic_mlp_layers", 1))
        self.gnn_mlp_layers = int(self.cfg.get("gnn_mlp_layers", 2))
        self.mlp_activation = (
            self.cfg.get("mlp_activation")
            or model_config.get("fcnet_activation")
            or "silu"
        )

        self._init_topology(model_config)
        self._init_critic_interface(obs_space)

        self.node_init = nn.Linear(2, self.h_dim)
        self.edge_init = nn.Linear(1, self.h_dim)
        self.global_init = nn.Linear(2, self.h_dim)
        self.id_encoder = nn.Linear(self.identity_dim, self.h_dim)

        self.layers = nn.ModuleList([
            SAGELayer(self.h_dim, self.gnn_mlp_layers, self.mlp_activation)
            for _ in range(self.k_layers)
        ])

        self.actor_head = self._build_mlp(self.h_dim * 3, num_outputs, self.h_dim, self.actor_mlp_layers)
        # 💡 针对最后一层 Linear 进行极小初始化
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01) 
        nn.init.constant_(self.actor_head[-1].bias, 0)
        
        self.critic_mlp = self._build_mlp(
            self.critic_expected_dim, 1, self.h_dim, self.critic_mlp_layers
        )

    def _flatten_cfg(self, cfg):
        if not isinstance(cfg, dict):
            return {}
        res = {}
        def recurse(c):
            for k, v in c.items():
                if k == "custom_model_config" and isinstance(v, dict):
                    recurse(v)
                else:
                    res[k] = v
        recurse(cfg)
        return res

    def _build_mlp(self, input_dim, output_dim, hidden_dim, num_layers):
        num_layers = max(1, num_layers)
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(
                ResidualFC(
                    in_size=prev_dim,
                    out_size=hidden_dim,
                    activation_fn=self.mlp_activation,
                )
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_topology(self, model_config):
        sources = [self.cfg, model_config.get("custom_model_config", {}), model_config]
        path = None
        for s in sources:
            path = s.get("graph_path") or s.get("env_args", {}).get("graph_path")
            if path:
                break

        self.agent_num = self.cfg.get("agent_num") or self.cfg.get("num_agents") or 3
        from utils.graph_utils import Graph
        graph_helper = Graph(path)
        self.static_node_num = len(graph_helper.nodes)
        self.num_nodes = self.static_node_num + self.agent_num

        num_static_edges = sum(len(graph_helper.adj_list[u]) for u in graph_helper.nodes)
        self.max_edges = num_static_edges + (self.agent_num * 2)

        self.role_ifm = self.cfg.get("role_imformation") or "agent-index"
        self.identity_dim = 2 if self.role_ifm == "decision" else self.agent_num

    def _init_critic_interface(self, obs_space):
        full_space = getattr(obs_space, "original_space", obs_space)
        self.critic_obs_dim = 0
        self.critic_state_dim = 0

        if isinstance(full_space, GymDict):
            obs_box = full_space.spaces.get("obs")
            if isinstance(obs_box, Box):
                self.critic_obs_dim = int(np.prod(obs_box.shape))
            state_box = None
            for key in ("state", "global_state"):
                candidate = full_space.spaces.get(key)
                if isinstance(candidate, Box):
                    state_box = candidate
                    break
            if state_box is not None:
                self.critic_state_dim = int(np.prod(state_box.shape))
        elif isinstance(full_space, Box):
            self.critic_obs_dim = int(np.prod(full_space.shape))

        self.critic_expected_dim = max(1, self.critic_obs_dim + self.critic_state_dim)

    def _format_tensor(self, source, device):
        if source is None:
            return None
        if isinstance(source, torch.Tensor):
            tensor = source.to(device)
        else:
            tensor = torch.as_tensor(source, dtype=torch.float32, device=device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.view(tensor.size(0), -1)

    def _match_batch(self, tensor, batch_size):
        if tensor is None:
            return None
        if tensor.size(0) == batch_size:
            return tensor
        if tensor.size(0) > batch_size:
            return tensor[:batch_size]
        repeat_factor = (batch_size + tensor.size(0) - 1) // tensor.size(0)
        return tensor.repeat(repeat_factor, 1)[:batch_size]

    def _pad_or_truncate(self, tensor, target_dim):
        if tensor is None or target_dim <= 0:
            return None
        current_dim = tensor.size(1)
        if current_dim == target_dim:
            return tensor
        if current_dim > target_dim:
            return tensor[:, :target_dim]
        pad = torch.zeros(tensor.size(0), target_dim - current_dim, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)

    def _compute_joint_embedding(self, obs_tensor):
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
        self_embed = h_reshaped[torch.arange(batch_size), self_node_idx, :]

        id_embed = F.silu(self.id_encoder(identity))

        graph_mean = scatter_mean(x, batch_idx, dim=0, dim_size=batch_size)
        graph_max = scatter_max(x, batch_idx, dim=0, dim_size=batch_size)
        if isinstance(graph_max, tuple):
            graph_max = graph_max[0]

        return self_embed, g, id_embed, graph_mean, graph_max

    def _get_critic_tensor(self, obs_mapping, default_obs):
        device = next(self.parameters()).device
        base_tensor = self._format_tensor(default_obs, device)
        batch_size = base_tensor.size(0)
        features = torch.zeros(batch_size, self.critic_expected_dim, device=device)

        obs_tensor = None
        if isinstance(obs_mapping, Mapping):
            obs_tensor = obs_mapping.get("obs")
        obs_tensor = self._format_tensor(obs_tensor, device) if obs_tensor is not None else base_tensor
        obs_tensor = self._match_batch(obs_tensor, batch_size)
        obs_tensor = self._pad_or_truncate(obs_tensor, self.critic_obs_dim)
        if obs_tensor is not None and self.critic_obs_dim > 0:
            features[:, :self.critic_obs_dim] = obs_tensor

        if self.critic_state_dim > 0:
            state_tensor = None
            if isinstance(obs_mapping, Mapping):
                for key in ("state", "global_state"):
                    if key in obs_mapping:
                        state_tensor = obs_mapping[key]
                        break
            state_tensor = self._format_tensor(state_tensor, device)
            if state_tensor is None:
                state_tensor = torch.zeros(batch_size, self.critic_state_dim, device=device)
            state_tensor = self._match_batch(state_tensor, batch_size)
            state_tensor = self._pad_or_truncate(state_tensor, self.critic_state_dim)
            features[:, self.critic_obs_dim : self.critic_obs_dim + self.critic_state_dim] = state_tensor

        return features

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_mapping = input_dict["obs"] if isinstance(input_dict["obs"], Mapping) else None
        obs = obs_mapping["obs"] if obs_mapping else input_dict["obs"]

        self_emb, g_emb, id_emb, g_mean, g_max = self._compute_joint_embedding(obs)

        actor_input = torch.cat([self_emb, g_emb, id_emb], dim=-1)
        logits = self.actor_head(actor_input)
        

        critic_tensor = self._get_critic_tensor(obs_mapping, obs)
        self._value_out = self.critic_mlp(critic_tensor).squeeze(1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out

    def central_value_function(self, state, opponent_actions=None):
        if isinstance(state, Mapping):
            default_obs = state.get("obs")
            if default_obs is None:
                default_obs = state.get("state") or state.get("global_state")
            if default_obs is None and len(state) > 0:
                default_obs = next(iter(state.values()))
            if default_obs is None:
                device = next(self.parameters()).device
                default_obs = torch.zeros(self.critic_expected_dim, device=device)
            critic_input = self._get_critic_tensor(state, default_obs)
            return self.critic_mlp(critic_input).squeeze(1)

        device = next(self.parameters()).device
        state_tensor = self._format_tensor(state, device)
        if state_tensor is None:
            state_tensor = torch.zeros(1, self.critic_expected_dim, device=device)
        features = self._pad_or_truncate(state_tensor, self.critic_expected_dim)
        if features is None:
            features = torch.zeros(state_tensor.size(0), self.critic_expected_dim, device=device)
        return self.critic_mlp(features).squeeze(1)
