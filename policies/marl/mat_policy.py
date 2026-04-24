"""MATMultiAgentPolicy — 封装 GATEncoder + MATDecoder 的联合多智能体策略。

接口设计为独立于 MultiAgentPolicy，避免继承不兼容的 per-agent 接口。

核心方法:
    compute_joint_actions() — 推理阶段（决策采样）
    evaluate_joint_actions() — 训练阶段（teacher forcing，重计算 log_prob）
    build_shift_action()    — 根据 active_mask 构造自回归上下文
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from networks.mat import GATEncoder, MATDecoder


class MATMultiAgentPolicy(nn.Module):
    """联合多智能体策略：GATEncoder + MATDecoder。

    不继承 MultiAgentPolicy（接口不兼容），但提供 device / set_training_mode
    属性，供 BaseAlgorithm 使用。

    shift_action 格式: (B, N, N, 2) float32
        shift_action[b, agent_slot, k, :] = agent k 当前目标节点的 2D 坐标
        自回归解码时，agent i 以前 i 个 agent 的目标作为上下文。
    """

    def __init__(
        self,
        encoder: GATEncoder,
        decoder: MATDecoder,
        n_agents: int,
        agent_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_agents = n_agents
        self.agent_ids = agent_ids or [f"agent_{i}" for i in range(n_agents)]

        self._training_mode = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    #  公共接口
    # -------------------------------------------------------------------------

    @property
    def is_recurrent(self) -> bool:
        return False

    def set_training_mode(self, mode: bool):
        self._training_mode = mode
        self.train(mode)

    def to(self, device):
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    # -------------------------------------------------------------------------
    #  推理阶段
    # -------------------------------------------------------------------------

    def compute_joint_actions(
        self,
        graph_state: np.ndarray,            # (B, N, G, 3)
        current_node_idx: np.ndarray,       # (B, N) int
        active_mask: np.ndarray,            # (B, N) 0/1
        last_shift: Optional[torch.Tensor], # (B, N, N, 2) 上次 shift_action
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样阶段联合前向传播。

        Returns:
            actions:         (B, N) int — graph index
            log_prob_full:   (B, N, G) — 完整 log 分布（供 PPO 存储）
            state_value:     (B, N, 1)
            shift_action_new: (B, N, N, 2) 更新后
        """
        device = self.device
        state_t = torch.as_tensor(graph_state, dtype=torch.float32, device=device)

        with torch.no_grad():
            node_emb, state_value = self.encoder(state_t)

        shift_action = self.build_shift_action(
            current_node_idx, active_mask, last_shift, device
        )

        # 临时切换为 greedy/sampling
        orig = self.decoder.select_type
        if deterministic:
            self.decoder.select_type = "greedy"

        with torch.no_grad():
            log_prob_full, actions, shift_new = self.decoder(
                node_emb, current_node_idx, active_mask, shift_action
            )

        if deterministic:
            self.decoder.select_type = orig

        return (
            actions.cpu().numpy(),   # (B, N)
            log_prob_full,           # (B, N, G)
            state_value,             # (B, N, 1)
            shift_new,               # (B, N, N, 2)
        )

    # -------------------------------------------------------------------------
    #  训练阶段（teacher forcing）
    # -------------------------------------------------------------------------

    def evaluate_joint_actions(
        self,
        graph_state: torch.Tensor,   # (T, N, G, 3)
        current_node_idx: np.ndarray,# (T, N) int
        shift_action: torch.Tensor,  # (T, N, N, 2)
        active_mask: np.ndarray,     # (T, N) 0/1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练阶段：用存储的 shift_action 重新计算 log_prob。

        Returns:
            log_prob_full: (T, N, G)   — 每节点的 log 概率
            state_value:  (T, N, 1)
            entropy:      (T, N, G)    — Categorical entropy（对所有节点的分布）
        """
        node_emb, state_value = self.encoder(graph_state)
        log_prob_full, _, _ = self.decoder(
            node_emb, current_node_idx, active_mask, shift_action
        )
        # entropy: - sum(p * log_p)
        prob = log_prob_full.exp().clamp(min=1e-8)
        entropy = -(prob * log_prob_full).sum(dim=-1, keepdim=True)  # (T, N, 1)

        return log_prob_full, state_value, entropy

    # -------------------------------------------------------------------------
    #  辅助：构造 shift_action
    # -------------------------------------------------------------------------

    def build_shift_action(
        self,
        current_node_idx: np.ndarray,  # (B, N) int — graph index
        active_mask: np.ndarray,       # (B, N) 0/1
        last_shift: Optional[torch.Tensor],  # (B, N, N, 2) or None
        device: torch.device,
    ) -> torch.Tensor:
        """构造 shift_action。

        规则（对应原版 get_shift_action）:
        - agent 0 的 context slot 0: 全零初始化（epoch 首步）或沿用 last_shift
        - ON_EDGE (active_mask=0) 的 agent: context slot 沿用上次目标坐标
        - READY (active_mask=1) 的 agent: 由 Decoder 自回归填写（这里先全零，
          Decoder 会在自回归循环中更新 agent i+1 ... N-1 的 context slot）

        返回 (B, N, N, 2) tensor。
        """
        B = current_node_idx.shape[0]
        N = self.n_agents
        shift = torch.zeros(B, N, N, 2, dtype=torch.float32, device=device)

        if last_shift is not None:
            # ON_EDGE agents 沿用上一步的目标坐标
            am = torch.as_tensor(active_mask, dtype=torch.bool, device=device)  # (B, N)
            for b in range(B):
                for k in range(N):
                    if not am[b, k]:
                        # 非决策 agent：所有 slot 关于 k 的条目沿用 last_shift
                        shift[b, :, k, :] = last_shift[b, :, k, :]

        return shift

    # -------------------------------------------------------------------------
    #  Checkpoint 协议
    # -------------------------------------------------------------------------

    def get_config_dict(self, *args) -> dict:
        return {
            "encoder": self.encoder.get_config_dict(
                self.encoder.input_dim, self.encoder.output_dim
            ),
            "decoder": self.decoder.get_config_dict(
                self.decoder.input_dim, self.decoder.output_dim
            ),
            "n_agents": self.n_agents,
            "agent_ids": self.agent_ids,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "MATMultiAgentPolicy":
        encoder = GATEncoder.from_config_dict(cfg["encoder"])
        decoder = MATDecoder.from_config_dict(cfg["decoder"])
        return cls(encoder, decoder, cfg["n_agents"], cfg.get("agent_ids"))

    def parameters_to_optimize(self):
        """供 optimizer 使用（encoder + decoder 所有参数）。"""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
