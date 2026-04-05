"""值分解 Mixer 网络。

BaseMixer           ← 抽象基类
├── QPLEXMixer      ← QPLEX-style dueling mixer (VDPPO)
├── SumMixer        ← VDN: Q_tot = Σ Q_i
└── QMIXMixer       ← QMIX: monotone hypernetwork mixing
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.mlp import layer_init


class BaseMixer(nn.Module, ABC):
    """值分解 Mixer 基类。"""

    def __init__(self, n_agents: int, state_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim


class QPLEXMixer(BaseMixer):
    """
    QPLEX-style dueling mixer。

    V_tot(τ)   = Σ_i V_i(τ)
    A_tot(τ,u) = Σ_i λ_i(s) * A_i(τ,u_i),  λ_i > 0
    Q_tot(τ,u) = V_tot + A_tot

    λ_i 由 lambda_net 输出并经 softplus 保证正值，
    满足 Advantage-IGM 约束。
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__(n_agents, state_dim)
        self.lambda_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, embed_dim), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, n_agents), std=1.0),
        )

    def forward(
        self,
        v_vals: torch.Tensor,   # (B, N)
        a_vals: torch.Tensor,   # (B, N)
        states: torch.Tensor,   # (B, state_dim)
    ) -> torch.Tensor:
        """返回 Q_tot (B,)。"""
        v_tot = v_vals.sum(dim=-1)                          # (B,)
        lam = F.softplus(self.lambda_net(states)) + 1e-6    # (B, N), 严格正
        a_tot = (lam * a_vals).sum(dim=-1)                  # (B,)
        return v_tot + a_tot


class SumMixer(BaseMixer):
    """
    VDN Mixer: Q_tot = Σ_i Q_i。

    无可学习参数，forward 签名与 QMIXMixer 统一（忽略 states）。
    """

    def __init__(self, n_agents: int, state_dim: int = 0):
        super().__init__(n_agents, state_dim)

    def forward(
        self,
        q_vals: torch.Tensor,               # (B, N)
        states: torch.Tensor = None,         # unused
    ) -> torch.Tensor:
        """返回 Q_tot (B,)。"""
        return q_vals.sum(dim=-1)


class QMIXMixer(BaseMixer):
    """
    QMIX Mixer: 超网络保证 ∂Q_tot/∂Q_i ≥ 0 (单调性约束)。

    hyper_w 的输出经 abs() 保证非负权重。
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__(n_agents, state_dim)
        self.embed_dim = embed_dim

        # 第一层超网络: state → w1 (N × embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # 第二层超网络: state → w2 (embed_dim × 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # b2 用两层 MLP 以增加表达力（原论文设计）
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        q_vals: torch.Tensor,               # (B, N)
        states: torch.Tensor,                # (B, state_dim)
    ) -> torch.Tensor:
        """返回 Q_tot (B,)。"""
        B = q_vals.shape[0]
        N = self.n_agents

        wdev = self.hyper_w1[0].weight.device
        if states.device != wdev:
            raise RuntimeError(
                f"QMIXMixer: states.device={states.device} != hyper_w1.weight.device={wdev} "
                f"(q_vals.device={q_vals.device})"
            )
        if states.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"QMIXMixer: states.dtype={states.dtype} 应为浮点类型")
        if states.dim() != 2 or states.shape[0] != B or states.shape[1] != self.state_dim:
            raise RuntimeError(
                f"QMIXMixer: states.shape={tuple(states.shape)} 期望 (B={B}, {self.state_dim})"
            )
        if not torch.isfinite(states).all():
            nf = (~torch.isfinite(states)).float().mean().item()
            raise RuntimeError(f"QMIXMixer: states 含非有限值, 比例={nf:.4f}")

        # 第一层: (B, 1, N) × (B, N, E) + (B, 1, E) → (B, 1, E)
        w1 = self.hyper_w1(states).abs().view(B, N, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_vals.unsqueeze(1), w1) + b1)

        # 第二层: (B, 1, E) × (B, E, 1) + (B, 1, 1) → (B, 1, 1)
        w2 = self.hyper_w2(states).abs().view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)
