"""值分解 Mixer 网络。

BaseMixer           ← 抽象基类
└── QPLEXMixer      ← QPLEX-style dueling mixer (VDPPO)

未来扩展:
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
