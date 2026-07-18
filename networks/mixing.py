"""Value-decomposition mixer networks.

``SumMixer`` implements VDN, ``QMIXMixer`` performs monotonic state-conditioned
mixing, and ``QPLEXMixer`` provides the dueling decomposition used by VDPPO.
All implementations share the abstract ``BaseMixer`` interface.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.mlp import layer_init

class BaseMixer(nn.Module, ABC):
    def __init__(self, n_agents: int, state_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

class QPLEXMixer(BaseMixer):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__(n_agents, state_dim)
        self.lambda_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, embed_dim), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, n_agents), std=1.0),
        )

    def forward(
        self,
        v_vals: torch.Tensor,   # (B, N).
        a_vals: torch.Tensor,   # (B, N).
        states: torch.Tensor,   # (B, state_dim).
    ) -> torch.Tensor:
        v_tot = v_vals.sum(dim=-1)                          # (B,).
        lam = F.softplus(self.lambda_net(states)) + 1e-6    # Shape: (B, N).
        a_tot = (lam * a_vals).sum(dim=-1)                  # (B,).
        return v_tot + a_tot

class SumMixer(BaseMixer):
    """
    VDN mixer computing ``Q_tot = sum_i Q_i``.
    """

    def __init__(self, n_agents: int, state_dim: int = 0):
        super().__init__(n_agents, state_dim)

    def forward(
        self,
        q_vals: torch.Tensor,               # (B, N).
        states: torch.Tensor = None,        # unused.
    ) -> torch.Tensor:
        return q_vals.sum(dim=-1)

class QMIXMixer(BaseMixer):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__(n_agents, state_dim)
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        q_vals: torch.Tensor,               # (B, N).
        states: torch.Tensor,                # (B, state_dim).
    ) -> torch.Tensor:
        B = q_vals.shape[0]
        N = self.n_agents

        wdev = self.hyper_w1[0].weight.device
        if states.device != wdev:
            raise RuntimeError(
                f"QMIXMixer: states.device={states.device} != hyper_w1.weight.device={wdev} "
                f"(q_vals.device={q_vals.device})"
            )
        if states.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"QMIXMixer: states.dtype={states.dtype} must be floating point")
        if states.dim() != 2 or states.shape[0] != B or states.shape[1] != self.state_dim:
            raise RuntimeError(
                f"QMIXMixer: states.shape={tuple(states.shape)}; expected (B={B}, {self.state_dim})"
            )
        if not torch.isfinite(states).all():
            nf = (~torch.isfinite(states)).float().mean().item()
            raise RuntimeError(f"QMIXMixer: states contain non-finite values; fraction={nf:.4f}")

        w1 = self.hyper_w1(states).abs().view(B, N, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_vals.unsqueeze(1), w1) + b1)

        w2 = self.hyper_w2(states).abs().view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)
