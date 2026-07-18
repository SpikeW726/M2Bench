"""Joint policy combining the MAT graph encoder and autoregressive decoder.

The policy remains separate from the per-agent ``MultiAgentPolicy`` interface.
``compute_joint_actions`` samples in inference, ``evaluate_joint_actions`` uses
teacher forcing during training, and ``build_shift_action`` constructs the
autoregressive context from active-agent masks.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from networks.mat import GATEncoder, MATDecoder

class MATMultiAgentPolicy(nn.Module):
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

    def compute_joint_actions(
        self,
        graph_state: np.ndarray,            # (B, N, G, 3).
        current_node_idx: np.ndarray,       # (B, N) int.
        active_mask: np.ndarray,            # (B, N) 0/1.
        last_shift: Optional[torch.Tensor], # Shape: (B, N, N, 2).
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        state_t = torch.as_tensor(graph_state, dtype=torch.float32, device=device)

        with torch.no_grad():
            node_emb, state_value = self.encoder(state_t)

        shift_action = self.build_shift_action(
            current_node_idx, active_mask, last_shift, device
        )

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
            actions.cpu().numpy(),   # (B, N).
            log_prob_full,           # (B, N, G).
            state_value,             # (B, N, 1).
            shift_new,               # (B, N, N, 2).
        )

    def evaluate_joint_actions(
        self,
        graph_state: torch.Tensor,   # (T, N, G, 3).
        current_node_idx: np.ndarray,# (T, N) int.
        shift_action: torch.Tensor,  # (T, N, N, 2).
        active_mask: np.ndarray,     # (T, N) 0/1.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_emb, state_value = self.encoder(graph_state)
        log_prob_full, _, _ = self.decoder(
            node_emb, current_node_idx, active_mask, shift_action
        )
        # entropy: - sum(p * log_p).
        prob = log_prob_full.exp().clamp(min=1e-8)
        entropy = -(prob * log_prob_full).sum(dim=-1, keepdim=True)  # (T, N, 1).

        return log_prob_full, state_value, entropy

    def build_shift_action(
        self,
        current_node_idx: np.ndarray,  # (B, N) int - graph index.
        active_mask: np.ndarray,       # (B, N) 0/1.
        last_shift: Optional[torch.Tensor],  # (B, N, N, 2) or None.
        device: torch.device,
    ) -> torch.Tensor:
        B = current_node_idx.shape[0]
        N = self.n_agents
        shift = torch.zeros(B, N, N, 2, dtype=torch.float32, device=device)

        if last_shift is not None:

            am = torch.as_tensor(active_mask, dtype=torch.bool, device=device)  # (B, N).
            for b in range(B):
                for k in range(N):
                    if not am[b, k]:

                        shift[b, :, k, :] = last_shift[b, :, k, :]

        return shift

    def get_config_dict(self, *args) -> dict:
        return {
            "type": "MATMultiAgentPolicy",
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
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
