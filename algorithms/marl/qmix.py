"""QMIX with monotonic value-function factorization.

QMIX extends VDN with a state-conditioned hypernetwork mixer constrained so
``dQ_tot / dQ_i >= 0``. Its optimizer and target updates include both the shared
Q-network and the learnable mixer.
"""

import copy

import torch
import torch.nn as nn

from algorithms.marl.vdn import VDNAlgo
from configs.algo_configs import QMIXParams
from networks.mixing import QMIXMixer
from policies.marl.marl_base import MultiAgentPolicy

class QMIXAlgo(VDNAlgo):
    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: QMIXParams,
        n_agents: int = 1,
        state_dim: int = 0,
    ):
        super().__init__(policy, params, n_agents, state_dim)

    def _init_mixer(self, n_agents: int, state_dim: int, params: QMIXParams):
        self.mixer = QMIXMixer(n_agents, state_dim, params.mixer_embed_dim).to(self.device)

        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.target_mixer.eval()
        for p in self.target_mixer.parameters():
            p.requires_grad = False

    def _init_optimizer(self, params: QMIXParams):
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=params.lr,
        )

    def _update_target_networks(self):
        super()._update_target_networks()
        if self.tau < 1.0:
            for tp, sp in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        else:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
