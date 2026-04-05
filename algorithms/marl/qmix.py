"""QMIX: Monotonic Value Function Factorisation for multi-agent environments.

Q_tot = f(Q_1, ..., Q_N, s)，超网络保证 ∂Q_tot/∂Q_i ≥ 0。

继承 VDNAlgo，override mixer / optimizer / target update 三个扩展点。
与 VDN 的区别：
- Mixer 含可学习参数（超网络），需要 global state 输入
- 优化器同时包含 Q-network 和 Mixer 参数
- Target 更新同时覆盖 target_q_network 和 target_mixer
"""

import copy

import torch
import torch.nn as nn

from algorithms.marl.vdn import VDNAlgo
from configs.algo_configs import QMIXParams
from networks.mixing import QMIXMixer
from policies.marl.marl_base import MultiAgentPolicy


class QMIXAlgo(VDNAlgo):
    """
    QMIX: Q-value Mixing Network，继承 VDNAlgo。

    通过超网络 Mixer 保证 IGM 条件（单调性约束），
    _compute_loss / update / epsilon 管理全部复用 VDN 基类逻辑。
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: QMIXParams,
        n_agents: int = 1,
        state_dim: int = 0,
    ):
        super().__init__(policy, params, n_agents, state_dim)

    # ====================================================================
    #                       override 扩展点
    # ====================================================================

    def _init_mixer(self, n_agents: int, state_dim: int, params: QMIXParams):
        """QMIX: 含超网络参数的 Mixer + deepcopy target。"""
        self.mixer = QMIXMixer(n_agents, state_dim, params.mixer_embed_dim).to(self.device)
        # deepcopy 在部分环境下会把副本留在 CPU；显式 .to(device) 避免 target 与 batch 设备不一致
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.target_mixer.eval()
        for p in self.target_mixer.parameters():
            p.requires_grad = False

    def _init_optimizer(self, params: QMIXParams):
        """QMIX: Q-network + Mixer 参数联合优化。"""
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=params.lr,
        )

    def _update_target_networks(self):
        """QMIX: 同时更新 target Q-network 和 target Mixer。"""
        super()._update_target_networks()
        if self.tau < 1.0:
            for tp, sp in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        else:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
