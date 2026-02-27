"""VDN: Value Decomposition Network for multi-agent environments.

Q_tot = Σ_i Q_i(o_i, a_i)  (SumMixer, 无可学习参数)

所有 agent 共享 Q-network (shared_policy=True)，
联合 TD loss 通过 SumMixer 汇聚后反向传播。

本类同时作为 QMIX 等值分解算法的基类，子类通过 override
_init_mixer / _init_optimizer / _update_target_networks 实现差异化。
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import BaseAlgorithm, TrainingStats
from configs.algo_configs import VDNParams
from data.batch import TransitionBatch
from networks.mixing import SumMixer
from policies.marl.marl_base import MultiAgentPolicy
from policies.rl.rl_base import ValuePolicy


class VDNAlgo(BaseAlgorithm):
    """
    VDN: Value Decomposition Network。

    CTDE 范式：分散执行时每个 agent 用自身 Q_i 做 argmax，
    集中训练时通过 Q_tot = Σ Q_i 的联合 TD loss 更新共享 Q-network。

    同时作为 QMIX 等值分解算法的基类，可 override 的扩展点：
    - _init_mixer(): Mixer 网络创建（VDN=SumMixer, QMIX=QMIXMixer）
    - _init_optimizer(): 优化器创建（QMIX 需包含 mixer 参数）
    - _update_target_networks(): Target 更新（QMIX 需额外更新 target_mixer）
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: VDNParams,
        n_agents: int = 1,
        state_dim: int = 0,
    ):
        super().__init__(policy)
        self.params = params
        self.gamma = params.gamma
        self.max_grad_norm = params.max_grad_norm
        self.tau = params.tau
        self.target_update_freq = params.target_update_freq
        self.n_agents = n_agents

        # 共享 Q-network（从第一个 agent 取出）
        self.q_network = policy.get_policy(policy.agent_ids[0]).q_network

        # Target Q-network
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.eval()
        for p in self.target_q_network.parameters():
            p.requires_grad = False

        # Mixer & Optimizer（子类可 override）
        self._init_mixer(n_agents, state_dim, params)
        self._init_optimizer(params)

        # Epsilon 管理
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay = params.epsilon_decay
        for aid in policy.agent_ids:
            policy.get_policy(aid).set_epsilon(params.epsilon_start)

        self._update_count = 0

    # ====================================================================
    #                       子类扩展点
    # ====================================================================

    def _init_mixer(self, n_agents: int, state_dim: int, params):
        """创建 Mixer 网络。VDN 使用无参数的 SumMixer。"""
        self.mixer = SumMixer(n_agents, state_dim)
        self.target_mixer = SumMixer(n_agents, state_dim)

    def _init_optimizer(self, params):
        """创建优化器。VDN 仅优化 Q-network（SumMixer 无参数）。"""
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=params.lr,
        )

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.q_network, "is_recurrent", False)

    # ====================================================================
    #                       compute joint loss
    # ====================================================================

    def _compute_loss(
        self,
        batch_dict: Dict[str, TransitionBatch],
    ) -> tuple[torch.Tensor, dict]:
        agents = self.policy.agent_ids
        first_batch = batch_dict[agents[0]]
        B = first_batch.obs.shape[0]
        N = self.n_agents

        # Stack per-agent data → (B, N, ...)
        obs_all = torch.stack([batch_dict[a].obs for a in agents], dim=1)
        act_all = torch.stack([batch_dict[a].act for a in agents], dim=1).long()
        rew_all = torch.stack([batch_dict[a].rew for a in agents], dim=1)
        next_obs_all = torch.stack([batch_dict[a].next_obs for a in agents], dim=1)
        done = first_batch.done
        state = first_batch.state
        next_state = first_batch.next_state

        # Action masks
        next_am_all = None
        if first_batch.next_action_mask is not None:
            next_am_all = torch.stack(
                [batch_dict[a].next_action_mask for a in agents], dim=1,
            )

        # --- Online Q-values ---
        q_all = self.q_network(obs_all.view(B * N, -1)).view(B, N, -1)
        q_chosen = q_all.gather(-1, act_all.unsqueeze(-1)).squeeze(-1)  # (B, N)
        q_tot = self.mixer(q_chosen, state)  # (B,)

        # --- Target Q-values ---
        with torch.no_grad():
            if self.params.use_double_dqn:
                q_next_online = self.q_network(next_obs_all.view(B * N, -1)).view(B, N, -1)
                if next_am_all is not None:
                    q_next_online = q_next_online.masked_fill(~next_am_all.bool(), float("-inf"))
                next_actions = q_next_online.argmax(dim=-1, keepdim=True)
                q_next_target = self.target_q_network(next_obs_all.view(B * N, -1)).view(B, N, -1)
                q_next_chosen = q_next_target.gather(-1, next_actions).squeeze(-1)
            else:
                q_next_target = self.target_q_network(next_obs_all.view(B * N, -1)).view(B, N, -1)
                if next_am_all is not None:
                    q_next_target = q_next_target.masked_fill(~next_am_all.bool(), float("-inf"))
                q_next_chosen = q_next_target.max(dim=-1)[0]

            q_tot_target = self.target_mixer(q_next_chosen, next_state)
            r_tot = rew_all.sum(dim=-1)  # (B,) — 联合奖励 = 各 agent 奖励之和
            td_target = r_tot + self.gamma * (1.0 - done) * q_tot_target

        loss = F.smooth_l1_loss(q_tot, td_target)

        info = {
            "q_tot_mean": q_tot.detach().mean().item(),
            "q_tot_max": q_tot.detach().max().item(),
            "td_error": (td_target - q_tot).detach().abs().mean().item(),
        }
        return loss, info

    # ====================================================================
    #                       update
    # ====================================================================

    def update(
        self,
        batch_dict: Dict[str, TransitionBatch],
        **kwargs,
    ) -> TrainingStats:
        batch_dict = {
            aid: batch.to_tensor(self.device)
            for aid, batch in batch_dict.items()
        }

        loss, info = self._compute_loss(batch_dict)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            params = [p for g in self.optimizer.param_groups for p in g['params']]
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

        self._decay_epsilon()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._update_target_networks()

        return TrainingStats(
            loss=loss.item(),
            extra={
                "q_tot_mean": info["q_tot_mean"],
                "q_tot_max": info["q_tot_max"],
                "td_error": info["td_error"],
                "epsilon": self._get_current_epsilon(),
            },
        )

    # ====================================================================
    #                       target network / epsilon
    # ====================================================================

    def _update_target_networks(self):
        """更新 target Q-network。子类 override 以同时更新 target_mixer。"""
        if self.tau < 1.0:
            for tp, sp in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        else:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _decay_epsilon(self):
        for aid in self.policy.agent_ids:
            vp: ValuePolicy = self.policy.get_policy(aid)
            new_eps = max(vp.get_epsilon() * self.epsilon_decay, self.epsilon_end)
            vp.set_epsilon(new_eps)

    def _get_current_epsilon(self) -> float:
        return self.policy.get_policy(self.policy.agent_ids[0]).get_epsilon()

    def set_epsilon(self, epsilon: float):
        for aid in self.policy.agent_ids:
            self.policy.get_policy(aid).set_epsilon(epsilon)
