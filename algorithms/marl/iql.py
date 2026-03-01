"""IQL: Independent Q-Learning for multi-agent environments.

每个 agent 独立 ValuePolicy + Q-network，无中心化 critic，无梯度协调。
epsilon 当前值存储在各 agent 的 ValuePolicy 中，衰减调度由本算法控制。

支持 MLP (QMLP) 和 RNN (QRNN) 两种 Q-network：
- MLP: 标准 flat TransitionBatch 训练
- RNN: SequenceBatch 序列训练（DRQN 风格，h0=zeros）
"""

import copy
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import BaseAlgorithm, TrainingStats
from configs.algo_configs import IQLParams
from policies.marl.marl_base import MultiAgentPolicy
from policies.rl.rl_base import ValuePolicy
from data.batch import BaseBatch, TransitionBatch, SequenceBatch


class IQLAlgo(BaseAlgorithm):
    """
    Independent Q-Learning。

    每个 agent 有独立的 Q-network (ValuePolicy) 和对应的 target network。
    update 接收 Dict[str, TransitionBatch/SequenceBatch]，逐 agent 计算 DQN loss 并更新。

    Args:
        policy: MultiAgentPolicy，内部 wrap ValuePolicy 实例
        params: IQLParams
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: IQLParams,
        total_updates: int = 0,
    ):
        if getattr(policy, "shared", False):
            raise ValueError(
                "IQLAlgo 不支持 shared_policy=True（多 optimizer 指向同一参数）。"
                "如需参数共享请使用 VDN / QMIX。"
            )

        super().__init__(policy)
        self.params = params
        self.gamma = params.gamma
        self.max_grad_norm = params.max_grad_norm
        self.tau = params.tau
        self.target_update_freq = params.target_update_freq
        self.seq_len = params.seq_len

        # ---- target networks (nn.ModuleDict → 可被 PyTorch 追踪) ----
        self.target_policies = nn.ModuleDict()
        for aid in policy.agent_ids:
            target = copy.deepcopy(policy.get_policy(aid))
            target.set_training_mode(False)
            for p in target.parameters():
                p.requires_grad = False
            self.target_policies[aid] = target

        # ---- per-agent optimizers ----
        self._optimizers: Dict[str, torch.optim.Adam] = {
            aid: torch.optim.Adam(policy.get_policy(aid).parameters(), lr=params.lr)
            for aid in policy.agent_ids
        }

        # ---- epsilon 衰减参数 ----
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay = params.epsilon_decay
        self.epsilon_decay_by_step = params.epsilon_decay_by_step
        self.exploration_fraction = params.exploration_fraction

        if self.epsilon_decay_by_step:
            if total_updates <= 0:
                raise ValueError(
                    "epsilon_decay_by_step=True 需要 total_updates > 0，"
                    "请确保 train.py 正确传递了 total_updates。"
                )
            self._exploration_updates = max(
                1, int(self.exploration_fraction * total_updates)
            )
        self._total_updates = total_updates

        for aid in policy.agent_ids:
            policy.get_policy(aid).set_epsilon(params.epsilon_start)

        self._update_count = 0

    def set_training_mode(self, mode: bool):
        """切换 train/eval；target 网络始终保持 eval。"""
        super().set_training_mode(mode)
        for target in self.target_policies.values():
            target.set_training_mode(False)

    @property
    def is_recurrent(self) -> bool:
        first_policy = self.policy.get_policy(self.policy.agent_ids[0])
        return getattr(first_policy, "is_recurrent", False)

    # ====================================================================
    #                       per-agent loss (MLP)
    # ====================================================================

    def _compute_agent_loss_flat(
        self,
        batch: TransitionBatch,
        agent_id: str,
    ) -> tuple[torch.Tensor, dict]:
        """MLP 路径：计算单个 agent 的 Double DQN TD loss。"""
        agent_policy: ValuePolicy = self.policy.get_policy(agent_id)
        target_policy: ValuePolicy = self.target_policies[agent_id]

        obs = batch.obs
        actions = batch.act.long()
        rewards = batch.rew
        next_obs = batch.next_obs
        dones = batch.done
        next_action_mask = getattr(batch, 'next_action_mask', None)
        active_mask = getattr(batch, 'active_mask', None)

        q_values = agent_policy.q_network(obs)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.params.use_double_dqn:
                next_q_main = agent_policy.compute_q_values(next_obs, next_action_mask)
                next_actions = next_q_main.argmax(dim=1, keepdim=True)
                next_q_target = target_policy.q_network(next_obs)
                q_next = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q_target = target_policy.compute_q_values(
                    next_obs, next_action_mask,
                )
                q_next = next_q_target.max(dim=1)[0]

            td_target = rewards + self.gamma * (1.0 - dones) * q_next

        per_sample_loss = F.mse_loss(q_current, td_target, reduction='none')

        # active_mask: 仅对 READY 决策步计算 loss，跳过 ON_EDGE no-op 步
        if active_mask is not None:
            am_sum = active_mask.sum().clamp(min=1)
            loss = (per_sample_loss * active_mask).sum() / am_sum
        else:
            loss = per_sample_loss.mean()

        info = {
            "q_mean": q_values.mean().item(),
            "q_max": q_values.max().item(),
            "td_error": (td_target - q_current).detach().abs().mean().item(),
        }
        return loss, info

    # ====================================================================
    #                       per-agent loss (RNN sequence)
    # ====================================================================

    def _compute_agent_loss_sequence(
        self,
        batch: SequenceBatch,
        agent_id: str,
    ) -> tuple[torch.Tensor, dict]:
        """RNN 路径：DRQN 序列 TD loss，h0=zeros，支持 R2D2 burn-in。

        forward 全序列预热 hidden state，loss 仅在 [burn_in_len:] 训练区间计算。
        burn_in_len=0 时行为与无 burn-in 完全一致。
        """
        agent_policy: ValuePolicy = self.policy.get_policy(agent_id)
        target_policy: ValuePolicy = self.target_policies[agent_id]

        B, T_total = batch.obs.shape[:2]
        device = batch.obs.device
        bi = getattr(batch, 'burn_in_len', 0)

        obs_seq = batch.obs.transpose(0, 1)            # (T, B, D)
        next_obs_seq = batch.next_obs.transpose(0, 1)  # (T, B, D)
        actions = batch.act.long()                      # (B, T)
        rewards = batch.rew                             # (B, T)
        dones = batch.done                              # (B, T)
        mask = batch.mask                               # (B, T)

        h0 = agent_policy.q_network.get_initial_hidden(B, device)

        # forward 全序列（含 burn-in），然后截取训练区间
        q_full, _ = agent_policy.q_network.forward_sequence(obs_seq, h0)  # (T, B, act_dim)
        q_train = q_full[bi:]  # (S, B, act_dim), S = T - bi
        act_train = actions[:, bi:].T.unsqueeze(-1)  # (S, B, 1)
        q_current = q_train.gather(2, act_train).squeeze(-1)  # (S, B)

        with torch.no_grad():
            h0_target = target_policy.q_network.get_initial_hidden(B, device)

            if self.params.use_double_dqn:
                next_q_online_full, _ = agent_policy.q_network.forward_sequence(next_obs_seq, h0)
                next_q_online = next_q_online_full[bi:]
                next_am = getattr(batch, 'next_action_mask', None)
                if next_am is not None:
                    next_am_train = next_am[:, bi:].transpose(0, 1)
                    mask_t = next_am_train.bool() if next_am_train.dtype != torch.bool else next_am_train
                    next_q_online = next_q_online.masked_fill(~mask_t, float("-inf"))
                next_actions = next_q_online.argmax(dim=-1, keepdim=True)
                next_q_target_full, _ = target_policy.q_network.forward_sequence(next_obs_seq, h0_target)
                q_next = next_q_target_full[bi:].gather(2, next_actions).squeeze(-1)
            else:
                next_q_target_full, _ = target_policy.q_network.forward_sequence(next_obs_seq, h0_target)
                next_q_target = next_q_target_full[bi:]
                next_am = getattr(batch, 'next_action_mask', None)
                if next_am is not None:
                    next_am_train = next_am[:, bi:].transpose(0, 1)
                    mask_t = next_am_train.bool() if next_am_train.dtype != torch.bool else next_am_train
                    next_q_target = next_q_target.masked_fill(~mask_t, float("-inf"))
                q_next = next_q_target.max(dim=-1)[0]

            rew_train = rewards[:, bi:].T  # (S, B)
            done_train = dones[:, bi:].T   # (S, B)
            td_target = rew_train + self.gamma * (1.0 - done_train) * q_next

        td_loss = F.mse_loss(q_current, td_target, reduction='none')  # (S, B)
        mask_train = mask[:, bi:].transpose(0, 1)  # (S, B)
        loss = (td_loss * mask_train).sum() / mask_train.sum().clamp(min=1)

        info = {
            "q_mean": q_train.mean().item(),
            "q_max": q_train.max().item(),
            "td_error": ((td_target - q_current).abs() * mask_train).sum().item()
            / mask_train.sum().clamp(min=1).item(),
        }
        return loss, info

    # ====================================================================
    #                       update (all agents)
    # ====================================================================

    def update(
        self,
        batch_dict: Dict[str, Union[TransitionBatch, SequenceBatch]],
        **kwargs,
    ) -> TrainingStats:
        """逐 agent 计算 loss、梯度更新、epsilon 衰减。"""
        all_loss = []
        all_q_mean = []
        all_q_max = []
        all_td_error = []
        all_epsilon = []

        for agent_id, batch in batch_dict.items():
            batch = batch.to_tensor(self.device)

            if self.is_recurrent:
                loss, info = self._compute_agent_loss_sequence(batch, agent_id)
            else:
                loss, info = self._compute_agent_loss_flat(batch, agent_id)

            optimizer = self._optimizers[agent_id]
            optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.policy.get_policy(agent_id).parameters(),
                    self.max_grad_norm,
                )
            optimizer.step()

            all_loss.append(loss.item())
            all_q_mean.append(info["q_mean"])
            all_q_max.append(info["q_max"])
            all_td_error.append(info["td_error"])

        # ---- epsilon 衰减（所有 agent 统一调度） ----
        if self.epsilon_decay_by_step:
            progress = min(1.0, self._update_count / self._exploration_updates)
            new_eps = self.epsilon_start + (
                self.epsilon_end - self.epsilon_start
            ) * progress
        else:
            first_vp = self.policy.get_policy(self.policy.agent_ids[0])
            new_eps = max(
                first_vp.get_epsilon() * self.epsilon_decay, self.epsilon_end
            )

        for aid in self.policy.agent_ids:
            self.policy.get_policy(aid).set_epsilon(new_eps)
        all_epsilon.append(new_eps)

        # ---- target network update ----
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._update_target_networks()

        return TrainingStats(
            loss=float(np.mean(all_loss)),
            extra={
                "q_mean": float(np.mean(all_q_mean)),
                "q_max": float(np.mean(all_q_max)),
                "td_error": float(np.mean(all_td_error)),
                "epsilon_mean": float(np.mean(all_epsilon)),
            },
        )

    # ====================================================================
    #                       target network update
    # ====================================================================

    def _update_target_networks(self):
        for aid in self.policy.agent_ids:
            source = self.policy.get_policy(aid)
            target = self.target_policies[aid]
            if self.tau < 1.0:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            else:
                target.load_state_dict(source.state_dict())

    # ====================================================================
    #                       epsilon 管理
    # ====================================================================

    def set_epsilon(self, epsilon: float, agent_id: Optional[str] = None):
        """设置 epsilon（None 表示所有 agent）。"""
        targets = [agent_id] if agent_id else self.policy.agent_ids
        for aid in targets:
            self.policy.get_policy(aid).set_epsilon(epsilon)

    def get_epsilon(self, agent_id: Optional[str] = None):
        if agent_id is not None:
            return self.policy.get_policy(agent_id).get_epsilon()
        return {
            aid: self.policy.get_policy(aid).get_epsilon()
            for aid in self.policy.agent_ids
        }
