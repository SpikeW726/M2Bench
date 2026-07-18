"""Independent Q-Learning for multi-agent environments.

Each agent owns an independent value policy and Q-network, with no centralized
critic or gradient coordination. The algorithm schedules each policy's epsilon.
MLPs train from flat transition batches; recurrent networks train DRQN-style
sequence batches from zero initial hidden states.
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
    """Train independent per-agent Q-networks and epsilon schedules.

    Flat batches use Double-DQN targets directly. Recurrent batches support
    burn-in, masked sequence loss, and optional synchronized multi-step targets
    that advance from one READY decision to the next.
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: IQLParams,
        total_updates: int = 0,
    ):
        if getattr(policy, "shared", False):
            raise ValueError(
                "IQLAlgo does not support shared_policy=True because multiple optimizers "
                "would reference the same parameters. Use VDN or QMIX for parameter sharing."
            )

        super().__init__(policy)
        self.params = params
        self.gamma = params.gamma
        self.max_grad_norm = params.max_grad_norm
        self.tau = params.tau
        self.target_update_freq = params.target_update_freq
        self.seq_len = params.seq_len

        # target networks.
        self.target_policies = nn.ModuleDict()
        for aid in policy.agent_ids:
            target = copy.deepcopy(policy.get_policy(aid))
            target.set_training_mode(False)
            for p in target.parameters():
                p.requires_grad = False
            self.target_policies[aid] = target

        # per-agent optimizers.
        self._optimizers: Dict[str, torch.optim.Adam] = {
            aid: torch.optim.Adam(policy.get_policy(aid).parameters(), lr=params.lr)
            for aid in policy.agent_ids
        }

        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay_steps = max(1, int(params.epsilon_decay_steps))

        for aid in policy.agent_ids:
            policy.get_policy(aid).set_epsilon(params.epsilon_start)

        self._update_count = 0

    def set_training_mode(self, mode: bool):
        super().set_training_mode(mode)
        for target in self.target_policies.values():
            target.set_training_mode(False)

    @property
    def is_recurrent(self) -> bool:
        first_policy = self.policy.get_policy(self.policy.agent_ids[0])
        return getattr(first_policy, "is_recurrent", False)

    # per-agent loss (MLP).

    def _compute_agent_loss_flat(
        self,
        batch: TransitionBatch,
        agent_id: str,
    ) -> tuple[torch.Tensor, dict]:
        agent_policy: ValuePolicy = self.policy.get_policy(agent_id)
        target_policy: ValuePolicy = self.target_policies[agent_id]

        obs = batch.obs
        actions = batch.act.long()
        rewards = batch.rew
        next_obs = batch.next_obs
        dones = batch.done
        next_action_mask = getattr(batch, 'next_action_mask', None)
        active_mask = getattr(batch, 'active_mask', None)
        gamma_power = getattr(batch, 'gamma_power', None)

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

            # clamp:action mask.
            q_next = q_next.clamp(min=-1e9, max=1e9)

            bootstrap = torch.where(dones.bool(), torch.zeros_like(q_next), q_next)
            if gamma_power is not None:
                td_target = rewards + gamma_power * bootstrap
            else:
                td_target = rewards + self.gamma * bootstrap

        per_sample_loss = F.mse_loss(q_current, td_target, reduction='none')

        if gamma_power is None and active_mask is not None:
            am_sum = active_mask.sum().clamp(min=1)
            loss = (per_sample_loss * active_mask).sum() / am_sum
        else:
            loss = per_sample_loss.mean()

        info = {
            "q_mean": q_values.detach().mean(),
            "q_max": q_values.detach().max(),
            "td_error": (td_target - q_current).detach().abs().mean(),
        }
        return loss, info

    # per-agent loss (RNN sequence).

    def _compute_agent_loss_sequence(
        self,
        batch: SequenceBatch,
        agent_id: str,
    ) -> tuple[torch.Tensor, dict]:
        agent_policy: ValuePolicy = self.policy.get_policy(agent_id)
        target_policy: ValuePolicy = self.target_policies[agent_id]

        B, T_total = batch.obs.shape[:2]
        device = batch.obs.device
        bi = getattr(batch, 'burn_in_len', 0)

        obs_seq = batch.obs.transpose(0, 1)            # (T, B, D).
        next_obs_seq = batch.next_obs.transpose(0, 1)  # (T, B, D).
        actions = batch.act.long()                      # (B, T).
        rewards = batch.rew                             # (B, T).
        dones = batch.done                              # (B, T).
        mask = batch.mask                               # (B, T).

        h0 = agent_policy.q_network.get_initial_hidden(B, device)

        T_total = T_total

        full_obs_seq = torch.cat([obs_seq, next_obs_seq[-1:]], dim=0)  # (T+1, B, D).
        q_full_ext, _ = agent_policy.q_network.forward_sequence(full_obs_seq, h0)  # (T+1, B, act_dim).

        q_train = q_full_ext[bi:T_total]  # (S, B, act_dim).
        act_train = actions[:, bi:].T.unsqueeze(-1)  # (S, B, 1).
        q_current = q_train.gather(2, act_train).squeeze(-1)  # (S, B).

        with torch.no_grad():
            h0_target = target_policy.q_network.get_initial_hidden(B, device)

            next_am = getattr(batch, 'next_action_mask', None)
            mask_t = None
            if next_am is not None:
                next_am_train = next_am[:, bi:].transpose(0, 1)  # (S, B, act_dim).
                mask_t = next_am_train.bool().to(device)

            if self.params.use_double_dqn:

                next_q_online = q_full_ext[bi + 1:T_total + 1].detach()  # (S, B, act_dim).
                if mask_t is not None:
                    next_q_online = next_q_online.masked_fill(~mask_t, float("-inf"))
                next_actions = next_q_online.argmax(dim=-1, keepdim=True)  # (S, B, 1).

                next_q_tgt_ext, _ = target_policy.q_network.forward_sequence(full_obs_seq, h0_target)
                q_next = next_q_tgt_ext[bi + 1:T_total + 1].gather(2, next_actions).squeeze(-1)
            else:
                next_q_tgt_ext, _ = target_policy.q_network.forward_sequence(full_obs_seq, h0_target)
                next_q_target = next_q_tgt_ext[bi + 1:T_total + 1]  # (S, B, act_dim).
                if mask_t is not None:
                    next_q_target = next_q_target.masked_fill(~mask_t, float("-inf"))
                q_next = next_q_target.max(dim=-1)[0]  # (S, B).

            # clamp:action mask.
            q_next = q_next.clamp(min=-1e9, max=1e9)

            rew_train = rewards[:, bi:].T  # (S, B).
            done_train = dones[:, bi:].T   # (S, B).

            active_mask = getattr(batch, 'active_mask', None)
            if active_mask is not None:
                am_train = active_mask[:, bi:].transpose(0, 1)  # (S, B).

                td_target = self._compute_multistep_td_targets(
                    rew_train, q_next, done_train, am_train,
                )
            else:

                bootstrap = torch.where(done_train.bool(), torch.zeros_like(q_next), q_next)
                td_target = rew_train + self.gamma * bootstrap

        td_loss = F.mse_loss(q_current, td_target, reduction='none')  # (S, B).
        mask_train = mask[:, bi:].transpose(0, 1)  # (S, B).

        if active_mask is not None:
            loss_mask = mask_train * am_train
        else:
            loss_mask = mask_train

        loss = (td_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        info = {
            "q_mean": q_train.detach().mean(),
            "q_max": q_train.detach().max(),
            "td_error": (
                ((td_target - q_current).detach().abs() * loss_mask).sum()
                / loss_mask.sum().clamp(min=1)
            ),
        }
        return loss, info

    def _compute_multistep_td_targets(
        self,
        rewards: torch.Tensor,
        q_values: torch.Tensor,
        dones: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        S, B = rewards.shape
        gamma = self.gamma
        td_targets = torch.zeros_like(rewards)

        acc_reward = torch.zeros(B, device=rewards.device)
        acc_gamma = torch.ones(B, device=rewards.device)
        next_active_q = torch.zeros(B, device=rewards.device)

        for s in range(S - 1, -1, -1):
            done_s = dones[s]
            am_s = active_mask[s]
            rew_s = rewards[s]

            reset = done_s > 0.5
            acc_reward = torch.where(reset, torch.zeros_like(acc_reward), acc_reward)
            acc_gamma = torch.where(reset, torch.ones_like(acc_gamma), acc_gamma)
            next_active_q = torch.where(reset, torch.zeros_like(next_active_q), next_active_q)

            is_active = am_s > 0.5

            td_targets[s] = torch.where(
                is_active,
                rew_s + gamma * (acc_reward + acc_gamma * next_active_q),
                torch.zeros_like(rew_s),
            )

            new_acc_reward = torch.where(is_active, torch.zeros_like(acc_reward), rew_s + gamma * acc_reward)
            new_acc_gamma = torch.where(is_active, torch.ones_like(acc_gamma), acc_gamma * gamma)
            new_next_active_q = torch.where(is_active, q_values[s], next_active_q)

            acc_reward = new_acc_reward
            acc_gamma = new_acc_gamma
            next_active_q = new_next_active_q

        return td_targets

    # update (all agents).

    def update(
        self,
        batch_dict: Dict[str, Union[TransitionBatch, SequenceBatch]],
        **kwargs,
    ) -> TrainingStats:
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

            all_loss.append(loss.detach())
            all_q_mean.append(info["q_mean"])
            all_q_max.append(info["q_max"])
            all_td_error.append(info["td_error"])

        global_step = int(kwargs.get("global_step", 0))
        warmup_steps = int(kwargs.get("warmup_steps", 0))
        effective_step = max(0, global_step - warmup_steps)
        progress = min(1.0, effective_step / self.epsilon_decay_steps)
        new_eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

        for aid in self.policy.agent_ids:
            self.policy.get_policy(aid).set_epsilon(new_eps)
        all_epsilon.append(new_eps)

        # target network update.
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._update_target_networks()

        def _mean_stat(vals) -> float:
            if vals and isinstance(vals[0], torch.Tensor):
                return float(torch.stack([v.detach() for v in vals]).mean().item())
            return float(np.mean(vals))

        return TrainingStats(
            loss=_mean_stat(all_loss),
            extra={
                "q_mean": _mean_stat(all_q_mean),
                "q_max": _mean_stat(all_q_max),
                "td_error": _mean_stat(all_td_error),
                "epsilon_mean": float(np.mean(all_epsilon)),
            },
        )

    # target network update.

    def _update_target_networks(self):
        for aid in self.policy.agent_ids:
            source = self.policy.get_policy(aid)
            target = self.target_policies[aid]
            if self.tau < 1.0:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            else:
                target.load_state_dict(source.state_dict())

    def set_epsilon(self, epsilon: float, agent_id: Optional[str] = None):
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
