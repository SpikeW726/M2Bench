"""Shared reinforcement-learning algorithm infrastructure.

The hierarchy separates reusable update mechanics from algorithm-specific
losses::

    BaseAlgorithm
    |-- ActorCriticOnPolicyAlgo
    |   `-- A2CBase
    |       |-- A2CAlgo
    |       |-- MAA2CAlgo + CentralizedCriticMixin
    |       `-- PPOBase
    |           |-- PPOAlgo
    |           |-- IPPOAlgo
    |           `-- MAPPOAlgo + CentralizedCriticMixin
    `-- QLearningOffPolicyAlgo

``ActorCriticOnPolicyAlgo`` owns GAE, value normalization, truncation
bootstrapping, and batch preparation. ``A2CBase`` supplies the update template,
while PPO-specific clipping and KL stopping are implemented by ``PPOBase``.
Centralized-training helpers live in ``CentralizedCriticMixin``. Concrete
subclasses create their own optimizers and learning-rate schedulers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from configs.algo_configs import OnPolicyParams
from policies.rl.rl_base import RLBasePolicy, ActorPolicy
from data.batch import BaseBatch, RolloutBatch, TransitionBatch

@dataclass
class TrainingStats:
    """Statistics returned by an algorithm update."""

    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

# Base Algorithm.

class BaseAlgorithm(nn.Module, ABC):
    """Base class responsible for loss computation and parameter updates."""

    def __init__(self, policy: RLBasePolicy):
        super().__init__()
        self.policy = policy
        self.device = policy.device

        # Created by subclasses.
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler = None

    @abstractmethod
    def update(self, batch, **kwargs) -> TrainingStats:
        pass

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)

# On-Policy Algorithm.

class ActorCriticOnPolicyAlgo(BaseAlgorithm):
    """Algorithm-independent infrastructure for on-policy actor-critic methods.

    This class implements standard and transparent GAE, truncation
    bootstrapping, value normalization, and rollout preprocessing. Concrete
    A2C and PPO classes provide the optimization step.
    """

    def __init__(
        self,
        policy,
        critic: Optional[nn.Module],
        params: OnPolicyParams,
        num_envs: int = 1,
        value_norm_config: Optional[Dict] = None,
    ):
        super().__init__(policy)
        self.critic = critic.to(self.device) if critic is not None else None
        self.num_envs = num_envs

        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.vf_coef = params.vf_coef
        self.ent_coef = params.ent_coef
        self.max_grad_norm = params.max_grad_norm
        self.normalize_advantage = params.normalize_advantage
        self.use_active_mask = params.use_active_mask
        self.use_value_norm = params.use_value_norm
        self.data_chunk_length = params.data_chunk_length

        # Value Normalization.
        self.ret_rms = None
        if self.use_value_norm:
            from utils.train_utils import RunningMeanStd
            self.ret_rms = RunningMeanStd(shape=(1,)).to(self.device)
            if value_norm_config is not None:
                self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
                self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
                self.ret_rms.count.fill_(float(value_norm_config.get('ret_count', 1)))

        # RNN critic hidden state.
        self._critic_hidden: Optional[torch.Tensor] = None

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.policy, "is_recurrent", False)

    @property
    def is_critic_recurrent(self) -> bool:
        if self.critic is None:
            return False
        return getattr(self.critic, "is_recurrent", False)

    @property
    def is_any_recurrent(self) -> bool:
        return self.is_recurrent or self.is_critic_recurrent

    def _gae_vectorized(
        self,
        rewards: torch.Tensor,                          # (T, N).
        values: torch.Tensor,                           # (T, N) real scale.
        dones: torch.Tensor,                            # (T, N).
        truncateds: Optional[torch.Tensor],             # (T, N) or None.
        trunc_bootstrap: torch.Tensor,                  # Shape: (T, N).
        active_mask: Optional[torch.Tensor] = None,     # (T, N) or None.
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE for all environments in parallel.

        With no ``active_mask``, this is standard GAE. When the mask is present,
        rewards from inactive transitions are accumulated into the preceding
        active transition. ``trunc_bootstrap`` contains precomputed ``V(s')``
        values at truncations and zero elsewhere.

        Returns flattened advantage and return tensors of shape ``(T * N,)``.
        """

        T, N = rewards.shape
        device = rewards.device

        if truncateds is not None:
            trunc_mask = truncateds > 0.5
        else:
            trunc_mask = torch.zeros(T, N, dtype=torch.bool, device=device)

        done_mask = dones > 0.5
        term_mask = done_mask & ~trunc_mask

        last_step_bootstrap = values[-1].clone()  # (N,).

        use_transparent = active_mask is not None
        active_bool = active_mask > 0.5 if use_transparent else None

        advantages = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)

        if not use_transparent:

            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_step_bootstrap.clone()
                else:
                    next_val = values[t + 1].clone()

                next_val = torch.where(term_mask[t], torch.zeros_like(next_val), next_val)
                if trunc_mask[t].any():
                    next_val = torch.where(trunc_mask[t], trunc_bootstrap[t], next_val)

                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = torch.where(
                    done_mask[t],
                    delta,
                    delta + self.gamma * self.gae_lambda * last_gae,
                )
                advantages[t] = last_gae
        else:

            next_active_val = last_step_bootstrap.clone()
            acc_reward = torch.zeros(N, device=device)
            acc_discount = torch.ones(N, device=device)
            acc_gae_decay = torch.ones(N, device=device)

            for t in reversed(range(T)):
                active_t = active_bool[t]

                # Handle episode boundaries.
                if done_mask[t].any():
                    done_val = torch.where(
                        term_mask[t],
                        torch.zeros_like(next_active_val),
                        torch.where(trunc_mask[t], trunc_bootstrap[t], next_active_val),
                    )
                    next_active_val = torch.where(done_mask[t], done_val, next_active_val)
                    last_gae = torch.where(done_mask[t], torch.zeros_like(last_gae), last_gae)
                    acc_reward = torch.where(done_mask[t], torch.zeros_like(acc_reward), acc_reward)
                    acc_discount = torch.where(done_mask[t], torch.ones_like(acc_discount), acc_discount)
                    acc_gae_decay = torch.where(done_mask[t], torch.ones_like(acc_gae_decay), acc_gae_decay)

                effective_next_val = acc_reward + acc_discount * next_active_val
                delta_active = rewards[t] + self.gamma * effective_next_val - values[t]
                gae_active = delta_active + self.gamma * self.gae_lambda * acc_gae_decay * last_gae

                new_acc_reward = rewards[t] + self.gamma * acc_reward
                new_acc_discount = self.gamma * acc_discount
                new_acc_gae_decay = self.gamma * self.gae_lambda * acc_gae_decay

                # Apply the update selectively.
                last_gae = torch.where(active_t, gae_active, last_gae)
                advantages[t] = torch.where(active_t, gae_active, torch.zeros_like(gae_active))
                next_active_val = torch.where(active_t, values[t], next_active_val)
                acc_reward = torch.where(active_t, torch.zeros_like(acc_reward), new_acc_reward)
                acc_discount = torch.where(active_t, torch.ones_like(acc_discount), new_acc_discount)
                acc_gae_decay = torch.where(active_t, torch.ones_like(acc_gae_decay), new_acc_gae_decay)

        returns = advantages + values
        return advantages.view(-1), returns.view(-1)

    def _compute_trunc_bootstrap(
        self,
        truncateds: torch.Tensor,       # (T, N) or None.
        final_global_states,             # T x N nested list, or None.
        critic_input: torch.Tensor,      # Shape: (T*N, critic_dim).
        T: int,
        N: int,
    ) -> torch.Tensor:
        device = critic_input.device
        trunc_bootstrap = torch.zeros(T, N, device=device)

        if truncateds is None or final_global_states is None:
            return trunc_bootstrap

        trunc_mask = truncateds > 0.5
        if not trunc_mask.any():
            return trunc_bootstrap

        trunc_positions = trunc_mask.nonzero(as_tuple=False)  # (K, 2).

        batch_states = []
        valid_k_indices = []
        for k in range(len(trunc_positions)):
            t_idx = trunc_positions[k, 0].item()
            e_idx = trunc_positions[k, 1].item()
            fs = final_global_states[t_idx][e_idx] if final_global_states else None
            if fs is not None:
                batch_states.append(
                    torch.as_tensor(fs, dtype=torch.float32, device=device)
                )
                valid_k_indices.append(k)

        if batch_states:
            states_t = torch.stack(batch_states)

            if self.is_critic_recurrent:

                zero_h = self.critic.get_initial_hidden(len(batch_states), device)
                v_raw, _ = self.critic(states_t, zero_h)
                v_raw = v_raw.squeeze(-1)
            else:
                v_raw = self.critic(states_t).squeeze(-1)

            if self.use_value_norm and self.ret_rms is not None:
                v_real = v_raw * self.ret_rms.std + self.ret_rms.mean
            else:
                v_real = v_raw

            for vi, k in enumerate(valid_k_indices):
                t_idx = trunc_positions[k, 0].item()
                e_idx = trunc_positions[k, 1].item()
                trunc_bootstrap[t_idx, e_idx] = v_real[vi]

        return trunc_bootstrap

    def _critic_rnn_values(
        self,
        critic_input: torch.Tensor,    # (T*N, critic_dim).
        done_2d: torch.Tensor,          # (T, N).
        T: int,
        N: int,
        agent_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_seq = critic_input.view(T, N, -1)

        # Select per-agent or shared hidden state.
        if agent_key is not None:
            if not hasattr(self, '_critic_hidden_per_agent'):
                self._critic_hidden_per_agent: Dict[str, torch.Tensor] = {}
            hidden = self._critic_hidden_per_agent.get(agent_key)
            if hidden is None or hidden.shape[1] != N:
                hidden = self.critic.get_initial_hidden(N, critic_input.device)
        else:
            if self._critic_hidden is None or self._critic_hidden.shape[1] != N:
                self._critic_hidden = self.critic.get_initial_hidden(N, critic_input.device)
            hidden = self._critic_hidden

        all_values = []
        all_hidden = []

        for t in range(T):
            if t > 0:
                done_prev = done_2d[t - 1] > 0.5      # (N,).
                if done_prev.any():
                    hidden = hidden.clone()
                    hidden[:, done_prev, :] = 0.0

            all_hidden.append(hidden)

            v_t, hidden = self.critic(critic_seq[t], hidden)    # (N, 1), (rN, N, H).
            all_values.append(v_t.squeeze(-1))                  # (N,).

        # Persist the final hidden state.
        if agent_key is not None:
            self._critic_hidden_per_agent[agent_key] = hidden.detach()
        else:
            self._critic_hidden = hidden.detach()

        values_norm = torch.stack(all_values, dim=0)            # (T, N).
        critic_rnn_hidden = torch.stack(all_hidden, dim=0)      # (T, rN, N, H).
        return values_norm, critic_rnn_hidden

    def prepare_batch(self, batch: RolloutBatch) -> RolloutBatch:
        final_gs = batch.final_global_state
        batch = batch.to_tensor(self.device)

        N = self.num_envs
        total_size = batch.obs.shape[0]
        T = total_size // N

        with torch.no_grad():
            critic_input = batch.global_state if batch.global_state is not None else batch.obs

            done_2d = batch.done.view(T, N)

            if self.is_critic_recurrent:
                values_norm, critic_rnn_h = self._critic_rnn_values(
                    critic_input, done_2d, T, N,
                )
                # Shape: (T, rN, N, H) -> (T*N, rN, H).
                rN, H = critic_rnn_h.shape[1], critic_rnn_h.shape[3]
                batch.critic_rnn_hidden = critic_rnn_h.permute(0, 2, 1, 3).reshape(T * N, rN, H)
            else:
                values_norm = self.critic(critic_input).squeeze(-1).view(T, N)

            if self.use_value_norm and self.ret_rms is not None:
                values = values_norm * self.ret_rms.std + self.ret_rms.mean
            else:
                values = values_norm

            rew_2d = batch.rew.view(T, N)
            truncated_2d = batch.truncated.view(T, N) if batch.truncated is not None else None

            trunc_bootstrap = self._compute_trunc_bootstrap(
                truncated_2d, final_gs, critic_input, T, N,
            )

            active_mask_2d = None
            if self.use_active_mask and batch.active_mask is not None:
                active_mask_2d = batch.active_mask.view(T, N)

            adv, ret = self._gae_vectorized(
                rew_2d, values, done_2d, truncated_2d, trunc_bootstrap, active_mask_2d,
            )
            values_flat = values_norm.view(-1)

        # Update value-normalization statistics.
        if self.use_value_norm and self.ret_rms is not None:
            if active_mask_2d is not None:
                active_flat = batch.active_mask > 0.5
                active_ret = ret[active_flat]
                if active_ret.numel() > 0:
                    self.ret_rms.update(active_ret)
            else:
                self.ret_rms.update(ret)

        batch.adv = adv
        batch.ret = ret
        batch.value = values_flat
        return batch

# Off-Policy Algorithm.

class QLearningOffPolicyAlgo(BaseAlgorithm):
    """Shared optimization and target-network logic for Q-learning methods.

    Subclasses implement ``compute_loss``. ``tau < 1`` performs soft target
    updates; otherwise a hard copy is made every ``target_update_freq`` updates.
    Both flat transition and recurrent sequence batches are supported by concrete
    algorithms.
    """

    def __init__(
        self,
        policy: RLBasePolicy,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_freq: int = 1,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(policy)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.target_update_freq = target_update_freq
        self._update_count = 0

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.target_policy: Optional[RLBasePolicy] = None

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.policy, "is_recurrent", False)

    # target network.

    def _create_target_network(self):
        import copy
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.set_training_mode(False)
        for param in self.target_policy.parameters():
            param.requires_grad = False

    def _soft_update_target(self):
        if self.target_policy is None:
            return
        for tp, p in zip(self.target_policy.parameters(), self.policy.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _hard_update_target(self):
        if self.target_policy is not None:
            self.target_policy.load_state_dict(self.policy.state_dict())

    def update(self, batch: BaseBatch, **kwargs) -> TrainingStats:
        batch = batch.to_tensor(self.device)
        loss, stats = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stats.loss = float(loss.detach().item())
        if stats.extra:
            stats.extra = {
                k: float(v.detach().item()) if isinstance(v, torch.Tensor) else v
                for k, v in stats.extra.items()
            }

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            if self.tau < 1.0:
                self._soft_update_target()
            else:
                self._hard_update_target()
        return stats

    @abstractmethod
    def compute_loss(self, batch: BaseBatch) -> tuple[torch.Tensor, TrainingStats]:
        pass
