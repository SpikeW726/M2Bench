"""MAPPO: Multi-Agent PPO with Parameter Sharing and Centralized Critic."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from algorithms.algorithm_base import TrainingStats
from algorithms.rl.ppo import PPOBase
from configs.algo_configs import MAPPOParams
from policies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class MAPPOAlgo(PPOBase):
    """
    MAPPO: 参数共享 + Centralized Critic + 双优化器。

    继承 PPOBase 的 PPO 超参，override 以下方法：
    - prepare_batch: per-agent 循环 + 合并为单 batch（参数共享需要）
    - critic_input = global_state + agent_one_hot（CTDE 范式）
    - update: 双优化器分别更新 actor/critic
    - _compute_trunc_bootstrap_with_onehot: one-hot 拼接 + value denorm
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        params: MAPPOParams,
        num_envs: int,
        total_iterations: Optional[int] = None,
        optimizer_steps_per_iter: Optional[int] = None,
        value_norm_config: Optional[Dict] = None,
    ):
        super().__init__(policy, critic, params, num_envs, value_norm_config=value_norm_config)

        # 双优化器
        self.actor_optimizer = torch.optim.Adam(policy.parameters(), lr=params.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=params.critic_lr)

        # LR scheduler
        self.use_lr_scheduler = params.use_lr_scheduler
        self.actor_scheduler = None
        self.critic_scheduler = None

        if params.use_lr_scheduler and total_iterations is not None and optimizer_steps_per_iter is not None:
            actor_decay_steps = int(total_iterations * params.actor_lr_decay_ratio * optimizer_steps_per_iter)
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_optimizer,
                start_factor=params.actor_lr_start_factor,
                end_factor=params.actor_lr_end_factor,
                total_iters=actor_decay_steps,
            )

            critic_decay_steps = int(total_iterations * params.critic_lr_decay_ratio * optimizer_steps_per_iter)
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_optimizer,
                start_factor=params.critic_lr_start_factor,
                end_factor=params.critic_lr_end_factor,
                total_iters=critic_decay_steps,
            )

    # ====================================================================
    #                MAPPO-specific: Truncation Bootstrap
    # ====================================================================

    def _compute_trunc_bootstrap_with_onehot(
        self,
        truncateds: Optional[torch.Tensor],     # (T, N) or None
        final_global_states,                     # T x N nested list
        agent_idx: int,
        num_agents: int,
        T: int,
        N: int,
    ) -> torch.Tensor:
        """计算 truncation 处 bootstrap V，使用 one-hot 拼接 + value denorm。"""
        device = self.device
        trunc_bootstrap = torch.zeros(T, N, device=device)

        if truncateds is None or final_global_states is None:
            return trunc_bootstrap

        trunc_mask = truncateds > 0.5
        if not trunc_mask.any():
            return trunc_bootstrap

        trunc_positions = trunc_mask.nonzero(as_tuple=False)

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
            one_hot = torch.zeros(len(batch_states), num_agents, device=device)
            one_hot[:, agent_idx] = 1.0
            critic_in = torch.cat([states_t, one_hot], dim=-1)

            if self.is_critic_recurrent:
                zero_h = self.critic.get_initial_hidden(len(batch_states), device)
                v_norm, _ = self.critic(critic_in, zero_h)
                v_norm = v_norm.squeeze(-1)
            else:
                v_norm = self.critic(critic_in).squeeze(-1)

            if self.use_value_norm and self.ret_rms is not None:
                v_real = v_norm * self.ret_rms.std + self.ret_rms.mean
            else:
                v_real = v_norm

            for vi, k in enumerate(valid_k_indices):
                t_idx = trunc_positions[k, 0].item()
                e_idx = trunc_positions[k, 1].item()
                trunc_bootstrap[t_idx, e_idx] = v_real[vi]

        return trunc_bootstrap

    # ====================================================================
    #                          Batch 预处理
    # ====================================================================

    def _critic_rnn_values_with_onehot(
        self,
        critic_input: torch.Tensor,    # (T*N, critic_dim) 已拼接 one-hot
        done_2d: torch.Tensor,          # (T, N)
        agent_key: str,
        T: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """逐步处理 RNN critic（per-agent），done 边界重置 hidden。"""
        critic_seq = critic_input.view(T, N, -1)

        if not hasattr(self, "_critic_hidden_per_agent"):
            self._critic_hidden_per_agent: Dict[str, torch.Tensor] = {}

        hidden = self._critic_hidden_per_agent.get(agent_key, None)
        if hidden is None or hidden.shape[1] != N:
            hidden = self.critic.get_initial_hidden(N, critic_input.device)

        all_values = []
        all_hidden = []

        for t in range(T):
            if t > 0:
                done_prev = done_2d[t - 1] > 0.5
                if done_prev.any():
                    hidden = hidden.clone()
                    hidden[:, done_prev, :] = 0.0

            all_hidden.append(hidden)
            v_t, hidden = self.critic(critic_seq[t], hidden)
            all_values.append(v_t.squeeze(-1))

        self._critic_hidden_per_agent[agent_key] = hidden.detach()

        values_norm = torch.stack(all_values, dim=0)        # (T, N)
        critic_rnn_h = torch.stack(all_hidden, dim=0)       # (T, rN, N, H)
        return values_norm, critic_rnn_h

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """
        Per-agent 向量化 GAE + 合并为单 RolloutBatch（参数共享需要）。

        batch.value 存储归一化尺度的值，确保 update 中 value clipping 在正确尺度上进行。
        RNN 时额外收集 rnn_hidden / critic_rnn_hidden，并记录 T/N/num_agents 供 chunk_split 使用。
        """
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs

        all_obs, all_act, all_log_prob = [], [], []
        all_adv, all_ret, all_value = [], [], []
        all_critic_input, all_action_mask, all_active_mask = [], [], []
        all_rnn_hidden = []
        all_critic_rnn_hidden = []

        saved_T = 0

        for i, agent in enumerate(agents):
            batch = batch_dict[agent]
            final_gs = batch.final_global_state
            batch = batch.to_tensor(self.device)
            total_size = batch.global_state.shape[0]
            T = total_size // N
            saved_T = T

            one_hot = torch.zeros(total_size, num_agents, device=self.device)
            one_hot[:, i] = 1.0
            critic_input = torch.cat([batch.global_state, one_hot], dim=-1)

            rew_2d = batch.rew.view(T, N)
            done_2d = batch.done.view(T, N)
            truncated_2d = batch.truncated.view(T, N) if batch.truncated is not None else None

            active_mask_2d = None
            if self.use_active_mask and batch.active_mask is not None:
                active_mask_2d = batch.active_mask.view(T, N)

            with torch.no_grad():
                if self.is_critic_recurrent:
                    values_norm, critic_rnn_h = self._critic_rnn_values_with_onehot(
                        critic_input, done_2d, agent, T, N,
                    )
                    # (T, rN, N, H) -> (T*N, rN, H)
                    rN, H = critic_rnn_h.shape[1], critic_rnn_h.shape[3]
                    all_critic_rnn_hidden.append(
                        critic_rnn_h.permute(0, 2, 1, 3).reshape(T * N, rN, H)
                    )
                else:
                    values_norm = self.critic(critic_input).squeeze(-1).view(T, N)

                if self.use_value_norm and self.ret_rms is not None:
                    values = values_norm * self.ret_rms.std + self.ret_rms.mean
                else:
                    values = values_norm

                trunc_bootstrap = self._compute_trunc_bootstrap_with_onehot(
                    truncated_2d, final_gs, i, num_agents, T, N,
                )

                adv, ret = self._gae_vectorized(
                    rew_2d, values, done_2d, truncated_2d, trunc_bootstrap, active_mask_2d,
                )
                values_flat = values_norm.view(-1)

            all_obs.append(batch.obs)
            all_act.append(batch.act)
            all_log_prob.append(batch.log_prob)
            all_adv.append(adv)
            all_ret.append(ret)
            all_value.append(values_flat)
            all_critic_input.append(critic_input)
            if batch.action_mask is not None:
                all_action_mask.append(batch.action_mask)
            if self.use_active_mask and batch.active_mask is not None:
                all_active_mask.append(batch.active_mask)
            if batch.rnn_hidden is not None:
                all_rnn_hidden.append(batch.rnn_hidden)

        # 更新 Value Normalization 统计量
        if self.use_value_norm and self.ret_rms is not None:
            all_ret_tensor = torch.cat(all_ret, dim=0)
            if all_active_mask:
                active_flat = torch.cat(all_active_mask, dim=0) > 0.5
                active_ret = all_ret_tensor[active_flat]
                if active_ret.numel() > 0:
                    self.ret_rms.update(active_ret)
            else:
                self.ret_rms.update(all_ret_tensor)

        self._last_T = saved_T
        self._last_N = N
        self._last_num_agents = num_agents

        return RolloutBatch(
            obs=torch.cat(all_obs, dim=0),
            act=torch.cat(all_act, dim=0),
            log_prob=torch.cat(all_log_prob, dim=0),
            adv=torch.cat(all_adv, dim=0),
            ret=torch.cat(all_ret, dim=0),
            value=torch.cat(all_value, dim=0),
            global_state=torch.cat(all_critic_input, dim=0),
            action_mask=torch.cat(all_action_mask, dim=0) if all_action_mask else None,
            active_mask=torch.cat(all_active_mask, dim=0) if all_active_mask else None,
            rnn_hidden=torch.cat(all_rnn_hidden, dim=0) if all_rnn_hidden else None,
            critic_rnn_hidden=torch.cat(all_critic_rnn_hidden, dim=0) if all_critic_rnn_hidden else None,
        )

    # ====================================================================
    #                   PPO Update（双优化器）
    # ====================================================================

    def update(
        self,
        batch: RolloutBatch,
        minibatch_size: int,
        update_epochs: int,
        update_actor: bool = True,
    ) -> TrainingStats:
        """
        双优化器 PPO update，支持 active_mask loss masking。
        RNN 时使用 chunk_split + evaluate_actions_sequence_flat。
        """
        all_pg_loss, all_v_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl = []
        all_actor_grad_norm, all_critic_grad_norm = [], []

        _any_rnn = self.is_any_recurrent

        for epoch in range(update_epochs):
            epoch_approx_kl = []

            if _any_rnn:
                mb_iter = batch.chunk_split(
                    chunk_len=self.data_chunk_length,
                    T=self._last_T, N=self._last_N,
                    num_agents=self._last_num_agents,
                    minibatch_size=minibatch_size,
                )
            else:
                mb_iter = batch.split(size=minibatch_size, shuffle=True, merge_last=True)

            for mb in mb_iter:
                # Active mask 权重
                if mb.active_mask is not None:
                    am = mb.active_mask.float()
                    if _any_rnn:
                        am = am.reshape(-1)
                    am_sum = am.sum().clamp(min=1.0)
                else:
                    am = None

                # Advantage normalization
                mb_adv = mb.adv
                if _any_rnn:
                    mb_adv = mb_adv.reshape(-1)
                if self.normalize_advantage:
                    if am is not None:
                        active_adv = mb_adv[am > 0.5]
                        if active_adv.numel() > 1:
                            mb_adv = (mb_adv - active_adv.mean()) / (active_adv.std() + 1e-8)
                    else:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ===== Actor update =====
                if self.is_recurrent:
                    new_log_prob, entropy = self.policy.evaluate_actions_sequence_flat(
                        mb.obs, mb.act, mb.rnn_hidden,
                        action_mask=mb.action_mask,
                    )
                    new_log_prob = new_log_prob.reshape(-1)
                    entropy = entropy.reshape(-1)
                else:
                    obs_flat = mb.obs.reshape(-1, mb.obs.shape[-1]) if _any_rnn else mb.obs
                    act_flat = mb.act.reshape(-1) if _any_rnn else mb.act
                    am_flat = mb.action_mask.reshape(-1, mb.action_mask.shape[-1]) if (_any_rnn and mb.action_mask is not None) else mb.action_mask
                    new_log_prob, entropy = self.policy.evaluate_actions_flat(
                        obs_flat, act_flat, action_mask=am_flat,
                    )
                mb_log_prob = mb.log_prob.reshape(-1) if _any_rnn else mb.log_prob

                logratio = new_log_prob - mb_log_prob
                ratio = logratio.exp()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)

                if am is not None:
                    pg_loss = (pg_loss_per_sample * am).sum() / am_sum
                    entropy_loss = (entropy * am).sum() / am_sum
                else:
                    pg_loss = pg_loss_per_sample.mean()
                    entropy_loss = entropy.mean()

                actor_loss = pg_loss - self.ent_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                if update_actor:
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    all_actor_grad_norm.append(actor_grad_norm.item())

                # ===== Critic update =====
                critic_in = mb.global_state
                if self.is_critic_recurrent:
                    new_value_seq, _ = self.critic.forward_sequence(
                        critic_in, mb.critic_rnn_hidden,
                    )
                    new_value = new_value_seq.squeeze(-1).reshape(-1)
                else:
                    if _any_rnn:
                        critic_in = critic_in.reshape(-1, critic_in.shape[-1])
                    new_value = self.critic(critic_in).squeeze(-1)

                mb_ret = mb.ret.reshape(-1) if _any_rnn else mb.ret
                mb_value = mb.value.reshape(-1) if _any_rnn and mb.value is not None else mb.value

                if self.use_value_norm and self.ret_rms is not None:
                    target = (mb_ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                else:
                    target = mb_ret

                if self.clip_vloss:
                    v_loss_unclipped = (new_value - target) ** 2
                    v_clipped = mb_value + torch.clamp(
                        new_value - mb_value, -self.clip_range, self.clip_range,
                    )
                    v_loss_clipped = (v_clipped - target) ** 2
                    v_loss_per_sample = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    v_loss_per_sample = 0.5 * (new_value - target) ** 2

                if am is not None:
                    v_loss = (v_loss_per_sample * am).sum() / am_sum
                else:
                    v_loss = v_loss_per_sample.mean()

                critic_loss = self.vf_coef * v_loss

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
                all_critic_grad_norm.append(critic_grad_norm.item())

                # LR scheduler
                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()

                # Stats
                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(v_loss.item())
                all_entropy.append(entropy_loss.item())

                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    all_clipfrac.append(clipfrac)
                    mb_approx_kl = ((ratio - 1) - logratio).mean().item()
                    epoch_approx_kl.append(mb_approx_kl)

            # KL early stopping
            if epoch_approx_kl:
                avg_epoch_kl = np.mean(epoch_approx_kl)
                all_approx_kl.append(avg_epoch_kl)
                if self.target_kl is not None and avg_epoch_kl > self.target_kl:
                    break

        return TrainingStats(
            loss=np.mean(all_pg_loss) + self.vf_coef * np.mean(all_v_loss),
            policy_loss=np.mean(all_pg_loss),
            value_loss=np.mean(all_v_loss),
            entropy=np.mean(all_entropy),
            extra={
                "clipfrac": np.mean(all_clipfrac),
                "approx_kl": np.mean(all_approx_kl) if all_approx_kl else 0.0,
                "actor_grad_norm": np.mean(all_actor_grad_norm) if all_actor_grad_norm else 0.0,
                "critic_grad_norm": np.mean(all_critic_grad_norm),
            },
        )

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)
