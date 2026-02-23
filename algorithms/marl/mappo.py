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
        **kwargs,
    ):
        super().__init__(policy, critic, params, num_envs)

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

        # Value Normalization
        if self.use_value_norm:
            from utils.train_utils import RunningMeanStd
            self.ret_rms = RunningMeanStd(shape=(1,)).to(self.device)

            if value_norm_config is not None:
                self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
                self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
                self.ret_rms.count.fill_(1.0)
                print(f"[MAPPO] Loaded value_norm stats: mean={self.ret_rms.mean.item():.4f}, std={self.ret_rms.std.item():.4f}")
            else:
                print(f"[MAPPO] Initialized value_norm with default: mean=0, std=1")

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

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """
        Per-agent 向量化 GAE + 合并为单 RolloutBatch（参数共享需要）。

        batch.value 存储归一化尺度的值，确保 update 中 value clipping 在正确尺度上进行。
        """
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs

        all_obs, all_act, all_log_prob = [], [], []
        all_adv, all_ret, all_value = [], [], []
        all_critic_input, all_action_mask, all_active_mask = [], [], []

        for i, agent in enumerate(agents):
            batch = batch_dict[agent]
            final_gs = batch.final_global_state
            batch = batch.to_tensor(self.device)
            total_size = batch.global_state.shape[0]
            T = total_size // N

            # critic_input = global_state + agent_one_hot
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
        """
        all_pg_loss, all_v_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl = []
        all_actor_grad_norm, all_critic_grad_norm = [], []

        for epoch in range(update_epochs):
            epoch_approx_kl = []

            for mb in batch.split(size=minibatch_size, shuffle=True, merge_last=True):
                # Active mask 权重
                if mb.active_mask is not None:
                    am = mb.active_mask.float()
                    am_sum = am.sum().clamp(min=1.0)
                else:
                    am = None

                # Advantage normalization
                mb_adv = mb.adv
                if self.normalize_advantage:
                    if am is not None:
                        active_adv = mb_adv[am > 0.5]
                        if active_adv.numel() > 1:
                            mb_adv = (mb_adv - active_adv.mean()) / (active_adv.std() + 1e-8)
                    else:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ===== Actor update =====
                new_log_prob, entropy = self.policy.evaluate_actions_flat(
                    mb.obs, mb.act, action_mask=mb.action_mask
                )
                logratio = new_log_prob - mb.log_prob
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
                new_value = self.critic(mb.global_state).squeeze(-1)

                if self.use_value_norm and self.ret_rms is not None:
                    target = (mb.ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                else:
                    target = mb.ret

                if self.clip_vloss:
                    v_loss_unclipped = (new_value - target) ** 2
                    v_clipped = mb.value + torch.clamp(
                        new_value - mb.value, -self.clip_range, self.clip_range
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
            }
        )

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)
