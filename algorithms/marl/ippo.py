"""IPPO: Independent PPO — 每个 agent 独立策略网络，共享 critic，单优化器。"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from algorithms.algorithm_base import TrainingStats
from algorithms.rl.ppo import PPOBase
from configs.algo_configs import IPPOParams
from policies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class IPPOAlgo(PPOBase):
    """
    IPPO: 独立策略 + 共享 Critic + 单优化器。

    继承 PPOBase 的 PPO 超参 (clip_range, clip_vloss, target_kl)，
    override prepare_batch 和 update 以支持 per-agent 独立策略：
    - prepare_batch: per-agent 循环调用基类 _gae_vectorized
    - update: 同步 minibatch 切分，per-agent 累加 PPO loss
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        params: IPPOParams,
        num_envs: int = 1,
        total_iterations: Optional[int] = None,
        optimizer_steps_per_iter: Optional[int] = None,
        value_norm_config: Optional[Dict] = None,
    ):
        super().__init__(policy, critic, params, num_envs, value_norm_config=value_norm_config)

        # 单优化器: 所有独立 policy + critic
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()),
            lr=params.lr,
        )

        # LR scheduler
        self.lr_scheduler = None
        if params.use_lr_scheduler and total_iterations and optimizer_steps_per_iter:
            decay_steps = int(total_iterations * params.lr_decay_ratio * optimizer_steps_per_iter)
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=params.lr_start_factor,
                end_factor=params.lr_end_factor,
                total_iters=decay_steps,
            )

    # ====================================================================
    #                          Batch 预处理
    # ====================================================================

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> Dict[str, RolloutBatch]:
        """
        Per-agent 向量化 GAE 预处理。

        每个 agent 独立计算: values → trunc bootstrap → _gae_vectorized
        结果保留 Dict[str, RolloutBatch] 供 update 使用。
        RNN 时记录 T/N 供 chunk_split 使用。
        """
        N = self.num_envs
        all_ret_for_norm = []

        for agent, batch in batch_dict.items():
            final_obs = batch.final_obs          # per-agent final obs，供 truncation bootstrap
            batch = batch.to_tensor(self.device)
            total_size = batch.obs.shape[0]
            T = total_size // N
            self._last_T = T
            self._last_N = N

            with torch.no_grad():
                # IPPO: critic 只用 per-agent 局部 obs，不使用全局 state
                critic_input = batch.obs
                done_2d = batch.done.view(T, N)

                if self.is_critic_recurrent:
                    values_norm, critic_rnn_h = self._critic_rnn_values(
                        critic_input, done_2d, T, N, agent_key=agent,
                    )
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
                    truncated_2d, final_obs, critic_input, T, N,
                )

                active_mask_2d = None
                if self.use_active_mask and batch.active_mask is not None:
                    active_mask_2d = batch.active_mask.view(T, N)

                adv, ret = self._gae_vectorized(
                    rew_2d, values, done_2d, truncated_2d, trunc_bootstrap, active_mask_2d,
                )
                values_flat = values_norm.view(-1)

            batch.adv = adv
            batch.ret = ret
            batch.value = values_flat

            if self.use_value_norm:
                if active_mask_2d is not None:
                    active_flat = batch.active_mask > 0.5
                    active_ret = ret[active_flat]
                    if active_ret.numel() > 0:
                        all_ret_for_norm.append(active_ret)
                else:
                    all_ret_for_norm.append(ret)

            batch_dict[agent] = batch

        if self.use_value_norm and self.ret_rms is not None and all_ret_for_norm:
            self.ret_rms.update(torch.cat(all_ret_for_norm, dim=0))

        return batch_dict

    # ====================================================================
    #                          PPO Update
    # ====================================================================

    def update(
        self,
        batch_dict: Dict[str, RolloutBatch],
        minibatch_size: int = -1,
        update_epochs: int = 1,
    ) -> TrainingStats:
        """
        同步 minibatch PPO 更新：所有 agent 的 batch 同步切分，
        per-minibatch 累加各 agent 的 loss 后做一次 optimizer step。
        RNN 时使用 chunk_split + evaluate_actions_sequence。
        """
        agents = list(batch_dict.keys())
        n_agents = len(agents)
        batch_dict = {k: v.to_tensor(self.device) for k, v in batch_dict.items()}

        all_pg_loss, all_v_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl = []
        all_grad_norm = []

        _any_rnn = self.is_any_recurrent

        for epoch in range(update_epochs):
            epoch_approx_kl = []

            if _any_rnn:
                agent_mbs = {
                    agent: list(batch_dict[agent].chunk_split(
                        chunk_len=self.data_chunk_length,
                        T=self._last_T, N=self._last_N,
                        num_agents=1, minibatch_size=minibatch_size,
                    ))
                    for agent in agents
                }
            else:
                agent_mbs = {
                    agent: list(batch_dict[agent].split(
                        size=minibatch_size, shuffle=True, merge_last=True,
                    ))
                    for agent in agents
                }
            n_minibatches = len(agent_mbs[agents[0]])

            for mb_idx in range(n_minibatches):
                total_pg_loss = torch.tensor(0.0, device=self.device)
                total_v_loss = torch.tensor(0.0, device=self.device)
                total_entropy = torch.tensor(0.0, device=self.device)
                mb_clipfrac = 0.0
                mb_approx_kl_val = 0.0

                for agent in agents:
                    mb = agent_mbs[agent][mb_idx]
                    agent_policy = self.policy.get_policy(agent)

                    # ---- Active mask ----
                    if self.use_active_mask and mb.active_mask is not None:
                        am = mb.active_mask.float()
                        if _any_rnn:
                            am = am.reshape(-1)
                        am_sum = am.sum().clamp(min=1.0)
                    else:
                        am = None

                    # ---- Advantage normalization ----
                    mb_adv = mb.adv
                    if _any_rnn:
                        mb_adv = mb_adv.reshape(-1)
                    if self.normalize_advantage:
                        if am is not None:
                            active_adv = mb_adv[am > 0.5]
                            if active_adv.numel() > 1:
                                mb_adv = (mb_adv - active_adv.mean()) / (active_adv.std() + 1e-8)
                        elif mb_adv.numel() > 1:
                            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # ---- Policy loss ----
                    if self.is_recurrent:
                        new_log_prob, entropy = agent_policy.evaluate_actions_sequence(
                            mb.obs, mb.act, mb.rnn_hidden,
                            action_mask=mb.action_mask,
                        )
                        new_log_prob = new_log_prob.reshape(-1)
                        entropy = entropy.reshape(-1)
                    else:
                        obs_flat = mb.obs.reshape(-1, mb.obs.shape[-1]) if _any_rnn else mb.obs
                        act_flat = mb.act.reshape(-1) if _any_rnn else mb.act
                        am_eval = mb.action_mask.reshape(-1, mb.action_mask.shape[-1]) if (_any_rnn and mb.action_mask is not None) else mb.action_mask
                        new_log_prob, entropy = agent_policy.evaluate_actions(
                            obs_flat, act_flat, action_mask=am_eval,
                        )
                    mb_log_prob = mb.log_prob.reshape(-1) if _any_rnn else mb.log_prob

                    logratio = new_log_prob - mb_log_prob
                    ratio = logratio.exp()

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)

                    if am is not None:
                        pg_loss = (pg_loss_per_sample * am).sum() / am_sum
                        ent_loss = (entropy * am).sum() / am_sum
                    else:
                        pg_loss = pg_loss_per_sample.mean()
                        ent_loss = entropy.mean()

                    # ---- Value loss ----
                    # IPPO: critic 只用 per-agent 局部 obs
                    critic_input = mb.obs
                    if self.is_critic_recurrent:
                        new_value_seq, _ = self.critic.forward_sequence(
                            critic_input, mb.critic_rnn_hidden,
                        )
                        new_value = new_value_seq.squeeze(-1).reshape(-1)
                    else:
                        if _any_rnn:
                            critic_input = critic_input.reshape(-1, critic_input.shape[-1])
                        new_value = self.critic(critic_input).squeeze(-1)

                    mb_ret = mb.ret.reshape(-1) if _any_rnn else mb.ret
                    mb_value = mb.value.reshape(-1) if _any_rnn and mb.value is not None else mb.value

                    if self.use_value_norm and self.ret_rms is not None:
                        target = (mb_ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                    else:
                        target = mb_ret

                    if self.clip_vloss and mb_value is not None:
                        v_unclipped = (new_value - target) ** 2
                        v_clipped = mb_value + torch.clamp(
                            new_value - mb_value, -self.clip_range, self.clip_range,
                        )
                        v_loss_clipped = (v_clipped - target) ** 2
                        v_loss_per_sample = 0.5 * torch.max(v_unclipped, v_loss_clipped)
                    else:
                        v_loss_per_sample = 0.5 * (new_value - target) ** 2

                    if am is not None:
                        v_loss = (v_loss_per_sample * am).sum() / am_sum
                    else:
                        v_loss = v_loss_per_sample.mean()

                    total_pg_loss = total_pg_loss + pg_loss
                    total_v_loss = total_v_loss + v_loss
                    total_entropy = total_entropy + ent_loss

                    with torch.no_grad():
                        mb_clipfrac += ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                        mb_approx_kl_val += ((ratio - 1) - logratio).mean().item()

                avg_pg = total_pg_loss / n_agents
                avg_v = total_v_loss / n_agents
                avg_ent = total_entropy / n_agents
                loss = avg_pg + self.vf_coef * avg_v - self.ent_coef * avg_ent

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                all_pg_loss.append(avg_pg.item())
                all_v_loss.append(avg_v.item())
                all_entropy.append(avg_ent.item())
                all_clipfrac.append(mb_clipfrac / n_agents)
                all_grad_norm.append(grad_norm.item())
                epoch_approx_kl.append(mb_approx_kl_val / n_agents)

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
                "grad_norm": np.mean(all_grad_norm),
            },
        )

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)
