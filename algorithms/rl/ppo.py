"""PPO 家族算法。

PPOBase(ActorCriticOnPolicyAlgo)
    PPO 家族共享的 update 逻辑: clipped surrogate, value clipping, KL early stopping。
    不创建 optimizer，留给最终子类。

PPOAlgo(PPOBase)
    单智能体 / 集中式多智能体 PPO，创建 optimizer / scheduler / value_norm。
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from algorithms.algorithm_base import ActorCriticOnPolicyAlgo, TrainingStats
from configs.algo_configs import PPOParams, IPPOParams
from policies.rl.rl_base import ActorPolicy
from data.batch import RolloutBatch


# =============================================================================
#                          PPOBase — PPO 家族中间基类
# =============================================================================

class PPOBase(ActorCriticOnPolicyAlgo):
    """
    PPO 家族共享基类。

    在 ActorCriticOnPolicyAlgo（GAE、prepare_batch）之上增加 PPO 特有逻辑：
    - clipped surrogate policy loss
    - value loss clipping (PPO2 style)
    - KL early stopping
    - multi-epoch minibatch update

    提供完整的单优化器 update()，PPOAlgo 直接继承。
    IPPOAlgo / MAPPOAlgo 可 override update() 实现自身需求。
    """

    def __init__(self, policy, critic: nn.Module, params: PPOParams, num_envs: int = 1):
        super().__init__(policy, critic, params, num_envs)

        # PPO 特有超参
        self.clip_range = params.clip_range
        self.clip_vloss = params.clip_vloss
        self.target_kl = params.target_kl

    # ====================================================================
    #                     PPO Update（单优化器版本）
    # ====================================================================

    def update(
        self,
        batch: RolloutBatch,
        minibatch_size: int = -1,
        update_epochs: int = 1,
    ) -> TrainingStats:
        """
        单优化器 PPO update。

        包含: minibatch split, advantage normalization, clipped surrogate,
        value loss clipping, KL early stopping, 详细统计量。
        """
        batch = batch.to_tensor(self.device)
        all_pg_loss, all_v_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl = []
        all_grad_norm = []

        for epoch in range(update_epochs):
            epoch_approx_kl = []

            for mb in batch.split(size=minibatch_size, shuffle=True, merge_last=True):
                # ---- Advantage normalization ----
                mb_adv = mb.adv
                if self.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ---- Policy evaluation ----
                new_log_prob, entropy = self.policy.evaluate_actions(
                    mb.obs, mb.act, action_mask=mb.action_mask,
                )
                logratio = new_log_prob - mb.log_prob
                ratio = logratio.exp()

                # ---- Clipped surrogate loss ----
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # ---- Value loss (with optional clipping) ----
                critic_input = mb.global_state if mb.global_state is not None else mb.obs
                new_value = self.critic(critic_input).squeeze(-1)

                if self.use_value_norm and self.ret_rms is not None:
                    target = (mb.ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                else:
                    target = mb.ret

                if self.clip_vloss and mb.value is not None:
                    v_loss_unclipped = (new_value - target) ** 2
                    v_clipped = mb.value + torch.clamp(
                        new_value - mb.value, -self.clip_range, self.clip_range,
                    )
                    v_loss_clipped = (v_clipped - target) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - target) ** 2).mean()

                # ---- Total loss + optimizer step ----
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # ---- Stats ----
                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(v_loss.item())
                all_entropy.append(entropy_loss.item())
                all_grad_norm.append(grad_norm.item())

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
                "grad_norm": np.mean(all_grad_norm),
            },
        )


# =============================================================================
#                          PPOAlgo — 单优化器 PPO
# =============================================================================

class PPOAlgo(PPOBase):
    """
    单优化器 PPO。

    继承 PPOBase 的 update（clipped surrogate + KL early stopping）
    和 ActorCriticOnPolicyAlgo 的 prepare_batch（向量化 GAE + truncation bootstrap）。
    """

    def __init__(
        self,
        policy: ActorPolicy,
        critic: nn.Module,
        params: IPPOParams,
        num_envs: int = 1,
        total_iterations: Optional[int] = None,
        optimizer_steps_per_iter: Optional[int] = None,
        value_norm_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(policy, critic, params, num_envs)

        # 单优化器
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

        # Value Normalization
        if self.use_value_norm:
            from utils.train_utils import RunningMeanStd
            self.ret_rms = RunningMeanStd(shape=(1,)).to(self.device)
            if value_norm_config is not None:
                self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
                self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
                self.ret_rms.count.fill_(1.0)
