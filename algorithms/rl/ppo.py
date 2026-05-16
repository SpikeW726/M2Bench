"""PPO 家族算法。

PPOBase(A2CBase)
    PPO 家族共享的 hook overrides: clipped surrogate, value clipping, KL early stopping。
    update 骨架继承自 A2CBase，不创建 optimizer。

PPOAlgo(PPOBase)
    单智能体 PPO，单优化器。
    继承 PPOBase 的 hook + A2CBase 的 update + ActorCriticOnPolicyAlgo 的 prepare_batch。
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from algorithms.rl.a2c import A2CBase
from algorithms.algorithm_base import TrainingStats
from configs.algo_configs import PPOParams, IPPOParams
from policies.rl.rl_base import ActorPolicy
from data.batch import RolloutBatch


# =============================================================================
#                          PPOBase — PPO 家族中间基类
# =============================================================================

class PPOBase(A2CBase):
    """PPO 家族共享基类。

    在 A2CBase（update 骨架 + hook methods）之上 override PPO 特有逻辑：
    - _compute_policy_loss: clipped surrogate
    - _compute_value_loss: optional value clipping
    - _on_epoch_end: KL early stopping
    """

    def __init__(self, policy, critic: nn.Module, params: PPOParams, num_envs: int = 1,
                 value_norm_config: Optional[Dict] = None):
        super().__init__(policy, critic, params, num_envs, value_norm_config=value_norm_config)

        self.clip_range = params.clip_range
        self.clip_vloss = params.clip_vloss
        self.target_kl = params.target_kl

    # ====================================================================
    #                     PPO Hook Overrides
    # ====================================================================

    def _compute_policy_loss(self, new_log_prob, entropy, mb_adv, mb_log_prob, am, am_sum):
        """Clipped surrogate policy loss。"""
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

        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean().detach()
            approx_kl = ((ratio - 1) - logratio).mean().detach()

        return pg_loss, ent_loss, {"clipfrac": clipfrac, "approx_kl": approx_kl}

    def _compute_value_loss(self, new_value, target, mb_value, am, am_sum):
        """PPO2-style clipped value loss。"""
        if self.clip_vloss and mb_value is not None:
            v_loss_unclipped = (new_value - target) ** 2
            v_clipped = mb_value + torch.clamp(
                new_value - mb_value, -self.clip_range, self.clip_range,
            )
            v_loss_clipped = (v_clipped - target) ** 2
            v_loss_per_sample = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
        else:
            v_loss_per_sample = 0.5 * (new_value - target) ** 2

        if am is not None:
            return (v_loss_per_sample * am).sum() / am_sum
        return v_loss_per_sample.mean()

    def _on_epoch_end(self, epoch_extra_list):
        """KL early stopping。"""
        kl_values = [e.get("approx_kl", 0.0) for e in epoch_extra_list if "approx_kl" in e]
        if kl_values and self.target_kl is not None:
            if isinstance(kl_values[0], torch.Tensor):
                return bool(torch.stack(kl_values).mean().item() > self.target_kl)
            return np.mean(kl_values) > self.target_kl
        return False


# =============================================================================
#                          PPOAlgo — 单智能体 PPO
# =============================================================================

class PPOAlgo(PPOBase):
    """单智能体 PPO，单优化器。

    继承 PPOBase 的 hook（clipped surrogate + KL early stopping）
    和 ActorCriticOnPolicyAlgo.prepare_batch（向量化 GAE + truncation bootstrap）。
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
    ):
        super().__init__(policy, critic, params, num_envs, value_norm_config=value_norm_config)

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()),
            lr=params.lr,
        )

        self.lr_scheduler = None
        if params.use_lr_scheduler and total_iterations and optimizer_steps_per_iter:
            decay_steps = int(total_iterations * params.lr_decay_ratio * optimizer_steps_per_iter)
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=params.lr_start_factor,
                end_factor=params.lr_end_factor,
                total_iters=decay_steps,
            )
