"""MAA2C: Multi-Agent A2C with Parameter Sharing and Centralized Critic."""

from typing import Dict, Optional
import torch
import torch.nn as nn

from algorithms.rl.a2c import A2CBase
from algorithms.marl.ctde_mixin import CentralizedCriticMixin
from configs.algo_configs import MAA2CParams
from policies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class MAA2CAlgo(A2CBase, CentralizedCriticMixin):
    """MAA2C: 参数共享 + Centralized Critic + 双优化器。

    继承 A2CBase 的默认 hook（vanilla PG + MSE + 无 KL stopping），
    通过 CentralizedCriticMixin 获得 CTDE prepare_batch。

    override:
    - prepare_batch → _prepare_batch_ctde (CTDE)
    - _eval_policy → evaluate_actions_flat (参数共享)
    - _do_optimizer_step → 双优化器
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        params: MAA2CParams,
        num_envs: int,
        total_iterations: Optional[int] = None,
        optimizer_steps_per_iter: Optional[int] = None,
        value_norm_config: Optional[Dict] = None,
    ):
        super().__init__(policy, critic, params, num_envs, value_norm_config=value_norm_config)

        self.actor_optimizer = torch.optim.Adam(policy.parameters(), lr=params.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=params.critic_lr)

        self.actor_scheduler = None
        self.critic_scheduler = None

        if params.use_lr_scheduler and total_iterations is not None and optimizer_steps_per_iter is not None:
            actor_decay = int(total_iterations * params.actor_lr_decay_ratio * optimizer_steps_per_iter)
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_optimizer,
                start_factor=params.actor_lr_start_factor,
                end_factor=params.actor_lr_end_factor,
                total_iters=actor_decay,
            )
            critic_decay = int(total_iterations * params.critic_lr_decay_ratio * optimizer_steps_per_iter)
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_optimizer,
                start_factor=params.critic_lr_start_factor,
                end_factor=params.critic_lr_end_factor,
                total_iters=critic_decay,
            )

    # ====================================================================
    #                     CTDE prepare_batch
    # ====================================================================

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        return self._prepare_batch_ctde(batch_dict)

    # ====================================================================
    #                     Hook Overrides
    # ====================================================================

    def _eval_policy(self, mb, _any_rnn):
        """参数共享 policy，使用 _flat 变体。"""
        if self.is_recurrent:
            new_log_prob, entropy = self.policy.evaluate_actions_sequence_flat(
                mb.obs, mb.act, mb.rnn_hidden,
                action_mask=mb.action_mask,
            )
            new_log_prob = new_log_prob.reshape(-1)
            entropy = entropy.reshape(-1)
        else:
            obs = mb.obs.reshape(-1, mb.obs.shape[-1]) if _any_rnn else mb.obs
            act = mb.act.reshape(-1) if _any_rnn else mb.act
            am = (
                mb.action_mask.reshape(-1, mb.action_mask.shape[-1])
                if (_any_rnn and mb.action_mask is not None)
                else mb.action_mask
            )
            new_log_prob, entropy = self.policy.evaluate_actions_flat(
                obs, act, action_mask=am,
            )
        return new_log_prob, entropy

    def _do_optimizer_step(self, pg_loss, ent_loss, v_loss, update_actor=True):
        """双优化器 step。"""
        actor_loss = pg_loss - self.ent_coef * ent_loss
        grad_info = {}

        self.actor_optimizer.zero_grad()
        if update_actor:
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm,
            )
            self.actor_optimizer.step()
            grad_info["actor_grad_norm"] = actor_grad.item()

        critic_loss = self.vf_coef * v_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm,
        )
        self.critic_optimizer.step()
        grad_info["critic_grad_norm"] = critic_grad.item()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        return grad_info

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)
