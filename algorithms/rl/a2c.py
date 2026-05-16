"""A2C 家族算法。

A2CBase(ActorCriticOnPolicyAlgo)
    Actor-Critic On-Policy update 骨架（Template Method 模式）。
    通过 hook methods 支持 A2C / PPO 及其多智能体变体：
    - _compute_policy_loss: vanilla PG (A2C) / clipped surrogate (PPO)
    - _compute_value_loss:  MSE (A2C) / clipped value loss (PPO)
    - _do_optimizer_step:   单优化器 / 双优化器
    - _on_epoch_end:        无 early stopping (A2C) / KL early stopping (PPO)

A2CAlgo(A2CBase)
    单智能体 A2C，单优化器。继承 A2CBase 默认 hook + prepare_batch。
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from algorithms.algorithm_base import ActorCriticOnPolicyAlgo, TrainingStats
from configs.algo_configs import OnPolicyParams, A2CParams
from policies.rl.rl_base import ActorPolicy
from data.batch import RolloutBatch


# =============================================================================
#                          A2CBase — AC On-Policy 中间基类
# =============================================================================

class A2CBase(ActorCriticOnPolicyAlgo):
    """Actor-Critic On-Policy update 骨架（Template Method）。

    在 ActorCriticOnPolicyAlgo（GAE, prepare_batch）之上提供完整 update() 实现，
    通过 hook methods 支持 A2C / PPO 差异化，同时复用全部公共逻辑：
    - minibatch splitting (RNN chunk_split / MLP split)
    - active_mask 支持
    - advantage normalization
    - value normalization
    - 统计量收集
    """

    # ====================================================================
    #                     Hook Methods（子类 override）
    # ====================================================================

    def _compute_policy_loss(
        self,
        new_log_prob: torch.Tensor,
        entropy: torch.Tensor,
        mb_adv: torch.Tensor,
        mb_log_prob: torch.Tensor,
        am: Optional[torch.Tensor],
        am_sum: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """计算 policy loss + entropy loss。

        Returns:
            (pg_loss, ent_loss, extra_dict)
            extra_dict 可包含子类特有统计量（如 PPO 的 clipfrac/approx_kl）
        """
        pg_loss_per_sample = -(mb_adv * new_log_prob)
        if am is not None:
            pg_loss = (pg_loss_per_sample * am).sum() / am_sum
            ent_loss = (entropy * am).sum() / am_sum
        else:
            pg_loss = pg_loss_per_sample.mean()
            ent_loss = entropy.mean()
        return pg_loss, ent_loss, {}

    def _compute_value_loss(
        self,
        new_value: torch.Tensor,
        target: torch.Tensor,
        mb_value: Optional[torch.Tensor],
        am: Optional[torch.Tensor],
        am_sum: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算 value loss（默认 MSE）。"""
        v_loss_per_sample = 0.5 * (new_value - target) ** 2
        if am is not None:
            return (v_loss_per_sample * am).sum() / am_sum
        return v_loss_per_sample.mean()

    def _do_optimizer_step(
        self,
        pg_loss: torch.Tensor,
        ent_loss: torch.Tensor,
        v_loss: torch.Tensor,
        update_actor: bool = True,
    ) -> dict:
        """优化器 step（默认：单优化器）。返回 grad_norm dict。"""
        loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {"grad_norm": grad_norm.detach()}

    def _on_epoch_end(self, epoch_extra_list: List[dict]) -> bool:
        """每个 epoch 结束后调用。返回 True 触发 early stopping。"""
        return False

    # ====================================================================
    #                     Policy / Critic 评估辅助
    # ====================================================================

    def _eval_policy(
        self,
        mb: RolloutBatch,
        _any_rnn: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """统一 RNN/MLP policy forward，返回 (new_log_prob, entropy)。

        单智能体使用 evaluate_actions / evaluate_actions_sequence。
        MAA2CAlgo / MAPPOAlgo override 为 _flat 变体。
        """
        if self.is_recurrent:
            new_log_prob, entropy = self.policy.evaluate_actions_sequence(
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
            new_log_prob, entropy = self.policy.evaluate_actions(
                obs, act, action_mask=am,
            )
        return new_log_prob, entropy

    def _eval_critic(
        self,
        mb: RolloutBatch,
        _any_rnn: bool,
    ) -> torch.Tensor:
        """统一 RNN/MLP critic forward，返回 new_value (flat)。"""
        critic_input = mb.global_state if mb.global_state is not None else mb.obs
        if self.is_critic_recurrent:
            new_value_seq, _ = self.critic.forward_sequence(
                critic_input, mb.critic_rnn_hidden,
            )
            return new_value_seq.squeeze(-1).reshape(-1)
        if _any_rnn:
            critic_input = critic_input.reshape(-1, critic_input.shape[-1])
        return self.critic(critic_input).squeeze(-1)

    # ====================================================================
    #                     Template Update（核心骨架）
    # ====================================================================

    def update(
        self,
        batch: RolloutBatch,
        minibatch_size: int = -1,
        update_epochs: int = 1,
        T: int = 0,
        N: int = 0,
        num_agents: int = 1,
        update_actor: bool = True,
    ) -> TrainingStats:
        """Actor-Critic on-policy update 骨架。

        A2C/PPO/MAPPO/MAA2C 共用此循环，差异通过 hook methods 实现。
        """
        batch = batch.to_tensor(self.device)

        all_pg_loss: List[float] = []
        all_v_loss: List[float] = []
        all_entropy: List[float] = []
        all_extra: List[dict] = []
        all_grad: List[dict] = []

        _any_rnn = self.is_any_recurrent

        if T == 0 and hasattr(self, "_last_T"):
            T = self._last_T
        if N == 0 and hasattr(self, "_last_N"):
            N = self._last_N
        if num_agents == 1 and hasattr(self, "_last_num_agents"):
            num_agents = self._last_num_agents

        for _epoch in range(update_epochs):
            epoch_extra: List[dict] = []

            if _any_rnn:
                mb_iter = batch.chunk_split(
                    chunk_len=self.data_chunk_length,
                    T=T, N=N, num_agents=num_agents,
                    minibatch_size=minibatch_size,
                )
            else:
                mb_iter = batch.split(
                    size=minibatch_size, shuffle=True, merge_last=True,
                )

            for mb in mb_iter:
                # ---- active mask ----
                if mb.active_mask is not None:
                    am = mb.active_mask.float()
                    if _any_rnn:
                        am = am.reshape(-1)
                    am_sum = am.sum().clamp(min=1.0)
                else:
                    am = None
                    am_sum = None

                # ---- advantage normalization ----
                mb_adv = mb.adv.reshape(-1) if _any_rnn else mb.adv
                if self.normalize_advantage:
                    if am is not None:
                        active_adv = mb_adv[am > 0.5]
                        if active_adv.numel() > 1:
                            mb_adv = (mb_adv - active_adv.mean()) / (active_adv.std() + 1e-8)
                    elif mb_adv.numel() > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ---- policy forward ----
                new_log_prob, entropy = self._eval_policy(mb, _any_rnn)
                mb_log_prob = mb.log_prob.reshape(-1) if _any_rnn else mb.log_prob

                # ---- policy loss (hook) ----
                pg_loss, ent_loss, pg_extra = self._compute_policy_loss(
                    new_log_prob, entropy, mb_adv, mb_log_prob, am, am_sum,
                )

                # ---- critic forward ----
                new_value = self._eval_critic(mb, _any_rnn)

                # ---- value target ----
                mb_ret = mb.ret.reshape(-1) if _any_rnn else mb.ret
                mb_value = (
                    mb.value.reshape(-1)
                    if (_any_rnn and mb.value is not None) else mb.value
                )
                if self.use_value_norm and self.ret_rms is not None:
                    target = (mb_ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                else:
                    target = mb_ret

                # ---- value loss (hook) ----
                v_loss = self._compute_value_loss(
                    new_value, target, mb_value, am, am_sum,
                )

                # ---- optimizer step (hook) ----
                grad_info = self._do_optimizer_step(
                    pg_loss, ent_loss, v_loss, update_actor,
                )

                # ---- collect stats ----
                all_pg_loss.append(pg_loss.detach())
                all_v_loss.append(v_loss.detach())
                all_entropy.append(ent_loss.detach())
                epoch_extra.append(pg_extra)
                all_grad.append(grad_info)

            # ---- epoch end (hook) ----
            all_extra.extend(epoch_extra)
            if self._on_epoch_end(epoch_extra):
                break

        # ---- aggregate stats ----
        def _mean_stat(vals) -> float:
            if vals and isinstance(vals[0], torch.Tensor):
                return float(torch.stack([v.detach() for v in vals]).mean().item())
            return float(np.mean(vals))

        extra: Dict[str, float] = {}
        if all_extra:
            all_keys = {k for e in all_extra for k in e}
            for key in all_keys:
                vals = [e[key] for e in all_extra if key in e]
                if vals:
                    extra[key] = _mean_stat(vals)
        if all_grad:
            all_keys = {k for g in all_grad for k in g}
            for key in all_keys:
                vals = [g[key] for g in all_grad if key in g]
                if vals:
                    extra[key] = _mean_stat(vals)

        return TrainingStats(
            loss=_mean_stat(all_pg_loss) + self.vf_coef * _mean_stat(all_v_loss),
            policy_loss=_mean_stat(all_pg_loss),
            value_loss=_mean_stat(all_v_loss),
            entropy=_mean_stat(all_entropy),
            extra=extra,
        )


# =============================================================================
#                          A2CAlgo — 单智能体 A2C
# =============================================================================

class A2CAlgo(A2CBase):
    """单智能体 A2C，单优化器。

    继承 A2CBase 默认 hook（vanilla PG + MSE + 无 KL stopping）
    和 ActorCriticOnPolicyAlgo.prepare_batch。
    """

    def __init__(
        self,
        policy: ActorPolicy,
        critic: nn.Module,
        params: A2CParams,
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
