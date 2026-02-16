"""IPPO: Independent PPO — 每个 agent 独立策略网络，共享 critic。"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import ActorCriticOnPolicyAlgo, TrainingStats
from configs.algo_configs import IPPOParams
from policies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class IPPOAlgo(ActorCriticOnPolicyAlgo):
    """
    IPPO 算法：独立策略，共享 critic，单优化器。

    继承 ActorCriticOnPolicyAlgo，复用 GAE 计算和 advantage 归一化。
    循环处理每个 agent，一次 backward 更新所有网络。
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        params: IPPOParams,
        # 运行时上下文（与 MAPPO 保持一致的签名）
        num_envs: int = 1,
        **kwargs,
    ):
        super().__init__(policy, critic, params)
        self.num_envs = num_envs

        # 单优化器：所有独立 policy + critic
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()),
            lr=params.lr,
        )

    # ---- Batch 预处理（per-agent GAE）----

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> Dict[str, RolloutBatch]:
        """为每个 agent 分别计算 GAE。"""
        for agent, batch in batch_dict.items():
            batch = batch.to_tensor(self.device)

            with torch.no_grad():
                critic_input = batch.global_state if batch.global_state is not None else batch.obs
                values = self.critic(critic_input).squeeze(-1)

                last_done = batch.done[-1]
                if last_done > 0.5:
                    next_value = torch.tensor(0.0, device=self.device)
                else:
                    next_value = self.critic(critic_input[-1:]).squeeze(-1)

            batch.adv, batch.ret = self.compute_gae(batch.rew, values, batch.done, next_value)
            batch.value = values
            batch_dict[agent] = batch

        return batch_dict

    # ---- Loss 计算（循环 agent 累加）----

    def compute_loss(self, batch_dict: Dict[str, RolloutBatch]) -> tuple[torch.Tensor, TrainingStats]:
        """循环每个 agent 计算 PPO loss 并累加。"""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for agent, batch in batch_dict.items():
            batch.adv = self._normalize_advantage(batch.adv)

            # 获取该 agent 的独立 policy
            agent_policy = self.policy.get_policy(agent)

            # PPO clipped policy loss
            new_log_prob, entropy = agent_policy.evaluate_actions(
                batch.obs, batch.act, action_mask=batch.action_mask,
            )
            ratio = torch.exp(new_log_prob - batch.log_prob)
            surr1 = ratio * batch.adv
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch.adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            critic_input = batch.global_state if batch.global_state is not None else batch.obs
            value_pred = self.critic(critic_input).squeeze(-1)
            value_loss = F.mse_loss(value_pred, batch.ret)

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy.mean()

        # 求均值
        n = len(batch_dict)
        avg_policy_loss = total_policy_loss / n
        avg_value_loss = total_value_loss / n
        avg_entropy = total_entropy / n

        loss = avg_policy_loss + self.vf_coef * avg_value_loss - self.ent_coef * avg_entropy

        stats = TrainingStats(
            policy_loss=avg_policy_loss.item(),
            value_loss=avg_value_loss.item(),
            entropy=avg_entropy.item(),
        )
        return loss, stats

    # ---- Multi-epoch update（兼容 OnPolicyTrainer 接口）----

    def update(
        self,
        batch_dict: Dict[str, RolloutBatch],
        minibatch_size: int = -1,
        update_epochs: int = 1,
    ) -> TrainingStats:
        """
        Multi-epoch 更新，兼容 OnPolicyTrainer.update() 调用签名。

        注：IPPO 处理 Dict[str, RolloutBatch]，minibatch 切分在 per-agent 级别
        意义不大，这里仅做 epoch 循环。若后续需要可在 compute_loss 内部添加。
        """
        batch_dict = {k: v.to_tensor(self.device) for k, v in batch_dict.items()}
        all_stats = []

        for _ in range(update_epochs):
            loss, stats = self.compute_loss(batch_dict)

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
            self.optimizer.step()

            stats.loss = loss.item()
            all_stats.append(stats)

        return TrainingStats(
            loss=np.mean([s.loss for s in all_stats]),
            policy_loss=np.mean([s.policy_loss for s in all_stats]),
            value_loss=np.mean([s.value_loss for s in all_stats]),
            entropy=np.mean([s.entropy for s in all_stats]),
        )
