"""RL 算法基类层次结构。

BaseAlgorithm
├── ActorCriticOnPolicyAlgo  (PPO/MAPPO/IPPO)
└── QLearningOffPolicyAlgo   (DQN/SAC, 待完善)

设计原则：
- 基类只管公共逻辑（loss 计算、GAE、advantage 归一化）
- optimizer / lr_scheduler 由子类创建（不同算法需求不同）
- 超参通过 dataclass Params 传入，避免散列参数
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.algo_configs import PPOParams
from policies.rl.rl_base import RLBasePolicy, ActorPolicy
from data.batch import BaseBatch, RolloutBatch, TransitionBatch


@dataclass
class TrainingStats:
    """Algorithm.update() 返回的训练统计量。"""
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


# ============================================================================
#                             Base Algorithm
# ============================================================================

class BaseAlgorithm(nn.Module, ABC):
    """
    RL 算法基类。

    职责：loss 计算、梯度更新、LR 调度。
    optimizer 和 lr_scheduler 由子类在自身 __init__ 中创建。
    """

    def __init__(self, policy: RLBasePolicy):
        super().__init__()
        self.policy = policy
        self.device = policy.device

        # 由子类创建
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler = None

    @abstractmethod
    def compute_loss(self, batch: BaseBatch) -> tuple[torch.Tensor, TrainingStats]:
        """计算 loss，返回 (loss_tensor, training_stats)。"""
        pass

    def update(self, batch: BaseBatch, **kwargs) -> TrainingStats:
        """单步梯度更新（子类通常会 override）。"""
        batch = batch.to_tensor(self.device)
        loss, stats = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        if getattr(self, "max_grad_norm", 0) > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        stats.loss = loss.item()
        return stats

    def set_training_mode(self, mode: bool):
        """切换 train/eval 模式。"""
        self.train(mode)
        self.policy.set_training_mode(mode)


# ============================================================================
#                          On-Policy Algorithm
# ============================================================================

class ActorCriticOnPolicyAlgo(BaseAlgorithm):
    """
    On-policy Actor-Critic 基类 (A2C / PPO / MAPPO / IPPO)。

    功能：
    - GAE 计算
    - Advantage 归一化
    - Minibatch 切分 + multi-epoch 更新

    构造参数通过 PPOParams (或其子类) 传入。
    optimizer 由子类创建，因为不同算法有不同优化器配置
    （单优化器 vs 双优化器，不同的参数组合）。
    """

    def __init__(self, policy, critic: nn.Module, params: PPOParams):
        super().__init__(policy)
        self.critic = critic.to(self.device)

        # 从 params 解包 PPO 系列共享超参
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.clip_range = params.clip_range
        self.vf_coef = params.vf_coef
        self.ent_coef = params.ent_coef
        self.max_grad_norm = params.max_grad_norm
        self.normalize_advantage = params.normalize_advantage

    # ---- GAE ----

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation.

        Args:
            rewards: (T,)
            values:  (T,) value estimates
            dones:   (T,) done flags
            next_value: scalar, V(s_{T})

        Returns:
            advantages: (T,)
            returns:    (T,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    # ---- Batch 预处理 ----

    def prepare_batch(self, batch: RolloutBatch) -> RolloutBatch:
        """用当前 critic 计算 value / adv / ret（采样后、更新前调用）。"""
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
        return batch

    # ---- Advantage 归一化 ----

    def _normalize_advantage(self, adv: torch.Tensor) -> torch.Tensor:
        if self.normalize_advantage and len(adv) > 1:
            return (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv

    # ---- Loss 计算 ----

    def compute_policy_loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 policy loss 和 entropy，子类必须 override。"""
        raise NotImplementedError

    def compute_value_loss(self, batch: RolloutBatch) -> torch.Tensor:
        """计算 value loss，子类可 override（如 centralized critic）。"""
        critic_input = batch.global_state if batch.global_state is not None else batch.obs
        value_pred = self.critic(critic_input).squeeze(-1)
        return F.mse_loss(value_pred, batch.ret)

    def compute_loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, TrainingStats]:
        """单个 minibatch 的总 loss。"""
        batch.adv = self._normalize_advantage(batch.adv)

        policy_loss, entropy = self.compute_policy_loss(batch)
        value_loss = self.compute_value_loss(batch)
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        stats = TrainingStats(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
        )
        return loss, stats

    # ---- Multi-epoch minibatch update ----

    def update(
        self,
        batch: RolloutBatch,
        minibatch_size: int = -1,
        update_epochs: int = 1,
    ) -> TrainingStats:
        """
        Multi-epoch minibatch 更新。

        Args:
            batch: 完整 rollout 数据
            minibatch_size: 每个 minibatch 的样本数，-1 = 不切分
            update_epochs: 对同一批数据的遍历轮数
        """
        batch = batch.to_tensor(self.device)
        all_stats = []

        for _ in range(update_epochs):
            for minibatch in batch.split(size=minibatch_size, shuffle=True, merge_last=True):
                loss, stats = self.compute_loss(minibatch)

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                stats.loss = loss.item()
                all_stats.append(stats)

        return TrainingStats(
            loss=np.mean([s.loss for s in all_stats]),
            policy_loss=np.mean([s.policy_loss for s in all_stats]),
            value_loss=np.mean([s.value_loss for s in all_stats]),
            entropy=np.mean([s.entropy for s in all_stats]),
        )


# ============================================================================
#                          Off-Policy Algorithm (待完善)
# ============================================================================

class QLearningOffPolicyAlgo(BaseAlgorithm):
    """
    Off-policy 算法基类 (DQN / SAC / DDPG)。

    功能：target network 管理、soft/hard update。
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

        # optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # target network（由子类调用 _create_target_network 创建）
        self.target_policy: Optional[RLBasePolicy] = None

    def _create_target_network(self):
        import copy
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.set_training_mode(False)
        for param in self.target_policy.parameters():
            param.requires_grad = False

    def _soft_update_target(self):
        if self.target_policy is None:
            return
        for target_param, param in zip(
            self.target_policy.parameters(), self.policy.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _hard_update_target(self):
        if self.target_policy is not None:
            self.target_policy.load_state_dict(self.policy.state_dict())

    def update(self, batch: TransitionBatch, **kwargs) -> TrainingStats:
        """梯度更新 + target network 更新。"""
        stats = super().update(batch)
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            if self.tau < 1.0:
                self._soft_update_target()
            else:
                self._hard_update_target()
        return stats
