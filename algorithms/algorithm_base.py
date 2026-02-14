from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass, field, fields
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.rl.rl_base import RLBasePolicy, ActorPolicy
from data.batch import BaseBatch, RolloutBatch, TransitionBatch


@dataclass
class TrainingStats:
    """Training statistics returned by Algorithm.update()."""
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
    Base class for RL algorithms.
    
    Responsibilities:
    - Loss computation
    - Network update (optimizer step)
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        policy: RLBasePolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.policy = policy
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.device = policy.device
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.lr_scheduler = None
    
    @abstractmethod
    def compute_loss(self, batch: BaseBatch) -> tuple[torch.Tensor, TrainingStats]:
        """Compute loss from batch. Returns (loss_tensor, training_stats)."""
        pass
    
    def update(self, batch: BaseBatch) -> TrainingStats:
        """Perform one gradient update step."""
        # Convert to tensor
        batch = batch.to_tensor(self.device)
        
        # Compute loss
        loss, stats = self.compute_loss(batch)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # LR scheduling
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        stats.loss = loss.item()
        return stats
    
    def set_training_mode(self, mode: bool):
        """Set training mode for algorithm and policy."""
        self.train(mode)
        self.policy.set_training_mode(mode)


# ============================================================================
#                          On-Policy Algorithm
# ============================================================================

class ActorCriticOnPolicyAlgo(BaseAlgorithm):
    """
    Base class for on-policy algorithms (A2C, TRPO, PPO, etc.).
    
    Features:
    - GAE computation
    - Advantage normalization
    - Internal minibatch splitting and multi-epoch updates
    """
    
    def __init__(
        self,
        policy: ActorPolicy,
        critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
    ):
        super().__init__(policy, lr, gamma, max_grad_norm)
        self.critic = critic.to(self.device)
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantage = normalize_advantage
        
        # Add critic params to optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: (T,) rewards
            values: (T,) value estimates
            dones: (T,) done flags
            next_value: scalar, value of last next_obs
        
        Returns:
            advantages: (T,)
            returns: (T,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def prepare_batch(self, batch: RolloutBatch) -> RolloutBatch:
        """
        用当前 critic 计算 value, adv, ret（采样后、更新前调用）
        
        确保使用采样时的 critic 参数，符合 on-policy 原理。
        """
        batch = batch.to_tensor(self.device)
        
        with torch.no_grad():
            # 计算所有 value
            critic_input = batch.obs
            if batch.global_state is not None:
                critic_input = batch.global_state
            
            values = self.critic(critic_input).squeeze(-1)
            
            # 计算最后一个 obs 的 next_value
            # 如果最后一步 done=True，next_value 应该是 0
            last_done = batch.done[-1]
            if last_done > 0.5:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_value = self.critic(critic_input[-1:]).squeeze(-1)
        
        # 计算 GAE
        batch.adv, batch.ret = self.compute_gae(batch.rew, values, batch.done, next_value)
        batch.value = values
        
        return batch
    
    def _normalize_advantage(self, adv: torch.Tensor) -> torch.Tensor:
        """Normalize advantages to zero mean and unit std."""
        if self.normalize_advantage and len(adv) > 1:
            return (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv
    
    def compute_policy_loss(
        self,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy loss and entropy. Override in subclass for specific loss.
        
        Returns:
            policy_loss: scalar tensor
            entropy: scalar tensor (mean entropy)
        """
        raise NotImplementedError
    
    def compute_value_loss(self, batch: RolloutBatch) -> torch.Tensor:
        """Compute value loss. Subclass can override for centralized critic."""
        # Default: use local obs
        critic_input = batch.obs
        if batch.global_state is not None:
            critic_input = batch.global_state  # For MAPPO
        
        value_pred = self.critic(critic_input).squeeze(-1)
        return F.mse_loss(value_pred, batch.ret)
    
    def compute_loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, TrainingStats]:
        """Compute total loss for a single minibatch."""
        # Normalize advantage at minibatch level
        batch.adv = self._normalize_advantage(batch.adv)
        
        # Policy loss
        policy_loss, entropy = self.compute_policy_loss(batch)
        
        # Value loss
        value_loss = self.compute_value_loss(batch)
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        
        stats = TrainingStats()
        stats.policy_loss = policy_loss.item()
        stats.value_loss = value_loss.item()
        stats.entropy = entropy.item()
        
        return loss, stats
    
    def update(
        self,
        batch: RolloutBatch,
        minibatch_size: int,
        update_epochs: int,
    ) -> TrainingStats:
        """
        Multi-epoch minibatch update.
        
        Args:
            batch: 完整 rollout 数据
            minibatch_size: 每个 minibatch 的样本数，-1 表示不切分
            update_epochs: 对同一批数据重复更新的轮数
        """
        import numpy as np
        
        batch = batch.to_tensor(self.device)
        all_stats = []
        
        for _ in range(update_epochs):
            for minibatch in batch.split(size=minibatch_size, shuffle=True, merge_last=True):
                # Compute loss and update
                loss, stats = self.compute_loss(minibatch)
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm
                    )
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                stats.loss = loss.item()
                all_stats.append(stats)
        
        # Aggregate stats
        return TrainingStats(
            loss=np.mean([s.loss for s in all_stats]),
            policy_loss=np.mean([s.policy_loss for s in all_stats]),
            value_loss=np.mean([s.value_loss for s in all_stats]),
            entropy=np.mean([s.entropy for s in all_stats]),
        )


# ============================================================================
#                          Off-Policy Algorithm (Not completed yet)
# ============================================================================

class QLearningOffPolicyAlgo(BaseAlgorithm):
    """
    Base class for off-policy algorithms (DQN, SAC, DDPG, etc.).
    
    Features:
    - Target network management
    - N-step returns
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
        super().__init__(policy, lr, gamma, max_grad_norm)
        self.tau = tau
        self.target_update_freq = target_update_freq
        self._update_count = 0
        
        # Target network (created by subclass)
        self.target_policy: Optional[RLBasePolicy] = None
    
    def _create_target_network(self):
        """Create target network as a copy of policy."""
        import copy
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.set_training_mode(False)
        for param in self.target_policy.parameters():
            param.requires_grad = False
    
    def _soft_update_target(self):
        """Soft update target network (Polyak averaging)."""
        if self.target_policy is None:
            return
        for target_param, param in zip(
            self.target_policy.parameters(),
            self.policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def _hard_update_target(self):
        """Hard update target network."""
        if self.target_policy is not None:
            self.target_policy.load_state_dict(self.policy.state_dict())
    
    def update(self, batch: TransitionBatch) -> TrainingStats:
        """Update with target network management."""
        stats = super().update(batch)
        
        # Update target network
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            if self.tau < 1.0:
                self._soft_update_target()
            else:
                self._hard_update_target()
        
        return stats