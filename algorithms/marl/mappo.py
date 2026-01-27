"""MAPPO: Multi-Agent PPO with Parameter Sharing and Centralized Critic."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import ActorCriticOnPolicyAlgo, TrainingStats
from polocies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class MAPPOAlgo(ActorCriticOnPolicyAlgo):
    """
    MAPPO算法: 共享策略 + Centralized Critic
    
    继承 ActorCriticOnPolicyAlgo，复用 GAE 计算和 advantage 归一化。
    输入 Dict[str, RolloutBatch]，内部 merge 后一次前向计算。
    """
    
    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
    ):
        # 跳过父类 __init__，直接初始化（因为签名不同）
        nn.Module.__init__(self)
        
        self.policy = policy
        self.critic = critic.to(policy.device)
        self.device = policy.device
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()), lr=lr
        )
    
    def update(self, batch_dict: Dict[str, RolloutBatch]) -> TrainingStats:
        """处理 Dict 输入，merge 后计算 loss 并更新。"""
        # 转换为 tensor
        batch_dict = {k: v.to_tensor(self.device) for k, v in batch_dict.items()}
        
        # Merge 并计算 loss
        merged = self._merge_batches(batch_dict)
        loss, stats = self.compute_loss(merged)
        
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        stats.loss = loss.item()
        return stats
    
    def _merge_batches(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """合并所有 agent 的数据为单个 batch。"""
        agents = list(batch_dict.keys())
        
        # 直接 cat，简单高效
        obs = torch.cat([batch_dict[a].obs for a in agents], dim=0)
        act = torch.cat([batch_dict[a].act for a in agents], dim=0)
        log_prob = torch.cat([batch_dict[a].log_prob for a in agents], dim=0)
        adv = torch.cat([batch_dict[a].adv for a in agents], dim=0)
        ret = torch.cat([batch_dict[a].ret for a in agents], dim=0)
        global_state = torch.cat([batch_dict[a].global_state for a in agents], dim=0)
        
        action_mask = None
        if batch_dict[agents[0]].action_mask is not None:
            action_mask = torch.cat([batch_dict[a].action_mask for a in agents], dim=0)
        
        return RolloutBatch(
            obs=obs, act=act, log_prob=log_prob,
            adv=adv, ret=ret, global_state=global_state,
            action_mask=action_mask
        )
    
    def compute_loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, TrainingStats]:
        """计算 PPO loss。"""
        # Normalize advantage
        batch.adv = self._normalize_advantage(batch.adv)
        
        # Policy loss (PPO clipped)
        new_log_prob, entropy = self.policy.evaluate_actions_flat(
            batch.obs, batch.act, action_mask=batch.action_mask
        )
        ratio = torch.exp(new_log_prob - batch.log_prob)
        surr1 = ratio * batch.adv
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch.adv
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = self.critic(batch.global_state).squeeze(-1)
        value_loss = F.mse_loss(value_pred, batch.ret)
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
        
        stats = TrainingStats(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.mean().item()
        )
        return loss, stats
