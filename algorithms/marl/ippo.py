"""IPPO: Independent PPO with separate policy networks per agent."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import ActorCriticOnPolicyAlgo, TrainingStats
from policies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class IPPOAlgo(ActorCriticOnPolicyAlgo):
    """
    IPPO算法: 独立策略，每个 agent 有自己的网络
    
    继承 ActorCriticOnPolicyAlgo，复用 GAE 计算和 advantage 归一化。
    循环处理每个 agent，一次 backward 更新所有网络。
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
        
        # 优化器：包含所有独立 policy + critic
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()), lr=lr
        )
    
    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> Dict[str, RolloutBatch]:
        """
        为每个 agent 计算 GAE
        
        根据 critic 类型使用 local obs 或 global_state
        """
        for agent, batch in batch_dict.items():
            batch = batch.to_tensor(self.device)
            
            with torch.no_grad():
                # 选择 critic 输入
                critic_input = batch.global_state if batch.global_state is not None else batch.obs
                values = self.critic(critic_input).squeeze(-1)
                
                # 计算 next_value
                last_done = batch.done[-1]
                if last_done > 0.5:
                    next_value = torch.tensor(0.0, device=self.device)
                else:
                    next_value = self.critic(critic_input[-1:]).squeeze(-1)
            
            # 计算 GAE
            batch.adv, batch.ret = self.compute_gae(batch.rew, values, batch.done, next_value)
            batch.value = values
            batch_dict[agent] = batch
        
        return batch_dict
    
    def update(self, batch_dict: Dict[str, RolloutBatch]) -> TrainingStats:
        """处理 Dict 输入，循环计算每个 agent 的 loss。"""
        batch_dict = {k: v.to_tensor(self.device) for k, v in batch_dict.items()}
        
        loss, stats = self.compute_loss(batch_dict)
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        stats.loss = loss.item()
        return stats
    
    def compute_loss(self, batch_dict: Dict[str, RolloutBatch]) -> tuple[torch.Tensor, TrainingStats]:
        """循环处理每个 agent，累加 loss。"""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for agent, batch in batch_dict.items():
            # Normalize advantage
            batch.adv = self._normalize_advantage(batch.adv)
            
            # 获取该 agent 的独立 policy
            agent_policy = self.policy.get_policy(agent)
            
            # Policy loss (PPO clipped)
            new_log_prob, entropy = agent_policy.evaluate_actions(
                batch.obs, batch.act, action_mask=batch.action_mask
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
        
        # 求平均
        n = len(batch_dict)
        loss = (total_policy_loss / n 
                + self.vf_coef * total_value_loss / n 
                - self.ent_coef * total_entropy / n)
        
        stats = TrainingStats(
            policy_loss=(total_policy_loss / n).item(),
            value_loss=(total_value_loss / n).item(),
            entropy=(total_entropy / n).item()
        )
        return loss, stats
