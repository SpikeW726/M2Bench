from json import encoder
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithm_base import ActorCriticOnPolicyAlgo
from policies.rl.rl_base import ActorPolicy
from data.batch import RolloutBatch

class PPOAlgo(ActorCriticOnPolicyAlgo):
    """
    Proximal Policy Optimization algorithm.
    
    Paper: https://arxiv.org/abs/1707.06347
    """
    
    def __init__(
        self,
        policy: ActorPolicy,
        critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
    ):
        super().__init__(
            policy, critic, lr, gamma, gae_lambda,
            vf_coef, ent_coef, max_grad_norm, normalize_advantage
        )
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
    
    def compute_policy_loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute PPO clipped policy loss."""
        # Evaluate actions under current policy
        new_log_prob, entropy = self.policy.evaluate_actions(
            batch.obs, batch.act, action_mask=batch.action_mask
        )
        
        # Importance sampling ratio
        ratio = torch.exp(new_log_prob - batch.log_prob)
        
        # Clipped surrogate loss
        surr1 = ratio * batch.adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch.adv
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss, entropy.mean()
    
    def compute_value_loss(self, batch: RolloutBatch) -> torch.Tensor:
        """Compute value loss with optional clipping."""
        # Critic input: local obs or global_state
        critic_input = batch.global_state if batch.global_state is not None else batch.obs
        value_pred = self.critic(critic_input).squeeze(-1)
        
        if self.clip_range_vf is not None and batch.value is not None:
            # Clipped value loss (PPO2 style)
            value_clipped = batch.value + torch.clamp(
                value_pred - batch.value,
                -self.clip_range_vf,
                self.clip_range_vf
            )
            vf_loss1 = F.mse_loss(value_pred, batch.ret, reduction='none')
            vf_loss2 = F.mse_loss(value_clipped, batch.ret, reduction='none')
            value_loss = torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = F.mse_loss(value_pred, batch.ret)
        
        return value_loss