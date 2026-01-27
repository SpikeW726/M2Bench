"""MAPPO: Multi-Agent PPO with Parameter Sharing and Centralized Critic."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from algorithms.algorithm_base import ActorCriticOnPolicyAlgo, TrainingStats
from polocies.marl.marl_base import MultiAgentPolicy
from data.batch import RolloutBatch


class MAPPOAlgo(ActorCriticOnPolicyAlgo):
    """
    MAPPO算法: 共享策略 + Centralized Critic
    
    参考 CleanRL PPO 实现：
    - Minibatch 级别 advantage normalization
    - 可选的 value loss clipping
    - 内部处理 minibatch 切分和多轮更新
    """
    
    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        num_envs: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        num_minibatches: int = 4,
        update_epochs: int = 10,
        clip_vloss: bool = True,
        target_kl: Optional[float] = None,
    ):
        nn.Module.__init__(self)
        
        self.policy = policy
        self.critic = critic.to(policy.device)
        self.device = policy.device
        self.num_envs = num_envs
        
        # PPO 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        
        # Minibatch 和更新轮数
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(critic.parameters()), lr=lr
        )
    
    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """
        计算 GAE 并合并所有 agent 数据
        
        处理向量化环境：数据排列为 (T*N, ...) 需要 reshape 为 (T, N, ...) 分别计算 GAE
        
        正确处理 truncation vs termination:
        - termination (真正结束): next_value = 0
        - truncation (时间截断): next_value = critic(next_state)
        """
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs
        
        all_obs, all_act, all_log_prob = [], [], []
        all_adv, all_ret, all_value = [], [], []
        all_critic_input, all_action_mask = [], []
        
        for i, agent in enumerate(agents):
            batch = batch_dict[agent].to_tensor(self.device)
            total_size = batch.global_state.shape[0]
            T = total_size // N  # num_steps
            
            # 构建 critic_input: global_state + agent_one_hot
            one_hot = torch.zeros(total_size, num_agents, device=self.device)
            one_hot[:, i] = 1.0
            critic_input = torch.cat([batch.global_state, one_hot], dim=-1)
            
            # Reshape 为 (T, N, ...) 用于按环境计算 GAE
            rew_2d = batch.rew.view(T, N)
            done_2d = batch.done.view(T, N)
            critic_2d = critic_input.view(T, N, -1)
            
            # 获取 truncated 信息用于正确的 value bootstrap
            truncated_2d = None
            if batch.truncated is not None:
                truncated_2d = batch.truncated.view(T, N)
            
            with torch.no_grad():
                values = self.critic(critic_input).squeeze(-1).view(T, N)
                
                # 为每个环境计算 next_value 和 GAE
                all_env_adv, all_env_ret = [], []
                for env_idx in range(N):
                    last_done = done_2d[-1, env_idx] > 0.5
                    # 判断最后一步是 truncation 还是 termination
                    last_truncated = truncated_2d is not None and truncated_2d[-1, env_idx] > 0.5
                    
                    if last_done and not last_truncated:
                        # 真正的 termination，next_value = 0
                        next_val = torch.tensor(0.0, device=self.device)
                    else:
                        # truncation 或未结束，使用 critic 估计 next_value
                        next_val = self.critic(critic_2d[-1, env_idx:env_idx+1]).squeeze()
                    
                    adv, ret = self.compute_gae(
                        rew_2d[:, env_idx], values[:, env_idx], 
                        done_2d[:, env_idx], next_val
                    )
                    all_env_adv.append(adv)
                    all_env_ret.append(ret)
                
                # (T, N) -> (T*N,) 保持原始数据顺序
                adv = torch.stack(all_env_adv, dim=1).view(-1)
                ret = torch.stack(all_env_ret, dim=1).view(-1)
                values_flat = values.view(-1)
            
            all_obs.append(batch.obs)
            all_act.append(batch.act)
            all_log_prob.append(batch.log_prob)
            all_adv.append(adv)
            all_ret.append(ret)
            all_value.append(values_flat)
            all_critic_input.append(critic_input)
            if batch.action_mask is not None:
                all_action_mask.append(batch.action_mask)
        
        return RolloutBatch(
            obs=torch.cat(all_obs, dim=0),
            act=torch.cat(all_act, dim=0),
            log_prob=torch.cat(all_log_prob, dim=0),
            adv=torch.cat(all_adv, dim=0),
            ret=torch.cat(all_ret, dim=0),
            value=torch.cat(all_value, dim=0),
            global_state=torch.cat(all_critic_input, dim=0),
            action_mask=torch.cat(all_action_mask, dim=0) if all_action_mask else None,
        )
    
    def update(self, batch: RolloutBatch) -> TrainingStats:
        """
        PPO 更新，参考 CleanRL 实现
        
        - Shuffle + minibatch 切分
        - Minibatch 级别 advantage normalization
        - 可选 value loss clipping
        - KL 早停使用整个 epoch 的平均 KL
        """
        batch_size = len(batch)
        minibatch_size = batch_size // self.num_minibatches
        
        all_pg_loss, all_v_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl = []  # 用于 KL 早停判断
        indices = np.arange(batch_size)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            epoch_approx_kl = []  # 当前 epoch 内所有 minibatch 的 KL
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # 取 minibatch 数据
                mb_obs = batch.obs[mb_inds]
                mb_act = batch.act[mb_inds]
                mb_log_prob = batch.log_prob[mb_inds]
                mb_adv = batch.adv[mb_inds]
                mb_ret = batch.ret[mb_inds]
                mb_value = batch.value[mb_inds]
                mb_critic_input = batch.global_state[mb_inds]
                mb_action_mask = batch.action_mask[mb_inds] if batch.action_mask is not None else None
                
                # Minibatch 级别 advantage normalization
                if self.normalize_advantage:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # Policy loss
                new_log_prob, entropy = self.policy.evaluate_actions_flat(
                    mb_obs, mb_act, action_mask=mb_action_mask
                )
                logratio = new_log_prob - mb_log_prob
                ratio = logratio.exp()
                
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                new_value = self.critic(mb_critic_input).squeeze(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (new_value - mb_ret) ** 2
                    v_clipped = mb_value + torch.clamp(
                        new_value - mb_value, -self.clip_range, self.clip_range
                    )
                    v_loss_clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()
                
                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # 记录统计
                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(v_loss.item())
                all_entropy.append(entropy_loss.item())
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    all_clipfrac.append(clipfrac)
                    # 计算当前 minibatch 的 approx KL
                    mb_approx_kl = ((ratio - 1) - logratio).mean().item()
                    epoch_approx_kl.append(mb_approx_kl)
            
            # KL 早停：使用整个 epoch 的平均 KL
            if self.target_kl is not None and epoch_approx_kl:
                avg_epoch_kl = np.mean(epoch_approx_kl)
                all_approx_kl.append(avg_epoch_kl)
                if avg_epoch_kl > self.target_kl:
                    break
        
        return TrainingStats(
            loss=np.mean(all_pg_loss) + self.vf_coef * np.mean(all_v_loss),
            policy_loss=np.mean(all_pg_loss),
            value_loss=np.mean(all_v_loss),
            entropy=np.mean(all_entropy),
            extra={
                "clipfrac": np.mean(all_clipfrac),
                "approx_kl": np.mean(all_approx_kl) if all_approx_kl else 0.0,
            }
        )
    
    def set_training_mode(self, mode: bool):
        """设置训练/评估模式"""
        self.train(mode)
        self.policy.set_training_mode(mode)
