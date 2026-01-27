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
        
        正确处理 truncation vs termination（包括中间的 done）:
        - termination (真正结束): next_value = 0
        - truncation (时间截断): next_value = critic(final_state)
        """
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs
        
        all_obs, all_act, all_log_prob = [], [], []
        all_adv, all_ret, all_value = [], [], []
        all_critic_input, all_action_mask = [], []
        
        for i, agent in enumerate(agents):
            batch = batch_dict[agent]
            # 提取 final_global_state（List[List[ndarray or None]]，不转为 tensor）
            final_gs = batch.final_global_state  # T x N 的嵌套列表
            
            batch = batch.to_tensor(self.device)
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
                
                # 为每个环境计算 GAE，正确处理中间 truncation
                all_env_adv, all_env_ret = [], []
                for env_idx in range(N):
                    adv, ret = self._compute_gae_with_truncation(
                        rewards=rew_2d[:, env_idx],
                        values=values[:, env_idx],
                        dones=done_2d[:, env_idx],
                        truncateds=truncated_2d[:, env_idx] if truncated_2d is not None else None,
                        final_global_states=[final_gs[t][env_idx] if final_gs else None for t in range(T)],
                        critic_input_last=critic_2d[-1, env_idx:env_idx+1],
                        agent_idx=i,
                        num_agents=num_agents,
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
    
    def _compute_gae_with_truncation(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        truncateds: Optional[torch.Tensor],
        final_global_states: List[Optional[np.ndarray]],
        critic_input_last: torch.Tensor,
        agent_idx: int,
        num_agents: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE，正确处理中间的 truncation
        
        Args:
            rewards: (T,) 奖励序列
            values: (T,) value 估计
            dones: (T,) done 标志
            truncateds: (T,) truncation 标志
            final_global_states: 长度为 T 的列表，每个元素是 final_state 或 None
            critic_input_last: 最后一步的 critic 输入（用于非 done 情况）
            agent_idx: 当前 agent 索引（用于构建 one_hot）
            num_agents: agent 总数
        
        Returns:
            advantages: (T,)
            returns: (T,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            is_done = dones[t] > 0.5
            is_truncated = truncateds is not None and truncateds[t] > 0.5
            
            if t == T - 1:
                # 最后一步
                if is_done and not is_truncated:
                    # termination: next_value = 0
                    next_val = torch.tensor(0.0, device=self.device)
                elif is_done and is_truncated:
                    # truncation: 使用 final_state 计算 next_value
                    next_val = self._get_truncation_value(
                        final_global_states[t], agent_idx, num_agents
                    )
                else:
                    # 未结束: 使用最后一步的 critic 输入
                    next_val = self.critic(critic_input_last).squeeze()
            else:
                # 中间步
                if is_done and not is_truncated:
                    # termination: next_value = 0，GAE 截断
                    next_val = torch.tensor(0.0, device=self.device)
                elif is_done and is_truncated:
                    # truncation: 使用 final_state 计算 next_value
                    next_val = self._get_truncation_value(
                        final_global_states[t], agent_idx, num_agents
                    )
                else:
                    # 正常情况: 使用下一步的 value
                    next_val = values[t + 1]
            
            # TD error
            if is_done and not is_truncated:
                # termination: 完全截断，不使用 next_val
                delta = rewards[t] - values[t]
                last_gae = delta
            elif is_done and is_truncated:
                # truncation: 使用 bootstrapped next_val，但 GAE 在这里截断
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = delta
            else:
                # 正常情况
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def _get_truncation_value(
        self,
        final_state: Optional[np.ndarray],
        agent_idx: int,
        num_agents: int,
    ) -> torch.Tensor:
        """获取 truncation 时的 bootstrapped value"""
        if final_state is None:
            # 如果没有 final_state，回退到 0
            return torch.tensor(0.0, device=self.device)
        
        # 构建 critic_input: final_state + agent_one_hot
        state_t = torch.as_tensor(final_state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        one_hot = torch.zeros(1, num_agents, device=self.device)
        one_hot[0, agent_idx] = 1.0
        critic_input = torch.cat([state_t, one_hot], dim=-1)
        
        return self.critic(critic_input).squeeze()
    
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
