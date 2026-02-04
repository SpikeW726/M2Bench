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
    MAPPO: Multi-Agent PPO with Parameter Sharing and Centralized Critic
    
    Features:
    - Separate optimizers for actor and critic (dual timescale update)
    - Minibatch-level advantage normalization
    - Optional value loss clipping
    - Entropy loss only added to policy loss (not critic loss)
    """
    
    def __init__(
        self,
        policy: MultiAgentPolicy,
        critic: nn.Module,
        num_envs: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
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
        # Learning rate scheduler parameters
        use_lr_scheduler: bool = False,
        actor_lr_start_factor: float = 1.0,
        actor_lr_end_factor: float = 0.1,
        actor_lr_decay_ratio: float = 0.8,
        critic_lr_start_factor: float = 1.0,
        critic_lr_end_factor: float = 0.1,
        critic_lr_decay_ratio: float = 0.8,
        total_iterations: Optional[int] = None,
        # Value Normalization parameters
        use_value_norm: bool = False,
        value_norm_config: Optional[Dict] = None,
    ):
        nn.Module.__init__(self)
        
        self.policy = policy
        self.critic = critic.to(policy.device)
        self.device = policy.device
        self.num_envs = num_envs
        
        # PPO hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        
        # Minibatch and epochs
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        # Learning rate schedulers
        self.use_lr_scheduler = use_lr_scheduler
        self.actor_scheduler = None
        self.critic_scheduler = None

        if use_lr_scheduler and total_iterations is not None:
            # 计算 total optimizer step 次数
            steps_per_iteration = self.update_epochs * self.num_minibatches

            # Actor scheduler
            actor_decay_steps = int(total_iterations * actor_lr_decay_ratio * steps_per_iteration)
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_optimizer,
                start_factor=actor_lr_start_factor,
                end_factor=actor_lr_end_factor,
                total_iters=actor_decay_steps,
            )

            # Critic scheduler
            critic_decay_steps = int(total_iterations * critic_lr_decay_ratio * steps_per_iteration)
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_optimizer,
                start_factor=critic_lr_start_factor,
                end_factor=critic_lr_end_factor,
                total_iters=critic_decay_steps,
            )

        # Value Normalization
        self.use_value_norm = use_value_norm
        self.ret_rms = None

        if self.use_value_norm:
            from utils.train_utils import RunningMeanStd
            self.ret_rms = RunningMeanStd(shape=(1,)).to(self.device)  # 移动到正确的设备

            # 从预训练 checkpoint 继承统计量
            if value_norm_config is not None:
                self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
                self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
                self.ret_rms.count.fill_(1.0)  # 标记为已初始化
                print(f"[MAPPO] Loaded value_norm stats: mean={self.ret_rms.mean.item():.4f}, std={self.ret_rms.std.item():.4f}")
            else:
                print(f"[MAPPO] Initialized value_norm with default: mean=0, std=1")
    
    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """
        计算 GAE 并合并所有 agent 数据
        [已修复 Bug]: batch.value 现在存储归一化尺度的值，以便在 PPO update 中与 critic 输出进行正确的 clipping。
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
                # Raw output from critic (Normalized Scale if use_value_norm=True)
                values_norm = self.critic(critic_input).squeeze(-1).view(T, N)

                # 反归一化 value 用于 GAE 计算 (Real Scale)
                if self.use_value_norm and self.ret_rms is not None:
                    values = values_norm * self.ret_rms.std + self.ret_rms.mean
                else:
                    values = values_norm
                
                # 为每个环境计算 GAE
                all_env_adv, all_env_ret = [], []
                for env_idx in range(N):
                    # 注意：这里传入的是 Real Scale 的 values，这是正确的！GAE 需要物理意义
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
                
                # [Fix Bug]: 存储到 Buffer 中的 Value 必须是归一化尺度的 (values_norm)
                # 这样在 PPO Update 时，它才能和 Critic 的输出 (new_value) 进行正确的 clip 操作
                values_flat = values_norm.view(-1) 
            
            all_obs.append(batch.obs)
            all_act.append(batch.act)
            all_log_prob.append(batch.log_prob)
            all_adv.append(adv)
            all_ret.append(ret)
            all_value.append(values_flat)
            all_critic_input.append(critic_input)
            if batch.action_mask is not None:
                all_action_mask.append(batch.action_mask)

        # 更新 Value Normalization 统计量
        if self.use_value_norm and self.ret_rms is not None:
            # Stack all returns: [num_agents, T*N] -> [T*N*num_agents]
            # 注意：这里的 ret 依然是 Real Scale (GAE 算出来的)，用来更新统计量是正确的
            all_ret_tensor = torch.cat(all_ret, dim=0)
            self.ret_rms.update(all_ret_tensor)

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
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            is_done = dones[t] > 0.5
            is_truncated = truncateds is not None and truncateds[t] > 0.5
            
            # --- 计算 next_val (Bootstrap Value) ---
            if t == T - 1:
                # 最后一步
                if is_done and not is_truncated:
                    # termination: next_value = 0
                    next_val = torch.tensor(0.0, device=self.device)
                elif is_done and is_truncated:
                    # truncation: 使用 final_state 计算 next_value (已修复内部反归一化)
                    next_val = self._get_truncation_value(
                        final_global_states[t], agent_idx, num_agents
                    )
                else:
                    # 未结束: 使用最后一步的 critic 输入 (修复：增加反归一化)
                    next_val = self._get_denormalized_value(critic_input_last)
            else:
                # 中间步
                if is_done and not is_truncated:
                    # termination: next_value = 0
                    next_val = torch.tensor(0.0, device=self.device)
                elif is_done and is_truncated:
                    # truncation: 使用 final_state 计算 next_value (已修复内部反归一化)
                    next_val = self._get_truncation_value(
                        final_global_states[t], agent_idx, num_agents
                    )
                else:
                    # 正常情况: 使用下一步的 value (注意：这里的 values 已经在 prepare_batch 里反归一化过了，直接用)
                    next_val = values[t + 1]
            
            # --- 计算 TD Error 和 GAE ---
            # ... (这部分逻辑保持不变，因为 next_val 现在的尺度是对的了)
            if is_done and not is_truncated:
                delta = rewards[t] - values[t]
                last_gae = delta
            elif is_done and is_truncated:
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = delta
            else:
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
        """获取 truncation 时的 bootstrapped value (已修复：增加反归一化)"""
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
        
        # 修复：使用统一的反归一化逻辑
        return self._get_denormalized_value(critic_input)
    
    def _get_denormalized_value(self, critic_input: torch.Tensor) -> torch.Tensor:
        """辅助函数：获取反归一化后的 Critic 价值"""
        with torch.no_grad():
            value_norm = self.critic(critic_input).squeeze()
            if self.use_value_norm and self.ret_rms is not None:
                return value_norm * self.ret_rms.std + self.ret_rms.mean
            return value_norm

    def update(self, batch: RolloutBatch, update_actor: bool = True) -> TrainingStats:
        """
        PPO update with separate actor/critic optimization.
        
        Args:
            batch: RolloutBatch with computed advantages and returns
            update_actor: whether to update actor (for dual timescale update)
        
        Features:
            - Shuffle + minibatch split
            - Minibatch-level advantage normalization
            - Optional value loss clipping
            - Entropy loss only added to actor loss
            - KL early stopping based on epoch average
        """
        batch_size = len(batch)
        minibatch_size = batch_size // self.num_minibatches
        
        all_pg_loss, all_v_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl = []
        all_actor_grad_norm, all_critic_grad_norm = [], []
        indices = np.arange(batch_size)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            epoch_approx_kl = []
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Get minibatch data
                mb_obs = batch.obs[mb_inds]
                mb_act = batch.act[mb_inds]
                mb_log_prob = batch.log_prob[mb_inds]
                mb_adv = batch.adv[mb_inds]
                mb_ret = batch.ret[mb_inds]
                mb_value = batch.value[mb_inds]
                mb_critic_input = batch.global_state[mb_inds]
                mb_action_mask = batch.action_mask[mb_inds] if batch.action_mask is not None else None
                
                # Minibatch-level advantage normalization
                if self.normalize_advantage:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # ===== Actor update =====
                new_log_prob, entropy = self.policy.evaluate_actions_flat(
                    mb_obs, mb_act, action_mask=mb_action_mask
                )
                logratio = new_log_prob - mb_log_prob
                ratio = logratio.exp()
                
                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Entropy loss (only for actor)
                entropy_loss = entropy.mean()
                actor_loss = pg_loss - self.ent_coef * entropy_loss
                
                self.actor_optimizer.zero_grad()
                if update_actor:
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    all_actor_grad_norm.append(actor_grad_norm.item())
                
                # ===== Critic update =====
                new_value = self.critic(mb_critic_input).squeeze(-1)

                # 归一化 target 用于 Critic loss
                if self.use_value_norm and self.ret_rms is not None:
                    target_norm = (mb_ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                    target = target_norm
                else:
                    target = mb_ret

                if self.clip_vloss:
                    v_loss_unclipped = (new_value - target) ** 2
                    v_clipped = mb_value + torch.clamp(
                        new_value - mb_value, -self.clip_range, self.clip_range
                    )
                    v_loss_clipped = (v_clipped - target) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - target) ** 2).mean()
                
                critic_loss = self.vf_coef * v_loss
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
                all_critic_grad_norm.append(critic_grad_norm.item())

                # Learning rate scheduler step
                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()
                
                # Record stats
                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(v_loss.item())
                all_entropy.append(entropy_loss.item())
                
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    all_clipfrac.append(clipfrac)
                    mb_approx_kl = ((ratio - 1) - logratio).mean().item()
                    epoch_approx_kl.append(mb_approx_kl)
            
            # KL early stopping based on epoch average
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
                "actor_grad_norm": np.mean(all_actor_grad_norm) if all_actor_grad_norm else 0.0,
                "critic_grad_norm": np.mean(all_critic_grad_norm),
            }
        )
    
    def set_training_mode(self, mode: bool):
        """设置训练/评估模式"""
        self.train(mode)
        self.policy.set_training_mode(mode)
