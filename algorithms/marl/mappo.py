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
            
            # Reshape 为 (T, N) 用于向量化 GAE
            rew_2d = batch.rew.view(T, N)
            done_2d = batch.done.view(T, N)
            
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
                
                # 向量化 GAE（沿 env 维度并行，消除 per-env Python 循环）
                adv, ret = self._compute_gae_vectorized(
                    rewards=rew_2d,          # (T, N) real scale
                    values=values,           # (T, N) real scale
                    dones=done_2d,           # (T, N)
                    truncateds=truncated_2d, # (T, N) or None
                    final_global_states=final_gs,  # T x N nested list
                    agent_idx=i,
                    num_agents=num_agents,
                )
                
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
    
    def _compute_gae_vectorized(
        self,
        rewards: torch.Tensor,                          # (T, N)
        values: torch.Tensor,                           # (T, N) real scale (已反归一化)
        dones: torch.Tensor,                            # (T, N)
        truncateds: Optional[torch.Tensor],             # (T, N) or None
        final_global_states,                            # T x N nested list
        agent_idx: int,
        num_agents: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        向量化 GAE 计算，沿 env 维度并行。
        
        逻辑与原 _compute_gae_with_truncation 完全等价，但消除了 per-env 的
        Python for-loop，仅保留时间维度 T 的反向循环。
        
        三种情况统一为:
            next_val:  terminated→0,  truncated→V(final_state),  normal→V[t+1]
            delta   =  r[t] + γ * next_val - V[t]   （统一公式）
            last_gae:  done→δ（重置）,  ¬done→δ + γλ·last_gae
        
        Returns:
            advantages: (T*N,)
            returns:    (T*N,)
        """
        T, N = rewards.shape
        device = rewards.device

        # ---- 1. 构建 truncation mask (T, N) ----
        if truncateds is not None:
            trunc_mask = truncateds > 0.5                # (T, N) bool
        else:
            trunc_mask = torch.zeros(T, N, dtype=torch.bool, device=device)

        done_mask = dones > 0.5                          # (T, N) bool
        term_mask = done_mask & ~trunc_mask              # terminated: done 且非 truncated

        # ---- 2. 批量预计算所有 truncation 处的 bootstrap value ----
        #   收集所有 (t, env) 处的 final_state，一次 critic forward 替代逐个调用
        trunc_bootstrap = torch.zeros(T, N, device=device)

        if trunc_mask.any():
            trunc_positions = trunc_mask.nonzero(as_tuple=False)  # (K, 2)

            # 收集有效的 final_state
            batch_states = []
            valid_k_indices = []                          # 在 trunc_positions 中的行号
            for k in range(len(trunc_positions)):
                t_idx = trunc_positions[k, 0].item()
                e_idx = trunc_positions[k, 1].item()
                fs = final_global_states[t_idx][e_idx] if final_global_states else None
                if fs is not None:
                    batch_states.append(
                        torch.as_tensor(fs, dtype=torch.float32, device=device)
                    )
                    valid_k_indices.append(k)
                # fs 为 None 时保持 trunc_bootstrap[t,e]=0（与原 _get_truncation_value 一致）

            if batch_states:
                states_t = torch.stack(batch_states)          # (K_valid, state_dim)
                one_hot = torch.zeros(len(batch_states), num_agents, device=device)
                one_hot[:, agent_idx] = 1.0
                critic_in = torch.cat([states_t, one_hot], dim=-1)

                v_norm = self.critic(critic_in).squeeze(-1)   # (K_valid,)
                if self.use_value_norm and self.ret_rms is not None:
                    v_real = v_norm * self.ret_rms.std + self.ret_rms.mean
                else:
                    v_real = v_norm

                # 填入 trunc_bootstrap
                for vi, k in enumerate(valid_k_indices):
                    t_idx = trunc_positions[k, 0].item()
                    e_idx = trunc_positions[k, 1].item()
                    trunc_bootstrap[t_idx, e_idx] = v_real[vi]

        # ---- 3. 最后一步 (t=T-1) 的 bootstrap value ----
        #   原始代码: _get_denormalized_value(critic_input_last)
        #   ≡ values[-1] (因为 critic 输入相同，denormalize 相同)
        last_step_bootstrap = values[-1].clone()              # (N,)

        # ---- 4. 反向循环（仅沿 T 维度，N 维度完全向量化） ----
        advantages = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            # 4a. 计算 next_val (N,)
            if t == T - 1:
                next_val = last_step_bootstrap.clone()
            else:
                next_val = values[t + 1].clone()              # normal: V[t+1]

            # terminated → next_val = 0
            next_val = torch.where(term_mask[t], torch.zeros_like(next_val), next_val)

            # truncated → next_val = V(final_state)
            if trunc_mask[t].any():
                next_val = torch.where(trunc_mask[t], trunc_bootstrap[t], next_val)

            # 4b. TD error: δ = r + γ·next_val − V  （三种情况的统一公式）
            delta = rewards[t] + self.gamma * next_val - values[t]

            # 4c. GAE 更新: done 时重置为 δ，否则 δ + γλ·last_gae
            last_gae = torch.where(
                done_mask[t],
                delta,
                delta + self.gamma * self.gae_lambda * last_gae,
            )

            advantages[t] = last_gae

        returns = advantages + values
        return advantages.view(-1), returns.view(-1)

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
            
            # 记录 epoch 平均 approx_kl
            if epoch_approx_kl:
                avg_epoch_kl = np.mean(epoch_approx_kl)
                all_approx_kl.append(avg_epoch_kl)
                # KL early stopping（仅在设置 target_kl 时生效）
                if self.target_kl is not None and avg_epoch_kl > self.target_kl:
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
