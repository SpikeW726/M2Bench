from json import encoder
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from networks.mlp_SUP_joint import build_joint_sup_mlp


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class PPOAgent(nn.Module):
    """
    PPO离散联合动作智能体,实现细节参考 https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#Schulman2017
    - 不继承 BaseAgent，暴露 act() 与 update() 接口供 centralized_joint_trainer 调用
    """

    def __init__(self, config: Dict[str, Any], env: Any):
        super().__init__()
        self.config = config
        self.env = env
        # 读取各配置段（命名更清晰）
        agent_config = config.get('agent_config', {})
        env_config = config.get('env_config', {})
        train_config = config.get('train_config', {})
        custom_config = config.get('custom_config', {})

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 超参（保持对外字段名不变，便于外部组件访问）
        self.lr = float(agent_config.get('learning_rate', 3e-4))
        self.min_lr = float(agent_config.get('min_lr', 0.0))  # 学习率退火的最小值
        self.use_lr_anneal = bool(agent_config.get('use_lr_anneal', True))  # 是否启用学习率退火
        self.entropy_coeff = float(agent_config.get('entropy_coeff', 0.01))
        self.entropy_coeff_start = float(agent_config.get('entropy_coeff', 0.01)) # 用于衰减的初始值
        self.entropy_end = float(agent_config.get('entropy_end', self.entropy_coeff))
        self.use_entropy_decay = bool(agent_config.get('use_entropy_decay', True))
        self.max_grad_norm = float(agent_config.get('max_grad_norm', 0.5))
        self.ratio_clip = float(agent_config.get('ratio_clip', 0.1))
        self.discount = float(agent_config.get('discount', 1.0))
        self.gae_lambda = float(agent_config.get('gae_lambda', 0.9))
        self.use_action_mask = bool(agent_config.get('use_action_mask', True))
        self.vf_coef = float(agent_config.get('value_loss_coeff', agent_config.get('vf_coef', 0.5)))
        self.value_clip = float(agent_config.get('value_clip', 0.2))  # PPO式价值裁剪阈值
        self.critic_lr_ratio = float(agent_config.get('critic_lr_ratio', 1.0))
        self.kl_threshold = float(agent_config.get('kl_threshold', 0.02))
        self.kl_lr_reduce = float(agent_config.get('kl_lr_reduce', 1.0))
        self.use_kl_early_stop = bool(agent_config.get('use_kl_early_stop', True))  # KL早停开关
        self.normalize_advantage = bool(agent_config.get('normalize_advantage', True)) # 优势归一化
        self.normalize_returns = bool(agent_config.get('normalize_returns', True)) # 回报归一化开关
        self.total_steps = int(train_config.get('total_steps', 100000))

        # 维度推断（根据环境动态计算动作维度）
        self.num_agents = int(env_config.get('num_agents', 1))
        num_target = int(self.env.action_space[0].spaces['target'].n)
        num_duration = int(self.env.action_space[0].spaces['action_duration'].n)
        
        # 输入拼接：用户可直接给定 input_dim；否则根据 SUP_env 推断
        input_dim = int(agent_config.get('input_dim', 0))
        if input_dim <= 0:
            obs_space = self.env.observation_space
            input_dim = sum([v.shape[0] for k, v in obs_space.spaces.items()])
            
        net_config = {
            'input_dim': input_dim,
            'hidden_dims': agent_config.get('hidden_dims', [256, 128]),
            'num_agents': self.num_agents,
            'num_target': num_target,
            'num_duration': num_duration,
        }

        self.encoder, self.actor, self.critic = build_joint_sup_mlp(net_config)
        self.to(self.device)

        # 为 critic 使用更小学习率（按子模块区分，避免参数张量比较）
        encoder_params = list(self.encoder.parameters())
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": encoder_params, "lr": self.lr},
            {"params": actor_params, "lr": self.lr},
            {"params": critic_params, "lr": self.lr * self.critic_lr_ratio},
        ])

    # ---------- 工具 ----------
    def _preprocess_obs(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """将字典观测拼接为单个扁平张量，形状为 (B, D)。"""
        # obs is a dict of numpy arrays, first dim is batch size
        batch_size = obs['continuous'].shape[0]
        continuous_part = obs['continuous'].reshape(batch_size, -1)
        discrete_part = obs['discrete'].reshape(batch_size, -1)
        
        input_vector = np.concatenate([continuous_part, discrete_part], axis=1)
        return _to_tensor(input_vector, self.device)

    @torch.no_grad()
    def estimate_value(self, obs: Dict[str, Any]) -> torch.Tensor:
        """公开的估值接口：从原始观测计算 V(s)。支持批处理"""
        x = self._preprocess_obs(obs)
        feat = self.encoder(x)
        value = self.critic(feat)
        return value

    # ---------- 接口 ----------
    @torch.no_grad()
    def act(self, obs: Dict[str, Any], evaluation: bool = False, is_single_env: bool = False, masks: Dict[str, Any] = None):
        if is_single_env:
            obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
            if masks is not None:
                masks = {k: np.expand_dims(v, axis=0) for k, v in masks.items()}

        mask_t_tensor = None
        mask_d_tensor = None
        if masks is not None:
            mask_t_tensor = torch.as_tensor(masks.get('mask_t'), device=self.device, dtype=torch.bool)
            mask_d_tensor = torch.as_tensor(masks.get('mask_d'), device=self.device, dtype=torch.bool)

        input_tensor = self._preprocess_obs(obs)
        features = self.encoder(input_tensor)
        target_logits, duration_logits = self.actor(features)
        values = self.critic(features)

        if self.use_action_mask and mask_t_tensor is not None and mask_d_tensor is not None:
            target_logits = target_logits.masked_fill(~mask_t_tensor, -1e9)
            duration_logits = duration_logits.masked_fill(~mask_d_tensor, -1e9)
        
        target_dist = Categorical(logits=target_logits)
        duration_dist = Categorical(logits=duration_logits)
        
        if evaluation:
            targets = torch.argmax(target_logits, dim=-1)
            durations = torch.argmax(duration_logits, dim=-1)
        else:
            targets = target_dist.sample()
            durations = duration_dist.sample()

        log_prob_targets = target_dist.log_prob(targets)
        log_prob_durations = duration_dist.log_prob(durations)
        
        # 仅当选择等待时，才考虑 duration 的 log_prob
        is_waiting = (targets == 0).float()
        log_probs = log_prob_targets + log_prob_durations * is_waiting

        if is_single_env:
            action_tuple = [{"target": int(t), "action_duration": int(d)} for t, d in zip(targets[0], durations[0])]
            return action_tuple, log_probs[0], values[0], {}
        else:
            env_actions = []
            for env_idx in range(targets.shape[0]):
                per_env = tuple({"target": int(t.item()), "action_duration": int(d.item())} for t, d in zip(targets[env_idx], durations[env_idx]))
                env_actions.append(per_env)
            return (targets, durations), values, log_probs, {'env_actions': env_actions}

    def update(self, batch: Dict[str, torch.Tensor], update_epochs: int = 4, minibatch_size: int = 0, global_step: int = 0) -> Dict[str, float]:
        """基于 RolloutBuffer 的批次进行多轮 PPO 风格更新。"""
        # --- 1) 学习率退火 ---
        if self.use_lr_anneal and self.total_steps > 0:
            progress = min(float(global_step) / self.total_steps, 1.0)
            current_lr = self.lr * (1.0 - progress) + self.min_lr * progress
            for param_group in self.optimizer.param_groups:
                if 'lr' in param_group: # critic might have different lr
                    param_group['lr'] = current_lr * self.critic_lr_ratio if 'critic' in param_group.get('name', '') else current_lr

        # --- 2) 可选的 entropy 系数衰减 ---
        if self.use_entropy_decay and self.total_steps > 0:
            progress = min(float(global_step) / self.total_steps, 1.0)
            self.entropy_coeff = self.entropy_coeff_start * (1.0 - progress) + self.entropy_end * progress
        
        # --- 3) 提取批数据 ---
        obs_batch = batch['obs']
        actions_t_batch = batch['actions_t']
        actions_d_batch = batch['actions_d']
        log_probs_old_batch = batch['log_probs']
        advantages_batch = batch['advantages']
        returns_batch = batch['returns']
        values_old_batch = batch['values']
        mask_t_batch = batch['mask_t']
        mask_d_batch = batch['mask_d']

        if self.normalize_advantage:
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            
        batch_size = actions_t_batch.shape[0]
        if minibatch_size <= 0:
            minibatch_size = batch_size
        
        # --- 4) 指标累计器 ---
        all_losses, all_policy_losses, all_value_losses, all_entropies, all_approx_kls = [], [], [], [], []

        # --- 5) 多 epoch 的小批次更新 ---
        for epoch in range(int(update_epochs)):
            shuffled_indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_indices = shuffled_indices[start:end]

                # --- a. 获取minibatch数据 ---
                obs_mini_batch = {k: v[minibatch_indices] for k, v in obs_batch.items()}
                actions_t_mini_batch = actions_t_batch[minibatch_indices]
                actions_d_mini_batch = actions_d_batch[minibatch_indices]
                log_probs_old_mini_batch = log_probs_old_batch[minibatch_indices]
                advantages_mini_batch = advantages_batch[minibatch_indices]
                returns_mini_batch = returns_batch[minibatch_indices]
                values_old_mini_batch = values_old_batch[minibatch_indices]
                target_mask_batch = mask_t_batch[minibatch_indices]
                duration_mask_batch = mask_d_batch[minibatch_indices]

                # --- b. 重新前向计算 ---
                input_tensor = self._preprocess_obs(obs_mini_batch)
                features = self.encoder(input_tensor)
                new_target_logits, new_duration_logits = self.actor(features)
                new_values = self.critic(features).squeeze(-1)

                if self.use_action_mask:
                    new_target_logits = new_target_logits.masked_fill(~target_mask_batch, -1e9)
                    new_duration_logits = new_duration_logits.masked_fill(~duration_mask_batch, -1e9)
                
                # --- c. 计算新logp和熵 ---
                target_dist = Categorical(logits=new_target_logits)
                duration_dist = Categorical(logits=new_duration_logits)
                
                log_prob_targets = target_dist.log_prob(actions_t_mini_batch)
                log_prob_durations = duration_dist.log_prob(actions_d_mini_batch)
                
                is_waiting = (actions_t_mini_batch == 0).float()
                if self.use_action_mask:
                    allowed_duration_count = duration_mask_batch.to(torch.int32).sum(dim=-1)
                    include_duration = (is_waiting.bool() & (allowed_duration_count > 1)).to(log_prob_durations.dtype)
                else:
                    include_duration = is_waiting
                log_probs_new = log_prob_targets + log_prob_durations * include_duration

                entropy = (target_dist.entropy() + duration_dist.entropy() * include_duration).mean()

                # --- d. 计算损失 ---
                log_ratio = (log_probs_new - log_probs_old_mini_batch).sum(dim=-1)
                ratio = torch.exp(log_ratio)

                clipped_ratio = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
                policy_loss = -torch.min(ratio * advantages_mini_batch, clipped_ratio * advantages_mini_batch).mean()

                value_diff = new_values - values_old_mini_batch
                values_clipped = values_old_mini_batch + value_diff.clamp(-self.value_clip, self.value_clip)
                value_loss_unclipped = (new_values - returns_mini_batch).pow(2)
                value_loss_clipped = (values_clipped - returns_mini_batch).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coeff * entropy

                # --- e. 优化 ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- f. 记录指标 ---
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                all_losses.append(loss.item())
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.item())
                all_approx_kls.append(approx_kl)
        
        metrics = {
            'loss': np.mean(all_losses),
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy': np.mean(all_entropies),
            'approx_kl': np.mean(all_approx_kls),
        }
        self.last_metrics = metrics
        return metrics


    # ---------- 序列化 ----------
    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            'model': super().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):  # type: ignore[override]
        incompatible_keys = super().load_state_dict(state.get('model', {}), strict=strict)
        if 'optimizer' in state:
            try:
                self.optimizer.load_state_dict(state['optimizer'])
            except Exception:
                pass
        return incompatible_keys


