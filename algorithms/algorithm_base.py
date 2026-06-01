"""RL 算法基类层次结构。

BaseAlgorithm
├── ActorCriticOnPolicyAlgo  ← AC On-Policy 通用基础设施 (GAE, prepare_batch)
│   └── A2CBase              ← update 骨架 + hook methods (在 rl/a2c.py 中定义)
│       ├── A2CAlgo          ← 单智能体 A2C
│       ├── MAA2CAlgo        ← 多智能体 A2C (+ CentralizedCriticMixin)
│       └── PPOBase          ← PPO hook overrides (在 rl/ppo.py 中定义)
│           ├── PPOAlgo
│           ├── IPPOAlgo
│           └── MAPPOAlgo    ← (+ CentralizedCriticMixin)
└── QLearningOffPolicyAlgo   (DQN/SAC, 待完善)

设计原则：
- ActorCriticOnPolicyAlgo 只含算法无关的公共逻辑 (GAE, value norm, prepare_batch)
- A2CBase 提供 update 骨架 (Template Method)，通过 hook 支持 A2C/PPO 差异化
- PPO 特有逻辑 (clipped surrogate, KL early stopping) 在 PPOBase 中 override
- CTDE 辅助方法在 CentralizedCriticMixin (marl/ctde_mixin.py) 中
- optimizer / lr_scheduler 由最终子类创建
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from configs.algo_configs import OnPolicyParams
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
    def update(self, batch, **kwargs) -> TrainingStats:
        pass

    def set_training_mode(self, mode: bool):
        """切换 train/eval 模式。"""
        self.train(mode)
        self.policy.set_training_mode(mode)


# ============================================================================
#                          On-Policy Algorithm
# ============================================================================

class ActorCriticOnPolicyAlgo(BaseAlgorithm):
    """
    On-policy Actor-Critic 通用基类。

    提供算法无关的公共基础设施，不含任何 PPO / A2C 特有逻辑：
    - _gae_vectorized():        向量化 GAE（标准 + A+ 透明 GAE）
    - _compute_trunc_bootstrap(): truncation 处的 V bootstrap
    - prepare_batch():          单 RolloutBatch 的 GAE 预处理

    update() 留给子类（PPOBase / A2CBase 等）实现。
    """

    def __init__(
        self,
        policy,
        critic: Optional[nn.Module],
        params: OnPolicyParams,
        num_envs: int = 1,
        value_norm_config: Optional[Dict] = None,
    ):
        super().__init__(policy)
        self.critic = critic.to(self.device) if critic is not None else None
        self.num_envs = num_envs

        # Actor-Critic On-Policy 通用超参
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.vf_coef = params.vf_coef
        self.ent_coef = params.ent_coef
        self.max_grad_norm = params.max_grad_norm
        self.normalize_advantage = params.normalize_advantage
        self.use_active_mask = params.use_active_mask
        self.use_value_norm = params.use_value_norm
        self.data_chunk_length = params.data_chunk_length

        # Value Normalization
        self.ret_rms = None
        if self.use_value_norm:
            from utils.train_utils import RunningMeanStd
            self.ret_rms = RunningMeanStd(shape=(1,)).to(self.device)
            if value_norm_config is not None:
                self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
                self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
                self.ret_rms.count.fill_(float(value_norm_config.get('ret_count', 1)))
                # self.ret_rms.count.fill_(1.0)

        # RNN critic hidden state（跨 collect 持久化）
        self._critic_hidden: Optional[torch.Tensor] = None

    @property
    def is_recurrent(self) -> bool:
        """Actor 是否为 RNN"""
        return getattr(self.policy, "is_recurrent", False)

    @property
    def is_critic_recurrent(self) -> bool:
        """Critic 是否为 RNN"""
        if self.critic is None:
            return False
        return getattr(self.critic, "is_recurrent", False)

    @property
    def is_any_recurrent(self) -> bool:
        """Actor 或 Critic 任一为 RNN 时，需要 chunk_split"""
        return self.is_recurrent or self.is_critic_recurrent

    # ====================================================================
    #                     向量化 GAE（核心共享方法）
    # ====================================================================

    def _gae_vectorized(
        self,
        rewards: torch.Tensor,                          # (T, N)
        values: torch.Tensor,                           # (T, N) real scale
        dones: torch.Tensor,                            # (T, N)
        truncateds: Optional[torch.Tensor],             # (T, N) or None
        trunc_bootstrap: torch.Tensor,                  # (T, N) 预计算的 truncation bootstrap V
        active_mask: Optional[torch.Tensor] = None,     # (T, N) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        向量化 GAE，沿 env 维度 N 并行计算。

        支持两种模式：
        1. 标准 GAE（active_mask=None）
        2. A+ 透明 GAE（active_mask 存在时，inactive 步的奖励累积到前一个 active 步）

        Args:
            trunc_bootstrap: 每个 truncation 点的 V(s') bootstrap 值，
                             非 truncation 位置为 0。由调用方预计算。

        Returns:
            advantages (T*N,), returns (T*N,)
        """
        T, N = rewards.shape
        device = rewards.device

        # ---- 1. 构建 mask ----
        if truncateds is not None:
            trunc_mask = truncateds > 0.5
        else:
            trunc_mask = torch.zeros(T, N, dtype=torch.bool, device=device)

        done_mask = dones > 0.5
        term_mask = done_mask & ~trunc_mask

        # ---- 2. 最后一步 bootstrap ----
        last_step_bootstrap = values[-1].clone()  # (N,)

        # ---- 3. 反向循环 ----
        use_transparent = active_mask is not None
        active_bool = active_mask > 0.5 if use_transparent else None

        advantages = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)

        if not use_transparent:
            # ===== 标准 GAE =====
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_step_bootstrap.clone()
                else:
                    next_val = values[t + 1].clone()

                next_val = torch.where(term_mask[t], torch.zeros_like(next_val), next_val)
                if trunc_mask[t].any():
                    next_val = torch.where(trunc_mask[t], trunc_bootstrap[t], next_val)

                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = torch.where(
                    done_mask[t],
                    delta,
                    delta + self.gamma * self.gae_lambda * last_gae,
                )
                advantages[t] = last_gae
        else:
            # ===== A+ 透明 GAE =====
            next_active_val = last_step_bootstrap.clone()
            acc_reward = torch.zeros(N, device=device)
            acc_discount = torch.ones(N, device=device)
            acc_gae_decay = torch.ones(N, device=device)

            for t in reversed(range(T)):
                active_t = active_bool[t]

                # done 边界处理
                if done_mask[t].any():
                    done_val = torch.where(
                        term_mask[t],
                        torch.zeros_like(next_active_val),
                        torch.where(trunc_mask[t], trunc_bootstrap[t], next_active_val),
                    )
                    next_active_val = torch.where(done_mask[t], done_val, next_active_val)
                    last_gae = torch.where(done_mask[t], torch.zeros_like(last_gae), last_gae)
                    acc_reward = torch.where(done_mask[t], torch.zeros_like(acc_reward), acc_reward)
                    acc_discount = torch.where(done_mask[t], torch.ones_like(acc_discount), acc_discount)
                    acc_gae_decay = torch.where(done_mask[t], torch.ones_like(acc_gae_decay), acc_gae_decay)

                # active 步：用累积量计算多步 δ + GAE
                effective_next_val = acc_reward + acc_discount * next_active_val
                delta_active = rewards[t] + self.gamma * effective_next_val - values[t]
                gae_active = delta_active + self.gamma * self.gae_lambda * acc_gae_decay * last_gae

                # inactive 步：累积奖励和折扣
                new_acc_reward = rewards[t] + self.gamma * acc_reward
                new_acc_discount = self.gamma * acc_discount
                new_acc_gae_decay = self.gamma * self.gae_lambda * acc_gae_decay

                # 选择性更新
                last_gae = torch.where(active_t, gae_active, last_gae)
                advantages[t] = torch.where(active_t, gae_active, torch.zeros_like(gae_active))
                next_active_val = torch.where(active_t, values[t], next_active_val)
                acc_reward = torch.where(active_t, torch.zeros_like(acc_reward), new_acc_reward)
                acc_discount = torch.where(active_t, torch.ones_like(acc_discount), new_acc_discount)
                acc_gae_decay = torch.where(active_t, torch.ones_like(acc_gae_decay), new_acc_gae_decay)

        returns = advantages + values
        return advantages.view(-1), returns.view(-1)

    def _compute_trunc_bootstrap(
        self,
        truncateds: torch.Tensor,       # (T, N) or None
        final_global_states,             # T x N nested list, or None
        critic_input: torch.Tensor,      # (T*N, critic_dim) — 已构建好的 critic 输入
        T: int,
        N: int,
    ) -> torch.Tensor:
        """
        计算 truncation 处的 bootstrap V(s')。

        默认实现：直接用 final_global_state 过 critic。
        MAPPO override 此方法以添加 agent one-hot 拼接和 value denorm。

        Returns:
            trunc_bootstrap: (T, N) tensor，非 truncation 位置为 0
        """
        device = critic_input.device
        trunc_bootstrap = torch.zeros(T, N, device=device)

        if truncateds is None or final_global_states is None:
            return trunc_bootstrap

        trunc_mask = truncateds > 0.5
        if not trunc_mask.any():
            return trunc_bootstrap

        trunc_positions = trunc_mask.nonzero(as_tuple=False)  # (K, 2)

        batch_states = []
        valid_k_indices = []
        for k in range(len(trunc_positions)):
            t_idx = trunc_positions[k, 0].item()
            e_idx = trunc_positions[k, 1].item()
            fs = final_global_states[t_idx][e_idx] if final_global_states else None
            if fs is not None:
                batch_states.append(
                    torch.as_tensor(fs, dtype=torch.float32, device=device)
                )
                valid_k_indices.append(k)

        if batch_states:
            states_t = torch.stack(batch_states)

            if self.is_critic_recurrent:
                # Truncation = 新 episode 开始，hidden 归零
                zero_h = self.critic.get_initial_hidden(len(batch_states), device)
                v_raw, _ = self.critic(states_t, zero_h)
                v_raw = v_raw.squeeze(-1)
            else:
                v_raw = self.critic(states_t).squeeze(-1)

            if self.use_value_norm and self.ret_rms is not None:
                v_real = v_raw * self.ret_rms.std + self.ret_rms.mean
            else:
                v_real = v_raw

            for vi, k in enumerate(valid_k_indices):
                t_idx = trunc_positions[k, 0].item()
                e_idx = trunc_positions[k, 1].item()
                trunc_bootstrap[t_idx, e_idx] = v_real[vi]

        return trunc_bootstrap

    # ====================================================================
    #                     Batch 预处理（通用 GAE 预处理）
    # ====================================================================

    # ====================================================================
    #                  Critic RNN 序列处理辅助方法
    # ====================================================================

    def _critic_rnn_values(
        self,
        critic_input: torch.Tensor,    # (T*N, critic_dim)
        done_2d: torch.Tensor,          # (T, N)
        T: int,
        N: int,
        agent_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        逐步处理 RNN critic，在 done 边界重置 hidden。

        agent_key 不为 None 时使用 per-agent hidden（IPPO/PPO 多智能体场景），
        为 None 时使用共享 self._critic_hidden（向后兼容）。

        Returns:
            values_norm: (T, N) 归一化尺度的值
            all_hidden: (T, recurrent_N, N, H) 每步的 critic hidden（用于 chunk_split）
        """
        critic_seq = critic_input.view(T, N, -1)

        # 选择 hidden: per-agent 或 shared
        if agent_key is not None:
            if not hasattr(self, '_critic_hidden_per_agent'):
                self._critic_hidden_per_agent: Dict[str, torch.Tensor] = {}
            hidden = self._critic_hidden_per_agent.get(agent_key)
            if hidden is None or hidden.shape[1] != N:
                hidden = self.critic.get_initial_hidden(N, critic_input.device)
        else:
            if self._critic_hidden is None or self._critic_hidden.shape[1] != N:
                self._critic_hidden = self.critic.get_initial_hidden(N, critic_input.device)
            hidden = self._critic_hidden

        all_values = []
        all_hidden = []

        for t in range(T):
            if t > 0:
                done_prev = done_2d[t - 1] > 0.5      # (N,)
                if done_prev.any():
                    hidden = hidden.clone()
                    hidden[:, done_prev, :] = 0.0

            all_hidden.append(hidden)

            v_t, hidden = self.critic(critic_seq[t], hidden)    # (N, 1), (rN, N, H)
            all_values.append(v_t.squeeze(-1))                  # (N,)

        # 持久化最终 hidden
        if agent_key is not None:
            self._critic_hidden_per_agent[agent_key] = hidden.detach()
        else:
            self._critic_hidden = hidden.detach()

        values_norm = torch.stack(all_values, dim=0)            # (T, N)
        critic_rnn_hidden = torch.stack(all_hidden, dim=0)      # (T, rN, N, H)
        return values_norm, critic_rnn_hidden

    # ====================================================================
    #                     Batch 预处理（通用 GAE 预处理）
    # ====================================================================

    def prepare_batch(self, batch: RolloutBatch) -> RolloutBatch:
        """
        单 RolloutBatch 的 GAE 预处理。

        流程: reshape → compute values → trunc bootstrap → vectorized GAE → value norm → flatten
        RNN critic 时逐步处理序列并记录 hidden，供 update 中 chunk_split 使用。
        """
        final_gs = batch.final_global_state
        batch = batch.to_tensor(self.device)

        N = self.num_envs
        total_size = batch.obs.shape[0]
        T = total_size // N

        with torch.no_grad():
            critic_input = batch.global_state if batch.global_state is not None else batch.obs

            done_2d = batch.done.view(T, N)

            if self.is_critic_recurrent:
                values_norm, critic_rnn_h = self._critic_rnn_values(
                    critic_input, done_2d, T, N,
                )
                # (T, rN, N, H) -> (T*N, rN, H) 与 batch flat 布局对齐
                rN, H = critic_rnn_h.shape[1], critic_rnn_h.shape[3]
                batch.critic_rnn_hidden = critic_rnn_h.permute(0, 2, 1, 3).reshape(T * N, rN, H)
            else:
                values_norm = self.critic(critic_input).squeeze(-1).view(T, N)

            if self.use_value_norm and self.ret_rms is not None:
                values = values_norm * self.ret_rms.std + self.ret_rms.mean
            else:
                values = values_norm

            rew_2d = batch.rew.view(T, N)
            truncated_2d = batch.truncated.view(T, N) if batch.truncated is not None else None

            trunc_bootstrap = self._compute_trunc_bootstrap(
                truncated_2d, final_gs, critic_input, T, N,
            )

            active_mask_2d = None
            if self.use_active_mask and batch.active_mask is not None:
                active_mask_2d = batch.active_mask.view(T, N)

            adv, ret = self._gae_vectorized(
                rew_2d, values, done_2d, truncated_2d, trunc_bootstrap, active_mask_2d,
            )
            values_flat = values_norm.view(-1)

        # 更新 Value Normalization 统计量
        if self.use_value_norm and self.ret_rms is not None:
            if active_mask_2d is not None:
                active_flat = batch.active_mask > 0.5
                active_ret = ret[active_flat]
                if active_ret.numel() > 0:
                    self.ret_rms.update(active_ret)
            else:
                self.ret_rms.update(ret)

        batch.adv = adv
        batch.ret = ret
        batch.value = values_flat
        return batch



# ============================================================================
#                          Off-Policy Algorithm
# ============================================================================

class QLearningOffPolicyAlgo(BaseAlgorithm):
    """
    单智能体 off-policy Q-learning 基类 (D3QN 等)。

    提供：
    - 单优化器 + 梯度裁剪
    - target network 创建 / soft-hard update
    - 标准 update 流程 (compute_loss → backward → optimizer.step → target update)

    policy 预期为 ValuePolicy（内含 Q-network + epsilon-greedy）。
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

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.target_policy: Optional[RLBasePolicy] = None

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.policy, "is_recurrent", False)

    # ---- target network 管理 ----

    def _create_target_network(self):
        import copy
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.set_training_mode(False)
        for param in self.target_policy.parameters():
            param.requires_grad = False

    def _soft_update_target(self):
        if self.target_policy is None:
            return
        for tp, p in zip(self.target_policy.parameters(), self.policy.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _hard_update_target(self):
        if self.target_policy is not None:
            self.target_policy.load_state_dict(self.policy.state_dict())

    # ---- 标准 update 流程 ----

    def update(self, batch: BaseBatch, **kwargs) -> TrainingStats:
        """compute_loss → backward → clip → step → target update。

        子类若需在 update 前后增加逻辑（如 epsilon 衰减），
        应 override 此方法并调用 super().update()。

        注意：compute_loss 接收的 batch 已转为 tensor。
        """
        batch = batch.to_tensor(self.device)
        loss, stats = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stats.loss = float(loss.detach().item())
        if stats.extra:
            stats.extra = {
                k: float(v.detach().item()) if isinstance(v, torch.Tensor) else v
                for k, v in stats.extra.items()
            }

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            if self.tau < 1.0:
                self._soft_update_target()
            else:
                self._hard_update_target()
        return stats

    @abstractmethod
    def compute_loss(self, batch: BaseBatch) -> tuple[torch.Tensor, TrainingStats]:
        """计算 TD loss。batch 已在 update() 中转为 device tensor。"""
        pass
