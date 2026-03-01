"""D3QN: Double Dueling DQN for discrete action spaces.

继承 QLearningOffPolicyAlgo，使用 ValuePolicy（内含 Q-network + epsilon-greedy）。
epsilon 当前值存储在 ValuePolicy 中，衰减调度由本算法控制。

支持 MLP (QMLP) 和 RNN (QRNN) 两种 Q-network：
- MLP: 标准 flat TransitionBatch 训练
- RNN: SequenceBatch 序列训练（DRQN 风格，h0=zeros）
"""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from algorithms.algorithm_base import QLearningOffPolicyAlgo, TrainingStats
from configs.algo_configs import D3QNParams
from policies.rl.rl_base import ValuePolicy
from data.batch import BaseBatch, TransitionBatch, SequenceBatch


class D3QNAlgo(QLearningOffPolicyAlgo):
    """
    Double Dueling DQN。

    - Double DQN: main net 选动作, target net 评估
    - Dueling: V/A 分离在网络架构层实现（不影响本类逻辑）
    - Epsilon-greedy: 委托给 ValuePolicy.forward()
    - Target network: 继承自基类的 soft/hard update
    - RNN: forward_sequence + zero hidden init (DRQN 风格)
    """

    def __init__(
        self,
        policy: ValuePolicy,
        params: D3QNParams,
    ):
        super().__init__(
            policy=policy,
            lr=params.lr,
            gamma=params.gamma,
            tau=params.tau,
            target_update_freq=params.target_update_freq,
            max_grad_norm=params.max_grad_norm,
        )
        self.params = params
        self.seq_len = params.seq_len
        self._create_target_network()

        # epsilon 衰减参数（调度逻辑在算法中，当前值同步到 policy）
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay = params.epsilon_decay
        self.exploration_fraction = params.exploration_fraction
        self.epsilon_decay_by_step = params.epsilon_decay_by_step
        self._current_epsilon = params.epsilon_start
        self.policy.set_epsilon(self._current_epsilon)

    # ====================================================================
    #                           compute_loss
    # ====================================================================

    def compute_loss(
        self, batch: BaseBatch,
    ) -> tuple[torch.Tensor, TrainingStats]:
        if self.is_recurrent:
            return self._compute_loss_sequence(batch)
        return self._compute_loss_flat(batch)

    def _compute_loss_flat(
        self, batch: TransitionBatch,
    ) -> tuple[torch.Tensor, TrainingStats]:
        """MLP 路径：标准 DQN TD loss。"""
        obs = batch.obs
        actions = batch.act.long()
        rewards = batch.rew
        next_obs = batch.next_obs
        dones = batch.done
        next_action_mask = getattr(batch, 'next_action_mask', None)

        q_values = self.policy.q_network(obs)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.params.use_double_dqn:
                next_q_main = self.policy.compute_q_values(next_obs, next_action_mask)
                next_actions = next_q_main.argmax(dim=1, keepdim=True)
                next_q_target = self.target_policy.q_network(next_obs)
                q_next = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q_target = self.target_policy.compute_q_values(
                    next_obs, next_action_mask,
                )
                q_next = next_q_target.max(dim=1)[0]

            td_target = rewards + self.gamma * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q_current, td_target)

        with torch.no_grad():
            stats = TrainingStats(
                loss=loss.item(),
                extra={
                    "epsilon": self._current_epsilon,
                    "q_mean": q_values.mean().item(),
                    "q_max": q_values.max().item(),
                    "td_error": (td_target - q_current).abs().mean().item(),
                },
            )
        return loss, stats

    def _compute_loss_sequence(
        self, batch: SequenceBatch,
    ) -> tuple[torch.Tensor, TrainingStats]:
        """RNN 路径：DRQN 序列 TD loss，h0=zeros，支持 R2D2 burn-in。

        forward 全序列预热 hidden state，loss 仅在 [burn_in_len:] 训练区间计算。
        burn_in_len=0 时行为与无 burn-in 完全一致。
        """
        B, T_total = batch.obs.shape[:2]
        device = batch.obs.device
        bi = getattr(batch, 'burn_in_len', 0)

        obs_seq = batch.obs.transpose(0, 1)            # (T, B, D)
        next_obs_seq = batch.next_obs.transpose(0, 1)  # (T, B, D)
        actions = batch.act.long()                      # (B, T)
        rewards = batch.rew                             # (B, T)
        dones = batch.done                              # (B, T)
        mask = batch.mask                               # (B, T)

        h0 = self.policy.q_network.get_initial_hidden(B, device)

        q_full, _ = self.policy.q_network.forward_sequence(obs_seq, h0)  # (T, B, act_dim)
        q_train = q_full[bi:]  # (S, B, act_dim)
        act_train = actions[:, bi:].T.unsqueeze(-1)  # (S, B, 1)
        q_current = q_train.gather(2, act_train).squeeze(-1)  # (S, B)

        with torch.no_grad():
            h0_target = self.target_policy.q_network.get_initial_hidden(B, device)

            if self.params.use_double_dqn:
                next_q_online_full, _ = self.policy.q_network.forward_sequence(next_obs_seq, h0)
                next_q_online = next_q_online_full[bi:]
                next_am = getattr(batch, 'next_action_mask', None)
                if next_am is not None:
                    next_am_train = next_am[:, bi:].transpose(0, 1)
                    mask_t = next_am_train.bool() if next_am_train.dtype != torch.bool else next_am_train
                    next_q_online = next_q_online.masked_fill(~mask_t, float("-inf"))
                next_actions = next_q_online.argmax(dim=-1, keepdim=True)
                next_q_target_full, _ = self.target_policy.q_network.forward_sequence(next_obs_seq, h0_target)
                q_next = next_q_target_full[bi:].gather(2, next_actions).squeeze(-1)
            else:
                next_q_target_full, _ = self.target_policy.q_network.forward_sequence(next_obs_seq, h0_target)
                next_q_target = next_q_target_full[bi:]
                next_am = getattr(batch, 'next_action_mask', None)
                if next_am is not None:
                    next_am_train = next_am[:, bi:].transpose(0, 1)
                    mask_t = next_am_train.bool() if next_am_train.dtype != torch.bool else next_am_train
                    next_q_target = next_q_target.masked_fill(~mask_t, float("-inf"))
                q_next = next_q_target.max(dim=-1)[0]

            rew_train = rewards[:, bi:].T
            done_train = dones[:, bi:].T
            td_target = rew_train + self.gamma * (1.0 - done_train) * q_next

        td_loss = F.smooth_l1_loss(q_current, td_target, reduction='none')
        mask_train = mask[:, bi:].transpose(0, 1)
        loss = (td_loss * mask_train).sum() / mask_train.sum().clamp(min=1)

        with torch.no_grad():
            stats = TrainingStats(
                loss=loss.item(),
                extra={
                    "epsilon": self._current_epsilon,
                    "q_mean": q_train.mean().item(),
                    "q_max": q_train.max().item(),
                    "td_error": ((td_target - q_current).abs() * mask_train).sum().item()
                    / mask_train.sum().clamp(min=1).item(),
                },
            )
        return loss, stats

    # ====================================================================
    #                       update (扩展基类)
    # ====================================================================

    def update(self, batch: BaseBatch, **kwargs) -> TrainingStats:
        """基类 update + epsilon 衰减（同步到 ValuePolicy）。"""
        stats = super().update(batch, **kwargs)

        if not self.epsilon_decay_by_step:
            self._decay_epsilon_exp()

        return stats

    # ====================================================================
    #                       Epsilon 衰减调度
    # ====================================================================

    def _decay_epsilon_exp(self):
        """指数衰减（per update）。"""
        if self._current_epsilon > self.epsilon_end:
            self._current_epsilon = max(
                self._current_epsilon * self.epsilon_decay, self.epsilon_end,
            )
            self.policy.set_epsilon(self._current_epsilon)

    def update_epsilon_by_step(self, step: int, total_steps: int):
        """线性衰减（per env step），需由外部 Trainer 调用。"""
        decay_steps = int(total_steps * self.exploration_fraction)
        if step < decay_steps:
            progress = step / decay_steps
            self._current_epsilon = self.epsilon_start - progress * (
                self.epsilon_start - self.epsilon_end
            )
        else:
            self._current_epsilon = self.epsilon_end
        self.policy.set_epsilon(self._current_epsilon)

    def set_epsilon(self, epsilon: float):
        self._current_epsilon = max(0.0, min(epsilon, 1.0))
        self.policy.set_epsilon(self._current_epsilon)

    def get_epsilon(self) -> float:
        return self._current_epsilon


DQNAlgo = D3QNAlgo
