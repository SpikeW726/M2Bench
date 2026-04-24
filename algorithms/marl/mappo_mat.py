"""MAPPOMATAlgo — 严格遵循 asy_ppo.py 源码的 MAT PPO 算法。

与框架现有 MAPPOAlgo 的关键差异:
- 缓冲: 只存 READY 步（决策步缓冲），由 MATOnPolicyCollector 完成
- 回报: MC return（n-step 反向累积），不使用 GAE-lambda
- Value: encoder 输出对所有 agents 取均值 → 单标量/步
- Advantage: 单标量/步（所有 agents 共享），无 per-agent 区分
- Policy loss: per-agent 独立 ratio，对所有 (T, N) 平均，与 asy_ppo.py 完全对应
- 无 active_mask loss masking（buffer 本身只含决策步，无需 masking）
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from algorithms.algorithm_base import BaseAlgorithm, TrainingStats
from policies.marl.mat_policy import MATMultiAgentPolicy


@dataclass
class MATBatch:
    """MATOnPolicyCollector 收集到的决策步 batch。

    所有字段维度均为 (T,) 或 (T, ...) 其中 T = 决策步数。
    """
    graph_state:  np.ndarray   # (T, N, G, 3)
    actions:      np.ndarray   # (T, N) int — graph index
    log_probs:    np.ndarray   # (T, N, G) — 完整 log 分布（供 teacher forcing）
    rewards:      np.ndarray   # (T,) — cumulated change_reward（决策步间累积）
    shift_action: np.ndarray   # (T, N, N, 2)
    node_last_idx: np.ndarray  # (T, N) int — graph index
    active_mask:  np.ndarray   # (T, N) — 记录哪些 agent 在该步 READY（debug 用）
    # 最后一步的 next state，供 MC return bootstrap
    last_graph_state: Optional[np.ndarray] = None  # (N, G, 3) 或 None
    last_node_idx:    Optional[np.ndarray] = None  # (N,) int
    last_active_mask: Optional[np.ndarray] = None  # (N,)
    last_shift:       Optional[np.ndarray] = None  # (N, N, 2)


class MAPPOMATAlgo(BaseAlgorithm):
    """MAT PPO 算法，严格遵循 asy_ppo.py 实现。

    所有超参在 MAPPOMATParams（configs/algo_configs.py）中定义。
    """

    def __init__(
        self,
        policy: MATMultiAgentPolicy,
        params,
        n_agents: int = 1,
        **kwargs,   # 忽略 create_algorithm 传来的多余 context_kwargs
    ):
        super().__init__(policy)
        self.policy: MATMultiAgentPolicy = policy
        self.n_agents = n_agents

        self.gamma = params.gamma
        self.clip_coef = params.clip_coef
        self.ent_coef = params.ent_coef
        self.vf_coef = params.vf_coef
        self.policy_update_steps = params.policy_update_steps
        self.max_grad_norm = getattr(params, "max_grad_norm", 0.5)

        # Adam + ReduceLROnPlateau（与 asy_ppo.py 一致）
        self.optimizer = torch.optim.Adam(
            policy.parameters_to_optimize(),
            lr=params.lr,
            eps=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=getattr(params, "factor", 0.5),
            patience=getattr(params, "patience", 100),
        )
        self._episode_reward_buf = []  # 供 scheduler.step() 使用

    # -------------------------------------------------------------------------
    #  prepare_batch: MC return 计算（与 asy_ppo.py gae() 对应）
    # -------------------------------------------------------------------------

    def prepare_batch(self, batch: MATBatch) -> MATBatch:
        """计算 MC return 并填入 batch.rewards（原地替换为 G_estimate）。

        对应 asy_ppo.py gae() 方法:
            1. 用最后一步 state 估算 next_value
            2. 反向累积 G_t = gamma * G_{t+1} + r_t
            3. 归一化: G = (G - mean) / (std + 1e-8)
        """
        device = self.device
        policy = self.policy

        # --- 1. bootstrap next_value ---
        if batch.last_graph_state is not None:
            last_state = torch.as_tensor(
                batch.last_graph_state[None], dtype=torch.float32, device=device
            )  # (1, N, G, 3)
            with torch.no_grad():
                _, sv = policy.encoder(last_state)   # (1, N, 1)
            next_value = float(sv.mean().cpu().item())
        else:
            next_value = 0.0

        # --- 2. 反向累积 MC return ---
        T = len(batch.rewards)
        G_list = []
        for r in reversed(batch.rewards):
            next_value = self.gamma * next_value + float(r)
            G_list.append(next_value)
        G_list.reverse()
        G = np.array(G_list, dtype=np.float32)

        # --- 3. 归一化 ---
        G = (G - G.mean()) / (G.std() + 1e-8)

        batch.rewards = G  # (T,) 归一化后的 MC return
        return batch

    # -------------------------------------------------------------------------
    #  update: per-agent ratio PPO（与 asy_ppo.py update() 对应）
    # -------------------------------------------------------------------------

    def update(self, batch: MATBatch, **kwargs) -> TrainingStats:
        """严格遵循 asy_ppo.py update() 的 policy_update_steps 内层循环。

        kwargs (minibatch_size, update_epochs) 来自 OnPolicyTrainer，忽略，
        使用 params.policy_update_steps 代替。
        """
        device = self.device
        policy = self.policy

        T = len(batch.rewards)
        N = self.n_agents
        norm_factor = T * N

        returns = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)  # (T,)
        graph_state = torch.as_tensor(batch.graph_state, dtype=torch.float32, device=device)  # (T,N,G,3)
        shift_action = torch.as_tensor(batch.shift_action, dtype=torch.float32, device=device)  # (T,N,N,2)
        actions = batch.actions   # (T, N) int, numpy
        node_last_idx = batch.node_last_idx  # (T, N) int, numpy
        active_mask = batch.active_mask      # (T, N) 0/1, numpy

        # --- 旧 log_prob（存储时的完整分布）---
        old_log_prob = torch.as_tensor(
            batch.log_probs, dtype=torch.float32, device=device
        )  # (T, N, G)
        # 索引到对应动作的 log_prob: old_log_prob_act[t, k] = old_log_prob[t, k, action[t,k]]
        act_t = torch.as_tensor(actions, dtype=torch.long, device=device)  # (T, N)
        old_log_prob_act = old_log_prob.gather(
            dim=2, index=act_t.unsqueeze(-1)
        ).squeeze(-1)  # (T, N)
        old_log_prob_act = old_log_prob_act.detach()

        # --- 用旧网络计算 value 基线（仅用于 advantage，不用于 clip）---
        with torch.no_grad():
            _, sv_old = policy.encoder(graph_state)  # (T, N, 1)
        sv_old_mean = sv_old.mean(dim=1).squeeze(-1).detach()  # (T,)

        advantage = returns - sv_old_mean  # (T,)  共享标量

        # 记录用于 scheduler
        episode_reward = float(returns.sum().item())
        self._episode_reward_buf.append(episode_reward)

        # ---- policy_update_steps 内层循环 ----
        loss_last = policy_loss_last = value_loss_last = entropy_last = 0.0
        mse = nn.MSELoss()

        for _ in range(self.policy_update_steps):
            # 重新前向传播
            new_log_prob_full, sv_new, entropy_full = policy.evaluate_joint_actions(
                graph_state, node_last_idx, shift_action, active_mask
            )  # (T,N,G), (T,N,1), (T,N,1)

            sv_new_mean = sv_new.mean(dim=1).squeeze(-1)  # (T,)

            # 新 log_prob of chosen action
            new_log_prob_act = new_log_prob_full.gather(
                dim=2, index=act_t.unsqueeze(-1)
            ).squeeze(-1)  # (T, N)

            # --- policy loss (per-agent ratio, 对 T*N 平均，与 asy_ppo.py 完全对应）---
            policy_loss = torch.tensor(0.0, device=device)
            for t in range(T):
                ad = advantage[t]  # 共享标量
                for k in range(N):
                    lp_new = new_log_prob_act[t, k]
                    lp_old = old_log_prob_act[t, k]
                    ratio = (lp_new - lp_old).exp()
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    # asy_ppo.py 用 torch.max(-ad*ratio, -ad*clipped_ratio)
                    policy_loss += torch.max(
                        -ad * ratio,
                        -ad * clipped_ratio,
                    ) / norm_factor

            # --- value loss (clipped, 与 asy_ppo.py 对应）---
            sv_old_for_clip = sv_old_mean.detach()
            value_unclipped = mse(sv_new_mean, returns)
            sv_new_clipped = sv_old_for_clip + torch.clamp(
                sv_new_mean - sv_old_for_clip, -0.2, 0.2
            )
            value_clipped = mse(sv_new_clipped, returns)
            value_loss = torch.max(value_clipped, value_unclipped)

            # --- entropy loss ---
            # new_log_prob_full: (T, N, G)；对应 Categorical(logits=log_prob_new).entropy()
            entropy_loss = Categorical(logits=new_log_prob_full).entropy().mean()

            loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy.parameters_to_optimize(), self.max_grad_norm
            )
            self.optimizer.step()

            loss_last = float(loss.item())
            policy_loss_last = float(policy_loss.item())
            value_loss_last = float(value_loss.item())
            entropy_last = float(entropy_loss.item())

        # LR scheduler（使用最近一批 episode reward 均值）
        if self._episode_reward_buf:
            self.scheduler.step(np.mean(self._episode_reward_buf))
            self._episode_reward_buf.clear()

        return TrainingStats(
            loss=loss_last,
            policy_loss=policy_loss_last,
            value_loss=value_loss_last,
            entropy=entropy_last,
            extra={"lr": self.optimizer.param_groups[0]["lr"]},
        )

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)
