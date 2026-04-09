"""VDPPO: Value-Decomposition Multi-Agent PPO (Ma & Luo, 2022).

PPO actor + QPLEX-style Q-decomposition + Double DQN TD targets + 双优化器。

Q-network 输入设计（与论文对齐）：
  - MLP 路径 (q_type=mlp): concat(global_state, one_hot_i)
  - RNN 路径 (q_type=rnn): concat(global_state, one_hot_i, prev_action_one_hot_i)，
    通过 forward_sequence 处理时序，hidden state h_0=zeros（每轮 rollout 重置）

局部观测 o_i **不**由算法层拼接。若某环境的局部观测与全局状态差异显著，
应在环境的 _get_global_state() / state() 函数中自行将 o_i 拼入 s，算法层不做处理。

Advantage 计算（与论文对齐）：
  A_i = Q_i(τ_i, u_i) - V_i(τ_i)，其中 V_i = max_a Q_i(τ_i, a)
  直接使用 Q-network 输出，不经过 GAE。
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import TrainingStats
from algorithms.rl.ppo import PPOBase
from configs.algo_configs import VDPPOParams
from data.batch import RolloutBatch
from networks.mixing import QPLEXMixer
from networks.rnn import QRNN
from policies.marl.marl_base import MultiAgentPolicy


class VDPPOAlgo(PPOBase):
    """
    VDPPO: PPO actor 更新 + QPLEX 值分解 + 双优化器。

    继承 PPOBase 获取 clip_range / clip_vloss / target_kl 等超参，
    完全 override prepare_batch (Q-decomposition, 直接 Q-advantage) 和 update (双优化器)。
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: VDPPOParams,
        num_envs: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        critic: Optional[nn.Module] = None,
        total_iterations: Optional[int] = None,
        optimizer_steps_per_iter: Optional[int] = None,
        value_norm_config: Optional[Dict] = None,
        q_network: Optional[nn.Module] = None,
    ):
        super().__init__(policy, None, params, num_envs, value_norm_config=value_norm_config)
        self.params = params
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Q-network 由外部工厂传入，若未提供则回退创建（MLP，input = state + one_hot）
        if q_network is not None:
            self.q_network = q_network.to(self.device)
        else:
            from networks.mlp import QMLP
            q_input_dim = state_dim + n_agents
            self.q_network = QMLP(q_input_dim, [64, 64], action_dim).to(self.device)

        self.is_q_recurrent = isinstance(self.q_network, QRNN)

        self.target_q_network = copy.deepcopy(self.q_network)
        for p in self.target_q_network.parameters():
            p.requires_grad = False

        # ---- Mixing network ----
        self.mixing_net = QPLEXMixer(
            n_agents, state_dim, params.mixer_embed_dim,
        ).to(self.device)

        self.target_mixing_net = copy.deepcopy(self.mixing_net)
        for p in self.target_mixing_net.parameters():
            p.requires_grad = False

        # ---- 双优化器 ----
        self.actor_optimizer = torch.optim.Adam(
            policy.parameters(), lr=params.actor_lr,
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixing_net.parameters()),
            lr=params.q_lr,
        )

        # ---- Target 网络更新: tau < 1.0 → soft, tau >= 1.0 → hard ----
        self.tau = params.tau
        self.target_update_freq = params.target_update_freq
        self.q_clip_range = params.q_clip_range
        self._update_count = 0

        # ---- LR schedulers ----
        self.actor_scheduler = None
        self.q_scheduler = None
        if params.use_lr_scheduler and total_iterations and optimizer_steps_per_iter:
            actor_decay = int(
                total_iterations * params.actor_lr_decay_ratio * optimizer_steps_per_iter
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_optimizer,
                start_factor=params.actor_lr_start_factor,
                end_factor=params.actor_lr_end_factor,
                total_iters=actor_decay,
            )
            q_decay = int(
                total_iterations * params.q_lr_decay_ratio * optimizer_steps_per_iter
            )
            self.q_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.q_optimizer,
                start_factor=params.q_lr_start_factor,
                end_factor=params.q_lr_end_factor,
                total_iters=q_decay,
            )

        # Q-network 热身期：前 N 次 update 冻结 actor
        self.freeze_actor_iters = params.freeze_actor_iters

        # prepare_batch 设置，update 消费
        self._q_data: Optional[Dict] = None

    # ====================================================================
    #                     Batch 预处理（Q-Decomposition）
    # ====================================================================

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """
        完全 override 基类:

        Actor 部分: A_i = Q_i(u_i) - V_i（直接 Q-advantage，无 GAE）
        Q 部分: Q-decomposition TD target（Q_tot → y_tot，用于 Q-network 更新）
        """
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs

        # ---- 公共字段（agent 间相同） ----
        first_agent = agents[0]
        final_gs = batch_dict[first_agent].final_global_state
        boundary_gs = batch_dict[first_agent].boundary_global_state  # (N, state_dim) or None
        ref = batch_dict[first_agent].to_tensor(self.device)
        total_size = ref.obs.shape[0]
        T = total_size // N

        global_state = ref.global_state                             # (T*N, state_dim)
        state_2d = global_state.view(T, N, -1)                     # (T, N, state_dim)
        done_2d = ref.done.view(T, N)

        if ref.truncated is not None:
            trunc_2d = ref.truncated.view(T, N)
            trunc_bool = trunc_2d > 0.5
            term_bool = (done_2d > 0.5) & ~trunc_bool
        else:
            trunc_2d = None
            trunc_bool = torch.zeros(T, N, dtype=torch.bool, device=self.device)
            term_bool = done_2d > 0.5

        # ---- 收集 per-agent 数据 ----
        all_obs, all_act, all_log_prob = [], [], []
        all_adv, all_ret, all_value = [], [], []
        all_action_mask, all_active_mask = [], []
        all_rnn_hidden = []
        per_agent_act: List[torch.Tensor] = []
        per_agent_rew: List[torch.Tensor] = []

        for agent in agents:
            b = batch_dict[agent].to_tensor(self.device)
            all_obs.append(b.obs)
            all_act.append(b.act)
            all_log_prob.append(b.log_prob)
            per_agent_act.append(b.act.long())
            per_agent_rew.append(b.rew)
            if b.action_mask is not None:
                all_action_mask.append(b.action_mask)
            if b.active_mask is not None:
                all_active_mask.append(b.active_mask)
            if b.rnn_hidden is not None:
                all_rnn_hidden.append(b.rnn_hidden)

        joint_actions = torch.stack(per_agent_act, dim=-1)          # (T*N, n_agents)

        # ---- 团队奖励 r_tot ----
        if self.params.reward_global:
            r_tot = per_agent_rew[0]                                 # 所有 agent reward 相同，取 agent_0
        else:
            r_tot = torch.stack(per_agent_rew, dim=-1).sum(dim=-1)  # 各 agent reward 求和

        with torch.no_grad():
            # ---- 构建 next_states（截断 episode 用 final_gs 替换，rollout 边界用 boundary_gs） ----
            next_states = torch.zeros_like(state_2d)                 # (T, N, state_dim)
            next_states[:-1] = state_2d[1:]

            # rollout 末尾：非 terminal/truncated 步的 next_state 应为下一次 rollout 的初始 state
            if boundary_gs is not None:
                boundary_t = torch.as_tensor(
                    boundary_gs, dtype=torch.float32, device=self.device,
                )  # (N, state_dim)
                # 仅对 rollout 末尾非 done 步使用 boundary state；terminal/truncated 步会在下方覆盖
                not_done_last = ~(done_2d[-1] > 0.5)                 # (N,)
                next_states[-1, not_done_last] = boundary_t[not_done_last]
                # done 步（terminal 或 truncated）：next_state 先设为 state_2d[-1]，
                # 下方的 terminal bootstrap=0 和 final_gs 替换会处理正确值
                done_last = done_2d[-1] > 0.5
                next_states[-1, done_last] = state_2d[-1, done_last]
            else:
                # 无 boundary_gs（不支持 global_state 的环境）：回退到当前 state 近似
                next_states[-1] = state_2d[-1]

            if final_gs is not None:
                for pos in trunc_bool.nonzero(as_tuple=False):
                    t_idx, e_idx = pos[0].item(), pos[1].item()
                    fs = final_gs[t_idx][e_idx]
                    if fs is not None:
                        next_states[t_idx, e_idx] = torch.as_tensor(
                            fs, dtype=torch.float32, device=self.device,
                        )

            if self.is_q_recurrent:
                per_agent_v, per_agent_qplex_a, old_q_tot, y_tot_flat, per_agent_prev_act_2d = \
                    self._eval_q_rnn(
                        state_2d, next_states, per_agent_act,
                        done_2d, term_bool, r_tot, T, N, num_agents,
                    )
            else:
                per_agent_v, per_agent_qplex_a, old_q_tot, y_tot_flat = \
                    self._eval_q_mlp(
                        global_state, next_states.view(-1, self.state_dim),
                        per_agent_act, term_bool, r_tot, T, N, num_agents,
                    )

        # ---- Advantage = A_i（直接 Q-advantage，无 GAE） ----
        for i in range(num_agents):
            adv_i = per_agent_qplex_a[i]                            # (T*N,)
            ret_i = per_agent_v[i] + adv_i                          # Q_i(u_i)，仅用于 batch 结构兼容
            all_adv.append(adv_i)
            all_ret.append(ret_i)
            all_value.append(per_agent_v[i])

        # ---- 存储 Q-update 辅助数据 ----
        if self.is_q_recurrent:
            self._q_data = {
                "is_rnn": True,
                "state_seq": state_2d,                               # (T, N, state_dim)
                "joint_actions_seq": joint_actions.view(T, N, -1),  # (T, N, n_agents)
                "prev_act_seqs": per_agent_prev_act_2d,             # list of (T, N) int tensors
                "y_tot": y_tot_flat,                                 # (T*N,)
                "old_q_tot": old_q_tot,                              # (T*N,)
                "T_q": T, "N_q": N,
            }
        else:
            self._q_data = {
                "is_rnn": False,
                "states": global_state,                              # (T*N, state_dim)
                "joint_actions": joint_actions,                      # (T*N, n_agents)
                "y_tot": y_tot_flat,                                 # (T*N,)
                "old_q_tot": old_q_tot,                              # (T*N,)
            }

        # ---- 记录 RNN chunk_split 所需的布局参数 ----
        self._last_T = T
        self._last_N = N
        self._last_num_agents = num_agents

        # ---- 合并为 actor-update 用 RolloutBatch（同 MAPPO 结构） ----
        all_critic_input = []
        for i in range(num_agents):
            one_hot = torch.zeros(total_size, num_agents, device=self.device)
            one_hot[:, i] = 1.0
            all_critic_input.append(torch.cat([global_state, one_hot], dim=-1))

        return RolloutBatch(
            obs=torch.cat(all_obs, dim=0),
            act=torch.cat(all_act, dim=0),
            log_prob=torch.cat(all_log_prob, dim=0),
            adv=torch.cat(all_adv, dim=0),
            ret=torch.cat(all_ret, dim=0),
            value=torch.cat(all_value, dim=0),
            global_state=torch.cat(all_critic_input, dim=0),
            action_mask=torch.cat(all_action_mask, dim=0) if all_action_mask else None,
            active_mask=torch.cat(all_active_mask, dim=0) if all_active_mask else None,
            rnn_hidden=torch.cat(all_rnn_hidden, dim=0) if all_rnn_hidden else None,
        )

    # ====================================================================
    #                     Q 评估：MLP 路径
    # ====================================================================

    def _eval_q_mlp(
        self,
        global_state: torch.Tensor,           # (T*N, state_dim)
        next_flat: torch.Tensor,              # (T*N, state_dim)
        per_agent_act: List[torch.Tensor],    # list of (T*N,) int
        term_bool: torch.Tensor,              # (T, N) bool
        r_tot: torch.Tensor,                  # (T*N,)
        T: int, N: int, num_agents: int,
    ):
        """MLP Q-network 的 prepare 逻辑：flat forward，无 prev_action。"""
        total_size = T * N
        per_agent_v, per_agent_qplex_a = [], []

        # ---- Per-agent V_i = max Q_i，A_i = Q_i(u_i) - V_i ----
        for i in range(num_agents):
            one_hot = torch.zeros(total_size, num_agents, device=self.device)
            one_hot[:, i] = 1.0
            q_in = torch.cat([global_state, one_hot], dim=-1)
            Q_i = self.q_network(q_in)                               # (T*N, action_dim)
            V_i = Q_i.max(dim=-1).values
            A_i = Q_i.gather(1, per_agent_act[i].unsqueeze(-1)).squeeze(-1) - V_i
            per_agent_v.append(V_i)
            per_agent_qplex_a.append(A_i)

        V_vals = torch.stack(per_agent_v, dim=-1)                    # (T*N, n_agents)
        A_vals = torch.stack(per_agent_qplex_a, dim=-1)
        old_q_tot = self.mixing_net(V_vals, A_vals, global_state)    # (T*N,)

        # ---- Double DQN: next-state 评估 ----
        target_v_next = []
        for i in range(num_agents):
            one_hot = torch.zeros(total_size, num_agents, device=self.device)
            one_hot[:, i] = 1.0
            q_in = torch.cat([next_flat, one_hot], dim=-1)
            greedy = self.q_network(q_in).argmax(dim=-1)
            Q_tgt = self.target_q_network(q_in)
            V_tgt_i = Q_tgt.gather(1, greedy.unsqueeze(-1)).squeeze(-1)
            target_v_next.append(V_tgt_i)

        V_tgt_all = torch.stack(target_v_next, dim=-1)               # (T*N, n_agents)
        A_tgt_all = torch.zeros_like(V_tgt_all)
        Q_tot_next = self.target_mixing_net(V_tgt_all, A_tgt_all, next_flat)  # (T*N,)

        # ---- y_tot = r_tot + γ * Q_tot_next（terminal 处 bootstrap = 0） ----
        rew_2d = r_tot.view(T, N)
        bootstrap = self.gamma * Q_tot_next.view(T, N)
        bootstrap[term_bool] = 0.0
        y_tot_flat = (rew_2d + bootstrap).view(-1)

        return per_agent_v, per_agent_qplex_a, old_q_tot, y_tot_flat

    # ====================================================================
    #                     Q 评估：RNN 路径
    # ====================================================================

    def _build_prev_act_seqs(
        self,
        per_agent_act: List[torch.Tensor],    # list of (T*N,) int
        done_2d: torch.Tensor,                # (T, N)
        T: int, N: int,
    ) -> List[torch.Tensor]:
        """构建每个 agent 的 prev_action 序列 (T, N)，在 done 边界重置为 0。"""
        seqs = []
        for i in range(len(per_agent_act)):
            act_2d = per_agent_act[i].view(T, N)                     # (T, N)
            prev_act = torch.zeros_like(act_2d)                      # t=0 置零
            prev_act[1:] = act_2d[:-1]
            # done 边界：若上一步 done，则当前步 prev_action 重置
            reset_mask = done_2d[:-1] > 0.5                          # (T-1, N)
            prev_act[1:][reset_mask] = 0
            seqs.append(prev_act)
        return seqs

    def _build_rnn_q_inputs(
        self,
        state_seq: torch.Tensor,              # (T, N, state_dim)
        prev_act_seqs: List[torch.Tensor],    # list of (T, N) int
        num_agents: int, T: int, N: int,
    ) -> torch.Tensor:
        """
        拼接所有 agent 的 RNN Q-network 输入，沿 batch 维合并。
        Returns: (T, N*n_agents, state_dim + n_agents + action_dim)
        """
        q_inputs = []
        for i in range(num_agents):
            one_hot = torch.zeros(T, N, num_agents, device=self.device)
            one_hot[:, :, i] = 1.0
            prev_act_oh = F.one_hot(prev_act_seqs[i], num_classes=self.action_dim).float()
            q_in_i = torch.cat([state_seq, one_hot, prev_act_oh], dim=-1)
            q_inputs.append(q_in_i)
        return torch.cat(q_inputs, dim=1)                            # (T, N*n_agents, input_dim)

    def _run_rnn_q(
        self,
        q_net: nn.Module,
        state_seq: torch.Tensor,              # (T, N, state_dim)
        prev_act_seqs: List[torch.Tensor],    # list of (T, N) int
        per_agent_act: List[torch.Tensor],    # list of (T*N,) int，用于提取 Q(u_i)
        num_agents: int, T: int, N: int,
    ):
        """
        对 RNN Q-network 做整序列前向。
        Returns:
            per_agent_v:  list of (T*N,) float  — V_i = max Q_i
            per_agent_a:  list of (T*N,) float  — A_i = Q_i(u_i) - V_i
        """
        q_in_all = self._build_rnn_q_inputs(state_seq, prev_act_seqs, num_agents, T, N)
        h0 = q_net.get_initial_hidden(N * num_agents, self.device)
        q_vals_all, _ = q_net.forward_sequence(q_in_all, h0)        # (T, N*n_agents, action_dim)

        # 分回 per-agent：chunk 沿 dim=1 切分
        q_vals_agents = q_vals_all.chunk(num_agents, dim=1)          # list of (T, N, action_dim)

        per_agent_v, per_agent_a = [], []
        for i, qv in enumerate(q_vals_agents):
            V_i = qv.max(dim=-1).values.view(-1)                     # (T*N,)
            act_2d = per_agent_act[i].view(T, N)
            A_i = qv.gather(2, act_2d.unsqueeze(-1)).squeeze(-1).view(-1) - V_i
            per_agent_v.append(V_i)
            per_agent_a.append(A_i)

        return per_agent_v, per_agent_a

    def _eval_q_rnn(
        self,
        state_2d: torch.Tensor,              # (T, N, state_dim)
        next_states: torch.Tensor,           # (T, N, state_dim)
        per_agent_act: List[torch.Tensor],   # list of (T*N,) int
        done_2d: torch.Tensor,              # (T, N)
        term_bool: torch.Tensor,            # (T, N) bool
        r_tot: torch.Tensor,                # (T*N,)
        T: int, N: int, num_agents: int,
    ):
        """RNN Q-network 的 prepare 逻辑：sequence forward，包含 prev_action。"""
        global_state = state_2d.view(-1, self.state_dim)            # (T*N, state_dim)

        # ---- 构建 prev_action 序列 ----
        prev_act_seqs = self._build_prev_act_seqs(per_agent_act, done_2d, T, N)

        # ---- online Q-network：评估当前 state ----
        per_agent_v, per_agent_qplex_a = self._run_rnn_q(
            self.q_network, state_2d, prev_act_seqs, per_agent_act, num_agents, T, N,
        )

        V_vals = torch.stack(per_agent_v, dim=-1)                    # (T*N, n_agents)
        A_vals = torch.stack(per_agent_qplex_a, dim=-1)
        old_q_tot = self.mixing_net(V_vals, A_vals, global_state)    # (T*N,)

        # ---- Double DQN：next-state 评估 ----
        # next-state 的 prev_action = current action（u_i^t 是 s_{t+1} 的前一步动作）
        next_prev_act_seqs = [per_agent_act[i].view(T, N) for i in range(num_agents)]
        next_state_flat = next_states.view(-1, self.state_dim)       # (T*N, state_dim)

        target_v_next = self._double_dqn_next_v_rnn(
            next_states, next_prev_act_seqs, num_agents, T, N,
        )

        V_tgt_all = torch.stack(target_v_next, dim=-1)               # (T*N, n_agents)
        A_tgt_all = torch.zeros_like(V_tgt_all)                      # 下一步 advantage 置零（QPLEX）
        Q_tot_next = self.target_mixing_net(V_tgt_all, A_tgt_all, next_state_flat)

        # ---- y_tot = r_tot + γ * Q_tot_next（terminal 处 bootstrap = 0） ----
        rew_2d = r_tot.view(T, N)
        bootstrap = self.gamma * Q_tot_next.view(T, N)
        bootstrap[term_bool] = 0.0
        y_tot_flat = (rew_2d + bootstrap).view(-1)

        return per_agent_v, per_agent_qplex_a, old_q_tot, y_tot_flat, prev_act_seqs

    def _double_dqn_next_v_rnn(
        self,
        next_states: torch.Tensor,           # (T, N, state_dim)
        next_prev_act_seqs: List[torch.Tensor],  # list of (T, N) int
        num_agents: int, T: int, N: int,
    ) -> List[torch.Tensor]:
        """
        Double DQN next-state V：online Q 选 greedy，target Q 评估值。
        Returns: list of n_agents * (T*N,) float tensors
        """
        q_in_all = self._build_rnn_q_inputs(next_states, next_prev_act_seqs, num_agents, T, N)

        # online Q forward
        h0_on = self.q_network.get_initial_hidden(N * num_agents, self.device)
        q_on_all, _ = self.q_network.forward_sequence(q_in_all, h0_on)  # (T, N*n_ag, act)

        # target Q forward
        h0_tgt = self.target_q_network.get_initial_hidden(N * num_agents, self.device)
        q_tgt_all, _ = self.target_q_network.forward_sequence(q_in_all, h0_tgt)

        q_on_agents = q_on_all.chunk(num_agents, dim=1)               # list of (T, N, action_dim)
        q_tgt_agents = q_tgt_all.chunk(num_agents, dim=1)

        target_v_next = []
        for q_on_i, q_tgt_i in zip(q_on_agents, q_tgt_agents):
            greedy = q_on_i.argmax(dim=-1)                            # (T, N)
            V_tgt = q_tgt_i.gather(2, greedy.unsqueeze(-1)).squeeze(-1).view(-1)  # (T*N,)
            target_v_next.append(V_tgt)

        return target_v_next

    # ====================================================================
    #                     Update（双优化器）
    # ====================================================================

    def update(
        self,
        batch: RolloutBatch,
        minibatch_size: int,
        update_epochs: int,
        update_actor: bool = True,
    ) -> TrainingStats:
        """
        双优化器 update:
          1) Q-network (q_network + mixing_net) — clipped TD loss on Q_tot
          2) Actor — PPO clipped surrogate with per-agent A_i + active_mask
        """
        # Q-network 热身期：前 freeze_actor_iters 次 update 强制冻结 actor
        if self.freeze_actor_iters > 0 and self._update_count < self.freeze_actor_iters:
            update_actor = False
            if self._update_count == 0:
                print(f"[VDPPO] Actor frozen for first {self.freeze_actor_iters} updates (Q-network warmup).")
        elif self.freeze_actor_iters > 0 and self._update_count == self.freeze_actor_iters:
            print(f"[VDPPO] Q-network warmup done. Actor unfrozen at update #{self._update_count}.")

        all_pg_loss, all_q_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl: list[float] = []
        all_actor_gn, all_q_gn = [], []

        q_data = self._q_data
        assert q_data is not None, "prepare_batch must be called before update"

        for epoch in range(update_epochs):
            epoch_approx_kl: list[float] = []

            # ============ 1. Q-Network Update ============
            if q_data["is_rnn"]:
                q_loss, q_gn = self._update_q_rnn(q_data)
                all_q_loss.append(q_loss)
                all_q_gn.append(q_gn)
            else:
                q_losses, q_gns = self._update_q_mlp(q_data, minibatch_size)
                all_q_loss.extend(q_losses)
                all_q_gn.extend(q_gns)

            # ============ 2. Actor Update (merged, 同 MAPPO) ============
            _actor_rnn = self.is_recurrent

            if _actor_rnn:
                actor_mb_iter = batch.chunk_split(
                    chunk_len=self.data_chunk_length,
                    T=self._last_T, N=self._last_N,
                    num_agents=self._last_num_agents,
                    minibatch_size=minibatch_size,
                )
            else:
                actor_mb_iter = batch.split(size=minibatch_size, shuffle=True, merge_last=True)

            for mb in actor_mb_iter:
                if mb.active_mask is not None:
                    am = mb.active_mask.float()
                    if _actor_rnn:
                        am = am.reshape(-1)
                    am_sum = am.sum().clamp(min=1.0)
                else:
                    am = None

                mb_adv = mb.adv
                if _actor_rnn:
                    mb_adv = mb_adv.reshape(-1)
                if self.normalize_advantage:
                    if am is not None:
                        active_adv = mb_adv[am > 0.5]
                        if active_adv.numel() > 1:
                            mb_adv = (mb_adv - active_adv.mean()) / (active_adv.std() + 1e-8)
                    elif mb_adv.numel() > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                if _actor_rnn:
                    new_lp, entropy = self.policy.evaluate_actions_sequence_flat(
                        mb.obs, mb.act, mb.rnn_hidden,
                        action_mask=mb.action_mask,
                    )
                    new_lp = new_lp.reshape(-1)
                    entropy = entropy.reshape(-1)
                else:
                    new_lp, entropy = self.policy.evaluate_actions_flat(
                        mb.obs, mb.act, action_mask=mb.action_mask,
                    )
                mb_log_prob = mb.log_prob.reshape(-1) if _actor_rnn else mb.log_prob

                logratio = new_lp - mb_log_prob
                ratio = logratio.exp()

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_per = torch.max(pg1, pg2)

                if am is not None:
                    pg_loss = (pg_per * am).sum() / am_sum
                    ent_loss = (entropy * am).sum() / am_sum
                else:
                    pg_loss = pg_per.mean()
                    ent_loss = entropy.mean()

                actor_loss = pg_loss - self.ent_coef * ent_loss

                self.actor_optimizer.zero_grad()
                if update_actor:
                    actor_loss.backward()
                    a_gn = nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm,
                    )
                    self.actor_optimizer.step()
                    all_actor_gn.append(a_gn.item())

                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()

                all_pg_loss.append(pg_loss.item())
                all_entropy.append(ent_loss.item())

                with torch.no_grad():
                    cf = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    all_clipfrac.append(cf)
                    epoch_approx_kl.append(((ratio - 1) - logratio).mean().item())

            # KL early stopping (actor)
            if epoch_approx_kl:
                avg_kl = np.mean(epoch_approx_kl)
                all_approx_kl.append(avg_kl)
                if self.target_kl is not None and avg_kl > self.target_kl:
                    break

        # ---- 目标网络更新 ----
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            if self.tau < 1.0:
                self._soft_update_targets()
            else:
                self._hard_update_targets()

        return TrainingStats(
            loss=np.mean(all_pg_loss) + np.mean(all_q_loss),
            policy_loss=np.mean(all_pg_loss),
            value_loss=np.mean(all_q_loss),
            entropy=np.mean(all_entropy),
            extra={
                "clipfrac": np.mean(all_clipfrac),
                "approx_kl": np.mean(all_approx_kl) if all_approx_kl else 0.0,
                "actor_grad_norm": np.mean(all_actor_gn) if all_actor_gn else 0.0,
                "q_grad_norm": np.mean(all_q_gn),
            },
        )

    # ====================================================================
    #                     Q Update 辅助
    # ====================================================================

    def _q_td_loss(
        self,
        Q_tot: torch.Tensor,    # (B,)
        y: torch.Tensor,        # (B,)
        oq: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        """带可选 clip 的 TD MSE loss。"""
        if self.q_clip_range is not None:
            Q_clip = oq + torch.clamp(Q_tot - oq, -self.q_clip_range, self.q_clip_range)
            return 0.5 * torch.max((y - Q_tot) ** 2, (y - Q_clip) ** 2).mean()
        return 0.5 * ((y - Q_tot) ** 2).mean()

    def _update_q_mlp(self, q_data: Dict, minibatch_size: int):
        """MLP Q-network update：随机 minibatch 平坦路径。"""
        T_N = q_data["states"].shape[0]
        q_mb_size = T_N if minibatch_size <= 0 else minibatch_size
        q_perm = np.random.permutation(T_N)
        losses, gns = [], []

        for start in range(0, T_N, q_mb_size):
            end = start + q_mb_size
            if end + q_mb_size > T_N and end < T_N:
                idx = q_perm[start:]
            else:
                idx = q_perm[start:min(end, T_N)]
            if len(idx) == 0:
                continue

            s_mb = q_data["states"][idx]
            ja_mb = q_data["joint_actions"][idx]
            y_mb = q_data["y_tot"][idx]
            oq_mb = q_data["old_q_tot"][idx]
            mb_len = len(idx)

            v_list, a_list = [], []
            for i in range(self.n_agents):
                oh = torch.zeros(mb_len, self.n_agents, device=self.device)
                oh[:, i] = 1.0
                q_in = torch.cat([s_mb, oh], dim=-1)
                Q_i = self.q_network(q_in)
                V_i = Q_i.max(dim=-1).values
                A_i = Q_i.gather(1, ja_mb[:, i].unsqueeze(-1)).squeeze(-1) - V_i
                v_list.append(V_i)
                a_list.append(A_i)

            Q_tot = self.mixing_net(
                torch.stack(v_list, dim=-1),
                torch.stack(a_list, dim=-1),
                s_mb,
            )
            q_loss = self._q_td_loss(Q_tot, y_mb, oq_mb)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            q_gn = nn.utils.clip_grad_norm_(
                list(self.q_network.parameters()) + list(self.mixing_net.parameters()),
                self.max_grad_norm,
            )
            self.q_optimizer.step()
            if self.q_scheduler is not None:
                self.q_scheduler.step()

            losses.append(q_loss.item())
            gns.append(q_gn.item())

        return losses, gns

    def _update_q_rnn(self, q_data: Dict):
        """
        RNN Q-network update：整序列一次 forward，不对 T 维度做 minibatch。
        每个 update epoch 调用一次。
        """
        state_seq = q_data["state_seq"]                              # (T, N, state_dim)
        ja_seq = q_data["joint_actions_seq"]                         # (T, N, n_agents)
        prev_act_seqs = q_data["prev_act_seqs"]                      # list of (T, N)
        y_flat = q_data["y_tot"]                                     # (T*N,)
        oq_flat = q_data["old_q_tot"]                                # (T*N,)
        T, N = q_data["T_q"], q_data["N_q"]
        global_state_flat = state_seq.view(-1, self.state_dim)

        q_in_all = self._build_rnn_q_inputs(state_seq, prev_act_seqs, self.n_agents, T, N)
        h0 = self.q_network.get_initial_hidden(N * self.n_agents, self.device)
        q_vals_all, _ = self.q_network.forward_sequence(q_in_all, h0)  # (T, N*n_ag, act)
        q_vals_agents = q_vals_all.chunk(self.n_agents, dim=1)

        v_list, a_list = [], []
        for i, qv in enumerate(q_vals_agents):
            V_i = qv.max(dim=-1).values.view(-1)                     # (T*N,)
            A_i = qv.gather(2, ja_seq[:, :, i].unsqueeze(-1)).squeeze(-1).view(-1) - V_i
            v_list.append(V_i)
            a_list.append(A_i)

        Q_tot = self.mixing_net(
            torch.stack(v_list, dim=-1),
            torch.stack(a_list, dim=-1),
            global_state_flat,
        )
        q_loss = self._q_td_loss(Q_tot, y_flat, oq_flat)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_gn = nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixing_net.parameters()),
            self.max_grad_norm,
        )
        self.q_optimizer.step()
        if self.q_scheduler is not None:
            self.q_scheduler.step()

        return q_loss.item(), q_gn.item()

    # ====================================================================
    #                     辅助方法
    # ====================================================================

    def _soft_update_targets(self):
        tau = self.tau
        for tp, p in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def _hard_update_targets(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.policy.set_training_mode(mode)
        self.q_network.train(mode)
        self.mixing_net.train(mode)
