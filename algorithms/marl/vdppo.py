"""VDPPO: Value-Decomposition Multi-Agent PPO (Ma & Luo, 2022).

PPO actor + QPLEX-style Q-decomposition + Double DQN TD targets + 双优化器。
"""

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from algorithms.algorithm_base import TrainingStats
from algorithms.rl.ppo import PPOBase
from configs.algo_configs import VDPPOParams
from data.batch import RolloutBatch
from networks.mixing import QPLEXMixer
from policies.marl.marl_base import MultiAgentPolicy


def _as_q_logits(q_forward_ret):
    """Q-network forward：QMLP 返回 Tensor；QRNN 等返回 (logits, hidden)。"""
    return q_forward_ret[0] if isinstance(q_forward_ret, tuple) else q_forward_ret


class VDPPOAlgo(PPOBase):
    """
    VDPPO: PPO actor 更新 + QPLEX 值分解 + 双优化器。

    继承 PPOBase 获取 clip_range / clip_vloss / target_kl 等超参，
    完全 override prepare_batch (Q-decomposition, 不用 GAE) 和 update (双优化器)。
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

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Q-network 由外部工厂传入，若未提供则回退创建
        if q_network is not None:
            self.q_network = q_network.to(self.device)
        else:
            from networks.mlp import QMLP
            q_input_dim = state_dim + n_agents
            self.q_network = QMLP(q_input_dim, [64, 64], action_dim).to(self.device)

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

        # prepare_batch 设置，update 消费
        self._q_data: Optional[Dict[str, torch.Tensor]] = None

    # ====================================================================
    #                     Batch 预处理（Q-Decomposition）
    # ====================================================================

    def prepare_batch(self, batch_dict: Dict[str, RolloutBatch]) -> RolloutBatch:
        """
        完全 override 基类:

        Actor 部分: per-agent GAE（V_i = max Q_i 作为 baseline，实际 reward 驱动）
        Q 部分: Q-decomposition TD target（Q_tot → y_tot，用于 Q-network 更新）
        """
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs

        # ---- 公共字段（agent 间相同） ----
        first_agent = agents[0]
        final_gs = batch_dict[first_agent].final_global_state
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
        per_agent_act: list[torch.Tensor] = []
        per_agent_rew: list[torch.Tensor] = []

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
        joint_rew = torch.stack(per_agent_rew, dim=-1).sum(dim=-1)  # (T*N,)

        with torch.no_grad():
            # ---- Per-agent V_i = max Q_i（GAE baseline） ----
            per_agent_v = []
            per_agent_qplex_a = []
            for i in range(num_agents):
                one_hot = torch.zeros(total_size, num_agents, device=self.device)
                one_hot[:, i] = 1.0
                q_in = torch.cat([global_state, one_hot], dim=-1)

                Q_i = _as_q_logits(self.q_network(q_in))            # (T*N, action_dim)
                V_i = Q_i.max(dim=-1).values                        # (T*N,)
                A_i = (
                    Q_i.gather(1, per_agent_act[i].unsqueeze(-1)).squeeze(-1) - V_i
                )
                per_agent_v.append(V_i)
                per_agent_qplex_a.append(A_i)

            # ---- 当前 Q_tot（Q-update clipping baseline） ----
            V_vals = torch.stack(per_agent_v, dim=-1)               # (T*N, n_agents)
            A_vals = torch.stack(per_agent_qplex_a, dim=-1)         # (T*N, n_agents)
            old_q_tot = self.mixing_net(V_vals, A_vals, global_state)  # (T*N,)

            # ---- 构建 next_states ----
            next_states = torch.zeros_like(state_2d)                # (T, N, state_dim)
            next_states[:-1] = state_2d[1:]
            next_states[-1] = state_2d[-1]

            if final_gs is not None:
                for pos in trunc_bool.nonzero(as_tuple=False):
                    t_idx, e_idx = pos[0].item(), pos[1].item()
                    fs = final_gs[t_idx][e_idx]
                    if fs is not None:
                        next_states[t_idx, e_idx] = torch.as_tensor(
                            fs, dtype=torch.float32, device=self.device,
                        )

            next_flat = next_states.view(-1, self.state_dim)        # (T*N, state_dim)

            # ---- Double DQN: current Q 选 action, target Q 评估 ----
            target_v_next = []
            for i in range(num_agents):
                one_hot = torch.zeros(total_size, num_agents, device=self.device)
                one_hot[:, i] = 1.0
                q_in = torch.cat([next_flat, one_hot], dim=-1)

                greedy = _as_q_logits(self.q_network(q_in)).argmax(dim=-1)  # (T*N,)
                Q_tgt = _as_q_logits(self.target_q_network(q_in))   # (T*N, action_dim)
                V_tgt_i = Q_tgt.gather(1, greedy.unsqueeze(-1)).squeeze(-1)
                target_v_next.append(V_tgt_i)

            V_tgt_all = torch.stack(target_v_next, dim=-1)          # (T*N, n_agents)
            A_tgt_all = torch.zeros_like(V_tgt_all)
            Q_tot_next = self.target_mixing_net(
                V_tgt_all, A_tgt_all, next_flat,
            )                                                       # (T*N,)

            # ---- y_tot = r + γ * Q_tot_next  (terminal 处 bootstrap = 0) ----
            rew_2d = joint_rew.view(T, N)
            bootstrap = self.gamma * Q_tot_next.view(T, N)
            bootstrap[term_bool] = 0.0
            y_tot_flat = (rew_2d + bootstrap).view(-1)              # (T*N,)

            # ---- Per-agent GAE: V_i 作为 baseline, 实际 r_i 驱动 ----
            for i, agent in enumerate(agents):
                b = batch_dict[agent].to_tensor(self.device)
                rew_i_2d = b.rew.view(T, N)
                values_i_2d = per_agent_v[i].view(T, N)

                active_mask_2d = None
                if self.use_active_mask and b.active_mask is not None:
                    active_mask_2d = b.active_mask.view(T, N)

                # Truncation bootstrap: V_i(final_state)
                trunc_bootstrap_i = self._compute_q_trunc_bootstrap(
                    trunc_2d, final_gs, i, num_agents, T, N,
                )

                adv_i, ret_i = self._gae_vectorized(
                    rew_i_2d, values_i_2d, done_2d, trunc_2d,
                    trunc_bootstrap_i, active_mask_2d,
                )
                all_adv.append(adv_i)
                all_ret.append(ret_i)
                all_value.append(values_i_2d.view(-1))

        # ---- 存储 Q-update 辅助数据 ----
        self._q_data = {
            "states": global_state,                                 # (T*N, state_dim)
            "joint_actions": joint_actions,                         # (T*N, n_agents)
            "y_tot": y_tot_flat,                                    # (T*N,)
            "old_q_tot": old_q_tot,                                 # (T*N,)
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
        all_pg_loss, all_q_loss, all_entropy, all_clipfrac = [], [], [], []
        all_approx_kl: list[float] = []
        all_actor_gn, all_q_gn = [], []

        q_data = self._q_data
        assert q_data is not None, "prepare_batch must be called before update"

        T_N = q_data["states"].shape[0]
        q_mb_size = T_N if minibatch_size <= 0 else minibatch_size

        for epoch in range(update_epochs):
            epoch_approx_kl: list[float] = []

            # ============ 1. Q-Network Update ============
            q_perm = np.random.permutation(T_N)
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
                    Q_i = _as_q_logits(self.q_network(q_in))
                    V_i = Q_i.max(dim=-1).values
                    A_i = Q_i.gather(1, ja_mb[:, i].unsqueeze(-1)).squeeze(-1) - V_i
                    v_list.append(V_i)
                    a_list.append(A_i)

                Q_tot = self.mixing_net(
                    torch.stack(v_list, dim=-1),
                    torch.stack(a_list, dim=-1),
                    s_mb,
                )

                if self.q_clip_range is not None:
                    Q_clip = oq_mb + torch.clamp(
                        Q_tot - oq_mb, -self.q_clip_range, self.q_clip_range,
                    )
                    q_loss = 0.5 * torch.max(
                        (y_mb - Q_tot) ** 2,
                        (y_mb - Q_clip) ** 2,
                    ).mean()
                else:
                    q_loss = 0.5 * ((y_mb - Q_tot) ** 2).mean()

                self.q_optimizer.zero_grad()
                q_loss.backward()
                q_gn = nn.utils.clip_grad_norm_(
                    list(self.q_network.parameters()) + list(self.mixing_net.parameters()),
                    self.max_grad_norm,
                )
                self.q_optimizer.step()
                if self.q_scheduler is not None:
                    self.q_scheduler.step()

                all_q_loss.append(q_loss.item())
                all_q_gn.append(q_gn.item())

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
    #                     辅助方法
    # ====================================================================

    def _compute_q_trunc_bootstrap(
        self,
        truncateds: Optional[torch.Tensor],
        final_global_states,
        agent_idx: int,
        num_agents: int,
        T: int,
        N: int,
    ) -> torch.Tensor:
        """Truncation 处用 V_i = max Q_i(final_state) 做 bootstrap。"""
        device = self.device
        trunc_bootstrap = torch.zeros(T, N, device=device)

        if truncateds is None or final_global_states is None:
            return trunc_bootstrap

        trunc_mask = truncateds > 0.5
        if not trunc_mask.any():
            return trunc_bootstrap

        trunc_positions = trunc_mask.nonzero(as_tuple=False)

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
            one_hot = torch.zeros(len(batch_states), num_agents, device=device)
            one_hot[:, agent_idx] = 1.0
            q_in = torch.cat([states_t, one_hot], dim=-1)
            Q_i = _as_q_logits(self.q_network(q_in))
            V_i = Q_i.max(dim=-1).values

            for vi, k in enumerate(valid_k_indices):
                t_idx = trunc_positions[k, 0].item()
                e_idx = trunc_positions[k, 1].item()
                trunc_bootstrap[t_idx, e_idx] = V_i[vi]

        return trunc_bootstrap

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
