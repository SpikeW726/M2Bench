"""VDN: Value Decomposition Network for multi-agent environments.

Q_tot = Σ_i Q_i(o_i, a_i)  (SumMixer, 无可学习参数)

所有 agent 共享 Q-network (shared_policy=True)，
联合 TD loss 通过 SumMixer 汇聚后反向传播。

本类同时作为 QMIX 等值分解算法的基类，子类通过 override
_init_mixer / _init_optimizer / _update_target_networks 实现差异化。
"""

import copy
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithm_base import BaseAlgorithm, TrainingStats
from configs.algo_configs import VDNParams
from data.batch import TransitionBatch, SequenceBatch


def _build_q_lambda_targets_seq(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    mask: torch.Tensor,
    exp_qvals: torch.Tensor,
    qvals: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Peng’s Q(λ) 反向 λ-return，与 PyMARL `utils.rl_utils.build_q_lambda_targets` 同构；时间维为 dim0 (S, B)。"""
    S, B = rewards.shape
    if S == 0:
        return rewards
    term = terminated.float() if terminated.dtype != torch.float32 else terminated
    ret = torch.zeros_like(exp_qvals)
    ret[S - 1] = exp_qvals[S - 1] * (1.0 - term[S - 1])
    if S >= 2:
        for t in range(S - 2, -1, -1):
            reward_eff = rewards[t] + exp_qvals[t] - qvals[t]
            ret[t] = lam * gamma * ret[t + 1] + mask[t] * (
                reward_eff
                + (1.0 - lam) * gamma * exp_qvals[t + 1] * (1.0 - term[t])
            )
    else:
        # S==1：无多步 λ 混合，退化为单步 TD（不在此施加 Peng 修正，避免边界双计）
        ret[0] = rewards[0] + gamma * exp_qvals[0] * (1.0 - term[0])
    return ret
from networks.mixing import QMIXMixer, SumMixer
from policies.marl.marl_base import MultiAgentPolicy
from policies.rl.rl_base import ValuePolicy


class VDNAlgo(BaseAlgorithm):
    """
    VDN: Value Decomposition Network

    CTDE 范式：分散执行时每个 agent 用自身 Q_i 做 argmax
    集中训练时通过 Q_tot = Σ Q_i 的联合 TD loss 更新共享 Q-network

    同时作为 QMIX 等值分解算法的基类，可 override 的扩展点：
    - _init_mixer(): Mixer 网络创建（VDN=SumMixer 无参数, QMIX=QMIXMixer 含 target_mixer）
    - _init_optimizer(): 优化器创建（QMIX 需包含 mixer 参数）
    - _update_target_networks(): Target 更新（QMIX 需额外更新 target_mixer）
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        params: VDNParams,
        n_agents: int = 1,
        state_dim: int = 0,
    ):
        super().__init__(policy)

        # VDN/QMIX 必须使用参数共享；独立网络模式下只有 agent_0 能被优化
        if not policy.shared:
            raise ValueError(
                f"{type(self).__name__} 要求 shared_policy=true。"
                "独立网络模式下其余 agent 的 Q-network 永远不会被更新，请在 YAML 中设置 shared_policy: true。"
            )

        self.params = params
        self.gamma = params.gamma
        self.max_grad_norm = params.max_grad_norm
        self.tau = params.tau
        self.target_update_freq = params.target_update_freq
        self.n_agents = n_agents

        # 共享 Q-network
        self.q_network = policy.get_policy(policy.agent_ids[0]).q_network

        # Target Q-network
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.eval()
        for p in self.target_q_network.parameters():
            p.requires_grad = False

        # Mixer & Optimizer,子类可 override
        self._init_mixer(n_agents, state_dim, params)
        self._init_optimizer(params)

        # Epsilon 管理
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay_steps = max(1, int(params.epsilon_decay_steps))
        for aid in policy.agent_ids:
            policy.get_policy(aid).set_epsilon(params.epsilon_start)

        self._update_count = 0

    # ====================================================================
    #                       子类扩展点
    # ====================================================================

    def _init_mixer(self, n_agents: int, state_dim: int, params):
        """创建 Mixer 网络,VDN 使用无参数的 SumMixer"""
        self.mixer = SumMixer(n_agents, state_dim)

    def _init_optimizer(self, params):
        """创建优化器,VDN 仅优化 Q-network"""
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=params.lr,
        )

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.q_network, "is_recurrent", False)

    # ====================================================================
    #               MLP 路径: compute joint loss (flat transitions)
    # ====================================================================

    def _compute_loss_flat(
        self,
        batch_dict: Dict[str, TransitionBatch],
    ) -> tuple[torch.Tensor, dict]:
        agents = self.policy.agent_ids
        first_batch = batch_dict[agents[0]]
        B = first_batch.obs.shape[0]
        N = self.n_agents

        # Stack per-agent data → (B, N, ...)
        obs_all = torch.stack([batch_dict[a].obs for a in agents], dim=1)
        act_all = torch.stack([batch_dict[a].act for a in agents], dim=1).long()
        rew_all = torch.stack([batch_dict[a].rew for a in agents], dim=1)
        next_obs_all = torch.stack([batch_dict[a].next_obs for a in agents], dim=1)
        done = first_batch.done
        state = first_batch.state
        next_state = first_batch.next_state

        if isinstance(self.mixer, QMIXMixer):
            if state is None or next_state is None:
                raise RuntimeError(
                    "QMIX 需要 TransitionBatch.state / next_state,当前为 None"
                    "请确认采集端 collect_state=True 且环境实现 state()"
                )
            if state.shape[0] != B or state.shape[-1] != self.mixer.state_dim:
                raise RuntimeError(
                    f"QMIX state 形状与 batch 不一致: state.shape={tuple(state.shape)}, "
                    f"B={B}, expected_state_dim={self.mixer.state_dim}"
                )
            if next_state.shape[0] != B or next_state.shape[-1] != self.mixer.state_dim:
                raise RuntimeError(
                    f"QMIX next_state 形状与 batch 不一致: next_state.shape={tuple(next_state.shape)}, "
                    f"B={B}, expected_state_dim={self.mixer.state_dim}"
                )

        # Action masks
        next_am_all = None
        if first_batch.next_action_mask is not None:
            next_am_all = torch.stack(
                [batch_dict[a].next_action_mask for a in agents], dim=1,
            )

        # Active mask: 仅 READY 决策步（ON_EDGE 步 active_mask=0）参与 loss。
        # MASUP 等事件驱动环境中约 80% 的 transition 为 ON_EDGE，不过滤会严重稀释梯度。
        active_mask = getattr(first_batch, "active_mask", None)  # (B,)

        # --- Online Q-values ---
        q_all = self.q_network(obs_all.view(B * N, -1)).view(B, N, -1)
        q_chosen = q_all.gather(-1, act_all.unsqueeze(-1)).squeeze(-1)  # (B, N)
        q_tot = self.mixer(q_chosen, state)  # (B,)

        # --- Target Q-values ---
        with torch.no_grad():
            if self.params.use_double_dqn:
                q_next_online = self.q_network(next_obs_all.view(B * N, -1)).view(B, N, -1)
                if next_am_all is not None:
                    q_next_online = q_next_online.masked_fill(~next_am_all.bool(), float("-inf"))
                next_actions = q_next_online.argmax(dim=-1, keepdim=True)
                q_next_target = self.target_q_network(next_obs_all.view(B * N, -1)).view(B, N, -1)
                q_next_chosen = q_next_target.gather(-1, next_actions).squeeze(-1)
            else:
                q_next_target = self.target_q_network(next_obs_all.view(B * N, -1)).view(B, N, -1)
                if next_am_all is not None:
                    q_next_target = q_next_target.masked_fill(~next_am_all.bool(), float("-inf"))
                q_next_chosen = q_next_target.max(dim=-1)[0]

            # clamp：action mask 全 False 时 max=-inf，进入 SumMixer 后乘 (1-done) 会产生 0*(-inf)=NaN
            q_next_chosen = q_next_chosen.clamp(min=-1e9, max=1e9)

            # VDN 无 target_mixer（SumMixer 无参数），QMIX 有独立 target_mixer
            target_mixer = getattr(self, "target_mixer", self.mixer)
            q_tot_target = target_mixer(q_next_chosen, next_state)

            if self.params.reward_global:
                r_tot = rew_all[:, 0]   # 环境各 agent 奖励相同，取第一个
            else:
                r_tot = rew_all.sum(dim=-1)   # 各 agent 奖励之和

            # where 替代 *(1-done)：done=1 时 0*(-inf)=NaN，显式置 0 更安全
            bootstrap = torch.where(done.bool(), torch.zeros_like(q_tot_target), q_tot_target)
            gamma_pow = getattr(first_batch, "gamma_power", None)
            if gamma_pow is not None:
                gamma_pow = gamma_pow.to(r_tot.device).float()
            else:
                gamma_pow = torch.full_like(r_tot, self.gamma, dtype=torch.float32)
            td_target = r_tot + gamma_pow * bootstrap

        # Huber TD loss：较 MSE 对大误差更温和，与 D3QN 一致，利于 QRNN 等多步 bootstrap 稳定
        per_sample_loss = F.smooth_l1_loss(q_tot, td_target, reduction="none")
        if active_mask is not None:
            am_sum = active_mask.sum().clamp(min=1)
            loss = (per_sample_loss * active_mask).sum() / am_sum
        else:
            loss = per_sample_loss.mean()

        td_err = (td_target - q_tot).detach().abs()
        if active_mask is not None:
            am_sum = active_mask.sum().clamp(min=1)
            td_err_scalar = (td_err * active_mask).sum() / am_sum
            q_active = q_tot.detach()[active_mask > 0.5]
        else:
            td_err_scalar = td_err.mean()
            q_active = q_tot.detach()
        info = {
            "q_tot_mean": q_active.mean() if q_active.numel() > 0 else torch.tensor(0.0, device=q_tot.device),
            "q_tot_max":  q_active.max()  if q_active.numel() > 0 else torch.tensor(0.0, device=q_tot.device),
            "td_error": td_err_scalar,
        }
        return loss, info

    # ====================================================================
    #               RNN 路径: compute joint loss (sequences)
    # ====================================================================

    def _compute_loss_seq(
        self,
        batch_dict: Dict[str, SequenceBatch],
    ) -> tuple[torch.Tensor, dict]:
        agents = self.policy.agent_ids
        first_batch = batch_dict[agents[0]]
        B, T = first_batch.obs.shape[:2]
        N = self.n_agents
        device = first_batch.obs.device
        bi = first_batch.burn_in_len
        S = T - bi          # 训练步数

        # === 1. 所有 agent 堆叠 → (B, T, N, *) ===
        obs_all       = torch.stack([batch_dict[a].obs       for a in agents], dim=2)   # (B,T,N,D)
        next_obs_all  = torch.stack([batch_dict[a].next_obs  for a in agents], dim=2)
        act_all       = torch.stack([batch_dict[a].act       for a in agents], dim=2).long()  # (B,T,N)
        rew_all       = torch.stack([batch_dict[a].rew       for a in agents], dim=2)   # (B,T,N)
        done          = first_batch.done    # (B, T)
        mask          = first_batch.mask    # (B, T)  1=有效步, 0=padding
        state_seq     = getattr(first_batch, "state",      None)  # (B, T, state_dim) or None
        next_state_seq= getattr(first_batch, "next_state", None)

        # === 2. 共享网络：以 B*N 为批次维度运行序列 ===
        obs_seq      = obs_all.permute(1, 0, 2, 3).contiguous().view(T, B * N, -1)
        next_obs_seq = next_obs_all.permute(1, 0, 2, 3).contiguous().view(T, B * N, -1)
        h0  = self.q_network.get_initial_hidden(B * N, device)

        # 全序列拼接：[obs₀…obs_{T-1}] + [obs_T] = [obs₀…obs_T]，长度 T+1
        # 使用同一条 hidden state 链路同时计算训练 Q 和 bootstrap target Q，
        # 消除 "对 next_obs_seq 单独从 zeros 起步" 导致的 hidden state 断裂问题。
        full_obs_seq = torch.cat([obs_seq, next_obs_seq[-1:]], dim=0)  # (T+1, B*N, D)
        q_full_ext, _ = self.q_network.forward_sequence(full_obs_seq, h0)  # (T+1, B*N, act_dim)

        # 训练 Q：output[bi:T]，hidden 链路与原实现完全一致
        q_train = q_full_ext.view(T + 1, B, N, -1)[bi:T]  # (S, B, N, act_dim)

        act_train = act_all[:, bi:, :].permute(1, 0, 2).unsqueeze(-1)   # (S, B, N, 1)
        q_chosen  = q_train.gather(3, act_train).squeeze(-1)             # (S, B, N)

        # Mix online → Q_tot
        q_chosen_flat = q_chosen.reshape(S * B, N)
        state_train_flat = None
        if state_seq is not None:
            state_train_flat = (
                state_seq[:, bi:, :].permute(1, 0, 2).contiguous().view(S * B, -1)
            )
        q_tot = self.mixer(q_chosen_flat, state_train_flat).view(S, B)  # (S, B)

        # === 3. 目标 Q-values ===
        with torch.no_grad():
            h0_tgt = self.target_q_network.get_initial_hidden(B * N, device)

            # 可选 action mask（切片对齐训练区间 S 步）
            next_am_train = None
            if getattr(first_batch, "next_action_mask", None) is not None:
                next_am_all = torch.stack(
                    [batch_dict[a].next_action_mask for a in agents], dim=2
                )   # (B, T, N, act_dim)
                next_am_train = next_am_all[:, bi:, :, :].permute(1, 0, 2, 3)  # (S,B,N,act_dim)

            if self.params.use_double_dqn:
                # online net 已在 no_grad 外跑过 full_obs_seq；
                # bootstrap 区间 output[bi+1:T+1]，detach 阻断 backward
                q_next_online = q_full_ext.view(T + 1, B, N, -1)[bi + 1:T + 1].detach()  # (S,B,N,A)
                if next_am_train is not None:
                    q_next_online = q_next_online.masked_fill(~next_am_train.bool(), float("-inf"))
                next_actions = q_next_online.argmax(dim=-1, keepdim=True)  # (S,B,N,1)

                # target net 同样跑 full_obs_seq，bootstrap 区间 [bi+1:T+1]
                q_tgt_ext, _ = self.target_q_network.forward_sequence(full_obs_seq, h0_tgt)
                q_next_tgt   = q_tgt_ext.view(T + 1, B, N, -1)[bi + 1:T + 1]  # (S,B,N,A)
                q_next_chosen = q_next_tgt.gather(3, next_actions).squeeze(-1)  # (S,B,N)
            else:
                q_tgt_ext, _ = self.target_q_network.forward_sequence(full_obs_seq, h0_tgt)
                q_next_tgt   = q_tgt_ext.view(T + 1, B, N, -1)[bi + 1:T + 1]  # (S,B,N,A)
                if next_am_train is not None:
                    q_next_tgt = q_next_tgt.masked_fill(~next_am_train.bool(), float("-inf"))
                q_next_chosen = q_next_tgt.max(dim=-1)[0]                       # (S,B,N)

            # clamp：action mask 全 False 或 padding 步 max=-inf，进入 SumMixer 后产生 0*(-inf)=NaN
            q_next_chosen = q_next_chosen.clamp(min=-1e9, max=1e9)

            # Mix target → Q_tot_target
            q_next_flat = q_next_chosen.reshape(S * B, N)
            next_state_flat = None
            if next_state_seq is not None:
                next_state_flat = (
                    next_state_seq[:, bi:, :].permute(1, 0, 2).contiguous().view(S * B, -1)
                )
            target_mixer  = getattr(self, "target_mixer", self.mixer)
            q_tot_target  = target_mixer(q_next_flat, next_state_flat).view(S, B)

            # 联合奖励: (S, B)
            rew_train = rew_all[:, bi:, :].permute(1, 0, 2)   # (S, B, N)
            if self.params.reward_global:
                r_tot = rew_train[..., 0]          # 取第一个 agent（环境奖励相同）
            else:
                r_tot = rew_train.sum(dim=-1)      # 各 agent 奖励之和

            done_train = done[:, bi:].T            # (S, B)
            mask_train = mask[:, bi:].T            # (S, B)，提前供 Peng Q(λ) 与 loss 共用

            if getattr(self.params, "peng_q_lambda", False):
                # 行为策略 Q_tot(s_{t+1}, a_{t+1})：前 S-1 步用 act_train[s+1]，末步无 a_{t+1} 则用 greedy Q（与 exp 一致 → Peng 修正为 0）
                beh_ag = torch.zeros(S, B, N, device=device, dtype=q_next_tgt.dtype)
                if S >= 2:
                    beh_ag[:-1] = q_next_tgt[:-1].gather(3, act_train[1:]).squeeze(-1)
                beh_ag[-1] = q_next_chosen[-1]
                q_tot_beh = target_mixer(
                    beh_ag.reshape(S * B, N), next_state_flat,
                ).view(S, B)
                lam = float(getattr(self.params, "peng_lambda", 0.6))
                td_target = _build_q_lambda_targets_seq(
                    r_tot,
                    done_train,
                    mask_train,
                    q_tot_target,
                    q_tot_beh,
                    self.gamma,
                    lam,
                )
            else:
                # where 替代 *(1-done)：done=1 时 0*(-inf)=NaN，显式置 0 更安全
                bootstrap = torch.where(done_train.bool(), torch.zeros_like(q_tot_target), q_tot_target)
                td_target = r_tot + self.gamma * bootstrap

        # === 4. 带序列 mask（+ 可选 active_mask）的 MSE loss ===
        td_loss    = F.smooth_l1_loss(q_tot, td_target, reduction="none")  # (S, B)
        # mask_train 已在目标计算处构造 (S, B)
        # 若 batch 含 active_mask，叠加过滤 ON_EDGE 步
        raw_am = getattr(first_batch, "active_mask", None)
        if raw_am is not None:
            mask_train = mask_train * raw_am[:, bi:].T                 # (S, B)
        denom = mask_train.sum().clamp(min=1)
        loss  = (td_loss * mask_train).sum() / denom

        info = {
            "q_tot_mean": q_tot.detach().mean(),
            "q_tot_max":  q_tot.detach().max(),
            "td_error":   ((td_target - q_tot).detach().abs() * mask_train).sum() / denom,
        }
        return loss, info

    # ====================================================================
    #                       update
    # ====================================================================

    def update(
        self,
        batch_dict: Dict[str, Union[TransitionBatch, SequenceBatch]],
        **kwargs,
    ) -> TrainingStats:
        batch_dict = {
            aid: batch.to_tensor(self.device)
            for aid, batch in batch_dict.items()
        }

        if self.is_recurrent:
            loss, info = self._compute_loss_seq(batch_dict)
        else:
            loss, info = self._compute_loss_flat(batch_dict)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            params = [p for g in self.optimizer.param_groups for p in g['params']]
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

        global_step = int(kwargs.get("global_step", 0))
        warmup_steps = int(kwargs.get("warmup_steps", 0))
        self._update_epsilon_linear(global_step, warmup_steps)

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._update_target_networks()

        def _as_float(value) -> float:
            return float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)

        return TrainingStats(
            loss=float(loss.detach().item()),
            extra={
                "q_tot_mean": _as_float(info["q_tot_mean"]),
                "q_tot_max": _as_float(info["q_tot_max"]),
                "td_error": _as_float(info["td_error"]),
                "epsilon": self._get_current_epsilon(),
            },
        )

    # ====================================================================
    #                       target network / epsilon
    # ====================================================================

    def _update_target_networks(self):
        """更新 target Q-network。子类 override 以同时更新 target_mixer。"""
        if self.tau < 1.0:
            for tp, sp in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        else:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _update_epsilon_linear(self, global_step: int, warmup_steps: int = 0):
        """按 global-step 线性衰减 epsilon。
        
        warmup_steps 期间（global_step < warmup_steps）epsilon 保持 epsilon_start，
        之后再按 epsilon_decay_steps 线性衰减到 epsilon_end。
        """
        effective_step = max(0, global_step - warmup_steps)
        progress = min(1.0, effective_step / self.epsilon_decay_steps)
        new_eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
        for aid in self.policy.agent_ids:
            vp: ValuePolicy = self.policy.get_policy(aid)
            vp.set_epsilon(new_eps)

    def _get_current_epsilon(self) -> float:
        return self.policy.get_policy(self.policy.agent_ids[0]).get_epsilon()

    def set_epsilon(self, epsilon: float):
        for aid in self.policy.agent_ids:
            self.policy.get_policy(aid).set_epsilon(epsilon)
