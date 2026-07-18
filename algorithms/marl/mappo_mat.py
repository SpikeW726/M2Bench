"""MAT PPO implementation aligned with the original ``asy_ppo.py``.

The collector stores only READY decision steps. Returns are Monte Carlo rather
than GAE, encoder values and advantages are shared scalars across agents, and
policy ratios remain per-agent before averaging over time and environments.
Active-mask loss filtering is unnecessary because inactive steps are not stored.
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
    graph_state:  np.ndarray   # (T, N, G, 3).
    actions:      np.ndarray   # (T, N) int - graph index.
    log_probs:    np.ndarray   # Shape: (T, N, G).
    rewards:      np.ndarray
    shift_action: np.ndarray   # (T, N, N, 2).
    node_last_idx: np.ndarray  # (T, N) int - graph index.
    active_mask:  np.ndarray   # Shape: (T, N).

    last_graph_state: Optional[np.ndarray] = None  # Shape: (N, G, 3).
    last_node_idx:    Optional[np.ndarray] = None  # (N,) int.
    last_active_mask: Optional[np.ndarray] = None  # (N,).
    last_shift:       Optional[np.ndarray] = None  # (N, N, 2).

class MAPPOMATAlgo(BaseAlgorithm):
    def __init__(
        self,
        policy: MATMultiAgentPolicy,
        params,
        n_agents: int = 1,
        **kwargs,
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

        # Adam + ReduceLROnPlateau.
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
        self._episode_reward_buf = []

    # prepare_batch: MC return.

    def prepare_batch(self, batch: MATBatch) -> MATBatch:
        device = self.device
        policy = self.policy

        # 1. bootstrap next_value.
        if batch.last_graph_state is not None:
            last_state = torch.as_tensor(
                batch.last_graph_state[None], dtype=torch.float32, device=device
            )  # (1, N, G, 3).
            with torch.no_grad():
                _, sv = policy.encoder(last_state)   # (1, N, 1).
            next_value = float(sv.mean().cpu().item())
        else:
            next_value = 0.0

        T = len(batch.rewards)
        G_list = []
        for r in reversed(batch.rewards):
            next_value = self.gamma * next_value + float(r)
            G_list.append(next_value)
        G_list.reverse()
        G = np.array(G_list, dtype=np.float32)

        G = (G - G.mean()) / (G.std() + 1e-8)

        batch.rewards = G
        return batch

    # update: per-agent ratio PPO.

    def update(self, batch: MATBatch, **kwargs) -> TrainingStats:
        device = self.device
        policy = self.policy

        T = len(batch.rewards)
        N = self.n_agents
        norm_factor = T * N

        returns = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)  # (T,).
        graph_state = torch.as_tensor(batch.graph_state, dtype=torch.float32, device=device)  # (T,N,G,3).
        shift_action = torch.as_tensor(batch.shift_action, dtype=torch.float32, device=device)  # (T,N,N,2).
        actions = batch.actions   # (T, N) int, numpy.
        node_last_idx = batch.node_last_idx  # (T, N) int, numpy.
        active_mask = batch.active_mask      # (T, N) 0/1, numpy.

        old_log_prob = torch.as_tensor(
            batch.log_probs, dtype=torch.float32, device=device
        )  # (T, N, G).

        act_t = torch.as_tensor(actions, dtype=torch.long, device=device)  # (T, N).
        old_log_prob_act = old_log_prob.gather(
            dim=2, index=act_t.unsqueeze(-1)
        ).squeeze(-1)  # (T, N).
        old_log_prob_act = old_log_prob_act.detach()

        with torch.no_grad():
            _, sv_old = policy.encoder(graph_state)  # (T, N, 1).
        sv_old_mean = sv_old.mean(dim=1).squeeze(-1).detach()  # (T,).

        advantage = returns - sv_old_mean

        episode_reward = float(returns.sum().item())
        self._episode_reward_buf.append(episode_reward)

        loss_last = policy_loss_last = value_loss_last = entropy_last = 0.0
        mse = nn.MSELoss()

        for _ in range(self.policy_update_steps):

            new_log_prob_full, sv_new, entropy_full = policy.evaluate_joint_actions(
                graph_state, node_last_idx, shift_action, active_mask
            )  # (T,N,G), (T,N,1), (T,N,1).

            sv_new_mean = sv_new.mean(dim=1).squeeze(-1)  # (T,).

            new_log_prob_act = new_log_prob_full.gather(
                dim=2, index=act_t.unsqueeze(-1)
            ).squeeze(-1)  # (T, N).

            # policy loss.
            policy_loss = torch.tensor(0.0, device=device)
            for t in range(T):
                ad = advantage[t]
                for k in range(N):
                    lp_new = new_log_prob_act[t, k]
                    lp_old = old_log_prob_act[t, k]
                    ratio = (lp_new - lp_old).exp()
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)

                    policy_loss += torch.max(
                        -ad * ratio,
                        -ad * clipped_ratio,
                    ) / norm_factor

            # value loss.
            sv_old_for_clip = sv_old_mean.detach()
            value_unclipped = mse(sv_new_mean, returns)
            sv_new_clipped = sv_old_for_clip + torch.clamp(
                sv_new_mean - sv_old_for_clip, -0.2, 0.2
            )
            value_clipped = mse(sv_new_clipped, returns)
            value_loss = torch.max(value_clipped, value_unclipped)

            # entropy loss.
            # new_log_prob_full: (T, N, G).
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
