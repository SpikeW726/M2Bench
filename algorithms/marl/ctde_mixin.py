"""Centralized-critic helpers shared by CTDE algorithms.

This stateless mixin expects an ``ActorCriticOnPolicyAlgo`` subclass to provide
the critic and device. MAPPO and MAA2C use it for agent-identity concatenation,
truncation bootstrapping, recurrent critic evaluation, and CTDE batch preparation.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from data.batch import RolloutBatch

class CentralizedCriticMixin:

    # Truncation Bootstrap.
    def _compute_trunc_bootstrap_with_onehot(
        self,
        truncateds: Optional[torch.Tensor],     # (T, N) or None.
        final_global_states,                     # T x N nested list.
        agent_idx: int,
        num_agents: int,
        T: int,
        N: int,
    ) -> torch.Tensor:
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
            critic_in = torch.cat([states_t, one_hot], dim=-1)

            if self.is_critic_recurrent:
                zero_h = self.critic.get_initial_hidden(len(batch_states), device)
                v_norm, _ = self.critic(critic_in, zero_h)
                v_norm = v_norm.squeeze(-1)
            else:
                v_norm = self.critic(critic_in).squeeze(-1)

            if self.use_value_norm and self.ret_rms is not None:
                v_real = v_norm * self.ret_rms.std + self.ret_rms.mean
            else:
                v_real = v_norm

            for vi, k in enumerate(valid_k_indices):
                t_idx = trunc_positions[k, 0].item()
                e_idx = trunc_positions[k, 1].item()
                trunc_bootstrap[t_idx, e_idx] = v_real[vi]

        return trunc_bootstrap

    def _critic_rnn_values_with_onehot(
        self,
        critic_input: torch.Tensor,    # Shape: (T*N, critic_dim).
        done_2d: torch.Tensor,          # (T, N).
        agent_key: str,
        T: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_seq = critic_input.view(T, N, -1)

        if not hasattr(self, "_critic_hidden_per_agent"):
            self._critic_hidden_per_agent: Dict[str, torch.Tensor] = {}

        hidden = self._critic_hidden_per_agent.get(agent_key, None)
        if hidden is None or hidden.shape[1] != N:
            hidden = self.critic.get_initial_hidden(N, critic_input.device)

        all_values = []
        all_hidden = []

        for t in range(T):
            if t > 0:
                done_prev = done_2d[t - 1] > 0.5
                if done_prev.any():
                    hidden = hidden.clone()
                    hidden[:, done_prev, :] = 0.0

            all_hidden.append(hidden)
            v_t, hidden = self.critic(critic_seq[t], hidden)
            all_values.append(v_t.squeeze(-1))

        self._critic_hidden_per_agent[agent_key] = hidden.detach()

        values_norm = torch.stack(all_values, dim=0)        # (T, N).
        critic_rnn_h = torch.stack(all_hidden, dim=0)       # (T, rN, N, H).
        return values_norm, critic_rnn_h

    # CTDE prepare_batch.

    def _prepare_batch_ctde(
        self,
        batch_dict: Dict[str, RolloutBatch],
    ) -> RolloutBatch:
        agents = list(batch_dict.keys())
        num_agents = len(agents)
        N = self.num_envs

        all_obs, all_act, all_log_prob = [], [], []
        all_adv, all_ret, all_value = [], [], []
        all_critic_input, all_action_mask, all_active_mask = [], [], []
        all_rnn_hidden = []
        all_critic_rnn_hidden = []

        saved_T = 0

        for i, agent in enumerate(agents):
            batch = batch_dict[agent]
            final_gs = batch.final_global_state
            batch = batch.to_tensor(self.device)
            total_size = batch.global_state.shape[0]
            T = total_size // N
            saved_T = T

            one_hot = torch.zeros(total_size, num_agents, device=self.device)
            one_hot[:, i] = 1.0
            critic_input = torch.cat([batch.global_state, one_hot], dim=-1)

            rew_2d = batch.rew.view(T, N)
            done_2d = batch.done.view(T, N)
            truncated_2d = batch.truncated.view(T, N) if batch.truncated is not None else None

            active_mask_2d = None
            if self.use_active_mask and batch.active_mask is not None:
                active_mask_2d = batch.active_mask.view(T, N)

            with torch.no_grad():
                if self.is_critic_recurrent:
                    values_norm, critic_rnn_h = self._critic_rnn_values_with_onehot(
                        critic_input, done_2d, agent, T, N,
                    )
                    rN, H = critic_rnn_h.shape[1], critic_rnn_h.shape[3]
                    all_critic_rnn_hidden.append(
                        critic_rnn_h.permute(0, 2, 1, 3).reshape(T * N, rN, H)
                    )
                else:
                    values_norm = self.critic(critic_input).squeeze(-1).view(T, N)

                if self.use_value_norm and self.ret_rms is not None:
                    values = values_norm * self.ret_rms.std + self.ret_rms.mean
                else:
                    values = values_norm

                trunc_bootstrap = self._compute_trunc_bootstrap_with_onehot(
                    truncated_2d, final_gs, i, num_agents, T, N,
                )

                adv, ret = self._gae_vectorized(
                    rew_2d, values, done_2d, truncated_2d, trunc_bootstrap, active_mask_2d,
                )
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
            if self.use_active_mask and batch.active_mask is not None:
                all_active_mask.append(batch.active_mask)
            if batch.rnn_hidden is not None:
                all_rnn_hidden.append(batch.rnn_hidden)

        if self.use_value_norm and self.ret_rms is not None:
            all_ret_tensor = torch.cat(all_ret, dim=0)
            if all_active_mask:
                active_flat = torch.cat(all_active_mask, dim=0) > 0.5
                active_ret = all_ret_tensor[active_flat]
                if active_ret.numel() > 0:
                    self.ret_rms.update(active_ret)
            else:
                self.ret_rms.update(all_ret_tensor)

        self._last_T = saved_T
        self._last_N = N
        self._last_num_agents = num_agents

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
            critic_rnn_hidden=torch.cat(all_critic_rnn_hidden, dim=0) if all_critic_rnn_hidden else None,
        )
