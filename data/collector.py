"""Collect environment experience for reinforcement-learning updates.

``OnPolicyCollector`` and ``OffPolicyCollector`` handle single-agent vector
environments. Their ``MA`` counterparts preserve per-agent batches and replay
buffers. ``MATOnPolicyCollector`` stores only joint READY decision steps for the
autoregressive Multi-Agent Transformer pipeline.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from algorithms.algorithm_base import BaseAlgorithm
from envs.venvs import BaseVectorEnv
from data.batch import RolloutBatch, TransitionBatch, SequenceBatch, CollectResult
from data.buffer import ReplayBuffer, SequenceReplayBuffer

class BaseCollector(ABC):
    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        self.algorithm = algorithm
        self.env = env
        self.num_envs = env.num_envs

        self._obs = None
        self._info = None
        self.profiler = None

        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._current_rewards = np.zeros(self.num_envs)
        self._current_lengths = np.zeros(self.num_envs, dtype=int)

        self._reset_buffer()

    @abstractmethod
    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
    ) -> CollectResult:
        pass

    @abstractmethod
    def _reset_buffer(self):
        pass

    def reset(self, **kwargs) -> Tuple[Any, Any]:
        self._obs, self._info = self.env.reset(**kwargs)
        self._current_rewards = np.zeros(self.num_envs)
        self._current_lengths = np.zeros(self.num_envs, dtype=int)
        self._episode_rewards = []
        self._episode_lengths = []
        return self._obs, self._info

    def reset_buffer(self):
        self._reset_buffer()
        self._episode_rewards = []
        self._episode_lengths = []

    def _handle_done(self, env_idx: int, reward_sum: float, length: int):
        self._episode_rewards.append(reward_sum)
        self._episode_lengths.append(length)
        self._current_rewards[env_idx] = 0.0
        self._current_lengths[env_idx] = 0

    def _profile(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.time_block(name)

# On-Policy Collectors.

class OnPolicyCollector(BaseCollector):
    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        if env.is_parallel_env:
            raise ValueError(
                "OnPolicyCollector supports only Gymnasium environments; "
                "use MAOnPolicyCollector for multi-agent environments"
            )
        super().__init__(algorithm, env)

        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._hidden: Optional[torch.Tensor] = None
        if self._is_recurrent:
            self._init_hidden()

    def _init_hidden(self):
        policy = self.algorithm.policy
        device = getattr(self.algorithm, "device", policy.device)
        self._hidden = policy.actor.get_initial_hidden(self.num_envs, device)

    def _reset_buffer(self):
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[np.ndarray] = []
        self._rew_buf: List[np.ndarray] = []
        self._done_buf: List[np.ndarray] = []
        self._log_prob_buf: List[np.ndarray] = []
        self._action_mask_buf: List[np.ndarray] = []
        self._rnn_hidden_buf: List[np.ndarray] = []

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._is_recurrent:
            self._init_hidden()
        return result

    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
    ) -> CollectResult:
        if n_steps is None and n_episodes is None:
            raise ValueError("Either n_steps or n_episodes must be specified")
        if n_episodes is not None:
            raise NotImplementedError("Episode-based collection is not supported yet")

        if self._obs is None:
            self.reset()

        self.algorithm.set_training_mode(True)
        policy = self.algorithm.policy
        device = self.algorithm.device

        step_count = 0
        while step_count < n_steps:
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
            action_mask = self._extract_action_mask(self._info, device)

            if self._is_recurrent:
                self._rnn_hidden_buf.append(
                    self._hidden.transpose(0, 1).cpu().numpy()
                )

            with self._profile("collect/policy_forward"):
                infer_ctx = torch.no_grad() if self._is_recurrent else torch.inference_mode()
                with infer_ctx:
                    output = policy.forward(
                        obs_t,
                        state=self._hidden if self._is_recurrent else None,
                        action_mask=action_mask,
                    )

                act = output['act'].cpu().numpy()
                log_prob = output['log_prob'].cpu().numpy()

            if self._is_recurrent:
                self._hidden = output['state']

            with self._profile("collect/buffer_write"):
                self._obs_buf.append(self._obs.copy())
                self._act_buf.append(act)
                self._log_prob_buf.append(log_prob)
                if action_mask is not None:
                    self._action_mask_buf.append(action_mask.cpu().numpy())

            with self._profile("collect/env_step"):
                next_obs, rew, term, trunc, info = self.env.step(act)
            done = term | trunc

            self._rew_buf.append(rew)
            self._done_buf.append(done)

            self._current_rewards += rew
            self._current_lengths += 1
            step_count += self.num_envs

            for i in range(self.num_envs):
                if done[i]:
                    self._handle_done(
                        i, self._current_rewards[i], self._current_lengths[i]
                    )
                    if self._is_recurrent:
                        self._hidden[:, i, :] = 0.0

            self._obs = next_obs
            self._info = info

        rnn_h = None
        if self._is_recurrent and self._rnn_hidden_buf:
            rnn_h = np.concatenate(self._rnn_hidden_buf, axis=0)

        with self._profile("collect/batch_build"):
            batch = RolloutBatch(
                obs=np.concatenate(self._obs_buf, axis=0),
                act=np.concatenate(self._act_buf, axis=0),
                rew=np.concatenate(self._rew_buf, axis=0),
                done=np.concatenate(self._done_buf, axis=0).astype(np.float32),
                log_prob=np.concatenate(self._log_prob_buf, axis=0),
                action_mask=(
                    np.concatenate(self._action_mask_buf, axis=0)
                    if self._action_mask_buf
                    else None
                ),
                rnn_hidden=rnn_h,
            )

        return CollectResult(
            batch=batch,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )

    def _extract_action_mask(
        self, info: Optional[np.ndarray], device: torch.device
    ) -> Optional[torch.Tensor]:
        if info is None:
            return None
        masks = []
        for i in range(self.num_envs):
            if isinstance(info[i], dict) and 'action_mask' in info[i]:
                masks.append(info[i]['action_mask'])
        if not masks:
            return None
        return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)

class MAOnPolicyCollector(BaseCollector):
    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        if not env.is_parallel_env:
            raise ValueError(
                "MAOnPolicyCollector supports only ParallelEnv; "
                "use OnPolicyCollector for single-agent environments"
            )
        self.agents = env.agents

        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        super().__init__(algorithm, env)

        self._hidden: Optional[Dict[str, torch.Tensor]] = None
        if self._is_recurrent:
            self._init_hidden()

    def _init_hidden(self):
        policy = self.algorithm.policy
        device = getattr(self.algorithm, "device", policy.device)
        self._hidden = {}
        for agent in self.agents:
            actor = policy.get_policy(agent).actor
            self._hidden[agent] = actor.get_initial_hidden(self.num_envs, device)

    def _reset_buffer(self):
        buf_keys = [
            'obs', 'act', 'rew', 'done', 'truncated', 'log_prob',
            'action_mask', 'active_mask', 'global_state', 'final_global_state',
            'final_obs',
        ]
        if getattr(self, '_is_recurrent', False):
            buf_keys.append('rnn_hidden')
        self._buffers: Dict[str, Dict[str, List]] = {
            agent: {k: [] for k in buf_keys}
            for agent in self.agents
        }

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._is_recurrent:
            self._init_hidden()
        return result

    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
    ) -> CollectResult:
        if n_steps is None and n_episodes is None:
            raise ValueError("Either n_steps or n_episodes must be specified")
        if n_episodes is not None:
            raise NotImplementedError("Episode-based collection is not supported yet")

        if self._obs is None:
            self.reset()

        self.algorithm.set_training_mode(True)
        policy = self.algorithm.policy

        step_count = 0
        while step_count < n_steps:
            with self._profile("collect/state_query"):
                global_states = self._get_global_states()

            if self._is_recurrent:

                hidden_np = torch.stack(
                    [self._hidden[agent].transpose(0, 1) for agent in self.agents],
                    dim=0,
                ).cpu().numpy()
                for agent_idx, agent in enumerate(self.agents):
                    self._buffers[agent]['rnn_hidden'].append(hidden_np[agent_idx])

            with self._profile("collect/policy_forward"):
                actions, outputs, new_hidden = policy.compute_actions(
                    self._obs, self._info, hidden_dict=self._hidden,
                )

            if self._is_recurrent and new_hidden is not None:
                self._hidden = new_hidden

            with self._profile("collect/buffer_write"):

                log_prob_np = torch.stack(
                    [outputs[agent]['log_prob'] for agent in self.agents],
                    dim=0,
                ).cpu().numpy()
                for agent_idx, agent in enumerate(self.agents):
                    buf = self._buffers[agent]
                    buf['obs'].append(self._obs[agent].copy())
                    buf['act'].append(actions[agent].copy())
                    buf['log_prob'].append(log_prob_np[agent_idx])

                    if global_states is not None:
                        buf['global_state'].append(global_states.copy())

                    if self._info is not None and agent in self._info:
                        info_arr = self._info[agent]
                        masks, actives = [], []
                        for i in range(self.num_envs):
                            if 'action_mask' in info_arr[i]:
                                masks.append(info_arr[i]['action_mask'])
                            actives.append(info_arr[i].get('active_mask', 1))
                        if masks:
                            buf['action_mask'].append(np.stack(masks))
                        buf['active_mask'].append(np.array(actives, dtype=np.float32))

            with self._profile("collect/env_step"):
                next_obs, rew, term, trunc, info = self.env.step(actions)

            for agent in self.agents:
                done = term[agent] | trunc[agent]
                self._buffers[agent]['rew'].append(rew[agent])
                self._buffers[agent]['done'].append(done.astype(np.float32))
                self._buffers[agent]['truncated'].append(trunc[agent].astype(np.float32))

            first_agent = self.agents[0]
            final_gs_list = []
            for i in range(self.num_envs):
                if trunc[first_agent][i]:
                    info_i = info[first_agent][i]
                    if isinstance(info_i, dict) and 'final_state' in info_i:
                        final_gs_list.append(info_i['final_state'].copy())
                    else:
                        final_gs_list.append(None)
                else:
                    final_gs_list.append(None)
            for agent in self.agents:
                self._buffers[agent]['final_global_state'].append(final_gs_list)

            # per-agent final obs.
            for agent in self.agents:
                final_obs_list = []
                for i in range(self.num_envs):
                    if trunc[agent][i]:
                        final_obs_list.append(next_obs[agent][i].copy())
                    else:
                        final_obs_list.append(None)
                self._buffers[agent]['final_obs'].append(final_obs_list)

            mean_rew = sum(rew[a] for a in self.agents) / len(self.agents)
            self._current_rewards += mean_rew
            self._current_lengths += 1
            step_count += self.num_envs

            done_arr = term[first_agent] | trunc[first_agent]
            for i in range(self.num_envs):
                if done_arr[i]:
                    self._handle_done(
                        i, self._current_rewards[i], self._current_lengths[i]
                    )

                    if self._is_recurrent:
                        for agent in self.agents:
                            self._hidden[agent][:, i, :] = 0.0

            self._obs = next_obs
            self._info = info

        with self._profile("collect/state_query"):
            boundary_gs = self._get_global_states()

        with self._profile("collect/batch_build"):
            batch_dict: Dict[str, RolloutBatch] = {}
            for agent in self.agents:
                buf = self._buffers[agent]
                final_gs = buf['final_global_state'] if buf['final_global_state'] else None
                rnn_h = None
                if self._is_recurrent and buf.get('rnn_hidden'):
                    rnn_h = np.concatenate(buf['rnn_hidden'], axis=0)
                final_obs = buf['final_obs'] if buf['final_obs'] else None
                batch_dict[agent] = RolloutBatch(
                    obs=np.concatenate(buf['obs'], axis=0),
                    act=np.concatenate(buf['act'], axis=0),
                    rew=np.concatenate(buf['rew'], axis=0),
                    done=np.concatenate(buf['done'], axis=0),
                    truncated=np.concatenate(buf['truncated'], axis=0),
                    log_prob=np.concatenate(buf['log_prob'], axis=0),
                    global_state=(
                        np.concatenate(buf['global_state'], axis=0)
                        if buf['global_state'] else None
                    ),
                    action_mask=(
                        np.concatenate(buf['action_mask'], axis=0)
                        if buf['action_mask'] else None
                    ),
                    active_mask=(
                        np.concatenate(buf['active_mask'], axis=0)
                        if buf['active_mask'] else None
                    ),
                    final_global_state=final_gs,
                    final_obs=final_obs,
                    rnn_hidden=rnn_h,
                    boundary_global_state=boundary_gs,
                )

        return CollectResult(
            batch=batch_dict,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )

    def _get_global_states(self) -> Optional[np.ndarray]:
        if hasattr(self.env, 'call_env_method'):
            states = self.env.call_env_method("state")
        else:
            states = self.env.get_env_attr("state")
            states = [s() if callable(s) else s for s in states]
        if states is None or states[0] is None:
            return None
        return np.stack(states)

# Off-Policy Collectors.

class OffPolicyCollector(BaseCollector):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseVectorEnv,
        buffer: Union["ReplayBuffer", "SequenceReplayBuffer"],
    ):
        if env.is_parallel_env:
            raise ValueError(
                "OffPolicyCollector supports only Gymnasium environments; "
                "use MAOffPolicyCollector for multi-agent environments"
            )
        self.buffer = buffer
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._hidden: Optional[torch.Tensor] = None
        super().__init__(algorithm, env)

    def _init_ep_buf(self) -> dict:
        buf = {"obs": [], "act": [], "rew": [], "next_obs": [], "done": []}
        if getattr(self.buffer, "has_action_mask", False):
            buf["action_mask"] = []
            buf["next_action_mask"] = []
        return buf

    def _reset_buffer(self):
        if getattr(self, '_is_recurrent', False):
            self._episode_buffers: List[dict] = [
                self._init_ep_buf() for _ in range(self.num_envs)
            ]

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._is_recurrent:
            self._init_hidden()
            self._episode_buffers = [
                self._init_ep_buf() for _ in range(self.num_envs)
            ]
        return result

    def _init_hidden(self):
        policy = self.algorithm.policy
        device = getattr(self.algorithm, "device", policy.device)
        self._hidden = policy.q_network.get_initial_hidden(self.num_envs, device)

    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
    ) -> CollectResult:
        if n_steps is None and n_episodes is None:
            raise ValueError("Either n_steps or n_episodes must be specified")
        if n_episodes is not None:
            raise NotImplementedError("Episode-based collection is not supported yet")

        if self._obs is None:
            self.reset()

        policy = self.algorithm.policy
        device = getattr(self.algorithm, 'device', policy.device)

        step_count = 0
        while step_count < n_steps:
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
            action_mask = _extract_action_mask_single(self._info, self.num_envs, device)

            with self._profile("collect/policy_forward"):
                infer_ctx = torch.no_grad() if self._is_recurrent else torch.inference_mode()
                with infer_ctx:
                    output = policy.forward(
                        obs_t,
                        state=self._hidden if self._is_recurrent else None,
                        action_mask=action_mask,
                    )

                act = output['act'].cpu().numpy()
                action_mask_np = action_mask.cpu().numpy() if action_mask is not None else None

            if self._is_recurrent:
                self._hidden = output['state']

            with self._profile("collect/env_step"):
                next_obs, rew, term, trunc, info = self.env.step(act)
            done = term | trunc

            next_action_mask_np = _extract_action_mask_single_np(
                info, self.num_envs
            )

            with self._profile("collect/buffer_write"):
                if self._is_recurrent:
                    for i in range(self.num_envs):
                        eb = self._episode_buffers[i]
                        eb["obs"].append(self._obs[i].copy())
                        eb["act"].append(act[i])
                        eb["rew"].append(rew[i])
                        eb["next_obs"].append(next_obs[i].copy())
                        eb["done"].append(float(done[i]))
                        if action_mask_np is not None:
                            eb["action_mask"].append(action_mask_np[i].copy())
                        if next_action_mask_np is not None:
                            eb["next_action_mask"].append(next_action_mask_np[i].copy())

                        if done[i]:
                            self._flush_episode(i)
                            self._hidden[:, i, :] = 0.0
                else:
                    self.buffer.add_batch(
                        obs=self._obs,
                        act=act,
                        rew=rew,
                        next_obs=next_obs,
                        done=done.astype(np.float32),
                        action_mask=action_mask_np,
                        next_action_mask=next_action_mask_np,
                    )

            self._current_rewards += rew
            self._current_lengths += 1
            step_count += self.num_envs

            for i in range(self.num_envs):
                if done[i]:
                    self._handle_done(
                        i, self._current_rewards[i], self._current_lengths[i]
                    )

            self._obs = next_obs
            self._info = info

        return CollectResult(
            batch=None,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )

    def _flush_episode(self, env_idx: int):
        eb = self._episode_buffers[env_idx]
        if not eb["obs"]:
            return
        episode = {
            "obs": np.stack(eb["obs"]).astype(np.float32),
            "act": np.array(eb["act"], dtype=np.float32),
            "rew": np.array(eb["rew"], dtype=np.float32),
            "next_obs": np.stack(eb["next_obs"]).astype(np.float32),
            "done": np.array(eb["done"], dtype=np.float32),
        }
        if eb.get("action_mask"):
            episode["action_mask"] = np.stack(eb["action_mask"])
            episode["next_action_mask"] = np.stack(eb["next_action_mask"])
        self.buffer.add_episode(episode)
        self._episode_buffers[env_idx] = self._init_ep_buf()

    def sample(self, batch_size: int) -> Union[TransitionBatch, SequenceBatch]:
        return self.buffer.sample(batch_size)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

class MAOffPolicyCollector(BaseCollector):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseVectorEnv,
        buffers: Dict[str, Union["ReplayBuffer", "SequenceReplayBuffer"]],
        collect_state: bool = False,
        sync_mode: bool = False,
        gamma: float = 0.99,
        shared_indices: Optional[bool] = None,
        shared_sync_mode: bool = False,
    ):
        if not env.is_parallel_env:
            raise ValueError(
                "MAOffPolicyCollector supports only ParallelEnv; "
                "use OffPolicyCollector for single-agent environments"
            )
        self.buffers = buffers
        self.agents = env.agents
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._collect_state = collect_state

        self._shared_indices = collect_state if shared_indices is None else shared_indices
        self._sync_mode = sync_mode
        self._shared_sync_mode = shared_sync_mode
        self._gamma = gamma
        self._hidden: Optional[Dict[str, torch.Tensor]] = None
        super().__init__(algorithm, env)

    def _init_ep_buf(self, agent: str) -> dict:
        buf = {"obs": [], "act": [], "rew": [], "next_obs": [], "done": []}
        if getattr(self.buffers.get(agent), "has_action_mask", False):
            buf["action_mask"] = []
            buf["next_action_mask"] = []
        if getattr(self.buffers.get(agent), "has_active_mask", False):
            buf["active_mask"] = []
        if getattr(self.buffers.get(agent), "has_state", False):
            buf["state"] = []
            buf["next_state"] = []
        return buf

    def _reset_buffer(self):
        if getattr(self, '_is_recurrent', False):
            self._episode_buffers: Dict[str, List[dict]] = {
                agent: [self._init_ep_buf(agent) for _ in range(self.num_envs)]
                for agent in self.agents
            }
        if getattr(self, '_sync_mode', False) and not getattr(self, '_is_recurrent', False):
            self._pending: Dict[str, List[Optional[dict]]] = {
                agent: [None] * getattr(self, 'num_envs', 0)
                for agent in getattr(self, 'agents', [])
            }
        if getattr(self, '_shared_sync_mode', False) and not getattr(self, '_is_recurrent', False):
            self._vtd_pending: Dict[str, List[Optional[dict]]] = {
                agent: [None] * getattr(self, 'num_envs', 0)
                for agent in getattr(self, 'agents', [])
            }

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._is_recurrent:
            self._init_hidden()
            self._episode_buffers = {
                agent: [self._init_ep_buf(agent) for _ in range(self.num_envs)]
                for agent in self.agents
            }
        if self._sync_mode and not self._is_recurrent:
            self._pending = {
                agent: [None] * self.num_envs
                for agent in self.agents
            }
        if self._shared_sync_mode and not self._is_recurrent:
            self._vtd_pending = {
                agent: [None] * self.num_envs
                for agent in self.agents
            }
        return result

    def _init_hidden(self):
        policy = self.algorithm.policy
        device = getattr(self.algorithm, "device", policy.device)
        self._hidden = {}
        for agent in self.agents:
            q_net = policy.get_policy(agent).q_network
            self._hidden[agent] = q_net.get_initial_hidden(self.num_envs, device)

    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
    ) -> CollectResult:
        if n_steps is None and n_episodes is None:
            raise ValueError("Either n_steps or n_episodes must be specified")
        if n_episodes is not None:
            raise NotImplementedError("Episode-based collection is not supported yet")

        if self._obs is None:
            self.reset()

        policy = self.algorithm.policy

        step_count = 0
        while step_count < n_steps:
            with self._profile("collect/policy_forward"):
                actions, _outputs, new_hidden = policy.compute_actions(
                    self._obs, self._info,
                    hidden_dict=self._hidden if self._is_recurrent else None,
                )

            if self._is_recurrent and new_hidden is not None:
                self._hidden = new_hidden

            cur_action_masks: Dict[str, Optional[np.ndarray]] = {}
            cur_active_masks: Dict[str, Optional[np.ndarray]] = {}
            for agent in self.agents:
                cur_action_masks[agent] = _extract_agent_action_mask_np(
                    self._info, agent, self.num_envs
                )
                cur_active_masks[agent] = _extract_agent_active_mask_np(
                    self._info, agent, self.num_envs
                )

            with self._profile("collect/state_query"):
                cur_state = self._get_global_states() if self._collect_state else None

            with self._profile("collect/env_step"):
                next_obs, rew, term, trunc, info = self.env.step(actions)

            with self._profile("collect/state_query"):
                next_state = self._get_global_states() if self._collect_state else None

            if self._collect_state and (cur_state is None or next_state is None):
                raise RuntimeError(
                    "Off-policy CTDE requires a global state, but VectorEnv returned no valid state(). "
                    "This would corrupt replay for QMIX-like algorithms; check "
                    "env.call_env_method('state') and the vector wrapper."
                )

            next_active_masks: Dict[str, Optional[np.ndarray]] = {}
            for agent in self.agents:
                next_active_masks[agent] = _extract_agent_active_mask_np(
                    info, agent, self.num_envs
                )

            with self._profile("collect/buffer_write"):
                if self._is_recurrent:
                    for agent in self.agents:
                        next_am = _extract_agent_action_mask_np(info, agent, self.num_envs)
                        am = cur_action_masks[agent]
                        actm = cur_active_masks[agent]
                        for i in range(self.num_envs):
                            eb = self._episode_buffers[agent][i]
                            eb["obs"].append(self._obs[agent][i].copy())
                            eb["act"].append(actions[agent][i])
                            eb["rew"].append(rew[agent][i])

                            if trunc[agent][i] and isinstance(info[agent][i], dict):
                                final_obs = info[agent][i].get("final_obs")
                                next_obs_i = final_obs if final_obs is not None else next_obs[agent][i].copy()
                            else:
                                next_obs_i = next_obs[agent][i].copy()
                            eb["next_obs"].append(next_obs_i)
                            eb["done"].append(float(term[agent][i]))
                            if am is not None:
                                eb["action_mask"].append(am[i].copy())
                            if next_am is not None:
                                eb["next_action_mask"].append(next_am[i].copy())
                            if "active_mask" in eb:
                                eb["active_mask"].append(
                                    actm[i] if actm is not None else 1.0
                                )
                            if "state" in eb:
                                eb["state"].append(
                                    cur_state[i].copy() if cur_state is not None else None
                                )
                            if "next_state" in eb:
                                eb["next_state"].append(
                                    next_state[i].copy() if next_state is not None else None
                                )
                elif self._sync_mode:
                    self._collect_sync_mlp(
                        actions, rew, next_obs, term, trunc, info,
                        cur_action_masks, cur_active_masks, next_active_masks,
                        cur_state, next_state,
                    )
                elif self._shared_sync_mode:
                    self._collect_shared_sync_mlp(
                        actions, rew, next_obs, term, trunc, info,
                        cur_action_masks, cur_active_masks, next_active_masks,
                        cur_state, next_state,
                    )
                else:
                    for agent in self.agents:

                        done_td = term[agent].astype(np.float32)
                        next_obs_td = next_obs[agent].copy()
                        for i in range(self.num_envs):
                            if trunc[agent][i] and isinstance(info[agent][i], dict):
                                final_obs = info[agent][i].get("final_obs")
                                if final_obs is not None:
                                    next_obs_td[i] = final_obs
                        next_am = _extract_agent_action_mask_np(info, agent, self.num_envs)
                        self.buffers[agent].add_batch(
                            obs=self._obs[agent],
                            act=actions[agent],
                            rew=rew[agent],
                            next_obs=next_obs_td,
                            done=done_td,
                            action_mask=cur_action_masks[agent],
                            next_action_mask=next_am,
                            state=cur_state,
                            next_state=next_state,
                            active_mask=cur_active_masks[agent],
                        )

            first_agent = self.agents[0]
            mean_rew = sum(rew[a] for a in self.agents) / len(self.agents)
            self._current_rewards += mean_rew
            self._current_lengths += 1
            step_count += self.num_envs

            done_arr = term[first_agent] | trunc[first_agent]
            for i in range(self.num_envs):
                if done_arr[i]:
                    self._handle_done(
                        i, self._current_rewards[i], self._current_lengths[i]
                    )
                    if self._is_recurrent:
                        self._flush_episodes(i)
                        for agent in self.agents:
                            self._hidden[agent][:, i, :] = 0.0

            self._obs = next_obs
            self._info = info

        return CollectResult(
            batch=None,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )

    def _flush_episodes(self, env_idx: int):
        for agent in self.agents:
            eb = self._episode_buffers[agent][env_idx]
            if not eb["obs"]:
                continue
            episode = {
                "obs": np.stack(eb["obs"]).astype(np.float32),
                "act": np.array(eb["act"], dtype=np.float32),
                "rew": np.array(eb["rew"], dtype=np.float32),
                "next_obs": np.stack(eb["next_obs"]).astype(np.float32),
                "done": np.array(eb["done"], dtype=np.float32),
            }
            if eb.get("action_mask"):
                episode["action_mask"] = np.stack(eb["action_mask"])
                episode["next_action_mask"] = np.stack(eb["next_action_mask"])
            if eb.get("active_mask") is not None and len(eb.get("active_mask", [])) > 0:
                episode["active_mask"] = np.array(eb["active_mask"], dtype=np.float32)
            if eb.get("state") and eb["state"][0] is not None:
                episode["state"] = np.stack(eb["state"]).astype(np.float32)
                episode["next_state"] = np.stack(eb["next_state"]).astype(np.float32)
            self.buffers[agent].add_episode(episode)
            self._episode_buffers[agent][env_idx] = self._init_ep_buf(agent)

    def _collect_shared_sync_mlp(
        self,
        actions: Dict[str, np.ndarray],
        rew: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        term: Dict[str, np.ndarray],
        trunc: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
        cur_action_masks: Dict[str, Optional[np.ndarray]],
        cur_active_masks: Dict[str, Optional[np.ndarray]],
        next_active_masks: Dict[str, Optional[np.ndarray]],
        cur_state: Optional[np.ndarray],
        next_state: Optional[np.ndarray],
    ):
        gamma = self._gamma
        for agent in self.agents:
            done_td = term[agent].astype(np.float32)
            next_obs_td = next_obs[agent].copy()
            for i in range(self.num_envs):
                if trunc[agent][i] and isinstance(info[agent][i], dict):
                    fo = info[agent][i].get("final_obs")
                    if fo is not None:
                        next_obs_td[i] = fo
            next_am = _extract_agent_action_mask_np(info, agent, self.num_envs)
            am = cur_active_masks[agent]
            next_am_arr = next_active_masks[agent]

            for i in range(self.num_envs):
                was_active = (am[i] > 0.5) if am is not None else True
                new_active = (next_am_arr[i] > 0.5) if next_am_arr is not None else True
                is_done = float(done_td[i]) > 0.5 or bool(trunc[agent][i])

                pend = self._vtd_pending[agent][i]
                if pend is not None and pend.get("ptr") is not None:
                    pend["acc_reward"] += pend["gamma_power"] * rew[agent][i]
                    pend["gamma_power"] *= gamma

                if pend is not None and pend.get("ptr") is not None and (
                    new_active or is_done
                ):
                    self.buffers[agent].overwrite(
                        pend["ptr"],
                        rew=pend["acc_reward"],
                        next_obs=next_obs_td[i].copy(),
                        gamma_power=pend["gamma_power"],
                        active_mask=1.0,
                        next_action_mask=(
                            next_am[i].copy() if next_am is not None else None
                        ),
                        next_state=(
                            next_state[i].copy()
                            if next_state is not None
                            else None
                        ),
                        done=float(done_td[i]),
                    )
                    self._vtd_pending[agent][i] = None

                cur_am_f = float(am[i]) if am is not None else 1.0
                ptr_slot = self.buffers[agent].peek_write_index()
                self.buffers[agent].add(
                    obs=self._obs[agent][i],
                    act=actions[agent][i],
                    rew=rew[agent][i],
                    next_obs=next_obs_td[i].copy(),
                    done=float(done_td[i]),
                    action_mask=(
                        cur_action_masks[agent][i].copy()
                        if cur_action_masks[agent] is not None
                        else None
                    ),
                    next_action_mask=(
                        next_am[i].copy() if next_am is not None else None
                    ),
                    state=cur_state[i].copy() if cur_state is not None else None,
                    next_state=(
                        next_state[i].copy()
                        if next_state is not None
                        else None
                    ),
                    active_mask=cur_am_f,
                    gamma_power=float(gamma),
                )

                if was_active:
                    self._vtd_pending[agent][i] = {
                        "ptr": ptr_slot,
                        "acc_reward": 0.0,
                        "gamma_power": 1.0,
                    }
                    p2 = self._vtd_pending[agent][i]
                    p2["acc_reward"] += p2["gamma_power"] * rew[agent][i]
                    p2["gamma_power"] *= gamma

                if is_done:
                    self._vtd_pending[agent][i] = None

    def _collect_sync_mlp(
        self,
        actions: Dict[str, np.ndarray],
        rew: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        term: Dict[str, np.ndarray],
        trunc: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
        cur_action_masks: Dict[str, Optional[np.ndarray]],
        cur_active_masks: Dict[str, Optional[np.ndarray]],
        next_active_masks: Dict[str, Optional[np.ndarray]],
        cur_state: Optional[np.ndarray],
        next_state: Optional[np.ndarray],
    ):
        gamma = self._gamma
        for agent in self.agents:
            done = (term[agent] | trunc[agent]).astype(np.float32)
            am = cur_active_masks[agent]
            next_am_arr = next_active_masks[agent]
            next_am = _extract_agent_action_mask_np(info, agent, self.num_envs)

            for i in range(self.num_envs):
                was_active = (am[i] > 0.5) if am is not None else True

                if was_active:
                    self._pending[agent][i] = {
                        'obs': self._obs[agent][i].copy(),
                        'act': actions[agent][i],
                        'action_mask': (
                            cur_action_masks[agent][i].copy()
                            if cur_action_masks[agent] is not None else None
                        ),
                        'state': cur_state[i].copy() if cur_state is not None else None,
                        'acc_reward': 0.0,
                        'gamma_power': 1.0,
                    }

                pend = self._pending[agent][i]
                if pend is not None:
                    pend['acc_reward'] += pend['gamma_power'] * rew[agent][i]
                    pend['gamma_power'] *= gamma

                new_active = (next_am_arr[i] > 0.5) if next_am_arr is not None else True
                is_done = done[i] > 0.5

                if pend is not None and (new_active or is_done):
                    self.buffers[agent].add(
                        obs=pend['obs'],
                        act=pend['act'],
                        rew=pend['acc_reward'],
                        next_obs=next_obs[agent][i].copy(),
                        done=done[i],
                        action_mask=pend['action_mask'],
                        next_action_mask=(
                            next_am[i].copy() if next_am is not None else None
                        ),
                        state=pend['state'],
                        next_state=(
                            next_state[i].copy() if next_state is not None else None
                        ),
                        gamma_power=pend['gamma_power'],
                    )
                    self._pending[agent][i] = None

                if is_done:
                    self._pending[agent][i] = None

    def _get_global_states(self) -> Optional[np.ndarray]:
        if hasattr(self.env, 'call_env_method'):
            states = self.env.call_env_method("state")
        else:
            states = self.env.get_env_attr("state")
            states = [s() if callable(s) else s for s in states]
        if states is None or states[0] is None:
            return None
        return np.stack(states)

    def sample(self, batch_size: int) -> Dict[str, Union[TransitionBatch, SequenceBatch]]:
        first_buf = next(iter(self.buffers.values()))
        # shared_indices: VDN/QMIX.

        if self._shared_indices:
            indices = np.random.choice(len(first_buf), size=batch_size, replace=False)
            return {
                aid: buf.sample_by_indices(indices)
                for aid, buf in self.buffers.items()
            }
        return {
            aid: buf.sample(batch_size)
            for aid, buf in self.buffers.items()
        }

    def can_sample(self, batch_size: int) -> bool:
        return all(len(buf) >= batch_size for buf in self.buffers.values())

def _extract_action_mask_single(
    info: Optional[np.ndarray],
    num_envs: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if info is None:
        return None
    masks = []
    for i in range(num_envs):
        if isinstance(info[i], dict) and 'action_mask' in info[i]:
            masks.append(info[i]['action_mask'])
    if not masks:
        return None
    return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)

def _extract_action_mask_single_np(
    info: Optional[np.ndarray],
    num_envs: int,
) -> Optional[np.ndarray]:
    if info is None:
        return None
    masks = []
    for i in range(num_envs):
        if isinstance(info[i], dict) and 'action_mask' in info[i]:
            masks.append(info[i]['action_mask'])
    if not masks:
        return None
    return np.stack(masks)

def _extract_agent_action_mask_np(
    info_dict: Optional[Dict[str, np.ndarray]],
    agent: str,
    num_envs: int,
) -> Optional[np.ndarray]:
    if info_dict is None or agent not in info_dict:
        return None
    info_arr = info_dict[agent]
    masks = []
    for i in range(num_envs):
        if isinstance(info_arr[i], dict) and 'action_mask' in info_arr[i]:
            masks.append(info_arr[i]['action_mask'])
    if not masks:
        return None
    return np.stack(masks)

def _extract_agent_active_mask_np(
    info_dict: Optional[Dict[str, np.ndarray]],
    agent: str,
    num_envs: int,
) -> Optional[np.ndarray]:
    if info_dict is None or agent not in info_dict:
        return None
    info_arr = info_dict[agent]
    masks = []
    for i in range(num_envs):
        if isinstance(info_arr[i], dict):
            masks.append(float(info_arr[i].get('active_mask', 1)))
        else:
            masks.append(1.0)
    return np.array(masks, dtype=np.float32)

Collector = OnPolicyCollector
MACollector = MAOnPolicyCollector

class MATOnPolicyCollector(BaseCollector):
    """Collect joint READY decisions for Multi-Agent Transformer PPO.

    Unlike standard collectors, it stores only event times at which at least one
    agent can decide. Graph state, active masks, autoregressive action context,
    joint log-probabilities, rewards, and done flags remain aligned per decision.
    """

    def __init__(self, algorithm, env: "BaseVectorEnv", n_agents: int):
        from algorithms.marl.mappo_mat import MATBatch
        self._MATBatch = MATBatch
        self.n_agents = n_agents

        self._last_shift: Optional[torch.Tensor] = None

        self._change_rewards: Optional[np.ndarray] = None
        super().__init__(algorithm, env)

    def _reset_buffer(self):
        self._graph_state_buf: List[np.ndarray] = []   # (N, G, 3).
        self._actions_buf: List[np.ndarray] = []        # (N,) int.
        self._log_probs_buf: List[np.ndarray] = []      # (N, G).
        self._rewards_buf: List[float] = []
        self._shift_buf: List[np.ndarray] = []          # (N, N, 2).
        self._node_idx_buf: List[np.ndarray] = []       # (N,) int.
        self._active_mask_buf: List[np.ndarray] = []    # (N,).

        self._last_graph_state: Optional[np.ndarray] = None
        self._last_node_idx: Optional[np.ndarray] = None
        self._last_active_mask: Optional[np.ndarray] = None
        self._last_shift_np: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        result = super().reset(**kwargs)

        device = self.algorithm.device
        self._last_shift = torch.zeros(
            self.num_envs, self.n_agents, self.n_agents, 2,
            dtype=torch.float32, device=device,
        )
        self._change_rewards = np.zeros(self.num_envs, dtype=np.float32)
        return result

    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
    ) -> "CollectResult":
        if n_steps is None:
            raise ValueError("MATOnPolicyCollector currently supports only n_steps mode")
        if self.num_envs != 1:
            raise ValueError(
                "MATOnPolicyCollector currently requires num_envs=1; "
                f"received num_envs={self.num_envs}. Set training.num_envs to 1 in the experiment YAML."
            )
        if self._obs is None:
            self.reset()

        policy = self.algorithm.policy
        device = self.algorithm.device

        self.algorithm.set_training_mode(False)

        sim_step_count = 0

        while sim_step_count < n_steps:

            with self._profile("collect/state_query"):
                graph_states = self.env.call_env_method("state_mat")      # List[(N,G,3)].
                node_idxs = self.env.call_env_method("get_current_node_indices")  # List[(N,)].

            graph_state = graph_states[0]   # (N, G, 3).
            node_idx    = node_idxs[0]      # (N,) int32.

            active_mask = self._extract_active_mask_all_agents()  # (N,) float.

            graph_state_batch = graph_state[np.newaxis]  # (1, N, G, 3).
            node_idx_batch    = node_idx[np.newaxis]      # (1, N).
            am_batch          = active_mask[np.newaxis]   # (1, N).
            last_shift_env    = self._last_shift[0:1]     # (1, N, N, 2).

            with self._profile("collect/policy_forward"):
                actions_np, log_probs_t, _, shift_new = policy.compute_joint_actions(
                    graph_state_batch, node_idx_batch, am_batch, last_shift_env
                )
            # actions_np: (1, N) log_probs_t: (1, N, G) shift_new: (1, N, N, 2).

            actions_env = actions_np[0]           # (N,) int.
            log_probs_env = log_probs_t[0].cpu().numpy()  # (N, G).
            self._last_shift[0] = shift_new[0]

            with self._profile("collect/state_query"):
                actions_env_mapped = self._map_graph_idx_to_env_actions(actions_env, node_idx)

            with self._profile("collect/env_step"):
                obs_dict, rew_dict, term_dict, trunc_dict, info_dict =\
                    self.env.step(actions_env_mapped)

            shared_rew = self._get_shared_reward(rew_dict)

            self._change_rewards[0] += shared_rew
            sim_step_count += self.num_envs

            done = self._check_done(term_dict, trunc_dict)
            if done:
                self._handle_done(0, self._current_rewards[0], self._current_lengths[0])

            self._current_rewards[0] += shared_rew
            self._current_lengths[0] += 1

            any_ready = bool(active_mask.any())
            if any_ready:
                with self._profile("collect/buffer_write"):
                    self._graph_state_buf.append(graph_state.copy())
                    self._actions_buf.append(actions_env.copy())
                    self._log_probs_buf.append(log_probs_env.copy())
                    self._rewards_buf.append(float(self._change_rewards[0]))
                    self._shift_buf.append(shift_new[0].cpu().numpy().copy())
                    self._node_idx_buf.append(node_idx.copy())
                    self._active_mask_buf.append(active_mask.copy())
                    self._change_rewards[0] = 0.0

            if done:
                self._change_rewards[0] = 0.0
                self._last_shift[0] = torch.zeros(
                    self.n_agents, self.n_agents, 2,
                    dtype=torch.float32, device=device,
                )

            self._obs = obs_dict
            self._info = info_dict

        with self._profile("collect/state_query"):
            last_gs = self.env.call_env_method("state_mat")[0]
            last_ni = self.env.call_env_method("get_current_node_indices")[0]
        last_am = self._extract_active_mask_all_agents()
        self._last_graph_state = last_gs
        self._last_node_idx = last_ni
        self._last_active_mask = last_am
        self._last_shift_np = self._last_shift[0].cpu().numpy()

        with self._profile("collect/batch_build"):
            T = len(self._rewards_buf)
            if T == 0:

                batch = self._MATBatch(
                    graph_state=np.zeros((1, self.n_agents, 1, 3), dtype=np.float32),
                    actions=np.zeros((1, self.n_agents), dtype=np.int64),
                    log_probs=np.zeros((1, self.n_agents, 1), dtype=np.float32),
                    rewards=np.zeros(1, dtype=np.float32),
                    shift_action=np.zeros((1, self.n_agents, self.n_agents, 2), dtype=np.float32),
                    node_last_idx=np.zeros((1, self.n_agents), dtype=np.int32),
                    active_mask=np.ones((1, self.n_agents), dtype=np.float32),
                )
            else:
                batch = self._MATBatch(
                    graph_state=np.stack(self._graph_state_buf),   # (T,N,G,3).
                    actions=np.stack(self._actions_buf),            # (T,N).
                    log_probs=np.stack(self._log_probs_buf),        # (T,N,G).
                    rewards=np.array(self._rewards_buf, dtype=np.float32),  # (T,).
                    shift_action=np.stack(self._shift_buf),         # (T,N,N,2).
                    node_last_idx=np.stack(self._node_idx_buf),     # (T,N).
                    active_mask=np.stack(self._active_mask_buf),    # (T,N).
                    last_graph_state=self._last_graph_state,
                    last_node_idx=self._last_node_idx,
                    last_active_mask=self._last_active_mask,
                    last_shift=self._last_shift_np,
                )

        return CollectResult(
            batch=batch,
            n_steps=sim_step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )

    def _extract_active_mask_all_agents(self) -> np.ndarray:
        info = self._info
        mask = np.ones(self.n_agents, dtype=np.float32)
        if info is None:
            return mask
        for k in range(self.n_agents):
            agent_key = f"agent_{k}"
            if agent_key in info:
                arr = info[agent_key]
                if isinstance(arr, np.ndarray) and len(arr) > 0:
                    entry = arr[0]
                elif isinstance(arr, dict):
                    entry = arr
                else:
                    continue
                if isinstance(entry, dict):
                    mask[k] = float(entry.get("active_mask", 1))
        return mask

    def _map_graph_idx_to_env_actions(
        self,
        graph_idx_actions: np.ndarray,  # (N,) int - graph index.
        current_node_idx: np.ndarray,   # (N,) int.
    ) -> Dict[str, np.ndarray]:
        results = self.env.call_env_method(
            "graph_idx_to_action",
            graph_idx_actions.tolist(),
        )
        # results[0]: Dict[str, int].
        per_env_dict = results[0]

        return {
            agent: np.array([act], dtype=np.int64)
            for agent, act in per_env_dict.items()
        }

    def _get_shared_reward(self, rew_dict) -> float:
        if rew_dict is None:
            return 0.0
        for key in ("agent_0", "agent_0"):
            if key in rew_dict:
                arr = rew_dict[key]
                if isinstance(arr, np.ndarray):
                    return float(arr[0])
                return float(arr)
        return 0.0

    def _check_done(self, term_dict, trunc_dict) -> bool:
        if term_dict is None and trunc_dict is None:
            return False
        for d in (term_dict, trunc_dict):
            if d is None:
                continue
            for v in d.values():
                if isinstance(v, np.ndarray) and len(v) > 0 and v[0]:
                    return True
                elif isinstance(v, bool) and v:
                    return True
        return False
