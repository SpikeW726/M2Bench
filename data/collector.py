"""数据采集器：从环境中采集数据用于 RL 训练。

On-policy:
    OnPolicyCollector       ← 单智能体，返回 RolloutBatch
    MAOnPolicyCollector     ← 多智能体，返回 Dict[str, RolloutBatch]

Off-policy:
    OffPolicyCollector      ← 单智能体，内持 ReplayBuffer
    MAOffPolicyCollector    ← 多智能体，内持 Dict[str, ReplayBuffer]
"""

from abc import ABC, abstractmethod
from re import M
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from algorithms.algorithm_base import BaseAlgorithm
from envs.venvs import BaseVectorEnv
from data.batch import RolloutBatch, TransitionBatch, SequenceBatch, CollectResult
from data.buffer import ReplayBuffer, EpisodeReplayBuffer


class BaseCollector(ABC):
    """
    采集器基类，定义公共接口。

    Args:
        algorithm: RL 算法实例（包含 policy）
        env: 向量化环境
    """

    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        self.algorithm = algorithm
        self.env = env
        self.num_envs = env.num_envs

        # 当前 obs（reset 后保存）
        self._obs = None
        self._info = None

        # Episode 统计
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
        """重置内部缓冲区"""
        pass

    def reset(self, **kwargs) -> Tuple[Any, Any]:
        """重置环境并返回初始 obs"""
        self._obs, self._info = self.env.reset(**kwargs)
        self._current_rewards = np.zeros(self.num_envs)
        self._current_lengths = np.zeros(self.num_envs, dtype=int)
        self._episode_rewards = []
        self._episode_lengths = []
        return self._obs, self._info

    def reset_buffer(self):
        """清空 buffer（供 Trainer 调用）"""
        self._reset_buffer()
        self._episode_rewards = []
        self._episode_lengths = []

    def _handle_done(self, env_idx: int, reward_sum: float, length: int):
        self._episode_rewards.append(reward_sum)
        self._episode_lengths.append(length)
        self._current_rewards[env_idx] = 0.0
        self._current_lengths[env_idx] = 0


# =============================================================================
#                          On-Policy Collectors
# =============================================================================

class OnPolicyCollector(BaseCollector):
    """
    单智能体 on-policy 采集器。

    处理 Gymnasium Env（ndarray 格式 I/O），返回 RolloutBatch。
    RNN 时维护 per-env hidden state 并在 episode 边界重置。
    """

    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        if env.is_parallel_env:
            raise ValueError(
                "OnPolicyCollector 只支持 Gymnasium Env，"
                "多智能体请用 MAOnPolicyCollector"
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
            raise ValueError("必须指定 n_steps 或 n_episodes")
        if n_episodes is not None:
            raise NotImplementedError("暂不支持按 episode 采集")

        if self._obs is None:
            self.reset()

        self.algorithm.set_training_mode(True)
        policy = self.algorithm.policy
        device = self.algorithm.device

        step_count = 0
        while step_count < n_steps:
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
            action_mask = self._extract_action_mask(self._info, device)

            # RNN: 存储当前步 hidden
            if self._is_recurrent:
                self._rnn_hidden_buf.append(
                    self._hidden.transpose(0, 1).cpu().numpy()
                )

            with torch.no_grad():
                output = policy.forward(
                    obs_t,
                    state=self._hidden if self._is_recurrent else None,
                    action_mask=action_mask,
                )

            act = output['act'].cpu().numpy()
            log_prob = output['log_prob'].cpu().numpy()

            # RNN: 更新 hidden
            if self._is_recurrent:
                self._hidden = output['state']

            self._obs_buf.append(self._obs.copy())
            self._act_buf.append(act)
            self._log_prob_buf.append(log_prob)
            if action_mask is not None:
                self._action_mask_buf.append(action_mask.cpu().numpy())

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
    """
    多智能体 on-policy 采集器。

    处理 PettingZoo ParallelEnv（Dict 格式 I/O），
    支持 global_state 采集用于 MAPPO centralized critic。
    RNN 时维护 per-agent per-env hidden state 并在 episode 边界重置。
    """

    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        if not env.is_parallel_env:
            raise ValueError(
                "MAOnPolicyCollector 只支持 ParallelEnv，"
                "单智能体请用 OnPolicyCollector"
            )
        self.agents = env.agents
        super().__init__(algorithm, env)

        # RNN hidden state 管理
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._hidden: Optional[Dict[str, torch.Tensor]] = None
        if self._is_recurrent:
            self._init_hidden()

    def _init_hidden(self):
        """初始化所有 agent 的 hidden state 为零。"""
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
            raise ValueError("必须指定 n_steps 或 n_episodes")
        if n_episodes is not None:
            raise NotImplementedError("暂不支持按 episode 采集")

        if self._obs is None:
            self.reset()

        self.algorithm.set_training_mode(True)
        policy = self.algorithm.policy

        step_count = 0
        while step_count < n_steps:
            global_states = self._get_global_states()

            # RNN: 存储当前步的 hidden（作为该步的 initial hidden）
            if self._is_recurrent:
                for agent in self.agents:
                    # (recurrent_N, num_envs, H) → (num_envs, recurrent_N, H)
                    self._buffers[agent]['rnn_hidden'].append(
                        self._hidden[agent].transpose(0, 1).cpu().numpy()
                    )

            actions, outputs, new_hidden = policy.compute_actions(
                self._obs, self._info, hidden_dict=self._hidden,
            )

            # RNN: 更新 hidden
            if self._is_recurrent and new_hidden is not None:
                self._hidden = new_hidden

            for agent in self.agents:
                buf = self._buffers[agent]
                buf['obs'].append(self._obs[agent].copy())
                buf['act'].append(actions[agent].copy())
                buf['log_prob'].append(outputs[agent]['log_prob'].cpu().numpy())

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

            self._current_rewards += rew[first_agent]
            self._current_lengths += 1
            step_count += self.num_envs

            done_arr = term[first_agent] | trunc[first_agent]
            for i in range(self.num_envs):
                if done_arr[i]:
                    self._handle_done(
                        i, self._current_rewards[i], self._current_lengths[i]
                    )
                    # RNN: episode 结束时重置对应 env 的 hidden
                    if self._is_recurrent:
                        for agent in self.agents:
                            self._hidden[agent][:, i, :] = 0.0

            self._obs = next_obs
            self._info = info

        batch_dict: Dict[str, RolloutBatch] = {}
        for agent in self.agents:
            buf = self._buffers[agent]
            final_gs = buf['final_global_state'] if buf['final_global_state'] else None
            rnn_h = None
            if self._is_recurrent and buf.get('rnn_hidden'):
                rnn_h = np.concatenate(buf['rnn_hidden'], axis=0)
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
                rnn_hidden=rnn_h,
            )

        return CollectResult(
            batch=batch_dict,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )

    def _get_global_states(self) -> Optional[np.ndarray]:
        try:
            if hasattr(self.env, 'call_env_method'):
                states = self.env.call_env_method("state")
            else:
                states = self.env.get_env_attr("state")
                states = [s() if callable(s) else s for s in states]
            if states is None or states[0] is None:
                return None
            return np.stack(states)
        except Exception:
            return None


# =============================================================================
#                          Off-Policy Collectors
# =============================================================================

class OffPolicyCollector(BaseCollector):
    """
    单智能体 off-policy 采集器。

    MLP 模式: 内持 ReplayBuffer，逐 step 存入 transition。
    RNN 模式: 内持 EpisodeReplayBuffer，按 episode 存储。

    collect() 将 transitions 存入 buffer，返回 CollectResult（仅统计量）。
    Trainer 通过 sample() / can_sample() 从 buffer 采样。
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseVectorEnv,
        buffer: Union["ReplayBuffer", "EpisodeReplayBuffer"],
    ):
        if env.is_parallel_env:
            raise ValueError(
                "OffPolicyCollector 只支持 Gymnasium Env，"
                "多智能体请用 MAOffPolicyCollector"
            )
        self.buffer = buffer
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._hidden: Optional[torch.Tensor] = None
        super().__init__(algorithm, env)

    def _reset_buffer(self):
        # RNN: 初始化 per-env episode 临时缓冲区
        if getattr(self, '_is_recurrent', False):
            self._episode_buffers: List[List[dict]] = [[] for _ in range(self.num_envs)]

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._is_recurrent:
            self._init_hidden()
            self._episode_buffers = [[] for _ in range(self.num_envs)]
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
            raise ValueError("必须指定 n_steps 或 n_episodes")
        if n_episodes is not None:
            raise NotImplementedError("暂不支持按 episode 采集")

        if self._obs is None:
            self.reset()

        policy = self.algorithm.policy
        device = getattr(self.algorithm, 'device', policy.device)

        step_count = 0
        while step_count < n_steps:
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
            action_mask = _extract_action_mask_single(self._info, self.num_envs, device)

            with torch.no_grad():
                output = policy.forward(
                    obs_t,
                    state=self._hidden if self._is_recurrent else None,
                    action_mask=action_mask,
                )

            act = output['act'].cpu().numpy()
            action_mask_np = action_mask.cpu().numpy() if action_mask is not None else None

            if self._is_recurrent:
                self._hidden = output['state']

            next_obs, rew, term, trunc, info = self.env.step(act)
            done = term | trunc

            next_action_mask_np = _extract_action_mask_single_np(
                info, self.num_envs
            )

            if self._is_recurrent:
                # RNN: 累积到 per-env episode buffer
                for i in range(self.num_envs):
                    transition = {
                        "obs": self._obs[i].copy(),
                        "act": act[i],
                        "rew": rew[i],
                        "next_obs": next_obs[i].copy(),
                        "done": float(done[i]),
                    }
                    if action_mask_np is not None:
                        transition["action_mask"] = action_mask_np[i].copy()
                    if next_action_mask_np is not None:
                        transition["next_action_mask"] = next_action_mask_np[i].copy()
                    self._episode_buffers[i].append(transition)

                    if done[i]:
                        self._flush_episode(i)
                        self._hidden[:, i, :] = 0.0
            else:
                # MLP: 逐 step 批量写入
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
        """将 env_idx 的 episode buffer 合并并推入 EpisodeReplayBuffer。"""
        ep_buf = self._episode_buffers[env_idx]
        if not ep_buf:
            return
        episode = {
            "obs": np.array([t["obs"] for t in ep_buf], dtype=np.float32),
            "act": np.array([t["act"] for t in ep_buf], dtype=np.float32),
            "rew": np.array([t["rew"] for t in ep_buf], dtype=np.float32),
            "next_obs": np.array([t["next_obs"] for t in ep_buf], dtype=np.float32),
            "done": np.array([t["done"] for t in ep_buf], dtype=np.float32),
        }
        if "action_mask" in ep_buf[0]:
            episode["action_mask"] = np.array([t["action_mask"] for t in ep_buf])
            episode["next_action_mask"] = np.array([t["next_action_mask"] for t in ep_buf])
        self.buffer.add_episode(episode)
        self._episode_buffers[env_idx] = []

    def sample(self, batch_size: int, seq_len: int = 0) -> Union[TransitionBatch, SequenceBatch]:
        if isinstance(self.buffer, EpisodeReplayBuffer):
            return self.buffer.sample(batch_size, seq_len)
        return self.buffer.sample(batch_size)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


class MAOffPolicyCollector(BaseCollector):
    """
    多智能体 off-policy 采集器。

    MLP 模式: 内持 Dict[str, ReplayBuffer]，逐 step 存入。
    RNN 模式: 内持 Dict[str, EpisodeReplayBuffer]，按 episode 存储。

    每个 agent 的 transitions 独立存入对应 buffer。
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseVectorEnv,
        buffers: Dict[str, Union["ReplayBuffer", "EpisodeReplayBuffer"]],
    ):
        if not env.is_parallel_env:
            raise ValueError(
                "MAOffPolicyCollector 只支持 ParallelEnv，"
                "单智能体请用 OffPolicyCollector"
            )
        self.buffers = buffers
        self.agents = env.agents
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._hidden: Optional[Dict[str, torch.Tensor]] = None
        super().__init__(algorithm, env)

    def _reset_buffer(self):
        if getattr(self, '_is_recurrent', False):
            # per-agent per-env episode 临时缓冲区
            self._episode_buffers: Dict[str, List[List[dict]]] = {
                agent: [[] for _ in range(self.num_envs)]
                for agent in self.agents
            }

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._is_recurrent:
            self._init_hidden()
            self._episode_buffers = {
                agent: [[] for _ in range(self.num_envs)]
                for agent in self.agents
            }
        return result

    def _init_hidden(self):
        """初始化所有 agent 的 hidden state 为零。"""
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
            raise ValueError("必须指定 n_steps 或 n_episodes")
        if n_episodes is not None:
            raise NotImplementedError("暂不支持按 episode 采集")

        if self._obs is None:
            self.reset()

        policy = self.algorithm.policy

        step_count = 0
        while step_count < n_steps:
            actions, _outputs, new_hidden = policy.compute_actions(
                self._obs, self._info,
                hidden_dict=self._hidden if self._is_recurrent else None,
            )

            if self._is_recurrent and new_hidden is not None:
                self._hidden = new_hidden

            cur_action_masks: Dict[str, Optional[np.ndarray]] = {}
            for agent in self.agents:
                cur_action_masks[agent] = _extract_agent_action_mask_np(
                    self._info, agent, self.num_envs
                )

            next_obs, rew, term, trunc, info = self.env.step(actions)

            if self._is_recurrent:
                # RNN: 累积到 per-agent per-env episode buffer
                for agent in self.agents:
                    done = (term[agent] | trunc[agent]).astype(np.float32)
                    next_am = _extract_agent_action_mask_np(info, agent, self.num_envs)
                    for i in range(self.num_envs):
                        transition = {
                            "obs": self._obs[agent][i].copy(),
                            "act": actions[agent][i],
                            "rew": rew[agent][i],
                            "next_obs": next_obs[agent][i].copy(),
                            "done": done[i],
                        }
                        am = cur_action_masks[agent]
                        if am is not None:
                            transition["action_mask"] = am[i].copy()
                        if next_am is not None:
                            transition["next_action_mask"] = next_am[i].copy()
                        self._episode_buffers[agent][i].append(transition)
            else:
                # MLP: 逐 step 批量写入
                for agent in self.agents:
                    done = (term[agent] | trunc[agent]).astype(np.float32)
                    next_am = _extract_agent_action_mask_np(info, agent, self.num_envs)
                    self.buffers[agent].add_batch(
                        obs=self._obs[agent],
                        act=actions[agent],
                        rew=rew[agent],
                        next_obs=next_obs[agent],
                        done=done,
                        action_mask=cur_action_masks[agent],
                        next_action_mask=next_am,
                    )

            first_agent = self.agents[0]
            self._current_rewards += rew[first_agent]
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
        """将 env_idx 的所有 agent episode buffer 合并推入对应 EpisodeReplayBuffer。"""
        for agent in self.agents:
            ep_buf = self._episode_buffers[agent][env_idx]
            if not ep_buf:
                continue
            episode = {
                "obs": np.array([t["obs"] for t in ep_buf], dtype=np.float32),
                "act": np.array([t["act"] for t in ep_buf], dtype=np.float32),
                "rew": np.array([t["rew"] for t in ep_buf], dtype=np.float32),
                "next_obs": np.array([t["next_obs"] for t in ep_buf], dtype=np.float32),
                "done": np.array([t["done"] for t in ep_buf], dtype=np.float32),
            }
            if "action_mask" in ep_buf[0]:
                episode["action_mask"] = np.array([t["action_mask"] for t in ep_buf])
                episode["next_action_mask"] = np.array([t["next_action_mask"] for t in ep_buf])
            self.buffers[agent].add_episode(episode)
            self._episode_buffers[agent][env_idx] = []

    def sample(self, batch_size: int, seq_len: int = 0) -> Dict[str, Union[TransitionBatch, SequenceBatch]]:
        first_buf = next(iter(self.buffers.values()))
        if isinstance(first_buf, EpisodeReplayBuffer):
            return {
                aid: buf.sample(batch_size, seq_len)
                for aid, buf in self.buffers.items()
            }
        return {
            aid: buf.sample(batch_size)
            for aid, buf in self.buffers.items()
        }

    def can_sample(self, batch_size: int) -> bool:
        return all(len(buf) >= batch_size for buf in self.buffers.values())


# =============================================================================
#                        辅助函数（action_mask 提取）
# =============================================================================

def _extract_action_mask_single(
    info: Optional[np.ndarray],
    num_envs: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """从单智能体 vectorized info 中提取 action_mask → tensor。"""
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
    """从单智能体 vectorized info 中提取 action_mask → numpy。"""
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
    """从多智能体 vectorized info 中提取某 agent 的 action_mask → numpy。"""
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


# =============================================================================
#                          向后兼容别名
# =============================================================================

Collector = OnPolicyCollector
MACollector = MAOnPolicyCollector
