"""数据采集器：从环境中采集数据用于 RL 训练。

On-policy:
    OnPolicyCollector       ← 单智能体，返回 RolloutBatch
    MAOnPolicyCollector     ← 多智能体，返回 Dict[str, RolloutBatch]

Off-policy:
    OffPolicyCollector      ← 单智能体，内持 ReplayBuffer
    MAOffPolicyCollector    ← 多智能体，内持 Dict[str, ReplayBuffer]
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from algorithms.algorithm_base import BaseAlgorithm
from envs.venvs import BaseVectorEnv
from data.batch import RolloutBatch, TransitionBatch, SequenceBatch, CollectResult
from data.buffer import ReplayBuffer, SequenceReplayBuffer


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
        # 必须在 super().__init__ 之前设置，_reset_buffer 依赖此标志
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        super().__init__(algorithm, env)

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

            # per-agent final obs（供 IPPO obs-based critic 的 truncation bootstrap 使用）
            for agent in self.agents:
                final_obs_list = []
                for i in range(self.num_envs):
                    if trunc[agent][i]:
                        final_obs_list.append(next_obs[agent][i].copy())
                    else:
                        final_obs_list.append(None)
                self._buffers[agent]['final_obs'].append(final_obs_list)

            # 所有智能体平均 reward
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
                    # RNN: episode 结束时重置对应 env 的 hidden
                    if self._is_recurrent:
                        for agent in self.agents:
                            self._hidden[agent][:, i, :] = 0.0

            self._obs = next_obs
            self._info = info

        # rollout 结束后立刻采集边界 global state（= 下一步的初始 state）。
        # 供 VDPPO 修正 next_states[-1]，其他算法忽略此字段。
        boundary_gs = self._get_global_states()

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
        """获取所有 env 的 global state。

        失败时显式抛出异常而非静默返回 None，避免 CTDE 算法
        （如 QMIX）用零值 state 通过超网络产生无意义的 Q_tot。
        """
        if hasattr(self.env, 'call_env_method'):
            states = self.env.call_env_method("state")
        else:
            states = self.env.get_env_attr("state")
            states = [s() if callable(s) else s for s in states]
        if states is None or states[0] is None:
            return None
        return np.stack(states)


# =============================================================================
#                          Off-Policy Collectors
# =============================================================================

class OffPolicyCollector(BaseCollector):
    """
    单智能体 off-policy 采集器。

    MLP 模式: 内持 ReplayBuffer，逐 step 存入 transition。
    RNN 模式: 内持 SequenceReplayBuffer，按 episode 存储（预切片 + 向量化采样）。

    collect() 将 transitions 存入 buffer，返回 CollectResult（仅统计量）。
    Trainer 通过 sample() / can_sample() 从 buffer 采样。
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseVectorEnv,
        buffer: Union["ReplayBuffer", "SequenceReplayBuffer"],
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

    def _init_ep_buf(self) -> dict:
        """创建空的 dict-of-lists episode 临时缓冲区。"""
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
        """将 env_idx 的 episode buffer 合并推入 SequenceReplayBuffer。"""
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
    """
    多智能体 off-policy 采集器。

    MLP 模式: 内持 Dict[str, ReplayBuffer]，逐 step 存入。
    RNN 模式: 内持 Dict[str, SequenceReplayBuffer]，按 episode 预切片存储。

    每个 agent 的 transitions 独立存入对应 buffer。
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseVectorEnv,
        buffers: Dict[str, Union["ReplayBuffer", "SequenceReplayBuffer"]],
        collect_state: bool = False,
        sync_mode: bool = False,
        gamma: float = 0.99,
        shared_indices: Optional[bool] = None,
    ):
        if not env.is_parallel_env:
            raise ValueError(
                "MAOffPolicyCollector 只支持 ParallelEnv，"
                "单智能体请用 OffPolicyCollector"
            )
        self.buffers = buffers
        self.agents = env.agents
        self._is_recurrent = getattr(algorithm.policy, "is_recurrent", False)
        self._collect_state = collect_state
        # VDN/QMIX 等值分解算法需要 shared_indices 保证多 agent buffer 时间对齐；
        # 若未显式传入，仅在 collect_state=True（QMIX 路径）时默认启用，
        # 调用方应根据算法语义显式传入该参数。
        self._shared_indices = collect_state if shared_indices is None else shared_indices
        self._sync_mode = sync_mode
        self._gamma = gamma
        self._hidden: Optional[Dict[str, torch.Tensor]] = None
        super().__init__(algorithm, env)

    def _init_ep_buf(self, agent: str) -> dict:
        """创建空的 dict-of-lists episode 临时缓冲区。"""
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
            cur_active_masks: Dict[str, Optional[np.ndarray]] = {}
            for agent in self.agents:
                cur_action_masks[agent] = _extract_agent_action_mask_np(
                    self._info, agent, self.num_envs
                )
                cur_active_masks[agent] = _extract_agent_active_mask_np(
                    self._info, agent, self.num_envs
                )

            cur_state = self._get_global_states() if self._collect_state else None

            next_obs, rew, term, trunc, info = self.env.step(actions)

            next_state = self._get_global_states() if self._collect_state else None

            if self._collect_state and (cur_state is None or next_state is None):
                raise RuntimeError(
                    "CTDE(off-policy) 需要 global state，但 VectorEnv 未返回有效 state()。"
                    "QMIX 等算法会因此写入错误 replay；请检查 env.call_env_method('state') 与并行封装。"
                )

            # 提取 step 后的 active_mask（用于判断 agent 是否到达）
            next_active_masks: Dict[str, Optional[np.ndarray]] = {}
            for agent in self.agents:
                next_active_masks[agent] = _extract_agent_active_mask_np(
                    info, agent, self.num_envs
                )

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
                        # 截断≠真实终止：截断步的 next_obs 应用真实 final_obs（截断前最后一个状态），
                        # 而非 auto-reset 后的初始 obs；done 只由 terminated 决定
                        if trunc[agent][i] and isinstance(info[agent][i], dict):
                            final_obs = info[agent][i].get("final_obs")
                            next_obs_i = final_obs if final_obs is not None else next_obs[agent][i].copy()
                        else:
                            next_obs_i = next_obs[agent][i].copy()
                        eb["next_obs"].append(next_obs_i)
                        eb["done"].append(float(term[agent][i]))  # trunc 不视为 terminal
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
            else:
                for agent in self.agents:
                    # TD 语义：截断≠真实终止，done 只由 term 决定；
                    # 截断步的 next_obs 替换为真实 final_obs（截断前最后一个状态）
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
        """将 env_idx 的所有 agent episode buffer 合并推入对应 SequenceReplayBuffer。"""
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
        """MLP 同步路径：跟踪 pending decision，累积折扣 reward，到达时 flush。"""
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
        """获取所有 env 的 global state。

        失败时显式抛出异常而非静默返回 None，避免 CTDE 算法
        （如 QMIX）用零值 state 通过超网络产生无意义的 Q_tot。
        """
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
        # shared_indices: VDN/QMIX 要求所有 agent 取同一批 episode/transition，
        # 保证多 agent 数据时间对齐，对 ReplayBuffer 和 SequenceReplayBuffer 均适用。
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


def _extract_agent_active_mask_np(
    info_dict: Optional[Dict[str, np.ndarray]],
    agent: str,
    num_envs: int,
) -> Optional[np.ndarray]:
    """从多智能体 vectorized info 中提取某 agent 的 active_mask → numpy。"""
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


# =============================================================================
#                          向后兼容别名
# =============================================================================

Collector = OnPolicyCollector
MACollector = MAOnPolicyCollector
