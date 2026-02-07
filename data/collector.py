"""数据采集器：从环境中采集数据用于 RL 训练"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from algorithms.algorithm_base import BaseAlgorithm
from envs.venvs import BaseVectorEnv
from data.batch import RolloutBatch, CollectResult


class BaseCollector(ABC):
    """
    采集器基类，定义公共接口
    
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
        n_episodes: Optional[int] = None
    ) -> CollectResult:
        """
        采集数据
        
        Args:
            n_steps: 采集的总步数（所有 env 累计）
            n_episodes: 采集的 episode 数
            
        Returns:
            CollectResult 包含 batch 和统计信息
        """
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
        """处理 episode 结束"""
        self._episode_rewards.append(reward_sum)
        self._episode_lengths.append(length)
        self._current_rewards[env_idx] = 0.0
        self._current_lengths[env_idx] = 0


class Collector(BaseCollector):
    """
    单智能体采集器
    
    处理 Gymnasium Env（ndarray 格式 I/O）
    """
    
    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        if env.is_parallel_env:
            raise ValueError("Collector 只支持 Gymnasium Env，多智能体请用 MACollector")
        super().__init__(algorithm, env)
    
    def _reset_buffer(self):
        """重置内部缓冲列表"""
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[np.ndarray] = []
        self._rew_buf: List[np.ndarray] = []
        self._done_buf: List[np.ndarray] = []
        self._log_prob_buf: List[np.ndarray] = []
        self._action_mask_buf: List[np.ndarray] = []
    
    def collect(
        self, 
        n_steps: Optional[int] = None, 
        n_episodes: Optional[int] = None
    ) -> CollectResult:
        """
        采集 n_steps 步数据
        
        Args:
            n_steps: 采集的总步数
            n_episodes: 采集的 episode 数（暂不支持）
        """
        if n_steps is None and n_episodes is None:
            raise ValueError("必须指定 n_steps 或 n_episodes")
        if n_episodes is not None:
            raise NotImplementedError("暂不支持按 episode 采集")
        
        # 确保环境已 reset
        if self._obs is None:
            self.reset()
        
        # training_mode=True 使策略使用采样而非 argmax（梯度由 torch.no_grad 控制）
        self.algorithm.set_training_mode(True)
        policy = self.algorithm.policy
        device = self.algorithm.device
        
        step_count = 0
        while step_count < n_steps:
            # 转为 tensor
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
            
            # 提取 action_mask（如果存在）
            action_mask = self._extract_action_mask(self._info, device)
            
            # 获取动作
            with torch.no_grad():
                output = policy.forward(obs_t, action_mask=action_mask)
            
            act = output['act'].cpu().numpy()
            log_prob = output['log_prob'].cpu().numpy()
            
            # 存储
            self._obs_buf.append(self._obs.copy())
            self._act_buf.append(act)
            self._log_prob_buf.append(log_prob)
            if action_mask is not None:
                self._action_mask_buf.append(action_mask.cpu().numpy())
            
            # 执行动作
            next_obs, rew, term, trunc, info = self.env.step(act)
            done = term | trunc
            
            self._rew_buf.append(rew)
            self._done_buf.append(done)
            
            # 更新统计
            self._current_rewards += rew
            self._current_lengths += 1
            step_count += self.num_envs
            
            # 处理 done
            for i in range(self.num_envs):
                if done[i]:
                    self._handle_done(i, self._current_rewards[i], self._current_lengths[i])
            
            self._obs = next_obs
            self._info = info
        
        # 构建 batch
        batch = RolloutBatch(
            obs=np.concatenate(self._obs_buf, axis=0),
            act=np.concatenate(self._act_buf, axis=0),
            rew=np.concatenate(self._rew_buf, axis=0),
            done=np.concatenate(self._done_buf, axis=0).astype(np.float32),
            log_prob=np.concatenate(self._log_prob_buf, axis=0),
            action_mask=np.concatenate(self._action_mask_buf, axis=0) if self._action_mask_buf else None,
        )
        
        result = CollectResult(
            batch=batch,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )
        
        return result
    
    def _extract_action_mask(
        self, 
        info: Optional[np.ndarray], 
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """从 info 中提取 action_mask"""
        if info is None:
            return None
        
        masks = []
        for i in range(self.num_envs):
            if isinstance(info[i], dict) and 'action_mask' in info[i]:
                masks.append(info[i]['action_mask'])
        
        if not masks:
            return None
        
        return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)


class MACollector(BaseCollector):
    """
    多智能体采集器
    
    处理 PettingZoo ParallelEnv（Dict 格式 I/O）
    支持 global_state 采集用于 MAPPO centralized critic
    """
    
    def __init__(self, algorithm: BaseAlgorithm, env: BaseVectorEnv):
        if not env.is_parallel_env:
            raise ValueError("MACollector 只支持 ParallelEnv，单智能体请用 Collector")
        
        self.agents = env.agents
        super().__init__(algorithm, env)
    
    def _reset_buffer(self):
        """为每个 agent 重置缓冲列表"""
        self._buffers: Dict[str, Dict[str, List]] = {
            agent: {
                'obs': [],
                'act': [],
                'rew': [],
                'done': [],
                'truncated': [],  # 区分 truncation 和 termination，用于正确的 value bootstrap
                'log_prob': [],
                'action_mask': [],
                'global_state': [],
                'final_global_state': [],  # 用于中间 truncation 的 value bootstrap
            }
            for agent in self.agents
        }
    
    def collect(
        self, 
        n_steps: Optional[int] = None, 
        n_episodes: Optional[int] = None
    ) -> CollectResult:
        """
        采集 n_steps 步数据
        
        Returns:
            CollectResult，其中 batch 为 Dict[str, RolloutBatch]
        """
        if n_steps is None and n_episodes is None:
            raise ValueError("必须指定 n_steps 或 n_episodes")
        if n_episodes is not None:
            raise NotImplementedError("暂不支持按 episode 采集")
        
        # 确保环境已 reset
        if self._obs is None:
            self.reset()
        
        # training_mode=True 使策略使用采样而非 argmax（梯度由 torch.no_grad 控制）
        self.algorithm.set_training_mode(True)
        policy = self.algorithm.policy  # MultiAgentPolicy
        
        step_count = 0
        while step_count < n_steps:
            # 获取 global_state（如果环境支持）
            global_states = self._get_global_states()
            
            # 获取动作（使用 MultiAgentPolicy.compute_actions）
            actions, outputs = policy.compute_actions(self._obs, self._info)
            
            # 存储每个 agent 的数据
            for agent in self.agents:
                buf = self._buffers[agent]
                # obs: (num_envs, *obs_shape)
                buf['obs'].append(self._obs[agent].copy())
                buf['act'].append(actions[agent].copy())
                buf['log_prob'].append(outputs[agent]['log_prob'].cpu().numpy())
                
                # global_state
                if global_states is not None:
                    buf['global_state'].append(global_states.copy())
                
                # action_mask from info
                if self._info is not None and agent in self._info:
                    info_arr = self._info[agent]
                    masks = []
                    for i in range(self.num_envs):
                        if 'action_mask' in info_arr[i]:
                            masks.append(info_arr[i]['action_mask'])
                    if masks:
                        buf['action_mask'].append(np.stack(masks))
            
            # 执行动作
            next_obs, rew, term, trunc, info = self.env.step(actions)
            
            # 存储 reward、done 和 truncated
            for agent in self.agents:
                done = term[agent] | trunc[agent]
                self._buffers[agent]['rew'].append(rew[agent])
                self._buffers[agent]['done'].append(done.astype(np.float32))
                self._buffers[agent]['truncated'].append(trunc[agent].astype(np.float32))
            
            # 提取 final_global_state（用于中间 truncation 的 value bootstrap）
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
            
            # 更新统计（用第一个 agent 的 reward 作为 episode reward）
            first_agent = self.agents[0]
            self._current_rewards += rew[first_agent]
            self._current_lengths += 1
            step_count += self.num_envs
            
            # 处理 done（假设所有 agent 同时 done）
            done_arr = term[first_agent] | trunc[first_agent]
            for i in range(self.num_envs):
                if done_arr[i]:
                    self._handle_done(i, self._current_rewards[i], self._current_lengths[i])
            
            self._obs = next_obs
            self._info = info
        
        # 构建每个 agent 的 batch
        batch_dict: Dict[str, RolloutBatch] = {}
        for agent in self.agents:
            buf = self._buffers[agent]
            # 处理 final_global_state: List[List[ndarray or None]] -> List[List]
            # 保持原始结构以便在 prepare_batch 中使用
            final_gs = buf['final_global_state'] if buf['final_global_state'] else None
            batch_dict[agent] = RolloutBatch(
                obs=np.concatenate(buf['obs'], axis=0),
                act=np.concatenate(buf['act'], axis=0),
                rew=np.concatenate(buf['rew'], axis=0),
                done=np.concatenate(buf['done'], axis=0),
                truncated=np.concatenate(buf['truncated'], axis=0),
                log_prob=np.concatenate(buf['log_prob'], axis=0),
                global_state=np.concatenate(buf['global_state'], axis=0) if buf['global_state'] else None,
                action_mask=np.concatenate(buf['action_mask'], axis=0) if buf['action_mask'] else None,
                final_global_state=final_gs,  # List[List[ndarray or None]]，保持原始结构
            )
        
        result = CollectResult(
            batch=batch_dict,
            n_steps=step_count,
            n_episodes=len(self._episode_rewards),
            episode_rewards=self._episode_rewards.copy(),
            episode_lengths=self._episode_lengths.copy(),
        )
        
        return result
    
    def _get_global_states(self) -> Optional[np.ndarray]:
        """获取 global_state（如果环境支持 state() 方法）"""
        try:
            # 优先使用 call_env_method（SubprocVectorEnv 并行执行，避免传 bound method）
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
