import random
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Dict as GymDict, Discrete, MultiDiscrete, Tuple as GymTuple

from agent.base_agent import BaseAgent
from networks.mlp_S4R1 import mlp_S4R1
from networks.mlp_MASUP_d3qn import MASUPDuelingMLP
from utils.graph_utils import Graph


NETWORK_REGISTRY = {
    "mlp_S4R1": mlp_S4R1,
    "mlp_MASUP_d3qn": MASUPDuelingMLP,
}


class DQNAgent(BaseAgent):
    """
    通用 D3QN 智能体：
    - 支持配置/环境自动推断的状态与动作维度。
    - 默认启用 Double DQN + Dueling 网络结构。
    - 兼容图巡逻类（S4R1 等）与 MASUP 事件驱动环境。
    """

    def __init__(self, agent_id: int, config: Dict):
        super().__init__(agent_id, config)

        agent_config = config['agent_config']
        env_config = config['env_config']
        agent_num = env_config.get('num_agents', 4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 学习参数
        self.network_type = agent_config.get('network_type', "mlp_S4R1")
        if self.network_type not in NETWORK_REGISTRY:
            raise ValueError(f"Unknown network type: {self.network_type}")

        self.gamma = agent_config.get('gamma', 0.99)
        self.epsilon = agent_config.get('epsilon', 1.0)
        self.epsilon_decay = agent_config.get('epsilon_decay', 0.992)
        self.epsilon_min = agent_config.get('epsilon_min', 0.01)
        self.learning_rate = agent_config.get('learning_rate', 0.001)
        self.use_double_dqn = agent_config.get('use_double_dqn', True)

        # 经验回放
        self.batch_size = agent_config.get('batch_size', 32)
        self.replay_buffer_size = agent_config.get('replay_buffer_size', 10000)
        self.learning_starts = agent_config.get('learning_starts', self.batch_size)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        # 目标网络同步
        self.target_update_freq = agent_config.get('target_update_freq', 100)
        self.tau = agent_config.get('tau', 0.005)
        self.use_soft_update = agent_config.get('use_soft_update', True)

        # 结构/维度
        self.hidden_dims = agent_config.get('hidden_dims')
        self.network_kwargs = agent_config.get('network_kwargs', {})
        self.state_size: Optional[int] = self._safe_int(agent_config.get('state_dim'))
        self.action_size: Optional[int] = self._safe_int(agent_config.get('action_dim'))

        # 兼容旧配置：根据图结构推断默认维度
        graph_path = env_config.get('graph_path')
        if graph_path and (self.state_size is None or self.action_size is None):
            graph = Graph(graph_path)
            inferred_action = graph.get_max_degree()
            if self.action_size is None:
                self.action_size = inferred_action
            if self.state_size is None:
                self.state_size = 2 * agent_num + self.action_size

        self.q_network: Optional[torch.nn.Module] = None
        self.target_network: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._net_initialized = False

        # 训练状态
        self.step_count = 0
        self.update_count = 0

        if self.state_size is not None and self.action_size is not None:
            self._build_networks()

        print(f"D3QN Agent {agent_id} initialized on {self.device}, "
              f"state_dim={self.state_size}, action_dim={self.action_size}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def set_environment(self, env: Any):
        """
        Trainer 会在环境构造后调用。用于补充缺省的状态/动作维度。
        """
        self.env = env
        
        # 🔥 关键修复：优先从环境推断维度（更准确）
        inferred_state = self._infer_state_dim_from_env(env)
        inferred_action = self._infer_action_dim_from_env(env)
        
        # 检查是否需要更新维度
        need_rebuild = False
        
        if inferred_action is not None and inferred_action != self.action_size:
            print(f"Agent {self.agent_id}: 更新 action_dim: {self.action_size} → {inferred_action}")
            self.action_size = inferred_action
            need_rebuild = True
            
        if inferred_state is not None and inferred_state != self.state_size:
            print(f"Agent {self.agent_id}: 更新 state_dim: {self.state_size} → {inferred_state}")
            self.state_size = inferred_state
            need_rebuild = True

        # 如果维度变化或网络未初始化，（重新）构建网络
        if need_rebuild or not self._net_initialized:
            if self.state_size is None or self.action_size is None:
                raise ValueError(
                    "无法初始化 DQN 网络：state_dim/action_dim 未知。"
                    "请在 agent_config 中显式指定，或确保环境可推断出空间维度。"
                )
            self._build_networks()
            if need_rebuild:
                print(f"Agent {self.agent_id}: 已重新构建网络，新维度 state={self.state_size}, action={self.action_size}")

    def select_action(self,
                      observation: Union[np.ndarray, Dict[str, np.ndarray]],
                      neighbors: Optional[List[int]],
                      evaluation_mode: bool = False) -> Optional[int]:
        self._ensure_network_ready()

        # 🔥 关键修复：优先使用环境的get_action_mask方法（如果存在）
        # 这确保了与环境动作空间定义的一致性
        action_mask = None
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'get_action_mask') and callable(self.env.get_action_mask):
                try:
                    action_mask = self.env.get_action_mask(self.agent_name)
                    if not isinstance(action_mask, np.ndarray):
                        action_mask = np.asarray(action_mask, dtype=bool)
                except Exception:
                    action_mask = None
        
        # 如果环境没有提供mask，使用简单的neighbors-based mask
        if action_mask is None:
            action_mask = self._build_action_mask(neighbors)
        
        valid_indices = np.where(action_mask)[0]
        if valid_indices.size == 0:
            return None

        processed_obs = self._process_observation(observation)
        obs_tensor = torch.from_numpy(processed_obs).unsqueeze(0).to(self.device)

        greedy = evaluation_mode or np.random.rand() > self.epsilon
        if greedy:
            with torch.no_grad():
                q_values = self.q_network(obs_tensor).cpu().numpy()[0]
            masked_q = np.where(action_mask, q_values, -np.inf)
            action_idx = int(np.argmax(masked_q))
        else:
            action_idx = int(np.random.choice(valid_indices))

        # 存储预处理后的观测，避免重复转换
        self.save_observation(processed_obs, action_idx)
        return action_idx

    def learn(self,
              reward: float,
              next_observation: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
              next_neighbors: List[int],
              discount_factor: float):
        if self.last_observation is None or self.last_action is None:
            return

        processed_next = None
        if next_observation is not None:
            processed_next = self._process_observation(next_observation)

        experience = (
            self.last_observation.copy(),
            int(self.last_action),
            float(reward),
            processed_next.copy() if processed_next is not None else None,
            processed_next is None,
            float(discount_factor),
        )
        self.replay_buffer.append(experience)

    def can_train(self) -> bool:
        threshold = max(self.batch_size, self.learning_starts)
        return len(self.replay_buffer) >= threshold

    def train_step(self) -> bool:
        if not self.can_train():
            return False

        self._ensure_network_ready()
        self._replay_experience()
        self.step_count += 1

        if self.use_soft_update:
            self._update_target_network(soft=True)
        elif self.step_count % self.target_update_freq == 0:
            self._update_target_network(soft=False)
        return True

    def save_model(self, filepath: str):
        self._ensure_network_ready()
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'state_dim': self.state_size,
            'action_dim': self.action_size,
            'network_type': self.network_type,
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.state_size = int(checkpoint.get('state_dim', self.state_size))
        self.action_size = int(checkpoint.get('action_dim', self.action_size))
        if not self._net_initialized:
            self._build_networks()

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        self.update_count = checkpoint.get('update_count', 0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get_stats(self) -> Dict[str, Union[int, float]]:
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'memory_size': len(self.replay_buffer),
            'state_dim': self.state_size,
            'action_dim': self.action_size,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _build_networks(self):
        net_cls = NETWORK_REGISTRY[self.network_type]
        net_kwargs = dict(self.network_kwargs)
        if 'hidden_dims' not in net_kwargs and self.hidden_dims:
            net_kwargs['hidden_dims'] = self.hidden_dims

        self.q_network = net_cls(self.state_size, self.action_size, **net_kwargs).to(self.device)
        self.target_network = net_cls(self.state_size, self.action_size, **net_kwargs).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self._net_initialized = True

    def _replay_experience(self):
        batch = random.sample(self.replay_buffer, self.batch_size)

        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32, device=self.device)
        next_states = np.array(
            [exp[3] if exp[3] is not None else np.zeros(self.state_size, dtype=np.float32) for exp in batch],
            dtype=np.float32,
        )
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool, device=self.device)
        discounts = torch.tensor([exp[5] for exp in batch], dtype=torch.float32, device=self.device)

        states_tensor = torch.from_numpy(states).to(self.device)
        next_states_tensor = torch.from_numpy(next_states).to(self.device)

        current_q = self.q_network(states_tensor).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states_tensor).argmax(dim=1, keepdim=True)
                next_q_target = self.target_network(next_states_tensor).gather(1, next_actions).squeeze(1)
            else:
                next_q_target = self.target_network(next_states_tensor).max(dim=1)[0]
            target_q = rewards + discounts * next_q_target * (~dones).float()

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

    def _update_target_network(self, soft: bool):
        if soft:
            for target_param, main_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _process_observation(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        if observation is None:
            return np.zeros(self.state_size, dtype=np.float32)

        if isinstance(observation, dict):
            flat_parts = []
            for key in sorted(observation.keys()):
                value = observation[key]
                if value is None:
                    continue
                flat_parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
            flat_obs = np.concatenate(flat_parts, axis=0) if flat_parts else np.zeros(0, dtype=np.float32)
        else:
            flat_obs = np.asarray(observation, dtype=np.float32).reshape(-1)

        if len(flat_obs) < self.state_size:
            padded = np.zeros(self.state_size, dtype=np.float32)
            padded[:len(flat_obs)] = flat_obs
            return padded
        if len(flat_obs) > self.state_size:
            return flat_obs[:self.state_size]
        return flat_obs

    def _build_action_mask(self, neighbors: Optional[List[int]]) -> np.ndarray:
        if self.action_size is None or self.action_size <= 0:
            raise ValueError("action_size 尚未初始化，无法构建动作掩码。")

        mask = np.ones(self.action_size, dtype=bool)
        if neighbors is not None and len(neighbors) > 0:
            mask[:] = False
            valid_len = min(len(neighbors), self.action_size)
            mask[:valid_len] = True
        return mask

    def _infer_state_dim_from_env(self, env: Any) -> Optional[int]:
        try:
            obs_space = getattr(env, 'observation_space', None)
            if callable(obs_space):
                obs_space = obs_space(self.agent_name)
            if obs_space is None:
                return None
            return self._calc_space_dim(obs_space)
        except Exception:
            return None

    def _infer_action_dim_from_env(self, env: Any) -> Optional[int]:
        try:
            action_space = getattr(env, 'action_space', None)
            if callable(action_space):
                action_space = action_space(self.agent_name)
            if action_space is None:
                return None
            return self._calc_action_dim(action_space)
        except Exception:
            return None

    def _calc_space_dim(self, space) -> Optional[int]:
        if isinstance(space, Box):
            return int(np.prod(space.shape))
        if isinstance(space, GymDict):
            total = 0
            for sub in space.spaces.values():
                dim = self._calc_space_dim(sub)
                if dim is None:
                    return None
                total += dim
            return total
        if isinstance(space, GymTuple):
            total = 0
            for sub in space.spaces:
                dim = self._calc_space_dim(sub)
                if dim is None:
                    return None
                total += dim
            return total
        if isinstance(space, MultiDiscrete):
            return int(len(space.nvec))
        if isinstance(space, Discrete):
            return 1
        return None

    def _calc_action_dim(self, space) -> Optional[int]:
        if isinstance(space, Discrete):
            return int(space.n)
        if isinstance(space, MultiDiscrete):
            return int(np.prod(space.nvec))
        if isinstance(space, GymTuple):
            size = 1
            for sub in space.spaces:
                sub_dim = self._calc_action_dim(sub)
                if sub_dim is None:
                    return None
                size *= sub_dim
            return size
        if isinstance(space, GymDict):
            size = 1
            for sub in space.spaces.values():
                sub_dim = self._calc_action_dim(sub)
                if sub_dim is None:
                    return None
                size *= sub_dim
            return size
        return None

    def _ensure_network_ready(self):
        if not self._net_initialized:
            if self.state_size is None or self.action_size is None:
                raise ValueError("DQN 网络尚未初始化，缺少 state_dim/action_dim。")
            self._build_networks()

    @staticmethod
    def _safe_int(value: Optional[Union[int, float]]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
