"""
HeuristicSampler: 使用启发式策略采集样本用于预训练 Actor/Critic 网络

数据格式 (npz):
- obs: [N, T, M, Obs_Dim] Float32 - Actor输入
- critic_states: [N, T, M, State_Dim+M] Float32 - Centralized Critic输入 (global_state + agent_one_hot)
- actions: [N, T, M, 1] Int64 - 动作索引
- action_masks: [N, T, M, Act_Dim] Int8 - 动作掩码
- rewards: [N, T, M, 1] Float32 - 每个智能体的奖励
- padded_mask: [N, T, 1] Int8 - 填充掩码 (1=真实, 0=填充)
- returns: [N, T, M, 1] Float32 - 每个智能体的累计折扣回报

支持并行采样：使用 num_workers 参数启用多进程并行
"""
import sys
from pathlib import Path
# 添加项目根目录到 Python 路径 (支持从任意目录运行)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import h5py

from policies.heuritic.heuristic_base import HeuriticBasePolicy
from policies.heuritic.er import ERPolicy
from envs.mdps.base_envs import EventDrivenEnv
from envs.mdps.masup import MASUPEnv


@dataclass
class EpisodeData:
    """单个 episode 的轨迹数据"""                                     # T_ep为此episode的step数量,M为智能体数量
    obs: List[np.ndarray] = field(default_factory=list)               # [T_ep, M, Obs_Dim]
    critic_states: List[np.ndarray] = field(default_factory=list)     # [T_ep, M, State_Dim+M]
    actions: List[np.ndarray] = field(default_factory=list)           # [T_ep, M, 1]
    action_masks: List[np.ndarray] = field(default_factory=list)      # [T_ep, M, Act_Dim]
    rewards: List[np.ndarray] = field(default_factory=list)           # [T_ep, M]
    
    @property
    def length(self) -> int:
        return len(self.rewards)


def _worker_collect_episodes(
    args: Tuple[int, int, float, Dict, Dict, Dict]
) -> List[Dict[str, np.ndarray]]:
    """
    Worker 进程：采集指定数量的 episodes (旧版本，保留向后兼容)
    
    Args:
        args: (num_episodes, worker_id, eps, env_config, custom_config, policy_config)
    
    Returns:
        List of episode data dicts (已转换为 numpy arrays)
    """
    num_episodes, worker_id, eps, env_config, custom_config, policy_config = args
    
    # 在 worker 进程中创建独立的环境和策略实例
    env = MASUPEnv(env_config, **custom_config)
    num_agents = env_config.get("num_agents", 3)
    policy = ERPolicy(num_agents, policy_config)
    
    # 缓存维度信息
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n
    
    # 获取 state_dim
    env.reset()
    state_dim = len(env.state())
    critic_state_dim = state_dim + num_agents
    
    episodes_data = []
    
    for _ in range(num_episodes):
        episode = _collect_single_episode(env, policy, num_agents, obs_dim, act_dim, eps)
        episodes_data.append(episode)
    
    return episodes_data


def _worker_collect_to_file(args: Tuple) -> str:
    """
    Worker 进程：采集 episodes 并直接写入临时文件（内存友好版本）
    
    Args:
        args: (num_episodes, worker_id, eps, env_config, custom_config, policy_config, 
               temp_dir, chunk_size, gamma)
    
    Returns:
        临时文件路径列表（逗号分隔）
    """
    (num_episodes, worker_id, eps, env_config, custom_config, policy_config, 
     temp_dir, chunk_size, gamma) = args
    
    # 在 worker 进程中创建独立的环境和策略实例
    env = MASUPEnv(env_config, **custom_config)
    num_agents = env_config.get("num_agents", 3)
    policy = ERPolicy(num_agents, policy_config)
    
    # 缓存维度信息
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n
    
    # 获取 state_dim
    env.reset()
    state_dim = len(env.state())
    critic_state_dim = state_dim + num_agents
    
    # 分 chunk 采集并保存
    temp_dir_path = Path(temp_dir)
    chunk_files = []
    
    collected = 0
    while collected < num_episodes:
        # 当前 chunk 要采集的 episodes 数量
        chunk_eps = min(chunk_size, num_episodes - collected)
        episodes_data = []
        
        for _ in range(chunk_eps):
            episode = _collect_single_episode(env, policy, num_agents, obs_dim, act_dim, eps)
            episodes_data.append(episode)
        
        # 保存当前 chunk 到临时文件
        chunk_file = temp_dir_path / f"worker_{worker_id}_chunk_{len(chunk_files):05d}.npz"
        _save_chunk_to_file(episodes_data, chunk_file, gamma, num_agents, obs_dim, act_dim, critic_state_dim)
        chunk_files.append(str(chunk_file))
        
        collected += chunk_eps
        
        # 释放内存
        del episodes_data
    
    # 返回 chunk 文件列表（用逗号分隔）
    return ",".join(chunk_files)


def _save_chunk_to_file(
    episodes_data: List[Dict[str, np.ndarray]], 
    save_path: Path,
    gamma: float,
    num_agents: int,
    obs_dim: int,
    act_dim: int,
    critic_state_dim: int,
) -> None:
    """将一个 chunk 的 episodes 保存到 npz 文件"""
    N = len(episodes_data)
    max_len = max(ep['obs'].shape[0] for ep in episodes_data)
    
    # 初始化数组
    obs = np.zeros([N, max_len, num_agents, obs_dim], dtype=np.float32)
    critic_states = np.zeros([N, max_len, num_agents, critic_state_dim], dtype=np.float32)
    actions = np.zeros([N, max_len, num_agents, 1], dtype=np.int64)
    action_masks = np.zeros([N, max_len, num_agents, act_dim], dtype=np.int8)
    rewards = np.zeros([N, max_len, num_agents, 1], dtype=np.float32)
    padded_mask = np.zeros([N, max_len, 1], dtype=np.int8)
    returns = np.zeros([N, max_len, num_agents, 1], dtype=np.float32)
    
    for i, ep in enumerate(episodes_data):
        L = ep['obs'].shape[0]
        obs[i, :L] = ep['obs']
        critic_states[i, :L] = ep['critic_states']
        actions[i, :L] = ep['actions']
        action_masks[i, :L] = ep['action_masks']
        rewards[i, :L, :, 0] = ep['rewards']
        padded_mask[i, :L, 0] = 1
        
        # 计算 returns
        ep_returns = _compute_returns_static(ep['rewards'], gamma)
        returns[i, :L, :, 0] = ep_returns
    
    np.savez(save_path, obs=obs, critic_states=critic_states, actions=actions,
             action_masks=action_masks, rewards=rewards, padded_mask=padded_mask, returns=returns)


def _compute_returns_static(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """静态方法：计算 returns（可在 worker 进程中调用）"""
    T, M = rewards.shape
    returns = np.zeros((T, M), dtype=np.float32)
    G = np.zeros(M, dtype=np.float32)
    
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    
    return returns


def _collect_single_episode(
    env: MASUPEnv,
    policy: HeuriticBasePolicy,
    num_agents: int,
    obs_dim: int,
    act_dim: int,
    eps: float
) -> Dict[str, np.ndarray]:
    """
    采集单个 episode，返回 numpy arrays 字典
    """
    obs_list = []
    critic_states_list = []
    actions_list = []
    action_masks_list = []
    rewards_list = []
    
    obs_rl, info = env.reset()
    policy.reset()
    done = False
    
    while not done:
        # 1. 获取启发式观测和决策
        h_obs = env.world.get_heuristic_obs()
        global_state = env.world.get_global_state_for_heuristic()
        h_actions = policy.compute_actions(h_obs, global_state)
        
        # 2. 转换动作 + epsilon-greedy
        masup_actions = {}
        for agent_str, neighbor_idx in h_actions.items():
            if np.random.random() < eps:
                valid_actions = env.get_valid_actions(agent_str)
                masup_actions[agent_str] = int(np.random.choice(valid_actions))
            else:
                masup_actions[agent_str] = env.convert_heuristic_action(agent_str, neighbor_idx)
        
        # 3. 填充 no-op
        for agent_str in env.agents:
            if agent_str not in masup_actions:
                masup_actions[agent_str] = act_dim - 1
        
        # 4. 记录数据
        # obs: [M, Obs_Dim]
        obs_arr = np.stack([obs_rl[f"agent_{i}"] for i in range(num_agents)], axis=0)
        obs_list.append(obs_arr)
        
        # critic_states: [M, State_Dim+M]
        g_state = env.state()
        critic_states = []
        for agent_id in range(num_agents):
            one_hot = np.zeros(num_agents, dtype=np.float32)
            one_hot[agent_id] = 1.0
            critic_states.append(np.concatenate([g_state, one_hot]))
        critic_states_list.append(np.stack(critic_states, axis=0))
        
        # actions: [M, 1]
        actions_arr = np.array([[masup_actions.get(f"agent_{i}", act_dim - 1)] for i in range(num_agents)], dtype=np.int64)
        actions_list.append(actions_arr)
        
        # action_masks: [M, Act_Dim]
        masks_arr = np.stack([info[f"agent_{i}"]['action_mask'] for i in range(num_agents)], axis=0).astype(np.int8)
        action_masks_list.append(masks_arr)
        
        # 5. 执行动作
        obs_rl, rewards, terms, truncs, info = env.step(masup_actions)
        
        # 6. 记录奖励: [M]
        rewards_arr = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(num_agents)], dtype=np.float32)
        rewards_list.append(rewards_arr)
        
        # 7. 检查结束
        done = any(truncs.values()) or any(terms.values())
    
    return {
        'obs': np.stack(obs_list, axis=0),              # [T, M, Obs_Dim]
        'critic_states': np.stack(critic_states_list, axis=0),  # [T, M, State_Dim+M]
        'actions': np.stack(actions_list, axis=0),      # [T, M, 1]
        'action_masks': np.stack(action_masks_list, axis=0),    # [T, M, Act_Dim]
        'rewards': np.stack(rewards_list, axis=0),      # [T, M]
    }


class HeuristicSampler:
    """
    使用启发式策略在 MASUPEnv 中采集 Actor-Critic 监督学习样本
    支持 epsilon-greedy 探索增加数据多样性
    支持多进程并行采样 (num_workers > 1)
    """
    def __init__(
        self, 
        policy: HeuriticBasePolicy, 
        env: MASUPEnv,
        env_config: Optional[Dict] = None,
        custom_config: Optional[Dict] = None,
        policy_config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            policy: 启发式策略实例
            env: MASUPEnv 环境实例
            env_config: 环境配置 (并行采样时用于创建 worker 环境)
            custom_config: 环境自定义配置 (并行采样时使用)
            policy_config: 策略配置 (并行采样时用于创建 worker 策略)
        """
        self.policy = policy
        self.env = env
        
        # 保存配置用于并行采样
        self._env_config = env_config
        self._custom_config = custom_config or {}
        self._policy_config = policy_config
        
        # 缓存环境维度信息
        self.num_agents = env.world.num_agents
        self.obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        self.act_dim = env.action_space(env.possible_agents[0]).n
        
        # state_dim 需要 reset 后从 env.state() 获取
        self._state_dim: Optional[int] = None
    
    @property
    def state_dim(self) -> int:
        """全局状态维度，首次访问时通过 reset 环境获取"""
        if self._state_dim is None:
            self.env.reset()
            self._state_dim = len(self.env.state())
        return self._state_dim
    
    @property
    def critic_state_dim(self) -> int:
        """Centralized Critic 输入维度: global_state + agent_one_hot"""
        return self.state_dim + self.num_agents
    
    def sample(
        self, 
        num_episodes: int, 
        save_path: str, 
        gamma: float = 0.999, 
        eps: float = 0.0, 
        batch_size: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        """
        采集指定数量的 episode 并保存
        
        Args:
            num_episodes: 采集的 episode 数量
            save_path: 保存路径 (.npz)
            gamma: 计算 returns 的折扣因子
            eps: epsilon-greedy 随机探索概率 (0.0=纯启发式, 1.0=纯随机)
            batch_size: 分批处理大小，None 表示一次性处理所有数据（可能内存溢出）
                       建议设置为 1000-5000，根据可用内存调整
            num_workers: 并行 worker 数量 (>1 启用多进程并行采样)
        """
        if num_workers > 1:
            # 并行采样
            if self._env_config is None or self._policy_config is None:
                raise ValueError(
                    "Parallel sampling requires env_config and policy_config. "
                    "Please pass them to HeuristicSampler constructor."
                )
            self._sample_parallel(num_episodes, save_path, gamma, eps, batch_size, num_workers)
            print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path} (parallel, {num_workers} workers)")
        elif batch_size is None or batch_size >= num_episodes:
            # 一次性处理（保持向后兼容）
            trajectories: List[EpisodeData] = []
            
            for ep_idx in tqdm(range(num_episodes), desc="Sampling episodes"):
                episode_data = self._collect_episode(eps)
                trajectories.append(episode_data)
            
            # 后处理并保存
            self._pad_and_save(trajectories, save_path, gamma)
            print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path}")
        else:
            # 分批处理，避免内存溢出
            self._sample_in_batches(num_episodes, save_path, gamma, eps, batch_size)
            print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path} (in batches)")
    
    def _collect_episode(self, eps: float) -> EpisodeData:
        """
        采集单个 episode 的数据
        
        Args:
            eps: epsilon-greedy 随机探索概率
        
        Returns:
            EpisodeData: 包含完整轨迹数据的对象
        """
        episode = EpisodeData()
        
        # 重置环境和策略
        obs_rl, info = self.env.reset()
        self.policy.reset()  # 重置策略内部状态(如ER的意图表)
        
        done = False
        
        while not done:
            # 1. 获取启发式观测和全局状态（直接从 PatrolWorld 获取），然后获取启发式决策
            h_obs = self.env.world.get_heuristic_obs()
            global_state = self.env.world.get_global_state_for_heuristic()
            h_actions = self.policy.compute_actions(h_obs, global_state)
            
            # 2. 转换动作格式 + epsilon-greedy 探索
            masup_actions = {}
            for agent_str, neighbor_idx in h_actions.items():
                if np.random.random() < eps:
                    # 随机选择：从有效动作中随机选一个
                    valid_actions = self.env.get_valid_actions(agent_str)
                    masup_actions[agent_str] = int(np.random.choice(valid_actions))
                else:
                    # 启发式选择
                    masup_actions[agent_str] = self.env.convert_heuristic_action(agent_str, neighbor_idx)
            
            # 3. 为未决策的智能体填充 no-op
            for agent_str in self.env.agents:
                if agent_str not in masup_actions:
                    masup_actions[agent_str] = self.act_dim - 1  # no-op
            
            # 4. 记录当前步数据 (在执行动作前记录)
            episode.obs.append(self._extract_obs(obs_rl))
            episode.critic_states.append(self._build_critic_states())
            episode.actions.append(self._extract_actions(masup_actions, info))
            episode.action_masks.append(self._extract_masks(info))
            
            # 5. 执行动作
            obs_rl, rewards, terms, truncs, info = self.env.step(masup_actions)
            
            # 6. 记录每个智能体的奖励
            episode.rewards.append(self._extract_rewards(rewards))
            
            # 7. 检查是否结束
            done = any(truncs.values()) or any(terms.values())
        
        return episode
    
    def _extract_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从环境观测字典中提取观测数组
        
        Args:
            obs_dict: {agent_str: obs_array}
        
        Returns:
            np.ndarray: [M, Obs_Dim]
        """
        obs_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            obs_list.append(obs_dict[agent_str])
        return np.stack(obs_list, axis=0)  # [M, Obs_Dim]
    
    def _build_critic_states(self) -> np.ndarray:
        """
        构建每个智能体的 Centralized Critic 输入: global_state + agent_one_hot
        
        Returns:
            np.ndarray: [M, State_Dim + M]
        """
        global_state = self.env.state()  # [State_Dim]
        critic_states = []
        
        for agent_id in range(self.num_agents):
            # 为每个智能体构建 one-hot 编码
            one_hot = np.zeros(self.num_agents, dtype=np.float32)
            one_hot[agent_id] = 1.0
            # 拼接 global_state + one_hot
            critic_state = np.concatenate([global_state, one_hot])
            critic_states.append(critic_state)
        
        return np.stack(critic_states, axis=0)  # [M, State_Dim + M]
    
    def _extract_actions(self, actions: Dict[str, int], info: Dict[str, Dict]) -> np.ndarray:
        """
        从动作字典中提取动作数组
        
        Args:
            actions: {agent_str: action_idx}
            info: {agent_str: {'active_mask': int, ...}}
        
        Returns:
            np.ndarray: [M, 1]
        """
        action_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            action_list.append([actions.get(agent_str, self.act_dim - 1)])
        return np.array(action_list, dtype=np.int64)  # [M, 1]
    
    def _extract_masks(self, info: Dict[str, Dict]) -> np.ndarray:
        """
        从 info 中提取动作掩码
        
        Args:
            info: {agent_str: {'action_mask': np.ndarray, ...}}
        
        Returns:
            np.ndarray: [M, Act_Dim]
        """
        mask_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            mask_list.append(info[agent_str]['action_mask'])
        return np.stack(mask_list, axis=0).astype(np.int8)  # [M, Act_Dim]
    
    def _extract_rewards(self, rewards: Dict[str, float]) -> np.ndarray:
        """
        从奖励字典中提取每个智能体的奖励
        
        Args:
            rewards: {agent_str: reward}
        
        Returns:
            np.ndarray: [M]
        """
        reward_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            reward_list.append(rewards.get(agent_str, 0.0))
        return np.array(reward_list, dtype=np.float32)  # [M]
    
    def _compute_returns(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        """
        为每个智能体计算累计折扣回报 G_t = r_t + gamma * G_{t+1}
        
        Args:
            rewards: [T, M] 每个时刻每个智能体的奖励
            gamma: 折扣因子
        
        Returns:
            np.ndarray: [T, M] 每个智能体的回报序列
        """
        T, M = rewards.shape
        returns = np.zeros((T, M), dtype=np.float32)
        G = np.zeros(M, dtype=np.float32)  # 每个智能体的累计回报
        
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
        
        return returns
    
    def _pad_and_save(self, trajectories: List[EpisodeData], save_path: str, gamma: float) -> None:
        """
        填充轨迹到统一长度并保存为 npz 格式
        
        Args:
            trajectories: episode 数据列表
            save_path: 保存路径
            gamma: 计算 returns 的折扣因子
        """
        N = len(trajectories)
        max_len = max(ep.length for ep in trajectories)
        M = self.num_agents
        
        # 初始化数组
        obs = np.zeros([N, max_len, M, self.obs_dim], dtype=np.float32)
        critic_states = np.zeros([N, max_len, M, self.critic_state_dim], dtype=np.float32)
        actions = np.zeros([N, max_len, M, 1], dtype=np.int64)
        action_masks = np.zeros([N, max_len, M, self.act_dim], dtype=np.int8)
        rewards = np.zeros([N, max_len, M, 1], dtype=np.float32)
        padded_mask = np.zeros([N, max_len, 1], dtype=np.int8)
        returns = np.zeros([N, max_len, M, 1], dtype=np.float32)
        
        # 填充数据
        for i, ep in enumerate(trajectories):
            L = ep.length
            obs[i, :L] = np.stack(ep.obs, axis=0)
            critic_states[i, :L] = np.stack(ep.critic_states, axis=0)
            actions[i, :L] = np.stack(ep.actions, axis=0)
            action_masks[i, :L] = np.stack(ep.action_masks, axis=0)
            
            # rewards: [L, M] -> [L, M, 1]
            ep_rewards = np.stack(ep.rewards, axis=0)  # [L, M]
            rewards[i, :L, :, 0] = ep_rewards
            
            padded_mask[i, :L, 0] = 1  # 真实数据标记为 1
            
            # 为每个智能体独立计算 returns
            ep_returns = self._compute_returns(ep_rewards, gamma)  # [L, M]
            returns[i, :L, :, 0] = ep_returns
        
        # 保存
        np.savez(
            save_path,
            obs=obs,
            critic_states=critic_states,
            actions=actions,
            action_masks=action_masks,
            rewards=rewards,
            padded_mask=padded_mask,
            returns=returns
        )
        
        # 打印统计信息
        print(f"[HeuristicSampler] Data shapes:")
        print(f"  obs:           {obs.shape}")
        print(f"  critic_states: {critic_states.shape}")
        print(f"  actions:       {actions.shape}")
        print(f"  action_masks:  {action_masks.shape}")
        print(f"  rewards:       {rewards.shape}")
        print(f"  padded_mask:   {padded_mask.shape}")
        print(f"  returns:       {returns.shape}")
        print(f"  max_episode_len: {max_len}")
    
    def _sample_parallel(
        self, 
        num_episodes: int, 
        save_path: str, 
        gamma: float, 
        eps: float, 
        batch_size: Optional[int],
        num_workers: int
    ) -> None:
        """
        使用多进程并行采样（内存友好版本：Worker 直接写入临时文件）
        
        Args:
            num_episodes: 总 episode 数量
            save_path: 保存路径
            gamma: 折扣因子
            eps: epsilon-greedy 探索概率
            batch_size: 批大小 (用于分批保存)
            num_workers: worker 进程数量
        """
        import time
        
        # 创建临时目录
        temp_dir = Path(save_path).parent / ".temp_parallel"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 每个 worker 的 chunk 大小（每个 chunk 保存一次，避免内存爆炸）
        # 建议：每个 chunk 500-1000 episodes，平衡内存和 I/O
        chunk_size = min(500, max(100, num_episodes // (num_workers * 10)))
        
        # 计算每个 worker 负责的 episode 数量
        episodes_per_worker = num_episodes // num_workers
        remainder = num_episodes % num_workers
        
        # 构造 worker 参数
        worker_args = []
        for worker_id in range(num_workers):
            n_eps = episodes_per_worker + (1 if worker_id < remainder else 0)
            if n_eps > 0:
                worker_args.append((
                    n_eps,
                    worker_id,
                    eps,
                    self._env_config,
                    self._custom_config,
                    self._policy_config,
                    str(temp_dir),
                    chunk_size,
                    gamma,
                ))
        
        print(f"[HeuristicSampler] Starting parallel sampling with {num_workers} workers...")
        print(f"[HeuristicSampler] Episodes per worker: {[a[0] for a in worker_args]}")
        print(f"[HeuristicSampler] Chunk size: {chunk_size} episodes/chunk")
        
        # 使用 spawn 方式避免 CUDA/fork 问题
        ctx = mp.get_context('spawn')
        
        # 启动进程池
        all_chunk_files = []
        start_time = time.time()
        
        with ctx.Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 以便尽早获取结果并显示进度
            # 每个 worker 完成后返回其 chunk 文件路径
            results = list(tqdm(
                pool.imap_unordered(_worker_collect_to_file, worker_args),
                total=len(worker_args),
                desc=f"Workers (each ~{episodes_per_worker} eps)"
            ))
            
            # 收集所有 chunk 文件路径
            for chunk_files_str in results:
                if chunk_files_str:
                    all_chunk_files.extend(chunk_files_str.split(","))
        
        elapsed = time.time() - start_time
        print(f"[HeuristicSampler] Collected {num_episodes} episodes in {elapsed:.1f}s ({num_episodes/elapsed:.1f} eps/s)")
        print(f"[HeuristicSampler] Generated {len(all_chunk_files)} chunk files, merging...")
        
        # 合并所有 chunk 文件
        chunk_paths = [Path(f) for f in all_chunk_files]
        
        # 获取全局最大长度
        actual_max_lens = []
        for chunk_path in chunk_paths:
            with np.load(chunk_path, mmap_mode='r') as data:
                actual_max_lens.append(data['obs'].shape[1])
        global_max_len = max(actual_max_lens)
        
        print(f"[HeuristicSampler] Episode lengths: min={min(actual_max_lens)}, max={global_max_len}, mean={np.mean(actual_max_lens):.1f}")
        
        # 根据文件后缀选择合并方法
        if save_path.endswith('.h5') or save_path.endswith('.hdf5'):
            self._merge_batches_to_hdf5(chunk_paths, save_path, global_max_len)
        else:
            self._merge_batches(chunk_paths, save_path, global_max_len)
        
        # 清理临时文件
        for chunk_path in chunk_paths:
            chunk_path.unlink()
        
        # 尝试删除临时目录（可能有子目录残留）
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    
    def _sample_in_batches(self, num_episodes: int, save_path: str, gamma: float, eps: float, batch_size: int) -> None:
        """
        分批采集并保存，避免内存溢出
        
        Args:
            num_episodes: 总 episode 数量
            save_path: 保存路径
            gamma: 折扣因子
            eps: epsilon-greedy 探索概率
            batch_size: 每批处理的 episode 数量
        """
        import tempfile
        import os
        
        # 使用临时文件存储各批次数据
        temp_dir = Path(save_path).parent / ".temp_batches"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        batch_paths = []
        all_max_lens = []
        
        # 分批采集
        num_batches = (num_episodes + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_episodes)
            batch_episodes = end_idx - start_idx
            
            trajectories: List[EpisodeData] = []
            
            for ep_idx in tqdm(range(batch_episodes), desc=f"Batch {batch_idx+1}/{num_batches}"):
                episode_data = self._collect_episode(eps)
                trajectories.append(episode_data)
            
            # 保存当前批次到临时文件
            batch_path = temp_dir / f"batch_{batch_idx:05d}.npz"
            self._pad_and_save(trajectories, str(batch_path), gamma)
            batch_paths.append(batch_path)
            
            # 记录最大长度
            max_len = max(ep.length for ep in trajectories)
            all_max_lens.append(max_len)
            
            # 清空内存
            del trajectories
        
        # 合并所有批次前，重新检查所有批次文件的实际最大长度（更可靠）
        # 这样可以确保使用真正的全局最大长度，避免形状不匹配
        actual_max_lens = []
        for batch_path in batch_paths:
            with np.load(batch_path, mmap_mode='r') as batch_data:
                actual_max_lens.append(batch_data['obs'].shape[1])  # T 维度
        global_max_len = max(actual_max_lens)
        
        print(f"[HeuristicSampler] Batch max lengths: min={min(actual_max_lens)}, max={global_max_len}, mean={np.mean(actual_max_lens):.1f}")
        
        # 根据文件后缀选择合并方法
        if save_path.endswith('.h5') or save_path.endswith('.hdf5'):
            self._merge_batches_to_hdf5(batch_paths, save_path, global_max_len)
        else:
            self._merge_batches(batch_paths, save_path, global_max_len)
        
        # 清理临时文件
        for batch_path in batch_paths:
            batch_path.unlink()
        temp_dir.rmdir()
    
    def _merge_batches(self, batch_paths: List[Path], save_path: str, max_len: int) -> None:
        """
        合并多个批次的 npz 文件
        
        Args:
            batch_paths: 批次文件路径列表
            save_path: 最终保存路径
            max_len: 所有批次中的最大 episode 长度
        """
        # 加载第一个批次获取形状信息
        with np.load(batch_paths[0], mmap_mode='r') as first_batch:
            M = first_batch['obs'].shape[2]
            obs_dim = first_batch['obs'].shape[3]
            critic_state_dim = first_batch['critic_states'].shape[3]
            act_dim = first_batch['action_masks'].shape[3]
        
        # 计算总 episode 数（在合并循环中计算，避免重复打开文件）
        total_episodes = 0
        for batch_path in batch_paths:
            with np.load(batch_path, mmap_mode='r') as batch_data:
                total_episodes += batch_data['obs'].shape[0]
        
        # 初始化最终数组
        obs = np.zeros([total_episodes, max_len, M, obs_dim], dtype=np.float32)
        critic_states = np.zeros([total_episodes, max_len, M, critic_state_dim], dtype=np.float32)
        actions = np.zeros([total_episodes, max_len, M, 1], dtype=np.int64)
        action_masks = np.zeros([total_episodes, max_len, M, act_dim], dtype=np.int8)
        rewards = np.zeros([total_episodes, max_len, M, 1], dtype=np.float32)
        padded_mask = np.zeros([total_episodes, max_len, 1], dtype=np.int8)
        returns = np.zeros([total_episodes, max_len, M, 1], dtype=np.float32)
        
        # 合并所有批次
        offset = 0
        for batch_path in tqdm(batch_paths, desc="Merging batches"):
            with np.load(batch_path, mmap_mode='r') as batch_data:
                N_batch = batch_data['obs'].shape[0]
                T_batch = batch_data['obs'].shape[1]
                
                # 复制数据
                obs[offset:offset+N_batch, :T_batch] = batch_data['obs']
                critic_states[offset:offset+N_batch, :T_batch] = batch_data['critic_states']
                actions[offset:offset+N_batch, :T_batch] = batch_data['actions']
                action_masks[offset:offset+N_batch, :T_batch] = batch_data['action_masks']
                rewards[offset:offset+N_batch, :T_batch] = batch_data['rewards']
                padded_mask[offset:offset+N_batch, :T_batch] = batch_data['padded_mask']
                returns[offset:offset+N_batch, :T_batch] = batch_data['returns']
                
                offset += N_batch
        
        # 保存最终文件
        np.savez(
            save_path,
            obs=obs,
            critic_states=critic_states,
            actions=actions,
            action_masks=action_masks,
            rewards=rewards,
            padded_mask=padded_mask,
            returns=returns
        )
        
        # 打印统计信息
        print(f"[HeuristicSampler] Merged data shapes:")
        print(f"  obs:           {obs.shape}")
        print(f"  critic_states: {critic_states.shape}")
        print(f"  actions:       {actions.shape}")
        print(f"  action_masks:  {action_masks.shape}")
        print(f"  rewards:       {rewards.shape}")
        print(f"  padded_mask:   {padded_mask.shape}")
        print(f"  returns:       {returns.shape}")
        print(f"  max_episode_len: {max_len}")

    def _merge_batches_to_hdf5(self, batch_paths: List[Path], save_path: str, max_len: int) -> None:
        """
        增量合并多个批次的 npz 文件到 HDF5（内存友好）
        
        与 _merge_batches 不同，此方法不会将所有数据加载到内存，
        而是逐批次直接写入 HDF5 文件。
        """
        # 1. 扫描获取形状信息和总 episode 数
        with np.load(batch_paths[0], mmap_mode='r') as first_batch:
            M = first_batch['obs'].shape[2]
            obs_dim = first_batch['obs'].shape[3]
            critic_state_dim = first_batch['critic_states'].shape[3]
            act_dim = first_batch['action_masks'].shape[3]
        
        total_episodes = 0
        for batch_path in batch_paths:
            with np.load(batch_path, mmap_mode='r') as data:
                total_episodes += data['obs'].shape[0]
        
        # 2. 创建 HDF5 文件并预分配 datasets
        with h5py.File(save_path, 'w') as hf:
            # 创建 datasets (chunked for efficient I/O)
            chunk_size = min(512, total_episodes)
            hf.create_dataset('obs', shape=(total_episodes, max_len, M, obs_dim),
                              dtype='float32', chunks=(chunk_size, max_len, M, obs_dim))
            hf.create_dataset('critic_states', shape=(total_episodes, max_len, M, critic_state_dim),
                              dtype='float32', chunks=(chunk_size, max_len, M, critic_state_dim))
            hf.create_dataset('actions', shape=(total_episodes, max_len, M, 1),
                              dtype='int64', chunks=(chunk_size, max_len, M, 1))
            hf.create_dataset('action_masks', shape=(total_episodes, max_len, M, act_dim),
                              dtype='int8', chunks=(chunk_size, max_len, M, act_dim))
            hf.create_dataset('rewards', shape=(total_episodes, max_len, M, 1),
                              dtype='float32', chunks=(chunk_size, max_len, M, 1))
            hf.create_dataset('padded_mask', shape=(total_episodes, max_len, 1),
                              dtype='int8', chunks=(chunk_size, max_len, 1))
            hf.create_dataset('returns', shape=(total_episodes, max_len, M, 1),
                              dtype='float32', chunks=(chunk_size, max_len, M, 1))
            
            # 3. 逐批次写入
            offset = 0
            for batch_path in tqdm(batch_paths, desc="Merging to HDF5"):
                with np.load(batch_path, mmap_mode='r') as batch_data:
                    N_batch = batch_data['obs'].shape[0]
                    T_batch = batch_data['obs'].shape[1]
                    
                    # 直接写入 HDF5 (零拷贝到磁盘)
                    hf['obs'][offset:offset+N_batch, :T_batch] = batch_data['obs']
                    hf['critic_states'][offset:offset+N_batch, :T_batch] = batch_data['critic_states']
                    hf['actions'][offset:offset+N_batch, :T_batch] = batch_data['actions']
                    hf['action_masks'][offset:offset+N_batch, :T_batch] = batch_data['action_masks']
                    hf['rewards'][offset:offset+N_batch, :T_batch] = batch_data['rewards']
                    hf['padded_mask'][offset:offset+N_batch, :T_batch] = batch_data['padded_mask']
                    hf['returns'][offset:offset+N_batch, :T_batch] = batch_data['returns']
                    
                    offset += N_batch
        
        # 打印统计信息
        print(f"[HeuristicSampler] Saved HDF5: {save_path}")
        print(f"  shape: ({total_episodes}, {max_len}, {M}, *)")
        print(f"  datasets: obs, critic_states, actions, action_masks, rewards, padded_mask, returns")


if __name__ == "__main__":
    import os
    import argparse

    # 切换工作目录到项目根目录（确保配置文件路径正确）
    os.chdir(_project_root)

    from configs.registry import load_env_config, _env_config_to_dicts

    parser = argparse.ArgumentParser(description="Heuristic Sampler for Actor-Critic Pre-training")
    parser.add_argument("--num_episodes", type=int, default=50000, help="采集 episode 数量")
    parser.add_argument("--save_path", type=str, default="dataset/samples.h5", help="保存路径 (.npz/.h5)")
    parser.add_argument("--gamma", type=float, default=0.999, help="折扣因子")
    parser.add_argument("--eps", type=float, default=0.0, help="Epsilon-greedy 随机探索概率")
    parser.add_argument("--batch_size", type=int, default=2048, help="分批大小（避免内存溢出）")
    parser.add_argument("--num_workers", type=int, default=1, help="并行 worker 数量 (>1 启用多进程)")
    parser.add_argument("--policy_config", type=str, default="configs/policies/ER.yaml",
                        help="策略配置 YAML")
    parser.add_argument("--env_config", type=str, default="configs/eval/masup_tsp12.yaml",
                        help="环境配置 YAML (experiment YAML 或独立 eval YAML)")
    args = parser.parse_args()

    # 加载策略配置
    with open(args.policy_config, 'r', encoding='utf-8') as f:
        policy_config = yaml.safe_load(f)

    # 加载环境配置（使用统一的 EnvConfig 体系）
    env_cfg = load_env_config(args.env_config)
    env_config, custom_config = _env_config_to_dicts(env_cfg)

    num_agents = env_config.get("num_agents", 3)
    policy = ERPolicy(num_agents, policy_config)
    env = MASUPEnv(env_config, **custom_config)

    # 创建采样器（传入 dict 配置以支持并行采样中 worker 进程创建独立环境）
    sampler = HeuristicSampler(
        policy=policy,
        env=env,
        env_config=env_config,
        custom_config=custom_config,
        policy_config=policy_config,
    )

    # 开始采样
    sampler.sample(
        num_episodes=args.num_episodes,
        save_path=args.save_path,
        gamma=args.gamma,
        eps=args.eps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 用法示例:
    #   python trainers/imitator/heuristic_sampler.py --num_episodes 50000 --num_workers 1
    #   python trainers/imitator/heuristic_sampler.py --num_episodes 50000 --num_workers 4
