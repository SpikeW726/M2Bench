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
"""
import sys
from pathlib import Path
# 添加项目根目录到 Python 路径 (支持从任意目录运行)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import yaml
from tqdm import tqdm

from polocies.heuritic.heuristic_base import HeuriticBasePolicy
from polocies.heuritic.er import ERPolicy
from envs.base_envs import EventDrivenEnv
from envs.masup_env import MASUPEnv


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


class HeuristicSampler:
    """
    使用启发式策略在 MASUPEnv 中采集 Actor-Critic 监督学习样本
    支持 epsilon-greedy 探索增加数据多样性
    """
    def __init__(self, policy: HeuriticBasePolicy, env: MASUPEnv) -> None:
        self.policy = policy
        self.env = env
        
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
    
    def sample(self, num_episodes: int, save_path: str, gamma: float = 0.999, eps: float = 0.0) -> None:
        """
        采集指定数量的 episode 并保存
        
        Args:
            num_episodes: 采集的 episode 数量
            save_path: 保存路径 (.npz)
            gamma: 计算 returns 的折扣因子
            eps: epsilon-greedy 随机探索概率 (0.0=纯启发式, 1.0=纯随机)
        """
        trajectories: List[EpisodeData] = []
        
        for ep_idx in tqdm(range(num_episodes), desc="Sampling episodes"):
            episode_data = self._collect_episode(eps)
            trajectories.append(episode_data)
        
        # 后处理并保存
        self._pad_and_save(trajectories, save_path, gamma)
        print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path}")
    
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
            # 1. 获取启发式观测和全局状态，然后获取启发式决策
            h_obs = self.env.get_heuristic_obs()
            global_state = self.env.get_global_state_for_heuristic()
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


if __name__ == "__main__":
    import os
    # 切换工作目录到项目根目录 (确保配置文件路径正确)
    os.chdir(_project_root)
    
    policy_config_path = "configs/ER.yaml"
    env_config_path = "configs/MASUPEnv.yaml"

    with open(policy_config_path, 'r', encoding='utf-8') as f:
        ER_config = yaml.safe_load(f)
    with open(env_config_path, 'r', encoding='utf-8') as f:
        MASUP_config = yaml.safe_load(f)
    custom_config = MASUP_config["custom_config"]
    MASUP_config = MASUP_config["env_config"]

    num_agents = MASUP_config.get("num_agents", 3)
    policy = ERPolicy(num_agents, ER_config)
    env = MASUPEnv(MASUP_config, **custom_config)

    sampler = HeuristicSampler(policy=policy, env=env)
    
    sampler.sample(num_episodes=100000, save_path="data/samples_eps01.npz", gamma=0.999, eps=0.1)
