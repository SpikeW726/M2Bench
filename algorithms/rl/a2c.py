import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from agent.base_agent import BaseAgent
from networks.mlp_SUN import SUN_Actor, SUN_Critic

class A2CAgent(BaseAgent):
    """
    通用A2C智能体，支持多种网络架构
    
    根据论文"Lightweight Decentralized Neural Network-Based Strategies for Multi-Robot Patrolling"实现
    支持SUN网络（图级观测）和标准MLP网络（局部观测）
    """

    def __init__(self, agent_id: int, config: dict):
        super().__init__(agent_id, config)

        # 添加GPU设备支持
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C Agent {agent_id} initialized with device: {self.device}")

        # 从配置中读取关键参数
        agent_config = config['agent_config']
        env_config = config.get('env_config', {})
        
        self.gamma = agent_config.get('gamma', 0.99)  # 固定gamma，动态gamma由trainer处理
        self.lr = agent_config.get('learning_rate', 0.001)
        self.entropy_coef = agent_config.get('entropy_coef', 0.01)
        self.max_grad_norm = agent_config.get('max_grad_norm', 0.5)
        self.network_type = agent_config.get('network_type', 'mlp_SUN')
        # TD校验开关（仅首次训练步打印一次校验信息）
        self.debug_td = agent_config.get('debug_td', False)
        self._debug_td_printed = False
        # 概率调试与告警开关（默认关闭，避免终端噪音）
        self.debug_probs = agent_config.get('debug_probs', False)
        
        # 根据网络类型初始化不同的网络和处理逻辑
        if self.network_type == 'mlp_SUN':
            self._init_sun_networks(agent_config, env_config)
        elif self.network_type == 'standard_mlp':
            self._init_mlp_networks(agent_config, env_config)
        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")

        # 轨迹数据存储（所有网络类型共用）
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.discount_factors = []
        self.entropies = []
        
    def _init_sun_networks(self, agent_config: dict, env_config: dict):
        """初始化SUN网络（图级观测）"""
        # SUN网络特有参数
        self.node_feat_dim = agent_config.get('node_feat_dim', 2)
        self.edge_feat_dim = agent_config.get('edge_feat_dim', 1)
        
        # 创建Actor和Critic SUN网络并移动到GPU
        self.actor = SUN_Actor(self.node_feat_dim, self.edge_feat_dim).to(self.device)
        self.critic = SUN_Critic(self.node_feat_dim, self.edge_feat_dim).to(self.device)
        
        # SUN需要图结构信息
        self.requires_graph_info = True
        self.adjacency_matrix: Optional[torch.Tensor] = None
        self.edge_weights: Optional[torch.Tensor] = None
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )
        
    def _init_mlp_networks(self, agent_config: dict, env_config: dict):
        """初始化标准MLP网络（局部观测）"""
        # MLP网络参数
        input_dim = agent_config.get('input_dim', 64)
        hidden_dims = agent_config.get('hidden_dims', [128, 64])
        max_neighbors = agent_config.get('max_neighbors', 10)
        
        # 创建Actor网络（输出动作logits）
        actor_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, max_neighbors))
        self.actor = nn.Sequential(*actor_layers)
        
        # 创建Critic网络（输出状态价值）
        critic_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # MLP不需要图信息
        self.requires_graph_info = False
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

    def _to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        """将 numpy 数组转换为 float32 类型的 PyTorch 张量并移动到正确设备"""
        return torch.as_tensor(np_array, dtype=torch.float32, device=self.device)

    def set_graph_info(self, adjacency_matrix: np.ndarray, edge_weights: np.ndarray):
        """设置图结构信息（仅SUN网络需要）"""
        if self.requires_graph_info:
            self.adjacency_matrix = self._to_tensor(adjacency_matrix)
            self.edge_weights = self._to_tensor(edge_weights)
            print(f"Graph info moved to {self.device}: adjacency_matrix shape={self.adjacency_matrix.shape}, edge_weights shape={self.edge_weights.shape}")

    def select_action(self, observation: np.ndarray, neighbors: List[int], evaluation_mode: bool = False) -> Optional[int]:
        """
        根据观测选择动作
        
        统一的A2C动作选择逻辑：
        1. Actor网络输出动作概率分布
        2. 根据neighbors做掩码处理，确保只选择有效动作
        3. 评估模式选最大概率，训练模式按概率采样
        
        Args:
            observation: 观测向量
            neighbors: 邻居节点列表  
            evaluation_mode: 是否评估模式
            
        Returns:
            选中的邻居索引
        """
        # 边界：无可选邻居时直接返回None（避免空分布采样）
        if neighbors is None or len(neighbors) == 0:
            return None

        # 获取动作概率分布
        if self.network_type == 'mlp_SUN':
            if self.adjacency_matrix is None or self.edge_weights is None:
                raise ValueError("Graph info must be set for SUN networks before selecting action.")
            
            # 恢复原始设计：SUN网络处理全图，智能体提取邻居概率
            node_features = self._to_tensor(observation)
            # 使用Actor前向得到动作概率，避免数值不稳定
            action_probs = self.actor(node_features, self.edge_weights, self.adjacency_matrix)
            
            # 提取邻居节点的概率（节点编号从1开始，索引从0开始）
            neighbor_indices = torch.tensor([max(0, n - 1) for n in neighbors], dtype=torch.long, device=self.device)
            # 边界检查
            neighbor_indices = torch.clamp(neighbor_indices, 0, len(action_probs) - 1)
            valid_probs = action_probs[neighbor_indices]

        else:  # standard_mlp或其他类型
            obs_tensor = torch.FloatTensor(observation.flatten()).to(self.device)
            action_logits = self.actor(obs_tensor)
            # MLP输出logits，需要转换为概率
            action_probs = F.softmax(action_logits, dim=-1)
            
            # 根据neighbors数量截取有效概率（action_mask处理）
            valid_probs = action_probs[:len(neighbors)]
            
        # 修复：添加数值稳定性检查
        # 检查是否有NaN或无穷大
        if torch.isnan(valid_probs).any() or torch.isinf(valid_probs).any():
            if self.debug_probs:
                print(f"警告: valid_probs包含无效值，使用均匀分布")
            # 使用非空均匀分布；若出现长度0已在前面return
            valid_probs = torch.ones_like(valid_probs) / max(len(valid_probs), 1)
        
        # 确保概率为正数
        valid_probs = torch.clamp(valid_probs, min=1e-8)
        
        # 重新归一化（保护：分母为0时回退为均匀分布）
        denom = valid_probs.sum()
        if denom <= 0 or torch.isnan(denom) or torch.isinf(denom):
            valid_probs = torch.ones_like(valid_probs) / len(valid_probs)
        else:
            valid_probs = valid_probs / denom

        if evaluation_mode:
            # 性能优化：评估模式下使用no_grad减少内存占用
            with torch.no_grad():
                action_idx = torch.argmax(valid_probs).item()
        else:
            m = Categorical(valid_probs)
            action_idx = m.sample().item()

            # 保存训练信息
            self.current_log_prob = m.log_prob(torch.tensor(action_idx, device=self.device))
            # 状态价值由Critic计算，保持A2C正确性
            if self.network_type == 'mlp_SUN':
                self.current_state_value = self.critic(node_features, self.edge_weights, self.adjacency_matrix)
            else:
                self.current_state_value = self.compute_state_value(observation)
            # 记录熵用于熵正则
            self.entropies.append(m.entropy())

        self.save_observation(observation, action_idx)
        return action_idx
    

    def compute_state_value(self, observation: np.ndarray) -> torch.Tensor:
        """
        计算状态价值
        
        统一的A2C状态价值计算逻辑：
        所有Critic网络都应该输出标量状态价值
        
        Args:
            observation: 观测向量
            
        Returns:
            状态价值（标量）
        """
        if self.network_type == 'mlp_SUN':
            if self.adjacency_matrix is None or self.edge_weights is None:
                raise ValueError("Graph info must be set for SUN networks before computing state value.")
            
            # 使用全图观测计算状态价值（Critic需要全局信息）
            node_features = self._to_tensor(observation)
            state_value = self.critic(node_features, self.edge_weights, self.adjacency_matrix)
            
            # 修复：添加数值稳定性检查
            if torch.isnan(state_value) or torch.isinf(state_value):
                print(f"警告: state_value包含无效值，使用0替代")
                state_value = torch.tensor(0.0, device=self.device)
            
            return state_value
        else:  # standard_mlp或其他类型
            obs_tensor = torch.FloatTensor(observation.flatten()).to(self.device)
            return self.critic(obs_tensor).squeeze()
    
    def learn(self, reward: float, next_observation: Optional[np.ndarray], next_neighbors: List[int], discount_factor: float):
        """
        A2C学习函数：收集轨迹数据
        
        根据论文实现，A2C需要收集完整的episode轨迹后进行批量更新
        实际的网络训练由training_strategy触发，通过调用train_step()完成
        """
        # 保存奖励和动态折扣因子
        self.rewards.append(reward)
        self.discount_factors.append(discount_factor)

        # 判断是否episode结束
        done = next_observation is None
        self.dones.append(done)

        # 保存对数概率和状态价值（保持计算图用于训练）
        if hasattr(self, 'current_log_prob') and hasattr(self, 'current_state_value'):
            self.log_probs.append(self.current_log_prob)
            self.state_values.append(self.current_state_value)
            # 计算并缓存 V(s_{t+1}) 供一步TD使用
            if next_observation is not None:
                with torch.no_grad():
                    if self.network_type == 'mlp_SUN':
                        node_features_next = self._to_tensor(next_observation)
                        next_v = self.critic(node_features_next, self.edge_weights, self.adjacency_matrix)
                    else:
                        obs_tensor_next = torch.FloatTensor(next_observation.flatten()).to(self.device)
                        next_v = self.critic(obs_tensor_next).squeeze()
            else:
                next_v = torch.tensor(0.0, device=self.device)

            if not hasattr(self, 'next_state_values'):
                self.next_state_values = []
            self.next_state_values.append(next_v)
        
        # 注意：不在这里自动触发训练，而是依赖training_strategy

    def train_step(self) -> bool:
        """
        执行一次A2C参数更新
        
        使用收集的轨迹数据计算策略梯度和价值函数更新
        支持动态折扣因子（基于SMDP的边权重）
        """
        if not self.can_train():
            return False

        try:
            # 检查数据一致性
            if len(self.log_probs) != len(self.state_values):
                print(f"A2C Agent {self.agent_id}: 数据不一致，跳过训练")
                return False
            
            # 将列表转换为张量，克隆以避免计算图冲突
            log_probs = torch.stack([lp.clone() for lp in self.log_probs])
            state_values = torch.stack([sv.clone() for sv in self.state_values])
            rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
            discount_factors = torch.tensor(self.discount_factors, dtype=torch.float32, device=self.device)
            dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

            # 一步TD目标：td_target_t = r_t + gamma_t * V(s_{t+1})
            # 为了获得V(s_{t+1})，我们沿时间推进一次，末步用0替代（或episode终止处置0）
            # 这里由于我们只在到达节点时收集一步transition，state_values已经是V(s_t)
            # 我们需要在learn()时也缓存next_state_value供TD使用；若未缓存，则退化为MC（但我们会实现缓存）
            # 先尝试从self中读取缓存的next_state_values
            if hasattr(self, 'next_state_values') and len(self.next_state_values) == len(self.rewards):
                next_state_values = torch.stack([v.clone() for v in self.next_state_values]).to(self.device)
            else:
                # 兜底：末步0，其他用自身V(s)右移一位近似（次优，但避免崩溃）
                shifted = torch.cat([state_values[1:], torch.zeros(1, device=self.device)])
                next_state_values = shifted.detach()

            # TD路径一次性校验打印（可选）
            if self.debug_td and not self._debug_td_printed:
                try:
                    print(f"[A2C TD DEBUG][agent {self.agent_id}] lens: log_probs={len(log_probs)}, V={len(state_values)}, r={len(rewards)}, gamma_tau={len(discount_factors)}, V_next={len(next_state_values)}, dones={len(dones)}")
                    n_preview = min(5, len(rewards))
                    for i in range(n_preview):
                        sv = state_values[i].item()
                        r = rewards[i].item()
                        g = discount_factors[i].item()
                        vn = next_state_values[i].item()
                        dn = dones[i].item()
                        print(f"  t={i}: V={sv:.4f}, r={r:.4f}, gamma_tau={g:.6f}, V_next={vn:.4f}, done={dn}")
                    # 数值异常检查
                    if (torch.isnan(state_values).any() or torch.isnan(rewards).any() or 
                        torch.isnan(discount_factors).any() or torch.isnan(next_state_values).any()):
                        print(f"[A2C TD DEBUG][agent {self.agent_id}] 警告: 存在NaN")
                    if (torch.isinf(state_values).any() or torch.isinf(rewards).any() or 
                        torch.isinf(discount_factors).any() or torch.isinf(next_state_values).any()):
                        print(f"[A2C TD DEBUG][agent {self.agent_id}] 警告: 存在Inf")
                finally:
                    self._debug_td_printed = True

            td_target = rewards + discount_factors * next_state_values * (1.0 - dones)
            advantages = td_target - state_values.detach()
            # 优势标准化以降低尺度敏感性（不改变期望）
            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp_min(1e-6)
            advantages = (advantages - adv_mean) / adv_std

            # 计算Actor损失（策略梯度损失）
            actor_loss = -(log_probs * advantages).mean()

            # 计算Critic损失（价值函数损失）
            critic_loss = F.mse_loss(state_values, td_target.detach())

            # 总损失（论文中使用0.5作为critic loss的系数）
            if len(self.entropies) > 0:
                entropy_term = torch.stack(self.entropies).mean()
            else:
                entropy_term = torch.tensor(0.0, device=self.device)
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_term

            # 数值健康检查：loss 非有限则跳过更新
            if (not torch.isfinite(total_loss) or
                not torch.isfinite(actor_loss) or
                not torch.isfinite(critic_loss)):
                self._reset_trajectory()
                return False

            # 反向传播和优化（带非有限梯度保护）
            self.optimizer.zero_grad()
            total_loss.backward()

            params = list(self.actor.parameters()) + list(self.critic.parameters())
            grads_finite = True
            for p in params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                # 放弃此次更新，防止参数被破坏
                self.optimizer.zero_grad()
                self._reset_trajectory()
                return False

            # 梯度裁剪（仅在梯度有限时执行）
            torch.nn.utils.clip_grad_norm_(
                params,
                self.max_grad_norm,
                error_if_nonfinite=False
            )
            self.optimizer.step()

            # 重置轨迹数据，准备下一个episode
            self._reset_trajectory()

            return True
        
        except Exception as e:
            print(f"A2C Agent {self.agent_id} training failed: {e}")
            self._reset_trajectory()  # 出错时也要重置数据
            return False
        
    def _reset_trajectory(self):
        """重置用于存储轨迹数据的列表"""
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.discount_factors = []
        self.dones = []
        self.entropies = []
        if hasattr(self, 'next_state_values'):
            self.next_state_values = []

    def reset(self):
        """
        重置智能体状态。
        """
        super().reset()
        self._reset_trajectory()

    def can_train(self) -> bool:
        """
        判断A2C是否可以训练
        
        A2C需要完整的轨迹数据才能进行策略梯度更新：
        - 有奖励数据
        - 有对数概率数据（用于计算策略梯度）
        - 有状态价值数据（用于计算advantage）
        """
        return (len(self.rewards) > 0 and 
                len(self.log_probs) > 0 and 
                len(self.state_values) > 0 and
                len(self.rewards) == len(self.log_probs) == len(self.state_values))

    def select_action_with_coordination(self, observation: np.ndarray, neighbors: List[int], 
                                       neighbor_intentions: Dict[int, int], 
                                       evaluation_mode: bool = False) -> Optional[int]:
        """
        带协作机制的动作选择 - 实现论文中的多智能体协作
        
        Args:
            observation: 观测向量
            neighbors: 邻居节点列表
            neighbor_intentions: 邻居节点的意图计数 {node_id: intention_count}
            evaluation_mode: 是否评估模式
            
        Returns:
            选中的邻居索引（在neighbors列表中的索引）
        """
        if self.network_type != 'mlp_SUN':
            # 非SUN网络使用标准方法
            return self.select_action(observation, neighbors, evaluation_mode)
        
        if self.adjacency_matrix is None or self.edge_weights is None:
            raise ValueError("Graph info must be set for SUN networks")
        
        # 边界：无可选邻居时直接返回None
        if neighbors is None or len(neighbors) == 0:
            return None

        # 获取所有节点的效用值（恢复原始设计）
        node_features = self._to_tensor(observation)  # observation已经是[num_nodes, 2]格式
        all_utilities = self.actor.compute_utilities(node_features, self.edge_weights, self.adjacency_matrix)
        
        # 只考虑邻居节点的效用，添加边界检查
        neighbor_indices = torch.tensor([n - 1 for n in neighbors], dtype=torch.long, device=self.device)  # 节点ID转数组索引
        neighbor_indices = torch.clamp(neighbor_indices, 0, len(all_utilities) - 1)
        neighbor_utilities = all_utilities[neighbor_indices]
        
        # 论文协作机制：剔除已被其他智能体声明的邻居（disregard）
        declared_mask = torch.tensor([neighbor_intentions.get(nei, 0) > 0 for nei in neighbors], 
                                     dtype=torch.bool, device=self.device)
        has_available = (~declared_mask).any().item()
        
        if evaluation_mode:
            if has_available:
                # 从未被声明的邻居中选最大utility
                filtered_utils = neighbor_utilities.masked_fill(declared_mask, float('-inf'))
                action_idx = torch.argmax(filtered_utils).item()
            else:
                # 回退：所有邻居都被声明，跳过剔除，按最大utility选择
                action_idx = torch.argmax(neighbor_utilities).item()
        else:
            # 训练阶段不启用协作剔除，保持通用A2C行为
            return self.select_action(observation, neighbors, evaluation_mode=False)
        
        self.save_observation(observation, action_idx)
        return action_idx