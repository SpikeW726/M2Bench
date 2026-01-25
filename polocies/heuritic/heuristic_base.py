from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np

class HeuriticBasePolicy(ABC):
    """
    Abstract base class for all heuristic policies.
    """
    
    def __init__(self, num_agents: int, config: Dict):
        self.num_agents = num_agents
        self.config = config
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
    
    @abstractmethod
    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为所有智能体计算动作,默认是同步决策框架,无需决策的智能体返回None
        
        Args:
            obs_dict: 每个智能体的局部观测
                {agent_id: {
                    'current_node': int,           # 当前节点
                    'neighbors': List[int],        # 邻居节点列表
                    'neighbor_idleness': List[float],  # 邻居节点空闲度
                    'on_edge': bool,               # 是否在边上移动中
                    ...
                }}
            
            global_state: 全局状态信息（启发式算法通常需要）
                {
                    'graph': Graph,                        # 图结构对象
                    'agent_positions': Dict[int, int],     # 所有智能体位置 {agent_idx: node_id}
                    'agents_target_node': Dict[int, int],  # 所有智能体目标 {agent_idx: node_id}
                    'node_idleness': Dict[int, float],     # 节点空闲度 {node_id: idleness}
                    'agents_on_edge': Dict[int, bool],     # 智能体是否在边上
                    'agent_speeds': List[float],           # 智能体速度列表 (来自物理世界)
                    ...
                }
            

        Returns:
            actions: {agent_id: action} 所有需要决策的智能体的动作
        """
        pass
    
    @abstractmethod
    def _compute_action(
        self,
        agent_idx: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Optional[int]:
        """
        为单个智能体计算动作
        
        Args:
            agent_idx: 智能体索引
            obs: 该智能体的局部观测
            global_state: 全局状态
            evaluation_mode: 评估模式
        
        Returns:
            action: 动作（邻居索引），如果不需要决策返回 None
        """
        pass
    
    def reset(self):
        """重置策略内部状态（如果有的话）"""
        pass
