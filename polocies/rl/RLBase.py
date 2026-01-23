from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np

class BaseAgent(ABC):
    """
    The abstract base class for all agents.
    Defines the interface that agents must implement so that the trainer can handle different types of agents in a unified manner.
    """
    
    def __init__(self, agent_id: int, config: dict):
        self.agent_id = agent_id
        self.agent_name = f'agent_{self.agent_id}'
        self.config = config
        
        # Velocity configuration prepared for consecutive time-step (not implemented yet)
        self.speed = config.get('agent_speeds', {}).get(agent_id, 1.0)  
        self.time_discrete = config.get('time_discrete', True)  
        
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, neighbors: List[int], evaluation_mode: bool = False) -> Optional[int]:
        """
        Select an action based on the current observation and available actions.
        
        Args:
            observation: current observation vector from environment
            neighbors: available neighbor nodes
            evaluation_mode: false = epsilon-greedy, true = greedy
            
        Returns:
            the chosen action
        """
        pass
    
    @abstractmethod
    def learn(self, reward: float, next_observation: Optional[np.ndarray], next_neighbors: List[int], discount_factor: float):
        """
        Learn from the experience gained.
        
        Args:
            reward: 获得的奖励
            next_observation: 下一个观测向量（None表示episode结束）
            next_neighbors: 下一个状态的邻居节点
            discount_factor: 折扣因子
        """
        pass
    
    def can_train(self) -> bool:
        """
        判断智能体是否可以进行训练
        
        对于不同类型的智能体有不同的实现：
        - Q-learning: 总是可以训练（因为是即时更新）
        - DQN: 检查经验回放缓冲区是否有足够的经验
        
        Returns:
            bool: 是否可以训练
        """
        # 默认实现：总是可以训练
        # 子类可以重写这个方法来实现更复杂的逻辑
        return True
    
    def train_step(self) -> bool:
        """
        Execute a training step.
        
        For agents that support training (like DQN), this method should be overridden
        to perform the actual training step.
        
        Returns:
            bool: Whether training was successful
        """
        return False
    
    def decay_epsilon(self):
        """
        Decay the exploration rate at the end of an episode.
        
        This method should be overridden by agents that use epsilon-greedy exploration.
        Default implementation does nothing.
        """
        pass
    
    def save_observation(self, observation: np.ndarray, action: int):
        """
        Save the current observation and action for subsequent learning updates.
        
        Args:
            observation: current observation vector
            action: chosen action
        """
        self.last_observation = observation.copy() if observation is not None else None
        self.last_action = action
    
    def reset(self):
        """
        Reset the agent's state at the start of each episode.
        """
        self.last_observation = None
        self.last_action = None
    
    def get_speed(self) -> float:
        """
        Get the velocity of the agent.
        (Reserved interface for continuous time step algorithm)
            
        Returns:
            velocity of the agent
        """
        return self.speed
    
    def set_speed(self, speed: float):
        """
        Set the velocity of the agent.
        (Reserved interface for continuous time step algorithm)
        
        Args:
            speed: new velocity
        """
        self.speed = speed
    
    def is_time_discrete(self) -> bool:
        """
        Checks whether discrete time step mode is currently used.
        
        Returns:
            True = discrete,False = continuous
        """
        return self.time_discrete
    
    def get_continuous_action_info(self, target_node: int, current_node: int, 
                                 edge_length: Union[int, float]) -> Optional[dict]:
        """
        Gets details of actions for continuous time-step algorithms.
        (Reserved interface for continuous time step algorithm)
        
        Args:
            target_node: 
            current_node: 
            edge_length: 
            
        Returns:
            Dictionary containing action details, or None (discrete timestep mode)
        """
        if not self.time_discrete:
            return {
                'target_node': target_node,
                'current_node': current_node,
                'edge_length': edge_length,
                'speed': self.speed,
                'estimated_time': edge_length / self.speed if self.speed > 0 else float('inf')
            }
        return None 

    def get_valid_actions_from_state(self, state: Tuple, neighbors: List[int]) -> List[int]:
        """
        根据状态和邻居信息计算有效动作
        
        Args:
            state: 当前状态
            neighbors: 邻居节点列表
            
        Returns:
            有效动作索引列表
        """
        if not neighbors:
            return []
        
        # 动作索引对应邻居列表的索引
        return list(range(len(neighbors))) 