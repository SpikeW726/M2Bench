from abc import abstractmethod
from pettingzoo import ParallelEnv
from envs.mdps.patrol_core import PatrolWorld, TickResult
from typing import Dict, Any, Optional
import gymnasium
import numpy as np

class BaseEnv(ParallelEnv):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world = PatrolWorld(config)
        self.enable_wait = config.get("enable_wait", False)
        
        # PettingZoo标准属性
        self.metadata = {"render_modes": [], "name": self.__class__.__name__}
        self.possible_agents = [f"agent_{i}" for i in range(config["num_agents"])]
        self.agents = self.possible_agents[:]

    @abstractmethod
    def step(self, actions: Dict[str, int]):
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        pass

    @abstractmethod
    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        """
        返回指定智能体的观测空间
        
        PettingZoo 标准：必须为相同 agent name 返回相同值
        """
        raise NotImplementedError
    
    @abstractmethod
    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        """
        返回指定智能体的动作空间
        
        PettingZoo 标准：必须为相同 agent name 返回相同值
        """
        raise NotImplementedError
    
    @property
    def observation_spaces(self) -> Dict[str, gymnasium.spaces.Space]:
        """返回所有智能体的观测空间字典"""
        return {agent: self.observation_space(agent) for agent in self.possible_agents}
    
    @property
    def action_spaces(self) -> Dict[str, gymnasium.spaces.Space]:
        """返回所有智能体的动作空间字典"""
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    
    @abstractmethod
    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        """构建所有智能体的观测"""
        pass
    
    @abstractmethod
    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        """
        构建所有智能体的 info
        
        Args:
            result: tick 的结果，reset 时为 None
        
        Note:
            info 应包含 'active_mask' 用于标记智能体是否真正需要决策
        """
        pass
    
    @abstractmethod
    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        """计算奖励"""
        pass
    
    def _compute_terminations(self) -> Dict[str, bool]:
        """计算终止状态，默认全为 False"""
        return {agent: False for agent in self.agents}
    
    def _compute_truncations(self) -> Dict[str, bool]:
        """计算截断状态，默认全为 False"""
        return {agent: False for agent in self.agents}
    
    @abstractmethod
    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        """将动作索引转换为目标节点"""
        pass

    def _dispatch_move(self, agent_id: int, target_node: int):
        """将目标节点提交给 PatrolWorld。

        默认为单跳邻居移动。全图动作空间的子类（如 SUNS）
        可覆盖为 set_route_action 以支持多跳路由。
        """
        self.world.set_move_action(agent_id, target_node)

    def get_episode_metrics(self) -> Optional[dict]:
        """返回上一个完成 episode 的终止指标。

        world.last_episode_metrics 在每次 reset 时从
        metrics_tracker.current 拷贝，因此调用时机应在 reset 之后。
        子类可覆盖以添加额外指标（如 MASUP 的 wait_ratio）。
        """
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}


class FixedStepEnv(BaseEnv):
    """固定时间步、同步决策环境"""
    
    def __init__(self, config):
        super().__init__(config)
    
    def step(self, actions: Dict[str, int]):
        """执行一步，返回 (obs, rewards, terminations, truncations, infos)"""
        # 1. 只为"可以决策的智能体"设置动作
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])
            
            if self.world.is_ready(agent_id):
                if self.enable_wait:
                    if action_idx == 0:
                        self.world.set_wait_action(agent_id)
                    else:
                        target = self._action_to_target(agent_id, action_idx)
                        self._dispatch_move(agent_id, target)
                else:
                    target = self._action_to_target(agent_id, action_idx)
                    self._dispatch_move(agent_id, target)
        
        # 2. 推进一个时间步
        result = self.world.tick(dt=1.0)
        
        # 3. 构建返回值
        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        truncations = self._compute_truncations()
        infos = self._build_info(result)
        
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        """重置环境，返回 (obs, infos)"""
        self.world.reset()
        self.agents = self.possible_agents[:]
        
        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)
        
        return obs, infos
    
    
class EventDrivenEnv(BaseEnv):
    """事件驱动、同步决策环境"""
    
    def __init__(self, config: Dict):
        super().__init__(config)

    def step(self, actions: Dict[str, int]):
        """执行一步，返回 (obs, rewards, terminations, truncations, infos)"""
        # 1. 只为"可以决策的智能体"设置动作
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])
            
            if self.world.is_ready(agent_id):
                if self.enable_wait:
                    if action_idx == 0:
                        self.world.set_wait_action(agent_id)
                    else:
                        target = self._action_to_target(agent_id, action_idx)
                        self._dispatch_move(agent_id, target)
                else:
                    target = self._action_to_target(agent_id, action_idx)
                    self._dispatch_move(agent_id, target)
        
        # 2. 推进到最近的事件
        result = self.world.tick_to_next_event()
        
        # 3. 构建返回值
        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        truncations = self._compute_truncations()
        infos = self._build_info(result)
        
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        """重置环境，返回 (obs, infos)"""
        self.world.reset()
        self.agents = self.possible_agents[:]
        
        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)
        
        return obs, infos