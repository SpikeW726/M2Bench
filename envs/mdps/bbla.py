"""
Paper Implementation: Multi-Agent Patrolling with Reinforcement Learning
Authors: Hugo Santana, Geber Ramalho, Vincent Corruble, Bohdana Ratitch
Year: 2004
Venue: AAMAS
Link: https://ieeexplore.ieee.org/document/1373634

Description:
    This script implements the Black-Box Learner Agent (BBLA) architecture 
    described in Section 4.4 of the paper.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import FixedStepEnv

class BBLAEnv(FixedStepEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        # Episode 截止模式开关
        # truncate_by_time=True (默认): 使用总时间 all_timer >= episode_len 截止
        # truncate_by_time=False: 使用 step 数量 step_cnt >= episode_len 截止
        self.truncate_by_time = kwargs.get('truncate_by_time', True)

        # 确定的物理特征
        self.episode_len = config['episode_len']

    def observation_space(self, agent):
        """
        BBLA observation space: [current_pos, last_pos, max_idleness_node, min_idleness_node]
        """        
        max_node_id = self.world.num_nodes - 1
        max_num_neighbor = self.world.max_neighbors

        low = np.array([-1, -1, -1, -1])
        high = np.array([max_node_id, max_num_neighbor, max_num_neighbor, max_num_neighbor])

        return Box(low=low, high=high, dtype=np.int32)

    def action_space(self, agent):
        return Discrete(self.world.max_neighbors + 1) # The last dimension means no-op


    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        obs = {}

        for agent_id in range(self.world.num_agents):
            if self.world.agents[agent_id].state == AgentState.ON_EDGE:
                # 智能体在边上，返回特殊观测
                obs[f"agent_{agent_id}"] = np.full(4, -1, dtype=np.int32)
                continue

            current_pos = self.world.agents[agent_id].position
            raw_last = self.world.agents[agent_id].last_position
            if raw_last == current_pos:
                # 初始状态：last_position == position，用 -1 标记
                last_pos = -1
            else:
                last_pos = self.world.graph.neighbor_to_edge(current_pos, raw_last)

            # 获取邻居信息
            neighbors = [n for n, _ in self.world.graph.adj_list.get(current_pos, [])]
            
            # 计算最大/最小空闲度邻居
            if neighbors:
                neighbor_idleness = [self.world.node_idleness[n] for n in neighbors]
                max_idle = max(neighbor_idleness)
                min_idle = min(neighbor_idleness)
                
                max_nodes = [n for n, idle in zip(neighbors, neighbor_idleness) if idle == max_idle]
                min_nodes = [n for n, idle in zip(neighbors, neighbor_idleness) if idle == min_idle]

                # 平局时随机选取
                max_node = self.world.graph.neighbor_to_edge(current_pos, random.choice(max_nodes))
                min_node = self.world.graph.neighbor_to_edge(current_pos, random.choice(max_nodes))
            else:
                max_node = min_node = -1
            
            single_obs = np.zeros(4, dtype=np.int32)
            single_obs[0] = current_pos
            single_obs[1] = last_pos
            single_obs[2] = max_node
            single_obs[3] = min_node
            obs[f"agent_{agent_id}"] = single_obs

        return obs

    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        """
        构建所有智能体的 info
        """
        infos = {}

        for agent_str in self.agents:
            agent_id = int(agent_str.split('_')[1])
            action_mask = self.get_action_mask(agent_str)
            active_mask = 1 if self.world.is_ready(agent_id) else 0 

            infos[agent_str] = {"action_mask": action_mask, "active_mask": active_mask}
        
        return infos

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        return {f"agent_{id}": reward for id, reward in result.raw_rewards.items()}

    def _compute_truncations(self) -> Dict[str, bool]:
        """计算截断状态"""
        if self.truncate_by_time:
            is_truncated = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            is_truncated = self.world.step_count >= self.episode_len
        return {agent: is_truncated for agent in self.agents}

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        """将动作索引转换为目标节点"""
        current_pos = self.world.get_position(agent_id)
        neighbors = self.world.graph.get_neighbors(current_pos)
        
        # action_idx: 0~N-1=neighbors, N=no-op
        neighbor_idx = action_idx
        if neighbor_idx < len(neighbors):
            return neighbors[neighbor_idx]
        else: # 前面已经跳过了还未完成上一动作的智能体,这里不会传入no-op动作的index
            raise ValueError(
                f"无效动作: agent_id={agent_id}, action_idx={action_idx}, "
                f"可用邻居数量={len(neighbors)}"
            )    

    def state(self) -> np.ndarray:
        """全局状态: 所有智能体位置 + 全节点空闲度 + 当前时间"""
        agent_metrics = []
        for agent_id in range(self.world.num_agents):
            agent = self.world.agents[agent_id]
            last_pos = float(agent.last_position)
            target = float(agent.target_node)
            time_left = float(agent.action_remaining)
            agent_metrics.extend([last_pos, target, time_left])

        idleness = [
            float(self.world.graph.phi.get(n, 1.0)) * float(self.world.node_idleness.get(n, 0.0))
            for n in self.world.graph.nodes
        ]

        return np.asarray(agent_metrics + idleness + [float(self.world.current_time)], dtype=np.float32)

    # ==================== 辅助方法 ====================
    
    def get_action_mask(self, agent_str: str) -> np.ndarray:
        """获取动作掩码"""
        agent_id = int(agent_str.split('_')[1])
        mask = np.zeros(self.world.max_neighbors+1, dtype=bool)

        if not self.world.is_ready(agent_id):
            # 正在执行动作,只能选 no-op
            mask[-1] = True
        else:
            # 可以决策: 邻居节点
            current_pos = self.world.get_position(agent_id)
            neighbors = self.world.graph.get_neighbors(current_pos)
            mask[:len(neighbors)] = True
                      
        return mask

    def get_valid_actions(self, agent_str: str) -> List[int]:
        """返回有效动作索引列表"""
        mask = self.get_action_mask(agent_str)
        return [int(x) for x in np.where(mask)[0]]