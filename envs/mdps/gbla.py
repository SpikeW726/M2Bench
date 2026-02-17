"""
Paper Implementation: Multi-Agent Patrolling with Reinforcement Learning
Authors: Hugo Santana, Geber Ramalho, Vincent Corruble, Bohdana Ratitch
Year: 2004
Venue: AAMAS
Link: https://ieeexplore.ieee.org/document/1373634

Description:
    This script implements the Grey-Box Learner Agent (GBLA) architecture 
    described in Section 4.4 of the paper.
"""

from typing import Any, Dict, Optional
import numpy as np
from gymnasium.spaces import Box

from bbla import BBLAEnv
from patrol_core import AgentState, TickResult


class GBLAEnv(BBLAEnv):

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        # agent_id -> 意图目标节点 (None 表示无意图)
        self.agent_intentions: Dict[int, Optional[int]] = {}

    def observation_space(self, agent: str):
        """
        GBLA obs = BBLA obs + neighbor_intentions
        [current_pos, last_pos, max_node, min_node, intent_0, ..., intent_{M-1}]
        size: 4 + max_neighbors
        """
        max_node_id = self.world.num_nodes - 1
        max_num_neighbor = self.world.max_neighbors

        low = np.concatenate([
            np.array([-1, -1, -1, -1]),                    # BBLA 部分
            np.full(max_num_neighbor, -1),                  # 意图向量 (-1=padding, 0=无意图, 1=有意图)
        ]).astype(np.int32)

        high = np.concatenate([
            np.array([max_node_id, max_node_id,
                      max_num_neighbor, max_num_neighbor]),  # BBLA 部分
            np.ones(max_num_neighbor),                       # 意图向量
        ]).astype(np.int32)

        return Box(low=low, high=high, dtype=np.int32)

    def step(self, actions: Dict[str, int]):
        """step 前从动作中提取并记录各智能体意图"""
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])
            if self.world.is_ready(agent_id):
                current_pos = self.world.get_position(agent_id)
                neighbors = self.world.graph.get_neighbors(current_pos)
                if action_idx < len(neighbors):
                    # 有效移动 → 意图 = 目标节点
                    self.agent_intentions[agent_id] = neighbors[action_idx]
                else:
                    # no-op → 意图 = 留在当前节点
                    self.agent_intentions[agent_id] = current_pos

        return super().step(actions)

    def reset(self, seed: Optional[int] = None):
        """重置时清空意图表"""
        self.agent_intentions = {}
        return super().reset(seed)
        

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        """构建 GBLA 观测: BBLA 观测 + 邻居意图向量 (-1=padding, 0=无意图, 1=有意图)"""
        bbla_obs_dict = super()._build_obs(result)
        M = self.world.max_neighbors

        obs = {}
        for agent_id in range(self.world.num_agents):
            agent_str = f"agent_{agent_id}"
            bbla_obs = bbla_obs_dict[agent_str]

            intention_vec = np.full(M, -1, dtype=np.int32)  # 默认 padding

            if self.world.agents[agent_id].state != AgentState.ON_EDGE:
                # 获取当前节点的邻居列表（已排序，与动作索引对齐）
                current_pos = self.world.agents[agent_id].position
                neighbors = self.world.graph.get_neighbors(current_pos)

                # 实际邻居范围初始化为 0（无意图）
                intention_vec[:len(neighbors)] = 0

                # 标记：是否有其他智能体意图访问该邻居
                for other_id, target in self.agent_intentions.items():
                    if other_id != agent_id and target in neighbors:
                        idx = neighbors.index(target)
                        intention_vec[idx] = 1

            obs[agent_str] = np.concatenate([bbla_obs, intention_vec])

        return obs



