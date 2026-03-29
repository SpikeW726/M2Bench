"""
Paper Implementation: Robust Multi-agent Patrolling Strategies Using Reinforcement Learning
Authors: Fabrice Lauri, Abderrafiaa Koukam 
Year: 2014
Venue: Swarm Intelligence Based Optimization
Link: https://link.springer.com/chapter/10.1007/978-3-319-12970-9_17

Description:
    Extended Gray-Box Learner Agent (Ex-GBLA) 在 GBLA 基础上做了两点改进:
    1. 状态空间: 用按 idleness 降序排列的邻居列表替代 GBLA 中的 max/min 邻居
    2. 奖励函数: 到达一个已被其他智能体当作目标的节点时, reward 置 0
"""

from typing import Any, Dict, Optional
import numpy as np
from gymnasium.spaces import Box

from envs.mdps.gbla import GBLAEnv
from envs.mdps.patrol_core import AgentState, TickResult


class ExGBLAEnv(GBLAEnv):
    """Extended Gray-Box Learner Agent 环境

    观测空间: [current_pos, last_pos, sorted_neighbors(M), intentions(M)]
        - sorted_neighbors: 邻居按 idleness 降序排列的边索引, 不足 M 个补 -1
        - intentions: 与 GBLA 相同的二进制意图向量
    奖励: 节点空闲时间, 但到达节点若已被其他智能体当作目标则为 0
    """

    def observation_space(self, agent: str):
        """
        Ex-GBLA obs: [current_pos, last_pos, sorted_neighbors(M), intentions(M)]
        size: 2 + 2 * max_neighbors
        """
        max_node_id = self.world.num_nodes - 1
        M = self.world.max_neighbors

        low = np.concatenate([
            np.array([-1, -1]),         # current_pos, last_pos
            np.full(M, -1),             # sorted_neighbors (边索引, -1=padding)
            np.full(M, -1),             # intentions (-1=padding, 0=无意图, 1=有意图)
        ]).astype(np.int32)

        high = np.concatenate([
            np.array([max_node_id, max_node_id]),
            np.full(M, M),              # 边索引上界, 与 BBLA 保持一致
            np.ones(M),
        ]).astype(np.int32)

        return Box(low=low, high=high, dtype=np.int32)

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        """构建 Ex-GBLA 观测: [current_pos, last_pos, sorted_neighbors, intentions]"""
        M = self.world.max_neighbors
        obs = {}

        for agent_id in range(self.world.num_agents):
            agent_str = f"agent_{agent_id}"

            # ---- 在边上: 全部 -1 ----
            if self.world.agents[agent_id].state == AgentState.ON_EDGE:
                obs[agent_str] = np.full(2 + 2 * M, -1, dtype=np.int32)
                continue

            current_pos = self.world.agents[agent_id].position
            raw_last = self.world.agents[agent_id].last_position
            if raw_last == current_pos:
                last_edge = -1
            else:
                last_edge = self.world.graph.neighbor_to_edge(current_pos, raw_last)

            # ---- 邻居按 phi 加权空闲度降序排列, 转为边索引 ----
            neighbors = [n for n, _ in self.world.graph.adj_list.get(current_pos, [])]
            sorted_edges = np.full(M, -1, dtype=np.int32)
            if neighbors:
                pairs = [
                    (n, self.world.graph.phi.get(n, 1.0) * self.world.node_idleness[n])
                    for n in neighbors
                ]
                pairs.sort(key=lambda x: x[1], reverse=True)
                for i, (n, _) in enumerate(pairs):
                    sorted_edges[i] = self.world.graph.neighbor_to_edge(current_pos, n)

            # ---- 意图向量 (与动作索引对齐, -1=padding, 0=无意图, 1=有意图) ----
            intentions = np.full(M, -1, dtype=np.int32)
            action_neighbors = self.world.graph.get_neighbors(current_pos)
            intentions[:len(action_neighbors)] = 0
            for other_id, target in self.agent_intentions.items():
                if other_id != agent_id and target in action_neighbors:
                    idx = action_neighbors.index(target)
                    intentions[idx] = 1

            # ---- 拼接 ----
            single_obs = np.empty(2 + 2 * M, dtype=np.int32)
            single_obs[0] = current_pos
            single_obs[1] = last_edge
            single_obs[2:2 + M] = sorted_edges
            single_obs[2 + M:] = intentions
            obs[agent_str] = single_obs

        return obs

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        """奖励 = 节点空闲时间; 若到达节点已被其他智能体当作目标则置 0"""
        rewards = super()._compute_rewards(result)

        for agent_id, arrived_node in result.arrivals.items():
            agent_str = f"agent_{agent_id}"
            # 检查是否有其他智能体也以该节点为目标
            for other_id, target in self.agent_intentions.items():
                if other_id != agent_id and target == arrived_node:
                    rewards[agent_str] = 0.0
                    break

        return rewards
