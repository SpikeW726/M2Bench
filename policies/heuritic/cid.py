# -*- coding: utf-8 -*-
"""
CID (Combining Idleness & Distance) 启发式策略 - 多智能体版本

分数越小越好（与旧 CIDAgent 保持一致）：
  score = ir * nidle_norm + (1 - ir) * ndist_norm
  - nidle_norm = 1 - minmax(idleness)  # 高空闲 → 小（更优先）
  - ndist_norm = minmax(distance)      # 短距离 → 小（更优先）
  - ir ∈ [0, 1]（默认 0.6，越大越重视空闲度）

距离计算使用基类 _neighbor_distances，支持 'edge' / 'sp' 两种模式。
"""
import random
from typing import Dict, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class CIDPolicy(HeuriticBasePolicy):
    """
    CID 启发式策略（多智能体版本）

    设计：
    - compute_actions: 并发处理所有 READY 智能体（非顺序）
    - _compute_action: 单智能体二因子打分 + argmin 选择
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # ir：空闲度权重（与旧 CIDAgent 默认值一致）
        self.ir: float = float(np.clip(config.get("ir", ap.get("ir", 0.6)), 0.0, 1.0))
        # 距离读取方式
        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "edge"))).lower()
        # 探索率
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.0)))

    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        为所有 READY 智能体计算动作（并发，非顺序）。

        Args:
            obs_dict: {agent_str: obs}，obs 包含 'current_node', 'neighbors',
                      'neighbor_idleness'（可选）, 'on_edge'
            global_state: 含 'graph', 'node_idleness', 'agents_on_edge' 等

        Returns:
            actions: {agent_str: action_idx}
        """
        actions: Dict[str, int] = {}

        for agent_str, obs in obs_dict.items():
            agent_id = int(agent_str.split("_")[1]) if isinstance(agent_str, str) else int(agent_str)

            on_edge = obs.get("on_edge", False)
            if global_state.get("agents_on_edge"):
                on_edge = global_state["agents_on_edge"].get(agent_id, on_edge)
            if on_edge:
                continue

            action = self._compute_action(agent_id, obs, global_state)
            if action is not None:
                actions[agent_str] = action

        return actions

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Optional[int]:
        """
        单智能体 CID 决策：二因子归一化打分，取最小分数的邻居。
        """
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        # ε-贪心探索
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            return random.randrange(len(neighbors))

        current_node = obs.get("current_node")

        # 邻居空闲度
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        # 距离（调用基类工具方法，自动使用 self.distance_mode）
        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph)

        # 归一化（更优 → 值越小）
        nidle_norm = self._norm_inverted(neighbor_idleness)   # 高空闲 → 小
        ndist_norm = self._norm_minmax(distances)              # 短距离 → 小

        score = self.ir * nidle_norm + (1.0 - self.ir) * ndist_norm

        return int(np.argmin(score))
