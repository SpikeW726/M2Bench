# -*- coding: utf-8 -*-
"""
Conscientious Cognitive 启发式策略 - 多智能体版本

打分公式（与旧 ConscientiousCognitiveAgent 保持一致）：
  score = idleness_weight * (idl / max_idl) + distance_weight * (1 / (1 + dist))
          + uniform(0, 0.01)   # 轻微噪声打破平局
  （越大越优先，取 argmax）

"Cognitive"：同时考虑空闲度与距离两个因素，优于纯 Reactive。
距离通过基类 _neighbor_distances 从图结构读取。
"""
import random
from typing import Dict, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class ConscientiousCognitivePolicy(HeuriticBasePolicy):
    """
    Conscientious Cognitive 启发式策略（多智能体版本）。

    设计：
    - compute_actions: 并发处理所有 READY agents
    - _compute_action: idl+dist 加权打分 + argmax
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.idleness_weight: float = float(config.get("idleness_weight", ap.get("idleness_weight", 1.0)))
        self.distance_weight: float = float(config.get("distance_weight", ap.get("distance_weight", 0.5)))
        # 距离模式（基类 _neighbor_distances 使用）
        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "edge"))).lower()

    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Dict[str, int]:
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
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        current_node = obs.get("current_node")

        # 邻居空闲度
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(nb, 0.0) for nb in neighbors]
        idl = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float64)

        max_idl = float(np.max(idl)) if len(idl) > 0 else 0.0

        # 距离（调用基类工具方法）
        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph).astype(np.float64)

        # 打分（与旧 ConscientiousCognitiveAgent 完全一致）
        idl_component = (idl / max_idl if max_idl > 0 else np.zeros_like(idl)) * self.idleness_weight
        dist_component = (1.0 / (1.0 + distances)) * self.distance_weight

        scores = idl_component + dist_component
        # 轻微噪声打破平局（与旧实现 random.uniform(0, 0.01) 一致）
        scores += np.random.uniform(0.0, 0.01, size=len(neighbors))

        return int(np.argmax(scores))
