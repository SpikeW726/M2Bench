# -*- coding: utf-8 -*-
"""
MSP (Multi-agent Segment Patrol) 启发式策略 - 多智能体版本

每个 agent 按照预先配置的固定路线循环巡逻。
路线配置格式：
  routes:
    - [0, 1, 2, 3]   # agent_0 的巡逻路线（节点 ID 列表）
    - [3, 2, 1, 0]   # agent_1 的巡逻路线
    - [1, 3, 0, 2]   # agent_2 的巡逻路线

若目标节点不在当前邻居列表中（因路线与图拓扑不完全匹配），
回退选择邻居列表中的第一个节点（与旧 MSPAgent 保持一致）。
"""
from typing import Dict, List, Optional, Any

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class MSPPolicy(HeuriticBasePolicy):
    """
    MSP 预定路线巡逻策略（多智能体版本）。

    内部维护 per-agent 路线指针 `_indices`，每次决策后指针递增（循环）。
    `reset()` 时清空指针，使每个 episode 从路线开头重新出发。
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # routes: List[List[int]]，每 agent 一条路线；缺失则报错
        routes = config.get("routes", ap.get("routes", None))
        if routes is None:
            raise ValueError(
                "MSPPolicy requires 'routes' in config (list of per-agent node sequences). "
                "Example: routes: [[0,1,2,3], [3,2,1,0]]"
            )
        self.routes: List[List[int]] = [list(r) for r in routes]

        # per-agent 路线指针
        self._indices: Dict[int, int] = {}

    def reset(self):
        """清空所有 agent 的路线指针，每个 episode 从路线头开始。"""
        self._indices.clear()

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

        # 若 agent_id 超出路线配置范围，回退随机
        if agent_id >= len(self.routes) or not self.routes[agent_id]:
            return 0

        route = self.routes[agent_id]
        idx = self._indices.get(agent_id, 0)
        target_node = route[idx % len(route)]
        self._indices[agent_id] = idx + 1

        # 在邻居中查找目标节点
        try:
            return neighbors.index(target_node)
        except ValueError:
            # 目标不在当前邻居中（拓扑不匹配），回退选第一个邻居
            return 0
