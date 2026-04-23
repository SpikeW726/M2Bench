# -*- coding: utf-8 -*-
"""
Random 启发式策略 - 多智能体版本

每个 READY agent 从可用邻居中均匀随机选择一个目标节点，无任何协调或学习。
"""
import random
from typing import Dict, Optional, Any

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class RandomPolicy(HeuriticBasePolicy):
    """随机巡逻策略，并发处理所有 READY agents。"""

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
        return random.randrange(len(neighbors))
