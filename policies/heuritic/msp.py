# -*- coding: utf-8 -*-
"""Multi-agent Segment Patrol (MSP) policy.

Each agent repeatedly follows its configured cyclic route. If the next route node
is not adjacent under the current graph, the policy falls back to the first
available neighbor. Route indices restart at every episode reset.
"""

from typing import Dict, List, Optional, Any

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class MSPPolicy(HeuriticBasePolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # routes: List[List[int]].
        routes = config.get("routes", ap.get("routes", None))
        if routes is None:
            raise ValueError(
                "MSPPolicy requires 'routes' in config (list of per-agent node sequences). "
                "Example: routes: [[0,1,2,3], [3,2,1,0]]"
            )
        self.routes: List[List[int]] = [list(r) for r in routes]

        self._indices: Dict[int, int] = {}

    def reset(self):
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

        if agent_id >= len(self.routes) or not self.routes[agent_id]:
            return 0

        route = self.routes[agent_id]
        idx = self._indices.get(agent_id, 0)
        target_node = route[idx % len(route)]
        self._indices[agent_id] = idx + 1

        try:
            return neighbors.index(target_node)
        except ValueError:

            return 0
