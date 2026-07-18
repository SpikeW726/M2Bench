# -*- coding: utf-8 -*-
"""Conscientious Cognitive heuristic policy.

Neighbors maximize a weighted sum of normalized idleness and inverse distance,
with small random noise for tie-breaking. Decisions are concurrent and use no
inter-agent coordination.
"""

import random
from typing import Dict, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class ConscientiousCognitivePolicy(HeuriticBasePolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.idleness_weight: float = float(config.get("idleness_weight", ap.get("idleness_weight", 1.0)))
        self.distance_weight: float = float(config.get("distance_weight", ap.get("distance_weight", 0.5)))

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

        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(nb, 0.0) for nb in neighbors]
        idl = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float64)

        max_idl = float(np.max(idl)) if len(idl) > 0 else 0.0

        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph).astype(np.float64)

        idl_component = (idl / max_idl if max_idl > 0 else np.zeros_like(idl)) * self.idleness_weight
        dist_component = (1.0 / (1.0 + distances)) * self.distance_weight

        scores = idl_component + dist_component

        scores += np.random.uniform(0.0, 0.01, size=len(neighbors))

        return int(np.argmax(scores))
