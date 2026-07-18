# -*- coding: utf-8 -*-
"""Heuristic Conscientious Reactive (HCR) policy.

This implementation follows Algorithm 2 from:
Portugal and Rocha, "Multi-robot patrolling algorithms: examining
performance and scalability", Advanced Robotics, 2013.

For each neighbor v_i of the current vertex:
    NormIdl[v_i] = Idl[v_i] / HIdl
    NormDist[v_i] = (MaxDist - edge_cost(current, v_i)) / MaxDist
    Decision[v_i] = NormIdl[v_i] + NormDist[v_i]

The selected action is argmax(Decision).
"""
from typing import Any, Dict, Optional

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class HCRPolicy(HeuriticBasePolicy):
    """Heuristic Conscientious Reactive policy."""

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
        graph = global_state.get("graph")

        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float64)

        distances = np.asarray(
            [self._edge_cost(graph, current_node, nb) for nb in neighbors],
            dtype=np.float64,
        )

        max_idleness = float(np.max(idleness)) if idleness.size else 0.0
        max_distance = float(np.max(distances)) if distances.size else 0.0

        norm_idleness = (
            idleness / max_idleness
            if max_idleness > 0.0
            else np.zeros_like(idleness, dtype=np.float64)
        )
        norm_distance = (
            (max_distance - distances) / max_distance
            if max_distance > 0.0
            else np.zeros_like(distances, dtype=np.float64)
        )

        decision = norm_idleness + norm_distance
        return int(np.argmax(decision))

    @staticmethod
    def _edge_cost(graph: Any, current_node: int, neighbor: int) -> float:
        if graph is None or current_node is None:
            return 1.0
        get_len = getattr(graph, "get_edge_length", None)
        if not callable(get_len):
            return 1.0
        edge_len = float(get_len(current_node, neighbor) or 0.0)
        return edge_len if edge_len > 0.0 else 1.0
