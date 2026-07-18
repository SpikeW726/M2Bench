# -*- coding: utf-8 -*-
"""Heuristic Pathfinder Conscientious Cognitive (HPCC) policy.

This implementation follows Algorithm 3 from:
Portugal and Rocha, "Multi-robot patrolling algorithms: examining
performance and scalability", Advanced Robotics, 2013.

At a decision point the policy:
  1. chooses a global target vertex by maximizing idleness-distance utility;
  2. builds dynamic edge costs from destination idleness and original edge cost;
  3. runs Dijkstra with those dynamic costs;
  4. follows the planned path one hop at a time through the existing
     neighbor-index heuristic interface.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class HPCCPolicy(HeuriticBasePolicy):
    """Heuristic Pathfinder Conscientious Cognitive policy."""

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)
        self._path_cache: Dict[int, List[int]] = {}

    def reset(self):
        self._path_cache.clear()

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
        current_node = obs.get("current_node")
        neighbors = obs.get("neighbors", [])
        if current_node is None or not neighbors:
            self._path_cache.pop(agent_id, None)
            return None

        cached_action = self._pop_cached_action(agent_id, neighbors)
        if cached_action is not None:
            return cached_action

        graph = global_state.get("graph")
        if graph is None:
            return None

        node_idleness = {
            node: float(value)
            for node, value in global_state.get("node_idleness", {}).items()
        }

        target = self._select_target(graph, current_node, node_idleness)
        if target is None:
            return None

        edge_costs = self._build_dynamic_edge_costs(graph, node_idleness)
        path = self._dijkstra_path(graph, current_node, target, edge_costs)
        if path is None or len(path) < 2:
            return None

        remaining = path[1:]
        next_node = remaining.pop(0)
        if next_node not in neighbors:
            return None

        if remaining:
            self._path_cache[agent_id] = remaining
        else:
            self._path_cache.pop(agent_id, None)

        return int(neighbors.index(next_node))

    def _pop_cached_action(
        self,
        agent_id: int,
        neighbors: List[int],
    ) -> Optional[int]:
        path = self._path_cache.get(agent_id)
        if not path:
            return None

        next_node = path[0]
        if next_node not in neighbors:
            self._path_cache.pop(agent_id, None)
            return None

        path.pop(0)
        if not path:
            self._path_cache.pop(agent_id, None)
        return int(neighbors.index(next_node))

    def _select_target(
        self,
        graph: Any,
        current_node: int,
        node_idleness: Dict[int, float],
    ) -> Optional[int]:
        nodes = list(getattr(graph, "nodes", []))
        candidates = [node for node in nodes if node != current_node]
        if not candidates:
            return None

        max_idleness = max((node_idleness.get(node, 0.0) for node in nodes), default=0.0)
        distances = {
            node: float(graph.shortest_path_length(current_node, node))
            for node in candidates
        }
        finite_distances = [dist for dist in distances.values() if math.isfinite(dist)]
        if not finite_distances:
            return None

        max_distance = max(finite_distances)
        best_node: Optional[int] = None
        best_decision = -math.inf

        for node in candidates:
            dist = distances[node]
            if not math.isfinite(dist):
                continue

            norm_idleness = (
                node_idleness.get(node, 0.0) / max_idleness
                if max_idleness > 0.0
                else 0.0
            )
            norm_distance = (
                (max_distance - dist) / max_distance
                if max_distance > 0.0
                else 0.0
            )
            decision = norm_idleness + norm_distance

            if decision > best_decision:
                best_decision = decision
                best_node = node

        return best_node

    def _build_dynamic_edge_costs(
        self,
        graph: Any,
        node_idleness: Dict[int, float],
    ) -> Dict[Tuple[int, int], float]:
        nodes = list(getattr(graph, "nodes", []))
        max_idleness = max((node_idleness.get(node, 0.0) for node in nodes), default=0.0)

        edges: List[Tuple[int, int, float]] = []
        for src in nodes:
            for dst, cost in graph.adj_list.get(src, []):
                edges.append((src, dst, float(cost)))

        if not edges:
            return {}

        edge_values = [cost for _, _, cost in edges]
        min_cost = min(edge_values)
        max_cost = max(edge_values)
        cost_range = max_cost - min_cost

        dynamic: Dict[Tuple[int, int], float] = {}
        for src, dst, cost in edges:
            idle_cost = (
                (max_idleness - node_idleness.get(dst, 0.0)) / max_idleness
                if max_idleness > 0.0
                else 0.0
            )
            dist_cost = (
                (cost - min_cost) / cost_range
                if cost_range > 0.0
                else 0.0
            )
            dynamic[(src, dst)] = idle_cost + dist_cost

        return dynamic

    def _dijkstra_path(
        self,
        graph: Any,
        source: int,
        target: int,
        edge_costs: Dict[Tuple[int, int], float],
    ) -> Optional[List[int]]:
        dist = {node: math.inf for node in graph.nodes}
        prev: Dict[int, int] = {}
        dist[source] = 0.0
        queue: List[Tuple[float, int]] = [(0.0, source)]
        visited = set()

        while queue:
            cur_dist, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)

            if node == target:
                break

            for nxt, _ in graph.adj_list.get(node, []):
                weight = edge_costs.get((node, nxt), math.inf)
                if not math.isfinite(weight):
                    continue
                new_dist = cur_dist + weight
                if new_dist < dist[nxt]:
                    dist[nxt] = new_dist
                    prev[nxt] = node
                    heapq.heappush(queue, (new_dist, nxt))

        if not math.isfinite(dist.get(target, math.inf)):
            return None

        path = [target]
        node = target
        while node != source:
            node = prev.get(node)
            if node is None:
                return None
            path.append(node)

        return list(reversed(path))
