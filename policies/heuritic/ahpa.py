# -*- coding: utf-8 -*-
"""Adaptive Heuristic Patrolling Agent (AHPA) policy.

The original distributed algorithm maintains local beliefs synchronized through
probabilistic communication. ``PatrolWorld`` exposes global idleness instead,
which is equivalent to perfect communication. Candidates minimize a weighted
sum of normalized idleness, distance, target conflicts, and backtracking.
"""

import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class AHPAPolicy(HeuriticBasePolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.w_idl: float = float(config.get("w_idl", ap.get("w_idl", 0.7)))
        self.w_dist: float = float(config.get("w_dist", ap.get("w_dist", 0.2)))
        self.w_conflict: float = float(config.get("w_conflict", ap.get("w_conflict", 0.3)))
        self.w_back: float = float(config.get("w_back", ap.get("w_back", 0.2)))

        # Sampling and exploration.
        self.softmax_tau: float = float(config.get("softmax_tau", ap.get("softmax_tau", 0.0)))
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.0)))

        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "edge"))).lower()

        self._last_vertex: Dict[int, Optional[int]] = {}

    def reset(self):
        self._last_vertex.clear()

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

        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            self._last_vertex[agent_id] = current_node
            return random.randrange(len(neighbors))

        node_idleness = global_state.get("node_idleness", {})
        idl_vals = np.asarray(
            [float(node_idleness.get(nb, 0.0)) for nb in neighbors], dtype=np.float32
        )

        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph)

        conflicts = self._neighbor_conflicts(agent_id, neighbors, global_state)

        last_v = self._last_vertex.get(agent_id)
        back = np.asarray(
            [1.0 if last_v is not None and nb == last_v else 0.0 for nb in neighbors],
            dtype=np.float32,
        )

        nidle = self._norm_inverted(idl_vals)
        ndist = self._norm_minmax(distances)

        score = (
            self.w_idl * nidle
            + self.w_dist * ndist
            + self.w_conflict * conflicts
            + self.w_back * back
        )

        # Remember the current node for the next decision.
        self._last_vertex[agent_id] = current_node

        if self.softmax_tau <= 0.0:
            return int(np.argmin(score))
        else:
            logits = -score / max(self.softmax_tau, 1e-6)
            logits -= np.max(logits)
            probs = np.exp(np.clip(logits, -60, 60))
            s = float(np.sum(probs))
            if not np.isfinite(s) or s <= 0:
                return int(np.argmin(score))
            probs /= s
            return int(np.random.choice(len(neighbors), p=probs))
