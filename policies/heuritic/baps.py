# -*- coding: utf-8 -*-
"""Bayesian Ant Patrolling Strategy (BAPS) policy.

Neighbor scores combine normalized idleness, pheromone, travel distance, and
target conflicts; lower scores are preferred. Pheromone values are read from
``global_state['baps_pheromone']`` and default to zero when unavailable.
"""

import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class BAPSPolicy(HeuriticBasePolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # Distance lookup mode.
        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "edge"))).lower()

        self.w_idl: float = float(config.get("w_idl", ap.get("w_idl", 0.6)))
        self.w_pher: float = float(config.get("w_pher", ap.get("w_pher", 0.2)))
        self.w_dist: float = float(config.get("w_dist", ap.get("w_dist", 0.2)))
        self.w_conflict: float = float(config.get("w_conflict", ap.get("w_conflict", 0.4)))

        # Exploration and sampling.
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.0)))

        self.softmax_tau: float = float(config.get("softmax_tau", ap.get("softmax_tau", 0.0)))

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

        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            return random.randrange(len(neighbors))

        current_node = obs.get("current_node")

        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        # Optional pheromone values; defaults to zero.
        baps_pheromone: Dict = global_state.get("baps_pheromone", {})
        pheromone = np.asarray(
            [float(baps_pheromone.get(nb, 0.0)) for nb in neighbors], dtype=np.float32
        )

        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph)
        conflicts = self._neighbor_conflicts(agent_id, neighbors, global_state)

        nidle = self._norm_inverted(neighbor_idleness)
        npher = self._norm_inverted(pheromone)
        ndist = self._norm_minmax(distances)
        nconf = conflicts

        score = (
            self.w_idl * nidle
            + self.w_pher * npher
            + self.w_dist * ndist
            + self.w_conflict * nconf
        )

        if self.softmax_tau <= 0.0:
            return int(np.argmin(score))
        else:
            logits = -score / max(self.softmax_tau, 1e-6)
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.clip(np.sum(probs), 1e-8, None)
            return int(np.random.choice(len(neighbors), p=probs))
