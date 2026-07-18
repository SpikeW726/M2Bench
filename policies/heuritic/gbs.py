# -*- coding: utf-8 -*-
"""Greedy Bayesian Strategy (GBS) policy.

Each neighbor receives a utility based on idleness and its deviation from global
mean idleness. Per-neighbor noise is applied before softmax sampling. ``G1`` and
``G2`` also receive per-agent perturbations to diversify behavior.
"""

import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class GBSPolicy(HeuriticBasePolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.G1: float = float(config.get("G1", ap.get("G1", 0.1))) * (0.9 + 0.2 * random.random())
        self.G2: float = float(config.get("G2", ap.get("G2", 100.0))) * (0.9 + 0.2 * random.random())
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.1)))

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

        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        node_idleness_dict = global_state.get("node_idleness", {})
        global_mean = float(np.mean(list(node_idleness_dict.values()))) if node_idleness_dict else 0.0

        utilities = np.array([
            (0.95 + 0.1 * random.random()) * (
                self.G1 * float(neighbor_idleness[i])
                + self.G2 * (float(neighbor_idleness[i]) - global_mean)
            )
            for i in range(len(neighbors))
        ], dtype=np.float64)

        utilities -= np.max(utilities)
        exp_u = np.exp(utilities)
        probs = exp_u / np.sum(exp_u)
        return int(np.random.choice(len(neighbors), p=probs))
