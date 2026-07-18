# -*- coding: utf-8 -*-
"""Cycle-Based strategy with per-agent tabu lists (CBLS).

Each agent excludes recently visited neighbors and samples the remainder from a
softmax over idleness. If every neighbor is tabu, all neighbors become eligible.
Tabu state is reset between episodes.
"""

import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy

class CBLSPolicy(HeuriticBasePolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.1)))

        self.idleness_weight: float = float(config.get("idleness_weight", ap.get("idleness_weight", 1.0)))

        self.tabu_length: int = int(config.get("tabu_length", ap.get("tabu_length", 3)))

        self._tabu_lists: Dict[int, List[int]] = {}

    def reset(self):
        self._tabu_lists.clear()

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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

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
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        tabu = self._tabu_lists.setdefault(agent_id, [])

        filtered = [
            (i, float(neighbor_idleness[i]))
            for i, nb in enumerate(neighbors)
            if nb not in tabu
        ]

        if not filtered:
            filtered = [(i, float(neighbor_idleness[i])) for i in range(len(neighbors))]

        indices = [pair[0] for pair in filtered]
        values = np.array([pair[1] for pair in filtered], dtype=np.float32)
        probs = self._softmax(values * self.idleness_weight)

        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            selected_idx = random.choice(indices)
        else:
            selected_idx = int(np.random.choice(indices, p=probs))

        if current_node is not None:
            tabu.append(int(current_node))
            if len(tabu) > self.tabu_length:
                tabu.pop(0)

        return selected_idx
