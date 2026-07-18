# -*- coding: utf-8 -*-
"""State Exchange Bayesian Strategy (SEBS) policy.

SEBS shares the GBS utility function but uses unperturbed ``G1`` and ``G2``.
It applies one scalar noise value to all candidates and selects an argmax with
random tie-breaking instead of GBS per-neighbor noise and softmax sampling.
"""

import random
from typing import Dict, Optional, Any

import numpy as np

from policies.heuritic.gbs import GBSPolicy

class SEBSPolicy(GBSPolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.G1 = float(config.get("G1", ap.get("G1", 0.1)))
        self.G2 = float(config.get("G2", ap.get("G2", 100.0)))
        self.exploration_rate = float(config.get("exploration_rate", ap.get("exploration_rate", 0.3)))

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

        utilities = self.G1 * neighbor_idleness + self.G2 * (neighbor_idleness - global_mean)
        utilities = utilities * (1.0 + 0.2 * (random.random() - 0.5))

        # argmax + random tie-breaking.
        max_u = float(np.max(utilities))
        best_indices = list(np.where(utilities == max_u)[0])
        return int(random.choice(best_indices))
