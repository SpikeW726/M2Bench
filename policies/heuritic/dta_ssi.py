# -*- coding: utf-8 -*-
"""Distributed Tabu-list Adaptive Sequential Single Item (DTA-SSI) policy.

DTA-SSI reuses the deterministic DTA-Greedy base score, then applies independent
uniform noise to each candidate. This noise is configured separately from the
DTA-Greedy stochastic path. Legacy bid and timeout settings do not affect scores.
"""

import random
import numpy as np
from typing import Any, Dict, Optional

from policies.heuritic.dta_greedy import DTAGreedyPolicy

class DTASSIPolicy(DTAGreedyPolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self._ssi_lo: float = float(
            ap.get("ssi_uniform_low", config.get("ssi_uniform_low", 0.9))
        )
        self._ssi_hi: float = float(
            ap.get("ssi_uniform_high", config.get("ssi_uniform_high", 1.1))
        )
        if self._ssi_lo > self._ssi_hi:
            self._ssi_lo, self._ssi_hi = self._ssi_hi, self._ssi_lo

        self.bid_weight: float = float(config.get("bid_weight", ap.get("bid_weight", 1.2)))
        self.auction_timeout: int = int(config.get("auction_timeout", ap.get("auction_timeout", 10)))

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Optional[int]:
        out = self._compute_base_scores(agent_id, obs, global_state)
        if out is None:
            return None

        scores, current_node, tabu = out
        scores = scores.copy()

        for i in range(len(scores)):
            scores[i] *= random.uniform(self._ssi_lo, self._ssi_hi)

        best_idx = int(np.argmax(scores))
        self._update_tabu_after_decision(current_node, tabu)
        return best_idx
