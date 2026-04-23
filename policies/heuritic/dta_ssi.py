# -*- coding: utf-8 -*-
"""
DTASSI (Distributed Tabu-list Adaptive Sequential Single Item) 启发式策略

与旧 MultiAgentPatrolling 中 DTASSIAgent 一致：
- 基础分与 DTAGreedy 相同（由基类 _compute_base_scores 计算）
- 在基础分之上，**每个邻居候选**再乘 Uniform(ssi_uniform_low, ssi_uniform_high)，默认 [0.9, 1.1]（与旧代码中固定 randomness 一致）
- **不**使用 DTAGreedy 的 stochastic/score_jitter 路径（SSI 的随机性独立配置）

原框架中 bid_weight / auction_timeout 未参与实际打分，此处仍可从配置读取以兼容，但不影响决策。
"""
import random
import numpy as np
from typing import Any, Dict, Optional

from policies.heuritic.dta_greedy import DTAGreedyPolicy


class DTASSIPolicy(DTAGreedyPolicy):
    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # SSI 专用：逐候选乘性扰动（旧实现固定 0.9~1.1，此处可配）
        self._ssi_lo: float = float(
            ap.get("ssi_uniform_low", config.get("ssi_uniform_low", 0.9))
        )
        self._ssi_hi: float = float(
            ap.get("ssi_uniform_high", config.get("ssi_uniform_high", 1.1))
        )
        if self._ssi_lo > self._ssi_hi:
            self._ssi_lo, self._ssi_hi = self._ssi_hi, self._ssi_lo

        # 兼容旧配置字段，不参与选边
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

        # 与旧 DTASSI 一致：每条 score 再乘独立 Uniform（默认 0.9~1.1）
        for i in range(len(scores)):
            scores[i] *= random.uniform(self._ssi_lo, self._ssi_hi)

        best_idx = int(np.argmax(scores))
        self._update_tabu_after_decision(current_node, tabu)
        return best_idx
