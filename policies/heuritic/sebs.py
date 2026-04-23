# -*- coding: utf-8 -*-
"""
SEBS (State Exchange Bayesian Strategy) 启发式策略 - 多智能体版本

继承自 GBSPolicy，使用相同的效用函数：
  U = G1*idl + G2*(idl - global_mean)
但有以下两点差异（与旧 SEBSAgent 保持一致）：
  1. G1/G2 不做 per-agent 扰动，直接使用配置值
  2. 对所有 utilities 施加整体标量噪声，选择用 argmax + random tie-breaking
     而非 GBS 的 per-neighbor 噪声 + softmax 采样
"""
import random
from typing import Dict, Optional, Any

import numpy as np

from policies.heuritic.gbs import GBSPolicy


class SEBSPolicy(GBSPolicy):
    """
    SEBS 启发式策略（多智能体版本）

    继承 GBSPolicy 以复用：
    - compute_actions（相同的并发调度逻辑）
    - G1/G2/exploration_rate 参数读取逻辑（__init__ 覆盖掉扰动部分）

    覆盖 _compute_action 以实现 SEBS 特有的整体标量噪声 + argmax 选择。
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # 不做 per-agent 扰动（与旧 SEBSAgent.__init__ 一致）
        self.G1 = float(config.get("G1", ap.get("G1", 0.1)))
        self.G2 = float(config.get("G2", ap.get("G2", 100.0)))
        self.exploration_rate = float(config.get("exploration_rate", ap.get("exploration_rate", 0.3)))

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Optional[int]:
        """
        单智能体 SEBS 决策：整体标量噪声 + argmax + random tie-breaking。
        """
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        # ε-贪心探索
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            return random.randrange(len(neighbors))

        # 邻居空闲度
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        # 全局平均空闲度
        node_idleness_dict = global_state.get("node_idleness", {})
        global_mean = float(np.mean(list(node_idleness_dict.values()))) if node_idleness_dict else 0.0

        # 效用（向量化），然后施加整体标量噪声
        utilities = self.G1 * neighbor_idleness + self.G2 * (neighbor_idleness - global_mean)
        utilities = utilities * (1.0 + 0.2 * (random.random() - 0.5))

        # argmax + random tie-breaking（与旧 SEBSAgent 完全一致）
        max_u = float(np.max(utilities))
        best_indices = list(np.where(utilities == max_u)[0])
        return int(random.choice(best_indices))
