# -*- coding: utf-8 -*-
"""
GBS (Greedy Bayesian Strategy) 启发式策略 - 多智能体版本

核心思想：
  对每个邻居 v_i 计算效用 U_i = noise_i * (G1*idl_i + G2*(idl_i - global_mean))
  其中 noise_i ∈ [0.95, 1.05] 为 per-neighbor 随机扰动。
  然后通过 softmax 按概率分布采样，而非贪心选择最大值。

G1/G2 参数在 __init__ 时加 ±10% per-agent 扰动以区分不同智能体行为。
"""
import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class GBSPolicy(HeuriticBasePolicy):
    """
    GBS 启发式策略（多智能体版本）

    设计：
    - compute_actions: 并发处理所有 READY 智能体（非顺序）
    - _compute_action: 单智能体 GBS 决策逻辑
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # per-agent ±10% 扰动（与旧 GBSAgent 保持一致）
        self.G1: float = float(config.get("G1", ap.get("G1", 0.1))) * (0.9 + 0.2 * random.random())
        self.G2: float = float(config.get("G2", ap.get("G2", 100.0))) * (0.9 + 0.2 * random.random())
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.1)))

    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        为所有 READY 智能体计算动作（并发，非顺序）。

        Args:
            obs_dict: {agent_str: obs}，obs 包含 'current_node', 'neighbors',
                      'neighbor_idleness'（可选）, 'on_edge'
            global_state: 含 'node_idleness', 'agents_on_edge' 等

        Returns:
            actions: {agent_str: action_idx}
        """
        actions: Dict[str, int] = {}

        for agent_str, obs in obs_dict.items():
            agent_id = int(agent_str.split("_")[1]) if isinstance(agent_str, str) else int(agent_str)

            # 在边上移动中的智能体跳过
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
        """
        单智能体 GBS 决策。

        Returns:
            邻居索引（0 ~ len(neighbors)-1），无邻居时返回 None
        """
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        # ε-贪心探索
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            return random.randrange(len(neighbors))

        # 邻居空闲度：优先取 obs 中预计算值，否则从 global_state 提取
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        # 全局平均空闲度
        node_idleness_dict = global_state.get("node_idleness", {})
        global_mean = float(np.mean(list(node_idleness_dict.values()))) if node_idleness_dict else 0.0

        # GBS 效用：per-neighbor 噪声 ∈ [0.95, 1.05]
        utilities = np.array([
            (0.95 + 0.1 * random.random()) * (
                self.G1 * float(neighbor_idleness[i])
                + self.G2 * (float(neighbor_idleness[i]) - global_mean)
            )
            for i in range(len(neighbors))
        ], dtype=np.float64)

        # 数值稳定 softmax 采样
        utilities -= np.max(utilities)
        exp_u = np.exp(utilities)
        probs = exp_u / np.sum(exp_u)
        return int(np.random.choice(len(neighbors), p=probs))
