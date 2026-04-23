# -*- coding: utf-8 -*-
"""
BAPS (Bayesian Ant Patrolling Strategy) 启发式策略 - 多智能体版本

分数越小越好（与旧 BAPSAgent 保持一致）：
  score = w_idl * nidle + w_pher * npher + w_dist * ndist + w_conflict * nconf
  - nidle    = 1 - minmax(idleness)   # 空闲越大 → 值越小（更优先）
  - npher    = 1 - minmax(pheromone)  # 信息素越高 → 值越小（更优先）
  - ndist    = minmax(distance)       # 距离越短 → 值越小（更优先）
  - nconf    = 0/1 (其他 agent 已将 target 指向该节点)

信息素由环境可选提供（global_state['baps_pheromone']），若不存在则全部为 0（优雅降级）。
冲突检测使用基类 _neighbor_conflicts，通过 global_state['agents_target_node'] 读取。
距离计算使用基类 _neighbor_distances，支持 'edge' / 'sp' 两种模式。
"""
import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class BAPSPolicy(HeuriticBasePolicy):
    """
    BAPS 启发式策略（多智能体版本）

    设计：
    - compute_actions: 并发处理所有 READY 智能体（非顺序）
    - _compute_action: 单智能体四因子打分 + argmin/softmax 选择
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # 距离读取方式
        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "edge"))).lower()

        # 四项权重（与旧 BAPSAgent 默认值一致）
        self.w_idl: float = float(config.get("w_idl", ap.get("w_idl", 0.6)))
        self.w_pher: float = float(config.get("w_pher", ap.get("w_pher", 0.2)))
        self.w_dist: float = float(config.get("w_dist", ap.get("w_dist", 0.2)))
        self.w_conflict: float = float(config.get("w_conflict", ap.get("w_conflict", 0.4)))

        # 探索与采样
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.0)))
        # softmax 温度：>0 时按 softmax(-score/tau) 采样；<=0 时取 argmin
        self.softmax_tau: float = float(config.get("softmax_tau", ap.get("softmax_tau", 0.0)))

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
            global_state: 含 'graph', 'node_idleness', 'agents_target_node',
                          'agents_on_edge', 'baps_pheromone'（可选）等

        Returns:
            actions: {agent_str: action_idx}
        """
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
        """
        单智能体 BAPS 决策：四因子归一化打分，取最小分数的邻居。
        """
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        # ε-贪心探索
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            return random.randrange(len(neighbors))

        current_node = obs.get("current_node")

        # 邻居空闲度
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        # 信息素（环境可选提供，默认全 0）
        baps_pheromone: Dict = global_state.get("baps_pheromone", {})
        pheromone = np.asarray(
            [float(baps_pheromone.get(nb, 0.0)) for nb in neighbors], dtype=np.float32
        )

        # 距离与冲突（调用基类工具方法）
        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph)
        conflicts = self._neighbor_conflicts(agent_id, neighbors, global_state)

        # 归一化（更优 → 值越小）
        nidle = self._norm_inverted(neighbor_idleness)   # 高空闲 → 小
        npher = self._norm_inverted(pheromone)            # 高信息素 → 小
        ndist = self._norm_minmax(distances)              # 短距离 → 小
        nconf = conflicts                                 # 0 或 1

        score = (
            self.w_idl * nidle
            + self.w_pher * npher
            + self.w_dist * ndist
            + self.w_conflict * nconf
        )

        # 选择策略：评估/tau<=0 → argmin；否则 softmax(-score/tau) 采样
        if self.softmax_tau <= 0.0:
            return int(np.argmin(score))
        else:
            logits = -score / max(self.softmax_tau, 1e-6)
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.clip(np.sum(probs), 1e-8, None)
            return int(np.random.choice(len(neighbors), p=probs))
