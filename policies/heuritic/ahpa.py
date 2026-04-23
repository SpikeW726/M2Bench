# -*- coding: utf-8 -*-
"""
AHPA (Adaptive Heuristic Patrolling Agent) 启发式策略 - 多智能体版本

原始设计：分布式 + 本地 belief + 概率通信同步。
新框架降级：PatrolWorld 不提供通信机制，直接使用 global_state['node_idleness']
作为各 agent 的"本地 belief"（等效于完美通信），保留完整的多因子打分结构。

打分（与旧 AHPAAgent 严格一致）：
  score = w_idl*nidle + w_dist*ndist + w_conflict*nconf + w_back*nback
  （越小越优先）
  - nidle: 1 - minmax(idleness)    — 高空闲优先
  - ndist: minmax(distance)        — 近邻优先
  - nconf: 0/1 目标冲突
  - nback: 0/1 上一步节点（防折返）
"""
import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class AHPAPolicy(HeuriticBasePolicy):
    """
    AHPA 启发式策略（多智能体版本，通信降级为完美信息）。

    设计：
    - compute_actions: 并发处理所有 READY agents
    - _compute_action: 四因子打分 + argmin 或 softmax 采样
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # 打分权重
        self.w_idl: float = float(config.get("w_idl", ap.get("w_idl", 0.7)))
        self.w_dist: float = float(config.get("w_dist", ap.get("w_dist", 0.2)))
        self.w_conflict: float = float(config.get("w_conflict", ap.get("w_conflict", 0.3)))
        self.w_back: float = float(config.get("w_back", ap.get("w_back", 0.2)))

        # 采样与探索
        self.softmax_tau: float = float(config.get("softmax_tau", ap.get("softmax_tau", 0.0)))
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.0)))

        # 距离模式
        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "edge"))).lower()

        # per-agent 上一步离开的节点（防折返）
        self._last_vertex: Dict[int, Optional[int]] = {}

    def reset(self):
        """清空 per-agent 上一步节点记录。"""
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

        # ε-贪心探索
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            self._last_vertex[agent_id] = current_node
            return random.randrange(len(neighbors))

        # 邻居空闲度（降级：直接使用全局真实值代替本地 belief）
        node_idleness = global_state.get("node_idleness", {})
        idl_vals = np.asarray(
            [float(node_idleness.get(nb, 0.0)) for nb in neighbors], dtype=np.float32
        )

        # 距离（调用基类工具方法）
        graph = global_state.get("graph")
        distances = self._neighbor_distances(current_node, neighbors, graph)

        # 冲突（调用基类工具方法，需 agents_target_node）
        conflicts = self._neighbor_conflicts(agent_id, neighbors, global_state)

        # 折返惩罚：上一步离开的节点 → 1.0
        last_v = self._last_vertex.get(agent_id)
        back = np.asarray(
            [1.0 if last_v is not None and nb == last_v else 0.0 for nb in neighbors],
            dtype=np.float32,
        )

        # 归一化（更优 → 更小）
        nidle = self._norm_inverted(idl_vals)
        ndist = self._norm_minmax(distances)

        score = (
            self.w_idl * nidle
            + self.w_dist * ndist
            + self.w_conflict * conflicts
            + self.w_back * back
        )

        # 记录当前节点供下次决策使用
        self._last_vertex[agent_id] = current_node

        # 选择策略
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
