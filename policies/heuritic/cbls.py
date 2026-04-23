# -*- coding: utf-8 -*-
"""
CBLS (Cycle-Based with Tabu List Strategy) 启发式策略 - 多智能体版本

核心思想（与旧 CBLSAgent 保持一致）：
  - 维护 per-agent tabu 列表，防止短期内重复访问同一节点
  - 过滤 tabu 中的邻居节点后，对剩余候选的空闲度做 softmax 采样
  - 若所有邻居均在 tabu 中，则回退使用全量候选（兜底）
  - 每次决策后将当前节点（正在离开的节点）加入该 agent 的 tabu 列表

per-agent 状态（tabu 列表）存储在 self._tabu_lists 字典中，
reset() 调用时一并清空，保证 episode 间状态独立。
"""
import random
from typing import Dict, List, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class CBLSPolicy(HeuriticBasePolicy):
    """
    CBLS 启发式策略（多智能体版本）

    设计：
    - compute_actions: 并发处理所有 READY 智能体（非顺序）
    - _compute_action: 单智能体 tabu 过滤 + softmax over idleness 采样
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # 探索率（训练时随机选择的概率；评估时建议 0）
        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.1)))
        # 空闲度权重（对 softmax 分数的缩放；与旧 CBLSAgent 接口一致）
        self.idleness_weight: float = float(config.get("idleness_weight", ap.get("idleness_weight", 1.0)))
        # tabu 列表最大长度
        self.tabu_length: int = int(config.get("tabu_length", ap.get("tabu_length", 3)))

        # per-agent tabu 列表：{agent_idx: List[node_id]}
        self._tabu_lists: Dict[int, List[int]] = {}

    def reset(self):
        """清空所有 agent 的 tabu 列表，episode 开始时调用"""
        self._tabu_lists.clear()

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
        """数值稳定的 softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Optional[int]:
        """
        单智能体 CBLS 决策：tabu 过滤 + softmax 采样。
        """
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        current_node = obs.get("current_node")

        # 邻居空闲度
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        neighbor_idleness = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float32)

        # 获取本 agent 的 tabu 列表
        tabu = self._tabu_lists.setdefault(agent_id, [])

        # 过滤 tabu 节点：(邻居索引, 空闲度) 对
        filtered = [
            (i, float(neighbor_idleness[i]))
            for i, nb in enumerate(neighbors)
            if nb not in tabu
        ]

        # 若全在 tabu 中，回退使用全量候选（与旧 CBLSAgent 一致）
        if not filtered:
            filtered = [(i, float(neighbor_idleness[i])) for i in range(len(neighbors))]

        indices = [pair[0] for pair in filtered]
        values = np.array([pair[1] for pair in filtered], dtype=np.float32)
        probs = self._softmax(values * self.idleness_weight)

        # ε-贪心探索 or softmax 采样
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            selected_idx = random.choice(indices)
        else:
            selected_idx = int(np.random.choice(indices, p=probs))

        # 将当前节点（正在离开的节点）加入 tabu，超长时弹出队首
        if current_node is not None:
            tabu.append(int(current_node))
            if len(tabu) > self.tabu_length:
                tabu.pop(0)

        return selected_idx
