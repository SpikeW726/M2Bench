# -*- coding: utf-8 -*-
"""
Conscientious Reactive 启发式策略 - 多智能体版本

选邻居时只依赖邻居节点的空闲度：
  - 评估模式：argmax(idleness)
  - 训练模式：乘性噪声 + 稳定 softmax 采样（与旧实现保持一致）

"Reactive"：只看当前观测，无记忆、无协调。
"""
import random
from typing import Dict, Optional, Any

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class ConscientiousReactivePolicy(HeuriticBasePolicy):
    """
    Conscientious Reactive 启发式策略（多智能体版本）。

    设计：
    - compute_actions: 并发处理所有 READY agents
    - _compute_action: argmax 或 softmax over neighbor idleness
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.exploration_rate: float = float(config.get("exploration_rate", ap.get("exploration_rate", 0.0)))
        # 乘性噪声强度（训练期打破平局；评估时建议 0）
        self.idleness_noise: float = float(config.get("idleness_noise", ap.get("idleness_noise", 0.0)))
        # softmax 温度（<=1e-8 退化为 argmax；评估时建议低温）
        self.temperature: float = float(config.get("temperature", ap.get("temperature", 1.0)))

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

    @staticmethod
    def _stable_softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        """数值稳定的 softmax，含温度参数。"""
        T = max(float(temperature), 1e-8)
        z = np.clip(x / T - np.max(x / T), -60.0, 60.0)
        e = np.exp(z)
        s = e.sum()
        if not np.isfinite(s) or s <= 0.0:
            probs = np.zeros_like(x, dtype=np.float64)
            probs[int(np.argmax(x))] = 1.0
            return probs
        return e / s

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Optional[int]:
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        # 邻居空闲度
        neighbor_idleness = obs.get("neighbor_idleness")
        if neighbor_idleness is None:
            node_idleness = global_state.get("node_idleness", {})
            neighbor_idleness = [node_idleness.get(nb, 0.0) for nb in neighbors]
        idl = np.asarray(neighbor_idleness[: len(neighbors)], dtype=np.float64)

        # 低温或评估模式：直接 argmax
        if self.temperature <= 1e-8:
            return int(np.argmax(idl))

        # ε-贪心探索
        if self.exploration_rate > 0.0 and random.random() < self.exploration_rate:
            return random.randrange(len(neighbors))

        # 乘性噪声（训练期轻微打破平局）
        if self.idleness_noise > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.idleness_noise, size=len(neighbors))
            idl = idl * (1.0 + noise)

        probs = self._stable_softmax(idl, self.temperature)
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0.0:
            return int(np.argmax(idl))
        return int(np.random.choice(len(neighbors), p=probs))
