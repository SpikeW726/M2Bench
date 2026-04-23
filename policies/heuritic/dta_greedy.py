# -*- coding: utf-8 -*-
"""
DTAGreedy (Distributed Tabu-list Adaptive Greedy) 启发式策略 - 多智能体版本

与旧 MultiAgentPatrolling 中 DTAGreedyAgent 对齐：
  base = 1.5 * idl + tabu_penalty + 0.2 * (idl - global_mean)
  若其他 agent 占据该节点：base *= 0.3
  随机扰动：仅当 stochastic=True 且 score_jitter>0 时，对每条候选分数乘
  Uniform(1-jitter, 1+jitter)（对应旧代码中 evaluation_mode=False 时的训练期扰动；
  评估/可复现运行请设 stochastic=False。）

per-agent 独立 tabu：每次决策后将 current_node 加入该 agent 的 tabu。
"""
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from policies.heuritic.heuristic_base import HeuriticBasePolicy


class DTAGreedyPolicy(HeuriticBasePolicy):
    """
    DTAGreedy 启发式策略（多智能体版本）。

    设计：
    - compute_actions: 并发处理所有 READY agents
    - _compute_base_scores: 无随机性的基础分（子类 SSI 可再叠一层噪声）
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        self.tabu_length: int = int(config.get("tabu_length", ap.get("tabu_length", 5)))
        self.score_jitter: float = float(config.get("score_jitter", ap.get("score_jitter", 0.1)))
        # 对应旧版 evaluation_mode=False 时才加 Greedy 扰动；为 False 时不加（可复现评估）
        self.stochastic: bool = bool(config.get("stochastic", ap.get("stochastic", False)))

        self._tabu_lists: Dict[int, List[int]] = {}

    def reset(self) -> None:
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

    def _compute_base_scores(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Optional[Tuple[np.ndarray, int, List[int]]]:
        """
        计算各邻居的基础分（无 Greedy/SSI 随机层）。

        Returns:
            (scores, current_node, tabu_list) 或 None
        """
        neighbors = obs.get("neighbors", [])
        if not neighbors:
            return None

        current_node = int(obs.get("current_node", -1))
        node_idleness = global_state.get("node_idleness", {})

        global_mean = float(np.mean(list(node_idleness.values()))) if node_idleness else 0.0

        agent_positions = global_state.get("agent_positions", {})
        other_positions = {
            int(pos) for idx, pos in agent_positions.items()
            if idx != agent_id and isinstance(pos, (int, np.integer)) and pos >= 0
        }

        tabu = self._tabu_lists.setdefault(agent_id, [])

        scores = np.zeros(len(neighbors), dtype=np.float64)
        for i, node in enumerate(neighbors):
            idl = float(node_idleness.get(node, 0.0))
            tabu_penalty = -1.0 if node in tabu else 0.0
            bonus = 0.2 * (idl - global_mean)
            base = 1.5 * idl + tabu_penalty + bonus
            if node in other_positions:
                base *= 0.3
            scores[i] = base

        return scores, current_node, tabu

    def _apply_dta_greedy_score_noise(self, scores: np.ndarray) -> None:
        """与旧版一致：仅 stochastic 且 score_jitter>0 时逐候选乘 (1±jitter) 均匀扰动。原地修改。"""
        if not self.stochastic or self.score_jitter <= 0.0:
            return
        lo, hi = 1.0 - self.score_jitter, 1.0 + self.score_jitter
        for i in range(len(scores)):
            scores[i] *= random.uniform(lo, hi)

    def _update_tabu_after_decision(self, current_node: int, tabu: List[int]) -> None:
        if current_node is not None and current_node >= 0:
            tabu.append(int(current_node))
            if len(tabu) > self.tabu_length:
                tabu.pop(0)

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

        self._apply_dta_greedy_score_noise(scores)
        best_idx = int(np.argmax(scores))
        self._update_tabu_after_decision(current_node, tabu)
        return best_idx
