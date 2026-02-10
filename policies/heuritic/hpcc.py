# -*- coding: utf-8 -*-
# agent/hpcc_agent.py
from typing import Dict, List, Optional, Any
import numpy as np
import math
from policies.heuritic.heuristic_base import HeuriticBasePolicy

class HPCCPolicy(HeuriticBasePolicy):
    """
    HPCC 启发式策略（多智能体版本）
    
    设计：
    - compute_actions: 遍历所有需要决策的智能体，调用 _compute_action
    - _compute_action: 单个智能体的决策逻辑，使用全局信息进行冲突协调
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)
        
        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # 距离估计方式
        self.distance_mode: str = str(config.get("distance_mode", ap.get("distance_mode", "sp"))).lower()

        # 权重
        self.w_idl: float = float(config.get("w_idl", ap.get("w_idl", 0.6)))
        self.w_dist: float = float(config.get("w_dist", ap.get("w_dist", 0.25)))
        self.w_conflict: float = float(config.get("w_conflict", ap.get("w_conflict", 0.5)))


        # 认知协调参数（ETA冲突）
        self.conflict_eta_margin: float = float(ap.get("conflict_eta_margin", 0.0))  # 允许我方稍慢/持平的余量
        self.eta_clip_min: float = float(ap.get("eta_clip_min", 1.0))  # ETA 最小值，避免 0
        self._eps = 1e-9


    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        为所有需要决策的智能体计算动作
        
        Args:
            obs_dict: {agent_str: obs} 每个智能体的局部观测
                obs 应包含:
                - 'current_node': int
                - 'neighbors': List[int]
                - 'neighbor_idleness': List[float] (可选，如果没有则从 global_state 获取)
                - 'on_edge': bool (可选，用于判断是否需要决策)
            
            global_state: 全局状态
                - 'graph': Graph 对象
                - 'agent_positions': Dict[int, int]
                - 'agents_target_node': Dict[int, int]
                - 'node_idleness': Dict[int, float]
                - 'agents_on_edge': Dict[int, bool]
        
        Returns:
            actions: {agent_str: action_idx} 所有需要决策的智能体的动作
        """
        actions = {}
        
        for agent_str, obs in obs_dict.items():
            # 提取智能体索引
            agent_id = int(agent_str.split("_")[1]) if isinstance(agent_str, str) else int(agent_str)
            
            # 检查是否需要决策（在边上移动中的智能体不需要决策）
            on_edge = obs.get('on_edge', False)
            if global_state.get('agents_on_edge'):
                on_edge = global_state['agents_on_edge'].get(agent_id, on_edge)
            
            if on_edge:
                # 在边上移动中，不需要决策（返回 no-op 或跳过）
                continue
            
            # 计算动作
            action = self._compute_action(agent_id, obs, global_state)
            
            if action is not None:
                actions[agent_str] = action
        
        return actions

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Optional[int]:
        """
        为单个智能体计算动作
        
        Args:
            agent_id: 智能体索引
            obs: 局部观测，包含 current_node, neighbors, neighbor_idleness
            global_state: 全局状态
            evaluation_mode: 评估模式
        
        Returns:
            action: 邻居索引0 ~ len(neighbors)-1 或 None
        """
        # 提取局部观测
        current_node = obs.get('current_node')
        neighbors = obs.get('neighbors', [])
        
        if not neighbors:
            return None
        
        # 获取邻居空闲度
        neighbor_idleness = obs.get('neighbor_idleness')
        if neighbor_idleness is None:
            # 从 global_state 获取
            node_idleness = global_state.get('node_idleness', {})
            neighbor_idleness = [node_idleness.get(n, 0.0) for n in neighbors]
        
        neighbor_idleness = np.asarray(neighbor_idleness, dtype=np.float32)
        
        # 计算距离
        graph = global_state.get('graph')
        distances = self._neighbor_distances(current_node, neighbors, graph)
        
        # 计算冲突分数
        nconf = self._neighbor_conflicts(
            agent_id, current_node, neighbors, distances, global_state
        )
        
        # 归一化
        nidle = self._norm_inverted(neighbor_idleness)  # 大空闲 → 小值（更好）
        ndist = self._norm_minmax(distances)            # 短距离 → 小值（更好）
        
        # 线性组合得分
        score = (self.w_idl * nidle +
                 self.w_dist * ndist +
                 self.w_conflict * nconf)

        logits = -score / max(self.softmax_tau, 1e-6)
        logits -= np.max(logits)  # 数值稳定
        probs = np.exp(logits)
        probs = probs / np.clip(np.sum(probs), 1e-8, None)
        return int(np.random.choice(len(neighbors), p=probs))

    
    def _norm_minmax(self, arr: np.ndarray) -> np.ndarray:
        """Min-Max 归一化"""
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if a_max - a_min < self._eps:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - a_min) / (a_max - a_min)

    def _norm_inverted(self, arr: np.ndarray) -> np.ndarray:
        """反向 Min-Max 归一化（值大→归一化后小）"""
        return 1.0 - self._norm_minmax(arr)

    def _neighbor_distances(
        self, 
        current: int, 
        neighbors: List[int], 
        graph: Any
    ) -> np.ndarray:
        """
        计算从当前节点到每个邻居的距离
        
        Args:
            current: 当前节点
            neighbors: 邻居节点列表
            graph: 图对象
        
        Returns:
            distances: 到每个邻居的距离数组
        """
        if graph is None:
            return np.ones(len(neighbors), dtype=np.float32)

        # 优先使用最短路
        if self.distance_mode == "sp" and hasattr(graph, "shortest_path_length"):
            out = []
            for nb in neighbors:
                try:
                    d = float(graph.shortest_path_length(current, nb))
                except Exception:
                    d = float(getattr(graph, "get_edge_length", lambda u, v: 1.0)(current, nb) or 1.0)
                out.append(max(d, 1.0))
            return np.asarray(out, dtype=np.float32)

        # 否则使用边长
        get_len = getattr(graph, "get_edge_length", None)
        out = []
        for nb in neighbors:
            if callable(get_len):
                d = float(get_len(current, nb) or 1.0)
            else:
                d = 1.0
            out.append(max(d, 1.0))
        return np.asarray(out, dtype=np.float32)

    def _neighbor_conflicts(
        self, 
        agent_id: int, 
        current: int, 
        neighbors: List[int], 
        distances: np.ndarray,
        global_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        计算每个候选邻居的冲突/拥挤度分数
        
        冲突项包括：
          - 即时冲突：有人已将 target 指向该邻居 → +1
          - ETA 冲突：他人更快/近似时间到达该邻居 → +0.5
        
        Args:
            agent_id: 当前智能体索引
            current: 当前节点
            neighbors: 邻居节点列表
            distances: 到每个邻居的距离
            global_state: 全局状态
        
        Returns:
            conflicts: 每个邻居的冲突分数数组（越大越差）
        """
        graph = global_state.get('graph')
        agents_target_node = global_state.get('agents_target_node', {})
        agent_positions = global_state.get('agent_positions', {})
        
        # 其他机器人的"已选目标"
        occupied = set()
        for idx, tgt in agents_target_node.items():
            if idx == agent_id:
                continue
            if isinstance(tgt, (int, np.integer)) and tgt >= 0:
                occupied.add(int(tgt))

        # 其他机器人的当前位置
        others_pos = []
        for idx, pos in agent_positions.items():
            if idx == agent_id:
                continue
            if isinstance(pos, (int, np.integer)) and pos >= 0:
                others_pos.append(int(pos))

        sp = getattr(graph, "shortest_path_length", None) if graph is not None else None

        res = []
        for j, nb in enumerate(neighbors):
            score = 0.0

            # 1) 即时冲突：别人已经把目标对准 nb
            if nb in occupied:
                score += 1.0

            # 2) ETA 冲突：别人可能更快或差不多时间到达 nb
            if others_pos and callable(sp):
                eta_self = max(float(distances[j]), self.eta_clip_min)
                
                # 估计他人最小 ETA 到此 nb
                eta_other_min = math.inf
                for pos in others_pos:
                    try:
                        d = float(sp(pos, nb))
                    except Exception:
                        d = math.inf
                    if d < eta_other_min:
                        eta_other_min = d

                if eta_other_min < math.inf:
                    # 若他人 ETA <= 我方 ETA + 边际，则存在"拥挤/抢占风险"
                    if eta_other_min <= eta_self + self.conflict_eta_margin:
                        # 惩罚强度可与“领先程度”相关；这里给个平滑型
                        # delta = max(0, (eta_self + margin) - eta_other_min)
                        # score += 0.5 + sigmoid(delta)
                        score += 0.5  # 简洁实现：固定额外惩罚
            res.append(score)

            # NOTE: 若没有 others_pos 或没有 sp，可只保留“即时冲突”
        return np.asarray(res, dtype=np.float32)
