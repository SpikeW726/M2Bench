from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

_EPS = 1e-9


class HeuriticBasePolicy(ABC):
    """
    Abstract base class for all heuristic policies.
    """

    def __init__(self, num_agents: int, config: Dict):
        self.num_agents = num_agents
        self.config = config
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

    # ------------------------------------------------------------------
    # 共享工具方法：归一化 + 距离 + 冲突（供子类直接继承，无需重复实现）
    # ------------------------------------------------------------------

    def _norm_minmax(self, arr: np.ndarray) -> np.ndarray:
        """Min-Max 归一化，全相等时返回全零"""
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if a_max - a_min < _EPS:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - a_min) / (a_max - a_min)).astype(np.float32)

    def _norm_inverted(self, arr: np.ndarray) -> np.ndarray:
        """反向 Min-Max 归一化（值越大 → 归一化后越小，用于"高空闲优先"）"""
        return 1.0 - self._norm_minmax(arr)

    def _neighbor_distances(
        self,
        current: int,
        neighbors: List[int],
        graph: Any,
        distance_mode: str = None,
    ) -> np.ndarray:
        """
        计算当前节点到每个邻居的距离。

        distance_mode 优先级：参数 > self.distance_mode > 'edge'
          - 'edge': 直接读取相邻边长
          - 'sp'  : 读取最短路长度（graph 需提供 shortest_path_length 方法）
        """
        if distance_mode is None:
            distance_mode = getattr(self, "distance_mode", "edge")

        if graph is None:
            return np.ones(len(neighbors), dtype=np.float32)

        if distance_mode == "sp" and hasattr(graph, "shortest_path_length"):
            out = []
            for nb in neighbors:
                try:
                    d = float(graph.shortest_path_length(current, nb))
                except Exception:
                    d = float(getattr(graph, "get_edge_length", lambda u, v: 1.0)(current, nb) or 1.0)
                out.append(max(d, 1.0))
            return np.asarray(out, dtype=np.float32)

        get_len = getattr(graph, "get_edge_length", None)
        out = []
        for nb in neighbors:
            d = float(get_len(current, nb) or 1.0) if callable(get_len) else 1.0
            out.append(max(d, 1.0))
        return np.asarray(out, dtype=np.float32)

    def _neighbor_conflicts(
        self,
        agent_id: int,
        neighbors: List[int],
        global_state: Dict[str, Any],
    ) -> np.ndarray:
        """
        即时目标冲突检测：其他 agent 已将 target 指向该邻居 → 1.0，否则 0.0。
        HPCC 等算法可在子类中覆盖以实现更复杂的 ETA 冲突逻辑。
        """
        agents_target_node = global_state.get("agents_target_node", {})
        occupied: set = set()
        for idx, tgt in agents_target_node.items():
            if idx == agent_id:
                continue
            if isinstance(tgt, (int, np.integer)) and tgt >= 0:
                occupied.add(int(tgt))
        return np.asarray(
            [1.0 if nb in occupied else 0.0 for nb in neighbors], dtype=np.float32
        )
    
    @abstractmethod
    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为所有智能体计算动作,默认是同步决策框架,无需决策的智能体返回None
        
        Args:
            obs_dict: 每个智能体的局部观测
                {agent_id: {
                    'current_node': int,           # 当前节点
                    'neighbors': List[int],        # 邻居节点列表
                    'neighbor_idleness': List[float],  # 邻居节点空闲度
                    'on_edge': bool,               # 是否在边上移动中
                    ...
                }}
            
            global_state: 全局状态信息（启发式算法通常需要）
                {
                    'graph': Graph,                        # 图结构对象
                    'agent_positions': Dict[int, int],     # 所有智能体位置 {agent_idx: node_id}
                    'agents_target_node': Dict[int, int],  # 所有智能体目标 {agent_idx: node_id}
                    'node_idleness': Dict[int, float],     # 节点空闲度 {node_id: idleness}
                    'agents_on_edge': Dict[int, bool],     # 智能体是否在边上
                    'agent_speeds': List[float],           # 智能体速度列表 (来自物理世界)
                    ...
                }
            

        Returns:
            actions: {agent_id: action} 所有需要决策的智能体的动作
        """
        pass
    
    @abstractmethod
    def _compute_action(
        self,
        agent_idx: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Optional[int]:
        """
        为单个智能体计算动作
        
        Args:
            agent_idx: 智能体索引
            obs: 该智能体的局部观测
            global_state: 全局状态
            evaluation_mode: 评估模式
        
        Returns:
            action: 动作（邻居索引），如果不需要决策返回 None
        """
        pass
    
    def reset(self):
        """重置策略内部状态（如果有的话）"""
        pass
