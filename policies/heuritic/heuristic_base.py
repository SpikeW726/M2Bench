from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

_EPS = 1e-9

class HeuriticBasePolicy(ABC):
    """Common interface and candidate utilities for heuristic policies."""

    def __init__(self, num_agents: int, config: Dict):
        self.num_agents = num_agents
        self.config = config
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

    def _norm_minmax(self, arr: np.ndarray) -> np.ndarray:
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if a_max - a_min < _EPS:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - a_min) / (a_max - a_min)).astype(np.float32)

    def _norm_inverted(self, arr: np.ndarray) -> np.ndarray:
        return 1.0 - self._norm_minmax(arr)

    def _neighbor_distances(
        self,
        current: int,
        neighbors: List[int],
        graph: Any,
        distance_mode: str = None,
    ) -> np.ndarray:
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
        """Return neighbor-index actions for agents that can currently decide.

        Implementations may decide concurrently or sequentially. Returned keys
        use ``agent_<index>`` and values index the current node's neighbor list.
        Agents that are moving need not appear in the result.
        """

        pass

    @abstractmethod
    def _compute_action(
        self,
        agent_idx: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Optional[int]:
        """Select one neighbor index, or ``None`` when no action is available."""

        pass

    def reset(self):
        pass
