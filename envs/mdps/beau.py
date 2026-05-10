"""BEAU — 论文《Balancing Efficiency and Unpredictability》图巡逻环境。

设计原则
--------
BEAU 只实现 MAT pipeline 真正需要的接口：
- 物理仿真：继承 EventDrivenEnv（PatrolWorld + tick_to_next_event）
- 奖励：R = -IGI(t)（负平均节点 idleness，全 agent 共享）
- 截止：按 episode_len（时间或步数）
- MAT 图接口：state_mat / get_adj / get_current_node_indices / graph_idx_to_action

MATOnPolicyCollector 完全不读 per-agent obs，只调用上面四个图接口，
因此 observation_space / _build_obs 是 PettingZoo 合约要求的最小占位实现。

关于"顺序决策"
-------------
agent 的先后顺序在 MATDecoder 内部自回归循环中体现，env 侧仍为标准
PettingZoo ParallelEnv 联合动作 dict，无需任何修改。

与官方 MARL_to_solve_patrol（栅格+可见性）相比的主要差异
----------------------------------------------------
- 地图：本仓库为 JSON 拓扑 + NetworkX 生成 2D 归一化坐标，非四邻栅格；节点特征为 [x,y,归一化 idleness]。
- 奖励：原文 env.py 为 shared_reward = -avg_since_visit/100 + 1；BEAU 用连续时间 IGI（TickResult.pre_arrival_igi）
  的负值再乘 reward_scale，**无 +1 偏置**（仅尺度与可训性不同，最优点均为压低 idleness）。
- 时间：PatrolWorld 为事件驱动真实旅行时间；原文为同步栅格时间步。决策步疏密、change_reward 累积与 asy_ppo
  中按决策步存 buffer 的设计一致，但数值轨迹不可逐格对齐。
- 可见性/局部观测：原文含视野与 heatmap；BEAU 不实现，GAT 输入为全图结构上的节点特征（与公开仓库图任务常见设定一致）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from gymnasium.spaces import Box, Discrete

from envs.mdps.base_envs import EventDrivenEnv
from envs.mdps.patrol_core import AgentState, TickResult

# 占位 obs 维度（1 维，内容无意义，MAT 从不读它）
_STUB_OBS_DIM = 1


class BEAU(EventDrivenEnv):
    """图巡逻 BEAU 环境（MAT/图注意力 pipeline 专用）。

    继承 EventDrivenEnv，step 逻辑与 base_envs.EventDrivenEnv 完全相同。
    observation_space / _build_obs 为最小占位，MAT 不使用它们。
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        self.episode_len: int = config["episode_len"]
        self.reward_scale: float = float(kwargs.get("reward_scale", 1.0))
        self.truncate_by_time: bool = kwargs.get("truncate_by_time", True)

        # ---- 邻居缓存（graph_idx_to_action 需要）----
        self._node_neighbor_count: Dict[int, int] = {
            n: len(self.world.graph.get_neighbors(n))
            for n in self.world.graph.nodes
        }

        # ---- 图布局：MAT 图接口所需 ----
        self._sorted_nodes: List = sorted(self.world.graph.nodes)
        self._node_to_idx: Dict[int, int] = {n: i for i, n in enumerate(self._sorted_nodes)}
        self.graph_size: int = len(self._sorted_nodes)

        graph_path = Path(config["graph_path"])
        coords_path = graph_path.with_name(graph_path.stem + "_coords.json")
        if not coords_path.exists():
            self._generate_coords(graph_path, coords_path)
        with open(coords_path) as f:
            raw_coords = json.load(f)
        self._node_coords = np.array(
            [raw_coords[str(n)] for n in self._sorted_nodes], dtype=np.float32
        )  # (G, 2)
        self._adj: np.ndarray = self._build_adj()

    # ------------------------------------------------------------------
    #  PettingZoo 最小合约（占位，MAT 不读）
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Box:
        """占位：返回 1 维 [0,1] Box。MAT 不使用 per-agent obs。"""
        return Box(low=0.0, high=1.0, shape=(_STUB_OBS_DIM,), dtype=np.float32)

    def action_space(self, agent: str) -> Discrete:
        """有效：邻居 index + wait + no-op，与 MASUPEnv 一致。"""
        return Discrete(self.world.max_neighbors + 2)

    def state(self) -> np.ndarray:
        """占位全局向量（_infer_dims 需要非空返回）。"""
        return np.zeros(1, dtype=np.float32)

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        """占位：每个 agent 返回 1 维零向量。"""
        stub = np.zeros(_STUB_OBS_DIM, dtype=np.float32)
        return {a: stub for a in self.agents}

    # ------------------------------------------------------------------
    #  核心逻辑
    # ------------------------------------------------------------------

    def _compute_truncations(self) -> Dict[str, bool]:
        if self.truncate_by_time:
            done = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            done = self.world.step_count >= self.episode_len
        return {a: done for a in self.agents}

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        """R = -IGI(t) * reward_scale，所有 agent 共享（cooperative）。"""
        r = -float(result.pre_arrival_igi) * self.reward_scale
        return {a: r for a in self.agents}

    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        """active_mask 供 MATOnPolicyCollector 判断哪些 agent 需决策。"""
        return {
            agent_str: {
                "action_mask": self._get_action_mask(agent_str),
                "active_mask": 1 if self.world.is_ready(int(agent_str.split("_")[1])) else 0,
            }
            for agent_str in self.agents
        }

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        """action_idx → 目标节点（0=wait 由 EventDrivenEnv 处理，1..N=第 k 邻居）。"""
        current_pos = self.world.get_position(agent_id)
        neighbors = self.world.graph.get_neighbors(current_pos)
        neighbor_idx = action_idx - 1
        if 0 <= neighbor_idx < len(neighbors):
            return neighbors[neighbor_idx]
        raise ValueError(
            f"无效动作: agent_id={agent_id}, action_idx={action_idx}, 邻居数={len(neighbors)}"
        )

    def _get_action_mask(self, agent_str: str) -> np.ndarray:
        agent_id = int(agent_str.split("_")[1])
        mask = np.zeros(self.world.max_neighbors + 2, dtype=bool)
        if not self.world.is_ready(agent_id):
            mask[-1] = True   # no-op
        else:
            n_nb = self._node_neighbor_count[self.world.get_position(agent_id)]
            mask[: n_nb + 1] = True   # 0=wait, 1..n_nb=neighbors
        return mask

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}

    # ------------------------------------------------------------------
    #  MAT 图接口
    # ------------------------------------------------------------------

    def state_mat(self) -> np.ndarray:
        """GATEncoder 输入：(N, G, 3)，dim-3 = [x, y, idleness_norm]。"""
        n_a = self.world.num_agents
        idle_vals = np.array(
            [float(self.world.node_idleness.get(n, 0.0)) for n in self._sorted_nodes],
            dtype=np.float32,
        )
        idle_norm = idle_vals / (idle_vals.max() + 1e-9)
        state = np.empty((n_a, self.graph_size, 3), dtype=np.float32)
        for s in range(n_a):
            state[s, :, 0] = self._node_coords[:, 0]
            state[s, :, 1] = self._node_coords[:, 1]
            state[s, :, 2] = idle_norm
        return state

    def get_adj(self) -> np.ndarray:
        """(G, G) float32 邻接矩阵。"""
        return self._adj

    def get_current_node_indices(self) -> np.ndarray:
        """(N,) int32：各 agent 当前/上次所在节点的 graph index。"""
        out = np.empty(self.world.num_agents, dtype=np.int32)
        for i in range(self.world.num_agents):
            st = self.world.agents[i]
            node = st.last_position if st.state == AgentState.ON_EDGE else st.position
            out[i] = self._node_to_idx.get(node, 0)
        return out

    def graph_idx_to_action(self, graph_idx_list: list) -> dict:
        """graph index 列表 → {agent_k: env_action_idx} dict（供 collector 调用）。"""
        actions = {}
        for k, gidx in enumerate(graph_idx_list):
            st = self.world.agents[k]
            if st.state == AgentState.ON_EDGE:
                actions[f"agent_{k}"] = self.world.max_neighbors + 1  # no-op
                continue
            target = self._sorted_nodes[int(gidx)]
            neighbors = self.world.graph.get_neighbors(st.position)
            actions[f"agent_{k}"] = neighbors.index(target) + 1 if target in neighbors else 0
        return actions

    # ------------------------------------------------------------------
    #  内部辅助
    # ------------------------------------------------------------------

    def _build_adj(self) -> np.ndarray:
        g = self.graph_size
        adj = np.zeros((g, g), dtype=np.float32)
        for n in self._sorted_nodes:
            i = self._node_to_idx[n]
            for nb, _ in self.world.graph.adj_list[n]:
                j = self._node_to_idx.get(nb, -1)
                if j >= 0:
                    adj[i, j] = adj[j, i] = 1.0
        return adj

    @staticmethod
    def _generate_coords(graph_path: Path, coords_path: Path) -> None:
        import networkx as nx

        with open(graph_path) as f:
            data = json.load(f)
        g = nx.Graph()
        for n in data["nodes"]:
            g.add_node(n)
        for e in data["edges"]:
            g.add_edge(e["from"], e["to"], weight=float(e["weight"]))
        pos = nx.kamada_kawai_layout(g, weight="weight")
        xs = np.array([pos[n][0] for n in g.nodes()])
        ys = np.array([pos[n][1] for n in g.nodes()])
        x_rng = (xs.max() - xs.min()) or 1.0
        y_rng = (ys.max() - ys.min()) or 1.0
        coords = {
            str(n): [
                float((pos[n][0] - xs.min()) / x_rng),
                float((pos[n][1] - ys.min()) / y_rng),
            ]
            for n in g.nodes()
        }
        with open(coords_path, "w") as f:
            json.dump(coords, f, indent=2)
        print(f"[BEAU] 已生成坐标文件: {coords_path}")
