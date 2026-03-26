"""
MAGECEnv — Multi-Agent Graph Embedding-based Coordination 环境。

完整复现论文:
  Goeckner et al., "Graph Neural Network-based Multi-agent Reinforcement
  Learning for Resilient Distributed Coordination of Multi-Robot Systems"

设计要点：
  - 继承 EventDrivenEnv（事件驱动，tick_to_next_event）
  - action_dim = max_degree + 1（最后维为专用 no-op 槽）
  - ON_EDGE agent: action_mask=[0..0,1], active_mask=0，动作被 env.step 忽略
  - 节点 idleness 均以 phi 加权后再归一化
  - 虚拟节点动态边完整复现论文 Section IV-E.1
  - 双模截断: truncate_by_time (物理时间) 或 step_count
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium.spaces import Box, Discrete

from envs.mdps.base_envs import EventDrivenEnv
from envs.mdps.patrol_core import AgentState, TickResult


class MAGECEnv(EventDrivenEnv):
    """MAGEC 多智能体巡逻环境。

    观测向量遵守 gnn.py 文件头的通用协议：
      [node_feats | edge_src | edge_dst | edge_attr | edge_mask | global_feat | identity]

    节点特征 (node_feat_dim=3): [nodeType, phi_weighted_idleness_norm, degree]
    边特征 (edge_feat_dim=2):   [weight_norm, neighborIndex_float]
    全局特征 (global_feat_dim=2): [avg_phi_idleness_norm, worst_phi_idleness_norm]
    identity: one-hot agent index (agent-index 模式)
    """

    NODE_TYPE_PATROL = 0.0   # 巡逻节点
    NODE_TYPE_AGENT  = 1.0   # 虚拟 agent 节点
    NEIGHBOR_IDX_NONE = -1.0 # 不可作为 action target 的边

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        w = self.world
        graph = w.graph

        # ---- 图结构基础量 ----
        self.static_node_num   = w.num_nodes
        self.static_edge_num   = w.num_edges           # 有向边数
        self.agent_num         = w.num_agents
        self.max_neighbor_deg  = w.max_neighbors       # 图中节点最大出度
        self.max_edge_length   = w.max_edge_length     # 边权归一化分母

        # action space: neighbor indices 0..max_degree-1 + no-op(最后维)
        self.action_dim = self.max_neighbor_deg + 1

        # 静态节点排序（保证索引确定性）
        self.sorted_nodes = sorted(graph.nodes)
        self.node_to_idx  = {n: i for i, n in enumerate(self.sorted_nodes)}

        # 每个节点的邻居按 neighbor_id 升序排列 → 决定 neighborIndex 顺序
        self.sorted_adj: Dict[int, List[Tuple[int, float]]] = {
            v: sorted(graph.adj_list[v], key=lambda x: x[0])
            for v in self.sorted_nodes
        }

        # ---- 图结构：静态有向边列表 ----
        self.static_edges: List[Tuple[int, int, float]] = []
        for v in self.sorted_nodes:
            for nbr, w_edge in self.sorted_adj[v]:
                self.static_edges.append((v, nbr, w_edge))

        # ---- 观测维度 ----
        self.node_feat_dim   = 3   # [nodeType, idleness_norm, degree]
        self.edge_feat_dim   = 2   # [weight_norm, neighborIndex]
        self.global_feat_dim = 2   # [avg_phi_idle_norm, worst_phi_idle_norm]
        self.identity_dim    = self.agent_num  # one-hot

        # 总节点数 = 静态 + 虚拟 agent 节点
        self.total_nodes = self.static_node_num + self.agent_num

        # 虚拟节点最大出边数: READY 时 1(到当前节点) + max_deg(到邻居)
        self.max_agent_edges   = 1 + self.max_neighbor_deg
        self.total_max_edges   = self.static_edge_num + self.agent_num * self.max_agent_edges

        self.obs_size = (
            self.total_nodes * self.node_feat_dim       # 节点特征
            + self.total_max_edges * 2                   # edge_src + edge_dst
            + self.total_max_edges * self.edge_feat_dim  # edge_attr
            + self.total_max_edges                       # edge_mask
            + self.global_feat_dim                       # 全局特征
            + self.identity_dim                          # identity
        )

        # ---- 截断参数（episode_len 来自 EnvConfig，其余来自 custom_configs）----
        self.episode_len      = config["episode_len"]
        self.truncate_by_time = kwargs.get("truncate_by_time", True)

        # ---- 奖励参数（论文 Section IV-E.2，来自 custom_configs）----
        self.alpha = kwargs.get("alpha", 1.0)
        self.beta  = kwargs.get("beta", 0.5)
        self.eps   = 1e-6

        # ---- 邻接矩阵（for state()）----
        n = self.static_node_num
        self._adj_matrix = np.zeros((n, n), dtype=np.float32)
        for v in self.sorted_nodes:
            vi = self.node_to_idx[v]
            for nbr, _ in graph.adj_list[v]:
                ni = self.node_to_idx[nbr]
                self._adj_matrix[vi, ni] = 1.0
        self._adj_flat = self._adj_matrix.flatten()

    # ------------------------------------------------------------------
    #  PettingZoo 接口
    # ------------------------------------------------------------------

    def observation_space(self, agent: str):
        n_node = self.total_nodes * self.node_feat_dim
        n_edge_idx = self.total_max_edges * 2
        n_edge_attr = self.total_max_edges * self.edge_feat_dim
        n_mask = self.total_max_edges
        n_global = self.global_feat_dim
        n_id = self.identity_dim

        total = n_node + n_edge_idx + n_edge_attr + n_mask + n_global + n_id

        low  = np.full(total, -np.inf, dtype=np.float32)
        high = np.full(total,  np.inf, dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def action_space(self, agent: str):
        return Discrete(self.action_dim)

    def state(self) -> np.ndarray:
        """MAPPO Critic 全局状态: phi 加权归一化空闲度 + 邻接矩阵展开。"""
        weighted = self._phi_weighted_idleness()
        w_min, w_max = weighted.min(), weighted.max()
        idleness_norm = (weighted - w_min) / (w_max - w_min + self.eps)
        return np.concatenate([idleness_norm, self._adj_flat]).astype(np.float32)

    # ------------------------------------------------------------------
    #  内部辅助：phi 加权空闲度
    # ------------------------------------------------------------------

    def _phi_weighted_idleness(self) -> np.ndarray:
        """返回每个静态节点的 phi * idleness，按 sorted_nodes 顺序。"""
        phi  = self.world.graph.phi
        idle = self.world.node_idleness
        return np.array(
            [phi.get(v, 1) * idle.get(v, 0.0) for v in self.sorted_nodes],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    #  观测构建
    # ------------------------------------------------------------------

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        """构建所有 agent 的图结构观测。"""
        w    = self.world
        phi  = w.graph.phi
        idle = w.node_idleness

        # ---- 1. phi 加权 idleness → min-max 归一化 ----
        weighted = self._phi_weighted_idleness()
        w_min, w_max = weighted.min(), weighted.max()
        denom = w_max - w_min + self.eps

        # ---- 2. 节点特征（静态节点） ----
        node_feats: List[float] = []
        for i, v in enumerate(self.sorted_nodes):
            idle_norm = float((weighted[i] - w_min) / denom)
            degree    = float(len(w.graph.adj_list[v]))
            node_feats += [self.NODE_TYPE_PATROL, idle_norm, degree]

        # 虚拟 agent 节点（固定 idleness=0，degree 动态）
        for i in range(self.agent_num):
            ag = w.agents[i]
            if ag.state == AgentState.READY:
                virt_deg = 1 + len(self.sorted_adj.get(ag.position, []))
            else:  # ON_EDGE
                virt_deg = 2
            node_feats += [self.NODE_TYPE_AGENT, 0.0, float(virt_deg)]

        # ---- 3. 静态边 ----
        e_src_list:  List[float] = []
        e_dst_list:  List[float] = []
        e_attr_list: List[List[float]] = []

        for v, nbr, w_edge in self.static_edges:
            v_idx   = float(self.node_to_idx[v])
            nbr_idx = float(self.node_to_idx[nbr])
            # neighborIndex: v 的邻居列表中 nbr 的位置
            nbr_pos = self._get_neighbor_index(v, nbr)
            e_src_list.append(v_idx)
            e_dst_list.append(nbr_idx)
            e_attr_list.append([w_edge / max(self.max_edge_length, self.eps),
                                 float(nbr_pos)])

        # ---- 4. 虚拟 agent 节点的动态边 ----
        for i in range(self.agent_num):
            ag        = w.agents[i]
            virt_idx  = float(self.static_node_num + i)

            if ag.state in (AgentState.READY, AgentState.WAITING):
                cur_node = ag.position
                cur_idx  = float(self.node_to_idx[cur_node])

                # 边1: virtual → current_node, weight=0, neighborIndex=-1
                e_src_list.append(virt_idx)
                e_dst_list.append(cur_idx)
                e_attr_list.append([0.0, self.NEIGHBOR_IDX_NONE])

                # 边2+: virtual → sorted_adj[cur_node][k], neighborIndex=k
                for k, (nbr, w_edge) in enumerate(self.sorted_adj.get(cur_node, [])):
                    nbr_idx = float(self.node_to_idx[nbr])
                    e_src_list.append(virt_idx)
                    e_dst_list.append(nbr_idx)
                    e_attr_list.append([w_edge / max(self.max_edge_length, self.eps),
                                        float(k)])

                # 填充至 max_agent_edges（ON_EDGE 只用 2 条，READY 可能少于 max_deg+1）
                used = 1 + len(self.sorted_adj.get(cur_node, []))
                for _ in range(self.max_agent_edges - used):
                    e_src_list.append(virt_idx)
                    e_dst_list.append(virt_idx)
                    e_attr_list.append([0.0, self.NEIGHBOR_IDX_NONE])

            else:  # ON_EDGE
                src_node = ag.position
                dst_node = ag.target_node
                src_idx  = float(self.node_to_idx.get(src_node, 0))
                dst_idx  = float(self.node_to_idx.get(dst_node, 0))

                try:
                    full_dist = w.graph.get_edge_length(src_node, dst_node)
                except Exception:
                    full_dist = self.max_edge_length

                remaining_dist  = float(ag.action_remaining) * float(ag.speed)
                dist_traveled   = max(0.0, full_dist - remaining_dist)

                # 边1: virtual → source, weight=dist_traveled
                e_src_list.append(virt_idx)
                e_dst_list.append(src_idx)
                e_attr_list.append([dist_traveled / max(self.max_edge_length, self.eps),
                                    self.NEIGHBOR_IDX_NONE])

                # 边2: virtual → target, weight=remaining_dist
                e_src_list.append(virt_idx)
                e_dst_list.append(dst_idx)
                e_attr_list.append([remaining_dist / max(self.max_edge_length, self.eps),
                                    self.NEIGHBOR_IDX_NONE])

                # 填充至 max_agent_edges
                for _ in range(self.max_agent_edges - 2):
                    e_src_list.append(virt_idx)
                    e_dst_list.append(virt_idx)
                    e_attr_list.append([0.0, self.NEIGHBOR_IDX_NONE])

        # ---- 5. 边掩码（仅标记非填充边） ----
        # 有效边 = 静态边 + 每个 agent 的真实边（1+deg 或 2）
        real_edge_count = len(self.static_edges)
        for i in range(self.agent_num):
            ag = w.agents[i]
            if ag.state in (AgentState.READY, AgentState.WAITING):
                real_edge_count += 1 + len(self.sorted_adj.get(ag.position, []))
            else:
                real_edge_count += 2

        edge_mask = ([1.0] * real_edge_count
                     + [0.0] * (self.total_max_edges - real_edge_count))

        # ---- 6. 全局特征 ----
        avg_phi_idle   = float(weighted.mean())
        worst_phi_idle = float(weighted.max())
        norm_denom     = worst_phi_idle + self.eps
        global_feat    = [avg_phi_idle / norm_denom, 1.0]  # avg归一, worst=1

        # ---- 7. 组装各 agent 的观测 ----
        e_attr_flat = [x for pair in e_attr_list for x in pair]

        obs: Dict[str, np.ndarray] = {}
        for i in range(self.agent_num):
            identity = [0.0] * self.agent_num
            identity[i] = 1.0

            single_obs = np.array(
                node_feats
                + e_src_list
                + e_dst_list
                + e_attr_flat
                + edge_mask
                + global_feat
                + identity,
                dtype=np.float32,
            )
            obs[f"agent_{i}"] = single_obs

        return obs

    # ------------------------------------------------------------------
    #  Info（action_mask + active_mask）
    # ------------------------------------------------------------------

    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        infos: Dict[str, Dict] = {}
        for i in range(self.agent_num):
            ag     = self.world.agents[i]
            is_rdy = self.world.is_ready(i)

            if is_rdy:
                cur_node = ag.position
                deg      = len(self.sorted_adj.get(cur_node, []))
                mask     = ([1.0] * deg
                            + [0.0] * (self.max_neighbor_deg - deg)
                            + [0.0])   # no-op 槽对 READY agent 始终=0
                active   = 1
            else:
                mask   = [0.0] * self.max_neighbor_deg + [1.0]  # 仅 no-op 槽=1
                active = 0

            infos[f"agent_{i}"] = {
                "action_mask": np.array(mask, dtype=np.float32),
                "active_mask": active,
            }
        return infos

    # ------------------------------------------------------------------
    #  奖励
    # ------------------------------------------------------------------

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        w      = self.world
        phi    = w.graph.phi
        idle   = w.node_idleness
        agents = self.agents

        weighted    = self._phi_weighted_idleness()
        avg_phi_idle = float(weighted.mean())

        # 是否为截断步（在奖励中同时附加 terminal reward）
        if self.truncate_by_time:
            is_terminal = w.current_time >= (self.episode_len - 1e-9)
        else:
            is_terminal = w.step_count >= self.episode_len

        rewards: Dict[str, float] = {}
        for i in range(self.agent_num):
            agent_str = f"agent_{i}"
            r = 0.0

            if i in result.arrivals:
                arrived_node  = result.arrivals[i]
                phi_v         = float(phi.get(arrived_node, 1))
                idle_v        = float(idle.get(arrived_node, 0.0))
                phi_idle_v    = phi_v * idle_v
                r_local       = self.alpha * phi_idle_v / (avg_phi_idle + self.eps)
                r += r_local

            if is_terminal:
                r_term = self.beta * float(w.current_time) / (avg_phi_idle + self.eps)
                r += r_term

            rewards[agent_str] = r

        return rewards

    # ------------------------------------------------------------------
    #  动作解码
    # ------------------------------------------------------------------

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        cur_node  = self.world.agents[agent_id].position
        neighbors = self.sorted_adj.get(cur_node, [])
        if action_idx < len(neighbors):
            return neighbors[action_idx][0]
        # 越界保底（action_mask 应已防止）
        return neighbors[0][0] if neighbors else cur_node

    # ------------------------------------------------------------------
    #  截断
    # ------------------------------------------------------------------

    def _compute_truncations(self) -> Dict[str, bool]:
        w = self.world
        if self.truncate_by_time:
            trunc = w.current_time >= (self.episode_len - 1e-9)
        else:
            trunc = w.step_count >= self.episode_len
        return {a: trunc for a in self.agents}

    # ------------------------------------------------------------------
    #  辅助
    # ------------------------------------------------------------------

    def _get_neighbor_index(self, v: int, nbr: int) -> int:
        """返回 nbr 在 sorted_adj[v] 中的位置（neighborIndex）。"""
        for k, (n, _) in enumerate(self.sorted_adj.get(v, [])):
            if n == nbr:
                return k
        return -1

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}
