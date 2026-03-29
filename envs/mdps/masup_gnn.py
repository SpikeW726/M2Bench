from typing import Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.masup import MASUPEnv
from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import EventDrivenEnv

class MASUPGraphEnv(MASUPEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        self.static_node_num = self.world.num_nodes
        self.static_edge_num = self.world.num_edges
        self.agent_num = self.world.num_agents

        self.static_edges = []
        for u in sorted(self.world.graph.nodes):
            for v, weight in self.world.graph.adj_list[u]:
                self.static_edges.append((u, v, weight))        

        # 节点总数 = 物理节点 + 虚拟智能体节点
        self.total_node_num = self.static_node_num + self.agent_num

        # 最大边数 = 静态边 + 每个智能体的动态边上限
        self.max_dynamic_edges = self.agent_num * 2
        self.total_max_edges = self.static_edge_num + self.max_dynamic_edges    

        self.node_feat_dim = kwargs.get("node_feat_dim", 2)     # [Type, Weighted_Idleness]
        self.edge_feat_dim = kwargs.get("edge_feat_dim", 1)     # [Edge Weight]
        self.global_feat_dim = kwargs.get("global_feat_dim", 2)  # [WI@T (worst_idleness_fromT), obs_timer]

        if self.role_ifm == "agent-index":
            identity_len = self.world.num_agents
        elif self.role_ifm == "position":
            identity_len = 1
        elif self.role_ifm == "decision":
            identity_len = 1 + self.world.num_agents

        self.obs_size = (
            self.total_node_num * self.node_feat_dim +
            self.total_max_edges * 2 +  # Edge Index (Src + Dst)
            self.total_max_edges * self.edge_feat_dim +
            self.total_max_edges +      # Mask
            self.global_feat_dim +
            identity_len
        )

        # ---- _build_obs 加速: 预计算静态结构 + 复用 buffer ----
        self._sorted_nodes_gnn = sorted(self.world.graph.nodes)
        self._node_to_idx_gnn = {n: i for i, n in enumerate(self._sorted_nodes_gnn)}
        self._default_node_gnn = self._sorted_nodes_gnn[0] if self._sorted_nodes_gnn else 0
        self._phi_vec_gnn = np.array(
            [float(self.world.graph.phi.get(n, 1.0)) for n in self._sorted_nodes_gnn],
            dtype=np.float32,
        )
        self._static_e_src_gnn = np.array(
            [float(self._node_to_idx_gnn[u]) for u, v, w in self.static_edges],
            dtype=np.float32,
        )
        self._static_e_dst_gnn = np.array(
            [float(self._node_to_idx_gnn[v]) for u, v, w in self.static_edges],
            dtype=np.float32,
        )
        self._static_e_w_gnn = np.array(
            [float(w) for u, v, w in self.static_edges],
            dtype=np.float32,
        )
        nm = self.total_max_edges
        self._e_src_buf = np.zeros(nm, dtype=np.float32)
        self._e_dst_buf = np.zeros(nm, dtype=np.float32)
        self._e_w_buf = np.zeros(nm, dtype=np.float32)
        self._e_mask_buf = np.zeros(nm, dtype=np.float32)
        if self.role_ifm == "agent-index":
            self._identity_rows_gnn = np.eye(self.agent_num, dtype=np.float32)

    def observation_space(self, agent):
        # node_num = self.total_node_num
        # --- 1. 节点特征空间 ---
        node_low = [0.0] * (self.total_node_num * self.node_feat_dim)
        node_high = [float('inf')] * (self.total_node_num * self.node_feat_dim)

        # --- 2. 边索引空间 (Src + Dst) ---
        edge_idx_low = [0.0] * (2 * self.total_max_edges)
        edge_idx_high = [float(self.total_node_num)] * (2 * self.total_max_edges)

        # --- 3. 边属性空间 ---
        edge_attr_low = [0.0] * self.total_max_edges
        edge_attr_high = [float('inf')] * self.total_max_edges

        # --- 4. 边掩码空间 ---
        mask_low = [0.0] * self.total_max_edges
        mask_high = [1.0] * self.total_max_edges

        # --- 5. 全局状态空间 ---
        global_low = [0.0] * self.global_feat_dim
        global_high = [float('inf')] * self.global_feat_dim

        # --- 6. Agent Identity
        if self.role_ifm == "agent-index":
            identity_low = [0.0] * self.agent_num
            identity_high = [1.0] * self.agent_num
        elif self.role_ifm == "position":
            identity_low = [0.0]
            identity_high = [float(self.static_node_num)]
        elif self.role_ifm == "decision":
            identity_low = [0.0] + [0.0] * self.agent_num
            identity_high = [float(self.static_node_num)] + [1.0] * self.agent_num

        low = np.array(node_low + edge_idx_low + edge_attr_low + mask_low + global_low + identity_low, dtype=np.float32)
        high = np.array(node_high + edge_idx_high + edge_attr_high + mask_high + global_high + identity_high, dtype=np.float32)

        return Box(low=low, high=high, dtype=np.float32)

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        """构建当前时刻的动态图观测（公共前缀只算一次，仅 identity 按 agent 不同）。"""
        node_to_idx = self._node_to_idx_gnn
        default_node = self._default_node_gnn

        # ======== 1. 节点特征 ========
        # 静态节点: [Type=1, phi*idle]  虚拟节点: [Type=0, 0]
        idle_vals = np.array(
            [float(self.world.node_idleness.get(n, 0.0)) for n in self._sorted_nodes_gnn],
            dtype=np.float32,
        )
        weighted = self._phi_vec_gnn * idle_vals             # (static_node_num,)
        nf = np.empty(self.total_node_num * self.node_feat_dim, dtype=np.float32)
        sn = self.static_node_num
        nf[0 : sn * 2 : 2] = 1.0                            # type = 1
        nf[1 : sn * 2 : 2] = weighted                       # phi * idle
        nf[sn * 2 :] = 0.0                                  # 虚拟节点全 0

        # ======== 2. 边 ========
        es = self._e_src_buf
        ed = self._e_dst_buf
        ew = self._e_w_buf
        n_static = self.static_edge_num
        es[:n_static] = self._static_e_src_gnn
        ed[:n_static] = self._static_e_dst_gnn
        ew[:n_static] = self._static_e_w_gnn

        pos = n_static

        for i in range(self.agent_num):
            virtual_node_idx = float(self.static_node_num + i)
            ag = self.world.agents[i]

            last_node = ag.last_position
            if last_node not in node_to_idx:
                last_node = default_node

            target_node = ag.target_node
            if target_node not in node_to_idx:
                target_node = ag.position

            time_left = float(ag.action_remaining)

            if ag.state == AgentState.ON_EDGE:
                u_idx = float(node_to_idx[last_node])
                v_idx = float(node_to_idx[target_node])
                full_dist = self.world.graph.get_edge_length(last_node, target_node)
                dist_to_go = time_left * float(ag.speed)
                dist_traveled = max(0.0, full_dist - dist_to_go)

                es[pos] = u_idx;  ed[pos] = virtual_node_idx;  ew[pos] = dist_traveled
                pos += 1
                es[pos] = virtual_node_idx;  ed[pos] = v_idx;  ew[pos] = dist_to_go
                pos += 1

            elif ag.state == AgentState.WAITING:
                u_idx = float(node_to_idx[last_node])
                es[pos] = u_idx;          ed[pos] = virtual_node_idx;  ew[pos] = 0.0
                pos += 1
                es[pos] = virtual_node_idx;  ed[pos] = u_idx;          ew[pos] = time_left
                pos += 1

            elif ag.state == AgentState.READY:
                u_idx = float(node_to_idx[last_node])
                es[pos] = u_idx;          ed[pos] = virtual_node_idx;  ew[pos] = 0.0
                pos += 1
                es[pos] = virtual_node_idx;  ed[pos] = u_idx;          ew[pos] = 0.0
                pos += 1

        # ======== 3. 掩码 ========
        current_edge_count = pos
        mask = self._e_mask_buf
        mask[:current_edge_count] = 1.0
        mask[current_edge_count:] = 0.0

        # 填充边（pad_size > 0 时才需要）
        if pos < self.total_max_edges:
            es[pos:] = 0.0;  ed[pos:] = 0.0;  ew[pos:] = 0.0

        # ======== 4. 全局特征 ========
        global_arr = np.array(
            [float(self.worst_idleness_fromT), float(self.obs_timer)],
            dtype=np.float32,
        )

        # ======== 5. 公共前缀（一次拼接）========
        prefix = np.concatenate((nf, es, ed, ew, mask, global_arr))

        # ======== 6. 每 agent 仅拼接 identity ========
        obs: Dict[str, np.ndarray] = {}
        for i in range(self.agent_num):
            if self.role_ifm == "agent-index":
                obs[f"agent_{i}"] = np.concatenate((prefix, self._identity_rows_gnn[i]))
            elif self.role_ifm == "position":
                obs[f"agent_{i}"] = np.append(prefix, float(self.world.agents[i].position))
            elif self.role_ifm == "decision":
                decision_idx = int(self._decision_index_map.get(i, 0)) if hasattr(self, '_decision_index_map') else 0
                one_hot = np.zeros(self.agent_num, dtype=np.float32)
                one_hot[decision_idx] = 1.0
                obs[f"agent_{i}"] = np.concatenate(
                    (prefix, [float(self.world.agents[i].position)], one_hot)
                )
        return obs