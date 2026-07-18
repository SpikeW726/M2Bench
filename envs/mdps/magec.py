"""Environment for MAGEC graph-based multi-agent coordination.

The environment is event-driven and uses ``max_degree + 1`` actions, reserving
the final slot for no-op. Agents on edges expose only no-op and have a zero active
mask. Node idleness is priority-weighted before normalization, virtual-agent
edges follow the paper's construction, and episodes may truncate by physical
time or environment steps.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium.spaces import Box, Discrete

from envs.mdps.base_envs import EventDrivenEnv
from envs.mdps.patrol_core import AgentState, TickResult

class MAGECEnv(EventDrivenEnv):
    NODE_TYPE_PATROL = 0.0
    NODE_TYPE_AGENT  = 1.0
    NEIGHBOR_IDX_NONE = -1.0

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        w = self.world
        graph = w.graph

        self.static_node_num   = w.num_nodes
        self.static_edge_num   = w.num_edges
        self.agent_num         = w.num_agents
        self.max_neighbor_deg  = w.max_neighbors
        self.max_edge_length   = w.max_edge_length

        # action space: neighbor indices 0.max_degree-1 + no-op.
        self.action_dim = self.max_neighbor_deg + 1

        self.sorted_nodes = sorted(graph.nodes)
        self.node_to_idx  = {n: i for i, n in enumerate(self.sorted_nodes)}

        self.sorted_adj: Dict[int, List[Tuple[int, float]]] = {
            v: sorted(graph.adj_list[v], key=lambda x: x[0])
            for v in self.sorted_nodes
        }

        self.static_edges: List[Tuple[int, int, float]] = []
        for v in self.sorted_nodes:
            for nbr, w_edge in self.sorted_adj[v]:
                self.static_edges.append((v, nbr, w_edge))

        self.node_feat_dim   = 3   # [nodeType, idleness_norm, degree].
        self.edge_feat_dim   = 2   # [weight_norm, neighborIndex].
        self.global_feat_dim = 2   # [avg_phi_idle_norm, worst_phi_idle_norm].
        self.identity_dim    = self.agent_num  # one-hot.

        self.total_nodes = self.static_node_num + self.agent_num

        self.max_agent_edges   = 1 + self.max_neighbor_deg
        self.total_max_edges   = self.static_edge_num + self.agent_num * self.max_agent_edges

        self.obs_size = (
            self.total_nodes * self.node_feat_dim
            + self.total_max_edges * 2                   # edge_src + edge_dst.
            + self.total_max_edges * self.edge_feat_dim  # edge_attr.
            + self.total_max_edges                       # edge_mask.
            + self.global_feat_dim
            + self.identity_dim                          # identity.
        )

        self.episode_len      = config["episode_len"]
        self.truncate_by_time = kwargs.get("truncate_by_time", True)

        self.alpha = kwargs.get("alpha", 1.0)
        self.beta  = kwargs.get("beta", 0.5)
        self.eps   = 1e-6

        n = self.static_node_num
        self._adj_matrix = np.zeros((n, n), dtype=np.float32)
        for v in self.sorted_nodes:
            vi = self.node_to_idx[v]
            for nbr, _ in graph.adj_list[v]:
                ni = self.node_to_idx[nbr]
                self._adj_matrix[vi, ni] = 1.0
        self._adj_flat = self._adj_matrix.flatten()

        self._neighbor_pos_of: Dict[Tuple[int, int], int] = {}
        for v in self.sorted_nodes:
            for k, (nbr, _) in enumerate(self.sorted_adj[v]):
                self._neighbor_pos_of[(v, nbr)] = k

        ml_static = max(self.max_edge_length, self.eps)
        self._static_e_src = np.empty((self.static_edge_num,), dtype=np.float32)
        self._static_e_dst = np.empty((self.static_edge_num,), dtype=np.float32)
        self._static_e_attr = np.empty((self.static_edge_num, 2), dtype=np.float32)
        for idx, (v, nbr, w_edge) in enumerate(self.static_edges):
            k = float(self._neighbor_pos_of[(v, nbr)])
            self._static_e_src[idx] = float(self.node_to_idx[v])
            self._static_e_dst[idx] = float(self.node_to_idx[nbr])
            self._static_e_attr[idx, 0] = w_edge / ml_static
            self._static_e_attr[idx, 1] = k

        self._phi_vec = np.array(
            [float(graph.phi.get(v, 1.0)) for v in self.sorted_nodes],
            dtype=np.float32,
        )
        self._static_deg = np.array(
            [float(len(graph.adj_list[v])) for v in self.sorted_nodes],
            dtype=np.float32,
        )

        nm = self.total_max_edges
        tn = self.total_nodes
        self._node_feats_buf = np.zeros((tn, 3), dtype=np.float32)
        self._edge_src_buf = np.zeros((nm,), dtype=np.float32)
        self._edge_dst_buf = np.zeros((nm,), dtype=np.float32)
        self._edge_attr_buf = np.zeros((nm, 2), dtype=np.float32)
        self._edge_mask_buf = np.zeros((nm,), dtype=np.float32)
        self._global_buf = np.zeros(2, dtype=np.float32)
        self._identity_rows = np.eye(self.agent_num, dtype=np.float32)

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
        weighted = self._phi_weighted_idleness()
        w_min, w_max = weighted.min(), weighted.max()
        idleness_norm = (weighted - w_min) / (w_max - w_min + self.eps)
        return np.concatenate([idleness_norm, self._adj_flat]).astype(np.float32)

    def _phi_weighted_idleness(self) -> np.ndarray:
        idle = self.world.node_idleness
        idle_vals = np.array(
            [float(idle.get(v, 0.0)) for v in self.sorted_nodes],
            dtype=np.float32,
        )
        return self._phi_vec * idle_vals

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        w = self.world
        ml = max(self.max_edge_length, self.eps)

        weighted = self._phi_weighted_idleness()
        w_min = float(weighted.min())
        w_max = float(weighted.max())
        denom = w_max - w_min + self.eps

        nf = self._node_feats_buf
        sn = self.static_node_num
        idle_norm = (weighted - w_min) / denom
        nf[:sn, 0] = self.NODE_TYPE_PATROL
        nf[:sn, 1] = idle_norm
        nf[:sn, 2] = self._static_deg
        for i in range(self.agent_num):
            ag = w.agents[i]
            row = sn + i
            if ag.state == AgentState.READY:
                virt_deg = 1.0 + float(len(self.sorted_adj.get(ag.position, [])))
            else:
                virt_deg = 2.0
            nf[row, 0] = self.NODE_TYPE_AGENT
            nf[row, 1] = 0.0
            nf[row, 2] = virt_deg

        es = self._edge_src_buf
        ed = self._edge_dst_buf
        ea = self._edge_attr_buf
        n_static = self.static_edge_num
        es[:n_static] = self._static_e_src
        ed[:n_static] = self._static_e_dst
        ea[:n_static] = self._static_e_attr

        pos = n_static
        real_edge_count = n_static

        for i in range(self.agent_num):
            ag = w.agents[i]
            virt_idx = float(self.static_node_num + i)

            if ag.state in (AgentState.READY, AgentState.WAITING):
                cur_node = ag.position
                cur_idx = float(self.node_to_idx[cur_node])
                nbrs = self.sorted_adj.get(cur_node, [])

                es[pos] = virt_idx
                ed[pos] = cur_idx
                ea[pos, 0] = 0.0
                ea[pos, 1] = self.NEIGHBOR_IDX_NONE
                pos += 1
                real_edge_count += 1

                for k, (nbr, w_edge) in enumerate(nbrs):
                    es[pos] = virt_idx
                    ed[pos] = float(self.node_to_idx[nbr])
                    ea[pos, 0] = w_edge / ml
                    ea[pos, 1] = float(k)
                    pos += 1
                    real_edge_count += 1

                used = 1 + len(nbrs)
                for _ in range(self.max_agent_edges - used):
                    es[pos] = virt_idx
                    ed[pos] = virt_idx
                    ea[pos, 0] = 0.0
                    ea[pos, 1] = self.NEIGHBOR_IDX_NONE
                    pos += 1
            else:
                src_node = ag.position
                dst_node = ag.target_node
                src_idx = float(self.node_to_idx.get(src_node, 0))
                dst_idx = float(self.node_to_idx.get(dst_node, 0))

                try:
                    full_dist = float(w.graph.get_edge_length(src_node, dst_node))
                except Exception:
                    full_dist = float(self.max_edge_length)

                remaining_dist = float(ag.nominal_action_remaining) * float(ag.speed)
                dist_traveled = max(0.0, full_dist - remaining_dist)

                es[pos] = virt_idx
                ed[pos] = src_idx
                ea[pos, 0] = dist_traveled / ml
                ea[pos, 1] = self.NEIGHBOR_IDX_NONE
                pos += 1

                es[pos] = virt_idx
                ed[pos] = dst_idx
                ea[pos, 0] = remaining_dist / ml
                ea[pos, 1] = self.NEIGHBOR_IDX_NONE
                pos += 1
                real_edge_count += 2

                for _ in range(self.max_agent_edges - 2):
                    es[pos] = virt_idx
                    ed[pos] = virt_idx
                    ea[pos, 0] = 0.0
                    ea[pos, 1] = self.NEIGHBOR_IDX_NONE
                    pos += 1

        mask = self._edge_mask_buf
        mask[:real_edge_count] = 1.0
        mask[real_edge_count:] = 0.0

        avg_phi_idle = float(weighted.mean())
        worst_phi_idle = float(weighted.max())
        norm_denom = worst_phi_idle + self.eps
        gb = self._global_buf
        gb[0] = avg_phi_idle / norm_denom
        gb[1] = 1.0

        prefix = np.concatenate(
            (nf.ravel(), es, ed, ea.reshape(-1), mask, gb),
        )

        obs: Dict[str, np.ndarray] = {}
        eye = self._identity_rows
        for i in range(self.agent_num):
            obs[f"agent_{i}"] = np.concatenate((prefix, eye[i]))

        return obs

    # Info(action_mask + active_mask).

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
                            + [0.0])
                active   = 1
            else:
                mask   = [0.0] * self.max_neighbor_deg + [1.0]
                active = 0

            infos[f"agent_{i}"] = {
                "action_mask": np.array(mask, dtype=np.float32),
                "active_mask": active,
            }
        return infos

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        w      = self.world
        phi    = w.graph.phi
        idle   = w.node_idleness
        agents = self.agents

        weighted    = self._phi_weighted_idleness()
        avg_phi_idle = float(weighted.mean())

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

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        cur_node  = self.world.agents[agent_id].position
        neighbors = self.sorted_adj.get(cur_node, [])
        if action_idx < len(neighbors):
            return neighbors[action_idx][0]

        return neighbors[0][0] if neighbors else cur_node

    def _compute_truncations(self) -> Dict[str, bool]:
        w = self.world
        if self.truncate_by_time:
            trunc = w.current_time >= (self.episode_len - 1e-9)
        else:
            trunc = w.step_count >= self.episode_len
        return {a: trunc for a in self.agents}

    def _get_neighbor_index(self, v: int, nbr: int) -> int:
        return int(self._neighbor_pos_of.get((v, nbr), -1))

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}
