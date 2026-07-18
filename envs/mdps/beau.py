"""Patrolling environment for *Balancing Efficiency and Unpredictability*.

BEAU uses event-driven ``PatrolWorld`` dynamics, a shared ``-IGI`` reward, and
the graph interfaces required by the MAT collector. Per-agent observations are
minimal PettingZoo placeholders because MAT reads graph state directly. Agent
ordering is modeled by autoregressive decoding, while the environment still
receives one standard joint action dictionary.

Compared with the original grid implementation, this environment uses JSON
topologies and normalized graph coordinates, continuous travel times, no reward
offset, and full-graph node features rather than visibility heatmaps. It preserves
the decision-step collection protocol, but trajectories are not stepwise identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from gymnasium.spaces import Box, Discrete

from envs.mdps.base_envs import EventDrivenEnv
from envs.mdps.patrol_core import AgentState, TickResult

_STUB_OBS_DIM = 1

class BEAU(EventDrivenEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        self.episode_len: int = config["episode_len"]
        self.reward_scale: float = float(kwargs.get("reward_scale", 1.0))
        self.truncate_by_time: bool = kwargs.get("truncate_by_time", True)

        self._node_neighbor_count: Dict[int, int] = {
            n: len(self.world.graph.get_neighbors(n))
            for n in self.world.graph.nodes
        }

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
        )  # (G, 2).
        self._adj: np.ndarray = self._build_adj()

    def observation_space(self, agent: str) -> Box:
        return Box(low=0.0, high=1.0, shape=(_STUB_OBS_DIM,), dtype=np.float32)

    def action_space(self, agent: str) -> Discrete:
        return Discrete(self.world.max_neighbors + 2)

    def state(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        stub = np.zeros(_STUB_OBS_DIM, dtype=np.float32)
        return {a: stub for a in self.agents}

    def _compute_truncations(self) -> Dict[str, bool]:
        if self.truncate_by_time:
            done = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            done = self.world.step_count >= self.episode_len
        return {a: done for a in self.agents}

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        r = -float(result.pre_arrival_igi) * self.reward_scale
        return {a: r for a in self.agents}

    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        return {
            agent_str: {
                "action_mask": self._get_action_mask(agent_str),
                "active_mask": 1 if self.world.is_ready(int(agent_str.split("_")[1])) else 0,
            }
            for agent_str in self.agents
        }

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        current_pos = self.world.get_position(agent_id)
        neighbors = self.world.graph.get_neighbors(current_pos)
        neighbor_idx = action_idx - 1
        if 0 <= neighbor_idx < len(neighbors):
            return neighbors[neighbor_idx]
        raise ValueError(
            f"Invalid action: agent_id={agent_id}, action_idx={action_idx}, neighbors={len(neighbors)}"
        )

    def _get_action_mask(self, agent_str: str) -> np.ndarray:
        agent_id = int(agent_str.split("_")[1])
        mask = np.zeros(self.world.max_neighbors + 2, dtype=bool)
        if not self.world.is_ready(agent_id):
            mask[-1] = True   # no-op.
        else:
            n_nb = self._node_neighbor_count[self.world.get_position(agent_id)]
            mask[: n_nb + 1] = True   # 0=wait, 1.n_nb=neighbors.
        return mask

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}

    def state_mat(self) -> np.ndarray:
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
        return self._adj

    def get_current_node_indices(self) -> np.ndarray:
        out = np.empty(self.world.num_agents, dtype=np.int32)
        for i in range(self.world.num_agents):
            st = self.world.agents[i]
            node = st.last_position if st.state == AgentState.ON_EDGE else st.position
            out[i] = self._node_to_idx.get(node, 0)
        return out

    def graph_idx_to_action(self, graph_idx_list: list) -> dict:
        actions = {}
        for k, gidx in enumerate(graph_idx_list):
            st = self.world.agents[k]
            if st.state == AgentState.ON_EDGE:
                actions[f"agent_{k}"] = self.world.max_neighbors + 1  # no-op.
                continue
            target = self._sorted_nodes[int(gidx)]
            neighbors = self.world.graph.get_neighbors(st.position)
            actions[f"agent_{k}"] = neighbors.index(target) + 1 if target in neighbors else 0
        return actions

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
        print(f"[BEAU] Generated coordinate file: {coords_path}")
