#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum

import numpy as np

from utils.graph_utils import Graph
from utils.log_utils import EpisodeMetricsTracker, IdlenessMetrics

class AgentState(Enum):
    READY = "ready"          # ready to make decision.
    WAITING = "waiting"      # waiting on node.
    ON_EDGE = "on_edge"      # moving on edge.

@dataclass
class AgentStatus:
    """Physical state of one patrol agent."""

    position: int = -1                     # Current node (source node if on an edge).
    last_position: int = -1             # Last node.
    state: AgentState = AgentState.READY
    target_node: int = -1
    action_remaining: float = 0.0  # Actual time remaining, including jitter.
    nominal_action_remaining: float = 0.0  # Time exposed in observations.
    planned_edge_duration: float = 0.0  # Actual duration used for animation.
    speed: float = 1.0

@dataclass
class TickResult:
    """Events, rewards, and metric snapshots produced by one time advance."""

    dt: float                                               # Actual elapsed time.
    arrivals: Dict[int, int] = field(default_factory=dict)  # {agent_id: arrived_node}.
    wait_completed: Set[int] = field(default_factory=set)   # Agents that finish waiting.
    raw_rewards: Dict[int, float] = field(default_factory=dict)  # arrived_node_idleness*phi or waitT*phi or 0.0.
    pre_arrival_igi: float = 0.0  # IGI before arrived nodes are reset.
    pre_arrival_weighted_iwi: float = 0.0  # Weighted IWI before reset.

    @property
    def ready_agents(self) -> Set[int]:
        return set(self.arrivals.keys()) | self.wait_completed

class PatrolWorld:
    """Physical simulator shared by the patrol MDPs.

    The world owns the graph, agent motion, node idleness, and time advancement;
    observation, action, and reward definitions remain in the MDP wrappers. It
    supports fixed-step advancement through :meth:`tick` and event-driven
    advancement through :meth:`tick_to_next_event`.
    """

    def __init__(self, cfg: Dict):
        graph_path = cfg["graph_path"]
        self.graph = Graph(graph_path)

        self.max_neighbors = self.graph.get_max_degree()
        self.max_edge_length = self.graph.get_max_edge_length()
        self.max_path_length = self.graph.max_shortest_path_len
        self.max_phi = self.graph.get_max_phi()
        self.num_nodes = len(self.graph.nodes)
        self.num_edges = self.graph.get_num_edges(True)

        self.num_agents = cfg["num_agents"]
        self.speeds = cfg.get("speeds", [1.0] * self.num_agents)
        self.agents: Dict[int, AgentStatus] = {}

        self.node_idleness: Dict[int, float] = {n: 0.0 for n in self.graph.nodes}
        self.current_time: float = 0.0
        self.worst_idleness: float = 0.0
        self.waitT = cfg.get("deltaT", 1.0)

        self._node_order: List[int] = list(self.graph.nodes)
        self._node_idx: Dict[int, int] = {n: i for i, n in enumerate(self._node_order)}

        self._phi_arr: np.ndarray = np.array(
            [float(self.graph.phi.get(n, 1.0)) for n in self._node_order],
            dtype=np.float64,
        )

        self._idleness_arr: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)

        self._weighted_arr: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)

        self._pre_arrival_weighted_arr: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)

        self._occupied_count: np.ndarray = np.zeros(self.num_nodes, dtype=np.int32)

        self.metrics_tracker = EpisodeMetricsTracker(
            training_mode=cfg.get("training_mode", False)
        )
        self.last_episode_metrics: Optional[IdlenessMetrics] = None
        self.step_count: int = 0

        self._occupied_nodes: Set[int] = set()

        self._routes: Dict[int, List[int]] = {}
        self._acc_rewards: Dict[int, float] = {}

        _mode = cfg.get("edge_time_jitter_mode", "none")
        self._jitter_mode    = _mode
        self._jitter_enabled = (_mode != "none")
        self._jitter_obs_real = (_mode == "full")
        self._jitter_frac    = float(cfg.get("edge_time_jitter_frac", 0.1))
        import random as _random_mod
        _seed = cfg.get("edge_time_jitter_seed", None)
        self._jitter_rng = _random_mod.Random(_seed)

    def reset(self, initial_positions: Optional[List[int]] = None, seed: Optional[int] = None) -> None:
        if self.metrics_tracker.has_data:
            self.last_episode_metrics = self.metrics_tracker.current

        self.current_time = 0.0
        self.step_count = 0
        self.worst_idleness = 0.0

        self.metrics_tracker.reset()

        if initial_positions is None:
            import random as _random_mod
            _pos_rng = _random_mod.Random(seed)
            initial_positions = _pos_rng.sample(list(self.graph.nodes), self.num_agents)

        self._occupied_nodes.clear()
        self._routes.clear()
        self._acc_rewards.clear()

        self._idleness_arr[:] = 0.0
        self._weighted_arr[:] = 0.0
        self._pre_arrival_weighted_arr[:] = 0.0
        self._occupied_count[:] = 0

        for i in range(self.num_agents):
            pos = initial_positions[i]
            self.agents[i] = AgentStatus(
                position=pos,
                state=AgentState.READY,
                last_position=pos,
                speed=self.speeds[i]
            )
            self._occupied_nodes.add(pos)
            self._occupied_count[self._node_idx[pos]] += 1

        self.node_idleness = {n: 0.0 for n in self._node_order}

        self.metrics_tracker.record(self._weighted_arr, self.step_count, self.current_time)

    def tick(self, dt: float) -> TickResult:
        """Advance physical time by ``dt`` and return all completed events.

        Idleness is advanced first. IGI and weighted IWI are captured before
        resetting nodes reached at this instant, so arrival peaks are preserved
        in evaluation metrics. Agent states and rewards are then updated, and
        post-arrival weighted idleness is cached for observations.
        """

        raw_rewards = {a:0.0 for a in range(self.num_agents)}
        if dt < 0:
            return TickResult(dt=0.0, raw_rewards=raw_rewards)

        result = TickResult(dt=dt, raw_rewards=raw_rewards)

        free_mask = self._occupied_count == 0
        self._idleness_arr[free_mask] += dt

        # Capture metric peaks before arrivals reset node idleness.
        result.pre_arrival_igi = float(self._idleness_arr.mean())
        current_worst_idleness = float(self._idleness_arr.max())
        self.worst_idleness = max(self.worst_idleness, current_worst_idleness)

        np.multiply(self._phi_arr, self._idleness_arr, out=self._pre_arrival_weighted_arr)
        result.pre_arrival_weighted_iwi = float(self._pre_arrival_weighted_arr.max())

        for agent_id, status in self.agents.items():
            if status.state in (AgentState.ON_EDGE, AgentState.WAITING):
                status.action_remaining         -= dt
                status.nominal_action_remaining -= dt

                if status.action_remaining <= 1e-9:
                    status.action_remaining         = 0.0
                    status.nominal_action_remaining = 0.0

                    if status.state == AgentState.ON_EDGE:
                        arrived_node = status.target_node
                        status.last_position = status.position
                        status.position = arrived_node
                        status.state = AgentState.READY
                        status.target_node = -1
                        status.planned_edge_duration = 0.0

                        ai = self._node_idx[arrived_node]
                        self._occupied_nodes.add(arrived_node)
                        self._occupied_count[ai] += 1

                        arrival_reward = float(self._idleness_arr[ai]) * float(self._phi_arr[ai])
                        self._idleness_arr[ai] = 0.0

                        if agent_id in self._routes and self._routes[agent_id]:
                            self._acc_rewards[agent_id] = (
                                self._acc_rewards.get(agent_id, 0.0) + arrival_reward
                            )
                            next_hop = self._routes[agent_id].pop(0)
                            self.set_move_action(agent_id, next_hop)
                            # Intermediate route rewards remain hidden until the destination.
                            result.raw_rewards[agent_id] = 0.0
                        else:
                            acc = self._acc_rewards.pop(agent_id, 0.0)
                            result.raw_rewards[agent_id] = acc + arrival_reward
                            result.arrivals[agent_id] = arrived_node

                    elif status.state == AgentState.WAITING:
                        status.state = AgentState.READY
                        waiting_node = self.agents[agent_id].position
                        result.raw_rewards[agent_id] = self.waitT * self.graph.phi[waiting_node]
                        result.wait_completed.add(agent_id)

        self.current_time += dt

        self.node_idleness.update(zip(self._node_order, self._idleness_arr.tolist()))

        self.step_count += 1
        np.multiply(self._phi_arr, self._idleness_arr, out=self._weighted_arr)
        self.metrics_tracker.record(self._pre_arrival_weighted_arr, self.step_count, self.current_time)

        return result

    def tick_to_next_event(self) -> TickResult:
        """Advance to the earliest arrival or completed wait event."""

        dt = self._compute_next_event_time()
        if dt < 0:
            raw_rewards = {a: 0.0 for a in range(self.num_agents)}
            return TickResult(dt=0.0, raw_rewards=raw_rewards)
        return self.tick(dt)

    def _compute_next_event_time(self) -> float:
        min_time = float('inf')

        for status in self.agents.values():
            if status.state in (AgentState.ON_EDGE, AgentState.WAITING, AgentState.READY):
                if status.action_remaining >= 0:
                    min_time = min(min_time, status.action_remaining)

        return min_time if min_time != float('inf') else -1.0

    def set_move_action(self, agent_id: int, target_node: int) -> bool:
        """Start moving a READY agent to an adjacent node."""

        status = self.agents[agent_id]

        if status.state != AgentState.READY:
            return False

        current_pos = status.position

        if target_node not in self.graph.get_neighbors(current_pos):
            return False

        status.state = AgentState.ON_EDGE
        status.target_node = target_node
        edge_length = self.graph.get_edge_length(current_pos, target_node)
        status.last_position = current_pos

        T_nom = float(edge_length) / max(status.speed, 1e-6)
        if self._jitter_enabled:
            frac = self._jitter_frac
            T_act = T_nom * self._jitter_rng.uniform(1.0 - frac, 1.0 + frac)
        else:
            T_act = T_nom

        # Full jitter exposes actual time; dual jitter exposes nominal time only.
        T_obs = T_act if self._jitter_obs_real else T_nom
        status.nominal_action_remaining = T_obs
        status.action_remaining         = T_act
        status.planned_edge_duration    = T_act

        self._occupied_nodes.discard(current_pos)
        ni = self._node_idx[current_pos]
        if self._occupied_count[ni] > 0:
            self._occupied_count[ni] -= 1

        return True

    def set_route_action(self, agent_id: int, target_node: int) -> bool:
        """Route a READY agent to any reachable node along a shortest path.

        Rewards from intermediate arrivals are accumulated, and the agent
        becomes READY only at the final destination. Adjacent targets delegate
        directly to :meth:`set_move_action`.
        """

        status = self.agents[agent_id]
        if status.state != AgentState.READY:
            return False

        current_pos = status.position

        if target_node == current_pos:
            return self.set_wait_action(agent_id)

        if target_node in self.graph.get_neighbors(current_pos):
            return self.set_move_action(agent_id, target_node)

        path = self.graph.get_shortest_path(current_pos, target_node)
        if path is None or len(path) < 2:
            return False

        # Skip the current node and start with the first hop.
        first_hop = path[1]
        remaining = path[2:]
        self._routes[agent_id] = remaining
        self._acc_rewards[agent_id] = 0.0
        return self.set_move_action(agent_id, first_hop)

    def set_wait_action(self, agent_id: int) -> bool:
        """Make a READY agent wait at its current node for ``waitT``."""

        status = self.agents[agent_id]

        if status.state != AgentState.READY:
            return False

        status.state = AgentState.WAITING
        status.action_remaining         = self.waitT
        status.nominal_action_remaining = self.waitT
        status.planned_edge_duration    = 0.0
        status.last_position = status.position
        status.target_node = status.position

        return True

    def is_ready(self, agent_id: int) -> bool:
        return self.agents[agent_id].state == AgentState.READY

    def get_ready_agents(self) -> List[int]:
        return [i for i in range(self.num_agents) if self.is_ready(i)]

    def get_position(self, agent_id: int) -> int:
        return self.agents[agent_id].position

    def get_neighbors(self, node: int) -> List[int]:
        return self.graph.get_neighbors(node)

    def get_node_idleness(self, node: int) -> float:
        return self.node_idleness.get(node, 0.0)

    def get_all_idleness(self) -> Dict[int, float]:
        return dict(self.node_idleness)

    def get_agent_status(self, agent_id: int) -> AgentStatus:
        return self.agents[agent_id]

    def snapshot_agent_positions(self) -> Dict[int, tuple]:
        """Return ``agent_id -> (source, target, progress)`` for visualization.

        Stationary agents use the same source and target with zero progress.
        Moving agents report progress in ``[0, 1]`` using actual travel time.
        """

        snapshot = {}
        for agent_id, status in self.agents.items():
            if status.state == AgentState.ON_EDGE:
                travel_time = status.planned_edge_duration
                if travel_time < 1e-12:
                    travel_time = self.graph.get_edge_length(status.position, status.target_node)\
                                  / max(status.speed, 1e-6)
                progress = 1.0 - status.action_remaining / travel_time if travel_time > 0 else 1.0
                progress = max(0.0, min(1.0, progress))
                snapshot[agent_id] = (status.position, status.target_node, progress)
            else:
                snapshot[agent_id] = (status.position, status.position, 0.0)
        return snapshot

    @property
    def current_metrics(self) -> IdlenessMetrics:
        return self.metrics_tracker.current

    def get_episode_metrics(self) -> Dict[str, List[float]]:
        return self.metrics_tracker.get_history_dict()

    def plot_episode_metrics(self, save_path: str = None, show: bool = True, use_time_axis: bool = False):
        self.metrics_tracker.plot(save_path=save_path, show=show, use_time_axis=use_time_axis)

    def export_metrics_to_csv(self, path: str):
        self.metrics_tracker.to_csv(path)

    def get_heuristic_obs(self) -> Dict[str, Dict]:
        obs_dict = {}
        for agent_id in range(self.num_agents):
            agent_status = self.agents[agent_id]
            current_pos = agent_status.position
            neighbors = self.graph.get_neighbors(current_pos)
            on_edge = agent_status.state == AgentState.ON_EDGE

            obs_dict[f"agent_{agent_id}"] = {
                'current_node': current_pos,
                'neighbors': neighbors,
                'on_edge': on_edge,
            }
        return obs_dict

    def get_global_state_for_heuristic(self) -> Dict:
        """Build the graph and agent state consumed by heuristic policies.

        The result contains the graph, positions, motion states, targets,
        simulation time, idleness and inferred last-visit times, speeds, and
        average edge length.
        """

        node_last_visit = {
            n: self.current_time - self.node_idleness[n]
            for n in self.graph.nodes
        }

        return {
            'graph': self.graph,
            'agent_positions': {
                i: self.agents[i].position
                for i in range(self.num_agents)
            },
            'agents_on_edge': {
                i: self.agents[i].state == AgentState.ON_EDGE
                for i in range(self.num_agents)
            },
            'agents_target_node': {
                i: self.agents[i].target_node
                for i in range(self.num_agents)
            },
            'current_time': self.current_time,
            'node_last_visit': node_last_visit,
            'node_idleness': dict(self.node_idleness),
            'agent_speeds': self.speeds,
            'er_avg_edge_len': self.graph.get_average_edge_length() if hasattr(self.graph, 'get_average_edge_length') else 1.0,
        }

    def step_heuristic(self, actions: Dict[str, int]) -> TickResult:
        """Apply neighbor-index actions for READY agents and advance one event."""

        for agent_str, neighbor_idx in actions.items():
            agent_id = int(agent_str.split("_")[1]) if isinstance(agent_str, str) else int(agent_str)

            if not self.is_ready(agent_id):
                continue

            current_pos = self.get_position(agent_id)
            neighbors = self.graph.get_neighbors(current_pos)

            if 0 <= neighbor_idx < len(neighbors):
                target_node = neighbors[neighbor_idx]
                self.set_move_action(agent_id, target_node)

        return self.tick_to_next_event()
