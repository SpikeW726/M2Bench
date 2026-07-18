# -*- coding: utf-8 -*-
"""
ER (Expected Idleness) Heuristic Policy - Multi-agent version

Core idea:
  For each neighbor v_i, estimate the "expected idleness" I_exp when arriving:
    - If another agent has intention (v_i, eta), then I_exp = t_next - eta
    - Otherwise I_exp = t_next - last_visit_time[v_i]
  Select the neighbor with maximum utility U = |I_exp| / travel_time
"""
from typing import Dict, Optional, Any
from policies.heuritic.heuristic_base import HeuriticBasePolicy

class ERPolicy(HeuriticBasePolicy):
    """
    ER (Expected Idleness) heuristic policy for multi-agent patrolling.

    Design:
    - compute_actions: SEQUENTIAL decision - agents decide one by one
    - _compute_action: single agent decision using intention-based coordination
    - Maintains intention table (_intention_eta) for conflict avoidance

    Sequential Decision Protocol:
        Agent 0 decides, its intention is recorded, and agent 1 observes it before
        deciding. This pattern continues for every later agent.

    This allows later agents to avoid selecting the same target as earlier agents.
    """

    def __init__(self, num_agents: int, config: Dict):
        super().__init__(num_agents, config)

        ap = config.get("algorithm_params", {}) if isinstance(config.get("algorithm_params", {}), dict) else {}

        # Whether to use average edge length as lower bound (avoid short-edge oscillation).
        self.use_avg_len_lower_bound: bool = bool(
            config.get("use_avg_len_lower_bound", ap.get("use_avg_len_lower_bound", True))
        )

        self._intention_eta: Dict[int, float] = {}

    def reset(self):
        self._intention_eta.clear()

    def compute_actions(
        self,
        obs_dict: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Compute actions for all agents using SEQUENTIAL decision-making.

        Sequential Decision Protocol:
            - Agents decide one by one in order
            - Each agent's decision (intention) is immediately recorded
            - Subsequent agents can see previous agents' intentions and avoid conflicts

        Args:
            obs_dict: {agent_str: obs} where obs contains:
                - 'current_node': int
                - 'neighbors': List[int]
                - 'on_edge': bool (optional)

            global_state: global information including:
                - 'graph': Graph object
                - 'agent_positions': Dict[int, int]
                - 'agents_on_edge': Dict[int, bool]
                - 'agent_speeds': List[float] (agent speeds from physical world)
                - 'current_time': float (current simulation time)
                - 'node_last_visit': Dict[int, float] (last visit time for each node)
                - 'er_avg_edge_len': float (average edge length, optional)

        Returns:
            actions: {agent_str: action_idx} for agents that need to decide
        """
        actions = {}

        # Sequential decision: process agents one by one.
        for agent_str, obs in obs_dict.items():
            agent_id = int(agent_str.split("_")[1]) if isinstance(agent_str, str) else int(agent_str)

            # Skip agents that are moving on edge.
            on_edge = obs.get('on_edge', False)
            if global_state.get('agents_on_edge'):
                on_edge = global_state['agents_on_edge'].get(agent_id, on_edge)

            if on_edge:
                continue

            # Compute action (can see previous agents' intentions via self._intention_eta).
            result = self._compute_action(agent_id, obs, global_state)

            if result is not None:
                action_idx, target_node, eta = result
                actions[agent_str] = action_idx

                # IMMEDIATELY update intention so subsequent agents can see it.
                if target_node is not None and eta is not None:
                    existing_eta = self._intention_eta.get(target_node)
                    if existing_eta is None or eta < existing_eta:
                        self._intention_eta[target_node] = eta

        return actions

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> Optional[tuple]:
        """
        Compute action for a single agent using Expected Idleness heuristic.

        Args:
            agent_id: agent index
            obs: local observation with 'current_node' and 'neighbors'
            global_state: global state information

        Returns:
            tuple of (action_idx, target_node, eta) or None if no valid action
        """
        current_node = obs.get('current_node')
        neighbors = obs.get('neighbors', [])

        if not neighbors:
            return None

        graph = global_state.get('graph')
        current_time = global_state.get('current_time', 0.0)
        node_last_visit = global_state.get('node_last_visit', {})

        # Average edge length lower bound (to avoid short-edge oscillation).
        avg_len_lb = 0.0
        if self.use_avg_len_lower_bound:
            avg_len_lb = global_state.get('er_avg_edge_len', 0.0)

        best_utility = -1.0
        best_idx = None
        best_eta = None
        best_node = None

        # Get agent's speed from global_state (provided by physical world).
        agent_speeds = global_state.get('agent_speeds', [])
        speed = agent_speeds[agent_id] if agent_id < len(agent_speeds) else 1.0

        for idx, neighbor in enumerate(neighbors):
            # Travel time = edge_length / speed, with lower bound.
            edge_len = 1.0
            if graph is not None and hasattr(graph, 'get_edge_length'):
                edge_len = float(graph.get_edge_length(current_node, neighbor) or 1.0)

            travel_time = max(edge_len, avg_len_lb) / speed
            if travel_time <= 0:
                travel_time = 1e-6  # Avoid division by zero.

            t_next = current_time + travel_time  # Expected arrival time.

            # Expected idleness: use internal intention table if exists, otherwise use last visit time.
            intention_eta = self._intention_eta.get(neighbor)
            if intention_eta is not None and intention_eta >= current_time:
                # Another agent has intention for this node.
                I_exp = t_next - intention_eta
            else:
                # Use last visit time.
                last_visit = node_last_visit.get(neighbor, 0.0)
                I_exp = t_next - last_visit

            # Utility = |expected_idleness| / travel_time.
            utility = abs(I_exp) / travel_time

            if utility > best_utility:
                best_utility = utility
                best_idx = idx
                best_eta = t_next
                best_node = neighbor

        if best_idx is None:
            return None

        return (best_idx, best_node, best_eta)
