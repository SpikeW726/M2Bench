"""Extended Gray-Box Learner Agent environment from Lauri and Koukam (2014).

Ex-GBLA extends GBLA by sorting all neighboring actions by decreasing idleness
instead of encoding only extrema. A reward is set to zero when another agent has
already selected the arrived node as its target.
"""

from typing import Any, Dict, Optional
import numpy as np
from gymnasium.spaces import Box

from envs.mdps.gbla import GBLAEnv
from envs.mdps.patrol_core import AgentState, TickResult

class ExGBLAEnv(GBLAEnv):
    def observation_space(self, agent: str):
        """
        Ex-GBLA obs: [current_pos, last_pos, sorted_neighbors(M), intentions(M)]
        size: 2 + 2 * max_neighbors
        """
        max_node_id = self.world.num_nodes - 1
        M = self.world.max_neighbors

        low = np.concatenate([
            np.array([-1, -1]),         # current_pos, last_pos.
            np.full(M, -1),
            np.full(M, -1),
        ]).astype(np.int32)

        high = np.concatenate([
            np.array([max_node_id, max_node_id]),
            np.full(M, M),
            np.ones(M),
        ]).astype(np.int32)

        return Box(low=low, high=high, dtype=np.int32)

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        M = self.world.max_neighbors
        obs = {}

        for agent_id in range(self.world.num_agents):
            agent_str = f"agent_{agent_id}"

            if self.world.agents[agent_id].state == AgentState.ON_EDGE:
                obs[agent_str] = np.full(2 + 2 * M, -1, dtype=np.int32)
                continue

            current_pos = self.world.agents[agent_id].position
            raw_last = self.world.agents[agent_id].last_position
            if raw_last == current_pos:
                last_edge = -1
            else:
                last_edge = self.world.graph.neighbor_to_edge(current_pos, raw_last)

            neighbors = [n for n, _ in self.world.graph.adj_list.get(current_pos, [])]
            sorted_edges = np.full(M, -1, dtype=np.int32)
            if neighbors:
                pairs = [
                    (n, self.world.graph.phi.get(n, 1.0) * self.world.node_idleness[n])
                    for n in neighbors
                ]
                pairs.sort(key=lambda x: x[1], reverse=True)
                for i, (n, _) in enumerate(pairs):
                    sorted_edges[i] = self.world.graph.neighbor_to_edge(current_pos, n)

            intentions = np.full(M, -1, dtype=np.int32)
            action_neighbors = self.world.graph.get_neighbors(current_pos)
            intentions[:len(action_neighbors)] = 0
            for other_id, target in self.agent_intentions.items():
                if other_id != agent_id and target in action_neighbors:
                    idx = action_neighbors.index(target)
                    intentions[idx] = 1

            single_obs = np.empty(2 + 2 * M, dtype=np.int32)
            single_obs[0] = current_pos
            single_obs[1] = last_edge
            single_obs[2:2 + M] = sorted_edges
            single_obs[2 + M:] = intentions
            obs[agent_str] = single_obs

        return obs

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        rewards = super()._compute_rewards(result)

        for agent_id, arrived_node in result.arrivals.items():
            agent_str = f"agent_{agent_id}"

            for other_id, target in self.agent_intentions.items():
                if other_id != agent_id and target == arrived_node:
                    rewards[agent_str] = 0.0
                    break

        return rewards
