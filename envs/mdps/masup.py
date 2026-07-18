from typing import Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import EventDrivenEnv

class MASUPEnv(EventDrivenEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        self.obs_timer = 0
        self.T_flag = False

        self._agent_idleness_reduction = {}

        self.episode_len = config['episode_len']
        self.T_time = kwargs.get("T", 0.0)
        self.role_ifm = kwargs.get('role_imformation', "agent-index")

        self.last_episode_wi_fromT: Optional[float] = None

        self._wi_fromT_history: List[float] = []

        # reward trick.
        self.contribution_scale = float(kwargs.get('contribution_scale', 0.0))
        self.idi_scale = float(kwargs.get('idi_scale', 0.0))

        self.reward_scale = float(kwargs.get('reward_scale', 1.0))

        self.truncate_by_time = kwargs.get('truncate_by_time', True)

        if self.truncate_by_time:
            self.max_time_for_obs = self.episode_len
        else:
            self.max_time_for_obs = config.get(
                'max_time_for_obs', self.episode_len * self.world.max_edge_length
            )

        if self.role_ifm == "agent-index":
            self.obs_size = 4*self.world.num_agents + self.world.num_nodes + 3
        elif self.role_ifm == "position":
            self.obs_size = 3*self.world.num_agents + self.world.num_nodes + 4
        elif self.role_ifm == "decision":
            self.obs_size = 3*self.world.num_agents + self.world.num_nodes + 4 + self.world.num_agents

        self._obs_node_order: List = list(self.world.graph.nodes)
        self._phi_arr = np.array(
            [float(self.world.graph.phi[n]) for n in self._obs_node_order],
            dtype=np.float32,
        )
        if self.role_ifm == "agent-index":
            self._identity_rows = np.eye(self.world.num_agents, dtype=np.float32)

        self._node_neighbor_count: Dict[int, int] = {
            n: len(self.world.graph.get_neighbors(n))
            for n in self.world.graph.nodes
        }

    def observation_space(self, agent):
        """
        MASUP observation space: [Position of each agent, Latency of each node, Self-action finish flag,
                                The worst idleness from time T to now, obs_timer, role_information]
        """

        idleness_upper_bound = self.max_time_for_obs * self.world.max_phi * 1.1

        if self.role_ifm == "agent-index":
            low = np.array(
                [0, 0, 0] * self.world.num_agents
                + [0] * self.world.num_nodes
                + [0, 0, 0]
                + [0] * self.world.num_agents
            )
            high = np.array(
                [self.world.num_nodes, self.world.num_nodes, max(self.world.max_edge_length, self.world.waitT)] * self.world.num_agents
                + [idleness_upper_bound] * self.world.num_nodes
                + [1, idleness_upper_bound, self.T_time]
                + [1] * self.world.num_agents
            )
        elif self.role_ifm == "position":
            low = np.array(
                [0, 0, 0] * self.world.num_agents
                + [0] * self.world.num_nodes
                + [0, 0, 0, 0]
            )
            high = np.array(
                [self.world.num_nodes, self.world.num_nodes, max(self.world.max_edge_length, self.world.waitT)] * self.world.num_agents
                + [idleness_upper_bound] * self.world.num_nodes
                + [1, idleness_upper_bound, self.T_time, self.world.num_nodes]
            )
        elif self.role_ifm == "decision":
            N = self.world.num_agents
            low = np.array(
                [0, 0, 0] * N
                + [0] * self.world.num_nodes
                + [0, 0, 0, 0]
                + [0] * N
            )
            high = np.array(
                [self.world.num_nodes, self.world.num_nodes, max(self.world.max_edge_length, self.world.waitT)] * N
                + [idleness_upper_bound] * self.world.num_nodes
                + [1, idleness_upper_bound, self.T_time, self.world.num_nodes]
                + [1] * N
            )

        obs = Box(low=low, high=high, dtype=np.float32)

        return obs

    def action_space(self, agent):
        """
        MASUP action space: [Wait, Next target node (neighbors), no-op]
        """
        act = Discrete(self.world.max_neighbors+2)
        return act

    def step(self, actions: Dict[str, int]):
        deciding_agents = []
        if self.idi_scale > 0:
            for aid, status in self.world.agents.items():
                if status.state == AgentState.READY:
                    deciding_agents.append(aid)

        self._set_actions_with_stats(actions)

        # 1.5 IDI.
        idi_baselines = {}
        if self.idi_scale > 0:
            for aid in deciding_agents:
                baseline_val = self._simulate_step_cumulative_max_idleness(override_agent_id=aid)
                idi_baselines[aid] = baseline_val

        result = self._advance_with_T_time()

        self._decision_index_map = {}
        node_to_deciders = {}
        for aid in self.world.get_ready_agents():
            node = self.world.get_position(aid)
            node_to_deciders.setdefault(node, []).append(aid)
        for node, aids in node_to_deciders.items():
            aids.sort()
            # random.shuffle(aids).
            for idx, aid in enumerate(aids, start=0):
                self._decision_index_map[aid] = idx

        truncations = self._compute_truncations()
        is_truncated = any(truncations.values())

        if is_truncated:
            self._print_episode_summary()

        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        infos = self._build_info(result)

        for i, agent_str in enumerate(self.agents):
            if self.idi_scale > 0 and i in idi_baselines:
                # IDI = (Baseline_Worst - Real_Worst) * Scale.

                real_global_max = self.worst_idleness_fromT
                baseline_global_max = idi_baselines[i]

                diff = baseline_global_max - real_global_max

                rewards[agent_str] += diff * self.idi_scale
                infos[agent_str]['idi_reward'] = float(diff * self.idi_scale) if (self.idi_scale > 0 and i in idi_baselines) else 0.0

            if self.contribution_scale > 0:

                contrib = self._agent_idleness_reduction.get(i, 0.0)
                if contrib > 1e-9:
                    rewards[agent_str] += contrib * self.contribution_scale

        try:
            self._decision_index_map = {}
        except Exception:
            pass

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        if hasattr(self, 'wait_action_count') and self.world.metrics_tracker.history:
            total_wait = sum(self.wait_action_count.values())
            total_move = sum(self.move_action_count.values())
            total_decisions = total_wait + total_move
            wait_ratio = total_wait / total_decisions if total_decisions > 0 else 0.0
            self.world.metrics_tracker.current.wait_ratio = wait_ratio

        initial_positions = self.init_pos if self.init_pos else None

        # worst_idleness_fromT=0.0.
        if (
            hasattr(self, "worst_idleness_fromT")
            and len(self.world.metrics_tracker.history) > 1
        ):
            self.last_episode_wi_fromT = float(self.worst_idleness_fromT)
        else:
            self.last_episode_wi_fromT = None

        self.world.reset(initial_positions=initial_positions, seed=seed)
        self.agents = self.possible_agents[:]

        self.obs_timer = 0.0
        self.T_flag = False
        self._agent_idleness_reduction = {}
        self._decision_index_map = {}

        self.worst_idleness_fromT = 0.0
        self.last_time_interval = 0.0

        self._wi_fromT_history = [float(self.worst_idleness_fromT)]

        self.wait_action_count = {i: 0 for i in range(self.world.num_agents)}
        self.move_action_count = {i: 0 for i in range(self.world.num_agents)}
        self.total_decision_count = {i: 0 for i in range(self.world.num_agents)}

        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)

        return obs, infos

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        out = {
            "igi": m.igi,
            "agi": m.agi,
            "iwi": m.iwi,
            "wi": m.wi,
            "wait_ratio": m.wait_ratio,
        }
        if self.last_episode_wi_fromT is not None:
            out["wi_fromT"] = self.last_episode_wi_fromT
        return out

    def get_current_metrics(self) -> dict:
        m = self.world.metrics_tracker.current

        total_wait = sum(self.wait_action_count.values())
        total_move = sum(self.move_action_count.values())
        total_decisions = total_wait + total_move
        current_wait_ratio = total_wait / total_decisions if total_decisions > 0 else 0.0
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi, "wait_ratio": current_wait_ratio}

    def state(self) -> np.ndarray:
        agent_metrics: List[float] = []
        for agent_id in range(self.world.num_agents):
            agent_status = self.world.agents[agent_id]
            last_pos = float(agent_status.last_position)
            target = float(agent_status.target_node)
            time_left = float(agent_status.nominal_action_remaining)
            agent_metrics.extend([last_pos, target, time_left])

        weighted_idleness = self.world._weighted_arr.tolist()

        obs_list = agent_metrics + weighted_idleness + [float(self.worst_idleness_fromT), float(self.obs_timer)]
        return np.asarray(obs_list, dtype=np.float32)

    def _set_actions_with_stats(self, actions: Dict[str, int]):
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])

            if not self.world.is_ready(agent_id):
                continue

            self.total_decision_count[agent_id] += 1

            if action_idx == 0:

                self.world.set_wait_action(agent_id)
                self.wait_action_count[agent_id] += 1
            else:

                target = self._action_to_target(agent_id, action_idx)
                self._dispatch_move(agent_id, target)
                self.move_action_count[agent_id] += 1

    def _advance_with_T_time(self):
        dt = self.world._compute_next_event_time()

        if self.T_time > 0 and self.obs_timer < self.T_time:
            dt = min(dt, self.T_time - self.obs_timer)

        self._agent_idleness_reduction = {}

        for agent_id, status in self.world.agents.items():
            if status.state == AgentState.WAITING:

                pos = status.position
                mnt_contrib = float(self.world.graph.phi[pos] * dt)
                self._agent_idleness_reduction[agent_id] = mnt_contrib

        result = self.world.tick(dt)
        self.last_time_interval = result.dt

        for agent_id, reward in result.raw_rewards.items():
            if reward > 1e-9:
                self._agent_idleness_reduction[agent_id] =\
                    self._agent_idleness_reduction.get(agent_id, 0.0) + reward

        self._update_timers(result.dt)

        self._update_worst_idleness_with_T(result)

        return result

    def _update_worst_idleness_with_T(self, result):
        if self.obs_timer < self.T_time:
            self.worst_idleness_fromT = 0.0
        else:

            current_iwi = result.pre_arrival_weighted_iwi
            if current_iwi > self.worst_idleness_fromT:
                self.worst_idleness_fromT = current_iwi
        self._wi_fromT_history.append(float(self.worst_idleness_fromT))

    def _update_timers(self, dt: float):
        if self.T_time > 0:
            new_obs_timer = self.obs_timer + dt
            if new_obs_timer < self.T_time:

                self.obs_timer = new_obs_timer
            elif not self.T_flag:

                self.obs_timer = self.T_time
                self.T_flag = True
        else:
            # T_time = 0, obs_timer.
            self.obs_timer = 0.0

    def _compute_truncations(self) -> Dict[str, bool]:
        if self.truncate_by_time:
            is_truncated = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            is_truncated = self.world.step_count >= self.episode_len
        return {agent: is_truncated for agent in self.agents}

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        weighted = self.world._weighted_arr.astype(np.float32)

        Na = self.world.num_agents
        pos_buf = np.empty(3 * Na, dtype=np.float32)
        for aid in range(Na):
            ag = self.world.agents[aid]
            i = aid * 3
            pos_buf[i] = ag.last_position
            pos_buf[i + 1] = ag.target_node
            pos_buf[i + 2] = ag.nominal_action_remaining
        shared = np.concatenate((pos_buf, weighted))

        worst = float(self.worst_idleness_fromT)
        timer = float(self.obs_timer)
        obs: Dict[str, np.ndarray] = {}

        if self.role_ifm == "agent-index":
            for agent_id in range(Na):
                ag = self.world.agents[agent_id]
                ready_flag = 1.0 if ag.state == AgentState.READY else 0.0
                mid = np.array([ready_flag, worst, timer], dtype=np.float32)
                obs[f"agent_{agent_id}"] = np.concatenate(
                    (shared, mid, self._identity_rows[agent_id])
                )
        elif self.role_ifm == "position":
            for agent_id in range(Na):
                ag = self.world.agents[agent_id]
                ready_flag = 1.0 if ag.state == AgentState.READY else 0.0
                tail = np.array(
                    [ready_flag, worst, timer, float(ag.position)],
                    dtype=np.float32,
                )
                obs[f"agent_{agent_id}"] = np.concatenate((shared, tail))
        elif self.role_ifm == "decision":
            N = Na
            for agent_id in range(Na):
                ag = self.world.agents[agent_id]
                ready_flag = 1.0 if ag.state == AgentState.READY else 0.0
                decision_idx = (
                    int(self._decision_index_map.get(agent_id, 0))
                    if hasattr(self, "_decision_index_map")
                    else 0
                )
                one_hot = np.zeros(N, dtype=np.float32)
                one_hot[decision_idx] = 1.0
                mid = np.array(
                    [ready_flag, worst, timer, float(ag.position)],
                    dtype=np.float32,
                )
                obs[f"agent_{agent_id}"] = np.concatenate((shared, mid, one_hot))

        return obs

    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        infos = {}

        for agent_str in self.agents:
            agent_id = int(agent_str.split('_')[1])
            action_mask = self.get_action_mask(agent_str)
            active_mask = 1 if self.world.is_ready(agent_id) else 0

            infos[agent_str] = {"action_mask": action_mask, "active_mask": active_mask}

        return infos

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        rewards = {}

        for agent_str in self.agents:
            if self.truncate_by_time:
                reward = - self.worst_idleness_fromT * result.dt
            else:
                reward = - self.worst_idleness_fromT
            rewards[agent_str] = reward * self.reward_scale

        return rewards

    def _print_episode_summary(self):
        truncate_mode = "TIME" if self.truncate_by_time else "STEP"
        print(f"[TRUNCATE_MODE]: {truncate_mode}")
        print(f"[USED_STEP_NUM]: {self.world.step_count}")
        print(f"[TOTAL_TIME]: {self.world.current_time:.2f}")
        print(f"[MAX_LATENCY]: {self.worst_idleness_fromT:.4f}")

        total_wait = sum(self.wait_action_count.values())
        total_move = sum(self.move_action_count.values())
        total_decisions = total_wait + total_move
        wait_ratio = total_wait / total_decisions if total_decisions > 0 else 0
        print(f"[WAIT_ACTIONS]: {total_wait}/{total_decisions} ({wait_ratio:.2%})")

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        current_pos = self.world.get_position(agent_id)
        neighbors = self.world.graph.get_neighbors(current_pos)

        # action_idx: 0=wait, 1~N=neighbors, N+1=no-op.
        neighbor_idx = action_idx - 1
        if neighbor_idx < len(neighbors):
            return neighbors[neighbor_idx]
        else:
            raise ValueError(
                f"Invalid action: agent_id={agent_id}, action_idx={action_idx}, "
                f"available_neighbors={len(neighbors)}"
            )

    def get_action_mask(self, agent_str: str) -> np.ndarray:
        agent_id = int(agent_str.split('_')[1])
        mask = np.zeros(self.world.max_neighbors + 2, dtype=bool)

        if not self.world.is_ready(agent_id):

            mask[-1] = True
        else:

            current_pos = self.world.get_position(agent_id)
            num_neighbors = self._node_neighbor_count[current_pos]
            mask[:num_neighbors + 1] = True

        return mask

    def get_valid_actions(self, agent_str: str) -> List[int]:
        mask = self.get_action_mask(agent_str)
        return [int(x) for x in np.where(mask)[0]]

    def _simulate_step_cumulative_max_idleness(self, override_agent_id: Optional[int] = None) -> float:
        sim_time_left = {aid: self.world.agents[aid].action_remaining for aid in range(self.world.num_agents)}
        sim_target_node = {aid: self.world.agents[aid].target_node for aid in range(self.world.num_agents)}
        sim_agent_positions = {aid: self.world.agents[aid].position for aid in range(self.world.num_agents)}
        sim_agents_on_edge = {aid: self.world.agents[aid].state == AgentState.ON_EDGE for aid in range(self.world.num_agents)}
        sim_node_idleness = self.world.node_idleness.copy()

        if override_agent_id is not None:

            curr_pos = sim_agent_positions[override_agent_id]
            sim_target_node[override_agent_id] = curr_pos
            sim_time_left[override_agent_id] = float(self.world.waitT)
            sim_agents_on_edge[override_agent_id] = False

        if not sim_time_left:
            t_interval = float(self.world.waitT)
        else:
            t_interval = min(sim_time_left.values())

        if self.obs_timer < self.T_time:
            time_to_T = self.T_time - self.obs_timer
            if time_to_T < t_interval:
                t_interval = time_to_T

        finished_agents = []
        for aid, t in sim_time_left.items():
            new_t = t - t_interval
            sim_time_left[aid] = new_t
            if abs(new_t) < 1e-9:
                finished_agents.append(aid)

        nodes_have_agent = set()

        moving_agents_arrival = []
        for agent in finished_agents:
            current_pos = sim_agent_positions[agent]
            target = sim_target_node[agent]
            if current_pos == target:
                nodes_have_agent.add(current_pos)
            else:
                moving_agents_arrival.append(agent)

        for aid, is_on_edge in sim_agents_on_edge.items():
            if not is_on_edge:
                nodes_have_agent.add(sim_agent_positions[aid])

        current_instant_max = -1.0

        for node in self.world.graph.nodes:
            idle = sim_node_idleness.get(node, 0.0)
            if node not in nodes_have_agent:
                idle += t_interval

            phi = self.world.graph.phi[node]
            val = phi * idle
            if val > current_instant_max:
                current_instant_max = val

        sim_cumulative_max = self.worst_idleness_fromT

        target_timer = self.obs_timer + t_interval

        if target_timer >= self.T_time:
            if current_instant_max > sim_cumulative_max:
                sim_cumulative_max = current_instant_max

        return float(sim_cumulative_max)

    def get_heuristic_obs(self) -> Dict[str, Dict]:
        return self.world.get_heuristic_obs()

    def get_global_state_for_heuristic(self) -> Dict:
        return self.world.get_global_state_for_heuristic()

    def convert_heuristic_action(self, agent_str: str, neighbor_idx: int) -> int:
        return neighbor_idx + 1
