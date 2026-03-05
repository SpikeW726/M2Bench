"""SUNSGymEnv — gymnasium.Env 版本的 SUNS 环境，用于单智能体 RL 训练。

与 SUNSEnv(ParallelEnv) 共享同一套观测/奖励/截断逻辑，
但遵循标准 gymnasium.Env 接口，可直接搭配 OnPolicyCollector + A2C/PPO。

多智能体评估仍使用 SUNSEnv(ParallelEnv)。
"""

from typing import Dict, Optional
import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import PatrolWorld, TickResult


class SUNSGymEnv(gymnasium.Env):

    metadata = {"render_modes": []}

    def __init__(self, config: Dict, **kwargs):
        super().__init__()
        self.world = PatrolWorld(config)
        assert self.world.num_agents == 1, (
            "SUNSGymEnv 仅支持 1 个智能体，多智能体请用 SUNSEnv(ParallelEnv)"
        )

        self.episode_len = config["episode_len"]
        self.init_pos = config.get("init_positions", [])
        self.spl_mat = self.world.graph.get_shotest_path_len_mat()

        N = self.world.num_nodes
        ordered_nodes = sorted(self.world.graph.nodes)
        self._node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
        self._ordered_nodes = ordered_nodes

        self.weight_mat = np.zeros((N, N), dtype=np.float32)
        for i in self.world.graph.nodes:
            for j, w in self.world.graph.adj_list[i]:
                self.weight_mat[self._node_to_idx[i], self._node_to_idx[j]] = w
        self._weight_mat_flat = self.weight_mat.flatten()

        self.truncate_by_time = kwargs.get("truncate_by_time", True)
        if self.truncate_by_time:
            self.max_time_for_obs = self.episode_len
        else:
            self.max_time_for_obs = config.get(
                "max_time_for_obs", self.episode_len * self.world.max_edge_length
            )

        # gymnasium 标准 spaces (property，非方法)
        idleness_upper = self.max_time_for_obs * self.world.max_phi * 1.1
        node_low = [0.0, 0.0] * N
        node_high = [idleness_upper, self.world.max_path_length] * N
        wmat_low = [0.0] * (N * N)
        wmat_high = [float(self.world.max_edge_length)] * (N * N)

        self.observation_space = Box(
            low=np.array(node_low + wmat_low, dtype=np.float32),
            high=np.array(node_high + wmat_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Discrete(N)

    # ------------------------------------------------------------------
    #  gymnasium.Env 标准接口
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_pos = self.init_pos if self.init_pos else None
        self.world.reset(initial_positions=init_pos)
        obs = self._build_obs()
        info = {"action_mask": np.ones(self.world.num_nodes, dtype=bool)}
        return obs, info

    def step(self, action: int):
        target_node = self._ordered_nodes[action]
        self.world.set_route_action(0, target_node)

        result = self.world.tick_to_next_event()

        obs = self._build_obs()
        reward = result.raw_rewards.get(0, 0.0)
        terminated = False
        truncated = self._is_truncated()
        info = {
            "action_mask": np.ones(self.world.num_nodes, dtype=bool),
            "active_mask": 1 if self.world.is_ready(0) else 0,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    #  内部方法（与 SUNSEnv 保持一致）
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        ordered_nodes = self._ordered_nodes
        weighted_idleness = [
            self.world.graph.phi[n] * self.world.node_idleness[n]
            for n in ordered_nodes
        ]
        pos_idx = self._node_to_idx[self.world.agents[0].position]
        node_feat_flat = []
        for i, _n in enumerate(ordered_nodes):
            node_feat_flat.append(weighted_idleness[i])
            node_feat_flat.append(self.spl_mat[pos_idx, i])

        return np.concatenate([
            np.asarray(node_feat_flat, dtype=np.float32),
            self._weight_mat_flat,
        ])

    def _is_truncated(self) -> bool:
        if self.truncate_by_time:
            return self.world.current_time >= (self.episode_len - 1e-9)
        return self.world.step_count >= self.episode_len

    def get_episode_metrics(self) -> Optional[dict]:
        """供 vec_env.call_env_method('get_episode_metrics') 使用。"""
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}
