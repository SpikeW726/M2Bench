from typing import Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import EventDrivenEnv

class SUNSEnv(EventDrivenEnv):
    # step 必须返回 np.ndarray 类型的 obs，在网络类内部处理成需要的格式
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        # 确定的物理特征
        self.episode_len = config['episode_len']
        self.spl_mat = self.world.graph.get_shotest_path_len_mat()

        # 预计算权重矩阵 (N, N)：无边=0，有边=权重
        N = self.world.num_nodes
        ordered_nodes = sorted(self.world.graph.nodes)
        self._node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
        self._ordered_nodes = ordered_nodes          # 缓存以避免每步重排序
        self.weight_mat = np.zeros((N, N), dtype=np.float32)
        for i in self.world.graph.nodes:
            for j, w in self.world.graph.adj_list[i]:
                self.weight_mat[self._node_to_idx[i], self._node_to_idx[j]] = w
        self._weight_mat_flat = self.weight_mat.flatten()
        # phi 向量（按 _ordered_nodes 顺序）
        self._phi_vec = np.array(
            [float(self.world.graph.phi.get(n, 1.0)) for n in ordered_nodes],
            dtype=np.float32,
        )
        # 每步复用的节点特征 buffer (2N,)
        self._node_feat_buf = np.empty(2 * N, dtype=np.float32)

        # 需跟踪的信息
        self.agent_intentions: Dict[int, Optional[int]] = {}

        # Episode 截止模式开关
        self.truncate_by_time = kwargs.get('truncate_by_time', True)

        if self.truncate_by_time:
            self.max_time_for_obs = self.episode_len
        else:
            self.max_time_for_obs = config.get(
                'max_time_for_obs', self.episode_len * self.world.max_edge_length  
            )

        self.obs_size = 2 * N + N ** 2


    def observation_space(self, agent):
        """obs 布局: [node_features_flat(2N), weight_mat_flat(N^2)]"""
        N = self.world.num_nodes
        idleness_upper = self.max_time_for_obs * self.world.max_phi * 1.1

        # node_features: (INI, dist) * N
        node_low = [0.0, 0.0] * N
        node_high = [idleness_upper, self.world.max_path_length] * N
        # weight_mat: N^2 (无边=0, 有边=边权)
        wmat_low = [0.0] * (N * N)
        wmat_high = [float(self.world.max_edge_length)] * (N * N)

        low = np.array(node_low + wmat_low, dtype=np.float32)
        high = np.array(node_high + wmat_high, dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def action_space(self, agent):
        """
        SUNS action space: all nodes in the graph
        """        
        act = Discrete(self.world.num_nodes)
        return act

    def step(self, actions: Dict[str, int]):
        """step 前从动作中提取并记录各智能体意图"""
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])
            if self.world.is_ready(agent_id):
                # action_idx 直接对应节点 id（sorted 顺序）
                ordered_nodes = sorted(self.world.graph.nodes)
                if action_idx < len(ordered_nodes):
                    self.agent_intentions[agent_id] = ordered_nodes[action_idx]
                else:
                    # 越界兜底：留在当前位置
                    self.agent_intentions[agent_id] = self.world.get_position(agent_id)

        return super().step(actions)

    def reset(self, seed: Optional[int] = None):
        """重置时清空意图表"""
        self.agent_intentions = {}
        return super().reset(seed)

    def state(self) -> np.ndarray:
        """全局状态: 所有智能体位置 + 全节点空闲度 + 当前时间"""
        agent_metrics = []
        for agent_id in range(self.world.num_agents):
            agent = self.world.agents[agent_id]
            last_pos = float(agent.last_position)
            target = float(agent.target_node)
            time_left = float(agent.nominal_action_remaining)  # 名义剩余时间
            agent_metrics.extend([last_pos, target, time_left])

        idleness = [
            float(self.world.graph.phi.get(n, 1.0)) * float(self.world.node_idleness.get(n, 0.0))
            for n in self.world.graph.nodes
        ]

        return np.asarray(agent_metrics + idleness + [float(self.world.current_time)], dtype=np.float32)

    def _dispatch_move(self, agent_id: int, target_node: int):
        """全图动作空间：通过最短路径路由到任意目标节点。"""
        self.world.set_route_action(agent_id, target_node)

    # ==================== SUNS 内部方法 ====================
    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        """构建观测: [node_features_flat(2N), weight_mat_flat(N^2)]"""
        idle_vals = np.array(
            [float(self.world.node_idleness[n]) for n in self._ordered_nodes],
            dtype=np.float32,
        )
        weighted = self._phi_vec * idle_vals   # (N,) 向量化，无 Python 循环

        buf = self._node_feat_buf
        obs: Dict[str, np.ndarray] = {}
        for agent_id in range(self.world.num_agents):
            pos_idx = self._node_to_idx[self.world.agents[agent_id].position]
            # node_features: 偶数位 = weighted_idle, 奇数位 = spl_dist  (交替)
            buf[0::2] = weighted
            buf[1::2] = self.spl_mat[pos_idx]     # (N,) 行向量，O(N) 而非 O(N) 个 append
            obs[f"agent_{agent_id}"] = np.concatenate((buf, self._weight_mat_flat))

        return obs
    
    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        return {f"agent_{id}": reward for id, reward in result.raw_rewards.items()}

    def _build_info(self, result) -> Dict:
        """构建 info。到达最终目标的 agent 清除其 intention，再计算 mask。"""
        # 到达最终目标（READY 且无剩余路由）的 agent：intention 已完成
        for aid in list(self.agent_intentions.keys()):
            if (self.world.is_ready(aid)
                    and not self.world._routes.get(aid)):
                del self.agent_intentions[aid]

        infos = {}
        for agent_str in self.agents:
            agent_id = int(agent_str.split('_')[1])
            is_ready = self.world.is_ready(agent_id)
            infos[agent_str] = {
                "action_mask": self.get_action_mask(agent_str),
                "active_mask": 1 if is_ready else 0,
            }
        return infos

    def _compute_truncations(self) -> Dict[str, bool]:
        """计算截断状态"""
        if self.truncate_by_time:
            is_truncated = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            is_truncated = self.world.step_count >= self.episode_len
        return {agent: is_truncated for agent in self.agents}

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        """动作索引即节点索引（sorted 顺序）"""
        ordered_nodes = sorted(self.world.graph.nodes)
        if action_idx < len(ordered_nodes):
            return ordered_nodes[action_idx]
        raise ValueError(
            f"无效动作: agent_id={agent_id}, action_idx={action_idx}, "
            f"图节点数={len(ordered_nodes)}"
        )

    # ==================== 辅助方法 ====================

    def get_action_mask(self, agent_str: str) -> np.ndarray:
        """获取动作掩码：动作空间为全图所有节点，已有 intention 的节点置 False。"""
        ordered_nodes = sorted(self.world.graph.nodes)
        mask = np.ones(len(ordered_nodes), dtype=bool)

        agent_id = int(agent_str.split('_')[1])
        # 其他 agent 已有 intention 的节点屏蔽
        intended_nodes = {
            node for aid, node in self.agent_intentions.items()
            if aid != agent_id
        }
        for idx, node in enumerate(ordered_nodes):
            if node in intended_nodes:
                mask[idx] = False

        return mask

    def get_valid_actions(self, agent_str: str) -> List[int]:
        """返回有效动作索引列表"""
        mask = self.get_action_mask(agent_str)
        return [int(x) for x in np.where(mask)[0]]


    

    

