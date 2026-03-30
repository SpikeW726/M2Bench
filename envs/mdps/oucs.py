from typing import Any, Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import FixedStepEnv

class OUCSEnv(FixedStepEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        # 确定的物理特征
        self.episode_len = config['episode_len']
        self.max_visit = config['episode_len']

        # Episode 截止模式开关
        # truncate_by_time=True (默认): 使用总时间 all_timer >= episode_len 截止
        # truncate_by_time=False: 使用 step 数量 step_cnt >= episode_len 截止
        self.truncate_by_time = kwargs.get('truncate_by_time', True)
        self.penalised_factor = kwargs.get('penalised_factor', 1.5)

        if self.truncate_by_time:
            self.max_time_for_obs = self.episode_len
        else:
            self.max_time_for_obs = config.get(
                'max_time_for_obs', self.episode_len * self.world.max_edge_length  
            )

        self.obs_size = 3*self.world.num_agents + 2*self.world.max_neighbors

        # 需要追踪的物理特征
        self.nodes_visit_times = {n: 0 for n in self.world.graph.nodes}

    def observation_space(self, agent):
        """
        OUCS observation space: [Position of each agent,
                                Number of visits agents have made to neighbor nodes,
                                Significance of neighbor nodes (phi)]
        """
        num_nodes = self.world.num_nodes
        num_agents = self.world.num_agents
        
        low = np.array([-1] * self.obs_size, dtype=np.float32)
        high = np.array(
            [num_nodes, num_nodes, self.world.max_edge_length]*num_agents
            + [self.max_visit]*self.world.max_neighbors
            + [self.world.max_phi]*self.world.max_neighbors, dtype=np.float32
        )

        return Box(low=low, high=high, dtype=np.float32)

    def action_space(self, agent):
        return Discrete(self.world.max_neighbors + 1) # The last dimension means no-op

    def reset(self, seed: Optional[int] = None):
        """重置时清空访问次数表"""
        self.nodes_visit_times = {n: 0 for n in self.world.graph.nodes}
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


    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        # 在构建观测前更新到达节点的访问次数
        if result is not None:
            for agent_id, arrived_node in result.arrivals.items():
                self.nodes_visit_times[arrived_node] += 1

        obs = {}
        num_agents = self.world.num_agents
        M = self.world.max_neighbors

        # 预计算所有智能体的位置三元组 (start, target, remaining_time)
        all_positions = np.zeros(3 * num_agents, dtype=np.float32)
        for a in range(num_agents):
            agent = self.world.agents[a]
            if agent.state == AgentState.ON_EDGE:
                all_positions[3*a]     = agent.position          # 出发节点
                all_positions[3*a + 1] = agent.target_node       # 目标节点
                all_positions[3*a + 2] = agent.nominal_action_remaining  # 名义剩余时间
            else:
                all_positions[3*a]     = agent.position
                all_positions[3*a + 1] = agent.position          # READY: 目标=当前位置
                all_positions[3*a + 2] = 0.0

        for agent_id in range(num_agents):
            # ON_EDGE 智能体返回全 -1 观测
            if self.world.agents[agent_id].state == AgentState.ON_EDGE:
                obs[f"agent_{agent_id}"] = np.full(self.obs_size, -1.0, dtype=np.float32)
                continue

            current_pos = self.world.agents[agent_id].position
            neighbors = self.world.graph.get_neighbors(current_pos)

            single_obs = np.full(self.obs_size, -1.0, dtype=np.float32)

            # [0, 3*N): 所有智能体的位置信息
            single_obs[:3 * num_agents] = all_positions

            # [3*N, 3*N+M): 邻居节点的访问次数
            pos_end = 3 * num_agents
            for j, n in enumerate(neighbors):
                single_obs[pos_end + j] = float(self.nodes_visit_times[n])

            # [3*N+M, 3*N+2*M): 邻居节点的优先级 (phi)
            phi_start = pos_end + M
            for k, n in enumerate(neighbors):
                single_obs[phi_start + k] = float(self.world.graph.phi.get(n, 1.0))

            obs[f"agent_{agent_id}"] = single_obs

        return obs

    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        """
        构建所有智能体的 info
        """
        infos = {}

        for agent_str in self.agents:
            agent_id = int(agent_str.split('_')[1])
            action_mask = self.get_action_mask(agent_str)
            active_mask = 1 if self.world.is_ready(agent_id) else 0 

            infos[agent_str] = {"action_mask": action_mask, "active_mask": active_mask}
        
        return infos

    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        igi = max(result.pre_arrival_igi, 1e-8)
        rewards = {}
        for agent_id in range(self.world.num_agents):
            agent_str = f"agent_{agent_id}"
            if agent_id in result.arrivals:
                arrived_node = result.arrivals[agent_id]
                phi = self.world.graph.phi.get(arrived_node, 1.0)
                ini = result.raw_rewards[agent_id] / phi if phi > 0 else 0.0
                rewards[agent_str] = (ini ** self.penalised_factor) / igi
                rewards[agent_str] *= phi
            else:
                rewards[agent_str] = 0.0
        return rewards

    def _compute_truncations(self) -> Dict[str, bool]:
        """计算截断状态"""
        if self.truncate_by_time:
            is_truncated = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            is_truncated = self.world.step_count >= self.episode_len
        return {agent: is_truncated for agent in self.agents}

    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        """将动作索引转换为目标节点"""
        current_pos = self.world.get_position(agent_id)
        neighbors = self.world.graph.get_neighbors(current_pos)
        
        # action_idx: 0~N-1=neighbors, N=no-op
        neighbor_idx = action_idx
        if neighbor_idx < len(neighbors):
            return neighbors[neighbor_idx]
        else: # 前面已经跳过了还未完成上一动作的智能体,这里不会传入no-op动作的index
            raise ValueError(
                f"无效动作: agent_id={agent_id}, action_idx={action_idx}, "
                f"可用邻居数量={len(neighbors)}"
            )    
    
    # ==================== 辅助方法 ====================
    
    def get_action_mask(self, agent_str: str) -> np.ndarray:
        """获取动作掩码"""
        agent_id = int(agent_str.split('_')[1])
        mask = np.zeros(self.world.max_neighbors+1, dtype=bool)

        if not self.world.is_ready(agent_id):
            # 正在执行动作,只能选 no-op
            mask[-1] = True
        else:
            # 可以决策: 邻居节点
            current_pos = self.world.get_position(agent_id)
            neighbors = self.world.graph.get_neighbors(current_pos)
            mask[:len(neighbors)] = True
                      
        return mask

    def get_valid_actions(self, agent_str: str) -> List[int]:
        """返回有效动作索引列表"""
        mask = self.get_action_mask(agent_str)
        return [int(x) for x in np.where(mask)[0]]