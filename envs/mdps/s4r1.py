from typing import Any, Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import FixedStepEnv

class S4R1Env(FixedStepEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        # 确定的物理特征
        self.episode_len = config['episode_len']

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

        self.obs_size = 2 * self.world.num_agents + self.world.max_neighbors

    def observation_space(self, agent):
        """
        S4R1 observation space: [self_start_node, self_target_node, 
                                INI of self_target_node's neighbors, 
                                start_node and target_node of all other agents]
        """
        # 给 idleness 上界增加 10% 余量,防止边界情况导致观测值越界
        idleness_upper_bound = self.max_time_for_obs * self.world.max_phi * 1.1

        max_node_id = self.world.num_nodes - 1

        low = np.array([-1] * self.obs_size, dtype=np.int32)
        high = np.array([max_node_id, max_node_id] + [idleness_upper_bound] * self.world.max_neighbors + [max_node_id, max_node_id] * (self.world.num_agents-1), dtype=np.int32)
        
        return Box(low=low, high=high, dtype=np.int32)

    def action_space(self, agent):
        return Discrete(self.world.max_neighbors + 1) # The last dimension means no-op

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
        obs = {}
        num_agents = self.world.num_agents
        all_agent_states = np.zeros((num_agents, 2), dtype=np.int32)
        
        for i in range(num_agents):
            agent = self.world.agents[i]
            v_s = agent.last_position
            v_t = agent.target_node
            
            if v_s == -1:
                v_s = v_t
            if v_t == -1:
                v_t = v_s
                
            all_agent_states[i] = [v_s, v_t]

        for agent_id in range(num_agents):
            # 获取当前智能体的数据
            curr_v_s, curr_v_t = all_agent_states[agent_id]
            
            # 获取其他智能体的数据 (S4: "other agents' source and target node positions" )
            other_agents_data = np.delete(all_agent_states, agent_id, axis=0).flatten()

            # 获取目标节点 (v_t) 的邻居及其 phi 加权空闲度
            target_neighbors = self.world.graph.get_neighbors(curr_v_t)
            neighbors_ini = [
                self.world.graph.phi.get(n, 1.0) * self.world.node_idleness[n]
                for n in target_neighbors
            ]
            
            # 组装观测向量
            single_obs = np.full(self.obs_size, -1, dtype=np.int32)
            
            # [0-1]: 自身的 v_s, v_t
            single_obs[0] = curr_v_s
            single_obs[1] = curr_v_t
            
            # [2 - 2+max_degree]: 目标节点邻居的空闲度
            num_neighbors = len(neighbors_ini)
            single_obs[2 : 2 + num_neighbors] = neighbors_ini
            
            # [2+max_degree - end]: 其他智能体的位置信息
            start_others_idx = 2 + self.world.max_neighbors
            single_obs[start_others_idx : start_others_idx + len(other_agents_data)] = other_agents_data
            
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