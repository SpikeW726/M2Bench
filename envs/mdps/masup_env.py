from typing import Dict, List, Optional, Tuple
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import EventDrivenEnv

class MASUPEnv(EventDrivenEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        # 需跟踪的物理状态
        self.obs_timer = 0    # 供观测使用,只记录到T
        # self.step_interval = 0 # 每次step物理环境会返回
        self.T_flag = False

        # 确定的物理特征
        self.episode_len = config['episode_len']
        self.T_time = kwargs.get("T", 0.0)
        self.role_ifm = kwargs.get('role_imformation', "agent-index")
        self.init_pos = config.get('init_positions', [])

        # 训练trick
        self.reward_scale = float(kwargs.get('reward_scale', 1.0))
        
        # Episode 截止模式开关
        # truncate_by_time=True (默认): 使用总时间 all_timer >= episode_len 截止
        # truncate_by_time=False: 使用 step 数量 step_cnt >= episode_len 截止
        self.truncate_by_time = kwargs.get('truncate_by_time', True)
        
        # 观测空间的 worst_idleness 上界
        # 当 truncate_by_time=False 时,实际运行时间可能远超 episode_len,
        # 需要使用 max_time_for_obs 参数来定义观测空间的 worst_idleness 上界
        # 如果不设置,按 step 截止时默认使用 episode_len * max_edge 作为估算值
        if self.truncate_by_time:
            self.max_time_for_obs = self.episode_len
        else:
            self.max_time_for_obs = config.get(
                'max_time_for_obs', self.episode_len * self.world.max_edge_length  
            )

        if self.role_ifm == "agent-index":
            self.obs_size = 4*self.world.num_agents + self.world.num_nodes + 2
        elif self.role_ifm == "position":
            self.obs_size = 3*self.world.num_agents + self.world.num_nodes + 3


    def observation_space(self, agent):
        """
        MASUP observation space: [Position of each agent, Latency of each node, Self-action finish flag,
                                The worst idleness from time T to now, obs_timer, role_information]
        """
        # 给 idleness 上界增加 10% 余量,防止边界情况导致观测值越界
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
        obs = Box(low=low, high=high, dtype=np.float32)

        return obs

    def action_space(self, agent):
        """
        MASUP action space: [Wait, Next target node (neighbors), no-op]
        """        
        act = Discrete(self.world.max_neighbors+2)
        return act

    # ==================== 核心方法:step 和 reset ====================
    
    def step(self, actions: Dict[str, int]):
        """
        使用自己的物理推进逻辑（考虑 T_time 截断）,而非直接调用 world.tick_to_next_event()
        """
        # 1. 设置动作并统计
        self._set_actions_with_stats(actions)

        # 2. 推进物理世界（考虑 T_time 截断）
        result = self._advance_with_T_time()

        # 4. 计算截断条件
        truncations = self._compute_truncations()
        is_truncated = any(truncations.values())
        
        # 5. Episode 结束时打印统计信息
        if is_truncated:
            self._print_episode_summary()

        # 6. 构建返回值（留给子类实现的抽象方法）
        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        infos = self._build_info(result)
        
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        """
        MASUP 环境的 reset 方法
        
        初始化物理世界和 MASUP 特有的状态变量
        """
        # 1. 确定初始位置
        if self.init_pos:
            initial_positions = self.init_pos
        else:
            initial_positions = random.sample(list(self.world.graph.nodes), self.world.num_agents)
        
        # 2. 重置物理世界
        self.world.reset(initial_positions=initial_positions)
        self.agents = self.possible_agents[:]
        
        # 3. 重置 MASUP 特有的状态变量
        self.obs_timer = 0.0
        self.T_flag = False
        
        # worst_idleness: 考虑 T_time 的最大加权 idleness（与 world.worst_idleness 不同！）
        self.worst_idleness_fromT = 0.0
        self.last_time_interval = 0.0
        
        # 动作统计
        self.wait_action_count = {i: 0 for i in range(self.world.num_agents)}
        self.move_action_count = {i: 0 for i in range(self.world.num_agents)}
        self.total_decision_count = {i: 0 for i in range(self.world.num_agents)}
        
        # 4. 构建返回值
        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)
        
        return obs, infos
    
    def state(self) -> np.ndarray:
        """为MAPPO/QMIX等算法提供全局观测"""
        agent_metrics: List[float] = []
        for agent_id in range(self.world.num_agents):
            agent_status = self.world.agents[agent_id]
            last_pos = float(agent_status.last_position)
            target = float(agent_status.target_node)
            time_left = float(agent_status.action_remaining)
            agent_metrics.extend([last_pos, target, time_left])

        weighted_idleness = [
            float(self.world.graph.phi.get(n, 1.0)) * float(self.world.node_idleness.get(n, 0.0))
            for n in self.world.graph.nodes
        ]

        obs_list = agent_metrics + weighted_idleness + [float(self.worst_idleness_fromT), float(self.obs_timer)]
        return np.asarray(obs_list, dtype=np.float32)

    # ==================== MASUP 内部方法 ====================
    
    def _set_actions_with_stats(self, actions: Dict[str, int]):
        """设置动作并统计等待/移动次数"""
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])
            
            if not self.world.is_ready(agent_id):
                continue  # 跳过正在执行动作的智能体
            
            self.total_decision_count[agent_id] += 1
            
            if action_idx == 0:
                # 等待动作
                self.world.set_wait_action(agent_id)
                self.wait_action_count[agent_id] += 1
            else:
                # 移动动作
                target = self._action_to_target(agent_id, action_idx)
                self.world.set_move_action(agent_id, target)
                self.move_action_count[agent_id] += 1

    def _advance_with_T_time(self):
        """
        推进物理世界,考虑 T_time 截断
        
        关键逻辑:
        - 如果 obs_timer < T_time,则 dt 不能超过 T_time - obs_timer
        - T_time 之前,worst_idleness 保持为 0
        - T_time 之后,开始记录 worst_idleness
        """
        # 计算下一个事件的时间
        dt = self.world._compute_next_event_time()
        
        # T_time 截断:确保不会跨过 T_time
        if self.T_time > 0 and self.obs_timer < self.T_time:
            dt = min(dt, self.T_time - self.obs_timer)
        
        # 推进物理世界
        result = self.world.tick(dt)
        self.last_time_interval = result.dt
        
        # 更新 worst_idleness（考虑 T_time）
        self._update_worst_idleness_with_T()
        
        # 更新计时器
        self._update_timers(result.dt)
        
        return result
    
    def _update_worst_idleness_with_T(self):
        """
        更新 worst_idleness,考虑 T_time
        
        关键:T_time 之前,worst_idleness = 0;T_time 之后才开始记录
        """
        if self.obs_timer < self.T_time:
            self.worst_idleness_fromT = 0.0
        else:
            # 使用 metrics_tracker 中的 iwi (Instantaneous Worst Idleness)
            current_iwi = self.world.current_metrics.iwi
            if current_iwi > self.worst_idleness_fromT:
                self.worst_idleness_fromT = current_iwi
    
    def _update_timers(self, dt: float):
        """更新 obs_timer"""
        if self.T_time > 0:
            new_obs_timer = self.obs_timer + dt
            if new_obs_timer < self.T_time:
                # 未到 T_time
                self.obs_timer = new_obs_timer
            elif not self.T_flag:
                # 刚到达 T_time
                self.obs_timer = self.T_time
                self.T_flag = True
        else:
            # T_time = 0, obs_timer恒为0
            self.obs_timer = 0.0


    def _compute_truncations(self) -> Dict[str, bool]:
        """计算截断状态"""
        if self.truncate_by_time:
            is_truncated = self.world.current_time >= (self.episode_len - 1e-9)
        else:
            is_truncated = self.world.step_count >= self.episode_len
        return {agent: is_truncated for agent in self.agents}

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        """构建所有智能体的观测"""
        obs = {}
        weighted_idleness = [self.world.graph.phi[node] * self.world.node_idleness[node] for node in self.world.graph.nodes]
        agent_positions = [
            val
            for aid in range(self.world.num_agents)
            for val in [
                self.world.agents[aid].last_position,
                self.world.agents[aid].target_node,
                self.world.agents[aid].action_remaining
            ]
        ]

        for agent_id in range(self.world.num_agents):
            agent_status = self.world.agents[agent_id]
            ready_flag = 1.0 if agent_status.state == AgentState.READY else 0.0
            agent_id_one_hot = [1.0 if idx == agent_id else 0.0 for idx in range(self.world.num_agents)]

            if self.role_ifm == "agent-index":
                single_obs = (
                    agent_positions
                    + weighted_idleness
                    + [ready_flag, self.worst_idleness_fromT, self.obs_timer]
                    + agent_id_one_hot
                )
            elif self.role_ifm == "position":
                single_obs = (
                    agent_positions
                    + weighted_idleness
                    + [ready_flag, self.worst_idleness_fromT, self.obs_timer, agent_status.position]
                )

            single_obs = np.asarray(single_obs, dtype=np.float32)
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
            active_mask = 1 if self.world.agents[agent_id].state == AgentState.READY else 0 

            infos[agent_str] = {"action_mask": action_mask, "active_mask": active_mask}
        
        return infos
    
    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        """计算所有智能体的奖励"""
        rewards = {}

        for agent_str in self.agents:
            if self.truncate_by_time:
                reward = - self.worst_idleness_fromT * result.dt
            else:
                reward = - self.worst_idleness_fromT
            rewards[agent_str] = reward * self.reward_scale

        return rewards


    def _print_episode_summary(self):
        """Episode 结束时打印统计信息"""
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
        """将动作索引转换为目标节点"""
        current_pos = self.world.get_position(agent_id)
        neighbors = self.world.graph.get_neighbors(current_pos)
        
        # action_idx: 0=wait, 1~N=neighbors, N+1=no-op
        neighbor_idx = action_idx - 1
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
        mask = np.zeros(self.world.max_neighbors + 2, dtype=bool)

        if not self.world.is_ready(agent_id):
            # 正在执行动作,只能选 no-op
            mask[-1] = True
        else:
            # 可以决策:等待 + 邻居节点
            current_pos = self.world.get_position(agent_id)
            neighbors = self.world.graph.get_neighbors(current_pos)
            mask[:len(neighbors) + 1] = True
                      
        return mask

    def get_valid_actions(self, agent_str: str) -> List[int]:
        """返回有效动作索引列表"""
        mask = self.get_action_mask(agent_str)
        return [int(x) for x in np.where(mask)[0]]

    # ==================== 启发式采样接口（委托给 PatrolWorld）====================
    
    def get_heuristic_obs(self) -> Dict[str, Dict]:
        """返回启发式算法需要的观测格式（委托给 PatrolWorld）"""
        return self.world.get_heuristic_obs()
    
    def get_global_state_for_heuristic(self) -> Dict:
        """返回启发式算法需要的全局状态（委托给 PatrolWorld）"""
        return self.world.get_global_state_for_heuristic()
    
    def convert_heuristic_action(self, agent_str: str, neighbor_idx: int) -> int:
        """
        将启发式动作(邻居索引)转换为 MASUPEnv 动作索引
        
        Args:
            agent_str: 智能体标识符,如 "agent_0"
            neighbor_idx: 邻居列表中的索引 (0-based)
        
        Returns:
            MASUPEnv 动作索引: 0=wait, 1~N=neighbors, N+1=no-op
            邻居索引 0 -> 动作索引 1
        """
        # MASUPEnv 动作空间: 0=wait, 1~N=neighbors, N+1=no-op
        # 启发式返回的是邻居索引 (0-based)
        return neighbor_idx + 1