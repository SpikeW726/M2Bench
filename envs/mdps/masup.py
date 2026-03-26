from typing import Dict, List, Optional
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
        self.T_flag = False
        # 记录每步每个智能体消除的 Idleness 量 (idleness * phi)
        self._agent_idleness_reduction = {}

        # 确定的物理特征
        self.episode_len = config['episode_len']
        self.T_time = kwargs.get("T", 0.0)
        self.role_ifm = kwargs.get('role_imformation', "agent-index")

        # 上一局 episode 结束时的 worst_idleness_fromT（供 get_episode_metrics / WandB）
        self.last_episode_wi_fromT: Optional[float] = None
        # 与 metrics_tracker 时间序列对齐的 worst_idleness_fromT 曲线（供评估作图）
        self._wi_fromT_history: List[float] = []

        # reward trick
        self.contribution_scale = float(kwargs.get('contribution_scale', 0.0))
        self.idi_scale = float(kwargs.get('idi_scale', 0.0))

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
            self.obs_size = 4*self.world.num_agents + self.world.num_nodes + 3
        elif self.role_ifm == "position":
            self.obs_size = 3*self.world.num_agents + self.world.num_nodes + 4
        elif self.role_ifm == "decision":
            self.obs_size = 3*self.world.num_agents + self.world.num_nodes + 4 + self.world.num_agents


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

    # ==================== 核心方法:step 和 reset ====================
    
    def step(self, actions: Dict[str, int]):
        """
        由于需要考虑 T_time 截断, 使用自己的step推进逻辑, 而非直接调用 world.tick_to_next_event()
        """
        # 0. 在设置动作前先获取这一步哪些智能体需要决策,用于计算IDI
        deciding_agents = []
        if self.idi_scale > 0:
            for aid, status in self.world.agents.items():
                if status.state == AgentState.READY:
                    deciding_agents.append(aid)

        # 1. 设置动作并统计
        self._set_actions_with_stats(actions)

        # 1.5 IDI 基线推演 (Counterfactuals)
        idi_baselines = {}
        if self.idi_scale > 0:
            for aid in deciding_agents:
                baseline_val = self._simulate_step_cumulative_max_idleness(override_agent_id=aid)
                idi_baselines[aid] = baseline_val

        # 2. 推进物理世界（考虑 T_time 截断）
        result = self._advance_with_T_time()

        # 3. 为本次决策计算每个 agent 的决策序号（同节点内随机顺序）
        self._decision_index_map = {}
        node_to_deciders = {}
        for aid in self.world.get_ready_agents():
            node = self.world.get_position(aid)
            node_to_deciders.setdefault(node, []).append(aid)
        for node, aids in node_to_deciders.items():
            random.shuffle(aids)
            for idx, aid in enumerate(aids, start=0):
                self._decision_index_map[aid] = idx

        # 4. 计算截断条件
        truncations = self._compute_truncations()
        is_truncated = any(truncations.values())
        
        # 5. Episode 结束时打印统计信息
        if is_truncated:
            self._print_episode_summary()

        # 6. 构建返回值
        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        infos = self._build_info(result)

        # 7. 加上IDI-reward (Individual Diligence Intrinsic reward) 以及个体贡献奖励 (Event-based Reward)
        for i, agent_str in enumerate(self.agents):
            if self.idi_scale > 0 and i in idi_baselines:
                # IDI = (Baseline_Worst - Real_Worst) * Scale
                # Baseline_Worst: 如果我偷懒，由于少了我干活，WorstIdleness 可能会变大 (更坏)。
                # Real_Worst: 我实际干了活 (或者也偷懒了)，实际的 WorstIdleness。
                # 差值 > 0 表示我的行为让世界变得比“没有我”更好。
                
                # [修正] 使用【累积历史最大 Idleness】作为外部状态统计量
                # 反映了 Agent 行为对“最终优化目标” (Max Latency) 的因果影响。
                real_global_max = self.worst_idleness_fromT
                baseline_global_max = idi_baselines[i]
                
                diff = baseline_global_max - real_global_max
                
                # 直接累加，不做截断 (Allow negative contribution if agent made things worse than staying)
                rewards[agent_str] += diff * self.idi_scale
                infos[agent_str]['idi_reward'] = float(diff * self.idi_scale) if (self.idi_scale > 0 and i in idi_baselines) else 0.0

            if self.contribution_scale > 0:
                # 获取该 Agent 本步消除的 Idleness
                contrib = self._agent_idleness_reduction.get(i, 0.0)
                if contrib > 1e-9:
                    rewards[agent_str] += contrib * self.contribution_scale

         # 在 step 结束前统一清空本轮构建的决策序号映射，避免在策略仍在采样时被提前清空。
        try:
            self._decision_index_map = {}
        except Exception:
            pass

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        """
        MASUP 环境的 reset 方法
        
        初始化物理世界和 MASUP 特有的状态变量
        """
        # 1. 在重置前计算并保存 wait_ratio 到当前 metrics（用于 last_episode_metrics）
        if hasattr(self, 'wait_action_count') and self.world.metrics_tracker.history:
            total_wait = sum(self.wait_action_count.values())
            total_move = sum(self.move_action_count.values())
            total_decisions = total_wait + total_move
            wait_ratio = total_wait / total_decisions if total_decisions > 0 else 0.0
            self.world.metrics_tracker.current.wait_ratio = wait_ratio
        
        # 2. 确定初始位置
        if self.init_pos:
            initial_positions = self.init_pos
        else:
            initial_positions = random.sample(list(self.world.graph.nodes), self.world.num_agents)

        # 在 world.reset 固化 last_episode_metrics 之前，保存上一局终值（与 last_episode_metrics 同步）
        # 注意：world.reset 后 metrics_tracker 只有 1 条初始 record；
        # len > 1 才说明本轮真正跑过 tick，是合法的完整 episode。
        # 若只检查 history 非空，双重 reset（dims 推断 + collector.reset）时会把
        # worst_idleness_fromT=0.0 误存为上一局终值，导致 WandB 出现低于理论下限的值。
        if (
            hasattr(self, "worst_idleness_fromT")
            and len(self.world.metrics_tracker.history) > 1
        ):
            self.last_episode_wi_fromT = float(self.worst_idleness_fromT)
        else:
            self.last_episode_wi_fromT = None
        
        # 3. 重置物理世界
        self.world.reset(initial_positions=initial_positions)
        self.agents = self.possible_agents[:]
        
        # 3. 重置 MASUP 特有的状态变量
        self.obs_timer = 0.0
        self.T_flag = False
        self._agent_idleness_reduction = {}
        self._decision_index_map = {}

        # worst_idleness: 考虑 T_time 的最大加权 idleness（与 world.worst_idleness 不同！）
        self.worst_idleness_fromT = 0.0
        self.last_time_interval = 0.0
        # 与本轮 metrics_tracker 首行（reset 后初始时刻）对齐
        self._wi_fromT_history = [float(self.worst_idleness_fromT)]
        
        # 动作统计
        self.wait_action_count = {i: 0 for i in range(self.world.num_agents)}
        self.move_action_count = {i: 0 for i in range(self.world.num_agents)}
        self.total_decision_count = {i: 0 for i in range(self.world.num_agents)}
        
        # 4. 构建返回值
        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)
        
        return obs, infos
    
    def get_episode_metrics(self) -> Optional[dict]:
        """返回上一个完成 episode 的指标（轻量方法，用于 SubprocVectorEnv 避免 pickle 整个 world）"""
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
        """返回当前 episode 的实时指标"""
        m = self.world.metrics_tracker.current
        # 实时计算当前 wait_ratio
        total_wait = sum(self.wait_action_count.values())
        total_move = sum(self.move_action_count.values())
        total_decisions = total_wait + total_move
        current_wait_ratio = total_wait / total_decisions if total_decisions > 0 else 0.0
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi, "wait_ratio": current_wait_ratio}
    
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
                self._dispatch_move(agent_id, target)
                self.move_action_count[agent_id] += 1

    def _advance_with_T_time(self):
        """
        推进物理世界,考虑 T_time 截断
        
        关键逻辑:
        - 如果 obs_timer < T_time,则 dt 不能超过 T_time - obs_timer
        - T_time 之前,worst_idleness 保持为 0
        - T_time 之后,开始记录 worst_idleness
        - 维护 _agent_idleness_reduction: 包含完成动作的奖励 + 等待中的维护奖励
        """
        # 计算下一个事件的时间
        dt = self.world._compute_next_event_time()
        
        # T_time 截断:确保不会跨过 T_time
        if self.T_time > 0 and self.obs_timer < self.T_time:
            dt = min(dt, self.T_time - self.obs_timer)
        
        # [重置] 每步重置 idleness reduction 记录
        self._agent_idleness_reduction = {}
        
        # [维护奖励] 计算正在等待/驻守的智能体的维护贡献（阻止 idleness 增长）
        # 注意：这是在 tick 之前计算的，因为我们需要使用 dt
        for agent_id, status in self.world.agents.items():
            if status.state == AgentState.WAITING:
                # 正在等待的智能体阻止了其所在节点的 idleness 增长
                pos = status.position
                mnt_contrib = float(self.world.graph.phi[pos] * dt)
                self._agent_idleness_reduction[agent_id] = mnt_contrib
        
        # 推进物理世界
        result = self.world.tick(dt)
        self.last_time_interval = result.dt
        
        # [完成奖励] 将 tick 返回的 raw_rewards（完成动作的奖励）累加到 reduction 中
        for agent_id, reward in result.raw_rewards.items():
            if reward > 1e-9:
                self._agent_idleness_reduction[agent_id] = \
                    self._agent_idleness_reduction.get(agent_id, 0.0) + reward
        
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
        self._wi_fromT_history.append(float(self.worst_idleness_fromT))
    
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
            elif self.role_ifm == "decision":
                N = self.world.num_agents
                decision_idx = int(self._decision_index_map.get(agent_id, 0)) if hasattr(self, '_decision_index_map') else 0
                one_hot = [0.0] * N
                one_hot[decision_idx] = 1.0
                single_obs = (
                    agent_positions
                    + weighted_idleness
                    + [ready_flag, self.worst_idleness_fromT, self.obs_timer, agent_status.position]
                    + one_hot
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
            active_mask = 1 if self.world.is_ready(agent_id) else 0 

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

    def _simulate_step_cumulative_max_idleness(self, override_agent_id: Optional[int] = None) -> float:
        """
        [新增] 模拟物理世界推演一步，用于计算因果推断 IDI。
        仅在内存副本上计算，不改变真实环境状态。
        返回模拟步骤后的【累积历史最大 Idleness】 (Cumulative Max Idleness)。
        """
        # 1. 浅拷贝相关状态 (数值类型安全)
        sim_time_left = {aid: self.world.agents[aid].action_remaining for aid in range(self.world.num_agents)}
        sim_target_node = {aid: self.world.agents[aid].target_node for aid in range(self.world.num_agents)}
        sim_agent_positions = {aid: self.world.agents[aid].position for aid in range(self.world.num_agents)}
        sim_agents_on_edge = {aid: self.world.agents[aid].state == AgentState.ON_EDGE for aid in range(self.world.num_agents)}
        sim_node_idleness = self.world.node_idleness.copy()

        # 2. 应用反事实干预 (Force Stay)
        if override_agent_id is not None:
            # 假设该 Agent 选择了 Stay 动作 (target=current, time=deltaT)
            # 注意：调用此逻辑的前提是该 Agent 本步实际上“有机会”做决策
            curr_pos = sim_agent_positions[override_agent_id]
            sim_target_node[override_agent_id] = curr_pos
            sim_time_left[override_agent_id] = float(self.world.waitT)
            sim_agents_on_edge[override_agent_id] = False

        # 3. 模拟 _advance_physical_world 的核心逻辑
        
        # 计算推进间隔
        if not sim_time_left:
            t_interval = float(self.world.waitT)
        else:
            t_interval = min(sim_time_left.values())
        
        # 为了与真实环境一致，必须考虑 T_time 截断
        if self.obs_timer < self.T_time:
            time_to_T = self.T_time - self.obs_timer
            if time_to_T < t_interval:
                t_interval = time_to_T
                
        # 更新时间并找出完成的 Agent
        finished_agents = []
        for aid, t in sim_time_left.items():
            new_t = t - t_interval
            sim_time_left[aid] = new_t
            if abs(new_t) < 1e-9:
                finished_agents.append(aid)
                
        # 识别驻守/到达节点 (Masking Idleness Growth)
        nodes_have_agent = set()
        
        # a) 完成动作的 Agent (Waiting completion or Arrival)
        moving_agents_arrival = []
        for agent in finished_agents:
            current_pos = sim_agent_positions[agent]
            target = sim_target_node[agent]
            if current_pos == target:
                nodes_have_agent.add(current_pos)
            else:
                moving_agents_arrival.append(agent)
                
        # b) 正在驻守的 Agent (Not on edge)
        # 包含了 override=Stay 的情况
        for aid, is_on_edge in sim_agents_on_edge.items():
            if not is_on_edge:
                nodes_have_agent.add(sim_agent_positions[aid])
                
        # 更新 Idleness 并计算 Instant Weighted Idleness
        current_instant_max = -1.0
        
        for node in self.world.graph.nodes: # 确保遍历顺序与完整列表一致
            idle = sim_node_idleness.get(node, 0.0)
            if node not in nodes_have_agent:
                idle += t_interval
            
            # 计算加权值
            phi = self.world.graph.phi[node]
            val = phi * idle
            if val > current_instant_max:
                current_instant_max = val
        
        # 4. 计算反事实的【累积历史最大 Idleness】
        # 逻辑依据 _advance_physical_world 中的更新规则
        sim_cumulative_max = self.worst_idleness_fromT  # 起点是当前的累积最大值
        
        # 判断时间是否就在 T 时刻之后
        # 模拟的时间步后，obs_timer 必定增加。如果增加后 >= T_time，则 max_idleness 有效
        # 简单判定：如果当前环境已经 >= T_time，或者本步跨越了 T_time
        target_timer = self.obs_timer + t_interval
        
        if target_timer >= self.T_time:
            if current_instant_max > sim_cumulative_max:
                sim_cumulative_max = current_instant_max
        
        # 注意：如果 target_timer < self.T_time，则 max_idleness 保持为 0 (或初始值)
        # 且在真实 Step 中也是如此，所以 Diff 会是 0，这是合理的。
        
        return float(sim_cumulative_max)

    # ==================== 启发式采样接口（委托给 PatrolWorld）====================
    
    def get_heuristic_obs(self) -> Dict[str, Dict]:
        """返回启发式算法需要的观测格式"""
        return self.world.get_heuristic_obs()
    
    def get_global_state_for_heuristic(self) -> Dict:
        """返回启发式算法需要的全局状态"""
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