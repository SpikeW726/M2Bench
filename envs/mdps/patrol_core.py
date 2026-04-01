#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum

import numpy as np

from utils.graph_utils import Graph
from utils.log_utils import EpisodeMetricsTracker, IdlenessMetrics

class AgentState(Enum):
    READY = "ready"          # ready to make decision
    WAITING = "waiting"      # waiting on node
    ON_EDGE = "on_edge"      # moving on edge

@dataclass
class AgentStatus:
    """Status of an agent"""
    position: int = -1                     # Current node (source node if on an edge)
    last_position: int = -1             # Last node
    state: AgentState = AgentState.READY
    target_node: int = -1            
    action_remaining: float = 0.0       # 实际剩余时间（物理到达判断用，含 jitter）
    nominal_action_remaining: float = 0.0  # 名义剩余时间（obs 暴露给 actor，始终是理想预测）
    planned_edge_duration: float = 0.0  # 本次 ON_EDGE 的实际计划总时长（动画进度分母）
    speed: float = 1.0
    

@dataclass
class TickResult:
    """Results that tick() return"""
    dt: float                                               # Actual elapsed time
    arrivals: Dict[int, int] = field(default_factory=dict)  # {agent_id: arrived_node}
    wait_completed: Set[int] = field(default_factory=set)   # Agents that finish waiting
    raw_rewards: Dict[int, float] = field(default_factory=dict)  # arrived_node_idleness*phi or waitT*phi or 0.0
    pre_arrival_igi: float = 0.0                            # 到达清零前的 IGI（所有节点空闲度均值）
    pre_arrival_weighted_iwi: float = 0.0                   # 到达清零前的加权 IWI（max phi*idleness），供 wi_fromT 正确峰值捕获
    
    @property
    def ready_agents(self) -> Set[int]:
        """返回所有可以做新决策的智能体"""
        return set(self.arrivals.keys()) | self.wait_completed

class PatrolWorld:
    """
    巡逻物理世界模拟器
    
    职责：
    - 管理图结构、智能体位置、节点空闲度
    - 提供时间推进逻辑
    - 不涉及 MDP 定义（观测空间、动作空间、奖励）
    
    支持的使用模式：
    - 固定时间步：tick(dt=1.0)
    - 事件驱动：tick_to_next_event()
    """
    
    def __init__(self, cfg: Dict):
        graph_path = cfg["graph_path"]
        self.graph = Graph(graph_path)

        # 图特征
        self.max_neighbors = self.graph.get_max_degree()
        self.max_edge_length = self.graph.get_max_edge_length()
        self.max_path_length = self.graph.max_shortest_path_len
        self.max_phi = self.graph.get_max_phi()
        self.num_nodes = len(self.graph.nodes)
        self.num_edges = self.graph.get_num_edges(True) # 有向图

        # 智能体状态
        self.num_agents = cfg["num_agents"]
        self.speeds = cfg.get("speeds", [1.0] * self.num_agents)
        self.agents: Dict[int, AgentStatus] = {}

        # 时间变量
        self.node_idleness: Dict[int, float] = {n: 0.0 for n in self.graph.nodes}
        self.current_time: float = 0.0
        self.worst_idleness: float = 0.0
        self.waitT = cfg.get("deltaT", 1.0)  # 一次等待动作的持续时间

        # ---- numpy 加速结构（优化 1+3）----
        # 固定节点顺序：与 list(graph.nodes) 一致，masup._obs_node_order 复用
        self._node_order: List[int] = list(self.graph.nodes)
        self._node_idx: Dict[int, int] = {n: i for i, n in enumerate(self._node_order)}
        # phi 向量（float64，与 _idleness_arr 同精度，乘积传给 metrics_tracker）
        self._phi_arr: np.ndarray = np.array(
            [float(self.graph.phi.get(n, 1.0)) for n in self._node_order],
            dtype=np.float64,
        )
        # 节点空闲度数组（主存储，tick 后同步到 node_idleness dict）
        self._idleness_arr: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)
        # phi * idleness 到达后（缓冲区，每 tick 计算一次，传给 masup._build_obs 和奖励）
        self._weighted_arr: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)
        # phi * idleness 到达前（到达清零前快照，传给 metrics_tracker，确保 IWI 捕获真实峰值）
        self._pre_arrival_weighted_arr: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)
        # 节点占据计数（>=1 表示有智能体在此节点）
        self._occupied_count: np.ndarray = np.zeros(self.num_nodes, dtype=np.int32)

        # Episode 指标追踪器
        self.metrics_tracker = EpisodeMetricsTracker(
            training_mode=cfg.get("training_mode", False)
        )
        self.last_episode_metrics: Optional[IdlenessMetrics] = None  # 上一个 episode 终止时的指标
        self.step_count: int = 0

        # 记录哪些节点当前有智能体"占据"（保留用于向后兼容）
        self._occupied_nodes: Set[int] = set()

        # 路由模式（全图动作空间）: 存储多跳路径剩余节点 & 中间累积奖励
        self._routes: Dict[int, List[int]] = {}
        self._acc_rewards: Dict[int, float] = {}

        # 边上运动时间扰动（双轨设计）
        self._jitter_enabled = bool(cfg.get("edge_time_jitter", False))
        self._jitter_frac    = float(cfg.get("edge_time_jitter_frac", 0.1))
        import random as _random_mod
        _seed = cfg.get("edge_time_jitter_seed", None)
        self._jitter_rng = _random_mod.Random(_seed)  # 独立实例，不干扰全局 random
    
    def reset(self, initial_positions: Optional[List[int]] = None) -> None:
        """重置物理世界"""
        # 保存上一个 episode 的终止指标（has_data 兼容 training_mode 下 history 为空的情况）
        if self.metrics_tracker.has_data:
            self.last_episode_metrics = self.metrics_tracker.current

        self.current_time = 0.0
        self.step_count = 0
        self.worst_idleness = 0.0

        # 重置指标追踪器
        self.metrics_tracker.reset()

        # 初始化智能体位置
        if initial_positions is None:
            import random
            initial_positions = random.sample(list(self.graph.nodes), self.num_agents)

        self._occupied_nodes.clear()
        self._routes.clear()
        self._acc_rewards.clear()

        # 重置 numpy 加速结构
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

        # 重置 dict（所有节点空闲度归零，与 _idleness_arr 保持一致）
        self.node_idleness = {n: 0.0 for n in self._node_order}

        # 记录初始状态（空闲度全 0，_weighted_arr 已清零，直接传入）
        self.metrics_tracker.record(self._weighted_arr, self.step_count, self.current_time)

    # Most important funciton
    def tick(self, dt: float) -> TickResult:
        """
        推进指定的时间量
        
        这是最底层的时间推进方法，其他方法都基于此实现。
        
        Args:
            dt: 要推进的时间量
            
        Returns:
            TickResult: 包含到达事件等信息
        """
        raw_rewards = {a:0.0 for a in range(self.num_agents)}
        if dt < 0:
            return TickResult(dt=0.0, raw_rewards=raw_rewards)

        result = TickResult(dt=dt, raw_rewards=raw_rewards)

        # 1. 向量化更新节点空闲度（非占据节点 += dt）
        free_mask = self._occupied_count == 0
        self._idleness_arr[free_mask] += dt

        # 1.5 快照：到达清零前的 IGI、worst_idleness（未加权）及加权 IWI（phi*idleness 峰值）
        # 必须在步骤 2（到达节点清零）之前计算，确保 IWI/wi_fromT 能捕获真实峰值
        result.pre_arrival_igi = float(self._idleness_arr.mean())
        current_worst_idleness = float(self._idleness_arr.max())
        self.worst_idleness = max(self.worst_idleness, current_worst_idleness)
        # 写入独立缓冲区 _pre_arrival_weighted_arr，步骤 5 将其传给 metrics_tracker
        np.multiply(self._phi_arr, self._idleness_arr, out=self._pre_arrival_weighted_arr)
        result.pre_arrival_weighted_iwi = float(self._pre_arrival_weighted_arr.max())

        # 2. 更新智能体状态
        for agent_id, status in self.agents.items():
            if status.state in (AgentState.ON_EDGE, AgentState.WAITING):
                status.action_remaining         -= dt
                status.nominal_action_remaining -= dt  # 名义时间与实际时间同步递减

                # 检查是否完成（以实际物理时间为准）
                if status.action_remaining <= 1e-9:
                    status.action_remaining         = 0.0
                    status.nominal_action_remaining = 0.0  # 到达后两者均归零

                    if status.state == AgentState.ON_EDGE:
                        # 到达目标节点
                        arrived_node = status.target_node
                        status.last_position = status.position
                        status.position = arrived_node
                        status.state = AgentState.READY
                        status.target_node = -1
                        status.planned_edge_duration = 0.0  # 清除动画分母缓存

                        # 更新占据状态（numpy + set 保持同步）
                        ai = self._node_idx[arrived_node]
                        self._occupied_nodes.add(arrived_node)
                        self._occupied_count[ai] += 1

                        # 清零到达节点的空闲度，计算 arrival_reward
                        arrival_reward = float(self._idleness_arr[ai]) * float(self._phi_arr[ai])
                        self._idleness_arr[ai] = 0.0

                        # 路由续航：中间节点累积奖励并自动跳下一跳
                        if agent_id in self._routes and self._routes[agent_id]:
                            self._acc_rewards[agent_id] = (
                                self._acc_rewards.get(agent_id, 0.0) + arrival_reward
                            )
                            next_hop = self._routes[agent_id].pop(0)
                            self.set_move_action(agent_id, next_hop)
                            # 中间节点不暴露为 arrival，reward 暂存
                            result.raw_rewards[agent_id] = 0.0
                        else:
                            # 最终目标到达：合并累积奖励
                            acc = self._acc_rewards.pop(agent_id, 0.0)
                            result.raw_rewards[agent_id] = acc + arrival_reward
                            result.arrivals[agent_id] = arrived_node

                    elif status.state == AgentState.WAITING:
                        # 等待完成
                        status.state = AgentState.READY
                        waiting_node = self.agents[agent_id].position
                        result.raw_rewards[agent_id] = self.waitT * self.graph.phi[waiting_node]
                        result.wait_completed.add(agent_id)

        # 3. 更新时间
        self.current_time += dt

        # 4. 同步 node_idleness dict（供外部 MDP 的 _build_obs / state() 等访问）
        #    使用 dict.update(zip(...)) 替代 Python 逐元素赋值，利用 C 层批量写入
        self.node_idleness.update(zip(self._node_order, self._idleness_arr.tolist()))

        # 5. 计算到达后的 phi 加权空闲度（供观测/奖励使用），并用到达前快照记录指标
        # _pre_arrival_weighted_arr：步骤 1.5 已计算，携带到达清零前的真实 IWI 峰值
        # _weighted_arr：到达后重新计算，供 masup._build_obs / _compute_rewards 等读取
        self.step_count += 1
        np.multiply(self._phi_arr, self._idleness_arr, out=self._weighted_arr)
        self.metrics_tracker.record(self._pre_arrival_weighted_arr, self.step_count, self.current_time)

        return result

    def tick_to_next_event(self) -> TickResult:
        """
        推进到最近的事件发生时刻
        
        事件包括：
        - 某个智能体到达目标节点
        - 某个智能体等待完成
        
        Returns:
            TickResult: 如果没有待处理事件，dt=0，raw_rewards 全零
        """
        dt = self._compute_next_event_time()
        if dt < 0:
            raw_rewards = {a: 0.0 for a in range(self.num_agents)}
            return TickResult(dt=0.0, raw_rewards=raw_rewards)
        return self.tick(dt)

    def _compute_next_event_time(self) -> float:
        """计算下一个事件发生的时间"""
        min_time = float('inf')
        
        for status in self.agents.values():
            if status.state in (AgentState.ON_EDGE, AgentState.WAITING, AgentState.READY):
                if status.action_remaining >= 0:
                    min_time = min(min_time, status.action_remaining)
        
        return min_time if min_time != float('inf') else -1.0

    def set_move_action(self, agent_id: int, target_node: int) -> bool:
        """
        设置智能体移动动作
        
        Args:
            agent_id: 智能体ID
            target_node: 目标节点（必须是当前位置的邻居）
            
        Returns:
            bool: 是否设置成功
        """
        status = self.agents[agent_id]
        
        # 检查是否可以决策
        if status.state != AgentState.READY:
            return False
        
        current_pos = status.position
        
        # 检查目标是否是邻居
        if target_node not in self.graph.get_neighbors(current_pos):
            return False
        
        # 设置移动状态
        status.state = AgentState.ON_EDGE
        status.target_node = target_node
        edge_length = self.graph.get_edge_length(current_pos, target_node)
        status.last_position = current_pos

        # 双轨时间：名义时间（actor 可观测）与实际时间（物理到达，含 jitter）
        T_nom = float(edge_length) / max(status.speed, 1e-6)
        if self._jitter_enabled:
            frac = self._jitter_frac
            T_act = T_nom * self._jitter_rng.uniform(1.0 - frac, 1.0 + frac)
        else:
            T_act = T_nom
        status.nominal_action_remaining = T_nom   # obs 始终用名义值
        status.action_remaining         = T_act   # 物理到达由实际值驱动
        status.planned_edge_duration    = T_act   # 动画进度分母（实际时长）

        # 离开当前节点（numpy + set 保持同步）
        self._occupied_nodes.discard(current_pos)
        ni = self._node_idx[current_pos]
        if self._occupied_count[ni] > 0:
            self._occupied_count[ni] -= 1

        return True
    
    def set_route_action(self, agent_id: int, target_node: int) -> bool:
        """设置路由移动：目标可以是任意可达节点，自动沿最短路径逐跳移动。

        中间节点到达时累积奖励并自动续航，直到最终目标才标记 READY。
        若目标是当前位置的直接邻居，退化为 set_move_action。

        Returns:
            bool: 是否设置成功
        """
        status = self.agents[agent_id]
        if status.state != AgentState.READY:
            return False

        current_pos = status.position

        if target_node == current_pos:
            return self.set_wait_action(agent_id)

        # 直接邻居：无需路由
        if target_node in self.graph.get_neighbors(current_pos):
            return self.set_move_action(agent_id, target_node)

        # 计算最短路径 [current, hop1, hop2, ..., target]
        path = self.graph.get_shortest_path(current_pos, target_node)
        if path is None or len(path) < 2:
            return False

        # path[0] == current_pos，path[1] 是第一跳，path[2:] 是后续路径点
        first_hop = path[1]
        remaining = path[2:]  # 可能为空（两跳路径时只有 target）
        self._routes[agent_id] = remaining
        self._acc_rewards[agent_id] = 0.0
        return self.set_move_action(agent_id, first_hop)

    def set_wait_action(self, agent_id: int) -> bool:
        """
        设置智能体在当前节点等待
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 是否设置成功
        """
        status = self.agents[agent_id]
        
        if status.state != AgentState.READY:
            return False
        
        status.state = AgentState.WAITING
        status.action_remaining         = self.waitT
        status.nominal_action_remaining = self.waitT  # 等待不扰动，双轨相同
        status.planned_edge_duration    = 0.0
        status.last_position = status.position
        status.target_node = status.position

        return True

    # ==================== 状态查询方法 ====================
    
    def is_ready(self, agent_id: int) -> bool:
        """智能体是否可以做决策"""
        return self.agents[agent_id].state == AgentState.READY
    
    def get_ready_agents(self) -> List[int]:
        """获取所有可以做决策的智能体"""
        return [i for i in range(self.num_agents) if self.is_ready(i)]
    
    def get_position(self, agent_id: int) -> int:
        """获取智能体位置（如果在边上，返回出发节点）"""
        return self.agents[agent_id].position
    
    def get_neighbors(self, node: int) -> List[int]:
        """获取节点的邻居"""
        return self.graph.get_neighbors(node)
    
    def get_node_idleness(self, node: int) -> float:
        """获取节点空闲度"""
        return self.node_idleness.get(node, 0.0)
    
    def get_all_idleness(self) -> Dict[int, float]:
        """获取所有节点的空闲度"""
        return dict(self.node_idleness)
    
    def get_agent_status(self, agent_id: int) -> AgentStatus:
        """获取智能体完整状态"""
        return self.agents[agent_id]

    def snapshot_agent_positions(self) -> Dict[int, tuple]:
        """
        获取当前时刻所有智能体的位置快照，用于动画可视化。
        
        Returns:
            {agent_id: (start_node, end_node, progress)}
            - 在节点上: (node, node, 0.0)
            - 在边上移动: (src, dst, progress ∈ [0,1])
        """
        snapshot = {}
        for agent_id, status in self.agents.items():
            if status.state == AgentState.ON_EDGE:
                # 用实际计划时长（含 jitter）作分母，保证动画进度与物理到达一致
                travel_time = status.planned_edge_duration
                if travel_time < 1e-12:
                    # 保底：回退到名义时长
                    travel_time = self.graph.get_edge_length(status.position, status.target_node) \
                                  / max(status.speed, 1e-6)
                progress = 1.0 - status.action_remaining / travel_time if travel_time > 0 else 1.0
                progress = max(0.0, min(1.0, progress))
                snapshot[agent_id] = (status.position, status.target_node, progress)
            else:
                # READY / WAITING: 在节点上
                snapshot[agent_id] = (status.position, status.position, 0.0)
        return snapshot

    # ==================== 指标相关方法 ====================
    
    @property
    def current_metrics(self) -> IdlenessMetrics:
        """获取当前时刻的指标"""
        return self.metrics_tracker.current
    
    def get_episode_metrics(self) -> Dict[str, List[float]]:
        """
        获取整个 episode 的指标历史
        
        Returns:
            {
                'step': [0, 1, 2, ...],
                'time': [0.0, 1.0, 2.5, ...],
                'igi': [...],
                'agi': [...],
                'iwi': [...],
                'wi': [...]
            }
        """
        return self.metrics_tracker.get_history_dict()
    
    def plot_episode_metrics(self, save_path: str = None, show: bool = True, use_time_axis: bool = False):
        """
        绘制 episode 指标曲线图
        
        Args:
            save_path: 保存路径（可选）
            show: 是否显示图形
            use_time_axis: True 使用时间作为 x 轴，False 使用 step
        """
        self.metrics_tracker.plot(save_path=save_path, show=show, use_time_axis=use_time_axis)
    
    def export_metrics_to_csv(self, path: str):
        """导出指标历史到 CSV 文件"""
        self.metrics_tracker.to_csv(path)
    
    # ==================== 启发式算法接口 ====================
    
    def get_heuristic_obs(self) -> Dict[str, Dict]:
        """
        返回启发式算法需要的观测格式,目前只适用于ER启发式算法
        
        Returns:
            {agent_str: {
                'current_node': int,      # 当前节点
                'neighbors': List[int],   # 邻居节点列表
                'on_edge': bool           # 是否在边上移动中
            }}
        """
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
        """
        返回启发式算法需要的全局状态,目前只适用于ER启发式算法
        
        Returns:
            {
                'graph': Graph,                       # 图结构对象
                'agent_positions': Dict[int, int],    # 智能体位置
                'agents_on_edge': Dict[int, bool],    # 智能体是否在边上
                'current_time': float,                # 当前仿真时间
                'node_last_visit': Dict[int, float],  # 节点上次访问时间
                'agent_speeds': List[float],          # 智能体速度
                'er_avg_edge_len': float,             # 平均边长
            }
        """
        # 从 idleness 反推 last_visit_time: last_visit[n] = current_time - idleness[n]
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
            'current_time': self.current_time,
            'node_last_visit': node_last_visit,
            'agent_speeds': self.speeds,
            'er_avg_edge_len': self.graph.get_average_edge_length() if hasattr(self.graph, 'get_average_edge_length') else 1.0,
        }
    
    def step_heuristic(self, actions: Dict[str, int]) -> TickResult:
        """
        执行启发式动作并推进环境（事件驱动模式）
        
        启发式算法直接与 PatrolWorld 交互的接口，无需经过 MDP 封装。
        
        Args:
            actions: {agent_str: neighbor_idx} 启发式动作，neighbor_idx 是邻居列表中的索引
                     只需要包含 READY 状态智能体的动作，其他智能体会被跳过
        
        Returns:
            TickResult: 包含时间推进结果
        """
        # 1. 为每个 READY 状态的智能体设置移动动作
        for agent_str, neighbor_idx in actions.items():
            agent_id = int(agent_str.split("_")[1]) if isinstance(agent_str, str) else int(agent_str)
            
            if not self.is_ready(agent_id):
                continue
            
            # 将邻居索引转换为目标节点
            current_pos = self.get_position(agent_id)
            neighbors = self.graph.get_neighbors(current_pos)
            
            if 0 <= neighbor_idx < len(neighbors):
                target_node = neighbors[neighbor_idx]
                self.set_move_action(agent_id, target_node)
        
        # 2. 推进到下一个事件
        return self.tick_to_next_event()