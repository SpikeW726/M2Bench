#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum

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
    action_remaining: float = 0.0       # Remaining time for last action
    speed: float = 1.0
    

@dataclass
class TickResult:
    """Results that tick() return"""
    dt: float                                               # Actual elapsed time
    arrivals: Dict[int, int] = field(default_factory=dict)  # {agent_id: arrived_node}
    wait_completed: Set[int] = field(default_factory=set)   # Agents that finish waiting
    raw_rewards: Dict[int, float] = field(default_factory=dict)  # arrived_node_idleness*phi or waitT*phi or 0.0
    pre_arrival_igi: float = 0.0                            # 到达清零前的 IGI（所有节点空闲度均值）
    
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
        self.max_phi = self.graph.get_max_phi()
        self.num_nodes = len(self.graph.nodes)

        # 智能体状态
        self.num_agents = cfg["num_agents"]
        self.speeds = cfg.get("speeds", [1.0] * self.num_agents)
        self.agents: Dict[int, AgentStatus] = {}

        # 时间变量
        self.node_idleness: Dict[int, float] = {n: 0.0 for n in self.graph.nodes}
        self.current_time: float = 0.0
        self.worst_idleness: float = 0.0
        self.waitT = cfg.get("deltaT", 1.0)  # 一次等待动作的持续时间
        
        # Episode 指标追踪器
        self.metrics_tracker = EpisodeMetricsTracker()
        self.last_episode_metrics: Optional[IdlenessMetrics] = None  # 上一个 episode 终止时的指标
        self.step_count: int = 0 
     
        # 记录哪些节点当前有智能体"占据"（恰好有智能体到达或有智能体在该节点等待）
        self._occupied_nodes: Set[int] = set()
    
    def reset(self, initial_positions: Optional[List[int]] = None) -> None:
        """重置物理世界"""
        # 保存上一个 episode 的终止指标（reset 前 history 非空时）
        if self.metrics_tracker.history:
            self.last_episode_metrics = self.metrics_tracker.current
        
        self.current_time = 0.0
        self.step_count = 0
        self.worst_idleness = 0.0
        self.node_idleness = {n: 0.0 for n in self.graph.nodes}
        
        # 重置指标追踪器
        self.metrics_tracker.reset()
        
        # 初始化智能体位置
        if initial_positions is None:
            import random
            initial_positions = random.sample(list(self.graph.nodes), self.num_agents)
        
        self._occupied_nodes.clear()
        for i in range(self.num_agents):
            pos = initial_positions[i]
            self.agents[i] = AgentStatus(
                position=pos,
                state=AgentState.READY,
                last_position=pos,
                speed=self.speeds[i]
            )
            self._occupied_nodes.add(pos)
        
        # 记录初始状态的指标
        self.metrics_tracker.record(self.node_idleness, self.step_count, self.current_time)

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
        
        # 1. 更新节点空闲度
        for node in self.graph.nodes:
            if node not in self._occupied_nodes:
                self.node_idleness[node] += dt
        current_worst_idleness = max(self.node_idleness.values())
        self.worst_idleness = max(self.worst_idleness, current_worst_idleness)
        
        # 1.5 快照：到达清零前的 IGI（所有节点空闲度均值）
        result.pre_arrival_igi = sum(self.node_idleness.values()) / len(self.node_idleness)
        
        # 2. 更新智能体状态
        for agent_id, status in self.agents.items():
            if status.state in (AgentState.ON_EDGE, AgentState.WAITING):
                status.action_remaining -= dt
                
                # 检查是否完成
                if status.action_remaining <= 1e-9:
                    status.action_remaining = 0.0
                    
                    if status.state == AgentState.ON_EDGE:
                        # 到达目标节点
                        arrived_node = status.target_node
                        status.last_position = status.position
                        status.position = arrived_node
                        status.state = AgentState.READY
                        status.target_node = -1
                        
                        # 更新占据状态
                        self._occupied_nodes.add(arrived_node)
                        
                        # 清零到达节点的空闲度
                        result.raw_rewards[agent_id] = self.node_idleness[arrived_node] * self.graph.phi[arrived_node]
                        self.node_idleness[arrived_node] = 0.0
                        
                        result.arrivals[agent_id] = arrived_node
                        
                    elif status.state == AgentState.WAITING:
                        # 等待完成
                        status.state = AgentState.READY
                        waiting_node = self.agents[agent_id].position
                        result.raw_rewards[agent_id] = self.waitT * self.graph.phi[waiting_node]
                        result.wait_completed.add(agent_id)
        
        # 3. 更新时间
        self.current_time += dt
        
        # 4. 记录指标
        self.step_count += 1
        weighted_idleness = {
            node: float(self.graph.phi.get(node, 1.0)) * float(idle_val)
            for node, idle_val in self.node_idleness.items()
        }
        self.metrics_tracker.record(weighted_idleness, self.step_count, self.current_time)
        
        return result

    def tick_to_next_event(self) -> TickResult:
        """
        推进到最近的事件发生时刻
        
        事件包括：
        - 某个智能体到达目标节点
        - 某个智能体等待完成
        
        Returns:
            TickResult: 如果没有待处理事件，dt=0
        """
        dt = self._compute_next_event_time()
        if dt < 0:
            return TickResult(dt=0.0)
        return self.tick(dt)

    def _compute_next_event_time(self) -> float:
        """计算下一个事件发生的时间"""
        min_time = float('inf')
        
        for status in self.agents.values():
            if status.state in (AgentState.ON_EDGE, AgentState.WAITING):
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
        status.action_remaining = float(edge_length)
        status.last_position = current_pos
        
        # 离开当前节点
        self._occupied_nodes.discard(current_pos)
        
        return True
    
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
        status.action_remaining = self.waitT
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
                edge_len = self.graph.get_edge_length(status.position, status.target_node)
                progress = 1.0 - status.action_remaining / edge_len if edge_len > 0 else 1.0
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