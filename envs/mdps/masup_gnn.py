from typing import Dict, List, Optional
import numpy as np
import random
from gymnasium.spaces import Box, Discrete

from envs.mdps.masup import MASUPEnv
from envs.mdps.patrol_core import AgentState, TickResult
from envs.mdps.base_envs import EventDrivenEnv

class MASUPGraphEnv(MASUPEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        self.static_node_num = self.world.num_nodes
        self.static_edge_num = self.world.num_edges
        self.agent_num = self.world.num_agents

        self.static_edges = []
        for u in sorted(self.world.graph.nodes):
            for v, weight in self.world.graph.adj_list[u]:
                self.static_edges.append((u, v, weight))        

        # 节点总数 = 物理节点 + 虚拟智能体节点
        self.total_node_num = self.static_node_num + self.agent_num

        # 最大边数 = 静态边 + 每个智能体的动态边上限
        self.max_dynamic_edges = self.agent_num * 2
        self.total_max_edges = self.static_edge_num + self.max_dynamic_edges    

        self.node_feat_dim = kwargs.get("node_feat_dim", 2)     # [Type, Weighted_Idleness]
        self.edge_feat_dim = kwargs.get("edge_feat_dim", 1)     # [Edge Weight]
        self.global_feat_dim = kwargs.get("global_feat_dim", 2) # [Max_Idleness, Timer]   

        if self.role_ifm == "agent-index":
            identity_len = self.world.num_agents
        elif self.role_ifm == "position":
            identity_len = 1
        elif self.role_ifm == "decision":
            identity_len = 1 + self.world.num_agents

        self.obs_size = (
            self.total_node_num * self.node_feat_dim +
            self.total_max_edges * 2 +  # Edge Index (Src + Dst)
            self.total_max_edges * self.edge_feat_dim +
            self.total_max_edges +      # Mask
            self.global_feat_dim +
            identity_len
        )

    def observation_space(self, agent):
        # node_num = self.total_node_num
        # --- 1. 节点特征空间 ---
        node_low = [0.0] * (self.total_node_num * self.node_feat_dim)
        node_high = [float('inf')] * (self.total_node_num * self.node_feat_dim)

        # --- 2. 边索引空间 (Src + Dst) ---
        edge_idx_low = [0.0] * (2 * self.total_max_edges)
        edge_idx_high = [float(self.total_node_num)] * (2 * self.total_max_edges)

        # --- 3. 边属性空间 ---
        edge_attr_low = [0.0] * self.total_max_edges
        edge_attr_high = [float('inf')] * self.total_max_edges

        # --- 4. 边掩码空间 ---
        mask_low = [0.0] * self.total_max_edges
        mask_high = [1.0] * self.total_max_edges

        # --- 5. 全局状态空间 ---
        global_low = [0.0] * self.global_feat_dim
        global_high = [float('inf')] * self.global_feat_dim

        # --- 6. Agent Identity
        if self.role_ifm == "agent-index":
            identity_low = [0.0] * self.agent_num
            identity_high = [1.0] * self.agent_num
        elif self.role_ifm == "position":
            identity_low = [0.0]
            identity_high = [float(self.static_node_num)]
        elif self.role_ifm == "decision":
            identity_low = [0.0] + [0.0] * self.agent_num
            identity_high = [float(self.static_node_num)] + [1.0] * self.agent_num

        low = np.array(node_low + edge_idx_low + edge_attr_low + mask_low + global_low + identity_low, dtype=np.float32)
        high = np.array(node_high + edge_idx_high + edge_attr_high + mask_high + global_high + identity_high, dtype=np.float32)

        return Box(low=low, high=high, dtype=np.float32)

    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, np.ndarray]:
        """构建当前时刻的动态图观测"""
        obs = {}
        # ======== 1. 构建节点特征 (Node Features) ========
        # 静态节点 (0 ~ N-1)
        # 特征: [Type=1, Idleness]
        sorted_nodes = sorted(list(self.world.graph.nodes))
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        default_node = sorted_nodes[0] if sorted_nodes else 0
        
        node_feats = []
        for node in sorted_nodes:
            phi_val = float(self.world.graph.phi.get(node, 1.0))
            idle_val = float(self.world.node_idleness.get(node, 0.0))
            node_feats.extend([1.0, phi_val * idle_val])
        
        # 虚拟节点 (N ~ N+K-1) 代表智能体
        # 特征: [Type=0, 0]
        # 虚拟节点索引从 self.static_node_num 开始
        for _ in range(self.agent_num):
            node_feats.extend([0.0, 0.0]) # 智能体节点 Idleness 设为 0

        # ======== 2. 构建边列表和权重 (Edges) ========
        edge_indices_src = []
        edge_indices_dst = []
        edge_weights = []
        
        # A. 添加物理边 (Static Edges)
        for u, v, w in self.static_edges:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_indices_src.append(float(u_idx))
            edge_indices_dst.append(float(v_idx))
            edge_weights.append(float(w))

        # B. 添加智能体动态边
        for i in range(self.agent_num):
            virtual_node_idx = self.static_node_num + i

            last_node = self.world.agents[i].last_position
            if last_node not in node_to_idx:
                last_node = default_node

            target_node = self.world.agents[i].target_node
            if target_node not in node_to_idx:
                target_node = self.world.agents[i].position

            time_left = float(self.world.agents[i].action_remaining)

            if self.world.agents[i].state == AgentState.ON_EDGE:
                # --- 情况 1: 正在边上移动 ---
                # 边连接: LeaveNode (last_node) -> VirtualNode -> TargetNode
                u_idx = node_to_idx[last_node]
                v_idx = node_to_idx[target_node]

                full_dist = self.world.graph.get_edge_length(last_node, target_node)
                dist_to_go = time_left * self.world.agents[i].speed
                dist_traveled = max(0.0, full_dist - dist_to_go)

                # 边 1: LeaveNode -> VirtualNode (已走距离)
                edge_indices_src.append(float(u_idx))
                edge_indices_dst.append(float(virtual_node_idx))
                edge_weights.append(float(dist_traveled))
                
                # 边 2: VirtualNode -> TargetNode (剩余距离)
                edge_indices_src.append(float(virtual_node_idx))
                edge_indices_dst.append(float(v_idx))
                edge_weights.append(float(dist_to_go))

            elif self.world.agents[i].state == AgentState.WAITING:
                # --- 情况 2: 在节点上等待 (Waiting) ---
                # 边连接: CurrentNode -> VirtualNode (已等待) 和 VirtualNode -> CurrentNode (剩余等待)
                u_idx = node_to_idx[last_node]

                # 暂时用 0.0 代表已等待时间，重点关注剩余时间
                # 边 1: CurrentNode -> VirtualNode
                edge_indices_src.append(float(u_idx))
                edge_indices_dst.append(float(virtual_node_idx))
                edge_weights.append(0.0) 
                
                # 边 2: VirtualNode -> CurrentNode
                edge_indices_src.append(float(virtual_node_idx))
                edge_indices_dst.append(float(u_idx))
                edge_weights.append(float(time_left))

            elif self.world.agents[i].state == AgentState.READY:
                # --- 情况 3: 就在节点上 (Finished) ---
                # 建立双向连接: VirtualNode <-> CurrentNode (权重均为0)
                u_idx = node_to_idx[last_node]

                # 边 1: CurrentNode -> VirtualNode
                edge_indices_src.append(float(u_idx))
                edge_indices_dst.append(float(virtual_node_idx))
                edge_weights.append(0.0)
                
                # 边 2: VirtualNode -> CurrentNode
                edge_indices_src.append(float(virtual_node_idx))
                edge_indices_dst.append(float(u_idx))
                edge_weights.append(0.0)

        # ======== 3. 填充与掩码 (Padding) 目前来看完全多余 ========
        current_edge_count = len(edge_weights)

        # 填充至 total_max_edges
        pad_size = self.total_max_edges - current_edge_count
        
        # 有效掩码: 真实边为 1, 填充边为 0
        edge_mask = [1.0] * current_edge_count + [0.0] * pad_size

        if pad_size > 0:
            edge_indices_src.extend([0.0] * pad_size) # Pad with 0
            edge_indices_dst.extend([0.0] * pad_size)
            edge_weights.extend([0.0] * pad_size)

        # ======== 4. 全局状态 ========
        global_feat = [float(self.world.worst_idleness), float(self.obs_timer)]

        for i in range(self.agent_num):   
            # ======== 5. Agent Identity ========  
            if self.role_ifm == "agent-index":
                identity = [0.0] * self.agent_num
                identity[i] = 1.0
            elif self.role_ifm == "position":
                identity = self.world.agents[i].position
            elif self.role_ifm == "decision":
                decision_idx = int(self._decision_index_map.get(i, 0)) if hasattr(self, '_decision_index_map') else 0
                one_hot = [0.0] * self.agent_num
                one_hot[decision_idx] = 1.0
                identity = self.world.agents[i].position + one_hot

            single_obs = np.concatenate([
                node_feats,
                edge_indices_src,
                edge_indices_dst,
                edge_weights,
                edge_mask,
                global_feat,
                identity
            ]).astype(np.float32)
            obs[f"agent_{i}"] = single_obs

        return obs