# load the topology G=(V,E,W,\phi) from graph_topology.json
# construct the graph in mathematical form

import json
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional

class Graph:
    def __init__(self, path: str): 
        with open(path, "r") as f:
            data = json.load(f)

            self.nodes: List[int] = data["nodes"]
            tmp_edges: List[Dict] = data["edges"]
            self.phi: Dict[int, int] = {int(k): int(v) for k, v in data["phi"].items()}

        # Initialize adjacency list
        self.adj_list: Dict[int, List[Tuple[int, int]]] = {node: [] for node in self.nodes}

        # Populate edges
        for edge in tmp_edges:
            src = edge["from"]
            dst = edge["to"]
            weight = edge["weight"]
            self.adj_list[src].append((dst, weight))
        
        self.total_node = len(self.nodes)
        self.total_edge = int()
        self.edges = dict()  # key is (nodeA,nodeB), value is edge index
        edge_index = 1
        for n in self.nodes:
            self.total_edge += len(self.adj_list[n])
            for neighbor, _ in self.adj_list[n]:
                if n < neighbor: 
                    self.edges[(n, neighbor)] = edge_index
                    edge_index += 1
        self.total_edge = int(self.total_edge / 2)
        
        # 预计算所有节点对之间的最短路径长度
        self._shortest_paths: Dict[int, Dict[int, float]] = {}
        self._precompute_shortest_paths()
        
    def _precompute_shortest_paths(self):
        """
        预计算所有节点对之间的最短路径长度（使用 Dijkstra 算法）
        
        时间复杂度: O(V * (V + E) * log V)
        空间复杂度: O(V^2)
        
        对于巡逻问题的图规模（通常 < 100 节点），这个开销是可接受的
        """
        for src in self.nodes:
            self._shortest_paths[src] = self._dijkstra(src)
    
    def _dijkstra(self, src: int) -> Dict[int, float]:
        """
        使用Dijkstra 算法从单个源节点计算到所有其他节点的最短路径
        
        Args:
            src: 源节点
        
        Returns:
            dist: {node: shortest_distance} 从 src 到每个节点的最短距离
        """
        dist = {node: float('inf') for node in self.nodes}
        dist[src] = 0.0
        
        # 优先队列: (distance, node)
        pq = [(0.0, src)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            for v, weight in self.adj_list[u]:
                if v not in visited:
                    new_dist = d + weight
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
        
        return dist
    
    def shortest_path_length(self, src: int, dst: int) -> float:
        """
        获取两个节点之间的最短路径长度
        
        Args:
            src: 源节点
            dst: 目标节点
        
        Returns:
            最短路径长度，如果不可达则返回 float('inf')
        """
        if src not in self._shortest_paths:
            return float('inf')
        return self._shortest_paths[src].get(dst, float('inf'))
    
    def get_shortest_path(self, src: int, dst: int) -> Optional[List[int]]:
        """
        获取两个节点之间的最短路径（节点序列）
        
        Args:
            src: 源节点
            dst: 目标节点
        
        Returns:
            路径节点列表 [src, ..., dst]，如果不可达则返回 None
        """
        if src == dst:
            return [src]
        
        if self.shortest_path_length(src, dst) == float('inf'):
            return None
        
        # 反向重建路径
        path = [dst]
        current = dst
        
        while current != src:
            # 找前驱节点：从邻居中找一个使得 dist[prev] + edge_weight == dist[current]
            found = False
            for prev in self.nodes:
                if prev == current:
                    continue
                edge_len = self.get_edge_length(prev, current)
                if edge_len > 0:  # 有边
                    expected_dist = self._shortest_paths[src].get(prev, float('inf')) + edge_len
                    if abs(expected_dist - self._shortest_paths[src][current]) < 1e-9:
                        path.append(prev)
                        current = prev
                        found = True
                        break
            
            if not found:
                return None  # 不应该发生
        
        return list(reversed(path))

    def get_edge_length(self, node1: int, node2: int) -> float:
        """
        获取两个相邻节点之间的边长度
        
        Args:
            node1: 节点1
            node2: 节点2
        
        Returns:
            边长度，如果没有直接连边则返回 0
        """
        for neighbor, weight in self.adj_list.get(node1, []):
            if neighbor == node2:
                return float(weight)
        return 0.0

    def get_max_edge_length(self):
        edge_length = []
        for n in self.nodes:
            for _, weight in self.adj_list[n]:
                edge_length.append(weight)
        return max(edge_length)
    
    def get_max_phi(self):
        return max(self.phi.values())

    def get_max_degree(self):
        return max(len(self.adj_list[n]) for n in self.nodes)

    def get_num_edges(self, is_directed) -> int:
        if is_directed:
            return 2 * self.total_edge
        else:
            return self.total_edge

    def get_neighbors(self, n:int) -> List[int]:
        neighbors = [m for m,_ in self.adj_list[n]]
        return sorted(neighbors)

    def get_adj_weight_mat(self):
        """
        Return np.ndarray Adjacent Matrix and Edge Weight Matrix
        """
        n = self.total_node
        adj_mat = np.full((n, n), 0, dtype=int)
        weight_mat = np.full((n, n), -1, dtype=float)

        # Node index starts from 0 or 1 are both compatible
        node_to_idx = {node: idx for idx, node in enumerate(sorted(self.nodes))}

        for i in self.nodes:
            for j, weight in self.adj_list[i]:
                adj_mat[node_to_idx[i], node_to_idx[j]] = 1
                weight_mat[node_to_idx[i], node_to_idx[j]] = weight

        return adj_mat, weight_mat
    
