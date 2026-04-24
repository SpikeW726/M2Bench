"""一次性离线工具：为 graphs/*.json 生成 2D 节点坐标。

运行:
    python utils/graph_layout.py                       # 处理所有 graphs/*.json
    python utils/graph_layout.py graphs/SFcrimemap.json  # 只处理指定文件

输出 graphs/{stem}_coords.json，格式:
    {"0": [x, y], "1": [x, y], ...}  坐标已归一化到 [0, 1]

选用 Kamada-Kawai layout（以边权重作为距离，适合巡逻拓扑图），结果完全确定，
无需固定随机种子。
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np


def generate_layout(graph_path: str, method: str = "kamada_kawai") -> dict:
    """计算图的 2D 节点坐标并归一化到 [0, 1]。

    Args:
        graph_path: JSON 图文件路径。
        method: "kamada_kawai"（默认，用边权作距离）或 "spring"（seed=42）。

    Returns:
        Dict[str, List[float]]  — 节点 ID（字符串）-> [x, y]
    """
    with open(graph_path) as f:
        data = json.load(f)

    G = nx.Graph()
    # 先添加所有节点（防止孤立节点丢失）
    for n in data["nodes"]:
        G.add_node(n)
    for e in data["edges"]:
        G.add_edge(e["from"], e["to"], weight=float(e["weight"]))

    if method == "kamada_kawai":
        # weight 参数指定用于距离计算的边属性名称
        pos = nx.kamada_kawai_layout(G, weight="weight")
    else:
        pos = nx.spring_layout(G, weight="weight", seed=42)

    # 归一化到 [0, 1]
    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    coords = {}
    for n in G.nodes():
        x = float((pos[n][0] - x_min) / x_range)
        y = float((pos[n][1] - y_min) / y_range)
        coords[str(n)] = [x, y]

    return coords


def process_graph_file(graph_path: Path, method: str = "kamada_kawai") -> Path:
    """处理单个 JSON 图文件，写出 _coords.json。"""
    coords = generate_layout(str(graph_path), method=method)
    out_path = graph_path.with_name(graph_path.stem + "_coords.json")
    with open(out_path, "w") as f:
        json.dump(coords, f, indent=2)
    return out_path


def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # 默认处理项目根目录下 graphs/ 的所有 .json（跳过已有的 _coords.json）
        repo_root = Path(__file__).parent.parent
        targets = [
            p for p in (repo_root / "graphs").glob("*.json")
            if "_coords" not in p.stem
        ]

    if not targets:
        print("未找到任何 JSON 图文件。")
        return

    for p in sorted(targets):
        if "_coords" in p.stem:
            print(f"  跳过（已是坐标文件）: {p}")
            continue
        try:
            out = process_graph_file(p)
            print(f"  Generated: {out}")
        except Exception as e:
            print(f"  ERROR processing {p}: {e}")


if __name__ == "__main__":
    main()
