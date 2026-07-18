"""Generate normalized 2D coordinates for graph JSON files.

The command writes ``graphs/{stem}_coords.json``. Kamada-Kawai layout uses edge
weights as distances and is deterministic; spring layout uses a fixed seed.
Coordinates are normalized to ``[0, 1]``.
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

def generate_layout(graph_path: str, method: str = "kamada_kawai") -> dict:
    with open(graph_path) as f:
        data = json.load(f)

    G = nx.Graph()
    # Add every node first to preserve isolated nodes.
    for n in data["nodes"]:
        G.add_node(n)
    for e in data["edges"]:
        G.add_edge(e["from"], e["to"], weight=float(e["weight"]))

    if method == "kamada_kawai":

        pos = nx.kamada_kawai_layout(G, weight="weight")
    else:
        pos = nx.spring_layout(G, weight="weight", seed=42)

    # Normalize to [0, 1].
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
    coords = generate_layout(str(graph_path), method=method)
    out_path = graph_path.with_name(graph_path.stem + "_coords.json")
    with open(out_path, "w") as f:
        json.dump(coords, f, indent=2)
    return out_path

def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:

        repo_root = Path(__file__).parent.parent
        targets = [
            p for p in (repo_root / "graphs").glob("*.json")
            if "_coords" not in p.stem
        ]

    if not targets:
        print("No JSON graph files found.")
        return

    for p in sorted(targets):
        if "_coords" in p.stem:
            print(f"  Skipping coordinate file: {p}")
            continue
        try:
            out = process_graph_file(p)
            print(f"  Generated: {out}")
        except Exception as e:
            print(f"  ERROR processing {p}: {e}")

if __name__ == "__main__":
    main()
