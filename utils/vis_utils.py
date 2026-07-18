import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Any, Union
from utils.graph_utils import Graph

def create_nx_layout(map_graph: Graph):
    G = nx.Graph()
    for node in map_graph.nodes:
        G.add_node(node)

    for node, neighbors in map_graph.adj_list.items():
        for neighbor, weight in neighbors:
            if node < neighbor:
                G.add_edge(node, neighbor, weight=weight)

    num_nodes = len(map_graph.nodes)
    scale_factor = 5.0 + np.sqrt(num_nodes)

    # weight='weight'.
    # scale=scale_factor.
    pos = nx.kamada_kawai_layout(G, weight='weight', scale=scale_factor)

    return pos

def _parse_history_input(history: Union[List[float], str]) -> List[float]:
    if isinstance(history, list):
        return history
    if isinstance(history, str) and os.path.isfile(history):
        values: List[float] = []
        try:
            with open(history, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        values.append(float(line.split(',')[0]))
                    except ValueError:
                        continue
        except Exception as exc:
            print(f"Warning: failed to parse history file {history}: {exc}")
        return values
    return []

def plot_training_curve(history: Union[List[float], str], title: str, save_dir: str,
                        use_normalization: bool = True, filename: str = 'training_curve.png'):
    history_values = _parse_history_input(history)
    if not history_values:
        print("No training-curve data available, skip plotting.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_values) + 1), history_values, 'b-', linewidth=1)
    plt.xlabel('Episode')
    ylabel = 'Normalized Average Idleness' if use_normalization else 'Average Idleness'
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curve saved to {file_path}")

def plot_training_curves(avg_idleness: List[float], max_latency: List[float], title: str, save_dir: str,
                         use_normalization: bool = True, filename: str = 'training_curve.png'):
    try:

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(range(1, len(avg_idleness) + 1), avg_idleness, 'b-', linewidth=1)
        ylabel = 'Normalized Average Idleness' if use_normalization else 'Average Idleness'
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(f"{title} ({ylabel})")
        axes[0].grid(True, alpha=0.3)

        if len(max_latency) > 0:
            axes[1].plot(range(1, len(max_latency) + 1), max_latency, 'r-', linewidth=1)
            axes[1].set_xlabel('Episode')
            ylabel_max = 'Normalized Max Idleness' if use_normalization else 'Max Idleness'
            axes[1].set_ylabel(ylabel_max)
            axes[1].set_title(f"{title} ({ylabel_max})")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {file_path}")
    except Exception as e:
        print(f"Error plotting training curves: {e}")

        try:
            if len(avg_idleness) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(avg_idleness) + 1), avg_idleness, 'b-', linewidth=1)
                ylabel = 'Normalized Average Idleness' if use_normalization else 'Average Idleness'
                plt.xlabel('Episode')
                plt.ylabel(ylabel)
                plt.title(f"{title} ({ylabel})")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                file_path = os.path.join(save_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Training curve saved to {file_path}")
        except Exception as e2:
            print(f"Error plotting fallback training curve: {e2}")

def plot_evaluation_metrics(avg_history: List[float], worst_history: List[float], title: str, save_dir: str,
                            use_normalization: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].plot(avg_history, color='b', linewidth=1.5)
    ylabel_avg = 'Normalized Average Idleness' if use_normalization else 'Average Idleness'
    axes[0].set_title(f"{title} - {ylabel_avg}", fontsize=18)
    axes[0].set_xlabel('Time Step', fontsize=14)
    axes[0].set_ylabel(ylabel_avg, fontsize=14)
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(worst_history, color='r', linewidth=1.5)
    ylabel_worst = 'Normalized Max Idleness' if use_normalization else 'Max Idleness'
    axes[1].set_title(f"{title} - {ylabel_worst}", fontsize=18)
    axes[1].set_xlabel('Time Step', fontsize=14)
    axes[1].set_ylabel(ylabel_worst, fontsize=14)
    axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    file_path = os.path.join(save_dir, 'evaluation_idleness_combined.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined evaluation chart saved to {file_path}")

def _downsample_frames(frames, max_frames):
    if max_frames is None or len(frames) <= max_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]

def create_animation(
    map_graph,
    agent_positions_history,
    total_frames,
    algorithm_name,
    map_name,
    save_dir=None,
    max_frames=None,
    plot_stem=None,
):
    if max_frames is not None and len(agent_positions_history) > max_frames:
        print(f"Downsampling animation: {len(agent_positions_history)} → {max_frames} frames")
        agent_positions_history = _downsample_frames(agent_positions_history, max_frames)
        total_frames = len(agent_positions_history)

    print("Starting animation...")

    num_nodes = len(map_graph.nodes)

    figure_scale_factor = 1.0 + num_nodes / 30.0

    baseline_nodes = 12.0
    visual_scale_factor = max(0.6, np.sqrt(baseline_nodes / num_nodes)) if num_nodes > 0 else 1.0

    node_markersize = 28 * visual_scale_factor
    node_fontsize = 14 * visual_scale_factor
    agent_markersize = 18 * visual_scale_factor
    agent_fontsize = 9 * visual_scale_factor
    edge_label_fontsize = 9 * visual_scale_factor

    # Ensure figure dimensions are even numbers (required for H.264 encoding).
    fig_width = 12 * figure_scale_factor
    fig_height = 10 * figure_scale_factor
    # Round to nearest even number.
    fig_width = round(fig_width / 2) * 2
    fig_height = round(fig_height / 2) * 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    node_positions = create_nx_layout(map_graph)

    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:
                pos1 = node_positions[node]
                pos2 = node_positions[neighbor]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.6, linewidth=2)

    for node, pos in node_positions.items():
        ax.plot(pos[0], pos[1], 'o', markersize=node_markersize, color='skyblue', markeredgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=node_fontsize, fontweight='bold')

    label_positions = _calculate_label_positions(map_graph, node_positions)
    for (node, neighbor), (label_x, label_y) in label_positions.items():
        weight = map_graph.get_edge_length(node, neighbor)
        if weight is not None:
            ax.text(label_x, label_y, str(weight), ha='center', va='center',
                   fontsize=edge_label_fontsize, fontweight='normal', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=0.5))

    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]
    x_margin = (max(x_coords) - min(x_coords)) * 0.1
    y_margin = (max(y_coords) - min(y_coords)) * 0.1

    padding = 0.25 * visual_scale_factor
    ax.set_xlim(min(x_coords) - x_margin - padding, max(x_coords) + x_margin + padding)
    ax.set_ylim(min(y_coords) - y_margin - padding, max(y_coords) + y_margin + padding)
    ax.set_aspect('equal')
    ax.set_title(f'{algorithm_name} on {map_name} - Agent Movement', fontsize=16, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    agent_markers = []
    agent_labels = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']

    rgb_colors = [_to_rgb(c) for c in colors]

    for i in range(len(agent_positions_history[0])):
        bg_color = rgb_colors[i % len(rgb_colors)]
        text_color = _get_text_color_for_bg(bg_color)
        marker, = ax.plot([], [], '^', markersize=agent_markersize, color=bg_color,
                         markeredgecolor='black', linewidth=2, label=f'Agent {i}', zorder=10)
        label = ax.text(0, 0, str(i), ha='center', va='center', color=text_color,
                        fontsize=agent_fontsize, fontweight='bold', zorder=11)
        agent_markers.append(marker)
        agent_labels.append(label)

    legend_handles = [Line2D([0], [0], marker='^', color='w', markerfacecolor=rgb_colors[i % len(rgb_colors)],
                              markeredgecolor='black', markersize=agent_markersize, linewidth=0)
                      for i in range(len(agent_markers))]
    legend_labels = [f'Agent {i}' for i in range(len(agent_markers))]
    fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.985, 0.98),
               framealpha=0.9, borderpad=0.6)

    def animate(frame):
        if frame >= len(agent_positions_history):
            return (*agent_markers, *agent_labels)

        current_positions = agent_positions_history[frame]

        for agent_id, (start_node, end_node, progress) in current_positions.items():
            if agent_id >= len(agent_markers):
                continue
            if start_node == end_node:
                pos = node_positions[start_node]
                agent_markers[agent_id].set_data([pos[0]], [pos[1]])
                agent_labels[agent_id].set_position((pos[0], pos[1]))
            else:
                start_pos = node_positions[start_node]
                end_pos = node_positions[end_node]
                current_x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                current_y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                agent_markers[agent_id].set_data([current_x], [current_y])
                agent_labels[agent_id].set_position((current_x, current_y))
        return (*agent_markers, *agent_labels)

    anim = FuncAnimation(fig, animate, frames=min(total_frames, len(agent_positions_history)),
                        interval=167, blit=False, repeat=True)

    if save_dir is None:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)

    stem_part = f"_{plot_stem}" if plot_stem else ""
    base_name = f"{algorithm_name}_animation_{map_name}{stem_part}"

    # Choose output format - set this to True for MP4, False for GIF.
    use_mp4_format = True

    if use_mp4_format:
        # MP4 output - higher quality, smaller file size, but slower rendering.
        animation_filename = os.path.join(save_dir, f"{base_name}.mp4")
        try:

            import matplotlib.animation as animation_mod
            try:

                Writer = animation_mod.FFMpegWriter
                writer = Writer(fps=6, metadata=dict(artist='MARLlib'), bitrate=2000)
                anim.save(animation_filename, writer=writer)
            except Exception:

                anim.save(animation_filename, writer='ffmpeg', fps=6, bitrate=2000)
        except Exception as e:
            print(f"Failed to save MP4: {e}")
            print("Falling back to GIF...")

            animation_filename = os.path.join(save_dir, f"{base_name}.gif")
            anim.save(animation_filename, writer='pillow', fps=6, dpi=90)
    else:
        # GIF output - faster rendering, larger file size.
        animation_filename = os.path.join(save_dir, f"{base_name}.gif")
        anim.save(animation_filename, writer='pillow', fps=6, dpi=90)

    plt.close()
    print(f"Animation saved as '{animation_filename}'")

def create_event_driven_animation(
    map_graph,
    agent_positions_history,
    time_intervals,
    algorithm_name,
    map_name,
    save_dir=None,
    max_frames=None,
    plot_stem=None,
):
    print("Starting event-driven animation...")

    if len(agent_positions_history) == 0:
        print("Warning: No agent position history provided.")
        return

    fps = 6
    frames_per_time_unit = fps  # 1 time unit approximately 1.

    event_frame_counts = []
    for interval in time_intervals:
        event_frame_counts.append(max(1, int(round(interval * frames_per_time_unit))))

    interpolated_positions = []

    interpolated_positions.append(agent_positions_history[0])

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    for i in range(1, len(agent_positions_history)):
        prev_state = agent_positions_history[i - 1]
        curr_state = agent_positions_history[i]
        num_frames = event_frame_counts[i - 1]

        all_agents = set(prev_state.keys()) | set(curr_state.keys())

        for j in range(1, num_frames + 1):
            alpha = j / num_frames

            current_frame = {}
            for agent_id in all_agents:
                if agent_id not in curr_state:
                    continue

                curr_start, curr_end, curr_prog = curr_state[agent_id]
                if agent_id in prev_state:
                    prev_start, prev_end, prev_prog = prev_state[agent_id]
                else:

                    prev_start, prev_end, prev_prog = curr_start, curr_end, 0.0

                same_edge = (prev_start == curr_start and prev_end == curr_end)
                prev_on_node = (prev_start == prev_end)
                curr_on_node = (curr_start == curr_end)

                if same_edge:
                    prog = _clamp(prev_prog + alpha * (curr_prog - prev_prog))
                    current_frame[agent_id] = (curr_start, curr_end, prog)

                elif (not prev_on_node) and curr_on_node and (prev_end == curr_start == curr_end):
                    prog = _clamp(prev_prog + alpha * (1.0 - prev_prog))
                    current_frame[agent_id] = (prev_start, prev_end, prog)

                elif prev_on_node and curr_on_node and (prev_end != curr_end):
                    prog = _clamp(alpha)
                    current_frame[agent_id] = (prev_end, curr_end, prog)

                elif prev_on_node and (not curr_on_node) and (prev_end == curr_start):
                    prog = _clamp(alpha * curr_prog)
                    current_frame[agent_id] = (curr_start, curr_end, prog)

                else:
                    prev_remaining = 1.0 - prev_prog

                    total_travel = prev_remaining + curr_prog
                    if total_travel > 0:
                        split = prev_remaining / total_travel
                    else:
                        split = 0.5

                    if alpha <= split:

                        local_alpha = alpha / split if split > 0 else 1.0
                        prog = _clamp(prev_prog + local_alpha * prev_remaining)
                        current_frame[agent_id] = (prev_start, prev_end, prog)
                    else:

                        local_alpha = (alpha - split) / (1.0 - split) if split < 1.0 else 1.0
                        prog = _clamp(local_alpha * curr_prog)
                        current_frame[agent_id] = (curr_start, curr_end, prog)

            interpolated_positions.append(current_frame)

    if max_frames is not None and len(interpolated_positions) > max_frames:
        print(f"Downsampling event-driven animation: {len(interpolated_positions)} → {max_frames} frames")
        interpolated_positions = _downsample_frames(interpolated_positions, max_frames)

    total_frames = len(interpolated_positions)
    create_animation(
        map_graph,
        interpolated_positions,
        total_frames,
        algorithm_name,
        map_name,
        save_dir,
        plot_stem=plot_stem,
    )

def _get_text_color_for_bg(bg_color_rgb):
    luminance = 0.299 * bg_color_rgb[0] + 0.587 * bg_color_rgb[1] + 0.114 * bg_color_rgb[2]

    return 'black' if luminance > 0.5 else 'white'

def _to_rgb(color_name):
    import matplotlib.colors as mcolors
    return mcolors.to_rgb(color_name)

def _calculate_label_positions(map_graph, node_positions):
    label_positions = {}

    edges = []
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:
                edges.append((node, neighbor, weight))

    edges.sort()

    for i, (node, neighbor, weight) in enumerate(edges):
        pos1 = np.array(node_positions[node])
        pos2 = np.array(node_positions[neighbor])

        edge_vector = pos2 - pos1

        if i % 2 == 0:

            position_ratio = 0.35
        else:

            position_ratio = 0.65

        final_label_pos = pos1 + edge_vector * position_ratio

        label_positions[(node, neighbor)] = (final_label_pos[0], final_label_pos[1])

    return label_positions

def create_enhanced_animation(graph: Any, positions_history: List[Dict[int, Tuple]],
                            idleness_history: List[Dict[int, Union[int, float]]],
                            title: str, save_dir: str, frames_to_render: int):
    fig, ax = plt.subplots(figsize=(12, 10))  # type: ignore.

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.nodes)
    nx_graph.add_edges_from(graph.edges)
    node_pos = graph.node_positions

    node_colors = ['lightblue'] * len(graph.nodes)
    nodes = nx.draw_networkx_nodes(nx_graph, node_pos, ax=ax, node_color=node_colors, node_size=500)  # type: ignore.
    nx.draw_networkx_edges(nx_graph, node_pos, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(nx_graph, node_pos, ax=ax, font_size=10)

    agent_colors = plt.cm.get_cmap('jet', len(positions_history[0]))
    agent_plots: Dict[int, Any] = {}
    for i in positions_history[0].keys():
        line, = ax.plot([], [], 'o', markersize=12, color=agent_colors(i),
                       markeredgecolor='black', markeredgewidth=2)  # type: ignore.
        agent_plots[i] = line

    def update(frame: int) -> List[Any]:
        ax.set_title(f"{title} - Time Step: {frame}", fontsize=14)  # type: ignore.

        if frame < len(idleness_history):
            current_idleness = idleness_history[frame]
            max_idleness = max(current_idleness.values()) if current_idleness else 1

            node_colors = []
            for node in graph.nodes:
                idleness = current_idleness.get(node, 0)

                color_intensity = min(1.0, idleness / max_idleness) if max_idleness > 0 else 0
                color = plt.cm.RdYlGn_r(color_intensity)  # type: ignore.
                node_colors.append(color)

            nodes.set_facecolor(node_colors)

        current_positions = positions_history[frame]
        for agent_id, pos_info in current_positions.items():
            start_node, end_node, progress = pos_info

            progress = max(0.0, min(1.0, float(progress)))

            start_pos = node_pos[start_node]
            end_pos = node_pos[end_node]

            x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

            agent_plots[agent_id].set_data(x, y)

        return [nodes] + list(agent_plots.values())

    ani = FuncAnimation(fig, update, frames=frames_to_render, blit=True, interval=100)

    file_path = os.path.join(save_dir, 'enhanced_patrol_animation.gif')
    ani.save(file_path, writer='pillow', fps=10)
    plt.close()
    print(f"Enhanced animation saved to {file_path}")
    print(f"Enhanced animation saved to {file_path}")

def plot_evaluation_max_latency(worst_history: List[float], cumulative_worst_history: List[float],
                                title: str, save_dir: str, use_normalization: bool = True):
    if not worst_history or not cumulative_worst_history:
        print("No max latency history available, skip plotting max latency curve.")
        return

    plt.figure(figsize=(10, 6))
    x_range = np.arange(1, len(worst_history) + 1)

    plt.plot(x_range, worst_history, color='r', linewidth=1.8, alpha=0.8,
             label='Instantaneous Max Idleness')

    plt.plot(x_range, cumulative_worst_history, 'k--', linewidth=2.0, alpha=1.0,
             label='Cumulative Worst Idleness (from T onwards)')

    ylabel = 'Normalized Max Idleness' if use_normalization else 'Max Idleness'

    plt.title(f"{title} - Cumulative Max Latency", fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    file_path = os.path.join(save_dir, 'evaluation_max_latency_cumulative.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Max latency evaluation chart saved to {file_path}")
