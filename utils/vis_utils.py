import os
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Any, Union
from utils.graph_utils import Graph

def create_nx_layout(map_graph: Graph):
    """
    使用 networkx 的 kamada_kawai_layout 算法创建高质量的节点布局。
    该实现确保了以下两点：
    1. 真实性: 明确使用边的'weight'作为距离度量，使得边的视觉长度与其权重成正比。
    2. 清晰性: 根据节点数量动态缩放整个布局，从根本上解决节点拥挤问题。

    Args:
        map_graph: 自定义的 Graph 对象

    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    # 1. 创建一个 networkx 图对象，并正确地添加带权重的边
    G = nx.Graph()
    for node in map_graph.nodes:
        G.add_node(node)
    
    for node, neighbors in map_graph.adj_list.items():
        for neighbor, weight in neighbors:
            if node < neighbor: # 避免重复添加
                G.add_edge(node, neighbor, weight=weight)
            
    # 2. 计算一个更温和、非线性的动态缩放因子
    #    这可以确保布局随节点数适度增长，而不会增长得过快
    #    从而让visualize_utils.py中的视觉缩放能够生效。
    num_nodes = len(map_graph.nodes)
    scale_factor = 5.0 + np.sqrt(num_nodes)
    
    # 3. 使用 kamada_kawai_layout 计算最终布局
    #    - weight='weight': 关键参数，强制算法将边的权重作为它们的目标长度。
    #    - scale=scale_factor: 将最终生成的坐标进行等比例放大，解决拥挤问题。
    pos = nx.kamada_kawai_layout(G, weight='weight', scale=scale_factor)
    
    return pos 

def _parse_history_input(history: Union[List[float], str]) -> List[float]:
    """支持传入list或包含数值的文件路径。"""
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
    """绘制并保存训练过程中的平均/未归一化空闲度曲线。"""
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
    """绘制并保存训练过程中的平均空闲度和最大空闲度双曲线。"""
    try:
        # 创建包含两个子图的图形
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 绘制平均空闲度曲线
        axes[0].plot(range(1, len(avg_idleness) + 1), avg_idleness, 'b-', linewidth=1)
        ylabel = 'Normalized Average Idleness' if use_normalization else 'Average Idleness'
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(f"{title} ({ylabel})")
        axes[0].grid(True, alpha=0.3)
        
        # 绘制max_latency曲线
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
        # 回退到原始的单图绘制
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
    """横向拼接评估结果的平均/最差空闲度曲线，保存为一张图。"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    # 平均空闲度
    axes[0].plot(avg_history, color='b', linewidth=1.5)
    ylabel_avg = 'Normalized Average Idleness' if use_normalization else 'Average Idleness'
    axes[0].set_title(f"{title} - {ylabel_avg}", fontsize=18)
    axes[0].set_xlabel('Time Step', fontsize=14)
    axes[0].set_ylabel(ylabel_avg, fontsize=14)
    axes[0].grid(True, alpha=0.4)
    # 最差空闲度
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
    """
    均匀降采样帧序列，保留首尾帧以确保动画完整性。
    
    Args:
        frames: 原始帧列表
        max_frames: 最大帧数限制
    Returns:
        降采样后的帧列表
    """
    if max_frames is None or len(frames) <= max_frames:
        return frames
    # 均匀选取 max_frames 个索引，始终包含首尾帧
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
    """
    创建agent移动的动画视频，保存到save_dir。
    Args:
        map_graph: 地图图结构
        agent_positions_history: agent位置历史记录（应为step数×sub_steps_per_step帧）
        total_frames: 总帧数
        algorithm_name: 算法名称
        map_name: 地图名称
        save_dir: 保存目录
        max_frames: 最大帧数限制（None=不限制）
        plot_stem: 可选；评估图 save_plot 的文件名 stem（不含扩展名），拼为
            {algorithm_name}_animation_{map_name}_{plot_stem}.mp4
    """
    # 帧数限制
    if max_frames is not None and len(agent_positions_history) > max_frames:
        print(f"Downsampling animation: {len(agent_positions_history)} → {max_frames} frames")
        agent_positions_history = _downsample_frames(agent_positions_history, max_frames)
        total_frames = len(agent_positions_history)

    print("Starting animation...")

    num_nodes = len(map_graph.nodes)
    # 移除动态画布缩放，使用固定的、合理的尺寸来避免内存溢出
    figure_scale_factor = 1.0 + num_nodes / 30.0

    baseline_nodes = 12.0
    visual_scale_factor = max(0.6, np.sqrt(baseline_nodes / num_nodes)) if num_nodes > 0 else 1.0

    # 计算动态大小
    node_markersize = 28 * visual_scale_factor
    node_fontsize = 14 * visual_scale_factor
    agent_markersize = 18 * visual_scale_factor
    agent_fontsize = 9 * visual_scale_factor
    edge_label_fontsize = 9 * visual_scale_factor

    # Ensure figure dimensions are even numbers (required for H.264 encoding)
    fig_width = 12 * figure_scale_factor
    fig_height = 10 * figure_scale_factor
    # Round to nearest even number
    fig_width = round(fig_width / 2) * 2
    fig_height = round(fig_height / 2) * 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    node_positions = create_nx_layout(map_graph)

    # 绘制边
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:
                pos1 = node_positions[node]
                pos2 = node_positions[neighbor]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.6, linewidth=2)

    # 绘制节点
    for node, pos in node_positions.items():
        ax.plot(pos[0], pos[1], 'o', markersize=node_markersize, color='skyblue', markeredgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=node_fontsize, fontweight='bold')

    # 绘制边权标签
    label_positions = _calculate_label_positions(map_graph, node_positions)
    for (node, neighbor), (label_x, label_y) in label_positions.items():
        weight = map_graph.get_edge_length(node, neighbor)
        if weight is not None:
            ax.text(label_x, label_y, str(weight), ha='center', va='center',
                   fontsize=edge_label_fontsize, fontweight='normal', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=0.5))

    # 设置图形范围 - 根据实际节点位置动态调整
    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]
    x_margin = (max(x_coords) - min(x_coords)) * 0.1
    y_margin = (max(y_coords) - min(y_coords)) * 0.1

    padding = 0.25 * visual_scale_factor
    ax.set_xlim(min(x_coords) - x_margin - padding, max(x_coords) + x_margin + padding)
    ax.set_ylim(min(y_coords) - y_margin - padding, max(y_coords) + y_margin + padding)
    ax.set_aspect('equal')
    ax.set_title(f'{algorithm_name} on {map_name} - Agent Movement', fontsize=16, fontweight='bold')

    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 绘制代表智能体的三角形
    agent_markers = []
    agent_labels = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    # 转换颜色名称为RGB元组，以便计算亮度
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
                continue  # 超出预分配的标记数量，跳过
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
                        interval=167, blit=False, repeat=True)  # 恢复原来的播放速度，约6fps
    
    # 保存动画到save_dir
    if save_dir is None:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)

    stem_part = f"_{plot_stem}" if plot_stem else ""
    base_name = f"{algorithm_name}_animation_{map_name}{stem_part}"
    
    # Choose output format - set this to True for MP4, False for GIF  
    use_mp4_format = True  # 使用MP4输出
    
    if use_mp4_format:
        # MP4 output - higher quality, smaller file size, but slower rendering
        animation_filename = os.path.join(save_dir, f"{base_name}.mp4")
        try:
            # 尝试使用ffmpeg保存MP4
            import matplotlib.animation as animation_mod
            try:
                # 尝试使用FFMpegWriter - 恢复正常帧率
                Writer = animation_mod.FFMpegWriter
                writer = Writer(fps=6, metadata=dict(artist='MARLlib'), bitrate=2000)
                anim.save(animation_filename, writer=writer)
            except Exception:
                # 备选方案：使用基础的ffmpeg
                anim.save(animation_filename, writer='ffmpeg', fps=6, bitrate=2000)
        except Exception as e:
            print(f"MP4保存失败，错误: {e}")
            print("回退到GIF格式...")
            # 回退到GIF格式
            animation_filename = os.path.join(save_dir, f"{base_name}.gif")
            anim.save(animation_filename, writer='pillow', fps=6, dpi=90)
    else:
        # GIF output - faster rendering, larger file size
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
    """
    创建事件驱动的智能体移动动画，考虑实际时间间隔。
    专门用于SUP_MDP等事件驱动的环境，其中每个事件之间的时间间隔可能不同。

    核心思路：
      对于相邻两个事件快照 prev_state 与 curr_state，根据 time_interval 计算
      需要插入的帧数 num_frames，然后用 alpha ∈ (0, 1] 在两者之间逐帧插值。
      alpha 从 1/num_frames 开始（跳过 0），避免与前一事件的末帧重复。

    Args:
        map_graph: 地图图结构
        agent_positions_history: list of dict, 每个元素是
            {agent_id: (start_node, end_node, progress)}
            表示每个事件 **结束时** 智能体的状态。
        time_intervals: list of float, 相邻事件之间的时间间隔
            （长度 = len(agent_positions_history) - 1）
        algorithm_name: 算法名称
        map_name: 地图名称
        save_dir: 保存目录
        max_frames: 最大帧数限制，None 表示不限制。超出时均匀降采样
        plot_stem: 可选，与 create_animation 一致，追加到输出文件名
    """
    print("Starting event-driven animation...")

    if len(agent_positions_history) == 0:
        print("Warning: No agent position history provided.")
        return

    # --- 视频参数 ---
    fps = 6
    frames_per_time_unit = fps  # 1 time unit ≈ 1秒 ≈ 6帧

    # 计算每个事件区间应播放的帧数（至少1帧）
    event_frame_counts = []
    for interval in time_intervals:
        event_frame_counts.append(max(1, int(round(interval * frames_per_time_unit))))

    # --- 生成插值位置序列 ---
    interpolated_positions = []

    # 放入第0个事件的状态作为初始帧
    interpolated_positions.append(agent_positions_history[0])

    def _clamp(v: float) -> float:
        """将 v 限制在 [0, 1] 范围内"""
        return max(0.0, min(1.0, v))

    # 遍历每对相邻事件，在它们之间插值生成中间帧
    for i in range(1, len(agent_positions_history)):
        prev_state = agent_positions_history[i - 1]  # 上一事件结束时的状态
        curr_state = agent_positions_history[i]       # 当前事件结束时的状态
        num_frames = event_frame_counts[i - 1]        # 本区间的帧数

        # 取所有涉及的智能体（并集）
        all_agents = set(prev_state.keys()) | set(curr_state.keys())

        # 生成 num_frames 帧，alpha ∈ (0, 1]，跳过 alpha=0 以避免与前一区间末帧重复
        for j in range(1, num_frames + 1):
            alpha = j / num_frames  # 插值系数，均匀分布在 (0, 1]

            current_frame = {}
            for agent_id in all_agents:
                if agent_id not in curr_state:
                    continue  # 智能体在当前事件中消失，跳过

                curr_start, curr_end, curr_prog = curr_state[agent_id]
                if agent_id in prev_state:
                    prev_start, prev_end, prev_prog = prev_state[agent_id]
                else:
                    # 新出现的智能体，从当前边起点开始
                    prev_start, prev_end, prev_prog = curr_start, curr_end, 0.0

                same_edge = (prev_start == curr_start and prev_end == curr_end)
                prev_on_node = (prev_start == prev_end)
                curr_on_node = (curr_start == curr_end)

                # ── Case 1: 同一条边上移动 ──
                # prev 和 curr 在同一条边上，progress 线性过渡
                if same_edge:
                    prog = _clamp(prev_prog + alpha * (curr_prog - prev_prog))
                    current_frame[agent_id] = (curr_start, curr_end, prog)

                # ── Case 2: 完成旧边到达节点 ──
                # 之前在边 (prev_start→prev_end) 上，现在到达 prev_end 并停留
                elif (not prev_on_node) and curr_on_node and (prev_end == curr_start == curr_end):
                    prog = _clamp(prev_prog + alpha * (1.0 - prev_prog))
                    current_frame[agent_id] = (prev_start, prev_end, prog)

                # ── Case 3: 节点间移动（相邻节点） ──
                # 前后都在节点上，但节点不同（动作掩码保证是相邻节点）
                elif prev_on_node and curr_on_node and (prev_end != curr_end):
                    prog = _clamp(alpha)
                    current_frame[agent_id] = (prev_end, curr_end, prog)

                # ── Case 4: 从节点出发进入新边 ──
                # 之前停在节点上，现在进入了一条新边
                elif prev_on_node and (not curr_on_node) and (prev_end == curr_start):
                    prog = _clamp(alpha * curr_prog)
                    current_frame[agent_id] = (curr_start, curr_end, prog)

                # ── Case 5: 旧边→新边过渡（兜底） ──
                # 之前在边 (prev_start→prev_end) 上，现在切换到新边 (curr_start→curr_end)
                # 分两阶段：前半段完成旧边剩余行程，后半段在新边上前进
                else:
                    prev_remaining = 1.0 - prev_prog  # 旧边剩余比例
                    # 按旧边剩余量与新边已走量的比例，确定分界点
                    total_travel = prev_remaining + curr_prog
                    if total_travel > 0:
                        split = prev_remaining / total_travel
                    else:
                        split = 0.5  # 两端都无移动量，均分

                    if alpha <= split:
                        # 前半段：在旧边上从 prev_prog 走到 1.0
                        local_alpha = alpha / split if split > 0 else 1.0
                        prog = _clamp(prev_prog + local_alpha * prev_remaining)
                        current_frame[agent_id] = (prev_start, prev_end, prog)
                    else:
                        # 后半段：在新边上从 0 走到 curr_prog
                        local_alpha = (alpha - split) / (1.0 - split) if split < 1.0 else 1.0
                        prog = _clamp(local_alpha * curr_prog)
                        current_frame[agent_id] = (curr_start, curr_end, prog)

            interpolated_positions.append(current_frame)

    # 帧数限制（在插值后对结果均匀降采样）
    if max_frames is not None and len(interpolated_positions) > max_frames:
        print(f"Downsampling event-driven animation: {len(interpolated_positions)} → {max_frames} frames")
        interpolated_positions = _downsample_frames(interpolated_positions, max_frames)

    # 调用底层动画函数
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
    """
    根据背景色的亮度决定使用黑色或白色文本以获得最佳对比度。
    Args:
        bg_color_rgb (tuple): 背景色的RGB元组, e.g., (1, 0, 0) for red.
    Returns:
        str: 'white' or 'black'.
    """
    # 计算颜色的感知亮度 (perceived luminance)
    # 公式: Y = 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * bg_color_rgb[0] + 0.587 * bg_color_rgb[1] + 0.114 * bg_color_rgb[2]
    
    # 如果亮度大于0.5，背景是亮的，使用黑色文字；否则使用白色文字。
    return 'black' if luminance > 0.5 else 'white'

def _to_rgb(color_name):
    """将颜色名称转换为RGB元组"""
    import matplotlib.colors as mcolors
    return mcolors.to_rgb(color_name)

def _calculate_label_positions(map_graph, node_positions):
    """
    计算边权标签的位置，使其保持在边上但避免重叠
    - 标签交替地放置在靠近边的一个端点的位置
      而不是都放在中点，以避免重叠
    
    Args:
        map_graph: 地图图结构
        node_positions: 节点位置字典
    
    Returns:
        dict: (node, neighbor) -> (x, y) 标签位置映射
    """
    label_positions = {}
    
    # 收集所有边
    edges = []
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:  # 避免重复
                edges.append((node, neighbor, weight))
    
    # 按节点对排序，以获得一致的交替效果
    edges.sort() 
    
    for i, (node, neighbor, weight) in enumerate(edges):
        pos1 = np.array(node_positions[node])
        pos2 = np.array(node_positions[neighbor])
        
        edge_vector = pos2 - pos1
        
        # 交替标签位置，一个靠近起点，一个靠近终点，避免全部挤在中间
        if i % 2 == 0:
            # 放在离起点35%的位置
            position_ratio = 0.35
        else:
            # 放在离起点65%的位置
            position_ratio = 0.65
            
        final_label_pos = pos1 + edge_vector * position_ratio
        
        label_positions[(node, neighbor)] = (final_label_pos[0], final_label_pos[1])
        
    return label_positions

def create_enhanced_animation(graph: Any, positions_history: List[Dict[int, Tuple]], 
                            idleness_history: List[Dict[int, Union[int, float]]], 
                            title: str, save_dir: str, frames_to_render: int):
    """
    创建增强版巡逻动画，包含节点空闲度的颜色变化。
    为未来支持连续时间步的算法预留接口。
    
    Args:
        graph (Any): 图对象。
        positions_history (List[Dict[int, Tuple]]): 每帧的智能体位置信息。
        idleness_history (List[Dict[int, Union[int, float]]]): 每帧的节点空闲度信息。
        title (str): 动画标题。
        save_dir (str): 保存目录。
        frames_to_render (int): 要渲染的总帧数。
    """
    fig, ax = plt.subplots(figsize=(12, 10))  # type: ignore
    
    # 绘制静态的图结构
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.nodes)
    nx_graph.add_edges_from(graph.edges)
    node_pos = graph.node_positions
    
    # 初始化节点颜色（基于空闲度）
    node_colors = ['lightblue'] * len(graph.nodes)
    nodes = nx.draw_networkx_nodes(nx_graph, node_pos, ax=ax, node_color=node_colors, node_size=500)  # type: ignore
    nx.draw_networkx_edges(nx_graph, node_pos, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(nx_graph, node_pos, ax=ax, font_size=10)

    # 初始化智能体的绘制对象
    agent_colors = plt.cm.get_cmap('jet', len(positions_history[0]))
    agent_plots: Dict[int, Any] = {}
    for i in positions_history[0].keys():
        line, = ax.plot([], [], 'o', markersize=12, color=agent_colors(i), 
                       markeredgecolor='black', markeredgewidth=2)  # type: ignore
        agent_plots[i] = line

    def update(frame: int) -> List[Any]:
        ax.set_title(f"{title} - Time Step: {frame}", fontsize=14)  # type: ignore
        
        # 更新节点颜色（基于空闲度）
        if frame < len(idleness_history):
            current_idleness = idleness_history[frame]
            max_idleness = max(current_idleness.values()) if current_idleness else 1
            
            node_colors = []
            for node in graph.nodes:
                idleness = current_idleness.get(node, 0)
                # 颜色从绿色（低空闲度）到红色（高空闲度）
                color_intensity = min(1.0, idleness / max_idleness) if max_idleness > 0 else 0
                color = plt.cm.RdYlGn_r(color_intensity)  # type: ignore
                node_colors.append(color)
            
            nodes.set_facecolor(node_colors)
        
        # 更新智能体位置
        current_positions = positions_history[frame]
        for agent_id, pos_info in current_positions.items():
            start_node, end_node, progress = pos_info
            
            # 确保progress在[0, 1]范围内
            progress = max(0.0, min(1.0, float(progress)))
            
            start_pos = node_pos[start_node]
            end_pos = node_pos[end_node]
            
            # 计算插值位置（支持连续进度）
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




# --------------------------------------------------------------------------
# 🔥🌟 新增函数: 专门绘制评估时的 Max Latency 曲线 (瞬时值 + 累积值)
# --------------------------------------------------------------------------
def plot_evaluation_max_latency(worst_history: List[float], cumulative_worst_history: List[float], 
                                title: str, save_dir: str, use_normalization: bool = True):
    """
    绘制评估结果中的最大延迟曲线，包含瞬时值和 T 后的累积最差值。
    
    Args:
        worst_history: 瞬时最大延迟（实线，对应 state 里的 maxlatency）。
        cumulative_worst_history: 环境内部计算的 T 之后的累积最差延迟（虚线）。
        title: 图表标题。
        save_dir: 保存目录。
        use_normalization: 是否使用归一化标签。
    """
    if not worst_history or not cumulative_worst_history:
        print("No max latency history available, skip plotting max latency curve.")
        return

    plt.figure(figsize=(10, 6))
    x_range = np.arange(1, len(worst_history) + 1)
    
    # 绘制瞬时最大延迟曲线 (实线)
    plt.plot(x_range, worst_history, color='r', linewidth=1.8, alpha=0.8, 
             label='Instantaneous Max Idleness')
    
    # 绘制累积最差延迟虚线 (虚线)
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