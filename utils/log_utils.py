import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


class RunningMeanStd:
    """
    在线计算数据流的 running mean 和 std。
    
    用于观测归一化，参考：
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    
    Args:
        mean: 初始均值估计
        std: 初始标准差估计
        clip_max: 归一化后的裁剪范围
    """
    
    def __init__(
        self,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        clip_max: float | None = 10.0,
    ):
        self.mean = mean
        self.var = std  # 初始方差 = std
        self.count = 0
        self.clip_max = clip_max
        self.eps = np.finfo(np.float32).eps.item()
    
    def update(self, data: np.ndarray) -> None:
        """用一批数据更新统计量。"""
        batch_mean = np.mean(data, axis=0)
        batch_var = np.var(data, axis=0)
        batch_count = len(data)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Parallel algorithm for combining statistics
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count
        
        self.mean, self.var = new_mean, new_var
        self.count = total_count
    
    def norm(self, data: np.ndarray) -> np.ndarray:
        """归一化数据。"""
        result = (data - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max is not None:
            result = np.clip(result, -self.clip_max, self.clip_max)
        return result

@dataclass
class IdlenessMetrics:
    """
    参考论文 https://jmvidal.cse.sc.edu/library/santana04a.pdf
    IGI: Instantaneous graph idleness  记录时刻全图节点瞬时idleness的均值 \sum{idleness} / N
    AGI: Average graph idleness  IGI关于时间的均值 \sum_{t=0}^T{IGI(t)} / T
    IWI: Instaneous worst idleness  记录时刻全图节点瞬时idleness的最大值
    WI: Worst idlenss  整个episode中历史最大IWI
    wait_ratio: 等待动作占比 (wait_actions / total_actions)
    """
    igi: float = 0.0
    agi: float = 0.0
    iwi: float = 0.0
    wi: float = 0.0
    step: int = 0
    time: float = 0.0
    wait_ratio: float = 0.0  # 等待动作占比


class EpisodeMetricsTracker:
    """
    Episode 级别的指标追踪器
    
    记录每个 step 的 idleness 指标，支持：
    - 历史数据存储
    - Episode 结束后的可视化
    - 导出到文件
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置历史记录"""
        self.history: List[IdlenessMetrics] = []
        self._igi_time_weighted_sum: float = 0.0  # 时间加权累积，用于计算 AGI
    
    def record(self, node_idleness: Dict[int, float], step: int, time: float):
        """
        记录当前 step 的指标
        
        Args:
            node_idleness: 当前所有节点的空闲度 {node_id: idleness}
            step: 当前步数
            time: 当前时间
        """
        idleness_values = list(node_idleness.values())
        n_nodes = len(idleness_values)
        
        if n_nodes == 0:
            return
        
        # 计算瞬时指标
        igi = sum(idleness_values) / n_nodes
        iwi = max(idleness_values)
        
        # 时间加权累积：乘以本次 tick 的实际时间间隔
        # 保证固定步长（dt=1.0）和事件驱动（dt 可变）下 AGI 均为正确的时间加权平均
        prev_time = self.history[-1].time if self.history else 0.0
        dt = time - prev_time
        self._igi_time_weighted_sum += igi * dt
        agi = self._igi_time_weighted_sum / time if time > 0 else 0.0
        wi = max(iwi, self.history[-1].wi if self.history else 0.0)
        
        metrics = IdlenessMetrics(
            igi=igi,
            agi=agi,
            iwi=iwi,
            wi=wi,
            step=step,
            time=time
        )
        self.history.append(metrics)
    
    @property
    def current(self) -> IdlenessMetrics:
        """返回最新的指标"""
        return self.history[-1] if self.history else IdlenessMetrics()
    
    def get_history_dict(self) -> Dict[str, List[float]]:
        """
        返回字典格式的历史数据，方便绘图
        
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
        return {
            'step': [m.step for m in self.history],
            'time': [m.time for m in self.history],
            'igi': [m.igi for m in self.history],
            'agi': [m.agi for m in self.history],
            'iwi': [m.iwi for m in self.history],
            'wi': [m.wi for m in self.history],
        }
    
    def plot(self, save_path: str = None, show: bool = True, use_time_axis: bool = False):
        """
        绘制 episode 指标曲线
        
        Args:
            save_path: 保存路径（可选）
            show: 是否显示图形
            use_time_axis: True 使用时间作为 x 轴，False 使用 step
        """
        if not self.history:
            print("No data to plot")
            return
        
        data = self.get_history_dict()
        x_key = 'time' if use_time_axis else 'step'
        x_label = 'Time' if use_time_axis else 'Step'
        x = data[x_key]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Episode Idleness Metrics', fontsize=14)
        
        # IGI
        axes[0, 0].plot(x, data['igi'], 'b-', linewidth=1)
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel('IGI')
        axes[0, 0].set_title('Instantaneous Graph Idleness (Mean)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # AGI
        axes[0, 1].plot(x, data['agi'], 'g-', linewidth=1)
        axes[0, 1].set_xlabel(x_label)
        axes[0, 1].set_ylabel('AGI')
        axes[0, 1].set_title('Average Graph Idleness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # IWI
        axes[1, 0].plot(x, data['iwi'], 'r-', linewidth=1)
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel('IWI')
        axes[1, 0].set_title('Instantaneous Worst Idleness (Max)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # WI
        axes[1, 1].plot(x, data['wi'], 'm-', linewidth=1)
        axes[1, 1].set_xlabel(x_label)
        axes[1, 1].set_ylabel('WI')
        axes[1, 1].set_title('Worst Idleness (Historical Max)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def to_csv(self, path: str):
        """导出历史数据到 CSV"""
        import csv
        
        data = self.get_history_dict()
        keys = list(data.keys())
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(self.history)):
                writer.writerow([data[k][i] for k in keys])
        
        print(f"Metrics exported to {path}")


# ==================== 跨 Episode 聚合可视化工具 ====================

def aggregate_episode_metrics(
    metrics_history: List[Dict[str, List[float]]],
    metric_keys: Optional[List[str]] = None,
    n_interp: int = 1000,
) -> Dict[str, Any]:
    """
    聚合多个 episode 的 metrics 历史数据。

    以真实仿真时间（'time' 字段）为 x 轴，将各 episode 插值到统一时间网格后计算
    均值与标准差，解决 event-driven 环境各 episode 步数/时刻不对齐的问题。

    Args:
        metrics_history: 多个 episode 的 metrics 历史列表，
            每个元素格式为 {'step': [...], 'time': [...], 'igi': [...], ...}
        metric_keys: 需要聚合的指标键；None 时自动排除 'step' / 'time'。
        n_interp: 统一时间网格的点数（默认 1000）。

    Returns:
        聚合后的数据字典，包含：
        - time_x: 统一的时间网格（float 列表，长度 n_interp）
        - {metric}_mean: 每个指标在时间网格上的均值
        - {metric}_std: 每个指标在时间网格上的标准差
    """
    if not metrics_history:
        return {}

    if metric_keys is None:
        first_ep = metrics_history[0]
        metric_keys = [k for k in first_ep.keys() if k not in ['step', 'time']]

    # 统一时间网格：0 到所有 episode 最大结束时间
    max_time = max(m['time'][-1] for m in metrics_history if m['time'])
    time_grid = np.linspace(0.0, max_time, n_interp)

    # 将每个 episode 的各指标插值到统一时间网格
    aggregated = {}
    interp_arrays = {key: [] for key in metric_keys}
    for ep_metrics in metrics_history:
        t = np.array(ep_metrics['time'], dtype=float)
        for key in metric_keys:
            vals = np.array(ep_metrics[key], dtype=float)
            # np.interp 在 t 范围外自动用端点值填充（左填首值，右填末值）
            interp_arrays[key].append(np.interp(time_grid, t, vals))

    for key in metric_keys:
        arr = np.array(interp_arrays[key])  # shape: (n_episodes, n_interp)
        aggregated[f'{key}_mean'] = np.mean(arr, axis=0)
        aggregated[f'{key}_std'] = np.std(arr, axis=0)

    aggregated['time_x'] = time_grid.tolist()
    return aggregated


def plot_aggregated_metrics(
    aggregated_data: Dict[str, Any],
    metric_configs: Optional[List[tuple]] = None,
    title: str = "Multi-Episode Metrics Evaluation",
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (14, 10)
):
    """
    绘制聚合后的 metrics（均值线 + 标准差阴影区域）
    
    Args:
        aggregated_data: 由 aggregate_episode_metrics() 返回的聚合数据
        metric_configs: 要绘制的指标配置列表，每个元素为 (key, title, label)
            如果为 None, 使用默认的 idleness metrics 配置
        title: 图表标题
        subtitle: 副标题，用于显示额外信息（如图名、智能体数、平均时间等）
        save_path: 保存路径（可选）
        show: 是否显示图形
        figsize: 图形大小
    """
    if not aggregated_data:
        print("No data to plot")
        return
    
    # 默认配置：idleness metrics
    if metric_configs is None:
        metric_configs = [
            ('igi', 'Instantaneous Graph Idleness (IGI)', 'IGI'),
            ('agi', 'Average Graph Idleness (AGI)', 'AGI'),
            ('iwi', 'Instantaneous Worst Idleness (IWI)', 'IWI'),
            ('wi', 'Worst Idleness (WI)', 'WI')
        ]
    
    # 创建子图
    n_metrics = len(metric_configs)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Title and subtitle
    if subtitle:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.text(0.52, 0.95, subtitle, ha='center', va='top', fontsize=10, color='gray')
    else:
        fig.suptitle(title, fontsize=14)
    
    x_data = aggregated_data.get('time_x', aggregated_data.get('step_x'))
    x_label = 'Simulation Time (s)' if 'time_x' in aggregated_data else 'Step'

    for ax, (metric_key, metric_title, metric_label) in zip(axes, metric_configs):
        mean_key = f'{metric_key}_mean'
        std_key = f'{metric_key}_std'
        
        if mean_key not in aggregated_data or std_key not in aggregated_data:
            print(f"Warning: {mean_key} or {std_key} not found in aggregated_data")
            continue
        
        mean = aggregated_data[mean_key]
        std = aggregated_data[std_key]
        
        ax.plot(x_data, mean, linewidth=2, label='Mean')
        ax.fill_between(x_data, mean - std, mean + std, alpha=0.3, label='±1 Std')
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 在图上标注最终的均值和标准差
        # mean[-1] 和 std[-1] 是所有 episode 在最后一个时间步的统计值
        # 由于 padding 使用最后一个值，这等价于所有 episode 最终值的统计
        final_mean = mean[-1]
        final_std = std[-1]
        summary_text = f'Final: {final_mean:.4f} ± {final_std:.4f}'
        
        # 将文本放在图的右上角，使用半透明背景框
        ax.text(0.98, 0.98, summary_text,
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='gray', linewidth=1))
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()