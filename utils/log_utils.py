import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class IdlenessMetrics:
    """
    参考论文 https://jmvidal.cse.sc.edu/library/santana04a.pdf
    IGI: Instantaneous graph idleness  记录时刻全图节点瞬时idleness的均值 \sum{idleness} / N
    AGI: Average graph idleness  IGI关于时间的均值 \sum_{t=0}^T{IGI(t)} / T
    IWI: Instaneous worst idleness  记录时刻全图节点瞬时idleness的最大值
    WI: Worst idlenss  整个episode中历史最大IWI
    """
    igi: float = 0.0
    agi: float = 0.0
    iwi: float = 0.0
    wi: float = 0.0
    step: int = 0
    time: float = 0.0


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
        self._igi_sum: float = 0.0  # 用于计算 AGI
    
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
        
        # 更新累积指标
        self._igi_sum += igi
        agi = self._igi_sum / time if time > 0 else 0.0
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
