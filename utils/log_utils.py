import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

class RunningMeanStd:
    """Maintain online mean and variance for observation normalization.

    Batch statistics are merged with the parallel variance algorithm. Normalized
    values are clipped to ``[-clip_max, clip_max]`` when clipping is enabled.
    """

    def __init__(
        self,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        clip_max: float | None = 10.0,
    ):
        self.mean = mean
        self.var = std  # Initialize the variance from std.
        self.count = 0
        self.clip_max = clip_max
        self.eps = np.finfo(np.float32).eps.item()

    def update(self, data: np.ndarray) -> None:
        batch_mean = np.mean(data, axis=0)
        batch_var = np.var(data, axis=0)
        batch_count = len(data)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Parallel algorithm for combining statistics.
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

    def norm(self, data: np.ndarray) -> np.ndarray:
        result = (data - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max is not None:
            result = np.clip(result, -self.clip_max, self.clip_max)
        return result

@dataclass
class IdlenessMetrics:
    """Idleness metrics recorded at one simulation time.

    IGI is the mean weighted node idleness, AGI is its time-weighted mean, IWI
    is the maximum weighted node idleness, and WI is the maximum IWI observed
    so far. ``wait_ratio`` is the fraction of actions that select waiting.
    """

    igi: float = 0.0
    agi: float = 0.0
    iwi: float = 0.0
    wi: float = 0.0
    step: int = 0
    time: float = 0.0
    wait_ratio: float = 0.0  # Fraction of wait actions.

class EpisodeMetricsTracker:
    """Track idleness metrics for one episode.

    Evaluation mode stores the complete history for plotting and export.
    Training mode updates only the current value to avoid per-tick allocations;
    history-dependent methods are therefore unavailable in that mode.
    """

    def __init__(self, training_mode: bool = False):
        self.training_mode = training_mode
        self.history: List[IdlenessMetrics] = []
        self._current: IdlenessMetrics = IdlenessMetrics()
        self._igi_time_weighted_sum: float = 0.0  # Time-weighted accumulator for AGI.

    def reset(self):
        self.history = []
        self._current = IdlenessMetrics()
        self._igi_time_weighted_sum = 0.0

    @property
    def has_data(self) -> bool:
        return bool(self.history) or self._current.step > 0

    def record(self, weighted_idleness: np.ndarray, step: int, time: float):
        """Record metrics from ``phi[i] * idleness[i]`` for every node.

        AGI accumulates ``IGI * dt`` using the actual interval since the previous
        record, which supports both fixed-step and event-driven simulation.
        """

        if weighted_idleness.size == 0:
            return

        igi = float(weighted_idleness.mean())
        iwi = float(weighted_idleness.max())

        prev_time = self._current.time
        dt = time - prev_time
        self._igi_time_weighted_sum += igi * dt
        agi = self._igi_time_weighted_sum / time if time > 0 else 0.0
        wi = max(iwi, self._current.wi)

        if self.training_mode:
            # Update in place to avoid allocations and list growth.
            self._current.igi = igi
            self._current.agi = agi
            self._current.iwi = iwi
            self._current.wi = wi
            self._current.step = step
            self._current.time = time
        else:
            metrics = IdlenessMetrics(
                igi=igi,
                agi=agi,
                iwi=iwi,
                wi=wi,
                step=step,
                time=time,
            )
            self.history.append(metrics)
            self._current = metrics

    @property
    def current(self) -> IdlenessMetrics:
        return self._current

    def get_history_dict(self) -> Dict[str, List[float]]:
        return {
            'step': [m.step for m in self.history],
            'time': [m.time for m in self.history],
            'igi': [m.igi for m in self.history],
            'agi': [m.agi for m in self.history],
            'iwi': [m.iwi for m in self.history],
            'wi': [m.wi for m in self.history],
        }

    def plot(self, save_path: str = None, show: bool = True, use_time_axis: bool = False):
        if not self.history:
            print("No data to plot")
            return

        data = self.get_history_dict()
        x_key = 'time' if use_time_axis else 'step'
        x_label = 'Time' if use_time_axis else 'Step'
        x = data[x_key]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Episode Idleness Metrics', fontsize=14)

        # IGI.
        axes[0, 0].plot(x, data['igi'], 'b-', linewidth=1)
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel('IGI')
        axes[0, 0].set_title('Instantaneous Graph Idleness (Mean)')
        axes[0, 0].grid(True, alpha=0.3)

        # AGI.
        axes[0, 1].plot(x, data['agi'], 'g-', linewidth=1)
        axes[0, 1].set_xlabel(x_label)
        axes[0, 1].set_ylabel('AGI')
        axes[0, 1].set_title('Average Graph Idleness')
        axes[0, 1].grid(True, alpha=0.3)

        # IWI.
        axes[1, 0].plot(x, data['iwi'], 'r-', linewidth=1)
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel('IWI')
        axes[1, 0].set_title('Instantaneous Worst Idleness (Max)')
        axes[1, 0].grid(True, alpha=0.3)

        # WI.
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
        import csv

        data = self.get_history_dict()
        keys = list(data.keys())

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(self.history)):
                writer.writerow([data[k][i] for k in keys])

        print(f"Metrics exported to {path}")

def aggregate_episode_metrics(
    metrics_history: List[Dict[str, List[float]]],
    metric_keys: Optional[List[str]] = None,
    n_interp: int = 1000,
) -> Dict[str, Any]:
    """Aggregate episodes on a common simulation-time grid.

    Event-driven episodes generally record at different times. Each metric is
    therefore interpolated onto ``n_interp`` shared points from zero to the
    latest episode end time before computing the mean and standard deviation.
    Values outside an episode's recorded range use its endpoint value.
    """

    if not metrics_history:
        return {}

    if metric_keys is None:
        first_ep = metrics_history[0]
        metric_keys = [k for k in first_ep.keys() if k not in ['step', 'time']]

    max_time = max(m['time'][-1] for m in metrics_history if m['time'])
    time_grid = np.linspace(0.0, max_time, n_interp)

    aggregated = {}
    interp_arrays = {key: [] for key in metric_keys}
    for ep_metrics in metrics_history:
        t = np.array(ep_metrics['time'], dtype=float)
        for key in metric_keys:
            vals = np.array(ep_metrics[key], dtype=float)

            interp_arrays[key].append(np.interp(time_grid, t, vals))

    for key in metric_keys:
        arr = np.array(interp_arrays[key])  # shape: (n_episodes, n_interp).
        aggregated[f'{key}_mean'] = np.mean(arr, axis=0)
        aggregated[f'{key}_std'] = np.std(arr, axis=0)

    aggregated['time_x'] = time_grid.tolist()
    return aggregated

def save_eval_plot_curves_csv(
    aggregated_data: Dict[str, Any],
    csv_path: str,
    plot_metric_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """Export the interpolated mean and standard-deviation curves used in plots."""

    if not aggregated_data:
        return None
    x_key = "time_x" if "time_x" in aggregated_data else (
        "step_x" if "step_x" in aggregated_data else None
    )
    if x_key is None:
        return None
    x_col = "time" if x_key == "time_x" else "step"
    x_vals = aggregated_data[x_key]
    n = len(x_vals)
    if n == 0:
        return None

    if plot_metric_keys is None:
        plot_metric_keys = [
            k[:-5]
            for k in aggregated_data
            if k.endswith("_mean") and not k.startswith(x_col)
        ]
        plot_metric_keys = sorted(set(plot_metric_keys))

    triples: List[tuple] = []
    for b in plot_metric_keys:
        mk, sk = f"{b}_mean", f"{b}_std"
        if mk in aggregated_data and sk in aggregated_data:
            triples.append((b, mk, sk))

    if not triples:
        return None

    cols = [x_col] + [c for _, mk, sk in triples for c in (mk, sk)]
    out = Path(csv_path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = [float(x_vals[i])]
            for _, mk, sk in triples:
                row.append(float(aggregated_data[mk][i]))
                row.append(float(aggregated_data[sk][i]))
            w.writerow(row)

    print(f"Plot data CSV saved to: {out}")
    return str(out)

def plot_aggregated_metrics(
    aggregated_data: Dict[str, Any],
    metric_configs: Optional[List[tuple]] = None,
    title: str = "Multi-Episode Metrics Evaluation",
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (14, 10)
):
    """Plot mean and standard-deviation bands for aggregated episode metrics."""

    if not aggregated_data:
        print("No data to plot")
        return

    if metric_configs is None:
        metric_configs = [
            ('igi', 'Instantaneous Graph Idleness (IGI)', 'IGI'),
            ('agi', 'Average Graph Idleness (AGI)', 'AGI'),
            ('iwi', 'Instantaneous Worst Idleness (IWI)', 'IWI'),
            ('wi', 'Worst Idleness (WI)', 'WI')
        ]

    plot_metric_keys = [t[0] for t in metric_configs]

    n_metrics = len(metric_configs)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Title and subtitle.
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

        final_mean = mean[-1]
        final_std = std[-1]
        summary_text = f'Final: {final_mean:.4f} ± {final_std:.4f}'

        ax.text(0.98, 0.98, summary_text,
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='gray', linewidth=1))

    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        csv_path = str(Path(save_path).with_name(f"{Path(save_path).stem}_plot_data.csv"))
        save_eval_plot_curves_csv(aggregated_data, csv_path, plot_metric_keys=plot_metric_keys)

    if show:
        plt.show()
    else:
        plt.close()
