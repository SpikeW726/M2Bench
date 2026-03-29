"""环境参数 dataclass。

注意：num_envs 和 use_subproc 属于训练并行度参数，
已移至 TrainerConfig；env_type 由 ExperimentConfig 管理。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class EnvConfig(ToStringMixin):
    """环境配置"""
    graph_path: str = "graphs/simple_TSP_12.json"

    enable_wait: bool = False
    deltaT: float = 0.5

    num_agents: int = 3
    speeds: Optional[List] = None
    init_positions: Optional[List] = None

    episode_len: int = 300             # step 数或最大时间 (取决于 truncate_by_time)
    max_time_for_obs: Optional[float] = None

    # 边上运动时间随机扰动（双轨设计：obs 始终用名义时间，物理到达由扰动时间驱动）
    edge_time_jitter: bool = False              # 是否开启扰动
    edge_time_jitter_frac: float = 0.1         # 扰动幅度 ε，实际时间 ∈ [T*(1-ε), T*(1+ε)]
    edge_time_jitter_seed: Optional[int] = None # 随机种子；None 表示非确定性

    custom_configs: Optional[Dict] = None
