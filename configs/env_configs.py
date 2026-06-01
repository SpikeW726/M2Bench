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

    # 奖励预处理开关：True 时使用 VectorEnvNormReward（只除 std，不减 mean）
    # False 时保持当前 pipeline 行为不变（默认）
    norm_reward: bool = False

    # 观测归一化开关：True 时使用 VectorEnvNormObs（running mean/std → clip ±10）
    # 适用于 obs 特征量级大（如 SUNS/BBLA 中未归一化的空闲度）的场景
    # MASUP 家族使用自身的观测预处理，不建议开启
    norm_obs: bool = False

    # 边上运动时间随机扰动
    # edge_time_jitter_mode 取值：
    #   "none" — 不扰动（默认）
    #   "dual" — 物理到达用扰动时间，obs 仍暴露名义时间
    #   "full" — obs 和物理到达均使用扰动后的真实时间
    edge_time_jitter_mode: str = "none"
    edge_time_jitter_frac: float = 0.1          # 扰动幅度 ε，实际时间 ∈ [T*(1-ε), T*(1+ε)]
    edge_time_jitter_seed: Optional[int] = None  # 随机种子；None 表示非确定性

    custom_configs: Optional[Dict] = None
