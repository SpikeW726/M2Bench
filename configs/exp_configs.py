"""顶层实验配置：整合分发字段、子 Config 和实验元信息。"""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sensai.util.string import ToStringMixin

from configs.env_configs import EnvConfig
from configs.algo_configs import AlgoParams, MAPPOParams
from configs.training_configs import TrainerConfig, OnPolicyTrainerConfig
from configs.network_configs import NetworkConfig


@dataclass(kw_only=True)
class ExperimentConfig(ToStringMixin):
    """
    顶层实验配置。

    包含三部分内容：
    1. 分发字段 — 决定使用哪个算法 / 环境 / 网络 / 训练器
    2. 子 Config — 各模块的具体参数
    3. 实验元信息 — 路径、日志、wandb 等
    """

    # ---- 分发字段 ----
    algo_name: str = "mappo"
    env_type: str = "masup"
    # 网络分发: actor / critic / q_network 三路独立配置
    actor_type: Optional[str] = "mlp"     # "mlp" | "rnn" | None
    critic_type: Optional[str] = "mlp"    # "mlp" | "rnn" | None
    q_type: Optional[str] = None          # "mlp" | "rnn" | None

    # ---- 子 Config ----
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoParams = field(default_factory=MAPPOParams)
    training: TrainerConfig = field(default_factory=OnPolicyTrainerConfig)
    # 网络配置: 各组件独立
    actor: Optional[NetworkConfig] = None
    critic: Optional[NetworkConfig] = None
    q_network: Optional[NetworkConfig] = None

    # ---- 实验元信息 ----
    exp_name: str = "default"
    graph_name: str = "TSP12"
    track_wandb: bool = True
    wandb_project: str = "MAP-RL"
    actor_path: Optional[str] = None
    critic_path: Optional[str] = None

    def __post_init__(self):
        """初始化时锁定时间戳，保证 run_name 在整个生命周期内不变。"""
        self._timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def run_name(self) -> str:
        return f"{self.algo_name}-{self.env_type}-{self.graph_name}-{self._timestamp}"

    @property
    def save_dir(self) -> Path:
        return Path(f"models/{self.algo_name}-{self.env_type}-{self.graph_name}/{self._timestamp}")

    @property
    def log_dir(self) -> Path:
        return Path(f"runs/{self.run_name}")
