"""顶层实验配置：整合分发字段、子 Config 和实验元信息。"""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sensai.util.string import ToStringMixin

from configs.env_configs import EnvConfig
from configs.algo_configs import AlgoParams, MAPPOParams
from configs.training_configs import TrainerConfig, OnPolicyTrainerConfig
from configs.network_configs import NetworkConfig, MLPNetworkConfig


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
    network_type: str = "mlp"

    # ---- 子 Config ----
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoParams = field(default_factory=MAPPOParams)
    training: TrainerConfig = field(default_factory=OnPolicyTrainerConfig)
    network: NetworkConfig = field(default_factory=MLPNetworkConfig)

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
        return f"{self.graph_name}_{self.exp_name}_{self._timestamp}"

    @property
    def save_dir(self) -> Path:
        return Path(f"models/{self.algo_name}-{self.graph_name}/{self.run_name}")

    @property
    def log_dir(self) -> Path:
        return Path(f"runs/{self.run_name}")
