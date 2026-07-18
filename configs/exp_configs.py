import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sensai.util.string import ToStringMixin

from configs.env_configs import EnvConfig
from utils.project_paths import DEFAULT_MODELS_DIR, DEFAULT_RUNS_DIR, user_path
from configs.algo_configs import AlgoParams, MAPPOParams
from configs.training_configs import TrainerConfig, OnPolicyTrainerConfig
from configs.network_configs import NetworkConfig

@dataclass(kw_only=True)
class ExperimentConfig(ToStringMixin):
    algo_name: str = "mappo"
    env_type: str = "masup"
    # Configuration.
    actor_type: Optional[str] = "mlp"     # "mlp" | "rnn" | None.
    critic_type: Optional[str] = "mlp"    # "mlp" | "rnn" | None.
    q_type: Optional[str] = None          # "mlp" | "rnn" | None.

    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoParams = field(default_factory=MAPPOParams)
    training: TrainerConfig = field(default_factory=OnPolicyTrainerConfig)

    actor: Optional[NetworkConfig] = None
    critic: Optional[NetworkConfig] = None
    q_network: Optional[NetworkConfig] = None

    exp_name: str = "default"
    graph_name: str = "TSP12"
    track_wandb: bool = True
    wandb_project: str = "MAP-RL"
    seed: Optional[int] = None
    actor_path: Optional[str] = None
    critic_path: Optional[str] = None

    eval_config_path: Optional[str] = None
    models_dir: str = str(DEFAULT_MODELS_DIR)
    runs_dir: str = str(DEFAULT_RUNS_DIR)
    save_dir_override: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        self._timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def run_name(self) -> str:
        return f"{self.algo_name}-{self.env_type}-{self.graph_name}-{self._timestamp}"

    @property
    def save_dir(self) -> Path:
        if self.save_dir_override is not None:
            return user_path(self.save_dir_override)
        return user_path(self.models_dir) / f"{self.algo_name}-{self.env_type}-{self.graph_name}" / self._timestamp

    @property
    def log_dir(self) -> Path:
        return user_path(self.runs_dir) / self.run_name
