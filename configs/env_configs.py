"""Environment parameter dataclasses.

Training parallelism belongs to ``TrainerConfig``, while ``env_type`` belongs to
``ExperimentConfig``. Environment configs contain only simulation behavior and
observation or reward preprocessing settings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from sensai.util.string import ToStringMixin

@dataclass(kw_only=True)
class EnvConfig(ToStringMixin):
    graph_path: str = "graphs/simple_TSP_12.json"

    enable_wait: bool = False
    deltaT: float = 0.5

    num_agents: int = 3
    speeds: Optional[List] = None
    init_positions: Optional[List] = None

    episode_len: int = 300
    max_time_for_obs: Optional[float] = None

    norm_reward: bool = False

    norm_obs: bool = False

    edge_time_jitter_mode: str = "none"
    edge_time_jitter_frac: float = 0.1
    edge_time_jitter_seed: Optional[int] = None

    custom_configs: Optional[Dict] = None
