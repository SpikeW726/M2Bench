"""网络架构参数 dataclass。"""

from dataclasses import dataclass, field
from typing import List
from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class NetworkConfig(ToStringMixin):
    """网络配置基类"""
    pass


@dataclass(kw_only=True)
class MLPNetworkConfig(NetworkConfig):
    """MLP 网络配置"""
    actor_hidden: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])
