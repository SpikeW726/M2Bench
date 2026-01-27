"""EnvWorker 基类，封装单个环境实例的交互逻辑。"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
import numpy as np
import gymnasium as gym


class EnvWorker(ABC):
    """环境工作器基类，每个 worker 管理一个环境实例。"""
    
    def __init__(self, env_fn: Callable[[], gym.Env]):
        self._env_fn = env_fn
        self.is_closed = False
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """重置环境，返回 (obs, info)。"""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行动作，返回 (obs, reward, terminated, truncated, info)。"""
        pass
    
    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        """获取环境属性。"""
        pass
    
    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        """设置环境属性。"""
        pass
    
    def seed(self, seed: int | None = None) -> list[int] | None:
        """设置随机种子。"""
        action_space = self.get_env_attr("action_space")
        if action_space is not None:
            return action_space.seed(seed)
        return None
    
    @abstractmethod
    def render(self, **kwargs) -> Any:
        """渲染环境。"""
        pass
    
    @abstractmethod
    def close_env(self) -> None:
        """关闭环境（内部实现）。"""
        pass
    
    def close(self) -> None:
        """关闭 worker。"""
        if self.is_closed:
            return
        self.is_closed = True
        self.close_env()
