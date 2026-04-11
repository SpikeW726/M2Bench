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
        """设置各 action_space 的随机种子。

        单智能体 Gymnasium：action_space 为 Space，直接 .seed。
        PettingZoo / MA：action_space 为 (agent) -> Space，需对每个 agent 分别 .seed。
        """
        results: list[int] = []

        def _collect_seed_ret(r) -> None:
            if r is None:
                return
            if isinstance(r, list):
                results.extend(r)
            elif isinstance(r, tuple):
                results.extend(r)
            else:
                results.append(r)

        # 多智能体：action_spaces 可能是方法，得到 dict[str, Space]
        action_spaces = self.get_env_attr("action_spaces")
        if action_spaces is not None and callable(action_spaces) and not hasattr(
            action_spaces, "seed"
        ):
            try:
                action_spaces = action_spaces()
            except TypeError:
                action_spaces = None
        if isinstance(action_spaces, dict):
            for sp in action_spaces.values():
                if sp is not None and hasattr(sp, "seed"):
                    _collect_seed_ret(sp.seed(seed))
            return results or None

        action_space = self.get_env_attr("action_space")
        if action_space is None:
            return None
        if hasattr(action_space, "seed"):
            _collect_seed_ret(action_space.seed(seed))
            return results or None
        # MA: bound method env.action_space(agent)，仅同进程 Dummy worker 可用
        if callable(action_space):
            agents = self.get_env_attr("possible_agents")
            if agents is None:
                agents = self.get_env_attr("agents")
            if agents:
                for agent in agents:
                    sp = action_space(agent)
                    if sp is not None and hasattr(sp, "seed"):
                        _collect_seed_ret(sp.seed(seed))
                return results or None
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
