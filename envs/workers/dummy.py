"""DummyEnvWorker: 在主进程中顺序执行环境，用于调试或单环境场景。"""

from typing import Any, Callable, Tuple
import numpy as np
import gymnasium as gym

from envs.workers.base import EnvWorker


class DummyEnvWorker(EnvWorker):
    """顺序执行的环境 worker，直接在主进程中运行。"""
    
    def __init__(self, env_fn: Callable[[], gym.Env]):
        super().__init__(env_fn)
        self.env = env_fn()
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return self.env.step(action)
    
    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env.unwrapped, key, None)
    
    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)
    
    def seed(self, seed: int | None = None) -> list[int] | None:
        super().seed(seed)
        # 尝试调用 env.seed()，如果不存在则通过 reset 设置
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        else:
            self.env.reset(seed=seed)
            return [seed] if seed is not None else None
    
    def render(self, **kwargs) -> Any:
        return self.env.render(**kwargs)
    
    def close_env(self) -> None:
        self.env.close()
