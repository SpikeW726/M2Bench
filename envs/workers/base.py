from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
import numpy as np
import gymnasium as gym

class EnvWorker(ABC):
    def __init__(self, env_fn: Callable[[], gym.Env]):
        self._env_fn = env_fn
        self.is_closed = False

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        pass

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def seed(self, seed: int | None = None) -> list[int] | None:
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
        # MA: bound method env.action_space(agent).
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
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.close_env()
