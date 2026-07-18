from typing import Any, List, Optional, Type
import numpy as np

from envs.venvs import BaseVectorEnv
from utils.log_utils import RunningMeanStd as NumpyRMS

def find_vec_wrapper(vec_env: Any, wrapper_type: Type[Any]) -> Optional[Any]:
    cur = vec_env
    seen = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, wrapper_type):
            return cur
        seen.add(id(cur))
        cur = getattr(cur, "venv", None)
    return None

class VectorEnvWrapper(BaseVectorEnv):
    def __init__(self, venv: BaseVectorEnv):
        # Delegate directly without calling super().__init__().
        self.venv = venv

    def __len__(self) -> int:
        return len(self.venv)

    @property
    def is_parallel_env(self) -> bool:
        return self.venv.is_parallel_env

    @property
    def agents(self):
        return self.venv.agents

    @property
    def num_envs(self) -> int:
        return self.venv.num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    @property
    def is_closed(self) -> bool:
        return self.venv.is_closed

    def reset(self, env_id=None, **kwargs):
        return self.venv.reset(env_id, **kwargs)

    def step(self, actions, env_id=None):
        return self.venv.step(actions, env_id)

    def seed(self, seed=None):
        return self.venv.seed(seed)

    def get_env_attr(self, key: str, env_id=None) -> List[Any]:
        return self.venv.get_env_attr(key, env_id)

    def call_env_method(self, method_name: str, *args, env_id=None, **kwargs):
        return self.venv.call_env_method(method_name, *args, env_id=env_id, **kwargs)

    def set_env_attr(self, key: str, value: Any, env_id=None) -> None:
        self.venv.set_env_attr(key, value, env_id)

    def render(self, **kwargs):
        return self.venv.render(**kwargs)

    def close(self) -> None:
        self.venv.close()

class VectorEnvNormObs(VectorEnvWrapper):
    """Normalize vector observations with running per-feature statistics.

    Statistics update only in training mode. Gymnasium observations share one
    accumulator; PettingZoo observations maintain one accumulator per agent.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        clip_max: float = 10.0,
    ):
        super().__init__(venv)
        self.update_obs_rms = update_obs_rms
        self.obs_rms = NumpyRMS(clip_max=clip_max)

    def _to_batch(self, obs) -> "np.ndarray":
        if isinstance(obs, dict):
            # PettingZoo: {agent: (num_envs, obs_dim)} -> (num_envs * num_agents, obs_dim).
            return np.concatenate(list(obs.values()), axis=0)
        return obs

    def _norm_obs(self, obs):
        if isinstance(obs, dict):
            return {k: self.obs_rms.norm(v) for k, v in obs.items()}
        return self.obs_rms.norm(obs)

    def reset(self, env_id=None, **kwargs):
        obs, info = self.venv.reset(env_id, **kwargs)
        if self.update_obs_rms:
            self.obs_rms.update(self._to_batch(obs))
        return self._norm_obs(obs), info

    def step(self, actions, env_id=None):
        obs, rew, term, trunc, info = self.venv.step(actions, env_id)
        if self.update_obs_rms:
            self.obs_rms.update(self._to_batch(obs))
        return self._norm_obs(obs), rew, term, trunc, info

    def set_obs_rms(self, obs_rms: NumpyRMS) -> None:
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> NumpyRMS:
        return self.obs_rms

class VectorEnvNormReward(VectorEnvWrapper):
    """Scale rewards by running standard deviation without subtracting the mean."""

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_rew_rms: bool = True,
        clip_max: float | None = None,
    ):
        super().__init__(venv)
        self.update_rew_rms = update_rew_rms

        self.rew_rms = NumpyRMS(clip_max=clip_max)

    def _collect_reward_batch(self, rew):
        if isinstance(rew, dict):
            parts = []
            for value in rew.values():
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    parts.append(arr)
            if parts:
                return np.concatenate(parts, axis=0)
            return np.asarray([0.0], dtype=np.float32)
        return np.asarray(rew, dtype=np.float32).reshape(-1)

    def _normalize_reward(self, rew, scale: float):
        if isinstance(rew, dict):
            return {k: np.asarray(v, dtype=np.float32) / scale for k, v in rew.items()}
        return np.asarray(rew, dtype=np.float32) / scale

    def step(self, actions, env_id=None):
        obs, rew, term, trunc, info = self.venv.step(actions, env_id)
        if self.update_rew_rms:
            reward_batch = self._collect_reward_batch(rew)
            self.rew_rms.update(reward_batch)
        scale = float(np.sqrt(self.rew_rms.var + self.rew_rms.eps))
        normed = self._normalize_reward(rew, scale)
        return obs, normed, term, trunc, info

    def set_reward_rms(self, rms: NumpyRMS) -> None:
        self.rew_rms = rms

    def get_reward_rms(self) -> NumpyRMS:
        return self.rew_rms
