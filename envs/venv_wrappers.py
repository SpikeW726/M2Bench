"""向量化环境包装器。"""

from typing import Any, List
import numpy as np

from envs.venvs import BaseVectorEnv
from utils.log_utils import RunningMeanStd


class VectorEnvWrapper(BaseVectorEnv):
    """向量化环境包装器基类。"""
    
    def __init__(self, venv: BaseVectorEnv):
        # 不调用 super().__init__，直接代理
        self.venv = venv
    
    def __len__(self) -> int:
        return len(self.venv)
    
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
    
    def set_env_attr(self, key: str, value: Any, env_id=None) -> None:
        self.venv.set_env_attr(key, value, env_id)
    
    def render(self, **kwargs):
        return self.venv.render(**kwargs)
    
    def close(self) -> None:
        self.venv.close()


class VectorEnvNormObs(VectorEnvWrapper):
    """
    观测归一化包装器。
    
    使用 running mean/std 将观测归一化到近似 N(0,1) 分布，有助于稳定训练。
    
    Args:
        venv: 被包装的向量化环境
        update_obs_rms: 是否在 reset/step 时更新统计量
        clip_max: 归一化后的裁剪范围，默认 10.0
    """
    
    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        clip_max: float = 10.0,
    ):
        super().__init__(venv)
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd(clip_max=clip_max)
    
    def reset(self, env_id=None, **kwargs):
        obs, info = self.venv.reset(env_id, **kwargs)
        if self.update_obs_rms:
            self.obs_rms.update(obs)
        return self.obs_rms.norm(obs), info
    
    def step(self, actions, env_id=None):
        obs, rew, term, trunc, info = self.venv.step(actions, env_id)
        if self.update_obs_rms:
            self.obs_rms.update(obs)
        return self.obs_rms.norm(obs), rew, term, trunc, info
    
    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """设置观测统计量（用于加载已保存的统计）。"""
        self.obs_rms = obs_rms
    
    def get_obs_rms(self) -> RunningMeanStd:
        """获取观测统计量（用于保存）。"""
        return self.obs_rms
