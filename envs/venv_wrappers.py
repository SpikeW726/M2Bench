"""向量化环境包装器。"""

from typing import Any, List
import numpy as np

from envs.venvs import BaseVectorEnv
from utils.log_utils import RunningMeanStd as NumpyRMS


class VectorEnvWrapper(BaseVectorEnv):
    """向量化环境包装器基类。"""
    
    def __init__(self, venv: BaseVectorEnv):
        # 不调用 super().__init__，直接代理
        self.venv = venv
    
    def __len__(self) -> int:
        return len(self.venv)

    @property
    def is_parallel_env(self) -> bool:
        """透传底层环境类型（Gym / PettingZoo Parallel）。"""
        return self.venv.is_parallel_env

    @property
    def agents(self):
        """透传并行环境 agent 列表（Gym 环境为 None）。"""
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
        """透传调用底层环境方法（如 state()）。"""
        return self.venv.call_env_method(method_name, *args, env_id=env_id, **kwargs)
    
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
        self.obs_rms = NumpyRMS(clip_max=clip_max)
    
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
    
    def set_obs_rms(self, obs_rms: NumpyRMS) -> None:
        """设置观测统计量（用于加载已保存的统计）。"""
        self.obs_rms = obs_rms
    
    def get_obs_rms(self) -> NumpyRMS:
        """获取观测统计量（用于保存）。"""
        return self.obs_rms


class VectorEnvNormReward(VectorEnvWrapper):
    """
    奖励标准差缩放包装器。

    只除以运行标准差，不减均值，保留 MDP 核心逻辑（生存惩罚等绝对语义）。

    Args:
        venv: 被包装的向量化环境
        update_rew_rms: 是否在 step 时更新统计量
        clip_max: 缩放后裁剪范围，None 表示不裁剪
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_rew_rms: bool = True,
        clip_max: float | None = None,
    ):
        super().__init__(venv)
        self.update_rew_rms = update_rew_rms
        # clip_max=None 不裁剪；初始 std=1 保证启动时无缩放
        self.rew_rms = NumpyRMS(clip_max=clip_max)

    def _collect_reward_batch(self, rew):
        """将不同结构的 reward 展平为 1D batch，用于更新 RMS。"""
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
        """按相同标准差缩放 reward，保持输入结构不变（dict 或 ndarray）。"""
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
        """设置奖励统计量（用于加载已保存的统计）。"""
        self.rew_rms = rms

    def get_reward_rms(self) -> NumpyRMS:
        """获取奖励统计量（用于保存）。"""
        return self.rew_rms
