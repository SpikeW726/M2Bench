"""Q-table 算法：纯 numpy 实现，用于复现 BBLA / GBLA / ExGBLA 论文。

QTablePolicy — 单 agent 的 Q-table + epsilon-greedy 动作选择
QTableAlgo   — 多 agent 独立 Q-learning 在线更新 + epsilon 衰减
"""

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from configs.algo_configs import QTableParams


class QTablePolicy:
    """单 agent 的 Q-table，纯 numpy 实现。

    状态编码: tuple(obs) 作为字典 key（适用于小整数离散观测）。
    """

    def __init__(self, action_dim: int, epsilon: float = 1.0):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim, dtype=np.float64)
        )

    def _obs_to_key(self, obs) -> tuple:
        if isinstance(obs, np.ndarray):
            return tuple(obs.flat)
        return (obs,)  # 标量直接包装成单元素 tuple

    def get_q(self, obs: np.ndarray) -> np.ndarray:
        """返回 Q(s, ·)，未见状态初始化为零向量。"""
        return self.q_table[self._obs_to_key(obs)]

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Epsilon-greedy，respect action_mask。"""
        valid = np.where(action_mask)[0]
        if len(valid) == 0:
            return 0
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid))
        q = self.get_q(obs).copy()
        q[~action_mask.astype(bool)] = -np.inf
        return int(np.argmax(q))

    def set_epsilon(self, eps: float):
        self.epsilon = eps

    def save(self, path: str):
        np.save(path, dict(self.q_table))

    def load(self, path: str):
        d = np.load(path, allow_pickle=True).item()
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_dim, dtype=np.float64), d
        )


class QTableAlgo:
    """多 agent 独立 Q-learning，纯 numpy。

    每个 agent 维护独立的 QTablePolicy，update_step 仅在决策步调用。
    """

    def __init__(self, policies: Dict[str, QTablePolicy], params: QTableParams):
        self.policies = policies
        self.params = params
        self.alpha = params.lr
        self.gamma = params.gamma
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay_steps = max(1, int(params.epsilon_decay_steps))
        for pol in self.policies.values():
            pol.set_epsilon(self.epsilon_start)

    def update_step(
        self,
        agent_id: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None,
        gamma_power: Optional[float] = None,
    ):
        """Q-learning 单步在线更新。

        gamma_power 非 None 时用于同步模式: td = reward + gamma_power * max Q(s')，
        其中 reward 已经是累积折扣奖励，gamma_power = γ^(K+1)。
        """
        pol = self.policies[agent_id]
        q = pol.get_q(obs)
        discount = gamma_power if gamma_power is not None else self.gamma

        if done:
            td_target = reward
        else:
            next_q = pol.get_q(next_obs).copy()
            if next_action_mask is not None:
                next_q[~next_action_mask.astype(bool)] = -np.inf
            td_target = reward + discount * np.max(next_q)

        q[action] += self.alpha * (td_target - q[action])

    def update_epsilon(self, global_step: int):
        """按 global-step 线性衰减 epsilon（所有 agent 统一）。"""
        progress = min(1.0, max(0.0, global_step / self.epsilon_decay_steps))
        new_eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
        for pol in self.policies.values():
            pol.set_epsilon(new_eps)

    def get_epsilon(self) -> float:
        return next(iter(self.policies.values())).epsilon

    def save(self, save_dir: str):
        """逐 agent 保存 Q-table 到指定目录。"""
        from pathlib import Path
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for aid, pol in self.policies.items():
            pol.save(str(d / f"{aid}_qtable.npy"))

    def load(self, save_dir: str):
        """从目录加载各 agent 的 Q-table。"""
        from pathlib import Path
        d = Path(save_dir)
        for aid, pol in self.policies.items():
            path = d / f"{aid}_qtable.npy"
            if path.exists():
                pol.load(str(path))
