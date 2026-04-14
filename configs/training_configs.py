"""训练器参数 dataclass。"""

import math
from dataclasses import dataclass
from typing import Optional

from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class TrainerConfig(ToStringMixin):
    """训练器基类配置"""
    num_envs: int = 16
    use_subproc: bool = True           # SubprocVectorEnv vs DummyVectorEnv
    num_steps: int = 1024              # 每个 env 每轮采集的步数
    max_iterations: int = 500          # 训练总轮数（total_steps 为 None 时使用）
    total_steps: Optional[int] = None    # 环境交互步数预算（RL：ceil 推导 max_iterations；Q-table：累计步数达预算即停）
    save_interval: int = 100           # 每隔多少轮保存 checkpoint
    verbose: bool = True
    # Inline eval：每隔 eval_interval 个 iteration 用当前权重跑 eval_episodes 个 episode 评估
    # 0 = 禁用；需在实验 YAML 中提供 eval_config_path 才能生效
    eval_interval: int = 0
    eval_episodes: int = 5

    @property
    def step_per_iteration(self) -> int:
        """每轮迭代的环境总步数 (num_envs * num_steps)"""
        return self.num_envs * self.num_steps

    @property
    def effective_max_iterations(self) -> int:
        """RL 训练实际迭代轮数。

        若设置 total_steps，则 max_iterations = ceil(total_steps / step_per_iteration)，
        保证实际总步数 >= total_steps。
        Q-table：由 QTableTrainer 读取 total_steps 作为累计 env 步数预算；未设 total_steps 时用 max_iterations 控制 episode 数。
        """
        if self.total_steps is not None:
            spi = self.step_per_iteration
            if spi <= 0:
                raise ValueError(
                    "step_per_iteration (num_envs * num_steps) must be positive when total_steps is set"
                )
            return max(1, math.ceil(self.total_steps / spi))
        return self.max_iterations


@dataclass(kw_only=True)
class OnPolicyTrainerConfig(TrainerConfig):
    """On-policy 训练器配置 (PPO, MAPPO 等)"""
    minibatch_size: int = 4096         # algo.update 内的 minibatch 大小
    update_epochs: int = 10            # algo.update 内的 epoch 数

    def compute_optimizer_steps_per_iter(self, num_agents: int = 1) -> int:
        """
        计算每轮迭代的优化器步数，供 LR scheduler 使用。

        需要 num_agents 因为多智能体场景下 batch 大小 = step_per_iteration * num_agents。
        """
        batch_size = self.step_per_iteration * num_agents
        num_minibatches = max(1, batch_size // self.minibatch_size)
        return self.update_epochs * num_minibatches


@dataclass(kw_only=True)
class OffPolicyTrainerConfig(TrainerConfig):
    """Off-policy 训练器配置 (DQN, SAC 等)"""
    batch_size: int = 256
    warmup_steps: int = 10000          # 训练前随机填充 buffer 的步数
    collect_per_step: int = 1          # 每次 collect 的环境步数
    update_per_step: int = 1           # 每次 collect 后的梯度更新次数
    buffer_size: int = 100_000         # ReplayBuffer 容量
