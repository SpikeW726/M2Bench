"""训练器参数 dataclass。"""

from dataclasses import dataclass
from sensai.util.string import ToStringMixin


@dataclass(kw_only=True)
class TrainerConfig(ToStringMixin):
    """训练器基类配置"""
    num_envs: int = 16
    use_subproc: bool = True           # SubprocVectorEnv vs DummyVectorEnv
    num_steps: int = 1024              # 每个 env 每轮采集的步数
    max_iterations: int = 500          # 训练总轮数
    save_interval: int = 100           # 每隔多少轮保存 checkpoint
    verbose: bool = True

    @property
    def step_per_iteration(self) -> int:
        """每轮迭代的环境总步数 (num_envs * num_steps)"""
        return self.num_envs * self.num_steps


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
