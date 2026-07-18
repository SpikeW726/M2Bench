import math
from dataclasses import dataclass
from typing import Optional

from sensai.util.string import ToStringMixin

@dataclass(kw_only=True)
class TrainerConfig(ToStringMixin):
    num_envs: int = 16
    use_subproc: bool = True           # SubprocVectorEnv vs DummyVectorEnv.
    num_steps: int = 1024
    max_iterations: int = 500
    total_steps: Optional[int] = None
    save_interval: int = 100
    verbose: bool = True
    # Evaluation.

    eval_interval: int = 0
    eval_episodes: int = 5

    @property
    def step_per_iteration(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def effective_max_iterations(self) -> int:
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
    minibatch_size: int = 4096
    update_epochs: int = 10

    def compute_optimizer_steps_per_iter(self, num_agents: int = 1) -> int:
        batch_size = self.step_per_iteration * num_agents
        num_minibatches = max(1, batch_size // self.minibatch_size)
        return self.update_epochs * num_minibatches

@dataclass(kw_only=True)
class OffPolicyTrainerConfig(TrainerConfig):
    batch_size: int = 256
    warmup_steps: int = 10000
    collect_per_step: int = 1
    update_per_step: int = 1
    buffer_size: int = 100_000
