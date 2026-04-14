"""RL Trainers: coordinate Collector and Algorithm for training."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import time
import numpy as np

from algorithms.algorithm_base import BaseAlgorithm, ActorCriticOnPolicyAlgo, TrainingStats
from configs.training_configs import (
    TrainerConfig, OnPolicyTrainerConfig, OffPolicyTrainerConfig,
)
from data.collector import BaseCollector
from data.batch import CollectResult


class BaseTrainer(ABC):
    """
    训练器基类。

    Terminology:
        - iteration: 一次 collect-update 循环
        - epoch: update 内部对同一批数据的遍历轮数（由 algorithm 控制）
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        config: TrainerConfig,
        # Callbacks（运行时对象，不适合放进 config）
        save_checkpoint_fn: Optional[Callable[[int], None]] = None,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        logger: Optional[Any] = None,
        # Inline eval callback：() -> Dict[str, float]，返回指标由 trainer 负责记录
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.algorithm = algorithm
        self.collector = collector

        # 从 config 解包（RL 使用 effective_max_iterations 支持 total_steps 预算）
        self.max_iteration = config.effective_max_iterations
        self.step_per_iteration = config.step_per_iteration
        self.save_interval = config.save_interval
        self.verbose = config.verbose
        self.eval_interval = getattr(config, "eval_interval", 0)

        # Callbacks
        self.save_checkpoint_fn = save_checkpoint_fn
        self.log_extra_fn = log_extra_fn
        self.stop_fn = stop_fn
        self.logger = logger
        self.eval_fn = eval_fn

        # State
        self.iteration = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.start_time: float = 0.0
        # eval 完成时的外部 hook：(metrics: dict, total_steps: int) -> None
        # 由 train.py 注入，用于将 eval 指标实时喂给 SweepEarlyStopper
        self.on_eval_complete: Optional[Callable] = None

    @abstractmethod
    def train(self) -> Dict[str, float]:
        """Execute training, return final stats."""
        pass

    @abstractmethod
    def _train_iteration(self) -> Dict[str, Any]:
        """Execute one iteration."""
        pass

    def _compute_sps(self) -> int:
        elapsed = time.time() - self.start_time
        return int(self.total_steps / elapsed) if elapsed > 0 else 0

    def _update_best(self, mean_reward: float) -> bool:
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            return True
        return False

    def _log(self, data: Dict[str, float]):
        if self.logger is not None:
            self.logger.log(data, step=self.total_steps)

    def _maybe_run_eval(self):
        """若配置了 eval_fn 且当前 iteration 命中间隔，则执行 inline eval 并记录指标。"""
        if not self.eval_fn or self.eval_interval <= 0:
            return
        if self.iteration % self.eval_interval != 0:
            return
        if self.verbose:
            print(f"[Eval] Running inline eval at iteration {self.iteration} ...")
        self.algorithm.set_training_mode(False)
        try:
            eval_metrics = self.eval_fn()
        except Exception as e:
            print(f"[Eval] Inline eval failed: {e}")
            eval_metrics = {}
        finally:
            self.algorithm.set_training_mode(True)
        if eval_metrics:
            self._log(eval_metrics)
            # 将 eval 指标喂给 early stopper（供 cross-trial 和 slope 判断）
            if self.on_eval_complete is not None:
                self.on_eval_complete(eval_metrics, self.total_steps)
            if self.verbose:
                parts = ", ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
                print(f"[Eval] {parts}")


class OnPolicyTrainer(BaseTrainer):
    """
    On-policy 训练器 (PPO, A2C, MAPPO 等)。

    每轮迭代：
        1. Collect step_per_iteration 步 (eval mode)
        2. Compute GAE (algorithm.prepare_batch)
        3. Update (algorithm.update)
        4. Clear buffer
    """

    def __init__(
        self,
        algorithm: ActorCriticOnPolicyAlgo,
        collector: BaseCollector,
        config: OnPolicyTrainerConfig,
        # Callbacks
        save_checkpoint_fn: Optional[Callable[[int], None]] = None,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        logger: Optional[Any] = None,
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        super().__init__(
            algorithm=algorithm,
            collector=collector,
            config=config,
            save_checkpoint_fn=save_checkpoint_fn,
            log_extra_fn=log_extra_fn,
            stop_fn=stop_fn,
            logger=logger,
            eval_fn=eval_fn,
        )
        # On-policy 特有参数
        self.minibatch_size = config.minibatch_size
        self.update_epochs = config.update_epochs

    def train(self) -> Dict[str, float]:
        """Execute full training."""
        self.collector.reset()
        self.start_time = time.time()

        for self.iteration in range(1, self.max_iteration + 1):
            # Checkpoint
            if self.save_checkpoint_fn and self.iteration % self.save_interval == 0:
                self.save_checkpoint_fn(self.iteration)

            # Train one iteration
            iter_result = self._train_iteration()

            # Log
            self._log_iteration(iter_result)

            # Inline eval
            self._maybe_run_eval()

            # Early stop
            mean_reward = iter_result.get("mean_reward")
            if mean_reward is not None and self.stop_fn and self.stop_fn(mean_reward):
                if self.verbose:
                    print(f"Early stopping at iteration {self.iteration}")
                break

        # Final checkpoint
        if self.save_checkpoint_fn:
            self.save_checkpoint_fn(self.iteration)

        total_time = time.time() - self.start_time
        final_stats = {
            "total_iterations": self.iteration,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
            "final_sps": self._compute_sps(),
        }

        if self.verbose:
            print(f"\nTraining complete! Steps: {self.total_steps}, "
                  f"Time: {total_time:.1f}s, SPS: {final_stats['final_sps']}")

        return final_stats

    def _train_iteration(self) -> Dict[str, Any]:
        """Execute one collect-update iteration."""
        # 1. Collect (eval mode)
        self.algorithm.set_training_mode(False)
        result = self.collector.collect(n_steps=self.step_per_iteration)
        self.total_steps += result.n_steps

        # 2. Prepare batch (compute GAE)
        batch = self.algorithm.prepare_batch(result.batch)

        # 3. Update (train mode)
        self.algorithm.set_training_mode(True)
        stats = self.algorithm.update(
            batch,
            minibatch_size=self.minibatch_size,
            update_epochs=self.update_epochs,
        )

        # 4. Clear buffer
        self.collector.reset_buffer()

        # 5. Build result dict
        iter_result = {
            "stats": stats,
            "collect_result": result,
            "sps": self._compute_sps(),
        }

        if result.episode_rewards:
            mean_reward = np.mean(result.episode_rewards)
            self._update_best(mean_reward)
            iter_result["mean_reward"] = mean_reward
            iter_result["n_episodes"] = result.n_episodes

        return iter_result

    def _log_iteration(self, iter_result: Dict[str, Any]):
        stats: TrainingStats = iter_result["stats"]
        result: CollectResult = iter_result["collect_result"]
        sps = iter_result["sps"]

        log_data = {
            "train/loss": stats.loss,
            "train/policy_loss": stats.policy_loss,
            "train/value_loss": stats.value_loss,
            "train/entropy": stats.entropy,
            "train/sps": sps,
            "train/total_steps": self.total_steps,
        }

        if stats.extra:
            for k, v in stats.extra.items():
                log_data[f"train/{k}"] = v

        if result.episode_rewards:
            log_data["rollout/episode_reward"] = np.mean(result.episode_rewards)
            log_data["rollout/episode_length"] = np.mean(result.episode_lengths)
            log_data["rollout/n_episodes"] = result.n_episodes

        if self.log_extra_fn:
            extra = self.log_extra_fn()
            if extra:
                log_data.update(extra)

        self._log(log_data)

        if self.verbose and (self.iteration % 10 == 0 or self.iteration == 1):
            reward_str = (
                f"{iter_result.get('mean_reward', 0):.2f}"
                if result.episode_rewards else "N/A"
            )
            print(
                f"[Iter {self.iteration}/{self.max_iteration}] "
                f"steps={self.total_steps}, reward={reward_str}, "
                f"pg_loss={stats.policy_loss:.4f}, v_loss={stats.value_loss:.4f}, "
                f"SPS={sps}"
            )


class OffPolicyTrainer(BaseTrainer):
    """
    Off-policy 训练器 (DQN, IQL, VDQN, QMIX 等)。

    ReplayBuffer 由 collector 持有（OffPolicyCollector / MAOffPolicyCollector），
    Trainer 通过 collector.sample() / collector.can_sample() 访问。

    每轮迭代：
        1. collect() 采集少量 step 并自动存入 collector 内部的 buffer
        2. collector.sample() 采样 → algorithm.update()
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        config: OffPolicyTrainerConfig,
        # Callbacks
        save_checkpoint_fn: Optional[Callable[[int], None]] = None,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        logger: Optional[Any] = None,
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        super().__init__(
            algorithm=algorithm,
            collector=collector,
            config=config,
            save_checkpoint_fn=save_checkpoint_fn,
            log_extra_fn=log_extra_fn,
            stop_fn=stop_fn,
            logger=logger,
            eval_fn=eval_fn,
        )
        self.collect_per_step = config.collect_per_step
        self.update_per_step = config.update_per_step
        self.batch_size = config.batch_size
        self.warmup_steps = config.warmup_steps

    def train(self) -> Dict[str, float]:
        self.collector.reset()
        self.start_time = time.time()

        # ---- Warmup: 填充 buffer 但不训练 ----
        # warmup 步数计入 total_steps，以便 epsilon 衰减能正确以 warmup 为偏移量起算
        if self.warmup_steps > 0:
            if self.verbose:
                print(f"Warming up buffer with {self.warmup_steps} steps...")
            self.algorithm.set_training_mode(True)
            warmup_collected = 0
            while warmup_collected < self.warmup_steps:
                result = self.collector.collect(n_steps=self.collect_per_step)
                warmup_collected += result.n_steps
                self.total_steps += result.n_steps
            if self.verbose:
                print(f"Warmup done, collected {warmup_collected} steps")

        # ---- 主训练循环 ----
        for self.iteration in range(1, self.max_iteration + 1):
            if self.save_checkpoint_fn and self.iteration % self.save_interval == 0:
                self.save_checkpoint_fn(self.iteration)

            iter_result = self._train_iteration()
            self._log_iteration(iter_result)

            # Inline eval
            self._maybe_run_eval()

            mean_reward = iter_result.get("mean_reward")
            if mean_reward is not None and self.stop_fn and self.stop_fn(mean_reward):
                if self.verbose:
                    print(f"Early stopping at iteration {self.iteration}")
                break

        if self.save_checkpoint_fn:
            self.save_checkpoint_fn(self.iteration)

        total_time = time.time() - self.start_time
        return {
            "total_iterations": self.iteration,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }

    def _train_iteration(self) -> Dict[str, Any]:
        all_stats: List[TrainingStats] = []
        all_rewards: List[float] = []
        iter_steps = 0

        while iter_steps < self.step_per_iteration:
            self.algorithm.set_training_mode(True)
            result = self.collector.collect(n_steps=self.collect_per_step)
            iter_steps += result.n_steps
            self.total_steps += result.n_steps
            all_rewards.extend(result.episode_rewards)

            if self.collector.can_sample(self.batch_size):
                for _ in range(self.update_per_step):
                    batch = self.collector.sample(self.batch_size)
                    stats = self.algorithm.update(
                        batch,
                        global_step=self.total_steps,
                        warmup_steps=self.warmup_steps,
                    )
                    all_stats.append(stats)

        iter_result: Dict[str, Any] = {
            "stats_list": all_stats,
            "sps": self._compute_sps(),
        }

        if all_rewards:
            mean_reward = np.mean(all_rewards)
            self._update_best(mean_reward)
            iter_result["mean_reward"] = mean_reward
            iter_result["n_episodes"] = len(all_rewards)

        return iter_result

    def _log_iteration(self, iter_result: Dict[str, Any]):
        stats_list: List[TrainingStats] = iter_result.get("stats_list", [])

        log_data: Dict[str, float] = {
            "train/sps": iter_result["sps"],
            "train/total_steps": self.total_steps,
        }

        if stats_list:
            log_data["train/loss"] = np.mean([s.loss for s in stats_list])
            # 提取 off-policy 特有统计量
            for key in ("epsilon", "epsilon_mean", "q_mean", "q_max", "td_error"):
                vals = [s.extra.get(key) for s in stats_list if s.extra.get(key) is not None]
                if vals:
                    log_data[f"train/{key}"] = np.mean(vals)

        if "mean_reward" in iter_result:
            log_data["rollout/episode_reward"] = iter_result["mean_reward"]
            log_data["rollout/n_episodes"] = iter_result["n_episodes"]

        if self.log_extra_fn:
            extra = self.log_extra_fn()
            if extra:
                log_data.update(extra)

        self._log(log_data)

        if self.verbose and (self.iteration % 10 == 0 or self.iteration == 1):
            loss_str = (
                f"{np.mean([s.loss for s in stats_list]):.4f}"
                if stats_list else "N/A"
            )
            reward_str = (
                f"{iter_result.get('mean_reward', 0):.2f}"
                if "mean_reward" in iter_result else "N/A"
            )
            print(
                f"[Iter {self.iteration}/{self.max_iteration}] "
                f"steps={self.total_steps}, reward={reward_str}, "
                f"loss={loss_str}, SPS={iter_result['sps']}"
            )
