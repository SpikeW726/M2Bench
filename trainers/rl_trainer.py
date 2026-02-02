"""RL Trainers: coordinate Collector and Algorithm for training"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import time
import numpy as np
import torch.optim as optim

from algorithms.algorithm_base import BaseAlgorithm, OnPolicyAlgorithm, TrainingStats
from data.collector import BaseCollector
from data.batch import CollectResult


class BaseTrainer(ABC):
    """
    Base trainer class.
    
    Terminology:
        - iteration: one collect-update cycle (call update() once)
        - epoch: inside update(), how many times to iterate over the same batch
                 (this is controlled by algorithm, not trainer)
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        max_iteration: int,
        step_per_iteration: int,
        # Callbacks
        save_checkpoint_fn: Optional[Callable[[int], None]] = None,
        save_interval: int = 100,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        # Logging
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        """
        Args:
            algorithm: RL algorithm instance
            collector: data collector
            max_iteration: total number of collect-update cycles
            step_per_iteration: env steps per iteration
            save_checkpoint_fn: callback(iteration) to save model
            save_interval: save every N iterations
            log_extra_fn: callback() -> dict of extra metrics to log
            stop_fn: callback(mean_reward) -> bool, return True to stop
            logger: logger instance (supports .log(dict, step))
            verbose: print progress
        """
        self.algorithm = algorithm
        self.collector = collector
        self.max_iteration = max_iteration
        self.step_per_iteration = step_per_iteration
        
        # Callbacks
        self.save_checkpoint_fn = save_checkpoint_fn
        self.save_interval = save_interval
        self.log_extra_fn = log_extra_fn
        self.stop_fn = stop_fn
        
        # Logging
        self.logger = logger
        self.verbose = verbose
        
        # State
        self.iteration = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.start_time: float = 0.0
    
    @abstractmethod
    def train(self) -> Dict[str, float]:
        """Execute training, return final stats"""
        pass
    
    @abstractmethod
    def _train_iteration(self) -> Dict[str, Any]:
        """Execute one iteration"""
        pass
    
    def _compute_sps(self) -> int:
        """Compute steps per second"""
        elapsed = time.time() - self.start_time
        return int(self.total_steps / elapsed) if elapsed > 0 else 0
    
    def _update_best(self, mean_reward: float) -> bool:
        """Update best reward, return True if updated"""
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            return True
        return False
    
    def _log(self, data: Dict[str, float]):
        """Log metrics"""
        if self.logger is not None:
            self.logger.log(data, step=self.total_steps)


class OnPolicyTrainer(BaseTrainer):
    """
    On-policy trainer (for PPO, A2C, MAPPO, etc.)
    
    Training loop per iteration:
        1. Collect step_per_iteration steps (eval mode)
        2. Compute GAE (algorithm.prepare_batch)
        3. Update (algorithm.update, handles minibatch and epochs internally)
        4. Clear buffer
    """
    
    def __init__(
        self,
        algorithm: OnPolicyAlgorithm,
        collector: BaseCollector,
        max_iteration: int,
        step_per_iteration: int,
        # Callbacks
        save_checkpoint_fn: Optional[Callable[[int], None]] = None,
        save_interval: int = 100,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        # Logging
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        super().__init__(
            algorithm=algorithm,
            collector=collector,
            max_iteration=max_iteration,
            step_per_iteration=step_per_iteration,
            save_checkpoint_fn=save_checkpoint_fn,
            save_interval=save_interval,
            log_extra_fn=log_extra_fn,
            stop_fn=stop_fn,
            logger=logger,
            verbose=verbose,
        )   

    def train(self) -> Dict[str, float]:
        """Execute full training"""
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
        """Execute one collect-update iteration"""
        # 1. Collect (eval mode)
        self.algorithm.set_training_mode(False)
        result = self.collector.collect(n_steps=self.step_per_iteration)
        self.total_steps += result.n_steps
        
        # 2. Prepare batch (compute GAE)
        batch = self.algorithm.prepare_batch(result.batch)
        
        # 3. Update (train mode)
        self.algorithm.set_training_mode(True)
        stats = self.algorithm.update(batch)
        
        # 4. Clear buffer
        self.collector.reset_buffer()
        
        # 5. Build result dict
        iter_result = {
            "stats": stats,
            "collect_result": result,
            "sps": self._compute_sps(),
        }
        
        # Update best reward
        if result.episode_rewards:
            mean_reward = np.mean(result.episode_rewards)
            self._update_best(mean_reward)
            iter_result["mean_reward"] = mean_reward
            iter_result["n_episodes"] = result.n_episodes
        
        return iter_result
    
    def _log_iteration(self, iter_result: Dict[str, Any]):
        """Log iteration metrics"""
        stats: TrainingStats = iter_result["stats"]
        result: CollectResult = iter_result["collect_result"]
        sps = iter_result["sps"]
        
        # Core metrics
        log_data = {
            "train/loss": stats.loss,
            "train/policy_loss": stats.policy_loss,
            "train/value_loss": stats.value_loss,
            "train/entropy": stats.entropy,
            "train/sps": sps,
            "train/total_steps": self.total_steps,
        }
        
        # Extra stats from algorithm (clipfrac, approx_kl, etc.)
        if stats.extra:
            for k, v in stats.extra.items():
                log_data[f"train/{k}"] = v
        
        # Episode stats
        if result.episode_rewards:
            log_data["rollout/episode_reward"] = np.mean(result.episode_rewards)
            log_data["rollout/episode_length"] = np.mean(result.episode_lengths)
            log_data["rollout/n_episodes"] = result.n_episodes
        
        # Extra metrics from callback (env-specific metrics)
        if self.log_extra_fn:
            extra = self.log_extra_fn()
            if extra:
                log_data.update(extra)
        
        # Log to logger
        self._log(log_data)
        
        # Console output
        if self.verbose and (self.iteration % 10 == 0 or self.iteration == 1):
            reward_str = f"{iter_result.get('mean_reward', 0):.2f}" if result.episode_rewards else "N/A"
            print(
                f"[Iter {self.iteration}/{self.max_iteration}] "
                f"steps={self.total_steps}, reward={reward_str}, "
                f"pg_loss={stats.policy_loss:.4f}, v_loss={stats.value_loss:.4f}, "
                f"SPS={sps}"
            )


class OffPolicyTrainer(BaseTrainer):
    """
    Off-policy trainer (for DQN, SAC, etc.)
    
    Training loop:
        1. Collect few steps into replay buffer
        2. Sample from buffer and update
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        buffer: Any,  # ReplayBuffer
        max_iteration: int,
        step_per_iteration: int = 1000,
        collect_per_step: int = 1,
        update_per_step: int = 1,
        batch_size: int = 256,
        warmup_steps: int = 10000,
        # Callbacks
        save_checkpoint_fn: Optional[Callable[[int], None]] = None,
        save_interval: int = 100,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        # Logging
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        super().__init__(
            algorithm=algorithm,
            collector=collector,
            max_iteration=max_iteration,
            step_per_iteration=step_per_iteration,
            save_checkpoint_fn=save_checkpoint_fn,
            save_interval=save_interval,
            log_extra_fn=log_extra_fn,
            stop_fn=stop_fn,
            logger=logger,
            verbose=verbose,
        )
        self.buffer = buffer
        self.collect_per_step = collect_per_step
        self.update_per_step = update_per_step
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
    
    def train(self) -> Dict[str, float]:
        """Execute full training"""
        self.collector.reset()
        self.start_time = time.time()
        
        # Warmup: fill buffer with random actions
        if self.warmup_steps > 0:
            if self.verbose:
                print(f"Warming up buffer with {self.warmup_steps} steps...")
            # TODO: collect with random policy
            pass
        
        for self.iteration in range(1, self.max_iteration + 1):
            if self.save_checkpoint_fn and self.iteration % self.save_interval == 0:
                self.save_checkpoint_fn(self.iteration)
            
            iter_result = self._train_iteration()
            self._log_iteration(iter_result)
            
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
        """One iteration: collect + multiple updates"""
        all_stats: List[TrainingStats] = []
        all_rewards: List[float] = []
        iter_steps = 0
        
        while iter_steps < self.step_per_iteration:
            # Collect
            result = self.collector.collect(n_steps=self.collect_per_step)
            iter_steps += result.n_steps
            self.total_steps += result.n_steps
            all_rewards.extend(result.episode_rewards)
            
            # Add to buffer
            # self.buffer.add(result.batch)
            
            # Update
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.update_per_step):
                    batch = self.buffer.sample(self.batch_size)
                    stats = self.algorithm.update(batch)
                    all_stats.append(stats)
        
        iter_result = {
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
        """Log iteration metrics"""
        stats_list = iter_result.get("stats_list", [])
        
        log_data = {
            "train/sps": iter_result["sps"],
            "train/total_steps": self.total_steps,
        }
        
        if stats_list:
            log_data["train/loss"] = np.mean([s.loss for s in stats_list])
        
        if "mean_reward" in iter_result:
            log_data["rollout/episode_reward"] = iter_result["mean_reward"]
            log_data["rollout/n_episodes"] = iter_result["n_episodes"]
        
        if self.log_extra_fn:
            extra = self.log_extra_fn()
            if extra:
                log_data.update(extra)
        
        self._log(log_data)
        
        if self.verbose and (self.iteration % 10 == 0 or self.iteration == 1):
            reward_str = f"{iter_result.get('mean_reward', 0):.2f}" if "mean_reward" in iter_result else "N/A"
            print(f"[Iter {self.iteration}/{self.max_iteration}] "
                  f"steps={self.total_steps}, reward={reward_str}, SPS={iter_result['sps']}")
