"""RL 训练器：协调 Collector 和 Algorithm 进行训练"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import time
import numpy as np

from algorithms.algorithm_base import BaseAlgorithm, ActorCriticOnPolicyAlgo, TrainingStats
from data.collector import BaseCollector, Collector, MACollector
from data.batch import RolloutBatch, CollectResult


class BaseTrainer(ABC):
    """
    训练器基类
    
    Args:
        algorithm: RL 算法实例
        collector: 数据采集器
        max_epoch: 最大训练轮数
        step_per_epoch: 每轮采集的步数
        logger: 日志记录器（可选）
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        max_epoch: int = 100,
        step_per_epoch: int = 2048,
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        self.algorithm = algorithm
        self.collector = collector
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.logger = logger
        self.verbose = verbose
        
        # 训练状态
        self.epoch = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # 统计
        self._epoch_stats: Dict[str, List[float]] = {}
    
    @abstractmethod
    def train(self) -> Dict[str, float]:
        """执行训练，返回最终统计"""
        pass
    
    @abstractmethod
    def _train_epoch(self) -> Dict[str, float]:
        """执行一轮训练"""
        pass
    
    def _log_epoch(self, stats: Dict[str, float]):
        """记录一轮训练的统计"""
        if self.verbose:
            msg = f"[Epoch {self.epoch}/{self.max_epoch}] "
            msg += " | ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            print(msg)
        
        if self.logger is not None:
            for k, v in stats.items():
                self.logger.log({f"train/{k}": v}, step=self.total_steps)
    
    def _update_best(self, mean_reward: float) -> bool:
        """更新最佳奖励，返回是否更新"""
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            return True
        return False


class OnPolicyTrainer(BaseTrainer):
    """
    On-policy 训练器（用于 PPO, A2C, MAPPO 等）
    
    训练流程：
    1. 采集 step_per_epoch 步数据
    2. 计算 GAE（algorithm.prepare_batch）
    3. 更新（algorithm.update，内部处理 minibatch 切分和多轮更新）
    4. 清空 buffer
    """
    
    def __init__(
        self,
        algorithm: ActorCriticOnPolicyAlgo,
        collector: BaseCollector,
        max_epoch: int = 100,
        step_per_epoch: int = 2048,
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        super().__init__(algorithm, collector, max_epoch, step_per_epoch, logger, verbose)
    
    def train(self) -> Dict[str, float]:
        """执行完整训练"""
        # 重置环境和 buffer
        self.collector.reset()
        
        start_time = time.time()
        
        for self.epoch in range(1, self.max_epoch + 1):
            epoch_stats = self._train_epoch()
            self._log_epoch(epoch_stats)
        
        total_time = time.time() - start_time
        
        final_stats = {
            "total_epochs": self.max_epoch,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }
        
        if self.verbose:
            print(f"\n训练完成! 总步数: {self.total_steps}, 耗时: {total_time:.1f}s")
        
        return final_stats
    
    def _train_epoch(self) -> Dict[str, float]:
        """执行一轮训练"""
        # 1. 采集数据
        self.algorithm.set_training_mode(False)
        result = self.collector.collect(n_steps=self.step_per_epoch)
        self.total_steps += result.n_steps
        
        # 2. 计算 GAE
        batch = self.algorithm.prepare_batch(result.batch)
        
        # 3. 更新（算法内部处理 minibatch 切分和多轮更新）
        self.algorithm.set_training_mode(True)
        stats = self.algorithm.update(batch)
        
        # 4. 清空 buffer
        self.collector.reset_buffer()
        
        # 5. 统计
        epoch_stats = self._aggregate_stats([stats], result)
        
        # 更新最佳奖励
        if result.episode_rewards:
            mean_reward = np.mean(result.episode_rewards)
            self._update_best(mean_reward)
            epoch_stats["mean_episode_reward"] = mean_reward
            epoch_stats["n_episodes"] = result.n_episodes
        
        return epoch_stats
    
    def _aggregate_stats(
        self, 
        stats_list: List[TrainingStats], 
        result: CollectResult
    ) -> Dict[str, float]:
        """聚合统计信息"""
        if not stats_list:
            return {}
        
        # 平均 loss
        epoch_stats = {
            "loss": np.mean([s.loss for s in stats_list]),
            "policy_loss": np.mean([s.policy_loss for s in stats_list]),
            "value_loss": np.mean([s.value_loss for s in stats_list]),
            "entropy": np.mean([s.entropy for s in stats_list]),
            "n_steps": result.n_steps,
        }
        
        return epoch_stats


class OffPolicyTrainer(BaseTrainer):
    """
    Off-policy 训练器（用于 DQN, SAC 等）
    
    训练流程：
    1. 采集少量数据存入 replay buffer
    2. 从 buffer 采样更新
    
    Args:
        algorithm: Off-policy 算法实例
        collector: 数据采集器
        buffer: Replay buffer
        step_per_collect: 每次采集的步数
        update_per_collect: 每次采集后更新的次数
        batch_size: 采样 batch 大小
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        buffer: Any,  # ReplayBuffer
        max_epoch: int = 100,
        step_per_epoch: int = 10000,
        step_per_collect: int = 10,
        update_per_collect: int = 10,
        batch_size: int = 256,
        start_timesteps: int = 10000,
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        super().__init__(algorithm, collector, max_epoch, step_per_epoch, logger, verbose)
        self.buffer = buffer
        self.step_per_collect = step_per_collect
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.start_timesteps = start_timesteps
    
    def train(self) -> Dict[str, float]:
        """执行完整训练"""
        self.collector.reset()
        start_time = time.time()
        
        # 预填充 buffer
        if self.start_timesteps > 0:
            if self.verbose:
                print(f"预填充 buffer: {self.start_timesteps} 步...")
            # TODO: 实现随机策略采集
        
        for self.epoch in range(1, self.max_epoch + 1):
            epoch_stats = self._train_epoch()
            self._log_epoch(epoch_stats)
        
        total_time = time.time() - start_time
        return {
            "total_epochs": self.max_epoch,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """执行一轮训练"""
        epoch_steps = 0
        all_stats: List[TrainingStats] = []
        all_rewards: List[float] = []
        
        while epoch_steps < self.step_per_epoch:
            # 1. 采集
            result = self.collector.collect(n_steps=self.step_per_collect)
            epoch_steps += result.n_steps
            self.total_steps += result.n_steps
            all_rewards.extend(result.episode_rewards)
            
            # 2. 存入 buffer（TODO: 需要实现）
            # self.buffer.add_batch(result.batch)
            
            # 3. 更新
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.update_per_collect):
                    batch = self.buffer.sample(self.batch_size)
                    stats = self.algorithm.update(batch)
                    all_stats.append(stats)
        
        # 统计
        epoch_stats = {
            "n_steps": epoch_steps,
        }
        
        if all_stats:
            epoch_stats["loss"] = np.mean([s.loss for s in all_stats])
        
        if all_rewards:
            epoch_stats["mean_episode_reward"] = np.mean(all_rewards)
            epoch_stats["n_episodes"] = len(all_rewards)
            self._update_best(epoch_stats["mean_episode_reward"])
        
        return epoch_stats
