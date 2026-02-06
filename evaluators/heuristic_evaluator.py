#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Evaluator for HeuristicBasePolicy Subclasses

直接与 PatrolWorld 交互，不依赖 MDP 封装。
Evaluates any HeuristicBasePolicy subclass over multiple episodes,
aggregates metrics, and plots results with mean (line) and std (shaded area).
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，并切换工作目录（确保配置文件路径正确）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import argparse
import yaml
import random
from typing import Dict, List, Any, Optional

from envs.mdps.patrol_core import PatrolWorld
from polocies.heuritic.heuristic_base import HeuriticBasePolicy
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


class HeuristicEvaluator:
    """
    Generic evaluator for HeuristicBasePolicy subclasses.
    
    直接与 PatrolWorld 交互，不依赖 任何 MDP 封装。
    """

    def __init__(
        self, 
        world: PatrolWorld, 
        policy: HeuriticBasePolicy, 
        num_episodes: int,
        episode_len: float = 5000,
        truncate_by_time: bool = True,
        init_positions: Optional[List[int]] = None,
    ):
        """
        Args:
            world: PatrolWorld 物理世界实例
            policy: 启发式策略实例
            num_episodes: 评估的 episode 数量
            episode_len: episode 长度（时间或步数，取决于 truncate_by_time）
            truncate_by_time: True=按物理时间截断，False=按步数截断
            init_positions: 初始位置（可选，None 则随机）
        """
        self.world = world
        self.policy = policy
        self.num_episodes = num_episodes
        self.episode_len = episode_len
        self.truncate_by_time = truncate_by_time
        self.init_positions = init_positions
        self.metrics_history: List[Dict[str, List[float]]] = []

    def _is_truncated(self) -> bool:
        """检查是否达到终止条件"""
        if self.truncate_by_time:
            return self.world.current_time >= self.episode_len
        else:
            return self.world.step_count >= self.episode_len

    def run_episode(self) -> Dict[str, List[float]]:
        """Run single episode, return time-series metrics"""
        # 重置物理世界和策略
        if self.init_positions:
            self.world.reset(initial_positions=self.init_positions)
        else:
            # 随机初始位置
            init_pos = random.sample(list(self.world.graph.nodes), self.world.num_agents)
            self.world.reset(initial_positions=init_pos)
        
        self.policy.reset()

        # 运行 episode 直到达到终止条件
        while not self._is_truncated():
            # 获取启发式观测和全局状态
            obs_dict = self.world.get_heuristic_obs()
            global_state = self.world.get_global_state_for_heuristic()

            # 计算启发式动作
            heuristic_actions = self.policy.compute_actions(obs_dict, global_state)

            # 执行动作并推进环境
            self.world.step_heuristic(heuristic_actions)

        # 返回整个 episode 的指标历史
        return self.world.metrics_tracker.get_history_dict()

    def evaluate(self) -> Dict[str, Any]:
        """Run all episodes and aggregate results"""
        truncate_mode = "time" if self.truncate_by_time else "steps"
        print(f"Running {self.num_episodes} episodes (episode_len={self.episode_len}, truncate_by={truncate_mode})...")

        for ep in range(self.num_episodes):
            metrics = self.run_episode()
            self.metrics_history.append(metrics)
            print(f"  Episode {ep + 1}/{self.num_episodes}: "
                  f"Final WI={metrics['wi'][-1]:.4f}, "
                  f"Final IGI={metrics['igi'][-1]:.4f}")

        return self._aggregate_metrics()

    def _aggregate_metrics(self) -> Dict[str, Any]:
        """Compute mean and std across episodes"""
        return aggregate_episode_metrics(self.metrics_history)


def create_policy(policy_name: str, num_agents: int, policy_config: Dict) -> HeuriticBasePolicy:
    """Factory function to create policy by name"""
    policy_map = {
        'ER': 'polocies.heuritic.er',
        'HPCC': 'polocies.heuritic.hpcc',
    }

    if policy_name not in policy_map:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policy_map.keys())}")

    module = __import__(policy_map[policy_name], fromlist=[policy_name + 'Policy'])
    policy_class = getattr(module, policy_name + 'Policy')

    return policy_class(num_agents, policy_config)


def main():
    parser = argparse.ArgumentParser(description='Evaluate heuristic policies (directly with PatrolWorld)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--policy', type=str, default='ER',
                        choices=['ER', 'HPCC'],
                        help='Policy to evaluate (default: ER)')
    parser.add_argument('--env_config', type=str, default='configs/heu_evaluate.yaml',
                        help='Path to environment config')
    parser.add_argument('--policy_config', type=str, default=None,
                        help='Path to policy config (default: configs/{POLICY}.yaml)')
    parser.add_argument('--save_plot', type=str, default='evaluators/results/ER_eval_grid_random.png',
                        help='Path to save plot (default: evaluators/results/heuristic_eval.png)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plot')

    args = parser.parse_args()

    # Load environment config
    with open(args.env_config) as f:
        env_config_data = yaml.safe_load(f)
        env_config = env_config_data['env_config']
        custom_config = env_config_data.get('custom_config', {})

    # 从配置中提取终止条件参数
    episode_len = env_config.get('episode_len', 5000)
    truncate_by_time = custom_config.get('truncate_by_time', True)
    init_positions = env_config.get('init_positions', None)

    # Load policy config
    policy_config_path = args.policy_config or f"configs/{args.policy}.yaml"
    with open(policy_config_path) as f:
        policy_config = yaml.safe_load(f)

    # 直接创建 PatrolWorld（不需要 MDP 封装）
    print(f"Creating PatrolWorld with graph: {env_config['graph_path']}")
    print(f"  episode_len: {episode_len}, truncate_by_time: {truncate_by_time}")
    print(f"  init_positions: {init_positions if init_positions else 'random'}")
    world = PatrolWorld(env_config)

    # Create policy
    print(f"Creating {args.policy} policy")
    policy = create_policy(args.policy, world.num_agents, policy_config)

    # Run evaluation
    evaluator = HeuristicEvaluator(
        world, policy, args.num_episodes,
        episode_len=episode_len,
        truncate_by_time=truncate_by_time,
        init_positions=init_positions
    )
    aggregated = evaluator.evaluate()

    # Print final statistics
    print("\n=== Final Statistics ===")
    for metric in ['igi', 'agi', 'iwi', 'wi']:
        final_mean = aggregated[f'{metric}_mean'][-1]
        final_std = aggregated[f'{metric}_std'][-1]
        print(f"{metric.upper()}: {final_mean:.4f} ± {final_std:.4f}")

    # Plot results
    print(f"\nPlotting results...")
    plot_aggregated_metrics(
        aggregated,
        title=f'{args.policy} Policy Evaluation ({args.num_episodes} episodes)',
        save_path=args.save_plot,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()
