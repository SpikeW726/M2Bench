#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Evaluator for HeuristicBasePolicy Subclasses

Evaluates any HeuristicBasePolicy subclass over multiple episodes,
aggregates metrics, and plots results with mean (line) and std (shaded area).
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import numpy as np
from typing import Dict, List, Any, Optional

from envs.mdps.masup_env import MASUPEnv
from polocies.heuritic.heuristic_base import HeuriticBasePolicy
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


class HeuristicEvaluator:
    """Generic evaluator for HeuristicBasePolicy subclasses"""

    def __init__(self, env: MASUPEnv, policy: HeuriticBasePolicy, num_episodes: int):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.metrics_history: List[Dict[str, List[float]]] = []

    def run_episode(self) -> Dict[str, List[float]]:
        """Run single episode, return time-series metrics"""
        obs, infos = self.env.reset()
        self.policy.reset()

        while True:
            # Get observations for heuristic policy
            obs_dict = self.env.get_heuristic_obs()
            global_state = self.env.get_global_state_for_heuristic()

            # Compute actions
            heuristic_actions = self.policy.compute_actions(obs_dict, global_state)

            # Convert to MASUPEnv action format
            env_actions = {}
            for agent_str, neighbor_idx in heuristic_actions.items():
                env_actions[agent_str] = self.env.convert_heuristic_action(agent_str, neighbor_idx)

            # Step environment
            obs, rewards, terminations, truncations, infos = self.env.step(env_actions)

            if any(truncations.values()):
                break

        # 直接使用环境的 metrics_tracker 获取整个 episode 的历史数据
        return self.env.world.metrics_tracker.get_history_dict()

    def evaluate(self) -> Dict[str, Any]:
        """Run all episodes and aggregate results"""
        print(f"Running {self.num_episodes} episodes...")

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
    parser = argparse.ArgumentParser(description='Evaluate heuristic policies')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--policy', type=str, default='ER',
                        choices=['ER', 'HPCC'],
                        help='Policy to evaluate (default: ER)')
    parser.add_argument('--env_config', type=str, default='configs/MASUPEnv.yaml',
                        help='Path to environment config')
    parser.add_argument('--policy_config', type=str, default=None,
                        help='Path to policy config (default: configs/{POLICY}.yaml)')
    parser.add_argument('--save_plot', type=str, default='evaluators/results/heuristic_eval.png',
                        help='Path to save plot (default: evaluators/results/heuristic_eval.png)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plot')

    args = parser.parse_args()

    # Load environment config
    with open(args.env_config) as f:
        env_config_data = yaml.safe_load(f)
        env_config = env_config_data['env_config']
        custom_config = env_config_data.get('custom_config', {})

    # Load policy config
    policy_config_path = args.policy_config or f"configs/{args.policy}.yaml"
    with open(policy_config_path) as f:
        policy_config = yaml.safe_load(f)

    # Create environment
    print(f"Creating MASUPEnv with graph: {env_config['graph_path']}")
    env = MASUPEnv(env_config, **custom_config)

    # Create policy (使用环境接口获取 num_agents)
    print(f"Creating {args.policy} policy")
    policy = create_policy(args.policy, env.world.num_agents, policy_config)

    # Run evaluation
    evaluator = HeuristicEvaluator(env, policy, args.num_episodes)
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
        title=f'Heuristic Policy Evaluation ({args.num_episodes} episodes)',
        save_path=args.save_plot,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()