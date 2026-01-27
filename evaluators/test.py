#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for trained RL actor policy

Loads trained actor weights and evaluates performance in MASUPEnv.
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import yaml
import torch
import numpy as np

from envs.mdps.masup_env import MASUPEnv
from polocies.rl.rl_base import ActorPolicy
from polocies.marl.marl_base import MultiAgentPolicy
from trainers.imitator.imitation_trainer import ActorMLP
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


def load_trained_actor(checkpoint_path: str, obs_dim: int, action_dim: int,
                       hidden_sizes: list = None) -> ActorMLP:
    """Load actor network from checkpoint."""
    if hidden_sizes is None:
        hidden_sizes = [256, 256]  # Default from training

    print(f"Loading actor from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Create actor network
    actor = ActorMLP(obs_dim, hidden_sizes, action_dim)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    print(f"Actor loss: {checkpoint.get('actor_loss', 'N/A')}")
    print(f"Actor accuracy: {checkpoint.get('actor_accuracy', 'N/A')}")
    print(f"Stopped at iteration: {checkpoint.get('stopped_iteration', 'N/A')}")

    return actor


def test_trained_policy(checkpoint_path: str = 'models/pure/imi_train__1769274544_actor_best.pt',
                       num_episodes: int = 5,
                       hidden_sizes: list = None,
                       save_plot: str = None,
                       show_plot: bool = True):
    """Test trained actor policy in MASUPEnv."""
    # Load environment config
    with open("configs/MASUPEnv.yaml") as f:
        env_config_data = yaml.safe_load(f)
        env_config = env_config_data['env_config']
        custom_config = env_config_data.get('custom_config', {})

    # Create environment
    print(f"\n=== Creating MASUPEnv ===")
    print(f"Graph: {env_config['graph_path']}")
    print(f"Num agents: {env_config['num_agents']}")
    env = MASUPEnv(env_config, **custom_config)

    # 使用环境接口获取网络维度
    sample_agent = env.possible_agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    action_dim = env.action_space(sample_agent).n

    print(f"\n=== Network Dimensions ===")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Max neighbors: {env.world.max_neighbors}")

    # Load trained actor
    actor = load_trained_actor(checkpoint_path, obs_dim, action_dim, hidden_sizes)

    # Create MultiAgentPolicy with shared strategy
    multi_policy = MultiAgentPolicy(
        agent_ids=env.possible_agents,
        obs_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        policy_class=ActorPolicy,
        policy_kwargs={'actor': actor, 'deterministic_eval': True},
        shared=True
    )
    multi_policy.set_training_mode(False)

    print(f"\n=== Running {num_episodes} episodes ===")

    # Track metrics across episodes
    episode_metrics = []
    metrics_history = []  # 用于聚合可视化

    for ep in range(num_episodes):
        obs, infos = env.reset()

        while True:
            # Get action masks
            action_masks = {aid: info['action_mask'] for aid, info in infos.items()}

            # Compute actions using multi-agent policy
            actions, _ = multi_policy.compute_actions(obs, action_masks=action_masks)

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)

            if any(truncations.values()):
                break

        # Get final metrics
        final_metrics = env.world.current_metrics
        episode_metrics.append(final_metrics)
        
        # 收集整个 episode 的 metrics 历史用于可视化
        episode_history = env.world.metrics_tracker.get_history_dict()
        metrics_history.append(episode_history)

        print(f"Episode {ep + 1}/{num_episodes}: "
              f"IGI={final_metrics.igi:.4f}, "
              f"AGI={final_metrics.agi:.4f}, "
              f"IWI={final_metrics.iwi:.4f}, "
              f"WI={final_metrics.wi:.4f}, "
              f"Time={final_metrics.time:.2f}, "
              f"Steps={final_metrics.step}")

    # Print summary statistics
    print(f"\n=== Summary Statistics ({num_episodes} episodes) ===")

    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nTime: {np.mean([m.time for m in episode_metrics]):.2f} ± {np.std([m.time for m in episode_metrics]):.2f}")
    print(f"Steps: {np.mean([m.step for m in episode_metrics]):.1f} ± {np.std([m.step for m in episode_metrics]):.1f}")

    # 聚合并可视化 metrics
    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        plot_aggregated_metrics(
            aggregated,
            title=f'RL Policy Evaluation ({num_episodes} episodes)',
            save_path=save_plot,
            show=show_plot
        )

    return episode_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained RL policy')
    parser.add_argument('--checkpoint', type=str,
                        default='models/pure/imi_train__1769274544_actor_best.pt',
                        help='Path to actor checkpoint')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=None,
                        help='Hidden layer sizes (default: [256, 256])')
    parser.add_argument('--save_plot', type=str, default='evaluators/results/rl_eval.png',
                        help='Path to save plot (default: evaluators/results/rl_eval.png)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plot')

    args = parser.parse_args()

    test_trained_policy(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        hidden_sizes=args.hidden_sizes,
        save_plot=args.save_plot,
        show_plot=not args.no_show
    )