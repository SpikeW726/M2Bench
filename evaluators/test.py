#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for trained RL actor policy

Supports two loading modes:
1. HuggingFace style: model_dir/ with config.yaml + policy.pt
2. Legacy: single checkpoint file (.pt)
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import yaml
import torch
import numpy as np

from envs.mdps.masup_env import MASUPEnv
from polocies.rl.rl_base import ActorPolicy
from polocies.marl.marl_base import MultiAgentPolicy
from networks.mlp import ActorMLP
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics
from utils.model_io import load_actor_only, get_model_config


def load_trained_actor(model_path: str, obs_dim: int, action_dim: int) -> ActorMLP:
    """
    Load actor network from model directory or legacy checkpoint file.
    
    Supports:
    1. HuggingFace style: model_path is a directory with config.yaml + policy.pt
    2. Legacy formats: model_path is a .pt file
       - Pretrain format: {'actor_state_dict': ..., 'hidden_sizes': ...}
       - MAPPO MultiAgentPolicy format: {'_shared_policy.actor.network.0.weight': ...}
       - Direct state_dict format: {'network.0.weight': ...}
    """
    model_path = Path(model_path)
    
    # Check if HuggingFace style directory
    if model_path.is_dir() and (model_path / 'config.yaml').exists():
        print(f"Loading from HuggingFace style directory: {model_path}")
        actor = load_actor_only(model_path, device='cpu')
        config = get_model_config(model_path)
        print(f"Actor config: {config.get('actor', {})}")
        return actor
    
    # Legacy: single checkpoint file
    print(f"Loading from legacy checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine format and extract actor state_dict
    if 'actor_state_dict' in checkpoint:
        # Pretrain format
        actor_sd = checkpoint['actor_state_dict']
        hidden_sizes = checkpoint.get('hidden_sizes', _infer_hidden_sizes(actor_sd))
        print(f"Format: Pretrain, hidden_sizes={hidden_sizes}")
    
    elif '_shared_policy.actor.network.0.weight' in checkpoint:
        # MAPPO MultiAgentPolicy format
        actor_sd = {}
        prefix = '_shared_policy.actor.'
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                actor_sd[k[len(prefix):]] = v
        hidden_sizes = _infer_hidden_sizes(actor_sd)
        print(f"Format: MAPPO MultiAgentPolicy, hidden_sizes={hidden_sizes}")
    
    else:
        # Direct state_dict format
        actor_sd = checkpoint
        hidden_sizes = _infer_hidden_sizes(actor_sd)
        print(f"Format: Direct state_dict, hidden_sizes={hidden_sizes}")
    
    # Create and load actor
    actor = ActorMLP(obs_dim, hidden_sizes, action_dim)
    actor.load_state_dict(actor_sd)
    actor.eval()
    return actor


def _infer_hidden_sizes(state_dict: dict) -> list:
    """Infer hidden sizes from MLP state_dict (fallback for legacy checkpoints)."""
    hidden_sizes = []
    idx = 0
    while f'network.{idx}.weight' in state_dict:
        hidden_sizes.append(state_dict[f'network.{idx}.weight'].shape[0])
        idx += 2
    return hidden_sizes[:-1] if hidden_sizes else []


def test_trained_policy(checkpoint_path: str = 'models/pure/imi_train__1769274544_actor_best.pt',
                       num_episodes: int = 5,
                       max_steps: int = 1000,
                       save_plot: str = None,
                       show_plot: bool = True):
    """
    Test trained actor policy in MASUPEnv.
    
    Args:
        checkpoint_path: Path to model directory or checkpoint file
        num_episodes: Number of episodes to run
        max_steps: Fixed step count per episode (ensures consistent length for visualization)
        save_plot: Path to save plot
        show_plot: Whether to display plot
    """
    # Load environment config
    with open("configs/MASUPEnv.yaml") as f:
        env_config_data = yaml.safe_load(f)
        env_config = env_config_data['env_config']
        custom_config = env_config_data.get('custom_config', {})

    # Extract info for plotting
    graph_path = env_config['graph_path']
    graph_name = Path(graph_path).stem  # Get filename without extension
    num_agents = env_config['num_agents']

    # Create environment
    print(f"\n=== Creating MASUPEnv ===")
    print(f"Graph: {graph_path} ({graph_name})")
    print(f"Num agents: {num_agents}")
    env = MASUPEnv(env_config, **custom_config)

    # Get network dimensions from environment
    sample_agent = env.possible_agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    action_dim = env.action_space(sample_agent).n

    print(f"\n=== Network Dimensions ===")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Max neighbors: {env.world.max_neighbors}")

    # Load trained actor (hidden_sizes inferred from checkpoint)
    actor = load_trained_actor(checkpoint_path, obs_dim, action_dim)

    # Create MultiAgentPolicy with shared strategy
    multi_policy = MultiAgentPolicy(
        agent_ids=env.possible_agents,
        obs_space=env.observation_space(env.possible_agents[0]),
        action_space=env.action_space(env.possible_agents[0]),
        policy_class=ActorPolicy,
        policy_kwargs={'actor': actor, 'deterministic_eval': True},
        shared=True
    )
    multi_policy.set_training_mode(False)

    print(f"\n=== Running {num_episodes} episodes (fixed {max_steps} steps each) ===")

    # Track metrics across episodes
    episode_metrics = []
    metrics_history = []
    episode_times = []  # Physical time for each episode

    for ep in range(num_episodes):
        obs, infos = env.reset()
        step_count = 0

        while step_count < max_steps:
            # Get action masks and convert to tensor
            action_masks = {
                aid: torch.as_tensor(info['action_mask'], dtype=torch.bool, device=multi_policy.device)
                for aid, info in infos.items()
            }
            
            # Convert obs to tensor
            obs_tensor = {
                aid: torch.as_tensor(o, dtype=torch.float32, device=multi_policy.device)
                for aid, o in obs.items()
            }
            
            # Forward pass with action_mask
            with torch.no_grad():
                outputs = multi_policy.forward(obs_tensor, action_mask=action_masks)
            
            # Extract actions
            actions = {aid: out['act'].cpu().numpy() for aid, out in outputs.items()}

            # Step environment
            obs, _, _, _, infos = env.step(actions)
            step_count += 1

        # Get final metrics
        final_metrics = env.world.current_metrics
        episode_metrics.append(final_metrics)
        episode_times.append(final_metrics.time)
        
        # Collect episode metrics history for visualization
        episode_history = env.world.metrics_tracker.get_history_dict()
        metrics_history.append(episode_history)

        print(f"Episode {ep + 1}/{num_episodes}: "
              f"IGI={final_metrics.igi:.4f}, "
              f"AGI={final_metrics.agi:.4f}, "
              f"IWI={final_metrics.iwi:.4f}, "
              f"WI={final_metrics.wi:.4f}, "
              f"time={final_metrics.time:.2f}s, "
              f"steps={step_count}")

    # Print summary statistics
    print(f"\n=== Summary Statistics ({num_episodes} episodes, {max_steps} steps) ===")

    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    # Aggregate and visualize metrics (no padding needed since all episodes have same length)
    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        
        # Build subtitle with extra info
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"
        
        plot_aggregated_metrics(
            aggregated,
            title=f'RL Policy Evaluation ({num_episodes} episodes, {max_steps} steps)',
            subtitle=subtitle,
            save_path=save_plot,
            show=show_plot
        )

    return episode_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained RL policy')
    parser.add_argument('--model', type=str,
                        default='models/mappo/imi/final',
                        help='Path to model directory (HuggingFace style) or checkpoint file (legacy)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Fixed steps per episode')
    parser.add_argument('--save_plot', type=str, default='evaluators/results/rl_eval.png',
                        help='Path to save plot')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plot')

    args = parser.parse_args()

    test_trained_policy(
        checkpoint_path=args.model,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        save_plot=args.save_plot,
        show_plot=not args.no_show
    )