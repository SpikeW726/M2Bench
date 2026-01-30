#!/usr/bin/env python3
"""Temporary script to evaluate legacy MAPPO checkpoints."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import os
os.chdir(project_root)

import yaml
import torch
import numpy as np

from envs.mdps.masup_env import MASUPEnv
from polocies.rl.rl_base import ActorPolicy
from polocies.marl.marl_base import MultiAgentPolicy
from networks.mlp import ActorMLP
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


def evaluate(policy_path: str, num_episodes: int = 10, max_steps: int = 1000, save_plot: str = None):
    """
    Evaluate legacy MAPPO checkpoint.
    
    Args:
        policy_path: Path to policy checkpoint
        num_episodes: Number of episodes to run
        max_steps: Fixed step count per episode (ensures consistent length for visualization)
        save_plot: Path to save plot
    """
    # 1. 显式检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed hidden sizes for legacy models
    HIDDEN_SIZES = [256, 256]
    
    # Load env config
    with open("configs/MASUPEnv.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Extract info for plotting
    graph_path = cfg['env_config']['graph_path']
    graph_name = Path(graph_path).stem
    num_agents = cfg['env_config']['num_agents']
    
    env = MASUPEnv(cfg['env_config'], **cfg.get('custom_config', {}))
    
    # Get dims
    agent = env.possible_agents[0]
    obs_dim = env.observation_space(agent).shape[0]
    action_dim = env.action_space(agent).n
    
    print(f"Loading: {policy_path}")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}, hidden={HIDDEN_SIZES}")
    
    # Load checkpoint
    # 保持 map_location='cpu' 以确保加载兼容性，随后再移动到 device
    ckpt = torch.load(policy_path, map_location='cpu', weights_only=True)
    
    # Create actor and load weights
    actor = ActorMLP(obs_dim, HIDDEN_SIZES, action_dim)
    
    # Handle different formats
    if '_shared_policy.actor.network.0.weight' in ckpt:
        # MAPPO MultiAgentPolicy format
        actor_sd = {}
        for k, v in ckpt.items():
            if k.startswith('_shared_policy.actor.'):
                actor_sd[k.replace('_shared_policy.actor.', '')] = v
        actor.load_state_dict(actor_sd)
        print("Loaded from MAPPO format")
    else:
        actor.load_state_dict(ckpt)
        print("Loaded from direct state_dict")
    
    # 2. 将模型移动到指定设备
    actor.to(device)
    actor.eval()
    
    # Create policy
    policy = MultiAgentPolicy(
        agent_ids=env.possible_agents,
        obs_space=env.observation_space(agent),
        action_space=env.action_space(agent),
        policy_class=ActorPolicy,
        policy_kwargs={'actor': actor, 'deterministic_eval': True},
        shared=True
    )
    policy.set_training_mode(False)
    
    # Run episodes with fixed step count
    print(f"\nRunning {num_episodes} episodes (fixed {max_steps} steps each)...")
    episode_metrics = []
    metrics_history = []
    episode_times = []
    
    for ep in range(num_episodes):
        obs, infos = env.reset()
        step_count = 0
        
        while step_count < max_steps:
            # 3. 将输入数据移动到同一设备
            action_masks = {
                aid: torch.as_tensor(info['action_mask'], dtype=torch.bool).to(device)
                for aid, info in infos.items()
            }
            obs_tensor = {
                aid: torch.as_tensor(o, dtype=torch.float32).to(device)
                for aid, o in obs.items()
            }
            
            with torch.no_grad():
                outputs = policy.forward(obs_tensor, action_mask=action_masks)
            
            # 将输出转回 CPU 以便后续处理 (numpy转换)
            actions = {
                aid: out['act'].cpu().numpy() 
                for aid, out in outputs.items()
            }
            obs, _, _, _, infos = env.step(actions)
            step_count += 1
        
        m = env.world.current_metrics
        episode_metrics.append(m)
        episode_times.append(m.time)
        metrics_history.append(env.world.metrics_tracker.get_history_dict())
        
        print(f"  Episode {ep+1}: IGI={m.igi:.4f}, IWI={m.iwi:.4f}, WI={m.wi:.4f}, time={m.time:.2f}s, steps={step_count}")
    
    # Summary
    print(f"\n=== Summary ({max_steps} steps/episode) ===")
    for name in ['igi', 'agi', 'iwi', 'wi']:
        vals = [getattr(m, name) for m in episode_metrics]
        print(f"{name.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Plot (no padding needed since all episodes have same length)
    if save_plot and metrics_history:
        Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
        aggregated = aggregate_episode_metrics(metrics_history)
        
        # Build subtitle with extra info
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"
        
        plot_aggregated_metrics(
            aggregated, 
            save_path=save_plot, 
            show=False,
            title=f'Evaluation ({num_episodes} episodes, {max_steps} steps)',
            subtitle=subtitle
        )
        print(f"Plot saved to {save_plot}")
    
    env.close()
    return episode_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('policy', type=str, help='Path to policy.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=1000, help='Fixed steps per episode')
    parser.add_argument('--save_plot', type=str, default=None)
    args = parser.parse_args()
    
    evaluate(args.policy, args.episodes, args.max_steps, args.save_plot)
