#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL 策略评估脚本。

支持两种模型加载方式：
1. HuggingFace 风格：model_dir/ 含 config.yaml + policy.pt
2. Legacy：单 checkpoint 文件 (.pt)

环境配置通过 --env_config 指定 YAML 文件加载（支持 experiment YAML 或独立 eval YAML）。
"""
import os
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import numpy as np

from envs.mdps.masup import MASUPEnv
from configs.registry import load_env_config, _env_config_to_dicts
from policies.rl.rl_base import ActorPolicy
from policies.marl.marl_base import MultiAgentPolicy
from networks.mlp import ActorMLP
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics
from utils.model_io import load_actor_only, get_model_config


# =============================================================================
#                          模型加载
# =============================================================================

def load_trained_actor(model_path: str, obs_dim: int, action_dim: int) -> ActorMLP:
    """
    从模型目录或 legacy checkpoint 文件加载 actor 网络。

    支持格式：
    1. HuggingFace 风格目录 (config.yaml + policy.pt)
    2. Pretrain 格式 {'actor_state_dict': ..., 'hidden_sizes': ...}
    3. MAPPO MultiAgentPolicy 格式 {'_shared_policy.actor.network.0.weight': ...}
    4. 直接 state_dict 格式 {'network.0.weight': ...}
    """
    model_path = Path(model_path)

    # HuggingFace 风格目录
    if model_path.is_dir() and (model_path / 'config.yaml').exists():
        print(f"Loading from HuggingFace style directory: {model_path}")
        actor = load_actor_only(model_path, device='cpu')
        config = get_model_config(model_path)
        print(f"Actor config: {config.get('actor', {})}")
        return actor

    # Legacy: 单 checkpoint 文件
    print(f"Loading from legacy checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'actor_state_dict' in checkpoint:
        actor_sd = checkpoint['actor_state_dict']
        hidden_sizes = checkpoint.get('hidden_sizes', _infer_hidden_sizes(actor_sd))
        print(f"Format: Pretrain, hidden_sizes={hidden_sizes}")

    elif '_shared_policy.actor.network.0.weight' in checkpoint:
        actor_sd = {}
        prefix = '_shared_policy.actor.'
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                actor_sd[k[len(prefix):]] = v
        hidden_sizes = _infer_hidden_sizes(actor_sd)
        print(f"Format: MAPPO MultiAgentPolicy, hidden_sizes={hidden_sizes}")

    else:
        actor_sd = checkpoint
        hidden_sizes = _infer_hidden_sizes(actor_sd)
        print(f"Format: Direct state_dict, hidden_sizes={hidden_sizes}")

    actor = ActorMLP(obs_dim, hidden_sizes, action_dim)
    actor.load_state_dict(actor_sd)
    actor.eval()
    return actor


def _infer_hidden_sizes(state_dict: dict) -> list:
    """从 MLP state_dict 推断 hidden_sizes（legacy checkpoint 降级路径）。"""
    hidden_sizes = []
    idx = 0
    while f'network.{idx}.weight' in state_dict:
        hidden_sizes.append(state_dict[f'network.{idx}.weight'].shape[0])
        idx += 2
    return hidden_sizes[:-1] if hidden_sizes else []


# =============================================================================
#                          评估主函数
# =============================================================================

def test_trained_policy(
    checkpoint_path: str,
    env_config_path: str,
    num_episodes: int = 5,
    max_steps: int = 1000,
    save_plot: str = None,
    show_plot: bool = True,
    record_animation: bool = False,
    event_driven: bool = True,
    max_frames: int = None,
):
    """
    在 MASUPEnv 中评估训练好的 actor 策略。

    Args:
        checkpoint_path: 模型目录或 checkpoint 文件路径
        env_config_path: 环境配置 YAML 路径（experiment YAML 或独立 eval YAML）
        num_episodes: 评估 episode 数量
        max_steps: 每个 episode 的固定步数
        save_plot: 图表保存路径
        show_plot: 是否显示图表
        record_animation: 是否录制最后一个 episode 的动画视频
        event_driven: True=事件驱动动画，False=固定步长动画
        max_frames: 动画最大帧数限制
    """
    # ---- 加载环境配置 ----
    env_cfg = load_env_config(env_config_path)
    cfg_dict, custom_dict = _env_config_to_dicts(env_cfg)

    graph_path = cfg_dict['graph_path']
    graph_name = Path(graph_path).stem
    num_agents = cfg_dict['num_agents']

    # ---- 创建环境 ----
    print(f"\n=== Creating MASUPEnv ===")
    print(f"Graph: {graph_path} ({graph_name})")
    print(f"Num agents: {num_agents}")
    env = MASUPEnv(cfg_dict, **custom_dict)

    # 从环境推断网络维度
    sample_agent = env.possible_agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    action_dim = env.action_space(sample_agent).n

    print(f"\n=== Network Dimensions ===")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Max neighbors: {env.world.max_neighbors}")

    # ---- 加载 actor ----
    actor = load_trained_actor(checkpoint_path, obs_dim, action_dim)

    # ---- 构建 MultiAgentPolicy ----
    multi_policy = MultiAgentPolicy(
        agent_ids=env.possible_agents,
        obs_space=env.observation_space(env.possible_agents[0]),
        action_space=env.action_space(env.possible_agents[0]),
        policy_class=ActorPolicy,
        policy_kwargs={'actor': actor, 'deterministic_eval': True},
        shared=True,
    )
    multi_policy.set_training_mode(False)

    print(f"\n=== Running {num_episodes} episodes (fixed {max_steps} steps each) ===")

    episode_metrics = []
    metrics_history = []
    episode_times = []

    # 动画录制数据（仅最后一个 episode）
    anim_positions_history = []
    anim_time_intervals = []

    for ep in range(num_episodes):
        obs, infos = env.reset()
        step_count = 0

        is_last = (ep == num_episodes - 1)
        record = record_animation and is_last
        if record:
            anim_positions_history = [env.world.snapshot_agent_positions()]
            anim_time_intervals = []

        while step_count < max_steps:
            action_masks = {
                aid: torch.as_tensor(info['action_mask'], dtype=torch.bool, device=multi_policy.device)
                for aid, info in infos.items()
            }
            obs_tensor = {
                aid: torch.as_tensor(o, dtype=torch.float32, device=multi_policy.device)
                for aid, o in obs.items()
            }

            with torch.no_grad():
                outputs = multi_policy.forward(obs_tensor, action_mask=action_masks)
            actions = {aid: out['act'].cpu().numpy() for aid, out in outputs.items()}

            obs, _, _, _, infos = env.step(actions)
            step_count += 1

            if record:
                anim_time_intervals.append(env.last_time_interval)
                anim_positions_history.append(env.world.snapshot_agent_positions())

        # Episode 结束：收集指标
        final_metrics = env.world.current_metrics
        episode_metrics.append(final_metrics)
        episode_times.append(final_metrics.time)
        metrics_history.append(env.world.metrics_tracker.get_history_dict())

        print(f"Episode {ep + 1}/{num_episodes}: "
              f"IGI={final_metrics.igi:.4f}, "
              f"AGI={final_metrics.agi:.4f}, "
              f"IWI={final_metrics.iwi:.4f}, "
              f"WI={final_metrics.wi:.4f}, "
              f"time={final_metrics.time:.2f}s, "
              f"steps={step_count}")

    # ---- 汇总统计 ----
    print(f"\n=== Summary Statistics ({num_episodes} episodes, {max_steps} steps) ===")
    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # ---- 可视化 ----
    save_dir = str(Path(save_plot).parent) if save_plot else 'evaluators/results'

    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"

        plot_aggregated_metrics(
            aggregated,
            title=f'RL Policy Evaluation ({num_episodes} episodes, {max_steps} steps)',
            subtitle=subtitle,
            save_path=save_plot,
            show=show_plot,
        )

    # ---- 动画 ----
    if record_animation and anim_positions_history:
        print(f"\n=== Generating animation for last episode ===")
        algorithm_name = Path(checkpoint_path).stem
        if event_driven:
            from utils.vis_utils import create_event_driven_animation
            create_event_driven_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                time_intervals=anim_time_intervals,
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=save_dir,
                max_frames=max_frames,
            )
        else:
            from utils.vis_utils import create_animation
            create_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                total_frames=len(anim_positions_history),
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=save_dir,
                max_frames=max_frames,
            )

    return episode_metrics


# =============================================================================
#                              CLI 入口
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RL 策略评估')
    parser.add_argument('--model', type=str,
                        default='models/mappo/imi/final',
                        help='模型目录 (HuggingFace 风格) 或 checkpoint 文件 (legacy)')
    parser.add_argument('--env_config', type=str,
                        default='configs/eval/masup_tsp12.yaml',
                        help='环境配置 YAML (experiment YAML 或独立 eval YAML)')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='评估 episode 数量')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每个 episode 的固定步数')
    parser.add_argument('--save_plot', type=str, default='evaluators/results/rl_eval.png',
                        help='图表保存路径')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图表')
    parser.add_argument('--animation', action='store_true',
                        help='录制最后一个 episode 的动画视频')
    parser.add_argument('--no_event_driven', action='store_true',
                        help='使用固定步长动画（默认为事件驱动动画）')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='动画最大帧数限制（默认不限制，推荐 300~600）')

    args = parser.parse_args()

    test_trained_policy(
        checkpoint_path=args.model,
        env_config_path=args.env_config,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        save_plot=args.save_plot,
        show_plot=not args.no_show,
        record_animation=args.animation,
        event_driven=not args.no_event_driven,
        max_frames=args.max_frames,
    )
