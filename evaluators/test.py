#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 RL 策略评估脚本。

支持所有算法 (MAPPO/IPPO/IQL/D3QN/VDN/QMIX 等)、
所有网络 (MLP/RNN) 和所有 MDP 环境 (MASUPEnv/S4R1Env/OUCSEnv 等)。

模型重建信息来自模型目录的 config.yaml (训练时自动保存);
环境类型和参数来自 eval YAML。

用法示例:
  1) 基础评估
     python evaluators/test.py --model models/iql-s4r1-TSP12/2026-03-01_17-40-36/final \
                               --env_config configs/eval/s4r1_tsp12.yaml

  2) 生成最后一个 episode 动画
     python evaluators/test.py --model models/mappo-masup-TSP12/2026-03-01_10-20-30/final \
                               --env_config configs/eval/masup_tsp12.yaml \
                               --animation --max_frames 500

  3) 仅保存图，不弹窗
     python evaluators/test.py --model models/xxx/final \
                               --env_config configs/eval/suns_tsp12.yaml \
                               --save_plot evaluators/results/eval.png --no_show
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import yaml
import torch
import numpy as np

from configs.registry import (
    ENV_REGISTRY,
    _import_class,
    load_eval_config,
    _env_config_to_dicts,
)
from utils.model_io import load_policy_for_eval, get_model_config
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


# =============================================================================
#                          环境创建
# =============================================================================

def _create_env(env_type: str, env_config):
    """通过 ENV_REGISTRY 动态创建单个环境实例。"""
    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])
    cfg_dict, custom_dict = _env_config_to_dicts(env_config)
    return env_cls(cfg_dict, **custom_dict)


# =============================================================================
#                          评估主函数
# =============================================================================

def test_trained_policy(
    model_dir: str,
    env_config_path: str,
    num_episodes: int = 5,
    max_steps: int = 1000,
    save_plot: str = None,
    show_plot: bool = True,
    record_animation: bool = False,
    event_driven: bool = True,
    max_frames: int = None,
    save_animation_dir: str = None,
):
    """
    评估训练好的策略，自动适配算法/网络/环境类型。

    Args:
        model_dir: 模型目录 (含 config.yaml + policy.pt)
        env_config_path: eval YAML 路径 (含 env_type + env 参数)
        num_episodes: 评估 episode 数量
        max_steps: 每个 episode 的固定步数
        save_plot: 图表保存路径
        show_plot: 是否显示图表
        record_animation: 是否录制最后一个 episode 的动画
        event_driven: True=事件驱动动画，False=固定步长动画
        max_frames: 动画最大帧数限制
        save_animation_dir: 动画保存目录；None 时回退到 save_plot 的父目录
    """
    # ---- 1. 加载环境配置 & 创建环境 ----
    env_type, env_cfg = load_eval_config(env_config_path)

    print(f"\n=== Creating {env_type} environment ===")
    env = _create_env(env_type, env_cfg)

    cfg_dict, _ = _env_config_to_dicts(env_cfg)
    graph_path = cfg_dict.get("graph_path", "unknown")
    graph_name = Path(graph_path).stem
    num_agents = cfg_dict.get("num_agents", len(env.possible_agents))

    print(f"Graph: {graph_path} ({graph_name})")
    print(f"Num agents: {num_agents}")

    # ---- 2. 从环境推断维度 ----
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_space(sample_agent)
    action_space = env.action_space(sample_agent)
    obs_dim = obs_space.shape[0]
    action_dim = action_space.n

    print(f"\n=== Network Dimensions ===")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # ---- 3. 加载策略 ----
    model_dir = Path(model_dir)
    model_config = get_model_config(model_dir)
    extra = model_config.get("extra", {})
    algo_name = extra.get("algo_name", "unknown")

    print(f"\n=== Loading model ({algo_name}) from {model_dir} ===")
    multi_policy = load_policy_for_eval(
        model_dir=model_dir,
        agent_ids=env.possible_agents,
        obs_space=obs_space,
        action_space=action_space,
        device='cpu',
    )

    # ---- 4. 评估循环 ----
    print(f"\n=== Running {num_episodes} episodes (fixed {max_steps} steps each) ===")

    episode_metrics = []
    metrics_history = []
    episode_times = []

    anim_positions_history = []
    anim_time_intervals = []

    for ep in range(num_episodes):
        obs, infos = env.reset()
        step_count = 0
        hidden_state = None  # episode 开始时重置 RNN hidden state

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
                outputs = multi_policy.forward(obs_tensor, state_dict=hidden_state, action_mask=action_masks)
            actions = {aid: out['act'].cpu().numpy() for aid, out in outputs.items()}
            # 保留 RNN hidden state 供下一步使用
            if multi_policy.is_recurrent:
                hidden_state = {aid: out['state'] for aid, out in outputs.items() if out.get('state') is not None}

            obs, _, _, _, infos = env.step(actions)
            step_count += 1

            if record:
                anim_time_intervals.append(getattr(env, 'last_time_interval', 1.0))
                anim_positions_history.append(env.world.snapshot_agent_positions())

        # Episode 结束: 收集指标
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

    # ---- 5. 汇总统计 ----
    print(f"\n=== Summary Statistics ({num_episodes} episodes, {max_steps} steps) ===")
    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # ---- 6. 可视化 ----
    # 动画目录：优先用显式传入的 save_animation_dir，回退到 save_plot 的父目录
    anim_dir = save_animation_dir or (str(Path(save_plot).parent) if save_plot else 'evaluators/results')

    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"

        plot_aggregated_metrics(
            aggregated,
            title=f'{algo_name.upper()} Evaluation ({num_episodes} episodes, {max_steps} steps)',
            subtitle=subtitle,
            save_path=save_plot,
            show=show_plot,
        )

    # ---- 7. 动画 ----
    if record_animation and anim_positions_history:
        print(f"\n=== Generating animation for last episode ===")
        algorithm_name = algo_name or Path(model_dir).stem
        if event_driven:
            from utils.vis_utils import create_event_driven_animation
            create_event_driven_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                time_intervals=anim_time_intervals,
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=anim_dir,
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
                save_dir=anim_dir,
                max_frames=max_frames,
            )

    return episode_metrics


# =============================================================================
#                     Q-table 评估（BBLA / GBLA / ExGBLA 等）
# =============================================================================

def test_qtable_policy(
    model_dir: str,
    env_config_path: str,
    num_episodes: int = 5,
    save_plot: str = None,
    show_plot: bool = True,
    record_animation: bool = False,
    event_driven: bool = True,
    max_frames: int = None,
    save_animation_dir: str = None,
):
    """评估训练好的 Q-table 策略。

    支持 ParallelEnv（BBLA / GBLA / ExGBLA，per-agent 独立 Q-table）
    和 Gymnasium Env（JointBaseEnv 系列，单一集中式 Q-table）。

    Args:
        model_dir: checkpoint 目录（含 *_qtable.npy 文件）
        env_config_path: eval YAML（含 env_type + env 参数 + algo_name: qtable）
        num_episodes: 评估 episode 数
        save_plot: 指标图保存路径（None = 不保存）
        show_plot: 是否弹窗显示图表
        record_animation: 是否录制最后一个 episode 的动画
        event_driven: True=事件驱动动画，False=固定步长动画
        max_frames: 动画最大帧数限制
        save_animation_dir: 动画保存目录；None 时回退到 save_plot 的父目录
    """
    from algorithms.tabular.qtable import QTablePolicy, QTableAlgo
    from configs.algo_configs import QTableParams

    # ---- 1. 创建环境 ----
    env_type, env_cfg = load_eval_config(env_config_path)
    print(f"\n=== Creating {env_type} environment ===")
    env = _create_env(env_type, env_cfg)

    cfg_dict, _ = _env_config_to_dicts(env_cfg)
    graph_path = cfg_dict.get("graph_path", "unknown")
    graph_name = Path(graph_path).stem
    num_agents = cfg_dict.get("num_agents", None)

    print(f"Graph: {graph_path} ({graph_name})")

    # ---- 2. 检测环境类型并构建 Q-table ----
    is_parallel = hasattr(env, "possible_agents")  # ParallelEnv vs Gymnasium Env

    if is_parallel:
        agent_ids = env.possible_agents
        action_dim = env.action_space(agent_ids[0]).n
        num_agents = num_agents or len(agent_ids)
    else:
        agent_ids = ["agent_0"]
        action_dim = env.action_space.n
        num_agents = num_agents or 1

    print(f"Num agents: {num_agents}, Action dim: {action_dim}")
    print(f"Mode: {'ParallelEnv (per-agent Q-table)' if is_parallel else 'GymnasiumEnv (joint Q-table)'}")

    # 评估时 epsilon=0（纯 greedy），lr/gamma 不影响 eval，填默认值即可
    dummy_params = QTableParams(lr=0.0, gamma=0.99,
                                epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0)
    policies = {aid: QTablePolicy(action_dim, epsilon=0.0) for aid in agent_ids}
    algo = QTableAlgo(policies, dummy_params)

    # ---- 3. 加载 Q-table ----
    model_dir = Path(model_dir)
    algo.load(str(model_dir))
    for pol in algo.policies.values():
        pol.set_epsilon(0.0)

    qtable_sizes = {aid: len(pol.q_table) for aid, pol in algo.policies.items()}
    print(f"\n=== Q-table loaded from {model_dir} ===")
    print(f"Q-table sizes: {qtable_sizes}")

    # ---- 4. 评估循环 ----
    print(f"\n=== Running {num_episodes} episodes (greedy, epsilon=0) ===")

    episode_metrics = []
    metrics_history = []
    episode_times = []
    anim_positions_history = []
    anim_time_intervals = []

    for ep in range(num_episodes):
        obs, infos = env.reset()
        truncated = False
        terminated = False

        is_last = (ep == num_episodes - 1)
        record = record_animation and is_last
        if record:
            anim_positions_history = [env.world.snapshot_agent_positions()]
            anim_time_intervals = []

        while not (truncated or terminated):
            if is_parallel:
                actions = {}
                for agent_str in env.agents:
                    info_i = infos[agent_str]
                    action_mask = info_i.get("action_mask", None)
                    pol = algo.policies[agent_str]

                    if info_i.get("active_mask", 1):
                        if action_mask is not None:
                            actions[agent_str] = pol.select_action(obs[agent_str], action_mask)
                        else:
                            actions[agent_str] = int(np.argmax(pol.get_q(obs[agent_str])))
                    else:
                        # ON_EDGE：选 action_mask 中最后一个有效位（no-op）
                        if action_mask is not None:
                            valid = np.where(action_mask)[0]
                            actions[agent_str] = int(valid[-1]) if len(valid) > 0 else 0
                        else:
                            actions[agent_str] = 0

                obs, _, terms, truncs, infos = env.step(actions)
                first = env.agents[0]
                truncated = bool(truncs[first])
                terminated = bool(terms[first])

            else:
                # Gymnasium Env（Joint 控制器）
                info_i = infos if isinstance(infos, dict) else {}
                action_mask = info_i.get("action_mask", None)
                pol = algo.policies["agent_0"]
                if action_mask is not None:
                    action = pol.select_action(obs, action_mask)
                else:
                    action = int(np.argmax(pol.get_q(obs)))
                obs, _, terminated, truncated, infos = env.step(action)

            if record:
                anim_time_intervals.append(getattr(env, 'last_time_interval', 1.0))
                anim_positions_history.append(env.world.snapshot_agent_positions())

        final_metrics = env.world.current_metrics
        episode_metrics.append(final_metrics)
        episode_times.append(final_metrics.time)
        metrics_history.append(env.world.metrics_tracker.get_history_dict())

        print(f"Episode {ep + 1}/{num_episodes}: "
              f"IGI={final_metrics.igi:.4f}, "
              f"AGI={final_metrics.agi:.4f}, "
              f"IWI={final_metrics.iwi:.4f}, "
              f"WI={final_metrics.wi:.4f}, "
              f"time={final_metrics.time:.2f}s")

    # ---- 5. 汇总统计 ----
    print(f"\n=== Summary Statistics ({num_episodes} episodes) ===")
    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # ---- 6. 可视化 ----
    # 动画目录：优先用显式传入的 save_animation_dir，回退到 save_plot 的父目录
    anim_dir = save_animation_dir or (str(Path(save_plot).parent) if save_plot else 'evaluators/results')
    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"
        plot_aggregated_metrics(
            aggregated,
            title=f'Q-table Evaluation ({num_episodes} episodes)',
            subtitle=subtitle,
            save_path=save_plot,
            show=show_plot,
        )

    # ---- 7. 动画 ----
    if record_animation and anim_positions_history:
        print(f"\n=== Generating animation for last episode ===")
        algorithm_name = "qtable"
        if event_driven:
            from utils.vis_utils import create_event_driven_animation
            create_event_driven_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                time_intervals=anim_time_intervals,
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=anim_dir,
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
                save_dir=anim_dir,
                max_frames=max_frames,
            )

    return episode_metrics


# =============================================================================
#                     自动化评估入口（供 train.py / sweep.py 调用）
# =============================================================================

def run_eval_from_config(model_dir: str, eval_config_path: str, extra_params: dict = None):
    """从 eval yaml 读取所有评估参数，自动调用对应评估函数。

    eval yaml 需包含 env_type、env 段（环境参数），以及可选的 eval 段：
        eval:
          num_episodes: 10
          max_steps: 1000
          save_plot: evaluators/results/auto_eval.png
          show_plot: false
          animation: false
          event_driven: true
          max_frames: null

    Args:
        model_dir: 模型目录。
        eval_config_path: eval YAML 路径。
        extra_params: 可选的覆盖字段，会合并到 yaml 的 eval 段之上（用于 sweep 批量评估时
                      为每个 trial 生成独立的 save_plot / save_animation_dir 等）。

    若未提供 eval 段，则使用各评估函数的默认参数。
    """
    with open(eval_config_path) as f:
        raw = yaml.safe_load(f)
    algo_name = raw.get("algo_name", None)
    eval_params = raw.get("eval", {})

    # extra_params 覆盖 yaml 中的 eval 字段
    if extra_params:
        eval_params.update(extra_params)

    # animation 字段在 CLI 中叫 record_animation，统一映射
    if "animation" in eval_params:
        eval_params.setdefault("record_animation", eval_params.pop("animation"))

    print(f"\n[Eval] model_dir={model_dir}, config={eval_config_path}")
    if algo_name == "qtable":
        test_qtable_policy(
            model_dir=model_dir,
            env_config_path=eval_config_path,
            **eval_params,
        )
    else:
        test_trained_policy(
            model_dir=model_dir,
            env_config_path=eval_config_path,
            **eval_params,
        )


# =============================================================================
#                              CLI 入口
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='通用 RL 策略评估')
    parser.add_argument('--model', type=str,
                        default='models/mappo/imi/final',
                        help='模型目录 (含 config.yaml + policy.pt)')
    parser.add_argument('--env_config', type=str,
                        default='configs/eval/masup_tsp12.yaml',
                        help='eval YAML (含 env_type + 环境参数)')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='评估 episode 数量（未指定时从 env_config 的 eval 段读取）')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='每个 episode 的固定步数（未指定时从 env_config 的 eval 段读取）')
    parser.add_argument('--save_plot', type=str, default='evaluators/results/eval.png',
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

    # 读取 yaml：algo_name + eval 段。CLI 显式传入覆盖 yaml，未传入则用 yaml
    with open(args.env_config) as _f:
        _raw = yaml.safe_load(_f)
    _algo_name = _raw.get("algo_name", None)
    _eval = _raw.get("eval", {})

    num_episodes = args.num_episodes if args.num_episodes is not None else _eval.get("num_episodes", 5)
    max_steps = args.max_steps if args.max_steps is not None else _eval.get("max_steps", 1000)

    if _algo_name == "qtable":
        test_qtable_policy(
            model_dir=args.model,
            env_config_path=args.env_config,
            num_episodes=num_episodes,
            save_plot=args.save_plot,
            show_plot=not args.no_show,
            record_animation=args.animation,
            event_driven=not args.no_event_driven,
            max_frames=args.max_frames,
        )
    else:
        test_trained_policy(
            model_dir=args.model,
            env_config_path=args.env_config,
            num_episodes=num_episodes,
            max_steps=max_steps,
            save_plot=args.save_plot,
            show_plot=not args.no_show,
            record_animation=args.animation,
            event_driven=not args.no_event_driven,
            max_frames=args.max_frames,
        )
