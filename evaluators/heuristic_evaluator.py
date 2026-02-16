#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启发式策略评估器。

直接与 PatrolWorld 交互，不依赖 MDP 封装。
支持多 episode 评估、指标聚合、可视化绘图和 MP4 动画生成。
环境配置通过 --env_config 指定 YAML 文件加载（支持 experiment YAML 或独立 eval YAML）。
"""
import os
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import argparse
import yaml
import random
from typing import Dict, List, Any, Optional, Tuple

from envs.mdps.patrol_core import PatrolWorld
from configs.registry import load_env_config, _env_config_to_dicts
from policies.heuritic.heuristic_base import HeuriticBasePolicy
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


# =============================================================================
#                          Evaluator 类
# =============================================================================

class HeuristicEvaluator:
    """
    启发式策略评估器，直接与 PatrolWorld 交互。

    支持功能：
    - 多 episode 评估
    - 时间/步数截断
    - 最后一个 episode 的动画录制
    """

    def __init__(
        self,
        world: PatrolWorld,
        policy: HeuriticBasePolicy,
        num_episodes: int,
        episode_len: float = 5000,
        truncate_by_time: bool = True,
        init_positions: Optional[List[int]] = None,
        record_animation: bool = False,
        event_driven: bool = True,
    ):
        self.world = world
        self.policy = policy
        self.num_episodes = num_episodes
        self.episode_len = episode_len
        self.truncate_by_time = truncate_by_time
        self.init_positions = init_positions
        self.record_animation = record_animation
        self.event_driven = event_driven
        self.metrics_history: List[Dict[str, List[float]]] = []
        # 动画录制数据（仅保留最后一个 episode）
        self.last_positions_history: List[Dict[int, Tuple[int, int, float]]] = []
        self.last_time_intervals: List[float] = []

    def _is_truncated(self) -> bool:
        if self.truncate_by_time:
            return self.world.current_time >= self.episode_len
        else:
            return self.world.step_count >= self.episode_len

    def run_episode(self, record: bool = False) -> Dict[str, List[float]]:
        """运行单个 episode，返回时序指标。"""
        # 重置物理世界和策略
        if self.init_positions:
            self.world.reset(initial_positions=self.init_positions)
        else:
            init_pos = random.sample(list(self.world.graph.nodes), self.world.num_agents)
            self.world.reset(initial_positions=init_pos)
        self.policy.reset()

        # 录制初始化
        positions_history: List[Dict[int, Tuple[int, int, float]]] = []
        time_intervals: List[float] = []
        if record:
            positions_history.append(self.world.snapshot_agent_positions())

        # 运行 episode
        while not self._is_truncated():
            obs_dict = self.world.get_heuristic_obs()
            global_state = self.world.get_global_state_for_heuristic()
            heuristic_actions = self.policy.compute_actions(obs_dict, global_state)
            result = self.world.step_heuristic(heuristic_actions)

            if record:
                time_intervals.append(result.dt)
                positions_history.append(self.world.snapshot_agent_positions())

        if record:
            self.last_positions_history = positions_history
            self.last_time_intervals = time_intervals

        return self.world.metrics_tracker.get_history_dict()

    def evaluate(self) -> Dict[str, Any]:
        """运行所有 episode 并聚合结果。"""
        truncate_mode = "time" if self.truncate_by_time else "steps"
        print(f"Running {self.num_episodes} episodes "
              f"(episode_len={self.episode_len}, truncate_by={truncate_mode})...")

        for ep in range(self.num_episodes):
            is_last = (ep == self.num_episodes - 1)
            record = self.record_animation and is_last
            metrics = self.run_episode(record=record)
            self.metrics_history.append(metrics)
            print(f"  Episode {ep + 1}/{self.num_episodes}: "
                  f"Final WI={metrics['wi'][-1]:.4f}, "
                  f"Final IGI={metrics['igi'][-1]:.4f}")

        return self._aggregate_metrics()

    def generate_animation(
        self, algorithm_name: str, map_name: str, save_dir: str, max_frames: int = None,
    ):
        """用录制的最后一个 episode 数据生成动画视频。"""
        if not self.last_positions_history:
            print("Warning: 没有录制数据，请先运行 evaluate() 且 record_animation=True")
            return

        if self.event_driven:
            from utils.vis_utils import create_event_driven_animation
            create_event_driven_animation(
                map_graph=self.world.graph,
                agent_positions_history=self.last_positions_history,
                time_intervals=self.last_time_intervals,
                algorithm_name=algorithm_name,
                map_name=map_name,
                save_dir=save_dir,
                max_frames=max_frames,
            )
        else:
            from utils.vis_utils import create_animation
            create_animation(
                map_graph=self.world.graph,
                agent_positions_history=self.last_positions_history,
                total_frames=len(self.last_positions_history),
                algorithm_name=algorithm_name,
                map_name=map_name,
                max_frames=max_frames,
                save_dir=save_dir,
            )

    def _aggregate_metrics(self) -> Dict[str, Any]:
        return aggregate_episode_metrics(self.metrics_history)


# =============================================================================
#                          策略工厂
# =============================================================================

def create_policy(policy_name: str, num_agents: int, policy_config: Dict) -> HeuriticBasePolicy:
    """按名称创建启发式策略实例。"""
    policy_map = {
        'ER': 'policies.heuritic.er',
        'HPCC': 'policies.heuritic.hpcc',
    }
    if policy_name not in policy_map:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policy_map.keys())}")

    module = __import__(policy_map[policy_name], fromlist=[policy_name + 'Policy'])
    policy_class = getattr(module, policy_name + 'Policy')
    return policy_class(num_agents, policy_config)


# =============================================================================
#                              CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='启发式策略评估 (直接与 PatrolWorld 交互)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='评估 episode 数量 (default: 10)')
    parser.add_argument('--policy', type=str, default='ER',
                        choices=['ER', 'HPCC'],
                        help='策略名称 (default: ER)')
    parser.add_argument('--env_config', type=str,
                        default='configs/eval/masup_grid.yaml',
                        help='环境配置 YAML (experiment YAML 或独立 eval YAML)')
    parser.add_argument('--policy_config', type=str, default=None,
                        help='策略配置路径 (default: configs/policies/{POLICY}.yaml)')
    parser.add_argument('--save_plot', type=str,
                        default='evaluators/results/heuristic_eval.png',
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

    # ---- 加载环境配置 ----
    env_cfg = load_env_config(args.env_config)
    cfg_dict, custom_dict = _env_config_to_dicts(env_cfg)

    episode_len = cfg_dict.get('episode_len', 5000)
    truncate_by_time = custom_dict.get('truncate_by_time', True)
    init_positions = cfg_dict.get('init_positions', None)
    graph_path = cfg_dict['graph_path']
    graph_name = Path(graph_path).stem

    # ---- 加载策略配置 ----
    policy_config_path = args.policy_config or f"configs/policies/{args.policy}.yaml"
    with open(policy_config_path) as f:
        policy_config = yaml.safe_load(f)

    # ---- 创建 PatrolWorld（不需要 MDP 封装）----
    print(f"Creating PatrolWorld with graph: {graph_path}")
    print(f"  episode_len: {episode_len}, truncate_by_time: {truncate_by_time}")
    print(f"  init_positions: {init_positions if init_positions else 'random'}")
    world = PatrolWorld(cfg_dict)

    # ---- 创建策略 ----
    print(f"Creating {args.policy} policy")
    policy = create_policy(args.policy, world.num_agents, policy_config)

    # ---- 运行评估 ----
    evaluator = HeuristicEvaluator(
        world, policy, args.num_episodes,
        episode_len=episode_len,
        truncate_by_time=truncate_by_time,
        init_positions=init_positions,
        record_animation=args.animation,
        event_driven=not args.no_event_driven,
    )
    aggregated = evaluator.evaluate()

    # ---- 汇总统计 ----
    print("\n=== Final Statistics ===")
    for metric in ['igi', 'agi', 'iwi', 'wi']:
        final_mean = aggregated[f'{metric}_mean'][-1]
        final_std = aggregated[f'{metric}_std'][-1]
        print(f"{metric.upper()}: {final_mean:.4f} ± {final_std:.4f}")

    # ---- 绘图 ----
    print(f"\nPlotting results...")
    save_dir = str(Path(args.save_plot).parent)
    plot_aggregated_metrics(
        aggregated,
        title=f'{args.policy} Policy Evaluation ({args.num_episodes} episodes)',
        save_path=args.save_plot,
        show=not args.no_show,
    )

    # ---- 动画 ----
    if args.animation:
        print(f"\nGenerating animation for last episode...")
        evaluator.generate_animation(
            algorithm_name=args.policy,
            map_name=graph_name,
            save_dir=save_dir,
            max_frames=args.max_frames,
        )


if __name__ == '__main__':
    main()
