#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one heuristic patrolling policy directly on PatrolWorld.

Heuristic YAML files only define policy/eval defaults. Runtime environment
settings are supplied by the command line.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from policies.heuritic.heuristic_base import HeuriticBasePolicy
from utils.project_paths import DEFAULT_RESULTS_DIR, user_path

matplotlib_config_dir = Path(
    os.environ.get(
        "MPLCONFIGDIR",
        Path(os.environ.get("TEMP", project_root / ".tmp")) / "map_imitation_matplotlib",
    )
)
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics


GRAPHS_DIR = project_root / "graphs"


def load_patrol_world_class():
    module_path = project_root / "envs" / "mdps" / "patrol_core.py"
    spec = importlib.util.spec_from_file_location("_heuristic_patrol_core", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PatrolWorld from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PatrolWorld


PatrolWorld = load_patrol_world_class()


ENV_DEFAULTS: Dict[str, Any] = {
    "enable_wait": False,
    "deltaT": 0.5,
    "max_time_for_obs": None,
    "norm_reward": False,
    "norm_obs": False,
    "edge_time_jitter_mode": "none",
    "edge_time_jitter_frac": 0.1,
    "edge_time_jitter_seed": None,
}


POLICY_MAP = {
    "ER": ("policies.heuritic.er", "ERPolicy"),
    "HPCC": ("policies.heuritic.hpcc", "HPCCPolicy"),
    "HCR": ("policies.heuritic.hcr", "HCRPolicy"),
    "GBS": ("policies.heuritic.gbs", "GBSPolicy"),
    "SEBS": ("policies.heuritic.sebs", "SEBSPolicy"),
    "BAPS": ("policies.heuritic.baps", "BAPSPolicy"),
    "CBLS": ("policies.heuritic.cbls", "CBLSPolicy"),
    "RANDOM": ("policies.heuritic.random", "RandomPolicy"),
    "MSP": ("policies.heuritic.msp", "MSPPolicy"),
    "DTAGREEDY": ("policies.heuritic.dta_greedy", "DTAGreedyPolicy"),
    "DTASSI": ("policies.heuritic.dta_ssi", "DTASSIPolicy"),
    "AHPA": ("policies.heuritic.ahpa", "AHPAPolicy"),
    "CR": ("policies.heuritic.conscientious_reactive", "ConscientiousReactivePolicy"),
    "CC": ("policies.heuritic.conscientious_cognitive", "ConscientiousCognitivePolicy"),
}


def _project_relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def resolve_graph_path(map_name_or_path: str) -> str:
    """Resolve a graph name/path to a usable path.

    Examples accepted by this function:
      mapA
      mapA.json
      graphs/mapA.json
      C:/somewhere/mapA.json
    """
    raw = Path(map_name_or_path)
    raw_variants = [raw]
    if not raw.suffix:
        raw_variants.append(raw.with_suffix(".json"))

    candidates: List[Path] = []
    for variant in raw_variants:
        if variant.is_absolute():
            candidates.append(variant)
        else:
            candidates.append(project_root / variant)
            candidates.append(GRAPHS_DIR / variant)
            candidates.append(GRAPHS_DIR / variant.name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return _project_relative_or_absolute(candidate)

    raise FileNotFoundError(
        f"Graph {map_name_or_path!r} was not found. Checked project root and {GRAPHS_DIR}."
    )


def parse_key_value_overrides(values: Optional[List[str]]) -> Dict[str, Any]:
    """Parse KEY=VALUE overrides, using YAML scalars/lists for VALUE."""
    overrides: Dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override must look like KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty in {item!r}")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def default_save_plot(
    policy_name: str,
    graph_path: str,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> str:
    map_name = Path(graph_path).stem
    return str(user_path(results_dir) / map_name / policy_name.lower() / f"{policy_name}_eval.png")


def build_runtime_env_config(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_dict = dict(ENV_DEFAULTS)
    cfg_dict = {k: v for k, v in cfg_dict.items() if v is not None}
    cfg_dict.update(parse_key_value_overrides(args.env_override))

    cfg_dict["graph_path"] = resolve_graph_path(args.map_name)
    cfg_dict["num_agents"] = args.num_agents
    cfg_dict["episode_len"] = args.episode_len

    if args.init_positions is not None:
        cfg_dict["init_positions"] = args.init_positions

    if args.speeds is not None:
        cfg_dict["speeds"] = args.speeds
    if args.deltaT is not None:
        cfg_dict["deltaT"] = args.deltaT
    if args.max_time_for_obs is not None:
        cfg_dict["max_time_for_obs"] = args.max_time_for_obs
    if args.enable_wait:
        cfg_dict["enable_wait"] = True
    if args.disable_wait:
        cfg_dict["enable_wait"] = False
    if args.edge_time_jitter_mode is not None:
        cfg_dict["edge_time_jitter_mode"] = args.edge_time_jitter_mode
    if args.edge_time_jitter_frac is not None:
        cfg_dict["edge_time_jitter_frac"] = args.edge_time_jitter_frac
    if args.edge_time_jitter_seed is not None:
        cfg_dict["edge_time_jitter_seed"] = args.edge_time_jitter_seed

    custom_dict = {"truncate_by_time": not args.truncate_by_steps}

    num_agents = int(cfg_dict["num_agents"])
    episode_len = float(cfg_dict["episode_len"])
    init_positions = cfg_dict.get("init_positions")
    speeds = cfg_dict.get("speeds")

    if num_agents <= 0:
        raise ValueError("num_agents must be greater than 0")
    if episode_len <= 0:
        raise ValueError("episode_len must be greater than 0")
    if init_positions is not None and len(init_positions) != num_agents:
        raise ValueError(
            f"init_positions length ({len(init_positions)}) must equal num_agents ({num_agents})"
        )
    if speeds is not None and len(speeds) != num_agents:
        raise ValueError(f"speeds length ({len(speeds)}) must equal num_agents ({num_agents})")
    if speeds is not None and any(float(speed) <= 0 for speed in speeds):
        raise ValueError("all speeds must be greater than 0")

    cfg_dict["num_agents"] = num_agents
    cfg_dict["episode_len"] = episode_len
    return cfg_dict, custom_dict


class HeuristicEvaluator:
    """Evaluate a heuristic policy by interacting directly with PatrolWorld."""

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
        self.last_positions_history: List[Dict[int, Tuple[int, int, float]]] = []
        self.last_time_intervals: List[float] = []

    def _is_truncated(self) -> bool:
        if self.truncate_by_time:
            return self.world.current_time >= self.episode_len
        return self.world.step_count >= self.episode_len

    def run_episode(self, record: bool = False) -> Dict[str, List[float]]:
        if self.init_positions:
            self.world.reset(initial_positions=self.init_positions)
        else:
            init_pos = random.sample(list(self.world.graph.nodes), self.world.num_agents)
            self.world.reset(initial_positions=init_pos)
        self.policy.reset()

        positions_history: List[Dict[int, Tuple[int, int, float]]] = []
        time_intervals: List[float] = []
        if record:
            positions_history.append(self.world.snapshot_agent_positions())

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
        truncate_mode = "time" if self.truncate_by_time else "steps"
        print(
            f"Running {self.num_episodes} episodes "
            f"(episode_len={self.episode_len}, truncate_by={truncate_mode})..."
        )

        for ep in range(self.num_episodes):
            is_last = ep == self.num_episodes - 1
            record = self.record_animation and is_last
            metrics = self.run_episode(record=record)
            self.metrics_history.append(metrics)
            print(
                f"  Episode {ep + 1}/{self.num_episodes}: "
                f"Final WI={metrics['wi'][-1]:.4f}, "
                f"Final IGI={metrics['igi'][-1]:.4f}"
            )

        return self._aggregate_metrics()

    def generate_animation(
        self,
        algorithm_name: str,
        map_name: str,
        save_dir: str,
        max_frames: Optional[int] = None,
    ) -> None:
        if not self.last_positions_history:
            print("Warning: no recorded episode data. Run evaluate() with animation enabled first.")
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


def create_policy(policy_name: str, num_agents: int, policy_config: Dict[str, Any]) -> HeuriticBasePolicy:
    """Create a heuristic policy by name."""
    if policy_name not in POLICY_MAP:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICY_MAP.keys())}")

    module_path, class_name = POLICY_MAP[policy_name]
    module = __import__(module_path, fromlist=[class_name])
    policy_class = getattr(module, class_name)
    return policy_class(num_agents, policy_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one heuristic policy directly on PatrolWorld.",
    )
    parser.add_argument("map_name", help="Map name/path, e.g. cumberland or graphs/cumberland.json.")
    parser.add_argument("num_agents", type=int, help="Number of agents.")
    parser.add_argument("episode_len", type=float, help="Episode length; time by default, steps with --truncate-by-steps.")
    parser.add_argument(
        "--policy",
        type=str,
        default="ER",
        choices=list(POLICY_MAP.keys()),
        help="Policy name (default: ER).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path (default: configs/heuristic/{POLICY}.yaml).",
    )

    parser.add_argument(
        "--init_positions",
        "--init-positions",
        dest="init_positions",
        type=int,
        nargs="+",
        default=None,
        metavar="NODE",
    )
    parser.add_argument("--speeds", type=float, nargs="+", default=None, metavar="SPEED")
    parser.add_argument("--deltaT", type=float, default=None)
    parser.add_argument("--max_time_for_obs", "--max-time-for-obs", dest="max_time_for_obs", type=float, default=None)
    parser.add_argument("--enable_wait", "--enable-wait", dest="enable_wait", action="store_true")
    parser.add_argument("--disable_wait", "--disable-wait", dest="disable_wait", action="store_true")
    parser.add_argument("--truncate_by_steps", "--truncate-by-steps", dest="truncate_by_steps", action="store_true")
    parser.add_argument(
        "--edge_time_jitter_mode",
        "--edge-time-jitter-mode",
        dest="edge_time_jitter_mode",
        choices=["none", "dual", "full"],
        default=None,
    )
    parser.add_argument(
        "--edge_time_jitter_frac",
        "--edge-time-jitter-frac",
        dest="edge_time_jitter_frac",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--edge_time_jitter_seed",
        "--edge-time-jitter-seed",
        dest="edge_time_jitter_seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--env",
        dest="env_override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Generic EnvConfig override, e.g. --env deltaT=1.0 --env speeds='[1,1,1]'.",
    )

    parser.add_argument("--num_episodes", type=int, default=None, help="Override eval.num_episodes.")
    parser.add_argument("--save_plot", type=str, default=None, help="Output plot path.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Evaluation output root (default: project evaluators/results).",
    )
    parser.add_argument("--no_show", action="store_true", help="Do not show plots.")
    parser.add_argument("--animation", action="store_true", help="Record animation for the last episode.")
    parser.add_argument("--no_event_driven", action="store_true", help="Use fixed-frame animation.")
    parser.add_argument("--max_frames", type=int, default=None, help="Override eval.max_frames.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.config_path = args.config or f"configs/heuristic/{args.policy}.yaml"

    if args.enable_wait and args.disable_wait:
        parser.error("--enable-wait and --disable-wait cannot be used together")

    try:
        with open(args.config_path, encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        cfg_dict, custom_dict = build_runtime_env_config(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    graph_path = cfg_dict["graph_path"]
    graph_name = Path(graph_path).stem
    episode_len = cfg_dict.get("episode_len", 5000)
    truncate_by_time = custom_dict.get("truncate_by_time", True)
    init_positions = cfg_dict.get("init_positions")

    eval_cfg = raw_cfg.get("eval", {})
    num_episodes = (
        args.num_episodes if args.num_episodes is not None else eval_cfg.get("num_episodes", 10)
    )

    save_plot = args.save_plot or default_save_plot(
        args.policy,
        graph_path,
        args.results_dir or DEFAULT_RESULTS_DIR,
    )

    show_plot = eval_cfg.get("show_plot", False) and not args.no_show
    animation = args.animation or eval_cfg.get("animation", False)
    event_driven = eval_cfg.get("event_driven", True) and not args.no_event_driven
    max_frames = args.max_frames if args.max_frames is not None else eval_cfg.get("max_frames", None)

    print(f"Config : {args.config_path}")
    print(f"Graph  : {graph_path}")
    print(f"Agents : {cfg_dict['num_agents']}")
    print(f"episode_len={episode_len}, truncate_by_time={truncate_by_time}")
    print(f"init_positions: {init_positions if init_positions else 'random'}")
    if cfg_dict.get("speeds") is not None:
        print(f"speeds: {cfg_dict['speeds']}")
    if custom_dict:
        print(f"custom_configs: {custom_dict}")

    world = PatrolWorld(cfg_dict)

    print(f"Policy : {args.policy}")
    policy = create_policy(args.policy, world.num_agents, raw_cfg)

    evaluator = HeuristicEvaluator(
        world,
        policy,
        num_episodes,
        episode_len=episode_len,
        truncate_by_time=truncate_by_time,
        init_positions=init_positions,
        record_animation=animation,
        event_driven=event_driven,
    )
    aggregated = evaluator.evaluate()

    print("\n=== Final Statistics ===")
    for metric in ["igi", "agi", "iwi", "wi"]:
        final_mean = aggregated[f"{metric}_mean"][-1]
        final_std = aggregated[f"{metric}_std"][-1]
        print(f"{metric.upper()}: {final_mean:.4f} +/- {final_std:.4f}")

    print("\nPlotting results...")
    save_dir = str(Path(save_plot).parent)
    plot_aggregated_metrics(
        aggregated,
        title=f"{args.policy} Policy Evaluation ({num_episodes} episodes)",
        save_path=save_plot,
        show=show_plot,
    )

    if animation:
        print("\nGenerating animation for last episode...")
        evaluator.generate_animation(
            algorithm_name=args.policy,
            map_name=graph_name,
            save_dir=save_dir,
            max_frames=max_frames,
        )


if __name__ == "__main__":
    main()
