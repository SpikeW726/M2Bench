#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键对所有启发式策略运行 heuristic_evaluator。

用法:
    python run_all_heuristics.py <map_name> <num_agents> <episode_len> [--init-positions N1 N2 ...]

示例:
    python run_all_heuristics.py mapA 6 500
    python run_all_heuristics.py mapA 6 500 --init-positions 0 10 20 30 40 49
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
HEURISTIC_CONFIG_DIR = PROJECT_ROOT / "configs" / "heuristic"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
EVALUATOR = PROJECT_ROOT / "evaluators" / "heuristic_evaluator.py"


def resolve_graph_path(map_name: str) -> Path:
    """将地图名解析为 graphs/ 下的 JSON 文件路径。"""
    stem = Path(map_name).stem
    candidates = [
        GRAPHS_DIR / f"{stem}.json",
        GRAPHS_DIR / map_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"未找到地图文件: {map_name!r}，请检查 {GRAPHS_DIR} 目录"
    )


def list_heuristic_policies() -> list[str]:
    """读取 configs/heuristic 下所有策略名。"""
    policies = sorted(p.stem for p in HEURISTIC_CONFIG_DIR.glob("*.yaml"))
    if not policies:
        raise FileNotFoundError(f"未找到启发式配置: {HEURISTIC_CONFIG_DIR}")
    return policies


def build_runtime_config(
    policy: str,
    graph_path: Path,
    num_agents: int,
    episode_len: float,
    init_positions: list[int] | None = None,
) -> dict:
    """基于模板配置生成本次评估用的运行时 YAML 内容。"""
    template_path = HEURISTIC_CONFIG_DIR / f"{policy}.yaml"
    with open(template_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(cfg)
    env = cfg.setdefault("env", {})
    env["graph_path"] = str(graph_path.relative_to(PROJECT_ROOT))
    env["num_agents"] = num_agents
    env["episode_len"] = episode_len
    if init_positions is not None:
        env["init_positions"] = init_positions
    else:
        env.pop("init_positions", None)

    custom = env.setdefault("custom_configs", {})
    custom["truncate_by_time"] = True

    eval_cfg = cfg.setdefault("eval", {})
    map_name = graph_path.stem
    eval_cfg["save_plot"] = (
        f"evaluators/results/{map_name}/{policy.lower()}/{policy}_eval.png"
    )
    return cfg


def run_policy(
    policy: str,
    graph_path: Path,
    num_agents: int,
    episode_len: float,
    tmp_dir: Path,
    init_positions: list[int] | None = None,
) -> int:
    """为单个策略生成临时配置并调用 heuristic_evaluator。"""
    runtime_cfg = build_runtime_config(
        policy, graph_path, num_agents, episode_len, init_positions,
    )
    config_path = tmp_dir / f"{policy}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False, allow_unicode=True)

    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--policy",
        policy,
        "--config",
        str(config_path),
        "--no_show",
    ]
    print(f"\n{'=' * 72}")
    print(f"Running {policy} ...")
    print(f"Config: {config_path}")
    print(f"{'=' * 72}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对所有启发式策略运行 heuristic_evaluator",
    )
    parser.add_argument(
        "map_name",
        help="目标地图名，对应 graphs/ 下的文件名（如 mapA 或 mapA.json）",
    )
    parser.add_argument(
        "num_agents",
        type=int,
        help="智能体数量",
    )
    parser.add_argument(
        "episode_len",
        type=float,
        help="按物理时间截断 episode 的时长",
    )
    parser.add_argument(
        "--init-positions",
        type=int,
        nargs="+",
        default=None,
        metavar="NODE",
        help="固定初始节点 ID 列表，长度须等于 num_agents；未指定则每个 episode 随机初始化",
    )
    args = parser.parse_args()

    if args.num_agents <= 0:
        parser.error("num_agents 必须大于 0")
    if args.episode_len <= 0:
        parser.error("episode_len 必须大于 0")
    if args.init_positions is not None and len(args.init_positions) != args.num_agents:
        parser.error(
            f"--init-positions 长度 ({len(args.init_positions)}) "
            f"须等于 num_agents ({args.num_agents})"
        )

    graph_path = resolve_graph_path(args.map_name)
    policies = list_heuristic_policies()

    init_desc = (
        str(args.init_positions) if args.init_positions is not None else "random"
    )
    print(f"Map         : {graph_path.relative_to(PROJECT_ROOT)}")
    print(f"Num agents  : {args.num_agents}")
    print(f"Episode len : {args.episode_len} (truncate_by_time=True)")
    print(f"Init pos    : {init_desc}")
    print(f"Policies    : {', '.join(policies)}")

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="heuristic_eval_") as tmp:
        tmp_dir = Path(tmp)
        for policy in policies:
            code = run_policy(
                policy,
                graph_path,
                args.num_agents,
                args.episode_len,
                tmp_dir,
                args.init_positions,
            )
            if code != 0:
                failures.append(policy)

    if failures:
        print(f"\n失败策略: {', '.join(failures)}")
        return 1

    print("\n全部启发式策略评估完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
