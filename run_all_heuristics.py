#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all heuristic policies with the same runtime environment settings.

Usage:
    python run_all_heuristics.py <map_name> <num_agents> <episode_len> [--init-positions N1 N2 ...]

Examples:
    python run_all_heuristics.py mapA 6 500
    python run_all_heuristics.py cumberland 6 500 --init-positions 0 10 20 30 40 49
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
HEURISTIC_CONFIG_DIR = PROJECT_ROOT / "configs" / "heuristic"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
EVALUATOR = PROJECT_ROOT / "evaluators" / "heuristic_evaluator.py"


def resolve_graph_path(map_name: str) -> Path:
    raw = Path(map_name)
    variants = [raw]
    if not raw.suffix:
        variants.append(raw.with_suffix(".json"))

    candidates: list[Path] = []
    for variant in variants:
        if variant.is_absolute():
            candidates.append(variant)
        else:
            candidates.append(PROJECT_ROOT / variant)
            candidates.append(GRAPHS_DIR / variant)
            candidates.append(GRAPHS_DIR / variant.name)

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"Graph {map_name!r} was not found under {GRAPHS_DIR}")


def list_heuristic_policies() -> list[str]:
    policies = sorted(p.stem for p in HEURISTIC_CONFIG_DIR.glob("*.yaml"))
    if not policies:
        raise FileNotFoundError(f"No heuristic configs found in {HEURISTIC_CONFIG_DIR}")
    return policies


def project_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def run_policy(
    policy: str,
    graph_path: Path,
    num_agents: int,
    episode_len: float,
    init_positions: list[int] | None = None,
) -> int:
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--policy",
        policy,
        project_relative_or_absolute(graph_path),
        str(num_agents),
        str(episode_len),
        "--no_show",
    ]
    if init_positions is not None:
        cmd.extend(["--init-positions", *[str(pos) for pos in init_positions]])

    print(f"\n{'=' * 72}")
    print(f"Running {policy} ...")
    print(f"{'=' * 72}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run heuristic_evaluator for every heuristic policy.",
    )
    parser.add_argument("map_name", help="Map name/path under graphs/, e.g. mapA or cumberland.")
    parser.add_argument("num_agents", type=int, help="Number of agents.")
    parser.add_argument("episode_len", type=float, help="Episode length in simulation time.")
    parser.add_argument(
        "--init-positions",
        type=int,
        nargs="+",
        default=None,
        metavar="NODE",
        help="Fixed initial node IDs; length must equal num_agents. Omit for random starts.",
    )
    args = parser.parse_args()

    if args.num_agents <= 0:
        parser.error("num_agents must be greater than 0")
    if args.episode_len <= 0:
        parser.error("episode_len must be greater than 0")
    if args.init_positions is not None and len(args.init_positions) != args.num_agents:
        parser.error(
            f"--init-positions length ({len(args.init_positions)}) "
            f"must equal num_agents ({args.num_agents})"
        )

    graph_path = resolve_graph_path(args.map_name)
    policies = list_heuristic_policies()

    print(f"Map         : {project_relative_or_absolute(graph_path)}")
    print(f"Num agents  : {args.num_agents}")
    print(f"Episode len : {args.episode_len} (truncate_by_time=True)")
    print(f"Init pos    : {args.init_positions if args.init_positions is not None else 'random'}")
    print(f"Policies    : {', '.join(policies)}")

    failures: list[str] = []
    for policy in policies:
        code = run_policy(
            policy,
            graph_path,
            args.num_agents,
            args.episode_len,
            args.init_positions,
        )
        if code != 0:
            failures.append(policy)

    if failures:
        print(f"\nFailed policies: {', '.join(failures)}")
        return 1

    print("\nAll heuristic policy evaluations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
