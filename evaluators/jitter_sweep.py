#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jitter 扰动扫描评估脚本。

以用户指定的 eval YAML 为模板，遍历不同 edge_time_jitter_frac（边上运动时间扰动幅度 ε），
对每个 frac 运行多次评估，汇总 WI 和 AGI 指标的均值/方差，
最终写入 CSV 文件，便于后续绘制「扰动程度 — 指标」曲线。

机制说明：
- 扰动幅度由 env.edge_time_jitter_frac 控制，实际运动时间 ∈ [T*(1-ε), T*(1+ε)]。
  具体见 envs/mdps/patrol_core.py 中 PatrolWorld._jitter_frac 的用法。
- 本脚本不修改用户的原始 yaml，而是基于其内容在内存中改写 frac 后写临时文件，
  交给 evaluators/test.py::test_trained_policy 完成单次评估。

输出文件：MAP-imitation-framework/evaluators/jitter/<原yaml文件名>.csv
列：jitter_frac, WI_mean, WI_std, AGI_mean, AGI_std
"""
import os
import sys
import csv
import copy
import tempfile
from pathlib import Path

# 添加项目根目录到 sys.path（与 evaluators/test.py 一致）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import yaml
import numpy as np
import argparse

from evaluators.test import test_trained_policy


# 默认扫描的扰动程度（与用户绘图横轴一致）
DEFAULT_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

# 输出目录（脚本所在目录的 jitter 子目录）
OUTPUT_DIR = Path(__file__).resolve().parent / "jitter"


def _build_temp_yaml(src_yaml_path: Path, frac: float) -> Path:
    """读取源 yaml，覆写 env.edge_time_jitter_frac，写入临时文件并返回路径。

    若源 yaml 未显式设置 edge_time_jitter_mode（即 "none"），强制改为 "full"，
    否则 frac 不起作用（patrol_core 仅在 mode != "none" 时启用扰动）。
    """
    with open(src_yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    env_section = raw.get("env", {})
    if not isinstance(env_section, dict):
        env_section = {}
        raw["env"] = env_section

    env_section["edge_time_jitter_frac"] = float(frac)
    # frac == 0 时仍允许 mode="full"，等价于无扰动（uniform(1,1) == 1）
    if frac == 0.0:
        env_section["edge_time_jitter_mode"] = "none"
    else:
        # 保留用户已设置的 mode（dual/full）；未设置则默认 full
        env_section.setdefault("edge_time_jitter_mode", "full")

    tmp_dir = Path(tempfile.gettempdir()) / "jitter_sweep_yamls"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_frac{frac:.4f}".replace(".", "p")
    tmp_path = tmp_dir / f"{src_yaml_path.stem}{suffix}.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    return tmp_path


def _summarize_episode_metrics(episode_metrics) -> dict:
    """从 test_trained_policy 返回的 episode_metrics 列表中提取 WI / AGI 均值方差。

    episode_metrics 元素含 .wi / .agi 属性（IdlenessMetrics dataclass）。
    std 采用总体标准差（ddof=0），与 evaluators/test.py 中 np.std 一致。
    """
    wi_vals = np.array([float(m.wi) for m in episode_metrics])
    agi_vals = np.array([float(m.agi) for m in episode_metrics])
    return {
        "WI_mean": float(np.mean(wi_vals)),
        "WI_std": float(np.std(wi_vals)),
        "AGI_mean": float(np.mean(agi_vals)),
        "AGI_std": float(np.std(agi_vals)),
        "n_episodes": len(episode_metrics),
    }


def _write_csv(rows: list, csv_path: Path) -> None:
    """将扫描结果写入 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["jitter_frac", "WI_mean", "WI_std", "AGI_mean", "AGI_std", "n_episodes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "jitter_frac": f"{r['jitter_frac']:.4f}",
                "WI_mean": f"{r['WI_mean']:.6f}",
                "WI_std": f"{r['WI_std']:.6f}",
                "AGI_mean": f"{r['AGI_mean']:.6f}",
                "AGI_std": f"{r['AGI_std']:.6f}",
                "n_episodes": r["n_episodes"],
            })
    print(f"\n[JitterSweep] 结果已保存至: {csv_path}")


def run_jitter_sweep(
    config_path: str,
    model_dir: str,
    fracs: list = None,
    num_episodes: int = None,
    episode_time: float = None,
    eval_seed: int = None,
) -> Path:
    """执行 jitter 扰动扫描。

    Args:
        config_path: 基础 eval YAML 路径（如 configs/eval/masup/masup_grid.yaml）。
        model_dir:   模型目录（含 config.yaml + policy.pt）。
        fracs:       扫描的扰动幅度列表，None 时使用 DEFAULT_FRACS。
        num_episodes: 单次评估的 episode 数；None 时从 yaml 的 eval 段读取。
        episode_time: 每个 episode 仿真时间上限；None 时从 yaml env.episode_len 读取。
        eval_seed:    评估种子；None 时保持 yaml 行为。

    Returns:
        输出 CSV 的路径。
    """
    if fracs is None:
        fracs = list(DEFAULT_FRACS)

    src_yaml = Path(config_path).resolve()
    if not src_yaml.exists():
        raise FileNotFoundError(f"Config YAML not found: {src_yaml}")

    print(f"[JitterSweep] base config : {src_yaml}")
    print(f"[JitterSweep] model dir   : {model_dir}")
    print(f"[JitterSweep] fracs       : {fracs}")
    print(f"[JitterSweep] num_episodes: {num_episodes} (None=use yaml)")
    print(f"[JitterSweep] episode_time: {episode_time} (None=use yaml)")
    print(f"[JitterSweep] eval_seed   : {eval_seed}")

    # 读 yaml 一次，确定 num_episodes 默认值（保持与 test.py CLI 行为一致）
    with open(src_yaml) as f:
        base_raw = yaml.safe_load(f) or {}
    base_eval = base_raw.get("eval", {}) or {}
    eff_num_episodes = num_episodes if num_episodes is not None else base_eval.get("num_episodes", 5)

    results = []
    for frac in fracs:
        tmp_yaml = _build_temp_yaml(src_yaml, float(frac))
        print(f"\n[JitterSweep] ===== frac={frac:.4f} | yaml={tmp_yaml.name} =====")

        # 关闭所有可视化产物，仅返回 episode_metrics 用于统计
        episode_metrics = test_trained_policy(
            model_dir=model_dir,
            env_config_path=str(tmp_yaml),
            num_episodes=eff_num_episodes,
            episode_time=episode_time,
            save_plot=None,
            show_plot=False,
            record_animation=False,
            event_driven=True,
            max_frames=None,
            eval_seed=eval_seed,
        )

        summary = _summarize_episode_metrics(episode_metrics)
        summary["jitter_frac"] = float(frac)
        results.append(summary)
        print(f"[JitterSweep] frac={frac:.4f} -> "
              f"WI={summary['WI_mean']:.4f}±{summary['WI_std']:.4f}, "
              f"AGI={summary['AGI_mean']:.4f}±{summary['AGI_std']:.4f} "
              f"(n={summary['n_episodes']})")

    csv_path = OUTPUT_DIR / f"{src_yaml.stem}.csv"
    _write_csv(results, csv_path)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Jitter 扰动扫描评估：遍历不同 edge_time_jitter_frac，汇总 WI/AGI 均值方差",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="基础评估 YAML 路径，如 configs/eval/masup/masup_grid.yaml")
    parser.add_argument("--model", type=str, required=True,
                        help="模型目录 (含 config.yaml + policy.pt)")
    parser.add_argument("--fracs", type=str, default=None,
                        help="逗号分隔的扰动幅度列表，如 0,0.05,0.1,0.15,0.2,0.25；"
                             "默认为 DEFAULT_FRACS")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="单次评估 episode 数（覆盖 yaml eval.num_episodes）")
    parser.add_argument("--episode_time", type=float, default=None,
                        help="每个 episode 仿真时间上限（覆盖 yaml env.episode_len）")
    parser.add_argument("--seed", type=int, default=None,
                        help="评估随机种子（所有 frac 共用，保证可复现）")

    args = parser.parse_args()

    fracs = None
    if args.fracs is not None:
        fracs = [float(x.strip()) for x in args.fracs.split(",") if x.strip() != ""]

    run_jitter_sweep(
        config_path=args.config,
        model_dir=args.model,
        fracs=fracs,
        num_episodes=args.num_episodes,
        episode_time=args.episode_time,
        eval_seed=args.seed,
    )


if __name__ == "__main__":
    main()
