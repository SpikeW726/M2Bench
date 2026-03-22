"""
WandB Sweep 超参数搜索脚本。

薄包装层：加载 base experiment YAML → 应用 wandb sweep 覆盖 → 调用 train.train()。

用法:
    # 创建 sweep（不运行）
    python sweep.py --base-config configs/experiments/mappo_tsp12_imi.yaml \\
                    --sweep-config configs/sweep/mappo_hparam.yaml --create-only

    # 加入已有 sweep 并行执行
    python sweep.py --base-config configs/experiments/mappo_tsp12_imi.yaml \\
                    --sweep-id abc12345 --count 10

    # 创建并立即运行（默认模式）
    python sweep.py --base-config configs/experiments/mappo_tsp12_imi.yaml \\
                    --sweep-config configs/sweep/mappo_hparam.yaml --count 20
"""

import argparse
from dataclasses import fields
from datetime import datetime

import yaml
import wandb

from configs.registry import load_config
from configs.exp_configs import ExperimentConfig
from train import train


# =============================================================================
#                          Sweep 覆盖逻辑
# =============================================================================

def apply_sweep_overrides(config: ExperimentConfig, sweep_cfg: dict):
    """
    将 wandb sweep 注入的扁平参数覆盖到嵌套 config 中。

    路由规则：
    1. key 在 algo dataclass 字段中 → 覆盖 config.algo
    2. key 在 training dataclass 字段中 → 覆盖 config.training
    3. key 在 ExperimentConfig 顶层字段中 → 覆盖顶层
    4. 其余忽略（wandb 内部 key 等）
    """
    algo_fields = {f.name for f in fields(type(config.algo))}
    training_fields = {f.name for f in fields(type(config.training))}
    top_fields = {f.name for f in fields(ExperimentConfig)}

    for key, value in sweep_cfg.items():
        if key in algo_fields:
            setattr(config.algo, key, value)
        elif key in training_fields:
            setattr(config.training, key, value)
        elif key in top_fields:
            setattr(config, key, value)
        elif key.startswith("custom_configs."):
            sub_key = key[len("custom_configs."):]
            if config.env.custom_configs is None:
                config.env.custom_configs = {}
            config.env.custom_configs[sub_key] = value


# =============================================================================
#                          Sweep 训练回调
# =============================================================================

# 全局变量：由 CLI 设置，供 sweep_train() 读取
_BASE_CONFIG_PATH: str = ""


def sweep_train():
    """
    wandb.agent() 回调函数。

    每次被调用时：
    1. wandb.init() 已由 wandb.agent 完成
    2. 从 base YAML 加载完整 config
    3. 用 wandb.config 中的 sweep 参数覆盖
    4. 设置有意义的 run name（算法_地图_时间戳_runID）
    5. 调用 train.train()
    6. 训练结束后将 save_dir 写入 wandb summary，供 eval_best_runs 查询
    """
    try:
        wandb.init()
        sweep_cfg = dict(wandb.config)

        config = load_config(_BASE_CONFIG_PATH)
        apply_sweep_overrides(config, sweep_cfg)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        config._timestamp = now
        config.exp_name = f"sweep-{wandb.run.id}"

        # 设置 wandb run name: algo-graph-timestamp-shortID
        wandb.run.name = f"{config.algo_name}-{config.env_type}-{config.graph_name}-{now}"

        config.track_wandb = True

        print(f"[Sweep] Run {wandb.run.name} with overrides:")
        for key, value in sweep_cfg.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")

        train(config)

        # 训练完成后记录 save_dir，供 eval_best_runs 事后查询
        final_dir = str(config.save_dir / "final")
        wandb.run.summary["save_dir"] = final_dir
        print(f"[Sweep] Recorded save_dir={final_dir}")

    except Exception as e:
        print(f"[Sweep] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
#                          Sweep 批量评估
# =============================================================================

def eval_best_runs(sweep_id: str, project: str, eval_config_path: str, top_n: int = 5):
    """用 wandb API 找出最优 N 个 trial，批量评估。

    每个 trial 的图像和视频保存到各自独立的路径，在 eval yaml 指定的路径基础上加序号后缀，
    避免互相覆盖：
        save_plot:  evaluators/results/auto_eval.png
          → trial 1: evaluators/results/auto_eval_1.png
          → trial 2: evaluators/results/auto_eval_2.png
        animation:  evaluators/results/auto_eval_1/，evaluators/results/auto_eval_2/，...

    Args:
        sweep_id: WandB sweep ID。
        project: WandB project 名称。
        eval_config_path: 评估 YAML 路径。
        top_n: 取最优 N 个 trial。
    """
    from pathlib import Path as _Path
    import yaml as _yaml

    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    metric_name = sweep.config.get("metric", {}).get("name", "env/wi")
    goal = sweep.config.get("metric", {}).get("goal", "minimize")

    runs = [r for r in sweep.runs if r.state == "finished"]
    if not runs:
        print(f"[Eval] No finished runs found in sweep {sweep_id}")
        return

    reverse = (goal == "maximize")
    runs_sorted = sorted(
        runs,
        key=lambda r: r.summary.get(metric_name, float("inf") if not reverse else float("-inf")),
        reverse=reverse,
    )
    best = runs_sorted[:top_n]

    # 读取 eval yaml 中的 save_plot 作为路径基准
    with open(eval_config_path) as f:
        _raw = _yaml.safe_load(f)
    base_save_plot = (_raw.get("eval") or {}).get("save_plot", None)
    record_animation = (_raw.get("eval") or {}).get("animation", False)

    print(f"\n[Eval] Evaluating top-{top_n} runs by {metric_name} ({goal})")
    from evaluators.test import run_eval_from_config
    for i, run in enumerate(best):
        model_dir = run.summary.get("save_dir")
        if not model_dir:
            print(f"  [Skip] Run {run.id}: no save_dir in summary")
            continue
        metric_val = run.summary.get(metric_name, "?")
        rank = i + 1
        print(f"\n  [{rank}/{top_n}] Run {run.id} | {metric_name}={metric_val} | {model_dir}")

        # 为当前 trial 生成独立的保存路径
        extra = {}
        if base_save_plot:
            p = _Path(base_save_plot)
            # e.g. auto_eval.png → auto_eval_1.png
            extra["save_plot"] = str(p.with_stem(f"{p.stem}_{rank}"))
        if record_animation:
            # 动画目录独立到与图像同目录下的 run_{rank}/ 子目录
            base_dir = str(_Path(base_save_plot).parent) if base_save_plot else "evaluators/results"
            extra["save_animation_dir"] = str(_Path(base_dir) / f"run_{rank}")

        try:
            run_eval_from_config(model_dir, eval_config_path, extra_params=extra)
        except Exception as e:
            print(f"  [Error] Run {run.id} eval failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
#                          Sweep 创建
# =============================================================================

def _resolve_project(args) -> str:
    """从 base config YAML 推导 sweep project 名称。

    优先级: CLI --project 显式指定 > 从 base config 自动推导。
    自动推导格式: "sweep-{algo_name}-{graph_name}"
    """
    if args.project:
        return args.project

    with open(args.base_config) as f:
        raw = yaml.safe_load(f)
    algo = raw.get("algo_name", "unknown")
    mdp = raw.get("env_type", "unknown")
    graph = raw.get("graph_name", "unknown")
    return f"sweep-{algo}-{mdp}-{graph}"


def create_sweep(args, project: str) -> str:
    """创建新的 WandB Sweep 并返回 sweep_id。"""
    if args.sweep_config:
        with open(args.sweep_config) as f:
            sweep_config = yaml.safe_load(f)
        print(f"[Main] Loaded sweep config from {args.sweep_config}")
    else:
        sweep_config = {
            "method": args.method,
            "metric": {"name": "env/wi", "goal": "minimize"},
            "parameters": {
                "actor_lr": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"},
                "critic_lr": {"min": 1e-4, "max": 1e-3, "distribution": "log_uniform_values"},
                "clip_range": {"min": 0.1, "max": 0.4},
                "vf_coef": {"min": 0.1, "max": 2.0},
                "ent_coef": {"min": 0.0, "max": 0.3},
                "gae_lambda": {"min": 0.9, "max": 1.0},
                "minibatch_size": {"values": [256, 512, 1024, 2048]},
                "update_epochs": {"values": [3, 5, 10]},
                "num_steps": {"values": [1024, 2048, 4096]},
                "gamma": {"value": 0.999},
                "total_steps": {"value": 50_000_000},
            },
        }

    sweep_id = wandb.sweep(sweep_config, project=project)
    return sweep_id


# =============================================================================
#                              CLI 入口
# =============================================================================

def main():
    """
    启动 WandB Sweep，支持三种模式:
      1) 仅创建 sweep:  --create-only
      2) 加入已有 sweep: --sweep-id <ID>
      3) 默认模式:       创建并立即运行

    并行用法:
      # 终端 1: 创建 sweep 并获取 ID
      python sweep.py --base-config configs/experiments/mappo_tsp12_imi.yaml \\
                      --sweep-config configs/sweep/mappo_hparam.yaml --create-only
      # => 输出 Sweep ID: abc12345

      # 终端 2/3/4: 各 tmux 窗口并行加入同一个 sweep
      python sweep.py --base-config configs/experiments/mappo_tsp12_imi.yaml \\
                      --sweep-id abc12345 --count 10
    """
    global _BASE_CONFIG_PATH

    parser = argparse.ArgumentParser(description="WandB Sweep 超参数搜索")

    # 必需参数
    parser.add_argument("--base-config", type=str, required=True,
                        help="Base experiment YAML 配置文件路径")

    # Sweep 运行模式
    parser.add_argument("--create-only", action="store_true",
                        help="仅创建 sweep 并打印 ID，不运行 agent")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="加入已有的 sweep（跳过创建），支持并行多终端")

    # 通用参数
    parser.add_argument("--project", type=str, default=None,
                        help="WandB project name (默认自动推导: sweep-{algo}-{graph})")
    parser.add_argument("--count", type=int, default=50,
                        help="本 agent 运行的 trial 数量")
    parser.add_argument("--method", type=str, default="bayes",
                        choices=["bayes", "random", "grid"],
                        help="Sweep method (仅创建时生效)")
    parser.add_argument("--sweep-config", type=str, default=None,
                        help="Sweep 配置 YAML 文件路径 (仅创建时生效)")
    parser.add_argument("--eval-config", type=str, default=None,
                        help="评估 YAML 路径；提供则 sweep 结束后评估最优 N 个 trial")
    parser.add_argument("--top-n", type=int, default=5,
                        help="sweep 结束后评估最优的 N 个 trial（默认 5）")

    args = parser.parse_args()
    _BASE_CONFIG_PATH = args.base_config
    project = _resolve_project(args)

    if args.sweep_id:
        # 模式 2: 加入已有 sweep
        sweep_id = args.sweep_id
        print(f"[Main] Joining existing sweep: {sweep_id}")
        print(f"[Main] Project: {project}")
        print(f"[Main] Will run {args.count} trials in this agent")
        wandb.agent(sweep_id, sweep_train, project=project, count=args.count)
        if args.eval_config:
            eval_best_runs(sweep_id, project, args.eval_config, top_n=args.top_n)

    elif args.create_only:
        # 模式 1: 仅创建 sweep
        sweep_id = create_sweep(args, project)
        print(f"\n{'=' * 60}")
        print(f"  Sweep created successfully!")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project:  {project}")
        print(f"{'=' * 60}")
        print(f"\n在各 tmux 终端运行以下命令来并行执行:")
        print(f"  python sweep.py --base-config {args.base_config} "
              f"--sweep-id {sweep_id} --count <N>")
        print()

    else:
        # 模式 3: 创建并立即运行（默认）
        sweep_id = create_sweep(args, project)
        print(f"[Main] Created sweep: {sweep_id} in project '{project}', "
              f"starting agent with {args.count} trials")
        wandb.agent(sweep_id, sweep_train, project=project, count=args.count)
        if args.eval_config:
            eval_best_runs(sweep_id, project, args.eval_config, top_n=args.top_n)


if __name__ == "__main__":
    main()
