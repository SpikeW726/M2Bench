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
    4. 调用 train.train()
    """
    try:
        wandb.init()
        sweep_cfg = dict(wandb.config)

        # 加载 base config
        config = load_config(_BASE_CONFIG_PATH)

        # 应用 sweep 覆盖
        apply_sweep_overrides(config, sweep_cfg)

        # 覆盖 run_name 以包含 sweep 标识
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config._timestamp = now
        config.exp_name = f"sweep-{wandb.run.id}"

        # 强制开启 wandb（sweep 模式下必须）
        config.track_wandb = True

        print(f"[Sweep] Run {wandb.run.id} with overrides:")
        for key, value in sweep_cfg.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")

        train(config)

    except Exception as e:
        print(f"[Sweep] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
#                          Sweep 创建
# =============================================================================

def create_sweep(args) -> str:
    """创建新的 WandB Sweep 并返回 sweep_id。"""
    if args.sweep_config:
        with open(args.sweep_config) as f:
            sweep_config = yaml.safe_load(f)
        print(f"[Main] Loaded sweep config from {args.sweep_config}")
    else:
        # 默认 sweep 配置
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
                "max_iterations": {"value": 500},
            },
        }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
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
    parser.add_argument("--project", type=str, default="MAP-Sweep",
                        help="WandB project name")
    parser.add_argument("--count", type=int, default=50,
                        help="本 agent 运行的 trial 数量")
    parser.add_argument("--method", type=str, default="bayes",
                        choices=["bayes", "random", "grid"],
                        help="Sweep method (仅创建时生效)")
    parser.add_argument("--sweep-config", type=str, default=None,
                        help="Sweep 配置 YAML 文件路径 (仅创建时生效)")

    args = parser.parse_args()
    _BASE_CONFIG_PATH = args.base_config

    if args.sweep_id:
        # 模式 2: 加入已有 sweep
        sweep_id = args.sweep_id
        print(f"[Main] Joining existing sweep: {sweep_id}")
        print(f"[Main] Will run {args.count} trials in this agent")
        wandb.agent(sweep_id, sweep_train, project=args.project, count=args.count)

    elif args.create_only:
        # 模式 1: 仅创建 sweep
        sweep_id = create_sweep(args)
        print(f"\n{'=' * 60}")
        print(f"  Sweep created successfully!")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project:  {args.project}")
        print(f"{'=' * 60}")
        print(f"\n在各 tmux 终端运行以下命令来并行执行:")
        print(f"  python sweep.py --base-config {args.base_config} "
              f"--sweep-id {sweep_id} --count <N>")
        print()

    else:
        # 模式 3: 创建并立即运行（默认）
        sweep_id = create_sweep(args)
        print(f"[Main] Created sweep: {sweep_id}, starting agent with {args.count} trials")
        wandb.agent(sweep_id, sweep_train, count=args.count)


if __name__ == "__main__":
    main()
