"""
统一训练入口：从 YAML 配置文件启动训练。

用法:
    python train.py configs/experiments/mappo_tsp12_imi.yaml
    python train.py configs/experiments/mappo_tsp12_scratch.yaml
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from configs.exp_configs import ExperimentConfig
from configs.training_configs import OnPolicyTrainerConfig
from configs.registry import (
    load_config,
    create_vec_env,
    create_networks,
    create_algorithm,
    create_trainer,
)
from data.collector import MACollector
from policies.marl.marl_base import MultiAgentPolicy
from policies.rl.rl_base import ActorPolicy
from utils.model_io import save_model


# =============================================================================
#                              Logger
# =============================================================================

class SimpleLogger:
    """TensorBoard + wandb 日志包装"""

    def __init__(self, tb_writer: SummaryWriter, use_wandb: bool = False):
        self.tb_writer = tb_writer
        self.use_wandb = use_wandb

    def log(self, data: Dict[str, float], step: int):
        for key, value in data.items():
            self.tb_writer.add_scalar(key, value, step)
        if self.use_wandb:
            import wandb
            wandb.log(data, step=step)


# =============================================================================
#                          辅助函数
# =============================================================================

def _infer_dims(vec_env) -> dict:
    """从向量化环境推断网络所需的各维度。"""
    agent_ids = vec_env.agents
    num_agents = len(agent_ids)
    obs_space = vec_env.observation_space[agent_ids[0]]
    action_space = vec_env.action_space[agent_ids[0]]

    obs_dim = obs_space.shape[0]
    action_dim = action_space.n

    # global state 维度（用于 centralized critic）
    # 通过 call_env_method 获取，不需要额外创建临时环境
    vec_env.reset()
    states = vec_env.call_env_method("state")
    state_dim = len(states[0])
    critic_input_dim = state_dim + num_agents  # state + agent one-hot

    return {
        "agent_ids": agent_ids,
        "num_agents": num_agents,
        "obs_space": obs_space,
        "action_space": action_space,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "critic_input_dim": critic_input_dim,
    }


def _load_pretrained(
    actor_net,
    critic_net,
    actor_path: Optional[str],
    critic_path: Optional[str],
    device,
) -> Optional[dict]:
    """加载预训练权重，返回 value_norm_config（如果存在）。"""
    value_norm_config = None

    if not actor_path or not critic_path:
        print("[Train] Training from scratch (random initialization)")
        return None

    actor_p, critic_p = Path(actor_path), Path(critic_path)
    if not actor_p.exists() or not critic_p.exists():
        print("[Train] Pretrained paths not found, training from scratch")
        return None

    actor_ckpt = torch.load(actor_p, map_location=device, weights_only=True)
    critic_ckpt = torch.load(critic_p, map_location=device, weights_only=True)
    actor_net.load_state_dict(actor_ckpt.get("actor_state_dict", actor_ckpt))
    critic_net.load_state_dict(critic_ckpt.get("critic_state_dict", critic_ckpt))
    print(f"[Train] Loaded pretrained weights from {actor_path}")

    # 尝试从 checkpoint 目录读取 value_norm 统计量
    config_file = actor_p.parent / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            saved = yaml.safe_load(f)
        vn = saved.get("value_normalization")
        if vn is not None:
            value_norm_config = vn
            print(f"[Train] Loaded value_norm: "
                  f"mean={vn.get('ret_mean', 0.0):.4f}, "
                  f"std={vn.get('ret_std', 1.0):.4f}")

    return value_norm_config


# =============================================================================
#                              主训练函数
# =============================================================================

def train(config: ExperimentConfig):
    """
    根据 ExperimentConfig 执行完整训练流程。

    流程：环境 → 维度推断 → 网络 → 预训练加载 → Policy → 算法 → Collector → Trainer → 训练
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = config.training  # TrainerConfig shortcut

    # ---- 1. 日志初始化 ----
    config.save_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(config.log_dir)

    if config.track_wandb:
        import wandb
        if wandb.run is None:  # sweep 模式下由 wandb.agent 负责 init
            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=asdict(config),
                sync_tensorboard=True,
            )

    logger = SimpleLogger(tb_writer, use_wandb=config.track_wandb)

    # ---- 2. 创建向量化环境 ----
    vec_env = create_vec_env(
        env_type=config.env_type,
        env_config=config.env,
        num_envs=tc.num_envs,
        use_subproc=tc.use_subproc,
    )
    print(f"[Train] Created {tc.num_envs} vectorized {config.env_type} envs "
          f"({'subproc' if tc.use_subproc else 'dummy'})")

    # ---- 3. 推断维度 ----
    dims = _infer_dims(vec_env)
    print(f"  Agents: {dims['agent_ids']}, Obs: {dims['obs_dim']}, "
          f"Act: {dims['action_dim']}, CriticIn: {dims['critic_input_dim']}")

    # ---- 4. 创建网络 ----
    actor_net, critic_net = create_networks(
        network_type=config.network_type,
        network_config=config.network,
        obs_dim=dims["obs_dim"],
        action_dim=dims["action_dim"],
        critic_input_dim=dims["critic_input_dim"],
        device=str(device),
    )

    # ---- 5. 加载预训练权重 ----
    value_norm_config = _load_pretrained(
        actor_net, critic_net,
        config.actor_path, config.critic_path,
        device,
    )

    # ---- 6. 构建 MultiAgentPolicy ----
    ma_policy = MultiAgentPolicy(
        agent_ids=dims["agent_ids"],
        obs_space=dims["obs_space"],
        action_space=dims["action_space"],
        policy_class=ActorPolicy,
        policy_kwargs={"actor": actor_net},
        shared=True,
    )

    # ---- 7. 构建算法 ----
    # 准备运行时上下文参数
    context_kwargs = dict(num_envs=tc.num_envs)
    context_kwargs["action_dim"] = dims["action_dim"]
    context_kwargs["state_dim"] = dims["state_dim"]
    context_kwargs["n_agents"] = dims["num_agents"]

    if isinstance(tc, OnPolicyTrainerConfig):
        context_kwargs["total_iterations"] = tc.max_iterations
        context_kwargs["optimizer_steps_per_iter"] = tc.compute_optimizer_steps_per_iter(
            num_agents=dims["num_agents"]
        )

    # value_norm_config 从预训练 checkpoint 传递
    if value_norm_config is not None:
        context_kwargs["value_norm_config"] = value_norm_config

    algorithm = create_algorithm(
        algo_name=config.algo_name,
        policy=ma_policy,
        critic=critic_net,
        algo_params=config.algo,
        **context_kwargs,
    )

    # ---- 8. 构建 Collector ----
    collector = MACollector(algorithm, vec_env)

    # ---- 9. 定义 Callbacks ----
    # 网络配置（用于 checkpoint 保存）
    actor_config_dict = {
        "type": type(actor_net).__name__,
        "input_dim": dims["obs_dim"],
        "hidden_sizes": getattr(actor_net, "hidden_sizes", []),
        "output_dim": dims["action_dim"],
    }
    critic_config_dict = {
        "type": type(critic_net).__name__,
        "input_dim": dims["critic_input_dim"],
        "hidden_sizes": getattr(critic_net, "hidden_sizes", []),
        "output_dim": 1,
    }

    def _get_value_norm_config():
        if hasattr(algorithm, "use_value_norm") and algorithm.use_value_norm and algorithm.ret_rms is not None:
            return {
                "enabled": True,
                "ret_mean": float(algorithm.ret_rms.mean.item()),
                "ret_std": float(algorithm.ret_rms.std.item()),
                "ret_count": float(algorithm.ret_rms.count.item()),
            }
        return {"enabled": False}

    def save_checkpoint_fn(iteration: int):
        ckpt_dir = config.save_dir / f"iter_{iteration}"
        save_model(
            save_dir=ckpt_dir,
            policy=ma_policy,
            critic=critic_net,
            actor_config=actor_config_dict,
            critic_config=critic_config_dict,
            extra_info={
                "iteration": iteration,
                "value_normalization": _get_value_norm_config(),
            },
        )
        print(f"[Checkpoint] Saved iteration {iteration} to {ckpt_dir}")

    def log_extra_fn() -> Dict[str, float]:
        try:
            metrics_list = vec_env.call_env_method("get_episode_metrics")
            finished = [m for m in metrics_list if m is not None]
            if finished:
                return {
                    "env/igi": np.mean([m["igi"] for m in finished]),
                    "env/agi": np.mean([m["agi"] for m in finished]),
                    "env/iwi": np.mean([m["iwi"] for m in finished]),
                    "env/wi": np.mean([m["wi"] for m in finished]),
                }
        except Exception:
            pass
        return {}

    # ---- 10. 构建 Trainer ----
    trainer = create_trainer(
        algo_name=config.algo_name,
        algorithm=algorithm,
        collector=collector,
        training_config=tc,
        save_checkpoint_fn=save_checkpoint_fn,
        log_extra_fn=log_extra_fn,
        logger=logger,
    )

    # ---- 11. 训练 ----
    print(f"\n[Train] Starting {config.algo_name.upper()} training")
    print(f"  Max iterations: {tc.max_iterations}")
    if isinstance(tc, OnPolicyTrainerConfig):
        batch_size = tc.step_per_iteration * dims["num_agents"]
        print(f"  Batch size: {batch_size}, Minibatch: {tc.minibatch_size}, "
              f"Epochs: {tc.update_epochs}")
    print(f"  Device: {device}")
    print(f"  Save dir: {config.save_dir}")

    trainer.train()

    # ---- 12. 保存最终模型 & 清理 ----
    final_dir = config.save_dir / "final"
    save_model(
        save_dir=final_dir,
        policy=ma_policy,
        critic=critic_net,
        actor_config=actor_config_dict,
        critic_config=critic_config_dict,
        extra_info={
            "iteration": tc.max_iterations,
            "value_normalization": _get_value_norm_config(),
        },
    )
    print(f"[Train] Saved final model to {final_dir}")

    tb_writer.close()
    if config.track_wandb:
        import wandb
        if wandb.run is not None and wandb.run.sweep_id is None:
            # 非 sweep 模式才主动 finish，sweep 由 wandb.agent 管理生命周期
            wandb.finish()
    vec_env.close()


# =============================================================================
#                              入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAP 统一训练入口")
    parser.add_argument(
        "config",
        type=str,
        help="YAML 配置文件路径 (例: configs/experiments/mappo_tsp12_imi.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[Train] Loaded config from {args.config}")
    train(cfg)
