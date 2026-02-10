"""WandB Sweep 超参数搜索脚本 - MAPPO 训练"""

import argparse
import time
from datetime import datetime
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb

from envs.mdps.masup_env import MASUPEnv
from envs.venvs import DummyVectorEnv, SubprocVectorEnv
from networks.mlp import ActorMLP, CriticMLP
from policies.rl.rl_base import ActorPolicy
from policies.marl.marl_base import MultiAgentPolicy
from algorithms.marl.mappo import MAPPOAlgo
from data.collector import MACollector
from utils.model_io import save_model


def train():
    """
    WandB Sweep 训练函数
    由 wandb.agent() 调用，config 由 sweep 配置注入
    """
    try:
        # 固定配置
        algo_name = "mappo-sweep"
        graph_name = "TSP12"
        now = datetime.now()
        run_name = f"{graph_name}_sweep_{now:%Y-%m-%d_%H-%M-%S}"

        # 显式初始化 wandb run，使用与本地目录一致的名称
        wandb.init(name=run_name)
        config = wandb.config

        num_envs = 12
        use_subproc = True
        actor_hidden = [256, 256]
        critic_hidden = [256, 256]

        # 预训练权重
        actor_path = config.get("actor_path", "")
        critic_path = config.get("critic_path", "")

        # 保存路径
        save_dir = Path(f"models/{algo_name}-{graph_name}/{run_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        log_dir = Path(f"runs/{run_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)

        print(f"[Sweep] Starting run with config:")
        for key, value in config.items():
            if key not in ["actor_path", "critic_path"]:
                print(f"  {key}: {value}")

        # ========== 环境 ==========
        with open("configs/MASUPEnv.yaml", 'r') as f:
            env_config_dict = yaml.safe_load(f)

        def make_env(env_config, custom_config):
            return lambda: MASUPEnv(env_config, **custom_config)

        env_fns = [make_env(env_config_dict["env_config"], env_config_dict["custom_config"])
                   for _ in range(num_envs)]

        if use_subproc:
            vec_env = SubprocVectorEnv(env_fns)
            print(f"[Sweep] Using SubprocVectorEnv (parallel, {num_envs} processes)")
        else:
            vec_env = DummyVectorEnv(env_fns)
            print(f"[Sweep] Using DummyVectorEnv (sequential, single process)")

        agent_ids = vec_env.agents
        num_agents = len(agent_ids)
        obs_space = vec_env.observation_space[agent_ids[0]]
        action_space = vec_env.action_space[agent_ids[0]]

        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        temp_env = MASUPEnv(env_config_dict["env_config"], **env_config_dict["custom_config"])
        temp_env.reset()
        state_dim = len(temp_env.state())
        critic_state_dim = state_dim + num_agents
        temp_env.close()

        print(f"[Sweep] Created {num_envs} vectorized MASUPEnv")
        print(f"  Agents: {agent_ids}, Obs dim: {obs_dim}, Action dim: {action_dim}")
        print(f"  Critic input dim: {critic_state_dim}")

        # ========== 网络 ==========
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actor_net = ActorMLP(obs_dim, actor_hidden, action_dim)
        critic_net = CriticMLP(critic_state_dim, critic_hidden, 1)

        # 加载预训练权重
        value_norm_config = None
        if actor_path and critic_path and Path(actor_path).exists() and Path(critic_path).exists():
            actor_ckpt = torch.load(actor_path, map_location=device, weights_only=True)
            critic_ckpt = torch.load(critic_path, map_location=device, weights_only=True)
            actor_sd = actor_ckpt.get("actor_state_dict", actor_ckpt)
            critic_sd = critic_ckpt.get("critic_state_dict", critic_ckpt)
            actor_net.load_state_dict(actor_sd)
            critic_net.load_state_dict(critic_sd)
            print(f"[Sweep] Loaded pretrained weights from {actor_path}")

            config_dir = Path(actor_path).parent
            config_file = config_dir / 'config.yaml'
            if config_file.exists():
                with open(config_file) as f:
                    saved_config = yaml.full_load(f)
                if saved_config.get('value_normalization') is not None:
                    value_norm_config = saved_config['value_normalization']
                    ret_mean = float(value_norm_config.get('ret_mean', 0.0))
                    ret_std = float(value_norm_config.get('ret_std', 1.0))
                    value_norm_config['ret_mean'] = ret_mean
                    value_norm_config['ret_std'] = ret_std
                    print(f"[Sweep] Loaded value_norm config: mean={ret_mean:.4f}, std={ret_std:.4f}")
            else:
                print(f"[Sweep] No config.yaml found, value normalization disabled")
        else:
            print(f"[Sweep] Training from scratch (random initialization)")

        # ========== Policy 和 Algorithm ==========
        ma_policy = MultiAgentPolicy(
            agent_ids=agent_ids,
            obs_space=obs_space,
            action_space=action_space,
            policy_class=ActorPolicy,
            policy_kwargs={"actor": actor_net},
            shared=True,
        )

        algorithm = MAPPOAlgo(
            policy=ma_policy,
            critic=critic_net,
            num_envs=num_envs,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            num_minibatches=config.num_minibatches,
            update_epochs=config.update_epochs,
            clip_vloss=True,
            use_value_norm=value_norm_config is not None,
            value_norm_config=value_norm_config,
        )

        # Model Config
        actor_config = {
            'type': 'ActorMLP',
            'input_dim': obs_dim,
            'hidden_sizes': actor_hidden,
            'output_dim': action_dim,
        }
        critic_config = {
            'type': 'CriticMLP',
            'input_dim': critic_state_dim,
            'hidden_sizes': critic_hidden,
            'output_dim': 1,
        }

        def get_value_norm_config():
            if algorithm.use_value_norm and algorithm.ret_rms is not None:
                return {
                    'enabled': True,
                    'ret_mean': float(algorithm.ret_rms.mean.item()),
                    'ret_std': float(algorithm.ret_rms.std.item()),
                    'ret_count': float(algorithm.ret_rms.count.item()),
                }
            return {'enabled': False}

        # 训练配置（用于 checkpoint 元数据）
        training_config = {
            'algorithm': 'MAPPO',
            'num_envs': num_envs,
            'use_subproc': use_subproc,
            'total_timesteps': config.total_timesteps,
            'num_steps': config.num_steps,
            'num_minibatches': config.num_minibatches,
            'update_epochs': config.update_epochs,
            'actor_lr': config.actor_lr,
            'critic_lr': config.critic_lr,
            'gamma': config.gamma,
            'gae_lambda': config.gae_lambda,
            'clip_range': config.clip_range,
            'vf_coef': config.vf_coef,
            'ent_coef': config.ent_coef,
        }

        def build_extra_info(iteration: int):
            return {
                'iteration': iteration,
                'training': training_config,
                'value_normalization': get_value_norm_config(),
            }

        # ========== 训练循环 ==========
        collector = MACollector(algorithm, vec_env)
        collector.reset()

        num_steps = config.num_steps
        total_timesteps = config.total_timesteps
        save_interval = config.get("save_interval", 500)  # 单位: iteration

        step_per_epoch = num_envs * num_steps
        num_iterations = total_timesteps // step_per_epoch
        global_step = 0
        start_time = time.time()

        print(f"\n[Sweep] Starting MAPPO training")
        print(f"  Total timesteps: {total_timesteps}, Iterations: {num_iterations}")
        print(f"  Batch size: {step_per_epoch * num_agents}, Device: {device}")

        for iteration in range(1, num_iterations + 1):
            # Checkpoint
            if iteration % save_interval == 0:
                ckpt_dir = save_dir / f"iter_{iteration}"
                save_model(
                    save_dir=ckpt_dir,
                    policy=ma_policy,
                    critic=critic_net,
                    actor_config=actor_config,
                    critic_config=critic_config,
                    extra_info=build_extra_info(iteration),
                )

            # 采集数据
            t0 = time.time()
            algorithm.set_training_mode(False)
            result = collector.collect(n_steps=step_per_epoch)
            global_step += result.n_steps
            t_collect = time.time() - t0

            # 计算 GAE 并更新
            t0 = time.time()
            batch = algorithm.prepare_batch(result.batch)
            algorithm.set_training_mode(True)
            stats = algorithm.update(batch)
            collector.reset_buffer()
            t_update = time.time() - t0

            # 获取 episode 指标
            metrics_list = vec_env.call_env_method("get_episode_metrics")
            finished = [m for m in metrics_list if m is not None]
            if finished:
                env_metrics_igi = np.mean([m["igi"] for m in finished])
                env_metrics_agi = np.mean([m["agi"] for m in finished])
                env_metrics_iwi = np.mean([m["iwi"] for m in finished])
                env_metrics_wi = np.mean([m["wi"] for m in finished])
                env_metrics_wait_ratio = np.mean([m["wait_ratio"] for m in finished])
            else:
                cur_list = vec_env.call_env_method("get_current_metrics")
                m = cur_list[0]
                env_metrics_igi, env_metrics_agi, env_metrics_iwi, env_metrics_wi = m["igi"], m["agi"], m["iwi"], m["wi"]
                env_metrics_wait_ratio = m["wait_ratio"]

            # 记录日志
            sps = int(global_step / (time.time() - start_time))

            log_data = {
                "losses/policy_loss": stats.policy_loss,
                "losses/value_loss": stats.value_loss,
                "losses/entropy": stats.entropy,
                "losses/total_loss": stats.loss,
                "env/igi": env_metrics_igi,
                "env/agi": env_metrics_agi,
                "env/iwi": env_metrics_iwi,
                "env/wi": env_metrics_wi,
                "env/wait_ratio": env_metrics_wait_ratio,
                "charts/SPS": sps,
                "charts/global_step": global_step,
            }

            if stats.extra:
                log_data["losses/clipfrac"] = stats.extra.get("clipfrac", 0)
                log_data["losses/approx_kl"] = stats.extra.get("approx_kl", 0)

            if result.episode_rewards:
                log_data["charts/episode_reward"] = np.mean(result.episode_rewards)
                log_data["charts/episode_length"] = np.mean(result.episode_lengths)

            # TensorBoard
            for key, value in log_data.items():
                writer.add_scalar(key, value, global_step)

            # Wandb (不指定 step，自动从 0 递增)
            wandb.log(log_data)

            # 打印进度
            if iteration % 10 == 0 or iteration == 1:
                reward_str = f"{np.mean(result.episode_rewards):.2f}" if result.episode_rewards else "N/A"
                print(f"[Iter {iteration}/{num_iterations}] "
                      f"steps={global_step}, reward={reward_str}, "
                      f"pg_loss={stats.policy_loss:.4f}, v_loss={stats.value_loss:.4f}, "
                      f"iwi={env_metrics_iwi:.2f}, SPS={sps} "
                      f"(collect={t_collect:.1f}s update={t_update:.1f}s)")

        # Save final model
        final_dir = save_dir / "final"
        save_model(
            save_dir=final_dir,
            policy=ma_policy,
            critic=critic_net,
            actor_config=actor_config,
            critic_config=critic_config,
            extra_info=build_extra_info(num_iterations),
        )
        print(f"\n[Sweep] Saved final model to {final_dir}")

        writer.close()
        vec_env.close()

    except Exception as e:
        print(f"[Sweep] Error during training: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        raise


def create_sweep(args) -> str:
    """创建新的 Sweep 并返回 sweep_id"""
    if args.sweep_config:
        with open(args.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)
        print(f"[Main] Loaded sweep config from {args.sweep_config}")
    else:
        sweep_config = {
            "method": args.method,
            "metric": {
                "name": "env/wi",
                "goal": "minimize"
            },
            "parameters": {
                "actor_lr": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"},
                "critic_lr": {"min": 1e-4, "max": 1e-3, "distribution": "log_uniform_values"},
                "clip_range": {"min": 0.1, "max": 0.4},
                "vf_coef": {"min": 0.1, "max": 2.0},
                "ent_coef": {"min": 0.0, "max": 0.3},
                "gae_lambda": {"min": 0.9, "max": 1.0},
                "num_minibatches": {"values": [4, 8, 16, 32]},
                "update_epochs": {"values": [3, 5, 10]},
                "num_steps": {"values": [1024, 2048, 4096]},
                "gamma": {"min": 0.99, "max": 0.9999},
                "total_timesteps": {"value": 50000000},
                "actor_path": {"values": [""]},
                "critic_path": {"values": [""]},
                "save_interval": {"value": 500},
            }
        }

    # 命令行覆盖
    if args.actor_path or args.critic_path:
        sweep_config["parameters"]["actor_path"] = {"value": args.actor_path}
        sweep_config["parameters"]["critic_path"] = {"value": args.critic_path}

    if args.total_timesteps is not None:
        sweep_config["parameters"]["total_timesteps"] = {"value": args.total_timesteps}

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    return sweep_id


def main():
    """
    启动 WandB Sweep，支持三种模式:
      1) 仅创建 sweep:  --create-only
      2) 加入已有 sweep: --sweep-id <ID>
      3) 默认模式:       创建并立即运行

    并行用法:
      # 终端 1: 创建 sweep 并获取 ID
      python sweep_mappo.py --create-only --sweep-config configs/sweep_config.yaml
      # => 输出 Sweep ID: abc12345

      # 终端 2/3/4: 各 tmux 窗口并行加入同一个 sweep
      python sweep_mappo.py --sweep-id abc12345 --count 10
      python sweep_mappo.py --sweep-id abc12345 --count 10
      python sweep_mappo.py --sweep-id abc12345 --count 10
    """
    parser = argparse.ArgumentParser(description="WandB Sweep for MAPPO training")
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
    parser.add_argument("--actor-path", type=str, default="",
                        help="预训练 actor 路径 (覆盖 sweep config)")
    parser.add_argument("--critic-path", type=str, default="",
                        help="预训练 critic 路径 (覆盖 sweep config)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="覆盖 total timesteps")

    args = parser.parse_args()

    if args.sweep_id:
        # 模式 2: 加入已有 sweep
        sweep_id = args.sweep_id
        print(f"[Main] Joining existing sweep: {sweep_id}")
        print(f"[Main] Will run {args.count} trials in this agent")
        wandb.agent(sweep_id, train, project=args.project, count=args.count)

    elif args.create_only:
        # 模式 1: 仅创建 sweep
        sweep_id = create_sweep(args)
        print(f"\n{'='*60}")
        print(f"  Sweep created successfully!")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project:  {args.project}")
        print(f"{'='*60}")
        print(f"\n在各 tmux 终端运行以下命令来并行执行:")
        print(f"  python sweep_mappo.py --sweep-id {sweep_id} --count <N>")
        print()

    else:
        # 模式 3: 创建并立即运行（默认，向后兼容）
        sweep_id = create_sweep(args)
        print(f"[Main] Created sweep: {sweep_id}, starting agent with {args.count} trials")
        wandb.agent(sweep_id, train, count=args.count)


if __name__ == "__main__":
    main()
