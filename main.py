"""MAPPO training: RL fine-tuning with pretrained Actor/Critic weights on MASUPEnv"""

import time
from datetime import datetime
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from envs.mdps.masup_env import MASUPEnv
from envs.venvs import DummyVectorEnv, SubprocVectorEnv
from networks.mlp import ActorMLP, CriticMLP
from polocies.rl.rl_base import ActorPolicy
from polocies.marl.marl_base import MultiAgentPolicy
from algorithms.marl.mappo import MAPPOAlgo
from data.collector import MACollector
from utils.model_io import save_model


def main():
    # ========== 配置 ==========
    # 环境
    num_envs = 12
    use_subproc = True  # True: 多进程并行 (SubprocVectorEnv), False: 单进程串行 (DummyVectorEnv)
    
    # 网络隐藏层（需与预训练时一致）
    actor_hidden = [256, 256]
    critic_hidden = [256, 256]
    
    # 训练超参数
    total_timesteps = 50000000
    num_steps = 2048  # 每个 env 每次采集的步数
    num_minibatches = 16
    update_epochs = 5
    actor_lr = 3e-5
    critic_lr = 3e-4
    gamma = 0.999
    gae_lambda = 1.0
    clip_range = 0.2
    vf_coef = 1.0
    ent_coef = 0.1
    save_internal = 1000 # 每过xx个iteration保存一次模型参数
    
    # 日志配置
    track_wandb = True  # 是否使用 wandb
    wandb_project = "MAP-RL"
    
    # 预训练权重路径
    # actor_path = "models/imi-pure-norm-fixed-init/imi_train__1770179612_final/policy.pt"
    # critic_path = "models/imi-pure-norm-fixed-init/imi_train__1770179612_final/critic.pt"
    actor_path = "models/imi-pure-norm-random-init/imi_train__1770181266_final/policy.pt"
    critic_path = "models/imi-pure-norm-random-init/imi_train__1770181266_final/critic.pt"

    # 保存路径
    algo_name = "mappo"
    exp_name = "imi-norm-random-init"
    now = datetime.now()
    run_name = f"{exp_name}_{now:%Y-%m-%d_%H-%M-%S}"
    save_dir = Path(f"models/{algo_name}/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 日志初始化 ==========
    log_dir = Path(f"runs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    if track_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "total_timesteps": total_timesteps,
                "num_envs": num_envs,
                "num_steps": num_steps,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "vf_coef": vf_coef,
                "ent_coef": ent_coef,
                "num_minibatches": num_minibatches,
                "update_epochs": update_epochs,
            },
            sync_tensorboard = True,
        )
    
    # ========== 环境 ==========
    with open("configs/MASUPEnv.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    def make_env(env_config, custom_config):
        return lambda: MASUPEnv(env_config, **custom_config)
    
    env_fns = [make_env(config["env_config"], config["custom_config"]) for _ in range(num_envs)]
    
    if use_subproc:
        vec_env = SubprocVectorEnv(env_fns)
        print(f"[Main] Using SubprocVectorEnv (parallel, {num_envs} processes)")
    else:
        vec_env = DummyVectorEnv(env_fns)
        print(f"[Main] Using DummyVectorEnv (sequential, single process)")
    
    # 从环境获取维度
    agent_ids = vec_env.agents
    num_agents = len(agent_ids)
    obs_space = vec_env.observation_space[agent_ids[0]]
    action_space = vec_env.action_space[agent_ids[0]]
    
    obs_dim = obs_space.shape[0]
    action_dim = action_space.n
    
    # global_state 维度
    temp_env = MASUPEnv(config["env_config"], **config["custom_config"])
    temp_env.reset()
    state_dim = len(temp_env.state())
    critic_state_dim = state_dim + num_agents
    temp_env.close()
    
    print(f"[Main] Created {num_envs} vectorized MASUPEnv")
    print(f"  Agents: {agent_ids}, Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Critic input dim: {critic_state_dim}")
    
    # ========== 创建网络 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor_net = ActorMLP(obs_dim, actor_hidden, action_dim)
    critic_net = CriticMLP(critic_state_dim, critic_hidden, 1)
    
    # 加载预训练权重（可选）
    value_norm_config = None  # 用于存储预训练时的归一化配置
    if actor_path and critic_path and Path(actor_path).exists() and Path(critic_path).exists():
        actor_ckpt = torch.load(actor_path, map_location=device, weights_only=True)
        critic_ckpt = torch.load(critic_path, map_location=device, weights_only=True)
        actor_sd = actor_ckpt.get("actor_state_dict", actor_ckpt)
        critic_sd = critic_ckpt.get("critic_state_dict", critic_ckpt)
        actor_net.load_state_dict(actor_sd)
        critic_net.load_state_dict(critic_sd)
        print(f"[Main] Loaded pretrained weights from {actor_path}")
        
        # 尝试读取 value_normalization 配置（从预训练 checkpoint 的 config.yaml）
        config_dir = Path(actor_path).parent
        config_file = config_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file) as f:
                saved_config = yaml.safe_load(f)
            if saved_config.get('value_normalization') is not None:
                value_norm_config = saved_config['value_normalization']
                print(f"[Main] Loaded value_norm config: mean={value_norm_config.get('ret_mean', 0.0):.4f}, "
                      f"std={value_norm_config.get('ret_std', 1.0):.4f}")
            else:
                print(f"[Main] No value_normalization config found, using raw values")
        else:
            print(f"[Main] No config.yaml found at {config_dir}, value normalization disabled")
    else:
        print(f"[Main] Training from scratch (random initialization)")
    
    # ========== 构建 Policy 和 Algorithm ==========
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
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        clip_vloss=True,
        # Value Normalization (从预训练 checkpoint 继承)
        use_value_norm=value_norm_config is not None,
        value_norm_config=value_norm_config,
    )
    
    # ========== Model Config (for saving) ==========
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
    training_config = {
        'algorithm': 'MAPPO',
        # 环境配置
        'num_envs': num_envs,
        'use_subproc': use_subproc,
        # 训练超参数
        'total_timesteps': total_timesteps,
        'num_steps': num_steps,
        'num_minibatches': num_minibatches,
        'update_epochs': update_epochs,
        'save_interval': save_internal,
        # 优化器参数
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        # PPO 参数
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'vf_coef': vf_coef,
        'ent_coef': ent_coef,
        # 学习率调度器 (如果配置了)
        # 'lr_scheduler': {...},  # 预留字段，如需启用可添加
    }
    
    def get_value_norm_config():
        """获取当前 value normalization 配置（从 algorithm 中提取）"""
        if algorithm.use_value_norm and algorithm.ret_rms is not None:
            return {
                'enabled': True,
                'ret_mean': float(algorithm.ret_rms.mean.item()),
                'ret_std': float(algorithm.ret_rms.std.item()),
                'ret_count': float(algorithm.ret_rms.count.item()),
            }
        return {'enabled': False}
    
    def build_extra_info(iteration: int):
        """构建保存时的 extra_info，包含 iteration、training_config 和 value_normalization"""
        return {
            'iteration': iteration,
            'training': training_config,
            'value_normalization': get_value_norm_config(),
        }

    # ========== 训练循环 ==========
    collector = MACollector(algorithm, vec_env)
    collector.reset()
    
    step_per_epoch = num_envs * num_steps
    num_iterations = total_timesteps // step_per_epoch
    global_step = 0
    start_time = time.time()
    
    print(f"\n[Main] Starting MAPPO training")
    print(f"  Total timesteps: {total_timesteps}, Iterations: {num_iterations}")
    print(f"  Batch size: {step_per_epoch * num_agents}, Device: {device}")
    
    for iteration in range(1, num_iterations + 1):
        # 0. Checkpoint (HuggingFace style)
        if (iteration + 1) % save_internal == 0:
            ckpt_dir = save_dir / f"iter_{iteration + 1}"
            save_model(
                save_dir=ckpt_dir,
                policy=ma_policy,
                critic=critic_net,
                actor_config=actor_config,
                critic_config=critic_config,
                extra_info=build_extra_info(iteration + 1),
            )

        # 1. 采集数据 (eval mode)
        algorithm.set_training_mode(False)
        result = collector.collect(n_steps=step_per_epoch)
        global_step += result.n_steps
        
        # 2. 计算 GAE 并更新
        batch = algorithm.prepare_batch(result.batch)
        algorithm.set_training_mode(True)
        stats = algorithm.update(batch)
        collector.reset_buffer()
        
        # 3. 从环境获取 idleness 指标（取第一个环境的当前指标）
        env_metrics = vec_env.get_env_attr("world")[0].current_metrics
        
        # 4. 记录日志
        sps = int(global_step / (time.time() - start_time))
        
        # 构建日志字典
        log_data = {
            "losses/policy_loss": stats.policy_loss,
            "losses/value_loss": stats.value_loss,
            "losses/entropy": stats.entropy,
            "losses/total_loss": stats.loss,
            "env/igi": env_metrics.igi,
            "env/agi": env_metrics.agi,
            "env/iwi": env_metrics.iwi,
            "env/wi": env_metrics.wi,
            "charts/SPS": sps,
            "charts/actor_lr": actor_lr,
            "charts/critic_lr": critic_lr,
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
        
        # Wandb
        if track_wandb:
            import wandb
            wandb.log(log_data, step=global_step)
        
        # 打印进度
        if iteration % 10 == 0 or iteration == 1:
            reward_str = f"{np.mean(result.episode_rewards):.2f}" if result.episode_rewards else "N/A"
            print(f"[Iter {iteration}/{num_iterations}] "
                  f"steps={global_step}, reward={reward_str}, "
                  f"pg_loss={stats.policy_loss:.4f}, v_loss={stats.value_loss:.4f}, "
                  f"iwi={env_metrics.iwi:.2f}, SPS={sps}")
    
    # ========== Save final model (HuggingFace style) ==========
    final_dir = save_dir / "final"
    save_model(
        save_dir=final_dir,
        policy=ma_policy,
        critic=critic_net,
        actor_config=actor_config,
        critic_config=critic_config,
        extra_info=build_extra_info(num_iterations),
    )
    print(f"\n[Main] Saved final model to {final_dir}")
    
    writer.close()
    if track_wandb:
        wandb.finish()
    vec_env.close()


if __name__ == "__main__":
    main()
