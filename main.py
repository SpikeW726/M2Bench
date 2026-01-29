"""MAPPO training: RL fine-tuning with pretrained Actor/Critic weights on MASUPEnv"""

import time
from datetime import datetime
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from envs.mdps.masup_env import MASUPEnv
from envs.venvs import DummyVectorEnv
from trainers.imitator.imitation_trainer import ActorMLP, CriticMLP
from polocies.rl.rl_base import ActorPolicy
from polocies.marl.marl_base import MultiAgentPolicy
from algorithms.marl.mappo import MAPPOAlgo
from data.collector import MACollector
from utils.model_io import save_model


def main():
    # ========== 配置 ==========
    # 环境
    num_envs = 4
    
    # 网络隐藏层（需与预训练时一致）
    actor_hidden = [256, 256]
    critic_hidden = [128, 256, 256, 128]
    
    # 训练超参数
    total_timesteps = 50000000
    num_steps = 1024  # 每个 env 每次采集的步数
    num_minibatches = 8
    update_epochs = 10
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
    actor_path = "models/imi_train__1769515607_actor_best.pt"
    critic_path = "models/imi_train__1769515607_critic.pt"
    # actor_path = ""
    # critic_path = ""

    # 保存路径
    algo_name = "mappo"
    exp_name = "imi"
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
            sync_tensorboard=True,
        )
    
    # ========== 环境 ==========
    with open("configs/MASUPEnv.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    def make_env(env_config, custom_config):
        return lambda: MASUPEnv(env_config, **custom_config)
    
    env_fns = [make_env(config["env_config"], config["custom_config"]) for _ in range(num_envs)]
    vec_env = DummyVectorEnv(env_fns)
    
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
    if actor_path and critic_path and Path(actor_path).exists() and Path(critic_path).exists():
        actor_ckpt = torch.load(actor_path, map_location=device, weights_only=True)
        critic_ckpt = torch.load(critic_path, map_location=device, weights_only=True)
        actor_sd = actor_ckpt.get("actor_state_dict", actor_ckpt)
        critic_sd = critic_ckpt.get("critic_state_dict", critic_ckpt)
        actor_net.load_state_dict(actor_sd)
        critic_net.load_state_dict(critic_sd)
        print(f"[Main] Loaded pretrained weights from {actor_path}")
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
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'vf_coef': vf_coef,
        'ent_coef': ent_coef,
        'num_minibatches': num_minibatches,
        'update_epochs': update_epochs,
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
                extra_info={'iteration': iteration + 1, 'training': training_config},
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
        extra_info={'iteration': num_iterations, 'training': training_config},
    )
    print(f"\n[Main] Saved final model to {final_dir}")
    
    writer.close()
    if track_wandb:
        wandb.finish()
    vec_env.close()


if __name__ == "__main__":
    main()
