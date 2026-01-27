"""MAPPO 后训练：使用预训练的 Actor/Critic 权重在 MASUPEnv 上进行强化学习"""

import os
import yaml
import torch
from pathlib import Path

from envs.mdps.masup_env import MASUPEnv
from envs.venvs import DummyVectorEnv
from trainers.imitator.imitation_trainer import ActorMLP, CriticMLP
from polocies.rl.rl_base import ActorPolicy
from polocies.marl.marl_base import MultiAgentPolicy
from algorithms.marl.mappo import MAPPOAlgo
from data.collector import MACollector
from trainers.rl_trainer import OnPolicyTrainer


def main():
    # ========== 配置 ==========
    # 环境
    num_envs = 4
    num_agents = 3
    
    # 网络维度（需与预训练时一致）
    obs_dim = 27
    critic_state_dim = 26  # global_state(23) + agent_one_hot(3)
    action_dim = 7
    actor_hidden = [256, 256]
    critic_hidden = [128, 256, 256, 128]
    
    # 训练超参数
    total_timesteps = 500_000
    num_steps = 128  # 每个 env 每次采集的步数
    num_minibatches = 4
    update_epochs = 10
    lr = 3e-6
    gamma = 0.999  # 与采样时一致
    gae_lambda = 1.0
    clip_range = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    
    # 预训练权重路径（修改为实际路径）
    actor_path = "models/xxx_actor.pt"
    critic_path = "models/xxx_critic.pt"
    
    # ========== 环境 ==========
    with open("configs/MASUPEnv.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用 lambda 的默认参数避免闭包问题
    def make_env(env_config, custom_config):
        return lambda: MASUPEnv(env_config, **custom_config)
    
    env_fns = [make_env(config["env_config"], config["custom_config"]) for _ in range(num_envs)]
    vec_env = DummyVectorEnv(env_fns)
    
    print(f"[Main] Created {num_envs} vectorized MASUPEnv")
    print(f"  Agents: {vec_env.agents}")
    print(f"  Obs space: {vec_env.observation_space[vec_env.agents[0]]}")
    print(f"  Action space: {vec_env.action_space[vec_env.agents[0]]}")
    
    # ========== 加载预训练权重 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor_net = ActorMLP(obs_dim, actor_hidden, action_dim)
    critic_net = CriticMLP(critic_state_dim, critic_hidden, 1)
    
    if Path(actor_path).exists() and Path(critic_path).exists():
        actor_net.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
        critic_net.load_state_dict(torch.load(critic_path, map_location=device, weights_only=True))
        print(f"[Main] Loaded pretrained weights from {actor_path}, {critic_path}")
    else:
        print(f"[Main] Warning: Pretrained weights not found, using random initialization")
    
    # ========== 构建 Policy 和 Algorithm ==========
    agent_ids = vec_env.agents
    obs_space = vec_env.observation_space[agent_ids[0]]
    action_space = vec_env.action_space[agent_ids[0]]
    
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
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        clip_vloss=True,
    )
    
    # ========== 训练 ==========
    collector = MACollector(algorithm, vec_env)
    
    step_per_epoch = num_envs * num_steps
    num_iterations = total_timesteps // step_per_epoch
    
    trainer = OnPolicyTrainer(
        algorithm=algorithm,
        collector=collector,
        max_epoch=num_iterations,
        step_per_epoch=step_per_epoch,
    )
    
    batch_size = step_per_epoch * num_agents
    print(f"\n[Main] Starting MAPPO training")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Num iterations: {num_iterations}")
    print(f"  Step per epoch: {step_per_epoch}")
    print(f"  Batch size: {batch_size}")
    print(f"  Minibatch size: {batch_size // num_minibatches}")
    print(f"  Update epochs: {update_epochs}")
    print(f"  Device: {device}")
    
    trainer.train()
    
    # ========== 保存 ==========
    save_dir = Path("models/mappo")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(ma_policy.state_dict(), save_dir / "mappo_policy.pt")
    torch.save(critic_net.state_dict(), save_dir / "mappo_critic.pt")
    print(f"\n[Main] Saved trained models to {save_dir}")
    
    vec_env.close()


if __name__ == "__main__":
    main()
