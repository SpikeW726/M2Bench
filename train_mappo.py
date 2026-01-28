"""MAPPO training using OnPolicyTrainer with callbacks"""

from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from algorithms.marl.mappo import MAPPOAlgo
from data.collector import MACollector
from envs.mdps.masup_env import MASUPEnv
from envs.venvs import DummyVectorEnv
from polocies.marl.marl_base import MultiAgentPolicy
from polocies.rl.rl_base import ActorPolicy
from trainers.imitator.imitation_trainer import ActorMLP, CriticMLP
from trainers.rl_trainer import OnPolicyTrainer


class SimpleLogger:
    """Simple logger wrapper for TensorBoard + wandb"""
    
    def __init__(self, tb_writer: SummaryWriter, use_wandb: bool = False):
        self.tb_writer = tb_writer
        self.use_wandb = use_wandb
    
    def log(self, data: Dict[str, float], step: int):
        # TensorBoard
        for key, value in data.items():
            self.tb_writer.add_scalar(key, value, step)
        
        # wandb
        if self.use_wandb:
            import wandb
            wandb.log(data, step=step)


def main():
    # ========== Config ==========
    num_envs = 4
    actor_hidden = [256, 256]
    critic_hidden = [128, 256, 256, 128]

    # Training hyperparams
    total_timesteps = 50_000_000
    num_steps = 1024  # steps per env per collection
    num_minibatches = 8
    update_epochs = 10  # epochs inside algorithm.update()
    actor_lr = 3e-5
    critic_lr = 3e-5  # can use different lr for dual timescale update
    gamma = 0.999
    gae_lambda = 1.0
    clip_range = 0.2
    vf_coef = 1.0
    ent_coef = 0.1
    save_interval = 100

    # Logging
    exp_name = "mappo_patrol"
    track_wandb = True
    wandb_project = "MAP-RL"

    # Pretrained weights
    actor_path = "models/imi_train__1769515607_actor_best.pt"
    critic_path = "models/imi_train__1769515607_critic.pt"

    # Save dir
    save_dir = Path("models/mappo/imi")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========== Logger init ==========
    now = datetime.now()
    run_name = f"{exp_name}_{now:%Y-%m-%d}"
    log_dir = Path(f"runs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir)

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

    logger = SimpleLogger(tb_writer, use_wandb=track_wandb)

    # ========== Environment ==========
    with open("configs/MASUPEnv.yaml", "r") as f:
        config = yaml.safe_load(f)

    def make_env(env_config, custom_config):
        return lambda: MASUPEnv(env_config, **custom_config)

    env_fns = [make_env(config["env_config"], config["custom_config"]) for _ in range(num_envs)]
    vec_env = DummyVectorEnv(env_fns)

    # Get dimensions from env
    agent_ids = vec_env.agents
    num_agents = len(agent_ids)
    obs_space = vec_env.observation_space[agent_ids[0]]
    action_space = vec_env.action_space[agent_ids[0]]

    obs_dim = obs_space.shape[0]
    action_dim = action_space.n

    # Global state dim
    temp_env = MASUPEnv(config["env_config"], **config["custom_config"])
    temp_env.reset()
    state_dim = len(temp_env.state())
    critic_state_dim = state_dim + num_agents
    temp_env.close()

    print(f"[Main] Created {num_envs} vectorized MASUPEnv")
    print(f"  Agents: {agent_ids}, Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Critic input dim: {critic_state_dim}")

    # ========== Networks ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor_net = ActorMLP(obs_dim, actor_hidden, action_dim)
    critic_net = CriticMLP(critic_state_dim, critic_hidden, 1)

    # Load pretrained weights
    if actor_path and critic_path and Path(actor_path).exists() and Path(critic_path).exists():
        actor_ckpt = torch.load(actor_path, map_location=device, weights_only=True)
        critic_ckpt = torch.load(critic_path, map_location=device, weights_only=True)
        actor_sd = actor_ckpt.get("actor_state_dict", actor_ckpt)
        critic_sd = critic_ckpt.get("critic_state_dict", critic_ckpt)
        actor_net.load_state_dict(actor_sd)
        critic_net.load_state_dict(critic_sd)
        print(f"[Main] Loaded pretrained weights from {actor_path}")
    else:
        print("[Main] Training from scratch (random initialization)")

    # ========== Policy & Algorithm ==========
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

    # ========== Collector ==========
    collector = MACollector(algorithm, vec_env)

    # ========== Callbacks ==========
    def save_checkpoint_fn(iteration: int):
        """Save model checkpoint"""
        torch.save(ma_policy.state_dict(), save_dir / f"iter_{iteration}_policy.pt")
        torch.save(critic_net.state_dict(), save_dir / f"iter_{iteration}_critic.pt")
        print(f"[Checkpoint] Saved iteration {iteration}")

    def log_extra_fn() -> Dict[str, float]:
        """Get environment-specific metrics (idleness)"""
        try:
            worlds = vec_env.get_env_attr("world")
            if worlds and hasattr(worlds[0], "current_metrics"):
                m = worlds[0].current_metrics
                return {
                    "env/igi": m.igi,
                    "env/agi": m.agi,
                    "env/iwi": m.iwi,
                    "env/wi": m.wi,
                }
        except Exception:
            pass
        return {}

    # ========== Trainer ==========
    step_per_iteration = num_envs * num_steps
    max_iteration = total_timesteps // step_per_iteration

    trainer = OnPolicyTrainer(
        algorithm=algorithm,
        collector=collector,
        max_iteration=max_iteration,
        step_per_iteration=step_per_iteration,
        save_checkpoint_fn=save_checkpoint_fn,
        save_interval=save_interval,
        log_extra_fn=log_extra_fn,
        logger=logger,
        verbose=True,
    )

    print(f"\n[Main] Starting MAPPO training")
    print(f"  Total timesteps: {total_timesteps}, Iterations: {max_iteration}")
    print(f"  Batch size: {step_per_iteration * num_agents}, Device: {device}")

    # ========== Train ==========
    trainer.train()

    # ========== Cleanup ==========
    tb_writer.close()
    if track_wandb:
        import wandb
        wandb.finish()
    vec_env.close()


if __name__ == "__main__":
    main()
