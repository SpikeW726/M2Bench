import sys
import os
import time
import argparse
from pathlib import Path
from dataclasses import fields
# 添加项目根目录到 Python 路径 (支持从任意目录运行)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import h5py

from networks.mlp import ActorMLP, CriticMLP
from configs.registry import (
    create_actor, ENV_REGISTRY, _import_class, _env_config_to_dicts, load_eval_config
)
from configs.network_configs import MLPConfig, MPNNConfig


def _filter_dataclass_kwargs(cls, raw: dict) -> dict:
    """只保留 cls 的 dataclass 字段，忽略 YAML 中的多余 key"""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _infer_dims_from_env(env_type: str, env_config_path: str) -> dict:
    """创建临时环境，自动推断 obs_dim / action_dim / critic_state_dim。"""
    env_type_actual, env_cfg = load_eval_config(env_config_path)
    if env_type == "masup":
        env_type = env_type_actual
    env_config_dict, custom_config = _env_config_to_dicts(env_cfg)

    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])
    env = env_cls(env_config_dict, **custom_config)
    env.reset()

    agent = env.possible_agents[0]
    obs_dim = env.observation_space(agent).shape[0]
    action_dim = env.action_space(agent).n
    state_dim = len(env.state())
    num_agents = len(env.possible_agents)
    critic_state_dim = state_dim + num_agents

    env.close() if hasattr(env, "close") else None

    print(f"[ImiTrainer] 自动推断维度 (env_type={env_type}):")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, critic_state_dim={critic_state_dim}")
    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "critic_state_dim": critic_state_dim,
    }


def load_imitator_config(yaml_path: str | Path) -> tuple:
    """
    从 YAML 加载模仿学习配置，返回 (actor, critic, actor_type, train_kwargs)。

    当 YAML 中提供 env_config 时，obs_dim / action_dim / critic_state_dim 会从
    环境接口自动推断，YAML 中手动填写的值将被忽略。
    未提供 env_config 时退回读取 YAML 中的显式值。

    YAML 必填字段:
        actor_type: mlp | mpnn
        env_config: configs/eval/xxx.yaml   # 推荐：自动推断维度
        env_type: masup_gnn                 # 与 env_config 配套（默认读 env_config 中的 env_type）

    YAML 网络结构字段:
        actor: { hidden: [...] }            # mlp 时使用
        mpnn_actor: { graph_path, ... }     # mpnn 时使用
        critic_hidden_sizes: [...]
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    actor_type = raw.get("actor_type", "mlp")

    # 优先从环境自动推断维度，避免手动配置错误
    env_config_path = raw.get("env_config")
    if env_config_path is not None:
        env_type = raw.get("env_type", "masup")
        dims = _infer_dims_from_env(env_type, env_config_path)
        obs_dim = dims["obs_dim"]
        action_dim = dims["action_dim"]
        critic_state_dim = dims["critic_state_dim"]
    else:
        obs_dim = int(raw["obs_dim"])
        action_dim = int(raw["action_dim"])
        critic_state_dim = int(raw["critic_state_dim"])

    # 构建 Actor 配置并创建网络
    if actor_type == "mlp":
        actor_raw = raw.get("actor", {})
        if not actor_raw and "actor_hidden_sizes" in raw:
            actor_raw = {"hidden": raw["actor_hidden_sizes"]}
        actor_raw = actor_raw or {"hidden": [512, 256, 128]}
        actor_config = MLPConfig(**_filter_dataclass_kwargs(MLPConfig, actor_raw))
    elif actor_type == "mpnn":
        actor_raw = raw.get("mpnn_actor", raw.get("actor", {}))
        actor_config = MPNNConfig(**_filter_dataclass_kwargs(MPNNConfig, actor_raw))
    else:
        raise ValueError(f"Unknown actor_type: {actor_type}")

    actor = create_actor(actor_type, actor_config, obs_dim, action_dim, device="cpu")

    # 构建 Critic
    critic_hidden = raw.get("critic_hidden_sizes", raw.get("critic", {}).get("hidden", [512, 256, 128]))
    critic = CriticMLP(critic_state_dim, critic_hidden, 1)

    # 训练相关参数
    train_kwargs = {
        "actor_lr": raw.get("actor_lr", 3e-4),
        "critic_lr": raw.get("critic_lr", 3e-4),
        "data_path": raw.get("data_path", "dataset/samples.h5"),
        "batch_size": raw.get("batch_size", 1024),
        "iteration": raw.get("iteration", 100),
        "actor_patience": raw.get("actor_patience", 5),
        "actor_min_delta": raw.get("actor_min_delta", 1e-5),
        "use_value_norm": raw.get("use_value_norm", True),
        "exp_name": raw.get("exp_name", "imi_train"),
        "track": raw.get("track", False),
        "wandb_project": raw.get("wandb_project", "MAP-imitation"),
        "save_dir": raw.get("save_dir", "models"),
        "save_model": raw.get("save_model", True),
    }
    return actor, critic, actor_type, train_kwargs


class imi_trainer:
    def __init__(self, actor:nn.Module, critic:nn.Module, actor_type: str = "mlp", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_type = actor_type
        
        # 使用两个独立的优化器（Actor和Critic是两个独立的监督学习任务）
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=kwargs.get("actor_lr", 3e-4), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=kwargs.get("critic_lr", 1e-3), eps=1e-5)
        
        # Actor 早停配置
        self.actor_patience = kwargs.get("actor_patience", 5)  # 连续多少个epoch没有改善就早停
        self.actor_min_delta = kwargs.get("actor_min_delta", 1e-5)  # 最小改善量
        self.actor_stopped = False  # Actor 是否已经早停
        self.best_actor_loss = float('inf')
        self.actor_no_improve_count = 0

        # Value Normalization 配置
        self.use_value_norm = kwargs.get("use_value_norm", False)
        self.ret_mean = 0.0
        self.ret_std = 1.0

        # Logging 配置
        self.track = kwargs.get("track", False)
        self.exp_name = kwargs.get("exp_name", "imi_train")
        self.run_name = f"{self.exp_name}__{int(time.time())}"
        
        # 模型保存配置
        self.save_dir = Path(kwargs.get("save_dir", "models"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_model = kwargs.get("save_model", True)
        
        # 初始化 tensorboard
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text("hyperparameters", 
            "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in kwargs.items()]))
        
        # 初始化 wandb (可选)
        if self.track:
            import wandb
            wandb.init(
                project=kwargs.get("wandb_project", "MAP-imitation"),
                name=self.run_name,
                config=kwargs,
                sync_tensorboard=True,
            )

    def train(self, data_path: str, batch_size: int = 32, iteration: int = 10):
        """根据文件格式自动选择训练方法"""
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            self._train_from_hdf5(data_path, batch_size, iteration)
        else:
            self._train_from_npz(data_path, batch_size, iteration)
    
    def _train_from_npz(self, data_path: str, batch_size: int, iteration: int):
        """从 NPZ 文件训练（全量加载到内存）"""
        with np.load(data_path, mmap_mode='r') as data:
            raw_mask = data["padded_mask"]    # [N, T, 1]
            N, T, _ = raw_mask.shape
            M = data["obs"].shape[2]
            
            # 提取并flatten掩码: [N, T, 1] -> [N*T]
            flat_mask = raw_mask[:, :, 0].reshape(-1).astype(bool)    # [N*T]
            
            # 计算总有效步数 K
            num_valid_steps = np.sum(flat_mask)    # K
            print(f"[ImiTrainer] Total valid time steps (K): {num_valid_steps}")

            flattened_data = {}

            for key in data.files:
                
                if key == "padded_mask":
                    continue
                
                arr = data[key]    # [N, T, M, D]
                
                arr_view = arr.reshape(N * T, M, -1)    # [N*T, M, D]
                
                # 提取有效步的数据: [N*T, M, D] -> [K, M, D]
                valid_step_data = arr_view[flat_mask]    # [K, M, D]
                
                # Flatten智能体维度: [K, M, D] -> [K*M, D]
                final_flat = valid_step_data.reshape(-1, valid_step_data.shape[-1])    # [K*M, D]
                
                flattened_data[key] = final_flat

            # 打印验证
            total_samples = len(flattened_data["obs"])
            print(f"[ImiTrainer] Loaded {total_samples} valid samples (K*M={num_valid_steps}*{M})")
            for key, arr in flattened_data.items():
                print(f"  {key}: {arr.shape}")
            
            # 打印 returns 统计信息
            returns_data = flattened_data["returns"]
            print(f"[ImiTrainer] Returns range: [{returns_data.min():.4f}, {returns_data.max():.4f}], "
                  f"mean={returns_data.mean():.2f}, std={returns_data.std():.2f}")

            # 计算 Value Normalization 统计量
            if self.use_value_norm:
                self.ret_mean = float(returns_data.mean())
                self.ret_std = float(returns_data.std())
                print(f"[ImiTrainer] Value Normalization: mean={self.ret_mean:.4f}, std={self.ret_std:.4f}")
            
            # 训练逻辑
            data_idx = np.arange(num_valid_steps * M)
            global_step = 0
            start_time = time.time()

            for itr in range(iteration):
                np.random.shuffle(data_idx)
                
                # 每个iteration的累计统计
                itr_actor_loss, itr_critic_loss, itr_correct, itr_total = 0.0, 0.0, 0, 0

                for start in range(0, len(data_idx), batch_size):
                    end = start + batch_size
                    batch_idx = data_idx[start:end]
                    global_step += len(batch_idx)

                    # 将numpy数组转换为tensor并移到设备上
                    obs_batch = torch.from_numpy(flattened_data["obs"][batch_idx]).float().to(self.device)
                    actions_batch = torch.from_numpy(flattened_data["actions"][batch_idx]).long().to(self.device)
                    action_masks_batch = torch.from_numpy(flattened_data["action_masks"][batch_idx]).long().to(self.device)
                    returns_batch = torch.from_numpy(flattened_data["returns"][batch_idx]).float().to(self.device)
                    critic_states_batch = torch.from_numpy(flattened_data["critic_states"][batch_idx]).float().to(self.device)

                    # Actor 前向传播和更新（如果未早停）
                    if not self.actor_stopped:
                        actor_logits = self.actor(obs_batch)  # [batch, action_dim]
                        actor_logits = actor_logits.masked_fill(~action_masks_batch.bool(), float('-inf'))
                        actor_loss = nn.functional.cross_entropy(actor_logits, actions_batch.squeeze(-1))
                        
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()
                    else:
                        # Actor 已早停，只做推理用于统计
                        with torch.no_grad():
                            actor_logits = self.actor(obs_batch)
                            actor_logits = actor_logits.masked_fill(~action_masks_batch.bool(), float('-inf'))
                            actor_loss = nn.functional.cross_entropy(actor_logits, actions_batch.squeeze(-1))
                    
                    # Critic 前向传播和更新（始终更新）
                    critic_pred = self.critic(critic_states_batch)  # [batch, 1]

                    if self.use_value_norm:
                        target_norm = (returns_batch - self.ret_mean) / (self.ret_std + 1e-8)
                        critic_loss = nn.functional.mse_loss(critic_pred, target_norm)
                    else:
                        critic_loss = nn.functional.mse_loss(critic_pred, returns_batch)
                    
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    
                    # 累计统计
                    itr_actor_loss += actor_loss.item() * len(batch_idx)
                    itr_critic_loss += critic_loss.item() * len(batch_idx)
                    pred_actions = actor_logits.argmax(dim=-1)
                    itr_correct += (pred_actions == actions_batch.squeeze(-1)).sum().item()
                    itr_total += len(batch_idx)
                
                # 每个iteration结束后记录日志
                avg_actor_loss = itr_actor_loss / itr_total
                avg_critic_loss = itr_critic_loss / itr_total
                actor_acc = itr_correct / itr_total
                sps = int(global_step / (time.time() - start_time))
                
                self.writer.add_scalar("losses/actor_loss", avg_actor_loss, itr)
                self.writer.add_scalar("losses/critic_loss", avg_critic_loss, itr)
                self.writer.add_scalar("losses/total_loss", avg_actor_loss + avg_critic_loss, itr)
                self.writer.add_scalar("metrics/actor_accuracy", actor_acc, itr)
                self.writer.add_scalar("charts/SPS", sps, itr)
                
                # 构建日志信息
                actor_status = "[STOPPED]" if self.actor_stopped else ""
                print(f"[Iter {itr+1}/{iteration}] actor_loss: {avg_actor_loss:.4f} {actor_status}, "
                      f"critic_loss: {avg_critic_loss:.4f}, acc: {actor_acc:.4f}, SPS: {sps}")
                
                # Actor 早停检查（如果还没停止）
                if not self.actor_stopped:
                    if avg_actor_loss < self.best_actor_loss - self.actor_min_delta:
                        # 有改善，重置计数器
                        self.best_actor_loss = avg_actor_loss
                        self.actor_no_improve_count = 0
                    else:
                        # 没有改善
                        self.actor_no_improve_count += 1
                        if self.actor_no_improve_count >= self.actor_patience:
                            self.actor_stopped = True
                            print(f"\n[ImiTrainer] Actor early stopped at iter {itr+1}! "
                                  f"Best actor_loss: {self.best_actor_loss:.6f}, acc: {actor_acc:.4f}")
                            # 保存 Actor 模型
                            self._save_actor(avg_actor_loss, actor_acc, itr + 1)
            
            # 保存最终模型（主要保存 Critic，因为 Actor 可能已经早停保存过了）
            if self.save_model:
                self._save_checkpoint(avg_actor_loss, avg_critic_loss, actor_acc, iteration)
                # 如果 Actor 没有早停，也单独保存一次
                if not self.actor_stopped:
                    self._save_actor(avg_actor_loss, actor_acc, iteration)
            
            self.writer.close()
    
    def _train_from_hdf5(self, data_path: str, batch_size: int, iteration: int):
        """
        从 HDF5 文件训练（按 episode 批量预读取）
        
        策略：每次从 HDF5 读取一批 episodes 到内存，训练完后释放，
        这样既避免一次性加载全部数据，又比逐条读取高效。
        """
        # 预读取批次大小（episodes），可根据内存调整
        episode_batch_size = min(500, 50000 // batch_size)
        
        with h5py.File(data_path, 'r') as hf:
            N, T, M = hf['obs'].shape[:3]
            raw_mask = hf['padded_mask'][:]  # [N, T, 1]
            flat_mask = raw_mask[:, :, 0].astype(bool)  # [N, T]
            
            # 计算每个 episode 的有效样本数
            valid_per_ep = flat_mask.sum(axis=1) * M  # [N]
            total_samples = valid_per_ep.sum()
            print(f"[ImiTrainer] HDF5 mode: {total_samples} valid samples from {N} episodes")
            
            # 计算 Value Normalization 统计量
            if self.use_value_norm:
                print("[ImiTrainer] Computing value normalization stats from HDF5...")
                sum_ret, sum_sq_ret, count = 0.0, 0.0, 0
                for n in range(N):
                    valid_t = np.where(flat_mask[n])[0]
                    if len(valid_t) == 0:
                        continue
                    ep_returns = hf['returns'][n, valid_t, :, 0].flatten()
                    sum_ret += ep_returns.sum()
                    sum_sq_ret += (ep_returns ** 2).sum()
                    count += len(ep_returns)
                self.ret_mean = sum_ret / count
                self.ret_std = np.sqrt(sum_sq_ret / count - self.ret_mean ** 2)
                print(f"[ImiTrainer] Value Normalization: mean={self.ret_mean:.4f}, std={self.ret_std:.4f}")
            
            global_step = 0
            start_time = time.time()
            
            for itr in range(iteration):
                # Shuffle episode order
                ep_perm = np.random.permutation(N)
                itr_actor_loss, itr_critic_loss, itr_correct, itr_total = 0.0, 0.0, 0, 0
                
                # 按 episode 批次读取
                for ep_start in range(0, N, episode_batch_size):
                    ep_end = min(ep_start + episode_batch_size, N)
                    ep_indices = ep_perm[ep_start:ep_end]
                    
                    # 预读取这批 episodes 的所有数据
                    batch_data = self._load_episodes_from_hdf5(hf, ep_indices, flat_mask)
                    if batch_data is None:
                        continue
                    
                    # Shuffle within this batch
                    n_samples = len(batch_data['obs'])
                    sample_perm = np.random.permutation(n_samples)
                    
                    for start in range(0, n_samples, batch_size):
                        end = min(start + batch_size, n_samples)
                        idx = sample_perm[start:end]
                        global_step += len(idx)
                        
                        obs_batch = torch.tensor(batch_data['obs'][idx], dtype=torch.float32, device=self.device)
                        actions_batch = torch.tensor(batch_data['actions'][idx], dtype=torch.long, device=self.device)
                        action_masks_batch = torch.tensor(batch_data['action_masks'][idx], dtype=torch.long, device=self.device)
                        returns_batch = torch.tensor(batch_data['returns'][idx], dtype=torch.float32, device=self.device)
                        critic_states_batch = torch.tensor(batch_data['critic_states'][idx], dtype=torch.float32, device=self.device)
                        
                        # Actor 前向传播和更新
                        if not self.actor_stopped:
                            actor_logits = self.actor(obs_batch)
                            actor_logits = actor_logits.masked_fill(~action_masks_batch.bool(), float('-inf'))
                            actor_loss = nn.functional.cross_entropy(actor_logits, actions_batch.squeeze(-1))
                            
                            self.actor_optimizer.zero_grad()
                            actor_loss.backward()
                            self.actor_optimizer.step()
                        else:
                            with torch.no_grad():
                                actor_logits = self.actor(obs_batch)
                                actor_logits = actor_logits.masked_fill(~action_masks_batch.bool(), float('-inf'))
                                actor_loss = nn.functional.cross_entropy(actor_logits, actions_batch.squeeze(-1))
                        
                        # Critic 前向传播和更新
                        critic_pred = self.critic(critic_states_batch)
                        if self.use_value_norm:
                            target_norm = (returns_batch - self.ret_mean) / (self.ret_std + 1e-8)
                            critic_loss = nn.functional.mse_loss(critic_pred, target_norm)
                        else:
                            critic_loss = nn.functional.mse_loss(critic_pred, returns_batch)
                        
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
                        
                        # 累计统计
                        itr_actor_loss += actor_loss.item() * len(idx)
                        itr_critic_loss += critic_loss.item() * len(idx)
                        pred_actions = actor_logits.argmax(dim=-1)
                        itr_correct += (pred_actions == actions_batch.squeeze(-1)).sum().item()
                        itr_total += len(idx)
                    
                    # 释放内存
                    del batch_data
                
                # 每个 iteration 结束后记录
                avg_actor_loss = itr_actor_loss / itr_total
                avg_critic_loss = itr_critic_loss / itr_total
                actor_acc = itr_correct / itr_total
                sps = int(global_step / (time.time() - start_time))
                
                self.writer.add_scalar("losses/actor_loss", avg_actor_loss, itr)
                self.writer.add_scalar("losses/critic_loss", avg_critic_loss, itr)
                self.writer.add_scalar("losses/total_loss", avg_actor_loss + avg_critic_loss, itr)
                self.writer.add_scalar("metrics/actor_accuracy", actor_acc, itr)
                self.writer.add_scalar("charts/SPS", sps, itr)
                
                actor_status = "[STOPPED]" if self.actor_stopped else ""
                print(f"[Iter {itr+1}/{iteration}] actor_loss: {avg_actor_loss:.4f} {actor_status}, "
                      f"critic_loss: {avg_critic_loss:.4f}, acc: {actor_acc:.4f}, SPS: {sps}")
                
                # Actor 早停检查
                if not self.actor_stopped:
                    if avg_actor_loss < self.best_actor_loss - self.actor_min_delta:
                        self.best_actor_loss = avg_actor_loss
                        self.actor_no_improve_count = 0
                    else:
                        self.actor_no_improve_count += 1
                        if self.actor_no_improve_count >= self.actor_patience:
                            self.actor_stopped = True
                            print(f"\n[ImiTrainer] Actor early stopped at iter {itr+1}! "
                                  f"Best actor_loss: {self.best_actor_loss:.6f}, acc: {actor_acc:.4f}")
                            self._save_actor(avg_actor_loss, actor_acc, itr + 1)
            
            # 保存最终模型
            if self.save_model:
                self._save_checkpoint(avg_actor_loss, avg_critic_loss, actor_acc, iteration)
                if not self.actor_stopped:
                    self._save_actor(avg_actor_loss, actor_acc, iteration)
            
            self.writer.close()
    
    def _load_episodes_from_hdf5(self, hf: h5py.File, ep_indices: np.ndarray, 
                                  flat_mask: np.ndarray) -> dict:
        """从 HDF5 加载指定 episodes 的有效数据，展平为训练样本"""
        obs_list, actions_list, masks_list, returns_list, critic_list = [], [], [], [], []
        
        for n in ep_indices:
            valid_t = np.where(flat_mask[n])[0]
            if len(valid_t) == 0:
                continue
            # 读取该 episode 的有效时间步数据
            obs_list.append(hf['obs'][n, valid_t, :, :].reshape(-1, hf['obs'].shape[-1]))
            actions_list.append(hf['actions'][n, valid_t, :, :].reshape(-1, 1))
            masks_list.append(hf['action_masks'][n, valid_t, :, :].reshape(-1, hf['action_masks'].shape[-1]))
            returns_list.append(hf['returns'][n, valid_t, :, :].reshape(-1, 1))
            critic_list.append(hf['critic_states'][n, valid_t, :, :].reshape(-1, hf['critic_states'].shape[-1]))
        
        if not obs_list:
            return None
        
        return {
            'obs': np.concatenate(obs_list, axis=0),
            'actions': np.concatenate(actions_list, axis=0),
            'action_masks': np.concatenate(masks_list, axis=0),
            'returns': np.concatenate(returns_list, axis=0),
            'critic_states': np.concatenate(critic_list, axis=0),
        }
    
    def _save_actor(self, actor_loss: float, actor_acc: float, stopped_iter: int):
        """Save best actor model (HuggingFace style + legacy format)."""
        # HuggingFace style: directory with config.yaml + weights
        actor_dir = self.save_dir / f"{self.run_name}_actor_best"
        actor_dir.mkdir(parents=True, exist_ok=True)
        
        actor_cfg = self.actor.get_config_dict(self.actor.input_dim, self.actor.output_dim)

        config = {
            'actor': actor_cfg,
            'extra': {
                'actor_type': self.actor_type,
                'actor_loss': actor_loss,
                'actor_accuracy': actor_acc,
                'stopped_iteration': stopped_iter,
                'run_name': self.run_name,
            },
            # Value Normalization 统计量（转换为 Python 原生类型以兼容 yaml.safe_load）
            'value_normalization': {
                'use_value_norm': self.use_value_norm,
                'ret_mean': float(self.ret_mean),
                'ret_std': float(self.ret_std),
            } if self.use_value_norm else None,
        }
        with open(actor_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save weights
        torch.save(self.actor.state_dict(), actor_dir / 'policy.pt')
        print(f"[ImiTrainer] Saved best actor to {actor_dir}")
    
    def _save_checkpoint(self, final_actor_loss: float, final_critic_loss: float, 
                        final_acc: float, total_iterations: int):
        """Save final checkpoint (HuggingFace style + legacy format)."""
        # HuggingFace style: directory with config.yaml + weights
        ckpt_dir = self.save_dir / f"{self.run_name}_final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        actor_cfg = self.actor.get_config_dict(self.actor.input_dim, self.actor.output_dim)
        critic_cfg = self.critic.get_config_dict(self.critic.input_dim, self.critic.output_dim)

        config = {
            'actor': actor_cfg,
            'critic': critic_cfg,
            'extra': {
                'actor_type': self.actor_type,
                'final_actor_loss': final_actor_loss,
                'final_critic_loss': final_critic_loss,
                'final_accuracy': final_acc,
                'total_iterations': total_iterations,
                'run_name': self.run_name,
            },
            # Value Normalization 统计量（转换为 Python 原生类型以兼容 yaml.safe_load）
            'value_normalization': {
                'use_value_norm': self.use_value_norm,
                'ret_mean': float(self.ret_mean),
                'ret_std': float(self.ret_std),
            } if self.use_value_norm else None,
        }
        with open(ckpt_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save weights
        torch.save(self.actor.state_dict(), ckpt_dir / 'policy.pt')
        torch.save(self.critic.state_dict(), ckpt_dir / 'critic.pt')
        print(f"[ImiTrainer] Saved final checkpoint to {ckpt_dir}")

if __name__ == "__main__":
    os.chdir(_project_root)

    parser = argparse.ArgumentParser(description="Imitation learning trainer (Actor-Critic from heuristic data)")
    parser.add_argument("--config", type=str, default="configs/imitator/masup_gnn_tsp12.yaml",
                        help="YAML 配置文件路径，支持 mlp / mpnn 等网络类型")
    parser.add_argument("--data_path", type=str, default=None,
                        help="覆盖 config 中的 data_path")
    args = parser.parse_args()

    actor, critic, actor_type, train_kwargs = load_imitator_config(args.config)
    if args.data_path is not None:
        train_kwargs["data_path"] = args.data_path

    trainer = imi_trainer(actor, critic, actor_type=actor_type, **train_kwargs)
    trainer.train(
        train_kwargs["data_path"],
        train_kwargs["batch_size"],
        train_kwargs["iteration"],
    )
