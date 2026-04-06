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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import h5py

from networks.mlp import ActorMLP, CriticMLP
from utils.model_io import _convert_to_native_types as _ensure_native_types
from configs.registry import (
    create_actor, create_q_network,
    ENV_REGISTRY, _import_class, _env_config_to_dicts, load_eval_config,
    get_policy_type,
)
from configs.network_configs import MLPConfig, MPNNConfig, RNNConfig, QMLPConfig, QRNNConfig


def _filter_dataclass_kwargs(cls, raw: dict) -> dict:
    """只保留 cls 的 dataclass 字段，忽略 YAML 中的多余 key"""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _infer_dims_from_env(env_type: str, env_config_path: str) -> dict:
    """创建临时环境，推断 obs_dim / action_dim / state_dim / num_agents。

    返回原始维度，调用方根据算法自行组合（如 critic_state_dim = state_dim + num_agents）。
    """
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

    env.close() if hasattr(env, "close") else None

    print(f"[ImiTrainer] 自动推断维度 (env_type={env_type}):")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, state_dim={state_dim}, num_agents={num_agents}")
    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "num_agents": num_agents,
    }


def _get_pretraining_strategy(algo_name: str, dims: dict) -> dict:
    """根据算法名称推断预训练策略。

    Returns:
        mode: "actor_critic" | "actor_only" | "q_net"
        value_input_key: HDF5 中 value 目标来源字段名
        value_input_dim: value 网络输入维度（q_net 模式下为 Q-net 输入维度）
        critic_state_dim: actor_critic 模式下 critic 输入维度（其他模式为 0）
    """
    policy_type = get_policy_type(algo_name)
    if policy_type == "actor":
        if algo_name == "vdppo":
            # VDPPO 只预训练 actor，Mixer 随机初始化时 TD target 为噪声，Q-net 预训练无意义
            mode = "actor_only"
            value_input_key = "critic_states"   # 占位，不会实际用到
            value_input_dim = 0
            critic_state_dim = 0
        elif algo_name == "mappo":
            mode = "actor_critic"
            value_input_key = "critic_states"   # MAPPO 集中式 critic 用 global state
            value_input_dim = dims["state_dim"] + dims["num_agents"]
            critic_state_dim = value_input_dim
        else:
            # ippo / ppo 等：独立 critic，用 obs 作为 value 输入
            mode = "actor_critic"
            value_input_key = "obs"
            value_input_dim = dims["obs_dim"]
            critic_state_dim = value_input_dim
    else:
        # iql / vdn / qmix 等 value-based 算法
        mode = "q_net"
        value_input_key = "obs"
        value_input_dim = dims["obs_dim"]
        critic_state_dim = 0

    return {
        "mode": mode,
        "value_input_key": value_input_key,
        "value_input_dim": value_input_dim,
        "critic_state_dim": critic_state_dim,
    }


def load_imitator_config(yaml_path: str | Path) -> dict:
    """从 YAML 加载模仿学习配置。

    返回 dict，包含：
        mode: "actor_critic" | "actor_only" | "q_net"
        actor: nn.Module | None
        critic: nn.Module | None
        q_network: nn.Module | None
        value_input_key: str
        train_kwargs: dict

    YAML 必填字段:
        algo_name: mappo | ippo | vdppo | iql | vdn | qmix
        actor_type: mlp | rnn | mpnn       # actor/vdppo 模式需要
        q_type: mlp | rnn                  # q_net 模式需要
        env_config: configs/eval/xxx.yaml  # 推荐：自动推断维度
        env_type: masup                    # 与 env_config 配套
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    algo_name = raw.get("algo_name", "mappo").lower()

    # 优先从环境自动推断维度
    env_config_path = raw.get("env_config")
    if env_config_path is not None:
        env_type = raw.get("env_type", "masup")
        dims = _infer_dims_from_env(env_type, env_config_path)
    else:
        obs_dim = int(raw["obs_dim"])
        action_dim = int(raw["action_dim"])
        state_dim = int(raw.get("state_dim", raw.get("critic_state_dim", obs_dim)))
        num_agents = int(raw.get("num_agents", 1))
        dims = {"obs_dim": obs_dim, "action_dim": action_dim,
                "state_dim": state_dim, "num_agents": num_agents}

    strategy = _get_pretraining_strategy(algo_name, dims)
    mode = strategy["mode"]
    value_input_key = strategy["value_input_key"]
    critic_state_dim = strategy["critic_state_dim"]

    obs_dim = dims["obs_dim"]
    action_dim = dims["action_dim"]

    actor_net = None
    critic_net = None
    q_net = None

    if mode in ("actor_critic", "actor_only"):
        actor_type = raw.get("actor_type", "mlp")
        if actor_type == "mlp":
            actor_raw = raw.get("actor", {})
            if not actor_raw and "actor_hidden_sizes" in raw:
                actor_raw = {"hidden": raw["actor_hidden_sizes"]}
            actor_raw = actor_raw or {"hidden": [512, 256, 128]}
            actor_config = MLPConfig(**_filter_dataclass_kwargs(MLPConfig, actor_raw))
        elif actor_type == "rnn":
            actor_raw = raw.get("actor", {})
            actor_config = RNNConfig(**_filter_dataclass_kwargs(RNNConfig, actor_raw))
        elif actor_type == "mpnn":
            actor_raw = raw.get("mpnn_actor", raw.get("actor", {}))
            actor_config = MPNNConfig(**_filter_dataclass_kwargs(MPNNConfig, actor_raw))
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")

        actor_net = create_actor(actor_type, actor_config, obs_dim, action_dim, device="cpu")

        if mode == "actor_critic":
            critic_hidden = raw.get("critic_hidden_sizes",
                                    raw.get("critic", {}).get("hidden", [512, 256, 128]))
            critic_net = CriticMLP(critic_state_dim, critic_hidden, 1)

    elif mode == "q_net":
        q_type = raw.get("q_type", "mlp")
        if q_type == "mlp":
            q_raw = raw.get("q_network", raw.get("q_net", {}))
            q_config = QMLPConfig(**_filter_dataclass_kwargs(QMLPConfig, q_raw))
        elif q_type == "rnn":
            q_raw = raw.get("q_network", raw.get("q_net", {}))
            q_config = QRNNConfig(**_filter_dataclass_kwargs(QRNNConfig, q_raw))
        else:
            raise ValueError(f"Unknown q_type: {q_type}")
        q_net = create_q_network(q_type, q_config, obs_dim, action_dim, device="cpu")

    # 训练相关参数
    train_kwargs = {
        "actor_lr": raw.get("actor_lr", 3e-4),
        "critic_lr": raw.get("critic_lr", 3e-4),
        "q_lr": raw.get("q_lr", raw.get("actor_lr", 3e-4)),
        "bc_lambda": raw.get("bc_lambda", 1.0),
        "value_lambda": raw.get("value_lambda", 0.5),
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

    return {
        "mode": mode,
        "actor": actor_net,
        "critic": critic_net,
        "q_network": q_net,
        "value_input_key": value_input_key,
        "train_kwargs": train_kwargs,
    }


class imi_trainer:
    def __init__(self, mode: str,
                 actor: nn.Module = None,
                 critic: nn.Module = None,
                 q_network: nn.Module = None,
                 value_input_key: str = "critic_states",
                 **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.value_input_key = value_input_key

        if mode in ("actor_critic", "actor_only"):
            assert actor is not None, "actor 不能为 None"
            self.actor = actor.to(self.device)
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=kwargs.get("actor_lr", 3e-4), eps=1e-5)

            # Actor 早停配置
            self.actor_patience = kwargs.get("actor_patience", 5)
            self.actor_min_delta = kwargs.get("actor_min_delta", 1e-5)
            self.actor_stopped = False
            self.best_actor_loss = float("inf")
            self.actor_no_improve_count = 0

            if mode == "actor_critic":
                assert critic is not None, "actor_critic 模式 critic 不能为 None"
                self.critic = critic.to(self.device)
                self.critic_optimizer = torch.optim.Adam(
                    self.critic.parameters(), lr=kwargs.get("critic_lr", 1e-3), eps=1e-5)

        elif mode == "q_net":
            assert q_network is not None, "q_network 不能为 None"
            self.q_network = q_network.to(self.device)
            self.q_optimizer = torch.optim.Adam(
                self.q_network.parameters(), lr=kwargs.get("q_lr", 3e-4), eps=1e-5)
            self.bc_lambda = kwargs.get("bc_lambda", 1.0)
            self.value_lambda = kwargs.get("value_lambda", 0.5)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Value Normalization（所有模式共用）
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

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def train(self, data_path: str, batch_size: int = 32, iteration: int = 10):
        """根据文件格式自动选择训练方法"""
        if data_path.endswith(".h5") or data_path.endswith(".hdf5"):
            self._train_from_hdf5(data_path, batch_size, iteration)
        else:
            self._train_from_npz(data_path, batch_size, iteration)

    # ------------------------------------------------------------------
    # NPZ 训练路径（全量加载）
    # ------------------------------------------------------------------
    def _train_from_npz(self, data_path: str, batch_size: int, iteration: int):
        """从 NPZ 文件训练（全量加载到内存）"""
        with np.load(data_path, mmap_mode="r") as data:
            raw_mask = data["padded_mask"]    # [N, T, 1]
            N, T, _ = raw_mask.shape
            M = data["obs"].shape[2]

            flat_mask = raw_mask[:, :, 0].reshape(-1).astype(bool)
            num_valid_steps = np.sum(flat_mask)
            print(f"[ImiTrainer] Total valid time steps (K): {num_valid_steps}")

            flattened_data = {}
            load_keys = {"obs", "actions", "action_masks", "returns"}
            if self.mode == "actor_critic":
                load_keys.add(self.value_input_key)
            elif self.mode == "q_net":
                load_keys.add(self.value_input_key)

            for key in data.files:
                if key == "padded_mask" or key not in load_keys:
                    continue
                arr = data[key].reshape(N * T, M, -1)
                valid_step_data = arr[flat_mask]
                flattened_data[key] = valid_step_data.reshape(-1, valid_step_data.shape[-1])

            total_samples = len(flattened_data["obs"])
            print(f"[ImiTrainer] Loaded {total_samples} valid samples (K*M={num_valid_steps}*{M})")

            returns_data = flattened_data["returns"]
            if self.use_value_norm:
                self.ret_mean = float(returns_data.mean())
                self.ret_std = float(returns_data.std())
                print(f"[ImiTrainer] Value Normalization: mean={self.ret_mean:.4f}, std={self.ret_std:.4f}")

            data_idx = np.arange(total_samples)
            global_step = 0
            start_time = time.time()

            for itr in range(iteration):
                np.random.shuffle(data_idx)
                itr_stats = {"actor_loss": 0.0, "value_loss": 0.0, "correct": 0, "total": 0}

                for start in range(0, len(data_idx), batch_size):
                    idx = data_idx[start:start + batch_size]
                    global_step += len(idx)
                    self._step(flattened_data, idx, itr_stats)

                self._log_and_check_early_stop(itr, iteration, itr_stats, global_step, start_time)

            if self.save_model:
                self._save_final(itr_stats, iteration)

            self.writer.close()

    # ------------------------------------------------------------------
    # HDF5 训练路径（按 episode 批量预读取）
    # ------------------------------------------------------------------
    def _train_from_hdf5(self, data_path: str, batch_size: int, iteration: int):
        """从 HDF5 文件训练（按 episode 批量预读取）"""
        episode_batch_size = min(500, 50000 // batch_size)

        with h5py.File(data_path, "r") as hf:
            N, T, M = hf["obs"].shape[:3]
            raw_mask = hf["padded_mask"][:]
            flat_mask = raw_mask[:, :, 0].astype(bool)

            valid_per_ep = flat_mask.sum(axis=1) * M
            total_samples = valid_per_ep.sum()
            print(f"[ImiTrainer] HDF5 mode: {total_samples} valid samples from {N} episodes")

            if self.use_value_norm:
                print("[ImiTrainer] Computing value normalization stats from HDF5...")
                sum_ret, sum_sq_ret, count = 0.0, 0.0, 0
                for n in range(N):
                    valid_t = np.where(flat_mask[n])[0]
                    if len(valid_t) == 0:
                        continue
                    ep_returns = hf["returns"][n, valid_t, :, 0].flatten()
                    sum_ret += ep_returns.sum()
                    sum_sq_ret += (ep_returns ** 2).sum()
                    count += len(ep_returns)
                self.ret_mean = sum_ret / count
                self.ret_std = np.sqrt(sum_sq_ret / count - self.ret_mean ** 2)
                print(f"[ImiTrainer] Value Normalization: mean={self.ret_mean:.4f}, std={self.ret_std:.4f}")

            global_step = 0
            start_time = time.time()

            for itr in range(iteration):
                ep_perm = np.random.permutation(N)
                itr_stats = {"actor_loss": 0.0, "value_loss": 0.0, "correct": 0, "total": 0}

                for ep_start in range(0, N, episode_batch_size):
                    ep_end = min(ep_start + episode_batch_size, N)
                    ep_indices = ep_perm[ep_start:ep_end]

                    batch_data = self._load_episodes_from_hdf5(hf, ep_indices, flat_mask)
                    if batch_data is None:
                        continue

                    n_samples = len(batch_data["obs"])
                    sample_perm = np.random.permutation(n_samples)

                    for start in range(0, n_samples, batch_size):
                        end = min(start + batch_size, n_samples)
                        idx = sample_perm[start:end]
                        global_step += len(idx)
                        self._step(batch_data, idx, itr_stats)

                    del batch_data

                self._log_and_check_early_stop(itr, iteration, itr_stats, global_step, start_time)

            if self.save_model:
                self._save_final(itr_stats, iteration)

            self.writer.close()

    # ------------------------------------------------------------------
    # 单 batch 更新（所有模式统一入口）
    # ------------------------------------------------------------------
    def _step(self, data: dict, idx: np.ndarray, stats: dict):
        obs_batch = torch.tensor(data["obs"][idx], dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(data["actions"][idx], dtype=torch.long, device=self.device)
        action_masks_batch = torch.tensor(data["action_masks"][idx], dtype=torch.long, device=self.device)
        returns_batch = torch.tensor(data["returns"][idx], dtype=torch.float32, device=self.device)

        normalized_returns = returns_batch
        if self.use_value_norm:
            normalized_returns = (returns_batch - self.ret_mean) / (self.ret_std + 1e-8)

        if self.mode in ("actor_critic", "actor_only"):
            self._step_actor(obs_batch, actions_batch, action_masks_batch, stats)

            if self.mode == "actor_critic":
                value_input = torch.tensor(
                    data[self.value_input_key][idx], dtype=torch.float32, device=self.device)
                self._step_critic(value_input, normalized_returns, stats)

        elif self.mode == "q_net":
            self._step_q_net(obs_batch, actions_batch, action_masks_batch,
                             normalized_returns, stats)

    def _step_actor(self, obs_batch, actions_batch, action_masks_batch, stats):
        """Actor BC 更新（含早停处理）"""
        if not self.actor_stopped:
            _out = self.actor(obs_batch)
            actor_logits = _out[0] if isinstance(_out, tuple) else _out
            actor_logits = actor_logits.masked_fill(~action_masks_batch.bool(), float("-inf"))
            actor_loss = F.cross_entropy(actor_logits, actions_batch.squeeze(-1))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        else:
            with torch.no_grad():
                _out = self.actor(obs_batch)
                actor_logits = _out[0] if isinstance(_out, tuple) else _out
                actor_logits = actor_logits.masked_fill(~action_masks_batch.bool(), float("-inf"))
                actor_loss = F.cross_entropy(actor_logits, actions_batch.squeeze(-1))

        n = len(obs_batch)
        stats["actor_loss"] += actor_loss.item() * n
        stats["total"] += n
        with torch.no_grad():
            _out2 = self.actor(obs_batch)
            logits2 = _out2[0] if isinstance(_out2, tuple) else _out2
            logits2 = logits2.masked_fill(~action_masks_batch.bool(), float("-inf"))
            pred = logits2.argmax(dim=-1)
        stats["correct"] += (pred == actions_batch.squeeze(-1)).sum().item()

    def _step_critic(self, value_input, normalized_returns, stats):
        """Critic MSE 更新"""
        critic_pred = self.critic(value_input)
        critic_loss = F.mse_loss(critic_pred, normalized_returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        stats["value_loss"] += critic_loss.item() * len(value_input)

    def _step_q_net(self, obs_batch, actions_batch, action_masks_batch,
                    normalized_returns, stats):
        """Q-network BC + Value MSE 联合更新。

        flat 训练：RNN 每步 hidden=None（自动 zero-init），不跨 timestep 传递 hidden state。
        BC loss 让专家动作的 Q 值排名最高；Value loss 让专家动作的 Q 值逼近 return。
        """
        _q_out = self.q_network(obs_batch)
        q_all = _q_out[0] if isinstance(_q_out, tuple) else _q_out   # [B, action_dim]

        # BC loss：对 Q logits 做 cross-entropy（等价于让专家动作的 Q 最大）
        bc_loss = F.cross_entropy(q_all, actions_batch.squeeze(-1))

        # Value loss：专家动作对应 Q 值逼近真实 return
        q_taken = q_all.gather(1, actions_batch).squeeze(-1)
        value_loss = F.mse_loss(q_taken, normalized_returns.squeeze(-1))

        total_loss = self.bc_lambda * bc_loss + self.value_lambda * value_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        n = len(obs_batch)
        stats["actor_loss"] += bc_loss.item() * n      # actor_loss 字段复用于 q_net bc_loss
        stats["value_loss"] += value_loss.item() * n
        stats["total"] += n
        with torch.no_grad():
            pred = q_all.argmax(dim=-1)
        stats["correct"] += (pred == actions_batch.squeeze(-1)).sum().item()

    # ------------------------------------------------------------------
    # 日志 & 早停
    # ------------------------------------------------------------------
    def _log_and_check_early_stop(self, itr: int, iteration: int,
                                  stats: dict, global_step: int, start_time: float):
        total = max(stats["total"], 1)
        avg_actor_loss = stats["actor_loss"] / total
        avg_value_loss = stats["value_loss"] / total
        accuracy = stats["correct"] / total
        sps = int(global_step / (time.time() - start_time + 1e-8))

        # 日志标签根据模式调整
        loss_tag = "q_bc_loss" if self.mode == "q_net" else "actor_loss"
        value_tag = "q_value_loss" if self.mode == "q_net" else "critic_loss"
        self.writer.add_scalar(f"losses/{loss_tag}", avg_actor_loss, itr)
        self.writer.add_scalar(f"losses/{value_tag}", avg_value_loss, itr)
        self.writer.add_scalar("losses/total_loss", avg_actor_loss + avg_value_loss, itr)
        self.writer.add_scalar("metrics/accuracy", accuracy, itr)
        self.writer.add_scalar("charts/SPS", sps, itr)

        actor_status = ""
        if self.mode in ("actor_critic", "actor_only"):
            actor_status = " [STOPPED]" if self.actor_stopped else ""

        print(f"[Iter {itr+1}/{iteration}] {loss_tag}: {avg_actor_loss:.4f}{actor_status}, "
              f"{value_tag}: {avg_value_loss:.4f}, acc: {accuracy:.4f}, SPS: {sps}")

        # 早停（仅 actor 模式）
        if self.mode in ("actor_critic", "actor_only") and not self.actor_stopped:
            if avg_actor_loss < self.best_actor_loss - self.actor_min_delta:
                self.best_actor_loss = avg_actor_loss
                self.actor_no_improve_count = 0
            else:
                self.actor_no_improve_count += 1
                if self.actor_no_improve_count >= self.actor_patience:
                    self.actor_stopped = True
                    print(f"\n[ImiTrainer] Actor early stopped at iter {itr+1}! "
                          f"Best actor_loss: {self.best_actor_loss:.6f}, acc: {accuracy:.4f}")
                    self._save_actor(avg_actor_loss, accuracy, itr + 1)

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_episodes_from_hdf5(self, hf: h5py.File, ep_indices: np.ndarray,
                                  flat_mask: np.ndarray) -> dict | None:
        """从 HDF5 加载指定 episodes 的有效数据，展平为训练样本。

        actor_critic 模式额外加载 value_input_key（critic_states 或 obs）。
        q_net 模式只需要 obs（value_input_key 也是 obs）。
        """
        obs_list, actions_list, masks_list, returns_list = [], [], [], []
        value_list = []

        need_value = self.mode in ("actor_critic", "q_net")
        value_key = self.value_input_key

        for n in ep_indices:
            valid_t = np.where(flat_mask[n])[0]
            if len(valid_t) == 0:
                continue
            obs_list.append(hf["obs"][n, valid_t, :, :].reshape(-1, hf["obs"].shape[-1]))
            actions_list.append(hf["actions"][n, valid_t, :, :].reshape(-1, 1))
            masks_list.append(
                hf["action_masks"][n, valid_t, :, :].reshape(-1, hf["action_masks"].shape[-1]))
            returns_list.append(hf["returns"][n, valid_t, :, :].reshape(-1, 1))
            if need_value:
                value_list.append(
                    hf[value_key][n, valid_t, :, :].reshape(-1, hf[value_key].shape[-1]))

        if not obs_list:
            return None

        result = {
            "obs": np.concatenate(obs_list, axis=0),
            "actions": np.concatenate(actions_list, axis=0),
            "action_masks": np.concatenate(masks_list, axis=0),
            "returns": np.concatenate(returns_list, axis=0),
        }
        if need_value:
            result[value_key] = np.concatenate(value_list, axis=0)
        return result

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def _save_actor(self, actor_loss: float, actor_acc: float, stopped_iter: int):
        """保存 best actor（HuggingFace style）"""
        actor_dir = self.save_dir / f"{self.run_name}_actor_best"
        actor_dir.mkdir(parents=True, exist_ok=True)

        actor_cfg = self.actor.get_config_dict(self.actor.input_dim, self.actor.output_dim)
        config = {
            "actor": actor_cfg,
            "extra": {
                "actor_loss": actor_loss,
                "actor_accuracy": actor_acc,
                "stopped_iteration": stopped_iter,
                "run_name": self.run_name,
            },
            "value_normalization": {
                "use_value_norm": self.use_value_norm,
                "ret_mean": float(self.ret_mean),
                "ret_std": float(self.ret_std),
            } if self.use_value_norm else None,
        }
        config = _ensure_native_types(config)
        with open(actor_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        torch.save(self.actor.state_dict(), actor_dir / "policy.pt")
        print(f"[ImiTrainer] Saved best actor to {actor_dir}")

    def _save_checkpoint(self, stats: dict, total_iterations: int):
        """保存最终 checkpoint（HuggingFace style）。

        actor_critic: 保存 actor (policy.pt) + critic (critic.pt)
        actor_only:   保存 actor (policy.pt)
        q_net:        保存 Q-network (policy.pt)，格式与 train.py _load_pretrained 兼容
        """
        ckpt_dir = self.save_dir / f"{self.run_name}_final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        total = max(stats["total"], 1)
        avg_loss = stats["actor_loss"] / total
        avg_value_loss = stats["value_loss"] / total
        accuracy = stats["correct"] / total

        if self.mode in ("actor_critic", "actor_only"):
            actor_cfg = self.actor.get_config_dict(self.actor.input_dim, self.actor.output_dim)
            config = {
                "actor": actor_cfg,
                "extra": {
                    "final_actor_loss": avg_loss,
                    "final_accuracy": accuracy,
                    "total_iterations": total_iterations,
                    "run_name": self.run_name,
                    "mode": self.mode,
                },
                "value_normalization": {
                    "use_value_norm": self.use_value_norm,
                    "ret_mean": float(self.ret_mean),
                    "ret_std": float(self.ret_std),
                } if self.use_value_norm else None,
            }
            if self.mode == "actor_critic":
                critic_cfg = self.critic.get_config_dict(
                    self.critic.input_dim, self.critic.output_dim)
                config["critic"] = critic_cfg
                config["extra"]["final_critic_loss"] = avg_value_loss
            config = _ensure_native_types(config)
            with open(ckpt_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            torch.save(self.actor.state_dict(), ckpt_dir / "policy.pt")
            if self.mode == "actor_critic":
                torch.save(self.critic.state_dict(), ckpt_dir / "critic.pt")

        elif self.mode == "q_net":
            q_cfg = self.q_network.get_config_dict(
                self.q_network.input_dim, self.q_network.output_dim)
            config = {
                "actor": q_cfg,   # 以 "actor" key 保存，保持与 train.py _load_pretrained 格式一致
                "extra": {
                    "final_bc_loss": avg_loss,
                    "final_value_loss": avg_value_loss,
                    "final_accuracy": accuracy,
                    "total_iterations": total_iterations,
                    "run_name": self.run_name,
                    "mode": self.mode,
                },
                "value_normalization": {
                    "use_value_norm": self.use_value_norm,
                    "ret_mean": float(self.ret_mean),
                    "ret_std": float(self.ret_std),
                } if self.use_value_norm else None,
            }
            config = _ensure_native_types(config)
            with open(ckpt_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            # 保存权重到 policy.pt，train.py 通过 actor_path 指向此文件加载到 q_net
            torch.save(self.q_network.state_dict(), ckpt_dir / "policy.pt")

        print(f"[ImiTrainer] Saved final checkpoint to {ckpt_dir}")

    def _save_final(self, stats: dict, iteration: int):
        """训练结束时的保存逻辑"""
        self._save_checkpoint(stats, iteration)
        if self.mode in ("actor_critic", "actor_only") and not self.actor_stopped:
            total = max(stats["total"], 1)
            self._save_actor(stats["actor_loss"] / total, stats["correct"] / total, iteration)


if __name__ == "__main__":
    os.chdir(_project_root)

    parser = argparse.ArgumentParser(
        description="Imitation learning trainer (支持 MAPPO/IPPO/VDPPO/IQL/VDN/QMIX)")
    parser.add_argument("--config", type=str, default="configs/imitator/masup_mlp_grid.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--data_path", type=str, default=None,
                        help="覆盖 config 中的 data_path")
    args = parser.parse_args()

    cfg = load_imitator_config(args.config)
    train_kwargs = cfg["train_kwargs"]
    if args.data_path is not None:
        train_kwargs["data_path"] = args.data_path

    trainer = imi_trainer(
        mode=cfg["mode"],
        actor=cfg["actor"],
        critic=cfg["critic"],
        q_network=cfg["q_network"],
        value_input_key=cfg["value_input_key"],
        **train_kwargs,
    )
    trainer.train(
        train_kwargs["data_path"],
        train_kwargs["batch_size"],
        train_kwargs["iteration"],
    )
