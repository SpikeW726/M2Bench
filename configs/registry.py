"""注册表 + 工厂函数 + YAML 配置加载。"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch.nn as nn
import yaml

from configs.env_configs import EnvConfig
from configs.algo_configs import AlgoParams, MAPPOParams, IPPOParams, VDPPOParams
from configs.training_configs import (
    TrainerConfig, OnPolicyTrainerConfig, OffPolicyTrainerConfig,
)
from configs.network_configs import NetworkConfig, MLPNetworkConfig
from configs.exp_configs import ExperimentConfig


# =============================================================================
#                              注册表
# =============================================================================

# ---- 算法 ----
ALGO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "module": "algorithms.rl.ppo",
        "class_name": "PPOAlgo",
        "params_class": IPPOParams,
        "trainer_type": "on_policy",
    },
    "mappo": {
        "module": "algorithms.marl.mappo",
        "class_name": "MAPPOAlgo",
        "params_class": MAPPOParams,
        "trainer_type": "on_policy",
    },
    "ippo": {
        "module": "algorithms.marl.ippo",
        "class_name": "IPPOAlgo",
        "params_class": IPPOParams,
        "trainer_type": "on_policy",
    },
    "vdppo": {
        "module": "algorithms.marl.vdppo",
        "class_name": "VDPPOAlgo",
        "params_class": VDPPOParams,
        "trainer_type": "on_policy",
    },
}

# ---- 环境 ----
ENV_REGISTRY: Dict[str, Dict[str, str]] = {
    "masup": {
        "module": "envs.mdps.masup",
        "class_name": "MASUPEnv",
    },
}

# ---- 网络 ----
NETWORK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "module": "networks.mlp",
        "actor_class_name": "ActorMLP",
        "critic_class_name": "CriticMLP",
        "config_class": MLPNetworkConfig,
    },
}

# ---- 训练器 (运行时实例) ----
TRAINER_REGISTRY: Dict[str, Dict[str, str]] = {
    "on_policy": {
        "module": "trainers.rl_trainer",
        "class_name": "OnPolicyTrainer",
    },
    "off_policy": {
        "module": "trainers.rl_trainer",
        "class_name": "OffPolicyTrainer",
    },
}

# ---- 训练器配置 (dataclass) ----
TRAINER_CONFIG_REGISTRY: Dict[str, Type[TrainerConfig]] = {
    "on_policy": OnPolicyTrainerConfig,
    "off_policy": OffPolicyTrainerConfig,
}

# ---- 网络配置 (dataclass) ----
NETWORK_CONFIG_REGISTRY: Dict[str, Type[NetworkConfig]] = {
    "mlp": MLPNetworkConfig,
}


# =============================================================================
#                              辅助函数
# =============================================================================

def _import_class(module_path: str, class_name: str) -> Type:
    """按模块路径和类名动态导入"""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _env_config_to_dicts(env_config: EnvConfig) -> Tuple[dict, dict]:
    """将 EnvConfig 拆为 (env_config_dict, custom_config_dict)。

    值为 None 的字段会被移除，以确保下游代码的 dict.get(key, default)
    能正确回退到默认值。
    """
    d = asdict(env_config)
    custom = d.pop("custom_configs", None) or {}
    d = {k: v for k, v in d.items() if v is not None}
    return d, custom


def _filter_dataclass_kwargs(cls: Type, raw: dict) -> dict:
    """只保留 cls 的 dataclass 字段名，忽略 YAML 中的多余 key。"""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


# =============================================================================
#                          YAML 配置加载
# =============================================================================

def load_config(yaml_path: str | Path) -> "ExperimentConfig":
    """
    从 YAML 文件加载 ExperimentConfig。

    YAML 中的 algo_name / env_type / network_type 字符串会自动映射到
    对应的 Params / Config dataclass 类，用户无需接触任何 Python 类名。

    用法:
        config = load_config("configs/experiments/mappo_tsp12.yaml")
        train(config)
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # ---- 1. 读取分发字段 ----
    algo_name = raw.get("algo_name", "mappo")
    env_type = raw.get("env_type", "masup")
    network_type = raw.get("network_type", "mlp")

    # ---- 2. 根据分发字段选择对应的 dataclass 类并实例化 ----
    # 算法参数
    params_cls = get_params_class(algo_name)
    algo_raw = raw.get("algo", {})
    algo_params = params_cls(**_filter_dataclass_kwargs(params_cls, algo_raw))

    # 训练器配置
    trainer_type = get_trainer_type(algo_name)
    trainer_cfg_cls = TRAINER_CONFIG_REGISTRY[trainer_type]
    training_raw = raw.get("training", {})
    training_config = trainer_cfg_cls(**_filter_dataclass_kwargs(trainer_cfg_cls, training_raw))

    # 网络配置
    net_cfg_cls = NETWORK_CONFIG_REGISTRY[network_type]
    network_raw = raw.get("network", {})
    network_config = net_cfg_cls(**_filter_dataclass_kwargs(net_cfg_cls, network_raw))

    # 环境配置
    env_raw = raw.get("env", {})
    env_config = EnvConfig(**_filter_dataclass_kwargs(EnvConfig, env_raw))

    # ---- 3. 顶层元信息 ----
    top_level_keys = {f.name for f in fields(ExperimentConfig)}
    sub_config_keys = {"env", "algo", "training", "network"}
    meta_kwargs = {
        k: v for k, v in raw.items()
        if k in top_level_keys and k not in sub_config_keys
    }

    return ExperimentConfig(
        env=env_config,
        algo=algo_params,
        training=training_config,
        network=network_config,
        **meta_kwargs,
    )


# =============================================================================
#                              工厂函数
# =============================================================================

def create_vec_env(
    env_type: str,
    env_config: EnvConfig,
    num_envs: int,
    use_subproc: bool = True,
):
    """
    创建向量化环境。

    Returns:
        BaseVectorEnv 实例
    """
    from envs.venvs import DummyVectorEnv, SubprocVectorEnv

    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])

    cfg_dict, custom_dict = _env_config_to_dicts(env_config)
    env_fns = [lambda: env_cls(cfg_dict, **custom_dict) for _ in range(num_envs)]

    if use_subproc:
        return SubprocVectorEnv(env_fns)
    return DummyVectorEnv(env_fns)


def create_networks(
    network_type: str,
    network_config: NetworkConfig,
    obs_dim: int,
    action_dim: int,
    critic_input_dim: int,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Module]:
    """
    创建 Actor 和 Critic 网络。

    Returns:
        (actor, critic)
    """
    entry = NETWORK_REGISTRY[network_type]
    actor_cls = _import_class(entry["module"], entry["actor_class_name"])
    critic_cls = _import_class(entry["module"], entry["critic_class_name"])

    if isinstance(network_config, MLPNetworkConfig):
        actor = actor_cls(obs_dim, network_config.actor_hidden, action_dim).to(device)
        critic = critic_cls(critic_input_dim, network_config.critic_hidden).to(device)
    else:
        raise ValueError(f"Unsupported network config type: {type(network_config)}")

    return actor, critic


def create_algorithm(
    algo_name: str,
    policy,
    critic: nn.Module,
    algo_params: AlgoParams,
    **context_kwargs,
):
    """
    创建算法实例。

    context_kwargs 传递运行时上下文参数（如 num_envs, total_iterations 等）。
    """
    entry = ALGO_REGISTRY[algo_name]
    algo_cls = _import_class(entry["module"], entry["class_name"])
    return algo_cls(policy=policy, critic=critic, params=algo_params, **context_kwargs)


def create_trainer(
    algo_name: str,
    algorithm,
    collector,
    training_config: TrainerConfig,
    **callbacks,
):
    """
    创建训练器实例。

    根据算法注册的 trainer_type 自动选择 OnPolicyTrainer / OffPolicyTrainer。
    callbacks: save_checkpoint_fn, log_extra_fn, stop_fn, logger 等。
    """
    trainer_type = ALGO_REGISTRY[algo_name]["trainer_type"]
    entry = TRAINER_REGISTRY[trainer_type]
    trainer_cls = _import_class(entry["module"], entry["class_name"])
    return trainer_cls(
        algorithm=algorithm,
        collector=collector,
        config=training_config,
        **callbacks,
    )


# =============================================================================
#                          环境配置加载（评估用）
# =============================================================================

def load_env_config(yaml_path: str | Path) -> EnvConfig:
    """
    从 YAML 加载 EnvConfig（供评估脚本使用）。

    支持两种 YAML 格式：
    1. 完整 experiment YAML（含 env: 嵌套） → 提取 env 部分
    2. 独立 eval YAML（仅含 env: 嵌套）  → 同上

    用法:
        env_cfg = load_env_config("configs/experiments/mappo_tsp12_imi.yaml")
        env_cfg = load_env_config("configs/eval/masup_tsp12.yaml")
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    env_raw = raw.get("env", raw)  # 有 env: 则取之，否则整个 YAML 作为 env 字段
    return EnvConfig(**_filter_dataclass_kwargs(EnvConfig, env_raw))


# =============================================================================
#                              查询函数
# =============================================================================

def get_trainer_type(algo_name: str) -> str:
    """查询算法对应的训练器类型"""
    return ALGO_REGISTRY[algo_name]["trainer_type"]


def get_params_class(algo_name: str) -> Type[AlgoParams]:
    """查询算法对应的 Params dataclass 类"""
    return ALGO_REGISTRY[algo_name]["params_class"]
