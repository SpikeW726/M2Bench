"""注册表 + 工厂函数 + YAML 配置加载。

网络系统采用三路独立配置: actor / critic / q_network，
由 YAML 中的 actor_type / critic_type / q_type 分发字段驱动。
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch.nn as nn
import yaml

from configs.env_configs import EnvConfig
from configs.algo_configs import (
    AlgoParams, MAPPOParams, IPPOParams, VDPPOParams, D3QNParams, IQLParams,
    VDNParams, QMIXParams,
)
from configs.training_configs import (
    TrainerConfig, OnPolicyTrainerConfig, OffPolicyTrainerConfig,
)
from configs.network_configs import (
    NetworkConfig, MLPConfig, RNNConfig, QMLPConfig, QRNNConfig,
)
from configs.exp_configs import ExperimentConfig


# =============================================================================
#                              注册表
# =============================================================================

# ---- 算法 ----
ALGO_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ---- on-policy (actor-critic) ----
    "ppo": {
        "module": "algorithms.rl.ppo",
        "class_name": "PPOAlgo",
        "params_class": IPPOParams,
        "trainer_type": "on_policy",
        "policy_type": "actor",
    },
    "mappo": {
        "module": "algorithms.marl.mappo",
        "class_name": "MAPPOAlgo",
        "params_class": MAPPOParams,
        "trainer_type": "on_policy",
        "policy_type": "actor",
    },
    "ippo": {
        "module": "algorithms.marl.ippo",
        "class_name": "IPPOAlgo",
        "params_class": IPPOParams,
        "trainer_type": "on_policy",
        "policy_type": "actor",
    },
    "vdppo": {
        "module": "algorithms.marl.vdppo",
        "class_name": "VDPPOAlgo",
        "params_class": VDPPOParams,
        "trainer_type": "on_policy",
        "policy_type": "actor",
    },
    # ---- off-policy (value-based) ----
    "d3qn": {
        "module": "algorithms.rl.d3qn",
        "class_name": "D3QNAlgo",
        "params_class": D3QNParams,
        "trainer_type": "off_policy",
        "policy_type": "value",
    },
    "iql": {
        "module": "algorithms.marl.iql",
        "class_name": "IQLAlgo",
        "params_class": IQLParams,
        "trainer_type": "off_policy",
        "policy_type": "value",
    },
    "vdn": {
        "module": "algorithms.marl.vdn",
        "class_name": "VDNAlgo",
        "params_class": VDNParams,
        "trainer_type": "off_policy",
        "policy_type": "value",
    },
    "qmix": {
        "module": "algorithms.marl.qmix",
        "class_name": "QMIXAlgo",
        "params_class": QMIXParams,
        "trainer_type": "off_policy",
        "policy_type": "value",
    },
}

# ---- 环境 ----
ENV_REGISTRY: Dict[str, Dict[str, str]] = {
    "masup": {
        "module": "envs.mdps.masup",
        "class_name": "MASUPEnv",
    },
    "oucs": {
        "module": "envs.mdps.oucs",
        "class_name": "OUCSEnv",
    },
    "s4r1": {
        "module": "envs.mdps.s4r1",
        "class_name": "S4R1Env",
    },
}

# ---- 网络 (三路独立注册表) ----
ACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {"module": "networks.mlp", "class_name": "ActorMLP", "config_class": MLPConfig},
    "rnn": {"module": "networks.rnn", "class_name": "ActorRNN", "config_class": RNNConfig},
}

CRITIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {"module": "networks.mlp", "class_name": "CriticMLP", "config_class": MLPConfig},
    "rnn": {"module": "networks.rnn", "class_name": "CriticRNN", "config_class": RNNConfig},
}

Q_NETWORK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {"module": "networks.mlp", "class_name": "QMLP", "config_class": QMLPConfig},
    "rnn": {"module": "networks.rnn", "class_name": "QRNN", "config_class": QRNNConfig},
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


# =============================================================================
#                              辅助函数
# =============================================================================

def _import_class(module_path: str, class_name: str) -> Type:
    """按模块路径和类名动态导入"""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _env_config_to_dicts(env_config: EnvConfig) -> Tuple[dict, dict]:
    """将 EnvConfig 拆为 (env_config_dict, custom_config_dict)。"""
    d = asdict(env_config)
    custom = d.pop("custom_configs", None) or {}
    d = {k: v for k, v in d.items() if v is not None}
    return d, custom


def _filter_dataclass_kwargs(cls: Type, raw: dict) -> dict:
    """只保留 cls 的 dataclass 字段名，忽略 YAML 中的多余 key。"""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _parse_network_config(
    net_type: Optional[str],
    raw_section: dict,
    registry: Dict[str, Dict[str, Any]],
) -> Optional[NetworkConfig]:
    """根据 type 字符串和 YAML section 解析网络配置。"""
    if net_type is None:
        return None
    entry = registry[net_type]
    cfg_cls = entry["config_class"]
    return cfg_cls(**_filter_dataclass_kwargs(cfg_cls, raw_section))


# =============================================================================
#                          YAML 配置加载
# =============================================================================

def load_config(yaml_path: str | Path) -> ExperimentConfig:
    """
    从 YAML 文件加载 ExperimentConfig。

    YAML 分发字段: algo_name, env_type, actor_type, critic_type, q_type。
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # ---- 1. 读取分发字段 ----
    algo_name = raw.get("algo_name", "mappo")
    env_type = raw.get("env_type", "masup")
    actor_type = raw.get("actor_type", None)
    critic_type = raw.get("critic_type", None)
    q_type = raw.get("q_type", None)

    # ---- 2. 算法参数 ----
    params_cls = get_params_class(algo_name)
    algo_raw = raw.get("algo", {})
    algo_params = params_cls(**_filter_dataclass_kwargs(params_cls, algo_raw))

    # ---- 3. 训练器配置 ----
    trainer_type = get_trainer_type(algo_name)
    trainer_cfg_cls = TRAINER_CONFIG_REGISTRY[trainer_type]
    training_raw = raw.get("training", {})
    training_config = trainer_cfg_cls(**_filter_dataclass_kwargs(trainer_cfg_cls, training_raw))

    # ---- 4. 网络配置 (三路独立) ----
    actor_config = _parse_network_config(
        actor_type, raw.get("actor", {}), ACTOR_REGISTRY,
    )
    critic_config = _parse_network_config(
        critic_type, raw.get("critic", {}), CRITIC_REGISTRY,
    )
    q_config = _parse_network_config(
        q_type, raw.get("q_network", {}), Q_NETWORK_REGISTRY,
    )

    # ---- 5. 环境配置 ----
    env_raw = raw.get("env", {})
    env_config = EnvConfig(**_filter_dataclass_kwargs(EnvConfig, env_raw))

    # ---- 6. 顶层元信息 ----
    top_level_keys = {f.name for f in fields(ExperimentConfig)}
    sub_config_keys = {"env", "algo", "training", "actor", "critic", "q_network"}
    meta_kwargs = {
        k: v for k, v in raw.items()
        if k in top_level_keys and k not in sub_config_keys
    }

    return ExperimentConfig(
        env=env_config,
        algo=algo_params,
        training=training_config,
        actor=actor_config,
        critic=critic_config,
        q_network=q_config,
        **meta_kwargs,
    )


# =============================================================================
#                              网络工厂函数
# =============================================================================

def create_actor(
    actor_type: str,
    actor_config: NetworkConfig,
    obs_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> nn.Module:
    """创建 Actor 网络 (ActorMLP / ActorRNN)。"""
    entry = ACTOR_REGISTRY[actor_type]
    cls = _import_class(entry["module"], entry["class_name"])

    if isinstance(actor_config, RNNConfig):
        return cls(
            input_dim=obs_dim,
            hidden_size=actor_config.hidden_size,
            output_dim=action_dim,
            num_layers=actor_config.num_layers,
            rnn_type=actor_config.rnn_type,
            fc_hidden=actor_config.fc_hidden or None,
        ).to(device)
    else:
        return cls(obs_dim, actor_config.hidden, action_dim).to(device)


def create_critic(
    critic_type: str,
    critic_config: NetworkConfig,
    critic_input_dim: int,
    device: str = "cpu",
) -> nn.Module:
    """创建 Critic 网络 (CriticMLP / CriticRNN)。"""
    entry = CRITIC_REGISTRY[critic_type]
    cls = _import_class(entry["module"], entry["class_name"])

    if isinstance(critic_config, RNNConfig):
        return cls(
            input_dim=critic_input_dim,
            hidden_size=critic_config.hidden_size,
            output_dim=1,
            num_layers=critic_config.num_layers,
            rnn_type=critic_config.rnn_type,
            fc_hidden=critic_config.fc_hidden or None,
        ).to(device)
    else:
        return cls(critic_input_dim, critic_config.hidden).to(device)


def create_q_network(
    q_type: str,
    q_config: NetworkConfig,
    input_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> nn.Module:
    """创建 Q-network (QMLP / QRNN)。"""
    entry = Q_NETWORK_REGISTRY[q_type]
    cls = _import_class(entry["module"], entry["class_name"])
    dueling = getattr(q_config, "dueling", False)

    if isinstance(q_config, QRNNConfig):
        return cls(
            input_dim=input_dim,
            hidden_size=q_config.hidden_size,
            output_dim=action_dim,
            num_layers=q_config.num_layers,
            rnn_type=q_config.rnn_type,
            dueling=dueling,
            fc_hidden=q_config.fc_hidden or None,
        ).to(device)
    else:
        return cls(input_dim, q_config.hidden, action_dim, dueling=dueling).to(device)


# =============================================================================
#                              其他工厂函数
# =============================================================================

def create_vec_env(
    env_type: str,
    env_config: EnvConfig,
    num_envs: int,
    use_subproc: bool = True,
):
    """创建向量化环境。"""
    from envs.venvs import DummyVectorEnv, SubprocVectorEnv

    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])

    cfg_dict, custom_dict = _env_config_to_dicts(env_config)
    env_fns = [lambda: env_cls(cfg_dict, **custom_dict) for _ in range(num_envs)]

    if use_subproc:
        return SubprocVectorEnv(env_fns)
    return DummyVectorEnv(env_fns)


def create_algorithm(
    algo_name: str,
    policy,
    algo_params: AlgoParams,
    **context_kwargs,
):
    """
    创建算法实例。

    通过 inspect 自动过滤: 只传递算法 __init__ 签名中声明的参数，
    多余的 context_kwargs 会被安全忽略，缺失的非必需参数使用默认值。
    新增算法时无需修改此函数，只需在 __init__ 中显式声明所需参数即可。

    context_kwargs 常见 key: critic, num_envs, n_agents, state_dim,
        action_dim, total_iterations, optimizer_steps_per_iter,
        value_norm_config, q_network 等。
    """
    entry = ALGO_REGISTRY[algo_name]
    algo_cls = _import_class(entry["module"], entry["class_name"])

    # 基于 __init__ 签名过滤参数
    sig = inspect.signature(algo_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in context_kwargs.items() if k in valid_params}

    return algo_cls(policy=policy, params=algo_params, **filtered)


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
    """从 YAML 加载 EnvConfig（供评估脚本使用）。"""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    env_raw = raw.get("env", raw)
    return EnvConfig(**_filter_dataclass_kwargs(EnvConfig, env_raw))


def load_eval_config(yaml_path: str | Path) -> Tuple[str, EnvConfig]:
    """从 eval YAML 加载 env_type 和 EnvConfig。

    Returns:
        (env_type, env_config) — env_type 缺省 "masup" 以兼容旧 YAML。
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    env_type = raw.get("env_type", "masup")
    env_raw = raw.get("env", raw)
    env_config = EnvConfig(**_filter_dataclass_kwargs(EnvConfig, env_raw))
    return env_type, env_config


# =============================================================================
#                              查询函数
# =============================================================================

def get_trainer_type(algo_name: str) -> str:
    """查询算法对应的训练器类型 ("on_policy" | "off_policy")"""
    return ALGO_REGISTRY[algo_name]["trainer_type"]


def get_policy_type(algo_name: str) -> str:
    """查询算法对应的策略类型 ("actor" | "value")"""
    return ALGO_REGISTRY[algo_name].get("policy_type", "actor")


def get_params_class(algo_name: str) -> Type[AlgoParams]:
    """查询算法对应的 Params dataclass 类"""
    return ALGO_REGISTRY[algo_name]["params_class"]
