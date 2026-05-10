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
    AlgoParams, A2CParams, MAA2CParams, MAPPOParams, IPPOParams, VDPPOParams,
    D3QNParams, IQLParams, VDNParams, QMIXParams, QTableParams, MAPPOMATParams,
)
from configs.training_configs import (
    TrainerConfig, OnPolicyTrainerConfig, OffPolicyTrainerConfig,
)
from configs.network_configs import (
    NetworkConfig, MLPConfig, RNNConfig, QMLPConfig, QRNNConfig, SUNConfig, MPNNConfig, SAGEConfig,
    MASUPBaseConfig,
    MASUPActorConfig, MASUPActorRNNConfig,
    MASUPCriticConfig, MASUPCriticRNNConfig,
    MASUPQConfig, MASUPQRNNConfig,
    MASUPVDPPOQConfig, MASUPVDPPOQRNNConfig,
)
from configs.exp_configs import ExperimentConfig


# =============================================================================
#                              注册表
# =============================================================================

# ---- 算法 ----
ALGO_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ---- on-policy (actor-critic): A2C 系列 ----
    "a2c": {
        "module": "algorithms.rl.a2c",
        "class_name": "A2CAlgo",
        "params_class": A2CParams,
        "trainer_type": "on_policy",
        "policy_type": "actor",
    },
    "maa2c": {
        "module": "algorithms.marl.maa2c",
        "class_name": "MAA2CAlgo",
        "params_class": MAA2CParams,
        "trainer_type": "on_policy",
        "policy_type": "actor",
    },
    # ---- on-policy (actor-critic): PPO 系列 ----
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
    # ---- tabular ----
    "qtable": {
        "params_class": QTableParams,
        "trainer_type": "tabular",
    },
    # ---- MAT（Multi-Agent Transformer）PPO（asy_ppo.py 路径）----
    "mappo_mat": {
        "module": "algorithms.marl.mappo_mat",
        "class_name": "MAPPOMATAlgo",
        "params_class": MAPPOMATParams,
        "trainer_type": "on_policy",
        "policy_type": "mat",   # 特殊路径：MATMultiAgentPolicy
    },
    # 论文中 HAPPO/顺序信用分配语境；实现与 mappo_mat 相同（MAT+联合 PPO）
    "happo": {
        "module": "algorithms.marl.mappo_mat",
        "class_name": "MAPPOMATAlgo",
        "params_class": MAPPOMATParams,
        "trainer_type": "on_policy",
        "policy_type": "mat",
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
    "bbla": {
        "module": "envs.mdps.bbla",
        "class_name": "BBLAEnv",
    },
    "gbla": {
        "module": "envs.mdps.gbla",
        "class_name": "GBLAEnv",
    },
    "ex_gbla": {
        "module": "envs.mdps.ex_gbla",
        "class_name": "ExGBLAEnv",
    },
    "suns": {
        "module": "envs.mdps.suns",
        "class_name": "SUNSEnv",
    },
    "suns_gym": {
        "module": "envs.mdps.suns_gym",
        "class_name": "SUNSGymEnv",
    },
    "nep": {
        "module": "envs.mdps.nep",
        "class_name": "NEPEnv",
    },
    "masup_gnn": {
        "module": "envs.mdps.masup_gnn",
        "class_name": "MASUPGraphEnv",
    },
    "magec": {
        "module": "envs.mdps.magec",
        "class_name": "MAGECEnv",
    },
    "beau": {
        "module": "envs.mdps.beau",
        "class_name": "BEAU",
    },
}

# ---- 网络 (三路独立注册表) ----
ACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {"module": "networks.mlp", "class_name": "ActorMLP", "config_class": MLPConfig},
    "rnn": {"module": "networks.rnn", "class_name": "ActorRNN", "config_class": RNNConfig},
    "sun": {"module": "networks.custom.suns", "class_name": "SUNActor", "config_class": SUNConfig},
    "mpnn": {"module": "networks.gnn", "class_name": "MPNNActor", "config_class": MPNNConfig},
    "sage": {"module": "networks.gnn", "class_name": "GraphSageActor", "config_class": SAGEConfig},
    # MASUP 专供网络
    "masup_mlp": {"module": "networks.custom.masup_nets", "class_name": "MASUPActorMLP", "config_class": MASUPActorConfig},
    "masup_rnn": {"module": "networks.custom.masup_nets", "class_name": "MASUPActorRNN", "config_class": MASUPActorRNNConfig},
}

CRITIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {"module": "networks.mlp", "class_name": "CriticMLP", "config_class": MLPConfig},
    "rnn": {"module": "networks.rnn", "class_name": "CriticRNN", "config_class": RNNConfig},
    "sun": {"module": "networks.custom.suns", "class_name": "SUNCritic", "config_class": SUNConfig},
    # MASUP 专供网络
    "masup_mlp": {"module": "networks.custom.masup_nets", "class_name": "MASUPCriticMLP", "config_class": MASUPCriticConfig},
    "masup_rnn": {"module": "networks.custom.masup_nets", "class_name": "MASUPCriticRNN", "config_class": MASUPCriticRNNConfig},
}

Q_NETWORK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {"module": "networks.mlp", "class_name": "QMLP", "config_class": QMLPConfig},
    "rnn": {"module": "networks.rnn", "class_name": "QRNN", "config_class": QRNNConfig},
    # MASUP 专供网络
    "masup_q_mlp": {"module": "networks.custom.masup_nets", "class_name": "MASUPQMLP", "config_class": MASUPQConfig},
    "masup_q_rnn": {"module": "networks.custom.masup_nets", "class_name": "MASUPQRNN", "config_class": MASUPQRNNConfig},
    "masup_vdppo_mlp": {"module": "networks.custom.masup_nets", "class_name": "MASUPVDPPOQmlp", "config_class": MASUPVDPPOQConfig},
    "masup_vdppo_rnn": {"module": "networks.custom.masup_nets", "class_name": "MASUPVDPPOQrnn", "config_class": MASUPVDPPOQRNNConfig},
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
    "tabular": TrainerConfig,
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
    """创建 Actor 网络 (ActorMLP / ActorRNN / SUNActor / MASUP 专供)。"""
    entry = ACTOR_REGISTRY[actor_type]
    cls = _import_class(entry["module"], entry["class_name"])

    if isinstance(actor_config, MASUPActorRNNConfig):
        return cls(
            graph_path=actor_config.graph_path,
            num_agents=actor_config.num_agents,
            num_nodes=actor_config.num_nodes,
            role_imformation=actor_config.role_imformation,
            gpe_dim=actor_config.gpe_dim,
            proj_dim=actor_config.proj_dim,
            use_log_idleness=actor_config.use_log_idleness,
            T_time=actor_config.T_time,
            hidden_size=actor_config.hidden_size,
            output_dim=action_dim,
            input_dim=obs_dim,
            num_layers=actor_config.num_layers,
            rnn_type=actor_config.rnn_type,
            fc_hidden=actor_config.fc_hidden or None,
        ).to(device)
    elif isinstance(actor_config, MASUPActorConfig):
        return cls(
            graph_path=actor_config.graph_path,
            num_agents=actor_config.num_agents,
            num_nodes=actor_config.num_nodes,
            role_imformation=actor_config.role_imformation,
            gpe_dim=actor_config.gpe_dim,
            proj_dim=actor_config.proj_dim,
            use_log_idleness=actor_config.use_log_idleness,
            T_time=actor_config.T_time,
            hidden=actor_config.hidden,
            output_dim=action_dim,
            input_dim=obs_dim,
        ).to(device)
    elif isinstance(actor_config, SUNConfig):
        return cls(
            obs_dim=obs_dim,
            num_nodes=actor_config.num_nodes,
            node_feat_dim=actor_config.node_feat_dim,
            f1_hidden=actor_config.f1_hidden,
            f2_hidden=actor_config.f2_hidden,
            num_layers=actor_config.num_layers,
        ).to(device)
    elif isinstance(actor_config, RNNConfig):
        return cls(
            input_dim=obs_dim,
            hidden_size=actor_config.hidden_size,
            output_dim=action_dim,
            num_layers=actor_config.num_layers,
            rnn_type=actor_config.rnn_type,
            fc_hidden=actor_config.fc_hidden or None,
        ).to(device)
    elif isinstance(actor_config, (MPNNConfig, SAGEConfig)):
        return cls(obs_dim=obs_dim, action_dim=action_dim, config=actor_config).to(device)
    else:
        return cls(obs_dim, actor_config.hidden, action_dim).to(device)


def create_critic(
    critic_type: str,
    critic_config: NetworkConfig,
    critic_input_dim: int,
    device: str = "cpu",
) -> nn.Module:
    """创建 Critic 网络 (CriticMLP / CriticRNN / SUNCritic / MASUP 专供)。"""
    entry = CRITIC_REGISTRY[critic_type]
    cls = _import_class(entry["module"], entry["class_name"])

    if isinstance(critic_config, MASUPCriticRNNConfig):
        return cls(
            graph_path=critic_config.graph_path,
            num_agents=critic_config.num_agents,
            num_nodes=critic_config.num_nodes,
            role_imformation=critic_config.role_imformation,
            gpe_dim=critic_config.gpe_dim,
            proj_dim=critic_config.proj_dim,
            use_log_idleness=critic_config.use_log_idleness,
            T_time=critic_config.T_time,
            hidden_size=critic_config.hidden_size,
            input_dim=critic_input_dim,
            input_mode=critic_config.input_mode,
            num_layers=critic_config.num_layers,
            rnn_type=critic_config.rnn_type,
            fc_hidden=critic_config.fc_hidden or None,
        ).to(device)
    elif isinstance(critic_config, MASUPCriticConfig):
        return cls(
            graph_path=critic_config.graph_path,
            num_agents=critic_config.num_agents,
            num_nodes=critic_config.num_nodes,
            role_imformation=critic_config.role_imformation,
            gpe_dim=critic_config.gpe_dim,
            proj_dim=critic_config.proj_dim,
            use_log_idleness=critic_config.use_log_idleness,
            T_time=critic_config.T_time,
            hidden=critic_config.hidden,
            input_dim=critic_input_dim,
            input_mode=critic_config.input_mode,
        ).to(device)
    elif isinstance(critic_config, SUNConfig):
        return cls(
            obs_dim=critic_input_dim,
            num_nodes=critic_config.num_nodes,
            node_feat_dim=critic_config.node_feat_dim,
            f1_hidden=critic_config.f1_hidden,
            f2_hidden=critic_config.f2_hidden,
            num_layers=critic_config.num_layers,
        ).to(device)
    elif isinstance(critic_config, RNNConfig):
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
    """创建 Q-network (QMLP / QRNN / MASUP 专供)。"""
    entry = Q_NETWORK_REGISTRY[q_type]
    cls = _import_class(entry["module"], entry["class_name"])
    dueling = getattr(q_config, "dueling", False)

    if isinstance(q_config, MASUPVDPPOQRNNConfig):
        return cls(
            graph_path=q_config.graph_path,
            num_agents=q_config.num_agents,
            num_nodes=q_config.num_nodes,
            gpe_dim=q_config.gpe_dim,
            proj_dim=q_config.proj_dim,
            use_log_idleness=q_config.use_log_idleness,
            T_time=q_config.T_time,
            hidden_size=q_config.hidden_size,
            output_dim=action_dim,
            input_dim=input_dim,
            num_layers=q_config.num_layers,
            rnn_type=q_config.rnn_type,
            dueling=dueling,
            fc_hidden=q_config.fc_hidden or None,
        ).to(device)
    elif isinstance(q_config, MASUPVDPPOQConfig):
        return cls(
            graph_path=q_config.graph_path,
            num_agents=q_config.num_agents,
            num_nodes=q_config.num_nodes,
            gpe_dim=q_config.gpe_dim,
            proj_dim=q_config.proj_dim,
            use_log_idleness=q_config.use_log_idleness,
            T_time=q_config.T_time,
            hidden=q_config.hidden,
            output_dim=action_dim,
            input_dim=input_dim,
            dueling=dueling,
        ).to(device)
    elif isinstance(q_config, MASUPQRNNConfig):
        return cls(
            graph_path=q_config.graph_path,
            num_agents=q_config.num_agents,
            num_nodes=q_config.num_nodes,
            role_imformation=q_config.role_imformation,
            gpe_dim=q_config.gpe_dim,
            proj_dim=q_config.proj_dim,
            use_log_idleness=q_config.use_log_idleness,
            T_time=q_config.T_time,
            hidden_size=q_config.hidden_size,
            output_dim=action_dim,
            input_dim=input_dim,
            num_layers=q_config.num_layers,
            rnn_type=q_config.rnn_type,
            dueling=dueling,
            fc_hidden=q_config.fc_hidden or None,
        ).to(device)
    elif isinstance(q_config, MASUPQConfig):
        return cls(
            graph_path=q_config.graph_path,
            num_agents=q_config.num_agents,
            num_nodes=q_config.num_nodes,
            role_imformation=q_config.role_imformation,
            gpe_dim=q_config.gpe_dim,
            proj_dim=q_config.proj_dim,
            use_log_idleness=q_config.use_log_idleness,
            T_time=q_config.T_time,
            hidden=q_config.hidden,
            output_dim=action_dim,
            input_dim=input_dim,
            dueling=dueling,
        ).to(device)
    elif isinstance(q_config, QRNNConfig):
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
    """创建向量化环境。

    包装优先级（由内到外）：基础 VectorEnv → NormObs → NormReward。

    当 env_config.norm_obs=True 时，外包 VectorEnvNormObs：
    同时适配 Gymnasium 单智能体（obs 为 ndarray）和 PettingZoo 多智能体
    （obs 为 Dict[str, ndarray]）两种接口；同构 agent 共享同一个 RMS。

    当 env_config.norm_reward=True 时，外包 VectorEnvNormReward：
    仅按运行标准差缩放 reward，不减均值。

    注意：norm_obs 对包含整数编码结构（如 GNN 环境的 edge_src/edge_dst）的
    obs 向量不适用，请勿在 MAGEC 等 GNN 环境中开启。
    """
    from envs.venvs import DummyVectorEnv, SubprocVectorEnv
    from envs.venv_wrappers import VectorEnvNormObs, VectorEnvNormReward

    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])

    cfg_dict, custom_dict = _env_config_to_dicts(env_config)
    env_fns = [lambda: env_cls(cfg_dict, **custom_dict) for _ in range(num_envs)]

    if use_subproc:
        vec_env = SubprocVectorEnv(env_fns)
    else:
        vec_env = DummyVectorEnv(env_fns)

    if getattr(env_config, "norm_obs", False):
        vec_env = VectorEnvNormObs(vec_env)

    if getattr(env_config, "norm_reward", False):
        vec_env = VectorEnvNormReward(vec_env)

    return vec_env


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
