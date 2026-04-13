"""
Model I/O utilities following HuggingFace/MMLab style.

Save format:
    model_dir/
    ├── config.yaml      # Network architecture config + eval metadata
    ├── policy.pt        # Policy weights (state_dict only)
    └── critic.pt        # Critic weights (optional)

Usage:
    # Save
    save_model(save_dir, policy=ma_policy, critic=critic_net,
               actor_config={...}, critic_config={...}, q_config={...})

    # Load for eval
    multi_policy = load_policy_for_eval(model_dir, agent_ids, obs_space, action_space)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import yaml
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


def _convert_to_native_types(obj: Any) -> Any:
    """递归将 numpy 类型转为原生 Python 类型，确保 YAML 可序列化。"""
    if isinstance(obj, dict):
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def _get_class_registry() -> Dict[str, type]:
    """延迟导入，构建网络类名 → 类对象的映射。新增网络时在此注册。"""
    from networks.mlp import ActorMLP, CriticMLP, QMLP
    from networks.rnn import ActorRNN, CriticRNN, QRNN
    from networks.custom.suns import SUNActor, SUNCritic
    from networks.gnn import MPNNActor, GraphSageActor
    from networks.custom.masup_nets import (
        MASUPActorMLP, MASUPActorRNN,
        MASUPCriticMLP, MASUPCriticRNN,
        MASUPQMLP, MASUPQRNN,
        MASUPVDPPOQmlp, MASUPVDPPOQrnn,
    )

    return {
        "ActorMLP": ActorMLP,
        "CriticMLP": CriticMLP,
        "QMLP": QMLP,
        "ActorRNN": ActorRNN,
        "CriticRNN": CriticRNN,
        "QRNN": QRNN,
        "SUNActor": SUNActor,
        "SUNCritic": SUNCritic,
        "MPNNActor": MPNNActor,
        "GraphSageActor": GraphSageActor,
        # MASUP 专供网络
        "MASUPActorMLP": MASUPActorMLP,
        "MASUPActorRNN": MASUPActorRNN,
        "MASUPCriticMLP": MASUPCriticMLP,
        "MASUPCriticRNN": MASUPCriticRNN,
        "MASUPQMLP": MASUPQMLP,
        "MASUPQRNN": MASUPQRNN,
        "MASUPVDPPOQmlp": MASUPVDPPOQmlp,
        "MASUPVDPPOQrnn": MASUPVDPPOQrnn,
    }


def _build_network_from_config(net_cfg: dict, registry: Dict[str, type]) -> nn.Module:
    """根据 config dict 实例化网络。

    每个网络类须实现 from_config_dict(cls, cfg) 类方法，此处仅做调度。
    """
    net_type = net_cfg["type"]
    cls = registry.get(net_type)
    if cls is None:
        raise ValueError(
            f"Unknown network type: '{net_type}'. "
            f"Register it in utils/model_io._get_class_registry."
        )
    if not hasattr(cls, "from_config_dict"):
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config_dict(cls, cfg) for evaluation loading."
        )
    return cls.from_config_dict(net_cfg)


# =========================================================================
#                              Save
# =========================================================================

def save_model(
    save_dir: str | Path,
    policy: nn.Module,
    critic: Optional[nn.Module] = None,
    actor_config: Optional[Dict[str, Any]] = None,
    critic_config: Optional[Dict[str, Any]] = None,
    q_config: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
):
    """
    Save model in HuggingFace style (config + weights separately).

    Args:
        save_dir: Directory to save model
        policy: Policy module (MultiAgentPolicy or single policy)
        critic: Critic module (optional)
        actor_config: Actor network config
        critic_config: Critic network config
        q_config: Q-network config (for value-based algorithms)
        extra_info: Additional metadata (algo_name, policy_type, etc.)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config: dict = {}

    if actor_config:
        config['actor'] = actor_config
    if critic_config:
        config['critic'] = critic_config
    if q_config:
        config['q_network'] = q_config
    if extra_info:
        config['extra'] = extra_info

    config = _convert_to_native_types(config)

    config_path = save_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    policy_path = save_dir / 'policy.pt'
    torch.save(policy.state_dict(), policy_path)

    if critic is not None:
        critic_path = save_dir / 'critic.pt'
        torch.save(critic.state_dict(), critic_path)


# =========================================================================
#                              Load (legacy)
# =========================================================================

def load_model(
    model_dir: str | Path,
    device: str | torch.device = 'cpu',
    actor_class: Optional[Type[nn.Module]] = None,
    critic_class: Optional[Type[nn.Module]] = None,
) -> Tuple[nn.Module, Optional[nn.Module], Dict[str, Any]]:
    """
    Load model from HuggingFace style directory (legacy API).

    Returns:
        actor: Loaded actor network
        critic: Loaded critic network (or None)
        config: Full config dict
    """
    model_dir = Path(model_dir)

    config_path = model_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    registry = _get_class_registry()

    # Build and load actor
    actor = None
    actor_config = config.get('actor', {})
    if actor_config:
        if actor_class is not None:
            # 调用方显式传入类时，仍走旧路径（向后兼容）
            actor = actor_class(
                input_dim=actor_config['input_dim'],
                hidden_sizes=actor_config['hidden_sizes'],
                output_dim=actor_config['output_dim'],
            )
        else:
            actor = _build_network_from_config(actor_config, registry)

        policy_path = model_dir / 'policy.pt'
        if policy_path.exists():
            state_dict = torch.load(policy_path, map_location=device, weights_only=True)
            if '_shared_policy.actor.network.0.weight' in state_dict:
                actor_sd = {}
                prefix = '_shared_policy.actor.'
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        actor_sd[k[len(prefix):]] = v
                state_dict = actor_sd
            actor.load_state_dict(state_dict)

        actor = actor.to(device)
        actor.eval()

    # Build and load critic
    critic = None
    critic_config = config.get('critic', {})
    critic_path = model_dir / 'critic.pt'
    if critic_config and critic_path.exists():
        if critic_class is not None:
            critic = critic_class(
                input_dim=critic_config['input_dim'],
                hidden_sizes=critic_config['hidden_sizes'],
                output_dim=critic_config.get('output_dim', 1),
            )
        else:
            critic = _build_network_from_config(critic_config, registry)

        state_dict = torch.load(critic_path, map_location=device, weights_only=True)
        critic.load_state_dict(state_dict)
        critic = critic.to(device)
        critic.eval()

    return actor, critic, config


# =========================================================================
#                     Load for Evaluation (通用)
# =========================================================================

def _propagate_device_to_policies(module: nn.Module, device: torch.device) -> None:
    """递归将 device 同步到所有 RLBasePolicy 子类，确保评估时 mask/obs 与网络同设备。"""
    from policies.rl.rl_base import RLBasePolicy
    if isinstance(module, RLBasePolicy):
        module.device = device
    for child in module.children():
        _propagate_device_to_policies(child, device)


def load_policy_for_eval(
    model_dir: str | Path,
    agent_ids: List[str],
    obs_space: gym.Space,
    action_space: gym.Space,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    """
    从模型目录自动重建 MultiAgentPolicy 并加载权重。

    读取 config.yaml 中的 extra 段获取 policy_type / shared_policy 等元信息,
    自动选择 ActorPolicy 或 ValuePolicy 并实例化对应网络。

    向后兼容: 若缺少 extra 段则回退到 actor + shared 默认值。
    """
    from policies.rl.rl_base import ActorPolicy, ValuePolicy
    from policies.marl.marl_base import MultiAgentPolicy

    model_dir = Path(model_dir)
    config = get_model_config(model_dir)
    extra = config.get('extra', {})
    registry = _get_class_registry()

    policy_type = extra.get('policy_type', 'actor')
    shared = extra.get('shared_policy', True)

    if policy_type == 'value':
        q_cfg = config.get('q_network')
        if q_cfg is None:
            raise ValueError(
                f"Value-based policy requires q_network config in {model_dir / 'config.yaml'}"
            )
        q_net = _build_network_from_config(q_cfg, registry)
        policy_class = ValuePolicy
        policy_kwargs = {"q_network": q_net}
    else:
        actor_cfg = config.get('actor')
        if actor_cfg is None:
            raise ValueError(
                f"Actor-based policy requires actor config in {model_dir / 'config.yaml'}"
            )
        actor_net = _build_network_from_config(actor_cfg, registry)
        policy_class = ActorPolicy
        policy_kwargs = {"actor": actor_net, "deterministic_eval": True}

    multi_policy = MultiAgentPolicy(
        agent_ids=agent_ids,
        obs_space=obs_space,
        action_space=action_space,
        policy_class=policy_class,
        policy_kwargs=policy_kwargs,
        shared=shared,
    )

    # 加载权重
    policy_path = model_dir / 'policy.pt'
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy weights not found: {policy_path}")

    state_dict = torch.load(policy_path, map_location=device, weights_only=True)

    # 兼容单体训练→多体评估场景（如 A2C+suns_gym → suns）：
    # 单体训练保存的是裸 Policy state_dict（key 无 _shared_policy./_policy_dict. 前缀），
    # 而 MultiAgentPolicy(shared=True) 期望 _shared_policy.* 前缀。
    if state_dict:
        first_key = next(iter(state_dict))
        is_bare_policy = not (
            first_key.startswith('_shared_policy.')
            or first_key.startswith('_policy_dict.')
        )
        if is_bare_policy and shared:
            state_dict = {f'_shared_policy.{k}': v for k, v in state_dict.items()}

    multi_policy.load_state_dict(state_dict)
    multi_policy.to(device)
    dev = torch.device(device)
    multi_policy.device = dev
    # 递归同步内层 policy 的 device，避免 RLBasePolicy.__init__ 中 hardcode cuda 导致 mask/obs 设备不一致
    _propagate_device_to_policies(multi_policy, dev)
    multi_policy.set_training_mode(False)

    print(f"[ModelIO] Loaded {policy_type}-based policy "
          f"(shared={shared}) from {model_dir}")
    return multi_policy


# =========================================================================
#                           Convenience
# =========================================================================

def load_actor_only(
    model_dir: str | Path,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    """Convenience function to load only the actor network."""
    actor, _, _ = load_model(model_dir, device=device)
    return actor


def get_model_config(model_dir: str | Path) -> Dict[str, Any]:
    """Load only the config without loading weights."""
    config_path = Path(model_dir) / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
