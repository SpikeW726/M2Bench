"""
Model I/O utilities following HuggingFace/MMLab style.

Save format:
    model_dir/
    ├── config.yaml      # Network architecture config
    ├── policy.pt        # Policy weights (state_dict only)
    └── critic.pt        # Critic weights (optional)

Usage:
    # Save
    save_model(save_dir, policy=ma_policy, critic=critic_net, 
               actor_config={...}, critic_config={...})
    
    # Load
    actor, critic = load_model(model_dir, device='cuda')
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import yaml
import torch
import torch.nn as nn


def save_model(
    save_dir: str | Path,
    policy: nn.Module,
    critic: Optional[nn.Module] = None,
    actor_config: Optional[Dict[str, Any]] = None,
    critic_config: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
):
    """
    Save model in HuggingFace style (config + weights separately).
    
    Args:
        save_dir: Directory to save model
        policy: Policy module (MultiAgentPolicy or single policy)
        critic: Critic module (optional)
        actor_config: Actor network config (type, input_dim, hidden_sizes, output_dim)
        critic_config: Critic network config
        extra_info: Additional info to save in config (training params, etc.)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config
    config = {}
    
    if actor_config:
        config['actor'] = actor_config
    
    if critic_config:
        config['critic'] = critic_config
    
    if extra_info:
        config['extra'] = extra_info
    
    # Save config
    config_path = save_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Save policy weights
    policy_path = save_dir / 'policy.pt'
    torch.save(policy.state_dict(), policy_path)
    
    # Save critic weights (if provided)
    if critic is not None:
        critic_path = save_dir / 'critic.pt'
        torch.save(critic.state_dict(), critic_path)


def load_model(
    model_dir: str | Path,
    device: str | torch.device = 'cpu',
    actor_class: Optional[Type[nn.Module]] = None,
    critic_class: Optional[Type[nn.Module]] = None,
) -> Tuple[nn.Module, Optional[nn.Module], Dict[str, Any]]:
    """
    Load model from HuggingFace style directory.
    
    Args:
        model_dir: Directory containing config.yaml and weights
        device: Device to load model to
        actor_class: Actor network class (if None, uses class from config)
        critic_class: Critic network class (if None, uses class from config)
    
    Returns:
        actor: Loaded actor network
        critic: Loaded critic network (or None)
        config: Full config dict
    """
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import network classes dynamically
    from trainers.imitator.imitation_trainer import ActorMLP, CriticMLP
    
    CLASS_REGISTRY = {
        'ActorMLP': ActorMLP,
        'CriticMLP': CriticMLP,
    }
    
    # Build and load actor
    actor = None
    actor_config = config.get('actor', {})
    if actor_config:
        if actor_class is None:
            actor_type = actor_config.get('type', 'ActorMLP')
            actor_class = CLASS_REGISTRY.get(actor_type)
            if actor_class is None:
                raise ValueError(f"Unknown actor type: {actor_type}")
        
        actor = actor_class(
            input_dim=actor_config['input_dim'],
            hidden_sizes=actor_config['hidden_sizes'],
            output_dim=actor_config['output_dim'],
        )
        
        # Load weights
        policy_path = model_dir / 'policy.pt'
        if policy_path.exists():
            state_dict = torch.load(policy_path, map_location=device, weights_only=True)
            # Handle MultiAgentPolicy format
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
        if critic_class is None:
            critic_type = critic_config.get('type', 'CriticMLP')
            critic_class = CLASS_REGISTRY.get(critic_type)
            if critic_class is None:
                raise ValueError(f"Unknown critic type: {critic_type}")
        
        critic = critic_class(
            input_dim=critic_config['input_dim'],
            hidden_sizes=critic_config['hidden_sizes'],
            output_dim=critic_config.get('output_dim', 1),
        )
        
        state_dict = torch.load(critic_path, map_location=device, weights_only=True)
        critic.load_state_dict(state_dict)
        critic = critic.to(device)
        critic.eval()
    
    return actor, critic, config


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
