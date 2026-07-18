"""
MLP network modules: ActorMLP, CriticMLP, QMLP.
"""

import numpy as np
import torch
import torch.nn as nn

def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize linear layer with orthogonal weights.

    Args:
        layer: nn.Linear layer
        std: Gain for orthogonal initialization
        bias_const: Constant value for bias
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorMLP(nn.Module):
    """
    Actor network with configurable hidden layers.
    Output layer gain = 0.01 for stable policy initialization.
    """
    is_recurrent = False

    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        # Store config for checkpoint saving.
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim

        # Build hidden layers.
        for h_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(current_dim, h_dim), std=np.sqrt(2)))
            layers.append(nn.Tanh())
            current_dim = h_dim

        # Output layer with small std for stable init.
        layers.append(layer_init(nn.Linear(current_dim, output_dim), std=0.01))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": self.hidden_sizes,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "ActorMLP":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_sizes=cfg["hidden_sizes"],
            output_dim=cfg["output_dim"],
        )

class CriticMLP(nn.Module):
    """
    Critic network (Value Function).
    Output layer gain = 1.0.
    """
    is_recurrent = False

    def __init__(self, input_dim, hidden_sizes, output_dim=1):
        super().__init__()
        # Store config for checkpoint saving.
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim

        # Build hidden layers.
        for h_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(current_dim, h_dim), std=np.sqrt(2)))
            layers.append(nn.Tanh())
            current_dim = h_dim

        # Output layer with std=1.0.
        layers.append(layer_init(nn.Linear(current_dim, output_dim), std=1.0))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": self.hidden_sizes,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "CriticMLP":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_sizes=cfg["hidden_sizes"],
            output_dim=cfg.get("output_dim", 1),
        )

class QMLP(nn.Module):
    is_recurrent = False

    def __init__(self, input_dim, hidden_sizes, output_dim, dueling=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim
        self.dueling = dueling

        hidden_layers = []
        current_dim = input_dim
        for h_dim in hidden_sizes:
            hidden_layers.append(layer_init(nn.Linear(current_dim, h_dim), std=np.sqrt(2)))
            hidden_layers.append(nn.Tanh())
            current_dim = h_dim
        self.shared = nn.Sequential(*hidden_layers)

        if dueling:
            self.v_stream = layer_init(nn.Linear(current_dim, 1), std=1.0)
            self.a_stream = layer_init(nn.Linear(current_dim, output_dim), std=1.0)
        else:
            self.q_head = layer_init(nn.Linear(current_dim, output_dim), std=1.0)

    def forward(self, x):
        features = self.shared(x)
        if self.dueling:
            v = self.v_stream(features)                          # (batch, 1).
            a = self.a_stream(features)                          # (batch, output_dim).
            return v + a - a.mean(dim=-1, keepdim=True)          # (batch, output_dim).
        return self.q_head(features)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": self.hidden_sizes,
            "dueling": self.dueling,
        }

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "QMLP":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_sizes=cfg["hidden_sizes"],
            output_dim=cfg["output_dim"],
            dueling=cfg.get("dueling", False),
        )
