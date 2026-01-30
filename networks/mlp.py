"""
MLP network modules for Actor and Critic.
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
    
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        # Store config for checkpoint saving
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim
        
        layers = []
        current_dim = input_dim

        # Build hidden layers
        for h_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(current_dim, h_dim), std=np.sqrt(2)))
            layers.append(nn.Tanh())
            current_dim = h_dim

        # Output layer with small std for stable init
        layers.append(layer_init(nn.Linear(current_dim, output_dim), std=0.01))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CriticMLP(nn.Module):
    """
    Critic network (Value Function).
    Output layer gain = 1.0.
    """
    
    def __init__(self, input_dim, hidden_sizes, output_dim=1):
        super().__init__()
        # Store config for checkpoint saving
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim
        
        layers = []
        current_dim = input_dim

        # Build hidden layers
        for h_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(current_dim, h_dim), std=np.sqrt(2)))
            layers.append(nn.Tanh())
            current_dim = h_dim

        # Output layer with std=1.0
        layers.append(layer_init(nn.Linear(current_dim, output_dim), std=1.0))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
