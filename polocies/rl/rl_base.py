from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete


class RLBasePolicy(nn.Module, ABC):
    """Base class for RL policies. Maps observations to actions."""
    
    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._training_mode = True
        
        # Determine action type
        if isinstance(action_space, (Discrete, MultiDiscrete, MultiBinary)):
            self._action_type: Literal["discrete", "continuous"] = "discrete"
        elif isinstance(action_space, Box):
            self._action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {action_space}")
    
    @property
    def action_type(self) -> Literal["discrete", "continuous"]:
        return self._action_type
    
    @property
    def is_discrete(self) -> bool:
        return self._action_type == "discrete"

    @abstractmethod
    def forward(self, obs: torch.Tensor, state: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Forward pass: obs -> action. Returns dict with 'act' and other info."""
        pass

    def compute_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        state: dict | np.ndarray | None = None,
        **kwargs
    ) -> np.ndarray | int:
        """Get action as int (for discrete env's) or array (for continuous ones) from an env's observation and info.

        :param obs: observation from the gym's env.
        :param info: information given by the gym's env.
        :param state: the hidden state of RNN policy, used for recurrent policy.
        :return: action as int (for discrete env's) or array (for continuous ones).
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output = self.forward(obs_tensor, state=state, **kwargs)
        
        act = output['act']
        
        # To numpy
        act_np = act.cpu().numpy() if isinstance(act, torch.Tensor) else act
        act_np = self.map_action(act_np)
        
        return act_np.squeeze(0) if obs.shape[0] == 1 else act_np, output

    def add_exploration_noise(self, act: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Add exploration noise. Override in subclass."""
        return act
    
    def map_action(self, act: np.ndarray) -> np.ndarray:
        """Map action to valid range. Override for custom mapping."""
        if isinstance(self.action_space, Box):
            act = np.clip(act, self.action_space.low, self.action_space.high)
        return act

    def set_training_mode(self, mode: bool):
        self._training_mode = mode
        self.train(mode)


class ActorPolicy(RLBasePolicy):
    """Policy for actor-critic algorithms (PPO, A2C, SAC, etc.)."""
    
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        actor: nn.Module,
        deterministic_eval:bool = False
    ):
        super().__init__(obs_space, action_space)
        self.deterministic_eval = deterministic_eval
        self.actor = actor.to(self.device)
    
    def forward(self, obs: torch.Tensor, state: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for actor-critic policy.
        
        Args:
            obs: (batch, *obs_shape) observation tensor
            state: hidden state for RNN (optional)
            kwargs: may contain 'action_mask' for discrete action masking
        
        Returns:
            dict with 'act', 'log_prob', 'dist', 'state'
        """
        if self.is_discrete:
            return self._forward_discrete(obs, state, **kwargs)
        else:
            return self._forward_continuous(obs, state, **kwargs)
    
    def _forward_discrete(self, obs: torch.Tensor, state: Optional[Dict], **kwargs) -> Dict[str, Any]:
        """Forward for discrete action space."""
        logits = self.actor(obs)
        
        # Apply action_mask if provided
        action_mask = kwargs.get('action_mask', None)
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask_t, float('-inf'))
        
        dist = torch.distributions.Categorical(logits=logits)
        act = (
            dist.sample()
            if not self.deterministic_eval and self._training_mode
            else logits.argmax(dim=-1)
        )
        log_prob = dist.log_prob(act)
        
        return {'act': act, 'log_prob': log_prob, 'logits': logits, 'dist': dist, 'state': state}
    
    def _forward_continuous(self, obs: torch.Tensor, state: Optional[Dict], **kwargs) -> Dict[str, Any]:
        """Forward for continuous action space."""
        out = self.actor(obs)
        
        # Actor outputs (mean, log_std) or just mean
        if isinstance(out, tuple):
            mean, log_std = out
        else:
            mean = out
            log_std = torch.zeros_like(mean)  # Deterministic if no log_std
        
        std = log_std.exp().clamp(min=1e-6)
        dist = torch.distributions.Normal(mean, std)
        act = (
            dist.rsample()
            if not self.deterministic_eval
            else mean
        )
        log_prob = dist.log_prob(act).sum(dim=-1)
        
        return {'act': act, 'log_prob': log_prob, 'mean': mean, 'std': std, 'dist': dist, 'state': state}

    def evaluate_actions(self, obs: torch.Tensor, act: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions. Used in Algorithm.update() for computing loss.
        
        Returns:
            log_prob: log probability of actions
            entropy: distribution entropy
        """
        if self.is_discrete:
            logits = self.actor(obs)
            action_mask = kwargs.get('action_mask', None)
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, float('-inf'))
            dist = torch.distributions.Categorical(logits=logits)
        else:
            out = self.actor(obs)
            if isinstance(out, tuple):
                mean, log_std = out
            else:
                mean, log_std = out, torch.zeros_like(out)
            std = log_std.exp().clamp(min=1e-6)
            dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(act)
        if not self.is_discrete:
            log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy()
        if not self.is_discrete:
            entropy = entropy.sum(dim=-1)
        
        return log_prob, entropy
