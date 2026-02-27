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

    @property
    def is_recurrent(self) -> bool:
        return False

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
    """Policy for actor-critic algorithms (PPO, A2C, SAC, etc.).

    当 actor 是 RNN 网络 (is_recurrent=True) 时，forward 自动穿透 hidden state，
    并提供 evaluate_actions_sequence 用于 chunk-based 训练。
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        actor: nn.Module,
        deterministic_eval: bool = False,
    ):
        super().__init__(obs_space, action_space)
        self.deterministic_eval = deterministic_eval
        self.actor = actor.to(self.device)

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.actor, "is_recurrent", False)

    # -----------------------------------------------------------------
    #  forward
    # -----------------------------------------------------------------

    def forward(self, obs: torch.Tensor, state=None, **kwargs) -> Dict[str, Any]:
        """
        Args:
            obs: (batch, *obs_shape)
            state: RNN hidden state (recurrent_N, batch, hidden_size) 或 None
            kwargs: 'action_mask' 等
        Returns:
            dict with 'act', 'log_prob', 'dist', 'state'
        """
        if self.is_discrete:
            return self._forward_discrete(obs, state, **kwargs)
        else:
            return self._forward_continuous(obs, state, **kwargs)

    def _forward_discrete(self, obs: torch.Tensor, state, **kwargs) -> Dict[str, Any]:
        if self.is_recurrent:
            logits, new_state = self.actor(obs, state)
        else:
            logits = self.actor(obs)
            new_state = state

        action_mask = kwargs.get("action_mask", None)
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask_t, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits)
        act = (
            dist.sample()
            if not self.deterministic_eval and self._training_mode
            else logits.argmax(dim=-1)
        )
        log_prob = dist.log_prob(act)

        return {"act": act, "log_prob": log_prob, "logits": logits, "dist": dist, "state": new_state}

    def _forward_continuous(self, obs: torch.Tensor, state, **kwargs) -> Dict[str, Any]:
        if self.is_recurrent:
            out, new_state = self.actor(obs, state)
        else:
            out = self.actor(obs)
            new_state = state

        if isinstance(out, tuple):
            mean, log_std = out
        else:
            mean = out
            log_std = torch.zeros_like(mean)

        std = log_std.exp().clamp(min=1e-6)
        dist = torch.distributions.Normal(mean, std)
        act = dist.rsample() if not self.deterministic_eval else mean
        log_prob = dist.log_prob(act).sum(dim=-1)

        return {"act": act, "log_prob": log_prob, "mean": mean, "std": std, "dist": dist, "state": new_state}

    # -----------------------------------------------------------------
    #  evaluate_actions — MLP flat 路径 (不变)
    # -----------------------------------------------------------------

    def evaluate_actions(self, obs: torch.Tensor, act: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (MLP flat 路径)。

        Returns:
            (log_prob, entropy)
        """
        if self.is_discrete:
            logits = self.actor(obs)
            action_mask = kwargs.get("action_mask", None)
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, float("-inf"))
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

    # -----------------------------------------------------------------
    #  evaluate_actions_sequence — RNN chunk 路径
    # -----------------------------------------------------------------

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        hidden_init: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RNN 序列 evaluate。用于 chunk-based PPO update。

        Args:
            obs_seq: (seq_len, batch, obs_dim)
            act_seq: (seq_len, batch)
            hidden_init: (recurrent_N, batch, hidden_size)
            kwargs: 'action_mask' (seq_len, batch, act_dim) 等
        Returns:
            (log_prob, entropy)  shape = (seq_len, batch)
        """
        logits_seq, _ = self.actor.forward_sequence(obs_seq, hidden_init)

        action_mask = kwargs.get("action_mask", None)

        if self.is_discrete:
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                logits_seq = logits_seq.masked_fill(~mask_t, float("-inf"))
            dist = torch.distributions.Categorical(logits=logits_seq)
            log_prob = dist.log_prob(act_seq)
            entropy = dist.entropy()
        else:
            if isinstance(logits_seq, tuple):
                mean, log_std = logits_seq
            else:
                mean, log_std = logits_seq, torch.zeros_like(logits_seq)
            std = log_std.exp().clamp(min=1e-6)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(act_seq).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

class ValuePolicy(RLBasePolicy):
    """
    Value-based policy: Q-network + epsilon-greedy 动作选择。

    用于 DQN / D3QN / IQL 等 off-policy value-based 算法。
    与 ActorPolicy 的核心区别：
    - 输出 Q 值而非概率分布
    - 探索靠 epsilon-greedy 而非分布采样
    - 不提供 evaluate_actions / log_prob / entropy

    当 q_network 是 RNN (is_recurrent=True) 时，forward 自动穿透 hidden state，
    并提供 compute_q_values_sequence 用于序列训练。
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        q_network: nn.Module,
        epsilon: float = 1.0,
    ):
        super().__init__(obs_space, action_space)
        if not self.is_discrete:
            raise ValueError("ValuePolicy only supports discrete action spaces")
        self.q_network = q_network.to(self.device)
        self.epsilon = epsilon

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.q_network, "is_recurrent", False)

    # -----------------------------------------------------------------
    #  forward (epsilon-greedy)
    # -----------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor, state=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Epsilon-greedy 动作选择。

        training_mode 时使用当前 epsilon 探索，eval 时纯 greedy。
        RNN 时 state 为 hidden state (recurrent_N, batch, H)。

        Returns:
            {'act': (batch,), 'q_values': (batch, num_actions), 'state': hidden}
        """
        if self.is_recurrent:
            hidden = state
            if hidden is None:
                hidden = self.q_network.get_initial_hidden(obs.shape[0], obs.device)
            q_values, new_hidden = self.q_network(obs, hidden)
        else:
            q_values = self.q_network(obs)
            new_hidden = state

        num_actions = q_values.shape[-1]

        action_mask = kwargs.get("action_mask", None)
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values_masked = q_values.masked_fill(~mask_t, float("-inf"))
        else:
            mask_t = None
            q_values_masked = q_values

        greedy = q_values_masked.argmax(dim=-1)             # (batch,)

        if self._training_mode and self.epsilon > 0:
            batch_size = obs.shape[0]
            rand_mask = torch.rand(batch_size, device=self.device) < self.epsilon

            if mask_t is not None:
                valid_counts = mask_t.sum(dim=-1).float()   # (batch,)
                rand_probs = mask_t.float() / valid_counts.unsqueeze(-1).clamp(min=1)
                random_act = torch.multinomial(rand_probs, 1).squeeze(-1)
            else:
                random_act = torch.randint(0, num_actions, (batch_size,), device=self.device)

            act = torch.where(rand_mask, random_act, greedy)
        else:
            act = greedy

        return {"act": act, "q_values": q_values, "state": new_hidden}

    # -----------------------------------------------------------------
    #  compute_q_values (单步, 供 compute_loss 使用)
    # -----------------------------------------------------------------

    def compute_q_values(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """直接返回 Q 值 (batch, num_actions)，供算法 compute_loss 使用。"""
        if self.is_recurrent:
            if hidden is None:
                hidden = self.q_network.get_initial_hidden(obs.shape[0], obs.device)
            q, _ = self.q_network(obs, hidden)
        else:
            q = self.q_network(obs)
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask_t, float("-inf"))
        return q

    # -----------------------------------------------------------------
    #  compute_q_values_sequence (RNN 序列训练)
    # -----------------------------------------------------------------

    def compute_q_values_sequence(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RNN 序列 Q 值计算。

        Args:
            obs_seq: (seq_len, batch, obs_dim)
            hidden: (recurrent_N, batch, H) 或 None（自动零初始化）
        Returns:
            q_seq: (seq_len, batch, act_dim)
            final_hidden: (recurrent_N, batch, H)
        """
        if hidden is None:
            hidden = self.q_network.get_initial_hidden(obs_seq.shape[1], obs_seq.device)
        q_seq, final_h = self.q_network.forward_sequence(obs_seq, hidden)
        return q_seq, final_h

    # ---- epsilon 管理 ----

    def set_epsilon(self, epsilon: float):
        self.epsilon = max(0.0, min(epsilon, 1.0))

    def get_epsilon(self) -> float:
        return self.epsilon
