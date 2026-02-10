"""Multi-Agent Policy: wraps single-agent policies for multi-agent use."""

from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from policies.rl.rl_base import RLBasePolicy


class MultiAgentPolicy(nn.Module):
    """
    Multi-agent policy wrapper supporting shared/independent modes.
    Only support homogeneous agents now
    
    Args:
        agent_ids: List of agent identifiers
        obs_space: Observation space (homogeneous agents)
        action_space: Action space (homogeneous agents)
        policy_class: Single-agent policy class (e.g., ActorPolicy)
        policy_kwargs: Additional kwargs for policy_class
        shared: If True, all agents share one policy (parameter sharing)
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        obs_space: gym.Space,
        action_space: gym.Space,
        policy_class: type,
        policy_kwargs: Optional[Dict] = None,
        shared: bool = False,
    ):
        super().__init__()
        self.agent_ids = agent_ids
        self.obs_space = obs_space
        self.action_space = action_space
        self.shared = shared
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        policy_kwargs = policy_kwargs or {}
        
        if shared:
            # All agents share one policy instance
            self._shared_policy = policy_class(
                obs_space, action_space, **policy_kwargs
            ).to(self.device)
        else:
            # Each agent has independent policy
            self._policy_dict = nn.ModuleDict()
            for aid in agent_ids:
                # Deep copy nn.Module in kwargs for independent params
                agent_kwargs = {
                    k: copy.deepcopy(v) if isinstance(v, nn.Module) else v
                    for k, v in policy_kwargs.items()
                }
                self._policy_dict[aid] = policy_class(
                    obs_space, action_space, **agent_kwargs
                ).to(self.device)
    
    def get_policy(self, agent_id: str) -> RLBasePolicy:
        """Get policy for specific agent."""
        if self.shared:
            return self._shared_policy
        return self._policy_dict[agent_id]
    
    @property
    def num_agents(self) -> int:
        return len(self.agent_ids)
    
    # =========================================================================
    #                              Forward
    # =========================================================================
    
    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        state_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Forward pass for all agents.
        
        Args:
            obs_dict: {agent_id: obs_tensor} with shape (batch, *obs_shape)
            state_dict: {agent_id: hidden_state} for RNN (optional)
            kwargs: may contain 'action_mask': {agent_id: mask}
        
        Returns:
            {agent_id: {'act', 'log_prob', 'dist', ...}}
        """
        if self.shared:
            return self._forward_shared(obs_dict, **kwargs)
        return self._forward_independent(obs_dict, state_dict, **kwargs)
    
    def _forward_shared(
        self,
        obs_dict: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Shared mode: batch all agents for single forward pass."""
        agents = list(obs_dict.keys())
        if not agents:
            return {}
        
        # Stack: [num_agents, batch, *obs_shape] -> [num_agents * batch, *obs_shape]
        obs_list = [obs_dict[a] for a in agents]
        first_obs = obs_list[0]
        single_sample = first_obs.dim() == len(self.obs_space.shape)
        
        if single_sample:
            obs_list = [o.unsqueeze(0) for o in obs_list]
        
        stacked = torch.stack(obs_list, dim=0)
        n_agents, batch = stacked.shape[:2]
        flat_obs = stacked.view(n_agents * batch, *stacked.shape[2:])
        
        # Handle action_mask
        action_mask = kwargs.get("action_mask", {})
        flat_mask = None
        if action_mask:
            masks = [action_mask.get(a) for a in agents]
            if all(m is not None for m in masks):
                if single_sample:
                    masks = [m.unsqueeze(0) for m in masks]
                flat_mask = torch.stack(masks, dim=0).view(n_agents * batch, -1)
        
        # Single forward pass
        out = self._shared_policy.forward(
            flat_obs, action_mask=flat_mask if flat_mask is not None else None
        )
        
        # Unflatten: [n_agents * batch, ...] -> {agent: [batch, ...]}
        results = {}
        for i, agent in enumerate(agents):
            agent_out = {}
            for key, val in out.items():
                if isinstance(val, torch.Tensor):
                    unflat = val.view(n_agents, batch, *val.shape[1:])[i]
                    agent_out[key] = unflat.squeeze(0) if single_sample else unflat
                else:
                    agent_out[key] = val
            results[agent] = agent_out
        return results
    
    def _forward_independent(
        self,
        obs_dict: Dict[str, torch.Tensor],
        state_dict: Optional[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Independent mode: call each agent's policy separately."""
        action_mask = kwargs.get("action_mask", {})
        state_dict = state_dict or {}
        results = {}
        
        for agent, obs in obs_dict.items():
            policy = self._policy_dict[agent]
            mask = action_mask.get(agent)
            state = state_dict.get(agent)
            results[agent] = policy.forward(obs, state=state, action_mask=mask)
        
        return results
    
    # =========================================================================
    #                         Environment Interaction
    # =========================================================================
    
    def compute_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        info_dict: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Compute actions from numpy observations (for env interaction).
        
        Args:
            obs_dict: {agent_id: obs_array} shape (num_envs, *obs_shape)
            info_dict: {agent_id: info_array} 每个元素是 dict（来自 VectorEnv）
        
        Returns:
            actions: {agent_id: action_array}
            outputs: {agent_id: policy_output}
        """
        info_dict = info_dict or {}
        
        # Convert to tensor
        obs_tensor = {
            a: torch.as_tensor(o, dtype=torch.float32, device=self.device)
            for a, o in obs_dict.items()
        }
        
        # Extract action_mask from vectorized info
        # info_dict[agent] 是一个 np.ndarray，每个元素是一个 dict
        action_mask = {}
        for agent, info_arr in info_dict.items():
            masks = []
            for info in info_arr:
                if isinstance(info, dict) and 'action_mask' in info:
                    masks.append(info['action_mask'])
            if masks:
                action_mask[agent] = torch.as_tensor(
                    np.stack(masks), dtype=torch.bool, device=self.device
                )
        
        with torch.no_grad():
            outputs = self.forward(obs_tensor, action_mask=action_mask, **kwargs)
        
        # Convert to numpy
        actions = {}
        for agent, out in outputs.items():
            act = out["act"]
            act_np = act.cpu().numpy() if isinstance(act, torch.Tensor) else act
            act_np = self.get_policy(agent).map_action(act_np)
            actions[agent] = act_np
        
        return actions, outputs
    
    # =========================================================================
    #                              Training
    # =========================================================================
    
    def evaluate_actions(
        self,
        obs_dict: Dict[str, torch.Tensor],
        act_dict: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Evaluate actions for all agents.
        
        Returns:
            log_probs: {agent_id: log_prob}
            entropies: {agent_id: entropy}
        """
        if self.shared:
            return self._evaluate_shared(obs_dict, act_dict, **kwargs)
        
        action_mask = kwargs.get("action_mask", {})
        log_probs, entropies = {}, {}
        
        for agent in obs_dict:
            policy = self._policy_dict[agent]
            lp, ent = policy.evaluate_actions(
                obs_dict[agent], act_dict[agent], 
                action_mask=action_mask.get(agent)
            )
            log_probs[agent] = lp
            entropies[agent] = ent
        
        return log_probs, entropies
    
    def _evaluate_shared(
        self,
        obs_dict: Dict[str, torch.Tensor],
        act_dict: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Shared mode: batch evaluation."""
        agents = list(obs_dict.keys())
        if not agents:
            return {}, {}
        
        # Stack obs and actions
        obs_list = [obs_dict[a] for a in agents]
        act_list = [act_dict[a] for a in agents]
        batch = obs_list[0].shape[0]
        n_agents = len(agents)
        
        flat_obs = torch.stack(obs_list, dim=0).view(n_agents * batch, -1)
        flat_act = torch.stack(act_list, dim=0).view(n_agents * batch, -1).squeeze(-1)
        
        # Handle action_mask
        action_mask = kwargs.get("action_mask", {})
        flat_mask = None
        if action_mask:
            masks = [action_mask.get(a) for a in agents]
            if all(m is not None for m in masks):
                flat_mask = torch.stack(masks, dim=0).view(n_agents * batch, -1)
        
        # Single evaluate call
        log_prob, entropy = self._shared_policy.evaluate_actions(
            flat_obs, flat_act, action_mask=flat_mask
        )
        
        # Unflatten
        log_prob = log_prob.view(n_agents, batch)
        entropy = entropy.view(n_agents, batch)
        
        log_probs = {agents[i]: log_prob[i] for i in range(n_agents)}
        entropies = {agents[i]: entropy[i] for i in range(n_agents)}
        return log_probs, entropies
    
    def evaluate_actions_flat(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate with flattened tensors (for MAPPO training).
        Only works with shared=True.
        """
        if not self.shared:
            raise ValueError("evaluate_actions_flat requires shared=True")
        return self._shared_policy.evaluate_actions(obs, act, **kwargs)
    
    def set_training_mode(self, mode: bool):
        """Set training/eval mode."""
        self.train(mode)
        if self.shared:
            self._shared_policy.set_training_mode(mode)
        else:
            for policy in self._policy_dict.values():
                policy.set_training_mode(mode)
