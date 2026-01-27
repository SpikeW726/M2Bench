"""
Multi-Agent Policy Base Module
多智能体策略基类模块，支持参数共享和独立网络两种模式。
"""

from abc import ABC
from typing import Any, Dict, Iterator, List, Optional, Type, Callable
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from polocies.rl.rl_base import RLBasePolicy


class MultiAgentPolicy(nn.Module):
    """
    多智能体策略容器，支持参数共享和独立网络两种模式。
    
    通过 agent_to_policy 映射和 shared 参数支持从 IQL 到 VDN/QMIX，
    从 IPPO 到 MAPPO/VDPPO 的所有算法。
    
    Args:
        agents: 智能体标识符列表，如 ["agent_0", "agent_1", ...]
        policy_class: 单体策略类（RLBasePolicy 的子类）
        obs_space: 观测空间（同构 agent 共享）
        action_space: 动作空间（同构 agent 共享）
        shared: 是否共享参数
            - True: 所有 agent 共用一个 policy 实例
            - False: 每个 agent 有独立的 policy 实例
        policy_kwargs: 传递给 policy_class 的额外参数
    
    Example:
        >>> # IPPO: 独立策略，不共享参数
        >>> ippo_policy = MultiAgentPolicy(
        ...     agents=["agent_0", "agent_1"],
        ...     policy_class=ActorPolicy,
        ...     obs_space=env.observation_space("agent_0"),
        ...     action_space=env.action_space("agent_0"),
        ...     shared=False,
        ...     policy_kwargs={"actor": actor_net}
        ... )
        >>> 
        >>> # MAPPO: 共享策略参数
        >>> mappo_policy = MultiAgentPolicy(
        ...     agents=["agent_0", "agent_1"],
        ...     policy_class=ActorPolicy,
        ...     obs_space=env.observation_space("agent_0"),
        ...     action_space=env.action_space("agent_0"),
        ...     shared=True,
        ...     policy_kwargs={"actor": actor_net}
        ... )
    """
    
    def __init__(
        self,
        agents: List[str],
        policy_class: Type[RLBasePolicy],
        obs_space: gym.Space,
        action_space: gym.Space,
        shared: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.agents = agents
        self.num_agents = len(agents)
        self.policy_class = policy_class
        self.obs_space = obs_space
        self.action_space = action_space
        self.shared = shared
        self.policy_kwargs = policy_kwargs or {}
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._training_mode = True
        
        # Create policy instances
        self._policies: nn.ModuleDict = nn.ModuleDict()
        self._agent_to_policy: Dict[str, str] = {}
        
        self._init_policies()
    
    def _init_policies(self):
        """
        根据 shared 参数初始化策略实例和映射关系。
        
        - shared=True: 创建一个共享策略 "shared_policy"
        - shared=False: 为每个 agent 创建独立的策略
        """
        if self.shared:
            # Shared mode: 所有 agent 共用一个 policy
            shared_policy = self.policy_class(
                self.obs_space,
                self.action_space,
                **self.policy_kwargs
            )
            self._policies["shared_policy"] = shared_policy
            
            # 所有 agent 映射到同一个 policy
            for agent in self.agents:
                self._agent_to_policy[agent] = "shared_policy"
        else:
            # Independent mode: 每个 agent 有独立的 policy
            for agent in self.agents:
                # 为每个 agent 深拷贝 policy_kwargs 中的网络
                agent_kwargs = self._deep_copy_kwargs(self.policy_kwargs)
                policy = self.policy_class(
                    self.obs_space,
                    self.action_space,
                    **agent_kwargs
                )
                # 使用 agent name 作为 policy key (替换特殊字符以兼容 ModuleDict)
                policy_key = agent.replace("-", "_")
                self._policies[policy_key] = policy
                self._agent_to_policy[agent] = policy_key
    
    def _deep_copy_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝 kwargs 中的 nn.Module 对象，确保独立网络参数。"""
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, nn.Module):
                new_kwargs[key] = copy.deepcopy(value)
            else:
                new_kwargs[key] = value
        return new_kwargs
    
    def get_policy(self, agent: str) -> RLBasePolicy:
        """获取指定 agent 对应的 policy 实例。"""
        policy_key = self._agent_to_policy[agent]
        return self._policies[policy_key]
    
    @property
    def policy_ids(self) -> List[str]:
        """返回所有 policy 的 key 列表（去重后）。"""
        return list(self._policies.keys())
    
    # ============================================================================
    #                             Forward Methods
    # ============================================================================
    
    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        state_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量前向推理，为所有 agent 计算动作。
        
        Args:
            obs_dict: {agent_str: obs_tensor} 格式的观测字典
                - obs_tensor shape: (batch, *obs_shape) 或 (*obs_shape)
            state_dict: {agent_str: state} 格式的 RNN 隐状态（可选）
            **kwargs: 传递给单体 policy 的额外参数
                - action_mask: {agent_str: mask_tensor}
        
        Returns:
            {agent_str: {act, log_prob, dist, state, ...}} 格式的输出字典
        """
        state_dict = state_dict or {}
        results = {}
        
        if self.shared:
            # Shared mode: 合并所有 agent 的 obs 进行批量计算
            results = self._forward_shared(obs_dict, state_dict, **kwargs)
        else:
            # Independent mode: 分别调用每个 agent 的 policy
            results = self._forward_independent(obs_dict, state_dict, **kwargs)
        
        return results
    
    def _forward_shared(
        self,
        obs_dict: Dict[str, torch.Tensor],
        state_dict: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """共享模式的前向推理 - 合并 batch 后一次计算。"""
        active_agents = list(obs_dict.keys())
        if not active_agents:
            return {}
        
        # Stack observations: [num_agents, batch, *obs_shape] or [num_agents, *obs_shape]
        obs_list = [obs_dict[agent] for agent in active_agents]
        
        # 确保所有 obs 有相同的维度
        first_obs = obs_list[0]
        if first_obs.dim() == 1:
            # Single sample: (*obs_shape) -> (1, *obs_shape)
            obs_list = [o.unsqueeze(0) for o in obs_list]
        
        # Stack: [num_agents, batch, *obs_shape]
        stacked_obs = torch.stack(obs_list, dim=0)
        num_agents, batch_size = stacked_obs.shape[:2]
        
        # Flatten agent and batch dims: [num_agents * batch, *obs_shape]
        flat_obs = stacked_obs.view(num_agents * batch_size, *stacked_obs.shape[2:])
        
        # Handle action_mask if provided
        action_mask = kwargs.get("action_mask")
        flat_mask = None
        if action_mask is not None:
            mask_list = [action_mask.get(agent) for agent in active_agents]
            if all(m is not None for m in mask_list):
                # 确保 mask 维度一致
                if mask_list[0].dim() == 1:
                    mask_list = [m.unsqueeze(0) for m in mask_list]
                stacked_mask = torch.stack(mask_list, dim=0)
                flat_mask = stacked_mask.view(num_agents * batch_size, -1)
        
        # Forward through shared policy
        shared_policy = self._policies["shared_policy"]
        forward_kwargs = {k: v for k, v in kwargs.items() if k != "action_mask"}
        if flat_mask is not None:
            forward_kwargs["action_mask"] = flat_mask
        
        # 合并 state（如果有 RNN）
        flat_state = None  # TODO: 支持 RNN state 合并
        
        output = shared_policy.forward(flat_obs, state=flat_state, **forward_kwargs)
        
        # Unflatten results back to per-agent format
        results = {}
        for i, agent in enumerate(active_agents):
            agent_result = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    # [num_agents * batch, ...] -> [batch, ...]
                    agent_value = value.view(num_agents, batch_size, *value.shape[1:])[i]
                    # 如果原始输入是单样本，去掉 batch 维度
                    if first_obs.dim() == 1:
                        agent_value = agent_value.squeeze(0)
                    agent_result[key] = agent_value
                else:
                    agent_result[key] = value
            results[agent] = agent_result
        
        return results
    
    def _forward_independent(
        self,
        obs_dict: Dict[str, torch.Tensor],
        state_dict: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """独立模式的前向推理 - 分别调用每个 agent 的 policy。"""
        results = {}
        
        # Handle action_mask
        action_mask = kwargs.get("action_mask", {})
        forward_kwargs = {k: v for k, v in kwargs.items() if k != "action_mask"}
        
        for agent, obs in obs_dict.items():
            policy = self.get_policy(agent)
            state = state_dict.get(agent)
            
            # Get agent-specific action_mask
            agent_kwargs = forward_kwargs.copy()
            if agent in action_mask:
                agent_kwargs["action_mask"] = action_mask[agent]
            
            output = policy.forward(obs, state=state, **agent_kwargs)
            results[agent] = output
        
        return results
    
    # ============================================================================
    #                         Environment Interaction
    # ============================================================================
    
    def compute_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        info_dict: Optional[Dict[str, Dict]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[Dict[str, np.ndarray | int], Dict[str, Dict[str, Any]]]:
        """
        环境交互接口 - 从 numpy 观测计算动作。
        
        Args:
            obs_dict: {agent_str: obs_array} 格式的观测字典
            info_dict: {agent_str: info} 格式的 info 字典
                - 可包含 action_mask 等信息
            state_dict: {agent_str: state} 格式的 RNN 隐状态
        
        Returns:
            actions: {agent_str: action} 格式的动作字典
            outputs: {agent_str: {log_prob, ...}} 格式的额外输出
        """
        info_dict = info_dict or {}
        state_dict = state_dict or {}
        
        # Convert numpy to tensor
        obs_tensor_dict = {}
        for agent, obs in obs_dict.items():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tensor_dict[agent] = obs_tensor
        
        # Extract action_mask from info_dict
        action_mask_dict = {}
        for agent, info in info_dict.items():
            if "action_mask" in info:
                mask = info["action_mask"]
                action_mask_dict[agent] = torch.as_tensor(
                    mask, dtype=torch.bool, device=self.device
                )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                obs_tensor_dict,
                state_dict=state_dict,
                action_mask=action_mask_dict if action_mask_dict else None,
                **kwargs
            )
        
        # Convert actions to numpy
        actions = {}
        for agent, output in outputs.items():
            act = output["act"]
            if isinstance(act, torch.Tensor):
                act_np = act.cpu().numpy()
            else:
                act_np = act
            
            # Map action to valid range
            policy = self.get_policy(agent)
            act_np = policy.map_action(act_np)
            
            # Squeeze if single sample
            if isinstance(act_np, np.ndarray) and act_np.ndim > 0 and act_np.shape[0] == 1:
                act_np = act_np.squeeze(0)
            
            actions[agent] = act_np
        
        return actions, outputs
    
    # ============================================================================
    #                            Training Methods
    # ============================================================================
    
    def evaluate_actions(
        self,
        obs_dict: Dict[str, torch.Tensor],
        act_dict: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        评估给定动作的 log_prob 和 entropy。用于 PPO 等算法的训练。
        
        Args:
            obs_dict: {agent_str: obs_tensor} 观测字典
            act_dict: {agent_str: act_tensor} 动作字典
            **kwargs: 额外参数（如 action_mask）
        
        Returns:
            {agent_str: (log_prob, entropy)} 格式的评估结果
        """
        action_mask = kwargs.get("action_mask", {})
        results = {}
        
        if self.shared:
            # Shared mode: 合并计算
            results = self._evaluate_shared(obs_dict, act_dict, action_mask)
        else:
            # Independent mode: 分别计算
            for agent in obs_dict.keys():
                policy = self.get_policy(agent)
                obs = obs_dict[agent]
                act = act_dict[agent]
                
                agent_mask = action_mask.get(agent)
                if hasattr(policy, "evaluate_actions"):
                    log_prob, entropy = policy.evaluate_actions(
                        obs, act, action_mask=agent_mask
                    )
                    results[agent] = (log_prob, entropy)
                else:
                    raise NotImplementedError(
                        f"Policy {type(policy)} does not support evaluate_actions"
                    )
        
        return results
    
    def _evaluate_shared(
        self,
        obs_dict: Dict[str, torch.Tensor],
        act_dict: Dict[str, torch.Tensor],
        action_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """共享模式的动作评估。"""
        active_agents = list(obs_dict.keys())
        if not active_agents:
            return {}
        
        # Stack observations and actions
        obs_list = [obs_dict[agent] for agent in active_agents]
        act_list = [act_dict[agent] for agent in active_agents]
        
        # Get batch size
        first_obs = obs_list[0]
        batch_size = first_obs.shape[0] if first_obs.dim() > 1 else 1
        
        # Ensure consistent dims
        if first_obs.dim() == 1:
            obs_list = [o.unsqueeze(0) for o in obs_list]
            act_list = [a.unsqueeze(0) if a.dim() == 0 else a for a in act_list]
        
        # Stack: [num_agents, batch, ...]
        stacked_obs = torch.stack(obs_list, dim=0)
        stacked_act = torch.stack(act_list, dim=0)
        num_agents = len(active_agents)
        
        # Flatten
        flat_obs = stacked_obs.view(num_agents * batch_size, *stacked_obs.shape[2:])
        flat_act = stacked_act.view(num_agents * batch_size, *stacked_act.shape[2:])
        
        # Handle action_mask
        flat_mask = None
        if action_mask:
            mask_list = [action_mask.get(agent) for agent in active_agents]
            if all(m is not None for m in mask_list):
                if mask_list[0].dim() == 1:
                    mask_list = [m.unsqueeze(0) for m in mask_list]
                stacked_mask = torch.stack(mask_list, dim=0)
                flat_mask = stacked_mask.view(num_agents * batch_size, -1)
        
        # Evaluate through shared policy
        shared_policy = self._policies["shared_policy"]
        if hasattr(shared_policy, "evaluate_actions"):
            log_prob, entropy = shared_policy.evaluate_actions(
                flat_obs, flat_act, action_mask=flat_mask
            )
        else:
            raise NotImplementedError(
                f"Policy {type(shared_policy)} does not support evaluate_actions"
            )
        
        # Unflatten results
        log_prob = log_prob.view(num_agents, batch_size)
        entropy = entropy.view(num_agents, batch_size)
        
        results = {}
        for i, agent in enumerate(active_agents):
            agent_log_prob = log_prob[i]
            agent_entropy = entropy[i]
            # 如果原始是单样本，squeeze
            if first_obs.dim() == 1:
                agent_log_prob = agent_log_prob.squeeze(0)
                agent_entropy = agent_entropy.squeeze(0)
            results[agent] = (agent_log_prob, agent_entropy)
        
        return results

    def evaluate_actions_flat(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """直接处理 flatten tensor, 用于 MAPPO 训练。仅 shared=True 时使用。"""
        if not self.shared:
            raise ValueError("evaluate_actions_flat only works with shared=True")
        return self._policies["shared_policy"].evaluate_actions(obs, act, **kwargs)
