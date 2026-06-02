"""
Paper Implementation: Reinforcement Learning for Multi-agent Patrol Policy
Authors: Zhaohui Hu, Dongbin Zhao
Year: 2010
Venue: IEEE International Conference on Cognitive Informatics (ICCI)
Link: https://ieeexplore.ieee.org/document/5599681

Description:
    This script implements the NEP(short for Node-Edge Position) mdp design 
    described in Section 3.C of the paper.
"""

from typing import Dict, Optional
import numpy as np
import math
import random
import gymnasium
from gymnasium.spaces import  Discrete

from envs.mdps.patrol_core import TickResult
from envs.mdps.base_envs import JointEventDrivenEnv
from utils.comb_utils import *

class NEPEnv(JointEventDrivenEnv):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

        self.episode_len = config["episode_len"]
        self.truncate_by_time = kwargs.get("truncate_by_time", True)
        self.init_pos = config.get("init_positions", [])
        self._current_decision_agent: int = -1  # 当前决策的 agent id
        self._last_obs = None
        # 评估时 reset(seed=...) 启用；训练不传 seed 则随机选决策智能体
        self._decision_rng: Optional[random.Random] = None

        self.obs_size = 1 # 由 num_agent 维映射到 1 维
        edge_combinations_count = math.comb(self.world.num_edges, self.world.num_agents-1)
        max_combination = self.world.num_nodes * edge_combinations_count
        
        self.observation_space = Discrete(max_combination+2, start=-2) # -2: 有智能体未出发; -1: 多智能体在同一条边上
        self.action_space = Discrete(self.world.max_neighbors)
    
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._decision_rng = random.Random(seed)
        else:
            self._decision_rng = None
        gymnasium.Env.reset(self, seed=seed)
        self.world.reset(initial_positions=self.init_pos if self.init_pos else None, seed=seed)
        return self._build_obs(None), self._build_info(None)

    def _pick_decision_agent(self, ready_agents: list) -> int:
        if self._decision_rng is not None:
            return self._decision_rng.choice(ready_agents)
        return random.choice(ready_agents)

    def _dispatch_actions(self, action: int):
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"action 必须是 int, 当前类型为 {type(action).__name__}。")

        cur_position = self.world.agents[self._current_decision_agent].position
        neighbors = self.world.get_neighbors(cur_position)
        if action < 0 or action >= len(neighbors):
            raise ValueError(
                f"action 越界: action={action}, 可选范围=[0, {len(neighbors) - 1}], "
                f"current_position={cur_position}。"
            )
        target = neighbors[action]
        self.world.set_move_action(self._current_decision_agent, target)

    def _build_obs(self, result: Optional[TickResult]) -> int:
        ready_agents = self.world.get_ready_agents()
        self._current_decision_agent = self._pick_decision_agent(ready_agents)
        cur_position = self.world.agents[self._current_decision_agent].position

        edge_combinations_count = math.comb(self.world.num_edges, self.world.num_agents-1)
        x = cur_position * edge_combinations_count

        n = self.world.num_agents
        m = self.world.num_edges

        edges = []
        for a in range(n):
            if a != self._current_decision_agent:
                dst = self.world.agents[a].target_node
                if dst == -1:
                    self._last_obs = -2
                    return -2
                src = self.world.agents[a].last_position
                edges.append(self.world.graph.get_edge_index(src, dst)+1)
        edges = sorted(edges)

        if len(edges) != len(set(edges)):
            obs = -1
        else:
            y = compute_edge_comb_index(m, edges)
            obs = x + y

        self._last_obs = obs
        return obs

    def _build_info(self, result: Optional[TickResult]) -> dict:
        info = {}

        action_mask = np.zeros(self.world.max_neighbors)
        current_pos = self.world.get_position(self._current_decision_agent)
        neighbors = self.world.graph.get_neighbors(current_pos)
        action_mask[:len(neighbors)] = True
        info["action_mask"] = action_mask

        if result is not None:
            info["active_mask"] = 1 if self.world.is_ready(self._current_decision_agent) else 0
        return info

    def _compute_reward(self, result: TickResult) -> float:
        if self._last_obs in (-2, -1):
            return 0.0
        return result.raw_rewards.get(self._current_decision_agent, 0.0)

    def _compute_truncation(self) -> bool:
        if self.truncate_by_time:
            return self.world.current_time >= (self.episode_len - 1e-9)
        return self.world.step_count >= self.episode_len