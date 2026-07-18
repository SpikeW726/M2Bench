from abc import abstractmethod
from pettingzoo import ParallelEnv
from envs.mdps.patrol_core import PatrolWorld, TickResult
from typing import Dict, Any, Optional
import gymnasium
import numpy as np

class BaseEnv(ParallelEnv):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world = PatrolWorld(config)
        self.enable_wait = config.get("enable_wait", False)
        self.init_pos: list = config.get("init_positions", [])

        self.metadata = {"render_modes": [], "name": self.__class__.__name__}
        self.possible_agents = [f"agent_{i}" for i in range(config["num_agents"])]
        self.agents = self.possible_agents[:]

    @abstractmethod
    def step(self, actions: Dict[str, int]):
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        pass

    @abstractmethod
    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        raise NotImplementedError

    @abstractmethod
    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> np.ndarray:
        pass

    @property
    def observation_spaces(self) -> Dict[str, gymnasium.spaces.Space]:
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self) -> Dict[str, gymnasium.spaces.Space]:
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    @abstractmethod
    def _build_obs(self, result: Optional[TickResult]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _build_info(self, result: Optional[TickResult]) -> Dict[str, Dict]:
        pass

    @abstractmethod
    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        pass

    def _compute_terminations(self) -> Dict[str, bool]:
        return {agent: False for agent in self.agents}

    def _compute_truncations(self) -> Dict[str, bool]:
        return {agent: False for agent in self.agents}

    @abstractmethod
    def _action_to_target(self, agent_id: int, action_idx: int) -> int:
        pass

    def _dispatch_move(self, agent_id: int, target_node: int):
        self.world.set_move_action(agent_id, target_node)

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}

class FixedStepEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

    def step(self, actions: Dict[str, int]):
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])

            if self.world.is_ready(agent_id):
                if self.enable_wait:
                    if action_idx == 0:
                        self.world.set_wait_action(agent_id)
                    else:
                        target = self._action_to_target(agent_id, action_idx)
                        self._dispatch_move(agent_id, target)
                else:
                    target = self._action_to_target(agent_id, action_idx)
                    self._dispatch_move(agent_id, target)

        result = self.world.tick(dt=1.0)

        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        truncations = self._compute_truncations()
        infos = self._build_info(result)

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        initial = self.init_pos if len(self.init_pos) == self.world.num_agents else None
        self.world.reset(initial_positions=initial, seed=seed)
        self.agents = self.possible_agents[:]

        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)

        return obs, infos

class EventDrivenEnv(BaseEnv):
    def __init__(self, config: Dict):
        super().__init__(config)

    def step(self, actions: Dict[str, int]):
        for agent_str, action_idx in actions.items():
            agent_id = int(agent_str.split('_')[1])

            if self.world.is_ready(agent_id):
                if self.enable_wait:
                    if action_idx == 0:
                        self.world.set_wait_action(agent_id)
                    else:
                        target = self._action_to_target(agent_id, action_idx)
                        self._dispatch_move(agent_id, target)
                else:
                    target = self._action_to_target(agent_id, action_idx)
                    self._dispatch_move(agent_id, target)

        result = self.world.tick_to_next_event()

        obs = self._build_obs(result)
        rewards = self._compute_rewards(result)
        terminations = self._compute_terminations()
        truncations = self._compute_truncations()
        infos = self._build_info(result)

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None):
        initial = self.init_pos if len(self.init_pos) == self.world.num_agents else None
        self.world.reset(initial_positions=initial, seed=seed)
        self.agents = self.possible_agents[:]

        obs = self._build_obs(result=None)
        infos = self._build_info(result=None)

        return obs, infos

# Joint Gymnasium Env.

# obs / action / reward.

class JointBaseEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.world = PatrolWorld(config)

        self.observation_space: gymnasium.spaces.Space = None
        self.action_space: gymnasium.spaces.Space = None

    @abstractmethod
    def step(self, action):
        pass

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.world.reset(seed=seed)
        return self._build_obs(None), self._build_info(None)

    @abstractmethod
    def _dispatch_actions(self, action):
        pass

    @abstractmethod
    def _build_obs(self, result: Optional[TickResult]) -> np.ndarray:
        pass

    @abstractmethod
    def _build_info(self, result: Optional[TickResult]) -> dict:
        pass

    @abstractmethod
    def _compute_reward(self, result: TickResult) -> float:
        pass

    def _compute_termination(self) -> bool:
        return False

    @abstractmethod
    def _compute_truncation(self) -> bool:
        pass

    def get_episode_metrics(self) -> Optional[dict]:
        m = self.world.last_episode_metrics
        if m is None:
            return None
        return {"igi": m.igi, "agi": m.agi, "iwi": m.iwi, "wi": m.wi}

class JointFixedStepEnv(JointBaseEnv):
    def __init__(self, config: Dict):
        super().__init__(config)

    def step(self, action):
        self._dispatch_actions(action)
        result = self.world.tick(dt=1.0)
        return (
            self._build_obs(result),
            self._compute_reward(result),
            self._compute_termination(),
            self._compute_truncation(),
            self._build_info(result),
        )

class JointEventDrivenEnv(JointBaseEnv):
    def __init__(self, config: Dict):
        super().__init__(config)

    def step(self, action):
        self._dispatch_actions(action)
        result = self.world.tick_to_next_event()
        return (
            self._build_obs(result),
            self._compute_reward(result),
            self._compute_termination(),
            self._compute_truncation(),
            self._build_info(result),
        )