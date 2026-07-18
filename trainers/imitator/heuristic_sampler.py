"""Collect heuristic demonstrations for actor and critic pretraining.

Saved datasets use layout ``(episodes, time, agents, features)`` and contain
actor observations, centralized critic states with agent identities, actions,
action masks, READY-step masks, rewards, padding masks, and discounted returns.
Collection can run in multiple worker processes and writes either NPZ or HDF5
data through the sampler's configured output path.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import h5py

from policies.heuritic.heuristic_base import HeuriticBasePolicy
from policies.heuritic.er import ERPolicy
from envs.mdps.base_envs import EventDrivenEnv
from configs.registry import ENV_REGISTRY, _import_class

_POLICY_REGISTRY: Dict[str, tuple] = {
    "er":           ("policies.heuritic.er",                    "ERPolicy"),
    "hpcc":         ("policies.heuritic.hpcc",                  "HPCCPolicy"),
    "random":       ("policies.heuritic.random",                "RandomPolicy"),
    "ahpa":         ("policies.heuritic.ahpa",                  "AHPAPolicy"),
    "baps":         ("policies.heuritic.baps",                  "BAPSPolicy"),
    "gbs":          ("policies.heuritic.gbs",                   "GBSPolicy"),
    "cc":           ("policies.heuritic.conscientious_cognitive","ConscientiousCognitivePolicy"),
    "cr":           ("policies.heuritic.conscientious_reactive", "ConscientiousReactivePolicy"),
    "msp":          ("policies.heuritic.msp",                   "MSPPolicy"),
    "hcr":          ("policies.heuritic.hcr",                   "HCRPolicy"),
    "sebs":         ("policies.heuritic.sebs",                  "SEBSPolicy"),
    "cbls":         ("policies.heuritic.cbls",                  "CBLSPolicy"),
    "dta_greedy":   ("policies.heuritic.dta_greedy",            "DTAGreedyPolicy"),
    "dta_ssi":      ("policies.heuritic.dta_ssi",               "DTASSIPolicy"),

    "dtassi":       ("policies.heuritic.dta_ssi",               "DTASSIPolicy"),
    "dtagreedy":    ("policies.heuritic.dta_greedy",            "DTAGreedyPolicy"),
}

def _default_policy_config_path(policy_type: str) -> str:
    return f"configs/heuristic/{policy_type.upper().replace('_', '')}.yaml"

def _create_policy(policy_type: str, num_agents: int, policy_config: dict) -> HeuriticBasePolicy:
    key = policy_type.lower()
    if key not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy_type '{policy_type}'. "
            f"Available: {list(_POLICY_REGISTRY.keys())}"
        )
    module_path, class_name = _POLICY_REGISTRY[key]
    cls = _import_class(module_path, class_name)
    return cls(num_agents, policy_config)

@dataclass
class EpisodeData:
    obs: List[np.ndarray] = field(default_factory=list)               # [T_ep, M, Obs_Dim].
    critic_states: List[np.ndarray] = field(default_factory=list)     # [T_ep, M, State_Dim+M].
    actions: List[np.ndarray] = field(default_factory=list)           # [T_ep, M, 1].
    action_masks: List[np.ndarray] = field(default_factory=list)      # [T_ep, M, Act_Dim].
    active_masks: List[np.ndarray] = field(default_factory=list)      # [T_ep, M, 1].
    rewards: List[np.ndarray] = field(default_factory=list)           # [T_ep, M].

    @property
    def length(self) -> int:
        return len(self.rewards)

def _worker_collect_episodes(
    args: Tuple
) -> List[Dict[str, np.ndarray]]:
    policy_type = "er"
    if len(args) == 8:
        num_episodes, worker_id, eps, env_config, custom_config, policy_config, env_type, policy_type = args
    elif len(args) == 7:
        num_episodes, worker_id, eps, env_config, custom_config, policy_config, env_type = args
    else:
        num_episodes, worker_id, eps, env_config, custom_config, policy_config = args
        env_type = "masup"

    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])
    env = env_cls(env_config, **custom_config)
    num_agents = env_config.get("num_agents", 3)
    policy = _create_policy(policy_type, num_agents, policy_config)

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n

    env.reset()
    state_dim = len(env.state())
    critic_state_dim = state_dim + num_agents

    episodes_data = []

    for _ in range(num_episodes):
        episode = _collect_single_episode(env, policy, num_agents, obs_dim, act_dim, eps)
        episodes_data.append(episode)

    return episodes_data

def _worker_collect_to_file(args: Tuple) -> str:
    policy_type = "er"
    if len(args) == 11:
        (num_episodes, worker_id, eps, env_config, custom_config, policy_config,
         temp_dir, chunk_size, gamma, env_type, policy_type) = args
    elif len(args) == 10:
        (num_episodes, worker_id, eps, env_config, custom_config, policy_config,
         temp_dir, chunk_size, gamma, env_type) = args
    else:
        (num_episodes, worker_id, eps, env_config, custom_config, policy_config,
         temp_dir, chunk_size, gamma) = args
        env_type = "masup"

    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])
    env = env_cls(env_config, **custom_config)
    num_agents = env_config.get("num_agents", 3)
    policy = _create_policy(policy_type, num_agents, policy_config)

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n

    env.reset()
    state_dim = len(env.state())
    critic_state_dim = state_dim + num_agents

    temp_dir_path = Path(temp_dir)
    chunk_files = []

    collected = 0
    while collected < num_episodes:

        chunk_eps = min(chunk_size, num_episodes - collected)
        episodes_data = []

        for _ in range(chunk_eps):
            episode = _collect_single_episode(env, policy, num_agents, obs_dim, act_dim, eps)
            episodes_data.append(episode)

        chunk_file = temp_dir_path / f"worker_{worker_id}_chunk_{len(chunk_files):05d}.npz"
        _save_chunk_to_file(episodes_data, chunk_file, gamma, num_agents, obs_dim, act_dim, critic_state_dim)
        chunk_files.append(str(chunk_file))

        collected += chunk_eps

        del episodes_data

    return ",".join(chunk_files)

def _save_chunk_to_file(
    episodes_data: List[Dict[str, np.ndarray]],
    save_path: Path,
    gamma: float,
    num_agents: int,
    obs_dim: int,
    act_dim: int,
    critic_state_dim: int,
) -> None:
    N = len(episodes_data)
    max_len = max(ep['obs'].shape[0] for ep in episodes_data)

    obs = np.zeros([N, max_len, num_agents, obs_dim], dtype=np.float32)
    critic_states = np.zeros([N, max_len, num_agents, critic_state_dim], dtype=np.float32)
    actions = np.zeros([N, max_len, num_agents, 1], dtype=np.int64)
    action_masks = np.zeros([N, max_len, num_agents, act_dim], dtype=np.int8)
    active_masks = np.zeros([N, max_len, num_agents, 1], dtype=np.int8)
    rewards = np.zeros([N, max_len, num_agents, 1], dtype=np.float32)
    padded_mask = np.zeros([N, max_len, 1], dtype=np.int8)
    returns = np.zeros([N, max_len, num_agents, 1], dtype=np.float32)

    for i, ep in enumerate(episodes_data):
        L = ep['obs'].shape[0]
        obs[i, :L] = ep['obs']
        critic_states[i, :L] = ep['critic_states']
        actions[i, :L] = ep['actions']
        action_masks[i, :L] = ep['action_masks']
        active_masks[i, :L] = ep.get(
            'active_masks',
            np.ones((L, num_agents, 1), dtype=np.int8),
        )
        rewards[i, :L, :, 0] = ep['rewards']
        padded_mask[i, :L, 0] = 1

        ep_returns = _compute_returns_static(ep['rewards'], gamma)
        returns[i, :L, :, 0] = ep_returns

    np.savez(save_path, obs=obs, critic_states=critic_states, actions=actions,
             action_masks=action_masks, active_masks=active_masks,
             rewards=rewards, padded_mask=padded_mask, returns=returns)

def _compute_returns_static(rewards: np.ndarray, gamma: float) -> np.ndarray:
    T, M = rewards.shape
    returns = np.zeros((T, M), dtype=np.float32)
    G = np.zeros(M, dtype=np.float32)

    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns

def _collect_single_episode(
    env: EventDrivenEnv,
    policy: HeuriticBasePolicy,
    num_agents: int,
    obs_dim: int,
    act_dim: int,
    eps: float
) -> Dict[str, np.ndarray]:
    obs_list = []
    critic_states_list = []
    actions_list = []
    action_masks_list = []
    active_masks_list = []
    rewards_list = []

    obs_rl, info = env.reset()
    policy.reset()
    done = False

    while not done:

        h_obs = env.world.get_heuristic_obs()
        global_state = env.world.get_global_state_for_heuristic()
        h_actions = policy.compute_actions(h_obs, global_state)

        masup_actions = {}
        for agent_str, neighbor_idx in h_actions.items():
            if np.random.random() < eps:
                valid_actions = env.get_valid_actions(agent_str)
                masup_actions[agent_str] = int(np.random.choice(valid_actions))
            else:
                masup_actions[agent_str] = env.convert_heuristic_action(agent_str, neighbor_idx)

        for agent_str in env.agents:
            if agent_str not in masup_actions:
                masup_actions[agent_str] = act_dim - 1

        # obs: [M, Obs_Dim].
        obs_arr = np.stack([obs_rl[f"agent_{i}"] for i in range(num_agents)], axis=0)
        obs_list.append(obs_arr)

        # critic_states: [M, State_Dim+M].
        g_state = env.state()
        critic_states = []
        for agent_id in range(num_agents):
            one_hot = np.zeros(num_agents, dtype=np.float32)
            one_hot[agent_id] = 1.0
            critic_states.append(np.concatenate([g_state, one_hot]))
        critic_states_list.append(np.stack(critic_states, axis=0))

        # actions: [M, 1].
        actions_arr = np.array([[masup_actions.get(f"agent_{i}", act_dim - 1)] for i in range(num_agents)], dtype=np.int64)
        actions_list.append(actions_arr)

        # action_masks: [M, Act_Dim].
        masks_arr = np.stack([info[f"agent_{i}"]['action_mask'] for i in range(num_agents)], axis=0).astype(np.int8)
        action_masks_list.append(masks_arr)

        # active_masks: [M, 1].
        active_arr = np.array(
            [[info[f"agent_{i}"].get('active_mask', 1)] for i in range(num_agents)],
            dtype=np.int8,
        )
        active_masks_list.append(active_arr)

        obs_rl, rewards, terms, truncs, info = env.step(masup_actions)

        rewards_arr = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(num_agents)], dtype=np.float32)
        rewards_list.append(rewards_arr)

        done = any(truncs.values()) or any(terms.values())

    return {
        'obs': np.stack(obs_list, axis=0),              # [T, M, Obs_Dim].
        'critic_states': np.stack(critic_states_list, axis=0),  # [T, M, State_Dim+M].
        'actions': np.stack(actions_list, axis=0),      # [T, M, 1].
        'action_masks': np.stack(action_masks_list, axis=0),    # [T, M, Act_Dim].
        'active_masks': np.stack(active_masks_list, axis=0),     # [T, M, 1].
        'rewards': np.stack(rewards_list, axis=0),      # [T, M].
    }

class HeuristicSampler:
    def __init__(
        self,
        policy: HeuriticBasePolicy,
        env: EventDrivenEnv,
        env_type: str = "masup",
        env_config: Optional[Dict] = None,
        custom_config: Optional[Dict] = None,
        policy_config: Optional[Dict] = None,
        policy_type: str = "er",
    ) -> None:
        self.policy = policy
        self.env = env
        self.env_type = env_type
        self.policy_type = policy_type

        self._env_config = env_config
        self._custom_config = custom_config or {}
        self._policy_config = policy_config

        self.num_agents = env.world.num_agents
        self.obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        self.act_dim = env.action_space(env.possible_agents[0]).n

        self._state_dim: Optional[int] = None

    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            self.env.reset()
            self._state_dim = len(self.env.state())
        return self._state_dim

    @property
    def critic_state_dim(self) -> int:
        return self.state_dim + self.num_agents

    def sample(
        self,
        num_episodes: int,
        save_path: str,
        gamma: float = 0.999,
        eps: float = 0.0,
        batch_size: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        if num_workers > 1:

            if self._env_config is None or self._policy_config is None:
                raise ValueError(
                    "Parallel sampling requires env_config and policy_config. "
                    "Please pass them to HeuristicSampler constructor."
                )
            self._sample_parallel(num_episodes, save_path, gamma, eps, batch_size, num_workers)
            print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path} (parallel, {num_workers} workers)")
        elif batch_size is None or batch_size >= num_episodes:

            trajectories: List[EpisodeData] = []

            for ep_idx in tqdm(range(num_episodes), desc="Sampling episodes"):
                episode_data = self._collect_episode(eps)
                trajectories.append(episode_data)

            self._pad_and_save(trajectories, save_path, gamma)
            print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path}")
        else:

            self._sample_in_batches(num_episodes, save_path, gamma, eps, batch_size)
            print(f"[HeuristicSampler] Saved {num_episodes} episodes to {save_path} (in batches)")

    def _collect_episode(self, eps: float) -> EpisodeData:
        episode = EpisodeData()

        obs_rl, info = self.env.reset()
        self.policy.reset()

        done = False

        while not done:

            h_obs = self.env.world.get_heuristic_obs()
            global_state = self.env.world.get_global_state_for_heuristic()
            h_actions = self.policy.compute_actions(h_obs, global_state)

            masup_actions = {}
            for agent_str, neighbor_idx in h_actions.items():
                if np.random.random() < eps:

                    valid_actions = self.env.get_valid_actions(agent_str)
                    masup_actions[agent_str] = int(np.random.choice(valid_actions))
                else:

                    masup_actions[agent_str] = self.env.convert_heuristic_action(agent_str, neighbor_idx)

            for agent_str in self.env.agents:
                if agent_str not in masup_actions:
                    masup_actions[agent_str] = self.act_dim - 1  # no-op.

            episode.obs.append(self._extract_obs(obs_rl))
            episode.critic_states.append(self._build_critic_states())
            episode.actions.append(self._extract_actions(masup_actions, info))
            episode.action_masks.append(self._extract_masks(info))
            episode.active_masks.append(self._extract_active_masks(info))

            obs_rl, rewards, terms, truncs, info = self.env.step(masup_actions)

            episode.rewards.append(self._extract_rewards(rewards))

            done = any(truncs.values()) or any(terms.values())

        return episode

    def _extract_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            obs_list.append(obs_dict[agent_str])
        return np.stack(obs_list, axis=0)  # [M, Obs_Dim].

    def _build_critic_states(self) -> np.ndarray:
        global_state = self.env.state()  # [State_Dim].
        critic_states = []

        for agent_id in range(self.num_agents):

            one_hot = np.zeros(self.num_agents, dtype=np.float32)
            one_hot[agent_id] = 1.0

            critic_state = np.concatenate([global_state, one_hot])
            critic_states.append(critic_state)

        return np.stack(critic_states, axis=0)  # [M, State_Dim + M].

    def _extract_actions(self, actions: Dict[str, int], info: Dict[str, Dict]) -> np.ndarray:
        action_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            action_list.append([actions.get(agent_str, self.act_dim - 1)])
        return np.array(action_list, dtype=np.int64)  # [M, 1].

    def _extract_masks(self, info: Dict[str, Dict]) -> np.ndarray:
        mask_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            mask_list.append(info[agent_str]['action_mask'])
        return np.stack(mask_list, axis=0).astype(np.int8)  # [M, Act_Dim].

    def _extract_active_masks(self, info: Dict[str, Dict]) -> np.ndarray:
        mask_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            mask_list.append([info[agent_str].get('active_mask', 1)])
        return np.array(mask_list, dtype=np.int8)  # [M, 1].

    def _extract_rewards(self, rewards: Dict[str, float]) -> np.ndarray:
        reward_list = []
        for i in range(self.num_agents):
            agent_str = f"agent_{i}"
            reward_list.append(rewards.get(agent_str, 0.0))
        return np.array(reward_list, dtype=np.float32)  # [M].

    def _compute_returns(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        T, M = rewards.shape
        returns = np.zeros((T, M), dtype=np.float32)
        G = np.zeros(M, dtype=np.float32)

        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        return returns

    def _pad_and_save(self, trajectories: List[EpisodeData], save_path: str, gamma: float) -> None:
        N = len(trajectories)
        max_len = max(ep.length for ep in trajectories)
        M = self.num_agents

        obs = np.zeros([N, max_len, M, self.obs_dim], dtype=np.float32)
        critic_states = np.zeros([N, max_len, M, self.critic_state_dim], dtype=np.float32)
        actions = np.zeros([N, max_len, M, 1], dtype=np.int64)
        action_masks = np.zeros([N, max_len, M, self.act_dim], dtype=np.int8)
        active_masks = np.zeros([N, max_len, M, 1], dtype=np.int8)
        rewards = np.zeros([N, max_len, M, 1], dtype=np.float32)
        padded_mask = np.zeros([N, max_len, 1], dtype=np.int8)
        returns = np.zeros([N, max_len, M, 1], dtype=np.float32)

        for i, ep in enumerate(trajectories):
            L = ep.length
            obs[i, :L] = np.stack(ep.obs, axis=0)
            critic_states[i, :L] = np.stack(ep.critic_states, axis=0)
            actions[i, :L] = np.stack(ep.actions, axis=0)
            action_masks[i, :L] = np.stack(ep.action_masks, axis=0)
            active_masks[i, :L] = np.stack(ep.active_masks, axis=0)

            # rewards: [L, M] -> [L, M, 1].
            ep_rewards = np.stack(ep.rewards, axis=0)  # [L, M].
            rewards[i, :L, :, 0] = ep_rewards

            padded_mask[i, :L, 0] = 1

            ep_returns = self._compute_returns(ep_rewards, gamma)  # [L, M].
            returns[i, :L, :, 0] = ep_returns

        np.savez(
            save_path,
            obs=obs,
            critic_states=critic_states,
            actions=actions,
            action_masks=action_masks,
            active_masks=active_masks,
            rewards=rewards,
            padded_mask=padded_mask,
            returns=returns
        )

        print(f"[HeuristicSampler] Data shapes:")
        print(f"  obs:           {obs.shape}")
        print(f"  critic_states: {critic_states.shape}")
        print(f"  actions:       {actions.shape}")
        print(f"  action_masks:  {action_masks.shape}")
        print(f"  active_masks:  {active_masks.shape}")
        print(f"  rewards:       {rewards.shape}")
        print(f"  padded_mask:   {padded_mask.shape}")
        print(f"  returns:       {returns.shape}")
        print(f"  max_episode_len: {max_len}")

    def _sample_parallel(
        self,
        num_episodes: int,
        save_path: str,
        gamma: float,
        eps: float,
        batch_size: Optional[int],
        num_workers: int
    ) -> None:
        import time

        temp_dir = Path(save_path).parent / ".temp_parallel"
        temp_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = min(500, max(100, num_episodes // (num_workers * 10)))

        episodes_per_worker = num_episodes // num_workers
        remainder = num_episodes % num_workers

        worker_args = []
        for worker_id in range(num_workers):
            n_eps = episodes_per_worker + (1 if worker_id < remainder else 0)
            if n_eps > 0:
                worker_args.append((
                    n_eps,
                    worker_id,
                    eps,
                    self._env_config,
                    self._custom_config,
                    self._policy_config,
                    str(temp_dir),
                    chunk_size,
                    gamma,
                    self.env_type,
                    self.policy_type,
                ))

        print(f"[HeuristicSampler] Starting parallel sampling with {num_workers} workers...")
        print(f"[HeuristicSampler] Episodes per worker: {[a[0] for a in worker_args]}")
        print(f"[HeuristicSampler] Chunk size: {chunk_size} episodes/chunk")

        ctx = mp.get_context('spawn')

        all_chunk_files = []
        start_time = time.time()

        with ctx.Pool(processes=num_workers) as pool:

            results = list(tqdm(
                pool.imap_unordered(_worker_collect_to_file, worker_args),
                total=len(worker_args),
                desc=f"Workers (each ~{episodes_per_worker} eps)"
            ))

            for chunk_files_str in results:
                if chunk_files_str:
                    all_chunk_files.extend(chunk_files_str.split(","))

        elapsed = time.time() - start_time
        print(f"[HeuristicSampler] Collected {num_episodes} episodes in {elapsed:.1f}s ({num_episodes/elapsed:.1f} eps/s)")
        print(f"[HeuristicSampler] Generated {len(all_chunk_files)} chunk files, merging...")

        chunk_paths = [Path(f) for f in all_chunk_files]

        actual_max_lens = []
        for chunk_path in chunk_paths:
            with np.load(chunk_path, mmap_mode='r') as data:
                actual_max_lens.append(data['obs'].shape[1])
        global_max_len = max(actual_max_lens)

        print(f"[HeuristicSampler] Episode lengths: min={min(actual_max_lens)}, max={global_max_len}, mean={np.mean(actual_max_lens):.1f}")

        if save_path.endswith('.h5') or save_path.endswith('.hdf5'):
            self._merge_batches_to_hdf5(chunk_paths, save_path, global_max_len)
        else:
            self._merge_batches(chunk_paths, save_path, global_max_len)

        for chunk_path in chunk_paths:
            chunk_path.unlink()

        try:
            temp_dir.rmdir()
        except OSError:
            pass

    def _sample_in_batches(self, num_episodes: int, save_path: str, gamma: float, eps: float, batch_size: int) -> None:
        import tempfile
        import os

        temp_dir = Path(save_path).parent / ".temp_batches"
        temp_dir.mkdir(parents=True, exist_ok=True)

        batch_paths = []
        all_max_lens = []

        num_batches = (num_episodes + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_episodes)
            batch_episodes = end_idx - start_idx

            trajectories: List[EpisodeData] = []

            for ep_idx in tqdm(range(batch_episodes), desc=f"Batch {batch_idx+1}/{num_batches}"):
                episode_data = self._collect_episode(eps)
                trajectories.append(episode_data)

            batch_path = temp_dir / f"batch_{batch_idx:05d}.npz"
            self._pad_and_save(trajectories, str(batch_path), gamma)
            batch_paths.append(batch_path)

            max_len = max(ep.length for ep in trajectories)
            all_max_lens.append(max_len)

            del trajectories

        actual_max_lens = []
        for batch_path in batch_paths:
            with np.load(batch_path, mmap_mode='r') as batch_data:
                actual_max_lens.append(batch_data['obs'].shape[1])
        global_max_len = max(actual_max_lens)

        print(f"[HeuristicSampler] Batch max lengths: min={min(actual_max_lens)}, max={global_max_len}, mean={np.mean(actual_max_lens):.1f}")

        if save_path.endswith('.h5') or save_path.endswith('.hdf5'):
            self._merge_batches_to_hdf5(batch_paths, save_path, global_max_len)
        else:
            self._merge_batches(batch_paths, save_path, global_max_len)

        for batch_path in batch_paths:
            batch_path.unlink()
        temp_dir.rmdir()

    def _merge_batches(self, batch_paths: List[Path], save_path: str, max_len: int) -> None:
        with np.load(batch_paths[0], mmap_mode='r') as first_batch:
            M = first_batch['obs'].shape[2]
            obs_dim = first_batch['obs'].shape[3]
            critic_state_dim = first_batch['critic_states'].shape[3]
            act_dim = first_batch['action_masks'].shape[3]

        total_episodes = 0
        for batch_path in batch_paths:
            with np.load(batch_path, mmap_mode='r') as batch_data:
                total_episodes += batch_data['obs'].shape[0]

        obs = np.zeros([total_episodes, max_len, M, obs_dim], dtype=np.float32)
        critic_states = np.zeros([total_episodes, max_len, M, critic_state_dim], dtype=np.float32)
        actions = np.zeros([total_episodes, max_len, M, 1], dtype=np.int64)
        action_masks = np.zeros([total_episodes, max_len, M, act_dim], dtype=np.int8)
        active_masks = np.zeros([total_episodes, max_len, M, 1], dtype=np.int8)
        rewards = np.zeros([total_episodes, max_len, M, 1], dtype=np.float32)
        padded_mask = np.zeros([total_episodes, max_len, 1], dtype=np.int8)
        returns = np.zeros([total_episodes, max_len, M, 1], dtype=np.float32)

        offset = 0
        for batch_path in tqdm(batch_paths, desc="Merging batches"):
            with np.load(batch_path, mmap_mode='r') as batch_data:
                N_batch = batch_data['obs'].shape[0]
                T_batch = batch_data['obs'].shape[1]

                obs[offset:offset+N_batch, :T_batch] = batch_data['obs']
                critic_states[offset:offset+N_batch, :T_batch] = batch_data['critic_states']
                actions[offset:offset+N_batch, :T_batch] = batch_data['actions']
                action_masks[offset:offset+N_batch, :T_batch] = batch_data['action_masks']
                if 'active_masks' in batch_data:
                    active_masks[offset:offset+N_batch, :T_batch] = batch_data['active_masks']
                else:
                    active_masks[offset:offset+N_batch, :T_batch] = 1
                rewards[offset:offset+N_batch, :T_batch] = batch_data['rewards']
                padded_mask[offset:offset+N_batch, :T_batch] = batch_data['padded_mask']
                returns[offset:offset+N_batch, :T_batch] = batch_data['returns']

                offset += N_batch

        np.savez(
            save_path,
            obs=obs,
            critic_states=critic_states,
            actions=actions,
            action_masks=action_masks,
            active_masks=active_masks,
            rewards=rewards,
            padded_mask=padded_mask,
            returns=returns
        )

        print(f"[HeuristicSampler] Merged data shapes:")
        print(f"  obs:           {obs.shape}")
        print(f"  critic_states: {critic_states.shape}")
        print(f"  actions:       {actions.shape}")
        print(f"  action_masks:  {action_masks.shape}")
        print(f"  active_masks:  {active_masks.shape}")
        print(f"  rewards:       {rewards.shape}")
        print(f"  padded_mask:   {padded_mask.shape}")
        print(f"  returns:       {returns.shape}")
        print(f"  max_episode_len: {max_len}")

    def _merge_batches_to_hdf5(self, batch_paths: List[Path], save_path: str, max_len: int) -> None:
        with np.load(batch_paths[0], mmap_mode='r') as first_batch:
            M = first_batch['obs'].shape[2]
            obs_dim = first_batch['obs'].shape[3]
            critic_state_dim = first_batch['critic_states'].shape[3]
            act_dim = first_batch['action_masks'].shape[3]

        total_episodes = 0
        for batch_path in batch_paths:
            with np.load(batch_path, mmap_mode='r') as data:
                total_episodes += data['obs'].shape[0]

        with h5py.File(save_path, 'w') as hf:

            chunk_size = min(512, total_episodes)
            hf.create_dataset('obs', shape=(total_episodes, max_len, M, obs_dim),
                              dtype='float32', chunks=(chunk_size, max_len, M, obs_dim))
            hf.create_dataset('critic_states', shape=(total_episodes, max_len, M, critic_state_dim),
                              dtype='float32', chunks=(chunk_size, max_len, M, critic_state_dim))
            hf.create_dataset('actions', shape=(total_episodes, max_len, M, 1),
                              dtype='int64', chunks=(chunk_size, max_len, M, 1))
            hf.create_dataset('action_masks', shape=(total_episodes, max_len, M, act_dim),
                              dtype='int8', chunks=(chunk_size, max_len, M, act_dim))
            hf.create_dataset('active_masks', shape=(total_episodes, max_len, M, 1),
                              dtype='int8', chunks=(chunk_size, max_len, M, 1))
            hf.create_dataset('rewards', shape=(total_episodes, max_len, M, 1),
                              dtype='float32', chunks=(chunk_size, max_len, M, 1))
            hf.create_dataset('padded_mask', shape=(total_episodes, max_len, 1),
                              dtype='int8', chunks=(chunk_size, max_len, 1))
            hf.create_dataset('returns', shape=(total_episodes, max_len, M, 1),
                              dtype='float32', chunks=(chunk_size, max_len, M, 1))

            offset = 0
            for batch_path in tqdm(batch_paths, desc="Merging to HDF5"):
                with np.load(batch_path, mmap_mode='r') as batch_data:
                    N_batch = batch_data['obs'].shape[0]
                    T_batch = batch_data['obs'].shape[1]

                    hf['obs'][offset:offset+N_batch, :T_batch] = batch_data['obs']
                    hf['critic_states'][offset:offset+N_batch, :T_batch] = batch_data['critic_states']
                    hf['actions'][offset:offset+N_batch, :T_batch] = batch_data['actions']
                    hf['action_masks'][offset:offset+N_batch, :T_batch] = batch_data['action_masks']
                    if 'active_masks' in batch_data:
                        hf['active_masks'][offset:offset+N_batch, :T_batch] = batch_data['active_masks']
                    else:
                        hf['active_masks'][offset:offset+N_batch, :T_batch] = 1
                    hf['rewards'][offset:offset+N_batch, :T_batch] = batch_data['rewards']
                    hf['padded_mask'][offset:offset+N_batch, :T_batch] = batch_data['padded_mask']
                    hf['returns'][offset:offset+N_batch, :T_batch] = batch_data['returns']

                    offset += N_batch

        print(f"[HeuristicSampler] Saved HDF5: {save_path}")
        print(f"  shape: ({total_episodes}, {max_len}, {M}, *)")
        print(f"  datasets: obs, critic_states, actions, action_masks, active_masks, rewards, padded_mask, returns")

if __name__ == "__main__":
    import os
    import argparse

    os.chdir(_project_root)

    from configs.registry import load_env_config, _env_config_to_dicts

    parser = argparse.ArgumentParser(description="Heuristic Sampler for Actor-Critic Pre-training")
    parser.add_argument("--num_episodes", type=int, default=50000, help="Number of episodes to collect")
    parser.add_argument("--save_path", type=str, default="dataset/samples.h5", help="Output path (.npz/.h5)")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--eps", type=float, default=0.0, help="Epsilon-greedy exploration probability")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size used to limit memory usage")
    parser.add_argument("--num_workers", type=int, default=1, help="Worker count; values above 1 enable multiprocessing")
    parser.add_argument("--policy_type", type=str, default="er",
                        help=f"Heuristic policy type; available: {list(_POLICY_REGISTRY.keys())}")
    parser.add_argument("--policy_config", type=str, default=None,
                        help="Policy YAML (default: configs/heuristic/<TYPE>.yaml inferred from policy_type)")
    parser.add_argument("--env_config", type=str, default="configs/eval/masup/masup_tsp12.yaml",
                        help="Environment configuration YAML (experiment or standalone evaluation YAML)")
    parser.add_argument("--env_type", type=str, default="masup",
                        help="Environment type such as masup or masup_gnn; must be registered in ENV_REGISTRY")
    args = parser.parse_args()

    if args.policy_config is None:
        args.policy_config = _default_policy_config_path(args.policy_type)

    # Configuration.
    with open(args.policy_config, 'r', encoding='utf-8') as f:
        policy_config = yaml.safe_load(f)

    env_cfg = load_env_config(args.env_config)
    env_config, custom_config = _env_config_to_dicts(env_cfg)

    num_agents = env_config.get("num_agents", 3)
    policy = _create_policy(args.policy_type, num_agents, policy_config)

    _entry = ENV_REGISTRY[args.env_type]
    _env_cls = _import_class(_entry["module"], _entry["class_name"])
    env = _env_cls(env_config, **custom_config)

    _agent0 = env.possible_agents[0]
    env.reset()
    _obs_dim = env.observation_space(_agent0).shape[0]
    _act_dim = env.action_space(_agent0).n
    _state_dim = len(env.state())
    _critic_state_dim = _state_dim + env.world.num_agents
    print(f"[HeuristicSampler] policy_type={args.policy_type}, env_type={args.env_type}")
    print(f"  obs_dim={_obs_dim}, action_dim={_act_dim}, critic_state_dim={_critic_state_dim}")

    sampler = HeuristicSampler(
        policy=policy,
        env=env,
        env_type=args.env_type,
        env_config=env_config,
        custom_config=custom_config,
        policy_config=policy_config,
        policy_type=args.policy_type,
    )

    sampler.sample(
        num_episodes=args.num_episodes,
        save_path=args.save_path,
        gamma=args.gamma,
        eps=args.eps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # python trainers/imitator/heuristic_sampler.py --num_episodes 50000 --num_workers 1.
    # python trainers/imitator/heuristic_sampler.py --num_episodes 50000 --num_workers 4 --policy_type ahpa --env_config configs/eval/masup/masup_island.yaml.
    # python trainers/imitator/heuristic_sampler.py --num_episodes 5000 --num_workers 4 --policy_type ahpa --env_config configs/eval/masup/masup_cumberland.yaml --gamma 0.999 --save_path /root/autodl-tmp/dataset/cumberland/ahpa_sample.h5.
