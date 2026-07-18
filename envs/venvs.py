from typing import Any, Callable, Dict, List, Literal, Sequence, Union, Optional
import numpy as np
import gymnasium as gym

from envs.workers import EnvWorker, DummyEnvWorker, SubprocEnvWorker

class BaseVectorEnv:
    """Synchronous vector wrapper for Gymnasium and PettingZoo environments.

    Workers may execute locally or in subprocesses. Observations and metadata are
    stacked across selected environment IDs. Completed environments are reset
    automatically while terminal observations, and truncation states when
    available, are retained in ``info`` for bootstrapping.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
    ):
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.num_envs = len(env_fns)
        self.is_closed = False

        self._worker_seeds: List[Optional[int]] = [None] * self.num_envs

        # Detect the environment type.
        self._is_parallel_env = self._detect_parallel_env()

        if self._is_parallel_env:

            self.agents = self.workers[0].get_env_attr("possible_agents")

            self._observation_spaces = {
                agent: self.workers[0].get_env_attr("observation_space")(agent)
                if callable(self.workers[0].get_env_attr("observation_space"))
                else self.workers[0].get_env_attr("observation_spaces")[agent]
                for agent in self.agents
            }
            self._action_spaces = {
                agent: self.workers[0].get_env_attr("action_space")(agent)
                if callable(self.workers[0].get_env_attr("action_space"))
                else self.workers[0].get_env_attr("action_spaces")[agent]
                for agent in self.agents
            }
        else:
            # Gymnasium Env.
            self.agents = None
            self._observation_space = self.workers[0].get_env_attr("observation_space")
            self._action_space = self.workers[0].get_env_attr("action_space")

    def _detect_parallel_env(self) -> bool:
        return self.workers[0].get_env_attr("possible_agents") is not None

    @property
    def is_parallel_env(self) -> bool:
        return self._is_parallel_env

    def __len__(self) -> int:
        return self.num_envs

    @property
    def observation_space(self) -> Union[gym.Space, Dict[str, gym.Space]]:
        if self._is_parallel_env:
            return self._observation_spaces
        return self._observation_space

    @property
    def action_space(self) -> Union[gym.Space, Dict[str, gym.Space]]:
        if self._is_parallel_env:
            return self._action_spaces
        return self._action_space

    def _wrap_id(self, env_id: int | List[int] | np.ndarray | None) -> List[int]:
        if env_id is None:
            return list(range(self.num_envs))
        if np.isscalar(env_id):
            return [env_id]
        return list(env_id)

    # Reset.

    def reset(
        self,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple:
        assert not self.is_closed, "Cannot reset closed VectorEnv"
        env_id = self._wrap_id(env_id)

        results = []
        for i in env_id:
            kw = dict(kwargs)
            if "seed" not in kw and self._worker_seeds[i] is not None:
                kw["seed"] = self._worker_seeds[i]
            results.append(self.workers[i].reset(**kw))
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]

        if self._is_parallel_env:
            return self._stack_dict_obs(obs_list), self._stack_dict_info(info_list)
        else:
            return self._stack_array_obs(obs_list), np.array(info_list)

    # Step.

    def step(
        self,
        actions: Union[np.ndarray, Dict[str, np.ndarray]],
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> tuple:
        """Step selected environments and auto-reset completed episodes.

        Parallel-environment actions are supplied as ``agent -> batch`` arrays;
        Gymnasium actions use one batched array. Returned observations belong to
        reset environments after termination, while ``info['final_obs']`` keeps
        the terminal observation.
        """

        assert not self.is_closed, "Cannot step closed VectorEnv"
        env_id = self._wrap_id(env_id)

        if self._is_parallel_env:

            results = []
            for j, i in enumerate(env_id):
                action_dict = {agent: actions[agent][j] for agent in self.agents}
                obs, rew, term, trunc, info = self.workers[i].step(action_dict)

                first_agent = self.agents[0]
                if term[first_agent] or trunc[first_agent]:
                    # Store the final observation in info.
                    for agent in self.agents:
                        info[agent]["final_obs"] = obs[agent]

                    # Store final_state for truncation bootstrapping.
                    if trunc[first_agent]:
                        try:
                            state_method = self.workers[i].get_env_attr("state")
                            if state_method is not None:
                                final_state = state_method() if callable(state_method) else state_method
                                for agent in self.agents:
                                    info[agent]["final_state"] = final_state
                        except Exception:
                            pass

                    # Reset the environment and replace the observation.
                    reset_kw = {} if self._worker_seeds[i] is None else {"seed": self._worker_seeds[i]}
                    obs, reset_info = self.workers[i].reset(**reset_kw)
                    for agent in self.agents:
                        info[agent].update(reset_info.get(agent, {}))

                results.append((obs, rew, term, trunc, info))
            return self._unpack_parallel_results(results, env_id)
        else:
            # Gymnasium Env.
            results = []
            assert len(actions) == len(env_id)
            for j, i in enumerate(env_id):
                obs, rew, term, trunc, info = self.workers[i].step(actions[j])

                # Autoreset.
                if term or trunc:
                    info["final_obs"] = obs
                    reset_kw = {} if self._worker_seeds[i] is None else {"seed": self._worker_seeds[i]}
                    obs, reset_info = self.workers[i].reset(**reset_kw)
                    info.update(reset_info)

                results.append((obs, rew, term, trunc, info))
            return self._unpack_gym_results(results, env_id)

    # Helper Methods.

    def _stack_array_obs(self, obs_list: List[np.ndarray]) -> np.ndarray:
        try:
            return np.stack(obs_list)
        except ValueError:
            return np.array(obs_list, dtype=object)

    def _stack_dict_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            agent: np.stack([obs[agent] for obs in obs_list])
            for agent in self.agents
        }

    def _stack_dict_info(self, info_list: List[Dict[str, Dict]]) -> Dict[str, np.ndarray]:
        """Stack PettingZoo info dictionaries by agent."""

        return {
            agent: np.array([info[agent] for info in info_list])
            for agent in self.agents
        }

    def _unpack_gym_results(self, results: List[tuple], env_id: List[int]) -> tuple:
        for j, i in enumerate(env_id):
            results[j][-1]["env_id"] = i

        obs_list, rew_list, term_list, trunc_list, info_list = zip(*results)
        return (
            self._stack_array_obs(obs_list),
            np.array(rew_list),
            np.array(term_list),
            np.array(trunc_list),
            np.array(info_list),
        )

    def _unpack_parallel_results(self, results: List[tuple], env_id: List[int]) -> tuple:

        # results[j] = (obs_dict, rew_dict, term_dict, trunc_dict, info_dict).
        for j, i in enumerate(env_id):
            for agent in self.agents:
                results[j][-1][agent]["env_id"] = i

        obs_list = [r[0] for r in results]
        rew_list = [r[1] for r in results]
        term_list = [r[2] for r in results]
        trunc_list = [r[3] for r in results]
        info_list = [r[4] for r in results]

        return (
            self._stack_dict_obs(obs_list),
            {agent: np.array([r[agent] for r in rew_list]) for agent in self.agents},
            {agent: np.array([r[agent] for r in term_list]) for agent in self.agents},
            {agent: np.array([r[agent] for r in trunc_list]) for agent in self.agents},
            self._stack_dict_info(info_list),
        )

    # Other Methods.

    def seed(self, seed: int | List[int] | None = None) -> List[Any]:
        assert not self.is_closed
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
        self._worker_seeds = seeds
        return [w.seed(s) for w, s in zip(self.workers, seeds)]

    def get_env_attr(
        self,
        key: str,
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> List[Any]:
        assert not self.is_closed
        env_id = self._wrap_id(env_id)
        return [self.workers[i].get_env_attr(key) for i in env_id]

    def call_env_method(
        self,
        method_name: str,
        *args,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> List[Any]:
        assert not self.is_closed
        env_id = self._wrap_id(env_id)
        results = []
        for i in env_id:
            method = self.workers[i].get_env_attr(method_name)
            if method is not None and callable(method):
                results.append(method(*args, **kwargs))
            else:
                results.append(None)
        return results

    def set_env_attr(
        self,
        key: str,
        value: Any,
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> None:
        assert not self.is_closed
        env_id = self._wrap_id(env_id)
        for i in env_id:
            self.workers[i].set_env_attr(key, value)

    def render(self, **kwargs) -> List[Any]:
        assert not self.is_closed
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        if self.is_closed:
            return
        for w in self.workers:
            w.close()
        self.is_closed = True

class DummyVectorEnv(BaseVectorEnv):
    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]):
        super().__init__(env_fns, DummyEnvWorker)

class SubprocVectorEnv(BaseVectorEnv):
    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        context: Literal["fork", "spawn"] | None = None,
    ):
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, context=context)

        super().__init__(env_fns, worker_fn)

    def reset(
        self,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple:
        assert not self.is_closed, "Cannot reset closed VectorEnv"
        env_id = self._wrap_id(env_id)

        for i in env_id:
            kw = dict(kwargs)
            if "seed" not in kw and self._worker_seeds[i] is not None:
                kw["seed"] = self._worker_seeds[i]
            self.workers[i].send_reset(**kw)

        results = [self.workers[i].recv_reset() for i in env_id]
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]

        if self._is_parallel_env:
            return self._stack_dict_obs(obs_list), self._stack_dict_info(info_list)
        else:
            return self._stack_array_obs(obs_list), np.array(info_list)

    def step(
        self,
        actions: Union[np.ndarray, Dict[str, np.ndarray]],
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> tuple:
        assert not self.is_closed, "Cannot step closed VectorEnv"
        env_id = self._wrap_id(env_id)

        if self._is_parallel_env:
            return self._parallel_step_parallel(actions, env_id)
        else:
            return self._parallel_step_gym(actions, env_id)

    def _parallel_step_parallel(
        self,
        actions: Dict[str, np.ndarray],
        env_id: List[int],
    ) -> tuple:
        first_agent = self.agents[0]

        # Phase 1: scatter step.
        for j, i in enumerate(env_id):
            action_dict = {agent: actions[agent][j] for agent in self.agents}
            self.workers[i].send_step(action_dict)

        # Phase 2: gather step.
        raw = [self.workers[i].recv_step() for _, i in enumerate(env_id)]

        done_pairs = []   # (result_idx, worker_idx).
        trunc_pairs = []
        results = []
        for j, i in enumerate(env_id):
            obs, rew, term, trunc, info = raw[j]
            if term[first_agent] or trunc[first_agent]:
                for agent in self.agents:
                    info[agent]["final_obs"] = obs[agent]
                done_pairs.append((j, i))
                if trunc[first_agent]:
                    trunc_pairs.append((j, i))
            results.append([obs, rew, term, trunc, info])

        for _, i in trunc_pairs:
            self.workers[i].send_call_method("state")
        for j_idx, i in trunc_pairs:
            try:
                final_state = self.workers[i].recv_call_method()
                if final_state is not None:
                    for agent in self.agents:
                        results[j_idx][4][agent]["final_state"] = final_state
            except Exception:
                pass

        for _, i in done_pairs:
            reset_kw = {} if self._worker_seeds[i] is None else {"seed": self._worker_seeds[i]}
            self.workers[i].send_reset(**reset_kw)
        for j_idx, i in done_pairs:
            new_obs, reset_info = self.workers[i].recv_reset()
            results[j_idx][0] = new_obs
            for agent in self.agents:
                results[j_idx][4][agent].update(reset_info.get(agent, {}))

        return self._unpack_parallel_results([tuple(r) for r in results], env_id)

    def _parallel_step_gym(
        self,
        actions: np.ndarray,
        env_id: List[int],
    ) -> tuple:
        assert len(actions) == len(env_id)

        # Phase 1: scatter step.
        for j, i in enumerate(env_id):
            self.workers[i].send_step(actions[j])

        # Phase 2: gather step.
        raw = [self.workers[i].recv_step() for _, i in enumerate(env_id)]

        done_pairs = []
        results = []
        for j, i in enumerate(env_id):
            obs, rew, term, trunc, info = raw[j]
            if term or trunc:
                info["final_obs"] = obs
                done_pairs.append((j, i))
            results.append([obs, rew, term, trunc, info])

        for _, i in done_pairs:
            reset_kw = {} if self._worker_seeds[i] is None else {"seed": self._worker_seeds[i]}
            self.workers[i].send_reset(**reset_kw)
        for j_idx, i in done_pairs:
            new_obs, reset_info = self.workers[i].recv_reset()
            results[j_idx][0] = new_obs
            results[j_idx][4].update(reset_info)

        return self._unpack_gym_results([tuple(r) for r in results], env_id)

    def call_env_method(
        self,
        method_name: str,
        *args,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> List[Any]:
        assert not self.is_closed
        env_id = self._wrap_id(env_id)

        # scatter.
        for i in env_id:
            self.workers[i].send_call_method(method_name, *args, **kwargs)

        # gather.
        return [self.workers[i].recv_call_method() for i in env_id]

    def get_env_attr(
        self,
        key: str,
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> List[Any]:
        assert not self.is_closed
        env_id = self._wrap_id(env_id)

        # scatter.
        for i in env_id:
            self.workers[i].parent_conn.send(("getattr", key))

        # gather.
        return [self.workers[i].parent_conn.recv() for i in env_id]
