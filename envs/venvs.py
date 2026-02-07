"""向量化环境：并行运行多个环境实例，支持 Gymnasium Env 和 PettingZoo ParallelEnv。"""

from typing import Any, Callable, Dict, List, Literal, Sequence, Union
import numpy as np
import gymnasium as gym

from envs.workers import EnvWorker, DummyEnvWorker, SubprocEnvWorker


class BaseVectorEnv:
    """
    向量化环境基类，同时支持 Gymnasium Env 和 PettingZoo ParallelEnv。
    
    根据环境类型自动检测并切换数据格式：
    - Gymnasium Env: obs/actions 为 np.ndarray
    - ParallelEnv: obs/actions 为 Dict[str, np.ndarray]
    
    Args:
        env_fns: 环境创建函数列表
        worker_fn: worker 创建函数
    """
    
    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
    ):
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.num_envs = len(env_fns)
        self.is_closed = False
        
        # 检测环境类型
        self._is_parallel_env = self._detect_parallel_env()
        
        if self._is_parallel_env:
            # ParallelEnv: 获取 agents 列表和 space 字典
            self.agents = self.workers[0].get_env_attr("possible_agents")
            # observation_space/action_space 是方法，需要调用
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
            # Gymnasium Env
            self.agents = None
            self._observation_space = self.workers[0].get_env_attr("observation_space")
            self._action_space = self.workers[0].get_env_attr("action_space")
    
    def _detect_parallel_env(self) -> bool:
        """检测是否为 PettingZoo ParallelEnv。"""
        return self.workers[0].get_env_attr("possible_agents") is not None
    
    @property
    def is_parallel_env(self) -> bool:
        return self._is_parallel_env
    
    def __len__(self) -> int:
        return self.num_envs
    
    @property
    def observation_space(self) -> Union[gym.Space, Dict[str, gym.Space]]:
        """Gymnasium Env 返回单个 Space，ParallelEnv 返回 Dict。"""
        if self._is_parallel_env:
            return self._observation_spaces
        return self._observation_space
    
    @property
    def action_space(self) -> Union[gym.Space, Dict[str, gym.Space]]:
        """Gymnasium Env 返回单个 Space，ParallelEnv 返回 Dict。"""
        if self._is_parallel_env:
            return self._action_spaces
        return self._action_space
    
    def _wrap_id(self, env_id: int | List[int] | np.ndarray | None) -> List[int]:
        """将 env_id 转换为列表格式。"""
        if env_id is None:
            return list(range(self.num_envs))
        if np.isscalar(env_id):
            return [env_id]
        return list(env_id)
    
    # =========================================================================
    #                               Reset
    # =========================================================================
    
    def reset(
        self,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple:
        """
        重置指定环境。
        
        Returns:
            Gymnasium Env: (obs: ndarray, info: ndarray)
            ParallelEnv: (obs: Dict[str, ndarray], info: Dict[str, ndarray])
        """
        assert not self.is_closed, "Cannot reset closed VectorEnv"
        env_id = self._wrap_id(env_id)
        
        results = [self.workers[i].reset(**kwargs) for i in env_id]
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        
        if self._is_parallel_env:
            return self._stack_dict_obs(obs_list), self._stack_dict_info(info_list)
        else:
            return self._stack_array_obs(obs_list), np.array(info_list)
    
    # =========================================================================
    #                               Step
    # =========================================================================
    
    def step(
        self,
        actions: Union[np.ndarray, Dict[str, np.ndarray]],
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> tuple:
        """
        执行动作。
        
        Args:
            actions: 
                Gymnasium Env: (num_envs,) 或 (num_envs, act_dim) 的 ndarray
                ParallelEnv: Dict[str, ndarray]，每个 agent 的动作数组，shape (num_envs,)
            env_id: 要执行的环境索引，None 表示全部
        
        Returns:
            Gymnasium Env: (obs, rew, term, trunc, info) - 都是 ndarray
            ParallelEnv: (obs, rew, term, trunc, info) - 都是 Dict[str, ndarray]
        """
        assert not self.is_closed, "Cannot step closed VectorEnv"
        env_id = self._wrap_id(env_id)
        
        if self._is_parallel_env:
            # ParallelEnv: 将 Dict[str, ndarray] 转为每个环境的 Dict[str, scalar]
            results = []
            for j, i in enumerate(env_id):
                action_dict = {agent: actions[agent][j] for agent in self.agents}
                obs, rew, term, trunc, info = self.workers[i].step(action_dict)
                
                # Autoreset: 任意 agent done 则 reset 整个环境
                first_agent = self.agents[0]
                if term[first_agent] or trunc[first_agent]:
                    # 保存 final obs 到 info
                    for agent in self.agents:
                        info[agent]["final_obs"] = obs[agent]
                    
                    # 保存 final_state（用于 truncation 的 value bootstrap）
                    if trunc[first_agent]:
                        try:
                            state_method = self.workers[i].get_env_attr("state")
                            if state_method is not None:
                                final_state = state_method() if callable(state_method) else state_method
                                for agent in self.agents:
                                    info[agent]["final_state"] = final_state
                        except Exception:
                            pass
                    
                    # Reset 环境，用新 obs 替换
                    obs, reset_info = self.workers[i].reset()
                    for agent in self.agents:
                        info[agent].update(reset_info.get(agent, {}))
                
                results.append((obs, rew, term, trunc, info))
            return self._unpack_parallel_results(results, env_id)
        else:
            # Gymnasium Env
            results = []
            assert len(actions) == len(env_id)
            for j, i in enumerate(env_id):
                obs, rew, term, trunc, info = self.workers[i].step(actions[j])
                
                # Autoreset
                if term or trunc:
                    info["final_obs"] = obs
                    obs, reset_info = self.workers[i].reset()
                    info.update(reset_info)
                
                results.append((obs, rew, term, trunc, info))
            return self._unpack_gym_results(results, env_id)
    
    # =========================================================================
    #                          Helper Methods
    # =========================================================================
    
    def _stack_array_obs(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Stack Gymnasium 观测。"""
        try:
            return np.stack(obs_list)
        except ValueError:
            return np.array(obs_list, dtype=object)
    
    def _stack_dict_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Stack ParallelEnv 观测为 Dict[str, ndarray(num_envs, *obs_shape)]。"""
        return {
            agent: np.stack([obs[agent] for obs in obs_list])
            for agent in self.agents
        }
    
    def _stack_dict_info(self, info_list: List[Dict[str, Dict]]) -> Dict[str, np.ndarray]:
        """Stack ParallelEnv info。"""
        return {
            agent: np.array([info[agent] for info in info_list])
            for agent in self.agents
        }
    
    def _unpack_gym_results(self, results: List[tuple], env_id: List[int]) -> tuple:
        """解包 Gymnasium step 结果。"""
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
        """解包 ParallelEnv step 结果为 Dict 格式。"""
        # results[j] = (obs_dict, rew_dict, term_dict, trunc_dict, info_dict)
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
    
    # =========================================================================
    #                          Other Methods
    # =========================================================================
    
    def seed(self, seed: int | List[int] | None = None) -> List[Any]:
        """设置随机种子。"""
        assert not self.is_closed
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = seed
        return [w.seed(s) for w, s in zip(self.workers, seeds)]
    
    def get_env_attr(
        self,
        key: str,
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> List[Any]:
        """获取指定环境的属性。"""
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
        """
        在环境中调用方法并返回结果（默认串行实现，DummyVectorEnv 用）。
        SubprocVectorEnv 覆写为 scatter-gather 并行版本，且只传回返回值而非 bound method。
        """
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
        """设置指定环境的属性。"""
        assert not self.is_closed
        env_id = self._wrap_id(env_id)
        for i in env_id:
            self.workers[i].set_env_attr(key, value)
    
    def render(self, **kwargs) -> List[Any]:
        """渲染所有环境。"""
        assert not self.is_closed
        return [w.render(**kwargs) for w in self.workers]
    
    def close(self) -> None:
        """关闭所有环境。"""
        if self.is_closed:
            return
        for w in self.workers:
            w.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """顺序执行的向量化环境（单进程），用于调试。"""
    
    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]):
        super().__init__(env_fns, DummyEnvWorker)


class SubprocVectorEnv(BaseVectorEnv):
    """
    多进程向量化环境，每个环境在独立子进程中运行。
    使用 scatter-gather 模式实现真正的并行 step/reset。
    
    Args:
        env_fns: 环境创建函数列表
        context: multiprocessing context，可选 "fork" 或 "spawn"
    """
    
    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        context: Literal["fork", "spawn"] | None = None,
    ):
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, context=context)
        
        super().__init__(env_fns, worker_fn)
    
    # ---- 覆写 reset: scatter-gather 并行 ----
    def reset(
        self,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple:
        assert not self.is_closed, "Cannot reset closed VectorEnv"
        env_id = self._wrap_id(env_id)
        
        # scatter: 发送 reset 给所有 worker
        for i in env_id:
            self.workers[i].send_reset(**kwargs)
        
        # gather: 接收所有结果
        results = [self.workers[i].recv_reset() for i in env_id]
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        
        if self._is_parallel_env:
            return self._stack_dict_obs(obs_list), self._stack_dict_info(info_list)
        else:
            return self._stack_array_obs(obs_list), np.array(info_list)
    
    # ---- 覆写 step: scatter-gather 并行 ----
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
        """ParallelEnv 的并行 step，含 auto-reset"""
        first_agent = self.agents[0]
        
        # Phase 1: scatter step
        for j, i in enumerate(env_id):
            action_dict = {agent: actions[agent][j] for agent in self.agents}
            self.workers[i].send_step(action_dict)
        
        # Phase 2: gather step（所有子进程并行计算 env.step）
        raw = [self.workers[i].recv_step() for _, i in enumerate(env_id)]
        
        # Phase 3: 识别 done/truncated 环境，保存 final_obs
        done_pairs = []   # (result_idx, worker_idx)
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
        
        # Phase 4: 并行获取 final_state（仅 truncated 环境，用 call_method 避免传 bound method）
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
        
        # Phase 5: 并行 reset done 的环境
        for _, i in done_pairs:
            self.workers[i].send_reset()
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
        """Gymnasium Env 的并行 step，含 auto-reset"""
        assert len(actions) == len(env_id)
        
        # Phase 1: scatter step
        for j, i in enumerate(env_id):
            self.workers[i].send_step(actions[j])
        
        # Phase 2: gather step
        raw = [self.workers[i].recv_step() for _, i in enumerate(env_id)]
        
        # Phase 3: 识别 done 环境
        done_pairs = []
        results = []
        for j, i in enumerate(env_id):
            obs, rew, term, trunc, info = raw[j]
            if term or trunc:
                info["final_obs"] = obs
                done_pairs.append((j, i))
            results.append([obs, rew, term, trunc, info])
        
        # Phase 4: 并行 reset
        for _, i in done_pairs:
            self.workers[i].send_reset()
        for j_idx, i in done_pairs:
            new_obs, reset_info = self.workers[i].recv_reset()
            results[j_idx][0] = new_obs
            results[j_idx][4].update(reset_info)
        
        return self._unpack_gym_results([tuple(r) for r in results], env_id)
    
    # ---- 覆写 call_env_method: scatter-gather 并行，避免传 bound method ----
    def call_env_method(
        self,
        method_name: str,
        *args,
        env_id: int | List[int] | np.ndarray | None = None,
        **kwargs,
    ) -> List[Any]:
        """在子进程中调用方法并行返回结果，只传回返回值（不传 bound method）。"""
        assert not self.is_closed
        env_id = self._wrap_id(env_id)
        
        # scatter
        for i in env_id:
            self.workers[i].send_call_method(method_name, *args, **kwargs)
        
        # gather
        return [self.workers[i].recv_call_method() for i in env_id]
    
    # ---- 覆写 get_env_attr: scatter-gather 并行 ----
    def get_env_attr(
        self,
        key: str,
        env_id: int | List[int] | np.ndarray | None = None,
    ) -> List[Any]:
        """并行获取指定环境的属性。"""
        assert not self.is_closed
        env_id = self._wrap_id(env_id)
        
        # scatter
        for i in env_id:
            self.workers[i].parent_conn.send(("getattr", key))
        
        # gather
        return [self.workers[i].parent_conn.recv() for i in env_id]