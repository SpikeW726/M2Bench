"""NumPy online Q-learning training loops.

These trainers intentionally avoid ``BaseTrainer`` and neural-network collector
dependencies while reusing vector environments, logging, evaluation, and
checkpoint facilities.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from algorithms.tabular.qtable import QTableAlgo
from configs.training_configs import TrainerConfig

def _build_qtable_yaml_extra(
    vec_env: Any,
    base_extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    from envs.venv_wrappers import VectorEnvNormObs, VectorEnvNormReward, find_vec_wrapper
    from utils.model_io import running_mean_std_to_yaml_dict

    extra = dict(base_extra or {})
    obs_w = find_vec_wrapper(vec_env, VectorEnvNormObs)
    if obs_w is not None:
        extra["obs_rms_state"] = running_mean_std_to_yaml_dict(obs_w.get_obs_rms())
    rew_w = find_vec_wrapper(vec_env, VectorEnvNormReward)
    if rew_w is not None:
        extra["reward_rms_state"] = running_mean_std_to_yaml_dict(rew_w.get_reward_rms())
    return extra if extra else None

def _metric_improved(
    value: float,
    best: Optional[float],
    *,
    minimize: bool,
) -> bool:
    if best is None:
        return True
    return value < best if minimize else value > best

def _save_qtable_checkpoint(
    algo: QTableAlgo,
    vec_env,
    save_dir: Path,
    checkpoint_name: str,
    episode: int,
    total_steps: int,
    verbose: bool,
    base_extra: Optional[Dict[str, Any]] = None,
    algo_name: str = "qtable",
    best_metric_name: Optional[str] = None,
    best_metric_value: Optional[float] = None,
) -> Path:
    ckpt_dir = Path(save_dir) / checkpoint_name
    algo.save(str(ckpt_dir))
    meta: Dict[str, Any] = {
        "algo_name": algo_name,
        "episode": episode,
        "total_steps": total_steps,
        "epsilon": algo.get_epsilon(),
        "qtable_sizes": {aid: len(pol.q_table) for aid, pol in algo.policies.items()},
    }
    if best_metric_name is not None:
        meta["best_metric_name"] = best_metric_name
        meta["best_metric_value"] = best_metric_value
    yaml_extra = _build_qtable_yaml_extra(vec_env, base_extra)
    if yaml_extra:
        meta["extra"] = yaml_extra
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    if verbose:
        metric_msg = (
            f", {best_metric_name}={best_metric_value:.4f}"
            if best_metric_name is not None and best_metric_value is not None
            else ""
        )
        print(f"[QTable] Saved best checkpoint to {ckpt_dir}{metric_msg}")
    return ckpt_dir

def _log_metrics(logger, data: Dict[str, float], step: int) -> None:
    if logger is not None:
        logger.log(data, step=step)

def _maybe_run_inline_eval(
    *,
    algo: QTableAlgo,
    eval_fn: Optional[Callable[[], Dict[str, float]]],
    eval_interval: int,
    iteration: int,
    verbose: bool,
    logger,
    total_steps: int,
    on_eval_complete: Optional[Callable],
    save_best_fn: Optional[Callable[[Dict[str, float], int], bool]] = None,
) -> None:
    if not eval_fn or eval_interval <= 0:
        return
    if iteration % eval_interval != 0:
        return
    if verbose:
        print(f"[Eval] Running inline eval at iteration {iteration} ...")
    algo.set_training_mode(False)
    try:
        eval_metrics = eval_fn()
    except Exception as e:
        print(f"[Eval] Inline eval failed: {e}")
        eval_metrics = {}
    finally:
        algo.set_training_mode(True)
    if eval_metrics:
        if save_best_fn is not None:
            save_best_fn(eval_metrics, iteration)
        _log_metrics(logger, eval_metrics, step=total_steps)
        if on_eval_complete is not None:
            on_eval_complete(eval_metrics, total_steps)
        if verbose:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
            print(f"[Eval] {parts}")

class QTableTrainer:
    """Online trainer for independent per-agent Q-tables.

    Training uses vectorized environment transitions, periodic evaluation, best
    and final checkpoints, optional normalization wrappers, and callback hooks
    shared with the neural training pipeline.
    """

    def __init__(
        self,
        algo: QTableAlgo,
        vec_env,
        config: TrainerConfig,
        save_dir: Path,
        logger=None,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        best_metric_name: str = "eval/wi",
        best_metric_goal: str = "minimize",
    ):
        self.algo = algo
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.agents = vec_env.agents
        self._step_budget: Optional[int] = config.total_steps
        self.max_episodes = config.max_iterations
        self.eval_interval = getattr(config, "eval_interval", 0)
        self.save_dir = Path(save_dir)
        self.logger = logger
        self.log_extra_fn = log_extra_fn
        self.eval_fn = eval_fn
        self.extra_info = extra_info or {}
        self.verbose = config.verbose
        self.best_metric_name = best_metric_name
        self._minimize_best = best_metric_goal == "minimize"

        self._sync_mode = getattr(algo.params, 'sync_update', False)
        self._gamma = algo.gamma

        self.iteration = 0
        self.total_steps = 0
        self.best_reward = -float("inf")
        self.on_eval_complete: Optional[Callable] = None
        self._best_metric_value: Optional[float] = None
        self._best_saved = False

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        episode_rewards: List[float] = []
        ep = 0

        while True:
            ep += 1
            self.iteration = ep
            ep_result = self._run_episode()
            self.total_steps += ep_result["steps"]
            episode_rewards.append(ep_result["mean_reward"])

            self.algo.update_epsilon(self.total_steps)

            if ep_result["mean_reward"] > self.best_reward:
                self.best_reward = ep_result["mean_reward"]

            self._log_episode(ep, ep_result, start_time)
            _maybe_run_inline_eval(
                algo=self.algo,
                eval_fn=self.eval_fn,
                eval_interval=self.eval_interval,
                iteration=self.iteration,
                verbose=self.verbose,
                logger=self.logger,
                total_steps=self.total_steps,
                on_eval_complete=self.on_eval_complete,
                save_best_fn=self.maybe_save_best_checkpoint,
            )

            if self._step_budget is not None:
                if self.total_steps >= self._step_budget:
                    break
            elif ep >= self.max_episodes:
                break

        total_time = time.time() - start_time
        if self.verbose:
            budget_msg = (
                f"step_budget={self._step_budget}"
                if self._step_budget is not None
                else f"episodes={self.max_episodes}"
            )
            print(f"\n[QTable] Training complete! {budget_msg}, "
                  f"ran_episodes={ep}, Steps: {self.total_steps}, Time: {total_time:.1f}s, "
                  f"Best reward: {self.best_reward:.2f}")
            if self._best_saved:
                print(f"  Best checkpoint: {self.save_dir / 'best'} "
                      f"({self.best_metric_name}={self._best_metric_value:.4f})")
            else:
                print("  No best checkpoint saved (inline eval never improved).")

        return {
            "total_episodes": ep,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }

    def maybe_save_best_checkpoint(self, metrics: Dict[str, float], episode: int) -> bool:
        val = metrics.get(self.best_metric_name)
        if val is None:
            return False
        val = float(val)
        if not _metric_improved(val, self._best_metric_value, minimize=self._minimize_best):
            return False
        self._best_metric_value = val
        _save_qtable_checkpoint(
            self.algo,
            self.vec_env,
            self.save_dir,
            "best",
            episode,
            self.total_steps,
            self.verbose,
            base_extra=self.extra_info,
            best_metric_name=self.best_metric_name,
            best_metric_value=val,
        )
        self._best_saved = True
        return True

    @property
    def has_best_checkpoint(self) -> bool:
        return self._best_saved

    def _run_episode(self) -> Dict[str, Any]:
        obs_dict, info_dict = self.vec_env.reset()
        done_flags = np.zeros(self.num_envs, dtype=bool)
        ep_rewards = np.zeros(self.num_envs)
        ep_steps = 0

        if self._sync_mode:
            self._pending: Dict[str, List[Optional[dict]]] = {
                agent: [None] * self.num_envs for agent in self.agents
            }

        while not done_flags.all():
            actions = self._select_actions(obs_dict, info_dict)

            next_obs, rew, term, trunc, next_info = self.vec_env.step(actions)

            first_agent = self.agents[0]
            for i in range(self.num_envs):
                if done_flags[i]:
                    continue
                for agent in self.agents:
                    ep_rewards[i] += rew[agent][i]

            self._update_qtables(
                obs_dict, actions, rew, next_obs, term, trunc,
                info_dict, next_info, done_flags,
            )

            for i in range(self.num_envs):
                if not done_flags[i] and (term[first_agent][i] or trunc[first_agent][i]):
                    done_flags[i] = True

            obs_dict = next_obs
            info_dict = next_info
            ep_steps += self.num_envs

        mean_agent_reward = ep_rewards / len(self.agents)

        return {
            "mean_reward": float(mean_agent_reward.mean()),
            "steps": ep_steps,
            "per_env_rewards": mean_agent_reward,
        }

    def _select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        info_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        actions = {}
        for agent in self.agents:
            agent_actions = np.zeros(self.num_envs, dtype=np.int64)
            for i in range(self.num_envs):
                info_i = info_dict[agent][i] if info_dict and agent in info_dict else {}
                am = info_i.get("active_mask", 1)
                action_mask = info_i.get("action_mask", None)

                if am and action_mask is not None:
                    agent_actions[i] = self.algo.policies[agent].select_action(
                        obs_dict[agent][i], action_mask,
                    )
                else:
                    if action_mask is not None:
                        valid = np.where(action_mask)[0]
                        agent_actions[i] = valid[-1] if len(valid) > 0 else 0
                    else:
                        agent_actions[i] = 0
            actions[agent] = agent_actions
        return actions

    def _update_qtables(
        self,
        obs_dict, actions, rew, next_obs, term, trunc,
        info_dict, next_info, done_flags,
    ):
        if self._sync_mode:
            self._update_qtables_sync(
                obs_dict, actions, rew, next_obs, term, trunc,
                info_dict, next_info, done_flags,
            )
        else:
            self._update_qtables_standard(
                obs_dict, actions, rew, next_obs, term, trunc,
                info_dict, next_info, done_flags,
            )

    def _update_qtables_standard(
        self,
        obs_dict, actions, rew, next_obs, term, trunc,
        info_dict, next_info, done_flags,
    ):
        for agent in self.agents:
            for i in range(self.num_envs):
                if done_flags[i]:
                    continue

                info_i = info_dict[agent][i] if info_dict and agent in info_dict else {}
                am = info_i.get("active_mask", 1)
                if not am:
                    continue

                done = bool(term[agent][i] or trunc[agent][i])
                next_info_i = next_info[agent][i] if next_info and agent in next_info else {}
                next_am = next_info_i.get("action_mask", None)

                self.algo.update_step(
                    agent_id=agent,
                    obs=obs_dict[agent][i],
                    action=int(actions[agent][i]),
                    reward=float(rew[agent][i]),
                    next_obs=next_obs[agent][i],
                    done=done,
                    next_action_mask=next_am,
                )

    def _update_qtables_sync(
        self,
        obs_dict, actions, rew, next_obs, term, trunc,
        info_dict, next_info, done_flags,
    ):
        gamma = self._gamma
        for agent in self.agents:
            for i in range(self.num_envs):
                if done_flags[i]:
                    continue

                info_i = info_dict[agent][i] if info_dict and agent in info_dict else {}
                was_active = bool(info_i.get("active_mask", 1))

                if was_active:
                    self._pending[agent][i] = {
                        'obs': obs_dict[agent][i].copy(),
                        'act': int(actions[agent][i]),
                        'acc_reward': 0.0,
                        'gamma_power': 1.0,
                    }

                pend = self._pending[agent][i]
                if pend is not None:
                    pend['acc_reward'] += pend['gamma_power'] * float(rew[agent][i])
                    pend['gamma_power'] *= gamma

                is_done = bool(term[agent][i] or trunc[agent][i])
                next_info_i = next_info[agent][i] if next_info and agent in next_info else {}
                now_active = bool(next_info_i.get("active_mask", 1))

                if pend is not None and (now_active or is_done):
                    next_am = next_info_i.get("action_mask", None)
                    self.algo.update_step(
                        agent_id=agent,
                        obs=pend['obs'],
                        action=pend['act'],
                        reward=pend['acc_reward'],
                        next_obs=next_obs[agent][i],
                        done=is_done,
                        next_action_mask=next_am,
                        gamma_power=pend['gamma_power'],
                    )
                    self._pending[agent][i] = None

                if is_done:
                    self._pending[agent][i] = None

    def _log_episode(self, ep: int, ep_result: Dict, start_time: float):
        elapsed = time.time() - start_time
        sps = int(self.total_steps / elapsed) if elapsed > 0 else 0

        log_data = {
            "rollout/episode_reward": ep_result["mean_reward"],
            "train/epsilon": self.algo.get_epsilon(),
            "train/total_steps": self.total_steps,
            "train/sps": sps,
            "train/qtable_size": sum(
                len(p.q_table) for p in self.algo.policies.values()
            ),
        }

        if self.log_extra_fn:
            extra = self.log_extra_fn()
            if extra:
                log_data.update(extra)

        _log_metrics(self.logger, log_data, step=self.total_steps)

        if self.verbose and (ep % 50 == 0 or ep == 1):
            ep_cap = (
                f"{self._step_budget} steps"
                if self._step_budget is not None
                else f"{self.max_episodes} ep"
            )
            print(
                f"[Ep {ep} | stop@{ep_cap}] "
                f"reward={ep_result['mean_reward']:.2f}, "
                f"eps={self.algo.get_epsilon():.4f}, "
                f"steps={self.total_steps}, SPS={sps}"
            )

# JointQTableTrainer - Gymnasium Env.

# obs/rew/term/trunc.

class JointQTableTrainer:
    """Online trainer for one joint Q-table over all agent actions."""

    POLICY_KEY = "agent_0"

    def __init__(
        self,
        algo: QTableAlgo,
        vec_env,
        config: TrainerConfig,
        save_dir: Path,
        logger=None,
        log_extra_fn=None,
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        best_metric_name: str = "eval/wi",
        best_metric_goal: str = "minimize",
    ):
        self.algo = algo
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self._step_budget: Optional[int] = config.total_steps
        self.max_episodes = config.max_iterations
        self.eval_interval = getattr(config, "eval_interval", 0)
        self.save_dir = Path(save_dir)
        self.logger = logger
        self.log_extra_fn = log_extra_fn
        self.eval_fn = eval_fn
        self.extra_info = extra_info or {}
        self.verbose = config.verbose
        self.best_metric_name = best_metric_name
        self._minimize_best = best_metric_goal == "minimize"

        self.iteration = 0
        self.total_steps = 0
        self.best_reward = -float("inf")
        self.on_eval_complete: Optional[Callable] = None
        self._best_metric_value: Optional[float] = None
        self._best_saved = False

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        episode_rewards: List[float] = []
        ep = 0

        while True:
            ep += 1
            self.iteration = ep
            ep_result = self._run_episode()
            self.total_steps += ep_result["steps"]
            episode_rewards.append(ep_result["mean_reward"])

            self.algo.update_epsilon(self.total_steps)

            if ep_result["mean_reward"] > self.best_reward:
                self.best_reward = ep_result["mean_reward"]

            self._log_episode(ep, ep_result, start_time)
            _maybe_run_inline_eval(
                algo=self.algo,
                eval_fn=self.eval_fn,
                eval_interval=self.eval_interval,
                iteration=self.iteration,
                verbose=self.verbose,
                logger=self.logger,
                total_steps=self.total_steps,
                on_eval_complete=self.on_eval_complete,
                save_best_fn=self.maybe_save_best_checkpoint,
            )

            if self._step_budget is not None:
                if self.total_steps >= self._step_budget:
                    break
            elif ep >= self.max_episodes:
                break

        total_time = time.time() - start_time
        if self.verbose:
            budget_msg = (
                f"step_budget={self._step_budget}"
                if self._step_budget is not None
                else f"episodes={self.max_episodes}"
            )
            print(f"\n[JointQTable] Training complete! {budget_msg}, "
                  f"ran_episodes={ep}, Steps: {self.total_steps}, Time: {total_time:.1f}s, "
                  f"Best reward: {self.best_reward:.2f}")
            if self._best_saved:
                print(f"  Best checkpoint: {self.save_dir / 'best'} "
                      f"({self.best_metric_name}={self._best_metric_value:.4f})")
            else:
                print("  No best checkpoint saved (inline eval never improved).")

        return {
            "total_episodes": ep,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }

    def maybe_save_best_checkpoint(self, metrics: Dict[str, float], episode: int) -> bool:
        val = metrics.get(self.best_metric_name)
        if val is None:
            return False
        val = float(val)
        if not _metric_improved(val, self._best_metric_value, minimize=self._minimize_best):
            return False
        self._best_metric_value = val
        _save_qtable_checkpoint(
            self.algo,
            self.vec_env,
            self.save_dir,
            "best",
            episode,
            self.total_steps,
            self.verbose,
            base_extra=self.extra_info,
            algo_name="qtable_joint",
            best_metric_name=self.best_metric_name,
            best_metric_value=val,
        )
        self._best_saved = True
        return True

    @property
    def has_best_checkpoint(self) -> bool:
        return self._best_saved

    def _run_episode(self) -> Dict[str, Any]:
        obs, info = self.vec_env.reset()   # obs: (num_envs, *obs_shape).
        done_flags = np.zeros(self.num_envs, dtype=bool)
        ep_rewards = np.zeros(self.num_envs)
        ep_steps = 0

        while not done_flags.all():
            actions = self._select_actions(obs, info)

            next_obs, rew, term, trunc, next_info = self.vec_env.step(actions)

            ep_rewards += np.where(done_flags, 0.0, rew)

            self._update_qtables(obs, actions, rew, next_obs, term, trunc,
                                 info, next_info, done_flags)

            done_flags |= term | trunc
            obs, info = next_obs, next_info
            ep_steps += self.num_envs

        return {
            "mean_reward": float(ep_rewards.mean()),
            "steps": ep_steps,
            "per_env_rewards": ep_rewards,
        }

    def _select_actions(self, obs: np.ndarray, info: np.ndarray) -> np.ndarray:
        policy = self.algo.policies[self.POLICY_KEY]
        actions = np.zeros(self.num_envs, dtype=np.int64)
        for i in range(self.num_envs):
            info_i = info[i] if isinstance(info[i], dict) else {}
            action_mask = info_i.get("action_mask", None)
            if action_mask is not None:
                actions[i] = policy.select_action(obs[i], action_mask)
            else:
                actions[i] = policy.select_action(
                    obs[i], np.ones(policy.action_dim, dtype=bool)
                )
        return actions

    def _update_qtables(
        self,
        obs, actions, rew, next_obs, term, trunc,
        info, next_info, done_flags,
    ):
        policy_key = self.POLICY_KEY
        for i in range(self.num_envs):
            if done_flags[i]:
                continue
            next_info_i = next_info[i] if isinstance(next_info[i], dict) else {}
            next_mask = next_info_i.get("action_mask", None)
            self.algo.update_step(
                agent_id=policy_key,
                obs=obs[i],
                action=int(actions[i]),
                reward=float(rew[i]),
                next_obs=next_obs[i],
                done=bool(term[i] or trunc[i]),
                next_action_mask=next_mask,
            )

    def _log_episode(self, ep: int, ep_result: Dict, start_time: float):
        elapsed = time.time() - start_time
        sps = int(self.total_steps / elapsed) if elapsed > 0 else 0

        log_data = {
            "rollout/episode_reward": ep_result["mean_reward"],
            "train/epsilon": self.algo.get_epsilon(),
            "train/total_steps": self.total_steps,
            "train/sps": sps,
            "train/qtable_size": len(self.algo.policies[self.POLICY_KEY].q_table),
        }

        if self.log_extra_fn:
            extra = self.log_extra_fn()
            if extra:
                log_data.update(extra)

        _log_metrics(self.logger, log_data, step=self.total_steps)

        if self.verbose and (ep % 50 == 0 or ep == 1):
            ep_cap = (
                f"{self._step_budget} steps"
                if self._step_budget is not None
                else f"{self.max_episodes} ep"
            )
            print(
                f"[Ep {ep} | stop@{ep_cap}] "
                f"reward={ep_result['mean_reward']:.2f}, "
                f"eps={self.algo.get_epsilon():.4f}, "
                f"steps={self.total_steps}, SPS={sps}"
            )
