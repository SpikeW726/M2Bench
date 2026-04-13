"""Q-table Trainer：纯 numpy 在线 Q-learning 训练循环。

不继承 BaseTrainer（避免 nn.Module / Collector 依赖），
但复用 SimpleLogger、VectorEnv 等外围设施。
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from algorithms.tabular.qtable import QTableAlgo
from configs.training_configs import TrainerConfig


def _save_qtable_final(
    algo: QTableAlgo,
    save_dir: Path,
    episode: int,
    total_steps: int,
    verbose: bool,
) -> None:
    """写入 save_dir/final/，与 RL checkpoint 约定一致（sweep summary、自动评估）。"""
    final_dir = Path(save_dir) / "final"
    algo.save(str(final_dir))
    meta = {
        "algo_name": "qtable",
        "episode": episode,
        "total_steps": total_steps,
        "epsilon": algo.get_epsilon(),
        "qtable_sizes": {aid: len(pol.q_table) for aid, pol in algo.policies.items()},
    }
    with open(final_dir / "config.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    if verbose:
        print(f"[QTable] Saved final Q-tables to {final_dir}")


class QTableTrainer:
    """Q-table 训练器，按 episode 迭代。

    每个 episode：
        1. vec_env.reset()
        2. 逐步选动作 → env.step → 对 active agent 做 Q-update
        3. 所有 env 结束后 decay epsilon
        4. 日志 & checkpoint

    停止条件（二选一）：
        - total_steps 非 None：累计环境步数（与 _run_episode 返回的 steps 一致）>= total_steps 后停止；
          每 episode 仍完整跑完再判停，故实际总步数可能略超预算。
        - total_steps 为 None：跑满 max_iterations 个 episode（与旧行为一致）。

    Args:
        algo: QTableAlgo 实例
        vec_env: 向量化环境（DummyVectorEnv 即可）
        config: TrainerConfig
        save_dir: 模型保存根目录
        logger: SimpleLogger 实例
        log_extra_fn: 获取环境指标的回调
    """

    def __init__(
        self,
        algo: QTableAlgo,
        vec_env,
        config: TrainerConfig,
        save_dir: Path,
        logger=None,
        log_extra_fn: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.algo = algo
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.agents = vec_env.agents
        self._step_budget: Optional[int] = config.total_steps
        self.max_episodes = config.max_iterations
        self.save_interval = config.save_interval
        self.save_dir = Path(save_dir)
        self.logger = logger
        self.log_extra_fn = log_extra_fn
        self.verbose = config.verbose

        self._sync_mode = getattr(algo.params, 'sync_update', False)
        self._gamma = algo.gamma

        self.total_steps = 0
        self.best_reward = -float("inf")

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        episode_rewards: List[float] = []
        ep = 0

        while True:
            ep += 1
            ep_result = self._run_episode()
            self.total_steps += ep_result["steps"]
            episode_rewards.append(ep_result["mean_reward"])

            self.algo.update_epsilon(self.total_steps)

            if ep_result["mean_reward"] > self.best_reward:
                self.best_reward = ep_result["mean_reward"]

            self._log_episode(ep, ep_result, start_time)

            if ep % self.save_interval == 0:
                self._save_checkpoint(ep)

            if self._step_budget is not None:
                if self.total_steps >= self._step_budget:
                    break
            elif ep >= self.max_episodes:
                break

        self._save_checkpoint(ep)
        _save_qtable_final(
            self.algo, self.save_dir, ep, self.total_steps, self.verbose
        )

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

        return {
            "total_episodes": ep,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }

    def _run_episode(self) -> Dict[str, Any]:
        """运行一个 episode（所有 vec_env 并行），返回统计量。"""
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
        """为所有 agent 的所有 env 选择动作。"""
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
        """对每个 agent 做 Q-learning 更新。

        sync_mode=False: 仅 active 步逐步更新（原始逻辑）。
        sync_mode=True:  决策→到达折叠为一次 Q-update，
                         中间累积折扣 reward，到达时用 γ^(K+1) 做 bootstrap。
        """
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
        """非同步：仅 active 步做单步 Q-update。"""
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
        """同步更新：决策→到达折叠为一次 Q-update。

        per agent per env 维护 pending:
          决策点(active): 记录 (s, a), 重置累积器
          每步: acc_reward += γ^k * r_k, gamma_power *= γ
          到达(next_active) 或 done: flush 一次 Q-update
        """
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

        if self.logger:
            self.logger.log(log_data, step=self.total_steps)

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

    def _save_checkpoint(self, episode: int):
        ckpt_dir = self.save_dir / f"ep_{episode}"
        self.algo.save(str(ckpt_dir))

        meta = {
            "algo_name": "qtable",
            "episode": episode,
            "total_steps": self.total_steps,
            "epsilon": self.algo.get_epsilon(),
            "qtable_sizes": {
                aid: len(pol.q_table) for aid, pol in self.algo.policies.items()
            },
        }
        with open(ckpt_dir / "config.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        if self.verbose:
            print(f"[Checkpoint] Saved episode {episode} to {ckpt_dir}")


# =============================================================================
#                     JointQTableTrainer — Gymnasium Env 版本
#
#  与 QTableTrainer 对称，但对接 gymnasium.Env 格式：
#  - obs/rew/term/trunc 均为 ndarray，无 agent 维度
#  - 使用虚拟 key "agent_0" 访问 algo.policies
# =============================================================================

class JointQTableTrainer:
    """Gymnasium 格式的 Q-table 训练器（对应 JointBaseEnv 系列环境）。

    每个 episode：
        1. vec_env.reset()
        2. 逐步选动作 → env.step → Q-update
        3. 所有 env 结束后 decay epsilon
        4. 日志 & checkpoint

    停止条件与 QTableTrainer 相同：total_steps 预算 或 max_iterations 个 episode。
    """

    POLICY_KEY = "agent_0"

    def __init__(
        self,
        algo: QTableAlgo,
        vec_env,
        config: TrainerConfig,
        save_dir: Path,
        logger=None,
        log_extra_fn=None,
    ):
        self.algo = algo
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self._step_budget: Optional[int] = config.total_steps
        self.max_episodes = config.max_iterations
        self.save_interval = config.save_interval
        self.save_dir = Path(save_dir)
        self.logger = logger
        self.log_extra_fn = log_extra_fn
        self.verbose = config.verbose

        self.total_steps = 0
        self.best_reward = -float("inf")

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        episode_rewards: List[float] = []
        ep = 0

        while True:
            ep += 1
            ep_result = self._run_episode()
            self.total_steps += ep_result["steps"]
            episode_rewards.append(ep_result["mean_reward"])

            self.algo.update_epsilon(self.total_steps)

            if ep_result["mean_reward"] > self.best_reward:
                self.best_reward = ep_result["mean_reward"]

            self._log_episode(ep, ep_result, start_time)

            if ep % self.save_interval == 0:
                self._save_checkpoint(ep)

            if self._step_budget is not None:
                if self.total_steps >= self._step_budget:
                    break
            elif ep >= self.max_episodes:
                break

        self._save_checkpoint(ep)
        _save_qtable_final(
            self.algo, self.save_dir, ep, self.total_steps, self.verbose
        )

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

        return {
            "total_episodes": ep,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "best_reward": self.best_reward,
        }

    def _run_episode(self) -> Dict[str, Any]:
        obs, info = self.vec_env.reset()   # obs: (num_envs, *obs_shape)
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
        """为所有 env 选动作，返回 (num_envs,) int 数组。"""
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
        """对每个未结束的 env 做单步 Q-update。"""
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

        if self.logger:
            self.logger.log(log_data, step=self.total_steps)

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

    def _save_checkpoint(self, episode: int):
        ckpt_dir = self.save_dir / f"ep_{episode}"
        self.algo.save(str(ckpt_dir))

        meta = {
            "algo_name": "qtable_joint",
            "episode": episode,
            "total_steps": self.total_steps,
            "epsilon": self.algo.get_epsilon(),
            "qtable_size": len(self.algo.policies[self.POLICY_KEY].q_table),
        }
        with open(ckpt_dir / "config.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        if self.verbose:
            print(f"[Checkpoint] Saved episode {episode} to {ckpt_dir}")
