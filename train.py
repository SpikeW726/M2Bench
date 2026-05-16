"""
统一训练入口：从 YAML 配置文件启动训练。

用法:
    python train.py configs/experiments/mappo_tsp12_imi.yaml
    python train.py configs/experiments/mappo_tsp12_scratch.yaml
"""

import math
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from configs.exp_configs import ExperimentConfig
from configs.training_configs import OnPolicyTrainerConfig, OffPolicyTrainerConfig
from configs.registry import (
    ALGO_REGISTRY,
    load_config,
    create_vec_env,
    create_actor,
    create_critic,
    create_q_network,
    create_algorithm,
    create_trainer,
    get_policy_type,
    get_trainer_type,
)
from data.collector import (
    OnPolicyCollector, MAOnPolicyCollector,
    OffPolicyCollector, MAOffPolicyCollector,
    MATOnPolicyCollector,
)
from data.buffer import ReplayBuffer, SequenceReplayBuffer
from policies.marl.marl_base import MultiAgentPolicy
from policies.rl.rl_base import ActorPolicy, ValuePolicy
from trainers.sweep_early_stopper import SweepEarlyStop
from utils.model_io import save_model
from utils.autodl_paths import resolve_models_path

# WandB：同一进程内多次 train 时按 run.id 重新注册 step 轴
_WANDB_ENV_STEP_AXIS_RUN_ID: Optional[str] = None


def _configure_wandb_env_step_axis() -> None:
    """注册 global_step 为自定义横轴指标；实际主横轴由 SimpleLogger 的 wandb.log(..., step=) 锁定为环境步数。

    不传 step= 时 WandB 会对每次 log 自增内部计数 → 横轴变成 iteration，与 sweep/直连 train 路径均会不一致。
    """
    global _WANDB_ENV_STEP_AXIS_RUN_ID
    import wandb

    if wandb.run is None:
        return
    rid = getattr(wandb.run, "id", None)
    if rid is None or rid == _WANDB_ENV_STEP_AXIS_RUN_ID:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    _WANDB_ENV_STEP_AXIS_RUN_ID = rid


# =============================================================================
#                          随机种子
# =============================================================================


def _set_training_random_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch RNG（训练可复现）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_seed_vec_env(vec_env, seed: Optional[int]) -> None:
    if seed is None:
        return
    vec_env.seed(seed)


# =============================================================================
#                              Logger
# =============================================================================

class SimpleLogger:
    """TensorBoard + wandb 日志包装"""

    def __init__(self, tb_writer: SummaryWriter, use_wandb: bool = False):
        self.tb_writer = tb_writer
        self.use_wandb = use_wandb

    def log(self, data: Dict[str, float], step: int):
        for key, value in data.items():
            self.tb_writer.add_scalar(key, value, step)
        if self.use_wandb:
            import wandb

            _configure_wandb_env_step_axis()
            # 必须传 step=：否则 WandB 用内部计数器（每次 log +1）→ 横轴变成 iter。
            # define_metric 仅辅助 UI 选轴；与 step= 同时存在时以 step= 为准。
            payload = dict(data)
            gs = int(step)
            payload["global_step"] = gs
            wandb.log(payload, step=gs)


class TrainingStageProfiler:
    """按 iteration 聚合训练阶段耗时，CUDA 路径会同步以避免低估 GPU update。"""

    def __init__(self, device: torch.device, sync_cuda: Optional[bool] = None):
        self.device = device
        self.sync_cuda = (
            device.type == "cuda" and torch.cuda.is_available()
            if sync_cuda is None
            else bool(sync_cuda)
        )
        self._iter_times: Dict[str, float] = defaultdict(float)
        self._total_times: Dict[str, float] = defaultdict(float)
        self._iter_inclusive_times: Dict[str, float] = defaultdict(float)
        self._total_inclusive_times: Dict[str, float] = defaultdict(float)
        self._last_times: Dict[str, float] = {}
        self._stack: list[list[object]] = []

    def _sync(self) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def time_block(self, name: str):
        self._sync()
        start = time.perf_counter()
        frame = [name, start, 0.0]  # name, start, nested_child_time
        self._stack.append(frame)
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            popped = self._stack.pop()
            child_time = float(popped[2])
            exclusive = max(0.0, elapsed - child_time)
            self._iter_times[name] += exclusive
            self._total_times[name] += exclusive
            self._iter_inclusive_times[name] += elapsed
            self._total_inclusive_times[name] += elapsed
            if self._stack:
                self._stack[-1][2] = float(self._stack[-1][2]) + elapsed

    def consume_iteration_metrics(self) -> Dict[str, float]:
        if not self._iter_times:
            return {}

        excl = dict(self._iter_times)          # exclusive：各阶段自身耗时，无重叠
        incl = dict(self._iter_inclusive_times) # inclusive：含所有子阶段的墙钟时间
        total_excl = sum(excl.values())
        metrics: Dict[str, float] = {
            "profile/iter_profiled_s": float(total_excl),
            "profile/cuda_synchronized": float(self.sync_cuda),
        }
        # profile/{name}_s = inclusive（含子阶段），便于直接与其他 inclusive 指标对比
        all_names = set(excl) | set(incl)
        for name in all_names:
            inc_s = incl.get(name, excl.get(name, 0.0))
            exc_s = excl.get(name, 0.0)
            metrics[f"profile/{name}_s"] = float(inc_s)
            # 百分比用 exclusive，保证所有叶节点之和 ≈ 100%
            metrics[f"profile_pct/{name}"] = (
                float(100.0 * exc_s / total_excl) if total_excl > 0 else 0.0
            )

        # _last_times 存 inclusive，供 format_last 打印有意义的时间
        self._last_times = {n: incl.get(n, excl.get(n, 0.0)) for n in all_names}
        self._iter_times.clear()
        self._iter_inclusive_times.clear()
        return metrics

    def format_last(self, top_k: int = 6) -> str:
        """打印 inclusive 耗时 top-k，百分比相对于 exclusive 总量（不重叠）。"""
        if not self._last_times:
            return ""
        excl = dict(self._total_times)  # 用累计 exclusive 做占比分母
        total_excl = sum(
            v for k, v in excl.items()
            if k in self._last_times
        )
        if total_excl == 0:
            total_excl = sum(self._last_times.values()) or 1.0
        ranked = sorted(self._last_times.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        parts = []
        for name, inc_s in ranked:
            exc_s = excl.get(name, inc_s)
            pct = 100.0 * exc_s / total_excl if total_excl > 0 else 0.0
            parts.append(f"{name}={inc_s:.3f}s({pct:.0f}%)")
        return ", ".join(parts)


# =============================================================================
#                          辅助函数
# =============================================================================

def _wandb_env_metrics_from_finished(finished: list) -> Dict[str, float]:
    """从各子环境 get_episode_metrics 返回值构造 WandB env/*。

    可选字段（子类实现时写入）：
    - wi_fromT → env/wi_fromT
    - wait_ratio → env/wait_ratio（与 MASUP 终端 [WAIT_ACTIONS] 比例一致，enable_wait 场景）
    """
    if not finished:
        return {}
    out: Dict[str, float] = {
        "env/igi": float(np.mean([m["igi"] for m in finished])),
        "env/agi": float(np.mean([m["agi"] for m in finished])),
        "env/iwi": float(np.mean([m["iwi"] for m in finished])),
        "env/wi": float(np.mean([m["wi"] for m in finished])),
    }
    wi_t = [m["wi_fromT"] for m in finished if m.get("wi_fromT") is not None]
    if wi_t:
        out["env/wi_fromT"] = float(np.mean(wi_t))
    # MASUP(enable_wait) 等在 get_episode_metrics 中返回 wait_ratio（上一完整 episode）
    wr = [m["wait_ratio"] for m in finished if "wait_ratio" in m]
    if wr:
        out["env/wait_ratio"] = float(np.mean(wr))
    return out


def _infer_dims(vec_env, algo_name: str) -> dict:
    """从向量化环境推断网络所需的各维度。

    根据 is_parallel_env 分两条路径：
    - PettingZoo (parallel): 多智能体 dict I/O，有 global_state
    - Gymnasium (single): 集中式环境，obs 即全局观测
    """
    if vec_env.is_parallel_env:
        return _infer_dims_parallel(vec_env, algo_name)
    else:
        return _infer_dims_single(vec_env)


def _infer_dims_parallel(vec_env, algo_name: str) -> dict:
    """PettingZoo 多智能体环境维度推断。"""
    agent_ids = vec_env.agents
    num_agents = len(agent_ids)
    obs_space = vec_env.observation_space[agent_ids[0]]
    action_space = vec_env.action_space[agent_ids[0]]

    obs_dim = obs_space.shape[0]
    action_dim = action_space.n

    vec_env.reset()
    states = vec_env.call_env_method("state")
    state_dim = len(states[0])

    # MAPPO/VDPPO: centralized critic = global_state + agent one-hot
    # IPPO: decentralized critic = per-agent obs（不依赖全局信息）
    # 其余算法（MAA2C 等）: critic = global_state（无 one-hot）
    if algo_name in ("mappo", "vdppo"):
        critic_input_dim = state_dim + num_agents
    elif algo_name == "ippo":
        critic_input_dim = obs_dim
    else:
        critic_input_dim = state_dim

    return {
        "agent_ids": agent_ids,
        "num_agents": num_agents,
        "obs_space": obs_space,
        "action_space": action_space,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "critic_input_dim": critic_input_dim,
    }


def _infer_dims_single(vec_env) -> dict:
    """Gymnasium 集中式环境维度推断。

    集中式环境将多智能体建模为单体问题：
    - obs = 全局联合观测
    - action = 联合动作
    - 无 global_state（obs 本身即全局状态）
    """
    obs_space = vec_env.observation_space
    action_space = vec_env.action_space

    obs_dim = obs_space.shape[0]
    action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

    return {
        "agent_ids": None,
        "num_agents": 1,
        "obs_space": obs_space,
        "action_space": action_space,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": obs_dim,
        "critic_input_dim": obs_dim,
    }


def _load_pretrained(
    actor_net,
    critic_net,
    actor_path: Optional[str],
    critic_path: Optional[str],
    device,
) -> Optional[dict]:
    """加载预训练权重，返回 value_norm_config（如果存在）。"""
    value_norm_config = None

    has_actor_path = actor_path and actor_net is not None
    has_critic_path = critic_path and critic_net is not None

    if not has_actor_path and not has_critic_path:
        print("[Train] Training from scratch (random initialization)")
        return None

    loaded_from = None
    if has_actor_path:
        actor_p = resolve_models_path(actor_path)
        if actor_p.exists():
            actor_ckpt = torch.load(actor_p, map_location=device, weights_only=True)
            actor_net.load_state_dict(actor_ckpt.get("actor_state_dict", actor_ckpt))
            loaded_from = actor_p
            print(f"[Train] Loaded pretrained actor from {actor_path}")

    if has_critic_path:
        critic_p = resolve_models_path(critic_path)
        if critic_p.exists():
            critic_ckpt = torch.load(critic_p, map_location=device, weights_only=True)
            critic_net.load_state_dict(critic_ckpt.get("critic_state_dict", critic_ckpt))
            if loaded_from is None:
                loaded_from = critic_p
            print(f"[Train] Loaded pretrained critic from {critic_path}")

    if loaded_from is not None:
        config_file = loaded_from.parent / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                saved = yaml.safe_load(f)
            vn = saved.get("value_normalization")
            if vn is not None:
                value_norm_config = vn
                print(f"[Train] Loaded value_norm: "
                      f"mean={vn.get('ret_mean', 0.0):.4f}, "
                      f"std={vn.get('ret_std', 1.0):.4f}")

    return value_norm_config


def _build_mat_policy(config: ExperimentConfig, dims: dict, device):
    """构建 MAT 专属 MATMultiAgentPolicy（GATEncoder + MATDecoder）。

    从 config.algo 获取超参（hidden_dim, num_heads, encoder_layers, decoder_heads）。
    图结构（adj, node_coords, sorted_nodes）从 config.env 的 graph_path 读取。
    """
    import json
    import numpy as np
    from networks.mat import GATEncoder, MATDecoder
    from policies.marl.mat_policy import MATMultiAgentPolicy

    algo_cfg = config.algo
    env_cfg = config.env
    custom = env_cfg.custom_configs or {}

    graph_path = custom.get("graph_path", getattr(env_cfg, "graph_path", None))
    if graph_path is None:
        raise ValueError("[MAT] config.env.custom_configs 中必须包含 graph_path 字段")

    from utils.graph_layout import process_graph_file
    from pathlib import Path
    gp = Path(graph_path)
    coords_path = gp.with_name(gp.stem + "_coords.json")
    if not coords_path.exists():
        process_graph_file(gp)

    with open(graph_path) as f:
        graph_data = json.load(f)
    with open(coords_path) as f:
        raw_coords = json.load(f)

    all_nodes = sorted(graph_data["nodes"])
    graph_size = len(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    node_coords = np.array([raw_coords[str(n)] for n in all_nodes], dtype=np.float32)

    adj = np.zeros((graph_size, graph_size), dtype=np.float32)
    for e in graph_data["edges"]:
        i, j = node_to_idx[e["from"]], node_to_idx[e["to"]]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    hidden_dim = getattr(algo_cfg, "hidden_dim", 64)
    num_heads = getattr(algo_cfg, "num_heads", 4)
    enc_layers = getattr(algo_cfg, "encoder_layers", 2)
    n_agents = dims["num_agents"]

    encoder = GATEncoder(
        input_dim=3,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=enc_layers,
        adj=adj,
        graph_size=graph_size,
    ).to(device)

    decoder = MATDecoder(
        hidden_dim=hidden_dim,
        n_agents=n_agents,
        graph_size=graph_size,
        n_heads=num_heads,
        adj=adj,
        node_coords=node_coords,
        node_id_to_idx=node_to_idx,
        sorted_nodes=all_nodes,
    ).to(device)

    policy = MATMultiAgentPolicy(
        encoder=encoder,
        decoder=decoder,
        n_agents=n_agents,
        agent_ids=dims["agent_ids"],
    ).to(device)

    print(f"[MAT] Built GATEncoder(graph_size={graph_size}, hidden={hidden_dim}, "
          f"heads={num_heads}, layers={enc_layers}) + MATDecoder(n_agents={n_agents})")
    return policy


def _build_policy(
    config: ExperimentConfig,
    dims: dict,
    is_parallel: bool,
    actor_net=None,
    q_net=None,
    device=None,
):
    """根据环境类型和算法类型构建 Policy。

    PettingZoo (parallel) → MultiAgentPolicy（包装多个 ActorPolicy / ValuePolicy）
    Gymnasium (single) → 裸 ActorPolicy / ValuePolicy
    """
    policy_type = get_policy_type(config.algo_name)

    if is_parallel:
        # ---- 多智能体路径 ----
        if policy_type == "mat":
            # MAT 专属路径：从 vec_env 获取图结构信息，构建 GATEncoder + MATDecoder
            return _build_mat_policy(config, dims, device)
        elif policy_type == "value":
            shared = getattr(config.algo, "shared_policy", False)
            return MultiAgentPolicy(
                agent_ids=dims["agent_ids"],
                obs_space=dims["obs_space"],
                action_space=dims["action_space"],
                policy_class=ValuePolicy,
                policy_kwargs={"q_network": q_net},
                shared=shared,
            )
        else:
            shared = (
                getattr(config.algo, "shared_policy", False)
                if config.algo_name == "ippo"
                else True
            )
            return MultiAgentPolicy(
                agent_ids=dims["agent_ids"],
                obs_space=dims["obs_space"],
                action_space=dims["action_space"],
                policy_class=ActorPolicy,
                policy_kwargs={"actor": actor_net},
                shared=shared,
            )
    else:
        # ---- 单体路径（集中式 Gymnasium 环境） ----
        if policy_type == "value":
            return ValuePolicy(
                obs_space=dims["obs_space"],
                action_space=dims["action_space"],
                q_network=q_net,
            )
        else:
            return ActorPolicy(
                obs_space=dims["obs_space"],
                action_space=dims["action_space"],
                actor=actor_net,
            )


def _build_collector(
    config: ExperimentConfig,
    algorithm,
    vec_env,
    dims: dict,
):
    """根据环境类型和算法类型构建 Collector。

    PettingZoo → MAOnPolicyCollector / MAOffPolicyCollector
    Gymnasium  → OnPolicyCollector / OffPolicyCollector

    Off-policy + RNN Q-network → SequenceReplayBuffer（R2D2 预切片）
    Off-policy + MLP Q-network → ReplayBuffer
    """
    trainer_type = get_trainer_type(config.algo_name)
    is_parallel = vec_env.is_parallel_env

    if trainer_type == "off_policy":
        tc: OffPolicyTrainerConfig = config.training
        is_q_recurrent = getattr(algorithm, "is_recurrent", False)

        needs_state = config.algo_name in ("qmix",)
        # VDN/QMIX 等值分解算法须用共享 indices 保证多 agent buffer 时间对齐
        needs_shared_indices = config.algo_name in ("vdn", "qmix")

        # sync_replay 仅 IQL 生效
        sync_replay = (
            getattr(config.algo, "sync_replay", False)
            and config.algo_name == "iql"
        )
        # VDN/QMIX MLP：shared_sync 与 IQL sync_replay 互斥（collector 路径二选一）
        shared_sync_vdq = (
            getattr(config.algo, "shared_sync", False)
            and needs_shared_indices
            and not is_q_recurrent
        )

        if is_parallel:
            if is_q_recurrent:
                burn_in_len = getattr(config.algo, "burn_in_len", 0)
                seq_len = getattr(config.algo, "seq_len", 20)
                max_seqs = getattr(config.algo, "max_episodes", 5000)
                buffers = {
                    aid: SequenceReplayBuffer(
                        obs_dim=dims["obs_dim"],
                        seq_len=seq_len,
                        burn_in_len=burn_in_len,
                        max_seqs=max_seqs,
                        has_action_mask=True,
                        action_dim=dims["action_dim"],
                        has_active_mask=sync_replay,
                        has_state=needs_state,
                        state_dim=dims["state_dim"] if needs_state else 0,
                    )
                    for aid in dims["agent_ids"]
                }
            else:
                buffers = {
                    aid: ReplayBuffer(
                        obs_shape=(dims["obs_dim"],),
                        act_shape=(1,),
                        max_size=tc.buffer_size,
                        has_action_mask=True,
                        action_dim=dims["action_dim"],
                        has_state=needs_state,
                        state_dim=dims["state_dim"] if needs_state else 0,
                        has_active_mask=not sync_replay,
                        has_gamma_power=(sync_replay or shared_sync_vdq),
                    )
                    for aid in dims["agent_ids"]
                }
            gamma = getattr(config.algo, "gamma", 0.99)
            return MAOffPolicyCollector(
                algorithm, vec_env, buffers,
                collect_state=needs_state,
                sync_mode=sync_replay and not is_q_recurrent,
                gamma=gamma,
                shared_indices=needs_shared_indices,
                shared_sync_mode=shared_sync_vdq,
            )
        else:
            if is_q_recurrent:
                burn_in_len = getattr(config.algo, "burn_in_len", 0)
                seq_len = getattr(config.algo, "seq_len", 20)
                max_seqs = getattr(config.algo, "max_episodes", 5000)
                buffer = SequenceReplayBuffer(
                    obs_dim=dims["obs_dim"],
                    seq_len=seq_len,
                    burn_in_len=burn_in_len,
                    max_seqs=max_seqs,
                    has_action_mask=True,
                    action_dim=dims["action_dim"],
                )
            else:
                buffer = ReplayBuffer(
                    obs_shape=(dims["obs_dim"],),
                    act_shape=(1,),
                    max_size=tc.buffer_size,
                    has_action_mask=True,
                    action_dim=dims["action_dim"],
                )
            return OffPolicyCollector(algorithm, vec_env, buffer)
    else:
        if get_policy_type(config.algo_name) == "mat":
            # MAT 专属：决策步缓冲 collector
            return MATOnPolicyCollector(algorithm, vec_env, dims["num_agents"])
        elif is_parallel:
            return MAOnPolicyCollector(algorithm, vec_env)
        else:
            return OnPolicyCollector(algorithm, vec_env)


# =============================================================================
#                          Q-table 独立训练路径
# =============================================================================

def _train_qtable(
    config: ExperimentConfig,
    eval_config_path: str = None,
    early_stopper=None,
):
    """Q-table 独立训练路径：复用日志/环境，绕过 nn.Module 流水线。

    根据环境类型自动分派：
    - ParallelEnv → QTableTrainer（per-agent 独立 Q-table）
    - Gymnasium Env → JointQTableTrainer（单一集中式 Q-table）
    """
    from algorithms.tabular.qtable import QTablePolicy, QTableAlgo
    from trainers.qtable_trainer import QTableTrainer, JointQTableTrainer

    tc = config.training

    if config.seed is not None:
        _set_training_random_seed(config.seed)

    config.save_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(config.log_dir)

    if config.track_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=asdict(config),
                # 与 sweep 一致：不同步 TB，避免重复曲线且横轴与 global_step 冲突
                sync_tensorboard=False,
            )
        _configure_wandb_env_step_axis()

    logger = SimpleLogger(tb_writer, use_wandb=config.track_wandb)

    vec_env = create_vec_env(
        env_type=config.env_type,
        env_config=config.env,
        num_envs=tc.num_envs,
        use_subproc=tc.use_subproc,
    )
    _maybe_seed_vec_env(vec_env, config.seed)
    vec_env.reset()

    def log_extra_fn() -> Dict[str, float]:
        try:
            metrics_list = vec_env.call_env_method("get_episode_metrics")
            finished = [m for m in metrics_list if m is not None]
            if finished:
                return _wandb_env_metrics_from_finished(finished)
        except Exception:
            pass
        return {}

    qtable_eval_meta = {
        "env_type": config.env_type,
        "agent_ids": list(vec_env.agents) if vec_env.is_parallel_env else ["agent_0"],
        # 训练时的 custom_configs（含 sweep 覆盖），供自动评估复用。
        "train_env_custom_configs": dict(config.env.custom_configs or {}),
    }

    inline_eval_fn = None
    if eval_config_path and tc.eval_interval > 0:
        from evaluators.test import eval_qtable_inline
        from configs.registry import load_eval_config
        from envs.venv_wrappers import VectorEnvNormObs, find_vec_wrapper

        _obs_norm_for_inline = find_vec_wrapper(vec_env, VectorEnvNormObs)
        _eval_env_type, _eval_env_cfg = load_eval_config(eval_config_path)
        _train_custom = dict(config.env.custom_configs or {})
        if _train_custom:
            _eval_env_cfg.custom_configs = {
                **(_eval_env_cfg.custom_configs or {}),
                **_train_custom,
            }
        _eval_episodes = tc.eval_episodes

        def inline_eval_fn():
            _rms = (
                _obs_norm_for_inline.get_obs_rms()
                if _obs_norm_for_inline is not None
                else None
            )
            return eval_qtable_inline(
                algo=algo,
                env_type=_eval_env_type,
                env_config=_eval_env_cfg,
                num_episodes=_eval_episodes,
                obs_rms=_rms,
            )

    if vec_env.is_parallel_env:
        # ParallelEnv：每个 agent 独立 Q-table
        action_dim = list(vec_env.action_space.values())[0].n
        agent_ids = vec_env.agents
        print(f"[Train] Q-table (parallel) mode: env={config.env_type}, "
              f"agents={agent_ids}, action_dim={action_dim}, num_envs={tc.num_envs}")
        policies = {
            aid: QTablePolicy(action_dim, config.algo.epsilon_start)
            for aid in agent_ids
        }
        algo = QTableAlgo(policies, config.algo)
        trainer = QTableTrainer(
            algo=algo, vec_env=vec_env, config=tc,
            save_dir=config.save_dir, logger=logger, log_extra_fn=log_extra_fn,
            eval_fn=inline_eval_fn, extra_info=qtable_eval_meta,
        )
    else:
        # Gymnasium Env：单一集中式 Q-table，虚拟 key "agent_0"
        action_dim = vec_env.action_space.n
        print(f"[Train] Q-table (joint) mode: env={config.env_type}, "
              f"action_dim={action_dim}, num_envs={tc.num_envs}")
        policies = {"agent_0": QTablePolicy(action_dim, config.algo.epsilon_start)}
        algo = QTableAlgo(policies, config.algo)
        trainer = JointQTableTrainer(
            algo=algo, vec_env=vec_env, config=tc,
            save_dir=config.save_dir, logger=logger, log_extra_fn=log_extra_fn,
            eval_fn=inline_eval_fn, extra_info=qtable_eval_meta,
        )

    if early_stopper is not None:
        _orig_lef = trainer.log_extra_fn

        def _patched_lef():
            metrics = _orig_lef() if _orig_lef else {}
            if metrics:
                early_stopper.record_metrics(metrics, trainer.total_steps)
            early_stopper.check_and_maybe_raise(trainer.total_steps)
            return metrics

        trainer.log_extra_fn = _patched_lef
        trainer.on_eval_complete = early_stopper.record_metrics

    print(f"\n[Train] Starting Q-table training")
    if tc.total_steps is not None:
        print(
            f"  Stop when total_steps >= {tc.total_steps} "
            f"(per-episode epsilon decay & Q-update unchanged)"
        )
        print(f"  max_iterations ({tc.max_iterations}) unused for stopping while total_steps is set")
    else:
        print(f"  Max episodes: {tc.max_iterations}")
    print(f"  Save dir: {config.save_dir}")

    trainer.train()

    if eval_config_path:
        from evaluators.test import run_eval_from_config
        qtable_dir = config.save_dir / "final"
        print(f"\n[Train] Starting auto-eval from {eval_config_path}")
        final_eval_metrics = run_eval_from_config(str(qtable_dir), eval_config_path)
        if final_eval_metrics:
            logger.log(final_eval_metrics, step=trainer.total_steps)
            if early_stopper is not None:
                early_stopper.record_metrics(final_eval_metrics, trainer.total_steps)
            if config.track_wandb:
                import wandb
                if wandb.run is not None:
                    for key, value in final_eval_metrics.items():
                        wandb.run.summary[key] = value

    tb_writer.close()
    if config.track_wandb:
        import wandb
        if wandb.run is not None and wandb.run.sweep_id is None:
            wandb.finish()
    vec_env.close()


# =============================================================================
#                              主训练函数
# =============================================================================

def train(config: ExperimentConfig, eval_config_path: str = None,
          early_stopper=None):
    """
    根据 ExperimentConfig 执行完整训练流程。

    流程：环境 → 维度推断 → 网络 → 预训练加载 → Policy → 算法 → Collector → Trainer → 训练

    Args:
        config: 实验配置。
        eval_config_path: 评估 YAML 路径；提供则训练结束后自动评估最终模型。
    """
    if config.algo_name == "qtable":
        _train_qtable(config, eval_config_path=eval_config_path, early_stopper=early_stopper)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = config.training
    trainer_type = get_trainer_type(config.algo_name)

    if config.seed is not None:
        _set_training_random_seed(config.seed)
        print(f"[Train] Random seed: {config.seed}")

    # ---- 1. 日志初始化 ----
    config.save_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(config.log_dir)

    if config.track_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=asdict(config),
                sync_tensorboard=False,
            )
        _configure_wandb_env_step_axis()

    logger = SimpleLogger(tb_writer, use_wandb=config.track_wandb)

    # ---- 2. 创建向量化环境 ----
    vec_env = create_vec_env(
        env_type=config.env_type,
        env_config=config.env,
        num_envs=tc.num_envs,
        use_subproc=tc.use_subproc,
    )
    _maybe_seed_vec_env(vec_env, config.seed)
    print(f"[Train] Created {tc.num_envs} vectorized {config.env_type} envs "
          f"({'subproc' if tc.use_subproc else 'dummy'})")

    # 若启用了 obs 归一化，保留内层 wrapper 引用，供 checkpoint 和 eval 使用
    from envs.venv_wrappers import VectorEnvNormObs, find_vec_wrapper
    _obs_norm_wrapper: Optional[VectorEnvNormObs] = (
        find_vec_wrapper(vec_env, VectorEnvNormObs)
    )

    # ---- 3. 推断维度 ----
    is_parallel = vec_env.is_parallel_env
    dims = _infer_dims(vec_env, config.algo_name)

    print(f"  Mode: {'parallel (MARL)' if is_parallel else 'single (centralized)'}")
    print(f"  Agents: {dims['agent_ids']}, Obs: {dims['obs_dim']}, "
          f"Act: {dims['action_dim']}, CriticIn: {dims['critic_input_dim']}")

    # ---- 4. 创建网络（三路独立） ----
    actor_net = None
    critic_net = None
    q_net = None

    if config.actor_type and config.actor is not None:
        actor_net = create_actor(
            actor_type=config.actor_type,
            actor_config=config.actor,
            obs_dim=dims["obs_dim"],
            action_dim=dims["action_dim"],
            device=str(device),
        )
    if config.critic_type and config.critic is not None:
        critic_net = create_critic(
            critic_type=config.critic_type,
            critic_config=config.critic,
            critic_input_dim=dims["critic_input_dim"],
            device=str(device),
        )
    if config.q_type and config.q_network is not None:
        q_input_dim = (
            dims["state_dim"] + dims["num_agents"]
            if get_policy_type(config.algo_name) == "actor"
            else dims["obs_dim"]
        )
        q_net = create_q_network(
            q_type=config.q_type,
            q_config=config.q_network,
            input_dim=q_input_dim,
            action_dim=dims["action_dim"],
            device=str(device),
        )

    # ---- 5. 加载预训练权重 ----
    value_norm_config = None
    if trainer_type == "on_policy":
        value_norm_config = _load_pretrained(
            actor_net, critic_net,
            config.actor_path, config.critic_path,
            device,
        )
    elif trainer_type == "off_policy" and config.actor_path and q_net is not None:
        # off-policy 加载模仿学习预训练的 Q-network 权重（复用 actor_path 字段）
        _load_pretrained(q_net, None, config.actor_path, None, device)
    else:
        print("[Train] Off-policy algorithm, skipping pretrained weight loading")

    # ---- 6. 构建 Policy ----
    policy = _build_policy(config, dims, is_parallel, actor_net=actor_net, q_net=q_net, device=device)

    # ---- 7. 构建算法 ----
    # context_kwargs 池: 放入所有可能需要的运行时参数，
    # create_algorithm 会根据算法 __init__ 签名自动过滤。
    context_kwargs = dict(
        num_envs=tc.num_envs,
        action_dim=dims["action_dim"],
        state_dim=dims["state_dim"],
        n_agents=dims["num_agents"],
    )

    if critic_net is not None:
        context_kwargs["critic"] = critic_net

    if q_net is not None:
        context_kwargs["q_network"] = q_net

    if isinstance(tc, OnPolicyTrainerConfig):
        context_kwargs["total_iterations"] = tc.effective_max_iterations
        context_kwargs["optimizer_steps_per_iter"] = tc.compute_optimizer_steps_per_iter(
            num_agents=dims["num_agents"]
        )

    if isinstance(tc, OffPolicyTrainerConfig):
        steps_per_collect = tc.num_envs * math.ceil(tc.collect_per_step / tc.num_envs)
        updates_per_iter = math.ceil(tc.step_per_iteration / steps_per_collect) * tc.update_per_step
        context_kwargs["total_updates"] = tc.effective_max_iterations * updates_per_iter

    if value_norm_config is not None:
        context_kwargs["value_norm_config"] = value_norm_config

    algorithm = create_algorithm(
        algo_name=config.algo_name,
        policy=policy,
        algo_params=config.algo,
        **context_kwargs,
    )

    # ---- 8. 构建 Collector ----
    collector = _build_collector(config, algorithm, vec_env, dims)
    profiler = TrainingStageProfiler(device)
    collector.profiler = profiler

    # ---- 9. 定义 Callbacks ----
    def _make_net_config_dict(net, input_dim, output_dim):
        if net is None:
            return None
        if hasattr(net, "get_config_dict"):
            return net.get_config_dict(input_dim, output_dim)
        raise NotImplementedError(
            f"{type(net).__name__} must implement get_config_dict(input_dim, output_dim) "
            f"and from_config_dict(cls, cfg) to support checkpoint saving/loading."
        )

    if get_policy_type(config.algo_name) == "mat":
        actor_config_dict = policy.get_config_dict(dims["obs_dim"], dims["action_dim"])
    else:
        actor_config_dict = _make_net_config_dict(
            actor_net, dims["obs_dim"], dims["action_dim"]
        )
    critic_config_dict = _make_net_config_dict(critic_net, dims["critic_input_dim"], 1)
    q_config_dict = _make_net_config_dict(q_net, dims["obs_dim"], dims["action_dim"])

    # 评估时需要的元信息，写入 config.yaml 的 extra 段
    _eval_meta = {
        "algo_name": config.algo_name,
        "env_type": config.env_type,
        "policy_type": get_policy_type(config.algo_name),
        "shared_policy": getattr(config.algo, "shared_policy", True),
        "agent_ids": dims["agent_ids"],
        # 训练时的 custom_configs（含 sweep 覆盖后的 idi_scale / contribution_scale 等）
        # 评估时会从 config.yaml 读回并覆盖 eval yaml 里的同名字段，确保评估环境与训练一致
        "train_env_custom_configs": dict(config.env.custom_configs or {}),
    }

    def _get_value_norm_config():
        if hasattr(algorithm, "use_value_norm") and algorithm.use_value_norm and algorithm.ret_rms is not None:
            return {
                "enabled": True,
                "ret_mean": float(algorithm.ret_rms.mean.item()),
                "ret_std": float(algorithm.ret_rms.std.item()),
                "ret_count": float(algorithm.ret_rms.count.item()),
            }
        return {"enabled": False}

    def _get_obs_rms_state():
        """序列化当前 obs_rms 统计量，供 checkpoint 写入和评估时还原。"""
        if _obs_norm_wrapper is None:
            return None
        from utils.model_io import running_mean_std_to_yaml_dict

        return running_mean_std_to_yaml_dict(_obs_norm_wrapper.get_obs_rms())

    def save_checkpoint_fn(iteration: int):
        ckpt_dir = config.save_dir / f"iter_{iteration}"
        save_model(
            save_dir=ckpt_dir,
            policy=policy,
            critic=critic_net,
            actor_config=actor_config_dict,
            critic_config=critic_config_dict,
            q_config=q_config_dict,
            extra_info={
                **_eval_meta,
                "iteration": iteration,
                "value_normalization": _get_value_norm_config(),
                "obs_rms_state": _get_obs_rms_state(),
            },
        )
        print(f"[Checkpoint] Saved iteration {iteration} to {ckpt_dir}")

    def log_extra_fn() -> Dict[str, float]:
        try:
            metrics_list = vec_env.call_env_method("get_episode_metrics")
            finished = [m for m in metrics_list if m is not None]
            if finished:
                return _wandb_env_metrics_from_finished(finished)
        except Exception:
            pass
        return {}

    # ---- 10. 构建 Trainer ----
    # 若配置了 eval_config_path 且 eval_interval > 0，构建 inline eval 回调
    inline_eval_fn = None
    if eval_config_path and tc.eval_interval > 0:
        from evaluators.test import eval_policy_inline
        from configs.registry import load_eval_config
        _eval_env_type, _eval_env_cfg = load_eval_config(eval_config_path)
        # 用训练时的 custom_configs 覆盖 eval yaml，保持 idi_scale 等参数一致
        _train_custom = dict(config.env.custom_configs or {})
        if _train_custom:
            _eval_env_cfg.custom_configs = {
                **(_eval_env_cfg.custom_configs or {}),
                **_train_custom,
            }
        _eval_episodes = tc.eval_episodes
        _eval_device = device

        def inline_eval_fn():
            return eval_policy_inline(
                policy=policy,
                env_type=_eval_env_type,
                env_config=_eval_env_cfg,
                num_episodes=_eval_episodes,
                device=_eval_device,
                obs_rms=_obs_norm_wrapper.get_obs_rms() if _obs_norm_wrapper is not None else None,
            )

    trainer = create_trainer(
        algo_name=config.algo_name,
        algorithm=algorithm,
        collector=collector,
        training_config=tc,
        save_checkpoint_fn=save_checkpoint_fn,
        log_extra_fn=log_extra_fn,
        logger=logger,
        eval_fn=inline_eval_fn,
        profiler=profiler,
    )

    # ---- 10.5. 若提供 early_stopper，patch trainer 的 log_extra_fn + on_eval_complete ----
    if early_stopper is not None:
        _orig_lef = trainer.log_extra_fn
        def _patched_lef():
            metrics = _orig_lef() if _orig_lef else {}
            if metrics:
                early_stopper.record_metrics(metrics, trainer.total_steps)
            # 每轮 log 都检查 slope；避免长 episode 下多轮无 env 指标时永远不触发 within-trial 逻辑
            early_stopper.check_and_maybe_raise(trainer.total_steps)
            return metrics
        trainer.log_extra_fn = _patched_lef
        # 将 inline eval 产生的 eval/... 指标也喂给 early stopper
        # 这样当 sweep metric 为 eval/wi_fromT 时，cross-trial 和 slope 检查均有数据来源
        trainer.on_eval_complete = early_stopper.record_metrics

    # ---- 11. 训练 ----
    print(f"\n[Train] Starting {config.algo_name.upper()} training")
    if tc.total_steps is not None:
        print(
            f"  total_steps budget: {tc.total_steps} → "
            f"max_iterations={tc.effective_max_iterations} "
            f"(ceil({tc.total_steps} / {tc.step_per_iteration}))"
        )
    else:
        print(f"  Max iterations: {tc.effective_max_iterations}")
    if isinstance(tc, OnPolicyTrainerConfig):
        batch_size = tc.step_per_iteration * dims["num_agents"]
        print(f"  Batch size: {batch_size}, Minibatch: {tc.minibatch_size}, "
              f"Epochs: {tc.update_epochs}")
    elif isinstance(tc, OffPolicyTrainerConfig):
        print(f"  Buffer size: {tc.buffer_size}, Batch size: {tc.batch_size}, "
              f"Warmup: {tc.warmup_steps}")
    print(f"  Device: {device}")
    print(f"  Profiling: enabled (cuda_sync={profiler.sync_cuda})")
    print(f"  Save dir: {config.save_dir}")

    final_dir = config.save_dir / "final"

    def _save_final_checkpoint(extra_info: dict):
        save_model(
            save_dir=final_dir,
            policy=policy,
            critic=critic_net,
            actor_config=actor_config_dict,
            critic_config=critic_config_dict,
            q_config=q_config_dict,
            extra_info={
                **_eval_meta,
                "value_normalization": _get_value_norm_config(),
                "obs_rms_state": _get_obs_rms_state(),
                **extra_info,
            },
        )

    try:
        trainer.train()
    except SweepEarlyStop as e:
        _save_final_checkpoint(
            {
                "iteration": getattr(trainer, "iteration", 0),
                "total_env_steps": getattr(trainer, "total_steps", 0),
                "early_stopped": True,
                "early_stop_reason": str(e),
            },
        )
        print(f"[Train] Sweep early stop — saved checkpoint to {final_dir}: {e}")
        raise
    else:
        _save_final_checkpoint({"iteration": tc.effective_max_iterations})
        print(f"[Train] Saved final model to {final_dir}")
    finally:
        tb_writer.close()
        if config.track_wandb:
            import wandb
            if wandb.run is not None and wandb.run.sweep_id is None:
                wandb.finish()
        vec_env.close()

    # 训练结束后自动评估（早停 re-raise 不会执行到此）
    if eval_config_path:
        from evaluators.test import run_eval_from_config
        print(f"\n[Train] Starting auto-eval from {eval_config_path}")
        run_eval_from_config(str(final_dir), eval_config_path)


# =============================================================================
#                              入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAP 统一训练入口")
    parser.add_argument(
        "config",
        type=str,
        help="YAML 配置文件路径 (例: configs/experiments/mappo_tsp12_imi.yaml)",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="评估配置 YAML 路径；提供则训练结束后自动评估最终模型",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[Train] Loaded config from {args.config}")
    eval_path = args.eval_config or getattr(cfg, "eval_config_path", None)
    train(cfg, eval_config_path=eval_path)
