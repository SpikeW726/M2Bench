#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 RL 策略评估脚本。

支持所有算法 (MAPPO/IPPO/IQL/D3QN/VDN/QMIX 等)、
所有网络 (MLP/RNN) 和所有 MDP 环境 (MASUPEnv/S4R1Env/OUCSEnv 等)。

模型重建信息来自模型目录的 config.yaml (训练时自动保存);
环境类型和参数来自 eval YAML。

用法示例:
  1) 基础评估
     python evaluators/test.py --model models/iql-s4r1-TSP12/2026-03-01_17-40-36/final \
                               --env_config configs/eval/s4r1/s4r1_tsp12.yaml

  2) 生成最后一个 episode 动画
     python evaluators/test.py --model models/mappo-masup-TSP12/2026-03-01_10-20-30/final \
                               --env_config configs/eval/masup/masup_tsp12.yaml \
                               --animation --max_frames 500

  3) 仅保存图，不弹窗（同目录会额外写入与图一致的 *_plot_data.csv，供多 run 叠加曲线）
     python evaluators/test.py --model models/xxx/final \
                               --env_config configs/eval/suns/suns_tsp12.yaml \
                               --save_plot evaluators/results/eval.png --no_show

  4) 记录每次决策的 logits（或 Q 值）以观察动作区分度
     python evaluators/test.py --model models/xxx/final --env_config configs/eval/masup/masup_SF.yaml \
                               --log_action_logits --action_logits_csv evaluators/results/action_logits.csv
     或在 eval yaml 的 eval 段设置 log_action_logits / action_logits_csv。
"""
import os
import sys
from pathlib import Path
from typing import TextIO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import csv
import yaml
import torch
import torch.nn.functional as F
import numpy as np

from configs.registry import (
    ENV_REGISTRY,
    _import_class,
    load_eval_config,
    _env_config_to_dicts,
)
from policies.marl.marl_base import MultiAgentPolicy
from policies.marl.mat_policy import MATMultiAgentPolicy
from utils.model_io import load_policy_for_eval, get_model_config
from utils.log_utils import aggregate_episode_metrics, plot_aggregated_metrics
from utils.autodl_paths import (
    AUTODL_MODELS_ROOT,
    AUTODL_RESULTS_ROOT,
    resolve_models_path,
    resolve_results_path,
)


# MASUP / MASUPGraphEnv：第四幅子图用 wi_fromT 替代全 episode 的 wi
# BEAU 无 T_time，不提供 _wi_fromT_history，不纳入此集合
_MASUP_LIKE_ENV_TYPES = frozenset({"masup", "masup_gnn"})


def _eval_metrics_history_for_plot(env, env_type: str) -> dict:
    hist = env.world.metrics_tracker.get_history_dict()
    if env_type not in _MASUP_LIKE_ENV_TYPES:
        return hist
    wfh = getattr(env, "_wi_fromT_history", None)
    tlen = len(hist.get("time", []))
    if wfh is not None and len(wfh) == tlen:
        h = dict(hist)
        h["wi_fromT"] = list(wfh)
        return h
    return hist


def _beau_mat_rollout_step(
    env,
    mat_policy: MATMultiAgentPolicy,
    infos: dict,
    last_shift: torch.Tensor,
) -> tuple[dict, dict, torch.Tensor]:
    """单步：图状态 + 联合采样 + graph_idx_to_action，供 BEAU+MAT 评估/对齐训练 collector。"""
    assert hasattr(env, "state_mat") and hasattr(env, "graph_idx_to_action")
    n = mat_policy.n_agents
    active = np.array(
        [float(infos[aid].get("active_mask", 1.0)) for aid in env.possible_agents],
        dtype=np.float32,
    )
    gs = env.state_mat()
    ni = env.get_current_node_indices()
    gs_b = gs[np.newaxis, ...]
    ni_b = ni[np.newaxis, ...]
    am_b = active[np.newaxis, ...]
    with torch.no_grad():
        act_np, _, _, shift_new = mat_policy.compute_joint_actions(
            gs_b, ni_b, am_b, last_shift, deterministic=True,
        )
    g_list = [int(x) for x in act_np[0].tolist()]
    raw_map = env.graph_idx_to_action(g_list)
    actions = {k: int(v) for k, v in raw_map.items()}
    obs, _, _, _, infos2 = env.step(actions)
    return obs, infos2, shift_new


def _summary_from_episode_metrics(episode_metrics, wi_fromT_finals: list | None = None) -> dict:
    """统一汇总为 logger / wandb 友好的 eval/* 指标。"""
    result = {
        "eval/igi": float(np.mean([m.igi for m in episode_metrics])),
        "eval/agi": float(np.mean([m.agi for m in episode_metrics])),
        "eval/iwi": float(np.mean([m.iwi for m in episode_metrics])),
        "eval/wi": float(np.mean([m.wi for m in episode_metrics])),
    }
    if wi_fromT_finals:
        result["eval/wi_fromT"] = float(np.mean(wi_fromT_finals))
    return result


def _policy_output_to_action_scores(out: dict) -> tuple[torch.Tensor | None, str]:
    """从 policy.forward 结果取出用于区分动作的标量向量（actor=logits, value=Q）。"""
    if out.get("logits") is not None:
        return out["logits"], "actor_logits"
    if out.get("q_values") is not None:
        return out["q_values"], "q_values"
    return None, ""


def _squeeze_scores(scores: torch.Tensor) -> torch.Tensor:
    s = scores.detach().float()
    if s.dim() > 1:
        s = s.squeeze(0)
    return s.view(-1)


def _valid_action_stats(scores_1d: torch.Tensor, mask_1d: torch.Tensor) -> dict:
    """仅在合法动作上计算 softmax 熵、top-2 logit 间隔等。"""
    valid = scores_1d[mask_1d]
    if valid.numel() == 0:
        return {
            "n_valid": 0,
            "max_logit": float("nan"),
            "min_logit": float("nan"),
            "logit_gap_top2": float("nan"),
            "entropy": float("nan"),
            "max_prob": float("nan"),
        }
    n = int(valid.numel())
    max_logit = float(valid.max().item())
    min_logit = float(valid.min().item())
    vs = torch.sort(valid, descending=True).values
    logit_gap_top2 = float(vs[0] - vs[1]) if n >= 2 else float("nan")
    logp = F.log_softmax(valid, dim=0)
    p = logp.exp()
    entropy = float((-(p * logp)).sum().item())
    max_prob = float(p.max().item())
    return {
        "n_valid": n,
        "max_logit": max_logit,
        "min_logit": min_logit,
        "logit_gap_top2": logit_gap_top2,
        "entropy": entropy,
        "max_prob": max_prob,
    }


def _open_action_logits_csv(path: str, action_dim: int) -> tuple[csv.DictWriter, TextIO]:
    """创建 CSV 并写入表头；返回 (writer, file_handle) 由调用方关闭。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "w", newline="", encoding="utf-8")
    logits_cols = [f"logit_a{i}" for i in range(action_dim)]
    fieldnames = [
        "episode",
        "step",
        "sim_time",
        "agent_id",
        "policy_head",
        "chosen_action",
        "active_mask",
        "n_valid",
        "max_logit",
        "min_logit",
        "logit_gap_top2",
        "entropy",
        "max_prob",
    ] + logits_cols
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return w, f


def _record_qtable_action_scores(
    *,
    writer,
    do_print: bool,
    print_count: int,
    print_limit: int,
    action_dim: int,
    episode: int,
    step: int,
    sim_time: float,
    agent_id: str,
    q_values: np.ndarray,
    action_mask,
    chosen_action: int,
    active: int,
    active_only: bool,
) -> int:
    """记录 Q-table 决策时的合法动作 Q 值与区分度统计。"""
    if writer is None and not do_print:
        return print_count
    if active_only and active != 1:
        return print_count

    q = np.asarray(q_values, dtype=np.float32).reshape(-1)
    if action_mask is None:
        mask = np.ones_like(q, dtype=bool)
    else:
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.size != q.size:
            return print_count

    scores_t = torch.as_tensor(q, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    st = _valid_action_stats(scores_t, mask_t)

    row = {
        "episode": episode,
        "step": step,
        "sim_time": f"{sim_time:.6f}",
        "agent_id": agent_id,
        "policy_head": "q_values",
        "chosen_action": chosen_action,
        "active_mask": active,
        "n_valid": st["n_valid"],
        "max_logit": f"{st['max_logit']:.6f}",
        "min_logit": f"{st['min_logit']:.6f}",
        "logit_gap_top2": f"{st['logit_gap_top2']:.6f}",
        "entropy": f"{st['entropy']:.6f}",
        "max_prob": f"{st['max_prob']:.6f}",
    }
    for i in range(action_dim):
        key = f"logit_a{i}"
        if i < q.size and i < mask.size and bool(mask[i]):
            row[key] = f"{float(q[i]):.6f}"
        else:
            row[key] = ""

    if writer is not None:
        writer.writerow(row)

    if do_print and print_count < print_limit:
        parts = [
            f"{float(q[i]):.3f}" if i < mask.size and bool(mask[i]) else "—"
            for i in range(min(q.size, action_dim))
        ]
        print(
            f"[action_scores] ep={episode} step={step} t={sim_time:.2f} "
            f"{agent_id} q_values a={chosen_action} "
            f"H={st['entropy']:.3f} max_p={st['max_prob']:.3f} "
            f"gap2={st['logit_gap_top2']:.3f} scores=[{', '.join(parts)}]"
        )
        return print_count + 1
    return print_count


def _eval_plot_metric_configs(env_type: str, sample_hist: dict):
    base = [
        ("igi", "Instantaneous Graph Idleness (IGI)", "IGI"),
        ("agi", "Average Graph Idleness (AGI)", "AGI"),
        ("iwi", "Instantaneous Worst Idleness (IWI)", "IWI"),
    ]
    if env_type in _MASUP_LIKE_ENV_TYPES and sample_hist and "wi_fromT" in sample_hist:
        return base + [
            (
                "wi_fromT",
                "Worst Idleness from T (max weighted IWI after T_time)",
                "WI from T",
            ),
        ]
    return base + [("wi", "Worst Idleness (WI)", "WI")]


# =============================================================================
#                          环境创建
# =============================================================================

def _create_env(env_type: str, env_config):
    """通过 ENV_REGISTRY 动态创建单个环境实例。"""
    entry = ENV_REGISTRY[env_type]
    env_cls = _import_class(entry["module"], entry["class_name"])
    cfg_dict, custom_dict = _env_config_to_dicts(env_config)
    return env_cls(cfg_dict, **custom_dict)


# =============================================================================
#                          评估主函数
# =============================================================================

def test_trained_policy(
    model_dir: str,
    env_config_path: str,
    num_episodes: int = 5,
    episode_time: float = None,
    save_plot: str = None,
    show_plot: bool = True,
    record_animation: bool = False,
    event_driven: bool = True,
    max_frames: int = None,
    save_animation_dir: str = None,
    env_custom_config_overrides: dict = None,
    log_action_logits: bool = False,
    log_action_logits_max_lines: int = 500,
    action_logits_csv: str = None,
    action_logits_active_only: bool = True,
):
    """
    评估训练好的策略，自动适配算法/网络/环境类型。

    Args:
        model_dir: 模型目录 (含 config.yaml + policy.pt)
        env_config_path: eval YAML 路径 (含 env_type + env 参数)
        num_episodes: 评估 episode 数量
        episode_time: 每个 episode 的仿真时间上限（秒）；
                      None 时从 env 配置的 episode_len 读取，依靠环境自身的
                      truncated/terminated 信号结束 episode
        save_plot: 图表保存路径
        show_plot: 是否显示图表
        record_animation: 是否录制最后一个 episode 的动画
        event_driven: True=事件驱动动画，False=固定步长动画
        max_frames: 动画最大帧数限制
        save_animation_dir: 动画保存目录；None 时回退到 save_plot 的父目录
        env_custom_config_overrides: 覆盖 eval yaml 中 custom_configs 的字段（优先级高于 yaml）；
            通常由 run_eval_from_config 从模型 config.yaml 的 train_env_custom_configs 中读取，
            以确保 idi_scale / contribution_scale 等 sweep 参数与训练时保持一致。
        log_action_logits: 是否在 stdout 打印每次（有效）决策的动作分数向量及区分度统计。
        log_action_logits_max_lines: 打印行数上限，避免刷屏；超出后仅提示截断。
        action_logits_csv: 若提供路径，将所有记录行写入 CSV（含各动作 logit/q 列）。
        action_logits_active_only: True 时仅记录 active_mask=1 的步（真正决策时刻）。
        动画文件名：若提供 save_plot，在「算法名_animation_图名」后追加 _ 与 save_plot 的文件名 stem，
        例如 best_eval.png → mappo_animation_long_edge_best_eval.mp4
    """
    model_dir = resolve_models_path(model_dir)
    if save_plot is not None:
        save_plot = resolve_results_path(save_plot)
    if save_animation_dir is not None:
        save_animation_dir = resolve_results_path(save_animation_dir)

    # ---- 1. 加载环境配置 & 创建环境 ----
    env_type, env_cfg = load_eval_config(env_config_path)

    # 用训练时的 custom_configs 覆盖 eval yaml 中的同名字段（如 idi_scale / contribution_scale）
    if env_custom_config_overrides:
        env_cfg.custom_configs = {**(env_cfg.custom_configs or {}), **env_custom_config_overrides}
        print(f"[Eval] Merged train-time custom_configs: {env_custom_config_overrides}")

    print(f"\n=== Creating {env_type} environment ===")
    env = _create_env(env_type, env_cfg)

    cfg_dict, _ = _env_config_to_dicts(env_cfg)
    graph_path = cfg_dict.get("graph_path", "unknown")
    graph_name = Path(graph_path).stem
    num_agents = cfg_dict.get("num_agents", len(env.possible_agents))

    # episode 时间上限：优先用显式传入的 episode_time，否则读 env 配置的 episode_len
    eff_episode_time = episode_time if episode_time is not None else cfg_dict.get("episode_len", None)

    print(f"Graph: {graph_path} ({graph_name})")
    print(f"Num agents: {num_agents}")

    # ---- 2. 从环境推断维度 ----
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_space(sample_agent)
    action_space = env.action_space(sample_agent)
    obs_dim = obs_space.shape[0]
    action_dim = action_space.n

    print(f"\n=== Network Dimensions ===")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # ---- 3. 加载策略 & obs_rms ----
    model_dir = Path(model_dir)
    model_config = get_model_config(model_dir)
    extra = model_config.get("extra", {})
    algo_name = extra.get("algo_name", "unknown")

    print(f"\n=== Loading model ({algo_name}) from {model_dir} ===")
    multi_policy = load_policy_for_eval(
        model_dir=model_dir,
        agent_ids=env.possible_agents,
        obs_space=obs_space,
        action_space=action_space,
        device='cpu',
    )

    # 加载训练时的 obs_rms（若训练时未开启 norm_obs，则为 None，评估循环不做归一化）
    from utils.model_io import load_obs_rms
    _eval_obs_rms = load_obs_rms(model_dir)
    if _eval_obs_rms is not None:
        print(f"[Eval] obs_rms loaded: mean≈{float(np.mean(_eval_obs_rms.mean)):.3f}, "
              f"std≈{float(np.mean(np.sqrt(_eval_obs_rms.var))):.3f}, count={_eval_obs_rms.count:.0f}")

    # ---- 4. 评估循环 ----
    time_desc = f"{eff_episode_time:.0f}s" if eff_episode_time is not None else "env-truncated"
    print(f"\n=== Running {num_episodes} episodes (episode_time={time_desc}) ===")

    do_logits_file = bool(action_logits_csv)
    do_logits_print = bool(log_action_logits)
    logits_csv_w = None
    logits_csv_f = None
    logits_print_count = 0
    if do_logits_file:
        logits_csv_w, logits_csv_f = _open_action_logits_csv(action_logits_csv, action_dim)
        print(f"[Eval] Writing per-decision action scores to {action_logits_csv}")
    if do_logits_print or do_logits_file:
        print(
            "[Eval] action_logits: actor→logits, value-based→Q；"
            f"active_only={action_logits_active_only}；"
            f"entropy/gap 仅在合法动作上按 softmax 计算。"
        )

    is_mat_policy = isinstance(multi_policy, MATMultiAgentPolicy)
    if is_mat_policy and (do_logits_print or do_logits_file):
        print("[Eval] BEAU+MAT 不支持 action_logits 记录，已跳过。")
        do_logits_file = do_logits_print = False
        logits_csv_w = logits_csv_f = None

    episode_metrics = []
    metrics_history = []
    episode_times = []
    wi_fromT_finals: list = []

    anim_positions_history = []
    anim_time_intervals = []

    for ep in range(num_episodes):
        obs, infos = env.reset()
        step_count = 0
        hidden_state = None  # episode 开始时重置 RNN hidden state

        is_last = (ep == num_episodes - 1)
        record = record_animation and is_last
        if record:
            anim_positions_history = [env.world.snapshot_agent_positions()]
            anim_time_intervals = []

        if is_mat_policy:
            _last_sh = torch.zeros(
                1, multi_policy.n_agents, multi_policy.n_agents, 2,
                dtype=torch.float32, device=multi_policy.device,
            )
        else:
            _last_sh = None

        # 以仿真时间为截止条件；如果未配置则依靠环境的 truncated/terminated 信号
        while env.agents:
            if eff_episode_time is not None and env.world.current_time >= eff_episode_time:
                break

            if is_mat_policy:
                obs, infos, _last_sh = _beau_mat_rollout_step(
                    env, multi_policy, infos, _last_sh
                )
                step_count += 1
                if record:
                    anim_time_intervals.append(getattr(env, "last_time_interval", 1.0))
                    anim_positions_history.append(env.world.snapshot_agent_positions())
                continue

            action_masks = {
                aid: torch.as_tensor(info['action_mask'], dtype=torch.bool, device=multi_policy.device)
                for aid, info in infos.items()
            }
            # obs 归一化：若训练时开启了 norm_obs，使用保存的 RMS 统计量
            if _eval_obs_rms is not None:
                obs_normed = {aid: _eval_obs_rms.norm(np.asarray(o)) for aid, o in obs.items()}
            else:
                obs_normed = obs
            obs_tensor = {
                aid: torch.as_tensor(o, dtype=torch.float32, device=multi_policy.device)
                for aid, o in obs_normed.items()
            }

            # BEAU: set _current_node_idx on actor before forward
            if hasattr(multi_policy, '_shared_policy') and hasattr(
                multi_policy._shared_policy, 'actor'
            ) and hasattr(multi_policy._shared_policy.actor, 'compute_value'):
                actor_net = multi_policy._shared_policy.actor
                n_agents_eval = len(env.agents)
                cn_idx_2d = np.zeros((n_agents_eval, 1), dtype=np.int64)
                for a_idx, aid in enumerate(env.agents):
                    cn_idx_2d[a_idx, 0] = infos[aid].get('current_node_idx', 0)
                actor_net._current_node_idx = torch.as_tensor(
                    cn_idx_2d, dtype=torch.long, device=multi_policy.device,
                )

            with torch.no_grad():
                outputs = multi_policy.forward(obs_tensor, state_dict=hidden_state, action_mask=action_masks)
            actions = {aid: out['act'].cpu().numpy() for aid, out in outputs.items()}

            if do_logits_print or do_logits_file:
                sim_time = float(getattr(getattr(env, "world", None), "current_time", step_count))
                for aid, out in outputs.items():
                    info_i = infos.get(aid) or {}
                    active = int(info_i.get("active_mask", 1))
                    if action_logits_active_only and active != 1:
                        continue
                    scores_t, head = _policy_output_to_action_scores(out)
                    if scores_t is None:
                        continue
                    s1 = _squeeze_scores(scores_t)
                    mask_t = action_masks[aid]
                    m1 = mask_t.detach().bool().cpu().view(-1)
                    if s1.numel() != m1.numel():
                        continue
                    st = _valid_action_stats(s1, m1)
                    act_t = out["act"]
                    chosen_action = int(act_t.detach().view(-1)[0].item())

                    row = {
                        "episode": ep,
                        "step": step_count,
                        "sim_time": f"{sim_time:.6f}",
                        "agent_id": aid,
                        "policy_head": head,
                        "chosen_action": chosen_action,
                        "active_mask": active,
                        "n_valid": st["n_valid"],
                        "max_logit": f"{st['max_logit']:.6f}",
                        "min_logit": f"{st['min_logit']:.6f}",
                        "logit_gap_top2": f"{st['logit_gap_top2']:.6f}",
                        "entropy": f"{st['entropy']:.6f}",
                        "max_prob": f"{st['max_prob']:.6f}",
                    }
                    for i in range(action_dim):
                        key = f"logit_a{i}"
                        if i < s1.numel() and bool(m1[i].item()):
                            row[key] = f"{float(s1[i].item()):.6f}"
                        else:
                            row[key] = ""

                    if logits_csv_w is not None:
                        logits_csv_w.writerow(row)

                    if do_logits_print and logits_print_count < log_action_logits_max_lines:
                        parts = []
                        for i in range(min(s1.numel(), action_dim)):
                            if bool(m1[i].item()):
                                parts.append(f"{float(s1[i].item()):.3f}")
                            else:
                                parts.append("—")
                        scores_compact = "[" + ", ".join(parts) + "]"
                        print(
                            f"[action_scores] ep={ep} step={step_count} t={sim_time:.2f} "
                            f"{aid} {head} a={chosen_action} "
                            f"H={st['entropy']:.3f} max_p={st['max_prob']:.3f} "
                            f"gap2={st['logit_gap_top2']:.3f} scores={scores_compact}"
                        )
                        logits_print_count += 1

            # 保留 RNN hidden state 供下一步使用
            if multi_policy.is_recurrent:
                hidden_state = {aid: out['state'] for aid, out in outputs.items() if out.get('state') is not None}

            obs, _, _, _, infos = env.step(actions)
            step_count += 1

            if record:
                anim_time_intervals.append(getattr(env, 'last_time_interval', 1.0))
                anim_positions_history.append(env.world.snapshot_agent_positions())

        # Episode 结束: 收集指标
        final_metrics = env.world.current_metrics
        episode_metrics.append(final_metrics)
        episode_times.append(final_metrics.time)
        metrics_history.append(_eval_metrics_history_for_plot(env, env_type))
        if env_type in _MASUP_LIKE_ENV_TYPES:
            wi_fromT_finals.append(float(getattr(env, "worst_idleness_fromT", 0.0)))

        ep_line = (
            f"Episode {ep + 1}/{num_episodes}: "
            f"IGI={final_metrics.igi:.4f}, "
            f"AGI={final_metrics.agi:.4f}, "
            f"IWI={final_metrics.iwi:.4f}, "
            f"WI={final_metrics.wi:.4f}, "
            f"time={final_metrics.time:.2f}s, "
            f"steps={step_count}"
        )
        if env_type in _MASUP_LIKE_ENV_TYPES:
            ep_line += f", WI@T={getattr(env, 'worst_idleness_fromT', 0.0):.4f}"
        print(ep_line)

    if logits_csv_f is not None:
        logits_csv_f.close()
        print(f"[Eval] action_logits CSV saved: {action_logits_csv}")

    if (
        do_logits_print
        and log_action_logits_max_lines > 0
        and logits_print_count >= log_action_logits_max_lines
    ):
        print(
            f"[Eval] action_scores 打印已截断（仅前 {log_action_logits_max_lines} 行）；"
            "完整记录请使用 action_logits_csv。"
        )

    # ---- 5. 汇总统计 ----
    print(f"\n=== Summary Statistics ({num_episodes} episodes, episode_time={time_desc}) ===")
    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    if env_type in _MASUP_LIKE_ENV_TYPES and wi_fromT_finals:
        print(
            f"WI@T: {np.mean(wi_fromT_finals):.4f} ± {np.std(wi_fromT_finals):.4f} "
            f"(worst_idleness_fromT at episode end)"
        )

    # ---- 6. 可视化 ----
    # 动画目录：优先用显式传入的 save_animation_dir，回退到 save_plot 的父目录
    _default_results = str(AUTODL_RESULTS_ROOT)
    anim_dir = save_animation_dir or (str(Path(save_plot).parent) if save_plot else _default_results)
    anim_plot_stem = Path(save_plot).stem if save_plot else None

    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"
        _mcfg = _eval_plot_metric_configs(
            env_type, metrics_history[0] if metrics_history else {}
        )
        plot_aggregated_metrics(
            aggregated,
            metric_configs=_mcfg,
            title=f'{algo_name.upper()} Evaluation ({num_episodes} episodes, {time_desc})',
            subtitle=subtitle,
            save_path=save_plot,
            show=show_plot,
        )

    # ---- 7. 动画 ----
    if record_animation and anim_positions_history:
        print(f"\n=== Generating animation for last episode ===")
        algorithm_name = algo_name or Path(model_dir).stem
        if event_driven:
            from utils.vis_utils import create_event_driven_animation
            create_event_driven_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                time_intervals=anim_time_intervals,
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=anim_dir,
                max_frames=max_frames,
                plot_stem=anim_plot_stem,
            )
        else:
            from utils.vis_utils import create_animation
            create_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                total_frames=len(anim_positions_history),
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=anim_dir,
                max_frames=max_frames,
                plot_stem=anim_plot_stem,
            )

    if hasattr(env, "close"):
        env.close()
    return episode_metrics


# =============================================================================
#                     Q-table 评估（BBLA / GBLA / ExGBLA 等）
# =============================================================================

def test_qtable_policy(
    model_dir: str,
    env_config_path: str,
    num_episodes: int = 5,
    save_plot: str = None,
    show_plot: bool = True,
    record_animation: bool = False,
    event_driven: bool = True,
    max_frames: int = None,
    save_animation_dir: str = None,
    env_custom_config_overrides: dict = None,
    log_action_logits: bool = False,
    log_action_logits_max_lines: int = 500,
    action_logits_csv: str = None,
    action_logits_active_only: bool = True,
):
    """评估训练好的 Q-table 策略。

    支持 ParallelEnv（BBLA / GBLA / ExGBLA，per-agent 独立 Q-table）
    和 Gymnasium Env（JointBaseEnv 系列，单一集中式 Q-table）。

    Args:
        model_dir: checkpoint 目录（含 *_qtable.npy 文件）
        env_config_path: eval YAML（含 env_type + env 参数 + algo_name: qtable）
        num_episodes: 评估 episode 数
        save_plot: 指标图保存路径（None = 不保存）
        show_plot: 是否弹窗显示图表
        record_animation: 是否录制最后一个 episode 的动画
        event_driven: True=事件驱动动画，False=固定步长动画
        max_frames: 动画最大帧数限制
        save_animation_dir: 动画保存目录；None 时回退到 save_plot 的父目录
        动画文件名：若提供 save_plot，在「算法名_animation_图名」后追加 _stem（同 test_trained_policy）
    """
    model_dir = resolve_models_path(model_dir)
    if save_plot is not None:
        save_plot = resolve_results_path(save_plot)
    if save_animation_dir is not None:
        save_animation_dir = resolve_results_path(save_animation_dir)
    if action_logits_csv is not None:
        action_logits_csv = resolve_results_path(action_logits_csv)

    from algorithms.tabular.qtable import QTablePolicy, QTableAlgo
    from configs.algo_configs import QTableParams

    # ---- 1. 创建环境 ----
    env_type, env_cfg = load_eval_config(env_config_path)
    if env_custom_config_overrides:
        env_cfg.custom_configs = {**(env_cfg.custom_configs or {}), **env_custom_config_overrides}
        print(f"[Eval] Merged train-time custom_configs: {env_custom_config_overrides}")
    print(f"\n=== Creating {env_type} environment ===")
    env = _create_env(env_type, env_cfg)

    cfg_dict, _ = _env_config_to_dicts(env_cfg)
    graph_path = cfg_dict.get("graph_path", "unknown")
    graph_name = Path(graph_path).stem
    num_agents = cfg_dict.get("num_agents", None)

    print(f"Graph: {graph_path} ({graph_name})")

    # ---- 2. 检测环境类型并构建 Q-table ----
    is_parallel = hasattr(env, "possible_agents")  # ParallelEnv vs Gymnasium Env

    if is_parallel:
        agent_ids = env.possible_agents
        action_dim = env.action_space(agent_ids[0]).n
        num_agents = num_agents or len(agent_ids)
    else:
        agent_ids = ["agent_0"]
        action_dim = env.action_space.n
        num_agents = num_agents or 1

    print(f"Num agents: {num_agents}, Action dim: {action_dim}")
    print(f"Mode: {'ParallelEnv (per-agent Q-table)' if is_parallel else 'GymnasiumEnv (joint Q-table)'}")

    # 评估时 epsilon=0（纯 greedy），lr/gamma 不影响 eval，填默认值即可
    dummy_params = QTableParams(
        lr=0.0,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    policies = {aid: QTablePolicy(action_dim, epsilon=0.0) for aid in agent_ids}
    algo = QTableAlgo(policies, dummy_params)

    # ---- 3. 加载 Q-table ----
    model_dir = Path(model_dir)
    algo.load(str(model_dir))
    for pol in algo.policies.values():
        pol.set_epsilon(0.0)

    qtable_sizes = {aid: len(pol.q_table) for aid, pol in algo.policies.items()}
    print(f"\n=== Q-table loaded from {model_dir} ===")
    print(f"Q-table sizes: {qtable_sizes}")

    # ---- 4. 评估循环 ----
    print(f"\n=== Running {num_episodes} episodes (greedy, epsilon=0) ===")

    do_logits_file = bool(action_logits_csv)
    do_logits_print = bool(log_action_logits)
    logits_csv_w = None
    logits_csv_f = None
    logits_print_count = 0
    if do_logits_file:
        logits_csv_w, logits_csv_f = _open_action_logits_csv(action_logits_csv, action_dim)
        print(f"[Eval] Writing per-decision Q values to {action_logits_csv}")
    if do_logits_print or do_logits_file:
        print(
            "[Eval] qtable action_scores: 记录 Q 值；"
            f"active_only={action_logits_active_only}。"
        )

    episode_metrics = []
    metrics_history = []
    episode_times = []
    wi_fromT_finals: list = []
    anim_positions_history = []
    anim_time_intervals = []

    for ep in range(num_episodes):
        obs, infos = env.reset()
        truncated = False
        terminated = False
        step_count = 0

        is_last = (ep == num_episodes - 1)
        record = record_animation and is_last
        if record:
            anim_positions_history = [env.world.snapshot_agent_positions()]
            anim_time_intervals = []

        while not (truncated or terminated):
            if is_parallel:
                actions = {}
                for agent_str in env.agents:
                    info_i = infos[agent_str]
                    action_mask = info_i.get("action_mask", None)
                    pol = algo.policies[agent_str]
                    active = int(info_i.get("active_mask", 1))

                    if active:
                        if action_mask is not None:
                            actions[agent_str] = pol.select_action(obs[agent_str], action_mask)
                        else:
                            actions[agent_str] = int(np.argmax(pol.get_q(obs[agent_str])))
                    else:
                        # ON_EDGE：选 action_mask 中最后一个有效位（no-op）
                        if action_mask is not None:
                            valid = np.where(action_mask)[0]
                            actions[agent_str] = int(valid[-1]) if len(valid) > 0 else 0
                        else:
                            actions[agent_str] = 0

                    logits_print_count = _record_qtable_action_scores(
                        writer=logits_csv_w,
                        do_print=do_logits_print,
                        print_count=logits_print_count,
                        print_limit=log_action_logits_max_lines,
                        action_dim=action_dim,
                        episode=ep,
                        step=step_count,
                        sim_time=float(getattr(env.world, "current_time", step_count)),
                        agent_id=agent_str,
                        q_values=pol.get_q(obs[agent_str]).copy(),
                        action_mask=action_mask,
                        chosen_action=int(actions[agent_str]),
                        active=active,
                        active_only=action_logits_active_only,
                    )

                obs, _, terms, truncs, infos = env.step(actions)
                first = env.agents[0]
                truncated = bool(truncs[first])
                terminated = bool(terms[first])

            else:
                # Gymnasium Env（Joint 控制器）
                info_i = infos if isinstance(infos, dict) else {}
                action_mask = info_i.get("action_mask", None)
                pol = algo.policies["agent_0"]
                if action_mask is not None:
                    action = pol.select_action(obs, action_mask)
                else:
                    action = int(np.argmax(pol.get_q(obs)))
                logits_print_count = _record_qtable_action_scores(
                    writer=logits_csv_w,
                    do_print=do_logits_print,
                    print_count=logits_print_count,
                    print_limit=log_action_logits_max_lines,
                    action_dim=action_dim,
                    episode=ep,
                    step=step_count,
                    sim_time=float(getattr(getattr(env, "world", None), "current_time", step_count)),
                    agent_id="agent_0",
                    q_values=pol.get_q(obs).copy(),
                    action_mask=action_mask,
                    chosen_action=int(action),
                    active=1,
                    active_only=action_logits_active_only,
                )
                obs, _, terminated, truncated, infos = env.step(action)

            if record:
                anim_time_intervals.append(getattr(env, 'last_time_interval', 1.0))
                anim_positions_history.append(env.world.snapshot_agent_positions())
            step_count += 1

        final_metrics = env.world.current_metrics
        episode_metrics.append(final_metrics)
        episode_times.append(final_metrics.time)
        metrics_history.append(_eval_metrics_history_for_plot(env, env_type))
        if env_type in _MASUP_LIKE_ENV_TYPES:
            wi_fromT_finals.append(float(getattr(env, "worst_idleness_fromT", 0.0)))

        ep_line = (
            f"Episode {ep + 1}/{num_episodes}: "
            f"IGI={final_metrics.igi:.4f}, "
            f"AGI={final_metrics.agi:.4f}, "
            f"IWI={final_metrics.iwi:.4f}, "
            f"WI={final_metrics.wi:.4f}, "
            f"time={final_metrics.time:.2f}s"
        )
        if env_type in _MASUP_LIKE_ENV_TYPES:
            ep_line += f", WI@T={getattr(env, 'worst_idleness_fromT', 0.0):.4f}"
        print(ep_line)

    if logits_csv_f is not None:
        logits_csv_f.close()
        print(f"[Eval] action_scores CSV saved: {action_logits_csv}")
    if (
        do_logits_print
        and log_action_logits_max_lines > 0
        and logits_print_count >= log_action_logits_max_lines
    ):
        print(
            f"[Eval] action_scores 打印已截断（仅前 {log_action_logits_max_lines} 行）；"
            "完整记录请使用 action_logits_csv。"
        )

    # ---- 5. 汇总统计 ----
    print(f"\n=== Summary Statistics ({num_episodes} episodes) ===")
    for metric_name in ['IGI', 'AGI', 'IWI', 'WI']:
        values = [getattr(m, metric_name.lower()) for m in episode_metrics]
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    if env_type in _MASUP_LIKE_ENV_TYPES and wi_fromT_finals:
        print(
            f"WI@T: {np.mean(wi_fromT_finals):.4f} ± {np.std(wi_fromT_finals):.4f} "
            f"(worst_idleness_fromT at episode end)"
        )

    # ---- 6. 可视化 ----
    # 动画目录：优先用显式传入的 save_animation_dir，回退到 save_plot 的父目录
    _default_results_q = str(AUTODL_RESULTS_ROOT)
    anim_dir = save_animation_dir or (str(Path(save_plot).parent) if save_plot else _default_results_q)
    anim_plot_stem = Path(save_plot).stem if save_plot else None
    if metrics_history:
        print(f"\n=== Generating aggregated visualization ===")
        aggregated = aggregate_episode_metrics(metrics_history)
        avg_time = np.mean(episode_times)
        subtitle = f"Graph: {graph_name} | Agents: {num_agents} | Avg Time: {avg_time:.2f}s"
        _mcfg_q = _eval_plot_metric_configs(
            env_type, metrics_history[0] if metrics_history else {}
        )
        plot_aggregated_metrics(
            aggregated,
            metric_configs=_mcfg_q,
            title=f'Q-table Evaluation ({num_episodes} episodes)',
            subtitle=subtitle,
            save_path=save_plot,
            show=show_plot,
        )

    # ---- 7. 动画 ----
    if record_animation and anim_positions_history:
        print(f"\n=== Generating animation for last episode ===")
        algorithm_name = "qtable"
        if event_driven:
            from utils.vis_utils import create_event_driven_animation
            create_event_driven_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                time_intervals=anim_time_intervals,
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=anim_dir,
                max_frames=max_frames,
                plot_stem=anim_plot_stem,
            )
        else:
            from utils.vis_utils import create_animation
            create_animation(
                map_graph=env.world.graph,
                agent_positions_history=anim_positions_history,
                total_frames=len(anim_positions_history),
                algorithm_name=algorithm_name,
                map_name=graph_name,
                save_dir=anim_dir,
                max_frames=max_frames,
                plot_stem=anim_plot_stem,
            )

    if hasattr(env, "close"):
        env.close()
    return episode_metrics


# =============================================================================
#                     自动化评估入口（供 train.py / sweep.py 调用）
# =============================================================================

def run_eval_from_config(model_dir: str, eval_config_path: str, extra_params: dict = None):
    """从 eval yaml 读取所有评估参数，自动调用对应评估函数。

    eval yaml 需包含 env_type、env 段（环境参数），以及可选的 eval 段：
        eval:
          num_episodes: 10
          episode_time: null   # null 时从 env.episode_len 读取，依靠环境自身截断
          save_plot: evaluators/results/auto_eval.png
          # 保存 PNG 时同目录自动写入 {stem}_plot_data.csv（与图曲线一致）
          show_plot: false
          animation: false
          event_driven: true
          max_frames: null
          # 决策时刻动作分数（actor=logits / value=Q）：打印与/或 CSV
          log_action_logits: false
          log_action_logits_max_lines: 500
          action_logits_csv: null   # 例: evaluators/results/run_action_logits.csv
          action_logits_active_only: true

    Args:
        model_dir: 模型目录。
        eval_config_path: eval YAML 路径。
        extra_params: 可选的覆盖字段，会合并到 yaml 的 eval 段之上（用于 sweep 批量评估时
                      为每个 trial 生成独立的 save_plot / save_animation_dir 等）。

    若未提供 eval 段，则使用各评估函数的默认参数。

    custom_configs 自动对齐：若模型 config.yaml 的 extra 段包含 train_env_custom_configs
    （由 train.py 在训练时写入），会将其合并到 eval 环境的 custom_configs 中，
    确保 idi_scale / contribution_scale 等 sweep 参数与训练时保持一致。
    eval yaml 中显式设置的同名字段以训练时值为准（保证一致性）。
    """
    with open(eval_config_path) as f:
        raw = yaml.safe_load(f)
    algo_name = raw.get("algo_name", None)
    eval_params = dict(raw.get("eval", {}))

    # extra_params 覆盖 yaml 中的 eval 字段
    if extra_params:
        eval_params.update(extra_params)

    model_dir = str(resolve_models_path(model_dir))
    sp = eval_params.get("save_plot")
    if sp is not None:
        eval_params["save_plot"] = resolve_results_path(sp)
    sad = eval_params.get("save_animation_dir")
    if sad is not None:
        eval_params["save_animation_dir"] = resolve_results_path(sad)

    # animation 字段在 CLI 中叫 record_animation，统一映射
    if "animation" in eval_params:
        eval_params.setdefault("record_animation", eval_params.pop("animation"))

    # 从模型 config.yaml 读取训练时的 custom_configs，覆盖 eval yaml 中的同名字段
    # 这样 sweep 时被搜索的 idi_scale / contribution_scale 等参数在评估时与训练保持一致
    train_custom_configs = None
    try:
        model_config_path = Path(model_dir) / "config.yaml"
        if model_config_path.exists():
            with open(model_config_path) as f:
                model_cfg = yaml.safe_load(f)
            train_custom_configs = (model_cfg.get("extra") or {}).get("train_env_custom_configs")
    except Exception:
        pass

    print(f"\n[Eval] model_dir={model_dir}, config={eval_config_path}")
    if algo_name == "qtable":
        # 过滤掉 DRL 专属字段，仅保留 Q-table 评估支持的参数。
        _QTABLE_SUPPORTED = {
            "num_episodes", "save_plot", "show_plot", "record_animation",
            "event_driven", "max_frames", "save_animation_dir",
            "log_action_logits", "log_action_logits_max_lines",
            "action_logits_csv", "action_logits_active_only",
        }
        qtable_params = {k: v for k, v in eval_params.items() if k in _QTABLE_SUPPORTED}
        episode_metrics = test_qtable_policy(
            model_dir=model_dir,
            env_config_path=eval_config_path,
            env_custom_config_overrides=train_custom_configs,
            **qtable_params,
        )
    else:
        episode_metrics = test_trained_policy(
            model_dir=model_dir,
            env_config_path=eval_config_path,
            env_custom_config_overrides=train_custom_configs,
            **eval_params,
        )
    return _summary_from_episode_metrics(episode_metrics)


def eval_qtable_inline(
    algo,
    env_type: str,
    env_config,
    num_episodes: int = 5,
) -> dict:
    """用内存中的当前 Q-table 直接评估，行为与 DRL 的 inline eval 对齐。"""
    env = _create_env(env_type, env_config)
    is_parallel = hasattr(env, "possible_agents")
    prev_eps = {aid: pol.epsilon for aid, pol in algo.policies.items()}
    for pol in algo.policies.values():
        pol.set_epsilon(0.0)

    episode_metrics = []
    wi_fromT_finals: list = []
    try:
        for _ in range(num_episodes):
            obs, infos = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                if is_parallel:
                    actions = {}
                    for agent_str in env.agents:
                        info_i = infos[agent_str]
                        action_mask = info_i.get("action_mask", None)
                        pol = algo.policies[agent_str]
                        if info_i.get("active_mask", 1):
                            if action_mask is not None:
                                actions[agent_str] = pol.select_action(obs[agent_str], action_mask)
                            else:
                                actions[agent_str] = int(np.argmax(pol.get_q(obs[agent_str])))
                        else:
                            if action_mask is not None:
                                valid = np.where(action_mask)[0]
                                actions[agent_str] = int(valid[-1]) if len(valid) > 0 else 0
                            else:
                                actions[agent_str] = 0
                    obs, _, terms, truncs, infos = env.step(actions)
                    first = env.agents[0]
                    truncated = bool(truncs[first])
                    terminated = bool(terms[first])
                else:
                    info_i = infos if isinstance(infos, dict) else {}
                    action_mask = info_i.get("action_mask", None)
                    pol = algo.policies["agent_0"]
                    if action_mask is not None:
                        action = pol.select_action(obs, action_mask)
                    else:
                        action = int(np.argmax(pol.get_q(obs)))
                    obs, _, terminated, truncated, infos = env.step(action)

            final_metrics = env.world.current_metrics
            episode_metrics.append(final_metrics)
            if env_type in _MASUP_LIKE_ENV_TYPES:
                wi_fromT_finals.append(float(getattr(env, "worst_idleness_fromT", 0.0)))
    finally:
        for aid, pol in algo.policies.items():
            pol.set_epsilon(prev_eps[aid])
        if hasattr(env, "close"):
            env.close()

    return _summary_from_episode_metrics(episode_metrics, wi_fromT_finals)


# =============================================================================
#                 内存内 inline eval（训练循环中定期调用，无需 checkpoint）
# =============================================================================


def _eval_policy_inline_beau_mat(
    policy: MATMultiAgentPolicy,
    env,
    num_episodes: int,
    device: torch.device,
    episode_time: float | None,
    env_type: str,
) -> dict:
    """BEAU + MAT：与 MATOnPolicyCollector 同构的逐步 rollout，返回 eval/* 指标。"""
    is_masup_like = env_type in _MASUP_LIKE_ENV_TYPES
    episode_metrics_list: list = []
    wi_fromT_finals: list = []

    for _ in range(num_episodes):
        obs, infos = env.reset()
        _last_sh = torch.zeros(
            1, policy.n_agents, policy.n_agents, 2,
            dtype=torch.float32, device=device,
        )
        while env.agents:
            if episode_time is not None and env.world.current_time >= episode_time:
                break
            obs, infos, _last_sh = _beau_mat_rollout_step(
                env, policy, infos, _last_sh
            )
        episode_metrics_list.append(env.world.current_metrics)
        if is_masup_like:
            wi_fromT_finals.append(
                float(getattr(env, "worst_idleness_fromT", 0.0))
            )

    return _summary_from_episode_metrics(episode_metrics_list, wi_fromT_finals or None)


def eval_policy_inline(
    policy,
    env_type: str,
    env_config,
    num_episodes: int = 5,
    device=None,
    obs_rms=None,
) -> dict:
    """用内存中的当前 policy 直接评估，不保存/加载 checkpoint，不生成图表。

    Args:
        policy:       训练中的 policy 实例。可以是 MultiAgentPolicy（多智能体训练），
                      也可以是裸 ActorPolicy / ValuePolicy（单智能体 Gym 训练）。
                      当单智能体训练 + 多智能体 eval（如 suns_gym 训→ suns 3-agent eval）时，
                      会为每个 agent 独立调用裸 policy，实现"共享同一套权重"的多 agent 部署。
        env_type:     环境类型字符串（如 "masup"、"suns"）。
        env_config:   EnvConfig 实例（从 eval yaml 读取）。
        num_episodes: 评估 episode 数量。
        device:       torch.device；None 时从 policy 自动检测。
        obs_rms:      utils.log_utils.RunningMeanStd 实例；训练时启用了 norm_obs 则传入，
                      确保评估时观测归一化与训练完全一致。None 表示不归一化。

    Returns:
        dict, key 均带 "eval/" 前缀，可直接传入 logger.log。
        指标包含 igi / agi / iwi / wi（及 masup 专属 wi_fromT）的 episode 均值。
    """
    if device is None:
        device = getattr(policy, "device", torch.device("cpu"))

    env = _create_env(env_type, env_config)
    is_masup_like = env_type in _MASUP_LIKE_ENV_TYPES
    cfg_dict, _ = _env_config_to_dicts(env_config)
    episode_time = cfg_dict.get("episode_len", None)

    is_recurrent = getattr(policy, "is_recurrent", False)
    episode_metrics_list = []
    wi_fromT_finals = []

    # 判断 eval env 的接口类型：
    #   is_pettingzoo=True  → PettingZoo ParallelEnv，obs/infos 为 per-agent dict
    #   is_pettingzoo=False → Gymnasium 单智能体，obs 为 ndarray，infos 为扁平 dict
    is_pettingzoo = hasattr(env, "possible_agents")
    # MultiAgentPolicy 使用统一的 forward(obs_dict, state_dict, action_mask) 接口；
    # 裸 ActorPolicy/ValuePolicy 使用 forward(obs_t, state, action_mask) 接口。
    is_multi_policy = isinstance(policy, MultiAgentPolicy)

    try:
        if isinstance(policy, MATMultiAgentPolicy):
            return _eval_policy_inline_beau_mat(
                policy, env, num_episodes, device, episode_time, env_type
            )

        for _ in range(num_episodes):
            obs, infos = env.reset()
            hidden_state = None

            if is_pettingzoo:
                # PettingZoo ParallelEnv 分支（多智能体，无论 policy 是否为 MultiAgentPolicy）
                while env.agents:
                    if episode_time is not None and env.world.current_time >= episode_time:
                        break

                    action_masks = {
                        aid: torch.as_tensor(
                            info["action_mask"], dtype=torch.bool, device=device
                        )
                        for aid, info in infos.items()
                    }
                    # obs 归一化：若训练时开启了 norm_obs，使用相同 RMS 统计量
                    if obs_rms is not None:
                        obs_normed = {aid: obs_rms.norm(np.asarray(o)) for aid, o in obs.items()}
                    else:
                        obs_normed = obs
                    obs_tensor = {
                        aid: torch.as_tensor(o, dtype=torch.float32, device=device)
                        for aid, o in obs_normed.items()
                    }

                    if is_multi_policy:
                        with torch.no_grad():
                            outputs = policy.forward(
                                obs_tensor, state_dict=hidden_state, action_mask=action_masks
                            )
                        actions = {aid: out["act"].cpu().numpy() for aid, out in outputs.items()}
                        if is_recurrent:
                            hidden_state = {
                                aid: out["state"]
                                for aid, out in outputs.items()
                                if out.get("state") is not None
                            }
                    else:
                        actions = {}
                        new_hidden = {} if is_recurrent else None
                        for aid in list(obs_tensor.keys()):
                            obs_t = obs_tensor[aid].unsqueeze(0)
                            am_t = action_masks[aid].unsqueeze(0)
                            h = hidden_state.get(aid) if (is_recurrent and hidden_state) else None
                            with torch.no_grad():
                                out = policy.forward(obs_t, state=h, action_mask=am_t)
                            # 提取标量整数动作（env.step 不接受 array([scalar])）
                            actions[aid] = int(out["act"].cpu().numpy().reshape(-1)[0])
                            if is_recurrent and out.get("state") is not None:
                                new_hidden[aid] = out["state"]
                        if is_recurrent:
                            hidden_state = new_hidden

                    obs, _, _, _, infos = env.step(actions)
            else:
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    if episode_time is not None and getattr(env, "world", None) is not None:
                        if env.world.current_time >= episode_time:
                            break

                    am = infos.get("action_mask")
                    if am is None:
                        raise KeyError(
                            "inline eval 需要 info['action_mask']；请在环境 _build_info 中提供"
                        )
                    action_mask_t = torch.as_tensor(
                        np.asarray(am), dtype=torch.bool, device=device,
                    ).unsqueeze(0)
                    obs_np = obs_rms.norm(np.asarray(obs)) if obs_rms is not None else np.asarray(obs)
                    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        out = policy.forward(
                            obs_t, state=hidden_state, action_mask=action_mask_t,
                        )
                    act = int(out["act"].cpu().numpy().reshape(-1)[0])
                    if is_recurrent and out.get("state") is not None:
                        hidden_state = out["state"]

                    obs, _rew, terminated, truncated, infos = env.step(act)
                    terminated = bool(terminated)
                    truncated = bool(truncated)

            final_metrics = env.world.current_metrics
            episode_metrics_list.append(final_metrics)
            if is_masup_like:
                wi_fromT_finals.append(
                    float(getattr(env, "worst_idleness_fromT", 0.0))
                )

        result = {
            "eval/igi": float(np.mean([m.igi for m in episode_metrics_list])),
            "eval/agi": float(np.mean([m.agi for m in episode_metrics_list])),
            "eval/iwi": float(np.mean([m.iwi for m in episode_metrics_list])),
            "eval/wi":  float(np.mean([m.wi  for m in episode_metrics_list])),
        }
        if is_masup_like and wi_fromT_finals:
            result["eval/wi_fromT"] = float(np.mean(wi_fromT_finals))

        return result
    finally:
        if hasattr(env, "close"):
            env.close()


# =============================================================================
#                              CLI 入口
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='通用 RL 策略评估')
    parser.add_argument('--model', type=str,
                        default=str(AUTODL_MODELS_ROOT / 'mappo' / 'imi' / 'final'),
                        help='模型目录 (含 config.yaml + policy.pt)')
    parser.add_argument('--env_config', type=str,
                        default='configs/eval/masup/masup_tsp12.yaml',
                        help='eval YAML (含 env_type + 环境参数)')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='评估 episode 数量（未指定时从 env_config 的 eval 段读取）')
    parser.add_argument('--episode_time', type=float, default=None,
                        help='每个 episode 的仿真时间上限（秒）；未指定时从 env 段的 episode_len 读取')
    parser.add_argument('--save_plot', type=str,
                        default=str(AUTODL_RESULTS_ROOT / 'eval.png'),
                        help='图表保存路径')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图表')
    parser.add_argument('--animation', action='store_true',
                        help='录制最后一个 episode 的动画视频')
    parser.add_argument('--no_event_driven', action='store_true',
                        help='使用固定步长动画（默认为事件驱动动画）')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='动画最大帧数限制（默认不限制，推荐 300~600）')
    parser.add_argument('--log_action_logits', action='store_true',
                        help='打印每次有效决策的动作 logits/Q 与区分度统计（有行数上限）')
    parser.add_argument('--log_action_logits_max_lines', type=int, default=None,
                        help='覆盖 yaml 中的打印行数上限')
    parser.add_argument('--action_logits_csv', type=str, default=None,
                        help='将每次决策的 logits/Q 写入该 CSV 路径')

    args = parser.parse_args()

    # 读取 yaml：algo_name + eval 段。CLI 显式传入覆盖 yaml，未传入则用 yaml
    with open(args.env_config) as _f:
        _raw = yaml.safe_load(_f)
    _algo_name = _raw.get("algo_name", None)
    _eval = _raw.get("eval", {})

    num_episodes = args.num_episodes if args.num_episodes is not None else _eval.get("num_episodes", 5)
    episode_time = args.episode_time if args.episode_time is not None else _eval.get("episode_time", None)

    _log_logits = args.log_action_logits or _eval.get("log_action_logits", False)
    _log_max = (
        args.log_action_logits_max_lines
        if args.log_action_logits_max_lines is not None
        else _eval.get("log_action_logits_max_lines", 500)
    )
    _csv = args.action_logits_csv if args.action_logits_csv is not None else _eval.get("action_logits_csv")
    _active_only = _eval.get("action_logits_active_only", True)

    if _algo_name == "qtable":
        test_qtable_policy(
            model_dir=args.model,
            env_config_path=args.env_config,
            num_episodes=num_episodes,
            save_plot=args.save_plot,
            show_plot=not args.no_show,
            record_animation=args.animation,
            event_driven=not args.no_event_driven,
            max_frames=args.max_frames,
        )
    else:
        test_trained_policy(
            model_dir=args.model,
            env_config_path=args.env_config,
            num_episodes=num_episodes,
            episode_time=episode_time,
            save_plot=args.save_plot,
            show_plot=not args.no_show,
            record_animation=args.animation,
            event_driven=not args.no_event_driven,
            max_frames=args.max_frames,
            log_action_logits=_log_logits,
            log_action_logits_max_lines=_log_max,
            action_logits_csv=_csv,
            action_logits_active_only=_active_only,
        )
