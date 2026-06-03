"""Sweep 自定义早停模块。

两种独立机制：
  1. Cross-trial: trial 达到 min_steps 时，指标不在历史 milestone pool 的前
     terminate_ratio 分位则早停（仅在 min_trial 个 warm-up 后启用）
  2. Within-trial slope: 有指标样本数 >= early_stop_iters 后（每个训练 iter 至多追加 1 点；
     长 episode 时点数可能远小于 trainer iteration），若所有 slope_horizons 窗口的斜率均平坦则早停

均通过 SweepEarlyStop 异常触发，与崩溃报错严格区分。
"""

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.stats import linregress


class SweepEarlyStop(Exception):
    """优雅早停异常。sweep_train() 单独 catch，不写 crash log，不 re-raise。

    reason:
        cross_trial          — 跨 trial 分位淘汰；sweep 会删除本 trial 权重目录
        within_trial_slope   — trial 内斜率平坦；保留 best/ 等 checkpoint
    """

    REASON_CROSS_TRIAL = "cross_trial"
    REASON_WITHIN_TRIAL_SLOPE = "within_trial_slope"

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


class SweepEarlyStopper:
    """管理 sweep 跨 trial 和 trial 内部的早停逻辑。

    Args:
        config:       early_terminate YAML 块（dict）
        metric_name:  目标指标名称，如 "env/wi_fromT"
        metric_goal:  "minimize" 或 "maximize"
        state_file:   跨 trial 共享的 JSON 状态文件路径
    """

    def __init__(self, config: dict, metric_name: str, metric_goal: str, state_file: str):
        self.metric_name     = metric_name
        self.minimize        = (metric_goal == "minimize")
        self.state_file      = Path(state_file)
        self.lock_file       = self.state_file.with_suffix(".lock")

        # 早停配置
        self.min_trial       = int(config.get("min_trial", 6))
        self.min_steps       = int(config.get("min_steps", 5_000_000))
        self.terminate_ratio = float(config.get("terminate_ratio", 0.4))
        self.early_stop_iters = int(config.get("early_stop_iters", 200))
        self.slope_horizons  = list(config.get("slope_horizons", [8, 32, 64]))
        self.slope_threshold = float(config.get("slope_threshold", 1e-4))

        # 本 trial 启动时读取共享状态（只读，无锁）
        state = self._load_state()
        self._trial_count_at_start  = state["trial_count"]
        self._pool_at_start         = list(state["milestone_pool"])

        # 本 trial 内部状态
        self._metric_history: List[float] = []
        self._milestone_value: Optional[float] = None
        self._cross_trial_checked = False   # cross-trial 检查只做一次

    # ------------------------------------------------------------------
    # 每 iteration 调用
    # ------------------------------------------------------------------

    def record_metrics(self, env_metrics: dict, total_steps: int) -> None:
        """记录本 iteration 的指标值，并在首次达到 min_steps 时存储 milestone。"""
        val = env_metrics.get(self.metric_name)
        if val is None:
            return
        self._metric_history.append(float(val))

        if self._milestone_value is None and total_steps >= self.min_steps:
            self._milestone_value = float(val)

    def check_and_maybe_raise(self, total_steps: int) -> None:
        """检查两种早停条件，若触发则 raise SweepEarlyStop。"""
        if not self._metric_history:
            return

        warmed_up = self._trial_count_at_start >= self.min_trial

        # ---- Cross-trial 检查（仅在 min_steps 首次到达后执行一次）----
        if (warmed_up
                and self._milestone_value is not None
                and not self._cross_trial_checked):
            self._cross_trial_checked = True
            pool = self._pool_at_start
            if pool:
                # minimize: 保留指标 <= q(r) 的 trial（约前 r 分位里最好的那些）
                # maximize: 对称地用 q(1-r)，保留指标 >= 该阈值的 trial（约前 r 分位）
                q = (
                    np.quantile(pool, self.terminate_ratio)
                    if self.minimize
                    else np.quantile(pool, 1.0 - self.terminate_ratio)
                )
                should_stop = (self._milestone_value > q
                               if self.minimize
                               else self._milestone_value < q)
                if should_stop:
                    raise SweepEarlyStop(
                        f"cross-trial: {self.metric_name}={self._milestone_value:.4f} "
                        f"未进入前 {self.terminate_ratio:.0%}（threshold={q:.4f}）"
                        f" at step={total_steps}",
                        reason=SweepEarlyStop.REASON_CROSS_TRIAL,
                    )

        # ---- Within-trial slope 检查 ----
        # 必须同时满足：跨 trial warm-up、样本数足够、且当前步数已过 min_steps 门槛
        # 否则 early_stop_iters 点可能在 min_steps 之前就凑齐，导致过早终止
        n = len(self._metric_history)
        max_horizon = max(self.slope_horizons)
        if (warmed_up
                and total_steps >= self.min_steps
                and n >= self.early_stop_iters
                and n >= max_horizon):
            all_flat = True
            for horizon in self.slope_horizons:
                window = self._metric_history[-horizon:]
                slope, *_ = linregress(np.arange(horizon), window)
                if not np.isfinite(slope):
                    all_flat = False
                    break
                # minimize: slope < -threshold 表示仍在改善；maximize 反之
                still_improving = (slope < -self.slope_threshold
                                   if self.minimize
                                   else slope > self.slope_threshold)
                if still_improving:
                    all_flat = False
                    break
            if all_flat:
                raise SweepEarlyStop(
                    f"within-trial slope: {self.metric_name} 连续 {n} iters 无改善"
                    f"（所有 horizons {self.slope_horizons} 均平坦）"
                    f" at step={total_steps}",
                    reason=SweepEarlyStop.REASON_WITHIN_TRIAL_SLOPE,
                )

    # ------------------------------------------------------------------
    # Trial 结束时调用（正常完成或早停都必须调用）
    # ------------------------------------------------------------------

    def finalize_trial(self) -> None:
        """将本 trial 的结果写入共享状态（文件锁保证并发安全）。"""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                state = self._load_state()
                state["trial_count"] += 1
                if self._milestone_value is not None:
                    state["milestone_pool"].append(self._milestone_value)
                self._save_state(state)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"trial_count": 0, "milestone_pool": []}

    def _save_state(self, state: dict) -> None:
        """原子写入：先写临时文件再 rename，避免部分写入。"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.state_file.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp_path, self.state_file)
        except Exception:
            os.unlink(tmp_path)
            raise
