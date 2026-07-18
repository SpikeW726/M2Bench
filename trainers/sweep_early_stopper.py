"""Early stopping for hyperparameter sweeps.

Cross-trial stopping removes runs below the historical milestone quantile after
warm-up. Within-trial stopping detects flat slopes over every configured horizon.
Both paths raise ``SweepEarlyStop`` so expected termination remains distinct from
training failures.
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
    REASON_CROSS_TRIAL = "cross_trial"
    REASON_WITHIN_TRIAL_SLOPE = "within_trial_slope"

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason

class SweepEarlyStopper:
    def __init__(self, config: dict, metric_name: str, metric_goal: str, state_file: str):
        self.metric_name     = metric_name
        self.minimize        = (metric_goal == "minimize")
        self.state_file      = Path(state_file)
        self.lock_file       = self.state_file.with_suffix(".lock")

        # Configuration.
        self.min_trial       = int(config.get("min_trial", 6))
        self.min_steps       = int(config.get("min_steps", 5_000_000))
        self.terminate_ratio = float(config.get("terminate_ratio", 0.4))
        self.early_stop_iters = int(config.get("early_stop_iters", 200))
        self.slope_horizons  = list(config.get("slope_horizons", [8, 32, 64]))
        self.slope_threshold = float(config.get("slope_threshold", 1e-4))

        state = self._load_state()
        self._trial_count_at_start  = state["trial_count"]
        self._pool_at_start         = list(state["milestone_pool"])

        self._metric_history: List[float] = []
        self._milestone_value: Optional[float] = None
        self._cross_trial_checked = False

    def record_metrics(self, env_metrics: dict, total_steps: int) -> None:
        val = env_metrics.get(self.metric_name)
        if val is None:
            return
        self._metric_history.append(float(val))

        if self._milestone_value is None and total_steps >= self.min_steps:
            self._milestone_value = float(val)

    def check_and_maybe_raise(self, total_steps: int) -> None:
        if not self._metric_history:
            return

        warmed_up = self._trial_count_at_start >= self.min_trial

        if (warmed_up
                and self._milestone_value is not None
                and not self._cross_trial_checked):
            self._cross_trial_checked = True
            pool = self._pool_at_start
            if pool:

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
                        f"outside the top {self.terminate_ratio:.0%} (threshold={q:.4f})"
                        f" at step={total_steps}",
                        reason=SweepEarlyStop.REASON_CROSS_TRIAL,
                    )

        # Within-trial slope.

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
                # minimize: slope < -threshold.
                still_improving = (slope < -self.slope_threshold
                                   if self.minimize
                                   else slope > self.slope_threshold)
                if still_improving:
                    all_flat = False
                    break
            if all_flat:
                raise SweepEarlyStop(
                    f"within-trial slope: {self.metric_name} did not improve for {n} iterations "
                    f"(all horizons {self.slope_horizons} are flat)"
                    f" at step={total_steps}",
                    reason=SweepEarlyStop.REASON_WITHIN_TRIAL_SLOPE,
                )

    def finalize_trial(self) -> None:
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

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"trial_count": 0, "milestone_pool": []}

    def _save_state(self, state: dict) -> None:
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
