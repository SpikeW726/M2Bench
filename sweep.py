"""Run Weights & Biases hyperparameter sweeps around ``train.train``.

The script loads a base experiment YAML, applies parameters supplied by a sweep,
and invokes the normal training entry point. It can create a sweep without
running it, join an existing sweep, or create and execute one immediately.
"""

import argparse
import shutil
import os
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import yaml
import wandb

from configs.registry import load_config
from configs.exp_configs import ExperimentConfig
from train import (
    train,
    _configure_wandb_env_step_axis,
    _wandb_fix_broken_local_proxy,
    _wandb_init_settings,
)
from utils.project_paths import DEFAULT_RESULTS_DIR, result_path
from utils.model_io import trial_checkpoint_dir
from trainers.sweep_early_stopper import SweepEarlyStop, SweepEarlyStopper

def apply_sweep_overrides(config: ExperimentConfig, sweep_cfg: dict):
    algo_fields = {f.name for f in fields(type(config.algo))}
    training_fields = {f.name for f in fields(type(config.training))}
    top_fields = {f.name for f in fields(ExperimentConfig)}

    for key, value in sweep_cfg.items():
        if key in algo_fields:
            setattr(config.algo, key, value)
        elif key in training_fields:
            setattr(config.training, key, value)
        elif key in top_fields:
            setattr(config, key, value)
        elif key == "custom_configs" and isinstance(value, dict):
            if config.env.custom_configs is None:
                config.env.custom_configs = {}
            config.env.custom_configs = {**config.env.custom_configs, **value}
        elif key.startswith("custom_configs."):
            sub_key = key[len("custom_configs."):]
            if config.env.custom_configs is None:
                config.env.custom_configs = {}
            config.env.custom_configs[sub_key] = value

_BASE_CONFIG_PATH: str = ""
_SWEEP_ID: str = ""
_SWEEP_RAW_CONFIG: dict = {}
_EVAL_CONFIG_PATH: str = ""
_MODELS_DIR: str = ""
_RESULTS_DIR: str = ""

def _record_trial_save_dir(config: ExperimentConfig) -> None:
    best_dir = config.save_dir / "best"
    final_dir = config.save_dir / "final"
    if best_dir.is_dir():
        ckpt_dir = best_dir
    elif final_dir.is_dir():
        ckpt_dir = final_dir
    else:
        print("[Sweep] No checkpoint found for this trial (skip save_dir in summary)")
        return
    wandb.run.summary["save_dir"] = str(ckpt_dir)
    print(f"[Sweep] Recorded save_dir={ckpt_dir}")

def delete_trial_artifacts(save_dir: Path) -> None:
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        return
    shutil.rmtree(save_dir)
    print(f"[Sweep] Deleted eliminated trial artifacts: {save_dir}", flush=True)

def sweep_train():
    early_stopper = None
    et_config = _SWEEP_RAW_CONFIG.get("early_terminate")
    if et_config and _SWEEP_ID:
        metric_name = _SWEEP_RAW_CONFIG.get("metric", {}).get("name", "env/wi")
        metric_goal = _SWEEP_RAW_CONFIG.get("metric", {}).get("goal", "minimize")
        state_file = et_config.get("state_file", f".sweep_es_{_SWEEP_ID}.json")
        early_stopper = SweepEarlyStopper(et_config, metric_name, metric_goal, state_file)

    config = None
    saved_proxy_env: dict = {}
    try:
        saved_proxy_env = _wandb_fix_broken_local_proxy()
        wandb.init(settings=_wandb_init_settings())

        _configure_wandb_env_step_axis()
        sweep_cfg = dict(wandb.config)

        config = load_config(_BASE_CONFIG_PATH)
        apply_sweep_overrides(config, sweep_cfg)
        if _MODELS_DIR:
            config.models_dir = _MODELS_DIR

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        config._timestamp = now
        config.exp_name = f"sweep-{wandb.run.id}"

        wandb.run.name = f"{config.algo_name}-{config.env_type}-{config.graph_name}-{now}"

        config.track_wandb = True

        print(f"[Sweep] Run {wandb.run.name} with overrides:")
        for key, value in sweep_cfg.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")

        _ecp = _EVAL_CONFIG_PATH or getattr(config, "eval_config_path", None) or None
        train(
            config,
            eval_config_path=_ecp,
            early_stopper=early_stopper,
            results_dir=_RESULTS_DIR or None,
        )

        _record_trial_save_dir(config)

    except SweepEarlyStop as e:

        print(f"\n[Sweep] *** EARLY STOP *** (not a crash): {e}", flush=True)
        try:
            wandb.run.summary["early_stopped"] = True
            wandb.run.summary["early_stop_reason"] = e.reason
            if config is None:
                pass
            elif e.reason == SweepEarlyStop.REASON_CROSS_TRIAL:
                delete_trial_artifacts(config.save_dir)
            else:

                _record_trial_save_dir(config)
        except Exception as cleanup_err:
            print(f"[Sweep] Failed to handle early stop cleanup: {cleanup_err}", flush=True)

    except Exception as e:

        import traceback
        import sys
        from pathlib import Path

        lines = [
            f"[Sweep] Error during training: {type(e).__name__}: {e!r}",
        ]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as sync_e:
            lines.append(f"  cuda_synchronize: {type(sync_e).__name__}: {sync_e!r}")
        if e.__cause__ is not None:
            lines.append(f"  __cause__: {type(e.__cause__).__name__}: {e.__cause__!r}")
        if e.__context__ is not None and e.__context__ is not e.__cause__:
            lines.append(f"  __context__: {type(e.__context__).__name__}: {e.__context__!r}")
        msg = "\n".join(lines)
        print(msg, file=sys.stderr)
        traceback.print_exception(type(e), e, e.__traceback__, chain=True, file=sys.stderr)

        log_path = Path("sweep_train_crash.log")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(msg + "\n")
                traceback.print_exception(
                    type(e), e, e.__traceback__, chain=True, file=f
                )
            print(f"[Sweep] Full traceback also appended to {log_path.resolve()}", file=sys.stderr)
        except OSError:
            pass
        raise

    finally:
        for k, v in saved_proxy_env.items():
            os.environ[k] = v

        if early_stopper is not None:
            early_stopper.finalize_trial()

def cleanup_sweep_early_stop_files(
    sweep_id: str,
    sweep_raw_config: dict,
    *,
    keep: bool = False,
) -> None:
    if keep:
        return
    et = sweep_raw_config.get("early_terminate")
    if not et:
        return
    state_path = Path(et.get("state_file", f".sweep_es_{sweep_id}.json"))
    lock_path = state_path.with_suffix(".lock")
    removed: list[str] = []
    for p in (lock_path, state_path):
        try:
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except OSError as e:
            print(f"[Sweep] Failed to remove {p}: {e}", flush=True)
    if removed:
        print(f"[Sweep] Removed early-stop state files: {', '.join(removed)}", flush=True)

# Batch evaluation.

def eval_best_runs(
    sweep_id: str,
    project: str,
    eval_config_path: str,
    top_n: int = 5,
    results_dir: str | None = None,
):
    from pathlib import Path as _Path
    import yaml as _yaml

    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    metric_name = sweep.config.get("metric", {}).get("name", "env/wi")
    goal = sweep.config.get("metric", {}).get("goal", "minimize")

    runs = [r for r in sweep.runs if r.state == "finished"]
    if not runs:
        print(f"[Eval] No finished runs found in sweep {sweep_id}")
        return

    reverse = (goal == "maximize")
    runs_sorted = sorted(
        runs,
        key=lambda r: r.summary.get(metric_name, float("inf") if not reverse else float("-inf")),
        reverse=reverse,
    )
    best = runs_sorted[:top_n]

    with open(eval_config_path) as f:
        _raw = _yaml.safe_load(f)
    _eval_raw = _raw.get("eval") or {}
    base_save_plot = _eval_raw.get("save_plot", None)
    base_action_logits_csv = _eval_raw.get("action_logits_csv", None)
    record_animation = _eval_raw.get("animation", False)

    print(f"\n[Eval] Evaluating top-{top_n} runs by {metric_name} ({goal})")
    from evaluators.test import run_eval_from_config
    for i, run in enumerate(best):
        model_dir = run.summary.get("save_dir")
        if not model_dir:
            print(f"  [Skip] Run {run.id}: no save_dir in summary")
            continue
        metric_val = run.summary.get(metric_name, "?")
        rank = i + 1
        print(f"\n  [{rank}/{top_n}] Run {run.id} | {metric_name}={metric_val} | {model_dir}")

        extra = {}
        if base_save_plot:
            p = result_path(base_save_plot, results_dir)
            # e.g. auto_eval.png -> auto_eval_1.png.
            extra["save_plot"] = str(p.with_stem(f"{p.stem}_{rank}"))
        if base_action_logits_csv:
            lp = result_path(base_action_logits_csv, results_dir)
            extra["action_logits_csv"] = str(lp.with_stem(f"{lp.stem}_{rank}"))
        if record_animation:

            base_dir = (
                str(result_path(base_save_plot, results_dir).parent)
                if base_save_plot
                else str(_Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR)
            )
            extra["save_animation_dir"] = str(_Path(base_dir) / f"run_{rank}")

        try:
            run_eval_from_config(model_dir, eval_config_path, extra_params=extra)
        except Exception as e:
            print(f"  [Error] Run {run.id} eval failed: {e}")
            import traceback
            traceback.print_exc()

def _resolve_project(args) -> str:
    if args.project:
        return args.project

    with open(args.base_config) as f:
        raw = yaml.safe_load(f)
    algo = raw.get("algo_name", "unknown")
    mdp = raw.get("env_type", "unknown")
    graph = raw.get("graph_name", "unknown")
    return f"sweep-{algo}-{mdp}-{graph}"

def create_sweep(args, project: str) -> str:
    if args.sweep_config:
        with open(args.sweep_config) as f:
            sweep_config = yaml.safe_load(f)
        print(f"[Main] Loaded sweep config from {args.sweep_config}")

        sweep_config = {k: v for k, v in sweep_config.items() if k != "early_terminate"}
    else:
        sweep_config = {
            "method": args.method,
            "metric": {"name": "env/wi", "goal": "minimize"},
            "parameters": {
                "actor_lr": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"},
                "critic_lr": {"min": 1e-4, "max": 1e-3, "distribution": "log_uniform_values"},
                "clip_range": {"min": 0.1, "max": 0.4},
                "vf_coef": {"min": 0.1, "max": 2.0},
                "ent_coef": {"min": 0.0, "max": 0.3},
                "gae_lambda": {"min": 0.9, "max": 1.0},
                "minibatch_size": {"values": [256, 512, 1024, 2048]},
                "update_epochs": {"values": [3, 5, 10]},
                "num_steps": {"values": [1024, 2048, 4096]},
                "gamma": {"value": 0.999},
                "total_steps": {"value": 50_000_000},
            },
        }

    sweep_id = wandb.sweep(sweep_config, project=project)
    return sweep_id

# Entry point.

def main():
    global _BASE_CONFIG_PATH, _SWEEP_ID, _SWEEP_RAW_CONFIG, _EVAL_CONFIG_PATH
    global _MODELS_DIR, _RESULTS_DIR

    parser = argparse.ArgumentParser(description="Weights & Biases hyperparameter sweep")

    parser.add_argument("--base-config", type=str, required=True,
                        help="Path to the base experiment YAML")

    parser.add_argument("--create-only", action="store_true",
                        help="Create the sweep and print its ID without running an agent")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Join an existing sweep without creating one; supports parallel terminals")

    parser.add_argument("--project", type=str, default=None,
                        help="WandB project name (default: sweep-{algo}-{graph})")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of trials run by this agent")
    parser.add_argument("--method", type=str, default="bayes",
                        choices=["bayes", "random", "grid"],
                        help="Sweep method (used only when creating a sweep)")
    parser.add_argument("--sweep-config", type=str, default=None,
                        help="Sweep YAML path (used only when creating a sweep)")
    parser.add_argument("--eval-config", type=str, default=None,
                        help="Evaluation YAML path; evaluates the top N trials after the sweep")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top trials to evaluate after the sweep (default: 5)")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Root directory for trial checkpoints (default: project models directory).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Root directory for evaluation outputs (default: project evaluators/results).",
    )
    parser.add_argument(
        "--keep-sweep-state",
        action="store_true",
        help="Keep early-termination .json/.lock files; use for workers that finish before parallel peers",
    )

    args = parser.parse_args()
    _BASE_CONFIG_PATH = args.base_config
    _EVAL_CONFIG_PATH = args.eval_config or ""
    _MODELS_DIR = args.models_dir or ""
    _RESULTS_DIR = args.results_dir or ""
    project = _resolve_project(args)

    if args.sweep_config:
        with open(args.sweep_config) as f:
            _SWEEP_RAW_CONFIG = yaml.safe_load(f) or {}

    if args.sweep_id:

        sweep_id = args.sweep_id
        _SWEEP_ID = sweep_id
        if not args.sweep_config:
            print(
                "[Main] Warning: --sweep-config was not provided, so custom early termination "
                "will not be loaded. Parallel workers should use the YAML that created the sweep."
            )
        print(f"[Main] Joining existing sweep: {sweep_id}")
        print(f"[Main] Project: {project}")
        print(f"[Main] Will run {args.count} trials in this agent")
        try:
            wandb.agent(sweep_id, sweep_train, project=project, count=args.count)
            if args.eval_config:
                eval_best_runs(
                    sweep_id,
                    project,
                    args.eval_config,
                    top_n=args.top_n,
                    results_dir=args.results_dir,
                )
        finally:
            cleanup_sweep_early_stop_files(
                sweep_id, _SWEEP_RAW_CONFIG, keep=args.keep_sweep_state
            )

    elif args.create_only:

        sweep_id = create_sweep(args, project)
        _SWEEP_ID = sweep_id
        print(f"\n{'=' * 60}")
        print(f"  Sweep created successfully!")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project:  {project}")
        print(f"{'=' * 60}")
        print("\nRun the following command in each tmux terminal for parallel execution:")
        print(f"  python sweep.py --base-config {args.base_config} "
              f"--sweep-id {sweep_id} --count <N>")
        print()

    else:

        sweep_id = create_sweep(args, project)
        _SWEEP_ID = sweep_id
        print(f"[Main] Created sweep: {sweep_id} in project '{project}', "
              f"starting agent with {args.count} trials")
        try:
            wandb.agent(sweep_id, sweep_train, project=project, count=args.count)
            if args.eval_config:
                eval_best_runs(
                    sweep_id,
                    project,
                    args.eval_config,
                    top_n=args.top_n,
                    results_dir=args.results_dir,
                )
        finally:
            cleanup_sweep_early_stop_files(
                sweep_id, _SWEEP_RAW_CONFIG, keep=args.keep_sweep_state
            )

if __name__ == "__main__":
    main()
