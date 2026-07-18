#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate MAPPO-MASUP robustness over edge-time jitter levels.

For each requested jitter fraction, the script derives a temporary evaluation
configuration from the supplied YAML and records every episode's ``WI_fromT``.
The source YAML is never modified. Actual edge duration is sampled from
``[T * (1 - fraction), T * (1 + fraction)]``.

The output CSV contains ``jitter_frac``, ``episode_idx``, and ``WI_fromT``.
"""


import os
import sys
import csv
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import yaml
import numpy as np
import argparse

from evaluators.test import test_trained_policy

DEFAULT_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

OUTPUT_DIR = Path(__file__).resolve().parent / "jitter"

def _build_temp_yaml(src_yaml_path: Path, frac: float) -> Path:
    with open(src_yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    env_section = raw.get("env", {})
    if not isinstance(env_section, dict):
        env_section = {}
        raw["env"] = env_section

    env_section["edge_time_jitter_frac"] = float(frac)

    if frac == 0.0:
        env_section["edge_time_jitter_mode"] = "none"
    else:

        env_section.setdefault("edge_time_jitter_mode", "full")

    tmp_dir = Path(tempfile.gettempdir()) / "jitter_sweep_yamls"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_frac{frac:.4f}".replace(".", "p")
    tmp_path = tmp_dir / f"{src_yaml_path.stem}{suffix}.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    return tmp_path

def _write_csv(rows: list, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["jitter_frac", "episode_idx", "WI_fromT"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "jitter_frac": f"{r['jitter_frac']:.4f}",
                "episode_idx": r["episode_idx"],
                "WI_fromT": f"{r['WI_fromT']:.6f}",
            })
    print(f"\n[JitterSweep] Results saved to: {csv_path}")

def run_jitter_sweep(
    config_path: str,
    model_dir: str,
    fracs: list = None,
    num_episodes: int = None,
    episode_time: float = None,
    eval_seed: int = None,
) -> Path:
    if fracs is None:
        fracs = list(DEFAULT_FRACS)

    src_yaml = Path(config_path).resolve()
    if not src_yaml.exists():
        raise FileNotFoundError(f"Config YAML not found: {src_yaml}")

    print(f"[JitterSweep] base config : {src_yaml}")
    print(f"[JitterSweep] model dir   : {model_dir}")
    print(f"[JitterSweep] fracs       : {fracs}")
    print(f"[JitterSweep] num_episodes: {num_episodes} (None=use yaml)")
    print(f"[JitterSweep] episode_time: {episode_time} (None=use yaml)")
    print(f"[JitterSweep] eval_seed   : {eval_seed}")

    with open(src_yaml) as f:
        base_raw = yaml.safe_load(f) or {}
    base_eval = base_raw.get("eval", {}) or {}
    eff_num_episodes = num_episodes if num_episodes is not None else base_eval.get("num_episodes", 5)

    rows = []
    for frac in fracs:
        tmp_yaml = _build_temp_yaml(src_yaml, float(frac))
        print(f"\n[JitterSweep] ===== frac={frac:.4f} | yaml={tmp_yaml.name} =====")

        _episode_metrics, wi_fromT_finals = test_trained_policy(
            model_dir=model_dir,
            env_config_path=str(tmp_yaml),
            num_episodes=eff_num_episodes,
            episode_time=episode_time,
            save_plot=None,
            show_plot=False,
            record_animation=False,
            event_driven=True,
            max_frames=None,
            eval_seed=eval_seed,
        )

        if not wi_fromT_finals:

            raise RuntimeError(
                f"frac={frac:.4f} got empty wi_fromT_finals; "
                "this script is for mappo+masup only, check env_type in yaml."
            )

        for ep_idx, wi_fromT in enumerate(wi_fromT_finals):
            rows.append({
                "jitter_frac": float(frac),
                "episode_idx": ep_idx,
                "WI_fromT": float(wi_fromT),
            })

        arr = np.asarray(wi_fromT_finals, dtype=float)
        print(f"[JitterSweep] frac={frac:.4f} -> "
              f"WI_fromT mean={arr.mean():.4f} ± {arr.std():.4f} "
              f"(n={len(wi_fromT_finals)})")

    csv_path = OUTPUT_DIR / f"{src_yaml.stem}.csv"
    _write_csv(rows, csv_path)
    return csv_path

def main():
    parser = argparse.ArgumentParser(
        description="Jitter sweep evaluation for MAPPO+MASUP: iterate over "
                    "edge_time_jitter_frac values and record per-episode WI_fromT",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Base evaluation YAML, e.g. configs/eval/masup/masup_mapB.yaml")
    parser.add_argument("--model", type=str, required=True,
                        help="Model directory containing config.yaml and policy.pt")
    parser.add_argument("--fracs", type=str, default=None,
                        help="Comma-separated jitter fractions, e.g. 0,0.05,0.1,0.15,0.2,0.25; "
                             "defaults to DEFAULT_FRACS")
    parser.add_argument("--num_episodes", type=int, required=True,
                        help="Episodes per fraction (required; controls the WI_fromT sample size)")
    parser.add_argument("--episode_time", type=float, default=None,
                        help="Simulation-time limit per episode (overrides yaml env.episode_len)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Evaluation seed shared by all fractions for reproducibility")

    args = parser.parse_args()

    fracs = None
    if args.fracs is not None:
        fracs = [float(x.strip()) for x in args.fracs.split(",") if x.strip() != ""]

    run_jitter_sweep(
        config_path=args.config,
        model_dir=args.model,
        fracs=fracs,
        num_episodes=args.num_episodes,
        episode_time=args.episode_time,
        eval_seed=args.seed,
    )

if __name__ == "__main__":
    main()
