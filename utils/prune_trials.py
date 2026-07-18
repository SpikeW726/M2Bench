#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TRIAL_RE = re.compile(r"^(\d{8})_(\d{6})$")

METRIC_RE = re.compile(r"^\s*best_metric_value:\s*([+-]?[\d.eE+-]+|null)\s*$")

ITER_SNAPSHOT_RE = re.compile(r"^(?:iter|ep)_(\d+)$")

KEEP_SUBDIRS = ("best", "final")

SKIP_PREFIXES = ("imi", "happo")

def parse_trial_time(name):
    m = TRIAL_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def read_metric(trial_dir):
    for sub in ("best", "final"):
        cfg = os.path.join(trial_dir, sub, "config.yaml")
        if not os.path.isfile(cfg):
            continue
        with open(cfg, "r") as f:
            for line in f:
                m = METRIC_RE.match(line)
                if m:
                    raw = m.group(1)
                    if raw == "null":
                        return None, f"{sub}/config.yaml: best_metric_value is null"
                    try:
                        return float(raw), sub
                    except ValueError:
                        return None, f"{sub}/config.yaml: cannot parse best_metric_value ({raw})"

    return None, "best_metric_value not found in best or final config"

def group_into_sweeps(trials, gap_hours):
    # trials: [(name, datetime)].
    sweeps = []
    cur = []
    prev_t = None
    for name, t in trials:
        if prev_t is not None and (t - prev_t).total_seconds() > gap_hours * 3600:
            sweeps.append(cur)
            cur = []
        cur.append((name, t))
        prev_t = t
    if cur:
        sweeps.append(cur)
    return sweeps

def cleanup_trial_dir(trial_dir, dry_run):
    removed = []
    if not os.path.isdir(trial_dir):
        return removed

    iter_snapshots = []
    has_keep_dir = False
    for sub in os.listdir(trial_dir):
        full = os.path.join(trial_dir, sub)
        if not os.path.isdir(full):
            continue
        if sub in KEEP_SUBDIRS:
            has_keep_dir = True
            continue
        m = ITER_SNAPSHOT_RE.match(sub)
        if m:
            iter_snapshots.append((int(m.group(1)), sub))

    if not iter_snapshots:
        return removed

    if not has_keep_dir:

        iter_snapshots.sort(key=lambda x: x[0])
        keep_name = iter_snapshots[-1][1]
        for _, sub in iter_snapshots:
            if sub == keep_name:
                continue
            removed.append(sub)
            if not dry_run:
                shutil.rmtree(os.path.join(trial_dir, sub))
    else:

        for _, sub in iter_snapshots:
            removed.append(sub)
            if not dry_run:
                shutil.rmtree(os.path.join(trial_dir, sub))
    return removed

def process_experiment(exp_dir, keep, gap_hours, dry_run):
    exp_name = os.path.basename(exp_dir)
    entries = sorted(os.listdir(exp_dir))

    trials = []
    for name in entries:
        full = os.path.join(exp_dir, name)
        if not os.path.isdir(full):
            continue
        t = parse_trial_time(name)
        if t is None:
            print(f"  [skip] Not a trial directory: {name}")
            continue
        trials.append((name, t))

    if not trials:
        print("  No valid trials; skipping")
        return

    trials.sort(key=lambda x: x[1])
    sweeps = group_into_sweeps(trials, gap_hours)

    print(f"  Found {len(trials)} trials grouped into {len(sweeps)} sweeps")

    to_delete = []
    for i, sweep in enumerate(sweeps, 1):
        span = f"{sweep[0][0]} ~ {sweep[-1][0]}"
        scored = []      # (value, name).
        unscored = []
        for name, _ in sweep:
            val, info = read_metric(os.path.join(exp_dir, name))
            if val is None:
                print(f"    [note] sweep#{i} {name}: {info} -> indeterminate, keeping")
                unscored.append(name)
            else:
                scored.append((val, name))

        scored.sort(key=lambda x: x[0])
        kept = scored[:keep]
        removed = scored[keep:]

        print(f"  -- sweep#{i} ({span}), {len(sweep)} trials --")
        for val, name in kept:
            pruned = cleanup_trial_dir(os.path.join(exp_dir, name), dry_run)
            extra = f", removed intermediate checkpoints: {', '.join(pruned)}" if pruned else ""
            print(f"    [keep] {name}  (best_metric_value={val}){extra}")
        for name in unscored:
            pruned = cleanup_trial_dir(os.path.join(exp_dir, name), dry_run)
            extra = f", removed intermediate checkpoints: {', '.join(pruned)}" if pruned else ""
            print(f"    [keep] {name}  (no metric){extra}")
        for val, name in removed:
            print(f"    [delete] {name}  (best_metric_value={val})")
            to_delete.append(os.path.join(exp_dir, name))

    if not dry_run:
        for path in to_delete:
            shutil.rmtree(path)
        if to_delete:
            print(f"  Deleted {len(to_delete)} trials")

def main():
    parser = argparse.ArgumentParser(description="Keep the best trials within each sweep group")
    parser.add_argument("--root", default=DEFAULT_MODELS_DIR, help="Models root directory")
    parser.add_argument("--keep", type=int, default=2, help="Number of trials to keep per sweep")
    parser.add_argument("--gap-hours", type=float, default=24.0,
                        help="Start a new sweep group when adjacent trials differ by more than this many hours")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files; without this flag, only print the dry-run actions")
    args = parser.parse_args()

    dry_run = not args.execute
    mode = "DRY-RUN (print only)" if dry_run else "EXECUTE (files will be deleted)"
    print(f"Mode: {mode}")
    print(f"Root: {args.root} | Keep per sweep: {args.keep} | Sweep gap: {args.gap_hours}h\n")

    if not os.path.isdir(args.root):
        print(f"Error: root directory does not exist: {args.root}")
        sys.exit(1)

    for exp_name in sorted(os.listdir(args.root)):
        exp_dir = os.path.join(args.root, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        if exp_name.startswith(SKIP_PREFIXES):
            print(f"[skip experiment] {exp_name} (imi/happo prefix)")
            continue
        print(f"[experiment] {exp_name}")
        process_experiment(exp_dir, args.keep, args.gap_hours, dry_run)
        print()

if __name__ == "__main__":
    main()
