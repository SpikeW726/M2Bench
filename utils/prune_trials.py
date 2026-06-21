#!/usr/bin/env python3
# 按 sweep 分组, 每组只保留指标最好的 N 个 trial, 其余删除
import argparse
import os
import re
import shutil
import sys
from datetime import datetime

# trial 文件夹命名: YYYYMMDD_HHMMSS
TRIAL_RE = re.compile(r"^(\d{8})_(\d{6})$")
# 从 config.yaml 中提取 best_metric_value
METRIC_RE = re.compile(r"^\s*best_metric_value:\s*([+-]?[\d.eE+-]+|null)\s*$")
# 中间迭代快照子目录: iter_<数字>(DRL) / ep_<数字>(Q-table epoch) 等
# 保留 trial 时需删掉这些中间快照, 只留 best/final
ITER_SNAPSHOT_RE = re.compile(r"^(?:iter|ep)_(\d+)$")

# 保留 trial 时, 内部只留下这些子目录
KEEP_SUBDIRS = ("best", "final")

SKIP_PREFIXES = ("imi", "happo")


def parse_trial_time(name):
    # 返回 trial 的起始时间 (datetime), 非法命名返回 None
    m = TRIAL_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def read_metric(trial_dir):
    # 优先 best 子目录, 否则 final 子目录; 返回 (value, source) 或 (None, reason)
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
                        return None, f"{sub}/config.yaml: best_metric_value 为 null"
                    try:
                        return float(raw), sub
                    except ValueError:
                        return None, f"{sub}/config.yaml: best_metric_value 无法解析 ({raw})"
        # 找到 config 但无该字段, 继续尝试下一个子目录
    return None, "未找到 best_metric_value 字段 (best/final config 均缺失)"


def group_into_sweeps(trials, gap_hours):
    # trials: [(name, datetime)] 已按时间排序; 相邻间隔超过阈值则切分为新 sweep
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
    # 删除 trial 内部除 best/final 外的中间快照子目录(iter_500 / ep_1000 等)
    # 特殊情况: 若 trial 内部没有任何 best/final, 只有 iter_*/ep_* 快照,
    #   则保留后缀数字最大的一个, 删除其余所有快照
    # 返回被清理的子目录相对路径列表(用于打印)
    removed = []
    if not os.path.isdir(trial_dir):
        return removed

    # 收集所有 iter_<数字> / ep_<数字> 子目录
    iter_snapshots = []  # [(int 后缀, 子目录名), ...]
    has_keep_dir = False  # 是否存在 best/final
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
        # 只有 iter_*/ep_* 没有 best/final: 保留数字最大的一个
        iter_snapshots.sort(key=lambda x: x[0])
        keep_name = iter_snapshots[-1][1]
        for _, sub in iter_snapshots:
            if sub == keep_name:
                continue
            removed.append(sub)
            if not dry_run:
                shutil.rmtree(os.path.join(trial_dir, sub))
    else:
        # 有 best/final: 所有中间快照都删掉
        for _, sub in iter_snapshots:
            removed.append(sub)
            if not dry_run:
                shutil.rmtree(os.path.join(trial_dir, sub))
    return removed


def process_experiment(exp_dir, keep, gap_hours, dry_run):
    exp_name = os.path.basename(exp_dir)
    entries = sorted(os.listdir(exp_dir))

    trials = []  # 合法 trial: (name, datetime)
    for name in entries:
        full = os.path.join(exp_dir, name)
        if not os.path.isdir(full):
            continue
        t = parse_trial_time(name)
        if t is None:
            print(f"  [跳过] 非 trial 文件夹: {name}")
            continue
        trials.append((name, t))

    if not trials:
        print(f"  无有效 trial, 跳过")
        return

    trials.sort(key=lambda x: x[1])
    sweeps = group_into_sweeps(trials, gap_hours)

    print(f"  共 {len(trials)} 个 trial, 划分为 {len(sweeps)} 个 sweep")

    to_delete = []
    for i, sweep in enumerate(sweeps, 1):
        span = f"{sweep[0][0]} ~ {sweep[-1][0]}"
        scored = []      # (value, name)
        unscored = []    # 无指标, 一律保留
        for name, _ in sweep:
            val, info = read_metric(os.path.join(exp_dir, name))
            if val is None:
                print(f"    [提示] sweep#{i} {name}: {info} -> 无法判断, 保留")
                unscored.append(name)
            else:
                scored.append((val, name))

        # 数值越小越好
        scored.sort(key=lambda x: x[0])
        kept = scored[:keep]
        removed = scored[keep:]

        print(f"  -- sweep#{i} ({span}), {len(sweep)} 个 trial --")
        for val, name in kept:
            pruned = cleanup_trial_dir(os.path.join(exp_dir, name), dry_run)
            extra = f", 清理中间快照: {', '.join(pruned)}" if pruned else ""
            print(f"    [保留] {name}  (best_metric_value={val}){extra}")
        for name in unscored:
            pruned = cleanup_trial_dir(os.path.join(exp_dir, name), dry_run)
            extra = f", 清理中间快照: {', '.join(pruned)}" if pruned else ""
            print(f"    [保留] {name}  (无指标){extra}")
        for val, name in removed:
            print(f"    [删除] {name}  (best_metric_value={val})")
            to_delete.append(os.path.join(exp_dir, name))

    if not dry_run:
        for path in to_delete:
            shutil.rmtree(path)
        if to_delete:
            print(f"  已删除 {len(to_delete)} 个 trial")


def main():
    parser = argparse.ArgumentParser(description="按 sweep 分组保留指标最好的若干 trial")
    parser.add_argument("--root", default="/root/autodl-tmp/models", help="models 根目录")
    parser.add_argument("--keep", type=int, default=2, help="每个 sweep 保留的 trial 数")
    parser.add_argument("--gap-hours", type=float, default=24.0,
                        help="相邻 trial 间隔超过该小时数则视为不同 sweep")
    parser.add_argument("--execute", action="store_true",
                        help="真正执行删除; 不加该参数则为 dry-run 仅打印")
    args = parser.parse_args()

    dry_run = not args.execute
    mode = "DRY-RUN (仅打印, 不删除)" if dry_run else "EXECUTE (将真实删除)"
    print(f"模式: {mode}")
    print(f"根目录: {args.root} | 每 sweep 保留: {args.keep} | sweep 间隔阈值: {args.gap_hours}h\n")

    if not os.path.isdir(args.root):
        print(f"错误: 根目录不存在 {args.root}")
        sys.exit(1)

    for exp_name in sorted(os.listdir(args.root)):
        exp_dir = os.path.join(args.root, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        if exp_name.startswith(SKIP_PREFIXES):
            print(f"[跳过实验] {exp_name} (imi/happo 开头)")
            continue
        print(f"[实验] {exp_name}")
        process_experiment(exp_dir, args.keep, args.gap_hours, dry_run)
        print()


if __name__ == "__main__":
    main()
