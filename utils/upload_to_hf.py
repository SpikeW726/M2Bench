#!/usr/bin/env python3
# 将 autodl-tmp/{models,results} 增量上传到 HuggingFace dataset repo
#
# 核心策略 (避免 HF commit 限流 128/h):
#   - models: 每个 best/final/*_final/*_actor_best 目录打包成 1 个 .zip 上传 (1 commit)
#   - results: 每个 <map>/<algo>/ 目录打包成 1 个 .zip 上传 (1 commit)
#             只保留 best_eval*.png 和 best_eval*plot_data.csv
#   - 跳过视频文件 (.mp4/.gif/...)
#   - npy 在打包时转 npz 压缩
#   - SIGALRM 超时 + 指数退避重试
import argparse
import io
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from huggingface_hub import HfApi, create_repo

# ---- 配置区 ----
DEFAULT_REPO_ID = "SpikeW726/M2Bench"
DEFAULT_TOKEN_ENV = "HF_TOKEN"

# 跳过的视频/动画扩展名
SKIP_EXTS = {".mp4", ".gif", ".avi", ".mov", ".mkv", ".webm"}
# npy -> npz 压缩
COMPRESS_EXTS = {".npy"}

# results 中只保留这些文件 (best_eval 相关), 其余跳过
RESULTS_KEEP_RE = re.compile(r"^best_eval.*\.(png|csv)$")

# models 中作为"打包单元"的目录名 (这些目录整体打成 1 个 zip)
MODELS_PACK_DIRNAMES = {"best", "final"}
MODELS_PACK_DIRNAME_RE = re.compile(r"^.*(_final|_actor_best)$")

# 单文件上传超时 (秒) - 超过此时间无响应判定卡死, 触发重试
UPLOAD_TIMEOUT = 300
MAX_RETRIES = 5


def upload_file_with_retry(api, *, path_or_fileobj, path_in_repo, repo_id, label=""):
    """带 SIGALRM 超时 + 指数退避重试的单文件上传。
    huggingface_hub.upload_file 无 timeout 参数, 底层 requests 无 read timeout,
    网络断开会无限等待。这里用 SIGALRM 强制中断。
    """
    def _alarm_handler(signum, frame):
        raise TimeoutError(f"upload 超过 {UPLOAD_TIMEOUT}s 无响应")

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(UPLOAD_TIMEOUT)
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=api.token,
            )
            return  # 成功
        except TimeoutError as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  [超时 重试 {attempt}/{MAX_RETRIES}] {label} 超过 {UPLOAD_TIMEOUT}s, {wait}s 后重试")
            time.sleep(wait)
        except ConnectionError as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  [断连 重试 {attempt}/{MAX_RETRIES}] {label} ({type(e).__name__}), {wait}s 后重试")
            time.sleep(wait)
        except Exception as e:
            last_err = e
            # 429 限流: 等久一点再重试 (HF 提示 Retry after 60s)
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 429:
                wait = 120  # 限流时等 2 分钟
                print(f"  [429限流 重试 {attempt}/{MAX_RETRIES}] {label}, 等 {wait}s 后重试")
                time.sleep(wait)
            elif status is not None and 400 <= status < 500:
                # 其他 4xx (鉴权/参数) 重试无意义
                raise
            else:
                wait = min(2 ** attempt, 60)
                print(f"  [重试 {attempt}/{MAX_RETRIES}] {label} ({type(e).__name__}: {str(e)[:80]}), {wait}s 后重试")
                time.sleep(wait)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    raise last_err


def collect_files_to_pack(local_dir: Path, section: str):
    """收集 local_dir 下应打包的文件。
    - section == 'results': 只保留 best_eval*.png/csv, 跳过视频
    - section == 'models': 保留所有非视频文件
    返回 [(绝对路径, 在 zip 内的相对路径), ...]
    """
    out = []
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir)
        ext = p.suffix.lower()
        if ext in SKIP_EXTS:
            continue
        if section == "results":
            # results 只保留 best_eval*.png 和 best_eval*plot_data.csv
            if not RESULTS_KEEP_RE.match(p.name):
                continue
        out.append((p, rel))
    return out


def build_zip_bytes(files, npy_compress=True):
    """把 files [(abs_path, zip_rel_path)] 打包成 zip bytes。
    npy 文件转 npz 压缩后写入 zip (省空间)。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for abs_path, zip_rel in files:
            ext = abs_path.suffix.lower()
            if ext == ".npy" and npy_compress:
                # npy -> npz
                arr = np.load(abs_path, allow_pickle=True)
                inner = io.BytesIO()
                np.savez_compressed(inner, data=arr)
                zf.writestr(str(zip_rel).rsplit(".npy", 1)[0] + ".npz", inner.getvalue())
            else:
                zf.write(abs_path, str(zip_rel))
    return buf.getvalue()


def fmt_size(n):
    """智能单位: B/kB/MB/GB, 保留 2 位有效数字"""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024


def upload_packed_dir(local_dir, repo_id, path_in_repo, token, section, dry_run):
    """把 local_dir 打包成 1 个 zip, 上传到 path_in_repo/<dirname>.zip"""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        print(f"[warn] 目录不存在: {local_dir}, 跳过")
        return 0

    files = collect_files_to_pack(local_dir, section)
    if not files:
        print(f"[info] {local_dir} 无可打包文件, 跳过")
        return 0

    zip_name = f"{local_dir.name}.zip"
    remote_path = f"{path_in_repo.rstrip('/')}/{zip_name}"
    total_size = sum(f[0].stat().st_size for f in files)

    if dry_run:
        print(f"  [dry-run] 打包 {len(files)} 文件 -> {remote_path} (原始 {fmt_size(total_size)})")
        for abs_p, rel in files:
            print(f"           - {rel}")
        return 1

    print(f"  打包 {len(files)} 文件 ({fmt_size(total_size)}) -> {remote_path}")
    try:
        zip_bytes = build_zip_bytes(files)
        api = HfApi(token=token)
        upload_file_with_retry(
            api,
            path_or_fileobj=zip_bytes,
            path_in_repo=remote_path,
            repo_id=repo_id,
            label=zip_name,
        )
        ratio = (1 - len(zip_bytes) / total_size) * 100 if total_size > 0 else 0
        print(f"  [ok] {remote_path} (压缩后 {fmt_size(len(zip_bytes))}, 省 {ratio:.0f}%)")
        return 1
    except Exception as e:
        print(f"  [error] {local_dir.name}: {e}")
        return 0


def upload_models_section(local_root, repo_id, token, dry_run):
    """models: 每个 best/final/*_final/*_actor_best 目录打包成一个 zip"""
    total = 0
    for exp_dir in sorted(local_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith(("imi", "happo")):
            print(f"[跳过] {exp_dir.name} (imi/happo)")
            continue
        print(f"--- {exp_dir.name} ---")
        for trial_dir in sorted(exp_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            # 找该 trial 下应打包的子目录 (best/final/*_final/*_actor_best)
            for sub in sorted(trial_dir.iterdir()):
                if not sub.is_dir():
                    continue
                if (sub.name in MODELS_PACK_DIRNAMES
                        or MODELS_PACK_DIRNAME_RE.match(sub.name)):
                    rir = f"models/{exp_dir.name}/{trial_dir.name}"
                    total += upload_packed_dir(
                        sub, repo_id, rir, token, "models", dry_run
                    )
    return total


def upload_results_section(local_root, repo_id, token, dry_run):
    """results: 每个 <map>/<algo>/ 目录打包成一个 zip (只保留 best_eval*)"""
    total = 0
    for map_dir in sorted(local_root.iterdir()):
        if not map_dir.is_dir():
            continue
        for algo_dir in sorted(map_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            rir = f"results/{map_dir.name}"
            total += upload_packed_dir(
                algo_dir, repo_id, rir, token, "results", dry_run
            )
    return total


def main():
    p = argparse.ArgumentParser(
        description="打包上传 models/results 到 HF dataset (按目录打包, 避免限流)"
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HF dataset repo id (默认 {DEFAULT_REPO_ID})")
    p.add_argument("--token", default=os.environ.get(DEFAULT_TOKEN_ENV),
                   help=f"HF token (默认从 ${DEFAULT_TOKEN_ENV} 读)")
    p.add_argument("--root", default="/root/autodl-tmp",
                   help="本地根目录")
    p.add_argument("--only", choices=["models", "results", "both"], default="both",
                   help="只上传哪部分")
    p.add_argument("--create", action="store_true",
                   help="仓库不存在时创建")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要打包上传的内容, 不实际上传")
    args = p.parse_args()

    if "YOUR_HF_USERNAME" in args.repo_id:
        print(f"[错误] 请用 --repo-id 指定你的仓库")
        sys.exit(1)
    if not args.token:
        print(f"[错误] 未找到 token. 请用 --token 或设置环境变量 {DEFAULT_TOKEN_ENV}")
        print(f"       获取 token: https://huggingface.co/settings/tokens (权限选 Write)")
        sys.exit(1)

    print(f"仓库: {args.repo_id} (dataset)")
    print(f"模式: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"超时: {UPLOAD_TIMEOUT}s, 重试: {MAX_RETRIES} 次\n")

    # 确保 repo 存在
    if args.create and not args.dry_run:
        url = create_repo(repo_id=args.repo_id, repo_type="dataset",
                          private=False, exist_ok=True, token=args.token)
        print(f"[ok] 仓库就绪: {url}\n")

    total = 0
    if args.only in ("models", "both"):
        models_root = Path(args.root) / "models"
        if models_root.is_dir():
            print(f"=== 上传 models/ (按 best/final 目录打包) ===")
            total += upload_models_section(models_root, args.repo_id, args.token, args.dry_run)
            print()
        else:
            print(f"[warn] {models_root} 不存在, 跳过\n")

    if args.only in ("results", "both"):
        results_root = Path(args.root) / "results"
        if results_root.is_dir():
            print(f"=== 上传 results/ (按 algo 目录打包, 只保留 best_eval*) ===")
            total += upload_results_section(results_root, args.repo_id, args.token, args.dry_run)
            print()
        else:
            print(f"[warn] {results_root} 不存在, 跳过\n")

    print(f"完成. 共上传 {total} 个 zip 包 (= {total} 次 API 请求).")


if __name__ == "__main__":
    main()
