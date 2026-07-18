#!/usr/bin/env python3

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

DEFAULT_REPO_ID = "SpikeW726/M2Bench"
DEFAULT_TOKEN_ENV = "HF_TOKEN"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluators" / "results"

SKIP_EXTS = {".mp4", ".gif", ".avi", ".mov", ".mkv", ".webm"}
# npy -> npz.
COMPRESS_EXTS = {".npy"}

RESULTS_KEEP_RE = re.compile(r"^best_eval.*\.(png|csv)$")

MODELS_PACK_DIRNAMES = {"best", "final"}
MODELS_PACK_DIRNAME_RE = re.compile(r"^.*(_final|_actor_best)$")

UPLOAD_TIMEOUT = 300
MAX_RETRIES = 5

def upload_file_with_retry(api, *, path_or_fileobj, path_in_repo, repo_id, label=""):
    def _alarm_handler(signum, frame):
        raise TimeoutError(f"Upload did not respond within {UPLOAD_TIMEOUT}s")

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
            return
        except TimeoutError as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  [timeout retry {attempt}/{MAX_RETRIES}] {label} exceeded {UPLOAD_TIMEOUT}s; retrying in {wait}s")
            time.sleep(wait)
        except ConnectionError as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  [connection retry {attempt}/{MAX_RETRIES}] {label} ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            last_err = e

            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 429:
                wait = 120
                print(f"  [rate-limit retry {attempt}/{MAX_RETRIES}] {label}; retrying in {wait}s")
                time.sleep(wait)
            elif status is not None and 400 <= status < 500:

                raise
            else:
                wait = min(2 ** attempt, 60)
                print(f"  [retry {attempt}/{MAX_RETRIES}] {label} ({type(e).__name__}: {str(e)[:80]}); retrying in {wait}s")
                time.sleep(wait)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    raise last_err

def collect_files_to_pack(local_dir: Path, section: str):
    out = []
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir)
        ext = p.suffix.lower()
        if ext in SKIP_EXTS:
            continue
        if section == "results":

            if not RESULTS_KEEP_RE.match(p.name):
                continue
        out.append((p, rel))
    return out

def build_zip_bytes(files, npy_compress=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for abs_path, zip_rel in files:
            ext = abs_path.suffix.lower()
            if ext == ".npy" and npy_compress:
                # npy -> npz.
                arr = np.load(abs_path, allow_pickle=True)
                inner = io.BytesIO()
                np.savez_compressed(inner, data=arr)
                zf.writestr(str(zip_rel).rsplit(".npy", 1)[0] + ".npz", inner.getvalue())
            else:
                zf.write(abs_path, str(zip_rel))
    return buf.getvalue()

def fmt_size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024

def upload_packed_dir(local_dir, repo_id, path_in_repo, token, section, dry_run):
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        print(f"[warn] Directory does not exist: {local_dir}; skipping")
        return 0

    files = collect_files_to_pack(local_dir, section)
    if not files:
        print(f"[info] No files to package in {local_dir}; skipping")
        return 0

    zip_name = f"{local_dir.name}.zip"
    remote_path = f"{path_in_repo.rstrip('/')}/{zip_name}"
    total_size = sum(f[0].stat().st_size for f in files)

    if dry_run:
        print(f"  [dry-run] Package {len(files)} files -> {remote_path} (original {fmt_size(total_size)})")
        for abs_p, rel in files:
            print(f"           - {rel}")
        return 1

    print(f"  Packaging {len(files)} files ({fmt_size(total_size)}) -> {remote_path}")
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
        print(f"  [ok] {remote_path} (compressed {fmt_size(len(zip_bytes))}, saved {ratio:.0f}%)")
        return 1
    except Exception as e:
        print(f"  [error] {local_dir.name}: {e}")
        return 0

def upload_models_section(local_root, repo_id, token, dry_run):
    total = 0
    for exp_dir in sorted(local_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith(("imi", "happo")):
            print(f"[skip] {exp_dir.name} (imi/happo)")
            continue
        print(f"--- {exp_dir.name} ---")
        for trial_dir in sorted(exp_dir.iterdir()):
            if not trial_dir.is_dir():
                continue

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
        description="Package and upload models/results to a Hugging Face dataset with one archive per directory"
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})")
    p.add_argument("--token", default=os.environ.get(DEFAULT_TOKEN_ENV),
                   help=f"Hugging Face token (default: ${DEFAULT_TOKEN_ENV})")
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR),
                   help="Local model root directory")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="Local evaluation-results root directory")
    p.add_argument("--only", choices=["models", "results", "both"], default="both",
                   help="Select which artifacts to upload")
    p.add_argument("--create", action="store_true",
                   help="Create the repository if it does not exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Print packages without uploading them")
    args = p.parse_args()

    if "YOUR_HF_USERNAME" in args.repo_id:
        print("[error] Specify your repository with --repo-id")
        sys.exit(1)
    if not args.token:
        print(f"[error] Token not found. Use --token or set {DEFAULT_TOKEN_ENV}")
        print("        Create a Write token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"Repository: {args.repo_id} (dataset)")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Timeout: {UPLOAD_TIMEOUT}s, retries: {MAX_RETRIES}\n")

    if args.create and not args.dry_run:
        url = create_repo(repo_id=args.repo_id, repo_type="dataset",
                          private=False, exist_ok=True, token=args.token)
        print(f"[ok] Repository ready: {url}\n")

    total = 0
    if args.only in ("models", "both"):
        models_root = Path(args.models_dir).expanduser()
        if models_root.is_dir():
            print("=== Upload models/ (one archive per best/final directory) ===")
            total += upload_models_section(models_root, args.repo_id, args.token, args.dry_run)
            print()
        else:
            print(f"[warn] {models_root} does not exist; skipping\n")

    if args.only in ("results", "both"):
        results_root = Path(args.results_dir).expanduser()
        if results_root.is_dir():
            print("=== Upload results/ (one archive per algorithm, best_eval* only) ===")
            total += upload_results_section(results_root, args.repo_id, args.token, args.dry_run)
            print()
        else:
            print(f"[warn] {results_root} does not exist; skipping\n")

    print(f"Done. Uploaded {total} zip archives using {total} API requests.")

if __name__ == "__main__":
    main()
