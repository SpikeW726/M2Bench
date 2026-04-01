"""AutoDL 数据盘路径约定（autodl 分支）。

将默认的 models/、evaluators/results/、runs/ 映射到 /root/autodl-tmp，
避免 checkpoint、TensorBoard 与评估产物占满系统盘。绝对路径保持不变。
"""
from __future__ import annotations

from pathlib import Path

AUTODL_MODELS_ROOT = Path("/root/autodl-tmp/models")
AUTODL_RESULTS_ROOT = Path("/root/autodl-tmp/results")
AUTODL_RUNS_ROOT = Path("/root/autodl-tmp/runs")


def resolve_models_path(path: str | Path | None) -> Path | None:
    """相对路径若以 models/ 开头（或仅为 models），映射到 AUTODL_MODELS_ROOT；绝对路径原样返回。"""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    s = str(path).replace("\\", "/").rstrip("/")
    if s == "models" or s.startswith("models/"):
        suffix = s[7:].lstrip("/") if s.startswith("models/") else ""
        return AUTODL_MODELS_ROOT / suffix if suffix else AUTODL_MODELS_ROOT
    return AUTODL_MODELS_ROOT / s


def resolve_results_path(path: str | None) -> str | None:
    """evaluators/results/... 或同目录结构的相对路径映射到 AUTODL_RESULTS_ROOT；绝对路径原样返回。"""
    if path is None:
        return None
    q = Path(path)
    if q.is_absolute():
        return str(q)
    s = str(path).replace("\\", "/").rstrip("/")
    prefix = "evaluators/results"
    if s == prefix or s.startswith(prefix + "/"):
        suffix = s[len(prefix) :].lstrip("/")
        return str(AUTODL_RESULTS_ROOT / suffix) if suffix else str(AUTODL_RESULTS_ROOT)
    return str(AUTODL_RESULTS_ROOT / s)


def resolve_runs_path(path: str | Path | None) -> Path | None:
    """相对路径若以 runs/ 开头（或仅为 runs），映射到 AUTODL_RUNS_ROOT；绝对路径原样返回。"""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    s = str(path).replace("\\", "/").rstrip("/")
    if s == "runs" or s.startswith("runs/"):
        suffix = s[5:].lstrip("/") if s.startswith("runs/") else ""
        return AUTODL_RUNS_ROOT / suffix if suffix else AUTODL_RUNS_ROOT
    return AUTODL_RUNS_ROOT / s
