from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluators" / "results"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

def user_path(path: str | Path) -> Path:
    """Expand a user-supplied path without silently relocating it."""
    return Path(path).expanduser()

def result_path(path: str | Path, results_dir: str | Path | None = None) -> Path:
    """Optionally relocate a standard evaluation path under a chosen results root."""
    source = user_path(path)
    if results_dir is None:
        return source

    if source.is_absolute():
        try:
            relative = source.relative_to(DEFAULT_RESULTS_DIR)
        except ValueError:
            relative = Path(source.name)
    else:
        try:
            relative = source.relative_to(Path("evaluators") / "results")
        except ValueError:
            relative = Path(source.name)
    return user_path(results_dir) / relative
