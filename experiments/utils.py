"""Utilities for experiments."""

from pathlib import Path


def ensure_parent_dir(file_path: str | Path) -> None:
    """Ensure the parent directory of a file path exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
