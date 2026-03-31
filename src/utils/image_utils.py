"""Image utility helpers used across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder: Path, extensions: List[str] | None = None) -> List[Path]:
    """List image files in a folder using a set of allowed extensions."""
    folder = Path(folder)
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]
    return [p for p in folder.iterdir() if p.suffix.lower() in extensions]
