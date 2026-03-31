"""Project scaffolding script for the Food Travel pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_STRUCTURE = {
    "data": ["raw", "processed", "vector_db"],
    "models": ["detection", "classification"],
    "notebooks": [],
    "src": [
        "module1_detect",
        "module2_classify",
        "module3_rag",
        "utils",
    ],
}

FILES_TO_CREATE = [
    "src/__init__.py",
    "src/module1_detect/__init__.py",
    "src/module1_detect/detector.py",
    "src/module2_classify/__init__.py",
    "src/module2_classify/classifier.py",
    "src/module3_rag/__init__.py",
    "src/module3_rag/rag_engine.py",
    "src/utils/__init__.py",
    "src/utils/image_utils.py",
    "src/utils/text_utils.py",
    "main_pipeline.py",
    "requirements.txt",
    "README.md",
]


def _create_directories(base_path: Path) -> None:
    for root, subfolders in PROJECT_STRUCTURE.items():
        root_path = base_path / root
        root_path.mkdir(parents=True, exist_ok=True)
        for subfolder in subfolders:
            (root_path / subfolder).mkdir(parents=True, exist_ok=True)


def _create_files(base_path: Path, files: Iterable[str]) -> None:
    for relative_path in files:
        file_path = base_path / relative_path
        if file_path.exists():
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")


def main() -> None:
    base_path = Path(__file__).resolve().parent
    _create_directories(base_path)
    _create_files(base_path, FILES_TO_CREATE)
    print("Project structure created at:", base_path)


if __name__ == "__main__":
    main()
