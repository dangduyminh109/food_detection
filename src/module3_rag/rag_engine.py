"""RAG engine scaffolding for food information retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class RAGEngine:
    """Mock RAG engine that simulates querying a vector database."""

    def __init__(self, vector_db_path: Path) -> None:
        self.vector_db_path = Path(vector_db_path)
        self._index_loaded: bool = False
        self._client: Optional[object] = None

    def load_index(self) -> None:
        """Load or connect to the vector database."""
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self._client = object()
        self._index_loaded = True

    def query(self, food_name: str) -> str:
        """Return a mock description for the given food name."""
        if not self._index_loaded:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        return f"{food_name} is a popular dish in the Food Travel dataset."
