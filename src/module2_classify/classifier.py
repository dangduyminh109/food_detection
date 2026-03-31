"""Fine-grained classifier scaffolding for food recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Classification:
    """Represents a model prediction for a single crop."""

    image_path: Path
    label: str
    confidence: float


class FoodClassifier:
    """Mock classifier for identifying food names from cropped images."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._model: Optional[object] = None

    def load_model(self) -> None:
        """Load a fine-grained classifier model.

        Replace this stub with PyTorch model loading logic.
        """
        self._model = object()

    def predict(self, image_paths: List[Path]) -> List[Classification]:
        """Generate mock predictions for each image path."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results: List[Classification] = []
        for image_path in image_paths:
            results.append(Classification(image_path=image_path, label="mock_food", confidence=0.5))
        return results
