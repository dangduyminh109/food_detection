"""YOLO detector wrapper for food item cropping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import shutil


try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency for scaffolding
    YOLO = None


@dataclass
class DetectionResult:
    """Container for crop outputs."""

    crop_paths: List[Path]


class YoloDetector:
    """Thin wrapper around a YOLO model to crop detected food items."""

    def __init__(self, model_path: Path, device: str = "cpu", conf: float = 0.25) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.conf = conf
        self._model: Optional[object] = None

    def load_model(self) -> None:
        """Load the YOLO model from disk."""
        if YOLO is None:
            raise ImportError("ultralytics is not installed")
        self._model = YOLO(str(self.model_path))

    def detect_and_crop(self, image_path: Path, output_dir: Path) -> DetectionResult:
        """Detect food items and write cropped images to the output directory.

        This scaffolding implementation simulates a single crop by copying the
        input image. Replace with real YOLO inference and cropping logic.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        simulated_crop = output_dir / "crop_0.jpg"
        shutil.copy2(image_path, simulated_crop)
        return DetectionResult(crop_paths=[simulated_crop])
