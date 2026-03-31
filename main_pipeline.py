"""Main orchestration for the Food Travel pipeline."""

from __future__ import annotations

from pathlib import Path

from src.module1_detect.detector import YoloDetector
from src.module2_classify.classifier import FoodClassifier
from src.module3_rag.rag_engine import RAGEngine
from src.utils.text_utils import format_rag_output


def run_pipeline(image_path: Path) -> None:
	"""Run the detection, classification, and RAG stages."""
	detector = YoloDetector(model_path=Path("models/detection/yolo.pt"))
	classifier = FoodClassifier(model_path=Path("models/classification/food_model.pt"))
	rag_engine = RAGEngine(vector_db_path=Path("data/vector_db"))

	detector.load_model()
	classifier.load_model()
	rag_engine.load_index()

	detection = detector.detect_and_crop(image_path=image_path, output_dir=Path("data/processed"))
	predictions = classifier.predict(detection.crop_paths)

	for prediction in predictions:
		description = rag_engine.query(prediction.label)
		formatted = format_rag_output(prediction.label, description)
		print(f"Image: {prediction.image_path}")
		print(f"Prediction: {prediction.label} (confidence={prediction.confidence:.2f})")
		print(f"RAG: {formatted}")
		print("-" * 40)


if __name__ == "__main__":
	try:
		run_pipeline(image_path=Path("data/raw/sample.jpg"))
	except Exception as exc:  # pragma: no cover - scaffolding guard
		print(f"Pipeline failed: {exc}")
