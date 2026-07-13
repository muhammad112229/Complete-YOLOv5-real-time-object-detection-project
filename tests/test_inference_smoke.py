"""Smoke test for the production YOLOv5 inference pipeline."""

from __future__ import annotations

from pathlib import Path

from src.common import project_root, require_python_package
from src.inference_engine import InferenceConfig, InferenceEngine


def _find_local_image(root: Path) -> Path | None:
    candidates = [
        root / "external" / "yolov5" / "data" / "images" / "bus.jpg",
        root / "external" / "yolov5" / "data" / "images" / "zidane.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for base in [root / "outputs" / "images", root / "data" / "smoke"]:
        if base.exists():
            for suffix in ("*.jpg", "*.jpeg", "*.png"):
                match = next(base.rglob(suffix), None)
                if match is not None:
                    return match
    return None


def _make_temporary_image(path: Path) -> Path:
    cv2 = require_python_package("cv2", "opencv-python")
    numpy = require_python_package("numpy")
    image = numpy.zeros((320, 320, 3), dtype=numpy.uint8)
    cv2.rectangle(image, (80, 80), (240, 240), (255, 255, 255), -1)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write temporary smoke image: {path}")
    return path


def test_production_checkpoint_inference_smoke(tmp_path: Path) -> None:
    root = project_root()
    model_path = root / "models" / "yolov5s_coco20k_best.pt"
    assert model_path.is_file(), f"Missing production checkpoint: {model_path}"

    source = _find_local_image(root) or _make_temporary_image(tmp_path / "generated_smoke.jpg")
    config = InferenceConfig(
        model_path=model_path,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        image_size=640,
        device="auto",
        class_filter=None,
        output_directory=tmp_path,
    )
    engine = InferenceEngine(config)
    result = engine.predict_image(source)
    output = engine.save_annotated_image(result, tmp_path / "annotated_smoke.jpg")

    assert output.is_file()
    assert isinstance(result.detections, tuple)
    assert result.detection_count == len(result.detections)
    for detection in result.detections:
        assert {
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        }.issubset(detection)
